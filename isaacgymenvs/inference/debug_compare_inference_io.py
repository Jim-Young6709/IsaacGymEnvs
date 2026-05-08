import argparse
from collections import OrderedDict

import torch

from isaacgymenvs.inference.inference_wbc_policy import WBCPolicyTransformer


def _clone_obs_dict(obs_dict):
    cloned = OrderedDict()
    for key, value in obs_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _print_diff(name, actual, expected):
    diff = actual - expected
    print(
        f"{name}: "
        f"mean_abs={diff.abs().mean().item():.8f} "
        f"max_abs={diff.abs().max().item():.8f}"
    )
    return diff.abs().mean().item(), diff.abs().max().item()


def _compare_record(policy, record, idx, q_hand_delta_scale):
    device = torch.device(policy.device)

    prev_abs_hand_actions = record["prev_abs_hand_actions"].to(device)
    prev_aux_anchor_state = None
    if getattr(policy, "has_aux_input", False) and record["aux_anchor_state"] is not None:
        prev_aux_anchor_state = record["aux_anchor_state"].unsqueeze(0).to(device)
        policy.aux_anchor_state[:] = prev_aux_anchor_state
    policy.abs_hand_actions[:] = prev_abs_hand_actions

    aux_inputs = None
    if record["aux_inputs"] is not None:
        aux_inputs = record["aux_inputs"].to(device)

    full_pcd = record["full_pcd_frankabase_frame_t"].to(device)
    eef_xyz = record["eef_xyz_frankabase_frame_t"].to(device)
    q_hand = record["q_hand"].to(device)
    q_arm_manip = record["q_arm_manip"].to(device)
    q_arm_vision = record["q_arm_vision"].to(device)
    prev_steps = int(policy.steps)

    base_vel_robot, franka_joint_pos, leap_joint_pos, arx_joint_pos, aux_pred_cpu, obs_dict_get_action = policy.get_action(
        full_pcd,
        eef_xyz,
        q_hand,
        q_arm_manip,
        q_arm_vision,
        aux_inputs=aux_inputs,
    )
    actual_actions_abs_get_action = torch.cat([
        torch.from_numpy(base_vel_robot),
        torch.from_numpy(franka_joint_pos),
        torch.from_numpy(leap_joint_pos),
        torch.from_numpy(arx_joint_pos),
    ], dim=0).to(torch.float32)
    actual_aux_pred_abs_get_action = (
        torch.from_numpy(aux_pred_cpu).to(torch.float32)
        if aux_pred_cpu is not None else None
    )

    policy.steps = prev_steps
    policy.abs_hand_actions[:] = prev_abs_hand_actions
    if prev_aux_anchor_state is not None:
        policy.aux_anchor_state[:] = prev_aux_anchor_state

    full_pcd_b = full_pcd.unsqueeze(0)
    eef_xyz_b = eef_xyz.unsqueeze(0)
    q_hand_b = policy.normalize_robot_joints(q_hand.unsqueeze(0), robot="leap", delta=False)
    q_hand_ctrl_delta_b = policy.normalize_robot_joints(
        (q_hand - policy.abs_hand_actions), robot="leap", delta=True
    ).unsqueeze(0)
    q_arm_manip_b = policy.normalize_robot_joints(q_arm_manip.unsqueeze(0), robot="franka", delta=False)
    q_arm_vision_b = policy.normalize_robot_joints(q_arm_vision.unsqueeze(0), robot="arx", delta=False)

    obs_dict = OrderedDict([
        ("full_pcd_frankabase_frame_t", full_pcd_b),
        ("eef_xyz_frankabase_frame_t", eef_xyz_b),
        ("aux_object_state", None if aux_inputs is None else aux_inputs.unsqueeze(0)),
        ("q_arm_manip", q_arm_manip_b),
        ("q_arm_vision", q_arm_vision_b),
        ("q_hand", q_hand_b),
        ("q_hand_ctrl_delta", q_hand_ctrl_delta_b * q_hand_delta_scale),
    ])

    policy.abs_hand_actions[:] = prev_abs_hand_actions
    if prev_aux_anchor_state is not None:
        policy.aux_anchor_state[:] = prev_aux_anchor_state
    step_action, aux_pred_t, obs_dict_after = policy.inference_policy(_clone_obs_dict(obs_dict))
    actual_actions_abs = step_action.clone()
    actual_actions_abs[:3] = step_action[:3]
    if policy.delta_franka_action:
        actual_actions_abs[3:10] = policy.unnormalize_robot_joints(
            actual_actions_abs[3:10], robot="franka", delta=True
        ) * policy.action_scale["franka"] * policy.dt + q_arm_manip
    else:
        actual_actions_abs[3:10] = policy.unnormalize_robot_joints(
            actual_actions_abs[3:10], robot="franka", delta=False
        )
    if policy.delta_leap_action:
        actual_actions_abs[10:26] = policy.unnormalize_robot_joints(
            actual_actions_abs[10:26], robot="leap", delta=True
        ) * policy.action_scale["leap"] * policy.dt + q_hand
    else:
        actual_actions_abs[10:26] = policy.unnormalize_robot_joints(
            actual_actions_abs[10:26], robot="leap", delta=False
        )
    if policy.delta_arx_action:
        actual_actions_abs[26:] = policy.unnormalize_robot_joints(
            actual_actions_abs[26:], robot="arx", delta=True
        ) * policy.action_scale["arx"] * policy.dt + q_arm_vision
    else:
        actual_actions_abs[26:] = policy.unnormalize_robot_joints(
            actual_actions_abs[26:], robot="arx", delta=False
        )
    actual_actions_abs = torch.max(
        torch.min(actual_actions_abs, policy.robot_dof_upper_limits), policy.robot_dof_lower_limits
    ).detach().cpu().to(torch.float32)

    expected_actions_abs = record["expected_actions_abs"].to(torch.float32)
    print(f"record idx={idx} episode={record['episode']} total_steps={record['total_steps']} env_id={record['env_id']}")
    stats = {}
    stats["actions_abs_get_action"] = _print_diff("actions_abs_get_action", actual_actions_abs_get_action, expected_actions_abs)
    stats["actions_abs_from_inference_policy"] = _print_diff(
        "actions_abs_from_inference_policy", actual_actions_abs, expected_actions_abs
    )

    if "obs_input_a0" in record:
        obs_input_exact = OrderedDict()
        for key, value in record["obs_input_a0"].items():
            if torch.is_tensor(value):
                obs_input_exact[key] = value.to(device)
            else:
                obs_input_exact[key] = value

        with torch.no_grad():
            policy.model.eval()
            exact_output = policy.model(_clone_obs_dict(obs_input_exact))

        expected_model_action_chunk = record["expected_model_action_chunk"].to(torch.float32)
        actual_model_action_chunk = exact_output["action"].detach().cpu().to(torch.float32)
        stats["model_action_chunk_from_saved_obs"] = _print_diff(
            "model_action_chunk_from_saved_obs", actual_model_action_chunk, expected_model_action_chunk
        )

        expected_model_step_action = torch.clamp(expected_model_action_chunk[0, 0, :32], -policy.clip_actions, policy.clip_actions)
        actual_model_step_action_tensor = torch.clamp(exact_output["action"][0, 0, :32], -policy.clip_actions, policy.clip_actions)
        actual_model_step_action = actual_model_step_action_tensor.detach().cpu().to(torch.float32)
        stats["step_action_from_saved_obs"] = _print_diff(
            "step_action_from_saved_obs", actual_model_step_action, expected_model_step_action
        )

        actual_actions_abs_from_saved_obs = actual_model_step_action_tensor.clone()
        actual_actions_abs_from_saved_obs[:3] = actual_model_step_action_tensor[:3]
        if policy.delta_franka_action:
            actual_actions_abs_from_saved_obs[3:10] = policy.unnormalize_robot_joints(
                actual_actions_abs_from_saved_obs[3:10], robot="franka", delta=True
            ) * policy.action_scale["franka"] * policy.dt + q_arm_manip
        else:
            actual_actions_abs_from_saved_obs[3:10] = policy.unnormalize_robot_joints(
                actual_actions_abs_from_saved_obs[3:10], robot="franka", delta=False
            )
        if policy.delta_leap_action:
            actual_actions_abs_from_saved_obs[10:26] = policy.unnormalize_robot_joints(
                actual_actions_abs_from_saved_obs[10:26], robot="leap", delta=True
            ) * policy.action_scale["leap"] * policy.dt + q_hand
        else:
            actual_actions_abs_from_saved_obs[10:26] = policy.unnormalize_robot_joints(
                actual_actions_abs_from_saved_obs[10:26], robot="leap", delta=False
            )
        if policy.delta_arx_action:
            actual_actions_abs_from_saved_obs[26:] = policy.unnormalize_robot_joints(
                actual_actions_abs_from_saved_obs[26:], robot="arx", delta=True
            ) * policy.action_scale["arx"] * policy.dt + q_arm_vision
        else:
            actual_actions_abs_from_saved_obs[26:] = policy.unnormalize_robot_joints(
                actual_actions_abs_from_saved_obs[26:], robot="arx", delta=False
            )
        actual_actions_abs_from_saved_obs = torch.max(
            torch.min(actual_actions_abs_from_saved_obs, policy.robot_dof_upper_limits), policy.robot_dof_lower_limits
        ).detach().cpu().to(torch.float32)
        stats["actions_abs_from_saved_obs"] = _print_diff(
            "actions_abs_from_saved_obs", actual_actions_abs_from_saved_obs, expected_actions_abs
        )

        if record["expected_model_aux_output"] is not None and "aux" in exact_output:
            expected_model_aux_output = record["expected_model_aux_output"].to(torch.float32)
            actual_model_aux_output = exact_output["aux"].detach().cpu().to(torch.float32)
            stats["model_aux_output_from_saved_obs"] = _print_diff(
                "model_aux_output_from_saved_obs", actual_model_aux_output, expected_model_aux_output
            )

        if "local_pcd_t" in obs_input_exact and "local_pcd_t" in obs_dict_get_action:
            actual_local_pcd_t = obs_dict_get_action["local_pcd_t"].detach().cpu().to(torch.float32)
            expected_local_pcd_t = record["obs_input_a0"]["local_pcd_t"].to(torch.float32)
            stats["local_pcd_t"] = _print_diff(
                "local_pcd_t", actual_local_pcd_t, expected_local_pcd_t
            )

    if record["expected_q_hand_ctrl_delta"] is not None and "q_hand_ctrl_delta" in obs_dict_get_action:
        actual_q_hand_ctrl_delta = obs_dict_get_action["q_hand_ctrl_delta"][0].detach().cpu().to(torch.float32)
        expected_q_hand_ctrl_delta = record["expected_q_hand_ctrl_delta"].to(torch.float32)
        stats["q_hand_ctrl_delta"] = _print_diff("q_hand_ctrl_delta", actual_q_hand_ctrl_delta, expected_q_hand_ctrl_delta)

    actual_step_action = step_action.detach().cpu().to(torch.float32)
    expected_step_action = record["expected_step_action"].to(torch.float32)
    stats["step_action"] = _print_diff("step_action", actual_step_action, expected_step_action)

    if record["expected_aux_pred_abs"] is not None and actual_aux_pred_abs_get_action is not None:
        expected_aux_pred_abs = record["expected_aux_pred_abs"].to(torch.float32)
        stats["aux_pred_abs_get_action"] = _print_diff(
            "aux_pred_abs_get_action", actual_aux_pred_abs_get_action, expected_aux_pred_abs
        )

    if record["expected_aux_pred_abs"] is not None and aux_pred_t is not None:
        actual_aux_pred_abs = aux_pred_t.detach().cpu().to(torch.float32)
        expected_aux_pred_abs = record["expected_aux_pred_abs"].to(torch.float32)
        stats["aux_pred_abs_from_inference_policy"] = _print_diff(
            "aux_pred_abs_from_inference_policy", actual_aux_pred_abs, expected_aux_pred_abs
        )

    print("")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="Path to debug_inference_io_pairs.pt")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path for inference_wbc_policy")
    parser.add_argument("--idx", type=int, default=None, help="Optional single record index inside the dump")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q-hand-delta-scale", type=float, default=1.0)
    args = parser.parse_args()

    records = torch.load(args.dump, map_location="cpu")
    if not records:
        raise ValueError(f"No records found in {args.dump}")
    policy = WBCPolicyTransformer({"seed": args.seed, "ckpt_path": args.ckpt})
    if args.idx is not None:
        if args.idx < 0 or args.idx >= len(records):
            raise IndexError(f"idx={args.idx} out of range for {len(records)} records")
        indices = [args.idx]
    else:
        indices = list(range(len(records)))

    summary_stats = {}
    for idx in indices:
        stats = _compare_record(policy, records[idx], idx, args.q_hand_delta_scale)
        for name, (mean_abs, max_abs) in stats.items():
            if name not in summary_stats:
                summary_stats[name] = {"mean_sum": 0.0, "count": 0, "max_abs": 0.0}
            summary_stats[name]["mean_sum"] += mean_abs
            summary_stats[name]["count"] += 1
            summary_stats[name]["max_abs"] = max(summary_stats[name]["max_abs"], max_abs)

    if len(indices) > 1:
        print("summary:")
        for name, stats in summary_stats.items():
            avg_mean_abs = stats["mean_sum"] / max(stats["count"], 1)
            print(
                f"{name}: "
                f"avg_mean_abs={avg_mean_abs:.8f} "
                f"max_abs={stats['max_abs']:.8f}"
            )


if __name__ == "__main__":
    main()
