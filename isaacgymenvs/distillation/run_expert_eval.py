import isaacgym
import torch
from isaacgym import gymtorch  # CODEX
from isaacgym.torch_utils import tensor_clamp  # CODEX

import json
from datetime import datetime
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, open_dict
from tqdm import tqdm

from isaacgymenvs.distillation.dagger_mobile_trainer import DaggerMobile
from isaacgymenvs.distillation.run_dagger_eval import (
    _cfg,
    _mean_logs,
    _print_eval_logs,
    _save_table_txt,
    _scalar,
)
from isaacgymenvs.utils import eef_ctrl  # CODEX
from isaacgymenvs.utils.training_utils import colorprint


def _apply_eval_overrides(cfg: DictConfig) -> None:
    debug_visuals = bool(_cfg(cfg.eval, "debug_visuals", False))
    with open_dict(cfg):
        cfg.dagger.load_ckpt_path = None

        enable_viewer = _cfg(cfg.eval, "enable_viewer")
        if enable_viewer is not None:
            cfg.headless = not bool(enable_viewer)
        elif debug_visuals:
            cfg.headless = False
        else:
            cfg.headless = True

        if debug_visuals:
            cfg.force_render = True


def _update_reaching_phase_collision(trainer, reaching_collision_flags, fabric_switch_flags):
    env = trainer.env
    reaching_collision_flags.logical_or_(env.env_collision.bool() & (~fabric_switch_flags))

    if getattr(env, "enable_fabric", False):
        switching_matching_err = env._get_eef_point_matching_err(
            curent_eef_pos7=env._eef_state[:, :7],
            target_eef_pos7=torch.cat([env.switching_target_pos, env.switching_target_quat], dim=-1),
        )
        fabric_switch_flags.logical_or_(switching_matching_err < env.switch_tol)


# CODEX
def _weighted_ik_cfg(cfg: DictConfig, key: str, default):
    weighted_ik_cfg = _cfg(cfg.eval, "weighted_ik", None)
    if weighted_ik_cfg is None:
        return default
    return _cfg(weighted_ik_cfg, key, default)


# CODEX
def _get_expert_reaching_controller(cfg: DictConfig) -> str:
    reaching_controller = str(_cfg(cfg.eval, "expert_reaching_controller", "fabric")).lower()
    if reaching_controller not in ["fabric", "weighted_ik"]:
        raise ValueError(
            "eval.expert_reaching_controller must be 'fabric' or 'weighted_ik', "
            f"got {reaching_controller}"
        )
    return reaching_controller


# CODEX
def _compute_weighted_ik_abs_targets(env, cfg: DictConfig):
    target_eef_quat = torch.cat([env.switching_target_pos, env.switching_target_quat], dim=-1)
    pos_error, axis_angle_error = eef_ctrl.get_pose_error(
        current_eef_pos=env.states["eef_pos"],
        current_eef_quat=env.states["eef_quat"],
        ctrl_target_eef_pos=target_eef_quat[:, :3],
        ctrl_target_eef_quat=target_eef_quat[:, 3:],
    )

    pos_gain = float(_weighted_ik_cfg(cfg, "pos_gain", 3.0))
    rot_gain = float(_weighted_ik_cfg(cfg, "rot_gain", 3.0))
    xdot_des = torch.cat([pos_gain * pos_error, rot_gain * axis_angle_error], dim=-1)

    max_task_speed = float(_weighted_ik_cfg(cfg, "max_task_speed", 2.0))
    xdot_norm = torch.linalg.norm(xdot_des, dim=-1, keepdim=True).clamp_min(1.0e-8)
    xdot_des = xdot_des * torch.clamp(max_task_speed / xdot_norm, max=1.0)

    jacobian = env._j_eef[:, :, :10]
    jacobian_t = jacobian.transpose(-1, -2)
    base_xy_weight = float(_weighted_ik_cfg(cfg, "base_xy_weight", 10.0))
    base_yaw_weight = float(_weighted_ik_cfg(cfg, "base_yaw_weight", 10.0))
    arm_weight = float(_weighted_ik_cfg(cfg, "arm_weight", 1.0))
    weights = torch.tensor(
        [base_xy_weight, base_xy_weight, base_yaw_weight] + [arm_weight] * 7,
        dtype=jacobian.dtype,
        device=jacobian.device,
    )
    regularization = float(_weighted_ik_cfg(cfg, "regularization", 1.0e-2))
    identity = torch.eye(10, dtype=jacobian.dtype, device=jacobian.device).unsqueeze(0)
    weighted_regularizer = regularization * (weights.square().view(1, 10, 1) * identity)
    a_matrix = jacobian_t @ jacobian + weighted_regularizer
    b_vector = jacobian_t @ xdot_des.unsqueeze(-1)
    qdot = torch.linalg.solve(a_matrix, b_vector).squeeze(-1)

    base_xy_vel_limit = float(_weighted_ik_cfg(cfg, "base_xy_vel_limit", 0.5))
    base_yaw_vel_limit = float(_weighted_ik_cfg(cfg, "base_yaw_vel_limit", 1.0))
    arm_vel_limit = float(_weighted_ik_cfg(cfg, "arm_vel_limit", 1.5))
    qdot[:, :2] = torch.clamp(qdot[:, :2], -base_xy_vel_limit, base_xy_vel_limit)
    qdot[:, 2] = torch.clamp(qdot[:, 2], -base_yaw_vel_limit, base_yaw_vel_limit)
    qdot[:, 3:10] = torch.clamp(qdot[:, 3:10], -arm_vel_limit, arm_vel_limit)

    abs_targets = env.states["q"].clone()
    abs_targets[:, :10] = env.states["q"][:, :10] + env.dt * qdot
    abs_targets[:, 10:26] = env.canonical_joint_config[:, 10:26]
    return tensor_clamp(abs_targets, env.robot_dof_lower_limits, env.robot_dof_upper_limits)


# CODEX
def _compute_teacher_abs_targets_without_fabric(env, expert_actions):
    enable_fabric = env.enable_fabric
    try:
        env.enable_fabric = False
        return env._pre_physics_step_teacher(expert_actions)
    finally:
        env.enable_fabric = enable_fabric


# CODEX
def _step_abs_joint_targets(env, abs_targets):
    env.abs_actions[:] = tensor_clamp(abs_targets, env.robot_dof_lower_limits, env.robot_dof_upper_limits)
    # CODEX
    env.delta_joint_actions[:] = env.abs_actions - env.states["q"]
    env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env.abs_actions))

    if env.object_wrench_args["enable"]:
        env._apply_object_wrench()

    for _ in range(env.control_freq_inv):
        if env.force_render:
            env.render()
        env.gym.simulate(env.sim)

    if env.device == "cpu":
        env.gym.fetch_results(env.sim, True)

    env.post_physics_step()
    env.control_steps += 1
    env.timeout_buf = (env.progress_buf >= env.max_episode_length - 1) & (env.reset_buf != 0)

    if env.dr_randomizations.get("observations", None):
        env.obs_buf = env.dr_randomizations["observations"]["noise_lambda"](env.obs_buf)

    env.extras["time_outs"] = env.timeout_buf.to(env.rl_device)
    env.obs_dict["obs"] = torch.clamp(env.obs_buf, -env.clip_obs, env.clip_obs).to(env.rl_device)
    if env.num_states > 0:
        env.obs_dict["states"] = env.get_state()
    return env.obs_dict, env.rew_buf.to(env.rl_device), env.reset_buf.to(env.rl_device), env.extras


# CODEX
def _compute_arm_manipulability(env):
    arm_jacobian = env._j_eef[:, :, 3:10]
    if arm_jacobian.shape[-1] != 7:
        raise ValueError(f"Expected Franka arm Jacobian with 7 columns, got shape {arm_jacobian.shape}")

    jj_t = torch.bmm(arm_jacobian, arm_jacobian.transpose(1, 2))
    return torch.sqrt(torch.clamp(torch.linalg.det(jj_t), min=0.0))


# CODEX
def _compute_franka_joint_limit_margin(env):
    q_arm = env.states["q"][:, 3:10]
    lower_limits, upper_limits = env.get_joint_limits_franka()
    return torch.minimum(q_arm - lower_limits, upper_limits - q_arm)


def _table_rows(env):
    rows = []
    if hasattr(env, "table_surface_height") and hasattr(env, "box_dims") and hasattr(env, "success_flags"):
        table_heights = env.table_surface_height.detach().cpu()
        box_dims = env.box_dims.detach().cpu()
        success_flags = env.success_flags.detach().cpu()
        for env_id in range(env.num_envs):
            rows.append({
                "env_id": env_id,
                "table_height": float(table_heights[env_id]),
                "compartment_dim_x": float(box_dims[env_id, 0]),
                "compartment_dim_y": float(box_dims[env_id, 1]),
                "compartment_dim_z": float(box_dims[env_id, 2]),
                "success": bool(success_flags[env_id] > 0),
            })
    return rows


def _run_expert_episode(trainer, cfg: DictConfig):
    env = trainer.env
    # CODEX
    reaching_controller = _get_expert_reaching_controller(cfg)
    env.distillation_mode = False
    env.reset_idx()
    env.compute_observations()
    env.abs_actions[:] = env.states["q"].clone()

    reaching_collision_flags = torch.zeros(env.num_envs, device=trainer.device, dtype=torch.bool)
    fabric_switch_flags = torch.zeros(env.num_envs, device=trainer.device, dtype=torch.bool)
    # CODEX
    arm_manipulability_sum = torch.zeros((), device=trainer.device)
    # CODEX
    joint_limit_margin_sum = torch.zeros((), device=trainer.device)
    # CODEX
    reaching_arm_manipulability_sum = torch.zeros((), device=trainer.device)
    # CODEX
    reaching_joint_limit_margin_sum = torch.zeros((), device=trainer.device)
    # CODEX
    reaching_metric_env_count = torch.zeros((), device=trainer.device)
    # CODEX
    metric_steps = 0
    _update_reaching_phase_collision(trainer, reaching_collision_flags, fabric_switch_flags)

    for _ in tqdm(
        range(trainer.steps_per_episode),
        desc="Evaluating expert",
        ncols=None,
        dynamic_ncols=True,
        disable=(trainer.multi_gpu and trainer.global_rank != 0),
    ):
        batch_dict = {
            "is_train": False,
            "obs": env.obs_buf.clone(),
            "prev_actions": None,
        }

        with torch.no_grad():
            res_dict = trainer.teacher_model(batch_dict)

        expert_actions = torch.clamp(res_dict["mus"], -env.clip_actions, env.clip_actions)
        env.progress_buf[:] = 0
        # CODEX
        if reaching_controller == "weighted_ik":
            # CODEX
            reaching_phase_mask = ~fabric_switch_flags
            # CODEX
            teacher_abs_targets = _compute_teacher_abs_targets_without_fabric(env, expert_actions)
            # CODEX
            weighted_ik_abs_targets = _compute_weighted_ik_abs_targets(env, cfg)
            # CODEX
            step_abs_targets = teacher_abs_targets.clone()
            # CODEX
            step_abs_targets[reaching_phase_mask] = weighted_ik_abs_targets[reaching_phase_mask]
            # CODEX
            _step_abs_joint_targets(env, step_abs_targets)
        else:
            env.step(expert_actions)
        with torch.no_grad():
            # CODEX
            reaching_phase_mask = ~fabric_switch_flags
            # CODEX
            arm_manipulability = _compute_arm_manipulability(env)
            # CODEX
            joint_limit_margin_by_env = _compute_franka_joint_limit_margin(env).mean(dim=-1)
            # CODEX
            arm_manipulability_sum += arm_manipulability.mean()
            # CODEX
            joint_limit_margin_sum += joint_limit_margin_by_env.mean()
            # CODEX
            if torch.any(reaching_phase_mask):
                # CODEX
                reaching_arm_manipulability_sum += arm_manipulability[reaching_phase_mask].sum()
                # CODEX
                reaching_joint_limit_margin_sum += joint_limit_margin_by_env[reaching_phase_mask].sum()
                # CODEX
                reaching_metric_env_count += reaching_phase_mask.float().sum()
            # CODEX
            metric_steps += 1
        _update_reaching_phase_collision(trainer, reaching_collision_flags, fabric_switch_flags)

    eval_reaching_collision_rate = torch.mean(reaching_collision_flags.float()).item()
    # CODEX
    eval_arm_manipulability = (arm_manipulability_sum / metric_steps).item()
    # CODEX
    eval_joint_limit_margin = (joint_limit_margin_sum / metric_steps).item()
    # CODEX
    eval_reaching_arm_manipulability = (
        reaching_arm_manipulability_sum / torch.clamp(reaching_metric_env_count, min=1.0)
    ).item()
    # CODEX
    eval_reaching_joint_limit_margin = (
        reaching_joint_limit_margin_sum / torch.clamp(reaching_metric_env_count, min=1.0)
    ).item()
    eval_logs = {
        "metrics/eval_success_rate_5cm_final_step": env.extras["metrics/success_rate_5cm_per_step"],
        "metrics/eval_success_rate_5cm_per_ep": env.extras["metrics/success_rate_5cm_per_ep"],
        "metrics/eval_lifting_rate_5cm_final_step": env.extras["metrics/lifting_rate_5cm_per_step"],
        "metrics/eval_lifting_rate_5cm_per_ep": env.extras["metrics/lifting_rate_5cm_per_ep"],
        "metrics/eval_reaching_collision_rate_before_switch_per_ep": eval_reaching_collision_rate,
        # CODEX
        "metrics/eval_arm_manipulability": eval_arm_manipulability,
        # CODEX
        "metrics/eval_franka_joint_limit_margin": eval_joint_limit_margin,
        # CODEX
        "metrics/eval_reaching_arm_manipulability_before_switch": eval_reaching_arm_manipulability,
        # CODEX
        "metrics/eval_reaching_franka_joint_limit_margin_before_switch": eval_reaching_joint_limit_margin,
    }
    return eval_logs, _table_rows(env)


def _run_expert_eval(trainer, num_episodes, include_env_extras):
    logs_per_episode = []
    table_rows = []
    for episode in range(num_episodes):
        logs, rows = _run_expert_episode(trainer, trainer.cfg)  # CODEX
        logs = {key: _scalar(value) for key, value in logs.items()}
        if include_env_extras:
            logs = {**{key: _scalar(value) for key, value in trainer.env.extras.items()}, **logs}
        logs_per_episode.append(logs)
        table_rows.extend({"episode": episode, **row} for row in rows)
    return _mean_logs(logs_per_episode), table_rows


def _save_expert_eval_logs(cfg: DictConfig, eval_logs: dict) -> Path:
    metrics_path = _cfg(cfg.eval, "save_metrics_path")
    if metrics_path is None:
        exp_name = str(cfg.experiment or cfg.wandb_name)
        exp_name = "".join(char if char.isalnum() or char in "-_." else "_" for char in exp_name)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metrics_path = Path("logs") / "eval" / f"{exp_name}_{timestamp}.json"
    else:
        metrics_path = Path(metrics_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "teacher_checkpoint": str(cfg.teacher.ckpt),
        "num_episodes": int(_cfg(cfg.eval, "num_episodes", 1)),
        "metrics": {key: _scalar(value) for key, value in sorted(eval_logs.items())},
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return metrics_path


@hydra.main(config_name="dagger_config", config_path="../cfg")
def main(cfg: DictConfig):
    _apply_eval_overrides(cfg)

    num_episodes = int(_cfg(cfg.eval, "num_episodes", 1))
    if num_episodes <= 0:
        raise ValueError(f"eval.num_episodes must be positive, got {num_episodes}")
    if cfg["task"]["type"] != "WBC":
        raise ValueError(
            "run_expert_eval.py supports single-GPU WBC mobile expert eval only. "
            f"Unsupported task.type: {cfg['task']['type']}"
        )

    trainer = DaggerMobile(cfg=cfg)
    eval_logs, table_rows = _run_expert_eval(
        trainer,
        num_episodes,
        bool(_cfg(cfg.eval, "include_env_extras", True)),
    )

    _print_eval_logs(eval_logs)
    if getattr(trainer, "use_wandb", False):
        wandb.log(eval_logs, step=trainer.total_steps)

    metrics_path = None
    if bool(_cfg(cfg.eval, "save_metrics", True)):
        metrics_path = _save_expert_eval_logs(cfg, eval_logs)
        colorprint(f"Saved expert eval metrics to {metrics_path}", color="cyan")
    if table_rows:
        table_path = _save_table_txt(cfg, table_rows, metrics_path)
        colorprint(f"Saved expert eval table log to {table_path}", color="cyan")

    if getattr(trainer, "use_wandb", False):
        wandb.finish()


if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.disable = True
    main()
