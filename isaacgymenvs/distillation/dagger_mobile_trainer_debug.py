import os
import torch
from pathlib import Path

from isaacgymenvs.utils.rotation_conversions import quaternion_to_matrix_ig
from isaacgymenvs.utils.simulate_depth_cam import simulate_depth_cam_render_from_pose
from isaacgymenvs.utils.training_utils import colorprint


class DaggerMobileDebugMixin:
    def _debug_init_inference_io_dump(self):
        self.dump_inference_io = os.getenv("DUMP_INFERENCE_IO", "0") == "1"
        self.dump_inference_io_max = int(os.getenv("DUMP_INFERENCE_IO_MAX", "8"))
        self.dump_inference_io_env = int(os.getenv("DUMP_INFERENCE_IO_ENV", "0"))
        self.dump_inference_io_saved = 0
        dump_path_override = os.getenv("DUMP_INFERENCE_IO_PATH", "").strip()
        self.dump_inference_io_path = Path(dump_path_override) if dump_path_override else (self.save_dir / "debug_inference_io_pairs.pt")
        self.dump_inference_io_records = []
        self.debug_replay_snapshot_path = os.getenv("DEBUG_REPLAY_SNAPSHOT_PATH", "").strip()
        self.debug_replay_snapshot_max = int(os.getenv("DEBUG_REPLAY_SNAPSHOT_MAX", "0"))
        self.debug_replay_snapshot_done = False
        self.debug_parallel_preprocess_compare_count = 0

    def _debug_env_id(self):  # CODEX NEW
        return min(2, int(self.env.num_envs) - 1)

    def _debug_draw_cross(self, point_world, color, env_id=None, cross_len=0.035, clear_lines=False):  # CODEX NEW
        if not self.debug_mode:
            return
        if self.multi_gpu and self.global_rank != 0:
            return
        if not hasattr(self.env, "viewer") or self.env.viewer is None:
            return
        if env_id is None:
            env_id = self._debug_env_id()
        if clear_lines:
            self.env.gym.clear_lines(self.env.viewer)
        p = point_world.detach()
        pxm = p + torch.tensor([-cross_len, 0.0, 0.0], device=self.device)
        pxp = p + torch.tensor([ cross_len, 0.0, 0.0], device=self.device)
        pym = p + torch.tensor([0.0, -cross_len, 0.0], device=self.device)
        pyp = p + torch.tensor([0.0,  cross_len, 0.0], device=self.device)
        pzm = p + torch.tensor([0.0, 0.0, -cross_len], device=self.device)
        pzp = p + torch.tensor([0.0, 0.0,  cross_len], device=self.device)
        verts = []
        colors = []
        verts.extend([float(pxm[0]), float(pxm[1]), float(pxm[2]), float(pxp[0]), float(pxp[1]), float(pxp[2])])
        colors.extend(color)
        verts.extend([float(pym[0]), float(pym[1]), float(pym[2]), float(pyp[0]), float(pyp[1]), float(pyp[2])])
        colors.extend(color)
        verts.extend([float(pzm[0]), float(pzm[1]), float(pzm[2]), float(pzp[0]), float(pzp[1]), float(pzp[2])])
        colors.extend(color)
        if not hasattr(self, "_debug_draw_once_logged"):
            self._debug_draw_once_logged = True
            print(f"[debug/draw] viewer cross active env_id={env_id} cross_len={cross_len}")
        self.env.gym.add_lines(self.env.viewer, self.env.envs[env_id], 3, verts, colors)

    def _debug_draw_axes(self, origin_world, rot_mat, env_id=None, axis_len=0.10):  # CODEX NEW
        if not self.debug_mode:
            return
        if self.multi_gpu and self.global_rank != 0:
            return
        if not hasattr(self.env, "viewer") or self.env.viewer is None:
            return
        if env_id is None:
            env_id = self._debug_env_id()
        origin = origin_world.detach()
        rot = rot_mat.detach()
        verts = []
        colors = []
        axis_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        for axis_idx in range(3):
            p1 = origin + rot[:, axis_idx] * axis_len
            verts.extend([float(origin[0]), float(origin[1]), float(origin[2]), float(p1[0]), float(p1[1]), float(p1[2])])
            colors.extend(axis_colors[axis_idx])
        self.env.gym.add_lines(self.env.viewer, self.env.envs[env_id], 3, verts, colors)

    def _debug_print_aux_error_only(self, tag, metrics):  # CODEX NEW
        return

    def _debug_print_metrics(self, tag, metrics):  # CODEX NEW
        return

    def _debug_should_run(self):  # CODEX NEW
        if not self.debug_mode:
            return False
        if self.multi_gpu and self.global_rank != 0:
            return False
        return (self.total_steps % self.debug_print_freq) == 0

    def _debug_dependency_tree(self):  # CODEX NEW
        return {
            "level0": {
                "name": "trusted_visual_state",
                "depends_on": [],
                "checks": [
                    "visualize_robot_base_pose7",
                    "visualize_object_center_world",
                ],
            },
            "level1": {
                "name": "geometry_transforms",
                "depends_on": ["level0"],
                "checks": [
                    "world_to_base_transform",
                    "base_to_world_transform",
                    "object_center_in_base_frame",
                ],
            },
            "level2": {
                "name": "semantic_decoders",
                "depends_on": ["level1"],
                "checks": [
                    "visualize_decoded_student_action_target",
                    "visualize_decoded_aux_anchor",
                ],
            },
            "level3": {
                "name": "end_to_end_invariants",
                "depends_on": ["level2"],
                "checks": [
                    "action_roundtrip",
                    "teacher_roundtrip",
                    "live_recode",
                    "hand_tracking_error",
                ],
            },
        }

    def _debug_l0_visualize_state(self, tag):  # CODEX NEW
        if not self.debug_mode:
            return {}
        metrics = {}
        object_ref_world = self.env.states["object_pos"].clone()
        base_pose7 = self.env.states["franka_base_pose7"].clone()
        base_pos_world = base_pose7[:, :3]
        base_rot_mat = quaternion_to_matrix_ig(base_pose7[:, 3:])
        metrics["object_world_norm"] = float(torch.linalg.norm(object_ref_world, dim=-1).mean().item())
        metrics["base_world_norm"] = float(torch.linalg.norm(base_pos_world, dim=-1).mean().item())
        env_id = self._debug_env_id()
        self._debug_draw_cross(object_ref_world[env_id], [0.0, 1.0, 0.0], env_id=env_id, cross_len=0.035, clear_lines=True)
        self._debug_draw_axes(base_pos_world[env_id], base_rot_mat[env_id], env_id=env_id, axis_len=0.50)
        if self._debug_should_run():
            self._debug_print_metrics(f"{tag}/l0", metrics)
        return metrics

    def _debug_l1_check_aux_transport(self, aux_world_src, dst_base_pose7):  # CODEX NEW
        transported = self._world_points_to_base_frame(aux_world_src, dst_base_pose7)
        reconstructed_world = self._base_points_to_world_frame(transported, dst_base_pose7)
        diff = torch.abs(reconstructed_world - aux_world_src)
        return {
            "aux_transport_mean": float(diff.mean().item()),
            "aux_transport_max": float(diff.max().item()),
        }

    def _debug_l1_check_object_ref_base(self, object_ref_world, base_pose7):  # CODEX NEW
        object_ref_base = self._world_points_to_base_frame(object_ref_world, base_pose7)
        reconstructed_world = self._base_points_to_world_frame(object_ref_base, base_pose7)
        diff = torch.abs(reconstructed_world - object_ref_world)
        return {
            "object_ref_base_mean": float(diff.mean().item()),
            "object_ref_base_max": float(diff.max().item()),
        }

    def _debug_l2_visualize_semantics(self, tag, student_actions_chunk=None, anchor_q=None, aux_pred=None, prev_abs_aux=None, aux_anchor_state=None, object_ref_world=None):  # CODEX NEW
        if not self.debug_mode:
            return {}
        point_specs = []
        metrics = {}

        if student_actions_chunk is not None and anchor_q is not None:
            decoded_abs = self.env._decode_student_actions_with_anchor(student_actions_chunk, anchor_q)
            point_specs.append((decoded_abs[0, :3], [1.0, 1.0, 0.0]))
            metrics["decoded_base_target_norm"] = float(torch.linalg.norm(decoded_abs[:, :3], dim=-1).mean().item())

        if aux_pred is not None and prev_abs_aux is not None:
            aux_pred_abs = self._decode_aux_prediction(aux_pred, prev_abs_aux)
            aux_world = self._base_points_to_world_frame(aux_pred_abs[:, 0, :], self.chunk_anchor_base_pose7)
            point_specs.append((aux_world[self._debug_env_id()], [1.0, 0.0, 1.0]))
            metrics["decoded_aux_target_norm"] = float(torch.linalg.norm(aux_pred_abs[:, 0, :], dim=-1).mean().item())

        if aux_anchor_state is not None and hasattr(self, "chunk_anchor_base_pose7"):
            aux_anchor_world = self._base_points_to_world_frame(aux_anchor_state, self.chunk_anchor_base_pose7)
            debug_env_id = self._debug_env_id()
            point_specs.append((aux_anchor_world[debug_env_id], [0.0, 1.0, 1.0]))
            if object_ref_world is not None:
                aux_anchor_err = torch.linalg.norm(aux_anchor_world - object_ref_world, dim=-1)
                metrics[f"aux_anchor_err_env{debug_env_id}"] = float(aux_anchor_err[debug_env_id].item())
            metrics["aux_anchor_world_norm"] = float(torch.linalg.norm(aux_anchor_world, dim=-1).mean().item())

        if len(point_specs) > 0:
            env_id = self._debug_env_id()
            for point_world, color in point_specs:
                self._debug_draw_cross(point_world, color, env_id=env_id, cross_len=0.035, clear_lines=False)
            self._debug_print_aux_error_only(f"{tag}/l2", metrics)
        return metrics

    def _debug_l3_check_action_roundtrip(self, student_actions_chunk, anchor_q):  # CODEX NEW
        decoded_abs = self.env._decode_student_actions_with_anchor(student_actions_chunk, anchor_q)
        reencoded = self.env._encode_abs_targets_to_student_space(decoded_abs, anchor_q)
        diff = torch.abs(reencoded - student_actions_chunk)
        return {
            "action_roundtrip_mean": float(diff.mean().item()),
            "action_roundtrip_max": float(diff.max().item()),
        }

    def _debug_l3_check_teacher_roundtrip(self, teacher_actions_abs, teacher_actions_anchor, anchor_q):  # CODEX NEW
        decoded_abs = self.env._decode_student_actions_with_anchor(teacher_actions_anchor, anchor_q)
        diff = torch.abs(decoded_abs - teacher_actions_abs)
        return {
            "teacher_roundtrip_mean": float(diff.mean().item()),
            "teacher_roundtrip_max": float(diff.max().item()),
        }

    def _debug_l3_check_live_recode(self, intended_abs, live_q):  # CODEX NEW
        recoded = self.env._encode_abs_targets_to_student_space(intended_abs, live_q)
        decoded_abs = self.env._decode_student_actions_with_anchor(recoded, live_q)
        diff = torch.abs(decoded_abs - intended_abs)
        return {
            "live_recode_mean": float(diff.mean().item()),
            "live_recode_max": float(diff.max().item()),
        }

    def _debug_pcd_world_to_base_frame(self, pcd_world, base_pose7):
        base_pos = base_pose7[:, :3]
        base_quat = base_pose7[:, 3:]
        base_rot_mat = quaternion_to_matrix_ig(base_quat)
        rot_global2base = base_rot_mat.transpose(1, 2)
        shifted = pcd_world - base_pos.unsqueeze(1)
        return torch.bmm(shifted, rot_global2base)

    def _debug_compute_inference_actions_abs(self, step_action, q_arm_manip, q_hand, q_arm_vision):
        actions_abs = step_action.clone()
        actions_abs[:3] = step_action[:3]

        if self.env.delta_franka_action:
            actions_abs[3:10] = self.env.unnormalize_robot_joints(
                actions_abs[3:10], robot="franka", delta=True
            ) * self.env.action_scale["franka"] * self.env.dt + q_arm_manip
        else:
            actions_abs[3:10] = self.env.unnormalize_robot_joints(
                actions_abs[3:10], robot="franka", delta=False
            )

        if self.env.delta_leap_action:
            actions_abs[10:26] = self.env.unnormalize_robot_joints(
                actions_abs[10:26], robot="leap", delta=True
            ) * self.env.action_scale["leap"] * self.env.dt + q_hand
        else:
            actions_abs[10:26] = self.env.unnormalize_robot_joints(
                actions_abs[10:26], robot="leap", delta=False
            )

        if self.env.delta_arx_action:
            actions_abs[26:] = self.env.unnormalize_robot_joints(
                actions_abs[26:], robot="arx", delta=True
            ) * self.env.action_scale["arx"] * self.env.dt + q_arm_vision
        else:
            actions_abs[26:] = self.env.unnormalize_robot_joints(
                actions_abs[26:], robot="arx", delta=False
            )

        return torch.max(torch.min(actions_abs, self.env.robot_dof_upper_limits), self.env.robot_dof_lower_limits)

    def _maybe_dump_inference_io_pair(
        self,
        mode,
        full_scene_pcd_t,
        robot_pcd_t,
        q_arm_manip,
        q_hand,
        q_arm_vision,
        obs_input_a0,
        student_actions_chunk,
        aux_output=None,
        obs_input_snapshot=None,
    ):
        if not self.dump_inference_io:
            return
        if self.chunk_size != 1:
            return
        if self.dump_inference_io_saved >= self.dump_inference_io_max:
            return
        if self.multi_gpu and self.global_rank != 0:
            return

        env_id = min(self.dump_inference_io_env, self.env.num_envs - 1)
        if obs_input_snapshot is None:
            obs_input_snapshot = {
                key: value.detach().clone() if torch.is_tensor(value) else value
                for key, value in obs_input_a0.items()
            }
        base_pose7 = self.env.states["franka_base_pose7"].clone()
        full_pcd_world = torch.cat([full_scene_pcd_t, robot_pcd_t], dim=1)
        if self.env.pcd_spec_dict["simulate_depth_cam"]:
            num_full_pcd_points = (
                self.env.pcd_spec_dict["num_static_points"]
                + self.env.pcd_spec_dict["num_robot_points"]
                + self.env.pcd_spec_dict["num_object_points"]
            )
            full_pcd_world, _ = simulate_depth_cam_render_from_pose(
                pcd=full_pcd_world,
                camera_pose=self.env.states["camera_pose7"].clone(),
                num_points=num_full_pcd_points,
            )

        full_pcd_base = self._debug_pcd_world_to_base_frame(full_pcd_world, base_pose7)
        eef_xyz_base = self._world_points_to_base_frame(self.env.states["eef_pos"].clone(), base_pose7)
        student_action = student_actions_chunk[env_id, 0, :32].detach().clone()
        step_action = torch.clamp(student_action, -self.env.clip_actions, self.env.clip_actions)
        actions_abs = self._debug_compute_inference_actions_abs(
            step_action,
            q_arm_manip[env_id].detach().clone(),
            q_hand[env_id].detach().clone(),
            q_arm_vision[env_id].detach().clone(),
        )

        aux_input = None
        aux_anchor_state = None
        aux_pred_abs = None
        if self.has_aux_input:
            aux_tensor = obs_input_snapshot["aux_object_state"]
            if aux_tensor.ndim == 3:
                aux_input = aux_tensor[env_id, 0, :].detach().clone()
            else:
                aux_input = aux_tensor[env_id].detach().clone()
            aux_anchor_state = self.aux_anchor_state[env_id].detach().clone()
            if aux_output is not None:
                aux_pred_abs = self._decode_aux_prediction(
                    aux_output[env_id:env_id + 1],
                    aux_tensor[env_id:env_id + 1],
                )[0, 0, :].detach().clone()

        record = {
            "episode": int(self.episode),
            "total_steps": int(self.total_steps),
            "mode": mode,
            "env_id": int(env_id),
            "full_pcd_frankabase_frame_t": full_pcd_base[env_id].detach().cpu(),
            "eef_xyz_frankabase_frame_t": eef_xyz_base[env_id].detach().cpu(),
            "q_hand": q_hand[env_id].detach().cpu(),
            "q_arm_manip": q_arm_manip[env_id].detach().cpu(),
            "q_arm_vision": q_arm_vision[env_id].detach().cpu(),
            "prev_abs_hand_actions": self.env.abs_actions[env_id, 10:26].detach().cpu(),
            "aux_inputs": aux_input.detach().cpu() if aux_input is not None else None,
            "aux_anchor_state": aux_anchor_state.detach().cpu() if aux_anchor_state is not None else None,
            "expected_q_hand_ctrl_delta": obs_input_snapshot["q_hand_ctrl_delta"][env_id].detach().cpu()
            if "q_hand_ctrl_delta" in obs_input_snapshot else None,
            "obs_input_a0": {
                key: value[env_id:env_id + 1].detach().cpu() if torch.is_tensor(value) else value
                for key, value in obs_input_snapshot.items()
            },
            "expected_model_action_chunk": student_actions_chunk[env_id:env_id + 1].detach().cpu(),
            "expected_model_aux_output": aux_output[env_id:env_id + 1].detach().cpu() if aux_output is not None else None,
            "expected_student_action": student_action.detach().cpu(),
            "expected_step_action": step_action.detach().cpu(),
            "expected_actions_abs": actions_abs.detach().cpu(),
            "expected_aux_pred_abs": aux_pred_abs.detach().cpu() if aux_pred_abs is not None else None,
        }
        self.dump_inference_io_records.append(record)
        self.dump_inference_io_saved += 1
        torch.save(self.dump_inference_io_records, self.dump_inference_io_path)
        if self.dump_inference_io_saved == 1:
            colorprint(f"Dumping inference IO pairs to {self.dump_inference_io_path}", color="yellow")

    def _debug_eval_saved_snapshots(self, student_model):
        if self.debug_replay_snapshot_done:
            return
        if not self.debug_replay_snapshot_path:
            return
        if self.multi_gpu and self.global_rank != 0:
            self.debug_replay_snapshot_done = True
            return

        snapshot_path = Path(self.debug_replay_snapshot_path)
        records = torch.load(snapshot_path, map_location="cpu")
        if not records:
            colorprint(f"No snapshot records found in {snapshot_path}", color="yellow")
            self.debug_replay_snapshot_done = True
            return

        max_records = self.debug_replay_snapshot_max if self.debug_replay_snapshot_max > 0 else len(records)
        records = records[:max_records]
        summary = {}
        student_model.eval()
        with torch.no_grad():
            for idx, record in enumerate(records):
                if "obs_input_a0" not in record:
                    continue
                obs_input = {}
                for key, value in record["obs_input_a0"].items():
                    if torch.is_tensor(value):
                        obs_input[key] = value.to(self.device)
                    else:
                        obs_input[key] = value
                output = student_model(obs_input)

                per_record = {
                    "model_action_chunk": (
                        output["action"].detach().cpu().to(torch.float32),
                        record["expected_model_action_chunk"].to(torch.float32),
                    ),
                }
                if record.get("expected_model_aux_output", None) is not None and "aux" in output:
                    per_record["model_aux_output"] = (
                        output["aux"].detach().cpu().to(torch.float32),
                        record["expected_model_aux_output"].to(torch.float32),
                    )

                print(f"[debug/snapshot_replay] idx={idx} episode={record['episode']} total_steps={record['total_steps']}")
                for name, (actual, expected) in per_record.items():
                    diff = (actual - expected).abs()
                    mean_abs = float(diff.mean().item())
                    max_abs = float(diff.max().item())
                    print(f"[debug/snapshot_replay] {name} mean_abs={mean_abs:.8f} max_abs={max_abs:.8f}")
                    if name not in summary:
                        summary[name] = {"mean_sum": 0.0, "count": 0, "max_abs": 0.0}
                    summary[name]["mean_sum"] += mean_abs
                    summary[name]["count"] += 1
                    summary[name]["max_abs"] = max(summary[name]["max_abs"], max_abs)

        if summary:
            for name, stats in summary.items():
                avg_mean_abs = stats["mean_sum"] / max(stats["count"], 1)
                print(
                    f"[debug/snapshot_replay/summary] {name} "
                    f"avg_mean_abs={avg_mean_abs:.8f} max_abs={stats['max_abs']:.8f}"
                )
        self.debug_replay_snapshot_done = True

    def _debug_dispatch_l1(self, tag, **kwargs):  # CODEX NEW
        metrics = {}
        if "aux_world_src" in kwargs and "dst_base_pose7" in kwargs and kwargs["aux_world_src"] is not None and kwargs["dst_base_pose7"] is not None:
            metrics.update(self._debug_l1_check_aux_transport(kwargs["aux_world_src"], kwargs["dst_base_pose7"]))
        if "object_ref_world" in kwargs and "base_pose7" in kwargs and kwargs["object_ref_world"] is not None and kwargs["base_pose7"] is not None:
            metrics.update(self._debug_l1_check_object_ref_base(kwargs["object_ref_world"], kwargs["base_pose7"]))
        return metrics

    def _debug_dispatch_l2(self, tag, **kwargs):  # CODEX NEW
        return self._debug_l2_visualize_semantics(
            tag,
            student_actions_chunk=kwargs.get("student_actions_chunk", None),
            anchor_q=kwargs.get("anchor_q", None),
            aux_pred=kwargs.get("aux_pred", None),
            prev_abs_aux=kwargs.get("prev_abs_aux", None),
            aux_anchor_state=kwargs.get("aux_anchor_state", None),
            object_ref_world=kwargs.get("object_ref_world", None),
        )

    def _debug_dispatch_l3(self, tag, **kwargs):  # CODEX NEW
        metrics = {}
        if "student_actions_chunk" in kwargs and "anchor_q" in kwargs and kwargs["student_actions_chunk"] is not None and kwargs["anchor_q"] is not None:
            metrics.update(self._debug_l3_check_action_roundtrip(kwargs["student_actions_chunk"], kwargs["anchor_q"]))
        if "teacher_actions_abs" in kwargs and "teacher_actions_anchor" in kwargs and "anchor_q" in kwargs and kwargs["teacher_actions_abs"] is not None and kwargs["teacher_actions_anchor"] is not None and kwargs["anchor_q"] is not None:
            metrics.update(self._debug_l3_check_teacher_roundtrip(
                kwargs["teacher_actions_abs"], kwargs["teacher_actions_anchor"], kwargs["anchor_q"]
            ))
        if "intended_abs" in kwargs and "live_q" in kwargs and kwargs["intended_abs"] is not None and kwargs["live_q"] is not None:
            metrics.update(self._debug_l3_check_live_recode(kwargs["intended_abs"], kwargs["live_q"]))
        return metrics
