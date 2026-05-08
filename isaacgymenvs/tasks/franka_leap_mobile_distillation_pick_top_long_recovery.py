import torch
from isaacgym.torch_utils import *
from curobo.types.math import Pose

from isaacgymenvs.tasks.franka_leap_mobile_distillation_pick_top_long import (
    FrankaLEAPMobileDistillationPickTopLong,
)
from isaacgymenvs.tasks.franka_leap_mobile_distillation_pick_top import (
    FrankaLEAPMobileDistillationPickTop,
)


class FrankaLEAPMobileDistillationPickTopLongRecovery(FrankaLEAPMobileDistillationPickTopLong):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        recovery_cfg = cfg["env"].get("failure_recovery", {})
        self.failure_recovery_enable = bool(recovery_cfg.get("enable", True))
        self.failure_recovery_prob = float(recovery_cfg.get("prob", 0.25))
        self.failure_recovery_eef_rel_object_min = torch.tensor(
            recovery_cfg.get("eef_rel_object_min", [-0.1178, 0.05, 0.0282]),
            dtype=torch.float32,
        )
        self.failure_recovery_eef_rel_object_max = torch.tensor(
            recovery_cfg.get("eef_rel_object_max", [0.2, 0.5, 0.3490]),
            dtype=torch.float32,
        )
        self.failure_recovery_base_rel_eef_min = torch.tensor(
            recovery_cfg.get("base_rel_eef_min", [-0.5523, 0.0161, -0.7124]),
            dtype=torch.float32,
        )
        self.failure_recovery_base_rel_eef_max = torch.tensor(
            recovery_cfg.get("base_rel_eef_max", [-0.3286, 0.3031, -0.1116]),
            dtype=torch.float32,
        )
        self.failure_recovery_base_yaw_noise_deg = float(recovery_cfg.get("base_yaw_noise_deg", 10.0))
        self.failure_recovery_target_quat_right = torch.tensor(
            recovery_cfg.get("target_quat_right", [-0.7071067811865475, 0.0, 0.0, 0.7071067811865476]),
            dtype=torch.float32,
        )
        self.failure_recovery_target_quat_left = torch.tensor(
            recovery_cfg.get("target_quat_left", [0.7071067811865475, 0.0, 0.0, 0.7071067811865476]),
            dtype=torch.float32,
        )
        self.failure_recovery_pre_align_height_offset = float(recovery_cfg.get("pre_align_height_offset", 0.18))
        self.failure_recovery_max_attempts = int(recovery_cfg.get("max_attempts", 32))
        self.failure_recovery_reference_table_height = float(
            recovery_cfg.get("reference_table_surface_height", cfg["env"].get("rl_reference_table_surface_height", 0.175))
        )
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

    def init_data(self, actor_num):
        super().init_data(actor_num)
        self.failure_recovery_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.failure_recovery_stage = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.failure_recovery_prealign_target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.failure_recovery_prealign_target_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.failure_recovery_prealign_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.failure_recovery_target_quat_right = self.failure_recovery_target_quat_right.to(
            device=self.device, dtype=torch.float32
        )
        self.failure_recovery_target_quat_right = self.failure_recovery_target_quat_right / torch.norm(
            self.failure_recovery_target_quat_right
        ).clamp_min(1.0e-8)
        self.failure_recovery_target_quat_left = self.failure_recovery_target_quat_left.to(
            device=self.device, dtype=torch.float32
        )
        self.failure_recovery_target_quat_left = self.failure_recovery_target_quat_left / torch.norm(
            self.failure_recovery_target_quat_left
        ).clamp_min(1.0e-8)

    def _solve_arm_ik_local(self, eef_pose_local):
        eef_pose_local = eef_pose_local.clone().contiguous()
        eef_pos = eef_pose_local[:, :3]
        eef_quat_xyzw = eef_pose_local[:, 3:]
        eef_quat_wxyz = eef_quat_xyzw[:, [3, 0, 1, 2]]
        goal = Pose(eef_pos, eef_quat_wxyz)
        result = self.ik_solver_reset.solve_batch(
            goal_pose=goal,
            retract_config=self.ik_regularization_config[: eef_pose_local.shape[0]],
        )
        return result.solution[:, 0], result.success.bool()

    def _yaw_to_quat_xyzw(self, yaw):
        half_yaw = 0.5 * yaw
        q = torch.zeros((yaw.shape[0], 4), device=yaw.device, dtype=yaw.dtype)
        q[:, 2] = torch.sin(half_yaw)
        q[:, 3] = torch.cos(half_yaw)
        return q

    def _world_pose_to_base_local(self, base_pose, world_pose):
        rel_pos = world_pose[:, :3].clone()
        rel_pos[:, 0] -= base_pose[:, 0]
        rel_pos[:, 1] -= base_pose[:, 1]
        base_yaw = base_pose[:, 2]
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        local_pos = rel_pos.clone()
        local_pos[:, 0] = cos_yaw * rel_pos[:, 0] + sin_yaw * rel_pos[:, 1]
        local_pos[:, 1] = -sin_yaw * rel_pos[:, 0] + cos_yaw * rel_pos[:, 1]

        base_quat = self._yaw_to_quat_xyzw(base_yaw)
        base_quat_inv = base_quat.clone()
        base_quat_inv[:, :3] = -base_quat_inv[:, :3]
        local_quat = quat_mul(base_quat_inv, world_pose[:, 3:7])
        local_quat = local_quat / torch.norm(local_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
        return torch.cat([local_pos, local_quat], dim=-1)

    def _arm_local_pose_to_world(self, base_pose, eef_pose_local):
        base_yaw = base_pose[:, 2]
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        world_pos = eef_pose_local[:, :3].clone()
        x_local = eef_pose_local[:, 0]
        y_local = eef_pose_local[:, 1]
        world_pos[:, 0] = cos_yaw * x_local - sin_yaw * y_local + base_pose[:, 0]
        world_pos[:, 1] = sin_yaw * x_local + cos_yaw * y_local + base_pose[:, 1]

        base_quat = self._yaw_to_quat_xyzw(base_yaw)
        world_quat = quat_mul(base_quat, eef_pose_local[:, 3:7])
        world_quat = world_quat / torch.norm(world_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
        return torch.cat([world_pos, world_quat], dim=-1)

    def _sample_failure_recovery_reset(self, env_ids):
        num_envs = int(env_ids.numel())
        if num_envs == 0:
            return torch.empty((0,), dtype=torch.bool, device=self.device), None

        left_mask = self._sample_side_mask(env_ids)
        rel_min = self.failure_recovery_eef_rel_object_min.to(device=self.device, dtype=self._q.dtype)
        rel_max = self.failure_recovery_eef_rel_object_max.to(device=self.device, dtype=self._q.dtype)
        base_rel_min = self.failure_recovery_base_rel_eef_min.to(device=self.device, dtype=self._q.dtype)
        base_rel_max = self.failure_recovery_base_rel_eef_max.to(device=self.device, dtype=self._q.dtype)
        target_quat_left = self.failure_recovery_target_quat_left.to(device=self.device, dtype=self._q.dtype)
        target_quat_right = self.failure_recovery_target_quat_right.to(device=self.device, dtype=self._q.dtype)
        base_yaw_noise_rad = torch.deg2rad(
            torch.tensor(self.failure_recovery_base_yaw_noise_deg, device=self.device, dtype=self._q.dtype)
        )

        final_joint_state = self._q[env_ids].clone()
        success_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        world_eef_pose_final = torch.zeros((num_envs, 7), dtype=self._q.dtype, device=self.device)
        left_mask_final = left_mask.clone()

        object_center_world = self._object_center_init_state[env_ids].to(dtype=self._q.dtype)

        for _ in range(max(1, self.failure_recovery_max_attempts)):
            pending = (~success_mask).nonzero(as_tuple=False).squeeze(-1)
            if pending.numel() == 0:
                break

            sample_left_mask = left_mask[pending]
            sample_object_center = object_center_world[pending]
            sample_table_height = self.table_surface_height[env_ids[pending]].to(dtype=self._q.dtype)
            table_height_delta = sample_table_height - self.failure_recovery_reference_table_height

            rel = torch.rand((pending.numel(), 3), device=self.device, dtype=self._q.dtype)
            rel = rel_min.unsqueeze(0) + rel * (rel_max - rel_min).unsqueeze(0)
            rel[:, 1] = torch.where(sample_left_mask, rel[:, 1], -rel[:, 1])
            eef_world_pos = sample_object_center + rel

            base_rel = torch.rand((pending.numel(), 3), device=self.device, dtype=self._q.dtype)
            base_rel_z_min = base_rel_min[2] - table_height_delta
            base_rel_z_max = base_rel_max[2] - table_height_delta
            base_rel[:, 0] = base_rel_min[0] + base_rel[:, 0] * (base_rel_max[0] - base_rel_min[0])
            base_rel[:, 1] = base_rel_min[1] + base_rel[:, 1] * (base_rel_max[1] - base_rel_min[1])
            base_rel[:, 2] = base_rel_z_min + base_rel[:, 2] * (base_rel_z_max - base_rel_z_min)
            base_rel[:, 1] = torch.where(sample_left_mask, base_rel[:, 1], -base_rel[:, 1])
            base_yaw = (torch.rand((pending.numel(),), device=self.device, dtype=self._q.dtype) * 2.0 - 1.0) * base_yaw_noise_rad
            base_pose = torch.zeros((pending.numel(), 3), device=self.device, dtype=self._q.dtype)
            base_pose[:, 2] = base_yaw
            cos_yaw = torch.cos(base_yaw)
            sin_yaw = torch.sin(base_yaw)
            base_rel_world_x = cos_yaw * base_rel[:, 0] - sin_yaw * base_rel[:, 1]
            base_rel_world_y = sin_yaw * base_rel[:, 0] + cos_yaw * base_rel[:, 1]
            base_pose[:, 0] = eef_world_pos[:, 0] + base_rel_world_x
            base_pose[:, 1] = eef_world_pos[:, 1] + base_rel_world_y

            side_init_quat = target_quat_right.unsqueeze(0).repeat(pending.numel(), 1)
            if torch.any(sample_left_mask):
                side_init_quat[sample_left_mask] = target_quat_left.unsqueeze(0).repeat(
                    int(sample_left_mask.sum().item()), 1
                )

            desired_dir = sample_object_center - eef_world_pos
            desired_dir_xy = desired_dir.clone()
            desired_dir_xy[:, 2] = 0.0
            desired_dir_xy = desired_dir_xy / torch.norm(desired_dir_xy, dim=-1, keepdim=True).clamp_min(1.0e-8)
            ref_dir_xy = torch.zeros_like(desired_dir_xy)
            ref_dir_xy[:, 1] = torch.where(
                sample_left_mask,
                -torch.ones_like(desired_dir_xy[:, 1]),
                torch.ones_like(desired_dir_xy[:, 1]),
            )
            cross_z = (ref_dir_xy[:, 0] * desired_dir_xy[:, 1] - ref_dir_xy[:, 1] * desired_dir_xy[:, 0]).unsqueeze(-1)
            dot_xy = torch.sum(ref_dir_xy[:, :2] * desired_dir_xy[:, :2], dim=-1, keepdim=True)
            yaw = torch.atan2(cross_z, dot_xy)
            q_align = self._yaw_to_quat_xyzw(yaw.squeeze(-1))
            eef_world_quat = quat_mul(q_align, side_init_quat)
            eef_world_quat = eef_world_quat / torch.norm(eef_world_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)

            eef_world_pose = torch.cat([eef_world_pos, eef_world_quat], dim=-1)
            eef_local_pose = self._world_pose_to_base_local(base_pose, eef_world_pose)
            arm_q, ik_success = self._solve_arm_ik_local(eef_local_pose)

            if not torch.any(ik_success):
                continue

            solved_pending = pending[ik_success]
            solved_base_pose = base_pose[ik_success]
            solved_arm_q = arm_q[ik_success]

            final_joint_state[solved_pending, :3] = solved_base_pose
            final_joint_state[solved_pending, 3:10] = solved_arm_q
            success_mask[solved_pending] = True

            actual_eef_local_pose = self.get_ee_from_joint(solved_arm_q)
            actual_eef_world_pose = self._arm_local_pose_to_world(solved_base_pose, actual_eef_local_pose)
            world_eef_pose_final[solved_pending] = actual_eef_world_pose

            left_mask_final[solved_pending] = sample_left_mask[ik_success]

        if not torch.all(success_mask):
            remaining = (~success_mask).nonzero(as_tuple=False).squeeze(-1)
            unresolved_env_ids = env_ids[remaining].detach().cpu().tolist()
            raise RuntimeError(
                "TopLongRecovery failed to sample valid side-style recovery resets for all selected envs. "
                f"unresolved_env_ids={unresolved_env_ids[:32]} "
                f"count={len(unresolved_env_ids)} "
                f"max_attempts={self.failure_recovery_max_attempts}"
            )

        return success_mask, {
            "joint_state": final_joint_state,
            "world_eef_pose": world_eef_pose_final,
            "side_is_left": left_mask_final,
        }

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return

        super().reset_idx(env_ids)

        self.failure_recovery_mode[env_ids] = False
        self.failure_recovery_stage[env_ids] = 0
        self.failure_recovery_prealign_latched[env_ids] = False

        if (not self.failure_recovery_enable) or self.failure_recovery_prob <= 0.0:
            return

        recovery_sample = torch.rand(env_ids.numel(), device=self.device) < self.failure_recovery_prob
        failure_env_ids = env_ids[recovery_sample]
        if failure_env_ids.numel() == 0:
            return

        success_mask, sampled = self._sample_failure_recovery_reset(failure_env_ids)
        success_env_ids = failure_env_ids
        sampled_joint_state = sampled["joint_state"]
        self.set_robot_joint_state(sampled_joint_state, env_ids=success_env_ids)
        self.side_is_left[success_env_ids] = sampled["side_is_left"]
        self.failure_recovery_mode[success_env_ids] = True
        self.failure_recovery_stage[success_env_ids] = 1
        self.failure_recovery_prealign_latched[success_env_ids] = False
        self.failure_recovery_prealign_target_pos[success_env_ids] = sampled["world_eef_pose"][:, :3]
        self.failure_recovery_prealign_target_quat[success_env_ids] = sampled["world_eef_pose"][:, 3:7]

    def _update_fabric_switching_target(self, object_center_pos):
        super()._update_fabric_switching_target(object_center_pos)

        prealign_mask = self.failure_recovery_mode & (self.failure_recovery_stage == 1)
        if not torch.any(prealign_mask):
            return

        unlatch_ids = (prealign_mask & (~self.failure_recovery_prealign_latched)).nonzero(as_tuple=False).squeeze(-1)
        if unlatch_ids.numel() > 0:
            object_top_z = self.table_surface_height[unlatch_ids] + self.mesh_aabb_extents[unlatch_ids, 2]
            pre_align_pos = self.failure_recovery_prealign_target_pos[unlatch_ids].clone()
            pre_align_pos[:, 2] = torch.maximum(
                pre_align_pos[:, 2],
                object_top_z + self.failure_recovery_pre_align_height_offset,
            )
            self.failure_recovery_prealign_target_pos[unlatch_ids] = pre_align_pos
            base_topdown_quat = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=self._q.dtype,
                device=self.device,
            ).repeat(unlatch_ids.numel(), 1)
            self.failure_recovery_prealign_target_quat[unlatch_ids] = base_topdown_quat
            self.failure_recovery_prealign_latched[unlatch_ids] = True

        self.switching_target_pos[prealign_mask] = self.failure_recovery_prealign_target_pos[prealign_mask]
        self.switching_target_quat[prealign_mask] = self.failure_recovery_prealign_target_quat[prealign_mask]

    def _update_states(self):
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._set_fixed_goal_target(all_env_ids, refresh_quat=False)
        refresh_quat_mask = (~self._target_quat_latched) | (self.progress_buf == 0)
        if bool(torch.any(refresh_quat_mask)):
            refresh_env_ids = refresh_quat_mask.nonzero(as_tuple=False).squeeze(-1)
            self._set_fixed_goal_target(refresh_env_ids, refresh_quat=True)

        FrankaLEAPMobileDistillationPickTop._update_states(self)

        if self.enable_fabric:
            switch_pos_err = torch.norm(self._eef_state[:, :3] - self.switching_target_pos, dim=-1)
            quat_dot = torch.abs(
                torch.sum(normalize(self._eef_state[:, 3:7]) * normalize(self.switching_target_quat), dim=-1)
            ).clamp(max=1.0)
            switch_rot_err_deg = 2.0 * torch.rad2deg(torch.acos(quat_dot))

            prealign_mask = self.failure_recovery_mode & (self.failure_recovery_stage == 1)
            prealign_ready = prealign_mask & (switch_pos_err < self.switch_pos_tol) & (switch_rot_err_deg < self.switch_rot_tol_deg)
            if torch.any(prealign_ready):
                self.failure_recovery_stage[prealign_ready] = 2
                self.fabric_switch_enable[prealign_ready] = True

            approach_mask = (~prealign_mask)
            switch_ready = approach_mask & (switch_pos_err < self.switch_pos_tol) & (switch_rot_err_deg < self.switch_rot_tol_deg)
            self.fabric_switch_enable[switch_ready] = False
            self.fabric_switch_enable[prealign_mask] = True
            self.fabric_switch_enable[self.progress_buf == 0] = True

            self.extras["fabric/switch_pos_err"] = torch.mean(switch_pos_err).item()
            self.extras["fabric/switch_rot_err_deg"] = torch.mean(switch_rot_err_deg).item()
            self.extras["fabric/recovery_prealign_frac"] = torch.mean(prealign_mask.float()).item()

        self._set_fixed_goal_target(all_env_ids, refresh_quat=False)
        self._refresh_target_state_components()

        local_z = torch.zeros((self.num_envs, 3), dtype=self._object_state.dtype, device=self.device)
        local_z[:, 2] = 1.0
        object_z_axis_world = quat_apply(self._object_state[:, 3:7], local_z)
        hand_z_axis_world = quat_apply(self._eef_state[:, 3:7], local_z)

        local_offset = torch.zeros((self.num_envs, 3), dtype=self._object_state.dtype, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[:, 2].to(dtype=self._object_state.dtype) * self.object_grasp_target_z_scale
        object_grasp_target_pos = self._object_state[:, :3] + quat_apply(self._object_state[:, 3:7], local_offset)
        object_grasp_target_to_eef_world = object_grasp_target_pos - self._eef_state[:, :3]

        if self.teacher_use_eef_frame:
            eef_rot_mat_t = quaternion_to_matrix_ig(self._eef_state[:, 3:7]).transpose(1, 2)
            object_grasp_target_to_eef = torch.matmul(
                eef_rot_mat_t,
                object_grasp_target_to_eef_world.unsqueeze(-1),
            ).squeeze(-1)
        else:
            object_grasp_target_to_eef = object_grasp_target_to_eef_world

        eef_table_dist = self._eef_state[:, 2:3] - self.table_surface_height.unsqueeze(-1)
        self.states.update(
            {
                "grasp_side_binary": self.side_is_left.to(dtype=torch.float32).unsqueeze(-1),
                "object_grasp_target_pos": object_grasp_target_pos,
                "object_grasp_target_to_eef": object_grasp_target_to_eef,
                "eef_table_dist": eef_table_dist,
                "object_z_axis_world": object_z_axis_world,
                "hand_z_axis_world": hand_z_axis_world,
            }
        )
