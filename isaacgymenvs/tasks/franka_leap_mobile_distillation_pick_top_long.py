"""
Mobile distillation env for side-table objects with a flat top-down teacher.

This intentionally stays close to the regular top-down mobile distillation env,
but exposes the side-table teacher observation layout and resets objects in the
flat orientation used by the side RL teacher when lie_flat_prob=1.0.
"""

import math

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.franka_leap_mobile_distillation_pick_top import (
    FrankaLEAPMobileDistillationPickTop,
)
from isaacgymenvs.tasks.franka_leap_mobile_distillation_pick_side import (
    FrankaLEAPMobileDistillationPickSide,
)
from isaacgymenvs.utils.pcd_utils import transform_pcds_to_world
from isaacgymenvs.utils.rotation_conversions import A2B_quaternion, matrix_to_rotation_6d, quaternion_to_matrix_ig


class FrankaLEAPMobileDistillationPickTopLong(FrankaLEAPMobileDistillationPickTop):
    VERIFIED_BANK_CATEGORY_NAMES = FrankaLEAPMobileDistillationPickSide.VERIFIED_BANK_CATEGORY_NAMES
    VERIFIED_BANK_AFAR = FrankaLEAPMobileDistillationPickSide.VERIFIED_BANK_AFAR
    VERIFIED_BANK_NEAR_RECOVERY = FrankaLEAPMobileDistillationPickSide.VERIFIED_BANK_NEAR_RECOVERY
    VERIFIED_BANK_FAR_RECOVERY = FrankaLEAPMobileDistillationPickSide.VERIFIED_BANK_FAR_RECOVERY

    _get_verified_teacher_bank_shard_path = FrankaLEAPMobileDistillationPickSide._get_verified_teacher_bank_shard_path
    _verified_teacher_bank_counts_tensor = FrankaLEAPMobileDistillationPickSide._verified_teacher_bank_counts_tensor
    get_verified_teacher_bank_category_counts_tensor = (
        FrankaLEAPMobileDistillationPickSide.get_verified_teacher_bank_category_counts_tensor
    )
    get_verified_teacher_bank_category_correct_counts_tensor = (
        FrankaLEAPMobileDistillationPickSide.get_verified_teacher_bank_category_correct_counts_tensor
    )
    _get_activation_snapshot_bank_shard_path = (
        FrankaLEAPMobileDistillationPickSide._get_activation_snapshot_bank_shard_path
    )
    _init_verified_teacher_bank_storage = FrankaLEAPMobileDistillationPickSide._init_verified_teacher_bank_storage
    _init_activation_snapshot_bank_storage = (
        FrankaLEAPMobileDistillationPickSide._init_activation_snapshot_bank_storage
    )
    get_activation_snapshot_bank_count_summary = (
        FrankaLEAPMobileDistillationPickSide.get_activation_snapshot_bank_count_summary
    )
    save_activation_snapshot_bank_hdf5 = FrankaLEAPMobileDistillationPickSide.save_activation_snapshot_bank_hdf5
    record_activation_snapshot_bank_entries = (
        FrankaLEAPMobileDistillationPickSide.record_activation_snapshot_bank_entries
    )
    configure_verified_teacher_bank_collection = (
        FrankaLEAPMobileDistillationPickSide.configure_verified_teacher_bank_collection
    )
    _verified_teacher_bank_category_full = (
        FrankaLEAPMobileDistillationPickSide._verified_teacher_bank_category_full
    )
    get_verified_teacher_bank_count_summary = (
        FrankaLEAPMobileDistillationPickSide.get_verified_teacher_bank_count_summary
    )
    save_verified_teacher_bank_hdf5 = FrankaLEAPMobileDistillationPickSide.save_verified_teacher_bank_hdf5
    load_verified_teacher_bank_hdf5 = FrankaLEAPMobileDistillationPickSide.load_verified_teacher_bank_hdf5
    _snapshot_verified_teacher_episode_starts = (
        FrankaLEAPMobileDistillationPickSide._snapshot_verified_teacher_episode_starts
    )
    _flush_pending_verified_teacher_episode_starts = (
        FrankaLEAPMobileDistillationPickSide._flush_pending_verified_teacher_episode_starts
    )
    record_verified_teacher_episode_outcomes = (
        FrankaLEAPMobileDistillationPickSide.record_verified_teacher_episode_outcomes
    )
    _choose_verified_teacher_bank_categories = (
        FrankaLEAPMobileDistillationPickSide._choose_verified_teacher_bank_categories
    )
    _sample_verified_teacher_bank_entries = (
        FrankaLEAPMobileDistillationPickSide._sample_verified_teacher_bank_entries
    )
    _filter_recovery_robot_object_penetration = (
        FrankaLEAPMobileDistillationPickSide._filter_recovery_robot_object_penetration
    )
    _get_franka_base_pose7_from_mobile_base_pose = (
        FrankaLEAPMobileDistillationPickSide._get_franka_base_pose7_from_mobile_base_pose
    )
    _solve_reset_arm_ik = FrankaLEAPMobileDistillationPickSide._solve_reset_arm_ik
    _apply_recovery_local_roll_pitch_noise = (
        FrankaLEAPMobileDistillationPickSide._apply_recovery_local_roll_pitch_noise
    )

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.object_grasp_target_z_scale = float(cfg["env"]["object_settings"]["object_grasp_target_z_scale"])
        self.lie_flat_prob = float(cfg["env"]["object_settings"].get("lie_flat_prob", 1.0))
        self.preserve_orientation_on_teleport = bool(
            cfg["env"].get("object_teleport", {}).get("preserve_orientation", True)
        )
        self._flat_object_quat_base_values = [0.0, 0.70710678, 0.0, 0.70710678]
        reference_table_height = float(cfg["env"].get("rl_reference_table_surface_height", 0.175))
        self._fixed_goal_z_offset_from_table = float(cfg["reward"]["params"]["target_pos"][2]) - reference_table_height
        self.post_lift_target_pos_base = torch.tensor(
            cfg["reward"]["params"].get(
                "target_pos_after_lift_base",
                [0.7, 0.0, float(cfg["reward"]["params"]["target_pos"][2])],
            ),
            dtype=torch.float32,
        )
        self.post_lift_target_z_from_table = float(
            cfg["reward"]["params"].get(
                "target_pos_after_lift_z_from_table",
                float(self.post_lift_target_pos_base[2].item()),
            )
        )
        self.side_mode = str(cfg["env"].get("eef_init", {}).get("side_mode", "both"))
        robot_init_cfg = cfg["env"]["robot_init"]
        self.switch_pos_tol = float(robot_init_cfg.get("switch_pos_tol", robot_init_cfg.get("switch_tol", 0.2)))
        self.switch_rot_tol_deg = float(robot_init_cfg.get("switch_rot_tol_deg", 180.0))
        self.switch_use_orientation_target = bool(robot_init_cfg.get("switch_use_orientation_target", False))
        self.use_goal_orientation_target = bool(cfg["reward"]["params"].get("use_goal_orientation_target", False))
        eef_init_cfg = cfg["env"].get("eef_init", {})
        self._reset_yaw_noise_deg = float(eef_init_cfg.get("yaw_noise_deg", 45.0))
        self._reset_yaw_noise_rad = float(math.radians(self._reset_yaw_noise_deg))
        self._reset_pitch_roll_noise_deg = float(eef_init_cfg.get("pitch_roll_noise_deg", 45.0))
        self._reset_pitch_roll_noise_rad = float(math.radians(self._reset_pitch_roll_noise_deg))
        self._reset_hand_joint_noise_deg = float(eef_init_cfg.get("hand_joint_noise_deg", 20.0))
        self._topdown_afar_rel_object_min_cfg = eef_init_cfg.get(
            "flat_topdown_rel_object_min",
            [-0.4, -0.4, 0.10],
        )
        self._topdown_afar_rel_object_max_cfg = eef_init_cfg.get(
            "flat_topdown_rel_object_max",
            [0.4, 0.4, 0.45],
        )
        self._topdown_near_recovery_rel_object_min_cfg = eef_init_cfg.get(
            "flat_topdown_near_recovery_rel_object_min",
            [-0.16, -0.16, 0.10],
        )
        self._topdown_near_recovery_rel_object_max_cfg = eef_init_cfg.get(
            "flat_topdown_near_recovery_rel_object_max",
            [0.16, 0.16, 0.28],
        )
        self._topdown_far_recovery_rel_object_min_cfg = eef_init_cfg.get(
            "flat_topdown_far_recovery_rel_object_min",
            [-0.32, -0.32, 0.10],
        )
        self._topdown_far_recovery_rel_object_max_cfg = eef_init_cfg.get(
            "flat_topdown_far_recovery_rel_object_max",
            [0.32, 0.32, 0.40],
        )
        self._topdown_recovery_min_palm_object_dist = float(
            eef_init_cfg.get("flat_topdown_min_palm_object_dist", 0.05)
        )
        self._topdown_recovery_max_resample_attempts = int(
            eef_init_cfg.get("flat_topdown_max_resample_attempts", 32)
        )
        teacher_state_bank_cfg = cfg.get("dagger", {}).get("teacher_state_bank", {})
        self._teacher_state_bank_collect_requested = bool(
            teacher_state_bank_cfg.get("collect_before_train", False)
            or teacher_state_bank_cfg.get("collect_only", False)
        )
        verified_bank_cfg = cfg["env"].get("verified_teacher_bank", {})
        self._verified_teacher_bank_enable = bool(verified_bank_cfg.get("enable", False))
        self._verified_teacher_bank_hdf5_path = str(verified_bank_cfg.get("hdf5_path", "") or "")
        self._verified_teacher_bank_capacity_per_env = int(verified_bank_cfg.get("capacity_per_env", 0))
        self._verified_teacher_bank_correct_region_margin = float(
            verified_bank_cfg.get("correct_region_margin", 0.03)
        )
        self._verified_teacher_bank_collection_region_filter = str(
            verified_bank_cfg.get("collection_region_filter", "all")
        )
        if self._verified_teacher_bank_collection_region_filter not in ("all", "correct_only", "wrong_only"):
            raise ValueError(
                "verified_teacher_bank.collection_region_filter must be one of "
                "['all', 'correct_only', 'wrong_only']"
            )
        sampling_probs_cfg = verified_bank_cfg.get(
            "sampling_probs",
            {"afar": 1.0 / 3.0, "near_recovery": 1.0 / 3.0, "far_recovery": 1.0 / 3.0},
        )
        self._verified_teacher_bank_sampling_probs_cfg = sampling_probs_cfg
        self._verified_teacher_bank_sampling_probs = torch.tensor(
            [
                float(sampling_probs_cfg.get("afar", 1.0 / 3.0)),
                float(sampling_probs_cfg.get("near_recovery", 1.0 / 3.0)),
                float(sampling_probs_cfg.get("far_recovery", 1.0 / 3.0)),
            ],
            dtype=torch.float32,
        )
        activation_snapshot_bank_cfg = cfg["env"].get("activation_snapshot_bank", {})
        self._activation_snapshot_bank_enable = bool(activation_snapshot_bank_cfg.get("enable", False))
        self._activation_snapshot_bank_hdf5_path = str(activation_snapshot_bank_cfg.get("hdf5_path", "") or "")
        self._activation_snapshot_bank_capacity_per_env = int(
            activation_snapshot_bank_cfg.get("capacity_per_env", 0)
        )
        if self._verified_teacher_bank_enable:
            if not self._verified_teacher_bank_hdf5_path:
                raise ValueError("verified_teacher_bank.enable=True but hdf5_path is empty")
            if self._verified_teacher_bank_capacity_per_env <= 0:
                raise ValueError("verified_teacher_bank.enable=True requires capacity_per_env > 0")
        if self._activation_snapshot_bank_enable:
            if not self._activation_snapshot_bank_hdf5_path:
                raise ValueError("activation_snapshot_bank.enable=True but hdf5_path is empty")
            if self._activation_snapshot_bank_capacity_per_env <= 0:
                raise ValueError(
                    "activation_snapshot_bank.enable=True requires capacity_per_env > 0"
                )
        self._verified_teacher_bank_loaded = False
        self._verified_teacher_bank_collection_enabled = False
        self._verified_teacher_bank_collection_category = -1
        self._palm_center_from_link7_local = torch.tensor([0.0, 0.0, 0.115], dtype=torch.float32)
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

    def _adjust_object_reset_xy_bounds(self, env_ids, xy_min, xy_max):
        xy_min = xy_min.clone()
        xy_max = xy_max.clone()
        robot_side_x_margin = float(self.cfg["env"]["object_settings"].get("robot_side_x_margin", 0.0))
        if robot_side_x_margin <= 0.0:
            return xy_min, xy_max

        table_front_x = self.table_pos[env_ids, 0] - 0.5 * self.table_size[env_ids, 0]
        object_half_x = 0.5 * self.mesh_aabb_extents[env_ids, 0]
        safe_x_min = table_front_x + object_half_x + robot_side_x_margin
        xy_min[:, 0] = torch.minimum(torch.maximum(xy_min[:, 0], safe_x_min), xy_max[:, 0])
        return xy_min, xy_max

    def _update_fabric_switching_target(self, object_center_pos):
        # Top-down fabric phase: move above the object center and hand over to the
        # teacher when the EEF reaches this approach distance.
        self.switching_target_pos = object_center_pos + self.switch_pos_offset
        if not self.switch_use_orientation_target:
            self.switching_target_quat = normalize(self._eef_state[:, 3:7])
            return

        base_topdown_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * self.num_envs,
            dtype=object_center_pos.dtype,
            device=self.device,
        )
        object_z_local = torch.zeros((self.num_envs, 3), dtype=object_center_pos.dtype, device=self.device)
        object_z_local[:, 2] = 1.0
        object_long_axis_world = quat_apply(self._object_state[:, 3:7], object_z_local)
        object_long_axis_xy = object_long_axis_world.clone()
        object_long_axis_xy[:, 2] = 0.0
        object_long_axis_xy_norm = torch.norm(object_long_axis_xy[:, :2], dim=-1, keepdim=True)
        valid_axis = object_long_axis_xy_norm.squeeze(-1) > 1.0e-6
        object_long_axis_xy[:, :2] = object_long_axis_xy[:, :2] / object_long_axis_xy_norm.clamp_min(1.0e-8)

        # Align the same local hand grasp axis used by the side RL env, instead of
        # hard-coding the hand local x-axis. For the default top-down setup this is
        # local +y, which fixes the 90-degree mismatch seen in fabric approach.
        hand_grasp_dir_local = self.hand_grasp_dir_local.to(dtype=object_center_pos.dtype).unsqueeze(0).repeat(self.num_envs, 1)
        hand_face_dir_local = self.hand_face_dir_local.to(dtype=object_center_pos.dtype).unsqueeze(0).repeat(self.num_envs, 1)
        ref_dir_xy = quat_apply(base_topdown_quat, hand_grasp_dir_local)
        ref_dir_xy[:, 2] = 0.0
        ref_dir_xy = ref_dir_xy / torch.norm(ref_dir_xy[:, :2], dim=-1, keepdim=True).clamp_min(1.0e-8)

        def align_grasp_axis(target_axis_xy):
            cross_z = (
                ref_dir_xy[:, 0] * target_axis_xy[:, 1] - ref_dir_xy[:, 1] * target_axis_xy[:, 0]
            ).unsqueeze(-1)
            dot_xy = torch.sum(ref_dir_xy[:, :2] * target_axis_xy[:, :2], dim=-1, keepdim=True)
            yaw = torch.atan2(cross_z, dot_xy)
            half_yaw = 0.5 * yaw
            q_align = torch.zeros((self.num_envs, 4), dtype=object_center_pos.dtype, device=self.device)
            q_align[:, 2:3] = torch.sin(half_yaw)
            q_align[:, 3:4] = torch.cos(half_yaw)
            aligned_quat = quat_mul(q_align, base_topdown_quat)
            return aligned_quat / torch.norm(aligned_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)

        aligned_topdown_quat_a = align_grasp_axis(object_long_axis_xy)
        aligned_topdown_quat_b = align_grasp_axis(-object_long_axis_xy)

        # The grasp-axis alignment has a 180-degree ambiguity. Resolve it using
        # the in-plane hand-facing direction, but against a fixed world-side
        # region instead of the instantaneous base position. For this task the
        # base always spawns from one side of the table, so choose the candidate
        # whose facing vector points more into that absolute half-plane.
        face_vec_a = quat_apply(aligned_topdown_quat_a, hand_face_dir_local)
        face_vec_b = quat_apply(aligned_topdown_quat_b, hand_face_dir_local)
        face_vec_a[:, 2] = 0.0
        face_vec_b[:, 2] = 0.0
        face_vec_a = face_vec_a / torch.norm(face_vec_a[:, :2], dim=-1, keepdim=True).clamp_min(1.0e-8)
        face_vec_b = face_vec_b / torch.norm(face_vec_b[:, :2], dim=-1, keepdim=True).clamp_min(1.0e-8)
        desired_region_sign = self.base_spawn_x_region_sign.to(dtype=object_center_pos.dtype)
        face_score_a = desired_region_sign * face_vec_a[:, 0]
        face_score_b = desired_region_sign * face_vec_b[:, 0]
        choose_b = valid_axis & (face_score_b > face_score_a)
        aligned_topdown_quat = torch.where(
            choose_b.unsqueeze(-1),
            aligned_topdown_quat_b,
            aligned_topdown_quat_a,
        )
        self.switching_target_quat = torch.where(
            valid_axis.unsqueeze(-1),
            aligned_topdown_quat,
            base_topdown_quat,
        )

    def init_data(self, actor_num):
        super().init_data(actor_num)
        hand_grasp_dir_local = torch.tensor(
            self.cfg["reward"]["params"].get("hand_grasp_dir_local", [0.0, 1.0, 0.0]),
            device=self.device,
            dtype=torch.float32,
        )
        self.hand_grasp_dir_local = hand_grasp_dir_local / torch.norm(hand_grasp_dir_local).clamp_min(1.0e-8)
        hand_face_dir_local = torch.tensor(
            self.cfg["reward"]["params"].get("hand_face_dir_local", [-1.0, 0.0, 0.0]),
            device=self.device,
            dtype=torch.float32,
        )
        self.hand_face_dir_local = hand_face_dir_local / torch.norm(hand_face_dir_local).clamp_min(1.0e-8)
        base_init_range = torch.tensor(
            self.cfg["env"]["robot_init"]["base_init_range"],
            device=self.device,
            dtype=torch.float32,
        )
        base_init_center_x = 0.5 * (base_init_range[0, 0] + base_init_range[1, 0])
        base_spawn_x_region_sign = -1.0 if float(base_init_center_x.item()) < 0.0 else 1.0
        self.base_spawn_x_region_sign = torch.tensor(base_spawn_x_region_sign, device=self.device, dtype=torch.float32)
        self.side_is_left = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_quat_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.canonical_flat_hand_config = torch.tensor(
            [
                0.0000, 0.0000, 0.0000, 0.0000,
                -0.0000, 0.0000, 1.0000, 0.5700,
                0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._reset_hand_joint_noise_rad = torch.full(
            (16,),
            float(math.radians(self._reset_hand_joint_noise_deg)),
            dtype=torch.float32,
            device=self.device,
        )
        self.current_episode_verified_bank_category = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.per_verified_bank_episode_counts = torch.zeros(
            (len(self.VERIFIED_BANK_CATEGORY_NAMES),), device=self.device, dtype=torch.long
        )
        self.per_verified_bank_success_counts = torch.zeros(
            (len(self.VERIFIED_BANK_CATEGORY_NAMES),), device=self.device, dtype=torch.long
        )
        self.post_lift_target_active = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._init_verified_teacher_bank_storage()
        if self._activation_snapshot_bank_enable:
            self._init_activation_snapshot_bank_storage()

    def _sample_side_mask(self, env_ids):
        if self.side_mode == "left":
            return torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
        if self.side_mode == "right":
            return torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        if self.side_mode == "both":
            return torch.rand(env_ids.numel(), device=self.device) < 0.5
        raise ValueError(f"Unsupported eef_init.side_mode={self.side_mode}. Expected one of: left, right, both.")

    def _table_center_for_target_quat(self, env_ids):
        table_center = torch.zeros((env_ids.numel(), 3), dtype=self._eef_state.dtype, device=self.device)
        table_center[:, :2] = self.table_pos[env_ids, :2].to(dtype=self._eef_state.dtype)
        table_center[:, 2] = self.table_surface_height[env_ids].to(dtype=self._eef_state.dtype)
        return table_center

    def _compute_topdown_target_quat(self, env_ids):
        eef_pos = self._eef_state[env_ids, :3]
        return self._compute_topdown_target_quat_from_palm_pos(env_ids, eef_pos)

    def _compute_topdown_target_quat_from_palm_pos(self, env_ids, palm_pos):
        table_center = self._table_center_for_target_quat(env_ids)
        target_quat = A2B_quaternion(palm_pos, table_center, max_angle_deg=20, right_axis="y").to(
            device=self.device,
            dtype=palm_pos.dtype,
        )
        flip_idx = palm_pos[:, 0] < table_center[:, 0]
        if bool(torch.any(flip_idx)):
            rot_local_z_180 = torch.tensor(
                [[0.0, 0.0, 1.0, 0.0]],
                dtype=palm_pos.dtype,
                device=self.device,
            ).repeat(int(flip_idx.sum().item()), 1)
            target_quat[flip_idx] = quat_mul(target_quat[flip_idx], rot_local_z_180)
        return target_quat / torch.norm(target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _compute_correct_region_mask(self, eef_pos, object_center_pos, side_is_left):
        return torch.ones((eef_pos.shape[0],), device=eef_pos.device, dtype=torch.bool)

    def _verified_teacher_bank_collection_region_match_bool(self, is_correct_region):
        region_filter = self._verified_teacher_bank_collection_region_filter
        if region_filter == "wrong_only":
            return False
        return True

    def _get_clipped_topdown_recovery_palm_rel_object_bounds(
        self,
        object_center,
        table_xy_min,
        table_xy_max,
        dtype,
        rel_object_min_cfg,
        rel_object_max_cfg,
    ):
        rel_min = torch.tensor(rel_object_min_cfg, device=self.device, dtype=dtype).unsqueeze(0).repeat(
            object_center.shape[0], 1
        )
        rel_max = torch.tensor(rel_object_max_cfg, device=self.device, dtype=dtype).unsqueeze(0).repeat(
            object_center.shape[0], 1
        )
        rel_min[:, :2] = torch.maximum(rel_min[:, :2], table_xy_min - object_center[:, :2])
        rel_max[:, :2] = torch.minimum(rel_max[:, :2], table_xy_max - object_center[:, :2])
        valid_xy = torch.all(rel_min[:, :2] <= rel_max[:, :2], dim=-1)
        return rel_min, rel_max, valid_xy

    def _apply_topdown_orientation_noise(self, nominal_target_quat):
        if nominal_target_quat.numel() == 0:
            return nominal_target_quat
        target_quat = nominal_target_quat
        yaw_noise_rad = float(self._reset_yaw_noise_rad)
        if yaw_noise_rad > 0.0:
            batch = int(target_quat.shape[0])
            yaw_axis = torch.zeros((batch, 3), device=self.device, dtype=target_quat.dtype)
            yaw_axis[:, 2] = 1.0
            yaw_angle = (torch.rand((batch,), device=self.device, dtype=target_quat.dtype) * 2.0 - 1.0) * yaw_noise_rad
            q_yaw_noise = quat_from_angle_axis(yaw_angle, yaw_axis)
            target_quat = quat_mul(q_yaw_noise, target_quat)
            target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
        target_quat = self._apply_recovery_local_roll_pitch_noise(target_quat)
        return target_quat / torch.norm(target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _sample_topdown_joint_config(self, env_ids, rel_object_min_cfg, rel_object_max_cfg, log_prefix):
        if env_ids.numel() == 0:
            return (
                torch.empty((0, self.num_dofs), device=self.device, dtype=self._q.dtype),
                torch.empty((0, 3), device=self.device, dtype=self._q.dtype),
                torch.empty((0, 4), device=self.device, dtype=self._q.dtype),
            )

        dtype = self._q.dtype
        device = self.device
        num_recovery = int(env_ids.numel())
        base_init_range = torch.tensor(
            self.cfg["env"]["robot_init"]["base_init_range"],
            device=device,
            dtype=dtype,
        )
        object_center = self._object_center_init_state[env_ids, :3].to(dtype=dtype).clone()
        table_center_xy = self.table_pos[env_ids, :2].to(dtype=dtype)
        table_half_xy = 0.5 * self.table_size[env_ids, :2].to(dtype=dtype)
        table_xy_min = table_center_xy - table_half_xy
        table_xy_max = table_center_xy + table_half_xy

        solved_mask = torch.zeros(num_recovery, device=device, dtype=torch.bool)
        solved_joint_config = self.canonical_joint_config[env_ids].clone()
        solved_palm_pos = torch.zeros((num_recovery, 3), device=device, dtype=dtype)
        solved_target_quat = torch.zeros((num_recovery, 4), device=device, dtype=dtype)
        solved_target_quat[:, 3] = 1.0

        palm_offset_local = self._palm_center_from_link7_local.to(device=device, dtype=dtype)
        hand_joint_noise = self._reset_hand_joint_noise_rad.to(device=device, dtype=dtype)
        warn_interval = max(1, int(self._topdown_recovery_max_resample_attempts))
        attempt_idx = 0
        while True:
            unresolved = (~solved_mask).nonzero(as_tuple=False).squeeze(-1)
            if unresolved.numel() == 0:
                break
            attempt_idx += 1
            if attempt_idx % warn_interval == 0:
                print(
                    f"[{log_prefix}] still sampling topdown reset "
                    f"unsolved={int(unresolved.numel())}/{num_recovery} "
                    f"round={attempt_idx}"
                )

            count = int(unresolved.numel())
            base_pose = base_init_range[0].unsqueeze(0) + torch.rand((count, 3), device=device, dtype=dtype) * (
                base_init_range[1] - base_init_range[0]
            ).unsqueeze(0)
            palm_rel_min, palm_rel_max, valid_xy = self._get_clipped_topdown_recovery_palm_rel_object_bounds(
                object_center[unresolved],
                table_xy_min[unresolved],
                table_xy_max[unresolved],
                dtype,
                rel_object_min_cfg,
                rel_object_max_cfg,
            )
            if not bool(torch.any(valid_xy)):
                continue

            palm_rel_object = palm_rel_min + torch.rand((count, 3), device=device, dtype=dtype) * (
                palm_rel_max - palm_rel_min
            )
            palm_pos = object_center[unresolved] + palm_rel_object
            palm_object_dist = torch.norm(palm_rel_object, dim=-1)
            valid_scene = (
                valid_xy
                & (palm_pos[:, 0] >= table_xy_min[unresolved, 0])
                & (palm_pos[:, 0] <= table_xy_max[unresolved, 0])
                & (palm_pos[:, 1] >= table_xy_min[unresolved, 1])
                & (palm_pos[:, 1] <= table_xy_max[unresolved, 1])
                & (palm_object_dist >= self._topdown_recovery_min_palm_object_dist)
            )
            if not bool(torch.any(valid_scene)):
                continue

            valid_idx = valid_scene.nonzero(as_tuple=False).squeeze(-1)
            valid_unresolved = unresolved[valid_idx]
            valid_palm_pos_world = palm_pos[valid_idx]
            nominal_target_quat = self._compute_topdown_target_quat_from_palm_pos(
                env_ids[valid_unresolved],
                valid_palm_pos_world,
            )
            ik_target_quat = self._apply_topdown_orientation_noise(nominal_target_quat)

            franka_base_pose7 = self._get_franka_base_pose7_from_mobile_base_pose(base_pose[valid_idx], dtype)
            franka_base_pos_world = franka_base_pose7[:, :3]
            franka_base_quat_world = franka_base_pose7[:, 3:7]
            inv_franka_base_quat = quat_conjugate(franka_base_quat_world)
            palm_pos_local = quat_apply(inv_franka_base_quat, valid_palm_pos_world - franka_base_pos_world)
            target_quat_local = quat_mul(inv_franka_base_quat, ik_target_quat)
            target_quat_local = target_quat_local / torch.norm(target_quat_local, dim=-1, keepdim=True).clamp_min(
                1.0e-8
            )
            link7_pos_local = palm_pos_local - quat_apply(
                target_quat_local,
                palm_offset_local.unsqueeze(0).repeat(valid_idx.numel(), 1),
            )
            eef_pose = torch.cat([link7_pos_local, target_quat_local], dim=-1)
            arm_q, success = self._solve_reset_arm_ik(eef_pose)
            if not bool(torch.any(success)):
                continue

            success_unresolved = valid_unresolved[success]
            success_joint_config = solved_joint_config[success_unresolved].clone()
            success_joint_config[:, :3] = base_pose[valid_idx][success]
            success_joint_config[:, 3:10] = arm_q[success]
            sampled_hand = self.canonical_flat_hand_config.unsqueeze(0).repeat(int(success.sum().item()), 1).to(
                dtype=dtype
            )
            sampled_hand = sampled_hand + (torch.rand_like(sampled_hand) * 2.0 - 1.0) * hand_joint_noise.unsqueeze(0)
            sampled_hand = tensor_clamp(
                sampled_hand,
                self.robot_dof_lower_limits[10:26].unsqueeze(0).repeat(int(success.sum().item()), 1),
                self.robot_dof_upper_limits[10:26].unsqueeze(0).repeat(int(success.sum().item()), 1),
            )
            success_joint_config[:, 10:26] = sampled_hand
            no_penetration = self._filter_recovery_robot_object_penetration(
                success_joint_config,
                object_center[valid_idx][success],
                env_ids[success_unresolved],
            )
            if not bool(torch.any(no_penetration)):
                continue

            solved_valid = success_unresolved[no_penetration]
            solved_joint_config[solved_valid] = success_joint_config[no_penetration]
            solved_palm_pos[solved_valid] = valid_palm_pos_world[success][no_penetration]
            solved_target_quat[solved_valid] = nominal_target_quat[success][no_penetration]
            solved_mask[solved_valid] = True

        return solved_joint_config, solved_palm_pos, solved_target_quat

    def _sample_topdown_afar_joint_config(self, env_ids):
        return self._sample_topdown_joint_config(
            env_ids,
            self._topdown_afar_rel_object_min_cfg,
            self._topdown_afar_rel_object_max_cfg,
            "TopLongAfar",
        )

    def _sample_topdown_recovery_joint_config(self, env_ids, force_far=False):
        rel_object_min_cfg = (
            self._topdown_far_recovery_rel_object_min_cfg
            if force_far
            else self._topdown_near_recovery_rel_object_min_cfg
        )
        rel_object_max_cfg = (
            self._topdown_far_recovery_rel_object_max_cfg
            if force_far
            else self._topdown_near_recovery_rel_object_max_cfg
        )
        log_prefix = "TopLongFarRecovery" if force_far else "TopLongNearRecovery"
        return self._sample_topdown_joint_config(
            env_ids,
            rel_object_min_cfg,
            rel_object_max_cfg,
            log_prefix,
        )

    def _apply_verified_teacher_bank_reset(self, env_ids):
        category_ids = self._choose_verified_teacher_bank_categories(env_ids)
        (
            sampled_env_ids,
            sampled_category_ids,
            joint_config,
            object_center_world,
            object_quat_world,
            side_is_left,
        ) = self._sample_verified_teacher_bank_entries(env_ids, category_ids)
        if sampled_env_ids.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
            )
        self._apply_object_center_state(sampled_env_ids, object_center_world, object_quat_world)
        self.set_robot_joint_state(joint_config, env_ids=sampled_env_ids)
        if self.enable_fabric:
            self.fabric_q[sampled_env_ids, :10] = joint_config[:, :10]
            self.fabric_q[sampled_env_ids, 10:] = joint_config[:, 26:]
            self.fabric_qd[sampled_env_ids, :] = 0.0
            self.fabric_qdd[sampled_env_ids, :] = 0.0
        self.side_is_left[sampled_env_ids] = side_is_left
        self.reward_settings["object_init_height"][sampled_env_ids] = object_center_world[:, 2]
        self._target_quat_latched[sampled_env_ids] = False
        return sampled_env_ids, sampled_category_ids

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return

        super().reset_idx(env_ids)
        self.reward_settings["object_init_height"][env_ids] = self._object_center_init_state[env_ids, 2]
        collection_category = int(self._verified_teacher_bank_collection_category)
        collection_enabled = bool(self._verified_teacher_bank_collection_enabled) and (collection_category >= 0)
        use_verified_teacher_bank = (
            self._verified_teacher_bank_enable
            and self._verified_teacher_bank_loaded
            and (not collection_enabled)
        )

        self.current_episode_verified_bank_category[env_ids] = -1
        if use_verified_teacher_bank:
            sampled_env_ids, sampled_category_ids = self._apply_verified_teacher_bank_reset(env_ids)
            if sampled_env_ids.numel() > 0:
                self.current_episode_verified_bank_category[sampled_env_ids] = sampled_category_ids
        else:
            should_apply_afar = (not collection_enabled) or (collection_category == self.VERIFIED_BANK_AFAR)
            if should_apply_afar:
                afar_joint_config, _afar_palm_pos, _afar_target_quat = self._sample_topdown_afar_joint_config(env_ids)
                self.set_robot_joint_state(afar_joint_config, env_ids=env_ids)
                if self.enable_fabric:
                    self.fabric_q[env_ids, :10] = afar_joint_config[:, :10]
                    self.fabric_q[env_ids, 10:] = afar_joint_config[:, 26:]
                    self.fabric_qd[env_ids, :] = 0.0
                    self.fabric_qdd[env_ids, :] = 0.0
            if collection_enabled and collection_category in (
                self.VERIFIED_BANK_NEAR_RECOVERY,
                self.VERIFIED_BANK_FAR_RECOVERY,
            ):
                recovery_joint_config, _recovery_palm_pos, _recovery_target_quat = self._sample_topdown_recovery_joint_config(
                    env_ids,
                    force_far=bool(collection_category == self.VERIFIED_BANK_FAR_RECOVERY),
                )
                self.set_robot_joint_state(recovery_joint_config, env_ids=env_ids)
                if self.enable_fabric:
                    self.fabric_q[env_ids, :10] = recovery_joint_config[:, :10]
                    self.fabric_q[env_ids, 10:] = recovery_joint_config[:, 26:]
                    self.fabric_qd[env_ids, :] = 0.0
                    self.fabric_qdd[env_ids, :] = 0.0
            if collection_enabled:
                self.current_episode_verified_bank_category[env_ids] = int(collection_category)

        if hasattr(self, "_verified_teacher_bank_snapshot_pending"):
            self._verified_teacher_bank_snapshot_pending[env_ids] = False
            self._verified_teacher_bank_snapshot_pending_category[env_ids] = -1
            if collection_enabled:
                self._verified_teacher_bank_snapshot_pending[env_ids] = True
                self._verified_teacher_bank_snapshot_pending_category[env_ids] = int(collection_category)

    def _set_object_pose_from_center_and_quat(self, env_ids, desired_center_xy, object_quat):
        if env_ids.numel() == 0:
            return

        dtype = self._object_state.dtype
        object_quat = object_quat.to(dtype=dtype)
        desired_center_xy = desired_center_xy.to(dtype=dtype)

        rot_mat = quaternion_to_matrix_ig(object_quat)
        local_half_extents = 0.5 * self.mesh_aabb_extents[env_ids].to(dtype=dtype)
        vertical_half_extent = torch.sum(torch.abs(rot_mat[:, 2, :]) * local_half_extents, dim=-1)

        local_offset = torch.zeros((env_ids.numel(), 3), dtype=dtype, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[env_ids, 2].to(dtype=dtype) * self.object_center_z_scale
        rotated_offset = quat_apply(object_quat, local_offset)

        object_center_world = torch.zeros((env_ids.numel(), 3), dtype=dtype, device=self.device)
        object_center_world[:, :2] = desired_center_xy
        object_center_world[:, 2] = self.table_surface_height[env_ids].to(dtype=dtype) + vertical_half_extent

        root_pos = object_center_world - rotated_offset
        self._object_state[env_ids, :3] = root_pos
        self._object_state[env_ids, 3:7] = object_quat
        self._object_center_init_state[env_ids] = object_center_world

    def _flat_object_robot_penetration_mask(self, env_ids, robot_q_for_collision, desired_center_xy, object_quat):
        if env_ids.numel() == 0:
            return torch.empty((0,), dtype=torch.bool, device=self.device)

        dtype = self._object_state.dtype
        object_quat = object_quat.to(dtype=dtype)
        desired_center_xy = desired_center_xy.to(dtype=dtype)

        rot_mat = quaternion_to_matrix_ig(object_quat)
        local_half_extents = 0.5 * self.mesh_aabb_extents[env_ids].to(dtype=dtype)
        vertical_half_extent = torch.sum(torch.abs(rot_mat[:, 2, :]) * local_half_extents, dim=-1)

        object_center_world = torch.zeros((env_ids.numel(), 3), dtype=dtype, device=self.device)
        object_center_world[:, :2] = desired_center_xy
        object_center_world[:, 2] = self.table_surface_height[env_ids].to(dtype=dtype) + vertical_half_extent

        if robot_q_for_collision is None:
            robot_q_collision = self._q[env_ids]
        else:
            robot_q_collision = robot_q_for_collision
        robot_pcd_world = self.robot_pcd_sampler.sample(robot_q_collision, self.torchurdf_to_isaac_idx)

        rel_world = robot_pcd_world - object_center_world.unsqueeze(1)
        num_points = int(robot_pcd_world.shape[1])
        inv_quat = quat_conjugate(object_quat)
        inv_quat_expanded = inv_quat.unsqueeze(1).repeat(1, num_points, 1).reshape(-1, 4)
        rel_local = quat_apply(inv_quat_expanded, rel_world.reshape(-1, 3)).reshape(env_ids.numel(), num_points, 3)
        local_margin = 0.01
        inside = torch.all(
            torch.abs(rel_local) <= (local_half_extents + local_margin).unsqueeze(1),
            dim=-1,
        )
        return torch.any(inside, dim=1)

    def _resample_flat_center_xy_for_collision(
        self,
        env_ids,
        desired_center_xy,
        object_quat,
        robot_q_for_collision=None,
        max_rounds=12,
    ):
        if env_ids.numel() == 0:
            return desired_center_xy

        dtype = self._object_state.dtype
        xy_min = self.obj_pos_range[env_ids][:, [0, 2]].to(dtype=dtype)
        xy_max = self.obj_pos_range[env_ids][:, [1, 3]].to(dtype=dtype)
        xy_min, xy_max = self._adjust_object_reset_xy_bounds(env_ids, xy_min, xy_max)

        center_xy = desired_center_xy.clone().to(dtype=dtype)
        penetration_mask = self._flat_object_robot_penetration_mask(
            env_ids,
            robot_q_for_collision,
            center_xy,
            object_quat,
        )
        for _ in range(max_rounds):
            if not bool(torch.any(penetration_mask)):
                break
            bad_idx = penetration_mask.nonzero(as_tuple=False).squeeze(-1)
            n_bad = int(bad_idx.numel())
            center_xy[bad_idx, 0] = (
                torch.rand(n_bad, device=self.device, dtype=dtype) * (xy_max[bad_idx, 0] - xy_min[bad_idx, 0])
                + xy_min[bad_idx, 0]
            )
            center_xy[bad_idx, 1] = (
                torch.rand(n_bad, device=self.device, dtype=dtype) * (xy_max[bad_idx, 1] - xy_min[bad_idx, 1])
                + xy_min[bad_idx, 1]
            )
            penetration_mask = self._flat_object_robot_penetration_mask(
                env_ids,
                robot_q_for_collision,
                center_xy,
                object_quat,
            )

        if bool(torch.any(penetration_mask)):
            bad_idx = penetration_mask.nonzero(as_tuple=False).squeeze(-1)
            robot_q_collision = self._q[env_ids] if robot_q_for_collision is None else robot_q_for_collision
            robot_pcd_world = self.robot_pcd_sampler.sample(robot_q_collision, self.torchurdf_to_isaac_idx)
            for local_i in bad_idx.tolist():
                x0, y0 = float(xy_min[local_i, 0].item()), float(xy_min[local_i, 1].item())
                x1, y1 = float(xy_max[local_i, 0].item()), float(xy_max[local_i, 1].item())
                corners = torch.tensor(
                    [[x0, y0], [x0, y1], [x1, y0], [x1, y1]],
                    device=self.device,
                    dtype=dtype,
                )
                robot_xy = robot_pcd_world[local_i, :, :2]
                corner_score = torch.cdist(corners.unsqueeze(0), robot_xy.unsqueeze(0)).amin(dim=-1).squeeze(0)
                center_xy[local_i] = corners[torch.argmax(corner_score)]

        return center_xy

    def _set_fixed_goal_target(self, env_ids, refresh_quat=False):
        if env_ids is None or env_ids.numel() == 0:
            return
        target_dtype = self.reward_settings["target_pos"].dtype
        target_pos = torch.zeros((env_ids.numel(), 3), dtype=target_dtype, device=self.device)
        target_pos[:, :2] = self._object_center_init_state[env_ids, :2].to(dtype=target_dtype)
        target_pos[:, 2] = self.table_surface_height[env_ids].to(dtype=target_dtype) + self._fixed_goal_z_offset_from_table
        self.reward_settings["target_pos"][env_ids] = target_pos

        if refresh_quat:
            if self.use_goal_orientation_target:
                target_quat = self._compute_topdown_target_quat(env_ids)
            else:
                target_quat = normalize(self._eef_state[env_ids, 3:7])
            target_quat = target_quat.to(dtype=self.reward_settings["target_quat"].dtype)
            self.reward_settings["target_quat"][env_ids] = target_quat
            self.reward_settings["target_rot_6d"][env_ids] = matrix_to_rotation_6d(quaternion_to_matrix_ig(target_quat))
            self._target_quat_latched[env_ids] = True

    def _get_current_mobile_base_pose7(self, env_ids):
        mobile_base_pose = self._q[env_ids, :3]
        dtype = self._q.dtype
        yaw = mobile_base_pose[:, 2]
        half_yaw = 0.5 * yaw
        mobile_quat = torch.zeros((mobile_base_pose.shape[0], 4), device=self.device, dtype=dtype)
        mobile_quat[:, 2] = torch.sin(half_yaw)
        mobile_quat[:, 3] = torch.cos(half_yaw)
        mobile_base_pos_world = torch.zeros((mobile_base_pose.shape[0], 3), device=self.device, dtype=dtype)
        mobile_base_pos_world[:, :2] = mobile_base_pose[:, :2]
        return torch.cat([mobile_base_pos_world, mobile_quat], dim=-1)

    def _compute_post_lift_target_world(self, env_ids):
        mobile_base_pose7 = self._get_current_mobile_base_pose7(env_ids)
        base_xy_offset = self.post_lift_target_pos_base[:2].to(
            device=self.device,
            dtype=mobile_base_pose7.dtype,
        ).unsqueeze(0).repeat(env_ids.numel(), 1)
        base_xy_offset_3d = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=mobile_base_pose7.dtype)
        base_xy_offset_3d[:, :2] = base_xy_offset
        target_world = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=mobile_base_pose7.dtype)
        target_world[:, :2] = mobile_base_pose7[:, :2] + quat_apply(
            mobile_base_pose7[:, 3:7],
            base_xy_offset_3d,
        )[:, :2]
        target_world[:, 2] = self.table_surface_height[env_ids].to(dtype=mobile_base_pose7.dtype) + self.post_lift_target_z_from_table
        return target_world

    def _refresh_target_state_components(self):
        eef_rot_mat = quaternion_to_matrix_ig(self._eef_state[:, 3:7])
        eef_rot_mat_t = eef_rot_mat.transpose(1, 2)
        target_rot_mat = quaternion_to_matrix_ig(self.reward_settings["target_quat"])
        target_rot_mat_in_eef_frame = torch.matmul(eef_rot_mat_t, target_rot_mat)
        target_to_eef_rot_6d = matrix_to_rotation_6d(target_rot_mat_in_eef_frame)

        target_to_eef_world = self.reward_settings["target_pos"] - self._eef_state[:, :3]
        if self.teacher_use_eef_frame:
            target_to_eef = torch.matmul(eef_rot_mat_t, target_to_eef_world.unsqueeze(-1)).squeeze(-1)
        else:
            target_to_eef = target_to_eef_world

        point_matching_err_target = self._get_eef_point_matching_err(
            curent_eef_pos7=self._eef_state[:, :7],
            target_eef_pos7=torch.cat([self.reward_settings["target_pos"], self.reward_settings["target_quat"]], dim=-1),
        )
        hand_eef_pos7_rot = torch.cat([self._eef_state[:, :3], self.reward_settings["target_quat"]], dim=-1)
        point_matching_err_hand = self._get_eef_point_matching_err(
            curent_eef_pos7=self._eef_state[:, :7],
            target_eef_pos7=hand_eef_pos7_rot,
        )
        self.states.update(
            {
                "target_to_eef": target_to_eef,
                "target_to_eef_rot_6d": target_to_eef_rot_6d,
                "point_matching_err_target": point_matching_err_target,
                "point_matching_err_hand": point_matching_err_hand,
            }
        )

    def _reset_object_state(
        self,
        object_reset_env_ids,
        apply_teleport_env_ids=None,
        robot_q_for_collision=None,
        eef_xy_for_collision=None,
    ):
        if object_reset_env_ids is None:
            prev_env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            prev_env_ids = object_reset_env_ids.clone()
        prev_object_quat = self._object_state[prev_env_ids, 3:7].clone() if prev_env_ids.numel() > 0 else None

        super()._reset_object_state(
            object_reset_env_ids,
            apply_teleport_env_ids=apply_teleport_env_ids,
            robot_q_for_collision=robot_q_for_collision,
            eef_xy_for_collision=eef_xy_for_collision,
        )

        if object_reset_env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = object_reset_env_ids.clone()
        if env_ids.numel() == 0:
            return
        if hasattr(self, "side_is_left"):
            self.side_is_left[env_ids] = self._sample_side_mask(env_ids)
        if hasattr(self, "_target_quat_latched"):
            self._target_quat_latched[env_ids] = False
        if self.lie_flat_prob <= 0.0:
            return

        tele_mask = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        if apply_teleport_env_ids is not None and apply_teleport_env_ids.numel() > 0:
            tele_mask = torch.isin(env_ids, apply_teleport_env_ids)

        updated_env_ids = []

        if self.preserve_orientation_on_teleport:
            tele_preserve_mask = tele_mask
            tele_resample_mask = torch.zeros_like(tele_mask)
        else:
            tele_preserve_mask = torch.zeros_like(tele_mask)
            tele_resample_mask = tele_mask

        # During in-episode teleport, optionally preserve the previous object orientation.
        if torch.any(tele_preserve_mask):
            tele_env_ids = env_ids[tele_preserve_mask]
            tele_prev_quat = prev_object_quat[tele_preserve_mask]
            tele_prev_quat = tele_prev_quat / torch.norm(tele_prev_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
            desired_center_xy = self._object_state[tele_env_ids, :2].clone()
            robot_q_collision = None if robot_q_for_collision is None else robot_q_for_collision[tele_preserve_mask]
            desired_center_xy = self._resample_flat_center_xy_for_collision(
                tele_env_ids,
                desired_center_xy,
                tele_prev_quat,
                robot_q_for_collision=robot_q_collision,
                max_rounds=int(self.object_teleport_args.get("min_xy_dist_resample_rounds", 12)),
            )
            self._set_object_pose_from_center_and_quat(tele_env_ids, desired_center_xy, tele_prev_quat)
            updated_env_ids.append(tele_env_ids)

        # Full episode resets still resample a flat random-yaw orientation.
        reset_mask = (~tele_mask) | tele_resample_mask
        if torch.any(reset_mask):
            reset_env_ids = env_ids[reset_mask]
            flat_mask = torch.rand(reset_env_ids.numel(), device=self.device) < self.lie_flat_prob
            if torch.any(flat_mask):
                flat_env_ids = reset_env_ids[flat_mask]
                n_flat = int(flat_env_ids.numel())

                yaw_axis = torch.zeros((n_flat, 3), dtype=self._object_state.dtype, device=self.device)
                yaw_axis[:, 2] = 1.0
                yaw_angle = torch.rand(n_flat, dtype=self._object_state.dtype, device=self.device) * 6.283185307179586
                q_yaw = quat_from_angle_axis(yaw_angle, yaw_axis)
                flat_base = torch.tensor(
                    self._flat_object_quat_base_values,
                    dtype=self._object_state.dtype,
                    device=self.device,
                ).unsqueeze(0).repeat(n_flat, 1)
                object_quat = quat_mul(q_yaw, flat_base)
                object_quat = object_quat / torch.norm(object_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
                desired_center_xy = self._object_state[flat_env_ids, :2].clone()
                robot_q_collision = None if robot_q_for_collision is None else robot_q_for_collision[reset_mask][flat_mask]
                desired_center_xy = self._resample_flat_center_xy_for_collision(
                    flat_env_ids,
                    desired_center_xy,
                    object_quat,
                    robot_q_for_collision=robot_q_collision,
                    max_rounds=int(self.object_teleport_args.get("min_xy_dist_resample_rounds", 12)),
                )
                self._set_object_pose_from_center_and_quat(flat_env_ids, desired_center_xy, object_quat)
                updated_env_ids.append(flat_env_ids)

        if len(updated_env_ids) == 0:
            return

        updated_env_ids = torch.cat(updated_env_ids)
        multi_env_ids_obj_int32 = self._global_indices[updated_env_ids, self._object_id].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_obj_int32),
            len(multi_env_ids_obj_int32),
        )

        object_pcds_world = transform_pcds_to_world(self.object_pcds, self._object_state[:, :7])
        if self.object_pcd_t0 is None:
            self.object_pcd_t0 = object_pcds_world.clone()
        self.object_pcd_t0[updated_env_ids] = object_pcds_world[updated_env_ids].clone()

    def _update_states(self):
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._set_fixed_goal_target(all_env_ids, refresh_quat=not self.use_goal_orientation_target)
        if self.use_goal_orientation_target:
            refresh_quat_mask = (~self._target_quat_latched) | (self.progress_buf == 0)
            if bool(torch.any(refresh_quat_mask)):
                refresh_env_ids = refresh_quat_mask.nonzero(as_tuple=False).squeeze(-1)
                self._set_fixed_goal_target(refresh_env_ids, refresh_quat=True)

        super()._update_states()

        if torch.any(self.object_reset_mask):
            reset_object_env_ids = self.object_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self.post_lift_target_active[reset_object_env_ids] = False

        self.post_lift_target_active = self.post_lift_target_active | self.states["lift"]
        if torch.any(self.post_lift_target_active):
            lifted_env_ids = self.post_lift_target_active.nonzero(as_tuple=False).squeeze(-1)
            self.reward_settings["target_pos"][lifted_env_ids] = self._compute_post_lift_target_world(lifted_env_ids)

        if self.enable_fabric:
            switch_pos_err = torch.norm(self._eef_state[:, :3] - self.switching_target_pos, dim=-1)
            if self.switch_use_orientation_target:
                quat_dot = torch.abs(
                    torch.sum(normalize(self._eef_state[:, 3:7]) * normalize(self.switching_target_quat), dim=-1)
                ).clamp(max=1.0)
                switch_rot_err_deg = 2.0 * torch.rad2deg(torch.acos(quat_dot))
                switch_ready = (switch_pos_err < self.switch_pos_tol) & (switch_rot_err_deg < self.switch_rot_tol_deg)
                self.extras["fabric/switch_rot_err_deg"] = torch.mean(switch_rot_err_deg).item()
            else:
                switch_ready = switch_pos_err < self.switch_pos_tol
                self.extras["fabric/switch_rot_err_deg"] = 0.0
            self.fabric_switch_enable[switch_ready] = False
            self.fabric_switch_enable[self.progress_buf == 0] = True
            self.extras["fabric/switch_pos_err"] = torch.mean(switch_pos_err).item()

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

    def _draw_top_long_debug_regions(self):
        if not self.debug_viz or self.viewer is None:
            return

        self.gym.clear_lines(self.viewer)
        def append_line(verts, colors, p0, p1, color):
            verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(p1[0]), float(p1[1]), float(p1[2])])
            colors.extend(color)

        def append_box(verts, colors, box_min, box_max, color):
            x0, y0, z0 = float(box_min[0]), float(box_min[1]), float(box_min[2])
            x1, y1, z1 = float(box_max[0]), float(box_max[1]), float(box_max[2])
            corners = [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
            edge_ids = (
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            )
            for i0, i1 in edge_ids:
                append_line(verts, colors, corners[i0], corners[i1], color)

        cross_len = 0.03

        for env_id in range(self.num_envs):
            env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
            dtype = self._q.dtype
            verts = []
            colors = []

            object_center = self._object_state[env_ids, :3]
            table_center_xy = self.table_pos[env_ids, :2]
            table_half_xy = 0.5 * self.table_size[env_ids, :2]
            table_xy_min = table_center_xy - table_half_xy
            table_xy_max = table_center_xy + table_half_xy

            spawn_xy_min = self.obj_pos_range[env_ids][:, [0, 2]].to(dtype=dtype)
            spawn_xy_max = self.obj_pos_range[env_ids][:, [1, 3]].to(dtype=dtype)
            spawn_xy_min, spawn_xy_max = self._adjust_object_reset_xy_bounds(env_ids, spawn_xy_min, spawn_xy_max)
            spawn_center_z = self.table_surface_height[env_ids].to(dtype=dtype) + self.mesh_aabb_extents[env_ids, 2] * self.object_center_z_scale
            spawn_box_min = torch.cat([spawn_xy_min[0], spawn_center_z[0:1]], dim=0)
            spawn_box_max = torch.cat([spawn_xy_max[0], spawn_center_z[0:1]], dim=0)

            near_rel_min, near_rel_max, near_valid = self._get_clipped_topdown_recovery_palm_rel_object_bounds(
                object_center,
                table_xy_min,
                table_xy_max,
                dtype,
                self._topdown_near_recovery_rel_object_min_cfg,
                self._topdown_near_recovery_rel_object_max_cfg,
            )
            far_rel_min, far_rel_max, far_valid = self._get_clipped_topdown_recovery_palm_rel_object_bounds(
                object_center,
                table_xy_min,
                table_xy_max,
                dtype,
                self._topdown_far_recovery_rel_object_min_cfg,
                self._topdown_far_recovery_rel_object_max_cfg,
            )

            # Yellow rectangle: valid object spawn XY range.
            p00 = (spawn_box_min[0], spawn_box_min[1], spawn_box_min[2])
            p10 = (spawn_box_max[0], spawn_box_min[1], spawn_box_min[2])
            p11 = (spawn_box_max[0], spawn_box_max[1], spawn_box_min[2])
            p01 = (spawn_box_min[0], spawn_box_max[1], spawn_box_min[2])
            append_line(verts, colors, p00, p10, [1.0, 0.9, 0.1])
            append_line(verts, colors, p10, p11, [1.0, 0.9, 0.1])
            append_line(verts, colors, p11, p01, [1.0, 0.9, 0.1])
            append_line(verts, colors, p01, p00, [1.0, 0.9, 0.1])

            # Orange box: near-recovery palm target region.
            if bool(near_valid[0].item()):
                near_box_min = object_center[0] + near_rel_min[0]
                near_box_max = object_center[0] + near_rel_max[0]
                append_box(verts, colors, near_box_min, near_box_max, [1.0, 0.55, 0.1])

            # Cyan box: far-recovery palm target region.
            if bool(far_valid[0].item()):
                far_box_min = object_center[0] + far_rel_min[0]
                far_box_max = object_center[0] + far_rel_max[0]
                append_box(verts, colors, far_box_min, far_box_max, [0.1, 0.9, 1.0])

            # Magenta cross: currently active goal target.
            target_pos = self.reward_settings["target_pos"][env_id]
            append_line(
                verts,
                colors,
                target_pos + torch.tensor([cross_len, 0.0, 0.0], device=self.device, dtype=dtype),
                target_pos - torch.tensor([cross_len, 0.0, 0.0], device=self.device, dtype=dtype),
                [1.0, 0.0, 1.0],
            )
            append_line(
                verts,
                colors,
                target_pos + torch.tensor([0.0, cross_len, 0.0], device=self.device, dtype=dtype),
                target_pos - torch.tensor([0.0, cross_len, 0.0], device=self.device, dtype=dtype),
                [1.0, 0.0, 1.0],
            )
            append_line(
                verts,
                colors,
                target_pos + torch.tensor([0.0, 0.0, cross_len], device=self.device, dtype=dtype),
                target_pos - torch.tensor([0.0, 0.0, cross_len], device=self.device, dtype=dtype),
                [1.0, 0.0, 1.0],
            )

            if verts:
                self.gym.add_lines(
                    self.viewer,
                    self.envs[env_id],
                    len(colors) // 3,
                    verts,
                    colors,
                )

    def post_physics_step(self):
        pending_snapshot_env_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        if hasattr(self, "_verified_teacher_bank_snapshot_pending"):
            pending_snapshot_env_ids = self._verified_teacher_bank_snapshot_pending.nonzero(as_tuple=False).squeeze(-1).clone()
        super().post_physics_step()
        self._flush_pending_verified_teacher_episode_starts(pending_snapshot_env_ids)
        self._draw_top_long_debug_regions()

    def compute_reward(self):
        super().compute_reward()
        done_envs = self.reset_buf > 0
        if torch.any(done_envs):
            done_env_ids = done_envs.nonzero(as_tuple=False).squeeze(-1)
            done_category_ids = self.current_episode_verified_bank_category[done_env_ids]
            valid_category_mask = done_category_ids >= 0
            if torch.any(valid_category_mask):
                done_category_increments = torch.bincount(
                    done_category_ids[valid_category_mask],
                    minlength=len(self.VERIFIED_BANK_CATEGORY_NAMES),
                )
                self.per_verified_bank_episode_counts += done_category_increments

            success_env_ids = (done_envs & self.success_long_enough).nonzero(as_tuple=False).squeeze(-1)
            success_done_category_ids = self.current_episode_verified_bank_category[success_env_ids]
            valid_success_category_mask = success_done_category_ids >= 0
            if torch.any(valid_success_category_mask):
                success_category_increments = torch.bincount(
                    success_done_category_ids[valid_success_category_mask],
                    minlength=len(self.VERIFIED_BANK_CATEGORY_NAMES),
                )
                self.per_verified_bank_success_counts += success_category_increments

        for category_id, category_name in enumerate(self.VERIFIED_BANK_CATEGORY_NAMES):
            category_total_eps = int(self.per_verified_bank_episode_counts[category_id].item())
            category_success_eps = int(self.per_verified_bank_success_counts[category_id].item())
            category_success_rate = (
                float(category_success_eps) / float(category_total_eps)
            ) if category_total_eps > 0 else 0.0
            self.extras[f"metrics/verified_bank_success_rate_5cm_per_ep_{category_name}"] = category_success_rate
            self.extras[f"metrics/verified_bank_success_rate_5cm_per_ep_{category_name}_count"] = category_total_eps

    def compute_observations(self):
        self._refresh()

        obs_components = [
            "q_hand",
            "grasp_side_binary",
            "eef_finger1_pos_relative",
            "eef_finger2_pos_relative",
            "eef_finger3_pos_relative",
            "eef_finger4_pos_relative",
            "object_to_eef",
            "object_to_eef_rot_6d",
            "object_grasp_target_to_eef",
            "eef_table_dist",
            "object_z_axis_world",
            "hand_z_axis_world",
            "target_to_eef",
            "target_to_eef_rot_6d",
        ]

        states_components = [
            "q",
            "qd",
            "grasp_side_binary",
            "eef_pos",
            "eef_rot_6d",
            "eef_vel",
            "eef_finger1_pos_relative",
            "eef_finger2_pos_relative",
            "eef_finger3_pos_relative",
            "eef_finger4_pos_relative",
            "object_to_eef",
            "object_to_eef_rot_6d",
            "object_grasp_target_to_eef",
            "eef_table_dist",
            "object_z_axis_world",
            "hand_z_axis_world",
            "target_to_eef",
            "target_to_eef_rot_6d",
        ]

        obs_buf = torch.cat([self.states[ob] for ob in obs_components], dim=-1)
        states_buf = torch.cat([self.states[st] for st in states_components], dim=-1)

        if self.cfg["observation"]["usage"]["use_xy"]:
            obj_xy_bbox = self.mesh_aabb_extents[:, :2]
            obs_buf = torch.cat([obs_buf, obj_xy_bbox], dim=-1)
            states_buf = torch.cat([states_buf, obj_xy_bbox], dim=-1)
        if self.cfg["observation"]["usage"]["use_z"]:
            obj_height = self.mesh_aabb_extents[:, 2:3]
            obs_buf = torch.cat([obs_buf, obj_height], dim=-1)
            states_buf = torch.cat([states_buf, obj_height], dim=-1)

        self.obs_buf = obs_buf
        self.states_buf = states_buf
        return self.obs_buf
