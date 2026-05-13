"""
Mobile distillation env for side-table objects with a flat top-down teacher.

This intentionally stays close to the regular top-down mobile distillation env,
but exposes the side-table teacher observation layout and resets objects in the
flat orientation used by the side RL teacher when lie_flat_prob=1.0.
"""

import math
import time

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
from tqdm import tqdm


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
        self.use_post_lift_base_target = bool(cfg["reward"]["params"].get("use_post_lift_base_target", False))
        self.side_mode = str(cfg["env"].get("eef_init", {}).get("side_mode", "both"))
        robot_init_cfg = cfg["env"]["robot_init"]
        self.switch_activate_radius = float(
            robot_init_cfg.get(
                "switch_activate_radius",
                robot_init_cfg.get("switch_pos_tol", robot_init_cfg.get("switch_tol", 0.2)),
            )
        )
        self.switch_activate_radius_exit = float(
            robot_init_cfg.get("switch_activate_radius_exit", self.switch_activate_radius)
        )
        self.switch_rot_tol_deg = float(robot_init_cfg.get("switch_rot_tol_deg", 180.0))
        self.switch_use_orientation_target = bool(robot_init_cfg.get("switch_use_orientation_target", False))
        self.use_goal_orientation_target = bool(cfg["reward"]["params"].get("use_goal_orientation_target", False))
        self.flat_object_axis_z_abs_max = float(cfg["reward"]["params"].get("flat_object_axis_z_abs_max", 0.5))
        target_waypoint_cfg = cfg["reward"]["params"].get("target_waypoint", {})
        self.use_target_waypoint = bool(target_waypoint_cfg.get("enable", False))
        self.target_waypoint_until_lift = bool(target_waypoint_cfg.get("until_lift", True))
        self._target_waypoint_eef_delta_abs_max_cfg = target_waypoint_cfg.get(
            "eef_delta_abs_max",
            [0.25, 0.25, 0.35],
        )
        eef_init_cfg = cfg["env"].get("eef_init", {})
        self._reset_yaw_noise_deg = float(eef_init_cfg.get("yaw_noise_deg", 45.0))
        self._reset_yaw_noise_rad = float(math.radians(self._reset_yaw_noise_deg))
        self._reset_pitch_roll_noise_deg = float(eef_init_cfg.get("pitch_roll_noise_deg", 45.0))
        self._reset_pitch_roll_noise_rad = float(math.radians(self._reset_pitch_roll_noise_deg))
        self._reset_hand_joint_noise_deg = float(eef_init_cfg.get("hand_joint_noise_deg", 20.0))
        self._topdown_recovery_base_init_range_cfg = eef_init_cfg.get(
            "flat_topdown_recovery_base_init_range",
            cfg["env"]["robot_init"]["base_init_range"],
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
        self._topdown_recovery_hard_max_resample_attempts = int(
            eef_init_cfg.get(
                "flat_topdown_hard_max_resample_attempts",
                max(256, self._topdown_recovery_max_resample_attempts * 32),
            )
        )
        self._topdown_recovery_bank_size = int(eef_init_cfg.get("flat_topdown_recovery_bank_size", 128))
        self._topdown_recovery_bank_max_ik_goals = int(
            eef_init_cfg.get("flat_topdown_recovery_bank_max_ik_goals", 2048)
        )
        self._topdown_debug_sampling = bool(eef_init_cfg.get("flat_topdown_debug_sampling", False))
        self._topdown_debug_sampling_min_rounds = int(
            eef_init_cfg.get("flat_topdown_debug_sampling_min_rounds", 64)
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
        self._topdown_recovery_bank_ready = False
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
        should_build_topdown_recovery_bank = (
            self._topdown_recovery_bank_size > 0
            and float(self._verified_teacher_bank_sampling_probs[1:].sum().item()) > 0.0
            and (not self._teacher_state_bank_collect_requested)
            and (not self._verified_teacher_bank_enable)
        )
        if should_build_topdown_recovery_bank:
            self._init_topdown_recovery_bank()
            self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def _adjust_object_reset_xy_bounds(self, env_ids, xy_min, xy_max):
        xy_min = xy_min.clone()
        xy_max = xy_max.clone()
        table_y_margin = float(self.cfg["env"]["object_settings"].get("table_y_margin", 0.0))
        if table_y_margin > 0.0:
            # CODEX TOP-LONG SPAWN: keep configured y sampling range inside each table's y edges.
            table_y_min = self.table_pos[env_ids, 1] - 0.5 * self.table_size[env_ids, 1]
            table_y_max = self.table_pos[env_ids, 1] + 0.5 * self.table_size[env_ids, 1]
            object_half_y = 0.5 * self.mesh_aabb_extents[env_ids, 1]
            safe_y_min = table_y_min + object_half_y + table_y_margin
            safe_y_max = table_y_max - object_half_y - table_y_margin
            safe_y_mid = 0.5 * (table_y_min + table_y_max)
            valid_y_bounds = safe_y_min <= safe_y_max
            safe_y_min = torch.where(valid_y_bounds, safe_y_min, safe_y_mid)
            safe_y_max = torch.where(valid_y_bounds, safe_y_max, safe_y_mid)
            xy_min[:, 1] = torch.minimum(torch.maximum(xy_min[:, 1], safe_y_min), safe_y_max)
            xy_max[:, 1] = torch.maximum(torch.minimum(xy_max[:, 1], safe_y_max), xy_min[:, 1])

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
            self._topdown_recovery_base_init_range_cfg,
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
        self.target_waypoint_active = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.target_waypoint_alpha = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32)
        self.final_target_pos = self.reward_settings["target_pos"].clone()
        self.reward_settings["success_tolerance"] = to_torch(
            self.cfg["reward"]["params"].get("success_tolerance", 0.1),
            device=self.device,
        )
        waypoint_eef_delta_abs_max = torch.tensor(
            self._target_waypoint_eef_delta_abs_max_cfg,
            device=self.device,
            dtype=torch.float32,
        ).flatten()
        if waypoint_eef_delta_abs_max.numel() != 3:
            raise ValueError("reward.params.target_waypoint.eef_delta_abs_max must have exactly 3 values")
        self.target_waypoint_eef_delta_abs_max = torch.clamp(waypoint_eef_delta_abs_max, min=1.0e-6)
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
        # Match the flat top-down teacher RL reset family: for top-down starts we
        # keep the nominal orientation and only randomize the hand joints.
        return nominal_target_quat / torch.norm(nominal_target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _sample_topdown_joint_config(
        self,
        env_ids,
        rel_object_min_cfg,
        rel_object_max_cfg,
        log_prefix,
        object_center_world=None,
        fallback_joint_config=None,
    ):
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
            self._topdown_recovery_base_init_range_cfg,
            device=device,
            dtype=dtype,
        )
        if object_center_world is None:
            object_center = self._object_center_init_state[env_ids, :3].to(dtype=dtype).clone()
        else:
            object_center = object_center_world.to(device=device, dtype=dtype).clone()
        table_center_xy = self.table_pos[env_ids, :2].to(dtype=dtype)
        table_half_xy = 0.5 * self.table_size[env_ids, :2].to(dtype=dtype)
        table_xy_min = table_center_xy - table_half_xy
        table_xy_max = table_center_xy + table_half_xy

        solved_mask = torch.zeros(num_recovery, device=device, dtype=torch.bool)
        # CODEX TOP-LONG RECOVERY BANK: preserve the superclass reset as fallback
        # for pathological IK samples instead of blocking forever on one env.
        if fallback_joint_config is None:
            solved_joint_config = self._q[env_ids].clone()
        else:
            solved_joint_config = fallback_joint_config.to(device=device, dtype=dtype).clone()
        solved_palm_pos = torch.zeros((num_recovery, 3), device=device, dtype=dtype)
        solved_target_quat = torch.zeros((num_recovery, 4), device=device, dtype=dtype)
        solved_target_quat[:, 3] = 1.0

        palm_offset_local = self._palm_center_from_link7_local.to(device=device, dtype=dtype)
        hand_joint_noise = self._reset_hand_joint_noise_rad.to(device=device, dtype=dtype)
        warn_interval = max(1, int(self._topdown_recovery_max_resample_attempts))
        hard_max_attempts = int(self._topdown_recovery_hard_max_resample_attempts)
        debug_enabled = bool(self._topdown_debug_sampling)
        debug_min_rounds = max(1, int(self._topdown_debug_sampling_min_rounds))
        debug_stats = {
            "candidates": 0,
            "valid_xy": 0,
            "valid_scene": 0,
            "ik_success": 0,
            "no_penetration": 0,
            "accepted": 0,
        }

        def _format_debug_rate(num, denom):
            if denom <= 0:
                return "0/0(0.0%)"
            return f"{num}/{denom}({100.0 * float(num) / float(denom):.1f}%)"

        def _print_sampling_debug(reason):
            if not debug_enabled or debug_stats["candidates"] <= 0:
                return
            solved = int(solved_mask.sum().item())
            pending = num_recovery - solved
            print(
                f"[{log_prefix}/debug] {reason} "
                f"round={attempt_idx} solved={solved}/{num_recovery} pending={pending} "
                f"xy={_format_debug_rate(debug_stats['valid_xy'], debug_stats['candidates'])} "
                f"scene={_format_debug_rate(debug_stats['valid_scene'], debug_stats['valid_xy'])} "
                f"ik={_format_debug_rate(debug_stats['ik_success'], debug_stats['valid_scene'])} "
                f"no_pen={_format_debug_rate(debug_stats['no_penetration'], debug_stats['ik_success'])} "
                f"accepted={_format_debug_rate(debug_stats['accepted'], debug_stats['candidates'])}"
            )

        attempt_idx = 0
        while True:
            unresolved = (~solved_mask).nonzero(as_tuple=False).squeeze(-1)
            if unresolved.numel() == 0:
                break
            if hard_max_attempts > 0 and attempt_idx >= hard_max_attempts:
                print(
                    f"[{log_prefix}] fallback to canonical reset "
                    f"unsolved={int(unresolved.numel())}/{num_recovery} "
                    f"after_round={attempt_idx}"
                )
                _print_sampling_debug("fallback")
                break
            attempt_idx += 1
            if attempt_idx % warn_interval == 0:
                print(
                    f"[{log_prefix}] still sampling topdown reset "
                    f"unsolved={int(unresolved.numel())}/{num_recovery} "
                    f"round={attempt_idx}"
                )
                if attempt_idx >= debug_min_rounds:
                    _print_sampling_debug("still_sampling")

            count = int(unresolved.numel())
            if debug_enabled:
                debug_stats["candidates"] += count
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
            if debug_enabled:
                debug_stats["valid_xy"] += int(valid_xy.sum().item())
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
            if debug_enabled:
                debug_stats["valid_scene"] += int(valid_scene.sum().item())
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
            if debug_enabled:
                debug_stats["ik_success"] += int(success.sum().item())
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
            if debug_enabled:
                debug_stats["no_penetration"] += int(no_penetration.sum().item())
            if not bool(torch.any(no_penetration)):
                continue

            solved_valid = success_unresolved[no_penetration]
            solved_joint_config[solved_valid] = success_joint_config[no_penetration]
            solved_palm_pos[solved_valid] = valid_palm_pos_world[success][no_penetration]
            solved_target_quat[solved_valid] = nominal_target_quat[success][no_penetration]
            solved_mask[solved_valid] = True
            if debug_enabled:
                debug_stats["accepted"] += int(solved_valid.numel())

        return solved_joint_config, solved_palm_pos, solved_target_quat

    def _sample_topdown_recovery_joint_config(
        self,
        env_ids,
        force_far=False,
        object_center_world=None,
        fallback_joint_config=None,
    ):
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
            object_center_world=object_center_world,
            fallback_joint_config=fallback_joint_config,
        )

    def _choose_topdown_synthetic_categories(self, env_ids, forced_category=None):
        if env_ids.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        if forced_category is not None and int(forced_category) >= 0:
            return torch.full((env_ids.numel(),), int(forced_category), dtype=torch.long, device=self.device)

        # CODEX TOP-LONG RESET: when no HDF5 bank is loaded, reuse verified-bank
        # sampling_probs to choose synthetic afar/near/far reset categories.
        probs = self._verified_teacher_bank_sampling_probs.to(device=self.device, dtype=torch.float32).clamp_min(0.0)
        if not bool(torch.any(probs > 0.0)):
            probs = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        probs = probs / probs.sum().clamp_min(1.0e-8)
        batched_probs = probs.unsqueeze(0).repeat(env_ids.numel(), 1)
        return torch.multinomial(batched_probs, num_samples=1).squeeze(-1).to(dtype=torch.long)

    def _apply_topdown_joint_reset(self, env_ids, joint_config):
        if env_ids.numel() == 0:
            return
        self.set_robot_joint_state(joint_config, env_ids=env_ids)
        if self.enable_fabric:
            self.fabric_q[env_ids, :10] = joint_config[:, :10]
            self.fabric_q[env_ids, 10:] = joint_config[:, 26:]
            self.fabric_qd[env_ids, :] = 0.0
            self.fabric_qdd[env_ids, :] = 0.0

    def _apply_object_center_state(self, env_ids, object_center_world, object_quat=None):
        # CODEX TOP-LONG RECOVERY BANK: use side's root-state helper, then keep
        # the top-long object pointcloud anchor consistent with the restored pose.
        FrankaLEAPMobileDistillationPickSide._apply_object_center_state(
            self,
            env_ids,
            object_center_world,
            object_quat,
        )
        if hasattr(self, "object_pcds") and hasattr(self, "object_pcd_t0"):
            object_pcds_world = transform_pcds_to_world(
                self.object_pcds[env_ids],
                self._object_state[env_ids, :7],
            )
            if self.object_pcd_t0 is None:
                self.object_pcd_t0 = transform_pcds_to_world(self.object_pcds, self._object_state[:, :7])
            self.object_pcd_t0[env_ids] = object_pcds_world.clone()

    def _make_topdown_recovery_bank_is_far(self):
        bank_size = int(self._topdown_recovery_bank_size)
        near_prob = float(max(0.0, self._verified_teacher_bank_sampling_probs[1].item()))
        far_prob = float(max(0.0, self._verified_teacher_bank_sampling_probs[2].item()))
        is_far = torch.zeros((bank_size,), device=self.device, dtype=torch.bool)
        if bank_size <= 0 or far_prob <= 0.0:
            return is_far
        if near_prob <= 0.0:
            is_far[:] = True
            return is_far

        num_far = int(round(bank_size * far_prob / max(near_prob + far_prob, 1.0e-8)))
        num_far = min(max(num_far, 1), bank_size - 1)
        is_far[:num_far] = True
        return is_far[torch.randperm(bank_size, device=self.device)]

    def _sample_topdown_bank_fallback_joint_config(self, env_ids, dtype):
        joint_config = self.canonical_joint_config[env_ids].clone().to(dtype=dtype)
        joint_config[:, :3] = self._sample_mobile_base_init_pose(env_ids, dtype=dtype)
        return joint_config

    def _sample_topdown_bank_object_state(self, env_ids, robot_q_for_collision):
        # CODEX TOP-LONG RECOVERY BANK: sample reset object states as tensors,
        # rather than calling reset_idx per bank slot. This mirrors side's bank
        # construction and avoids simulator root-state writes during cache build.
        dtype = self._object_state.dtype
        num_samples = int(env_ids.numel())
        xy_min = self.obj_pos_range[env_ids][:, [0, 2]].to(dtype=dtype)
        xy_max = self.obj_pos_range[env_ids][:, [1, 3]].to(dtype=dtype)
        xy_min, xy_max = self._adjust_object_reset_xy_bounds(env_ids, xy_min, xy_max)
        desired_center_xy = xy_min + torch.rand((num_samples, 2), device=self.device, dtype=dtype) * (xy_max - xy_min)

        yaw_axis = torch.zeros((num_samples, 3), dtype=dtype, device=self.device)
        yaw_axis[:, 2] = 1.0
        yaw_angle = torch.rand(num_samples, dtype=dtype, device=self.device) * 6.283185307179586
        q_yaw = quat_from_angle_axis(yaw_angle, yaw_axis)
        flat_base = torch.tensor(
            self._flat_object_quat_base_values,
            dtype=dtype,
            device=self.device,
        ).unsqueeze(0).repeat(num_samples, 1)
        object_quat = quat_mul(q_yaw, flat_base)
        object_quat = object_quat / torch.norm(object_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)

        desired_center_xy = self._resample_flat_center_xy_for_collision(
            env_ids,
            desired_center_xy,
            object_quat,
            robot_q_for_collision=robot_q_for_collision,
            max_rounds=int(self.object_teleport_args.get("min_xy_dist_resample_rounds", 12)),
        )

        rot_mat = quaternion_to_matrix_ig(object_quat)
        local_half_extents = 0.5 * self.mesh_aabb_extents[env_ids].to(dtype=dtype)
        vertical_half_extent = torch.sum(torch.abs(rot_mat[:, 2, :]) * local_half_extents, dim=-1)
        object_center_world = torch.zeros((num_samples, 3), dtype=dtype, device=self.device)
        object_center_world[:, :2] = desired_center_xy
        object_center_world[:, 2] = self.table_surface_height[env_ids].to(dtype=dtype) + vertical_half_extent
        return object_center_world, object_quat

    def _init_topdown_recovery_bank(self):
        bank_size = int(self._topdown_recovery_bank_size)
        num_envs = int(self.num_envs)
        dtype = self._q.dtype
        if bank_size <= 0 or num_envs <= 0:
            return

        self._topdown_recovery_bank_joint_config_cpu = torch.empty(
            (num_envs, bank_size, self.num_dofs), dtype=dtype, device="cpu"
        )
        self._topdown_recovery_bank_object_center_world_cpu = torch.empty(
            (num_envs, bank_size, 3), dtype=dtype, device="cpu"
        )
        self._topdown_recovery_bank_object_quat_world_cpu = torch.empty(
            (num_envs, bank_size, 4), dtype=dtype, device="cpu"
        )
        self._topdown_recovery_bank_palm_pos_cpu = torch.empty(
            (num_envs, bank_size, 3), dtype=dtype, device="cpu"
        )
        self._topdown_recovery_bank_palm_quat_cpu = torch.empty(
            (num_envs, bank_size, 4), dtype=dtype, device="cpu"
        )
        self._topdown_recovery_bank_is_far_cpu = torch.empty(
            (num_envs, bank_size), dtype=torch.bool, device="cpu"
        )

        env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
        goals_per_round = max(1, int(self._topdown_recovery_bank_max_ik_goals))
        goals_per_round = min(goals_per_round, max(num_envs * 16, 64))
        slots_per_round = max(1, goals_per_round // max(num_envs, 1))
        bank_is_far = self._make_topdown_recovery_bank_is_far()
        t0 = time.time()
        progress = tqdm(total=bank_size, desc="Building TopLong Recovery Bank")
        for start in range(0, bank_size, slots_per_round):
            cur_slots = min(slots_per_round, bank_size - start)
            end = start + cur_slots
            batched_env_ids = env_ids.repeat(cur_slots)
            batched_is_far = bank_is_far[start:end].repeat_interleave(num_envs)
            fallback_joint_config = self._sample_topdown_bank_fallback_joint_config(batched_env_ids, dtype=dtype)
            object_center_world, object_quat_world = self._sample_topdown_bank_object_state(
                batched_env_ids,
                robot_q_for_collision=fallback_joint_config,
            )

            joint_config = fallback_joint_config.clone()
            palm_pos = torch.zeros((batched_env_ids.numel(), 3), device=self.device, dtype=dtype)
            palm_quat = torch.zeros((batched_env_ids.numel(), 4), device=self.device, dtype=dtype)
            palm_quat[:, 3] = 1.0
            for force_far in (False, True):
                category_mask = batched_is_far == bool(force_far)
                if not bool(torch.any(category_mask)):
                    continue
                category_joint_config, category_palm_pos, category_palm_quat = self._sample_topdown_recovery_joint_config(
                    batched_env_ids[category_mask],
                    force_far=force_far,
                    object_center_world=object_center_world[category_mask],
                    fallback_joint_config=fallback_joint_config[category_mask],
                )
                joint_config[category_mask] = category_joint_config
                palm_pos[category_mask] = category_palm_pos
                palm_quat[category_mask] = category_palm_quat

            joint_config = joint_config.reshape(cur_slots, num_envs, self.num_dofs).transpose(0, 1).contiguous().cpu()
            object_center_world = object_center_world.reshape(cur_slots, num_envs, 3).transpose(0, 1).contiguous().cpu()
            object_quat_world = object_quat_world.reshape(cur_slots, num_envs, 4).transpose(0, 1).contiguous().cpu()
            palm_pos = palm_pos.reshape(cur_slots, num_envs, 3).transpose(0, 1).contiguous().cpu()
            palm_quat = palm_quat.reshape(cur_slots, num_envs, 4).transpose(0, 1).contiguous().cpu()
            is_far = batched_is_far.reshape(cur_slots, num_envs).transpose(0, 1).contiguous().cpu()

            self._topdown_recovery_bank_joint_config_cpu[:, start:end].copy_(joint_config)
            self._topdown_recovery_bank_object_center_world_cpu[:, start:end].copy_(object_center_world)
            self._topdown_recovery_bank_object_quat_world_cpu[:, start:end].copy_(object_quat_world)
            self._topdown_recovery_bank_palm_pos_cpu[:, start:end].copy_(palm_pos)
            self._topdown_recovery_bank_palm_quat_cpu[:, start:end].copy_(palm_quat)
            self._topdown_recovery_bank_is_far_cpu[:, start:end].copy_(is_far)
            progress.update(cur_slots)
        progress.close()

        elapsed = time.time() - t0
        print(
            f"Built top-long recovery bank: num_envs={num_envs}, bank_size={bank_size}, "
            f"slots_per_round={slots_per_round}, elapsed={elapsed:.1f}s"
        )
        self._topdown_recovery_bank_ready = True

    def _sample_topdown_recovery_from_bank(self, env_ids, force_far=None):
        env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
        bank_idx_cpu = torch.zeros((env_ids.numel(),), device="cpu", dtype=torch.long)
        for i in range(int(env_ids_cpu.numel())):
            env_id = int(env_ids_cpu[i].item())
            candidate_mask = torch.ones((self._topdown_recovery_bank_size,), dtype=torch.bool, device="cpu")
            if force_far is not None:
                candidate_mask &= self._topdown_recovery_bank_is_far_cpu[env_id] == bool(force_far)
            candidate_ids = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
            if candidate_ids.numel() == 0:
                candidate_ids = torch.arange(self._topdown_recovery_bank_size, device="cpu", dtype=torch.long)
            rand_idx = torch.randint(
                low=0,
                high=int(candidate_ids.numel()),
                size=(1,),
                device="cpu",
                dtype=torch.long,
            )[0]
            bank_idx_cpu[i] = candidate_ids[rand_idx]

        joint_config = self._topdown_recovery_bank_joint_config_cpu[env_ids_cpu, bank_idx_cpu].to(
            device=self.device,
            dtype=self._q.dtype,
        )
        object_center_world = self._topdown_recovery_bank_object_center_world_cpu[env_ids_cpu, bank_idx_cpu].to(
            device=self.device,
            dtype=self._q.dtype,
        )
        object_quat_world = self._topdown_recovery_bank_object_quat_world_cpu[env_ids_cpu, bank_idx_cpu].to(
            device=self.device,
            dtype=self._q.dtype,
        )
        palm_pos = self._topdown_recovery_bank_palm_pos_cpu[env_ids_cpu, bank_idx_cpu].to(
            device=self.device,
            dtype=self._q.dtype,
        )
        palm_quat = self._topdown_recovery_bank_palm_quat_cpu[env_ids_cpu, bank_idx_cpu].to(
            device=self.device,
            dtype=self._q.dtype,
        )
        is_far = self._topdown_recovery_bank_is_far_cpu[env_ids_cpu, bank_idx_cpu].to(device=self.device)
        return joint_config, object_center_world, object_quat_world, palm_pos, palm_quat, is_far

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
        collection_enabled = bool(self._verified_teacher_bank_collection_enabled)
        forced_collection_category = collection_category if collection_category >= 0 else None
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
            sampled_category_ids = self._choose_topdown_synthetic_categories(
                env_ids,
                forced_category=forced_collection_category,
            )
            for category_id in (
                self.VERIFIED_BANK_NEAR_RECOVERY,
                self.VERIFIED_BANK_FAR_RECOVERY,
            ):
                category_env_ids = env_ids[sampled_category_ids == int(category_id)]
                if category_env_ids.numel() == 0:
                    continue
                # CODEX TOP-LONG RESET: afar intentionally keeps the superclass
                # canonical/default reset, matching side distillation. Only recovery
                # categories override the reset with top-down IK samples.
                force_far = bool(category_id == self.VERIFIED_BANK_FAR_RECOVERY)
                if self._topdown_recovery_bank_ready and (not collection_enabled):
                    (
                        joint_config,
                        object_center_world,
                        object_quat_world,
                        _palm_pos,
                        _target_quat,
                        _is_far,
                    ) = self._sample_topdown_recovery_from_bank(category_env_ids, force_far=force_far)
                    self._apply_object_center_state(category_env_ids, object_center_world, object_quat_world)
                    self.reward_settings["object_init_height"][category_env_ids] = object_center_world[:, 2]
                    self._target_quat_latched[category_env_ids] = False
                else:
                    joint_config, _palm_pos, _target_quat = self._sample_topdown_recovery_joint_config(
                        category_env_ids,
                        force_far=force_far,
                    )
                self._apply_topdown_joint_reset(category_env_ids, joint_config)
            self.current_episode_verified_bank_category[env_ids] = sampled_category_ids

        if hasattr(self, "_verified_teacher_bank_snapshot_pending"):
            self._verified_teacher_bank_snapshot_pending[env_ids] = False
            self._verified_teacher_bank_snapshot_pending_category[env_ids] = -1
            if collection_enabled:
                self._verified_teacher_bank_snapshot_pending[env_ids] = True
                self._verified_teacher_bank_snapshot_pending_category[env_ids] = self.current_episode_verified_bank_category[
                    env_ids
                ]

        self.post_lift_target_active[env_ids] = False
        self._target_quat_latched[env_ids] = False
        self._resample_target_pos(env_ids, refresh_quat=True)

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

    def _resample_target_pos(self, env_ids, refresh_quat=False):
        if env_ids is None or env_ids.numel() == 0:
            return
        target_dtype = self.reward_settings["target_pos"].dtype
        target_pos = self._compute_post_lift_target_world(env_ids).to(dtype=target_dtype)
        # CODEX TOP-LONG TARGET: keep the distillation goal deterministic.
        # Match side distillation's base-relative goal construction instead of
        # tying the lift goal to object spawn XY. This makes the final goal clear
        # from robot proprioception and independent of object placement.
        final_target_pos = target_pos
        self.final_target_pos[env_ids] = final_target_pos
        self.target_waypoint_alpha[env_ids] = 1.0
        self.target_waypoint_active[env_ids] = False
        self.reward_settings["target_pos"][env_ids] = final_target_pos

        if refresh_quat:
            if self.use_goal_orientation_target:
                target_quat = self._compute_topdown_target_quat(env_ids)
            else:
                target_quat = normalize(self._eef_state[env_ids, 3:7])
            target_quat = target_quat.to(dtype=self.reward_settings["target_quat"].dtype)
            self.reward_settings["target_quat"][env_ids] = target_quat
            self.reward_settings["target_rot_6d"][env_ids] = matrix_to_rotation_6d(quaternion_to_matrix_ig(target_quat))
            self._target_quat_latched[env_ids] = True

    def _set_fixed_goal_target(self, env_ids, refresh_quat=False):
        # Backward-compatible wrapper for older debug paths; normal resets use
        # the deterministic base-relative target sampled in _resample_target_pos.
        self._resample_target_pos(env_ids, refresh_quat=refresh_quat)

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

    def _apply_target_waypoint_clipping(self):
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.final_target_pos[:] = self._compute_post_lift_target_world(all_env_ids).to(
            dtype=self.final_target_pos.dtype
        )
        if not self.use_target_waypoint:
            # Parent _update_states() rewrites reward_settings["target_pos"]
            # through obj_pos_target to the low lift-threshold target every
            # step. Restore the top-long final goal even when waypointing is off.
            self.reward_settings["target_pos"] = self.final_target_pos.to(dtype=self.reward_settings["target_pos"].dtype)
            self.target_waypoint_active[:] = False
            self.target_waypoint_alpha[:] = 1.0
            return
        eligible = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        if self.target_waypoint_until_lift:
            eligible = ~self.states["lift"]

        final_delta_world = self.final_target_pos - self._eef_state[:, :3]
        if self.teacher_use_eef_frame:
            eef_rot_mat_t = quaternion_to_matrix_ig(self._eef_state[:, 3:7]).transpose(1, 2)
            final_delta = torch.matmul(eef_rot_mat_t, final_delta_world.unsqueeze(-1)).squeeze(-1)
        else:
            final_delta = final_delta_world

        abs_max = self.target_waypoint_eef_delta_abs_max.to(
            device=self.device,
            dtype=final_delta.dtype,
        ).unsqueeze(0)
        ratio = abs_max / final_delta.abs().clamp_min(1.0e-6)
        alpha = torch.clamp(torch.min(ratio, dim=-1).values, max=1.0)
        needs_waypoint = eligible & (alpha < 1.0)
        waypoint_pos = self._eef_state[:, :3] + final_delta_world * alpha.unsqueeze(-1)
        self.reward_settings["target_pos"] = torch.where(
            needs_waypoint.unsqueeze(-1),
            waypoint_pos.to(dtype=self.reward_settings["target_pos"].dtype),
            self.final_target_pos.to(dtype=self.reward_settings["target_pos"].dtype),
        )
        self.target_waypoint_active[:] = needs_waypoint
        self.target_waypoint_alpha[:] = alpha.to(dtype=self.target_waypoint_alpha.dtype)

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

        if len(updated_env_ids) > 0:
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

        if hasattr(self, "post_lift_target_active"):
            self.post_lift_target_active[env_ids] = False
        if hasattr(self, "_target_quat_latched"):
            self._target_quat_latched[env_ids] = False
        if hasattr(self, "reward_settings") and "target_pos" in self.reward_settings:
            self._resample_target_pos(env_ids, refresh_quat=True)

    def _update_states(self):
        super()._update_states()

        if torch.any(self.object_reset_mask):
            reset_object_env_ids = self.object_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self.post_lift_target_active[reset_object_env_ids] = False

        if self.use_post_lift_base_target:
            self.post_lift_target_active = self.post_lift_target_active | self.states["lift"]
            if torch.any(self.post_lift_target_active):
                lifted_env_ids = self.post_lift_target_active.nonzero(as_tuple=False).squeeze(-1)
                self.reward_settings["target_pos"][lifted_env_ids] = self._compute_post_lift_target_world(lifted_env_ids)

        if self.enable_fabric:
            # CODEX TOP-LONG SWITCH: match side simple_anchor handoff: activate teacher inside
            # an object-centered radius, with a larger exit radius for hysteresis.
            teacher_active_prev = ~self.fabric_switch_enable
            switch_center_pos = self.states["object_center_pos"]
            switch_pos_err = torch.norm(self._eef_state[:, :3] - switch_center_pos, dim=-1)
            switch_radius = torch.full_like(switch_pos_err, float(self.switch_activate_radius))
            switch_radius[teacher_active_prev] = float(self.switch_activate_radius_exit)
            if self.switch_use_orientation_target:
                quat_dot = torch.abs(
                    torch.sum(normalize(self._eef_state[:, 3:7]) * normalize(self.switching_target_quat), dim=-1)
                ).clamp(max=1.0)
                switch_rot_err_deg = 2.0 * torch.rad2deg(torch.acos(quat_dot))
                switch_ready = (switch_pos_err < switch_radius) & (switch_rot_err_deg < self.switch_rot_tol_deg)
                self.extras["fabric/switch_rot_err_deg"] = torch.mean(switch_rot_err_deg).item()
            else:
                switch_ready = switch_pos_err < switch_radius
                self.extras["fabric/switch_rot_err_deg"] = 0.0
            teacher_active = switch_ready | self.states["lift"]
            self.fabric_switch_enable[:] = ~teacher_active
            self.fabric_switch_enable[self.progress_buf == 0] = True
            self.extras["fabric/switch_pos_err"] = torch.mean(switch_pos_err).item()
            self.extras["fabric/switch_target_pos_err"] = torch.mean(
                torch.norm(self._eef_state[:, :3] - self.switching_target_pos, dim=-1)
            ).item()

        # CODEX TOP-LONG TARGET: keep the RL-style reset target stable during the
        # episode unless the optional post-lift base target is explicitly enabled.
        self._apply_target_waypoint_clipping()
        if self.use_target_waypoint:
            self.extras["metrics/target_waypoint_active_frac"] = self.target_waypoint_active.float().mean().item()
            self.extras["metrics/target_waypoint_alpha_mean"] = self.target_waypoint_alpha.mean().item()
        self._refresh_target_state_components()

        local_z = torch.zeros((self.num_envs, 3), dtype=self._object_state.dtype, device=self.device)
        local_z[:, 2] = 1.0
        object_z_axis_world = quat_apply(self._object_state[:, 3:7], local_z)
        hand_z_axis_world = quat_apply(self._eef_state[:, 3:7], local_z)

        local_offset = torch.zeros((self.num_envs, 3), dtype=self._object_state.dtype, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[:, 2].to(dtype=self._object_state.dtype) * self.object_grasp_target_z_scale
        object_grasp_target_pos = self._object_state[:, :3] + quat_apply(self._object_state[:, 3:7], local_offset)
        flat_object_like = torch.abs(object_z_axis_world[:, 2]) <= self.flat_object_axis_z_abs_max
        object_grasp_target_pos = torch.where(
            flat_object_like.unsqueeze(-1),
            self.states["object_center_pos"],
            object_grasp_target_pos,
        )
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

        def append_circle(verts, colors, center, radius, color, plane="xy", segments=48):
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
            radius = float(radius)
            if radius <= 0.0:
                return
            for i in range(segments):
                a0 = 2.0 * math.pi * float(i) / float(segments)
                a1 = 2.0 * math.pi * float(i + 1) / float(segments)
                if plane == "xy":
                    p0 = (cx + radius * math.cos(a0), cy + radius * math.sin(a0), cz)
                    p1 = (cx + radius * math.cos(a1), cy + radius * math.sin(a1), cz)
                elif plane == "xz":
                    p0 = (cx + radius * math.cos(a0), cy, cz + radius * math.sin(a0))
                    p1 = (cx + radius * math.cos(a1), cy, cz + radius * math.sin(a1))
                else:
                    p0 = (cx, cy + radius * math.cos(a0), cz + radius * math.sin(a0))
                    p1 = (cx, cy + radius * math.cos(a1), cz + radius * math.sin(a1))
                append_line(verts, colors, p0, p1, color)

        def append_cross(verts, colors, center, length, color):
            append_line(
                verts,
                colors,
                center + torch.tensor([length, 0.0, 0.0], device=self.device, dtype=center.dtype),
                center - torch.tensor([length, 0.0, 0.0], device=self.device, dtype=center.dtype),
                color,
            )
            append_line(
                verts,
                colors,
                center + torch.tensor([0.0, length, 0.0], device=self.device, dtype=center.dtype),
                center - torch.tensor([0.0, length, 0.0], device=self.device, dtype=center.dtype),
                color,
            )
            append_line(
                verts,
                colors,
                center + torch.tensor([0.0, 0.0, length], device=self.device, dtype=center.dtype),
                center - torch.tensor([0.0, 0.0, length], device=self.device, dtype=center.dtype),
                color,
            )

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

            if self.use_post_lift_base_target:
                # White cross/line: optional post-lift goal target defined relative to the mobile base.
                post_lift_goal_pos = self._compute_post_lift_target_world(env_ids)[0]
                mobile_base_pos = self._get_current_mobile_base_pose7(env_ids)[0, :3]
                append_line(verts, colors, mobile_base_pos, post_lift_goal_pos, [1.0, 1.0, 1.0])
                append_line(
                    verts,
                    colors,
                    post_lift_goal_pos + torch.tensor([cross_len, 0.0, 0.0], device=self.device, dtype=dtype),
                    post_lift_goal_pos - torch.tensor([cross_len, 0.0, 0.0], device=self.device, dtype=dtype),
                    [1.0, 1.0, 1.0],
                )
                append_line(
                    verts,
                    colors,
                    post_lift_goal_pos + torch.tensor([0.0, cross_len, 0.0], device=self.device, dtype=dtype),
                    post_lift_goal_pos - torch.tensor([0.0, cross_len, 0.0], device=self.device, dtype=dtype),
                    [1.0, 1.0, 1.0],
                )
                append_line(
                    verts,
                    colors,
                    post_lift_goal_pos + torch.tensor([0.0, 0.0, cross_len], device=self.device, dtype=dtype),
                    post_lift_goal_pos - torch.tensor([0.0, 0.0, cross_len], device=self.device, dtype=dtype),
                    [1.0, 1.0, 1.0],
                )

            # Orange cross: final sampled target. Always draw it; waypoint mode
            # only changes whether the magenta active target is clipped before
            # reaching this final goal.
            final_target_pos = self.final_target_pos[env_id]
            final_cross_len = cross_len * 2.2
            append_cross(verts, colors, final_target_pos, final_cross_len, [1.0, 0.55, 0.0])

            # Magenta cross: currently active goal target. If waypointing is
            # active, this is the waypoint; otherwise it coincides with the final target.
            target_pos = self.reward_settings["target_pos"][env_id]
            append_cross(verts, colors, target_pos, cross_len, [1.0, 0.0, 1.0])

            if self.use_target_waypoint and bool(self.target_waypoint_active[env_id].item()):
                # Orange line: active waypoint to final target.
                final_target_pos = self.final_target_pos[env_id]
                append_line(verts, colors, target_pos, final_target_pos, [1.0, 0.55, 0.0])

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
