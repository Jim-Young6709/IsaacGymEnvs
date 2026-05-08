"""
Franka + LEAP Hand Pick Env
TODO:1. clean up
"""

import time
import json
import os

import hydra
import h5py
import isaacgym
import numpy as np
import torch
from curobo.types.math import Pose
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.pcd_utils import *
from isaacgymenvs.utils.rotation_conversions import *
from isaacgymenvs.tasks import FrankaLEAPMobileDistillation
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

class FrankaLEAPMobileDistillationPickSide(FrankaLEAPMobileDistillation):
    VERIFIED_BANK_AFAR = 0
    VERIFIED_BANK_NEAR_RECOVERY = 1
    VERIFIED_BANK_FAR_RECOVERY = 2
    VERIFIED_BANK_CATEGORY_NAMES = ("afar", "near_recovery", "far_recovery")

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # @ray unlike the rl env where we keep object spawn location fixed, let object spawn location vary
        # xyz_range = cfg["env"]["object_settings"]["xyz_range"]
        # avg_xyz = [
        #     0.5 * (xyz_range[0][0] + xyz_range[1][0]),
        #     0.5 * (xyz_range[0][1] + xyz_range[1][1]),
        #     0.5 * (xyz_range[0][2] + xyz_range[1][2]),
        # ]
        # cfg["env"]["object_settings"]["xyz_range"] = [avg_xyz, avg_xyz]
        self.object_grasp_target_z_scale = float(cfg["env"]["object_settings"]["object_grasp_target_z_scale"])
        self.use_center_tracking_switch_target = False  # CODEX
        self.fabric_switch_cfg = cfg["env"]["fabric_switch"]
        self.fabric_switch_mode = str(self.fabric_switch_cfg.get("mode", "staged"))
        if self.fabric_switch_mode not in ("staged", "simple_anchor"):
            raise ValueError(
                f"Unsupported env.fabric_switch.mode={self.fabric_switch_mode}. "
                "Expected one of: staged, simple_anchor."
            )
        self.switch_tol = float(self.fabric_switch_cfg["lateral_offset"])
        self.rl_target_xy_offset = torch.tensor([0.05, 0.0], dtype=torch.float32)
        reference_table_height = float(cfg["env"].get("rl_reference_table_surface_height", 0.175))
        self.rl_target_z_offset = float(cfg["reward"]["params"]["target_pos"][2]) - reference_table_height
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
        self._reset_upright_near_table_recovery_prob = float(
            cfg["env"]["eef_init"].get("upright_near_table_recovery_prob", 0.0)
        )
        self._reset_upright_near_table_recovery_prob_boot = self._reset_upright_near_table_recovery_prob
        self._reset_upright_near_table_recovery_prob = 0.0
        self._reset_upright_near_table_base_init_range_cfg = cfg["env"]["eef_init"].get(
            "upright_near_table_base_init_range",
            cfg["env"]["robot_init"]["base_init_range"],
        )
        self._reset_upright_near_table_palm_rel_object_min_cfg = cfg["env"]["eef_init"].get(
            "upright_near_table_palm_rel_object_min",
            [-0.20, -0.20, 0.00],
        )
        self._reset_upright_near_table_palm_rel_object_max_cfg = cfg["env"]["eef_init"].get(
            "upright_near_table_palm_rel_object_max",
            [0.20, 0.20, 0.20],
        )
        self._reset_upright_near_table_far_recovery_prob = float(
            cfg["env"]["eef_init"].get("upright_near_table_far_recovery_prob", 0.5)
        )
        self._reset_upright_near_table_palm_rel_object_far_min_cfg = cfg["env"]["eef_init"].get(
            "upright_near_table_palm_rel_object_far_min",
            [-0.40, -0.40, 0.00],
        )
        self._reset_upright_near_table_palm_rel_object_far_max_cfg = cfg["env"]["eef_init"].get(
            "upright_near_table_palm_rel_object_far_max",
            [0.40, 0.40, 0.20],
        )
        self._reset_upright_near_table_palm_rel_base_min_cfg = cfg["env"]["eef_init"].get(
            "upright_near_table_palm_rel_base_min",
            [0.20, -0.35, 0.02],
        )
        self._reset_upright_near_table_palm_rel_base_max_cfg = cfg["env"]["eef_init"].get(
            "upright_near_table_palm_rel_base_max",
            [0.65, 0.35, 0.16],
        )
        self._reset_upright_near_table_min_palm_object_dist = float(
            cfg["env"]["eef_init"].get("upright_near_table_min_palm_object_dist", 0.05)
        )
        self._reset_upright_near_table_max_resample_attempts = int(
            cfg["env"]["eef_init"].get("upright_near_table_max_resample_attempts", 32)
        )
        self._reset_upright_near_table_bank_size = int(
            cfg["env"]["eef_init"].get("upright_near_table_bank_size", 128)
        )
        self._reset_upright_near_table_bank_max_ik_goals = int(
            cfg["env"]["eef_init"].get("upright_near_table_bank_max_ik_goals", 2048)
        )
        self._reset_pitch_roll_noise_deg = float(cfg["env"]["eef_init"].get("pitch_roll_noise_deg", 45.0))
        self._reset_pitch_roll_noise_rad = float(np.deg2rad(self._reset_pitch_roll_noise_deg))
        teacher_state_bank_cfg = cfg.get("dagger", {}).get("teacher_state_bank", {})
        self._teacher_state_bank_collect_requested = bool(
            teacher_state_bank_cfg.get("collect_before_train", False)
            or teacher_state_bank_cfg.get("collect_only", False)
        )
        verified_bank_cfg = cfg["env"]["verified_teacher_bank"]
        self._verified_teacher_bank_enable = bool(verified_bank_cfg["enable"])
        self._verified_teacher_bank_hdf5_path = str(verified_bank_cfg["hdf5_path"])
        self._verified_teacher_bank_capacity_per_env = int(verified_bank_cfg["capacity_per_env"])
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
        self._verified_teacher_bank_far_recovery_hand_noise_scale = float(
            verified_bank_cfg.get("far_recovery_hand_noise_scale", 0.0)
        )
        self._verified_teacher_bank_far_recovery_hand_bank_size = int(
            verified_bank_cfg.get("far_recovery_hand_bank_size", 256)
        )
        self._verified_teacher_bank_far_recovery_hand_bank_candidate_batch_size = int(
            verified_bank_cfg.get("far_recovery_hand_bank_candidate_batch_size", 64)
        )
        self._verified_teacher_bank_far_recovery_hand_bank_max_rounds = int(
            verified_bank_cfg.get("far_recovery_hand_bank_max_rounds", 64)
        )
        self._verified_teacher_bank_far_recovery_hand_bank_points_per_link = int(
            verified_bank_cfg.get("far_recovery_hand_bank_points_per_link", 24)
        )
        self._verified_teacher_bank_far_recovery_hand_bank_min_link_distance = float(
            verified_bank_cfg.get("far_recovery_hand_bank_min_link_distance", 0.004)
        )
        self._verified_teacher_bank_sampling_probs_cfg = verified_bank_cfg["sampling_probs"]
        self._verified_teacher_bank_sampling_probs = torch.tensor(
            [
                float(self._verified_teacher_bank_sampling_probs_cfg["afar"]),
                float(self._verified_teacher_bank_sampling_probs_cfg["near_recovery"]),
                float(self._verified_teacher_bank_sampling_probs_cfg["far_recovery"]),
            ],
            dtype=torch.float32,
        )
        activation_snapshot_bank_cfg = cfg["env"].get("activation_snapshot_bank", {})
        self._activation_snapshot_bank_enable = bool(activation_snapshot_bank_cfg.get("enable", False))
        self._activation_snapshot_bank_hdf5_path = str(activation_snapshot_bank_cfg.get("hdf5_path", "") or "")
        self._activation_snapshot_bank_capacity_per_env = int(
            activation_snapshot_bank_cfg.get("capacity_per_env", 0)
        )
        if self._activation_snapshot_bank_enable:
            if not self._activation_snapshot_bank_hdf5_path:
                raise ValueError("activation_snapshot_bank.enable=True but hdf5_path is empty")
            if self._activation_snapshot_bank_capacity_per_env <= 0:
                raise ValueError(
                    "activation_snapshot_bank.enable=True requires capacity_per_env > 0"
                )
        self._recovery_reset_bank_ready = False
        self._verified_teacher_bank_loaded = False
        self._verified_teacher_bank_collection_enabled = False
        self._verified_teacher_bank_collection_category = -1
        # cuRobo IK solves for panda_link7, while the near-table reset sampler reasons
        # about the palm center. Convert palm targets to link7 targets with the fixed URDF offset.
        self._palm_center_from_link7_local = torch.tensor([0.0, 0.0, 0.115], dtype=torch.float32)
        # Fixed URDF mount from tidybot2_base_link to panda_link0.
        self._franka_mount_offset_from_mobile_base = torch.tensor([0.178, 0.0, 0.444775], dtype=torch.float32)
        self.bad_state_xy_margin = float(self.fabric_switch_cfg.get("bad_state_xy_margin", 0.05))
        self.debug_teacher_state_stats_enabled = bool(cfg["env"].get("enableDebugVis", False))
        if self.debug_teacher_state_stats_enabled:
            self.debug_teacher_state_stats_print_freq = int(os.getenv("DEBUG_TEACHER_STATE_STATS_PRINT_FREQ", "500"))
            self.debug_teacher_state_stats_path = os.path.join(
                os.getcwd(),
                f"debug_teacher_state_stats_{os.getpid()}.json",
            )
            self.debug_teacher_state_stats_raw_path = os.path.join(
                os.getcwd(),
                f"debug_teacher_state_stats_raw_{os.getpid()}.jsonl",
            )
            self.debug_teacher_state_stats_raw_stride = max(1, int(os.getenv("DEBUG_TEACHER_STATE_STATS_RAW_STRIDE", "1")))
            self.debug_teacher_state_stats_raw_max_records = int(os.getenv("DEBUG_TEACHER_STATE_STATS_RAW_MAX_RECORDS", "200000"))
            self._debug_teacher_state_stats = self._init_debug_teacher_state_stats()
        else:
            self.debug_teacher_state_stats_print_freq = 0
            self.debug_teacher_state_stats_path = None
            self.debug_teacher_state_stats_raw_path = None
            self.debug_teacher_state_stats_raw_stride = 1
            self.debug_teacher_state_stats_raw_max_records = 0
            self._debug_teacher_state_stats = None
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )
        self._reset_upright_near_table_recovery_prob = self._reset_upright_near_table_recovery_prob_boot
        should_build_recovery_bank = (
            self._reset_upright_near_table_recovery_prob > 0.0
            and not (
                self._verified_teacher_bank_enable
                and self._teacher_state_bank_collect_requested
            )
        )
        if should_build_recovery_bank:
            self._init_upright_near_table_recovery_bank()
            self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._init_far_recovery_safe_hand_joint_bank()

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

    def _get_clipped_recovery_palm_rel_object_bounds(
        self,
        object_center,
        table_xy_min,
        table_xy_max,
        dtype,
        rel_object_min_cfg=None,
        rel_object_max_cfg=None,
    ):
        if rel_object_min_cfg is None:
            rel_object_min_cfg = self._reset_upright_near_table_palm_rel_object_min_cfg
        if rel_object_max_cfg is None:
            rel_object_max_cfg = self._reset_upright_near_table_palm_rel_object_max_cfg
        palm_rel_object_min = torch.tensor(
            rel_object_min_cfg,
            device=self.device,
            dtype=dtype,
        )
        palm_rel_object_max = torch.tensor(
            rel_object_max_cfg,
            device=self.device,
            dtype=dtype,
        )
        batch = int(object_center.shape[0])
        rel_min = palm_rel_object_min.unsqueeze(0).repeat(batch, 1)
        rel_max = palm_rel_object_max.unsqueeze(0).repeat(batch, 1)
        rel_min[:, :2] = torch.maximum(rel_min[:, :2], table_xy_min - object_center[:, :2])
        rel_max[:, :2] = torch.minimum(rel_max[:, :2], table_xy_max - object_center[:, :2])
        valid_xy = torch.all(rel_min[:, :2] <= rel_max[:, :2], dim=-1)
        return rel_min, rel_max, valid_xy

    def _filter_recovery_robot_object_penetration(self, joint_config, object_center_world, env_ids):
        if joint_config.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=torch.bool)

        robot_pcd_world = self.robot_pcd_sampler.sample(joint_config, self.torchurdf_to_isaac_idx)
        bbox_center = object_center_world
        bbox_half_extents = 0.5 * self.mesh_aabb_extents[env_ids].to(dtype=joint_config.dtype)
        bbox_half_extents = bbox_half_extents + 0.01
        rel = torch.abs(robot_pcd_world - bbox_center.unsqueeze(1))
        inside = torch.all(rel <= bbox_half_extents.unsqueeze(1), dim=-1)
        penetration = torch.any(inside, dim=1)
        return ~penetration

    def _get_franka_base_pose7_from_mobile_base_pose(self, mobile_base_pose, dtype=None):
        if dtype is None:
            dtype = mobile_base_pose.dtype
        if mobile_base_pose.numel() == 0:
            return torch.empty((0, 7), device=self.device, dtype=dtype)

        yaw = mobile_base_pose[:, 2]
        half_yaw = 0.5 * yaw
        mobile_quat = torch.zeros((mobile_base_pose.shape[0], 4), device=self.device, dtype=dtype)
        mobile_quat[:, 2] = torch.sin(half_yaw)
        mobile_quat[:, 3] = torch.cos(half_yaw)

        mobile_base_pos_world = torch.zeros((mobile_base_pose.shape[0], 3), device=self.device, dtype=dtype)
        mobile_base_pos_world[:, :2] = mobile_base_pose[:, :2]
        mount_offset = self._franka_mount_offset_from_mobile_base.to(
            device=self.device,
            dtype=dtype,
        ).unsqueeze(0).repeat(mobile_base_pose.shape[0], 1)
        franka_base_pos_world = mobile_base_pos_world + quat_apply(mobile_quat, mount_offset)
        return torch.cat([franka_base_pos_world, mobile_quat], dim=-1)

    def _filter_recovery_ik_realization(
        self,
        arm_q,
        target_palm_pos_world,
        target_palm_quat_world,
        franka_base_pose7,
    ):
        if arm_q.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=torch.bool)

        arm_fk = self.get_ee_from_joint(arm_q)
        link7_pos_local = arm_fk[:, :3]
        link7_quat_local = arm_fk[:, 3:7]

        palm_offset_local = self._palm_center_from_link7_local.to(device=self.device, dtype=arm_q.dtype)
        palm_pos_local = link7_pos_local + quat_apply(link7_quat_local, palm_offset_local.unsqueeze(0).repeat(arm_q.shape[0], 1))
        palm_quat_local = link7_quat_local

        base_pos_world = franka_base_pose7[:, :3]
        base_quat = franka_base_pose7[:, 3:7]

        realized_palm_pos_world = base_pos_world + quat_apply(base_quat, palm_pos_local)
        realized_palm_quat_world = quat_mul(base_quat, palm_quat_local)
        pos_err = torch.norm(realized_palm_pos_world - target_palm_pos_world, dim=-1)
        quat_dot = torch.abs(torch.sum(realized_palm_quat_world * target_palm_quat_world, dim=-1))
        quat_err = 1.0 - quat_dot

        return (pos_err <= 0.02) & (quat_err <= 0.05)

    def _get_verified_teacher_bank_shard_path(self, rank=0):
        base_path = self._verified_teacher_bank_hdf5_path
        root, ext = os.path.splitext(base_path)
        if ext == "":
            ext = ".hdf5"
        return f"{root}_rank{int(rank)}{ext}"

    def _get_activation_snapshot_bank_shard_path(self, rank=0):
        base_path = self._activation_snapshot_bank_hdf5_path
        root, ext = os.path.splitext(base_path)
        if ext == "":
            ext = ".hdf5"
        return f"{root}_rank{int(rank)}{ext}"

    def _verified_teacher_bank_counts_tensor(self):
        return self._verified_teacher_bank_counts_cpu.to(device=self.device, dtype=torch.int64)

    def get_verified_teacher_bank_category_counts_tensor(self, category_id):
        local_counts = self._verified_teacher_bank_counts_cpu[int(category_id)].to(
            device=self.device,
            dtype=torch.int64,
        )
        variation_ids = getattr(self, "teacher_bank_variation_ids", None)
        num_variants_global = int(getattr(self, "teacher_bank_num_variants_global", int(self.num_envs)))
        if variation_ids is None or variation_ids.numel() != local_counts.numel():
            return local_counts

        global_counts = torch.zeros(
            (num_variants_global,),
            dtype=torch.int64,
            device=self.device,
        )
        global_counts.scatter_add_(0, variation_ids.to(device=self.device, dtype=torch.long), local_counts)
        return global_counts

    def get_verified_teacher_bank_category_correct_counts_tensor(self, category_id):
        valid_counts = self._verified_teacher_bank_counts_cpu[int(category_id)].to(
            device=self.device,
            dtype=torch.long,
        )
        correct_region = self._verified_teacher_bank_correct_region_cpu[int(category_id)].to(
            device=self.device,
            dtype=torch.long,
        )
        slot_ids = torch.arange(correct_region.shape[1], device=self.device, dtype=torch.long).unsqueeze(0)
        local_correct_counts = (correct_region * (slot_ids < valid_counts.unsqueeze(1)).to(dtype=torch.long)).sum(dim=1)

        variation_ids = getattr(self, "teacher_bank_variation_ids", None)
        num_variants_global = int(getattr(self, "teacher_bank_num_variants_global", int(self.num_envs)))
        if variation_ids is None or variation_ids.numel() != local_correct_counts.numel():
            return local_correct_counts.to(dtype=torch.int64)

        global_counts = torch.zeros(
            (num_variants_global,),
            dtype=torch.int64,
            device=self.device,
        )
        global_counts.scatter_add_(
            0,
            variation_ids.to(device=self.device, dtype=torch.long),
            local_correct_counts.to(dtype=torch.int64),
        )
        return global_counts

    def _compute_correct_region_mask(self, eef_pos, object_center_world, side_is_left):
        dy = eef_pos[:, 1] - object_center_world[:, 1]
        margin = float(self._verified_teacher_bank_correct_region_margin)
        # "Correct" means the intended grasp side. The opposite side is the
        # cross-object recovery side, which is usually the harder side.
        return torch.where(side_is_left, dy >= margin, dy <= -margin)

    def _verified_teacher_bank_collection_region_match_bool(self, is_correct_region):
        region_filter = self._verified_teacher_bank_collection_region_filter
        if region_filter == "all":
            return True
        if region_filter == "correct_only":
            return bool(is_correct_region)
        if region_filter == "wrong_only":
            return not bool(is_correct_region)
        raise ValueError(f"Unsupported verified teacher bank collection_region_filter={region_filter}")

    def _apply_collection_region_filter_to_recovery_bounds(self, rel_min, rel_max, side_is_left):
        region_filter = self._verified_teacher_bank_collection_region_filter
        if region_filter == "all":
            return rel_min, rel_max, torch.ones((rel_min.shape[0],), device=rel_min.device, dtype=torch.bool)

        rel_min = rel_min.clone()
        rel_max = rel_max.clone()
        side_is_left = side_is_left.to(device=rel_min.device, dtype=torch.bool)
        margin = float(self._verified_teacher_bank_correct_region_margin)
        margin_tensor = torch.full((rel_min.shape[0],), margin, device=rel_min.device, dtype=rel_min.dtype)
        neg_margin_tensor = -margin_tensor

        if region_filter == "correct_only":
            if bool(torch.any(side_is_left)):
                rel_min[side_is_left, 1] = torch.maximum(rel_min[side_is_left, 1], margin_tensor[side_is_left])
            right_mask = ~side_is_left
            if bool(torch.any(right_mask)):
                rel_max[right_mask, 1] = torch.minimum(rel_max[right_mask, 1], neg_margin_tensor[right_mask])
        elif region_filter == "wrong_only":
            if bool(torch.any(side_is_left)):
                rel_max[side_is_left, 1] = torch.minimum(rel_max[side_is_left, 1], neg_margin_tensor[side_is_left])
            right_mask = ~side_is_left
            if bool(torch.any(right_mask)):
                rel_min[right_mask, 1] = torch.maximum(rel_min[right_mask, 1], margin_tensor[right_mask])
        else:
            raise ValueError(f"Unsupported verified teacher bank collection_region_filter={region_filter}")

        valid_region = torch.all(rel_min <= rel_max, dim=-1)
        return rel_min, rel_max, valid_region

    def _compute_directional_teleport_y_interval(self, global_env_ids, xy_min_y, xy_max_y, prev_y):
        region_filter = self._verified_teacher_bank_collection_region_filter
        if region_filter not in ("correct_only", "wrong_only"):
            return None

        global_env_ids = global_env_ids.to(device=self.device, dtype=torch.long)
        side_is_left = self.side_is_left[global_env_ids].to(dtype=torch.bool)
        eps = torch.full_like(prev_y, 1e-4)

        if region_filter == "correct_only":
            y_lo = torch.where(side_is_left, xy_min_y, torch.maximum(xy_min_y, prev_y + eps))
            y_hi = torch.where(side_is_left, torch.minimum(xy_max_y, prev_y - eps), xy_max_y)
        else:
            y_lo = torch.where(side_is_left, torch.maximum(xy_min_y, prev_y + eps), xy_min_y)
            y_hi = torch.where(side_is_left, xy_max_y, torch.minimum(xy_max_y, prev_y - eps))

        return y_lo, y_hi

    def _init_verified_teacher_bank_storage(self):
        capacity = int(self._verified_teacher_bank_capacity_per_env)
        num_categories = len(self.VERIFIED_BANK_CATEGORY_NAMES)
        dtype = self._q.dtype
        self._verified_teacher_bank_counts_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._verified_teacher_bank_failure_counts_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._verified_teacher_bank_attempt_counts_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._verified_teacher_bank_success_counts_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._verified_teacher_bank_joint_config_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, self.num_dofs), dtype=dtype, device="cpu"
        )
        self._verified_teacher_bank_failure_joint_config_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, self.num_dofs), dtype=dtype, device="cpu"
        )
        self._verified_teacher_bank_object_center_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 3), dtype=dtype, device="cpu"
        )
        self._verified_teacher_bank_failure_object_center_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 3), dtype=dtype, device="cpu"
        )
        self._verified_teacher_bank_object_quat_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 4), dtype=dtype, device="cpu"
        )
        self._verified_teacher_bank_failure_object_quat_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 4), dtype=dtype, device="cpu"
        )
        self._verified_teacher_bank_side_is_left_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.bool, device="cpu"
        )
        self._verified_teacher_bank_failure_side_is_left_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.bool, device="cpu"
        )
        self._verified_teacher_bank_correct_region_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.bool, device="cpu"
        )
        self._verified_teacher_bank_failure_correct_region_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.bool, device="cpu"
        )
        self._verified_teacher_bank_episode_steps_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.int32, device="cpu"
        )
        self._verified_teacher_bank_failure_episode_steps_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.int32, device="cpu"
        )
        self._verified_teacher_bank_attempt_episode_step_sum_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int64, device="cpu"
        )
        self._verified_teacher_bank_success_episode_step_sum_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int64, device="cpu"
        )
        self._verified_teacher_bank_failure_episode_step_sum_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int64, device="cpu"
        )
        self._verified_teacher_bank_attempt_episode_step_max_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._verified_teacher_bank_success_episode_step_max_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._verified_teacher_bank_failure_episode_step_max_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._verified_teacher_bank_episode_active = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._verified_teacher_bank_episode_category = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._verified_teacher_bank_episode_joint_config = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=dtype, device=self.device
        )
        self._verified_teacher_bank_episode_object_center_world = torch.zeros(
            (self.num_envs, 3), dtype=dtype, device=self.device
        )
        self._verified_teacher_bank_episode_object_quat_world = torch.zeros(
            (self.num_envs, 4), dtype=dtype, device=self.device
        )
        self._verified_teacher_bank_episode_object_quat_world[:, 3] = 1.0
        self._verified_teacher_bank_episode_side_is_left = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._verified_teacher_bank_episode_correct_region = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._verified_teacher_bank_episode_start_progress = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._verified_teacher_bank_snapshot_pending = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._verified_teacher_bank_snapshot_pending_category = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

    def _init_activation_snapshot_bank_storage(self):
        capacity = int(self._activation_snapshot_bank_capacity_per_env)
        num_categories = len(self.VERIFIED_BANK_CATEGORY_NAMES)
        dtype = self._q.dtype
        self._activation_snapshot_bank_counts_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._activation_snapshot_bank_failure_counts_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._activation_snapshot_bank_event_counts_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._activation_snapshot_bank_joint_config_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, self.num_dofs), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_failure_joint_config_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, self.num_dofs), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_object_center_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 3), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_failure_object_center_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 3), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_object_quat_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 4), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_failure_object_quat_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 4), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_side_is_left_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.bool, device="cpu"
        )
        self._activation_snapshot_bank_failure_side_is_left_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.bool, device="cpu"
        )
        self._activation_snapshot_bank_correct_region_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.bool, device="cpu"
        )
        self._activation_snapshot_bank_failure_correct_region_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.bool, device="cpu"
        )
        self._activation_snapshot_bank_table_pos_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 3), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_failure_table_pos_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 3), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_table_quat_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 4), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_table_quat_world_cpu[..., 3] = 1.0
        self._activation_snapshot_bank_failure_table_quat_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 4), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_failure_table_quat_world_cpu[..., 3] = 1.0
        self._activation_snapshot_bank_target_quat_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 4), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_target_quat_world_cpu[..., 3] = 1.0
        self._activation_snapshot_bank_failure_target_quat_world_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity, 4), dtype=dtype, device="cpu"
        )
        self._activation_snapshot_bank_failure_target_quat_world_cpu[..., 3] = 1.0
        self._activation_snapshot_bank_episode_steps_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.int32, device="cpu"
        )
        self._activation_snapshot_bank_failure_episode_steps_cpu = torch.zeros(
            (num_categories, self.num_envs, capacity), dtype=torch.int32, device="cpu"
        )
        self._activation_snapshot_bank_event_step_sum_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int64, device="cpu"
        )
        self._activation_snapshot_bank_success_step_sum_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int64, device="cpu"
        )
        self._activation_snapshot_bank_failure_step_sum_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int64, device="cpu"
        )
        self._activation_snapshot_bank_event_step_max_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._activation_snapshot_bank_success_step_max_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._activation_snapshot_bank_failure_step_max_cpu = torch.zeros(
            (num_categories, self.num_envs), dtype=torch.int32, device="cpu"
        )
        self._activation_snapshot_bank_episode_active = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._activation_snapshot_bank_episode_category = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._activation_snapshot_bank_episode_joint_config = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=dtype, device=self.device
        )
        self._activation_snapshot_bank_episode_object_center_world = torch.zeros(
            (self.num_envs, 3), dtype=dtype, device=self.device
        )
        self._activation_snapshot_bank_episode_object_quat_world = torch.zeros(
            (self.num_envs, 4), dtype=dtype, device=self.device
        )
        self._activation_snapshot_bank_episode_object_quat_world[:, 3] = 1.0
        self._activation_snapshot_bank_episode_side_is_left = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._activation_snapshot_bank_episode_correct_region = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device
        )
        self._activation_snapshot_bank_episode_table_pos_world = torch.zeros(
            (self.num_envs, 3), dtype=dtype, device=self.device
        )
        self._activation_snapshot_bank_episode_table_quat_world = torch.zeros(
            (self.num_envs, 4), dtype=dtype, device=self.device
        )
        self._activation_snapshot_bank_episode_table_quat_world[:, 3] = 1.0
        self._activation_snapshot_bank_episode_target_quat_world = torch.zeros(
            (self.num_envs, 4), dtype=dtype, device=self.device
        )
        self._activation_snapshot_bank_episode_target_quat_world[:, 3] = 1.0
        self._activation_snapshot_bank_episode_start_progress = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )

    def get_activation_snapshot_bank_count_summary(self):
        if not hasattr(self, "_activation_snapshot_bank_counts_cpu"):
            return {
                name: {
                    "stored_total": 0,
                    "failure_stored_total": 0,
                    "stored_min_per_env": 0,
                    "stored_max_per_env": 0,
                    "event_total": 0,
                    "event_mean_steps": 0.0,
                    "success_mean_steps": 0.0,
                    "failure_mean_steps": 0.0,
                    "event_max_steps": 0,
                }
                for name in self.VERIFIED_BANK_CATEGORY_NAMES
            }

        counts = self._activation_snapshot_bank_counts_cpu.clone()
        failure_counts = self._activation_snapshot_bank_failure_counts_cpu.clone()
        event_counts = self._activation_snapshot_bank_event_counts_cpu.clone()
        event_step_sums = self._activation_snapshot_bank_event_step_sum_cpu.clone()
        success_step_sums = self._activation_snapshot_bank_success_step_sum_cpu.clone()
        failure_step_sums = self._activation_snapshot_bank_failure_step_sum_cpu.clone()
        event_step_max = self._activation_snapshot_bank_event_step_max_cpu.clone()
        out = {}
        for category_id, name in enumerate(self.VERIFIED_BANK_CATEGORY_NAMES):
            event_total = int(event_counts[category_id].sum().item())
            success_total = int(counts[category_id].sum().item())
            failure_total = int(failure_counts[category_id].sum().item())
            out[name] = {
                "stored_total": success_total,
                "failure_stored_total": failure_total,
                "stored_min_per_env": int(counts[category_id].min().item()),
                "stored_max_per_env": int(counts[category_id].max().item()),
                "correct_stored_total": int(
                    self._activation_snapshot_bank_correct_region_cpu[category_id].sum().item()
                ),
                "event_total": event_total,
                "event_mean_steps": (
                    float(event_step_sums[category_id].sum().item()) / float(event_total)
                ) if event_total > 0 else 0.0,
                "success_mean_steps": (
                    float(success_step_sums[category_id].sum().item()) / float(success_total)
                ) if success_total > 0 else 0.0,
                "failure_mean_steps": (
                    float(failure_step_sums[category_id].sum().item()) / float(failure_total)
                ) if failure_total > 0 else 0.0,
                "event_max_steps": int(event_step_max[category_id].max().item()) if event_step_max.shape[1] > 0 else 0,
            }
        return out

    def save_activation_snapshot_bank_hdf5(self, rank=0):
        if not self._activation_snapshot_bank_enable:
            return
        if not hasattr(self, "_activation_snapshot_bank_counts_cpu"):
            return

        shard_path = self._get_activation_snapshot_bank_shard_path(rank=rank)
        os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
        with h5py.File(shard_path, "w") as f:
            f.attrs["num_envs"] = int(self.num_envs)
            f.attrs["num_dofs"] = int(self.num_dofs)
            f.attrs["capacity_per_env"] = int(self._activation_snapshot_bank_capacity_per_env)
            f.attrs["category_names_json"] = json.dumps(list(self.VERIFIED_BANK_CATEGORY_NAMES))
            f.attrs["num_variants_global"] = int(
                getattr(self, "teacher_bank_num_variants_global", int(self.num_envs))
            )
            f.create_dataset("counts", data=self._activation_snapshot_bank_counts_cpu.numpy())
            f.create_dataset("failure_counts", data=self._activation_snapshot_bank_failure_counts_cpu.numpy())
            f.create_dataset("activation_counts", data=self._activation_snapshot_bank_event_counts_cpu.numpy())
            f.create_dataset("episode_steps", data=self._activation_snapshot_bank_episode_steps_cpu.numpy())
            f.create_dataset("failure_episode_steps", data=self._activation_snapshot_bank_failure_episode_steps_cpu.numpy())
            f.create_dataset("event_episode_step_sum", data=self._activation_snapshot_bank_event_step_sum_cpu.numpy())
            f.create_dataset("success_episode_step_sum", data=self._activation_snapshot_bank_success_step_sum_cpu.numpy())
            f.create_dataset("failure_episode_step_sum", data=self._activation_snapshot_bank_failure_step_sum_cpu.numpy())
            f.create_dataset("event_episode_step_max", data=self._activation_snapshot_bank_event_step_max_cpu.numpy())
            f.create_dataset("success_episode_step_max", data=self._activation_snapshot_bank_success_step_max_cpu.numpy())
            f.create_dataset("failure_episode_step_max", data=self._activation_snapshot_bank_failure_step_max_cpu.numpy())
            f.create_dataset("joint_config", data=self._activation_snapshot_bank_joint_config_cpu.numpy())
            f.create_dataset(
                "failure_joint_config",
                data=self._activation_snapshot_bank_failure_joint_config_cpu.numpy(),
            )
            f.create_dataset("object_center_world", data=self._activation_snapshot_bank_object_center_world_cpu.numpy())
            f.create_dataset(
                "failure_object_center_world",
                data=self._activation_snapshot_bank_failure_object_center_world_cpu.numpy(),
            )
            f.create_dataset("object_quat_world", data=self._activation_snapshot_bank_object_quat_world_cpu.numpy())
            f.create_dataset(
                "failure_object_quat_world",
                data=self._activation_snapshot_bank_failure_object_quat_world_cpu.numpy(),
            )
            f.create_dataset(
                "side_is_left",
                data=self._activation_snapshot_bank_side_is_left_cpu.numpy().astype(np.uint8),
            )
            f.create_dataset(
                "correct_region",
                data=self._activation_snapshot_bank_correct_region_cpu.numpy().astype(np.uint8),
            )
            f.create_dataset(
                "failure_side_is_left",
                data=self._activation_snapshot_bank_failure_side_is_left_cpu.numpy().astype(np.uint8),
            )
            f.create_dataset(
                "failure_correct_region",
                data=self._activation_snapshot_bank_failure_correct_region_cpu.numpy().astype(np.uint8),
            )
            f.create_dataset("table_pos_world", data=self._activation_snapshot_bank_table_pos_world_cpu.numpy())
            f.create_dataset(
                "failure_table_pos_world",
                data=self._activation_snapshot_bank_failure_table_pos_world_cpu.numpy(),
            )
            f.create_dataset("table_quat_world", data=self._activation_snapshot_bank_table_quat_world_cpu.numpy())
            f.create_dataset(
                "failure_table_quat_world",
                data=self._activation_snapshot_bank_failure_table_quat_world_cpu.numpy(),
            )
            f.create_dataset("target_quat_world", data=self._activation_snapshot_bank_target_quat_world_cpu.numpy())
            f.create_dataset(
                "failure_target_quat_world",
                data=self._activation_snapshot_bank_failure_target_quat_world_cpu.numpy(),
            )
            f.create_dataset("env_object_ids", data=self.env_object_ids.detach().cpu().numpy())
            if getattr(self, "teacher_bank_variation_ids", None) is not None:
                f.create_dataset("variation_id", data=self.teacher_bank_variation_ids.detach().cpu().numpy())
            if hasattr(self, "teacher_bank_height_bin_idx"):
                f.create_dataset("height_bin_idx", data=self.teacher_bank_height_bin_idx.detach().cpu().numpy())
            f.create_dataset("table_surface_height", data=self.table_surface_height.detach().cpu().numpy())
            f.create_dataset("table_size", data=self.table_size.detach().cpu().numpy())
            if hasattr(self, "object_mass"):
                f.create_dataset("object_mass", data=self.object_mass.detach().cpu().numpy())

    def record_activation_snapshot_bank_entries(self, env_ids, category_id):
        if (
            (not self._activation_snapshot_bank_enable)
            or env_ids.numel() == 0
            or int(category_id) < 0
        ):
            return
        if not hasattr(self, "_activation_snapshot_bank_counts_cpu"):
            self._init_activation_snapshot_bank_storage()

        category_id = int(category_id)
        record_env_ids = env_ids[~self._activation_snapshot_bank_episode_active[env_ids]]
        if record_env_ids.numel() == 0:
            return
        self._activation_snapshot_bank_episode_active[record_env_ids] = True
        self._activation_snapshot_bank_episode_category[record_env_ids] = category_id
        self._activation_snapshot_bank_episode_joint_config[record_env_ids] = self._q[record_env_ids].clone()
        self._activation_snapshot_bank_episode_object_center_world[record_env_ids] = self.states["object_center_pos"][
            record_env_ids
        ].clone()
        self._activation_snapshot_bank_episode_object_quat_world[record_env_ids] = self._object_state[
            record_env_ids, 3:7
        ].clone()
        self._activation_snapshot_bank_episode_side_is_left[record_env_ids] = self.side_is_left[record_env_ids].clone()
        self._activation_snapshot_bank_episode_correct_region[record_env_ids] = self._compute_correct_region_mask(
            self.states["eef_pos"][record_env_ids],
            self.states["object_center_pos"][record_env_ids],
            self.side_is_left[record_env_ids],
        )
        self._activation_snapshot_bank_episode_table_pos_world[record_env_ids] = self.table_pos[record_env_ids].clone()
        if hasattr(self, "cuboid_quats") and self.cuboid_quats is not None:
            table_quat_world = self.cuboid_quats[record_env_ids, 0].to(
                device=self._activation_snapshot_bank_episode_table_quat_world.device,
                dtype=self._activation_snapshot_bank_episode_table_quat_world.dtype,
            )
            self._activation_snapshot_bank_episode_table_quat_world[record_env_ids] = table_quat_world.clone()
        else:
            self._activation_snapshot_bank_episode_table_quat_world[record_env_ids] = 0.0
            self._activation_snapshot_bank_episode_table_quat_world[record_env_ids, 3] = 1.0
        self._activation_snapshot_bank_episode_target_quat_world[record_env_ids] = self.reward_settings["target_quat"][
            record_env_ids
        ].clone()
        self._activation_snapshot_bank_episode_start_progress[record_env_ids] = self.progress_buf[record_env_ids].clone()

    def configure_verified_teacher_bank_collection(self, enabled, category_name=None):
        self._verified_teacher_bank_collection_enabled = bool(enabled)
        if category_name is None:
            self._verified_teacher_bank_collection_category = -1
            return
        if category_name not in self.VERIFIED_BANK_CATEGORY_NAMES:
            raise ValueError(
                f"Unsupported verified teacher bank category {category_name}. "
                f"Expected one of {self.VERIFIED_BANK_CATEGORY_NAMES}."
            )
        self._verified_teacher_bank_collection_category = int(
            self.VERIFIED_BANK_CATEGORY_NAMES.index(category_name)
        )

    def _verified_teacher_bank_category_full(self, category_id):
        counts = self._verified_teacher_bank_counts_cpu[category_id]
        return bool(torch.all(counts >= int(self._verified_teacher_bank_capacity_per_env)).item())

    def get_verified_teacher_bank_count_summary(self):
        counts = self._verified_teacher_bank_counts_cpu.clone()
        failure_counts = self._verified_teacher_bank_failure_counts_cpu.clone()
        attempts = self._verified_teacher_bank_attempt_counts_cpu.clone()
        successes = self._verified_teacher_bank_success_counts_cpu.clone()
        attempt_step_sums = self._verified_teacher_bank_attempt_episode_step_sum_cpu.clone()
        success_step_sums = self._verified_teacher_bank_success_episode_step_sum_cpu.clone()
        failure_step_sums = self._verified_teacher_bank_failure_episode_step_sum_cpu.clone()
        attempt_step_max = self._verified_teacher_bank_attempt_episode_step_max_cpu.clone()
        out = {}
        for category_id, name in enumerate(self.VERIFIED_BANK_CATEGORY_NAMES):
            attempt_total = int(attempts[category_id].sum().item())
            success_total = int(successes[category_id].sum().item())
            failure_stored_total = int(failure_counts[category_id].sum().item())
            failure_total = max(attempt_total - success_total, 0)
            out[name] = {
                "stored_total": int(counts[category_id].sum().item()),
                "failure_stored_total": failure_stored_total,
                "failure_total": failure_total,
                "stored_min_per_env": int(counts[category_id].min().item()),
                "stored_max_per_env": int(counts[category_id].max().item()),
                "correct_stored_total": int(self._verified_teacher_bank_correct_region_cpu[category_id].sum().item()),
                "correct_stored_min_per_env": int(
                    self._verified_teacher_bank_correct_region_cpu[category_id].sum(dim=1).min().item()
                ),
                "correct_stored_max_per_env": int(
                    self._verified_teacher_bank_correct_region_cpu[category_id].sum(dim=1).max().item()
                ),
                "attempt_total": attempt_total,
                "success_total": success_total,
                "success_rate": (float(success_total) / float(attempt_total)) if attempt_total > 0 else 0.0,
                "attempt_mean_steps": (
                    float(attempt_step_sums[category_id].sum().item()) / float(attempt_total)
                ) if attempt_total > 0 else 0.0,
                "success_mean_steps": (
                    float(success_step_sums[category_id].sum().item()) / float(success_total)
                ) if success_total > 0 else 0.0,
                "failure_mean_steps": (
                    float(failure_step_sums[category_id].sum().item()) / float(failure_total)
                ) if failure_total > 0 else 0.0,
                "attempt_max_steps": int(attempt_step_max[category_id].max().item()) if attempt_step_max.shape[1] > 0 else 0,
            }
        return out

    def save_verified_teacher_bank_hdf5(self, rank=0):
        shard_path = self._get_verified_teacher_bank_shard_path(rank=rank)
        os.makedirs(os.path.dirname(shard_path) or ".", exist_ok=True)
        with h5py.File(shard_path, "w") as f:
            f.attrs["num_envs"] = int(self.num_envs)
            f.attrs["num_dofs"] = int(self.num_dofs)
            f.attrs["capacity_per_env"] = int(self._verified_teacher_bank_capacity_per_env)
            f.attrs["category_names_json"] = json.dumps(list(self.VERIFIED_BANK_CATEGORY_NAMES))
            f.attrs["num_variants_global"] = int(getattr(self, "teacher_bank_num_variants_global", int(self.num_envs)))
            f.create_dataset("counts", data=self._verified_teacher_bank_counts_cpu.numpy())
            f.create_dataset("failure_counts", data=self._verified_teacher_bank_failure_counts_cpu.numpy())
            f.create_dataset("attempt_counts", data=self._verified_teacher_bank_attempt_counts_cpu.numpy())
            f.create_dataset("success_counts", data=self._verified_teacher_bank_success_counts_cpu.numpy())
            f.create_dataset("episode_steps", data=self._verified_teacher_bank_episode_steps_cpu.numpy())
            f.create_dataset("failure_episode_steps", data=self._verified_teacher_bank_failure_episode_steps_cpu.numpy())
            f.create_dataset("attempt_episode_step_sum", data=self._verified_teacher_bank_attempt_episode_step_sum_cpu.numpy())
            f.create_dataset("success_episode_step_sum", data=self._verified_teacher_bank_success_episode_step_sum_cpu.numpy())
            f.create_dataset("failure_episode_step_sum", data=self._verified_teacher_bank_failure_episode_step_sum_cpu.numpy())
            f.create_dataset("attempt_episode_step_max", data=self._verified_teacher_bank_attempt_episode_step_max_cpu.numpy())
            f.create_dataset("success_episode_step_max", data=self._verified_teacher_bank_success_episode_step_max_cpu.numpy())
            f.create_dataset("failure_episode_step_max", data=self._verified_teacher_bank_failure_episode_step_max_cpu.numpy())
            f.create_dataset("joint_config", data=self._verified_teacher_bank_joint_config_cpu.numpy())
            f.create_dataset(
                "failure_joint_config",
                data=self._verified_teacher_bank_failure_joint_config_cpu.numpy(),
            )
            f.create_dataset("object_center_world", data=self._verified_teacher_bank_object_center_world_cpu.numpy())
            f.create_dataset(
                "failure_object_center_world",
                data=self._verified_teacher_bank_failure_object_center_world_cpu.numpy(),
            )
            f.create_dataset("object_quat_world", data=self._verified_teacher_bank_object_quat_world_cpu.numpy())
            f.create_dataset(
                "failure_object_quat_world",
                data=self._verified_teacher_bank_failure_object_quat_world_cpu.numpy(),
            )
            f.create_dataset("side_is_left", data=self._verified_teacher_bank_side_is_left_cpu.numpy().astype(np.uint8))
            f.create_dataset(
                "correct_region",
                data=self._verified_teacher_bank_correct_region_cpu.numpy().astype(np.uint8),
            )
            f.create_dataset(
                "failure_side_is_left",
                data=self._verified_teacher_bank_failure_side_is_left_cpu.numpy().astype(np.uint8),
            )
            f.create_dataset(
                "failure_correct_region",
                data=self._verified_teacher_bank_failure_correct_region_cpu.numpy().astype(np.uint8),
            )
            f.create_dataset("env_object_ids", data=self.env_object_ids.detach().cpu().numpy())
            if getattr(self, "teacher_bank_variation_ids", None) is not None:
                f.create_dataset("variation_id", data=self.teacher_bank_variation_ids.detach().cpu().numpy())
            if hasattr(self, "teacher_bank_height_bin_idx"):
                f.create_dataset("height_bin_idx", data=self.teacher_bank_height_bin_idx.detach().cpu().numpy())
            f.create_dataset("table_surface_height", data=self.table_surface_height.detach().cpu().numpy())
            f.create_dataset("table_size", data=self.table_size.detach().cpu().numpy())
            if hasattr(self, "object_mass"):
                f.create_dataset("object_mass", data=self.object_mass.detach().cpu().numpy())

    def load_verified_teacher_bank_hdf5(self, rank=0, strict=True):
        shard_path = self._get_verified_teacher_bank_shard_path(rank=rank)
        if not os.path.exists(shard_path):
            if strict:
                raise FileNotFoundError(f"Verified teacher bank shard not found: {shard_path}")
            return False
        if not hasattr(self, "_verified_teacher_bank_counts_cpu"):
            self._init_verified_teacher_bank_storage()
        with h5py.File(shard_path, "r") as f:
            file_num_envs = int(f.attrs["num_envs"])
            file_num_dofs = int(f.attrs["num_dofs"])
            file_capacity = int(f.attrs["capacity_per_env"])
            if file_num_envs != int(self.num_envs) or file_num_dofs != int(self.num_dofs):
                raise RuntimeError(
                    f"Verified teacher bank shard shape mismatch: path={shard_path} "
                    f"file_num_envs={file_num_envs} file_num_dofs={file_num_dofs} "
                    f"expected_num_envs={self.num_envs} expected_num_dofs={self.num_dofs}"
                )
            if file_capacity != int(self._verified_teacher_bank_capacity_per_env):
                raise RuntimeError(
                    f"Verified teacher bank capacity mismatch: path={shard_path} "
                    f"file_capacity={file_capacity} expected_capacity={self._verified_teacher_bank_capacity_per_env}"
                )
            if "env_object_ids" in f:
                file_env_object_ids = torch.from_numpy(f["env_object_ids"][...]).to(dtype=torch.long)
                current_env_object_ids = self.env_object_ids.detach().to(device="cpu", dtype=torch.long)
                if not torch.equal(file_env_object_ids, current_env_object_ids):
                    raise RuntimeError(
                        f"Verified teacher bank env_object_ids mismatch: path={shard_path}"
                    )
            if "variation_id" in f and getattr(self, "teacher_bank_variation_ids", None) is not None:
                file_variation_ids = torch.from_numpy(f["variation_id"][...]).to(dtype=torch.long)
                current_variation_ids = self.teacher_bank_variation_ids.detach().to(device="cpu", dtype=torch.long)
                if not torch.equal(file_variation_ids, current_variation_ids):
                    raise RuntimeError(
                        f"Verified teacher bank variation_id mismatch: path={shard_path}"
                    )
            if "table_surface_height" in f:
                file_table_surface_height = torch.from_numpy(f["table_surface_height"][...]).to(dtype=self._q.dtype)
                current_table_surface_height = self.table_surface_height.detach().to(device="cpu", dtype=self._q.dtype)
                if not torch.allclose(file_table_surface_height, current_table_surface_height, atol=1.0e-6, rtol=0.0):
                    raise RuntimeError(
                        f"Verified teacher bank table_surface_height mismatch: path={shard_path}"
                    )
            if "table_size" in f:
                file_table_size = torch.from_numpy(f["table_size"][...]).to(dtype=self._q.dtype)
                current_table_size = self.table_size.detach().to(device="cpu", dtype=self._q.dtype)
                if not torch.allclose(file_table_size, current_table_size, atol=1.0e-6, rtol=0.0):
                    raise RuntimeError(
                        f"Verified teacher bank table_size mismatch: path={shard_path}"
                    )
            if "object_mass" in f and hasattr(self, "object_mass"):
                file_object_mass = torch.from_numpy(f["object_mass"][...]).to(dtype=self._q.dtype)
                current_object_mass = self.object_mass.detach().to(device="cpu", dtype=self._q.dtype)
                if not torch.allclose(file_object_mass, current_object_mass, atol=1.0e-6, rtol=0.0):
                    raise RuntimeError(
                        f"Verified teacher bank object_mass mismatch: path={shard_path}"
                    )
            self._verified_teacher_bank_counts_cpu.copy_(torch.from_numpy(f["counts"][...]).to(dtype=torch.int32))
            if "failure_counts" in f:
                self._verified_teacher_bank_failure_counts_cpu.copy_(torch.from_numpy(f["failure_counts"][...]).to(dtype=torch.int32))
            else:
                self._verified_teacher_bank_failure_counts_cpu.zero_()
            self._verified_teacher_bank_attempt_counts_cpu.copy_(torch.from_numpy(f["attempt_counts"][...]).to(dtype=torch.int32))
            self._verified_teacher_bank_success_counts_cpu.copy_(torch.from_numpy(f["success_counts"][...]).to(dtype=torch.int32))
            if "episode_steps" in f:
                self._verified_teacher_bank_episode_steps_cpu.copy_(torch.from_numpy(f["episode_steps"][...]).to(dtype=torch.int32))
            else:
                self._verified_teacher_bank_episode_steps_cpu.zero_()
            if "failure_episode_steps" in f:
                self._verified_teacher_bank_failure_episode_steps_cpu.copy_(torch.from_numpy(f["failure_episode_steps"][...]).to(dtype=torch.int32))
            else:
                self._verified_teacher_bank_failure_episode_steps_cpu.zero_()
            if "attempt_episode_step_sum" in f:
                self._verified_teacher_bank_attempt_episode_step_sum_cpu.copy_(torch.from_numpy(f["attempt_episode_step_sum"][...]).to(dtype=torch.int64))
            else:
                self._verified_teacher_bank_attempt_episode_step_sum_cpu.zero_()
            if "success_episode_step_sum" in f:
                self._verified_teacher_bank_success_episode_step_sum_cpu.copy_(torch.from_numpy(f["success_episode_step_sum"][...]).to(dtype=torch.int64))
            else:
                self._verified_teacher_bank_success_episode_step_sum_cpu.zero_()
            if "failure_episode_step_sum" in f:
                self._verified_teacher_bank_failure_episode_step_sum_cpu.copy_(torch.from_numpy(f["failure_episode_step_sum"][...]).to(dtype=torch.int64))
            else:
                self._verified_teacher_bank_failure_episode_step_sum_cpu.zero_()
            if "attempt_episode_step_max" in f:
                self._verified_teacher_bank_attempt_episode_step_max_cpu.copy_(torch.from_numpy(f["attempt_episode_step_max"][...]).to(dtype=torch.int32))
            else:
                self._verified_teacher_bank_attempt_episode_step_max_cpu.zero_()
            if "success_episode_step_max" in f:
                self._verified_teacher_bank_success_episode_step_max_cpu.copy_(torch.from_numpy(f["success_episode_step_max"][...]).to(dtype=torch.int32))
            else:
                self._verified_teacher_bank_success_episode_step_max_cpu.zero_()
            if "failure_episode_step_max" in f:
                self._verified_teacher_bank_failure_episode_step_max_cpu.copy_(torch.from_numpy(f["failure_episode_step_max"][...]).to(dtype=torch.int32))
            else:
                self._verified_teacher_bank_failure_episode_step_max_cpu.zero_()
            self._verified_teacher_bank_joint_config_cpu.copy_(torch.from_numpy(f["joint_config"][...]).to(dtype=self._q.dtype))
            if "failure_joint_config" in f:
                self._verified_teacher_bank_failure_joint_config_cpu.copy_(torch.from_numpy(f["failure_joint_config"][...]).to(dtype=self._q.dtype))
            else:
                self._verified_teacher_bank_failure_joint_config_cpu.zero_()
            self._verified_teacher_bank_object_center_world_cpu.copy_(torch.from_numpy(f["object_center_world"][...]).to(dtype=self._q.dtype))
            if "failure_object_center_world" in f:
                self._verified_teacher_bank_failure_object_center_world_cpu.copy_(torch.from_numpy(f["failure_object_center_world"][...]).to(dtype=self._q.dtype))
            else:
                self._verified_teacher_bank_failure_object_center_world_cpu.zero_()
            self._verified_teacher_bank_object_quat_world_cpu.copy_(torch.from_numpy(f["object_quat_world"][...]).to(dtype=self._q.dtype))
            if "failure_object_quat_world" in f:
                self._verified_teacher_bank_failure_object_quat_world_cpu.copy_(torch.from_numpy(f["failure_object_quat_world"][...]).to(dtype=self._q.dtype))
            else:
                self._verified_teacher_bank_failure_object_quat_world_cpu.zero_()
            self._verified_teacher_bank_side_is_left_cpu.copy_(torch.from_numpy(f["side_is_left"][...].astype(np.bool_)))
            if "correct_region" in f:
                self._verified_teacher_bank_correct_region_cpu.copy_(
                    torch.from_numpy(f["correct_region"][...].astype(np.bool_))
                )
            else:
                self._verified_teacher_bank_correct_region_cpu.zero_()
            if "failure_side_is_left" in f:
                self._verified_teacher_bank_failure_side_is_left_cpu.copy_(torch.from_numpy(f["failure_side_is_left"][...].astype(np.bool_)))
            else:
                self._verified_teacher_bank_failure_side_is_left_cpu.zero_()
            if "failure_correct_region" in f:
                self._verified_teacher_bank_failure_correct_region_cpu.copy_(
                    torch.from_numpy(f["failure_correct_region"][...].astype(np.bool_))
                )
            else:
                self._verified_teacher_bank_failure_correct_region_cpu.zero_()
        self._verified_teacher_bank_loaded = True
        return True

    def _snapshot_verified_teacher_episode_starts(self, env_ids, category_id):
        if env_ids.numel() == 0:
            return
        self._verified_teacher_bank_episode_active[env_ids] = True
        self._verified_teacher_bank_episode_category[env_ids] = int(category_id)
        self._verified_teacher_bank_episode_joint_config[env_ids] = self._q[env_ids].clone()
        self._verified_teacher_bank_episode_object_center_world[env_ids] = self.states["object_center_pos"][env_ids].clone()
        self._verified_teacher_bank_episode_object_quat_world[env_ids] = self._object_state[env_ids, 3:7].clone()
        self._verified_teacher_bank_episode_side_is_left[env_ids] = self.side_is_left[env_ids].clone()
        self._verified_teacher_bank_episode_correct_region[env_ids] = self._compute_correct_region_mask(
            self.states["eef_pos"][env_ids],
            self.states["object_center_pos"][env_ids],
            self.side_is_left[env_ids],
        )
        self._verified_teacher_bank_episode_start_progress[env_ids] = self.progress_buf[env_ids].clone()

    def _flush_pending_verified_teacher_episode_starts(self, env_ids):
        if env_ids is None or env_ids.numel() == 0:
            return
        valid_env_ids = env_ids[self._verified_teacher_bank_snapshot_pending[env_ids]]
        if valid_env_ids.numel() == 0:
            return

        category_ids = self._verified_teacher_bank_snapshot_pending_category[valid_env_ids]
        valid_mask = category_ids >= 0
        valid_env_ids = valid_env_ids[valid_mask]
        category_ids = category_ids[valid_mask]
        if valid_env_ids.numel() == 0:
            return

        for category_id in torch.unique(category_ids).tolist():
            category_env_ids = valid_env_ids[category_ids == int(category_id)]
            self._snapshot_verified_teacher_episode_starts(category_env_ids, int(category_id))

        self._verified_teacher_bank_snapshot_pending[valid_env_ids] = False
        self._verified_teacher_bank_snapshot_pending_category[valid_env_ids] = -1

    def record_verified_teacher_episode_outcomes(self, done_env_ids):
        if done_env_ids is None or done_env_ids.numel() == 0:
            return
        done_env_ids_cpu = done_env_ids.detach().to(device="cpu", dtype=torch.long)
        done_success_cpu = self.success_long_enough[done_env_ids].detach().to(device="cpu", dtype=torch.bool)
        done_active_cpu = self._verified_teacher_bank_episode_active[done_env_ids].detach().to(device="cpu", dtype=torch.bool)
        done_category_cpu = self._verified_teacher_bank_episode_category[done_env_ids].detach().to(device="cpu", dtype=torch.long)
        joint_cpu = self._verified_teacher_bank_episode_joint_config[done_env_ids].detach().to(device="cpu", dtype=self._q.dtype)
        object_center_cpu = self._verified_teacher_bank_episode_object_center_world[done_env_ids].detach().to(device="cpu", dtype=self._q.dtype)
        object_quat_cpu = self._verified_teacher_bank_episode_object_quat_world[done_env_ids].detach().to(device="cpu", dtype=self._q.dtype)
        side_cpu = self._verified_teacher_bank_episode_side_is_left[done_env_ids].detach().to(device="cpu", dtype=torch.bool)
        correct_region_cpu = self._verified_teacher_bank_episode_correct_region[done_env_ids].detach().to(
            device="cpu",
            dtype=torch.bool,
        )
        start_progress_cpu = self._verified_teacher_bank_episode_start_progress[done_env_ids].detach().to(
            device="cpu",
            dtype=torch.long,
        )
        done_progress_cpu = self.progress_buf[done_env_ids].detach().to(device="cpu", dtype=torch.long)
        done_episode_steps_cpu = torch.clamp(done_progress_cpu - start_progress_cpu + 1, min=1).to(dtype=torch.int32)

        if self._activation_snapshot_bank_enable and hasattr(self, "_activation_snapshot_bank_episode_active"):
            activation_active_cpu = self._activation_snapshot_bank_episode_active[done_env_ids].detach().to(
                device="cpu",
                dtype=torch.bool,
            )
            activation_category_cpu = self._activation_snapshot_bank_episode_category[done_env_ids].detach().to(
                device="cpu",
                dtype=torch.long,
            )
            activation_joint_cpu = self._activation_snapshot_bank_episode_joint_config[done_env_ids].detach().to(
                device="cpu",
                dtype=self._q.dtype,
            )
            activation_object_center_cpu = self._activation_snapshot_bank_episode_object_center_world[
                done_env_ids
            ].detach().to(device="cpu", dtype=self._q.dtype)
            activation_object_quat_cpu = self._activation_snapshot_bank_episode_object_quat_world[
                done_env_ids
            ].detach().to(device="cpu", dtype=self._q.dtype)
            activation_side_cpu = self._activation_snapshot_bank_episode_side_is_left[done_env_ids].detach().to(
                device="cpu",
                dtype=torch.bool,
            )
            activation_correct_region_cpu = self._activation_snapshot_bank_episode_correct_region[
                done_env_ids
            ].detach().to(device="cpu", dtype=torch.bool)
            activation_table_pos_cpu = self._activation_snapshot_bank_episode_table_pos_world[done_env_ids].detach().to(
                device="cpu",
                dtype=self._q.dtype,
            )
            activation_table_quat_cpu = self._activation_snapshot_bank_episode_table_quat_world[
                done_env_ids
            ].detach().to(device="cpu", dtype=self._q.dtype)
            activation_target_quat_cpu = self._activation_snapshot_bank_episode_target_quat_world[
                done_env_ids
            ].detach().to(device="cpu", dtype=self._q.dtype)
            activation_start_progress_cpu = self._activation_snapshot_bank_episode_start_progress[
                done_env_ids
            ].detach().to(device="cpu", dtype=torch.long)
            activation_episode_steps_cpu = torch.clamp(
                done_progress_cpu - activation_start_progress_cpu + 1,
                min=1,
            ).to(dtype=torch.int32)
        else:
            activation_active_cpu = None

        for i in range(int(done_env_ids_cpu.numel())):
            env_id = int(done_env_ids_cpu[i].item())
            is_success = bool(done_success_cpu[i].item())
            matches_region_filter = self._verified_teacher_bank_collection_region_match_bool(
                bool(correct_region_cpu[i].item())
            )
            if bool(done_active_cpu[i].item()):
                category_id = int(done_category_cpu[i].item())
                episode_steps = int(done_episode_steps_cpu[i].item())
                self._verified_teacher_bank_attempt_counts_cpu[category_id, env_id] += 1
                self._verified_teacher_bank_attempt_episode_step_sum_cpu[category_id, env_id] += episode_steps
                self._verified_teacher_bank_attempt_episode_step_max_cpu[category_id, env_id] = max(
                    int(self._verified_teacher_bank_attempt_episode_step_max_cpu[category_id, env_id].item()),
                    episode_steps,
                )
                if is_success:
                    self._verified_teacher_bank_success_counts_cpu[category_id, env_id] += 1
                    self._verified_teacher_bank_success_episode_step_sum_cpu[category_id, env_id] += episode_steps
                    self._verified_teacher_bank_success_episode_step_max_cpu[category_id, env_id] = max(
                        int(self._verified_teacher_bank_success_episode_step_max_cpu[category_id, env_id].item()),
                        episode_steps,
                    )
                    write_idx = int(self._verified_teacher_bank_counts_cpu[category_id, env_id].item())
                    if matches_region_filter and (write_idx < int(self._verified_teacher_bank_capacity_per_env)):
                        self._verified_teacher_bank_joint_config_cpu[category_id, env_id, write_idx].copy_(joint_cpu[i])
                        self._verified_teacher_bank_object_center_world_cpu[category_id, env_id, write_idx].copy_(
                            object_center_cpu[i]
                        )
                        self._verified_teacher_bank_object_quat_world_cpu[category_id, env_id, write_idx].copy_(
                            object_quat_cpu[i]
                        )
                        self._verified_teacher_bank_side_is_left_cpu[category_id, env_id, write_idx] = side_cpu[i]
                        self._verified_teacher_bank_correct_region_cpu[category_id, env_id, write_idx] = (
                            correct_region_cpu[i]
                        )
                        self._verified_teacher_bank_episode_steps_cpu[category_id, env_id, write_idx] = episode_steps
                        self._verified_teacher_bank_counts_cpu[category_id, env_id] += 1
                else:
                    self._verified_teacher_bank_failure_episode_step_sum_cpu[category_id, env_id] += episode_steps
                    self._verified_teacher_bank_failure_episode_step_max_cpu[category_id, env_id] = max(
                        int(self._verified_teacher_bank_failure_episode_step_max_cpu[category_id, env_id].item()),
                        episode_steps,
                    )
                    write_idx = int(self._verified_teacher_bank_failure_counts_cpu[category_id, env_id].item())
                    if matches_region_filter and (write_idx < int(self._verified_teacher_bank_capacity_per_env)):
                        self._verified_teacher_bank_failure_joint_config_cpu[category_id, env_id, write_idx].copy_(
                            joint_cpu[i]
                        )
                        self._verified_teacher_bank_failure_object_center_world_cpu[
                            category_id, env_id, write_idx
                        ].copy_(object_center_cpu[i])
                        self._verified_teacher_bank_failure_object_quat_world_cpu[
                            category_id, env_id, write_idx
                        ].copy_(object_quat_cpu[i])
                        self._verified_teacher_bank_failure_side_is_left_cpu[category_id, env_id, write_idx] = side_cpu[i]
                        self._verified_teacher_bank_failure_correct_region_cpu[category_id, env_id, write_idx] = (
                            correct_region_cpu[i]
                        )
                        self._verified_teacher_bank_failure_episode_steps_cpu[category_id, env_id, write_idx] = episode_steps
                        self._verified_teacher_bank_failure_counts_cpu[category_id, env_id] += 1

            if activation_active_cpu is not None and bool(activation_active_cpu[i].item()):
                category_id = int(activation_category_cpu[i].item())
                activation_episode_steps = int(activation_episode_steps_cpu[i].item())
                activation_matches_region_filter = self._verified_teacher_bank_collection_region_match_bool(
                    bool(activation_correct_region_cpu[i].item())
                )
                self._activation_snapshot_bank_event_counts_cpu[category_id, env_id] += 1
                self._activation_snapshot_bank_event_step_sum_cpu[category_id, env_id] += activation_episode_steps
                self._activation_snapshot_bank_event_step_max_cpu[category_id, env_id] = max(
                    int(self._activation_snapshot_bank_event_step_max_cpu[category_id, env_id].item()),
                    activation_episode_steps,
                )
                if is_success:
                    self._activation_snapshot_bank_success_step_sum_cpu[category_id, env_id] += activation_episode_steps
                    self._activation_snapshot_bank_success_step_max_cpu[category_id, env_id] = max(
                        int(self._activation_snapshot_bank_success_step_max_cpu[category_id, env_id].item()),
                        activation_episode_steps,
                    )
                    write_idx = int(self._activation_snapshot_bank_counts_cpu[category_id, env_id].item())
                    if activation_matches_region_filter and (
                        write_idx < int(self._activation_snapshot_bank_capacity_per_env)
                    ):
                        self._activation_snapshot_bank_joint_config_cpu[category_id, env_id, write_idx].copy_(
                            activation_joint_cpu[i]
                        )
                        self._activation_snapshot_bank_object_center_world_cpu[category_id, env_id, write_idx].copy_(
                            activation_object_center_cpu[i]
                        )
                        self._activation_snapshot_bank_object_quat_world_cpu[category_id, env_id, write_idx].copy_(
                            activation_object_quat_cpu[i]
                        )
                        self._activation_snapshot_bank_side_is_left_cpu[category_id, env_id, write_idx] = (
                            activation_side_cpu[i]
                        )
                        self._activation_snapshot_bank_correct_region_cpu[category_id, env_id, write_idx] = (
                            activation_correct_region_cpu[i]
                        )
                        self._activation_snapshot_bank_table_pos_world_cpu[category_id, env_id, write_idx].copy_(
                            activation_table_pos_cpu[i]
                        )
                        self._activation_snapshot_bank_table_quat_world_cpu[category_id, env_id, write_idx].copy_(
                            activation_table_quat_cpu[i]
                        )
                        self._activation_snapshot_bank_target_quat_world_cpu[category_id, env_id, write_idx].copy_(
                            activation_target_quat_cpu[i]
                        )
                        self._activation_snapshot_bank_episode_steps_cpu[category_id, env_id, write_idx] = activation_episode_steps
                        self._activation_snapshot_bank_counts_cpu[category_id, env_id] += 1
                else:
                    self._activation_snapshot_bank_failure_step_sum_cpu[category_id, env_id] += activation_episode_steps
                    self._activation_snapshot_bank_failure_step_max_cpu[category_id, env_id] = max(
                        int(self._activation_snapshot_bank_failure_step_max_cpu[category_id, env_id].item()),
                        activation_episode_steps,
                    )
                    write_idx = int(self._activation_snapshot_bank_failure_counts_cpu[category_id, env_id].item())
                    if activation_matches_region_filter and (
                        write_idx < int(self._activation_snapshot_bank_capacity_per_env)
                    ):
                        self._activation_snapshot_bank_failure_joint_config_cpu[
                            category_id, env_id, write_idx
                        ].copy_(activation_joint_cpu[i])
                        self._activation_snapshot_bank_failure_object_center_world_cpu[
                            category_id, env_id, write_idx
                        ].copy_(activation_object_center_cpu[i])
                        self._activation_snapshot_bank_failure_object_quat_world_cpu[
                            category_id, env_id, write_idx
                        ].copy_(activation_object_quat_cpu[i])
                        self._activation_snapshot_bank_failure_side_is_left_cpu[category_id, env_id, write_idx] = (
                            activation_side_cpu[i]
                        )
                        self._activation_snapshot_bank_failure_correct_region_cpu[
                            category_id, env_id, write_idx
                        ] = activation_correct_region_cpu[i]
                        self._activation_snapshot_bank_failure_table_pos_world_cpu[
                            category_id, env_id, write_idx
                        ].copy_(activation_table_pos_cpu[i])
                        self._activation_snapshot_bank_failure_table_quat_world_cpu[
                            category_id, env_id, write_idx
                        ].copy_(activation_table_quat_cpu[i])
                        self._activation_snapshot_bank_failure_target_quat_world_cpu[
                            category_id, env_id, write_idx
                        ].copy_(activation_target_quat_cpu[i])
                        self._activation_snapshot_bank_failure_episode_steps_cpu[
                            category_id, env_id, write_idx
                        ] = activation_episode_steps
                        self._activation_snapshot_bank_failure_counts_cpu[category_id, env_id] += 1

        self._verified_teacher_bank_episode_active[done_env_ids] = False
        self._verified_teacher_bank_episode_category[done_env_ids] = -1
        if self._activation_snapshot_bank_enable and hasattr(self, "_activation_snapshot_bank_episode_active"):
            self._activation_snapshot_bank_episode_active[done_env_ids] = False
            self._activation_snapshot_bank_episode_category[done_env_ids] = -1

    def _choose_verified_teacher_bank_categories(self, env_ids):
        counts = self._verified_teacher_bank_counts_tensor()[:, env_ids]
        available = counts > 0
        if env_ids.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        base_probs = self._verified_teacher_bank_sampling_probs.to(device=self.device).unsqueeze(1).repeat(1, env_ids.numel())
        base_probs = base_probs * available.to(dtype=base_probs.dtype)
        probs_sum = base_probs.sum(dim=0, keepdim=True)
        category_ids = torch.full((env_ids.numel(),), -1, dtype=torch.long, device=self.device)
        valid_envs = probs_sum.squeeze(0) > 0
        if torch.any(valid_envs):
            valid_probs = base_probs[:, valid_envs].transpose(0, 1)
            valid_probs = valid_probs / valid_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            category_ids[valid_envs] = torch.multinomial(valid_probs, num_samples=1).squeeze(-1)
        return category_ids

    def _sample_verified_teacher_bank_entries(self, env_ids, category_ids):
        valid_mask = category_ids >= 0
        if not torch.any(valid_mask):
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0, self.num_dofs), dtype=self._q.dtype, device=self.device),
                torch.empty((0, 3), dtype=self._q.dtype, device=self.device),
                torch.empty((0, 4), dtype=self._q.dtype, device=self.device),
                torch.empty((0,), dtype=torch.bool, device=self.device),
            )

        valid_env_ids = env_ids[valid_mask]
        valid_category_ids = category_ids[valid_mask].detach().to(device="cpu", dtype=torch.long)
        valid_env_ids_cpu = valid_env_ids.detach().to(device="cpu", dtype=torch.long)

        sample_indices_cpu = torch.zeros((valid_env_ids_cpu.numel(),), dtype=torch.long, device="cpu")
        for i in range(int(valid_env_ids_cpu.numel())):
            category_id = int(valid_category_ids[i].item())
            env_id = int(valid_env_ids_cpu[i].item())
            count = int(self._verified_teacher_bank_counts_cpu[category_id, env_id].item())
            if count <= 0:
                raise RuntimeError(
                    f"Verified teacher bank requested empty slot: category={category_id} env_id={env_id}"
                )
            sample_indices_cpu[i] = torch.randint(low=0, high=count, size=(1,), device="cpu", dtype=torch.long)[0]

        joint_config = self._verified_teacher_bank_joint_config_cpu[
            valid_category_ids,
            valid_env_ids_cpu,
            sample_indices_cpu,
        ].to(device=self.device, dtype=self._q.dtype)
        object_center_world = self._verified_teacher_bank_object_center_world_cpu[
            valid_category_ids,
            valid_env_ids_cpu,
            sample_indices_cpu,
        ].to(device=self.device, dtype=self._q.dtype)
        object_quat_world = self._verified_teacher_bank_object_quat_world_cpu[
            valid_category_ids,
            valid_env_ids_cpu,
            sample_indices_cpu,
        ].to(device=self.device, dtype=self._q.dtype)
        side_is_left = self._verified_teacher_bank_side_is_left_cpu[
            valid_category_ids,
            valid_env_ids_cpu,
            sample_indices_cpu,
        ].to(device=self.device)
        return (
            valid_env_ids,
            valid_category_ids.to(device=self.device),
            joint_config,
            object_center_world,
            object_quat_world,
            side_is_left,
        )

    def _resample_verified_teacher_bank_afar_hand_joints(self, env_ids, category_ids, joint_config):
        afar_mask = category_ids == self.VERIFIED_BANK_AFAR
        if not torch.any(afar_mask):
            return joint_config

        afar_env_ids = env_ids[afar_mask]
        hand_reset_noise = torch.rand((afar_env_ids.numel(), 16), device=self.device, dtype=joint_config.dtype)
        hand_reset_noise = 2.0 * (hand_reset_noise - 0.5)

        if self.reset_noise_scale["leap"] is None:
            hand_reset_noise = self.unnormalize_robot_joints(hand_reset_noise, robot="leap", delta=False)
            hand_reset_noise -= self.canonical_joint_config[afar_env_ids, 10:26]
        else:
            hand_reset_noise *= self.reset_noise_scale["leap"]

        joint_config = joint_config.clone()
        joint_config[afar_mask, 10:26] = self.canonical_joint_config[afar_env_ids, 10:26] + hand_reset_noise
        joint_config = tensor_clamp(
            joint_config,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )
        return joint_config

    def _get_far_recovery_safe_hand_joint_bank_collision_pairs(self):
        adjacency = {
            frozenset(("palm_lower", "mcp_1")),
            frozenset(("palm_lower", "mcp_2")),
            frozenset(("palm_lower", "mcp_3")),
            frozenset(("palm_lower", "thumb_temp_base")),
            frozenset(("mcp_1", "pip_1")),
            frozenset(("pip_1", "dip_1")),
            frozenset(("dip_1", "fingertip_1")),
            frozenset(("mcp_2", "pip_2")),
            frozenset(("pip_2", "dip_2")),
            frozenset(("dip_2", "fingertip_2")),
            frozenset(("mcp_3", "pip_3")),
            frozenset(("pip_3", "dip_3")),
            frozenset(("dip_3", "fingertip_3")),
            frozenset(("thumb_temp_base", "pip_4")),
            frozenset(("pip_4", "dip_4")),
            frozenset(("dip_4", "fingertip_4")),
        }

        hand_links = [link.name for link in self.robot_pcd_sampler.hand_links if link.name in self.robot_pcd_sampler.points]
        pairs = []
        for i, name_a in enumerate(hand_links):
            for name_b in hand_links[i + 1 :]:
                if frozenset((name_a, name_b)) in adjacency:
                    continue
                pairs.append((name_a, name_b))
        return pairs

    def _apply_recovery_local_roll_pitch_noise(self, target_quat):
        if target_quat.numel() == 0:
            return target_quat
        noise_rad = float(self._reset_pitch_roll_noise_rad)
        if noise_rad <= 0.0:
            return target_quat

        device = target_quat.device
        dtype = target_quat.dtype
        batch = target_quat.shape[0]
        hand_x_axis = quat_apply(
            target_quat,
            torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(batch, 1),
        )
        hand_y_axis = quat_apply(
            target_quat,
            torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype).repeat(batch, 1),
        )
        roll_angle = (torch.rand((batch,), device=device, dtype=dtype) * 2.0 - 1.0) * noise_rad
        pitch_angle = (torch.rand((batch,), device=device, dtype=dtype) * 2.0 - 1.0) * noise_rad
        q_roll_noise = quat_from_angle_axis(roll_angle, hand_x_axis)
        q_pitch_noise = quat_from_angle_axis(pitch_angle, hand_y_axis)
        target_quat_noisy = quat_mul(q_pitch_noise, quat_mul(q_roll_noise, target_quat))
        return target_quat_noisy / torch.norm(target_quat_noisy, dim=-1, keepdim=True).clamp_min(1.0e-8)

    def _sample_far_recovery_safe_hand_joint_bank_candidate_batch(self, batch_size):
        base_joint_config = self.canonical_joint_config[:1].repeat(batch_size, 1).clone()
        hand_noise = torch.rand((batch_size, 16), device=self.device, dtype=base_joint_config.dtype)
        hand_noise = 2.0 * (hand_noise - 0.5)

        if self.reset_noise_scale["leap"] is None:
            hand_noise = self.unnormalize_robot_joints(hand_noise, robot="leap", delta=True)
            hand_noise *= self._verified_teacher_bank_far_recovery_hand_noise_scale
        else:
            hand_noise *= float(self.reset_noise_scale["leap"]) * self._verified_teacher_bank_far_recovery_hand_noise_scale

        base_joint_config[:, 10:26] = base_joint_config[:, 10:26] + hand_noise
        base_joint_config = tensor_clamp(
            base_joint_config,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )
        return base_joint_config

    def _filter_far_recovery_safe_hand_joint_bank_candidates(self, joint_config):
        if joint_config.numel() == 0:
            return torch.empty((0,), device=self.device, dtype=torch.bool)

        candidate_q = joint_config[:, self.torchurdf_to_isaac_idx]
        fk = self.robot_pcd_sampler.robot.visual_geometry_fk_batch(candidate_q)
        valid = torch.ones((joint_config.shape[0],), device=self.device, dtype=torch.bool)

        link_pcds = {}
        points_per_link = max(1, int(self._verified_teacher_bank_far_recovery_hand_bank_points_per_link))
        for link in self.robot_pcd_sampler.hand_links:
            if link.name not in self.robot_pcd_sampler.points:
                continue
            geom = link.visuals[0].geometry
            T = fk[geom]
            local_points = self.robot_pcd_sampler.points[link.name][:, :points_per_link].repeat(joint_config.shape[0], 1, 1)
            link_pcds[link.name] = transform_pointcloud(local_points, T)

        min_link_distance = float(self._verified_teacher_bank_far_recovery_hand_bank_min_link_distance)
        for name_a, name_b in self._far_recovery_safe_hand_joint_bank_collision_pairs:
            if name_a not in link_pcds or name_b not in link_pcds:
                continue
            active_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            if active_idx.numel() == 0:
                break
            pa = link_pcds[name_a][active_idx]
            pb = link_pcds[name_b][active_idx]
            min_dist = torch.cdist(pa, pb).amin(dim=(1, 2))
            valid_active = min_dist >= min_link_distance
            valid[active_idx] = valid_active
        return valid

    def _init_far_recovery_safe_hand_joint_bank(self):
        self._far_recovery_safe_hand_joint_bank = torch.empty((0, 16), device=self.device, dtype=self._q.dtype)
        self._far_recovery_safe_hand_joint_bank_collision_pairs = []
        if self._verified_teacher_bank_far_recovery_hand_noise_scale <= 0.0:
            return
        if self._verified_teacher_bank_far_recovery_hand_bank_size <= 0:
            return

        self._far_recovery_safe_hand_joint_bank_collision_pairs = (
            self._get_far_recovery_safe_hand_joint_bank_collision_pairs()
        )

        accepted = []
        target_size = int(self._verified_teacher_bank_far_recovery_hand_bank_size)
        candidate_batch_size = max(1, int(self._verified_teacher_bank_far_recovery_hand_bank_candidate_batch_size))
        max_rounds = max(1, int(self._verified_teacher_bank_far_recovery_hand_bank_max_rounds))

        for round_idx in range(max_rounds):
            candidates = self._sample_far_recovery_safe_hand_joint_bank_candidate_batch(candidate_batch_size)
            valid_mask = self._filter_far_recovery_safe_hand_joint_bank_candidates(candidates)
            if bool(torch.any(valid_mask)):
                accepted.append(candidates[valid_mask, 10:26].clone())
            accepted_count = sum(int(x.shape[0]) for x in accepted)
            if accepted_count >= target_size:
                break
            if (round_idx + 1) == max_rounds and accepted_count == 0:
                print(
                    "[FarRecoverySafeHandBank] failed to find any safe hand-joint samples; "
                    "far recovery hand randomization will be disabled."
                )

        if len(accepted) == 0:
            return

        bank = torch.cat(accepted, dim=0)[:target_size]
        self._far_recovery_safe_hand_joint_bank = bank
        print(
            "[FarRecoverySafeHandBank] built "
            f"count={int(bank.shape[0])} target={target_size} "
            f"noise_scale={self._verified_teacher_bank_far_recovery_hand_noise_scale:.3f} "
            f"min_link_distance={self._verified_teacher_bank_far_recovery_hand_bank_min_link_distance:.4f}"
        )

    def _perturb_verified_teacher_bank_far_recovery_hand_joints(self, env_ids, category_ids, joint_config):
        if self._far_recovery_safe_hand_joint_bank.shape[0] <= 0:
            return joint_config

        far_mask = category_ids == self.VERIFIED_BANK_FAR_RECOVERY
        if not torch.any(far_mask):
            return joint_config

        far_env_ids = env_ids[far_mask]
        sample_idx = torch.randint(
            low=0,
            high=int(self._far_recovery_safe_hand_joint_bank.shape[0]),
            size=(far_env_ids.numel(),),
            device=self.device,
            dtype=torch.long,
        )
        hand_joint_samples = self._far_recovery_safe_hand_joint_bank[sample_idx].to(
            device=self.device,
            dtype=joint_config.dtype,
        )

        joint_config = joint_config.clone()
        joint_config[far_mask, 10:26] = hand_joint_samples
        joint_config = tensor_clamp(
            joint_config,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )
        return joint_config

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
        joint_config = self._resample_verified_teacher_bank_afar_hand_joints(
            sampled_env_ids,
            sampled_category_ids,
            joint_config,
        )
        joint_config = self._perturb_verified_teacher_bank_far_recovery_hand_joints(
            sampled_env_ids,
            sampled_category_ids,
            joint_config,
        )
        self._apply_object_center_state(sampled_env_ids, object_center_world, object_quat_world)
        self.set_robot_joint_state(joint_config, env_ids=sampled_env_ids)
        if self.enable_fabric:
            self.fabric_q[sampled_env_ids, :10] = joint_config[:, :10]
            self.fabric_q[sampled_env_ids, 10:] = joint_config[:, 26:]
            self.fabric_qd[sampled_env_ids, :] = 0.0
            self.fabric_qdd[sampled_env_ids, :] = 0.0
        self.side_is_left[sampled_env_ids] = side_is_left
        self._set_reward_target_quat_from_side_mask(sampled_env_ids)
        self.reward_settings["target_pos"][sampled_env_ids] = self._build_fixed_target_pos(sampled_env_ids)
        self.debug_recovery_reset_active[sampled_env_ids] = False
        return sampled_env_ids, sampled_category_ids

    def _sample_side_mask(self, num_samples, device):
        side_mode = str(self.cfg["env"]["eef_init"].get("side_mode", "left"))
        if side_mode == "left":
            return torch.ones(num_samples, device=device, dtype=torch.bool)
        if side_mode == "right":
            return torch.zeros(num_samples, device=device, dtype=torch.bool)
        if side_mode == "both":
            return torch.rand(num_samples, device=device) < 0.5
        raise ValueError(f"Unsupported eef_init.side_mode={side_mode}")
        
    def _build_fixed_target_pos(self, env_ids=None):
        if env_ids is None:
            object_init_xy = self._object_center_init_state[:, :2]
            table_height = self.table_surface_height
        else:
            object_init_xy = self._object_center_init_state[env_ids, :2]
            table_height = self.table_surface_height[env_ids]

        target_xy_offset = self.rl_target_xy_offset.to(device=self.device, dtype=object_init_xy.dtype).unsqueeze(0)
        target_pos = torch.zeros((object_init_xy.shape[0], 3), device=self.device, dtype=object_init_xy.dtype)
        target_pos[:, :2] = object_init_xy + target_xy_offset
        target_pos[:, 2] = table_height + self.rl_target_z_offset
        return target_pos

    def _apply_object_center_state(self, env_ids, object_center_world, object_quat=None):
        num_resets = int(env_ids.numel())
        sampled_object_state = torch.zeros((num_resets, 13), device=self.device, dtype=self._object_state.dtype)
        if object_quat is None:
            object_quat = torch.zeros((num_resets, 4), device=self.device, dtype=self._object_state.dtype)
            object_quat[:, 3] = 1.0
        local_offset = torch.zeros((num_resets, 3), device=self.device, dtype=self._object_state.dtype)
        local_offset[:, 2] = self.mesh_aabb_extents[env_ids, 2] * self.object_center_z_scale
        object_root_pos = object_center_world - quat_apply(object_quat, local_offset)
        sampled_object_state[:, :3] = object_root_pos
        sampled_object_state[:, 3:7] = object_quat
        self._object_state[env_ids] = sampled_object_state
        self._object_center_init_state[env_ids] = object_center_world

        multi_env_ids_int32 = self._global_indices[env_ids, self._object_id].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def _sample_upright_near_table_recovery_states(self, env_ids):
        dtype = self._q.dtype
        device = self.device
        num_recovery = int(env_ids.numel())
        if num_recovery == 0:
            return (
                torch.empty((0, self.num_dofs), device=device, dtype=dtype),
                torch.empty((0,), device=device, dtype=torch.bool),
                torch.empty((0, 3), device=device, dtype=dtype),
                torch.empty((0, 4), device=device, dtype=dtype),
                torch.empty((0, 3), device=device, dtype=dtype),
                torch.empty((0, 4), device=device, dtype=dtype),
                torch.empty((0,), device=device, dtype=torch.bool),
            )

        side_mode = str(self.cfg["env"]["eef_init"].get("side_mode", "left"))
        if side_mode == "left":
            recovery_left_mask = torch.ones(num_recovery, device=device, dtype=torch.bool)
        elif side_mode == "right":
            recovery_left_mask = torch.zeros(num_recovery, device=device, dtype=torch.bool)
        elif side_mode == "both":
            recovery_left_mask = torch.rand(num_recovery, device=device) < 0.5
        else:
            raise ValueError(f"Unsupported eef_init.side_mode={side_mode}")
        recovery_far_mask = torch.rand(num_recovery, device=device) < self._reset_upright_near_table_far_recovery_prob

        base_init_range = torch.tensor(
            self._reset_upright_near_table_base_init_range_cfg,
            device=device,
            dtype=dtype,
        )
        palm_rel_object_min = torch.tensor(
            self._reset_upright_near_table_palm_rel_object_min_cfg,
            device=device,
            dtype=dtype,
        )
        palm_rel_object_max = torch.tensor(
            self._reset_upright_near_table_palm_rel_object_max_cfg,
            device=device,
            dtype=dtype,
        )
        table_height = self.table_surface_height[env_ids].to(dtype=dtype)
        table_center_xy = self.table_pos[env_ids, :2].to(dtype=dtype)
        table_half_xy = 0.5 * self.table_size[env_ids, :2].to(dtype=dtype)
        table_xy_min = table_center_xy - table_half_xy
        table_xy_max = table_center_xy + table_half_xy
        object_half_xy = 0.5 * self.mesh_aabb_extents[env_ids, :2].to(dtype=dtype)
        object_top_z = table_height + self.mesh_aabb_extents[env_ids, 2].to(dtype=dtype)

        solved_mask = torch.zeros(num_recovery, device=device, dtype=torch.bool)
        solved_joint_config = self.canonical_joint_config[env_ids].clone()
        solved_palm_pos = torch.zeros((num_recovery, 3), device=device, dtype=dtype)
        solved_palm_quat = torch.zeros((num_recovery, 4), device=device, dtype=dtype)
        solved_palm_quat[:, 3] = 1.0
        solved_object_center = torch.zeros((num_recovery, 3), device=device, dtype=dtype)
        solved_object_quat = torch.zeros((num_recovery, 4), device=device, dtype=dtype)
        solved_object_quat[:, 3] = 1.0
        solved_is_far = torch.zeros((num_recovery,), device=device, dtype=torch.bool)

        palm_offset_local = self._palm_center_from_link7_local.to(device=device, dtype=dtype)
        target_quat_left = self.target_quat_left.to(device=device, dtype=dtype)
        target_quat_right = self.target_quat_right.to(device=device, dtype=dtype)

        xy_min = self.obj_pos_range[env_ids][:, [0, 2]].to(dtype=dtype)
        xy_max = self.obj_pos_range[env_ids][:, [1, 3]].to(dtype=dtype)
        xy_min, xy_max = self._adjust_object_reset_xy_bounds(env_ids, xy_min, xy_max)

        warn_interval = max(1, self._reset_upright_near_table_max_resample_attempts)
        attempt_idx = 0
        while True:
            unresolved = (~solved_mask).nonzero(as_tuple=False).squeeze(-1)
            if unresolved.numel() == 0:
                break
            attempt_idx += 1
            if attempt_idx % warn_interval == 0:
                print(
                    "[SideRecoveryBank] still sampling recovery states "
                    f"unsolved={int(unresolved.numel())}/{num_recovery} "
                    f"round={attempt_idx}"
                )

            count = int(unresolved.numel())
            base_pose = base_init_range[0].unsqueeze(0) + torch.rand((count, 3), device=device, dtype=dtype) * (
                base_init_range[1] - base_init_range[0]
            ).unsqueeze(0)

            base_yaw = base_pose[:, 2]
            base_half_yaw = 0.5 * base_yaw
            base_quat = torch.zeros((count, 4), device=device, dtype=dtype)
            base_quat[:, 2] = torch.sin(base_half_yaw)
            base_quat[:, 3] = torch.cos(base_half_yaw)

            object_xy_rand = torch.rand((count, 2), device=device, dtype=dtype)
            object_center = torch.zeros((count, 3), device=device, dtype=dtype)
            object_center[:, :2] = xy_min[unresolved] + object_xy_rand * (xy_max[unresolved] - xy_min[unresolved])
            object_center[:, 2] = table_height[unresolved] + self.mesh_aabb_extents[env_ids[unresolved], 2].to(dtype=dtype) * self.object_center_z_scale
            close_rel_min_clip, close_rel_max_clip, valid_close_clip_xy = self._get_clipped_recovery_palm_rel_object_bounds(
                object_center,
                table_xy_min[unresolved],
                table_xy_max[unresolved],
                dtype,
            )
            close_rel_min_clip, close_rel_max_clip, valid_close_region = (
                self._apply_collection_region_filter_to_recovery_bounds(
                    close_rel_min_clip,
                    close_rel_max_clip,
                    recovery_left_mask[unresolved],
                )
            )
            valid_close_clip_xy = valid_close_clip_xy & valid_close_region
            far_rel_min_clip, far_rel_max_clip, valid_far_clip_xy = self._get_clipped_recovery_palm_rel_object_bounds(
                object_center,
                table_xy_min[unresolved],
                table_xy_max[unresolved],
                dtype,
                rel_object_min_cfg=self._reset_upright_near_table_palm_rel_object_far_min_cfg,
                rel_object_max_cfg=self._reset_upright_near_table_palm_rel_object_far_max_cfg,
            )
            far_rel_min_clip, far_rel_max_clip, valid_far_region = (
                self._apply_collection_region_filter_to_recovery_bounds(
                    far_rel_min_clip,
                    far_rel_max_clip,
                    recovery_left_mask[unresolved],
                )
            )
            valid_far_clip_xy = valid_far_clip_xy & valid_far_region
            use_far = recovery_far_mask[unresolved]
            far_region_available = valid_far_clip_xy & torch.any(
                (far_rel_min_clip < close_rel_min_clip) | (far_rel_max_clip > close_rel_max_clip),
                dim=-1,
            )
            use_far = use_far & far_region_available
            active_valid_clip_xy = torch.where(use_far, valid_far_clip_xy, valid_close_clip_xy)
            if not bool(torch.any(active_valid_clip_xy)):
                continue
            palm_rel_object = close_rel_min_clip + torch.rand((count, 3), device=device, dtype=dtype) * (
                close_rel_max_clip - close_rel_min_clip
            )
            if bool(torch.any(use_far)):
                far_palm_rel_object = far_rel_min_clip + torch.rand((count, 3), device=device, dtype=dtype) * (
                    far_rel_max_clip - far_rel_min_clip
                )
                palm_rel_object[use_far] = far_palm_rel_object[use_far]
            palm_pos = object_center + palm_rel_object

            palm_rel_object_xy = palm_rel_object[:, :2]
            palm_object_dist = torch.norm(palm_rel_object, dim=-1)
            inside_close_box = torch.all(
                (palm_rel_object >= close_rel_min_clip) & (palm_rel_object <= close_rel_max_clip),
                dim=-1,
            )
            inside_object_xy = (
                (torch.abs(palm_rel_object_xy[:, 0]) <= object_half_xy[unresolved, 0] + 0.02)
                & (torch.abs(palm_rel_object_xy[:, 1]) <= object_half_xy[unresolved, 1] + 0.02)
            )
            below_object_top = palm_pos[:, 2] <= (object_top_z[unresolved] + 0.02)
            valid_scene = (
                (palm_pos[:, 0] >= table_xy_min[unresolved, 0])
                & (palm_pos[:, 0] <= table_xy_max[unresolved, 0])
                & (palm_pos[:, 1] >= table_xy_min[unresolved, 1])
                & (palm_pos[:, 1] <= table_xy_max[unresolved, 1])
                & (object_center[:, 0] >= table_xy_min[unresolved, 0])
                & (object_center[:, 0] <= table_xy_max[unresolved, 0])
                & (object_center[:, 1] >= table_xy_min[unresolved, 1])
                & (object_center[:, 1] <= table_xy_max[unresolved, 1])
                & active_valid_clip_xy
                & (palm_object_dist >= self._reset_upright_near_table_min_palm_object_dist)
                & ((~use_far) | (~inside_close_box))
                & (~(inside_object_xy & below_object_top))
            )
            if not bool(torch.any(valid_scene)):
                continue

            valid_idx = valid_scene.nonzero(as_tuple=False).squeeze(-1)
            valid_unresolved = unresolved[valid_idx]
            valid_left_mask = recovery_left_mask[valid_unresolved]
            valid_palm_pos = palm_pos[valid_idx]

            base_quat_nominal = target_quat_right.repeat(valid_idx.numel(), 1)
            if int(valid_left_mask.sum().item()) > 0:
                base_quat_nominal[valid_left_mask] = target_quat_left.repeat(int(valid_left_mask.sum().item()), 1)
            target_quat = quat_mul(base_quat[valid_idx], base_quat_nominal)
            target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
            target_quat = self._apply_recovery_local_roll_pitch_noise(target_quat)

            franka_base_pose7 = self._get_franka_base_pose7_from_mobile_base_pose(base_pose[valid_idx], dtype)
            franka_base_pos_world = franka_base_pose7[:, :3]
            franka_base_quat_world = franka_base_pose7[:, 3:7]
            inv_franka_base_quat = quat_conjugate(franka_base_quat_world)
            valid_palm_pos_local = quat_apply(inv_franka_base_quat, valid_palm_pos - franka_base_pos_world)
            target_quat_local = quat_mul(inv_franka_base_quat, target_quat)
            target_quat_local = target_quat_local / torch.norm(target_quat_local, dim=-1, keepdim=True).clamp_min(1.0e-8)
            link7_pos_local = valid_palm_pos_local - quat_apply(
                target_quat_local,
                palm_offset_local.unsqueeze(0).repeat(valid_idx.numel(), 1),
            )
            eef_pose = torch.cat([link7_pos_local, target_quat_local], dim=-1)
            arm_q, success = self._solve_reset_arm_ik(eef_pose)
            if not bool(torch.any(success)):
                continue

            success_unresolved = valid_unresolved[success]
            success_base_pose = base_pose[valid_idx][success]
            success_palm_pos_world = valid_palm_pos[success]
            success_target_quat_world = target_quat[success]
            success_joint_config = solved_joint_config[success_unresolved].clone()
            success_joint_config[:, :3] = success_base_pose
            success_joint_config[:, 3:10] = arm_q[success]
            success_object_center = object_center[valid_idx][success]
            no_penetration = self._filter_recovery_robot_object_penetration(
                success_joint_config,
                success_object_center,
                env_ids[success_unresolved],
            )
            if not bool(torch.any(no_penetration)):
                continue

            solved_valid = success_unresolved[no_penetration]
            solved_joint_config[solved_valid] = success_joint_config[no_penetration]
            solved_palm_pos[solved_valid] = success_palm_pos_world[no_penetration]
            solved_palm_quat[solved_valid] = success_target_quat_world[no_penetration]
            solved_object_center[solved_valid] = success_object_center[no_penetration]
            solved_is_far[solved_valid] = use_far[valid_idx][success][no_penetration]
            solved_mask[solved_valid] = True

        return (
            solved_joint_config,
            recovery_left_mask,
            solved_object_center,
            solved_object_quat,
            solved_palm_pos,
            solved_palm_quat,
            solved_is_far,
        )

    def _init_upright_near_table_recovery_bank(self):
        bank_size = int(self._reset_upright_near_table_bank_size)
        num_envs = int(self.num_envs)
        dtype = self._q.dtype

        self._recovery_bank_joint_config_cpu = torch.empty((num_envs, bank_size, self.num_dofs), dtype=dtype, device="cpu")
        self._recovery_bank_side_is_left_cpu = torch.empty((num_envs, bank_size), dtype=torch.bool, device="cpu")
        self._recovery_bank_object_center_world_cpu = torch.empty((num_envs, bank_size, 3), dtype=dtype, device="cpu")
        self._recovery_bank_object_quat_world_cpu = torch.empty((num_envs, bank_size, 4), dtype=dtype, device="cpu")
        self._recovery_bank_palm_pos_cpu = torch.empty((num_envs, bank_size, 3), dtype=dtype, device="cpu")
        self._recovery_bank_palm_quat_cpu = torch.empty((num_envs, bank_size, 4), dtype=dtype, device="cpu")
        self._recovery_bank_is_far_cpu = torch.empty((num_envs, bank_size), dtype=torch.bool, device="cpu")

        env_ids = torch.arange(num_envs, device=self.device, dtype=torch.long)
        goals_per_round = max(1, int(self._reset_upright_near_table_bank_max_ik_goals))
        goals_per_round = min(goals_per_round, max(num_envs * 16, 64))
        slots_per_round = max(1, goals_per_round // max(num_envs, 1))
        t0 = time.time()
        progress = tqdm(total=bank_size, desc="Building Side Recovery Bank")
        for start in range(0, bank_size, slots_per_round):
            cur_slots = min(slots_per_round, bank_size - start)
            batched_env_ids = env_ids.repeat(cur_slots)
            (
                joint_config,
                left_mask,
                object_center_world,
                object_quat_world,
                palm_pos,
                palm_quat,
                is_far,
            ) = self._sample_upright_near_table_recovery_states(batched_env_ids)

            end = start + cur_slots
            joint_config = joint_config.reshape(cur_slots, num_envs, self.num_dofs).transpose(0, 1).contiguous().cpu()
            left_mask = left_mask.reshape(cur_slots, num_envs).transpose(0, 1).contiguous().cpu()
            object_center_world = object_center_world.reshape(cur_slots, num_envs, 3).transpose(0, 1).contiguous().cpu()
            object_quat_world = object_quat_world.reshape(cur_slots, num_envs, 4).transpose(0, 1).contiguous().cpu()
            palm_pos = palm_pos.reshape(cur_slots, num_envs, 3).transpose(0, 1).contiguous().cpu()
            palm_quat = palm_quat.reshape(cur_slots, num_envs, 4).transpose(0, 1).contiguous().cpu()
            is_far = is_far.reshape(cur_slots, num_envs).transpose(0, 1).contiguous().cpu()

            self._recovery_bank_joint_config_cpu[:, start:end].copy_(joint_config)
            self._recovery_bank_side_is_left_cpu[:, start:end].copy_(left_mask)
            self._recovery_bank_object_center_world_cpu[:, start:end].copy_(object_center_world)
            self._recovery_bank_object_quat_world_cpu[:, start:end].copy_(object_quat_world)
            self._recovery_bank_palm_pos_cpu[:, start:end].copy_(palm_pos)
            self._recovery_bank_palm_quat_cpu[:, start:end].copy_(palm_quat)
            self._recovery_bank_is_far_cpu[:, start:end].copy_(is_far)
            progress.update(cur_slots)
        progress.close()

        elapsed = time.time() - t0
        print(
            f"Built side recovery bank: num_envs={num_envs}, bank_size={bank_size}, "
            f"slots_per_round={slots_per_round}, elapsed={elapsed:.1f}s"
        )
        self._recovery_reset_bank_ready = True

    def _sample_upright_near_table_recovery_from_bank(self, env_ids, force_far=None):
        env_ids_cpu = env_ids.detach().to(device="cpu", dtype=torch.long)
        bank_idx_cpu = torch.zeros((env_ids.numel(),), device="cpu", dtype=torch.long)
        for i in range(int(env_ids_cpu.numel())):
            env_id = int(env_ids_cpu[i].item())
            candidate_mask = torch.ones((self._reset_upright_near_table_bank_size,), dtype=torch.bool, device="cpu")
            if force_far is not None:
                candidate_mask &= (self._recovery_bank_is_far_cpu[env_id] == bool(force_far))
            candidate_ids = candidate_mask.nonzero(as_tuple=False).squeeze(-1)
            if candidate_ids.numel() == 0:
                candidate_ids = torch.arange(self._reset_upright_near_table_bank_size, device="cpu", dtype=torch.long)
            bank_idx_cpu[i] = candidate_ids[torch.randint(low=0, high=int(candidate_ids.numel()), size=(1,), device="cpu", dtype=torch.long)[0]]

        joint_config = self._recovery_bank_joint_config_cpu[env_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        left_mask = self._recovery_bank_side_is_left_cpu[env_ids_cpu, bank_idx_cpu].to(device=self.device)
        object_center_world = self._recovery_bank_object_center_world_cpu[env_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        object_quat_world = self._recovery_bank_object_quat_world_cpu[env_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        palm_pos = self._recovery_bank_palm_pos_cpu[env_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        palm_quat = self._recovery_bank_palm_quat_cpu[env_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        is_far = self._recovery_bank_is_far_cpu[env_ids_cpu, bank_idx_cpu].to(device=self.device)
        return joint_config, left_mask, object_center_world, object_quat_world, palm_pos, palm_quat, is_far

    def _make_debug_teacher_state_bucket(self):
        return {
            "count": 0,
            "raw_seen": 0,
            "raw_written": 0,
            "base_rel_table_xyz": self._make_debug_tensor_stats(3),
            "base_rel_object_xyz": self._make_debug_tensor_stats(3),
            "eef_rel_object_xyz": self._make_debug_tensor_stats(3),
            "eef_target_quat_deg": self._make_debug_tensor_stats(1),
            "q_arm": self._make_debug_tensor_stats(7),
            "q_hand": self._make_debug_tensor_stats(16),
        }

    def _make_debug_tensor_stats(self, dim):
        return {
            "sum": torch.zeros(dim, dtype=torch.float64),
            "min": torch.full((dim,), float("inf"), dtype=torch.float64),
            "max": torch.full((dim,), float("-inf"), dtype=torch.float64),
        }

    def _init_debug_teacher_state_stats(self):
        return {
            "activation": self._make_debug_teacher_state_bucket(),
            "teleport": self._make_debug_teacher_state_bucket(),
        }

    def _write_debug_teacher_state_raw_records(self, tag, env_ids, payload):
        if not self.debug_teacher_state_stats_enabled:
            return
        bucket = self._debug_teacher_state_stats[tag]
        if bucket["raw_written"] >= self.debug_teacher_state_stats_raw_max_records:
            return

        records = []
        n = int(env_ids.numel())
        for local_idx in range(n):
            bucket["raw_seen"] += 1
            if (bucket["raw_seen"] - 1) % self.debug_teacher_state_stats_raw_stride != 0:
                continue
            if bucket["raw_written"] >= self.debug_teacher_state_stats_raw_max_records:
                break
            records.append(
                {
                    "tag": tag,
                    "sim_steps": int(self.sim_steps),
                    "env_id": int(env_ids[local_idx].item()),
                    "base_rel_table_xyz": payload["base_rel_table_xyz"][local_idx].detach().cpu().tolist(),
                    "base_rel_object_xyz": payload["base_rel_object_xyz"][local_idx].detach().cpu().tolist(),
                    "eef_rel_object_xyz": payload["eef_rel_object_xyz"][local_idx].detach().cpu().tolist(),
                    "eef_target_quat_deg": float(payload["eef_target_quat_deg"][local_idx].item()),
                    "q_arm": payload["q_arm"][local_idx].detach().cpu().tolist(),
                    "q_hand": payload["q_hand"][local_idx].detach().cpu().tolist(),
                }
            )
            bucket["raw_written"] += 1

        if not records:
            return

        with open(self.debug_teacher_state_stats_raw_path, "a", encoding="ascii") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def _update_debug_tensor_stats(self, bucket, key, values):
        if not self.debug_teacher_state_stats_enabled:
            return
        values_cpu = values.detach().to(dtype=torch.float64).cpu()
        if values_cpu.ndim == 1:
            values_cpu = values_cpu.unsqueeze(-1)
        if values_cpu.shape[0] == 0:
            return
        bucket[key]["sum"] += values_cpu.sum(dim=0)
        bucket[key]["min"] = torch.minimum(bucket[key]["min"], values_cpu.amin(dim=0))
        bucket[key]["max"] = torch.maximum(bucket[key]["max"], values_cpu.amax(dim=0))

    def _collect_debug_teacher_state_stats(self, tag, env_ids):
        if not self.debug_teacher_state_stats_enabled:
            return
        if env_ids.numel() == 0:
            return

        bucket = self._debug_teacher_state_stats[tag]
        q = self.states["q"][env_ids]
        base_pos = self.states["franka_base_pose7"][env_ids, :3]
        object_center = self.states["object_center_pos"][env_ids]
        eef_pos = self._eef_state[env_ids, :3]
        eef_quat = self._eef_state[env_ids, 3:7]
        target_quat = self.reward_settings["target_quat"][env_ids]
        table_height = self.table_surface_height[env_ids]

        base_rel_table = base_pos.clone()
        base_rel_table[:, 2] -= table_height
        base_rel_object = base_pos - object_center
        eef_rel_object = eef_pos - object_center
        quat_dot = torch.abs(torch.sum(normalize(eef_quat) * normalize(target_quat), dim=-1)).clamp(max=1.0)
        eef_target_quat_deg = 2.0 * torch.rad2deg(torch.acos(quat_dot))

        raw_payload = {
            "base_rel_table_xyz": base_rel_table,
            "base_rel_object_xyz": base_rel_object,
            "eef_rel_object_xyz": eef_rel_object,
            "eef_target_quat_deg": eef_target_quat_deg,
            "q_arm": q[:, 3:10],
            "q_hand": q[:, 10:26],
        }

        self._update_debug_tensor_stats(bucket, "base_rel_table_xyz", raw_payload["base_rel_table_xyz"])
        self._update_debug_tensor_stats(bucket, "base_rel_object_xyz", raw_payload["base_rel_object_xyz"])
        self._update_debug_tensor_stats(bucket, "eef_rel_object_xyz", raw_payload["eef_rel_object_xyz"])
        self._update_debug_tensor_stats(bucket, "eef_target_quat_deg", raw_payload["eef_target_quat_deg"])
        self._update_debug_tensor_stats(bucket, "q_arm", raw_payload["q_arm"])
        self._update_debug_tensor_stats(bucket, "q_hand", raw_payload["q_hand"])
        bucket["count"] += int(env_ids.numel())
        self._write_debug_teacher_state_raw_records(tag, env_ids, raw_payload)

    def _debug_teacher_state_bucket_to_jsonable(self, bucket):
        count = int(bucket["count"])
        out = {
            "count": count,
            "raw_seen": int(bucket.get("raw_seen", 0)),
            "raw_written": int(bucket.get("raw_written", 0)),
        }
        for key, stats in bucket.items():
            if key in {"count", "raw_seen", "raw_written"}:
                continue
            if count > 0:
                mean = (stats["sum"] / float(count)).tolist()
                min_v = stats["min"].tolist()
                max_v = stats["max"].tolist()
            else:
                dim = int(stats["sum"].numel())
                mean = [0.0] * dim
                min_v = [0.0] * dim
                max_v = [0.0] * dim
            out[key] = {"mean": mean, "min": min_v, "max": max_v}
        return out

    def _flush_debug_teacher_state_stats(self, force=False):
        if not self.debug_teacher_state_stats_enabled:
            return
        if (not force) and (int(self.sim_steps) % self.debug_teacher_state_stats_print_freq != 0):
            return
        payload = {
            tag: self._debug_teacher_state_bucket_to_jsonable(bucket)
            for tag, bucket in self._debug_teacher_state_stats.items()
        }
        payload["sim_steps"] = int(self.sim_steps)
        with open(self.debug_teacher_state_stats_path, "w", encoding="ascii") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[debug_teacher_state_stats] step={int(self.sim_steps)} "
            f"activation_n={payload['activation']['count']} "
            f"teleport_n={payload['teleport']['count']} "
            f"raw_path={self.debug_teacher_state_stats_raw_path} "
            f"path={self.debug_teacher_state_stats_path}"
        )

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return

        if self.debug_viz:
            pre_reset_color_state = torch.where(
                self.flat_table_long_enough[env_ids],
                torch.full((env_ids.numel(),), 2, device=self.device, dtype=self._table_debug_color_state_prev.dtype),
                torch.where(
                    self.success_long_enough[env_ids],
                    torch.full((env_ids.numel(),), 1, device=self.device, dtype=self._table_debug_color_state_prev.dtype),
                    torch.zeros((env_ids.numel(),), device=self.device, dtype=self._table_debug_color_state_prev.dtype),
                ),
            )
            success_count = int((pre_reset_color_state == 1).sum().item())
            failure_count = int((pre_reset_color_state == 2).sum().item())
            labeled_count = success_count + failure_count
            if labeled_count > 0:
                self.debug_success_total += success_count
                self.debug_failure_total += failure_count

            self._table_debug_outcome_state[env_ids] = pre_reset_color_state
            self._table_debug_outcome_steps_left[env_ids] = torch.where(
                pre_reset_color_state > 0,
                torch.full((env_ids.numel(),), self._table_debug_outcome_hold_steps, device=self.device, dtype=torch.int32),
                torch.zeros((env_ids.numel(),), device=self.device, dtype=torch.int32),
            )

        super().reset_idx(env_ids)
        collection_category = int(self._verified_teacher_bank_collection_category)
        collection_enabled = bool(self._verified_teacher_bank_collection_enabled) and (collection_category >= 0)
        use_verified_teacher_bank = (
            self._verified_teacher_bank_enable
            and self._verified_teacher_bank_loaded
            and (not collection_enabled)
        )

        if use_verified_teacher_bank:
            recovery_env_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        elif collection_enabled and (collection_category in (self.VERIFIED_BANK_NEAR_RECOVERY, self.VERIFIED_BANK_FAR_RECOVERY)):
            recovery_env_ids = env_ids.clone()
        elif self._reset_upright_near_table_recovery_prob > 0.0:
            select_mask = torch.rand(env_ids.numel(), device=self.device) < self._reset_upright_near_table_recovery_prob
            recovery_env_ids = env_ids[select_mask]
        else:
            recovery_env_ids = torch.empty((0,), dtype=torch.long, device=self.device)

        recovery_palm_pos = torch.empty((0, 3), dtype=self._q.dtype, device=self.device)
        recovery_left_mask = torch.empty((0,), dtype=torch.bool, device=self.device)
        recovery_palm_quat = torch.empty((0, 4), dtype=self._q.dtype, device=self.device)
        if use_verified_teacher_bank:
            sampled_env_ids, sampled_category_ids = self._apply_verified_teacher_bank_reset(env_ids)
            recovery_env_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        elif recovery_env_ids.numel() > 0:
            if collection_enabled and (collection_category in (self.VERIFIED_BANK_NEAR_RECOVERY, self.VERIFIED_BANK_FAR_RECOVERY)):
                force_far = bool(collection_category == self.VERIFIED_BANK_FAR_RECOVERY)
                recovery_env_ids, recovery_palm_pos, recovery_left_mask, recovery_palm_quat = (
                    self._apply_upright_near_table_recovery_reset_for_env_ids(
                        recovery_env_ids,
                        force_far=force_far,
                    )
                )
            elif self._recovery_reset_bank_ready:
                force_far = None
                if collection_enabled:
                    force_far = bool(collection_category == self.VERIFIED_BANK_FAR_RECOVERY)
                (
                    recovery_joint_config,
                    recovery_left_mask,
                    recovery_object_center,
                    recovery_object_quat,
                    recovery_palm_pos,
                    recovery_palm_quat,
                    _recovery_is_far,
                ) = self._sample_upright_near_table_recovery_from_bank(recovery_env_ids, force_far=force_far)
                self._apply_object_center_state(recovery_env_ids, recovery_object_center, recovery_object_quat)
                self.set_robot_joint_state(recovery_joint_config, env_ids=recovery_env_ids)
                if self.enable_fabric:
                    self.fabric_q[recovery_env_ids, :10] = recovery_joint_config[:, :10]
                    self.fabric_q[recovery_env_ids, 10:] = recovery_joint_config[:, 26:]
                    self.fabric_qd[recovery_env_ids, :] = 0.0
                    self.fabric_qdd[recovery_env_ids, :] = 0.0
            else:
                recovery_env_ids, recovery_palm_pos, recovery_left_mask, recovery_palm_quat = (
                    self._apply_upright_near_table_recovery_reset(env_ids)
                )

        reset_related_env_ids = env_ids
        snapshot_env_ids = reset_related_env_ids
        self.current_episode_verified_bank_category[reset_related_env_ids] = -1
        if collection_enabled and (
            collection_category in (self.VERIFIED_BANK_NEAR_RECOVERY, self.VERIFIED_BANK_FAR_RECOVERY)
        ):
            # Only solved recovery resets belong in the recovery-state banks. Any envs
            # that fell back to the superclass reset remain ordinary afar-like starts.
            snapshot_env_ids = recovery_env_ids
        if use_verified_teacher_bank and sampled_env_ids.numel() > 0:
            self.current_episode_verified_bank_category[sampled_env_ids] = sampled_category_ids
        elif collection_enabled and snapshot_env_ids.numel() > 0:
            self.current_episode_verified_bank_category[snapshot_env_ids] = int(collection_category)
        self.debug_recovery_reset_active[reset_related_env_ids] = False

        self.switch_target_quat_latched[reset_related_env_ids] = False  # CODEX: force switching target refresh after teleport/reset.
        self.switching_eef_init_pos[reset_related_env_ids] = self._eef_state[reset_related_env_ids, :3]  # CODEX
        if recovery_env_ids.numel() > 0:
            self.switching_eef_init_pos[recovery_env_ids] = recovery_palm_pos
            self.debug_recovery_reset_active[recovery_env_ids] = True
            self.debug_recovery_reset_palm_pos[recovery_env_ids] = recovery_palm_pos
            self.debug_recovery_reset_palm_quat[recovery_env_ids] = recovery_palm_quat
        self.flat_table_duration[reset_related_env_ids] = 0
        self.flat_table_long_enough[reset_related_env_ids] = False
        self.post_lift_target_active[reset_related_env_ids] = False
        self.reward_settings["target_pos"][reset_related_env_ids] = self._build_fixed_target_pos(reset_related_env_ids)
        if self.use_center_tracking_switch_target:  # CODEX
            self.side_is_left[reset_related_env_ids] = True  # CODEX: center-tracking mode is left-only by design.
        elif collection_enabled and (collection_category == self.VERIFIED_BANK_AFAR):
            self.side_is_left[reset_related_env_ids] = self._sample_side_mask(reset_related_env_ids.numel(), self.device)
        elif recovery_env_ids.numel() > 0:
            self.side_is_left[recovery_env_ids] = recovery_left_mask
        self._set_reward_target_quat_from_side_mask(reset_related_env_ids)  # CODEX: refresh teacher target quat on teleport/reset.
        if hasattr(self, "_verified_teacher_bank_snapshot_pending"):
            self._verified_teacher_bank_snapshot_pending[reset_related_env_ids] = False
            self._verified_teacher_bank_snapshot_pending_category[reset_related_env_ids] = -1
        if collection_enabled and snapshot_env_ids.numel() > 0:
            self._verified_teacher_bank_snapshot_pending[snapshot_env_ids] = True
            self._verified_teacher_bank_snapshot_pending_category[snapshot_env_ids] = int(collection_category)

    def _solve_reset_arm_ik(self, eef_pose):
        eef_pose = eef_pose.clone().contiguous()
        batch = int(eef_pose.shape[0])
        if batch == 0:
            return (
                torch.empty((0, 7), device=self.device, dtype=self._q.dtype),
                torch.empty((0,), device=self.device, dtype=torch.bool),
            )

        max_batch = int(self.num_envs)
        if batch <= max_batch:
            eef_pos = eef_pose[:, :3]
            eef_quat_xyzw = eef_pose[:, 3:]
            eef_quat_wxyz = eef_quat_xyzw[:, [3, 0, 1, 2]]

            batch_pad = max_batch - batch
            if batch_pad > 0:
                eef_pos_dummy = torch.tensor([[0.3, 0.0, 0.3]] * batch_pad, dtype=torch.float, device=self.device)
                eef_quat_wxyz_dummy = torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]] * batch_pad, dtype=torch.float, device=self.device
                )
                eef_pos = torch.cat((eef_pos, eef_pos_dummy), dim=0)
                eef_quat_wxyz = torch.cat((eef_quat_wxyz, eef_quat_wxyz_dummy), dim=0)

            goal = Pose(eef_pos, eef_quat_wxyz)
            result = self.ik_solver.solve_batch(
                goal_pose=goal,
                retract_config=self.ik_regularization_config,
            )
            success = result.success[:batch]
            if success.ndim > 1:
                success = success[:, 0]
            return result.solution[:batch, 0], success.reshape(-1).bool()

        # The recovery bank builder can request many more IK goals than `self.num_envs`.
        # cuRobo was initialized with batch capacity `self.num_envs`, so solve these in
        # chunks instead of feeding a mismatched retract_config batch.
        solutions = []
        successes = []
        for start in range(0, batch, max_batch):
            end = min(start + max_batch, batch)
            chunk_solution, chunk_success = self._solve_reset_arm_ik(eef_pose[start:end])
            solutions.append(chunk_solution)
            successes.append(chunk_success)
        return torch.cat(solutions, dim=0), torch.cat(successes, dim=0)

    def _apply_upright_near_table_recovery_reset_for_env_ids(self, recovery_env_ids, force_far=None):
        if recovery_env_ids.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0, 3), dtype=self._q.dtype, device=self.device),
                torch.empty((0,), dtype=torch.bool, device=self.device),
                torch.empty((0, 4), dtype=self._q.dtype, device=self.device),
            )

        dtype = self._q.dtype
        device = self.device
        num_recovery = int(recovery_env_ids.numel())

        side_mode = str(self.cfg["env"]["eef_init"].get("side_mode", "left"))
        if side_mode == "left":
            recovery_left_mask = torch.ones(num_recovery, device=device, dtype=torch.bool)
        elif side_mode == "right":
            recovery_left_mask = torch.zeros(num_recovery, device=device, dtype=torch.bool)
        elif side_mode == "both":
            recovery_left_mask = torch.rand(num_recovery, device=device) < 0.5
        else:
            raise ValueError(f"Unsupported eef_init.side_mode={side_mode}")
        if force_far is None:
            recovery_far_mask = torch.rand(num_recovery, device=device) < self._reset_upright_near_table_far_recovery_prob
        else:
            recovery_far_mask = torch.full(
                (num_recovery,),
                bool(force_far),
                device=device,
                dtype=torch.bool,
            )

        base_init_range = torch.tensor(
            self._reset_upright_near_table_base_init_range_cfg,
            device=device,
            dtype=dtype,
        )
        palm_rel_object_min = torch.tensor(
            self._reset_upright_near_table_palm_rel_object_min_cfg,
            device=device,
            dtype=dtype,
        )
        palm_rel_object_max = torch.tensor(
            self._reset_upright_near_table_palm_rel_object_max_cfg,
            device=device,
            dtype=dtype,
        )

        table_height = self.table_surface_height[recovery_env_ids].to(dtype=dtype)
        table_center_xy = self.table_pos[recovery_env_ids, :2].to(dtype=dtype)
        table_half_xy = 0.5 * self.table_size[recovery_env_ids, :2].to(dtype=dtype)
        table_xy_min = table_center_xy - table_half_xy
        table_xy_max = table_center_xy + table_half_xy
        object_center = self._object_center_init_state[recovery_env_ids, :3].to(dtype=dtype)
        object_half_xy = 0.5 * self.mesh_aabb_extents[recovery_env_ids, :2].to(dtype=dtype)
        object_top_z = table_height + self.mesh_aabb_extents[recovery_env_ids, 2].to(dtype=dtype)

        solved_mask = torch.zeros(num_recovery, device=device, dtype=torch.bool)
        solved_joint_config = self.canonical_joint_config[recovery_env_ids].clone()
        solved_palm_pos = torch.zeros((num_recovery, 3), device=device, dtype=dtype)
        solved_palm_quat = torch.zeros((num_recovery, 4), device=device, dtype=dtype)
        solved_palm_quat[:, 3] = 1.0

        palm_offset_local = self._palm_center_from_link7_local.to(device=device, dtype=dtype)
        target_quat_left = self.target_quat_left.to(device=device, dtype=dtype)
        target_quat_right = self.target_quat_right.to(device=device, dtype=dtype)

        warn_interval = max(1, self._reset_upright_near_table_max_resample_attempts)
        attempt_idx = 0
        while True:
            unresolved = (~solved_mask).nonzero(as_tuple=False).squeeze(-1)
            if unresolved.numel() == 0:
                break
            attempt_idx += 1
            if attempt_idx % warn_interval == 0:
                print(
                    "[SideRecoveryOnline] still sampling recovery reset "
                    f"unsolved={int(unresolved.numel())}/{num_recovery} "
                    f"round={attempt_idx}"
                )

            count = int(unresolved.numel())
            base_pose = base_init_range[0].unsqueeze(0) + torch.rand((count, 3), device=device, dtype=dtype) * (
                base_init_range[1] - base_init_range[0]
            ).unsqueeze(0)

            base_half_yaw = 0.5 * base_pose[:, 2]
            base_quat = torch.zeros((count, 4), device=device, dtype=dtype)
            base_quat[:, 2] = torch.sin(base_half_yaw)
            base_quat[:, 3] = torch.cos(base_half_yaw)

            close_rel_min_clip, close_rel_max_clip, valid_close_clip_xy = self._get_clipped_recovery_palm_rel_object_bounds(
                object_center[unresolved],
                table_xy_min[unresolved],
                table_xy_max[unresolved],
                dtype,
            )
            close_rel_min_clip, close_rel_max_clip, valid_close_region = (
                self._apply_collection_region_filter_to_recovery_bounds(
                    close_rel_min_clip,
                    close_rel_max_clip,
                    recovery_left_mask[unresolved],
                )
            )
            valid_close_clip_xy = valid_close_clip_xy & valid_close_region
            far_rel_min_clip, far_rel_max_clip, valid_far_clip_xy = self._get_clipped_recovery_palm_rel_object_bounds(
                object_center[unresolved],
                table_xy_min[unresolved],
                table_xy_max[unresolved],
                dtype,
                rel_object_min_cfg=self._reset_upright_near_table_palm_rel_object_far_min_cfg,
                rel_object_max_cfg=self._reset_upright_near_table_palm_rel_object_far_max_cfg,
            )
            far_rel_min_clip, far_rel_max_clip, valid_far_region = (
                self._apply_collection_region_filter_to_recovery_bounds(
                    far_rel_min_clip,
                    far_rel_max_clip,
                    recovery_left_mask[unresolved],
                )
            )
            valid_far_clip_xy = valid_far_clip_xy & valid_far_region
            use_far = recovery_far_mask[unresolved]
            far_region_available = valid_far_clip_xy & torch.any(
                (far_rel_min_clip < close_rel_min_clip) | (far_rel_max_clip > close_rel_max_clip),
                dim=-1,
            )
            use_far = use_far & far_region_available
            active_valid_clip_xy = torch.where(use_far, valid_far_clip_xy, valid_close_clip_xy)
            if not bool(torch.any(active_valid_clip_xy)):
                continue
            palm_rel_object = close_rel_min_clip + torch.rand((count, 3), device=device, dtype=dtype) * (
                close_rel_max_clip - close_rel_min_clip
            )
            if bool(torch.any(use_far)):
                far_palm_rel_object = far_rel_min_clip + torch.rand((count, 3), device=device, dtype=dtype) * (
                    far_rel_max_clip - far_rel_min_clip
                )
                palm_rel_object[use_far] = far_palm_rel_object[use_far]
            palm_pos = object_center[unresolved] + palm_rel_object
            palm_rel_object_xy = palm_rel_object[:, :2]
            palm_object_dist = torch.norm(palm_rel_object, dim=-1)
            inside_close_box = torch.all(
                (palm_rel_object >= close_rel_min_clip) & (palm_rel_object <= close_rel_max_clip),
                dim=-1,
            )
            inside_object_xy = (
                (torch.abs(palm_rel_object_xy[:, 0]) <= object_half_xy[unresolved, 0] + 0.02)
                & (torch.abs(palm_rel_object_xy[:, 1]) <= object_half_xy[unresolved, 1] + 0.02)
            )
            below_object_top = palm_pos[:, 2] <= (object_top_z[unresolved] + 0.02)
            valid_scene = (
                (palm_pos[:, 0] >= table_xy_min[unresolved, 0])
                & (palm_pos[:, 0] <= table_xy_max[unresolved, 0])
                & (palm_pos[:, 1] >= table_xy_min[unresolved, 1])
                & (palm_pos[:, 1] <= table_xy_max[unresolved, 1])
                & active_valid_clip_xy
                & (palm_object_dist >= self._reset_upright_near_table_min_palm_object_dist)
                & ((~use_far) | (~inside_close_box))
                & (~(inside_object_xy & below_object_top))
            )
            if not bool(torch.any(valid_scene)):
                continue

            valid_idx = valid_scene.nonzero(as_tuple=False).squeeze(-1)
            valid_unresolved = unresolved[valid_idx]
            valid_left_mask = recovery_left_mask[valid_unresolved]
            valid_palm_pos = palm_pos[valid_idx]

            base_quat_nominal = target_quat_right.repeat(valid_idx.numel(), 1)
            if int(valid_left_mask.sum().item()) > 0:
                base_quat_nominal[valid_left_mask] = target_quat_left.repeat(int(valid_left_mask.sum().item()), 1)
            target_quat = quat_mul(base_quat[valid_idx], base_quat_nominal)
            target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
            target_quat = self._apply_recovery_local_roll_pitch_noise(target_quat)

            franka_base_pose7 = self._get_franka_base_pose7_from_mobile_base_pose(base_pose[valid_idx], dtype)
            franka_base_pos_world = franka_base_pose7[:, :3]
            franka_base_quat_world = franka_base_pose7[:, 3:7]
            inv_franka_base_quat = quat_conjugate(franka_base_quat_world)
            valid_palm_pos_local = quat_apply(inv_franka_base_quat, valid_palm_pos - franka_base_pos_world)
            target_quat_local = quat_mul(inv_franka_base_quat, target_quat)
            target_quat_local = target_quat_local / torch.norm(target_quat_local, dim=-1, keepdim=True).clamp_min(1.0e-8)
            link7_pos_local = valid_palm_pos_local - quat_apply(
                target_quat_local,
                palm_offset_local.unsqueeze(0).repeat(valid_idx.numel(), 1),
            )
            eef_pose = torch.cat([link7_pos_local, target_quat_local], dim=-1)
            arm_q, success = self._solve_reset_arm_ik(eef_pose)
            if not bool(torch.any(success)):
                continue

            success_unresolved = valid_unresolved[success]
            success_base_pose = base_pose[valid_idx][success]
            success_palm_pos_world = valid_palm_pos[success]
            success_target_quat_world = target_quat[success]
            success_joint_config = solved_joint_config[success_unresolved].clone()
            success_joint_config[:, :3] = success_base_pose
            success_joint_config[:, 3:10] = arm_q[success]
            success_object_center = object_center[valid_idx][success]
            no_penetration = self._filter_recovery_robot_object_penetration(
                success_joint_config,
                success_object_center,
                recovery_env_ids[success_unresolved],
            )
            if not bool(torch.any(no_penetration)):
                continue

            solved_valid = success_unresolved[no_penetration]
            solved_joint_config[solved_valid] = success_joint_config[no_penetration]
            solved_palm_pos[solved_valid] = success_palm_pos_world[no_penetration]
            solved_palm_quat[solved_valid] = success_target_quat_world[no_penetration]
            solved_mask[solved_valid] = True

        if not bool(torch.any(solved_mask)):
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0, 3), dtype=self._q.dtype, device=self.device),
                torch.empty((0,), dtype=torch.bool, device=self.device),
                torch.empty((0, 4), dtype=self._q.dtype, device=self.device),
            )

        solved_env_ids = recovery_env_ids[solved_mask]
        self.set_robot_joint_state(solved_joint_config[solved_mask], env_ids=solved_env_ids)

        if self.enable_fabric:
            self.fabric_q[solved_env_ids, :10] = solved_joint_config[solved_mask, :10]
            self.fabric_q[solved_env_ids, 10:] = solved_joint_config[solved_mask, 26:]
            self.fabric_qd[solved_env_ids, :] = 0.0
            self.fabric_qdd[solved_env_ids, :] = 0.0

        return solved_env_ids, solved_palm_pos[solved_mask], recovery_left_mask[solved_mask], solved_palm_quat[solved_mask]

    def _apply_upright_near_table_recovery_reset(self, env_ids):
        if self._reset_upright_near_table_recovery_prob <= 0.0:
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0, 3), dtype=self._q.dtype, device=self.device),
                torch.empty((0,), dtype=torch.bool, device=self.device),
                torch.empty((0, 4), dtype=self._q.dtype, device=self.device),
            )

        select_mask = torch.rand(env_ids.numel(), device=self.device) < self._reset_upright_near_table_recovery_prob
        recovery_env_ids = env_ids[select_mask]
        return self._apply_upright_near_table_recovery_reset_for_env_ids(recovery_env_ids, force_far=None)

    def _set_reward_target_quat_from_side_mask(self, env_ids):
        if env_ids.numel() == 0:
            return
        left_mask = self.side_is_left[env_ids]
        target_quat = self.target_quat_right.repeat(len(env_ids), 1)
        if int(left_mask.sum().item()) > 0:
            target_quat[left_mask] = self.target_quat_left.repeat(int(left_mask.sum().item()), 1)
        self.reward_settings["target_quat"][env_ids] = target_quat
        self.reward_settings["target_rot_6d"][env_ids] = matrix_to_rotation_6d(
            quaternion_to_matrix_ig(target_quat)
        )

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
        target_world[:, :2] = mobile_base_pose7[:, :2] + quat_apply(mobile_base_pose7[:, 3:7], base_xy_offset_3d)[:, :2]
        target_world[:, 2] = self.table_surface_height[env_ids].to(dtype=mobile_base_pose7.dtype) + self.post_lift_target_z_from_table
        return target_world

    def _refresh_target_state_for_env_ids(self, env_ids):
        if env_ids.numel() == 0:
            return

        eef_pos = self._eef_state[env_ids, :3]
        eef_quat = self._eef_state[env_ids, 3:7]
        eef_rot_mat = quaternion_to_matrix_ig(eef_quat)
        eef_rot_mat_t = eef_rot_mat.transpose(1, 2)
        target_pos = self.reward_settings["target_pos"][env_ids]
        target_quat = self.reward_settings["target_quat"][env_ids]

        target_to_eef_world = target_pos - eef_pos
        if self.teacher_use_eef_frame:
            target_to_eef = torch.matmul(eef_rot_mat_t, target_to_eef_world.unsqueeze(-1)).squeeze(-1)
        else:
            target_to_eef = target_to_eef_world

        target_rot_mat = quaternion_to_matrix_ig(target_quat)
        target_rot_mat_in_eef = torch.matmul(eef_rot_mat_t, target_rot_mat)
        target_to_eef_rot_6d = matrix_to_rotation_6d(target_rot_mat_in_eef)

        point_matching_err_target = self._get_eef_point_matching_err(
            curent_eef_pos7=self._eef_state[env_ids, :7],
            target_eef_pos7=torch.cat([target_pos, target_quat], dim=-1),
        )
        hand_eef_pos7_rot = torch.cat([self._eef_state[env_ids, :3], target_quat], dim=-1)
        point_matching_err_hand = self._get_eef_point_matching_err(
            curent_eef_pos7=self._eef_state[env_ids, :7],
            target_eef_pos7=hand_eef_pos7_rot,
        )

        self.states["target_to_eef"][env_ids] = target_to_eef
        self.states["target_to_eef_rot_6d"][env_ids] = target_to_eef_rot_6d
        self.states["point_matching_err_target"][env_ids] = point_matching_err_target
        self.states["point_matching_err_hand"][env_ids] = point_matching_err_hand

    def _get_center_tracking_target_pos(self, object_center_pos, env_ids=None):
        # CODEX: single source of truth for center-tracking switch target position.
        if env_ids is None:
            target_pos = object_center_pos.clone()
        else:
            target_pos = object_center_pos[env_ids].clone()
        target_pos[:, 2] = target_pos[:, 2] * 2.0 + self.switching_target_z_offset
        return target_pos
 
    def _update_fabric_switching_target(self, object_center_pos):
        if self.use_center_tracking_switch_target:  # CODEX
            # CODEX: latch switch target pose once and keep fixed during fabric.
            self.switching_target_z_offset = 0.1  # CODEX

            if self.enable_fabric:  # CODEX
                unlatch_ids = (~self.switch_target_quat_latched).nonzero(as_tuple=False).squeeze(-1)  # CODEX
                if unlatch_ids.numel() > 0:  # CODEX
                    eef_pos = self.switching_eef_init_pos[unlatch_ids]  # CODEX
                    obj_pos = self._get_center_tracking_target_pos(object_center_pos, unlatch_ids)  # CODEX
                    desired_dir = obj_pos - eef_pos  # CODEX
                    desired_dir_xy = desired_dir.clone()  # CODEX
                    desired_dir_xy[:, 2] = 0.0  # CODEX
                    desired_dir_xy = desired_dir_xy / torch.norm(desired_dir_xy, dim=-1, keepdim=True).clamp_min(1e-6)  # CODEX

                    # CODEX: yaw-only alignment around world Z so palm stays parallel to table.
                    ref_dir_xy = torch.zeros((unlatch_ids.numel(), 3), device=self.device, dtype=desired_dir_xy.dtype)  # CODEX
                    ref_dir_xy[:, 1] = -1.0  # CODEX: left-side canonical reference direction in XY.
                    cross_z = (ref_dir_xy[:, 0] * desired_dir_xy[:, 1] - ref_dir_xy[:, 1] * desired_dir_xy[:, 0]).unsqueeze(-1)  # CODEX
                    dot_xy = torch.sum(ref_dir_xy[:, :2] * desired_dir_xy[:, :2], dim=-1, keepdim=True)  # CODEX
                    yaw = torch.atan2(cross_z, dot_xy)  # CODEX
                    half_yaw = 0.5 * yaw  # CODEX
                    q_align = torch.zeros((unlatch_ids.numel(), 4), device=self.device, dtype=desired_dir_xy.dtype)  # CODEX
                    q_align[:, 2:3] = torch.sin(half_yaw)  # CODEX
                    q_align[:, 3:4] = torch.cos(half_yaw)  # CODEX
                    base_quat_left = self.target_quat_left.repeat(unlatch_ids.numel(), 1)  # CODEX
                    latched_quat = quat_mul(q_align, base_quat_left)  # CODEX
                    latched_quat = latched_quat / torch.norm(latched_quat, dim=-1, keepdim=True).clamp_min(1e-8)  # CODEX
                    self.switching_target_quat_latched_value[unlatch_ids] = latched_quat  # CODEX
                    self.switching_target_pos_latched_value[unlatch_ids] = obj_pos  # CODEX
                    self.switch_target_quat_latched[unlatch_ids] = True  # CODEX

                self.switching_target_quat = self.switching_target_quat_latched_value  # CODEX
                self.switching_target_pos = self.switching_target_pos_latched_value  # CODEX
            else:  # CODEX
                self.switching_target_pos = self._get_center_tracking_target_pos(object_center_pos)  # CODEX
                self.switching_target_quat = self.target_quat_left.repeat(self.num_envs, 1)  # CODEX

            # CODEX: radius-based trigger around object center.
            center_radius = float(self.fabric_switch_cfg["lateral_offset"])  # CODEX
            self.switch_activate_radius = torch.full((self.num_envs,), center_radius, device=self.device, dtype=object_center_pos.dtype)  # CODEX
            return  # CODEX

        if self.fabric_switch_mode == "simple_anchor":
            switch_geom = self._get_simple_switch_geometry(object_center_pos)
            anchor_target_pos = switch_geom["anchor_target_pos"]
            self.switching_target_pos = anchor_target_pos
            self.switching_target_quat = self.target_quat_left.repeat(self.num_envs, 1)
            self.switch_activate_radius = torch.full(
                (self.num_envs,),
                float(
                    self.fabric_switch_cfg.get(
                        "simple_activate_radius",
                        self.fabric_switch_cfg.get("simple_lateral_offset", self.fabric_switch_cfg["lateral_offset"]),
                    )
                ),
                device=self.device,
                dtype=object_center_pos.dtype,
            )
            return

        # CODEX: pure region-based switching target for non-center mode.
        switch_geom = self._get_noncenter_switch_geometry(object_center_pos)
        high_z = switch_geom["high_z"]
        high_z_boundary = switch_geom["high_z_boundary"]
        stage2_target_pos = switch_geom["stage2_target_pos"]
        stage3_target_pos = switch_geom["stage3_target_pos"]

        stage_target_quat = self.target_quat_left.repeat(self.num_envs, 1)

        eef_pos = self._eef_state[:, :3]
        # Non-center side logic uses the opposite Y-side convention from the earlier draft.
        eef_on_right = eef_pos[:, 1] >= object_center_pos[:, 1]
        eef_on_left = ~eef_on_right
        eef_below_high = eef_pos[:, 2] < high_z_boundary
        high_region = ~eef_below_high
        left_lower_region = eef_on_left & eef_below_high
        left_lower_target_pos = eef_pos.clone()
        left_lower_target_pos[:, 2] = high_z
        stage2_xy_dist = torch.norm(eef_pos[:, :2] - stage2_target_pos[:, :2], dim=-1)
        high_aligned_region = high_region & (stage2_xy_dist <= float(self.fabric_switch_cfg["stage2_xy_tol"]))
        high_far_region = high_region & (~high_aligned_region)

        self.switching_target_pos = torch.where(
            left_lower_region.unsqueeze(-1),
            left_lower_target_pos,
            torch.where(high_far_region.unsqueeze(-1), stage2_target_pos, stage3_target_pos),
        )
        teacher_active = ~self.fabric_switch_enable
        if torch.any(teacher_active):
            self.switching_target_pos[teacher_active] = object_center_pos[teacher_active]
        self.switching_target_quat = stage_target_quat

        switch_radius = torch.norm(self.switching_target_pos - object_center_pos, dim=-1)
        self.switch_activate_radius = switch_radius

    def _get_noncenter_switch_geometry(self, object_center_pos):
        lateral_offset = float(self.fabric_switch_cfg.get("simple_lateral_offset", self.fabric_switch_cfg["lateral_offset"]))
        self.switching_target_z_offset = float(self.fabric_switch_cfg["high_z_offset"])
        high_z_tol = float(self.fabric_switch_cfg["high_z_tol"])
        object_top_z = self.table_surface_height + self.mesh_aabb_extents[:, 2]
        high_z = object_top_z + self.switching_target_z_offset
        high_z_boundary = high_z - high_z_tol
        teacher_z_max = object_top_z + float(self.fabric_switch_cfg["teacher_activate_z_offset"])

        stage2_target_pos = object_center_pos.clone()
        stage2_target_pos[:, 1] += lateral_offset
        stage2_target_pos[:, 2] = high_z

        left_lower_target_pos = object_center_pos.clone()
        left_lower_target_pos[:, 1] -= lateral_offset
        left_lower_target_pos[:, 2] = high_z

        stage3_target_pos = object_center_pos.clone()
        stage3_target_pos[:, 1] += lateral_offset
        stage3_target_pos[:, 2] = object_top_z + float(self.fabric_switch_cfg["stage3_z_offset"])

        return {
            "lateral_offset": lateral_offset,
            "high_z": high_z,
            "high_z_boundary": high_z_boundary,
            "teacher_z_max": teacher_z_max,
            "left_lower_target_pos": left_lower_target_pos,
            "stage2_target_pos": stage2_target_pos,
            "stage3_target_pos": stage3_target_pos,
        }

    def _get_simple_switch_geometry(self, object_center_pos):
        lateral_offset = float(self.fabric_switch_cfg["lateral_offset"])
        object_top_z = self.table_surface_height + self.mesh_aabb_extents[:, 2]

        anchor_target_pos = object_center_pos.clone()
        anchor_target_pos[:, 1] += lateral_offset
        anchor_target_pos[:, 2] = object_top_z

        return {
            "lateral_offset": lateral_offset,
            "anchor_target_pos": anchor_target_pos,
        }

    def _get_noncenter_switch_region_codes(self, object_center_pos):
        switch_geom = self._get_noncenter_switch_geometry(object_center_pos)
        eef_pos = self._eef_state[:, :3]
        teacher_activation_z = self._get_teacher_activation_z()
        eef_on_right = eef_pos[:, 1] >= object_center_pos[:, 1]
        eef_on_left = ~eef_on_right
        eef_below_high = eef_pos[:, 2] < switch_geom["high_z_boundary"]
        stage2_xy_dist = torch.norm(eef_pos[:, :2] - switch_geom["stage2_target_pos"][:, :2], dim=-1)
        object_xy_dist = torch.norm(eef_pos[:, :2] - object_center_pos[:, :2], dim=-1)
        stage3_xy_dist = torch.norm(eef_pos[:, :2] - switch_geom["stage3_target_pos"][:, :2], dim=-1)
        object_close = object_xy_dist <= float(self.fabric_switch_cfg["lateral_offset"])
        stage3_close = stage3_xy_dist <= self.pre_teacher_stage3_xy_tol
        right_close = eef_on_right & (teacher_activation_z <= switch_geom["teacher_z_max"]) & (object_close | stage3_close)
        high_region = ~eef_below_high
        high_aligned = high_region & (stage2_xy_dist <= float(self.fabric_switch_cfg["stage2_xy_tol"]))
        high_far = high_region & (~high_aligned)
        left_lower = eef_on_left & eef_below_high
        right_lower = eef_on_right & eef_below_high & (~right_close)

        region_codes = torch.full((self.num_envs,), 3, device=self.device, dtype=torch.long)
        region_codes[left_lower] = 0
        region_codes[high_far] = 1
        region_codes[high_aligned] = 2
        region_codes[right_lower] = 3
        region_codes[right_close] = 4
        return region_codes, switch_geom

    def _get_teacher_activation_z(self):
        return self._eef_state[:, 2]

    def _debug_print_joint_limits(self, env_id=2):
        return

    def init_data(self, actor_num):
        super().init_data(actor_num)
        self.side_is_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.switch_target_quat_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # CODEX
        self.switching_eef_init_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)  # CODEX
        self.switching_target_quat_latched_value = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)  # CODEX
        self.switching_target_pos_latched_value = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)  # CODEX
        self.ep_hist_window_sizes = [max(1, self.num_envs // 2), self.num_envs, self.num_envs * 2, self.num_envs * 4]  # CODEX
        self.ep_hist_max = int(self.ep_hist_window_sizes[-1])  # CODEX
        self.ep_hist_success = torch.full((self.ep_hist_max,), -1, device=self.device, dtype=torch.int8)  # CODEX
        self.ep_hist_lifting = torch.full((self.ep_hist_max,), -1, device=self.device, dtype=torch.int8)  # CODEX
        self.ep_hist_category = torch.full((self.ep_hist_max,), -1, device=self.device, dtype=torch.long)
        self.ep_hist_ptr = 0  # CODEX
        self.ep_hist_count = 0  # CODEX
        self.current_episode_verified_bank_category = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.per_verified_bank_episode_counts = torch.zeros(
            (len(self.VERIFIED_BANK_CATEGORY_NAMES),), device=self.device, dtype=torch.long
        )
        self.per_verified_bank_success_counts = torch.zeros(
            (len(self.VERIFIED_BANK_CATEGORY_NAMES),), device=self.device, dtype=torch.long
        )
        self.pre_teacher_stage3_xy_tol = float(self.fabric_switch_cfg["stage3_xy_tol"])
        self.pre_teacher_stage3_xy_tol_exit = float(self.fabric_switch_cfg["stage3_xy_tol_exit"])
        self.pre_teacher_object_xy_tol = float(self.fabric_switch_cfg["lateral_offset"])
        self.pre_teacher_object_xy_tol_exit = float(
            self.fabric_switch_cfg.get(
                "object_xy_tol_exit",
                self.pre_teacher_object_xy_tol + (self.pre_teacher_stage3_xy_tol_exit - self.pre_teacher_stage3_xy_tol),
            )
        )
        self.flat_table_reset_steps = int(self.fabric_switch_cfg["flat_table_reset_steps"])
        self.debug_last_region_code = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.debug_viz_env_ids = [0,1,2,3]
        self.flat_table_duration = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        self.flat_table_long_enough = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.post_lift_target_active = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.debug_recovery_reset_active = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.debug_recovery_reset_palm_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.debug_recovery_reset_palm_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.debug_recovery_reset_palm_quat[:, 3] = 1.0
        self._init_verified_teacher_bank_storage()
        if self._activation_snapshot_bank_enable:
            self._init_activation_snapshot_bank_storage()
        if self.debug_viz:
            self.debug_success_total = 0
            self.debug_failure_total = 0
            self._table_debug_color_state_prev = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
            self._table_debug_outcome_hold_steps = 12
            self._table_debug_outcome_state = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
            self._table_debug_outcome_steps_left = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)

        # Teacher-consistent target position semantics:
        # fixed per reset, anchored directly above object-init XY with table-height-relative Z.
        self.reward_settings["target_pos"] = self._build_fixed_target_pos()


        # @ray we use target_quat for reward computation and construct it based on the mode "left/right/both"
        self.target_quat_right = to_torch(self.cfg["reward"]["params"]["target_quat_right"], device=self.device).unsqueeze(0)
        self.target_quat_right = self.target_quat_right / (
            torch.norm(self.target_quat_right, dim=1, keepdim=True) + 1e-10
        )
        self.target_quat_left = to_torch(self.cfg["reward"]["params"]["target_quat_left"], device=self.device).unsqueeze(0)
        self.target_quat_left = self.target_quat_left / (
            torch.norm(self.target_quat_left, dim=1, keepdim=True) + 1e-10
        )
        
        self.reward_settings["beta_hand_orientation"] = to_torch(self.cfg["reward"]["exp"]["beta_hand_orientation"], device=self.device)
        self.reward_settings["w_hand_orientation"] = to_torch(self.cfg["reward"]["weights"]["w_hand_orientation"], device=self.device)
        # @ray to force the hand to grasp on the correct region on the object (in terms of z-axis), we gate the lift and to-goal rewards with a sigmoid based on z-difference 
        # between the grasp_target's z height and the averaged finger height on obejct. 
        # @ray we add an offset to the grasp_target z_height to match the height of the stacked fingers. 
        # This is crucial to make reward consistent across objects of varying height.
        # @ray we also add a floor value to make the -gate not 0, otherwise the policy loses incentive to lift early on when its grasp style is not good enough
        self.reward_settings["grasp_on_object_z_height_tolerance"] = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_tolerance"], device=self.device)
        self.reward_settings["grasp_on_object_z_height_slope"] = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_slope"], device=self.device)
        self.reward_settings["grasp_on_object_z_height_offset"] = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_offset"], device=self.device)
        self.reward_settings["grasp_on_object_z_height_gate_floor"]  = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_gate_floor"], device=self.device)
        self.reward_settings["grasp_on_object_z_height_gate_enabled"] = to_torch(1.0 if bool(self.cfg["reward"]["params"]["grasp_on_object_z_height_gate_enabled"]) else 0.0, device=self.device)
        self.reward_settings["success_tolerance"] = to_torch(self.cfg["reward"]["params"]["success_tolerance"], device=self.device)

        # Initialize right-side-only target quaternions for all envs.
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.side_is_left[:] = True
        self._set_reward_target_quat_from_side_mask(all_env_ids)


    def post_physics_step(self):
        pending_snapshot_env_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        if hasattr(self, "_verified_teacher_bank_snapshot_pending"):
            pending_snapshot_env_ids = self._verified_teacher_bank_snapshot_pending.nonzero(as_tuple=False).squeeze(-1).clone()
        super().post_physics_step()
        self._flush_pending_verified_teacher_episode_starts(pending_snapshot_env_ids)
        if self.debug_viz:
            self._update_table_lift_colors()
        # Visualize switching target frame for debugging in viewer mode.
        if self.debug_viz:
            self.gym.clear_lines(self.viewer)
            for debug_env_id in self.debug_viz_env_ids:
                # self._draw_base_init_pose_grid(env_id=debug_env_id, clear_lines=False)
                # self._draw_switching_target_pose(env_id=debug_env_id, axis_len=0.10, clear_lines=False)
                # self._draw_eef_quat_at_teacher_target_pose(env_id=debug_env_id, axis_len=0.095, clear_lines=False)  # CODEX
                # self._draw_teacher_target_pose(env_id=debug_env_id, axis_len=0.08, clear_lines=False)
                self._draw_fabric_switch_regions(env_id=debug_env_id, clear_lines=False)
                self._draw_base_relative_target(env_id=debug_env_id, axis_len=0.08, clear_lines=False)
                self._draw_recovery_reset_pose(env_id=debug_env_id, axis_len=0.07, clear_lines=False)
            # self._draw_object_grasp_center(env_id=debug_env_id, cross_len=0.2, clear_lines=False)
            # self._draw_observation_rays_from_eef(env_id=debug_env_id, clear_lines=False)  # CODEX
                pass

    def _update_table_lift_colors(self):
        if (not self.debug_viz) or (not hasattr(self, "tables")) or len(self.tables) != self.num_envs:
            return

        # 0=default gray, 1=success green, 2=failure red. Failure takes priority.
        color_state_live = torch.where(
            self.flat_table_long_enough,
            torch.full_like(self._table_debug_color_state_prev, 2),
            torch.where(
                self.success_long_enough,
                torch.full_like(self._table_debug_color_state_prev, 1),
                torch.zeros_like(self._table_debug_color_state_prev),
            ),
        )
        show_last_outcome = self._table_debug_outcome_steps_left > 0
        color_state = torch.where(show_last_outcome, self._table_debug_outcome_state, color_state_live)
        changed_mask = color_state != self._table_debug_color_state_prev
        if not torch.any(changed_mask):
            self._table_debug_outcome_steps_left[show_last_outcome] -= 1
            return

        changed_ids = changed_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        default_color = gymapi.Vec3(0.55, 0.55, 0.55)
        success_color = gymapi.Vec3(0.0, 0.85, 0.0)
        failure_color = gymapi.Vec3(0.9, 0.0, 0.0)
        for env_id in changed_ids:
            state = int(color_state[env_id].item())
            if state == 2:
                color = failure_color
            elif state == 1:
                color = success_color
            else:
                color = default_color
            self.gym.set_rigid_body_color(
                self.envs[env_id],
                self.tables[env_id],
                0,
                gymapi.MESH_VISUAL_AND_COLLISION,
                color,
            )
        self._table_debug_color_state_prev.copy_(color_state)
        self._table_debug_outcome_steps_left[show_last_outcome] -= 1

    def _draw_fabric_switch_regions(self, env_id=0, clear_lines=False):
        if self.viewer is None or self.use_center_tracking_switch_target:
            return
        if clear_lines:
            self.gym.clear_lines(self.viewer)

        object_center_pos = self.states["object_center_pos"]
        if self.fabric_switch_mode == "simple_anchor":
            obj = object_center_pos[env_id]
            anchor_target = self.switching_target_pos[env_id]
            enter_r = float(self.fabric_switch_cfg.get("simple_activate_radius", self.fabric_switch_cfg["lateral_offset"]))
            exit_r = float(
                self.fabric_switch_cfg.get(
                    "simple_activate_radius_exit",
                    self.fabric_switch_cfg.get("object_xy_tol_exit", enter_r + 0.03),
                )
            )

            verts = []
            colors = []

            def add_line(p0, p1, color):
                verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(p1[0]), float(p1[1]), float(p1[2])])
                colors.extend(color)

            def add_sphere(radius, color, num_seg=32):
                cx = float(obj[0].item())
                cy = float(obj[1].item())
                cz = float(obj[2].item())
                for i in range(num_seg):
                    a0 = 2.0 * np.pi * float(i) / float(num_seg)
                    a1 = 2.0 * np.pi * float(i + 1) / float(num_seg)
                    add_line(
                        [cx + radius * np.cos(a0), cy + radius * np.sin(a0), cz],
                        [cx + radius * np.cos(a1), cy + radius * np.sin(a1), cz],
                        color,
                    )
                    add_line(
                        [cx + radius * np.cos(a0), cy, cz + radius * np.sin(a0)],
                        [cx + radius * np.cos(a1), cy, cz + radius * np.sin(a1)],
                        color,
                    )
                    add_line(
                        [cx, cy + radius * np.cos(a0), cz + radius * np.sin(a0)],
                        [cx, cy + radius * np.cos(a1), cz + radius * np.sin(a1)],
                        color,
                    )

            add_sphere(enter_r, [1.0, 0.8, 0.2])
            add_sphere(exit_r, [1.0, 1.0, 0.0])

            cross_len = 0.03
            add_line(anchor_target + torch.tensor([-cross_len, 0.0, 0.0], device=self.device), anchor_target + torch.tensor([cross_len, 0.0, 0.0], device=self.device), [0.0, 0.8, 1.0])
            add_line(anchor_target + torch.tensor([0.0, -cross_len, 0.0], device=self.device), anchor_target + torch.tensor([0.0, cross_len, 0.0], device=self.device), [0.0, 0.8, 1.0])
            add_line(anchor_target + torch.tensor([0.0, 0.0, -cross_len], device=self.device), anchor_target + torch.tensor([0.0, 0.0, cross_len], device=self.device), [0.0, 0.8, 1.0])

            self.gym.add_lines(self.viewer, self.envs[env_id], len(verts) // 6, verts, colors)
            return

        region_codes, switch_geom = self._get_noncenter_switch_region_codes(object_center_pos)

        obj = object_center_pos[env_id]
        eef = self._eef_state[env_id, :3]
        teacher_activation_z = float(self._get_teacher_activation_z()[env_id].item())
        high_z = float(switch_geom["high_z"][env_id].item())
        teacher_z_max = float(switch_geom["teacher_z_max"][env_id].item())
        stage2_target = switch_geom["stage2_target_pos"][env_id]
        stage3_target = switch_geom["stage3_target_pos"][env_id]
        xy_tol = float(self.pre_teacher_stage3_xy_tol)
        stage2_xy_tol = float(self.fabric_switch_cfg["stage2_xy_tol"])

        x_half_span = 0.18
        y_half_span = max(0.02, float(switch_geom["lateral_offset"]) + 0.05)
        z_min = float(obj[2].item()) - 0.02
        z_max = max(high_z, float(eef[2].item()), teacher_z_max) + 0.05
        boundary_y = float(obj[1].item())

        verts = []
        colors = []

        def add_line(p0, p1, color):
            verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(p1[0]), float(p1[1]), float(p1[2])])
            colors.extend(color)

        # Left/right boundary through the object center.
        add_line(
            [float(obj[0]) - x_half_span, boundary_y, z_min],
            [float(obj[0]) - x_half_span, boundary_y, z_max],
            [1.0, 1.0, 0.0],
        )
        add_line(
            [float(obj[0]) + x_half_span, boundary_y, z_min],
            [float(obj[0]) + x_half_span, boundary_y, z_max],
            [1.0, 1.0, 0.0],
        )
        add_line(
            [float(obj[0]) - x_half_span, boundary_y, high_z],
            [float(obj[0]) + x_half_span, boundary_y, high_z],
            [0.0, 0.8, 1.0],
        )

        # Stage 2 and Stage 3 target markers.
        for target_pos, color in ((stage2_target, [0.0, 0.8, 1.0]), (stage3_target, [1.0, 0.0, 1.0])):
            cross_len = 0.03
            add_line(target_pos + torch.tensor([-cross_len, 0.0, 0.0], device=self.device), target_pos + torch.tensor([cross_len, 0.0, 0.0], device=self.device), color)
            add_line(target_pos + torch.tensor([0.0, -cross_len, 0.0], device=self.device), target_pos + torch.tensor([0.0, cross_len, 0.0], device=self.device), color)
            add_line(target_pos + torch.tensor([0.0, 0.0, -cross_len], device=self.device), target_pos + torch.tensor([0.0, 0.0, cross_len], device=self.device), color)

        # High-aligned XY boundary around the Stage 2 target at high_z.
        stage2_box_min_x = float(stage2_target[0].item()) - stage2_xy_tol
        stage2_box_max_x = float(stage2_target[0].item()) + stage2_xy_tol
        stage2_box_min_y = float(stage2_target[1].item()) - stage2_xy_tol
        stage2_box_max_y = float(stage2_target[1].item()) + stage2_xy_tol
        add_line([stage2_box_min_x, stage2_box_min_y, high_z], [stage2_box_max_x, stage2_box_min_y, high_z], [0.2, 1.0, 0.2])
        add_line([stage2_box_max_x, stage2_box_min_y, high_z], [stage2_box_max_x, stage2_box_max_y, high_z], [0.2, 1.0, 0.2])
        add_line([stage2_box_max_x, stage2_box_max_y, high_z], [stage2_box_min_x, stage2_box_max_y, high_z], [0.2, 1.0, 0.2])
        add_line([stage2_box_min_x, stage2_box_max_y, high_z], [stage2_box_min_x, stage2_box_min_y, high_z], [0.2, 1.0, 0.2])

        # Object-centered close region at the teacher activation height.
        close_z = teacher_z_max
        object_close_radius = float(switch_geom["lateral_offset"])
        obj_box_min_x = float(obj[0].item()) - object_close_radius
        obj_box_max_x = float(obj[0].item()) + object_close_radius
        obj_box_min_y = float(obj[1].item()) - object_close_radius
        obj_box_max_y = float(obj[1].item()) + object_close_radius
        add_line([obj_box_min_x, obj_box_min_y, close_z], [obj_box_max_x, obj_box_min_y, close_z], [1.0, 0.5, 0.0])
        add_line([obj_box_max_x, obj_box_min_y, close_z], [obj_box_max_x, obj_box_max_y, close_z], [1.0, 0.5, 0.0])
        add_line([obj_box_max_x, obj_box_max_y, close_z], [obj_box_min_x, obj_box_max_y, close_z], [1.0, 0.5, 0.0])
        add_line([obj_box_min_x, obj_box_max_y, close_z], [obj_box_min_x, obj_box_min_y, close_z], [1.0, 0.5, 0.0])

        # Small Stage 3 tolerance region attached to the object-centered close region.
        stage3_box_min_x = float(stage3_target[0].item()) - xy_tol
        stage3_box_max_x = float(stage3_target[0].item()) + xy_tol
        stage3_box_min_y = float(stage3_target[1].item()) - xy_tol
        stage3_box_max_y = float(stage3_target[1].item()) + xy_tol
        add_line([stage3_box_min_x, stage3_box_min_y, close_z], [stage3_box_max_x, stage3_box_min_y, close_z], [1.0, 0.8, 0.2])
        add_line([stage3_box_max_x, stage3_box_min_y, close_z], [stage3_box_max_x, stage3_box_max_y, close_z], [1.0, 0.8, 0.2])
        add_line([stage3_box_max_x, stage3_box_max_y, close_z], [stage3_box_min_x, stage3_box_max_y, close_z], [1.0, 0.8, 0.2])
        add_line([stage3_box_min_x, stage3_box_max_y, close_z], [stage3_box_min_x, stage3_box_min_y, close_z], [1.0, 0.8, 0.2])

        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts) // 6, verts, colors)

        region_names = ["left_lower", "high_far", "high_aligned", "right_lower", "right_close"]
        region_code = int(region_codes[env_id].item())
        if int(self.debug_last_region_code[env_id].item()) != region_code:
            self.debug_last_region_code[env_id] = region_code

    def _draw_base_init_pose_grid(self, env_id=0, clear_lines=False, num_div_x=8, num_div_y=6):
        if self.viewer is None:
            return
        if clear_lines:
            self.gym.clear_lines(self.viewer)

        base_init_range = self.cfg["env"]["robot_init"]["base_init_range"]
        x_min, y_min, _ = base_init_range[0]
        x_max, y_max, _ = base_init_range[1]

        # Match reset sampling convention where Y is shifted by box position.
        y_shift = 0.0
        if hasattr(self, "box_pos"):
            y_shift = float(self.box_pos[env_id, 1].item())
        y_min += y_shift
        y_max += y_shift

        z = float(self.table_surface_height[env_id].item()) + 0.01
        num_div_x = max(1, int(num_div_x))
        num_div_y = max(1, int(num_div_y))

        verts = []
        colors = []

        for i in range(num_div_x + 1):
            t = i / float(num_div_x)
            x = x_min + t * (x_max - x_min)
            verts.extend([x, y_min, z, x, y_max, z])
            is_border = (i == 0) or (i == num_div_x)
            c = [1.0, 1.0, 1.0] if is_border else [0.7, 0.7, 0.7]
            colors.extend(c)

        for j in range(num_div_y + 1):
            t = j / float(num_div_y)
            y = y_min + t * (y_max - y_min)
            verts.extend([x_min, y, z, x_max, y, z])
            is_border = (j == 0) or (j == num_div_y)
            c = [1.0, 1.0, 1.0] if is_border else [0.7, 0.7, 0.7]
            colors.extend(c)

        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts) // 6, verts, colors)

    def _draw_recovery_reset_pose(self, env_id=0, axis_len=0.07, clear_lines=False, num_div_x=8, num_div_y=6):
        if self.viewer is None:
            return
        if clear_lines:
            self.gym.clear_lines(self.viewer)

        object_center = self.states["object_center_pos"][env_id]
        verts = []
        colors = []

        def add_line(p0, p1, color):
            verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(p1[0]), float(p1[1]), float(p1[2])])
            colors.extend(color)

        def add_xy_grid(x_min, x_max, y_min, y_max, z, border_color, interior_color):
            num_x = max(1, int(num_div_x))
            num_y = max(1, int(num_div_y))
            for i in range(num_x + 1):
                t = i / float(num_x)
                x = x_min + t * (x_max - x_min)
                color = border_color if (i == 0 or i == num_x) else interior_color
                add_line([x, y_min, z], [x, y_max, z], color)
            for j in range(num_y + 1):
                t = j / float(num_y)
                y = y_min + t * (y_max - y_min)
                color = border_color if (j == 0 or j == num_y) else interior_color
                add_line([x_min, y, z], [x_max, y, z], color)

        table_center_xy = self.table_pos[env_id:env_id + 1, :2].to(dtype=object_center.dtype)
        table_half_xy = 0.5 * self.table_size[env_id:env_id + 1, :2].to(dtype=object_center.dtype)
        table_xy_min = table_center_xy - table_half_xy
        table_xy_max = table_center_xy + table_half_xy
        object_box_min_batch, object_box_max_batch, valid_clip_xy = self._get_clipped_recovery_palm_rel_object_bounds(
            object_center.unsqueeze(0),
            table_xy_min,
            table_xy_max,
            object_center.dtype,
        )
        outer_box_min_batch, outer_box_max_batch, valid_outer_clip_xy = self._get_clipped_recovery_palm_rel_object_bounds(
            object_center.unsqueeze(0),
            table_xy_min,
            table_xy_max,
            object_center.dtype,
            rel_object_min_cfg=self._reset_upright_near_table_palm_rel_object_far_min_cfg,
            rel_object_max_cfg=self._reset_upright_near_table_palm_rel_object_far_max_cfg,
        )
        if (not bool(valid_clip_xy[0].item())) and (not bool(valid_outer_clip_xy[0].item())):
            return
        def add_box(min_corner, max_corner, origin_xy, quat_xyzw, color):
            local_corners = torch.tensor(
                [
                    [min_corner[0], min_corner[1], min_corner[2]],
                    [max_corner[0], min_corner[1], min_corner[2]],
                    [max_corner[0], max_corner[1], min_corner[2]],
                    [min_corner[0], max_corner[1], min_corner[2]],
                    [min_corner[0], min_corner[1], max_corner[2]],
                    [max_corner[0], min_corner[1], max_corner[2]],
                    [max_corner[0], max_corner[1], max_corner[2]],
                    [min_corner[0], max_corner[1], max_corner[2]],
                ],
                device=self.device,
                dtype=object_center.dtype,
            )
            world_corners = quat_apply(quat_xyzw.repeat(8, 1), local_corners)
            world_corners[:, 0] += origin_xy[0]
            world_corners[:, 1] += origin_xy[1]
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for i0, i1 in edges:
                add_line(world_corners[i0], world_corners[i1], color)

        # Mobile-base recovery init XY range. The config includes yaw too, but the
        # draw focuses on the translational region since that is the relevant footprint.
        recovery_base_init_range = self._reset_upright_near_table_base_init_range_cfg
        base_x_min, base_y_min, _ = recovery_base_init_range[0]
        base_x_max, base_y_max, _ = recovery_base_init_range[1]
        base_grid_z = float(self.table_surface_height[env_id].item()) + 0.015
        add_xy_grid(
            float(min(base_x_min, base_x_max)),
            float(max(base_x_min, base_x_max)),
            float(min(base_y_min, base_y_max)),
            float(max(base_y_min, base_y_max)),
            base_grid_z,
            [0.2, 0.9, 1.0],
            [0.1, 0.45, 0.55],
        )

        identity_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device, dtype=object_center.dtype)[0:1]
        # Object-relative outer far-sampling box.
        if bool(valid_outer_clip_xy[0].item()):
            outer_box_min = outer_box_min_batch[0].clone()
            outer_box_max = outer_box_max_batch[0].clone()
            outer_box_min[2] = object_center[2] + outer_box_min[2]
            outer_box_max[2] = object_center[2] + outer_box_max[2]
            add_box(
                outer_box_min,
                outer_box_max,
                object_center[:2],
                identity_quat,
                [1.0, 0.9, 0.2],
            )

        # Object-relative inner near-sampling box.
        if bool(valid_clip_xy[0].item()):
            object_box_min = object_box_min_batch[0].clone()
            object_box_max = object_box_max_batch[0].clone()
            object_box_min[2] = object_center[2] + object_box_min[2]
            object_box_max[2] = object_center[2] + object_box_max[2]
            add_box(
                object_box_min,
                object_box_max,
                object_center[:2],
                identity_quat,
                [1.0, 0.6, 0.1],
            )

        # Current palm center in world (green cross).
        p_cur = self._eef_state[env_id, :3]
        cross = float(axis_len)
        add_line(
            p_cur + torch.tensor([cross, 0.0, 0.0], device=self.device),
            p_cur - torch.tensor([cross, 0.0, 0.0], device=self.device),
            [0.1, 1.0, 0.1],
        )
        add_line(
            p_cur + torch.tensor([0.0, cross, 0.0], device=self.device),
            p_cur - torch.tensor([0.0, cross, 0.0], device=self.device),
            [0.1, 1.0, 0.1],
        )
        add_line(
            p_cur + torch.tensor([0.0, 0.0, cross], device=self.device),
            p_cur - torch.tensor([0.0, 0.0, cross], device=self.device),
            [0.1, 1.0, 0.1],
        )

        # Optional sampled recovery palm target, if the current episode actually used one.
        if bool(self.debug_recovery_reset_active[env_id].item()):
            p = self.debug_recovery_reset_palm_pos[env_id]
            add_line(p + torch.tensor([cross, 0.0, 0.0], device=self.device), p - torch.tensor([cross, 0.0, 0.0], device=self.device), [1.0, 1.0, 1.0])
            add_line(p + torch.tensor([0.0, cross, 0.0], device=self.device), p - torch.tensor([0.0, cross, 0.0], device=self.device), [1.0, 1.0, 1.0])
            add_line(p + torch.tensor([0.0, 0.0, cross], device=self.device), p - torch.tensor([0.0, 0.0, cross], device=self.device), [1.0, 1.0, 1.0])

        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts) // 6, verts, colors)

    def _draw_switching_target_pose(self, env_id=0, axis_len=0.10, clear_lines=False):
        if self.viewer is None:
            return
        if clear_lines:
            self.gym.clear_lines(self.viewer)

        pos = self.switching_target_pos[env_id]
        quat = self.switching_target_quat[env_id:env_id + 1]
        rot = quaternion_to_matrix_ig(quat)[0]

        x_axis = rot[:, 0]
        y_axis = rot[:, 1]
        z_axis = rot[:, 2]

        p0 = pos
        px = pos + axis_len * x_axis
        py = pos + axis_len * y_axis
        pz = pos + axis_len * z_axis

        # Red=X, Green=Y, Blue=Z frame visualization.
        verts = []
        colors = []
        verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(px[0]), float(px[1]), float(px[2])])
        colors.extend([1.0, 0.0, 0.0])
        verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(py[0]), float(py[1]), float(py[2])])
        colors.extend([0.0, 1.0, 0.0])
        verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(pz[0]), float(pz[1]), float(pz[2])])
        colors.extend([0.0, 0.0, 1.0])

        self.gym.add_lines(self.viewer, self.envs[env_id], 3, verts, colors)

    def _draw_eef_quat_at_teacher_target_pose(self, env_id=0, axis_len=0.095, clear_lines=False):  # CODEX
        # CODEX: overlay live EEF orientation at teacher-target position for direct visual alignment check.
        if self.viewer is None:  # CODEX
            return  # CODEX
        if clear_lines:  # CODEX
            self.gym.clear_lines(self.viewer)  # CODEX

        pos = self.reward_settings["target_pos"][env_id]  # CODEX
        quat = self._eef_state[env_id:env_id + 1, 3:7]  # CODEX
        rot = quaternion_to_matrix_ig(quat)[0]  # CODEX

        x_axis = rot[:, 0]  # CODEX
        y_axis = rot[:, 1]  # CODEX
        z_axis = rot[:, 2]  # CODEX

        p0 = pos  # CODEX
        px = pos + axis_len * x_axis  # CODEX
        py = pos + axis_len * y_axis  # CODEX
        pz = pos + axis_len * z_axis  # CODEX

        verts = []  # CODEX
        colors = []  # CODEX
        # CODEX: orange/cyan/purple so this EEF overlay is distinguishable from target frames.
        verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(px[0]), float(px[1]), float(px[2])])  # CODEX
        colors.extend([1.0, 0.6, 0.0])  # CODEX
        verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(py[0]), float(py[1]), float(py[2])])  # CODEX
        colors.extend([0.0, 1.0, 1.0])  # CODEX
        verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(pz[0]), float(pz[1]), float(pz[2])])  # CODEX
        colors.extend([0.8, 0.2, 1.0])  # CODEX

        self.gym.add_lines(self.viewer, self.envs[env_id], 3, verts, colors)  # CODEX

    def _draw_teacher_target_pose(self, env_id=0, axis_len=0.08, clear_lines=False):
        # Draw the actual teacher target pose used in observations/reward_settings.
        if clear_lines:
            self.gym.clear_lines(self.viewer)

        pos = self.reward_settings["target_pos"][env_id]
        quat = self.reward_settings["target_quat"][env_id:env_id + 1]
        rot = quaternion_to_matrix_ig(quat)[0]

        x_axis = rot[:, 0]
        y_axis = rot[:, 1]
        z_axis = rot[:, 2]

        p0 = pos
        px = pos + axis_len * x_axis
        py = pos + axis_len * y_axis
        pz = pos + axis_len * z_axis

        verts = []
        colors = []
        # Brighter RGB so teacher target frame is distinguishable from switching target frame.
        verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(px[0]), float(px[1]), float(px[2])])
        colors.extend([1.0, 0.3, 0.3])
        verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(py[0]), float(py[1]), float(py[2])])
        colors.extend([0.3, 1.0, 0.3])
        verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(pz[0]), float(pz[1]), float(pz[2])])
        colors.extend([0.3, 0.7, 1.0])

        self.gym.add_lines(self.viewer, self.envs[env_id], 3, verts, colors)

    def _draw_base_relative_target(self, env_id=0, axis_len=0.08, clear_lines=False):
        if self.viewer is None:
            return
        if clear_lines:
            self.gym.clear_lines(self.viewer)

        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        base_pose7 = self._get_current_mobile_base_pose7(env_ids)[0]
        base_pos = base_pose7[:3]
        base_quat = base_pose7[3:7].unsqueeze(0)
        base_rot = quaternion_to_matrix_ig(base_quat)[0]

        base_relative_target_world = self._compute_post_lift_target_world(env_ids)[0]
        current_target_world = self.reward_settings["target_pos"][env_id]

        x_axis = base_rot[:, 0]
        y_axis = base_rot[:, 1]
        z_axis = base_rot[:, 2]

        verts = []
        colors = []

        def add_line(p0, p1, color):
            verts.extend([float(p0[0]), float(p0[1]), float(p0[2]), float(p1[0]), float(p1[1]), float(p1[2])])
            colors.extend(color)

        # Base frame at the mobile-base root.
        add_line(base_pos, base_pos + axis_len * x_axis, [1.0, 0.0, 0.0])
        add_line(base_pos, base_pos + axis_len * y_axis, [0.0, 1.0, 0.0])
        add_line(base_pos, base_pos + axis_len * z_axis, [0.0, 0.0, 1.0])

        # White line: base origin to the fixed final goal target in world.
        add_line(base_pos, base_relative_target_world, [1.0, 1.0, 1.0])

        # Cyan line: fixed final target to the currently active waypoint/reward target.
        add_line(base_relative_target_world, current_target_world, [0.0, 1.0, 1.0])

        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts) // 6, verts, colors)

        cross_len = 0.03
        target_verts = []
        target_colors = []
        for center, color in (
            (base_relative_target_world, [1.0, 1.0, 1.0]),
            (current_target_world, [1.0, 0.0, 1.0]),
        ):
            target_verts.extend([
                float(center[0] - cross_len), float(center[1]), float(center[2]),
                float(center[0] + cross_len), float(center[1]), float(center[2]),
            ])
            target_colors.extend(color)
            target_verts.extend([
                float(center[0]), float(center[1] - cross_len), float(center[2]),
                float(center[0]), float(center[1] + cross_len), float(center[2]),
            ])
            target_colors.extend(color)
            target_verts.extend([
                float(center[0]), float(center[1]), float(center[2] - cross_len),
                float(center[0]), float(center[1]), float(center[2] + cross_len),
            ])
            target_colors.extend(color)

        self.gym.add_lines(self.viewer, self.envs[env_id], len(target_verts) // 6, target_verts, target_colors)

    
    def _draw_object_grasp_center(self, env_id=0, cross_len=0.035, clear_lines=False):
        if self.viewer is None:
            return
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        if "object_grasp_target_pos" not in self.states:
            return
        c = self.states["object_grasp_target_pos"][env_id]
        c_center = self.states["object_center_pos"][env_id] if "object_center_pos" in self.states else None
        # Also plot raw object root position to validate grasp-target offset visually.
        c_root = self._object_state[env_id, :3]

        p_xm = c + torch.tensor([-cross_len, 0.0, 0.0], device=self.device)
        p_xp = c + torch.tensor([ cross_len, 0.0, 0.0], device=self.device)
        p_ym = c + torch.tensor([0.0, -cross_len, 0.0], device=self.device)
        p_yp = c + torch.tensor([0.0,  cross_len, 0.0], device=self.device)
        p_zm = c + torch.tensor([0.0, 0.0, -cross_len], device=self.device)
        p_zp = c + torch.tensor([0.0, 0.0,  cross_len], device=self.device)

        verts = []
        colors = []
        # Yellow XYZ cross at object grasp center.
        verts.extend([float(p_xm[0]), float(p_xm[1]), float(p_xm[2]), float(p_xp[0]), float(p_xp[1]), float(p_xp[2])])
        colors.extend([1.0, 1.0, 0.0])
        verts.extend([float(p_ym[0]), float(p_ym[1]), float(p_ym[2]), float(p_yp[0]), float(p_yp[1]), float(p_yp[2])])
        colors.extend([1.0, 1.0, 0.0])
        verts.extend([float(p_zm[0]), float(p_zm[1]), float(p_zm[2]), float(p_zp[0]), float(p_zp[1]), float(p_zp[2])])
        colors.extend([1.0, 1.0, 0.0])

        self.gym.add_lines(self.viewer, self.envs[env_id], 3, verts, colors)

        # Green XYZ cross at object center position.
        if c_center is not None:
            cxm = c_center + torch.tensor([-cross_len, 0.0, 0.0], device=self.device)
            cxp = c_center + torch.tensor([ cross_len, 0.0, 0.0], device=self.device)
            cym = c_center + torch.tensor([0.0, -cross_len, 0.0], device=self.device)
            cyp = c_center + torch.tensor([0.0,  cross_len, 0.0], device=self.device)
            czm = c_center + torch.tensor([0.0, 0.0, -cross_len], device=self.device)
            czp = c_center + torch.tensor([0.0, 0.0,  cross_len], device=self.device)

            center_verts = []
            center_colors = []
            center_verts.extend([float(cxm[0]), float(cxm[1]), float(cxm[2]), float(cxp[0]), float(cxp[1]), float(cxp[2])])
            center_colors.extend([0.0, 1.0, 0.0])
            center_verts.extend([float(cym[0]), float(cym[1]), float(cym[2]), float(cyp[0]), float(cyp[1]), float(cyp[2])])
            center_colors.extend([0.0, 1.0, 0.0])
            center_verts.extend([float(czm[0]), float(czm[1]), float(czm[2]), float(czp[0]), float(czp[1]), float(czp[2])])
            center_colors.extend([0.0, 1.0, 0.0])
            self.gym.add_lines(self.viewer, self.envs[env_id], 3, center_verts, center_colors)

        # Red XYZ cross at raw object root pose.
        rxm = c_root + torch.tensor([-cross_len, 0.0, 0.0], device=self.device)
        rxp = c_root + torch.tensor([ cross_len, 0.0, 0.0], device=self.device)
        rym = c_root + torch.tensor([0.0, -cross_len, 0.0], device=self.device)
        ryp = c_root + torch.tensor([0.0,  cross_len, 0.0], device=self.device)
        rzm = c_root + torch.tensor([0.0, 0.0, -cross_len], device=self.device)
        rzp = c_root + torch.tensor([0.0, 0.0,  cross_len], device=self.device)

        root_verts = []
        root_colors = []
        root_verts.extend([float(rxm[0]), float(rxm[1]), float(rxm[2]), float(rxp[0]), float(rxp[1]), float(rxp[2])])
        root_colors.extend([1.0, 0.0, 0.0])
        root_verts.extend([float(rym[0]), float(rym[1]), float(rym[2]), float(ryp[0]), float(ryp[1]), float(ryp[2])])
        root_colors.extend([1.0, 0.0, 0.0])
        root_verts.extend([float(rzm[0]), float(rzm[1]), float(rzm[2]), float(rzp[0]), float(rzp[1]), float(rzp[2])])
        root_colors.extend([1.0, 0.0, 0.0])
        self.gym.add_lines(self.viewer, self.envs[env_id], 3, root_verts, root_colors)

    def _draw_observation_rays_from_eef(self, env_id=0, clear_lines=False):
        if self.viewer is None:
            return
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        if "eef_pos" not in self.states:
            return
        if "object_to_eef" not in self.states:
            return
        if "object_grasp_target_to_eef" not in self.states:
            return
        if "target_to_eef" not in self.states:
            return

        eef_pos = self.states["eef_pos"][env_id]
        object_to_eef = self.states["object_to_eef"][env_id]
        object_grasp_target_to_eef = self.states["object_grasp_target_to_eef"][env_id]
        target_to_eef = self.states["target_to_eef"][env_id]

        if self.teacher_use_eef_frame:
            # Observation vectors are in EEF frame; rotate back to world for debug drawing.
            eef_rot_mat = quaternion_to_matrix_ig(self.states["eef_quat"][env_id:env_id + 1])[0] 
            object_center_pos = eef_pos + torch.matmul(eef_rot_mat, object_to_eef.unsqueeze(-1)).squeeze(-1) 
            object_grasp_target_pos = eef_pos + torch.matmul(eef_rot_mat, object_grasp_target_to_eef.unsqueeze(-1)).squeeze(-1) 
            target_pos = eef_pos + torch.matmul(eef_rot_mat, target_to_eef.unsqueeze(-1)).squeeze(-1) 
        else: 
            # CODEX
            # Legacy world-frame observation vectors.
            object_center_pos = eef_pos + object_to_eef 
            object_grasp_target_pos = eef_pos + object_grasp_target_to_eef 
            target_pos = eef_pos + target_to_eef 

        # # Reward target position reconstructed from observation vector.
        # target_pos = eef_pos + torch.matmul(eef_rot_mat, target_to_eef.unsqueeze(-1)).squeeze(-1)

        verts = []
        colors = []
        # White: EEF -> object center (from object_to_eef)
        verts.extend([
            float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]),
            float(object_center_pos[0]), float(object_center_pos[1]), float(object_center_pos[2]),
        ])
        colors.extend([1.0, 1.0, 1.0])
        # Orange: EEF -> object grasp target (from object_grasp_target_to_eef)
        verts.extend([
            float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]),
            float(object_grasp_target_pos[0]), float(object_grasp_target_pos[1]), float(object_grasp_target_pos[2]),
        ])
        colors.extend([1.0, 0.5, 0.0])

        # Cyan: EEF -> reward target (from target_to_eef)
        verts.extend([
            float(eef_pos[0]), float(eef_pos[1]), float(eef_pos[2]),
            float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
        ])
        colors.extend([0.0, 1.0, 1.0])
        cyan_thickness = 0.004
        cyan_offsets = [
            torch.tensor([ cyan_thickness, 0.0, 0.0], device=self.device),
            torch.tensor([-cyan_thickness, 0.0, 0.0], device=self.device),
            torch.tensor([0.0,  cyan_thickness, 0.0], device=self.device),
            torch.tensor([0.0, -cyan_thickness, 0.0], device=self.device),
        ]
        for off in cyan_offsets:
            p0 = eef_pos + off
            p1 = target_pos + off
            verts.extend([
                float(p0[0]), float(p0[1]), float(p0[2]),
                float(p1[0]), float(p1[1]), float(p1[2]),
            ])
            colors.extend([0.0, 1.0, 1.0])

        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts) // 6, verts, colors)

        # Draw a magenta cross at the reward target endpoint for visibility.
        cross_len = 0.03
        txm = target_pos + torch.tensor([-cross_len, 0.0, 0.0], device=self.device)
        txp = target_pos + torch.tensor([ cross_len, 0.0, 0.0], device=self.device)
        tym = target_pos + torch.tensor([0.0, -cross_len, 0.0], device=self.device)
        typ = target_pos + torch.tensor([0.0,  cross_len, 0.0], device=self.device)
        tzm = target_pos + torch.tensor([0.0, 0.0, -cross_len], device=self.device)
        tzp = target_pos + torch.tensor([0.0, 0.0,  cross_len], device=self.device)

        t_verts = []
        t_colors = []
        t_verts.extend([float(txm[0]), float(txm[1]), float(txm[2]), float(txp[0]), float(txp[1]), float(txp[2])])
        t_colors.extend([1.0, 0.0, 1.0])
        t_verts.extend([float(tym[0]), float(tym[1]), float(tym[2]), float(typ[0]), float(typ[1]), float(typ[2])])
        t_colors.extend([1.0, 0.0, 1.0])
        t_verts.extend([float(tzm[0]), float(tzm[1]), float(tzm[2]), float(tzp[0]), float(tzp[1]), float(tzp[2])])
        t_colors.extend([1.0, 0.0, 1.0])
        self.gym.add_lines(self.viewer, self.envs[env_id], 3, t_verts, t_colors)


    def _update_states(self):
        super()._update_states()
        self._debug_print_joint_limits(env_id=2)
        target_changed_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        if torch.any(self.object_reset_mask):
            reset_object_env_ids = self.object_reset_mask.nonzero(as_tuple=False).squeeze(-1)
            self.post_lift_target_active[reset_object_env_ids] = False
            self.reward_settings["target_pos"][reset_object_env_ids] = self._build_fixed_target_pos(reset_object_env_ids)
            self._set_reward_target_quat_from_side_mask(reset_object_env_ids)
            target_changed_mask[reset_object_env_ids] = True

        self.post_lift_target_active = self.post_lift_target_active | self.states["lift"]
        if torch.any(self.post_lift_target_active):
            lifted_env_ids = self.post_lift_target_active.nonzero(as_tuple=False).squeeze(-1)
            self.reward_settings["target_pos"][lifted_env_ids] = self._compute_post_lift_target_world(lifted_env_ids)
            target_changed_mask[lifted_env_ids] = True
        # Side-grasp switching override:
        # 1) Activate teacher when EEF pose is within switch_tol of switching target pose.
        # 2) Keep teacher active while EEF remains within side_distance + side_distance_noise of object center.
        # 3) Reset-step envs always stay on fabric.
        activated_now = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if self.enable_fabric:
            eef_pos = self._eef_state[:, :3]
            teacher_activation_z = self._get_teacher_activation_z()
            obj_center = self.states["object_center_pos"]
            radial_dist = torch.norm(eef_pos - obj_center, dim=-1)

            teacher_active_prev = ~self.fabric_switch_enable
            if self.use_center_tracking_switch_target:  # CODEX
                activate_now = radial_dist <= self.switch_activate_radius  # CODEX
            elif self.fabric_switch_mode == "simple_anchor":
                enter_r = float(
                    self.fabric_switch_cfg.get(
                        "simple_activate_radius",
                        self.fabric_switch_cfg.get("simple_lateral_offset", self.fabric_switch_cfg["lateral_offset"]),
                    )
                )
                exit_r = float(
                    self.fabric_switch_cfg.get(
                        "simple_activate_radius_exit",
                        self.fabric_switch_cfg.get("object_xy_tol_exit", enter_r + 0.03),
                    )
                )
                activate_radius = torch.full_like(radial_dist, enter_r)
                activate_radius[teacher_active_prev] = exit_r
                activate_now = radial_dist <= activate_radius
            else:  # CODEX
                object_top_z = self.table_surface_height + self.mesh_aabb_extents[:, 2]
                high_z = object_top_z + self.switching_target_z_offset
                teacher_z_max = object_top_z + float(self.fabric_switch_cfg["teacher_activate_z_offset"])
                stage3_target_pos = obj_center.clone()
                stage3_target_pos[:, 1] += float(self.fabric_switch_cfg["lateral_offset"])
                stage3_target_pos[:, 2] = object_top_z + float(self.fabric_switch_cfg["stage3_z_offset"])
                object_xy_dist = torch.norm(eef_pos[:, :2] - obj_center[:, :2], dim=-1)
                stage3_xy_dist = torch.norm(eef_pos[:, :2] - stage3_target_pos[:, :2], dim=-1)
                eef_on_right = eef_pos[:, 1] >= obj_center[:, 1]
                stage3_xy_tol = torch.full_like(stage3_xy_dist, self.pre_teacher_stage3_xy_tol)
                stage3_xy_tol[teacher_active_prev] = self.pre_teacher_stage3_xy_tol_exit
                object_close = object_xy_dist <= float(self.fabric_switch_cfg["lateral_offset"])
                stage3_close = stage3_xy_dist <= stage3_xy_tol
                activate_now = eef_on_right & (teacher_activation_z <= teacher_z_max) & (object_close | stage3_close)
            teacher_active = activate_now & (self.progress_buf > 0)
            # CODEX: latch teacher after lift so we do not switch back to fabric mid-manipulation.
            # `states["lift"]` comes from parent `_update_states()` and is updated every step.
            teacher_active = teacher_active | self.states["lift"]

            # CODEX: after teleport/reset this step, if object is on hand-right, force re-entry to fabric two-stage.
            if (not self.use_center_tracking_switch_target) and self.fabric_switch_mode == "staged":
                reset_event_mask = self.object_reset_pending_mask.clone()
                if torch.any(reset_event_mask):
                    object_on_right = obj_center[:, 1] < eef_pos[:, 1]
                    force_fabric_mask = reset_event_mask & object_on_right
                    teacher_active[force_fabric_mask] = False

            self.fabric_switch_enable[:] = ~teacher_active
            self.fabric_switch_enable[self.progress_buf == 0] = True
            activated_now = (~teacher_active_prev) & teacher_active
            # if self.debug_teacher_state_stats_enabled:
            #     teleport_env_ids = getattr(self, "debug_last_teleport_env_ids", torch.empty((0,), dtype=torch.long, device=self.device))
            #     if teleport_env_ids.numel() > 0:
            #         self._collect_debug_teacher_state_stats("teleport", teleport_env_ids)
            #         self.debug_last_teleport_env_ids = torch.empty((0,), dtype=torch.long, device=self.device)
            #     activated_ids = activated_now.nonzero(as_tuple=False).squeeze(-1)
            #     self._collect_debug_teacher_state_stats("activation", activated_ids)
            #     self._flush_debug_teacher_state_stats(force=False)

        active_ids = activated_now.nonzero(as_tuple=False).squeeze(-1)
        if active_ids.numel() > 0:
            obj_center_active = self.states["object_center_pos"][active_ids]
            num_active = active_ids.numel()
            eef_pos_active = self._eef_state[active_ids, :3]
            desired_dir = obj_center_active - eef_pos_active
            desired_dir_xy = desired_dir.clone()
            desired_dir_xy[:, 2] = 0.0
            desired_dir_xy = desired_dir_xy / torch.norm(desired_dir_xy, dim=-1, keepdim=True).clamp_min(1e-6)

            left_mask = self.side_is_left[active_ids]
            base_quat_right = self.target_quat_right.repeat(num_active, 1)
            base_quat_left = self.target_quat_left.repeat(num_active, 1)
            base_quat = torch.where(left_mask.unsqueeze(-1), base_quat_left, base_quat_right)

            ref_dir_xy = torch.zeros((num_active, 3), device=self.device, dtype=desired_dir_xy.dtype)
            ref_dir_xy[:, 1] = torch.where(
                left_mask,
                -torch.ones_like(left_mask, dtype=desired_dir_xy.dtype),
                torch.ones_like(left_mask, dtype=desired_dir_xy.dtype),
            )
            cross_z = (
                ref_dir_xy[:, 0] * desired_dir_xy[:, 1] - ref_dir_xy[:, 1] * desired_dir_xy[:, 0]
            ).unsqueeze(-1)
            dot_xy = torch.sum(ref_dir_xy[:, :2] * desired_dir_xy[:, :2], dim=-1, keepdim=True)
            yaw = torch.atan2(cross_z, dot_xy)
            half_yaw = 0.5 * yaw
            q_align = torch.zeros((num_active, 4), device=self.device, dtype=desired_dir_xy.dtype)
            q_align[:, 2:3] = torch.sin(half_yaw)
            q_align[:, 3:4] = torch.cos(half_yaw)
            target_quat_active = quat_mul(q_align, base_quat)
            target_quat_active = target_quat_active / torch.norm(target_quat_active, dim=-1, keepdim=True).clamp_min(1e-8)
            self.reward_settings["target_quat"][active_ids] = target_quat_active
            self.reward_settings["target_rot_6d"][active_ids] = matrix_to_rotation_6d(
                quaternion_to_matrix_ig(target_quat_active)
            )
            target_changed_mask[active_ids] = True

        changed_env_ids = target_changed_mask.nonzero(as_tuple=False).squeeze(-1)
        if changed_env_ids.numel() > 0:
            self._refresh_target_state_for_env_ids(changed_env_ids)

        if self._activation_snapshot_bank_enable:
            collection_category = int(self._verified_teacher_bank_collection_category)
            collection_enabled = bool(self._verified_teacher_bank_collection_enabled) and (collection_category >= 0)
            if collection_enabled:
                activated_ids = activated_now.nonzero(as_tuple=False).squeeze(-1)
                if (
                    activated_ids.numel() > 0
                    and collection_category in (self.VERIFIED_BANK_NEAR_RECOVERY, self.VERIFIED_BANK_FAR_RECOVERY)
                ):
                    activated_ids = activated_ids[self.debug_recovery_reset_active[activated_ids]]
                if activated_ids.numel() > 0:
                    self.record_activation_snapshot_bank_entries(activated_ids, collection_category)

        object_grasp_target_pos = self._object_state[:, :3].clone()
        local_offset = torch.zeros([self.num_envs, 3], dtype=torch.float, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[:, 2] * self.object_grasp_target_z_scale
        object_rot_mat = quaternion_to_matrix_ig(self._object_state[:, 3:7])
        rotated_offset = torch.matmul(object_rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)
        object_grasp_target_pos += rotated_offset
        object_grasp_target_to_eef_world = object_grasp_target_pos - self._eef_state[:, :3] 
        if self.teacher_use_eef_frame:
            eef_rot_mat = quaternion_to_matrix_ig(self._eef_state[:, 3:7])
            eef_rot_mat_t = eef_rot_mat.transpose(1, 2)
            object_grasp_target_to_eef = torch.matmul( # CODEX
                eef_rot_mat_t, object_grasp_target_to_eef_world.unsqueeze(-1) 
            ).squeeze(-1) 
        else: 
            object_grasp_target_to_eef = object_grasp_target_to_eef_world 

        eef_table_dist = (self._eef_state[:, 2] - self.table_surface_height).unsqueeze(-1)
        hand_z_axis = quaternion_to_matrix_ig(self._eef_state[:, 3:7])[:, :, 2]
        object_z_axis = object_rot_mat[:, :, 2]

        # @ray not just update but also create new keys here
        # Binary grasp-side indicator for policy observation: 1=left, 0=right.
        grasp_side_binary = self.side_is_left.to(dtype=object_grasp_target_to_eef.dtype).unsqueeze(-1)
        self.states.update({
            "object_grasp_target_pos": object_grasp_target_pos, # @ray reward-only grasp target
            "object_grasp_target_to_eef":  object_grasp_target_to_eef, # @ray for policy observation
            "eef_table_dist": eef_table_dist,
            "object_z_axis_world": object_z_axis,
            "hand_z_axis_world": hand_z_axis,
            "grasp_side_binary": grasp_side_binary,
            "grasp_side": grasp_side_binary,
        })

    def compute_observations(self):
        self._refresh() # @ray checks table collision and updates states

        obs_components = ["q_hand",
                          "grasp_side_binary",
                          "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                          "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                          "object_to_eef", "object_to_eef_rot_6d",
                          "object_grasp_target_to_eef",
                          "eef_table_dist",
                          "object_z_axis_world", "hand_z_axis_world",
                          "target_to_eef", "target_to_eef_rot_6d"]

        states_components = ["q", "qd",
                             "grasp_side_binary",
                             "eef_pos", "eef_rot_6d", "eef_vel",
                             "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                             "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                             "object_to_eef", "object_to_eef_rot_6d",
                             "object_grasp_target_to_eef",
                             "eef_table_dist",
                             "object_z_axis_world", "hand_z_axis_world",
                             "target_to_eef", "target_to_eef_rot_6d"]

        obs_buf = torch.cat([self.states[ob] for ob in obs_components], dim=-1)
        states_buf = torch.cat([self.states[st] for st in states_components], dim=-1)

        # @ray optionally append object bbox info
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
        

        # def _dump_rel_obs(env_id=0, tag=""):
        #     o = self.obs_buf[env_id].detach().cpu()
        #     blocks = {
        #         "grasp_side": o[16:17],
        #         "object_to_eef": o[29:32],
        #         "object_to_eef_rot6": o[32:38],
        #         "object_grasp_target_to_eef": o[38:41],
        #         "target_to_eef": o[41:44],
        #         "target_to_eef_rot6": o[44:50],
        #         "bbox_xyz": o[50:53],  # if use_xy/use_z are enabled
        #     }
        #     print(f"\n[{tag}] env={env_id} step={int(self.sim_steps)}")
        #     for k, v in blocks.items():
        #         print(f"{k:28s} {v.numpy()}")
        # _dump_rel_obs(env_id=0, tag="DISTILL")
        # keys = [
        #     "q", "object_to_eef", "object_to_eef_rot_6d",
        #     "object_grasp_target_to_eef",
        #     "target_to_eef", "target_to_eef_rot_6d",
        # ]
        # for k in keys:
        #     print(k, self.states[k][0].detach().cpu().numpy())


        return self.obs_buf

    
    def compute_reward(self):
        # Distillation only needs reset logic and success/lifting tracking.
        # Skip the dense reward decomposition used in RL.
        if not self.debug_viz:
            self.reset_buf[:] = torch.where(
                (self.progress_buf >= self.max_episode_length - 1),
                torch.ones_like(self.reset_buf),
                self.reset_buf,
            )
        self.rew_buf.zero_()

        d_eef_point_goal = torch.norm(
            self._eef_state[:, :3] - self.reward_settings["target_pos"],
            dim=-1,
        )

        # Use object-table contact for lifting metric instead of pure height.
        self.lifting_5cm_per_step = ~self.table_collision
        self.lifting_flags_instant[self.lifting_5cm_per_step] = 1
        self.success_5cm_per_step = (
            d_eef_point_goal < self.reward_settings["success_tolerance"]
        ) & self.lifting_5cm_per_step
        self.success_flags_instant[self.success_5cm_per_step] = 1
        success_timeout_steps = self.reward_settings["success_timeout_steps"]
        lifting_timeout_steps = self.reward_settings["lifting_timeout_steps"]
        self.success_duration = torch.where(
            self.success_5cm_per_step,
            self.success_duration + 1,
            torch.zeros_like(self.success_duration),
        )
        self.lifting_duration = torch.where(
            self.lifting_5cm_per_step,
            self.lifting_duration + 1,
            torch.zeros_like(self.lifting_duration),
        )
        # Latch success/lifting once achieved anywhere in an episode.
        self.success_long_enough = self.success_long_enough | (self.success_duration >= success_timeout_steps)
        self.lifting_long_enough = self.lifting_long_enough | (self.lifting_duration >= lifting_timeout_steps)
        object_center_pos = self.states["object_center_pos"]
        table_xy_delta = torch.abs(object_center_pos[:, :2] - self.table_pos[:, :2])
        table_xy_limit = 0.5 * self.table_size[:, :2] + self.bad_state_xy_margin
        far_from_table_xy = torch.any(table_xy_delta > table_xy_limit, dim=-1)
        below_table = object_center_pos[:, 2] < self.table_surface_height - 0.1
        low_object_height = object_center_pos[:, 2] <= (
            self.table_surface_height + 0.5 * (self._object_center_init_state[:, 2] - self.table_surface_height) + 0.01
        )
        bad_state = below_table | far_from_table_xy
        self.flat_table_duration = torch.where(
            low_object_height,
            self.flat_table_duration + 1,
            torch.zeros_like(self.flat_table_duration),
        )
        self.flat_table_long_enough = self.flat_table_long_enough | (
            self.flat_table_duration >= self.flat_table_reset_steps
        )
        self.reset_buf[bad_state] = 1
        self.reset_buf[self.flat_table_long_enough] = 1
        done_envs = self.reset_buf > 0

        if torch.any(done_envs):
            done_env_ids = done_envs.nonzero(as_tuple=False).squeeze(-1)
            done_object_ids = self.env_object_ids[done_env_ids]
            episode_increments = torch.bincount(done_object_ids, minlength=self.num_objects)
            success_env_ids = (done_envs & self.success_long_enough).nonzero(as_tuple=False).squeeze(-1)
            success_object_ids = self.env_object_ids[success_env_ids]
            success_increments = torch.bincount(success_object_ids, minlength=self.num_objects)
            lifting_env_ids = (done_envs & self.lifting_long_enough).nonzero(as_tuple=False).squeeze(-1)
            lifting_object_ids = self.env_object_ids[lifting_env_ids]
            lifting_increments = torch.bincount(lifting_object_ids, minlength=self.num_objects)
            self.per_object_episode_counts += episode_increments
            self.per_object_success_counts += success_increments
            self.per_object_lifting_counts += lifting_increments
            self.per_object_episode_counts_interval += episode_increments
            self.per_object_success_counts_interval += success_increments
            self.per_object_lifting_counts_interval += lifting_increments

            # CODEX: maintain rolling per-episode outcomes for windowed success metrics.
            done_success_bits = self.success_long_enough[done_env_ids].to(torch.int8)
            done_lifting_bits = self.lifting_long_enough[done_env_ids].to(torch.int8)
            done_category_ids = self.current_episode_verified_bank_category[done_env_ids]
            valid_category_mask = done_category_ids >= 0
            if torch.any(valid_category_mask):
                done_category_increments = torch.bincount(
                    done_category_ids[valid_category_mask],
                    minlength=len(self.VERIFIED_BANK_CATEGORY_NAMES),
                )
                self.per_verified_bank_episode_counts += done_category_increments
            success_done_category_ids = self.current_episode_verified_bank_category[success_env_ids]
            valid_success_category_mask = success_done_category_ids >= 0
            if torch.any(valid_success_category_mask):
                success_category_increments = torch.bincount(
                    success_done_category_ids[valid_success_category_mask],
                    minlength=len(self.VERIFIED_BANK_CATEGORY_NAMES),
                )
                self.per_verified_bank_success_counts += success_category_increments
            n_done = int(done_env_ids.numel())
            if n_done > 0:
                write_idx = (torch.arange(n_done, device=self.device) + self.ep_hist_ptr) % self.ep_hist_max
                self.ep_hist_success[write_idx] = done_success_bits
                self.ep_hist_lifting[write_idx] = done_lifting_bits
                self.ep_hist_category[write_idx] = done_category_ids
                self.ep_hist_ptr = (self.ep_hist_ptr + n_done) % self.ep_hist_max
                self.ep_hist_count = min(self.ep_hist_count + n_done, self.ep_hist_max)

        # @ray log per-object per-interval success rates locally and a histograom to wandb
        if self.log_per_object_success and self.sim_steps > 0 and (self.sim_steps % self.log_per_object_success_freq == 0):
            # @ray prevent inf from division by zero if some objects are not in any envs
            success_interval_rates = torch.where(
                self.per_object_episode_counts_interval > 0,
                self.per_object_success_counts_interval.float() / self.per_object_episode_counts_interval.float(),
                torch.zeros_like(self.per_object_success_counts_interval, dtype=torch.float32),
            )
            lifting_interval_rates = torch.where(
                self.per_object_episode_counts_interval > 0,
                self.per_object_lifting_counts_interval.float() / self.per_object_episode_counts_interval.float(),
                torch.zeros_like(self.per_object_lifting_counts_interval, dtype=torch.float32),
            )
            if wandb.run is not None:
                hist_step = int(self.distillation_steps) if self.distillation_mode else int(self.sim_steps)  # CODEX
                if wandb.run.step is not None:  # CODEX
                    hist_step = max(hist_step, int(wandb.run.step))  # CODEX
                wandb.log(  # CODEX
                    {"per_object_success_rate_hist": wandb.Histogram(success_interval_rates.detach().cpu().numpy(), num_bins=20)},
                    step=hist_step,
                )
            interval_snapshot = {
                "sim_steps": int(self.sim_steps),
                "log_interval_steps": int(self.log_per_object_success_freq),
                "per_object_success_rates": {
                    str(obj_id): {
                        "episodes": int(self.per_object_episode_counts_interval[obj_id].item()),
                        "successes": int(self.per_object_success_counts_interval[obj_id].item()),
                        "lifting": int(self.per_object_lifting_counts_interval[obj_id].item()),
                        "success_rate": float(success_interval_rates[obj_id].item()),
                        "lifting_rate": float(lifting_interval_rates[obj_id].item()),
                        "episodes_total": int(self.per_object_episode_counts[obj_id].item()),
                        "successes_total": int(self.per_object_success_counts[obj_id].item()),
                        "lifting_successes_total": int(self.per_object_lifting_counts[obj_id].item()),
                    }
                    for obj_id in range(self.num_objects)
                },
            }
            json_path = os.path.join(self.log_per_object_success_dir, f"per_object_success_step{int(self.sim_steps)}.json")
            with open(json_path, "w") as f:
                json.dump(interval_snapshot, f, indent=2)
            if wandb.run is not None:
                artifact = wandb.Artifact(self.log_per_object_success_artifact, type="per_object_success")
                artifact.add_file(json_path)
                wandb.log_artifact(artifact)
            self.per_object_episode_counts_interval.zero_()
            self.per_object_success_counts_interval.zero_()
            self.per_object_lifting_counts_interval.zero_()

        # Episode success rates from reset-event counters (length-invariant).
        total_eps = max(int(self.per_object_episode_counts.sum().item()), 1)
        success_eps = int(self.per_object_success_counts.sum().item())
        lifting_eps = int(self.per_object_lifting_counts.sum().item())
        self.extras["dis/d_eef_point_goal"] = torch.mean(d_eef_point_goal).item()
        self.extras["metrics/success_rate_5cm_per_ep"] = float(success_eps) / float(total_eps)
        self.extras["metrics/success_rate_5cm_per_ep_instant"] = torch.mean(self.success_flags_instant).item()
        self.extras["metrics/success_rate_5cm_per_step"] = torch.mean(self.success_5cm_per_step.float()).item()
        self.extras["metrics/lifting_rate_5cm_per_ep"] = float(lifting_eps) / float(total_eps)
        self.extras["metrics/lifting_rate_5cm_per_ep_instant"] = torch.mean(self.lifting_flags_instant).item()
        self.extras["metrics/lifting_rate_5cm_per_step"] = torch.mean(self.lifting_5cm_per_step.float()).item()
        for category_id, category_name in enumerate(self.VERIFIED_BANK_CATEGORY_NAMES):
            category_total_eps = int(self.per_verified_bank_episode_counts[category_id].item())
            category_success_eps = int(self.per_verified_bank_success_counts[category_id].item())
            category_success_rate = (
                float(category_success_eps) / float(category_total_eps)
            ) if category_total_eps > 0 else 0.0
            self.extras[f"metrics/verified_bank_success_rate_5cm_per_ep_{category_name}"] = category_success_rate
            self.extras[f"metrics/verified_bank_success_rate_5cm_per_ep_{category_name}_count"] = category_total_eps

        # CODEX: windowed episode metrics (empty slots are ignored, never counted as failures).
        window_key_suffixes = ["envs_div2", "envs", "envs_x2", "envs_x4"]  # CODEX
        for win_size, key_suffix in zip(self.ep_hist_window_sizes, window_key_suffixes):  # CODEX
            win_len = min(int(win_size), int(self.ep_hist_count))  # CODEX
            if win_len > 0:  # CODEX
                idx = (torch.arange(win_len, device=self.device) + (self.ep_hist_ptr - win_len)) % self.ep_hist_max  # CODEX
                succ_vals = self.ep_hist_success[idx]  # CODEX
                lift_vals = self.ep_hist_lifting[idx]  # CODEX
                category_vals = self.ep_hist_category[idx]
                valid_s = succ_vals >= 0  # CODEX
                valid_l = lift_vals >= 0  # CODEX
                succ_rate = float(succ_vals[valid_s].float().mean().item()) if bool(torch.any(valid_s)) else 0.0  # CODEX
                lift_rate = float(lift_vals[valid_l].float().mean().item()) if bool(torch.any(valid_l)) else 0.0  # CODEX
                valid_count_s = int(valid_s.sum().item())  # CODEX
                valid_count_l = int(valid_l.sum().item())  # CODEX
            else:  # CODEX
                succ_rate = 0.0  # CODEX
                lift_rate = 0.0  # CODEX
                valid_count_s = 0  # CODEX
                valid_count_l = 0  # CODEX

            self.extras[f"metrics/success_rate_5cm_per_ep_win_{key_suffix}"] = succ_rate  # CODEX
            self.extras[f"metrics/lifting_rate_5cm_per_ep_win_{key_suffix}"] = lift_rate  # CODEX
            self.extras[f"metrics/success_rate_5cm_per_ep_win_{key_suffix}_count"] = valid_count_s  # CODEX
            self.extras[f"metrics/lifting_rate_5cm_per_ep_win_{key_suffix}_count"] = valid_count_l  # CODEX
            if win_len > 0:
                for category_id, category_name in enumerate(self.VERIFIED_BANK_CATEGORY_NAMES):
                    category_mask = valid_s & (category_vals == category_id)
                    category_count = int(category_mask.sum().item())
                    category_rate = (
                        float(succ_vals[category_mask].float().mean().item())
                    ) if category_count > 0 else 0.0
                    self.extras[
                        f"metrics/verified_bank_success_rate_5cm_per_ep_win_{key_suffix}_{category_name}"
                    ] = category_rate
                    self.extras[
                        f"metrics/verified_bank_success_rate_5cm_per_ep_win_{key_suffix}_{category_name}_count"
                    ] = category_count
            else:
                for category_name in self.VERIFIED_BANK_CATEGORY_NAMES:
                    self.extras[
                        f"metrics/verified_bank_success_rate_5cm_per_ep_win_{key_suffix}_{category_name}"
                    ] = 0.0
                    self.extras[
                        f"metrics/verified_bank_success_rate_5cm_per_ep_win_{key_suffix}_{category_name}_count"
                    ] = 0

        if self.debug_viz:
            # Debug-only memory stats.
            mem_allocated_GB = float(torch.cuda.memory_allocated() / 1024**3)
            mem_reserved_GB = float(torch.cuda.memory_reserved() / 1024**3)
            self.extras["mem/allocated_GB"] = mem_allocated_GB
            self.extras["mem/reserved_GB"] = mem_reserved_GB

@torch.jit.script
def compute_franka_leap_reward(states, reward_settings):
    # type: (Dict[str, Tensor], Dict[str, Tensor]) -> Dict[str, Tensor]

    # R1: Hand (palm, fingers) to object distance
    d_palm = torch.norm(states["object_grasp_target_pos"] - states["eef_pos"], dim=-1)
    d_finger1 = torch.norm(states["object_grasp_target_pos"] - states["eef_finger1_pos"], dim=-1)
    d_finger2 = torch.norm(states["object_grasp_target_pos"] - states["eef_finger2_pos"], dim=-1)
    d_finger3 = torch.norm(states["object_grasp_target_pos"] - states["eef_finger3_pos"], dim=-1)
    d_finger4 = torch.norm(states["object_grasp_target_pos"] - states["eef_finger4_pos"], dim=-1)

    # R1: Max dist component to object: max_i∈{palm_pos,fingertips} ||x^i - x^obj||
    d_hand_obj = torch.stack([d_palm, d_finger1, d_finger2, d_finger3, d_finger4], dim=1)
    d_hand_obj = torch.max(d_hand_obj, dim=1)[0]

    # R1: Hand object distance reward
    beta_hand_object = reward_settings["beta_hand_object"]
    r_hand_obj = torch.exp(-beta_hand_object * d_hand_obj)

    # R2: Lifting bonus: r_lift = 1.0 if object is lifted
    target_pos = reward_settings["target_pos"].squeeze(-1)
    object_height = states["object_center_pos"][:, 2] - reward_settings["object_init_height"].squeeze(-1)
    beta_lift = reward_settings["beta_lift"]
    if beta_lift > 0:
        object_vertical_err = torch.abs(states["object_center_pos"][:, 2] - target_pos[2])
        r_lift = torch.exp(-beta_lift * object_vertical_err)
        r_lift = torch.where(states["lift"], r_lift, 0.0)
    else:
        r_lift = torch.where(states["lift"], 1.0, torch.zeros_like(object_height))

    n_env = states["object_quat"].shape[0]
    local_z = torch.zeros((n_env, 3), dtype=states["object_quat"].dtype, device=states["object_quat"].device)
    local_z[:, 2] = 1.0
    object_z_axis_world = quat_apply(states["object_quat"], local_z)

    finger_positions = torch.stack(
        [
            states["eef_finger1_pos"],
            states["eef_finger2_pos"],
            states["eef_finger3_pos"],
            states["eef_finger4_pos"],
        ],
        dim=1,
    )
    gate_offset = reward_settings["grasp_on_object_z_height_offset"]
    gate_target_pos = states["object_grasp_target_pos"] + gate_offset * object_z_axis_world
    target_pos_rep = gate_target_pos.unsqueeze(1)
    fingertip_to_target = finger_positions - target_pos_rep
    finger_signed_height_err = torch.sum(fingertip_to_target * object_z_axis_world.unsqueeze(1), dim=-1)
    # @ray use max signed height to find heighest finger
    # force this finger to be within a certain height range on the object by passing its absolute height to a sigmoid
    # this prevents competing rewards between lift+goal vs. grasp style, since the hand need to place its fingers correctly and thus cannot eargly exploit the lift with a bad style
    # also, the absolute height prevents the policy from overlapping its fingers on the grasp_target to exploit the reward after it learns a good style, thus we retain the good style throughout training
    top_finger_signed_height = torch.max(finger_signed_height_err, dim=1)[0]
    d_finger_height_align = torch.abs(top_finger_signed_height)

    # @ray sigmoid gate on lift and to-goal reward to force the hand to grasp on the right part of the object
    # not adding this would result in the hand eargerly converging on grasping the top part of long cylindrical objects, undesirable
    x = d_finger_height_align
    x_low = reward_settings["grasp_on_object_z_height_tolerance"][0]
    x_high = reward_settings["grasp_on_object_z_height_tolerance"][1]
    s = reward_settings["grasp_on_object_z_height_slope"]
    x_mid = 0.5 * (x_low + x_high)
    lift_height_gate = 1.0 - 1.0 / (1.0 + torch.exp(s * (x_mid - x)))
    gate_floor = reward_settings["grasp_on_object_z_height_gate_floor"]
    lift_height_gate_soft = gate_floor + (1.0 - gate_floor) * lift_height_gate

    gate_enabled = bool(reward_settings["grasp_on_object_z_height_gate_enabled"])
    if gate_enabled:
        lift_height_gate_apply = lift_height_gate_soft
    else:
        lift_height_gate_apply = torch.ones_like(lift_height_gate_soft)
    
    r_lift_before_gate = r_lift.clone()
    r_lift = r_lift * lift_height_gate_apply

    # R3: Object goal distance reward (based on average point matching distance)
    d_eef_point_goal_target = states["point_matching_err_target"]
    beta_object_goal = reward_settings["beta_object_goal"]
    r_obj_goal = torch.exp(-beta_object_goal * d_eef_point_goal_target)
    r_obj_goal = torch.where(states["lift"], r_obj_goal, 0.0)
    # Apply the same grasp-height gate to object-goal reward to reduce top-grasp exploitation.
    r_obj_goal = r_obj_goal * lift_height_gate_apply

    # R4: Hand orientation reward (based on average point matching distance)
    d_eef_point_goal_hand = states["point_matching_err_hand"]
    beta_hand_orientation = reward_settings["beta_hand_orientation"]
    r_hand_orientation = torch.exp(-beta_hand_orientation * d_eef_point_goal_hand)

    # R5: Finger curl
    hand_dof_pos = states["q"][:, 10:26] # hand joint angles
    near_object = (d_hand_obj <= reward_settings["curl_reaching_threshold"])
    finger_pos_diff = torch.sum((hand_dof_pos - reward_settings["grasp_finger_dof_pos"]) ** 2, dim=1)

    beta_curl = reward_settings["beta_curl"]
    r_curl= torch.exp(-beta_curl * finger_pos_diff)
    r_curl = torch.where(near_object, r_curl, 0.0)

    # R6: Colli Penalty
    r_colli = torch.where(states["collision"], 1.0, 0.0)

    # R7: Velocity Regularization/Penalty
    actionreg = states["actionreg"]
    r_actionreg = torch.sum(actionreg**2, dim=-1)

    w_hand_obj = reward_settings["w_hand_obj"]
    w_obj_goal = reward_settings["w_obj_goal"]
    w_hand_orientation = reward_settings["w_hand_orientation"]
    w_lift = reward_settings["w_lift"]
    w_curl = reward_settings["w_curl"]
    w_colli = reward_settings["w_colli"]
    w_actionreg = reward_settings["w_actionreg"]

    use_curl = bool(reward_settings["use_curl"])
    # @ray 
    # use activated rewards only
    # but compute all rewards anyways for logging
    r_total = w_hand_obj*r_hand_obj + w_obj_goal*r_obj_goal + w_hand_orientation*r_hand_orientation + \
              w_curl*r_curl * float(use_curl) + \
              w_lift*r_lift + w_colli*r_colli + w_actionreg*r_actionreg
    rewards = {
        "r_hand_obj": w_hand_obj*r_hand_obj,
        "r_lift": w_lift*r_lift,
        "r_obj_goal": w_obj_goal*r_obj_goal,
        "r_obj_goal_rot": w_hand_orientation*r_hand_orientation,
        "r_curl": w_curl*r_curl,
        "r_colli": w_colli*r_colli,
        "r_actionreg": w_actionreg*r_actionreg,
        "r_total": r_total,
        "d_hand_obj": d_hand_obj,
        "d_lift": object_height,
        "d_eef_point_goal": d_eef_point_goal_target,
        "d_eef_point_goal_rot": d_eef_point_goal_hand,
        "lift_height_gate": lift_height_gate,
        "lift_height_gate_soft": lift_height_gate_soft,
    }

    return rewards


@hydra.main(config_name="config", config_path="../cfg/")
def launch_test(cfg: DictConfig):
    np.random.seed(0)
    torch.manual_seed(0)
    cfg_dict = omegaconf_to_dict(cfg)
    cfg_task = cfg_dict["task"]
    rl_device = cfg_dict["rl_device"]
    sim_device = cfg_dict["sim_device"]
    headless = cfg_dict["headless"]
    graphics_device_id = 0
    virtual_screen_capture = False
    force_render = False
    env = FrankaLEAPMobileDistillationPickSide(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    for i in tqdm(range(1000)):
        env.reset_idx()
        # env.set_robot_joint_state(env.canonical_joint_config)
        # env.set_robot_joint_state(env.canonical_grasp_config)
        env.step_sim_multi(1, False)
        env.compute_observations()

        # test fk, ik # need to set eef to panda_link7, otherwise will have offset
        ee_pos = env.get_ee_from_joint(env.states['q'][:, :7])
        fk_pos_err = torch.any((ee_pos[:, :3] - env.states['eef_pos']) > 1e-4)
        fk_ori_err1 = (ee_pos[:, 3:] - env.states['eef_quat']) > 1e-4
        fk_ori_err2 = (ee_pos[:, 3:] + env.states['eef_quat']) > 1e-4
        fk_ori_err = torch.any(fk_ori_err1 & fk_ori_err2)

        q_config = env.get_joint_from_ee(ee_pos)
        ee_pos_resolve = env.get_ee_from_joint(q_config)
        ik_pos_err = torch.any((ee_pos_resolve[:, :3] - env.states['eef_pos']) > 1e-4)
        ik_quat_err1 = (ee_pos_resolve[:, 3:] - env.states['eef_quat']) > 1e-4
        ik_quat_err2 = (ee_pos_resolve[:, 3:] + env.states['eef_quat']) > 1e-4
        ik_quat_err = torch.any(ik_quat_err1 & ik_quat_err2)

        import ipdb ; ipdb.set_trace()
        env.render()


if __name__ == "__main__":
    launch_test()
