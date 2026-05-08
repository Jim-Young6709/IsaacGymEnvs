"""
Franka + LEAP Hand Pick Env
"""
import time
import json
import os
import csv
from pathlib import Path

import hydra
import h5py
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.pcd_utils import *
from isaacgymenvs.utils.rotation_conversions import *
from isaacgymenvs.tasks import FrankaLEAP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from tqdm import tqdm
import wandb


SNAPSHOT_RESET_MODE_NAMES = ("afar", "near_recovery", "far_recovery")


def _quat_xyzw_to_rot_np(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1.0e-12)
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _sample_box_surface_np(center, dims, quat_xyzw, num_points):
    center = np.asarray(center, dtype=np.float32).reshape(3)
    dims = np.asarray(dims, dtype=np.float32).reshape(3)
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    num_points = int(num_points)
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    half = 0.5 * dims
    face_areas = np.asarray(
        [
            dims[1] * dims[2],
            dims[1] * dims[2],
            dims[0] * dims[2],
            dims[0] * dims[2],
            dims[0] * dims[1],
            dims[0] * dims[1],
        ],
        dtype=np.float64,
    )
    total_area = float(face_areas.sum())
    if total_area <= 0.0 or not np.isfinite(total_area):
        return np.repeat(center[None, :], num_points, axis=0).astype(np.float32)

    # Avoid geometrout.Cuboid.sample_surface here; some NumPy/geometrout builds
    # reject otherwise valid face probabilities due to strict sum-to-one checks.
    probs = face_areas / total_area
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    faces = np.searchsorted(cdf, np.random.random(num_points), side="right")
    uv = np.random.uniform(-1.0, 1.0, size=(num_points, 2)).astype(np.float32)

    points = np.zeros((num_points, 3), dtype=np.float32)
    for face_idx in range(6):
        mask = faces == face_idx
        if not np.any(mask):
            continue
        if face_idx < 2:
            points[mask, 0] = half[0] if face_idx == 0 else -half[0]
            points[mask, 1] = uv[mask, 0] * half[1]
            points[mask, 2] = uv[mask, 1] * half[2]
        elif face_idx < 4:
            points[mask, 0] = uv[mask, 0] * half[0]
            points[mask, 1] = half[1] if face_idx == 2 else -half[1]
            points[mask, 2] = uv[mask, 1] * half[2]
        else:
            points[mask, 0] = uv[mask, 0] * half[0]
            points[mask, 1] = uv[mask, 1] * half[1]
            points[mask, 2] = half[2] if face_idx == 4 else -half[2]

    rot = _quat_xyzw_to_rot_np(quat_xyzw)
    return (points @ rot.T + center[None, :]).astype(np.float32)


class FrankaLEAPPickTableSide(FrankaLEAP):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.object_grasp_target_z_scale = float(cfg["env"]["object_settings"]["object_grasp_target_z_scale"])
        self.lie_flat_prob = float(cfg["env"]["object_settings"].get("lie_flat_prob", 0.0))
        self._reset_object_mirror_y_prob = float(cfg["env"]["object_settings"].get("mirror_y_prob", 0.5))
        self._reset_object_xyz_range_cfg = cfg["env"]["object_settings"]["xyz_range"]
        self._fixed_table_surface_height = float(cfg["env"]["scene"]["table_surface_height"])
        self._fixed_table_size = torch.tensor(
            cfg["env"]["scene"].get(
                "table_size",
                [0.7, 1.2, float(cfg["env"]["table_thickness"])],
            ),
            dtype=torch.float32,
        )
        self._franka_mount_offset_from_mobile_base = torch.tensor([0.178, 0.0, 0.444775], dtype=torch.float32)
        self._snapshot_variation_assignment_json = cfg["env"]["scene"].get("teacher_bank_variation_json", None)
        self.side_mode = str(cfg["env"]["eef_init"]["side_mode"])
        self._reset_flat_side_recovery_prob = float(cfg["env"]["eef_init"].get("flat_side_recovery_prob", 0.25))
        self._reset_eef_rel_object_min_cfg = cfg["env"]["eef_init"].get(
            "rel_object_min",
            [-0.1178, 0.05, 0.0282],
        )
        self._reset_eef_rel_object_max_cfg = cfg["env"]["eef_init"].get(
            "rel_object_max",
            [0.2, 0.5, 0.3490],
        )
        self._reset_flat_eef_rel_object_min_cfg = cfg["env"]["eef_init"].get(
            "flat_rel_object_min",
            [-0.08, -0.08, 0.16],
        )
        self._reset_flat_eef_rel_object_max_cfg = cfg["env"]["eef_init"].get(
            "flat_rel_object_max",
            [0.08, 0.08, 0.32],
        )
        self._reset_yaw_noise_deg = float(cfg["env"]["eef_init"].get("yaw_noise_deg", 45.0))
        self._reset_pitch_roll_noise_deg = float(cfg["env"]["eef_init"].get("pitch_roll_noise_deg", 45.0))
        self._reset_min_palm_object_dist_cfg = float(cfg["env"]["eef_init"].get("min_palm_object_dist", 0.05))
        snapshot_bank_cfg = cfg["env"].get("activation_snapshot_bank", {})
        self._activation_snapshot_bank_enable = bool(snapshot_bank_cfg.get("enable", False))
        self._activation_snapshot_bank_hdf5_path = str(snapshot_bank_cfg.get("hdf5_path", "") or "")
        self._activation_snapshot_bank_sampling_probs_cfg = snapshot_bank_cfg.get("sampling_probs", {})
        snapshot_reset_noise_cfg = snapshot_bank_cfg.get("reset_noise", {})
        self._activation_snapshot_object_xy_noise_cfg = snapshot_reset_noise_cfg.get("object_xy", [0.0, 0.0])
        self._activation_snapshot_arm_joint_noise_deg = float(snapshot_reset_noise_cfg.get("arm_joint_deg", 0.0))
        self._activation_snapshot_hand_joint_noise_deg = float(snapshot_reset_noise_cfg.get("hand_joint_deg", 0.0))
        self._activation_snapshot_hand_yaw_noise_deg = float(snapshot_reset_noise_cfg.get("hand_yaw_deg", 0.0))
        self._activation_snapshot_hand_pitch_noise_deg = float(snapshot_reset_noise_cfg.get("hand_pitch_deg", 0.0))
        self._activation_snapshot_hand_roll_noise_deg = float(snapshot_reset_noise_cfg.get("hand_roll_deg", 0.0))
        self._activation_snapshot_debug_print_limit_cfg = int(snapshot_bank_cfg.get("debug_print_limit", 0))
        self._copy_debug_env_ids = set()
        self._copy_debug_all4 = False
        self._copy_debug_print_limit = 0
        self._copy_debug_print_count = 0
        self._copy_debug_follow_steps = 0
        self._activation_snapshot_bank_loaded = False
        self._post_reset_grace_steps = 0
        self._snapshot_table_pin_steps_after_reset = 0
        self._init_reset_config()
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

    def _post_init_buffers(self):
        super()._post_init_buffers()
        self.side_is_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._desired_table_pos_world = self.cuboid_pos[:, 0, :].clone()
        self._desired_table_quat_world = self.cuboid_quats[:, 0, :].clone()
        self._snapshot_object_center_nominal = self._object_center_init_state.clone()
        self._snapshot_table_pin_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._post_reset_grace_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._copy_debug_follow_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._activation_snapshot_debug_print_limit = (
            self._activation_snapshot_debug_print_limit_cfg
            if self._activation_snapshot_debug_print_limit_cfg > 0
            else (8 if self.debug_viz else 0)
        )
        self._activation_snapshot_debug_print_count = 0
        self._activation_snapshot_debug_pending = None
        if self._activation_snapshot_bank_enable:
            self._load_activation_snapshot_bank_hdf5()
        else:
            self._init_reset_bank()

    def _finalize_reset_bookkeeping(self, env_ids, joint_config, target_quat, left_mask, object_center_world):
        self.reward_settings["object_init_height"][env_ids] = object_center_world[:, 2]
        self._resample_target_pos(env_ids)
        self.side_is_left[env_ids] = left_mask
        self.reward_settings["target_quat"][env_ids] = target_quat
        self.reward_settings["target_rot_6d"][env_ids] = matrix_to_rotation_6d(quaternion_to_matrix_ig(target_quat))
        self.set_robot_joint_state(joint_config, env_ids=env_ids)

        self.success_flags_instant[env_ids] = 0
        self.lifting_flags_instant[env_ids] = 0
        self.success_flags[env_ids] = self.success_long_enough[env_ids].float()
        self.lifting_flags[env_ids] = self.lifting_long_enough[env_ids].float()
        self.success_duration[env_ids] = 0
        self.lifting_duration[env_ids] = 0
        self.success_long_enough[env_ids] = False
        self.lifting_long_enough[env_ids] = False
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._post_reset_grace_buf[env_ids] = int(self._post_reset_grace_steps)
        if self._copy_debug_env_ids and self._copy_debug_follow_steps > 0:
            debug_env_ids = [int(x) for x in env_ids.detach().cpu().tolist() if int(x) in self._copy_debug_env_ids]
            if debug_env_ids:
                debug_env_ids_t = torch.tensor(debug_env_ids, device=self.device, dtype=torch.long)
                self._copy_debug_follow_buf[debug_env_ids_t] = int(self._copy_debug_follow_steps)

        if self.object_wrench_args["enable"]:
            self.object_applied_forces[env_ids] = 0.0
            self.object_applied_torques[env_ids] = 0.0
            self.rigid_body_forces[env_ids] = 0
            self.rigid_body_torques[env_ids] = 0

    def _init_reset_config(self):
        # Empirical activation ranges. Widen these manually when needed.
        object_xyz_range = self._reset_object_xyz_range_cfg
        self._reset_object_xy_min = torch.tensor(object_xyz_range[0][:2], dtype=torch.float32)
        self._reset_object_xy_max = torch.tensor(object_xyz_range[1][:2], dtype=torch.float32)
        self._reset_eef_rel_object_min = torch.tensor(self._reset_eef_rel_object_min_cfg, dtype=torch.float32)
        self._reset_eef_rel_object_max = torch.tensor(self._reset_eef_rel_object_max_cfg, dtype=torch.float32)
        self._reset_flat_eef_rel_object_min = torch.tensor(self._reset_flat_eef_rel_object_min_cfg, dtype=torch.float32)
        self._reset_flat_eef_rel_object_max = torch.tensor(self._reset_flat_eef_rel_object_max_cfg, dtype=torch.float32)
        self._reset_base_rel_eef_min = torch.tensor([-0.5523, 0.0161, -0.7124], dtype=torch.float32)
        self._reset_base_rel_eef_max = torch.tensor([-0.3286, 0.3031, -0.1116], dtype=torch.float32)
        self._reset_flat_base_rel_eef_min = torch.tensor([-0.65, -0.55, -0.80], dtype=torch.float32)
        self._reset_flat_base_rel_eef_max = torch.tensor([-0.20, 0.55, -0.05], dtype=torch.float32)
        self._reset_flat_topdown_box_size_min = torch.tensor([0.25, 0.25, 0.15], dtype=torch.float32)
        self._reset_flat_topdown_box_size_max = torch.tensor([0.5, 0.5, 0.25], dtype=torch.float32)
        self._reset_flat_topdown_dis_open_range = torch.tensor([0.3, 0.35], dtype=torch.float32)
        self._reset_flat_topdown_dis_side_range = -0.1
        self._reset_flat_topdown_obj_wall_tol = 0.03
        self._reset_yaw_noise_rad = float(np.deg2rad(self._reset_yaw_noise_deg))
        self._reset_pitch_roll_noise_rad = float(np.deg2rad(self._reset_pitch_roll_noise_deg))
        self._reset_min_palm_object_dist = self._reset_min_palm_object_dist_cfg
        self._reset_side_clearance = 0.025
        self._reset_hand_joint_noise_deg = 20.0
        self._activation_snapshot_object_xy_noise = torch.tensor(
            self._activation_snapshot_object_xy_noise_cfg,
            dtype=torch.float32,
        )
        self._activation_snapshot_arm_joint_noise_rad = float(np.deg2rad(self._activation_snapshot_arm_joint_noise_deg))
        self._activation_snapshot_hand_joint_noise_rad = float(np.deg2rad(self._activation_snapshot_hand_joint_noise_deg))
        self._activation_snapshot_hand_yaw_noise_rad = float(np.deg2rad(self._activation_snapshot_hand_yaw_noise_deg))
        self._activation_snapshot_hand_pitch_noise_rad = float(np.deg2rad(self._activation_snapshot_hand_pitch_noise_deg))
        self._activation_snapshot_hand_roll_noise_rad = float(np.deg2rad(self._activation_snapshot_hand_roll_noise_deg))
        self._reset_hand_joint_noise_rad = torch.full(
            (16,),
            float(np.deg2rad(self._reset_hand_joint_noise_deg)),
            dtype=torch.float32,
        )
        # `eef_rel_object` stats were logged from the palm_center rigid body, while cuRobo IK
        # solves for panda_link7. In the URDF, palm_center is attached to panda_link7 by a fixed
        # joint with xyz="0 0 0.115" and no rotation, so convert palm targets to link7 targets by
        # subtracting this local z offset rotated into world.
        self._palm_center_from_link7_local = torch.tensor([0.0, 0.0, 0.115], dtype=torch.float32)
        self._flat_object_quat_base = torch.tensor([0.0, 0.70710678, 0.0, 0.70710678], dtype=torch.float32)
        self._reset_bank_size = 16384
        self._reset_bank_max_ik_goals = 4096

    def _init_reset_bank(self):
        bank_size = int(self._reset_bank_size)
        num_envs = int(self.num_envs)
        num_objects = int(self.num_objects)
        dtype = self._q.dtype

        self._reset_bank_joint_config_cpu = torch.empty((num_objects, bank_size, self.num_dofs), dtype=dtype, device="cpu")
        self._reset_bank_target_quat_cpu = torch.empty((num_objects, bank_size, 4), dtype=dtype, device="cpu")
        self._reset_bank_object_center_world_cpu = torch.empty((num_objects, bank_size, 3), dtype=dtype, device="cpu")
        self._reset_bank_object_quat_world_cpu = torch.empty((num_objects, bank_size, 4), dtype=dtype, device="cpu")
        self._reset_bank_side_is_left_cpu = torch.empty((num_objects, bank_size), dtype=torch.bool, device="cpu")

        rep_env_ids_cpu = torch.full((num_objects,), -1, dtype=torch.long)
        env_object_ids_cpu = self.env_object_ids.detach().to(device="cpu", dtype=torch.long)
        for env_id in range(num_envs):
            object_id = int(env_object_ids_cpu[env_id].item())
            if rep_env_ids_cpu[object_id] < 0:
                rep_env_ids_cpu[object_id] = env_id
        if bool(torch.any(rep_env_ids_cpu < 0)):
            missing = (rep_env_ids_cpu < 0).nonzero(as_tuple=False).squeeze(-1).tolist()
            raise RuntimeError(f"Missing representative envs for object ids: {missing}")
        self._reset_bank_rep_env_ids_cpu = rep_env_ids_cpu.clone()

        rep_env_ids = rep_env_ids_cpu.to(device=self.device, dtype=torch.long)
        goals_per_round = max(1, int(self._reset_bank_max_ik_goals))
        slots_per_round = max(1, goals_per_round // max(num_objects, 1))
        t0 = time.time()
        progress = tqdm(total=bank_size, desc="Building Reset Bank")
        for start in range(0, bank_size, slots_per_round):
            cur_slots = min(slots_per_round, bank_size - start)
            batched_env_ids = rep_env_ids.repeat(cur_slots)
            (
                joint_config,
                target_quat,
                left_mask,
                object_center_world,
                object_quat_world,
            ) = self._sample_reset_joint_and_target_quat(
                batched_env_ids,
                use_reset_solver=True,
            )

            end = start + cur_slots
            joint_config = joint_config.reshape(cur_slots, num_objects, self.num_dofs).transpose(0, 1).contiguous().cpu()
            target_quat = target_quat.reshape(cur_slots, num_objects, 4).transpose(0, 1).contiguous().cpu()
            left_mask = left_mask.reshape(cur_slots, num_objects).transpose(0, 1).contiguous().cpu()
            object_center_world = object_center_world.reshape(cur_slots, num_objects, 3).transpose(0, 1).contiguous().cpu()
            object_quat_world = object_quat_world.reshape(cur_slots, num_objects, 4).transpose(0, 1).contiguous().cpu()

            self._reset_bank_joint_config_cpu[:, start:end].copy_(joint_config)
            self._reset_bank_target_quat_cpu[:, start:end].copy_(target_quat)
            self._reset_bank_side_is_left_cpu[:, start:end].copy_(left_mask)
            self._reset_bank_object_center_world_cpu[:, start:end].copy_(object_center_world)
            self._reset_bank_object_quat_world_cpu[:, start:end].copy_(object_quat_world)
            progress.update(cur_slots)
        progress.close()

        bank_bytes = (
            self._reset_bank_joint_config_cpu.element_size() * self._reset_bank_joint_config_cpu.numel() +
            self._reset_bank_target_quat_cpu.element_size() * self._reset_bank_target_quat_cpu.numel() +
            self._reset_bank_object_center_world_cpu.element_size() * self._reset_bank_object_center_world_cpu.numel() +
            self._reset_bank_object_quat_world_cpu.element_size() * self._reset_bank_object_quat_world_cpu.numel() +
            self._reset_bank_side_is_left_cpu.element_size() * self._reset_bank_side_is_left_cpu.numel()
        )
        elapsed = time.time() - t0
        print(
            f"Built side reset bank: num_envs={num_envs}, num_objects={num_objects}, bank_size={bank_size}, slots_per_round={slots_per_round}, "
            f"cpu_mem={bank_bytes / (1024 ** 3):.3f} GB, elapsed={elapsed:.1f}s"
        )

    def _load_snapshot_variation_assignment_json(self, assignment_json_path, num_objects):
        assignment_path = Path(assignment_json_path).expanduser()
        if not assignment_path.is_absolute():
            assignment_path = Path(os.getcwd()) / assignment_path
        if not assignment_path.exists():
            raise FileNotFoundError(f"snapshot variation assignment JSON not found: {assignment_path}")

        with assignment_path.open("r") as f:
            payload = json.load(f)

        def _slice_global_entries(num_entries):
            local_num_envs = int(self.num_envs)
            if num_entries == local_num_envs:
                return 0, local_num_envs
            world_size = max(int(getattr(self, "world_size", 1)), 1)
            global_rank = int(getattr(self, "global_rank", 0))
            expected_global_entries = local_num_envs * world_size
            if num_entries == expected_global_entries:
                start = global_rank * local_num_envs
                return start, start + local_num_envs
            raise ValueError(
                "snapshot variation assignment JSON has wrong length: "
                f"expected local {local_num_envs} or global {expected_global_entries}, got {num_entries}"
            )

        if isinstance(payload, dict) and "entries" in payload:
            entries = payload["entries"]
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ValueError(f"Unsupported snapshot variation assignment JSON format: {assignment_path}")

        if len(entries) == 0 or not isinstance(entries[0], dict):
            raise ValueError(f"snapshot variation assignment JSON must contain object entries: {assignment_path}")

        start_idx, end_idx = _slice_global_entries(len(entries))
        entries_local = entries[start_idx:end_idx]

        if "variant_id" in entries_local[0]:
            variation_ids = torch.tensor(
                [int(entry["variant_id"]) for entry in entries_local],
                dtype=torch.long,
                device=self.device,
            )
        else:
            variation_ids = torch.arange(int(self.num_envs), dtype=torch.long, device=self.device)

        if "object_id" not in entries_local[0] and "object_name" not in entries_local[0]:
            raise ValueError(
                "snapshot variation assignment JSON entries must contain object_id or object_name: "
                f"{assignment_path}"
            )

        object_name_to_index = {
            str(name): int(idx)
            for idx, name in enumerate(getattr(self, "object_id_to_name", []))
        }
        variant_object_name_map, variant_object_name_map_path = self._discover_snapshot_variant_object_name_map(
            assignment_path
        )
        resolved_object_ids = []
        stable_resolved = 0
        legacy_fallback = 0
        missing_object_names = set()
        for entry in entries_local:
            resolved_idx = None
            object_name = None
            if "object_name" in entry and entry["object_name"] not in (None, ""):
                object_name = str(entry["object_name"])
            elif variant_object_name_map and "variant_id" in entry:
                object_name = variant_object_name_map.get(int(entry["variant_id"]))

            if object_name is not None:
                resolved_idx = object_name_to_index.get(object_name)
                if resolved_idx is None:
                    missing_object_names.add(object_name)

            if resolved_idx is None:
                if "object_id" not in entry:
                    raise ValueError(
                        "snapshot variation assignment JSON cannot resolve object without object_id fallback: "
                        f"path={assignment_path}"
                    )
                resolved_idx = int(entry["object_id"])
                legacy_fallback += 1
            else:
                stable_resolved += 1

            resolved_object_ids.append(resolved_idx)

        object_ids = torch.tensor(
            resolved_object_ids,
            dtype=torch.long,
            device=self.device,
        )
        if bool(torch.any((object_ids < 0) | (object_ids >= int(num_objects)))):
            raise ValueError(
                "snapshot variation assignment JSON resolved object_id outside available range: "
                f"path={assignment_path} valid_ids=[0, {int(num_objects) - 1}]"
            )
        if stable_resolved > 0:
            source_text = str(variant_object_name_map_path) if variant_object_name_map_path is not None else "inline object_name"
            print(
                "[SnapshotAssignment] "
                f"resolved_by_name={stable_resolved} legacy_fallback={legacy_fallback} "
                f"source={source_text}",
                flush=True,
            )
        if missing_object_names:
            missing_sorted = sorted(missing_object_names)
            preview = missing_sorted[:8]
            suffix = " ..." if len(missing_sorted) > len(preview) else ""
            print(
                "[SnapshotAssignment] WARNING: "
                "stable object names not present in current mesh catalog; "
                f"falling back to legacy object_id for {preview}{suffix}",
                flush=True,
            )

        if "table_surface_height" in entries_local[0]:
            table_surface_height = torch.tensor(
                [float(entry["table_surface_height"]) for entry in entries_local],
                dtype=torch.float32,
                device=self.device,
            )
        elif "z_shift" in entries_local[0]:
            table_surface_height = torch.tensor(
                [float(entry["z_shift"]) for entry in entries_local],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            table_surface_height = torch.full(
                (int(self.num_envs),),
                float(self._fixed_table_surface_height),
                dtype=torch.float32,
                device=self.device,
            )
        if "table_size" in entries_local[0]:
            table_size = torch.tensor(
                [entry["table_size"] for entry in entries_local],
                dtype=torch.float32,
                device=self.device,
            )
            if table_size.ndim != 2 or int(table_size.shape[-1]) != 3:
                raise ValueError(
                    f"snapshot variation assignment JSON table_size must be shape [N,3]: {assignment_path}"
                )
        else:
            table_size = self._fixed_table_size.to(device=self.device).unsqueeze(0).repeat(int(self.num_envs), 1)
        return table_surface_height, object_ids, variation_ids, table_size

    def _discover_snapshot_variant_object_name_map(self, assignment_path):
        object_name_to_index = getattr(self, "object_id_to_name", None)
        if not object_name_to_index:
            return {}, None

        root_candidates = [assignment_path.parent]
        if assignment_path.parent.parent != assignment_path.parent:
            root_candidates.append(assignment_path.parent.parent)

        best_map = {}
        best_path = None
        for root in root_candidates:
            if not root.exists():
                continue
            for csv_path in sorted(root.rglob("*.csv")):
                try:
                    with csv_path.open("r", newline="") as f:
                        reader = csv.DictReader(f)
                        if reader.fieldnames is None:
                            continue
                        header = set(reader.fieldnames)
                        if "variant_id" not in header or "object_name" not in header:
                            continue
                        candidate_map = {}
                        inconsistent = False
                        for row in reader:
                            variant_raw = row.get("variant_id", "")
                            object_name_raw = row.get("object_name", "")
                            if variant_raw in (None, "") or object_name_raw in (None, ""):
                                continue
                            variant_id = int(variant_raw)
                            object_name = str(object_name_raw)
                            prev_name = candidate_map.get(variant_id)
                            if prev_name is None:
                                candidate_map[variant_id] = object_name
                            elif prev_name != object_name:
                                inconsistent = True
                                break
                        if inconsistent or len(candidate_map) == 0:
                            continue
                        if len(candidate_map) > len(best_map):
                            best_map = candidate_map
                            best_path = csv_path
                except Exception:
                    continue
            if best_map:
                break
        return best_map, best_path

    def _get_activation_snapshot_bank_path(self):
        if not self._activation_snapshot_bank_hdf5_path:
            raise ValueError("activation_snapshot_bank.enable=True but hdf5_path is empty")
        bank_path = Path(self._activation_snapshot_bank_hdf5_path).expanduser()
        if not bank_path.is_absolute():
            bank_path = Path(os.getcwd()) / bank_path
        return bank_path

    def _load_activation_snapshot_bank_hdf5(self):
        bank_path = self._get_activation_snapshot_bank_path()
        if not bank_path.exists():
            raise FileNotFoundError(f"activation snapshot bank not found: {bank_path}")

        with h5py.File(bank_path, "r") as f:
            counts_np = f["counts"][...]
            if counts_np.ndim != 2:
                raise ValueError(f"activation snapshot bank counts must be rank-2: {bank_path}")
            if counts_np.shape[0] != len(SNAPSHOT_RESET_MODE_NAMES):
                raise ValueError(
                    "activation snapshot bank counts has wrong mode dimension: "
                    f"path={bank_path} expected={len(SNAPSHOT_RESET_MODE_NAMES)} got={counts_np.shape[0]}"
                )
            self._activation_snapshot_bank_counts_cpu = torch.from_numpy(counts_np.astype(np.int32)).to(dtype=torch.int32)
            self._activation_snapshot_bank_num_variants = int(self._activation_snapshot_bank_counts_cpu.shape[1])

            if "joint_config" in f:
                joint_np = f["joint_config"][...]
                last_dim = int(joint_np.shape[-1])
                mobile_base_pose_cpu = None
                if last_dim == int(self.num_dofs):
                    joint_config_cpu = torch.from_numpy(joint_np).to(dtype=self._q.dtype)
                elif last_dim >= int(self.num_dofs) + 3:
                    mobile_base_pose_cpu = torch.from_numpy(joint_np[..., :3]).to(dtype=self._q.dtype)
                    joint_config_cpu = torch.from_numpy(
                        joint_np[..., 3: 3 + int(self.num_dofs)]
                    ).to(dtype=self._q.dtype)
                else:
                    raise ValueError(
                        "activation snapshot bank joint_config has incompatible last dim: "
                        f"path={bank_path} got={last_dim} expected={self.num_dofs}, "
                        f"{self.num_dofs + 3}, or a mobile layout with arm+hand slice [3:26]"
                    )
            elif ("q_arm" in f) and ("q_hand" in f):
                q_arm_cpu = torch.from_numpy(f["q_arm"][...]).to(dtype=self._q.dtype)
                q_hand_cpu = torch.from_numpy(f["q_hand"][...]).to(dtype=self._q.dtype)
                if q_arm_cpu.shape[:-1] != q_hand_cpu.shape[:-1]:
                    raise ValueError(f"activation snapshot bank q_arm/q_hand shape mismatch: {bank_path}")
                if int(q_arm_cpu.shape[-1]) != 7 or int(q_hand_cpu.shape[-1]) != 16:
                    raise ValueError(f"activation snapshot bank q_arm/q_hand dims must be 7 and 16: {bank_path}")
                joint_config_cpu = torch.zeros(
                    (*q_arm_cpu.shape[:-1], self.num_dofs),
                    dtype=self._q.dtype,
                    device="cpu",
                )
                joint_config_cpu[..., :7] = q_arm_cpu
                joint_config_cpu[..., 7:23] = q_hand_cpu
                mobile_base_pose_cpu = None
            else:
                raise ValueError(
                    f"activation snapshot bank must contain joint_config or (q_arm, q_hand): {bank_path}"
                )

            self._activation_snapshot_bank_joint_config_cpu = joint_config_cpu.contiguous()
            self._activation_snapshot_bank_mobile_base_pose_cpu = (
                mobile_base_pose_cpu.contiguous() if mobile_base_pose_cpu is not None else None
            )
            self._activation_snapshot_bank_object_center_world_cpu = torch.from_numpy(
                f["object_center_world"][...]
            ).to(dtype=self._q.dtype)
            self._activation_snapshot_bank_object_quat_world_cpu = torch.from_numpy(
                f["object_quat_world"][...]
            ).to(dtype=self._q.dtype)
            self._activation_snapshot_bank_side_is_left_cpu = torch.from_numpy(
                f["side_is_left"][...].astype(np.bool_)
            ).to(dtype=torch.bool)

            if "table_pos_world" in f:
                self._activation_snapshot_bank_table_pos_world_cpu = torch.from_numpy(
                    f["table_pos_world"][...]
                ).to(dtype=self._q.dtype)
            else:
                if "table_surface_height" not in f:
                    raise ValueError(
                        f"activation snapshot bank must contain table_pos_world or table_surface_height: {bank_path}"
                    )
                table_surface_height_cpu = torch.from_numpy(f["table_surface_height"][...]).to(dtype=self._q.dtype)
                table_pos_cpu = torch.zeros(
                    (*table_surface_height_cpu.shape, 3),
                    dtype=self._q.dtype,
                    device="cpu",
                )
                table_pos_cpu[..., 0] = 0.5
                table_pos_cpu[..., 1] = 0.0
                table_pos_cpu[..., 2] = table_surface_height_cpu - 0.5 * float(self.table_thickness)
                self._activation_snapshot_bank_table_pos_world_cpu = table_pos_cpu

            if "table_quat_world" in f:
                self._activation_snapshot_bank_table_quat_world_cpu = torch.from_numpy(
                    f["table_quat_world"][...]
                ).to(dtype=self._q.dtype)
            else:
                table_quat_cpu = torch.zeros(
                    (*self._activation_snapshot_bank_table_pos_world_cpu.shape[:-1], 4),
                    dtype=self._q.dtype,
                    device="cpu",
                )
                table_quat_cpu[..., 3] = 1.0
                self._activation_snapshot_bank_table_quat_world_cpu = table_quat_cpu

            if "target_quat_world" in f:
                self._activation_snapshot_bank_target_quat_world_cpu = torch.from_numpy(
                    f["target_quat_world"][...]
                ).to(dtype=self._q.dtype)
            else:
                self._activation_snapshot_bank_target_quat_world_cpu = None

            if "table_size" in f:
                file_table_size_cpu = torch.from_numpy(f["table_size"][...]).to(dtype=self._q.dtype)
                if file_table_size_cpu.ndim != 2 or int(file_table_size_cpu.shape[-1]) != 3:
                    raise ValueError(f"activation snapshot bank table_size must be shape [N,3]: {bank_path}")
                if "variation_id" in f:
                    file_variation_ids_cpu = torch.from_numpy(f["variation_id"][...]).to(dtype=torch.long)
                    if file_variation_ids_cpu.ndim != 1 or int(file_variation_ids_cpu.numel()) != int(file_table_size_cpu.shape[0]):
                        raise ValueError(
                            "activation snapshot bank variation_id/table_size shape mismatch: "
                            f"path={bank_path}"
                        )
                    variant_table_size_cpu = torch.full(
                        (self._activation_snapshot_bank_num_variants, 3),
                        float("nan"),
                        dtype=self._q.dtype,
                        device="cpu",
                    )
                    for i in range(int(file_variation_ids_cpu.numel())):
                        variation_id = int(file_variation_ids_cpu[i].item())
                        if variation_id < 0 or variation_id >= self._activation_snapshot_bank_num_variants:
                            raise ValueError(
                                f"activation snapshot bank variation_id out of range for table_size: path={bank_path}"
                            )
                        table_size_i = file_table_size_cpu[i]
                        existing_i = variant_table_size_cpu[variation_id]
                        if torch.isnan(existing_i).any():
                            variant_table_size_cpu[variation_id] = table_size_i
                        elif not torch.allclose(existing_i, table_size_i, atol=1.0e-6, rtol=0.0):
                            raise ValueError(
                                "activation snapshot bank has inconsistent table_size for one variation: "
                                f"path={bank_path} variation_id={variation_id}"
                            )
                    env_expected_table_size_cpu = variant_table_size_cpu[
                        self.env_variation_ids.detach().to(device="cpu", dtype=torch.long)
                    ]
                    if torch.isnan(env_expected_table_size_cpu).any():
                        raise ValueError(
                            f"activation snapshot bank missing table_size for some env variation ids: {bank_path}"
                        )
                    if not torch.allclose(
                        env_expected_table_size_cpu,
                        self.env_table_size_init.detach().to(device="cpu", dtype=self._q.dtype),
                        atol=1.0e-6,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            f"activation snapshot bank table_size mismatch against current env variation layout: {bank_path}"
                        )
                elif int(file_table_size_cpu.shape[0]) == int(self._activation_snapshot_bank_num_variants):
                    env_expected_table_size_cpu = file_table_size_cpu[
                        self.env_variation_ids.detach().to(device="cpu", dtype=torch.long)
                    ]
                    if not torch.allclose(
                        env_expected_table_size_cpu,
                        self.env_table_size_init.detach().to(device="cpu", dtype=self._q.dtype),
                        atol=1.0e-6,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            f"activation snapshot bank table_size mismatch against current env variation layout: {bank_path}"
                        )
                elif int(file_table_size_cpu.shape[0]) == int(self.num_envs):
                    if not torch.allclose(
                        file_table_size_cpu,
                        self.env_table_size_init.detach().to(device="cpu", dtype=self._q.dtype),
                        atol=1.0e-6,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            f"activation snapshot bank per-env table_size mismatch: {bank_path}"
                        )
                else:
                    raise ValueError(
                        "activation snapshot bank table_size has unsupported leading dimension: "
                        f"path={bank_path} shape={tuple(file_table_size_cpu.shape)}"
                    )

        max_env_variation_id = int(self.env_variation_ids.max().item()) if self.env_variation_ids.numel() > 0 else -1
        if max_env_variation_id >= self._activation_snapshot_bank_num_variants:
            raise ValueError(
                "activation snapshot bank does not cover current env variation ids: "
                f"path={bank_path} max_env_variation_id={max_env_variation_id} "
                f"num_variants={self._activation_snapshot_bank_num_variants}"
            )

        sampling_probs = torch.tensor(
            [
                float(self._activation_snapshot_bank_sampling_probs_cfg.get(mode_name, 1.0 / len(SNAPSHOT_RESET_MODE_NAMES)))
                for mode_name in SNAPSHOT_RESET_MODE_NAMES
            ],
            dtype=torch.float32,
            device="cpu",
        )
        if bool(torch.any(sampling_probs < 0.0)):
            raise ValueError(f"activation snapshot bank sampling_probs must be non-negative: {bank_path}")
        if float(sampling_probs.sum().item()) <= 0.0:
            raise ValueError(f"activation snapshot bank sampling_probs must sum to > 0: {bank_path}")
        self._activation_snapshot_bank_mode_probs_cpu = sampling_probs / sampling_probs.sum()
        self._activation_snapshot_bank_loaded = True

    def _sample_activation_snapshot_bank_entries(self, env_ids):
        env_variation_ids_cpu = self.env_variation_ids[env_ids].detach().to(device="cpu", dtype=torch.long)
        num_envs = int(env_ids.numel())
        chosen_mode_cpu = torch.zeros((num_envs,), dtype=torch.long, device="cpu")
        chosen_slot_cpu = torch.zeros((num_envs,), dtype=torch.long, device="cpu")

        for i in range(num_envs):
            variation_id = int(env_variation_ids_cpu[i].item())
            counts_for_variation = self._activation_snapshot_bank_counts_cpu[:, variation_id].to(dtype=torch.float32)
            probs = self._activation_snapshot_bank_mode_probs_cpu.clone()
            probs[counts_for_variation <= 0] = 0.0
            if float(probs.sum().item()) <= 0.0:
                raise RuntimeError(
                    "No activation snapshot entries available for variation: "
                    f"variation_id={variation_id}"
                )
            probs = probs / probs.sum()
            mode_id = int(torch.multinomial(probs, num_samples=1).item())
            count = int(self._activation_snapshot_bank_counts_cpu[mode_id, variation_id].item())
            chosen_mode_cpu[i] = mode_id
            chosen_slot_cpu[i] = int(torch.randint(low=0, high=count, size=(1,), device="cpu", dtype=torch.long).item())

        joint_config = self._activation_snapshot_bank_joint_config_cpu[
            chosen_mode_cpu, env_variation_ids_cpu, chosen_slot_cpu
        ].to(device=self.device, dtype=self._q.dtype)
        object_center_world = self._activation_snapshot_bank_object_center_world_cpu[
            chosen_mode_cpu, env_variation_ids_cpu, chosen_slot_cpu
        ].to(device=self.device, dtype=self._q.dtype)
        object_quat_world = self._activation_snapshot_bank_object_quat_world_cpu[
            chosen_mode_cpu, env_variation_ids_cpu, chosen_slot_cpu
        ].to(device=self.device, dtype=self._q.dtype)
        side_is_left = self._activation_snapshot_bank_side_is_left_cpu[
            chosen_mode_cpu, env_variation_ids_cpu, chosen_slot_cpu
        ].to(device=self.device, dtype=torch.bool)
        table_pos_world = self._activation_snapshot_bank_table_pos_world_cpu[
            chosen_mode_cpu, env_variation_ids_cpu, chosen_slot_cpu
        ].to(device=self.device, dtype=self._q.dtype)
        table_quat_world = self._activation_snapshot_bank_table_quat_world_cpu[
            chosen_mode_cpu, env_variation_ids_cpu, chosen_slot_cpu
        ].to(device=self.device, dtype=self._q.dtype)
        raw_table_pos_world = table_pos_world.clone()
        raw_table_quat_world = table_quat_world.clone()
        raw_object_center_world = object_center_world.clone()
        raw_object_quat_world = object_quat_world.clone()

        if self._activation_snapshot_bank_target_quat_world_cpu is not None:
            target_quat_world = self._activation_snapshot_bank_target_quat_world_cpu[
                chosen_mode_cpu, env_variation_ids_cpu, chosen_slot_cpu
            ].to(device=self.device, dtype=self._q.dtype)
        else:
            target_quat_world = None
        raw_target_quat_world = target_quat_world.clone() if target_quat_world is not None else None

        mobile_base_pose = None
        franka_base_pose7 = None

        if self._activation_snapshot_bank_mobile_base_pose_cpu is not None:
            mobile_base_pose = self._activation_snapshot_bank_mobile_base_pose_cpu[
                chosen_mode_cpu, env_variation_ids_cpu, chosen_slot_cpu
            ].to(device=self.device, dtype=self._q.dtype)
            franka_base_pose7 = self._get_franka_base_pose7_from_mobile_base_pose(
                mobile_base_pose,
                dtype=self._q.dtype,
            )
            franka_base_pos_world = franka_base_pose7[:, :3]
            franka_base_quat_world = franka_base_pose7[:, 3:7]
            franka_base_quat_inv = self._quat_conjugate_tensor(franka_base_quat_world)

            table_pos_world = quat_apply(franka_base_quat_inv, table_pos_world - franka_base_pos_world)
            object_center_world = quat_apply(franka_base_quat_inv, object_center_world - franka_base_pos_world)
            table_quat_world = self._normalize_quat_tensor(quat_mul(franka_base_quat_inv, table_quat_world))
            object_quat_world = self._normalize_quat_tensor(quat_mul(franka_base_quat_inv, object_quat_world))
            if target_quat_world is not None:
                target_quat_world = self._normalize_quat_tensor(
                    quat_mul(franka_base_quat_inv, target_quat_world)
                )

        debug_payload = {
            "env_ids": env_ids.clone(),
            "variation_id": self.env_variation_ids[env_ids].detach().clone(),
            "mode_id": chosen_mode_cpu.to(device=self.device, dtype=torch.long),
            "slot_id": chosen_slot_cpu.to(device=self.device, dtype=torch.long),
            "mobile_base_pose": mobile_base_pose.clone() if mobile_base_pose is not None else None,
            "franka_base_pose7": franka_base_pose7.clone() if franka_base_pose7 is not None else None,
            "raw_table_pos_world": raw_table_pos_world.clone(),
            "raw_table_quat_world": raw_table_quat_world.clone(),
            "raw_object_center_world": raw_object_center_world.clone(),
            "raw_object_quat_world": raw_object_quat_world.clone(),
            "raw_target_quat_world": raw_target_quat_world.clone() if raw_target_quat_world is not None else None,
            "table_pos_replay_world": table_pos_world.clone(),
            "table_quat_replay_world": table_quat_world.clone(),
            "object_center_replay_world": object_center_world.clone(),
            "object_quat_replay_world": object_quat_world.clone(),
            "target_quat_replay_world": target_quat_world.clone() if target_quat_world is not None else None,
            "joint_config": joint_config.clone(),
        }

        return (
            joint_config,
            side_is_left,
            object_center_world,
            object_quat_world,
            table_pos_world,
            table_quat_world,
            target_quat_world,
            debug_payload,
        )

    def _apply_activation_snapshot_reset_noise(
        self,
        joint_config,
        object_center_world,
        object_quat_world,
        table_quat_world,
    ):
        num_envs = int(joint_config.shape[0])
        if num_envs == 0:
            return joint_config, object_center_world, object_quat_world

        dtype = self._q.dtype
        device = self.device
        joint_config = joint_config.clone()
        object_center_world = object_center_world.clone()
        object_quat_world = object_quat_world.clone()

        object_xy_noise = self._activation_snapshot_object_xy_noise.to(device=device, dtype=dtype)
        if bool(torch.any(object_xy_noise > 0.0)):
            object_noise_local = (torch.rand((num_envs, 2), device=device, dtype=dtype) * 2.0 - 1.0) * object_xy_noise.unsqueeze(0)
            object_noise_local_3 = torch.zeros((num_envs, 3), device=device, dtype=dtype)
            object_noise_local_3[:, :2] = object_noise_local
            object_noise_world = quat_apply(table_quat_world.to(device=device, dtype=dtype), object_noise_local_3)
            object_center_world = object_center_world + object_noise_world

        hand_yaw_noise = self._activation_snapshot_hand_yaw_noise_rad
        hand_pitch_noise = self._activation_snapshot_hand_pitch_noise_rad
        hand_roll_noise = self._activation_snapshot_hand_roll_noise_rad
        if max(hand_yaw_noise, hand_pitch_noise, hand_roll_noise) > 0.0:
            eef_pose = self.get_ee_from_joint(joint_config[:, :7])
            eef_pos = eef_pose[:, :3]
            eef_quat = eef_pose[:, 3:7]

            yaw_axis = torch.zeros((num_envs, 3), device=device, dtype=dtype)
            yaw_axis[:, 2] = 1.0
            yaw_angle = (torch.rand((num_envs,), device=device, dtype=dtype) * 2.0 - 1.0) * hand_yaw_noise
            q_yaw_noise = quat_from_angle_axis(yaw_angle, yaw_axis)

            hand_x_axis = quat_apply(
                eef_quat,
                torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(num_envs, 1),
            )
            hand_y_axis = quat_apply(
                eef_quat,
                torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype).repeat(num_envs, 1),
            )
            roll_angle = (torch.rand((num_envs,), device=device, dtype=dtype) * 2.0 - 1.0) * hand_roll_noise
            pitch_angle = (torch.rand((num_envs,), device=device, dtype=dtype) * 2.0 - 1.0) * hand_pitch_noise
            q_roll_noise = quat_from_angle_axis(roll_angle, hand_x_axis)
            q_pitch_noise = quat_from_angle_axis(pitch_angle, hand_y_axis)
            noisy_eef_quat = quat_mul(q_pitch_noise, quat_mul(q_roll_noise, quat_mul(q_yaw_noise, eef_quat)))
            noisy_eef_quat = noisy_eef_quat / torch.norm(noisy_eef_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
            noisy_eef_pose = torch.cat([eef_pos, noisy_eef_quat], dim=-1)
            with torch.enable_grad():
                arm_q_ik, success = self.get_joint_from_ee(
                    noisy_eef_pose,
                    return_success=True,
                    use_reset_solver=True,
                )
            success = success.bool().reshape(-1)
            if bool(torch.any(success)):
                joint_config[success, :7] = arm_q_ik[success]

        return joint_config, object_center_world, object_quat_world

    def _queue_activation_snapshot_debug(self, env_ids, debug_payload):
        if self._activation_snapshot_debug_print_limit <= 0:
            return
        remaining = self._activation_snapshot_debug_print_limit - self._activation_snapshot_debug_print_count
        if remaining <= 0:
            return
        take = min(int(env_ids.numel()), int(remaining))
        if take <= 0:
            return
        idx = slice(0, take)
        queued = {}
        for key, value in debug_payload.items():
            if value is None:
                queued[key] = None
            elif torch.is_tensor(value):
                queued[key] = value[idx].detach().clone()
            else:
                queued[key] = value
        self._activation_snapshot_debug_pending = queued

    def _debug_env_selected(self, env_id):
        return (
            env_id in self._copy_debug_env_ids
            and self._copy_debug_print_count < self._copy_debug_print_limit
        )

    def _debug_print_copy_reset(self, env_ids, joint_config, object_center_world, object_quat_world, table_pos_world, table_quat_world):
        if not self._copy_debug_env_ids:
            return
        for local_i, env_id_t in enumerate(env_ids.detach().cpu().tolist()):
            env_id = int(env_id_t)
            if not self._debug_env_selected(env_id):
                continue
            lines = [
                f"[CopyResetDebug/reset] step={int(self.sim_steps)} env={env_id}",
                f"  variation_id={int(self.env_variation_ids[env_id].item())}",
                f"  table_root_target={table_pos_world[local_i].detach().cpu().tolist()}",
                f"  table_quat_target={table_quat_world[local_i].detach().cpu().tolist()}",
                f"  object_center_target={object_center_world[local_i].detach().cpu().tolist()}",
                f"  object_quat_target={object_quat_world[local_i].detach().cpu().tolist()}",
                f"  object_root_written={self._root_state[env_id, self._object_id, :3].detach().cpu().tolist()}",
                f"  table_root_written={self._root_state[env_id, 1, :3].detach().cpu().tolist()}",
                f"  arm_q_target={joint_config[local_i, :7].detach().cpu().tolist()}",
                f"  arm_q_written={self._q[env_id, :7].detach().cpu().tolist()}",
                f"  table_surface_height={float(self.table_surface_height[env_id].item()):.6f}",
                f"  post_reset_grace={int(self._post_reset_grace_buf[env_id].item())}",
            ]
            print("\n".join(lines), flush=True)
            self._copy_debug_print_count += 1
            if self._copy_debug_print_count >= self._copy_debug_print_limit:
                break

    def _debug_print_copy_done_reason(self, reason, env_ids, below_table=None, dist_to_table_center_xy=None):
        if not self._copy_debug_env_ids:
            return
        for env_id_t in env_ids.detach().cpu().tolist():
            env_id = int(env_id_t)
            if not self._debug_env_selected(env_id):
                continue
            lines = [
                f"[CopyResetDebug/done] step={int(self.sim_steps)} env={env_id} reason={reason}",
                f"  variation_id={int(self.env_variation_ids[env_id].item())}",
                f"  object_center_world={self.states['object_center_pos'][env_id].detach().cpu().tolist()}",
                f"  object_root_world={self._object_state[env_id, :3].detach().cpu().tolist()}",
                f"  table_root_world={self._root_state[env_id, 1, :3].detach().cpu().tolist()}",
                f"  table_surface_height={float(self.table_surface_height[env_id].item()):.6f}",
                f"  eef_pos_world={self.states['eef_pos'][env_id].detach().cpu().tolist()}",
                f"  progress_buf={int(self.progress_buf[env_id].item())}",
                f"  post_reset_grace={int(self._post_reset_grace_buf[env_id].item())}",
            ]
            if below_table is not None:
                lines.append(f"  below_table={bool(below_table[env_id].item())}")
            if dist_to_table_center_xy is not None:
                lines.append(f"  dist_to_table_center_xy={float(dist_to_table_center_xy[env_id].item()):.6f}")
            print("\n".join(lines), flush=True)
            self._copy_debug_print_count += 1
            if self._copy_debug_print_count >= self._copy_debug_print_limit:
                break

    def _debug_print_reset_dispatch(self, env_ids):
        if not self._copy_debug_env_ids:
            return
        for env_id_t in env_ids.detach().cpu().tolist():
            env_id = int(env_id_t)
            if not self._debug_env_selected(env_id):
                continue
            lines = [
                f"[CopyResetDebug/dispatch] step={int(self.sim_steps)} env={env_id}",
                f"  variation_id={int(self.env_variation_ids[env_id].item())}",
                f"  reset_buf={int(self.reset_buf[env_id].item())}",
                f"  progress_buf={int(self.progress_buf[env_id].item())}",
                f"  post_reset_grace={int(self._post_reset_grace_buf[env_id].item())}",
                f"  object_center_world={self.states['object_center_pos'][env_id].detach().cpu().tolist()}",
                f"  table_surface_height={float(self.table_surface_height[env_id].item()):.6f}",
            ]
            print("\n".join(lines), flush=True)
            self._copy_debug_print_count += 1
            if self._copy_debug_print_count >= self._copy_debug_print_limit:
                break

    def _debug_print_follow_after_reset(self):
        if not self._copy_debug_env_ids:
            return
        active_env_ids = (self._copy_debug_follow_buf > 0).nonzero(as_tuple=False).squeeze(-1)
        if active_env_ids.numel() == 0:
            return
        table_center_xy = self.cuboid_pos[:, 0, :2]
        object_center_xy = self.states["object_center_pos"][:, :2]
        dist_to_table_center_xy = torch.norm(object_center_xy - table_center_xy, dim=-1)
        for env_id_t in active_env_ids.detach().cpu().tolist():
            env_id = int(env_id_t)
            if not self._debug_env_selected(env_id):
                continue
            lines = [
                f"[CopyResetDebug/follow] step={int(self.sim_steps)} env={env_id}",
                f"  follow_steps_left={int(self._copy_debug_follow_buf[env_id].item())}",
                f"  object_center_world={self.states['object_center_pos'][env_id].detach().cpu().tolist()}",
                f"  object_root_world={self._object_state[env_id, :3].detach().cpu().tolist()}",
                f"  table_root_world={self._root_state[env_id, 1, :3].detach().cpu().tolist()}",
                f"  table_surface_height={float(self.table_surface_height[env_id].item()):.6f}",
                f"  eef_pos_world={self.states['eef_pos'][env_id].detach().cpu().tolist()}",
                f"  table_collision={bool(self.table_collision[env_id].item())}",
                f"  dist_to_table_center_xy={float(dist_to_table_center_xy[env_id].item()):.6f}",
                f"  progress_buf={int(self.progress_buf[env_id].item())}",
                f"  post_reset_grace={int(self._post_reset_grace_buf[env_id].item())}",
            ]
            print("\n".join(lines), flush=True)
            self._copy_debug_print_count += 1
            if self._copy_debug_print_count >= self._copy_debug_print_limit:
                break

    def _debug_print_all4_object_table(self):
        if self.num_envs != 4:
            return
        lines = [f"[CopyResetDebug/all4] step={int(self.sim_steps)}"]
        for env_id in range(self.num_envs):
            reset_object_center_world = self._object_center_init_state[env_id].detach().cpu().tolist()
            live_object_center_world = self.states["object_center_pos"][env_id].detach().cpu().tolist()
            object_center_minus_table = float(self.states["object_center_pos"][env_id, 2].item()) - float(self.table_surface_height[env_id].item())
            lines.append(
                "  "
                f"env={env_id} "
                f"variation_id={int(self.env_variation_ids[env_id].item())} "
                f"reset_object_center_world={reset_object_center_world} "
                f"live_object_center_world={live_object_center_world} "
                f"table_surface_height_world={float(self.table_surface_height[env_id].item()):.6f} "
                f"obj_center_minus_table={object_center_minus_table:.6f} "
                f"reset_buf={int(self.reset_buf[env_id].item())} "
                f"progress_buf={int(self.progress_buf[env_id].item())} "
                f"post_reset_grace={int(self._post_reset_grace_buf[env_id].item())}"
            )
        print("\n".join(lines), flush=True)

    def _print_activation_snapshot_debug(self):
        payload = self._activation_snapshot_debug_pending
        if payload is None or self._activation_snapshot_debug_print_limit <= 0:
            return
        num_entries = int(payload["env_ids"].numel())
        for i in range(num_entries):
            env_id = int(payload["env_ids"][i].item())
            variation_id = int(payload["variation_id"][i].item())
            mode_id = int(payload["mode_id"][i].item())
            slot_id = int(payload["slot_id"][i].item())
            mode_name = SNAPSHOT_RESET_MODE_NAMES[mode_id]
            lines = [
                (
                    "[SnapshotDebug] "
                    f"env={env_id} variation_id={variation_id} mode={mode_name} slot={slot_id}"
                ),
            ]
            if payload["mobile_base_pose"] is not None:
                lines.append(
                    f"  mobile_base_pose_xyyaw={payload['mobile_base_pose'][i].detach().cpu().tolist()}"
                )
            if payload["franka_base_pose7"] is not None:
                lines.append(
                    f"  distill_franka_base_pose7={payload['franka_base_pose7'][i].detach().cpu().tolist()}"
                )
            lines.append(
                f"  raw_table_center_world={payload['raw_table_pos_world'][i].detach().cpu().tolist()}"
            )
            lines.append(
                f"  replay_table_center_world={payload['table_pos_replay_world'][i].detach().cpu().tolist()}"
            )
            lines.append(
                f"  applied_table_root_world={self._root_state[env_id, 1, :3].detach().cpu().tolist()}"
            )
            lines.append(
                f"  raw_object_center_world={payload['raw_object_center_world'][i].detach().cpu().tolist()}"
            )
            lines.append(
                f"  replay_object_center_world={payload['object_center_replay_world'][i].detach().cpu().tolist()}"
            )
            lines.append(
                f"  applied_object_center_world={self.states['object_center_pos'][env_id].detach().cpu().tolist()}"
            )
            lines.append(
                f"  eef_pos_world={self.states['eef_pos'][env_id].detach().cpu().tolist()}"
            )
            lines.append(
                f"  table_surface_height={float(self.table_surface_height[env_id].item()):.6f}"
            )
            lines.append(
                f"  object_root_world={self._object_state[env_id, :3].detach().cpu().tolist()}"
            )
            lines.append(
                f"  sampled_arm_q={payload['joint_config'][i, :7].detach().cpu().tolist()}"
            )
            lines.append(
                f"  live_arm_q={self._q[env_id, :7].detach().cpu().tolist()}"
            )
            print("\n".join(lines), flush=True)
            self._activation_snapshot_debug_print_count += 1
            if self._activation_snapshot_debug_print_count >= self._activation_snapshot_debug_print_limit:
                break
        self._activation_snapshot_debug_pending = None

    def _compose_snapshot_target_quat_world(self, side_is_left, table_quat_world):
        num_envs = int(side_is_left.numel())
        target_quat_world = self._compose_table_frame_quat_to_world(
            table_quat_world,
            self.target_quat_right_canonical.to(device=self.device, dtype=self._q.dtype).repeat(num_envs, 1),
        )
        if int(side_is_left.sum().item()) > 0:
            target_quat_world[side_is_left] = self._compose_table_frame_quat_to_world(
                table_quat_world[side_is_left],
                self.target_quat_left_canonical.to(device=self.device, dtype=self._q.dtype).repeat(int(side_is_left.sum().item()), 1),
            )
        return target_quat_world

    def _sample_reset_from_bank(self, env_ids):
        object_ids_cpu = self.env_object_ids[env_ids].detach().to(device="cpu", dtype=torch.long)
        bank_idx_cpu = torch.randint(
            low=0,
            high=self._reset_bank_size,
            size=(object_ids_cpu.numel(),),
            device="cpu",
            dtype=torch.long,
        )

        joint_config = self._reset_bank_joint_config_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        target_quat = self._reset_bank_target_quat_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        left_mask = self._reset_bank_side_is_left_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device)
        object_center_world = self._reset_bank_object_center_world_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        object_quat_world = self._reset_bank_object_quat_world_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        return joint_config, target_quat, left_mask, object_center_world, object_quat_world

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

        self.teleport_buf[env_ids] = 0

    def _apply_snapshot_scene_state(
        self,
        env_ids,
        table_center_world,
        table_quat_world,
        object_center_world,
        object_quat_world,
    ):
        num_resets = int(env_ids.numel())
        if num_resets == 0:
            return

        dtype = self._root_state.dtype
        table_center_world = table_center_world.to(device=self.device, dtype=dtype)
        table_quat_world = self._normalize_quat_tensor(
            table_quat_world.to(device=self.device, dtype=dtype)
        )
        object_center_world = object_center_world.to(device=self.device, dtype=dtype)
        object_quat_world = self._normalize_quat_tensor(
            object_quat_world.to(device=self.device, dtype=dtype)
        )

        local_offset = torch.zeros((num_resets, 3), device=self.device, dtype=dtype)
        local_offset[:, 2] = self.mesh_aabb_extents[env_ids, 2] * self.object_center_z_scale
        object_root_pos = object_center_world - quat_apply(object_quat_world, local_offset)

        self._root_state[env_ids, 1, :3] = table_center_world
        self._root_state[env_ids, 1, 3:7] = table_quat_world
        self._root_state[env_ids, 1, 7:13] = 0.0

        self._root_state[env_ids, self._object_id, :3] = object_root_pos
        self._root_state[env_ids, self._object_id, 3:7] = object_quat_world
        self._root_state[env_ids, self._object_id, 7:13] = 0.0

        self.cuboid_pos[env_ids, 0] = table_center_world
        self.cuboid_quats[env_ids, 0] = table_quat_world
        self.table_surface_height[env_ids] = table_center_world[:, 2] + 0.5 * self.cuboid_dims[env_ids, 0, 2]
        self._object_center_init_state[env_ids] = object_center_world

        if hasattr(self, "_static_pcd_local"):
            local_static = self._static_pcd_local[env_ids]
            table_rot_mat = quaternion_to_matrix_ig(table_quat_world)
            static_pcd_world = torch.matmul(table_rot_mat, local_static.transpose(1, 2)).transpose(1, 2)
            static_pcd_world = static_pcd_world + table_center_world.unsqueeze(1)
            self.static_pcds[env_ids] = static_pcd_world
            num_static_points = self.pcd_spec_dict["num_static_points"]
            self.combined_pcds[env_ids, :num_static_points] = static_pcd_world

        actor_ids = torch.stack(
            [
                self._global_indices[env_ids, 1],
                self._global_indices[env_ids, self._object_id],
            ],
            dim=1,
        ).reshape(-1)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

        self.teleport_buf[env_ids] = 0

    @staticmethod
    def _normalize_quat_tensor(quat):
        return quat / torch.norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)

    @staticmethod
    def _quat_conjugate_tensor(quat):
        quat_conj = quat.clone()
        quat_conj[..., :3] = -quat_conj[..., :3]
        return quat_conj

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

    def _get_table_frame_data(self):
        table_pos_world = self.cuboid_pos[:, 0, :]
        table_quat_world = self.cuboid_quats[:, 0, :]
        table_rot_mat = quaternion_to_matrix_ig(table_quat_world)
        table_rot_mat_t = table_rot_mat.transpose(1, 2)
        return table_pos_world, table_quat_world, table_rot_mat, table_rot_mat_t

    def _world_vectors_to_table_frame(self, vectors_world, table_rot_mat_t=None):
        if table_rot_mat_t is None:
            _, _, _, table_rot_mat_t = self._get_table_frame_data()
        return torch.matmul(table_rot_mat_t, vectors_world.unsqueeze(-1)).squeeze(-1)

    def _world_points_to_table_frame(self, points_world, table_pos_world=None, table_rot_mat_t=None):
        if table_pos_world is None or table_rot_mat_t is None:
            table_pos_world, _, _, table_rot_mat_t = self._get_table_frame_data()
        return torch.matmul(
            table_rot_mat_t,
            (points_world - table_pos_world).unsqueeze(-1),
        ).squeeze(-1)

    def _compose_table_frame_quat_to_world(self, table_quat_world, local_quat):
        return self._normalize_quat_tensor(quat_mul(table_quat_world, local_quat))

    def _apply_table_state(self, env_ids, table_center_world, table_quat_world=None):
        num_resets = int(env_ids.numel())
        if num_resets == 0:
            return

        if table_quat_world is None:
            table_quat_world = torch.zeros((num_resets, 4), device=self.device, dtype=self._root_state.dtype)
            table_quat_world[:, 3] = 1.0
        table_quat_world = self._normalize_quat_tensor(table_quat_world.to(device=self.device, dtype=self._root_state.dtype))
        table_center_world = table_center_world.to(device=self.device, dtype=self._root_state.dtype)

        self._root_state[env_ids, 1, :3] = table_center_world
        self._root_state[env_ids, 1, 3:7] = table_quat_world
        self._root_state[env_ids, 1, 7:13] = 0.0

        self.cuboid_pos[env_ids, 0] = table_center_world
        self.cuboid_quats[env_ids, 0] = table_quat_world
        self.table_surface_height[env_ids] = table_center_world[:, 2] + 0.5 * self.cuboid_dims[env_ids, 0, 2]

        if hasattr(self, "_static_pcd_local"):
            local_static = self._static_pcd_local[env_ids]
            table_rot_mat = quaternion_to_matrix_ig(table_quat_world)
            static_pcd_world = torch.matmul(table_rot_mat, local_static.transpose(1, 2)).transpose(1, 2)
            static_pcd_world = static_pcd_world + table_center_world.unsqueeze(1)
            self.static_pcds[env_ids] = static_pcd_world
            num_static_points = self.pcd_spec_dict["num_static_points"]
            self.combined_pcds[env_ids, :num_static_points] = static_pcd_world

        multi_env_ids_int32 = self._global_indices[env_ids, 1].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def _create_movable_table_cube(self, pos, size, quat=[0, 0, 0, 1]):
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        opts.disable_gravity = True
        asset = self.gym.create_box(self.sim, *size, opts)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(*quat)
        self.cuboid_dims.append(size)
        self.cuboid_pos.append(pos)
        self.cuboid_quats.append(quat)
        return asset, start_pose

    def _get_reset_object_xy_bounds(self, env_ids, mirror_mask, dtype):
        table_center_xy = self.cuboid_pos[env_ids, 0, :2].to(dtype=dtype)
        table_half_xy = 0.5 * self.cuboid_dims[env_ids, 0, :2].to(dtype=dtype)
        table_xy_min = table_center_xy - table_half_xy
        table_xy_max = table_center_xy + table_half_xy

        canonical_min = self._reset_object_xy_min.to(device=self.device, dtype=dtype)
        canonical_max = self._reset_object_xy_max.to(device=self.device, dtype=dtype)

        x_min = torch.full((env_ids.numel(),), canonical_min[0], device=self.device, dtype=dtype)
        x_max = torch.full((env_ids.numel(),), canonical_max[0], device=self.device, dtype=dtype)
        y_min = torch.where(
            mirror_mask,
            torch.full((env_ids.numel(),), -canonical_max[1], device=self.device, dtype=dtype),
            torch.full((env_ids.numel(),), canonical_min[1], device=self.device, dtype=dtype),
        )
        y_max = torch.where(
            mirror_mask,
            torch.full((env_ids.numel(),), -canonical_min[1], device=self.device, dtype=dtype),
            torch.full((env_ids.numel(),), canonical_max[1], device=self.device, dtype=dtype),
        )

        empirical_xy_min = torch.stack([x_min, y_min], dim=-1)
        empirical_xy_max = torch.stack([x_max, y_max], dim=-1)
        object_xy_min = torch.maximum(empirical_xy_min, table_xy_min)
        object_xy_max = torch.minimum(empirical_xy_max, table_xy_max)
        return object_xy_min, object_xy_max

    def _sample_reset_joint_and_target_quat(self, env_ids, use_reset_solver=False):
        num_envs = int(env_ids.numel())
        device = self.device
        dtype = self._q.dtype
        table_quat_world = self.cuboid_quats[env_ids, 0, :].to(device=device, dtype=dtype)

        side_mode = self.side_mode
        if side_mode == "left":
            left_mask = torch.ones(num_envs, device=device, dtype=torch.bool)
        elif side_mode == "right":
            left_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
        elif side_mode == "both":
            left_mask = torch.rand(num_envs, device=device) < 0.5
        else:
            raise ValueError(f"Unsupported eef_init.side_mode={side_mode}. Expected one of: left, right, both.")
        object_mirror_mask = torch.rand(num_envs, device=device, dtype=dtype) < self._reset_object_mirror_y_prob

        target_quat_right_world = self._compose_table_frame_quat_to_world(
            table_quat_world,
            self.target_quat_right_canonical.to(device=device, dtype=dtype).repeat(num_envs, 1),
        )
        target_quat_left_world = self._compose_table_frame_quat_to_world(
            table_quat_world,
            self.target_quat_left_canonical.to(device=device, dtype=dtype).repeat(num_envs, 1),
        )
        target_quat_flat_world = self._compose_table_frame_quat_to_world(
            table_quat_world,
            self.target_quat_flat_canonical.to(device=device, dtype=dtype).repeat(num_envs, 1),
        )

        reward_target_quat_out = target_quat_right_world.clone()
        if int(left_mask.sum().item()) > 0:
            reward_target_quat_out[left_mask] = target_quat_left_world[left_mask]
        side_init_quat = reward_target_quat_out.clone()
        lie_flat_mask = torch.rand(num_envs, device=device) < self.lie_flat_prob
        if bool(torch.any(lie_flat_mask)):
            reward_target_quat_out[lie_flat_mask] = target_quat_flat_world[lie_flat_mask]

        joint_config = torch.zeros((num_envs, self.num_dofs), device=device, dtype=dtype)
        joint_config[:, 7:23] = self.canonical_flat_hand_config.unsqueeze(0).repeat(num_envs, 1).to(dtype=dtype)

        object_center_world = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        object_quat_world = torch.zeros((num_envs, 4), device=device, dtype=dtype)
        object_quat_world[:, 3] = 1.0
        table_surface_height = self.table_surface_height[env_ids].to(dtype=dtype)
        remaining = torch.arange(num_envs, device=device)
        rel_min = self._reset_eef_rel_object_min.to(device=device, dtype=dtype)
        rel_max = self._reset_eef_rel_object_max.to(device=device, dtype=dtype)
        flat_rel_min = self._reset_flat_eef_rel_object_min.to(device=device, dtype=dtype)
        flat_rel_max = self._reset_flat_eef_rel_object_max.to(device=device, dtype=dtype)
        base_rel_eef_min = self._reset_base_rel_eef_min.to(device=device, dtype=dtype)
        base_rel_eef_max = self._reset_base_rel_eef_max.to(device=device, dtype=dtype)
        flat_base_rel_eef_min = self._reset_flat_base_rel_eef_min.to(device=device, dtype=dtype)
        flat_base_rel_eef_max = self._reset_flat_base_rel_eef_max.to(device=device, dtype=dtype)
        flat_box_size_min = self._reset_flat_topdown_box_size_min.to(device=device, dtype=dtype)
        flat_box_size_max = self._reset_flat_topdown_box_size_max.to(device=device, dtype=dtype)
        flat_dis_open_range = self._reset_flat_topdown_dis_open_range.to(device=device, dtype=dtype)
        flat_dis_side_range = torch.tensor(self._reset_flat_topdown_dis_side_range, device=device, dtype=dtype)
        flat_obj_wall_tol = torch.tensor(self._reset_flat_topdown_obj_wall_tol, device=device, dtype=dtype)
        hand_joint_noise = self._reset_hand_joint_noise_rad.to(device=device, dtype=dtype)
        table_center_xy = self.cuboid_pos[env_ids, 0, :2].to(dtype=dtype)
        table_half_xy = 0.5 * self.cuboid_dims[env_ids, 0, :2].to(dtype=dtype)
        object_xy_min, object_xy_max = self._get_reset_object_xy_bounds(env_ids, object_mirror_mask, dtype)
        while remaining.numel() > 0:
            batch = int(remaining.numel())
            remaining_flat_mask = lie_flat_mask[remaining]
            remaining_flat_side_mask = remaining_flat_mask & (
                torch.rand(batch, device=device, dtype=dtype) < self._reset_flat_side_recovery_prob
            )
            remaining_flat_topdown_mask = remaining_flat_mask & (~remaining_flat_side_mask)
            flat_box_dims = flat_box_size_min.unsqueeze(0) + torch.rand((batch, 3), device=device, dtype=dtype) * (
                flat_box_size_max - flat_box_size_min
            ).unsqueeze(0)
            rel = torch.rand((batch, 3), device=device, dtype=dtype)
            rel_upright = rel_min.unsqueeze(0) + rel * (rel_max - rel_min).unsqueeze(0)
            rel_flat = flat_rel_min.unsqueeze(0) + rel * (flat_rel_max - flat_rel_min).unsqueeze(0)
            grasp_side_sign = torch.where(
                left_mask[remaining],
                torch.ones(batch, device=device, dtype=dtype),
                -torch.ones(batch, device=device, dtype=dtype),
            )
            mirror_sign = torch.where(
                object_mirror_mask[remaining],
                -torch.ones(batch, device=device, dtype=dtype),
                torch.ones(batch, device=device, dtype=dtype),
            )
            rel_y_sign = grasp_side_sign * mirror_sign
            rel_upright[:, 1] = rel_y_sign * rel_upright[:, 1]
            rel_flat[:, 1] = rel_y_sign * rel_flat[:, 1]

            object_xy_rand = torch.rand((batch, 2), device=device, dtype=dtype)
            upright_object_xy = object_xy_min[remaining] + object_xy_rand * (object_xy_max[remaining] - object_xy_min[remaining])
            object_half_xy = 0.5 * self.mesh_aabb_extents[env_ids[remaining], :2].to(dtype=dtype)
            flat_object_xy_min = table_center_xy[remaining] - 0.5 * flat_box_dims[:, :2] + object_half_xy + flat_obj_wall_tol
            flat_object_xy_max = table_center_xy[remaining] + 0.5 * flat_box_dims[:, :2] - object_half_xy - flat_obj_wall_tol
            flat_object_xy = flat_object_xy_min + object_xy_rand * (flat_object_xy_max - flat_object_xy_min).clamp_min(0.0)
            flat_object_range_valid = torch.all(flat_object_xy_min <= flat_object_xy_max, dim=-1)
            object_xy = torch.where(remaining_flat_topdown_mask.unsqueeze(-1), flat_object_xy, upright_object_xy)

            candidate_object_quat = torch.zeros((batch, 4), device=device, dtype=dtype)
            candidate_object_quat[:, 3] = 1.0
            if bool(torch.any(remaining_flat_mask)):
                flat_yaw_axis = torch.zeros((batch, 3), device=device, dtype=dtype)
                flat_yaw_axis[:, 2] = 1.0
                flat_yaw_angle = torch.rand(batch, device=device, dtype=dtype) * (2.0 * torch.pi)
                flat_q_yaw = quat_from_angle_axis(flat_yaw_angle, flat_yaw_axis)
                flat_quat_base = self._flat_object_quat_base.to(device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1)
                flat_object_quat = quat_mul(flat_q_yaw, flat_quat_base)
                flat_object_quat = flat_object_quat / torch.norm(flat_object_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
                candidate_object_quat = torch.where(
                    remaining_flat_mask.unsqueeze(-1),
                    flat_object_quat,
                    candidate_object_quat,
                )
            local_offset = torch.zeros((batch, 3), device=device, dtype=dtype)
            local_offset[:, 2] = self.mesh_aabb_extents[env_ids[remaining], 2].to(dtype=dtype) * self.object_center_z_scale
            rotated_offset = quat_apply(candidate_object_quat, local_offset)

            object_center_candidate = torch.zeros((batch, 3), device=device, dtype=dtype)
            object_center_candidate[:, :2] = object_xy
            object_center_candidate[:, 2] = table_surface_height[remaining] + rotated_offset[:, 2]

            solved_batch_mask = torch.zeros(batch, device=device, dtype=torch.bool)

            eef_pos_upright = object_center_candidate + rel_upright
            flat_eef_xy_span = (0.5 * flat_box_dims[:, :2] + flat_dis_side_range).clamp_min(0.0)
            flat_eef_xy = table_center_xy[remaining] + (torch.rand((batch, 2), device=device, dtype=dtype) * 2.0 - 1.0) * flat_eef_xy_span
            flat_dis_open = flat_dis_open_range[0] + torch.rand(batch, device=device, dtype=dtype) * (
                flat_dis_open_range[1] - flat_dis_open_range[0]
            )
            eef_pos_flat = torch.zeros((batch, 3), device=device, dtype=dtype)
            eef_pos_flat[:, :2] = flat_eef_xy
            eef_pos_flat[:, 2] = table_surface_height[remaining] + flat_box_dims[:, 2] + flat_dis_open
            eef_pos = torch.where(remaining_flat_topdown_mask.unsqueeze(-1), eef_pos_flat, eef_pos_upright)
            rel = eef_pos - object_center_candidate
            base_rel_eef = -eef_pos
            table_xy_min = table_center_xy[remaining] - table_half_xy[remaining]
            table_xy_max = table_center_xy[remaining] + table_half_xy[remaining]
            upright_base_rel_eef_y_min = torch.where(
                object_mirror_mask[remaining],
                -base_rel_eef_max[1],
                base_rel_eef_min[1],
            )
            upright_base_rel_eef_y_max = torch.where(
                object_mirror_mask[remaining],
                -base_rel_eef_min[1],
                base_rel_eef_max[1],
            )
            base_rel_eef_x_min = torch.where(remaining_flat_topdown_mask, flat_base_rel_eef_min[0], base_rel_eef_min[0])
            base_rel_eef_x_max = torch.where(remaining_flat_topdown_mask, flat_base_rel_eef_max[0], base_rel_eef_max[0])
            base_rel_eef_y_min = torch.where(remaining_flat_topdown_mask, flat_base_rel_eef_min[1], upright_base_rel_eef_y_min)
            base_rel_eef_y_max = torch.where(remaining_flat_topdown_mask, flat_base_rel_eef_max[1], upright_base_rel_eef_y_max)
            base_rel_eef_z_min = torch.where(remaining_flat_topdown_mask, flat_base_rel_eef_min[2], base_rel_eef_min[2])
            base_rel_eef_z_max = torch.where(remaining_flat_topdown_mask, flat_base_rel_eef_max[2], base_rel_eef_max[2])
            object_half_y = 0.5 * self.mesh_aabb_extents[env_ids[remaining], 1].to(dtype=dtype)
            palm_object_dist = torch.norm(rel, dim=-1)
            side_clearance = torch.abs(rel[:, 1]) - object_half_y
            valid_scene = (
                (object_center_candidate[:, 0] >= table_xy_min[:, 0]) &
                (object_center_candidate[:, 0] <= table_xy_max[:, 0]) &
                (object_center_candidate[:, 1] >= table_xy_min[:, 1]) &
                (object_center_candidate[:, 1] <= table_xy_max[:, 1]) &
                (base_rel_eef[:, 0] >= base_rel_eef_x_min) &
                (base_rel_eef[:, 0] <= base_rel_eef_x_max) &
                (base_rel_eef[:, 1] >= base_rel_eef_y_min) &
                (base_rel_eef[:, 1] <= base_rel_eef_y_max) &
                (base_rel_eef[:, 2] >= base_rel_eef_z_min) &
                (base_rel_eef[:, 2] <= base_rel_eef_z_max) &
                (palm_object_dist >= self._reset_min_palm_object_dist) &
                ((~remaining_flat_topdown_mask) | flat_object_range_valid) &
                (remaining_flat_mask | (side_clearance >= self._reset_side_clearance))
            )
            if not bool(torch.any(valid_scene)):
                continue

            left_mask_valid = left_mask[remaining[valid_scene]]
            flat_mask_valid = remaining_flat_mask[valid_scene]
            flat_side_mask_valid = remaining_flat_side_mask[valid_scene]
            flat_topdown_mask_valid = remaining_flat_topdown_mask[valid_scene]
            side_init_quat_valid = side_init_quat[remaining[valid_scene]]
            # Keep the side-grasp hand orientation tied only to the grasp type.
            # Mirroring the object/table-side should move the sampled reset pose,
            # not yaw-flip the hand to face the opposite direction.
            aligned_side_quat = side_init_quat_valid.clone()
            valid_count = side_init_quat_valid.shape[0]
            flat_box_center = torch.zeros((valid_count, 3), device=device, dtype=dtype)
            flat_box_center[:, :2] = table_center_xy[remaining[valid_scene]]
            flat_box_center[:, 2] = table_surface_height[remaining[valid_scene]]
            flat_topdown_quat = A2B_quaternion(
                eef_pos[valid_scene],
                flat_box_center,
                max_angle_deg=20,
                right_axis="y",
            ).to(dtype=dtype)
            flip_idx = eef_pos[valid_scene, 0] < flat_box_center[:, 0]
            if bool(torch.any(flip_idx)):
                rot_local_z_180 = torch.tensor(
                    [[0.0, 0.0, 1.0, 0.0]],
                    device=device,
                    dtype=dtype,
                ).repeat(int(flip_idx.sum().item()), 1)
                flat_topdown_quat[flip_idx] = quat_mul(flat_topdown_quat[flip_idx], rot_local_z_180)
            nominal_ik_quat = torch.where(
                flat_topdown_mask_valid.unsqueeze(-1),
                flat_topdown_quat,
                aligned_side_quat,
            )
            nominal_ik_quat = nominal_ik_quat / torch.norm(nominal_ik_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)

            yaw_axis = torch.zeros((nominal_ik_quat.shape[0], 3), device=device, dtype=dtype)
            yaw_axis[:, 2] = 1.0
            yaw_angle = (torch.rand((nominal_ik_quat.shape[0],), device=device, dtype=dtype) * 2.0 - 1.0) * self._reset_yaw_noise_rad
            yaw_angle = torch.where(
                flat_topdown_mask_valid,
                torch.zeros_like(yaw_angle),
                yaw_angle,
            )
            q_yaw_noise = quat_from_angle_axis(yaw_angle, yaw_axis)

            hand_x_axis = quat_apply(
                nominal_ik_quat,
                torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).repeat(nominal_ik_quat.shape[0], 1),
            )
            hand_y_axis = quat_apply(
                nominal_ik_quat,
                torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype).repeat(nominal_ik_quat.shape[0], 1),
            )
            roll_angle = (torch.rand((nominal_ik_quat.shape[0],), device=device, dtype=dtype) * 2.0 - 1.0) * self._reset_pitch_roll_noise_rad
            pitch_angle = (torch.rand((nominal_ik_quat.shape[0],), device=device, dtype=dtype) * 2.0 - 1.0) * self._reset_pitch_roll_noise_rad
            roll_angle = torch.where(
                flat_topdown_mask_valid,
                torch.zeros_like(roll_angle),
                roll_angle,
            )
            pitch_angle = torch.where(
                flat_topdown_mask_valid,
                torch.zeros_like(pitch_angle),
                pitch_angle,
            )
            q_roll_noise = quat_from_angle_axis(roll_angle, hand_x_axis)
            q_pitch_noise = quat_from_angle_axis(pitch_angle, hand_y_axis)
            ik_target_quat = quat_mul(q_pitch_noise, quat_mul(q_roll_noise, quat_mul(q_yaw_noise, nominal_ik_quat)))
            ik_target_quat = ik_target_quat / torch.norm(ik_target_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
            reward_target_quat_valid = nominal_ik_quat.clone()
            reward_target_quat_valid[flat_mask_valid] = flat_topdown_quat[flat_mask_valid]

            palm_offset_world = quat_apply(
                ik_target_quat,
                self._palm_center_from_link7_local.to(device=device, dtype=dtype).unsqueeze(0).repeat(int(valid_scene.sum().item()), 1),
            )
            link7_pos = eef_pos[valid_scene] - palm_offset_world
            eef_pose = torch.cat([link7_pos, ik_target_quat], dim=-1)
            with torch.enable_grad():
                arm_q, success = self.get_joint_from_ee(eef_pose, return_success=True, use_reset_solver=use_reset_solver)
            success = success.bool().reshape(-1)
            if bool(torch.any(success)):
                solved_remaining = remaining[valid_scene][success]
                joint_config[solved_remaining, :7] = arm_q[success]
                sampled_hand = self.canonical_flat_hand_config.unsqueeze(0).repeat(int(success.sum().item()), 1).to(dtype=dtype)
                sampled_hand = sampled_hand + (torch.rand_like(sampled_hand) * 2.0 - 1.0) * hand_joint_noise.unsqueeze(0)
                sampled_hand = tensor_clamp(
                    sampled_hand,
                    self.robot_dof_lower_limits[7:23].unsqueeze(0).repeat(int(success.sum().item()), 1),
                    self.robot_dof_upper_limits[7:23].unsqueeze(0).repeat(int(success.sum().item()), 1),
                )
                joint_config[solved_remaining, 7:23] = sampled_hand
                reward_target_quat_out[solved_remaining] = reward_target_quat_valid[success]
                object_center_world[solved_remaining] = object_center_candidate[valid_scene][success]
                object_quat_world[solved_remaining] = candidate_object_quat[valid_scene][success]
                valid_scene_idx = valid_scene.nonzero(as_tuple=False).squeeze(-1)
                solved_batch_mask[valid_scene_idx[success]] = True

            if bool(torch.any(solved_batch_mask)):
                remaining = remaining[~solved_batch_mask]

        return joint_config, reward_target_quat_out, left_mask, object_center_world, object_quat_world

    def pre_physics_step(self, actions):
        if (
            getattr(self, "_activation_snapshot_bank_loaded", False)
            and hasattr(self, "_desired_table_pos_world")
            and hasattr(self, "_desired_table_quat_world")
            and hasattr(self, "_snapshot_table_pin_buf")
        ):
            env_ids = (self._snapshot_table_pin_buf > 0).nonzero(as_tuple=False).squeeze(-1)
            if env_ids.numel() > 0:
                self._apply_table_state(
                    env_ids,
                    self._desired_table_pos_world[env_ids],
                    self._desired_table_quat_world[env_ids],
                )
        super().pre_physics_step(actions)

    def init_data(self, actor_num):
        super().init_data(actor_num)
        # @ray we use target_quat for reward computation and construct it based on the mode "left/right/both"
        self.target_quat_right = to_torch(self.cfg["reward"]["params"]["target_quat_right"], device=self.device).unsqueeze(0)
        self.target_quat_right = self.target_quat_right / (
            torch.norm(self.target_quat_right, dim=1, keepdim=True) + 1e-10
        )
        self.target_quat_left = to_torch(self.cfg["reward"]["params"]["target_quat_left"], device=self.device).unsqueeze(0)
        self.target_quat_left = self.target_quat_left / (
            torch.norm(self.target_quat_left, dim=1, keepdim=True) + 1e-10
        )
        self.target_quat_flat = to_torch(self.cfg["reward"]["params"].get("target_quat_flat", [1.0, 0.0, 0.0, 0.0]), device=self.device).unsqueeze(0)
        self.target_quat_flat = self.target_quat_flat / (
            torch.norm(self.target_quat_flat, dim=1, keepdim=True) + 1e-10
        )
        self.target_quat_right_canonical = self.target_quat_right.clone()
        self.target_quat_left_canonical = self.target_quat_left.clone()
        self.target_quat_flat_canonical = self.target_quat_flat.clone()
        
        self.reward_settings["beta_hand_orientation"] = to_torch(self.cfg["reward"]["exp"]["beta_hand_orientation"], device=self.device)
        self.reward_settings["w_hand_orientation"] = to_torch(self.cfg["reward"]["weights"]["w_hand_orientation"], device=self.device)
        orientation_reward_mode = str(self.cfg["reward"]["params"]["orientation_reward_mode"])  # CODEX
        if orientation_reward_mode == "old_targetquat":
            self.orientation_reward_mode_id = 0
        elif orientation_reward_mode == "canonical_axis":
            self.orientation_reward_mode_id = 1
        elif orientation_reward_mode == "object_axis":
            self.orientation_reward_mode_id = 2
        else:
            raise ValueError(
                f"Unsupported reward.params.orientation_reward_mode={orientation_reward_mode}. "
                "Expected one of: old_targetquat, canonical_axis, object_axis."
            )
        self.reward_settings["orientation_reward_mode"] = to_torch(float(self.orientation_reward_mode_id), device=self.device)  # CODEX
        self.reward_settings["object_upright_reward_floor"] = to_torch(
            float(self.cfg["reward"]["params"]["object_upright_reward_floor"]),
            device=self.device,
        )  # CODEX
        self.reward_settings["goal_align_gate_floor"] = to_torch(
            float(self.cfg["reward"]["params"].get("goal_align_gate_floor", 0.2)),
            device=self.device,
        )
        self.reward_settings["beta_goal_align"] = to_torch(
            float(self.cfg["reward"]["exp"].get("beta_goal_align", self.cfg["reward"]["exp"]["beta_hand_orientation"])),
            device=self.device,
        )
        self.reward_settings["w_goal_align"] = to_torch(
            float(self.cfg["reward"]["weights"].get("w_goal_align", 1.0)),
            device=self.device,
        )
        self.reward_settings["grasp_side_split_y_offset"] = to_torch(
            float(self.cfg["reward"]["params"].get("grasp_side_split_y_offset", 0.03)),
            device=self.device,
        )
        self.reward_settings["beta_recovery_region"] = to_torch(
            float(self.cfg["reward"]["exp"].get("beta_recovery_region", 10.0)),
            device=self.device,
        )
        self.reward_settings["w_recovery_region"] = to_torch(
            float(self.cfg["reward"]["weights"].get("w_recovery_region", 0.2)),
            device=self.device,
        )
        self.reward_settings["wrong_side_proximity_threshold"] = to_torch(
            float(self.cfg["reward"]["params"].get("wrong_side_proximity_threshold", 0.12)),
            device=self.device,
        )
        self.reward_settings["wrong_side_proximity_penalty"] = to_torch(
            float(self.cfg["reward"]["params"].get("wrong_side_proximity_penalty", 0.5)),
            device=self.device,
        )
        self.reward_settings["wrong_side_forbidden_box_enable"] = to_torch(
            1.0 if bool(self.cfg["reward"]["params"].get("wrong_side_forbidden_box_enable", True)) else 0.0,
            device=self.device,
        )
        self.reward_settings["wrong_side_forbidden_box_side_only"] = to_torch(
            1.0 if bool(self.cfg["reward"]["params"].get("wrong_side_forbidden_box_side_only", True)) else 0.0,
            device=self.device,
        )
        self.reward_settings["wrong_side_forbidden_box_flat_axis_z_abs_max"] = to_torch(
            float(self.cfg["reward"]["params"].get("wrong_side_forbidden_box_flat_axis_z_abs_max", 0.5)),
            device=self.device,
        )

        self.reward_settings["hand_obj_use_midheight_xy"] = to_torch(
            1.0 if bool(self.cfg["reward"]["params"]["hand_obj_use_midheight_xy"]) else 0.0,
            device=self.device,
        )  # CODEX
        self.reward_settings["hand_obj_midheight_height_weight"] = to_torch(
            float(self.cfg["reward"]["params"]["hand_obj_midheight_height_weight"]),
            device=self.device,
        )  # CODEX
        self.reward_settings["success_bonus_threshold"] = to_torch(
            float(self.cfg["reward"]["params"]["success_bonus_threshold"]),
            device=self.device,
        )
        self.reward_settings["w_success_bonus"] = to_torch(
            float(self.cfg["reward"]["weights"]["w_success_bonus"]),
            device=self.device,
        )
        hand_grasp_dir_local = torch.tensor(
            self.cfg["reward"]["params"]["hand_grasp_dir_local"],
            device=self.device,
            dtype=torch.float32,
        )  # CODEX
        self.hand_grasp_dir_local = hand_grasp_dir_local / torch.norm(hand_grasp_dir_local).clamp_min(1.0e-8)  # CODEX
        self.canonical_flat_hand_config = torch.tensor(
            [
                0.0000, 0.0000, 0.0000, 0.0000,
                -0.0000, 0.0000, 1.0000, 0.5700,
                0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000,
            ],
            device=self.device,
        )
        self.reward_settings["curl_dof_weight"] = torch.ones(16, device=self.device)

        self.reward_settings["w_obj_goal_base"] = self.reward_settings["w_obj_goal"].clone()
        self.reward_settings["w_lift_base"] = self.reward_settings["w_lift"].clone()
        self.target_pos_z_center = self.reward_settings["target_pos"][:, 2].clone()
        self.target_pos_z_offset_from_table = self.target_pos_z_center - float(self._fixed_table_surface_height)

    def _resample_target_pos(self, env_ids):
        if env_ids is None or env_ids.numel() == 0:
            return
        target_dtype = self.reward_settings["target_pos"].dtype
        target_noise = torch.tensor(
            self.cfg["reward"]["params"]["target_pos_noise"],
            device=self.device,
            dtype=target_dtype,
        )
        target_center = torch.zeros((env_ids.numel(), 3), device=self.device, dtype=target_dtype)
        target_center[:, :2] = self._object_center_init_state[env_ids, :2].to(dtype=target_dtype)
        target_center[:, 2] = (
            self.table_surface_height[env_ids].to(dtype=target_dtype)
            + self.target_pos_z_offset_from_table[env_ids].to(dtype=target_dtype)
        )
        offsets = (torch.rand((env_ids.numel(), 3), device=self.device, dtype=target_dtype) * 2.0 - 1.0) * target_noise.unsqueeze(0)
        self.reward_settings["target_pos"][env_ids] = target_center + offsets
    
    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return
        self._debug_print_reset_dispatch(env_ids)

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        if getattr(self, "_activation_snapshot_bank_loaded", False):
            (
                joint_config,
                left_mask,
                object_center_world,
                object_quat_world,
                table_pos_world,
                table_quat_world,
                target_quat_world,
                debug_payload,
            ) = self._sample_activation_snapshot_bank_entries(env_ids)
            target_quat = (
                target_quat_world
                if target_quat_world is not None
                else self._compose_snapshot_target_quat_world(left_mask, table_quat_world)
            )
            self._snapshot_object_center_nominal[env_ids] = object_center_world.clone()
            joint_config, object_center_world, object_quat_world = self._apply_activation_snapshot_reset_noise(
                joint_config,
                object_center_world,
                object_quat_world,
                table_quat_world,
            )
            debug_payload["joint_config"] = joint_config.clone()
            debug_payload["object_center_replay_world"] = object_center_world.clone()
            debug_payload["object_quat_replay_world"] = object_quat_world.clone()
            self._desired_table_pos_world[env_ids] = table_pos_world.clone()
            self._desired_table_quat_world[env_ids] = table_quat_world.clone()
            self._snapshot_table_pin_buf[env_ids] = int(self._snapshot_table_pin_steps_after_reset)
            self._queue_activation_snapshot_debug(env_ids, debug_payload)
            self._apply_snapshot_scene_state(
                env_ids,
                table_pos_world,
                table_quat_world,
                object_center_world,
                object_quat_world,
            )
            self._debug_print_copy_reset(
                env_ids,
                joint_config,
                object_center_world,
                object_quat_world,
                table_pos_world,
                table_quat_world,
            )
            self._finalize_reset_bookkeeping(
                env_ids,
                joint_config,
                target_quat,
                left_mask,
                object_center_world,
            )
        else:
            joint_config, target_quat, left_mask, object_center_world, object_quat_world = self._sample_reset_from_bank(env_ids)
            self._snapshot_object_center_nominal[env_ids] = object_center_world.clone()
            self._apply_object_center_state(env_ids, object_center_world, object_quat_world)
            self._finalize_reset_bookkeeping(
                env_ids,
                joint_config,
                target_quat,
                left_mask,
                object_center_world,
            )

    def set_viewer(self):
        super().set_viewer(
            pos=[1.5, -1.0, 0.7],
            target=[0.5, 0.0, 0.1],
        )

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        # @ray visualize debugging stuff
        if self.debug_viz and self.viewer is not None:
            self._draw_object_xy_range_grid(clear_lines=True)
            self._draw_wrong_side_proximity_box(clear_lines=False)
        self.compute_reward()
        self._debug_print_all4_object_table()
        self._debug_print_follow_after_reset()
        self._print_activation_snapshot_debug()

        if self.video_logging["capture"]:
            self.video_logger()
        self._snapshot_table_pin_buf = torch.clamp(self._snapshot_table_pin_buf - 1, min=0)
        self._post_reset_grace_buf = torch.clamp(self._post_reset_grace_buf - 1, min=0)
        self._copy_debug_follow_buf = torch.clamp(self._copy_debug_follow_buf - 1, min=0)
        self.sim_steps += 1

    def _create_envs(self, spacing, num_per_row):
        """
        loading Franka + LEAP + a table in the environment, this is for debugging purposes only
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # setup params
        table_thickness = self.cfg["env"]["table_thickness"]
        self.table_thickness = float(table_thickness)
        self.cuboid_dims = []  # xyz
        self.cuboid_pos = []
        self.cuboid_quats = []

        self.mesh_aabb_extents = None  # xyz, axis-aligned bounding box full extents

        # setup robot (franka + leap)
        robot_dof_props = self._create_franka_leap()
        robot_asset = self.robot_asset
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # make sure robot spawns at the origin, this matches the IK setting with cuRobo
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        max_agg_bodies = num_robot_bodies + 1 + 1  # 1 for table, 1 for object
        max_agg_shapes = num_robot_shapes + 1 + 1  # 1 for table, 1 for object

        self.robots = []
        self.tables = []
        self.objects = []
        self.envs = []
        self._object_center_init_state = torch.zeros((self.num_envs, 3), device=self.device)
        self.table_surface_height = torch.zeros((self.num_envs,), device=self.device)

        # load all meshes first
        all_meshes_list = self.create_all_meshes()
        if self._snapshot_variation_assignment_json is not None:
            # Snapshot-driven runs may use a small num_envs debug layout while still
            # referencing arbitrary object ids from the full mesh set. Keep the full
            # object catalog available in that case.
            self.num_objects = len(all_meshes_list)
        else:
            self.num_objects = min(len(all_meshes_list), self.num_envs) # @ray record number of objects for per-object success rate tracking
            all_meshes_list = all_meshes_list[:self.num_objects]
        self.env_object_ids = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
        self.env_variation_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.env_table_surface_height_init = torch.full(
            (self.num_envs,),
            float(self._fixed_table_surface_height),
            dtype=torch.float32,
            device=self.device,
        )
        self.env_table_size_init = self._fixed_table_size.to(device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.snapshot_variation_object_ids = None
        if self._snapshot_variation_assignment_json is not None and self.num_objects > 0:
            (
                self.env_table_surface_height_init,
                self.snapshot_variation_object_ids,
                self.env_variation_ids,
                self.env_table_size_init,
            ) = self._load_snapshot_variation_assignment_json(
                self._snapshot_variation_assignment_json,
                self.num_objects,
            )

        # Create environments
        # @ray tensors on object info should be created here
        for i in tqdm(range(self.num_envs), desc="Creating Envs"):
            # grasp object
            if self.snapshot_variation_object_ids is not None:
                object_idx = int(self.snapshot_variation_object_ids[i].item())
            else:
                object_idx = i % len(all_meshes_list)
            object_asset, object_start_pose, object_scale, object_id, mesh_id = all_meshes_list[object_idx]
            self.env_object_ids[i] = object_idx

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create robot (franka + leap)
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_surface_height_i = float(self.env_table_surface_height_init[i].item())
            table_size_i = self.env_table_size_init[i].detach().cpu().tolist()
            table_asset, table_start_pose = self._create_movable_table_cube(
                pos=[0.5, 0.0, table_surface_height_i - table_size_i[2] / 2],
                size=table_size_i,
            )
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, 1, 0
            )
            self.table_surface_height[i] = table_surface_height_i

            # Create object
            self._object_id = self.gym.create_actor(
                env_ptr, object_asset, object_start_pose, "object", i, 2, 0
            )
            self._object_center_init_state[i, :3] = torch.tensor([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z], device=self.device)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.tables.append(table_actor)
            self.objects.append(self._object_id)

            # Precompute static and object point cloud
            # TODO: now this is hardcoded to current simple settings, need to adapt later
            static_pcd_i = torch.from_numpy(
                _sample_box_surface_np(
                    center=[table_start_pose.p.x, table_start_pose.p.y, table_start_pose.p.z],
                    dims=self.cuboid_dims[-1],
                    quat_xyzw=[table_start_pose.r.x, table_start_pose.r.y, table_start_pose.r.z, table_start_pose.r.w],
                    num_points=self.pcd_spec_dict["num_static_points"],
                )
            ).to(self.device)
            self.static_pcds.append(static_pcd_i)

            object_pcd_i = torch.from_numpy(compute_scene_oracle_pcd(
                num_obstacle_points=self.pcd_spec_dict["num_object_points"],
                mesh_position=np.array([[0.0, 0.0, 0.0]]),
                mesh_scale=np.array([object_scale]),
                mesh_quaternion=np.array([[0.0, 0.0, 0.0, 1.0]]),
                obj_id=np.array([object_id]),
                mesh_id=np.array([mesh_id]),
                meshes_dir=self.mesh_args["mesh_dir"],
            )).to(self.device)
            self.object_pcds.append(object_pcd_i)

        self.cuboid_dims = np.asarray(self.cuboid_dims, dtype=np.float32).reshape(self.num_envs, -1, 3)
        self.cuboid_pos = np.asarray(self.cuboid_pos, dtype=np.float32).reshape(self.num_envs, -1, 3)
        self.cuboid_quats = np.asarray(self.cuboid_quats, dtype=np.float32).reshape(self.num_envs, -1, 4)

        self.table_rigid_body_idx = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        for i in range(self.num_envs):
            rb_names = self.gym.get_actor_rigid_body_names(self.envs[i], self.tables[i])
            rb_index = self.gym.find_actor_rigid_body_index(
                self.envs[i], self.tables[i], rb_names[0], gymapi.DOMAIN_ENV
            )
            self.table_rigid_body_idx[i] = rb_index

        self.cuboid_dims = torch.from_numpy(self.cuboid_dims).to(self.device)
        self.cuboid_pos = torch.from_numpy(self.cuboid_pos).to(self.device)
        self.cuboid_quats = torch.from_numpy(self.cuboid_quats).to(self.device)
        self.table_size = self.cuboid_dims[:, 0, :].clone()

        self.static_pcds = torch.stack(self.static_pcds, dim=0).to(self.device).to(torch.float32) # (num_envs, num_points, 3)
        # Keep the copied RL env consistent with distillation snapshot geometry.
        # Distillation scales object pcds by 0.9 before deriving AABB extents, and the
        # snapshot bank's object_center_world is consistent with that convention.
        self.object_pcds = torch.stack(self.object_pcds, dim=0).to(self.device).to(torch.float32) * 0.9
        self.combined_pcds = torch.cat([self.static_pcds, self.object_pcds], dim=1).to(self.device) # (num_envs, num_static_points + num_object_points, 3)
        self._static_pcd_local = self.static_pcds - self.cuboid_pos[:, 0, :].unsqueeze(1)

        # get mesh AABB (axis-aligned bounding box) extents
        min_xyz = self.object_pcds.min(axis=1).values
        max_xyz = self.object_pcds.max(axis=1).values
        self.mesh_aabb_extents = max_xyz - min_xyz
        self._object_center_init_state[:, 2] += self.mesh_aabb_extents[:, 2] * self.object_center_z_scale

        # Setup data
        actor_num = 1 + 1 + 1  # robot, table, object
        self.init_data(actor_num=actor_num)

    def _update_states(self):
        super()._update_states()
        table_pos_world, _, table_rot_mat, table_rot_mat_t = self._get_table_frame_data()
        object_grasp_target_pos = self._object_state[:, :3].clone()
        local_offset = torch.zeros([self.num_envs, 3], dtype=torch.float, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[:, 2] * self.object_grasp_target_z_scale
        object_rot_mat = quaternion_to_matrix_ig(self._object_state[:, 3:7])
        rotated_offset = torch.matmul(object_rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)
        object_grasp_target_pos += rotated_offset

        eef_rot_mat = quaternion_to_matrix_ig(self._eef_state[:, 3:7])
        eef_rot_mat_t = eef_rot_mat.transpose(1, 2)
        object_grasp_target_to_eef_world = object_grasp_target_pos - self._eef_state[:, :3]
        object_grasp_target_to_eef = torch.matmul(
            eef_rot_mat_t, object_grasp_target_to_eef_world.unsqueeze(-1)
        ).squeeze(-1)
        eef_table_dist = (self._eef_state[:, 2] - self.table_surface_height).unsqueeze(-1)
        hand_grasp_axis = quat_apply(
            self._eef_state[:, 3:7],
            self.hand_grasp_dir_local.unsqueeze(0).repeat(self.num_envs, 1),
        )  # CODEX
        object_z_axis = object_rot_mat[:, :, 2]
        object_z_axis_table = torch.matmul(table_rot_mat_t, object_z_axis.unsqueeze(-1)).squeeze(-1)
        # Mode 1: old orientation (target-quat full point-matching; includes XY guidance).
        hand_eef_pos7_rot = torch.cat([self._eef_state[:, :3], self.reward_settings["target_quat"]], dim=-1)
        hand_target_quat_err = self._get_eef_point_matching_err(
            curent_eef_pos7=self._eef_state[:, :7],
            target_eef_pos7=hand_eef_pos7_rot,
        )
        # Mode 2: canonical axis alignment only (no XY guidance), using hand/object canonical z-axes.
        hand_z_axis = eef_rot_mat[:, :, 2]
        hand_z_axis_table = torch.matmul(table_rot_mat_t, hand_z_axis.unsqueeze(-1)).squeeze(-1)
        target_z_axis = quaternion_to_matrix_ig(self.reward_settings["target_quat"])[:, :, 2]
        hand_target_axis_dot = torch.sum(hand_z_axis * target_z_axis, dim=-1).clamp(-1.0, 1.0)
        hand_canonical_axis_err = 1.0 - torch.abs(hand_target_axis_dot)
        # Mode 3: object-axis alignment using configured hand grasp direction and object z-axis.
        hand_object_axis_dot = torch.sum(hand_grasp_axis * object_z_axis, dim=-1).clamp(-1.0, 1.0)
        hand_object_axis_err = 1.0 - torch.abs(hand_object_axis_dot)
        if self.orientation_reward_mode_id == 0:
            hand_orientation_err = hand_target_quat_err
        elif self.orientation_reward_mode_id == 1:
            hand_orientation_err = hand_canonical_axis_err
        else:
            hand_orientation_err = hand_object_axis_err

        world_z = torch.zeros((self.num_envs, 3), device=self.device, dtype=self._object_state.dtype)
        world_z[:, 2] = 1.0
        object_upright_score = torch.sum(object_z_axis * world_z, dim=-1).clamp(0.0, 1.0)  # CODEX
        eef_rel_object_table = self._world_vectors_to_table_frame(
            self._eef_state[:, :3] - self.states["object_center_pos"],
            table_rot_mat_t=table_rot_mat_t,
        )

        # @ray not just update but also create new keys here
        self.states.update({
            # Table Contact Status, check whether the object is lifted
            "lift": ~self.table_collision,
            "object_grasp_target_pos": object_grasp_target_pos, # @ray reward-only grasp target
            "object_grasp_target_to_eef": object_grasp_target_to_eef, # @ray for policy observation
            "eef_table_dist": eef_table_dist,
            "eef_rel_object_table": eef_rel_object_table,
            "object_z_axis_world": object_z_axis,
            "hand_z_axis_world": hand_z_axis,
            "object_z_axis_table": object_z_axis_table,
            "hand_z_axis_table": hand_z_axis_table,
            "grasp_side_binary": self.side_is_left.float().unsqueeze(-1),
            "object_bbox_extent": self.mesh_aabb_extents,
            "point_matching_err_hand_axis": hand_object_axis_err,  # CODEX
            "point_matching_err_hand_canonical_axis": hand_canonical_axis_err,  # CODEX
            "point_matching_err_hand_targetquat": hand_target_quat_err,  # CODEX
            "point_matching_err_hand": hand_orientation_err,
            "object_upright_score": object_upright_score,  # CODEX
        })

    def check_robot_collision(self):
        super().check_robot_collision()
        # TODO @ray this is fragile as it assumes the table to be the 30th body in the env, need to check this
        table_forces = self.contact_forces[torch.arange(self.num_envs, device=self.device), self.table_rigid_body_idx]
        self.table_collision = torch.any(table_forces.view(self.num_envs, -1) != 0, dim=1)
        # self.table_collision = torch.any(self.contact_forces[:, 30].view(self.num_envs, -1) != 0, dim=1)

    def _draw_object_xy_range_grid(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        table_pos = self.cuboid_pos[env_id, 0].to(dtype=torch.float32)
        table_quat = self.cuboid_quats[env_id, 0].to(dtype=torch.float32)
        if getattr(self, "_activation_snapshot_bank_loaded", False):
            noise_xy = self._activation_snapshot_object_xy_noise.to(device=self.device, dtype=torch.float32)
            nominal_center = self._snapshot_object_center_nominal[env_id].to(dtype=torch.float32)
            nominal_center_table = self._world_points_to_table_frame(
                nominal_center.unsqueeze(0),
                table_pos_world=self.cuboid_pos[env_id:env_id + 1, 0, :],
                table_rot_mat_t=quaternion_to_matrix_ig(self.cuboid_quats[env_id:env_id + 1, 0, :]).transpose(1, 2),
            )[0].to(dtype=torch.float32)
            local_box_center = torch.tensor(
                [
                    float(nominal_center_table[0].item()),
                    float(nominal_center_table[1].item()),
                    0.5 * float(self.cuboid_dims[env_id, 0, 2].item()) + 0.001,
                ],
                device=self.device,
                dtype=torch.float32,
            )
            world_box_center = table_pos + quat_apply(table_quat.unsqueeze(0), local_box_center.unsqueeze(0))[0]
            box_half_extents = torch.tensor(
                [
                    float(noise_xy[0].item()),
                    float(noise_xy[1].item()),
                    0.001,
                ],
                device=self.device,
                dtype=torch.float32,
            )
            self._draw_oriented_wire_box(env_id, world_box_center, box_half_extents, table_quat, [0.0, 1.0, 0.0])
            return

        table_half_xy = 0.5 * self.cuboid_dims[env_id, 0, :2].to(dtype=torch.float32)
        object_center_table = self._world_points_to_table_frame(
            self.states["object_center_pos"][env_id:env_id + 1],
            table_pos_world=self.cuboid_pos[env_id:env_id + 1, 0, :],
            table_rot_mat_t=quaternion_to_matrix_ig(self.cuboid_quats[env_id:env_id + 1, 0, :]).transpose(1, 2),
        )[0].to(dtype=torch.float32)
        is_mirrored = float(object_center_table[1].item()) > 0.0

        canonical_min = self._reset_object_xy_min.to(device=self.device, dtype=torch.float32)
        canonical_max = self._reset_object_xy_max.to(device=self.device, dtype=torch.float32)
        if is_mirrored:
            local_xy_min = torch.tensor(
                [canonical_min[0].item(), -canonical_max[1].item()],
                device=self.device,
                dtype=torch.float32,
            )
            local_xy_max = torch.tensor(
                [canonical_max[0].item(), -canonical_min[1].item()],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            local_xy_min = canonical_min.clone()
            local_xy_max = canonical_max.clone()

        local_xy_min = torch.maximum(local_xy_min, -table_half_xy)
        local_xy_max = torch.minimum(local_xy_max, table_half_xy)
        local_center = 0.5 * (local_xy_min + local_xy_max)
        local_half_extents = 0.5 * (local_xy_max - local_xy_min)
        local_box_center = torch.tensor(
            [
                float(local_center[0].item()),
                float(local_center[1].item()),
                0.5 * float(self.cuboid_dims[env_id, 0, 2].item()) + 0.001,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        world_box_center = table_pos + quat_apply(table_quat.unsqueeze(0), local_box_center.unsqueeze(0))[0]
        box_half_extents = torch.tensor(
            [
                float(local_half_extents[0].item()),
                float(local_half_extents[1].item()),
                0.001,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        self._draw_oriented_wire_box(env_id, world_box_center, box_half_extents, table_quat, [0.0, 1.0, 0.0])

    def _draw_object_center_cross(self, half_extent=0.2, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)

        centers = self.states["object_center_pos"]
        grasp_targets = self.states["object_grasp_target_pos"]
        goal_targets = self.reward_settings["target_pos"]
        thick_eps = 0.006

        centers_np = centers.detach().cpu().numpy()
        grasp_targets_np = grasp_targets.detach().cpu().numpy()
        goal_targets_np = goal_targets.detach().cpu().numpy()

        # helper for thicker debug crosses (add 3 offset copies per axis).
        def add_thick_cross(verts_list, colors_list, p, cross_half, color_rgb):
            px, py, pz = float(p[0]), float(p[1]), float(p[2])
            offsets = ((0.0, 0.0, 0.0), (thick_eps, 0.0, 0.0), (-thick_eps, 0.0, 0.0))
            for ox, oy, oz in offsets:
                # x-axis segment
                verts_list.extend([px - cross_half + ox, py + oy, pz + oz, px + cross_half + ox, py + oy, pz + oz])
                colors_list.extend(color_rgb)
                # y-axis segment
                verts_list.extend([px + ox, py - cross_half + oy, pz + oz, px + ox, py + cross_half + oy, pz + oz])
                colors_list.extend(color_rgb)
                # z-axis segment
                verts_list.extend([px + ox, py + oy, pz - cross_half + oz, px + ox, py + oy, pz + cross_half + oz])
                colors_list.extend(color_rgb)

        for i in range(self.num_envs):
            c = centers_np[i]
            g = grasp_targets_np[i]
            t = goal_targets_np[i]

            verts_flat = []
            colors_flat = []

            # object center cross (red)
            add_thick_cross(verts_flat, colors_flat, c, half_extent, [1.0, 0.0, 0.0])

            # grasp target cross (cyan)
            add_thick_cross(verts_flat, colors_flat, g, half_extent, [0.0, 0.8, 1.0])

            # reward target cross (magenta)
            add_thick_cross(verts_flat, colors_flat, t, half_extent, [1.0, 0.0, 1.0])

            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                len(verts_flat) // 6,
                verts_flat,
                colors_flat,
            )

    def _draw_target_sampling_box(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        target_dtype = self.reward_settings["target_pos"].dtype
        target_noise = torch.tensor(
            self.cfg["reward"]["params"]["target_pos_noise"],
            device=self.device,
            dtype=target_dtype,
        )
        center = torch.zeros((3,), device=self.device, dtype=target_dtype)
        center[:2] = self._object_center_init_state[env_id, :2].to(dtype=target_dtype)
        center[2] = self.target_pos_z_center[env_id].to(dtype=target_dtype)
        limits_min = center - target_noise
        limits_max = center + target_noise
        x_min, y_min, z_min = [float(limits_min[0].item()), float(limits_min[1].item()), float(limits_min[2].item())]
        x_max, y_max, z_max = [float(limits_max[0].item()), float(limits_max[1].item()), float(limits_max[2].item())]
        verts = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_min, z_min], [x_max, y_max, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_max, z_min], [x_min, y_min, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_min, z_max], [x_max, y_max, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
            [x_min, y_max, z_max], [x_min, y_min, z_max],
            [x_min, y_min, z_min], [x_min, y_min, z_max],
            [x_max, y_min, z_min], [x_max, y_min, z_max],
            [x_max, y_max, z_min], [x_max, y_max, z_max],
            [x_min, y_max, z_min], [x_min, y_max, z_max],
        ]
        verts_flat = [v for seg in verts for v in seg]
        colors_flat = [1.0, 0.0, 1.0] * (len(verts_flat) // 6)
        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts_flat) // 6, verts_flat, colors_flat)

    def _draw_wire_box(self, env_id, limits_min, limits_max, color_rgb):
        x_min, y_min, z_min = [float(limits_min[0].item()), float(limits_min[1].item()), float(limits_min[2].item())]
        x_max, y_max, z_max = [float(limits_max[0].item()), float(limits_max[1].item()), float(limits_max[2].item())]
        verts = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_min, z_min], [x_max, y_max, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_max, z_min], [x_min, y_min, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_min, z_max], [x_max, y_max, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
            [x_min, y_max, z_max], [x_min, y_min, z_max],
            [x_min, y_min, z_min], [x_min, y_min, z_max],
            [x_max, y_min, z_min], [x_max, y_min, z_max],
            [x_max, y_max, z_min], [x_max, y_max, z_max],
            [x_min, y_max, z_min], [x_min, y_max, z_max],
        ]
        verts_flat = [v for seg in verts for v in seg]
        colors_flat = list(color_rgb) * (len(verts_flat) // 6)
        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts_flat) // 6, verts_flat, colors_flat)

    def _draw_oriented_wire_box(self, env_id, center, half_extents, quat, color_rgb):
        local_corners = torch.tensor(
            [
                [-1.0, -1.0, -1.0],
                [ 1.0, -1.0, -1.0],
                [ 1.0,  1.0, -1.0],
                [-1.0,  1.0, -1.0],
                [-1.0, -1.0,  1.0],
                [ 1.0, -1.0,  1.0],
                [ 1.0,  1.0,  1.0],
                [-1.0,  1.0,  1.0],
            ],
            device=self.device,
            dtype=torch.float32,
        ) * half_extents.unsqueeze(0)
        world_corners = quat_apply(
            quat.unsqueeze(0).repeat(local_corners.shape[0], 1),
            local_corners,
        ) + center.unsqueeze(0)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        verts_flat = []
        for i0, i1 in edges:
            p0 = world_corners[i0]
            p1 = world_corners[i1]
            verts_flat.extend([
                float(p0[0].item()), float(p0[1].item()), float(p0[2].item()),
                float(p1[0].item()), float(p1[1].item()), float(p1[2].item()),
            ])
        colors_flat = list(color_rgb) * len(edges)
        self.gym.add_lines(self.viewer, self.envs[env_id], len(edges), verts_flat, colors_flat)

    def _mirror_box_y_about_table(self, limits_min, limits_max, table_center_y):
        mirrored_min = limits_min.clone()
        mirrored_max = limits_max.clone()
        mirrored_min[1] = 2.0 * table_center_y - limits_max[1]
        mirrored_max[1] = 2.0 * table_center_y - limits_min[1]
        return mirrored_min, mirrored_max

    def _draw_wrong_side_proximity_box(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        center = self.states["object_center_pos"][env_id].to(dtype=torch.float32)
        quat = self._object_state[env_id, 3:7].to(dtype=torch.float32)
        half_extents = 0.5 * self.mesh_aabb_extents[env_id].to(dtype=torch.float32)
        threshold = float(self.reward_settings["wrong_side_proximity_threshold"].item())
        expanded_half_extents = half_extents + threshold
        self._draw_oriented_wire_box(env_id, center, half_extents, quat, [0.9, 0.9, 0.9])
        self._draw_oriented_wire_box(env_id, center, expanded_half_extents, quat, [1.0, 0.35, 0.0])

    def _draw_reset_eef_range_box(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        center = self.states["object_center_pos"][env_id].to(dtype=torch.float32)
        table_center_y = float(self.cuboid_pos[env_id, 0, 1].item())
        table_center_y_t = torch.tensor(table_center_y, device=self.device, dtype=torch.float32)
        is_mirrored = float(center[1].item()) > table_center_y
        canonical_center = center.clone()
        if is_mirrored:
            canonical_center[1] = 2.0 * table_center_y_t - center[1]
        grasp_side_sign = 1.0 if bool(self.side_is_left[env_id].item()) else -1.0
        rel_min_raw = self._reset_eef_rel_object_min.to(device=self.device, dtype=torch.float32)
        rel_max_raw = self._reset_eef_rel_object_max.to(device=self.device, dtype=torch.float32)
        x_min, x_max = rel_min_raw[0], rel_max_raw[0]
        y_abs_min, y_abs_max = rel_min_raw[1], rel_max_raw[1]
        z_min, z_max = rel_min_raw[2], rel_max_raw[2]
        if grasp_side_sign > 0.0:
            rel_min = torch.tensor([x_min.item(), y_abs_min.item(), z_min.item()], device=self.device, dtype=torch.float32)
            rel_max = torch.tensor([x_max.item(), y_abs_max.item(), z_max.item()], device=self.device, dtype=torch.float32)
        else:
            rel_min = torch.tensor([x_min.item(), -y_abs_max.item(), z_min.item()], device=self.device, dtype=torch.float32)
            rel_max = torch.tensor([x_max.item(), -y_abs_min.item(), z_max.item()], device=self.device, dtype=torch.float32)
        canonical_limits_min = canonical_center + rel_min
        canonical_limits_max = canonical_center + rel_max
        mirrored_limits_min, mirrored_limits_max = self._mirror_box_y_about_table(
            canonical_limits_min, canonical_limits_max, table_center_y_t
        )
        self._draw_wire_box(env_id, canonical_limits_min, canonical_limits_max, [0.35, 0.55, 1.0])
        self._draw_wire_box(env_id, mirrored_limits_min, mirrored_limits_max, [0.0, 0.7, 1.0])

    def _draw_hand_init_regions(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        center = self.states["object_center_pos"][env_id].to(dtype=torch.float32)
        table_center_y = float(self.cuboid_pos[env_id, 0, 1].item())
        table_center_y_t = torch.tensor(table_center_y, device=self.device, dtype=torch.float32)
        is_mirrored = float(center[1].item()) > table_center_y
        canonical_center = center.clone()
        if is_mirrored:
            canonical_center[1] = 2.0 * table_center_y_t - center[1]
        grasp_side_sign = 1.0 if bool(self.side_is_left[env_id].item()) else -1.0
        rel_min_raw = self._reset_eef_rel_object_min.to(device=self.device, dtype=torch.float32)
        rel_max_raw = self._reset_eef_rel_object_max.to(device=self.device, dtype=torch.float32)

        x_min, x_max = rel_min_raw[0], rel_max_raw[0]
        y_abs_min, y_abs_max = rel_min_raw[1], rel_max_raw[1]
        z_min, z_max = rel_min_raw[2], rel_max_raw[2]

        left_grasp_rel_min = torch.tensor(
            [x_min.item(), y_abs_min.item(), z_min.item()],
            device=self.device,
            dtype=torch.float32,
        )
        left_grasp_rel_max = torch.tensor(
            [x_max.item(), y_abs_max.item(), z_max.item()],
            device=self.device,
            dtype=torch.float32,
        )
        if grasp_side_sign > 0.0:
            box_rel_min, box_rel_max = left_grasp_rel_min, left_grasp_rel_max
        else:
            box_rel_min = torch.tensor(
                [x_min.item(), -y_abs_max.item(), z_min.item()],
                device=self.device,
                dtype=torch.float32,
            )
            box_rel_max = torch.tensor(
                [x_max.item(), -y_abs_min.item(), z_max.item()],
                device=self.device,
                dtype=torch.float32,
            )

        canonical_box_min = canonical_center + box_rel_min
        canonical_box_max = canonical_center + box_rel_max
        mirrored_box_min, mirrored_box_max = self._mirror_box_y_about_table(
            canonical_box_min, canonical_box_max, table_center_y_t
        )

        self._draw_wire_box(env_id, canonical_box_min, canonical_box_max, [0.35, 0.55, 1.0])
        self._draw_wire_box(env_id, mirrored_box_min, mirrored_box_max, [0.0, 0.7, 1.0])

        split_offset = float(self.reward_settings["grasp_side_split_y_offset"].item())
        sep_x_min = float((canonical_center[0] + x_min).item())
        sep_x_max = float((canonical_center[0] + x_max).item())
        sep_z_min = float((canonical_center[2] + z_min).item())
        sep_z_max = float((canonical_center[2] + z_max).item())
        canonical_sep_y = float(canonical_center[1].item() + grasp_side_sign * split_offset)
        mirrored_sep_y = float(2.0 * table_center_y - canonical_sep_y)

        canonical_sep_verts = [
            sep_x_min, canonical_sep_y, sep_z_min, sep_x_max, canonical_sep_y, sep_z_min,
            sep_x_max, canonical_sep_y, sep_z_min, sep_x_max, canonical_sep_y, sep_z_max,
            sep_x_max, canonical_sep_y, sep_z_max, sep_x_min, canonical_sep_y, sep_z_max,
            sep_x_min, canonical_sep_y, sep_z_max, sep_x_min, canonical_sep_y, sep_z_min,
        ]
        mirrored_sep_verts = [
            sep_x_min, mirrored_sep_y, sep_z_min, sep_x_max, mirrored_sep_y, sep_z_min,
            sep_x_max, mirrored_sep_y, sep_z_min, sep_x_max, mirrored_sep_y, sep_z_max,
            sep_x_max, mirrored_sep_y, sep_z_max, sep_x_min, mirrored_sep_y, sep_z_max,
            sep_x_min, mirrored_sep_y, sep_z_max, sep_x_min, mirrored_sep_y, sep_z_min,
        ]
        self.gym.add_lines(
            self.viewer,
            self.envs[env_id],
            4,
            canonical_sep_verts,
            [0.85, 0.85, 0.25] * 4,
        )
        self.gym.add_lines(
            self.viewer,
            self.envs[env_id],
            4,
            mirrored_sep_verts,
            [1.0, 1.0, 0.0] * 4,
        )

        table_x_min = float((self.cuboid_pos[env_id, 0, 0] - 0.5 * self.cuboid_dims[env_id, 0, 0]).item())
        table_x_max = float((self.cuboid_pos[env_id, 0, 0] + 0.5 * self.cuboid_dims[env_id, 0, 0]).item())
        table_z = float((self.table_surface_height[env_id] + 0.02).item())
        table_center_line = [
            table_x_min, table_center_y, table_z, table_x_max, table_center_y, table_z,
        ]
        self.gym.add_lines(
            self.viewer,
            self.envs[env_id],
            1,
            table_center_line,
            [1.0, 1.0, 1.0],
        )

    def _draw_grasp_direction_line(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        line_len = 0.12  # CODEX
        eef_pos = self._eef_state[env_id, :3]
        object_pos = self.states["object_center_pos"][env_id]

        hand_dir = quat_apply(
            self._eef_state[env_id:env_id + 1, 3:7],
            self.hand_grasp_dir_local.unsqueeze(0),
        )[0]
        hand_dir = hand_dir / torch.norm(hand_dir).clamp_min(1.0e-8)

        object_z = quat_apply(
            self._object_state[env_id:env_id + 1, 3:7],
            torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=eef_pos.dtype),
        )[0]
        object_z = object_z / torch.norm(object_z).clamp_min(1.0e-8)

        p0 = eef_pos
        p1 = eef_pos + line_len * hand_dir
        q0 = object_pos
        q1 = object_pos + line_len * object_z
        verts_flat = [
            float(p0[0]), float(p0[1]), float(p0[2]), float(p1[0]), float(p1[1]), float(p1[2]),
            float(q0[0]), float(q0[1]), float(q0[2]), float(q1[0]), float(q1[1]), float(q1[2]),
        ]
        colors_flat = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]  # hand-dir cyan, object-z yellow
        self.gym.add_lines(self.viewer, self.envs[env_id], 2, verts_flat, colors_flat)

    def compute_observations(self):
        self._refresh() # @ray checks table collision and updates states

        obs_components = ["q_hand",
                          "grasp_side_binary",
                          "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                          "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                          "object_to_eef", "object_to_eef_rot_6d",
                          "object_grasp_target_to_eef",
                          "eef_table_dist",
                          "object_z_axis_table", "hand_z_axis_table",
                          "target_to_eef", "target_to_eef_rot_6d"]

        states_components = ["q", "qd",
                             "grasp_side_binary",
                             "eef_pos", "eef_rot_6d", "eef_vel",
                             "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                             "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                             "object_to_eef", "object_to_eef_rot_6d",
                             "object_grasp_target_to_eef",
                             "eef_table_dist",
                             "object_z_axis_table", "hand_z_axis_table",
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

        return self.obs_buf

    def compute_reward(self):
        # @ray states used are updated in compute_observations(), called right before compute_reward()

        self.reset_buf[:] = torch.where((self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)
        allow_terminal_reset = self._post_reset_grace_buf <= 0
        below_table = self.states['object_center_pos'][:, 2] < self.table_surface_height - 0.1
        self.reset_buf[allow_terminal_reset & below_table] = 1
        debug_below_env_ids = (allow_terminal_reset & below_table).nonzero(as_tuple=False).squeeze(-1)
        if debug_below_env_ids.numel() > 0:
            self._debug_print_copy_done_reason("below_table", debug_below_env_ids, below_table=below_table)
        # CODEX: reset scene if object drifts too far in XY from table center (fly-away guard).
        table_center_xy = self.cuboid_pos[:, 0, :2]
        object_center_xy = self.states["object_center_pos"][:, :2]
        dist_to_table_center_xy = torch.norm(object_center_xy - table_center_xy, dim=-1)
        max_xy_dist = 5.0
        self.reset_buf[allow_terminal_reset & (dist_to_table_center_xy > max_xy_dist)] = 1
        debug_far_env_ids = (allow_terminal_reset & (dist_to_table_center_xy > max_xy_dist)).nonzero(as_tuple=False).squeeze(-1)
        if debug_far_env_ids.numel() > 0:
            self._debug_print_copy_done_reason(
                "xy_drift",
                debug_far_env_ids,
                below_table=below_table,
                dist_to_table_center_xy=dist_to_table_center_xy,
            )

        # Lift/goal curriculum (hardcoded here by request; edit these values directly).
        # Curriculum with delayed start:
        # 1) hold start scale for `curri_delay_steps`
        # 2) linearly ramp for `curri_steps`
        curri_enable = False
        curri_start_scale = 1.0
        curri_end_scale = 10.0
        curri_delay_steps = 50_000
        curri_steps = 200_000
        if curri_enable:
            curri_progress_steps = max(float(self.sim_steps) - float(curri_delay_steps), 0.0)
            curri_t = min(curri_progress_steps / float(max(curri_steps, 1)), 1.0)
            curri_scale = curri_start_scale + (curri_end_scale - curri_start_scale) * curri_t
        else:
            curri_scale = 1.0
        self.reward_settings["w_obj_goal"] = self.reward_settings["w_obj_goal_base"] * curri_scale
        self.reward_settings["w_lift"] = self.reward_settings["w_lift_base"] * curri_scale
        reward_dict = compute_franka_leap_reward(self.states, self.reward_settings)
        # Keep the target fixed within an episode. Only reset-time sampling changes it.

        self.rew_buf[:] = reward_dict["r_total"]
        self.extras["sep_reward/r_hand_obj"] = torch.mean(reward_dict["r_hand_obj"]).item()
        self.extras["sep_reward/r_obj_goal"] = torch.mean(reward_dict["r_obj_goal"]).item()
        self.extras["sep_reward/r_goal_align"] = torch.mean(reward_dict["r_goal_align"]).item()
        self.extras["sep_reward/r_recovery_region"] = torch.mean(reward_dict["r_recovery_region"]).item()
        self.extras["sep_reward/r_wrong_side_proximity"] = torch.mean(reward_dict["r_wrong_side_proximity"]).item()
        self.extras["metrics/wrong_side_forbidden_box_rate"] = torch.mean(
            reward_dict["inside_wrong_side_forbidden_box"].float()
        ).item()
        self.extras["sep_reward/r_hand_rot"] = torch.mean(reward_dict["r_hand_rot"]).item()
        self.extras["sep_reward/r_lift"] = torch.mean(reward_dict["r_lift"]).item()
        self.extras["sep_reward/r_curl"] = torch.mean(reward_dict["r_curl"]).item()
        self.extras["sep_reward/r_success_bonus"] = torch.mean(reward_dict["r_success_bonus"]).item()
        self.extras["sep_reward/r_actionreg"] = torch.mean(reward_dict["r_actionreg"]).item()
        self.extras["dis/d_hand_obj"] = torch.mean(reward_dict["d_hand_obj"]).item()
        self.extras["dis/d_lift"] = torch.mean(reward_dict["d_lift"]).item()
        self.extras["dis/d_eef_point_goal"] = torch.mean(reward_dict["d_eef_point_goal"]).item()
        self.extras["dis/d_eef_point_goal_rot"] = torch.mean(reward_dict["d_eef_point_goal_rot"]).item()
        self.extras["dis/d_goal_align"] = torch.mean(reward_dict["d_goal_align"]).item()
        self.extras["dis/d_recovery_region"] = torch.mean(reward_dict["d_recovery_region"]).item()

        # log metrics
        self.lifting_5cm_per_step = self.states["lift"]
        self.lifting_flags_instant[self.lifting_5cm_per_step] = 1
        self.success_5cm_per_step = (reward_dict["d_eef_point_goal"] < 0.05) & self.lifting_5cm_per_step
        self.success_flags_instant[self.success_5cm_per_step] = 1
        self.success_duration = torch.where(
            self.success_5cm_per_step,
            self.success_duration + self.dt,
            torch.zeros_like(self.success_duration),
        )
        self.lifting_duration = torch.where(
            self.lifting_5cm_per_step,
            self.lifting_duration + self.dt,
            torch.zeros_like(self.lifting_duration),
        )
        success_reached_now = self.success_duration >= self.reward_settings["success_timeout"]
        lifting_reached_now = self.lifting_duration >= self.reward_settings["lifting_timeout"]
        newly_success_long_enough = (~self.success_long_enough) & success_reached_now
        self.success_long_enough = self.success_long_enough | success_reached_now
        self.lifting_long_enough = self.lifting_long_enough | lifting_reached_now
        done_envs = self.reset_buf > 0

        if bool(self.cfg["reward"]["params"].get("resample_target_on_success", False)):
            resample_env_ids = (newly_success_long_enough & (~done_envs)).nonzero(as_tuple=False).squeeze(-1)
            if resample_env_ids.numel() > 0:
                self._resample_target_pos(resample_env_ids)
                # Start measuring success duration against the new target on the next step.
                self.success_duration[resample_env_ids] = 0.0

        if torch.any(done_envs):
            done_env_ids = done_envs.nonzero(as_tuple=False).squeeze(-1)
            done_object_ids = self.env_object_ids[done_env_ids]
            episode_increments = torch.bincount(done_object_ids, minlength=self.num_objects)
            success_env_ids = (done_envs & self.success_long_enough).nonzero(as_tuple=False).squeeze(-1)
            success_object_ids = self.env_object_ids[success_env_ids]
            success_increments = torch.bincount(success_object_ids, minlength=self.num_objects)
            self.per_object_episode_counts += episode_increments
            self.per_object_success_counts += success_increments
            self.per_object_episode_counts_interval += episode_increments
            self.per_object_success_counts_interval += success_increments

        # @ray log per-object per-interval success rates locally and a histograom to wandb
        if self.sim_steps > 0 and (self.sim_steps % self.log_per_object_success_freq == 0):
            # @ray prevent inf from division by zero if some objects are not in any envs
            interval_rates = torch.where(
                self.per_object_episode_counts_interval > 0,
                self.per_object_success_counts_interval.float() / self.per_object_episode_counts_interval.float(),
                torch.zeros_like(self.per_object_success_counts_interval, dtype=torch.float32),
            )
            if wandb.run is not None:
                print("logging per-object success rate histogram to wandb")
                wandb.log({"per_object_success_rate_hist": wandb.Histogram(interval_rates.detach().cpu().numpy(), num_bins=20)},)
            interval_snapshot = {
                "sim_steps": int(self.sim_steps),
                "log_interval_steps": int(self.log_per_object_success_freq),
                "per_object_success_rates": {
                    str(obj_id): {
                        "episodes": int(self.per_object_episode_counts_interval[obj_id].item()),
                        "successes": int(self.per_object_success_counts_interval[obj_id].item()),
                        "success_rate": float(interval_rates[obj_id].item()),
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

        episode_success_for_metrics = torch.maximum(self.success_flags, self.success_long_enough.float())
        episode_lifting_for_metrics = torch.maximum(self.lifting_flags, self.lifting_long_enough.float())
        self.extras["metrics/success_rate_5cm_per_ep"] = torch.mean(episode_success_for_metrics).item()
        self.extras["metrics/success_rate_5cm_per_ep_instant"] = torch.mean(self.success_flags_instant).item()
        self.extras["metrics/success_rate_5cm_per_step"] = torch.mean(self.success_5cm_per_step.float()).item()
        self.extras["metrics/lifting_rate_5cm_per_ep"] = torch.mean(episode_lifting_for_metrics).item()
        self.extras["metrics/lifting_rate_5cm_per_ep_instant"] = torch.mean(self.lifting_flags_instant).item()
        self.extras["metrics/lifting_rate_5cm_per_step"] = torch.mean(self.lifting_5cm_per_step.float()).item()
        self._update_object_wrench_success_curriculum(self.extras["metrics/success_rate_5cm_per_ep"])
        self._update_object_teleport_success_curriculum(self.extras["metrics/success_rate_5cm_per_ep"])
        self.extras["object_wrench/curriculum_stage"] = float(self.object_wrench_curriculum_stage)
        self.extras["object_wrench/curriculum_scale"] = float(self.object_wrench_curriculum_scale)
        self.extras["object_teleport/curriculum_stage"] = float(self.object_teleport_curriculum_stage)
        self.extras["object_teleport/curriculum_scale"] = float(self.object_teleport_curriculum_scale)

        # CODEX: success-based right-section curriculum update + logging.
        # log memory usage TODO: debug utils, cleanup later
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
    d_hand_obj_max = torch.stack([d_palm, d_finger1, d_finger2, d_finger3, d_finger4], dim=1)
    d_hand_obj_max = torch.max(d_hand_obj_max, dim=1)[0]

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
    hand_positions = torch.stack(
        [
            states["eef_wrist_pos"],
            states["eef_pos"],
            states["eef_finger1_pos"],
            states["eef_finger2_pos"],
            states["eef_finger3_pos"],
            states["eef_finger4_pos"],
        ],
        dim=1,
    )

    inside_wrong_side_forbidden_box = torch.zeros_like(states["lift"], dtype=torch.bool)
    if bool(reward_settings["wrong_side_forbidden_box_enable"]):
        object_quat_conj = quat_conjugate(states["object_quat"])
        object_bbox_half_extents = 0.5 * states["object_bbox_extent"]
        expanded_bbox_half_extents = object_bbox_half_extents + reward_settings["wrong_side_proximity_threshold"].unsqueeze(-1)
        hand_rel_object_world = hand_positions - states["object_center_pos"].unsqueeze(1)
        object_quat_conj_rep = object_quat_conj.unsqueeze(1).repeat(1, hand_positions.shape[1], 1).reshape(-1, 4)
        hand_rel_object_local = quat_apply(
            object_quat_conj_rep,
            hand_rel_object_world.reshape(-1, 3),
        ).view(n_env, hand_positions.shape[1], 3)
        inside_wrong_side_forbidden_box = torch.any(
            torch.all(torch.abs(hand_rel_object_local) <= expanded_bbox_half_extents.unsqueeze(1), dim=-1),
            dim=1,
        )

    # CODEX: Optional hand-object branch: fingertip-only height, palm+fingertip radial distance.
    use_midheight_xy = bool(reward_settings["hand_obj_use_midheight_xy"])
    if use_midheight_xy:
        object_target_pos = states["object_grasp_target_pos"]
        finger_to_target = finger_positions - object_target_pos.unsqueeze(1)
        finger_signed_height = torch.sum(finger_to_target * object_z_axis_world.unsqueeze(1), dim=-1)
        non_thumb_signed_height = finger_signed_height[:, :3]
        thumb_signed_height = finger_signed_height[:, 3]
        mid_non_thumb_height = 0.5 * (
            torch.max(non_thumb_signed_height, dim=1)[0] + torch.min(non_thumb_signed_height, dim=1)[0]
        )
        # CODEX: tangential (radial-to-axis) component in object frame.
        hand_to_target = hand_positions - object_target_pos.unsqueeze(1)
        hand_radial = hand_to_target - torch.sum(
            hand_to_target * object_z_axis_world.unsqueeze(1), dim=-1, keepdim=True
        ) * object_z_axis_world.unsqueeze(1)
        d_xy = torch.max(torch.norm(hand_radial, dim=-1), dim=1)[0]
        # CODEX: thumb gets its own height term so it cannot "cheat" the shared
        # midpoint target for the other three fingers.
        d_height = 0.75 * torch.abs(mid_non_thumb_height) + 0.25 * torch.abs(thumb_signed_height)
        d_hand_obj = d_xy + reward_settings["hand_obj_midheight_height_weight"] * d_height
    else:
        d_hand_obj = d_hand_obj_max

    # R1: Hand object distance reward
    beta_hand_object = reward_settings["beta_hand_object"]
    r_hand_obj = torch.exp(-beta_hand_object * d_hand_obj)

    # Pre-lift side disambiguation:
    # - on the configured grasp side, keep the normal grasping reward
    # - on the opposite side, replace it with a small recovery penalty that
    #   grows with distance from the grasp-side region
    eef_rel_object = states["eef_rel_object_table"]
    grasp_side_binary = states["grasp_side_binary"].squeeze(-1) > 0.5
    grasp_side_sign = torch.where(
        grasp_side_binary,
        torch.ones_like(eef_rel_object[:, 1]),
        -torch.ones_like(eef_rel_object[:, 1]),
    )
    grasp_side_split_y_offset = reward_settings["grasp_side_split_y_offset"]
    signed_y_to_grasp_side = grasp_side_sign * eef_rel_object[:, 1]
    on_grasp_side = signed_y_to_grasp_side >= grasp_side_split_y_offset
    prelift_recovery_mode = (~states["lift"]) & (~on_grasp_side)
    d_recovery_region = torch.clamp(grasp_side_split_y_offset - signed_y_to_grasp_side, min=0.0)
    beta_recovery_region = reward_settings["beta_recovery_region"]
    r_recovery_region = -(1.0 - torch.exp(-beta_recovery_region * d_recovery_region))
    r_recovery_region = torch.where(prelift_recovery_mode, r_recovery_region, torch.zeros_like(r_recovery_region))
    wrong_side_proximity_penalty = reward_settings["wrong_side_proximity_penalty"]
    if bool(reward_settings["wrong_side_forbidden_box_enable"]):
        r_wrong_side_proximity = torch.where(
            prelift_recovery_mode & inside_wrong_side_forbidden_box,
            -wrong_side_proximity_penalty * torch.ones_like(d_hand_obj),
            torch.zeros_like(d_hand_obj),
        )
    else:
        wrong_side_proximity_threshold = reward_settings["wrong_side_proximity_threshold"]
        d_wrong_side_proximity = torch.clamp(wrong_side_proximity_threshold - d_hand_obj, min=0.0)
        r_wrong_side_proximity = torch.where(
            prelift_recovery_mode & (d_wrong_side_proximity > 0.0),
            -wrong_side_proximity_penalty * torch.ones_like(d_wrong_side_proximity),
            torch.zeros_like(d_wrong_side_proximity),
        )

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

    r_lift = r_lift

    # R3: Object goal distance reward (based on average point matching distance)
    d_eef_point_goal_target = states["point_matching_err_target"]
    beta_object_goal = reward_settings["beta_object_goal"]
    d_goal_align = states["point_matching_err_hand_axis"]
    beta_goal_align = reward_settings["beta_goal_align"]
    goal_align_gate_floor = reward_settings["goal_align_gate_floor"]
    r_goal_align = torch.exp(-beta_goal_align * d_goal_align)
    r_goal_align = torch.where(states["lift"], r_goal_align, torch.zeros_like(r_goal_align))
    goal_align_gate = goal_align_gate_floor + (1.0 - goal_align_gate_floor) * r_goal_align
    r_obj_goal = torch.exp(-beta_object_goal * d_eef_point_goal_target)
    r_obj_goal = torch.where(states["lift"], r_obj_goal * goal_align_gate, 0.0)

    # R4: Hand orientation reward (based on average point matching distance)
    d_eef_point_goal_hand = states["point_matching_err_hand"]
    beta_hand_orientation = reward_settings["beta_hand_orientation"]
    r_hand_orientation = torch.exp(-beta_hand_orientation * d_eef_point_goal_hand)

    # R5: Finger curl
    hand_dof_pos = states["q"][:, 7:] # hand joint angles
    near_object = (d_hand_obj <= reward_settings["curl_reaching_threshold"])
    dof_err = hand_dof_pos - reward_settings["grasp_finger_dof_pos"]
    finger_pos_diff = torch.sum((dof_err ** 2) * reward_settings["curl_dof_weight"], dim=1)

    beta_curl = reward_settings["beta_curl"]
    r_curl= torch.exp(-beta_curl * finger_pos_diff)
    r_curl = torch.where(near_object, r_curl, 0.0)

    r_hand_obj = torch.where(prelift_recovery_mode, torch.zeros_like(r_hand_obj), r_hand_obj)
    r_hand_orientation = torch.where(prelift_recovery_mode, torch.zeros_like(r_hand_orientation), r_hand_orientation)
    r_curl = torch.where(prelift_recovery_mode, torch.zeros_like(r_curl), r_curl)

    # R6: Velocity Regularization/Penalty
    actionreg = states["actionreg"]
    r_actionreg = torch.sum(actionreg**2, dim=-1)

    w_hand_obj = reward_settings["w_hand_obj"]
    w_obj_goal = reward_settings["w_obj_goal"]
    w_goal_align = reward_settings["w_goal_align"]
    w_recovery_region = reward_settings["w_recovery_region"]
    w_hand_orientation = reward_settings["w_hand_orientation"]
    w_lift = reward_settings["w_lift"]
    w_curl = reward_settings["w_curl"]
    w_success_bonus = reward_settings["w_success_bonus"]
    w_actionreg = reward_settings["w_actionreg"]


    use_curl = bool(reward_settings["use_curl"])
    # @ray 
    # use activated rewards only
    # but compute all rewards anyways for logging
    # @ray using the same weight for obj_goal position and rotation, can be changed later if needed
    r_total =w_hand_obj*r_hand_obj + w_obj_goal*r_obj_goal + w_goal_align*r_goal_align + w_recovery_region*r_recovery_region + \
              r_wrong_side_proximity + w_hand_orientation*r_hand_orientation + \
              w_curl*r_curl * float(use_curl) + \
              w_lift*r_lift + w_actionreg*r_actionreg

    success_bonus_threshold = reward_settings["success_bonus_threshold"]
    success_region = states["lift"] & (d_eef_point_goal_target < success_bonus_threshold) & (r_goal_align > 0.5)
    r_success_bonus = torch.where(
        success_region,
        torch.ones_like(r_total) * w_success_bonus,
        torch.zeros_like(r_total),
    )
    r_total = r_total + r_success_bonus
    
    rewards = {
        "r_hand_obj": w_hand_obj*r_hand_obj,
        "r_lift": w_lift*r_lift,
        "r_obj_goal": w_obj_goal*r_obj_goal,
        "r_goal_align": w_goal_align*r_goal_align,
        "r_recovery_region": w_recovery_region*r_recovery_region,
        "r_wrong_side_proximity": r_wrong_side_proximity,
        "r_hand_rot": w_hand_orientation*r_hand_orientation,
        "r_curl": w_curl*r_curl,
        "r_success_bonus": r_success_bonus,
        "r_actionreg": w_actionreg*r_actionreg,
        "r_total": r_total,
        "inside_wrong_side_forbidden_box": inside_wrong_side_forbidden_box.float(),
        "d_hand_obj": d_hand_obj,
        "d_lift": object_height,
        "d_eef_point_goal": d_eef_point_goal_target,
        "d_eef_point_goal_rot": d_eef_point_goal_hand,
        "d_goal_align": d_goal_align,
        "d_recovery_region": d_recovery_region,
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
    env = FrankaLEAPPickTableSide(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    for i in tqdm(range(1000)):
        t1 = time.time()
        env.reset_idx()
        # env.set_robot_joint_state(env.canonical_joint_config)
        env.set_robot_joint_state(env.canonical_grasp_config)
        env.step_sim_multi(1, False)
        env.compute_observations()

        # test fk, ik # need to set eef to panda_link7, otherwise will have offset
        ee_pos = env.get_ee_from_joint(env.states['q'][:, :7])
        fk_pos_err = torch.any((ee_pos[:, :3] - env.states['eef_pos']) > 1e-4)
        fk_ori_err1 = (ee_pos[:, 3:] - env.states['eef_quat']) > 1e-4
        fk_ori_err2 = (ee_pos[:, 3:] + env.states['eef_quat']) > 1e-4
        fk_ori_err = torch.any(fk_ori_err1 & fk_ori_err2)
        print(f"FK pos error: {fk_pos_err}, FK ori error: {fk_ori_err}")

        q_config = env.get_joint_from_ee(ee_pos)
        ee_pos_resolve = env.get_ee_from_joint(q_config)
        ik_pos_err = torch.any((ee_pos_resolve[:, :3] - env.states['eef_pos']) > 1e-4)
        ik_quat_err1 = (ee_pos_resolve[:, 3:] - env.states['eef_quat']) > 1e-4
        ik_quat_err2 = (ee_pos_resolve[:, 3:] + env.states['eef_quat']) > 1e-4
        ik_quat_err = torch.any(ik_quat_err1 & ik_quat_err2)
        print(f"IK pos error: {ik_pos_err}, IK ori error: {ik_quat_err}")

        import ipdb ; ipdb.set_trace()
        t2 = time.time()
        print(f"Reset time: {t2 - t1}")
        env.render()


if __name__ == "__main__":
    launch_test()
