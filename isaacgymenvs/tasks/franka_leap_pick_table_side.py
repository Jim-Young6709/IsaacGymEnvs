"""
Franka + LEAP Hand Pick Env
"""
import ast
import time
import json
import os
import random
from collections.abc import Sequence

import hydra
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


def _cfg_to_float_array(value, name="value"):
    def _normalize(v):
        if isinstance(v, str):
            s = v.strip()
            if len(s) == 0:
                raise ValueError(f"Empty string in {name}")
            if s[0] in "[(" and s[-1] in "])":
                return _normalize(ast.literal_eval(s))
            if "=" in s:
                raise ValueError(
                    f"Invalid numeric token {s!r} in {name}. "
                    "This usually means the Hydra override string is malformed."
                )
            return float(s)
        if isinstance(v, np.ndarray):
            return _normalize(v.tolist())
        if torch.is_tensor(v):
            return _normalize(v.detach().cpu().tolist())
        if isinstance(v, Sequence) and not isinstance(v, (bytes, bytearray)):
            return [_normalize(x) for x in v]
        return v

    try:
        return np.asarray(_normalize(value), dtype=np.float32)
    except Exception as exc:
        raise ValueError(
            f"Could not parse {name}={value!r} as a float array. "
            'Expected syntax like "[[0.45, -0.10, 0.2], [0.62, 0.10, 0.25]]".'
        ) from exc


class FrankaLEAPPickTableSide(FrankaLEAP):
    def _top_long_mode(self):
        return self.lie_flat_prob >= 1.0 - 1.0e-6

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.object_grasp_target_z_scale = float(cfg["env"]["object_settings"]["object_grasp_target_z_scale"])
        self.lie_flat_prob = float(cfg["env"]["object_settings"].get("lie_flat_prob", 0.0))
        self._reset_object_mirror_y_prob = float(cfg["env"]["object_settings"].get("mirror_y_prob", 0.5))
        self._reset_object_xyz_range_cfg = cfg["env"]["object_settings"]["xyz_range"]
        self._reset_flat_object_mirror_y_prob = float(cfg["env"]["object_settings"].get("flat_mirror_y_prob", 0.0))
        self._reset_flat_object_xyz_range_cfg = cfg["env"]["object_settings"].get(
            "flat_xyz_range",
            self._reset_object_xyz_range_cfg,
        )
        self._fixed_table_surface_height = float(cfg["env"]["scene"]["table_surface_height"])
        self.side_mode = str(cfg["env"]["eef_init"]["side_mode"])
        self._reset_flat_side_recovery_prob = float(cfg["env"]["eef_init"].get("flat_side_recovery_prob", 0.25))
        reset_bank_cfg = cfg["env"].get("reset_bank", {})
        self._reset_bank_size_cfg = max(int(reset_bank_cfg.get("size", 4096)), 1)
        self._reset_bank_max_ik_goals_cfg = max(int(reset_bank_cfg.get("max_ik_goals", 4096)), 1)
        wrong_side_curriculum_cfg = cfg["env"]["eef_init"].get("wrong_side_curriculum", {})
        wrong_side_num_increments = max(int(wrong_side_curriculum_cfg.get("num_increments", 1)), 1)
        wrong_side_start_stage = int(wrong_side_curriculum_cfg.get("start_stage", 0))
        wrong_side_start_stage = max(0, min(wrong_side_start_stage, wrong_side_num_increments))
        self.wrong_side_curriculum_stage = wrong_side_start_stage
        self.wrong_side_curriculum_stable_steps = 0
        self.wrong_side_sample_prob = 0.0
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
        self._reset_flat_topdown_rel_object_min_cfg = cfg["env"]["eef_init"].get(
            "flat_topdown_rel_object_min",
            [-0.08, -0.08, 0.30],
        )
        self._reset_flat_topdown_rel_object_max_cfg = cfg["env"]["eef_init"].get(
            "flat_topdown_rel_object_max",
            [0.08, 0.08, 0.45],
        )
        self._reset_yaw_noise_deg = float(cfg["env"]["eef_init"].get("yaw_noise_deg", 45.0))
        self._reset_pitch_roll_noise_deg = float(cfg["env"]["eef_init"].get("pitch_roll_noise_deg", 45.0))
        self._init_reset_config()
        if self.lie_flat_prob <= 1.0e-6:
            wrong_side_max_prob = min(max(float(wrong_side_curriculum_cfg.get("max_prob", 0.0)), 0.0), 1.0)
            if bool(wrong_side_curriculum_cfg.get("enable", False)):
                self.wrong_side_sample_prob = wrong_side_max_prob * min(
                    float(wrong_side_start_stage) / float(wrong_side_num_increments),
                    1.0,
                )
            else:
                self.wrong_side_sample_prob = wrong_side_max_prob
        hand_obj_gate_success_cfg = cfg["reward"]["params"].get("hand_obj_gate_success_curriculum", {})
        hand_obj_gate_num_increments = max(int(hand_obj_gate_success_cfg.get("num_increments", 1)), 1)
        hand_obj_gate_start_stage = int(hand_obj_gate_success_cfg.get("start_stage", 0))
        hand_obj_gate_start_stage = max(0, min(hand_obj_gate_start_stage, hand_obj_gate_num_increments))
        self.hand_obj_gate_curriculum_stage = hand_obj_gate_start_stage
        self.hand_obj_gate_curriculum_stable_steps = 0
        self.hand_obj_gate_curriculum_scale = (
            min(float(hand_obj_gate_start_stage) / float(hand_obj_gate_num_increments), 1.0)
            if bool(hand_obj_gate_success_cfg.get("enable", False))
            else 1.0
        )
        self.disable_wrong_side_logic = bool(self.lie_flat_prob > 0.0) and (
            not bool(cfg["reward"]["params"].get("wrong_side_logic_for_flat_objects", False))
        )
        self._top_long_distillation_teleport = self._top_long_mode() and bool(
            cfg["env"]["object_teleport"].get("top_long_match_distillation", True)
        )
        if self._top_long_distillation_teleport:
            tele_cfg = cfg["env"]["object_teleport"]
            # Keep top-long RL teleport schedule and sample count aligned with
            # DexMobileDistillationTopLong when teleport is enabled.
            tele_cfg["mode"] = "schedule"
            tele_cfg["env_proportion"] = 1.0
            tele_cfg["swap_frequency"] = 1
            tele_cfg["swap_freq"] = 1
            tele_cfg["curri_steps"] = 1
            tele_cfg["n0"] = 20
            tele_cfg["n1"] = 105
            tele_cfg["n2"] = 105
            tele_cfg["min_xy_dist_to_eef"] = 0.05
            tele_cfg["min_xy_dist_resample_rounds"] = 12
            tele_cfg["force_right_of_eef"] = False
            tele_cfg["success_curriculum"]["enable"] = False
        if str(cfg["env"]["object_teleport"].get("mode", "schedule")) == "fixed_prob":
            raise ValueError(
                "DexExpTableSide no longer supports object_teleport.mode=fixed_prob. "
                "Use schedule teleport with n0/n1/n2 instead."
            )
        self.profile_step_timing = os.getenv("ISAACGYM_SIDE_PROFILE", "0") == "1"
        self.profile_step_timing_every = max(1, int(os.getenv("ISAACGYM_SIDE_PROFILE_EVERY", "100")))
        print(
            "[FrankaLEAPPickTableSide/init] "
            f"patch=toplong_disable_wrong_side_logic "
            f"lie_flat_prob={self.lie_flat_prob} "
            f"wrong_side_logic_for_flat_objects={bool(cfg['reward']['params'].get('wrong_side_logic_for_flat_objects', False))} "
            f"disable_wrong_side_logic={self.disable_wrong_side_logic} "
            f"wrong_side_sample_prob={self.wrong_side_sample_prob:.3f} "
            f"reset_bank_size={self._reset_bank_size} "
            f"reset_bank_max_ik_goals={self._reset_bank_max_ik_goals} "
            f"teleport_enable={bool(cfg['env']['object_teleport'].get('enable', False))} "
            f"teleport_mode={str(cfg['env']['object_teleport'].get('mode', 'schedule'))} "
            f"top_long_distillation_teleport={self._top_long_distillation_teleport} "
            f"profile_step_timing={self.profile_step_timing} "
            f"profile_every={self.profile_step_timing_every}"
        )
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
        self.teleport_swap_frequency = max(
            1,
            int(self.object_teleport_args.get("swap_frequency", self.object_teleport_args.get("swap_freq", 1))),
        )
        self.teleport_cached_env_ids = torch.empty((0,), dtype=torch.long, device=self.device)
        self.teleport_cached_event_count = 0
        self.object_teleport_last_count = 0
        self.object_teleport_total_events = 0
        self.object_teleport_total_env_teleports = 0
        self.flat_reset_active_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_wrong_side_active_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._init_reset_bank()

    def _init_reset_config(self):
        # Empirical activation ranges. Widen these manually when needed.
        object_xyz_range = _cfg_to_float_array(self._reset_object_xyz_range_cfg, "object_settings.xyz_range")
        self._reset_object_xy_min = torch.tensor(object_xyz_range[0][:2], dtype=torch.float32)
        self._reset_object_xy_max = torch.tensor(object_xyz_range[1][:2], dtype=torch.float32)
        flat_object_xyz_range = _cfg_to_float_array(self._reset_flat_object_xyz_range_cfg, "object_settings.flat_xyz_range")
        self._reset_flat_object_xy_min = torch.tensor(flat_object_xyz_range[0][:2], dtype=torch.float32)
        self._reset_flat_object_xy_max = torch.tensor(flat_object_xyz_range[1][:2], dtype=torch.float32)
        self._reset_eef_rel_object_min = torch.tensor(_cfg_to_float_array(self._reset_eef_rel_object_min_cfg, "eef_init.rel_object_min"), dtype=torch.float32)
        self._reset_eef_rel_object_max = torch.tensor(_cfg_to_float_array(self._reset_eef_rel_object_max_cfg, "eef_init.rel_object_max"), dtype=torch.float32)
        self._reset_flat_eef_rel_object_min = torch.tensor(_cfg_to_float_array(self._reset_flat_eef_rel_object_min_cfg, "eef_init.flat_rel_object_min"), dtype=torch.float32)
        self._reset_flat_eef_rel_object_max = torch.tensor(_cfg_to_float_array(self._reset_flat_eef_rel_object_max_cfg, "eef_init.flat_rel_object_max"), dtype=torch.float32)
        self._reset_base_rel_eef_min = torch.tensor([-0.5523, 0.0161, -0.7124], dtype=torch.float32)
        self._reset_base_rel_eef_max = torch.tensor([-0.3286, 0.3031, -0.1116], dtype=torch.float32)
        self._reset_flat_base_rel_eef_min = torch.tensor([-0.65, -0.55, -0.80], dtype=torch.float32)
        self._reset_flat_base_rel_eef_max = torch.tensor([-0.20, 0.55, -0.05], dtype=torch.float32)
        self._reset_flat_topdown_rel_object_min = torch.tensor(_cfg_to_float_array(self._reset_flat_topdown_rel_object_min_cfg, "eef_init.flat_topdown_rel_object_min"), dtype=torch.float32)
        self._reset_flat_topdown_rel_object_max = torch.tensor(_cfg_to_float_array(self._reset_flat_topdown_rel_object_max_cfg, "eef_init.flat_topdown_rel_object_max"), dtype=torch.float32)
        self._reset_yaw_noise_rad = float(np.deg2rad(self._reset_yaw_noise_deg))
        self._reset_pitch_roll_noise_rad = float(np.deg2rad(self._reset_pitch_roll_noise_deg))
        self._reset_min_palm_object_dist = 0.14
        self._reset_side_clearance = 0.025
        self._reset_hand_joint_noise_deg = 20.0
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
        self._reset_bank_size = int(self._reset_bank_size_cfg)
        self._reset_bank_max_ik_goals = int(self._reset_bank_max_ik_goals_cfg)

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
        self._reset_bank_flat_reset_active_cpu = torch.empty((num_objects, bank_size), dtype=torch.bool, device="cpu")
        self._reset_bank_wrong_side_active_cpu = torch.empty((num_objects, bank_size), dtype=torch.bool, device="cpu")
        if self._wrong_side_sampling_used_for_run() and bank_size >= 2:
            self._reset_bank_correct_size = (bank_size + 1) // 2
            self._reset_bank_wrong_size = bank_size - self._reset_bank_correct_size
        else:
            self._reset_bank_correct_size = bank_size
            self._reset_bank_wrong_size = 0

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

        def _fill_bank_range(range_start, range_end, force_wrong_side_value):
            for start in range(range_start, range_end, slots_per_round):
                cur_slots = min(slots_per_round, range_end - start)
                batched_env_ids = rep_env_ids.repeat(cur_slots)
                force_wrong_side = torch.full(
                    (batched_env_ids.numel(),),
                    bool(force_wrong_side_value),
                    device=self.device,
                    dtype=torch.bool,
                )
                (
                    joint_config,
                    target_quat,
                    left_mask,
                    flat_reset_active,
                    object_center_world,
                    object_quat_world,
                    wrong_side_active,
                ) = self._sample_reset_joint_and_target_quat(
                    batched_env_ids,
                    use_reset_solver=True,
                    force_wrong_side=force_wrong_side,
                )

                end = start + cur_slots
                joint_config = joint_config.reshape(cur_slots, num_objects, self.num_dofs).transpose(0, 1).contiguous().cpu()
                target_quat = target_quat.reshape(cur_slots, num_objects, 4).transpose(0, 1).contiguous().cpu()
                left_mask = left_mask.reshape(cur_slots, num_objects).transpose(0, 1).contiguous().cpu()
                flat_reset_active = flat_reset_active.reshape(cur_slots, num_objects).transpose(0, 1).contiguous().cpu()
                object_center_world = object_center_world.reshape(cur_slots, num_objects, 3).transpose(0, 1).contiguous().cpu()
                object_quat_world = object_quat_world.reshape(cur_slots, num_objects, 4).transpose(0, 1).contiguous().cpu()
                wrong_side_active = wrong_side_active.reshape(cur_slots, num_objects).transpose(0, 1).contiguous().cpu()

                self._reset_bank_joint_config_cpu[:, start:end].copy_(joint_config)
                self._reset_bank_target_quat_cpu[:, start:end].copy_(target_quat)
                self._reset_bank_side_is_left_cpu[:, start:end].copy_(left_mask)
                self._reset_bank_flat_reset_active_cpu[:, start:end].copy_(flat_reset_active)
                self._reset_bank_object_center_world_cpu[:, start:end].copy_(object_center_world)
                self._reset_bank_object_quat_world_cpu[:, start:end].copy_(object_quat_world)
                self._reset_bank_wrong_side_active_cpu[:, start:end].copy_(wrong_side_active)
                progress.update(cur_slots)

        _fill_bank_range(0, self._reset_bank_correct_size, False)
        if self._reset_bank_wrong_size > 0:
            _fill_bank_range(self._reset_bank_correct_size, bank_size, True)
        progress.close()

        bank_bytes = (
            self._reset_bank_joint_config_cpu.element_size() * self._reset_bank_joint_config_cpu.numel() +
            self._reset_bank_target_quat_cpu.element_size() * self._reset_bank_target_quat_cpu.numel() +
            self._reset_bank_object_center_world_cpu.element_size() * self._reset_bank_object_center_world_cpu.numel() +
            self._reset_bank_object_quat_world_cpu.element_size() * self._reset_bank_object_quat_world_cpu.numel() +
            self._reset_bank_side_is_left_cpu.element_size() * self._reset_bank_side_is_left_cpu.numel() +
            self._reset_bank_flat_reset_active_cpu.element_size() * self._reset_bank_flat_reset_active_cpu.numel() +
            self._reset_bank_wrong_side_active_cpu.element_size() * self._reset_bank_wrong_side_active_cpu.numel()
        )
        elapsed = time.time() - t0
        print(
            f"Built side reset bank: num_envs={num_envs}, num_objects={num_objects}, bank_size={bank_size}, slots_per_round={slots_per_round}, "
            f"correct_bank={self._reset_bank_correct_size}, wrong_bank={self._reset_bank_wrong_size}, "
            f"cpu_mem={bank_bytes / (1024 ** 3):.3f} GB, elapsed={elapsed:.1f}s"
        )

    def _sample_reset_from_bank(self, env_ids):
        object_ids_cpu = self.env_object_ids[env_ids].detach().to(device="cpu", dtype=torch.long)
        num_samples = int(object_ids_cpu.numel())
        bank_idx_cpu = torch.randint(
            low=0,
            high=self._reset_bank_correct_size,
            size=(num_samples,),
            device="cpu",
            dtype=torch.long,
        )
        if self._reset_bank_wrong_size > 0:
            wrong_prob = min(max(float(self.wrong_side_sample_prob), 0.0), 1.0)
            use_wrong_cpu = torch.rand((num_samples,), device="cpu") < wrong_prob
            wrong_idx_cpu = self._reset_bank_correct_size + torch.randint(
                low=0,
                high=self._reset_bank_wrong_size,
                size=(num_samples,),
                device="cpu",
                dtype=torch.long,
            )
            bank_idx_cpu = torch.where(use_wrong_cpu, wrong_idx_cpu, bank_idx_cpu)

        joint_config = self._reset_bank_joint_config_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        target_quat = self._reset_bank_target_quat_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        left_mask = self._reset_bank_side_is_left_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device)
        flat_reset_active = self._reset_bank_flat_reset_active_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device)
        object_center_world = self._reset_bank_object_center_world_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        object_quat_world = self._reset_bank_object_quat_world_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device, dtype=self._q.dtype)
        wrong_side_active = self._reset_bank_wrong_side_active_cpu[object_ids_cpu, bank_idx_cpu].to(device=self.device)
        if self.lie_flat_prob <= 0.0:
            object_quat_world.zero_()
            object_quat_world[:, 3] = 1.0
        return joint_config, target_quat, left_mask, flat_reset_active, object_center_world, object_quat_world, wrong_side_active

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

    def _sample_step_teleport_env_ids(self, exclude_env_ids=None):
        if not self.object_teleport_args["enable"] or self.num_teleport_envs <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        curri_factor = self._get_object_teleport_curriculum_scale()
        self.object_teleport_curriculum_scale = curri_factor
        schedule_len = int(self.teleport_probs.shape[0])
        if schedule_len <= 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        event_idx = int(self.sim_steps) % schedule_len
        event_prob = float(self.teleport_probs[event_idx].item()) * float(curri_factor)
        if event_prob <= 0.0 or random.random() >= event_prob:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        candidate_env_ids = torch.where(self.reset_buf == 0)[0]
        if exclude_env_ids is not None and exclude_env_ids.numel() > 0:
            candidate_env_ids = candidate_env_ids[~torch.isin(candidate_env_ids, exclude_env_ids)]
        if candidate_env_ids.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        refresh_subset = (
            self.teleport_cached_env_ids.numel() == 0
            or self.teleport_cached_event_count >= self.teleport_swap_frequency
        )
        if refresh_subset:
            num_pick = min(int(self.num_teleport_envs), int(candidate_env_ids.numel()))
            perm = torch.randperm(int(candidate_env_ids.numel()), device=self.device)[:num_pick]
            self.teleport_cached_env_ids = candidate_env_ids[perm]
            self.teleport_cached_event_count = 0

        cached_mask = self.reset_buf[self.teleport_cached_env_ids] == 0
        if exclude_env_ids is not None and exclude_env_ids.numel() > 0:
            cached_mask = cached_mask & (~torch.isin(self.teleport_cached_env_ids, exclude_env_ids))
        teleport_env_ids = self.teleport_cached_env_ids[cached_mask]
        if teleport_env_ids.numel() > 0:
            self.teleport_cached_event_count += 1
            return teleport_env_ids

        num_pick = min(int(self.num_teleport_envs), int(candidate_env_ids.numel()))
        perm = torch.randperm(int(candidate_env_ids.numel()), device=self.device)[:num_pick]
        self.teleport_cached_env_ids = candidate_env_ids[perm]
        self.teleport_cached_event_count = 1
        return self.teleport_cached_env_ids

    def _get_object_teleport_expected_envs_per_step(self):
        if not self.object_teleport_args["enable"]:
            return 0.0
        curri_factor = float(self._get_object_teleport_curriculum_scale())
        schedule_len = int(self.teleport_probs.shape[0])
        if schedule_len <= 0:
            return 0.0
        return float(self.num_teleport_envs) * float(torch.mean(self.teleport_probs).item()) * curri_factor

    def _wrong_side_sampling_used_for_run(self):
        return self.lie_flat_prob <= 1.0e-6

    def _wrong_side_sample_curriculum_enabled(self):
        curriculum_cfg = self.cfg["env"]["eef_init"].get("wrong_side_curriculum", {})
        return self._wrong_side_sampling_used_for_run() and bool(curriculum_cfg.get("enable", False))

    def _get_wrong_side_sample_prob(self):
        curriculum_cfg = self.cfg["env"]["eef_init"].get("wrong_side_curriculum", {})
        max_prob = min(max(float(curriculum_cfg.get("max_prob", 0.0)), 0.0), 1.0)
        if not self._wrong_side_sampling_used_for_run():
            return 0.0
        if not self._wrong_side_sample_curriculum_enabled():
            return max_prob
        num_increments = max(int(curriculum_cfg.get("num_increments", 1)), 1)
        return max_prob * min(float(self.wrong_side_curriculum_stage) / float(num_increments), 1.0)

    def _update_wrong_side_sample_curriculum(self, success_rate):
        if not self._wrong_side_sampling_used_for_run():
            self.wrong_side_sample_prob = 0.0
            return
        if not self._wrong_side_sample_curriculum_enabled():
            self.wrong_side_sample_prob = self._get_wrong_side_sample_prob()
            return

        curriculum_cfg = self.cfg["env"]["eef_init"].get("wrong_side_curriculum", {})
        success_threshold = float(curriculum_cfg.get("success_threshold", 0.5))
        success_steps = max(int(curriculum_cfg.get("success_steps", 1)), 1)
        num_increments = max(int(curriculum_cfg.get("num_increments", 1)), 1)

        if self.wrong_side_curriculum_stage >= num_increments:
            self.wrong_side_curriculum_stable_steps = 0
        else:
            if float(success_rate) >= success_threshold:
                self.wrong_side_curriculum_stable_steps += 1
            else:
                self.wrong_side_curriculum_stable_steps = 0

            if self.wrong_side_curriculum_stable_steps >= success_steps:
                self.wrong_side_curriculum_stage += 1
                self.wrong_side_curriculum_stable_steps = 0
                self.wrong_side_sample_prob = self._get_wrong_side_sample_prob()
                print(
                    f"[wrong_side_sample_curriculum] stage={self.wrong_side_curriculum_stage}/{num_increments} "
                    f"prob={self.wrong_side_sample_prob:.3f} "
                    f"success_rate={float(success_rate):.3f}"
                )

        self.wrong_side_sample_prob = self._get_wrong_side_sample_prob()

    def _hand_obj_gate_success_curriculum_enabled(self):
        success_cfg = self.cfg["reward"]["params"].get("hand_obj_gate_success_curriculum", {})
        return bool(success_cfg.get("enable", False))

    def _hand_obj_gate_used_for_run(self):
        return self.lie_flat_prob < 1.0 - 1.0e-6

    def _get_hand_obj_gate_curriculum_scale(self):
        if self._hand_obj_gate_success_curriculum_enabled():
            success_cfg = self.cfg["reward"]["params"].get("hand_obj_gate_success_curriculum", {})
            num_increments = max(int(success_cfg.get("num_increments", 1)), 1)
            return min(float(self.hand_obj_gate_curriculum_stage) / float(num_increments), 1.0)
        return 1.0

    def _update_hand_obj_gate_success_curriculum(self, success_rate):
        if not self._hand_obj_gate_used_for_run():
            return
        if not self._hand_obj_gate_success_curriculum_enabled():
            self.hand_obj_gate_curriculum_scale = self._get_hand_obj_gate_curriculum_scale()
            self.reward_settings["hand_obj_gate_curriculum_scale"] = to_torch(
                float(self.hand_obj_gate_curriculum_scale),
                device=self.device,
            )
            return

        success_cfg = self.cfg["reward"]["params"].get("hand_obj_gate_success_curriculum", {})
        success_threshold = float(success_cfg.get("success_threshold", 0.5))
        success_steps = max(int(success_cfg.get("success_steps", 1)), 1)
        num_increments = max(int(success_cfg.get("num_increments", 1)), 1)

        if self.hand_obj_gate_curriculum_stage >= num_increments:
            self.hand_obj_gate_curriculum_stable_steps = 0
            self.hand_obj_gate_curriculum_scale = 1.0
        else:
            if float(success_rate) >= success_threshold:
                self.hand_obj_gate_curriculum_stable_steps += 1
            else:
                self.hand_obj_gate_curriculum_stable_steps = 0

            if self.hand_obj_gate_curriculum_stable_steps >= success_steps:
                self.hand_obj_gate_curriculum_stage += 1
                self.hand_obj_gate_curriculum_stable_steps = 0
                self.hand_obj_gate_curriculum_scale = self._get_hand_obj_gate_curriculum_scale()
                print(
                    f"[hand_obj_gate_curriculum] stage={self.hand_obj_gate_curriculum_stage}/{num_increments} "
                    f"scale={self.hand_obj_gate_curriculum_scale:.3f} "
                    f"success_rate={float(success_rate):.3f}"
                )
            else:
                self.hand_obj_gate_curriculum_scale = self._get_hand_obj_gate_curriculum_scale()

        self.reward_settings["hand_obj_gate_curriculum_scale"] = to_torch(
            float(self.hand_obj_gate_curriculum_scale),
            device=self.device,
        )

    def _adjust_top_long_teleport_xy_bounds(self, env_ids, xy_min, xy_max, dtype):
        xy_min = xy_min.clone()
        xy_max = xy_max.clone()

        table_y_margin = float(self.cfg["env"]["object_settings"].get("table_y_margin", 0.0))
        if table_y_margin > 0.0:
            table_y_min = self.cuboid_pos[env_ids, 0, 1].to(dtype=dtype) - 0.5 * self.cuboid_dims[env_ids, 0, 1].to(dtype=dtype)
            table_y_max = self.cuboid_pos[env_ids, 0, 1].to(dtype=dtype) + 0.5 * self.cuboid_dims[env_ids, 0, 1].to(dtype=dtype)
            object_half_y = 0.5 * self.mesh_aabb_extents[env_ids, 1].to(dtype=dtype)
            safe_y_min = table_y_min + object_half_y + table_y_margin
            safe_y_max = table_y_max - object_half_y - table_y_margin
            safe_y_mid = 0.5 * (table_y_min + table_y_max)
            valid_y_bounds = safe_y_min <= safe_y_max
            safe_y_min = torch.where(valid_y_bounds, safe_y_min, safe_y_mid)
            safe_y_max = torch.where(valid_y_bounds, safe_y_max, safe_y_mid)
            xy_min[:, 1] = torch.minimum(torch.maximum(xy_min[:, 1], safe_y_min), safe_y_max)
            xy_max[:, 1] = torch.maximum(torch.minimum(xy_max[:, 1], safe_y_max), xy_min[:, 1])

        robot_side_x_margin = float(self.cfg["env"]["object_settings"].get("robot_side_x_margin", 0.0))
        if robot_side_x_margin > 0.0:
            table_front_x = self.cuboid_pos[env_ids, 0, 0].to(dtype=dtype) - 0.5 * self.cuboid_dims[env_ids, 0, 0].to(dtype=dtype)
            object_half_x = 0.5 * self.mesh_aabb_extents[env_ids, 0].to(dtype=dtype)
            safe_x_min = table_front_x + object_half_x + robot_side_x_margin
            xy_min[:, 0] = torch.minimum(torch.maximum(xy_min[:, 0], safe_x_min), xy_max[:, 0])

        return xy_min, xy_max

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

        robot_q_collision = self._q[env_ids] if robot_q_for_collision is None else robot_q_for_collision
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

    def _resample_top_long_center_xy_for_collision(
        self,
        env_ids,
        desired_center_xy,
        object_quat,
        xy_min,
        xy_max,
        robot_q_for_collision=None,
        max_rounds=12,
    ):
        if env_ids.numel() == 0:
            return desired_center_xy

        dtype = self._object_state.dtype
        center_xy = desired_center_xy.clone().to(dtype=dtype)
        xy_min = xy_min.to(dtype=dtype)
        xy_max = xy_max.to(dtype=dtype)

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

    def _teleport_object_state(self, teleport_env_ids):
        if teleport_env_ids is None or teleport_env_ids.numel() == 0:
            return

        env_ids = teleport_env_ids.clone()
        num_resets = int(env_ids.numel())
        device = self.device
        dtype = self._q.dtype
        top_long_mode = self._top_long_mode()

        if top_long_mode:
            mirror_mask = torch.zeros(num_resets, device=device, dtype=torch.bool)
            flat_mask = torch.ones(num_resets, device=device, dtype=torch.bool)
            xy_min, xy_max = self._get_reset_object_xy_bounds(env_ids, mirror_mask, dtype, flat_mask=flat_mask)
            xy_min, xy_max = self._adjust_top_long_teleport_xy_bounds(env_ids, xy_min, xy_max, dtype)
        else:
            table_center_y = self.cuboid_pos[env_ids, 0, 1].to(dtype=dtype)
            current_object_center_y = self.states["object_center_pos"][env_ids, 1].to(dtype=dtype)
            mirror_mask = current_object_center_y > table_center_y
            xy_min, xy_max = self._get_reset_object_xy_bounds(env_ids, mirror_mask, dtype)
        eef_xy = self._eef_state[env_ids, :2]
        min_xy_dist_to_eef = float(self.object_teleport_args.get("min_xy_dist_to_eef", 0.05))
        min_xy_dist_to_hand_points = float(
            self.object_teleport_args.get("min_xy_dist_to_hand_points", min_xy_dist_to_eef)
        )
        min_xy_resample_rounds = int(self.object_teleport_args.get("min_xy_dist_resample_rounds", 12))
        force_right_of_eef = bool(self.object_teleport_args.get("force_right_of_eef", False))

        reset_xy = torch.zeros((num_resets, 2), device=device, dtype=dtype)

        def _sample_xy_for_rows(row_mask):
            if not torch.any(row_mask):
                return
            row_idx = row_mask.nonzero(as_tuple=False).squeeze(-1)
            n = int(row_idx.numel())
            reset_xy[row_idx, 0] = (
                torch.rand(n, device=device, dtype=dtype) * (xy_max[row_idx, 0] - xy_min[row_idx, 0]) + xy_min[row_idx, 0]
            )
            reset_xy[row_idx, 1] = (
                torch.rand(n, device=device, dtype=dtype) * (xy_max[row_idx, 1] - xy_min[row_idx, 1]) + xy_min[row_idx, 1]
            )

        _sample_xy_for_rows(torch.ones(num_resets, dtype=torch.bool, device=device))

        if min_xy_dist_to_eef > 0.0:
            for _ in range(min_xy_resample_rounds):
                too_close = torch.norm(reset_xy - eef_xy, dim=-1) < min_xy_dist_to_eef
                if not torch.any(too_close):
                    break
                _sample_xy_for_rows(too_close)
            too_close = torch.norm(reset_xy - eef_xy, dim=-1) < min_xy_dist_to_eef
            if top_long_mode and torch.any(too_close):
                bad_idx = too_close.nonzero(as_tuple=False).squeeze(-1)
                for local_i in bad_idx.tolist():
                    x0, y0 = float(xy_min[local_i, 0].item()), float(xy_min[local_i, 1].item())
                    x1, y1 = float(xy_max[local_i, 0].item()), float(xy_max[local_i, 1].item())
                    corners = torch.tensor(
                        [[x0, y0], [x0, y1], [x1, y0], [x1, y1]],
                        device=device,
                        dtype=dtype,
                    )
                    d = torch.norm(corners - eef_xy[local_i].unsqueeze(0), dim=-1)
                    reset_xy[local_i] = corners[torch.argmax(d)]

        if force_right_of_eef:
            prev_xy = self._object_state[env_ids, :2].clone()
            prev_y = prev_xy[:, 1]
            y_lo = torch.maximum(xy_min[:, 1], prev_y + 1.0e-4)
            y_hi = xy_max[:, 1]
            can_move = y_hi > y_lo
            if torch.any(can_move):
                move_idx = can_move.nonzero(as_tuple=False).squeeze(-1)
                reset_xy[move_idx, 0] = prev_xy[move_idx, 0]
                reset_xy[move_idx, 1] = (
                    torch.rand(int(move_idx.numel()), device=device, dtype=dtype)
                    * (y_hi[move_idx] - y_lo[move_idx])
                    + y_lo[move_idx]
                )
            if torch.any(~can_move):
                stay_idx = (~can_move).nonzero(as_tuple=False).squeeze(-1)
                reset_xy[stay_idx] = prev_xy[stay_idx]

        object_quat_world = torch.zeros((num_resets, 4), device=device, dtype=dtype)
        object_quat_world[:, 3] = 1.0
        if top_long_mode:
            if bool(self.object_teleport_args.get("preserve_orientation", False)):
                object_quat_world = self._object_state[env_ids, 3:7].clone().to(dtype=dtype)
            else:
                yaw_axis = torch.zeros((num_resets, 3), device=device, dtype=dtype)
                yaw_axis[:, 2] = 1.0
                yaw_angle = torch.rand(num_resets, device=device, dtype=dtype) * (2.0 * torch.pi)
                flat_q_yaw = quat_from_angle_axis(yaw_angle, yaw_axis)
                flat_quat_base = self._flat_object_quat_base.to(device=device, dtype=dtype).unsqueeze(0).repeat(num_resets, 1)
                object_quat_world = quat_mul(flat_q_yaw, flat_quat_base)
            object_quat_world = object_quat_world / torch.norm(object_quat_world, dim=-1, keepdim=True).clamp_min(1.0e-8)
        elif self.lie_flat_prob > 0.0:
            local_z = torch.zeros((num_resets, 3), device=device, dtype=dtype)
            local_z[:, 2] = 1.0
            current_z_axis = quat_apply(self._object_state[env_ids, 3:7], local_z)
            current_flat_mask = torch.abs(current_z_axis[:, 2]) <= self.reward_settings["flat_object_axis_z_abs_max"]
            if torch.any(current_flat_mask):
                n_flat = int(current_flat_mask.sum().item())
                yaw_axis = torch.zeros((n_flat, 3), device=device, dtype=dtype)
                yaw_axis[:, 2] = 1.0
                yaw_angle = torch.rand(n_flat, device=device, dtype=dtype) * (2.0 * torch.pi)
                flat_q_yaw = quat_from_angle_axis(yaw_angle, yaw_axis)
                flat_quat_base = self._flat_object_quat_base.to(device=device, dtype=dtype).unsqueeze(0).repeat(n_flat, 1)
                flat_object_quat = quat_mul(flat_q_yaw, flat_quat_base)
                flat_object_quat = flat_object_quat / torch.norm(flat_object_quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
                object_quat_world[current_flat_mask] = flat_object_quat

        local_half_extents = 0.5 * self.mesh_aabb_extents[env_ids]
        object_rot_mat = quaternion_to_matrix_ig(object_quat_world)
        vertical_half_extent = torch.sum(torch.abs(object_rot_mat[:, 2, :]) * local_half_extents, dim=-1)
        object_xy_half_extent = torch.sum(
            torch.abs(object_rot_mat[:, :2, :]) * local_half_extents.unsqueeze(1),
            dim=-1,
        )
        object_xy_radius = torch.linalg.norm(object_xy_half_extent, dim=-1)

        object_center_world = torch.zeros((num_resets, 3), device=device, dtype=dtype)
        object_center_world[:, :2] = reset_xy
        object_center_world[:, 2] = self.table_surface_height[env_ids] + vertical_half_extent

        if top_long_mode:
            reset_xy = self._resample_top_long_center_xy_for_collision(
                env_ids,
                reset_xy,
                object_quat_world,
                xy_min,
                xy_max,
                robot_q_for_collision=None,
                max_rounds=min_xy_resample_rounds,
            )
            object_center_world[:, :2] = reset_xy
        else:
            hand_points_xy = torch.stack(
                [
                    self._eef_state[env_ids, :2],
                    self._eef_wrist_state[env_ids, :2],
                    self._eef_finger1_state[env_ids, :2],
                    self._eef_finger2_state[env_ids, :2],
                    self._eef_finger3_state[env_ids, :2],
                    self._eef_finger4_state[env_ids, :2],
                ],
                dim=1,
            )
            min_center_to_hand_xy = object_xy_radius + min_xy_dist_to_hand_points

            def _hand_clearance_mask(center_world):
                center_xy = center_world[:, :2]
                dists_xy = torch.norm(hand_points_xy - center_xy.unsqueeze(1), dim=-1)
                return torch.any(dists_xy < min_center_to_hand_xy.unsqueeze(1), dim=-1)

            hand_overlap_mask = _hand_clearance_mask(object_center_world)
            if torch.any(hand_overlap_mask):
                for _ in range(min_xy_resample_rounds):
                    _sample_xy_for_rows(hand_overlap_mask)
                    if force_right_of_eef:
                        prev_xy = self._object_state[env_ids, :2].clone()
                        prev_y = prev_xy[:, 1]
                        y_lo = torch.maximum(xy_min[:, 1], prev_y + 1.0e-4)
                        y_hi = xy_max[:, 1]
                        can_move = y_hi > y_lo
                        move_mask = hand_overlap_mask & can_move
                        if torch.any(move_mask):
                            move_idx = move_mask.nonzero(as_tuple=False).squeeze(-1)
                            reset_xy[move_idx, 0] = prev_xy[move_idx, 0]
                            reset_xy[move_idx, 1] = (
                                torch.rand(int(move_idx.numel()), device=device, dtype=dtype)
                                * (y_hi[move_idx] - y_lo[move_idx])
                                + y_lo[move_idx]
                            )
                        stay_mask = hand_overlap_mask & (~can_move)
                        if torch.any(stay_mask):
                            stay_idx = stay_mask.nonzero(as_tuple=False).squeeze(-1)
                            reset_xy[stay_idx] = prev_xy[stay_idx]
                    object_center_world[:, :2] = reset_xy
                    hand_overlap_mask = _hand_clearance_mask(object_center_world)
                    if not torch.any(hand_overlap_mask):
                        break

        self._apply_object_center_state(env_ids, object_center_world, object_quat_world)
        self.reward_settings["object_init_height"][env_ids] = object_center_world[:, 2]
        self._resample_target_pos(env_ids)
        # Teleport is an in-episode disturbance, not a full reset. Keep the
        # episode-level lifting / success buffers latched so metrics are not
        # wiped mid-episode. The next reward step will naturally recompute the
        # current lifting / success conditions against the teleported object.

        if self.object_wrench_args["enable"]:
            self.object_applied_forces[env_ids] = 0.0
            self.object_applied_torques[env_ids] = 0.0
            self.rigid_body_forces[env_ids] = 0
            self.rigid_body_torques[env_ids] = 0

    def _get_reset_object_xy_bounds(self, env_ids, mirror_mask, dtype, flat_mask=None):
        table_center_xy = self.cuboid_pos[env_ids, 0, :2].to(dtype=dtype)
        table_half_xy = 0.5 * self.cuboid_dims[env_ids, 0, :2].to(dtype=dtype)
        table_xy_min = table_center_xy - table_half_xy
        table_xy_max = table_center_xy + table_half_xy

        if flat_mask is None:
            flat_mask = torch.zeros(env_ids.numel(), device=self.device, dtype=torch.bool)
        else:
            flat_mask = flat_mask.to(device=self.device, dtype=torch.bool)

        canonical_min_normal = self._reset_object_xy_min.to(device=self.device, dtype=dtype)
        canonical_max_normal = self._reset_object_xy_max.to(device=self.device, dtype=dtype)
        canonical_min_flat = self._reset_flat_object_xy_min.to(device=self.device, dtype=dtype)
        canonical_max_flat = self._reset_flat_object_xy_max.to(device=self.device, dtype=dtype)

        canonical_min = torch.where(
            flat_mask.unsqueeze(-1),
            canonical_min_flat.unsqueeze(0).repeat(env_ids.numel(), 1),
            canonical_min_normal.unsqueeze(0).repeat(env_ids.numel(), 1),
        )
        canonical_max = torch.where(
            flat_mask.unsqueeze(-1),
            canonical_max_flat.unsqueeze(0).repeat(env_ids.numel(), 1),
            canonical_max_normal.unsqueeze(0).repeat(env_ids.numel(), 1),
        )

        x_min = canonical_min[:, 0]
        x_max = canonical_max[:, 0]
        y_min = torch.where(
            mirror_mask,
            -canonical_max[:, 1],
            canonical_min[:, 1],
        )
        y_max = torch.where(
            mirror_mask,
            -canonical_min[:, 1],
            canonical_max[:, 1],
        )

        empirical_xy_min = torch.stack([x_min, y_min], dim=-1)
        empirical_xy_max = torch.stack([x_max, y_max], dim=-1)
        object_xy_min = torch.maximum(empirical_xy_min, table_xy_min)
        object_xy_max = torch.minimum(empirical_xy_max, table_xy_max)
        return object_xy_min, object_xy_max

    def _sample_reset_joint_and_target_quat(self, env_ids, use_reset_solver=False, force_wrong_side=None):
        num_envs = int(env_ids.numel())
        device = self.device
        dtype = self._q.dtype
        if force_wrong_side is None:
            force_wrong_side = torch.zeros(num_envs, device=device, dtype=torch.bool)
        else:
            force_wrong_side = force_wrong_side.to(device=device, dtype=torch.bool).reshape(num_envs)

        side_mode = self.side_mode
        if side_mode == "left":
            left_mask = torch.ones(num_envs, device=device, dtype=torch.bool)
        elif side_mode == "right":
            left_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
        elif side_mode == "both":
            left_mask = torch.rand(num_envs, device=device) < 0.5
        else:
            raise ValueError(f"Unsupported eef_init.side_mode={side_mode}. Expected one of: left, right, both.")
        reward_target_quat_out = self.target_quat_right.repeat(num_envs, 1)
        if int(left_mask.sum().item()) > 0:
            reward_target_quat_out[left_mask] = self.target_quat_left.repeat(int(left_mask.sum().item()), 1)
        side_init_quat = reward_target_quat_out.clone()
        lie_flat_mask = torch.rand(num_envs, device=device) < self.lie_flat_prob
        if bool(torch.any(lie_flat_mask)):
            reward_target_quat_out[lie_flat_mask] = self.target_quat_flat.repeat(int(lie_flat_mask.sum().item()), 1)
        object_mirror_prob = torch.full((num_envs,), self._reset_object_mirror_y_prob, device=device, dtype=dtype)
        if self._reset_flat_object_mirror_y_prob != self._reset_object_mirror_y_prob:
            object_mirror_prob = torch.where(
                lie_flat_mask,
                torch.full((num_envs,), self._reset_flat_object_mirror_y_prob, device=device, dtype=dtype),
                object_mirror_prob,
            )
        object_mirror_mask = torch.rand(num_envs, device=device, dtype=dtype) < object_mirror_prob

        joint_config = torch.zeros((num_envs, self.num_dofs), device=device, dtype=dtype)
        joint_config[:, 7:23] = self.canonical_flat_hand_config.unsqueeze(0).repeat(num_envs, 1).to(dtype=dtype)

        object_center_world = torch.zeros((num_envs, 3), device=device, dtype=dtype)
        object_quat_world = torch.zeros((num_envs, 4), device=device, dtype=dtype)
        object_quat_world[:, 3] = 1.0
        wrong_side_active_out = torch.zeros(num_envs, device=device, dtype=torch.bool)
        table_surface_height = self.table_surface_height[env_ids].to(dtype=dtype)
        remaining = torch.arange(num_envs, device=device)
        rel_min = self._reset_eef_rel_object_min.to(device=device, dtype=dtype)
        rel_max = self._reset_eef_rel_object_max.to(device=device, dtype=dtype)
        flat_rel_min = self._reset_flat_eef_rel_object_min.to(device=device, dtype=dtype)
        flat_rel_max = self._reset_flat_eef_rel_object_max.to(device=device, dtype=dtype)
        flat_topdown_rel_min = self._reset_flat_topdown_rel_object_min.to(device=device, dtype=dtype)
        flat_topdown_rel_max = self._reset_flat_topdown_rel_object_max.to(device=device, dtype=dtype)
        base_rel_eef_min = self._reset_base_rel_eef_min.to(device=device, dtype=dtype)
        base_rel_eef_max = self._reset_base_rel_eef_max.to(device=device, dtype=dtype)
        flat_base_rel_eef_min = self._reset_flat_base_rel_eef_min.to(device=device, dtype=dtype)
        flat_base_rel_eef_max = self._reset_flat_base_rel_eef_max.to(device=device, dtype=dtype)
        hand_joint_noise = self._reset_hand_joint_noise_rad.to(device=device, dtype=dtype)
        table_center_xy = self.cuboid_pos[env_ids, 0, :2].to(dtype=dtype)
        table_half_xy = 0.5 * self.cuboid_dims[env_ids, 0, :2].to(dtype=dtype)
        object_xy_min, object_xy_max = self._get_reset_object_xy_bounds(
            env_ids,
            object_mirror_mask,
            dtype,
            flat_mask=lie_flat_mask,
        )
        while remaining.numel() > 0:
            batch = int(remaining.numel())
            remaining_flat_mask = lie_flat_mask[remaining]
            remaining_flat_side_mask = remaining_flat_mask & (
                torch.rand(batch, device=device, dtype=dtype) < self._reset_flat_side_recovery_prob
            )
            remaining_flat_topdown_mask = remaining_flat_mask & (~remaining_flat_side_mask)
            rel = torch.rand((batch, 3), device=device, dtype=dtype)
            rel_upright = rel_min.unsqueeze(0) + rel * (rel_max - rel_min).unsqueeze(0)
            rel_flat = flat_rel_min.unsqueeze(0) + rel * (flat_rel_max - flat_rel_min).unsqueeze(0)
            rel_flat_topdown = flat_topdown_rel_min.unsqueeze(0) + rel * (flat_topdown_rel_max - flat_topdown_rel_min).unsqueeze(0)
            grasp_side_sign = torch.where(
                left_mask[remaining],
                torch.ones(batch, device=device, dtype=dtype),
                -torch.ones(batch, device=device, dtype=dtype),
            )
            rel_y_sign = grasp_side_sign
            wrong_side_mask = (~remaining_flat_mask) & force_wrong_side[remaining]
            sampled_rel_y_sign = torch.where(wrong_side_mask, -rel_y_sign, rel_y_sign)
            rel_upright[:, 1] = sampled_rel_y_sign * torch.abs(rel_upright[:, 1])
            rel_flat[:, 1] = rel_y_sign * torch.abs(rel_flat[:, 1])

            object_xy_rand = torch.rand((batch, 2), device=device, dtype=dtype)
            upright_object_xy = object_xy_min[remaining] + object_xy_rand * (object_xy_max[remaining] - object_xy_min[remaining])
            object_xy = upright_object_xy

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
            local_half_extents = 0.5 * self.mesh_aabb_extents[env_ids[remaining]].to(dtype=dtype)
            object_rot_mat = quaternion_to_matrix_ig(candidate_object_quat)
            vertical_half_extent = torch.sum(torch.abs(object_rot_mat[:, 2, :]) * local_half_extents, dim=-1)

            object_center_candidate = torch.zeros((batch, 3), device=device, dtype=dtype)
            object_center_candidate[:, :2] = object_xy
            object_center_candidate[:, 2] = table_surface_height[remaining] + vertical_half_extent

            solved_batch_mask = torch.zeros(batch, device=device, dtype=torch.bool)

            eef_pos_upright = object_center_candidate + rel_upright
            eef_pos_flat = object_center_candidate + rel_flat_topdown
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
                wrong_side_active_out[solved_remaining] = wrong_side_mask[valid_scene][success]
                valid_scene_idx = valid_scene.nonzero(as_tuple=False).squeeze(-1)
                solved_batch_mask[valid_scene_idx[success]] = True

            if bool(torch.any(solved_batch_mask)):
                remaining = remaining[~solved_batch_mask]

        return joint_config, reward_target_quat_out, left_mask, lie_flat_mask, object_center_world, object_quat_world, wrong_side_active_out

    def pre_physics_step(self, actions):
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
        self.reward_settings["goal_align_gate_full_reward"] = to_torch(
            float(self.cfg["reward"]["params"].get("goal_align_gate_full_reward", 0.5)),
            device=self.device,
        )
        self.reward_settings["hand_obj_align_gate_for_side"] = to_torch(
            1.0 if bool(self.cfg["reward"]["params"].get("hand_obj_align_gate_for_side", False)) else 0.0,
            device=self.device,
        )
        self.reward_settings["object_knockdown_penalty_enable"] = to_torch(
            1.0 if bool(self.cfg["reward"]["params"].get("object_knockdown_penalty_enable", False)) else 0.0,
            device=self.device,
        )
        self.reward_settings["object_knockdown_axis_z_abs_max"] = to_torch(
            float(
                np.sin(
                    np.deg2rad(
                        float(self.cfg["reward"]["params"].get("object_knockdown_flat_angle_deg", 30.0))
                    )
                )
            ),
            device=self.device,
        )
        self.reward_settings["object_knockdown_penalty"] = to_torch(
            float(self.cfg["reward"]["params"].get("object_knockdown_penalty", 1.0)),
            device=self.device,
        )
        self.reward_settings["hand_obj_gate_floor"] = to_torch(
            float(self.cfg["reward"]["params"].get("hand_obj_gate_floor", 0.2)),
            device=self.device,
        )
        self.reward_settings["hand_obj_gate_curriculum_scale"] = to_torch(
            float(self.hand_obj_gate_curriculum_scale),
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
        self.reward_settings["wrong_side_logic_for_flat_objects"] = to_torch(
            1.0 if bool(self.cfg["reward"]["params"].get("wrong_side_logic_for_flat_objects", False)) else 0.0,
            device=self.device,
        )
        self.reward_settings["disable_wrong_side_logic"] = to_torch(
            1.0 if self.lie_flat_prob > 0.0 else 0.0,
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
        self.reward_settings["flat_object_axis_z_abs_max"] = to_torch(
            float(self.cfg["reward"]["params"].get("flat_object_axis_z_abs_max", 0.5)),
            device=self.device,
        )
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
        self.target_pos_z_offset_from_table = self.target_pos_z_center - self.table_surface_height

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
        target_center[:, 2] = self.target_pos_z_center[env_ids].to(dtype=target_dtype)
        offsets = (torch.rand((env_ids.numel(), 3), device=self.device, dtype=target_dtype) * 2.0 - 1.0) * target_noise.unsqueeze(0)
        self.reward_settings["target_pos"][env_ids] = target_center + offsets
    
    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        (
            joint_config,
            target_quat,
            left_mask,
            flat_reset_active,
            object_center_world,
            object_quat_world,
            wrong_side_active,
        ) = self._sample_reset_from_bank(env_ids)
        self._apply_object_center_state(env_ids, object_center_world, object_quat_world)
        self.reward_settings["object_init_height"][env_ids] = object_center_world[:, 2]
        self._resample_target_pos(env_ids)
        self.side_is_left[env_ids] = left_mask
        self.flat_reset_active_buf[env_ids] = flat_reset_active
        self.reset_wrong_side_active_buf[env_ids] = wrong_side_active
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

        if self.object_wrench_args["enable"]:
            self.object_applied_forces[env_ids] = 0.0
            self.object_applied_torques[env_ids] = 0.0
            self.rigid_body_forces[env_ids] = 0
            self.rigid_body_torques[env_ids] = 0

    def set_viewer(self):
        super().set_viewer(
            pos=[1.5, -1.0, 0.7],
            target=[0.5, 0.0, 0.1],
        )

    def post_physics_step(self):
        profile_this_step = self.profile_step_timing and ((self.sim_steps % self.profile_step_timing_every) == 0)

        def _profile_sync():
            if profile_this_step and torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)

        if profile_this_step:
            _profile_sync()
            t_step_start = time.perf_counter()

        self.progress_buf += 1

        if profile_this_step:
            _profile_sync()
            t0 = time.perf_counter()
        teleport_env_ids = self._sample_step_teleport_env_ids()
        if profile_this_step:
            _profile_sync()
            t1 = time.perf_counter()
        self.object_teleport_last_count = int(teleport_env_ids.numel())
        if teleport_env_ids.numel() > 0:
            self.object_teleport_total_events += 1
            self.object_teleport_total_env_teleports += int(teleport_env_ids.numel())
            self._teleport_object_state(teleport_env_ids)
        if profile_this_step:
            _profile_sync()
            t2 = time.perf_counter()

        if profile_this_step:
            _profile_sync()
            t3 = time.perf_counter()
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            self.reset_idx(env_ids)
        if profile_this_step:
            _profile_sync()
            t4 = time.perf_counter()

        if profile_this_step:
            _profile_sync()
            t5 = time.perf_counter()
        self.compute_observations()
        if profile_this_step:
            _profile_sync()
            t6 = time.perf_counter()
        # @ray visualize debugging stuff
        if self.debug_viz and self.viewer is not None:
            self._draw_object_xy_range_grid(clear_lines=True)
            self._draw_hand_init_regions(clear_lines=False)
            self._draw_wrong_side_proximity_box(clear_lines=False)
            # self._draw_object_center_cross(clear_lines=False)
            self._draw_target_sampling_box(clear_lines=False)
            # self._draw_workspace_limits_box(clear_lines=False)
            self._draw_grasp_direction_line(clear_lines=False)  # CODEX
            pass
        if profile_this_step:
            _profile_sync()
            t7 = time.perf_counter()
        self.compute_reward()
        if profile_this_step:
            _profile_sync()
            t8 = time.perf_counter()

        if self.video_logging["capture"]:
            self.video_logger()
        if profile_this_step:
            _profile_sync()
            t9 = time.perf_counter()
        self.sim_steps += 1

        if profile_this_step:
            print(
                "[SideProfile] "
                f"step={self.sim_steps} "
                f"tele_sample_ms={(t1 - t0) * 1e3:.2f} "
                f"tele_apply_ms={(t2 - t1) * 1e3:.2f} "
                f"tele_n={int(teleport_env_ids.numel())} "
                f"reset_ms={(t4 - t3) * 1e3:.2f} "
                f"reset_n={int(env_ids.numel())} "
                f"obs_ms={(t6 - t5) * 1e3:.2f} "
                f"reward_ms={(t8 - t7) * 1e3:.2f} "
                f"video_ms={(t9 - t8) * 1e3:.2f} "
                f"total_ms={(t9 - t_step_start) * 1e3:.2f}",
                flush=True,
            )

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
        self.num_objects = min(len(all_meshes_list), self.num_envs) # @ray record number of objects for per-object success rate tracking
        all_meshes_list = all_meshes_list[:self.num_objects]
        self.env_object_ids = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device) 

        # Create environments
        # @ray tensors on object info should be created here
        for i in tqdm(range(self.num_envs), desc="Creating Envs"):
            # grasp object
            object_asset, object_start_pose, object_scale, object_id, mesh_id = all_meshes_list[i % len(all_meshes_list)]
            self.env_object_ids[i] = i % len(all_meshes_list)

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
            table_surface_height_i = float(self._fixed_table_surface_height)
            table_asset, table_start_pose = self._create_cube(
                pos=[0.5, 0.0, table_surface_height_i - table_thickness / 2],
                size=[0.7, 1.2, table_thickness],
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

        self.static_pcds = torch.stack(self.static_pcds, dim=0).to(self.device).to(torch.float32) # (num_envs, num_points, 3)
        self.object_pcds = torch.stack(self.object_pcds, dim=0).to(self.device).to(torch.float32)
        self.combined_pcds = torch.cat([self.static_pcds, self.object_pcds], dim=1).to(self.device) # (num_envs, num_static_points + num_object_points, 3)

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
        object_grasp_target_pos = self._object_state[:, :3].clone()
        local_offset = torch.zeros([self.num_envs, 3], dtype=torch.float, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[:, 2] * self.object_grasp_target_z_scale
        object_rot_mat = quaternion_to_matrix_ig(self._object_state[:, 3:7])
        object_z_axis = object_rot_mat[:, :, 2]
        flat_object_like = torch.abs(object_z_axis[:, 2]) <= self.reward_settings["flat_object_axis_z_abs_max"]
        rotated_offset = torch.matmul(object_rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)
        object_grasp_target_pos += rotated_offset
        object_grasp_target_pos = torch.where(
            flat_object_like.unsqueeze(-1),
            self.states["object_center_pos"],
            object_grasp_target_pos,
        )

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
        # Mode 1: old orientation (target-quat full point-matching; includes XY guidance).
        hand_eef_pos7_rot = torch.cat([self._eef_state[:, :3], self.reward_settings["target_quat"]], dim=-1)
        hand_target_quat_err = self._get_eef_point_matching_err(
            curent_eef_pos7=self._eef_state[:, :7],
            target_eef_pos7=hand_eef_pos7_rot,
        )
        # Mode 2: canonical axis alignment only (no XY guidance), using hand/object canonical z-axes.
        hand_z_axis = eef_rot_mat[:, :, 2]
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
        base_grasp_side_sign = torch.where(
            self.side_is_left,
            torch.ones(self.num_envs, device=self.device, dtype=self._object_state.dtype),
            -torch.ones(self.num_envs, device=self.device, dtype=self._object_state.dtype),
        )
        effective_grasp_side_sign = base_grasp_side_sign

        # @ray not just update but also create new keys here
        self.states.update({
            # Table Contact Status, check whether the object is lifted
            "lift": ~self.table_collision,
            "object_grasp_target_pos": object_grasp_target_pos, # @ray reward-only grasp target
            "object_grasp_target_to_eef": object_grasp_target_to_eef, # @ray for policy observation
            "eef_table_dist": eef_table_dist,
            "object_z_axis_world": object_z_axis,
            "hand_z_axis_world": hand_z_axis,
            "grasp_side_binary": self.side_is_left.float().unsqueeze(-1),
            "grasp_side_sign_effective": effective_grasp_side_sign.unsqueeze(-1),
            "point_matching_err_hand_axis": hand_object_axis_err,  # CODEX
            "point_matching_err_hand_canonical_axis": hand_canonical_axis_err,  # CODEX
            "point_matching_err_hand_targetquat": hand_target_quat_err,  # CODEX
            "point_matching_err_hand": hand_orientation_err,
            "object_upright_score": object_upright_score,  # CODEX
            "flat_reset_active": self.flat_reset_active_buf.float().unsqueeze(-1),
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
        pure_flat_mode = self.lie_flat_prob >= 1.0 - 1.0e-6
        mirror_prob = self._reset_flat_object_mirror_y_prob if pure_flat_mode else self._reset_object_mirror_y_prob
        flat_mask = torch.tensor([pure_flat_mode], device=self.device, dtype=torch.bool)
        draw_canonical = mirror_prob < 1.0
        draw_mirrored = mirror_prob > 0.0

        if draw_canonical:
            canonical_xy_min, canonical_xy_max = self._get_reset_object_xy_bounds(
                torch.tensor([env_id], device=self.device, dtype=torch.long),
                torch.tensor([False], device=self.device, dtype=torch.bool),
                torch.float32,
                flat_mask=flat_mask,
            )
            limits_min = torch.tensor(
                [canonical_xy_min[0, 0].item(), canonical_xy_min[0, 1].item(), self.table_surface_height[env_id].item()],
                device=self.device,
                dtype=torch.float32,
            )
            limits_max = torch.tensor(
                [canonical_xy_max[0, 0].item(), canonical_xy_max[0, 1].item(), self.table_surface_height[env_id].item()],
                device=self.device,
                dtype=torch.float32,
            )
            self._draw_wire_box(env_id, limits_min, limits_max, [0.0, 1.0, 0.0])

        if draw_mirrored:
            mirrored_xy_min, mirrored_xy_max = self._get_reset_object_xy_bounds(
                torch.tensor([env_id], device=self.device, dtype=torch.long),
                torch.tensor([True], device=self.device, dtype=torch.bool),
                torch.float32,
                flat_mask=flat_mask,
            )
            limits_min = torch.tensor(
                [mirrored_xy_min[0, 0].item(), mirrored_xy_min[0, 1].item(), self.table_surface_height[env_id].item()],
                device=self.device,
                dtype=torch.float32,
            )
            limits_max = torch.tensor(
                [mirrored_xy_max[0, 0].item(), mirrored_xy_max[0, 1].item(), self.table_surface_height[env_id].item()],
                device=self.device,
                dtype=torch.float32,
            )
            self._draw_wire_box(env_id, limits_min, limits_max, [0.2, 0.9, 0.2])

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
        colors_flat = [1.0, 0.65, 0.0] * (len(verts_flat) // 6)
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
        mirrored_center = canonical_center.clone()
        mirrored_center[1] = 2.0 * table_center_y_t - canonical_center[1]
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
        mirrored_limits_min = mirrored_center + rel_min
        mirrored_limits_max = mirrored_center + rel_max
        self._draw_wire_box(env_id, canonical_limits_min, canonical_limits_max, [0.35, 0.55, 1.0])
        self._draw_wire_box(env_id, mirrored_limits_min, mirrored_limits_max, [0.0, 0.7, 1.0])

    def _draw_hand_init_regions(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        center = self.states["object_center_pos"][env_id].to(dtype=torch.float32)
        pure_flat_mode = self.lie_flat_prob >= 1.0 - 1.0e-6
        table_center = self.cuboid_pos[env_id, 0].to(dtype=torch.float32)
        table_surface_height = self.table_surface_height[env_id].to(dtype=torch.float32)
        table_center_y = float(self.cuboid_pos[env_id, 0, 1].item())
        table_center_y_t = torch.tensor(table_center_y, device=self.device, dtype=torch.float32)
        is_mirrored = float(center[1].item()) > table_center_y
        canonical_center = center.clone()
        if is_mirrored:
            canonical_center[1] = 2.0 * table_center_y_t - center[1]
        mirrored_center = canonical_center.clone()
        mirrored_center[1] = 2.0 * table_center_y_t - canonical_center[1]
        grasp_side_sign = 1.0 if bool(self.side_is_left[env_id].item()) else -1.0

        def _signed_rel_box(rel_min_raw, rel_max_raw, sign):
            x_min, x_max = rel_min_raw[0], rel_max_raw[0]
            y_min_raw, y_max_raw = rel_min_raw[1], rel_max_raw[1]
            z_min, z_max = rel_min_raw[2], rel_max_raw[2]
            y_abs_max = max(abs(float(y_min_raw.item())), abs(float(y_max_raw.item())))
            if float(y_min_raw.item()) <= 0.0 <= float(y_max_raw.item()):
                y_abs_min = 0.0
            else:
                y_abs_min = min(abs(float(y_min_raw.item())), abs(float(y_max_raw.item())))
            if sign > 0.0:
                box_rel_min = torch.tensor(
                    [x_min.item(), y_abs_min, z_min.item()],
                    device=self.device,
                    dtype=torch.float32,
                )
                box_rel_max = torch.tensor(
                    [x_max.item(), y_abs_max, z_max.item()],
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                box_rel_min = torch.tensor(
                    [x_min.item(), -y_abs_max, z_min.item()],
                    device=self.device,
                    dtype=torch.float32,
                )
                box_rel_max = torch.tensor(
                    [x_max.item(), -y_abs_min, z_max.item()],
                    device=self.device,
                    dtype=torch.float32,
                )
            return box_rel_min, box_rel_max

        if pure_flat_mode:
            if self._reset_flat_side_recovery_prob > 0.0:
                flat_rel_min_raw = self._reset_flat_eef_rel_object_min.to(device=self.device, dtype=torch.float32)
                flat_rel_max_raw = self._reset_flat_eef_rel_object_max.to(device=self.device, dtype=torch.float32)
                flat_box_rel_min, flat_box_rel_max = _signed_rel_box(flat_rel_min_raw, flat_rel_max_raw, grasp_side_sign)
                actual_box_min = center + flat_box_rel_min
                actual_box_max = center + flat_box_rel_max
                self._draw_wire_box(env_id, actual_box_min, actual_box_max, [0.2, 0.85, 1.0])

            if self._reset_flat_side_recovery_prob < 1.0:
                flat_topdown_rel_min = self._reset_flat_topdown_rel_object_min.to(device=self.device, dtype=torch.float32)
                flat_topdown_rel_max = self._reset_flat_topdown_rel_object_max.to(device=self.device, dtype=torch.float32)
                topdown_limits_min = center + flat_topdown_rel_min
                topdown_limits_max = center + flat_topdown_rel_max
                self._draw_wire_box(env_id, topdown_limits_min, topdown_limits_max, [1.0, 0.3, 1.0])
        else:
            rel_min_raw = self._reset_eef_rel_object_min.to(device=self.device, dtype=torch.float32)
            rel_max_raw = self._reset_eef_rel_object_max.to(device=self.device, dtype=torch.float32)
            box_rel_min, box_rel_max = _signed_rel_box(rel_min_raw, rel_max_raw, grasp_side_sign)

            canonical_box_min = canonical_center + box_rel_min
            canonical_box_max = canonical_center + box_rel_max
            mirrored_box_min = mirrored_center + box_rel_min
            mirrored_box_max = mirrored_center + box_rel_max

            self._draw_wire_box(env_id, canonical_box_min, canonical_box_max, [0.35, 0.55, 1.0])
            self._draw_wire_box(env_id, mirrored_box_min, mirrored_box_max, [0.0, 0.7, 1.0])

            split_offset = float(self.reward_settings["grasp_side_split_y_offset"].item())
            sep_x_min = float(canonical_box_min[0].item())
            sep_x_max = float(canonical_box_max[0].item())
            sep_z_min = float(canonical_box_min[2].item())
            sep_z_max = float(canonical_box_max[2].item())
            canonical_sep_y = float(canonical_center[1].item() - grasp_side_sign * split_offset)
            mirrored_sep_y = float(mirrored_center[1].item() - grasp_side_sign * split_offset)

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

        self.states["object_bbox_extent"] = self.mesh_aabb_extents
        self.obs_buf = obs_buf
        self.states_buf = states_buf

        return self.obs_buf

    def compute_reward(self):
        # @ray states used are updated in compute_observations(), called right before compute_reward()

        self.reset_buf[:] = torch.where((self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf[self.states['object_center_pos'][:, 2] < self.table_surface_height-0.1] = 1
        # CODEX: reset scene if object drifts too far in XY from table center (fly-away guard).
        table_center_xy = self.cuboid_pos[:, 0, :2]
        object_center_xy = self.states["object_center_pos"][:, :2]
        dist_to_table_center_xy = torch.norm(object_center_xy - table_center_xy, dim=-1)
        max_xy_dist = 5.0
        self.reset_buf[dist_to_table_center_xy > max_xy_dist] = 1

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
        reward_dict = compute_franka_leap_reward(
            self.states,
            self.reward_settings,
            self.disable_wrong_side_logic,
        )
        # Keep the target fixed within an episode. Only reset-time sampling changes it.

        self.rew_buf[:] = reward_dict["r_total"]
        self.extras["sep_reward/r_hand_obj"] = torch.mean(reward_dict["r_hand_obj"]).item()
        self.extras["sep_reward/r_obj_goal"] = torch.mean(reward_dict["r_obj_goal"]).item()
        self.extras["sep_reward/r_goal_align"] = torch.mean(reward_dict["r_goal_align"]).item()
        self.extras["sep_reward/r_recovery_region"] = torch.mean(reward_dict["r_recovery_region"]).item()
        self.extras["sep_reward/r_wrong_side_proximity"] = torch.mean(reward_dict["r_wrong_side_proximity"]).item()
        self.extras["sep_reward/r_object_knockdown"] = torch.mean(reward_dict["r_object_knockdown"]).item()
        self.extras["sep_reward/r_hand_rot"] = torch.mean(reward_dict["r_hand_rot"]).item()
        self.extras["sep_reward/r_lift"] = torch.mean(reward_dict["r_lift"]).item()
        self.extras["sep_reward/r_curl"] = torch.mean(reward_dict["r_curl"]).item()
        self.extras["sep_reward/r_success_bonus"] = torch.mean(reward_dict["r_success_bonus"]).item()
        self.extras["sep_reward/r_actionreg"] = torch.mean(reward_dict["r_actionreg"]).item()
        self.extras["metrics/wrong_side_forbidden_box_rate"] = torch.mean(
            reward_dict["inside_wrong_side_forbidden_box"].float()
        ).item()
        self.extras["metrics/object_knockdown_rate"] = torch.mean(reward_dict["object_knockdown"].float()).item()
        if self._hand_obj_gate_used_for_run():
            self.extras["metrics/hand_obj_gate"] = torch.mean(reward_dict["hand_obj_gate"]).item()
        else:
            self.extras.pop("metrics/hand_obj_gate", None)
        self.extras["metrics/goal_align_gate"] = torch.mean(reward_dict["goal_align_gate"]).item()
        if bool(self.cfg["reward"]["params"].get("hand_obj_align_gate_for_side", False)):
            self.extras["metrics/hand_obj_align_gate"] = torch.mean(reward_dict["hand_obj_align_gate"]).item()
        else:
            self.extras.pop("metrics/hand_obj_align_gate", None)
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
        self._update_wrong_side_sample_curriculum(self.extras["metrics/success_rate_5cm_per_ep"])
        if self._hand_obj_gate_used_for_run():
            self._update_hand_obj_gate_success_curriculum(self.extras["metrics/success_rate_5cm_per_ep"])
        self.extras["object_wrench/curriculum_stage"] = float(self.object_wrench_curriculum_stage)
        self.extras["object_wrench/curriculum_scale"] = float(self.object_wrench_curriculum_scale)
        self.extras["object_teleport/curriculum_stage"] = float(self.object_teleport_curriculum_stage)
        self.extras["object_teleport/curriculum_scale"] = float(self.object_teleport_curriculum_scale)
        if self._wrong_side_sampling_used_for_run():
            self.extras["wrong_side_sampling/curriculum_stage"] = float(self.wrong_side_curriculum_stage)
            self.extras["wrong_side_sampling/prob"] = float(self.wrong_side_sample_prob)
            self.extras["metrics/reset_wrong_side_rate"] = torch.mean(self.reset_wrong_side_active_buf.float()).item()
        else:
            self.extras.pop("wrong_side_sampling/curriculum_stage", None)
            self.extras.pop("wrong_side_sampling/prob", None)
            self.extras.pop("metrics/reset_wrong_side_rate", None)
        if self._hand_obj_gate_used_for_run():
            self.extras["hand_obj_gate/curriculum_stage"] = float(self.hand_obj_gate_curriculum_stage)
            self.extras["hand_obj_gate/curriculum_scale"] = float(self.hand_obj_gate_curriculum_scale)
        else:
            self.extras.pop("hand_obj_gate/curriculum_stage", None)
            self.extras.pop("hand_obj_gate/curriculum_scale", None)
        self.extras["object_teleport/expected_envs_step"] = float(
            self._get_object_teleport_expected_envs_per_step()
        )
        self.extras["object_teleport/events_total"] = float(
            getattr(self, "object_teleport_total_events", 0)
        )
        self.extras["object_teleport/env_teleports_total"] = float(
            getattr(self, "object_teleport_total_env_teleports", 0)
        )

        # CODEX: success-based right-section curriculum update + logging.
        # log memory usage TODO: debug utils, cleanup later
        mem_allocated_GB = float(torch.cuda.memory_allocated() / 1024**3)
        mem_reserved_GB = float(torch.cuda.memory_reserved() / 1024**3)
        self.extras["mem/allocated_GB"] = mem_allocated_GB
        self.extras["mem/reserved_GB"] = mem_reserved_GB

@torch.jit.script
def compute_franka_leap_reward(states, reward_settings, skip_wrong_side_logic: bool):
    # type: (Dict[str, Tensor], Dict[str, Tensor], bool) -> Dict[str, Tensor]

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
    flat_reset_active = states["flat_reset_active"].squeeze(-1) > 0.5
    if skip_wrong_side_logic:
        wrong_side_logic_active = torch.zeros_like(states["lift"], dtype=torch.bool)
        any_wrong_side_logic_active = False
    elif bool(reward_settings["wrong_side_logic_for_flat_objects"]):
        wrong_side_logic_active = torch.ones_like(states["lift"], dtype=torch.bool)
        any_wrong_side_logic_active = True
    else:
        wrong_side_logic_active = ~flat_reset_active
        any_wrong_side_logic_active = bool(torch.any(wrong_side_logic_active).item())
    use_midheight_xy = bool(reward_settings["hand_obj_use_midheight_xy"])

    finger_positions = torch.zeros((n_env, 4, 3), dtype=states["eef_pos"].dtype, device=states["eef_pos"].device)
    if use_midheight_xy:
        finger_positions = torch.stack(
            [
                states["eef_finger1_pos"],
                states["eef_finger2_pos"],
                states["eef_finger3_pos"],
                states["eef_finger4_pos"],
            ],
            dim=1,
        )

    inside_wrong_side_forbidden_box = torch.zeros_like(states["lift"], dtype=torch.bool)
    if bool(reward_settings["wrong_side_forbidden_box_enable"]) and any_wrong_side_logic_active:
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
    eef_rel_object = states["eef_pos"] - states["object_center_pos"]
    grasp_side_sign = states["grasp_side_sign_effective"].squeeze(-1)
    grasp_side_split_y_offset = reward_settings["grasp_side_split_y_offset"]
    signed_y_to_grasp_side = grasp_side_sign * eef_rel_object[:, 1]
    split_y = -grasp_side_split_y_offset
    on_grasp_side = signed_y_to_grasp_side >= split_y
    prelift_recovery_mode = (~states["lift"]) & (~on_grasp_side) & wrong_side_logic_active
    d_recovery_region = torch.clamp(split_y - signed_y_to_grasp_side, min=0.0)
    beta_recovery_region = reward_settings["beta_recovery_region"]
    r_recovery_region = -(1.0 - torch.exp(-beta_recovery_region * d_recovery_region))
    r_recovery_region = torch.where(prelift_recovery_mode, r_recovery_region, torch.zeros_like(r_recovery_region))
    wrong_side_proximity_threshold = reward_settings["wrong_side_proximity_threshold"]
    wrong_side_proximity_penalty = reward_settings["wrong_side_proximity_penalty"]
    wrong_side_forbidden_mask = torch.zeros_like(states["lift"], dtype=torch.bool)
    if bool(reward_settings["wrong_side_forbidden_box_enable"]):
        wrong_side_forbidden_mask = inside_wrong_side_forbidden_box
        if bool(reward_settings["wrong_side_forbidden_box_side_only"]):
            wrong_side_forbidden_mask = wrong_side_forbidden_mask & (~flat_reset_active)
        wrong_side_forbidden_mask = wrong_side_forbidden_mask & wrong_side_logic_active
        r_wrong_side_proximity = torch.where(
            prelift_recovery_mode & wrong_side_forbidden_mask,
            -wrong_side_proximity_penalty * torch.ones_like(d_hand_obj),
            torch.zeros_like(d_hand_obj),
        )
    else:
        d_wrong_side_proximity = torch.clamp(wrong_side_proximity_threshold - d_hand_obj, min=0.0)
        r_wrong_side_proximity = torch.where(
            prelift_recovery_mode & (d_wrong_side_proximity > 0.0),
            -wrong_side_proximity_penalty * torch.ones_like(d_wrong_side_proximity),
            torch.zeros_like(d_wrong_side_proximity),
        )
    object_knockdown_penalty_enable = reward_settings["object_knockdown_penalty_enable"] > 0.5
    object_knockdown_axis_z_abs_max = reward_settings["object_knockdown_axis_z_abs_max"]
    object_knockdown = (
        object_knockdown_penalty_enable
        & (~flat_reset_active)
        & (torch.abs(object_z_axis_world[:, 2]) <= object_knockdown_axis_z_abs_max)
    )
    r_object_knockdown = torch.where(
        object_knockdown,
        -reward_settings["object_knockdown_penalty"] * torch.ones_like(d_hand_obj),
        torch.zeros_like(d_hand_obj),
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
    goal_align_gate_full_reward = reward_settings["goal_align_gate_full_reward"]
    hand_obj_align_gate_for_side = reward_settings["hand_obj_align_gate_for_side"] > 0.5
    hand_obj_gate_floor = reward_settings["hand_obj_gate_floor"]
    hand_obj_gate_curriculum_scale = reward_settings["hand_obj_gate_curriculum_scale"]
    r_goal_align_raw = torch.exp(-beta_goal_align * d_goal_align)
    r_goal_align = torch.where(states["lift"], r_goal_align_raw, torch.zeros_like(r_goal_align_raw))
    goal_align_gate_denom = torch.maximum(
        goal_align_gate_full_reward,
        torch.ones_like(goal_align_gate_full_reward) * 1.0e-6,
    )
    goal_align_gate_alpha = torch.clamp(r_goal_align / goal_align_gate_denom, 0.0, 1.0)
    goal_align_gate_raw = goal_align_gate_floor + (1.0 - goal_align_gate_floor) * goal_align_gate_alpha
    goal_align_gate_active = states["lift"] & (~flat_reset_active)
    goal_align_gate = torch.where(goal_align_gate_active, goal_align_gate_raw, torch.ones_like(goal_align_gate_raw))
    hand_obj_align_gate_alpha = torch.clamp(r_goal_align_raw / goal_align_gate_denom, 0.0, 1.0)
    hand_obj_align_gate_raw = goal_align_gate_floor + (1.0 - goal_align_gate_floor) * hand_obj_align_gate_alpha
    hand_obj_align_gate_active = (~flat_reset_active) & hand_obj_align_gate_for_side
    hand_obj_align_gate = torch.where(
        hand_obj_align_gate_active,
        hand_obj_align_gate_raw,
        torch.ones_like(hand_obj_align_gate_raw),
    )
    hand_obj_gate_raw = hand_obj_gate_floor + (1.0 - hand_obj_gate_floor) * r_hand_obj
    hand_obj_gate = 1.0 - hand_obj_gate_curriculum_scale * (1.0 - hand_obj_gate_raw)
    hand_obj_gate_active = states["lift"] & (~flat_reset_active)
    hand_obj_gate = torch.where(hand_obj_gate_active, hand_obj_gate, torch.ones_like(hand_obj_gate))
    r_obj_goal = torch.exp(-beta_object_goal * d_eef_point_goal_target)
    r_obj_goal = torch.where(states["lift"], r_obj_goal * goal_align_gate * hand_obj_gate, 0.0)
    r_lift = r_lift * goal_align_gate * hand_obj_gate

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
    r_hand_obj = r_hand_obj * hand_obj_align_gate
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
              r_wrong_side_proximity + r_object_knockdown + w_hand_orientation*r_hand_orientation + \
              w_curl*r_curl * float(use_curl) + \
              w_lift*r_lift + w_actionreg*r_actionreg

    success_bonus_threshold = reward_settings["success_bonus_threshold"]
    success_region = states["lift"] & (d_eef_point_goal_target < success_bonus_threshold) & (r_goal_align > 0.5)
    r_success_bonus = torch.where(
        success_region,
        goal_align_gate * hand_obj_gate * w_success_bonus,
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
        "r_object_knockdown": r_object_knockdown,
        "r_hand_rot": w_hand_orientation*r_hand_orientation,
        "r_curl": w_curl*r_curl,
        "r_success_bonus": r_success_bonus,
        "r_actionreg": w_actionreg*r_actionreg,
        "r_total": r_total,
        "inside_wrong_side_forbidden_box": (prelift_recovery_mode & wrong_side_forbidden_mask).float(),
        "object_knockdown": object_knockdown.float(),
        "hand_obj_gate": hand_obj_gate,
        "goal_align_gate": goal_align_gate,
        "hand_obj_align_gate": hand_obj_align_gate,
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
