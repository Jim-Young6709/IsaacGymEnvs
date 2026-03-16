"""
Franka + LEAP Hand Pick Env
"""
import time
import json
import os

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.utils.pcd_utils import *
from isaacgymenvs.utils.rotation_conversions import *
from isaacgymenvs.tasks import FrankaLEAP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from tqdm import tqdm
import wandb


class FrankaLEAPPickTableSide(FrankaLEAP):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        # @ray we actually manually design the hand reset position range so avoid ik solver failures, so we need a fixed object spawn position
        # also, since we use eef position control, the policy is agnostic to object position and franka priorioception in the world frame
        # so fixed position won't affect learning
        xyz_range = cfg["env"]["object_settings"]["xyz_range"]
        avg_xyz = [
            0.5 * (xyz_range[0][0] + xyz_range[1][0]),
            0.5 * (xyz_range[0][1] + xyz_range[1][1]),
            0.5 * (xyz_range[0][2] + xyz_range[1][2]),
        ]
        cfg["env"]["object_settings"]["xyz_range"] = [avg_xyz, avg_xyz]
        self.object_grasp_target_z_scale = float(cfg["env"]["object_settings"]["object_grasp_target_z_scale"])
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
        self._right_section_last_log_step = -1  # CODEX
        self._right_section_active_sections = 0  # CODEX
        self._right_section_success_hold_counter = 0  # CODEX
        self._right_section_last_success_rate = 0.0  # CODEX
        # @ray precompute a reset pose bank to reuse
        # this is better than resampling during each reset, far less compute
        # however, need to potentially modify bank size for different sampling regions
        self._create_reset_pose_bank()
        # CODEX: initialize curriculum state after pose bank is ready.
        num_sections = int(self.eef_init["right_section_num"])
        force_active = int(self.eef_init["right_section_curriculum_force_active_sections"])
        if force_active > 0:
            self._right_section_active_sections = max(0, min(num_sections, force_active))
        else:
            self._right_section_active_sections = 0

    def _get_active_right_sections(self):
        # CODEX: helper for right-section curriculum progress/debug visualization.
        num_sections = int(self.eef_init["right_section_num"])
        if not bool(self.eef_init["right_section_curriculum_enable"]):
            active_sections = num_sections
        else:
            curriculum_mode = str(self.eef_init["right_section_curriculum_mode"])
            if curriculum_mode == "steps":
                curri_start_steps = int(self.eef_init["right_section_curriculum_start_steps"])
                curri_steps = int(self.eef_init["right_section_curriculum_steps"])
                progress_steps = max(float(self.sim_steps) - float(curri_start_steps), 0.0)
                curri_t = min(progress_steps / float(max(curri_steps, 1)), 1.0)
                active_sections = min(num_sections, max(0, int(np.floor(curri_t * float(num_sections)))))
            elif curriculum_mode == "success":
                active_sections = int(self._right_section_active_sections)
            else:
                raise ValueError(
                    f"Unsupported eef_init.right_section_curriculum_mode={curriculum_mode}. "
                    f"Expected one of: steps, success."
                )
        force_active = int(self.eef_init["right_section_curriculum_force_active_sections"])
        if force_active > 0:
            active_sections = max(0, min(num_sections, force_active))
        return active_sections

    def _update_right_section_curriculum_from_success(self):
        # CODEX: success-gated curriculum: when success_rate_5cm_per_ep stays above threshold
        # for hold_steps consecutive sim steps, unlock one additional right section.
        if not bool(self.eef_init["right_section_curriculum_enable"]):
            return
        if str(self.eef_init["right_section_curriculum_mode"]) != "success":
            return
        if int(self.eef_init["right_section_curriculum_force_active_sections"]) > 0:
            return

        num_sections = int(self.eef_init["right_section_num"])
        if self._right_section_active_sections >= num_sections:
            self._right_section_success_hold_counter = 0
            return

        success_threshold = float(self.eef_init["right_section_curriculum_success_threshold"])
        hold_steps = int(self.eef_init["right_section_curriculum_success_hold_steps"])
        if self._right_section_last_success_rate >= success_threshold:
            self._right_section_success_hold_counter += 1
        else:
            self._right_section_success_hold_counter = 0

        if self._right_section_success_hold_counter >= hold_steps:
            self._right_section_active_sections = min(num_sections, self._right_section_active_sections + 1)
            self._right_section_success_hold_counter = 0
            print(
                f"[right-section-curriculum] sim_steps={int(self.sim_steps)} "
                f"success_rate_5cm_per_ep={self._right_section_last_success_rate:.4f} "
                f"active_sections={self._right_section_active_sections}/{num_sections}"
            )
    
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
        
        self.reward_settings["beta_hand_orientation"] = to_torch(self.cfg["reward"]["exp"]["beta_hand_orientation"], device=self.device)
        self.reward_settings["w_hand_orientation"] = to_torch(self.cfg["reward"]["weights"]["w_hand_orientation"], device=self.device)

        # @ray to force the hand to grasp on the correct region on the object (in terms of z-axis), we gate the lift and to-goal rewards with a sigmoid based on z-difference 
        # between the grasp_target's z height and the averaged finger height on obejct. 
        # @ray we add an offset to the grasp_target z_height to match the height of the stacked fingers. 
        # This is crucial to make reward consistent across objects of varying height.
        # @ray we also add a floor value to make the gate not 0, otherwise the policy loses incentive to lift early on when its grasp style is not good enough
        self.reward_settings["grasp_on_object_z_height_tolerance"] = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_tolerance"], device=self.device)
        self.reward_settings["grasp_on_object_z_height_slope"] = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_slope"], device=self.device)
        self.reward_settings["grasp_on_object_z_height_offset"] = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_offset"], device=self.device)
        self.reward_settings["grasp_on_object_z_height_gate_floor"]  = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_gate_floor"], device=self.device)
        self.reward_settings["grasp_on_object_z_height_gate_enabled"] = to_torch(1.0 if bool(self.cfg["reward"]["params"]["grasp_on_object_z_height_gate_enabled"]) else 0.0, device=self.device)
        self.reward_settings["hand_obj_use_midheight_xy"] = to_torch(
            1.0 if bool(self.cfg["reward"]["params"]["hand_obj_use_midheight_xy"]) else 0.0,
            device=self.device,
        )  # CODEX
        self.reward_settings["hand_obj_midheight_height_weight"] = to_torch(
            float(self.cfg["reward"]["params"]["hand_obj_midheight_height_weight"]),
            device=self.device,
        )  # CODEX

        self.reward_settings["w_obj_goal_base"] = self.reward_settings["w_obj_goal"].clone()
        self.reward_settings["w_lift_base"] = self.reward_settings["w_lift"].clone()

    def _create_reset_pose_bank(self):
        self.pose_bank_size = int(self.eef_init["pose_bank_size"])
        bank_batch = self.pose_bank_size
        max_rounds = int(self.eef_init["pose_bank_max_rounds"])
        num_groups = int(self.num_objects * 2)

        # @ray build left/right/both pose banks per object
        # This assumes that all instances of the same object shares the same scale
        side_mode = self.eef_init["side_mode"]
        right_section_sampling_enable = bool(self.eef_init["right_section_sampling_enable"])  # CODEX
        debug_disable_left = bool(self.eef_init["debug_disable_left_pose_bank"])  # CODEX
        if side_mode == "left":
            # CODEX: for left grasp mode, still build right-section bank when enabled (for curriculum/debug viz),
            # while reset sampling can remain left-only unless debug_disable_left_pose_bank is set.
            side_build = [True, False] if right_section_sampling_enable else [True]
        elif side_mode == "right":
            side_build = [True, False] if right_section_sampling_enable else [False]
        else:
            side_build = [True, False]

        # @ray initialize left+right banks, we may only use half if side_mode!=both
        # but no need to optimize for memory as this is pretty cheap 
        # (~30mb) for 4096 bank size of 69 objects
        self.pose_bank_joint = torch.zeros((num_groups, self.pose_bank_size, self.num_dofs), device=self.device)
        self.pose_bank_quat = torch.zeros((num_groups, self.pose_bank_size, 4), device=self.device)
        self.pose_bank_eef_pos = torch.zeros((num_groups, self.pose_bank_size, 3), device=self.device)  # CODEX
        self.pose_bank_section = torch.full((num_groups, self.pose_bank_size), -1, dtype=torch.long, device=self.device)  # CODEX
        self.pose_bank_count = torch.zeros((num_groups,), dtype=torch.long, device=self.device)

        all_env_ids = torch.arange(self.num_envs, device=self.device)
        pbar = tqdm(total=self.num_objects * len(side_build), desc="Building Reset Pose Bank")

        for obj_id in range(self.num_objects):
            matches = (self.env_object_ids == obj_id)
            rep_env_id = int(all_env_ids[matches][0].item())
            for left_side in side_build:
                side_idx = 1 if left_side else 0
                group_id = obj_id * 2 + side_idx
                rounds = 0
                # @ray sampling strategy
                # 1) sample candidate points on a sphere shell centered around the object, with radius = side_distance + noise
                # 2) filter candidates that are outside workspace limits, a xyz bounding box
                # 3) solve for ik, if failure retry, if repeated retries still result in failure, errors
                limits_min_rel = torch.tensor(self.eef_init["limits_xyz_min"], device=self.device)
                limits_max_rel = torch.tensor(self.eef_init["limits_xyz_max"], device=self.device)
                if left_side:
                    # Mirror Y bounds for left-side sampling.
                    y_min_r = limits_min_rel[1].clone()
                    y_max_r = limits_max_rel[1].clone()
                    limits_min_rel[1] = -y_max_r
                    limits_max_rel[1] = -y_min_r
                # @ray HACK assumes fixed object start position at table center
                table_center = torch.tensor(
                    [
                        self.cuboid_pos[rep_env_id, 0, 0].item(),
                        self.cuboid_pos[rep_env_id, 0, 1].item(),
                        self.table_surface_height[rep_env_id].item(),
                    ],
                    device=self.device)
                limits_min = table_center + limits_min_rel
                limits_max = table_center + limits_max_rel
                side_distance = float(self.eef_init["side_distance"])
                side_distance_noise = float(self.eef_init["side_distance_noise"])
                target_quat_noise_deg = float(self.eef_init["target_quat_noise_deg"])
                target_quat_noise_rad = float(np.deg2rad(target_quat_noise_deg))
                pos_offset_min = torch.tensor(
                    self.eef_init["pos_offset_xyz_min"],
                    device=self.device,
                    dtype=torch.float32,
                )  # CODEX
                pos_offset_max = torch.tensor(
                    self.eef_init["pos_offset_xyz_max"],
                    device=self.device,
                    dtype=torch.float32,
                )  # CODEX

                # Orientation sampling mode for IK target quaternions:
                # - "facing_object": canonical side quat aligned to object-facing direction (default)
                # - "random_full": uniform random quaternion
                target_quat_sampling_mode = str(self.eef_init["target_quat_sampling_mode"])
                if target_quat_sampling_mode not in ("facing_object", "random_full"):
                    raise ValueError(
                        f"Unsupported eef_init.target_quat_sampling_mode={target_quat_sampling_mode}. "
                        f"Expected one of: facing_object, random_full."
                    )
            
                phi_range_deg = self.eef_init["phi_range_deg"]
                phi_start_deg = float(phi_range_deg[0])
                phi_end_deg = float(phi_range_deg[1])
                if left_side:
                    phi_start_deg, phi_end_deg = -phi_end_deg, -phi_start_deg
                phi_start_deg = phi_start_deg % 360.0
                phi_end_deg = phi_end_deg % 360.0
                phi_sweep_deg = (phi_end_deg - phi_start_deg) % 360.0


                center = self._object_center_init_state[rep_env_id]
                # CODEX: keep left bank generation unchanged; synthesize right bank by Y-sections from left poses.
                if (not left_side) and bool(self.eef_init["right_section_sampling_enable"]):
                    left_group_id = obj_id * 2 + 1
                    left_count = int(self.pose_bank_count[left_group_id].item())
                    if left_count <= 0:
                        raise RuntimeError(f"right-section bank build requires left bank first for obj_id={obj_id}")
                    left_eef_pos = self.pose_bank_eef_pos[left_group_id, :left_count]
                    left_quat = self.pose_bank_quat[left_group_id, :left_count]
                    num_sections = int(self.eef_init["right_section_num"])
                    right_y_rel_max = float(-float(self.eef_init["limits_xyz_min"][1]))
                    if right_y_rel_max <= 0.0:
                        raise ValueError("right_section_num requires -limits_xyz_min[1] > 0")
                    # CODEX: infer right-side Y direction as opposite of left-bank mean side.
                    left_y_mean_rel = torch.mean(left_eef_pos[:, 1] - center[1])
                    left_sign = 1.0 if float(left_y_mean_rel.item()) >= 0.0 else -1.0
                    right_sign = -left_sign
                    section_width = right_y_rel_max / float(num_sections)
                    # Mirror left density per Y-length onto right side.
                    left_y_span = right_y_rel_max
                    left_density = float(left_count) / max(left_y_span, 1.0e-6)
                    counts = torch.full((num_sections,), int(left_density * section_width), dtype=torch.long, device=self.device)
                    counts = torch.clamp(counts, min=1)
                    total_counts = int(counts.sum().item())
                    if total_counts != self.pose_bank_size:
                        scaled = torch.floor(counts.float() * (float(self.pose_bank_size) / float(max(total_counts, 1)))).long()
                        counts = torch.clamp(scaled, min=1)
                        diff = int(self.pose_bank_size - counts.sum().item())
                        if diff > 0:
                            counts[:diff] += 1
                        elif diff < 0:
                            dec = min(-diff, int((counts > 1).sum().item()))
                            if dec > 0:
                                dec_idx = (counts > 1).nonzero(as_tuple=False).squeeze(-1)[:dec]
                                counts[dec_idx] -= 1

                    section_cursor = 0
                    for section_idx in range(num_sections):
                        target_count = int(counts[section_idx].item())
                        if target_count <= 0:
                            continue
                        y_lo_rel = section_idx * section_width
                        y_hi_rel = (section_idx + 1) * section_width
                        filled = 0
                        rounds_section = 0
                        while filled < target_count:
                            rounds_section += 1
                            if rounds_section > max_rounds:
                                raise RuntimeError(
                                    f"right-section pose bank build failed for group {group_id}, section {section_idx}, "
                                    f"filled={filled}/{target_count}"
                                )
                            batch = max(target_count - filled, 32) * 2
                            base_idx = torch.randint(0, left_count, (batch,), device=self.device)
                            eef_pos = left_eef_pos[base_idx].clone()
                            quat_seed = left_quat[base_idx].clone()
                            y_samples_rel = y_lo_rel + torch.rand(batch, device=self.device) * (y_hi_rel - y_lo_rel)
                            eef_pos[:, 1] = center[1] + right_sign * y_samples_rel
                            # keep XZ inside existing workspace limits; Y follows section bounds by construction.
                            valid_xz = (
                                (eef_pos[:, 0] >= limits_min[0]) &
                                (eef_pos[:, 0] <= limits_max[0]) &
                                (eef_pos[:, 2] >= limits_min[2]) &
                                (eef_pos[:, 2] <= limits_max[2])
                            )
                            if not torch.any(valid_xz):
                                continue
                            eef_pos = eef_pos[valid_xz]
                            quat_seed = quat_seed[valid_xz]
                            eef_pose = torch.cat([eef_pos, quat_seed], dim=-1)
                            n = eef_pose.shape[0]
                            chunk_size = int(self.num_envs)
                            arm_q_chunks = []
                            success_chunks = []
                            with torch.enable_grad():
                                for start in range(0, n, chunk_size):
                                    end = min(start + chunk_size, n)
                                    arm_q_i, success_i = self.get_joint_from_ee(eef_pose[start:end], return_success=True)
                                    arm_q_chunks.append(arm_q_i)
                                    success_chunks.append(success_i)
                            arm_q = torch.cat(arm_q_chunks, dim=0)
                            ok = torch.cat(success_chunks, dim=0).bool().reshape(-1)
                            if not torch.any(ok):
                                continue
                            arm_ok = arm_q[ok]
                            quat_ok = quat_seed[ok]
                            eef_ok = eef_pos[ok]
                            take = min(target_count - filled, int(arm_ok.shape[0]))
                            joint_block = torch.zeros((take, self.num_dofs), device=self.device)
                            joint_block[:, :7] = arm_ok[:take]
                            start_i = section_cursor + filled
                            end_i = start_i + take
                            self.pose_bank_joint[group_id, start_i:end_i] = joint_block
                            self.pose_bank_quat[group_id, start_i:end_i] = quat_ok[:take]
                            self.pose_bank_eef_pos[group_id, start_i:end_i] = eef_ok[:take]
                            self.pose_bank_section[group_id, start_i:end_i] = section_idx
                            filled += take
                        section_cursor += target_count
                    self.pose_bank_count[group_id] = self.pose_bank_size
                    pbar.update(1)
                    continue

                while int(self.pose_bank_count[group_id].item()) < self.pose_bank_size:
                    rounds += 1
                    if rounds > max_rounds:
                        raise RuntimeError(f"pose bank build failed for group {group_id}, count={int(self.pose_bank_count[group_id].item())}")

                    # sample candidate side-approach positions for this object group.
                    out = []
                    total = 0
                    batch_size = bank_batch * 2
                    while total < bank_batch:
                        radius = side_distance + (torch.rand(batch_size, device=self.device) * 2 - 1.0) * side_distance_noise
                        radius = radius.clamp_min(1e-4)
                        # Sample directions on the unit sphere.
                        cos_theta = torch.rand(batch_size, device=self.device) * 2.0 - 1.0
                        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
                        # TODO @ray lifted the constraint on left/right hemisphere, should add a change to flip the xyz ,limits for left
                        # if left_side:
                        #     # left fan: [-overlap, +pi/2]
                        #     phi = torch.rand(batch_size, device=self.device) * torch.pi
                        # else:
                        #     # right fan: [-pi/2, +overlap]
                        #     phi = -torch.pi + torch.rand(batch_size, device=self.device) * torch.pi
                        if phi_range_deg is not None:
                            u = torch.rand(batch_size, device=self.device)
                            phi_deg = (phi_start_deg + u * phi_sweep_deg) % 360.0
                            phi = phi_deg * (torch.pi / 180.0)
                        else:
                            phi = -torch.pi + torch.rand(batch_size, device=self.device) * (2 * torch.pi)
                            
                        dirs = torch.stack(
                            [torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)],
                            dim=-1,
                        )
                        points = center + dirs * radius.unsqueeze(-1)
                        # CODEX: apply optional per-axis uniform XYZ offset before validity filtering.
                        if bool(torch.any(pos_offset_max > pos_offset_min)):
                            offsets = torch.rand((batch_size, 3), device=self.device, dtype=points.dtype)
                            offsets = pos_offset_min.unsqueeze(0) + offsets * (pos_offset_max - pos_offset_min).unsqueeze(0)
                            points = points + offsets
                        xyz_valid = ((points >= limits_min) & (points <= limits_max)).all(dim=-1)
                        # CODEX: enforce final sampled point radius around object center within side_distance +/- side_distance_noise.
                        radius_from_center = torch.norm(points - center.unsqueeze(0), dim=-1)
                        radius_min = max(1.0e-4, side_distance - side_distance_noise)
                        radius_max = side_distance + side_distance_noise
                        radius_valid = (radius_from_center >= radius_min) & (radius_from_center <= radius_max)
                        valid = xyz_valid & radius_valid
                        v = points[valid]
                        out.append(v)
                        total += int(v.shape[0])
                    eef_pos = torch.cat(out, dim=0)[:bank_batch]

                    n = eef_pos.shape[0]
                    env_ids_rep = torch.full((n,), rep_env_id, dtype=torch.long, device=self.device)
                    obj_center = self._object_center_init_state[env_ids_rep]

                    # convert our sampled eef_pos to target_quat of the hand pointing at the object center
                    # solve ik to get feasible joint configurations
                    base_quat = self.target_quat_right.repeat(n, 1)
                    if left_side:
                        base_quat = self.target_quat_left.repeat(n, 1)
                    desired_dir = obj_center - eef_pos
                    desired_dir = desired_dir / torch.norm(desired_dir, dim=-1, keepdim=True)
                    ref_dir = torch.zeros((n, 3), device=self.device, dtype=eef_pos.dtype)
                    ref_dir[:, 1] = -1.0 if left_side else 1.0

                    if target_quat_sampling_mode == "random_full":
                        # Uniform random quaternion in xyzw format.
                        u1 = torch.rand((n,), device=self.device, dtype=eef_pos.dtype)
                        u2 = torch.rand((n,), device=self.device, dtype=eef_pos.dtype)
                        u3 = torch.rand((n,), device=self.device, dtype=eef_pos.dtype)
                        two_pi = 2.0 * torch.pi
                        qx = torch.sqrt(1.0 - u1) * torch.sin(two_pi * u2)
                        qy = torch.sqrt(1.0 - u1) * torch.cos(two_pi * u2)
                        qz = torch.sqrt(u1) * torch.sin(two_pi * u3)
                        qw = torch.sqrt(u1) * torch.cos(two_pi * u3)
                        target_quat = torch.stack([qx, qy, qz, qw], dim=-1)
                    else:
                        # Canonical side quat + alignment toward object center.
                        cross = torch.cross(ref_dir, desired_dir, dim=-1)
                        dot = torch.sum(ref_dir * desired_dir, dim=-1, keepdim=True)
                        q_align = torch.cat([cross, 1.0 + dot], dim=-1)
                        q_align = q_align / torch.norm(q_align, dim=-1, keepdim=True)
                        target_quat = quat_mul(q_align, base_quat)
                    if target_quat_noise_rad > 0.0:
                        noise_axis = torch.randn((n, 3), device=self.device, dtype=eef_pos.dtype)
                        noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True).clamp_min(1e-8)
                        noise_angle = (torch.rand((n,), device=self.device, dtype=eef_pos.dtype) * 2.0 - 1.0) * target_quat_noise_rad
                        q_noise = quat_from_angle_axis(noise_angle, noise_axis)
                        target_quat = quat_mul(q_noise, target_quat)

                    target_quat = target_quat / torch.norm(target_quat, dim=-1, keepdim=True)
                    # @ray reward target uses only XY-facing yaw on top of canonical side-flat quaternion.
                    desired_dir_xy = desired_dir.clone()
                    desired_dir_xy[:, 2] = 0.0
                    desired_dir_xy = desired_dir_xy / torch.norm(desired_dir_xy, dim=-1, keepdim=True)
                    ref_yaw = torch.atan2(ref_dir[:, 1], ref_dir[:, 0])
                    desired_yaw = torch.atan2(desired_dir_xy[:, 1], desired_dir_xy[:, 0])
                    yaw_delta = desired_yaw - ref_yaw
                    yaw_axis = torch.zeros((n, 3), device=self.device, dtype=eef_pos.dtype)
                    yaw_axis[:, 2] = 1.0
                    q_yaw = quat_from_angle_axis(yaw_delta, yaw_axis)
                    reward_quat = quat_mul(q_yaw, base_quat)
                    reward_quat = reward_quat / torch.norm(reward_quat, dim=-1, keepdim=True)

                    eef_pose = torch.cat([eef_pos, target_quat], dim=-1)
                    chunk_size = int(self.num_envs)
                    arm_q_chunks = []
                    success_chunks = []
                    with torch.enable_grad():
                        for start in range(0, n, chunk_size):
                            end = min(start + chunk_size, n)
                            arm_q_i, success_i = self.get_joint_from_ee(eef_pose[start:end],return_success=True)
                            arm_q_chunks.append(arm_q_i)
                            success_chunks.append(success_i)
                    arm_q = torch.cat(arm_q_chunks, dim=0)
                    ok = torch.cat(success_chunks, dim=0).bool().reshape(-1)
                    arm_ok = arm_q[ok]
                    quat_ok = reward_quat[ok]
                    eef_ok = eef_pos[ok]
                    num_ok = int(arm_ok.shape[0])
                    count = int(self.pose_bank_count[group_id].item())
                    take = min(self.pose_bank_size - count, num_ok)
                    joint_block = torch.zeros((take, self.num_dofs), device=self.device)
                    joint_block[:, :7] = arm_ok[:take]
                    self.pose_bank_joint[group_id, count:count + take] = joint_block
                    self.pose_bank_quat[group_id, count:count + take] = quat_ok[:take]
                    self.pose_bank_eef_pos[group_id, count:count + take] = eef_ok[:take]  # CODEX
                    self.pose_bank_section[group_id, count:count + take] = -1  # CODEX
                    self.pose_bank_count[group_id] = count + take
                pbar.update(1)
        pbar.close()
    
    def reset_idx(self, env_ids=None):
        super().reset_idx(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return
        num_envs = len(env_ids)
        side_mode = self.eef_init["side_mode"]
        # CODEX: active-pose union logic
        # - right poses are active when right-section sampling+curriculum are enabled
        # - left poses are active when debug_disable_left_pose_bank is False
        right_active = bool(self.eef_init["right_section_sampling_enable"]) and bool(self.eef_init["right_section_curriculum_enable"])
        left_active = not bool(self.eef_init["debug_disable_left_pose_bank"])
        # Pose-bank-only reset sampling from merged active pose sets (density-proportional, not 50/50).
        obj_ids = self.env_object_ids[env_ids].long()
        left_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)  # CODEX
        chosen = torch.zeros(num_envs, device=self.device, dtype=torch.long)  # CODEX
        num_sections = int(self.eef_init["right_section_num"])  # CODEX
        active_sections = self._get_active_right_sections() if bool(self.eef_init["right_section_curriculum_enable"]) else num_sections  # CODEX
        for local_id in range(num_envs):  # CODEX
            obj_id = int(obj_ids[local_id].item())
            left_gid = obj_id * 2 + 1
            right_gid = obj_id * 2 + 0
            left_count = int(self.pose_bank_count[left_gid].item())
            right_count = int(self.pose_bank_count[right_gid].item())
            right_valid_ids = torch.empty((0,), dtype=torch.long, device=self.device)
            if right_active and right_count > 0:
                bank_sections = self.pose_bank_section[right_gid, :right_count]
                valid_bank = (bank_sections >= 0) & (bank_sections < active_sections)
                right_valid_ids = valid_bank.nonzero(as_tuple=False).squeeze(-1)
            use_left = left_active and left_count > 0
            use_right = right_active and right_valid_ids.numel() > 0
            if not use_left and not use_right:
                # fallback to original side_mode
                if side_mode == "left" and left_count > 0:
                    use_left = True
                elif side_mode == "right" and right_count > 0:
                    use_right = True
                    right_valid_ids = torch.arange(right_count, device=self.device, dtype=torch.long)
                elif left_count > 0:
                    use_left = True
                elif right_count > 0:
                    use_right = True
                    right_valid_ids = torch.arange(right_count, device=self.device, dtype=torch.long)
                else:
                    raise RuntimeError(f"No available reset poses for obj_id={obj_id}")

            left_pool = left_count if use_left else 0
            right_pool = int(right_valid_ids.numel()) if use_right else 0
            total_pool = left_pool + right_pool
            pick = int(torch.randint(total_pool, (1,), device=self.device).item())
            if pick < left_pool:
                left_mask[local_id] = True
                chosen[local_id] = pick
            else:
                left_mask[local_id] = False
                chosen[local_id] = right_valid_ids[pick - left_pool]

        group_ids = obj_ids * 2 + left_mask.long()  # CODEX
        # update the target_quat of envs based on sampled side mask
        self.side_is_left[env_ids] = left_mask
        target_quat = self.target_quat_right.repeat(num_envs, 1)
        if int(left_mask.sum().item()) > 0:
            target_quat[left_mask] = self.target_quat_left.repeat(int(left_mask.sum().item()), 1)
        self.reward_settings["target_quat"][env_ids] = target_quat
        self.reward_settings["target_rot_6d"][env_ids] = matrix_to_rotation_6d(quaternion_to_matrix_ig(target_quat))

        # CODEX: debug logs for right-section curriculum progress and sampled section distribution.
        debug_log_enable = bool(self.eef_init["right_section_debug_log_enable"])
        if debug_log_enable:
            debug_log_interval = int(self.eef_init["right_section_debug_log_interval_steps"])
        if debug_log_enable and (self.sim_steps - self._right_section_last_log_step >= debug_log_interval):
            sampled_sections = self.pose_bank_section[group_ids, chosen]
            sampled_right_sections = sampled_sections[~left_mask]
            if sampled_right_sections.numel() > 0:
                hist = torch.bincount(sampled_right_sections.clamp_min(0), minlength=num_sections).cpu().tolist()
                print(
                    f"[right-section-debug] sim_steps={int(self.sim_steps)} "
                    f"active_sections={active_sections}/{num_sections} "
                    f"right_samples={int(sampled_right_sections.numel())} "
                    f"section_hist={hist}"
                )
            else:
                print(
                    f"[right-section-debug] sim_steps={int(self.sim_steps)} "
                    f"active_sections={active_sections}/{num_sections} right_samples=0"
                )
            self._right_section_last_log_step = int(self.sim_steps)
        joint_config = self.pose_bank_joint[group_ids, chosen]
        target_quat = self.pose_bank_quat[group_ids, chosen]

        # Optional hand-joint perturbation at reset (radians, uniform in [-noise, noise]).
        hand_joint_reset_noise_deg = float(self.eef_init["hand_joint_reset_noise"])
        hand_joint_reset_noise = float(np.deg2rad(hand_joint_reset_noise_deg))
        if hand_joint_reset_noise > 0.0:
            base = joint_config[:, 7:23]
            lower = self.robot_dof_lower_limits[7:23].unsqueeze(0)
            upper = self.robot_dof_upper_limits[7:23].unsqueeze(0)
            # Sample uniformly in the feasible interval directly (avoid sample-then-clip bias).
            sample_low = torch.max(base - hand_joint_reset_noise, lower)
            sample_high = torch.min(base + hand_joint_reset_noise, upper)
            u = torch.rand((num_envs, 16), device=self.device)
            joint_config[:, 7:23] = sample_low + u * (sample_high - sample_low)

        self.reward_settings["target_quat"][env_ids] = target_quat
        self.reward_settings["target_rot_6d"][env_ids] = matrix_to_rotation_6d(quaternion_to_matrix_ig(target_quat))
        self.set_robot_joint_state(joint_config, env_ids=env_ids)

    def post_physics_step(self):
        super().post_physics_step()
        # @ray visualize debugging stuff
        if self.debug_viz and self.viewer is not None:
            # self._draw_object_xy_range_grid(clear_lines=False)
            self._draw_eef_candidate_points(clear_lines=False)
            self._draw_object_center_cross(clear_lines=False)
            self._draw_workspace_limits_box(clear_lines=False)
            self._draw_right_section_workspace(clear_lines=False)  # CODEX
            # self._draw_ik_reachability_grid(clear_lines=False)
            pass

    def _create_envs(self, spacing, num_per_row):
        """
        loading Franka + LEAP + a table in the environment, this is for debugging purposes only
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # setup params
        table_thickness = self.cfg["env"]["table_thickness"]
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

        obj_xyz_range = self.cfg["env"]["object_settings"]["xyz_range"]
        self.obj_pos_range = torch.zeros((self.num_envs, 4), device=self.device) # x-min, x-max, y-min, y-max
        self.obj_pos_range[:, 0] = obj_xyz_range[0][0] # x-min
        self.obj_pos_range[:, 1] = obj_xyz_range[1][0] # x-max
        self.obj_pos_range[:, 2] = obj_xyz_range[0][1] # y-min
        self.obj_pos_range[:, 3] = obj_xyz_range[1][1] # y-max

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
            # NOTE
            # @ray intentioanlly move it away from base a bit
            table_asset, table_start_pose = self._create_cube(
                pos=[0.5, 0.0, -table_thickness/2],
                size=[0.7, 1.2, table_thickness],
            )
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, 1, 0
            )

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
            static_pcd_i = torch.from_numpy(compute_scene_oracle_pcd(
                num_obstacle_points=self.pcd_spec_dict["num_static_points"],
                cuboid_dims=self.cuboid_dims,
                cuboid_centers=np.array([[table_start_pose.p.x, table_start_pose.p.y, table_start_pose.p.z]]),
                cuboid_quats=np.array([[table_start_pose.r.x, table_start_pose.r.y, table_start_pose.r.z, table_start_pose.r.w]]),
            )).to(self.device)
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

        self.cuboid_dims = np.array(self.cuboid_dims).reshape(self.num_envs, -1, 3)
        self.cuboid_pos = np.array(self.cuboid_pos).reshape(self.num_envs, -1, 3)
        self.cuboid_quats = np.array(self.cuboid_quats).reshape(self.num_envs, -1, 4)

        self.table_surface_height = torch.tensor([table_start_pose.p.z + table_thickness / 2] * self.num_envs, device=self.device)
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
        rotated_offset = torch.matmul(object_rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)
        object_grasp_target_pos += rotated_offset

        eef_rot_mat = quaternion_to_matrix_ig(self._eef_state[:, 3:7])
        eef_rot_mat_t = eef_rot_mat.transpose(1, 2)
        object_grasp_target_to_eef_world = object_grasp_target_pos - self._eef_state[:, :3]
        object_grasp_target_to_eef = torch.matmul(
            eef_rot_mat_t, object_grasp_target_to_eef_world.unsqueeze(-1)
        ).squeeze(-1)

        # @ray not just update but also create new keys here
        self.states.update({
            # Table Contact Status, check whether the object is lifted
            "lift": ~self.table_collision,
            "object_grasp_target_pos": object_grasp_target_pos, # @ray reward-only grasp target
            "object_grasp_target_to_eef": object_grasp_target_to_eef, # @ray for policy observation
            "grasp_side_binary": self.side_is_left.float().unsqueeze(-1),
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
        x_min = float(self.obj_pos_range[env_id, 0].item())
        x_max = float(self.obj_pos_range[env_id, 1].item())
        y_min = float(self.obj_pos_range[env_id, 2].item())
        y_max = float(self.obj_pos_range[env_id, 3].item())
        z = float(self.table_surface_height[env_id].item())
        num_lines = int(self.eef_init.get("range_grid_lines", 6))
        xs = np.linspace(x_min, x_max, num_lines)
        ys = np.linspace(y_min, y_max, num_lines)
        verts_flat = []
        for x in xs:
            verts_flat.extend([x, y_min, z, x, y_max, z])
        for y in ys:
            verts_flat.extend([x_min, y, z, x_max, y, z])
        colors_flat = [0.0, 1.0, 0.0] * (len(verts_flat) // 6)
        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts_flat) // 6, verts_flat, colors_flat)

    def _draw_eef_candidate_points(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        max_draw = int(1000)
        cross_size = float(0.01)
        obj_id = int(self.env_object_ids[env_id].item())
        left_group_id = obj_id * 2 + 1
        right_group_id = obj_id * 2 + 0
        draw_groups = []
        if self.eef_init["side_mode"] == "left" and bool(self.eef_init["right_section_sampling_enable"]):  # CODEX
            # CODEX: in left-grasp mode, visualize both left bank and right-section bank.
            draw_groups = [(left_group_id, [0.0, 1.0, 0.0]), (right_group_id, [1.0, 1.0, 0.0])]
        else:
            left_side = bool(self.side_is_left[env_id].item()) if hasattr(self, "side_is_left") else False
            group_id = obj_id * 2 + (1 if left_side else 0)
            draw_groups = [(group_id, [0.0, 1.0, 0.0])]

        verts_flat = []
        colors_flat = []
        for group_id, color in draw_groups:
            count = int(self.pose_bank_count[group_id].item())
            if count <= 0:
                continue
            draw_n = min(max_draw // max(len(draw_groups), 1), count)
            sample_idx = torch.randperm(count, device=self.device)[:draw_n]
            arm_joint = self.pose_bank_joint[group_id, sample_idx, :7]
            eef_pose = self.get_ee_from_joint(arm_joint)
            points = eef_pose[:, :3].detach().cpu().numpy()
            for p in points:
                verts_flat.extend([p[0] - cross_size, p[1], p[2], p[0] + cross_size, p[1], p[2]])
                colors_flat.extend(color)
                verts_flat.extend([p[0], p[1] - cross_size, p[2], p[0], p[1] + cross_size, p[2]])
                colors_flat.extend(color)
                verts_flat.extend([p[0], p[1], p[2] - cross_size, p[0], p[1], p[2] + cross_size])
                colors_flat.extend(color)
        if len(verts_flat) == 0:
            return
        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts_flat) // 6, verts_flat, colors_flat)

    # Draws:
    # 1) object center and grasp target crosses
    # 2) x_low/x_high bands along object local z-axis (both + and - directions)
    # 3) projected fingertip points on object z-axis
    # 4) average absolute fingertip height-error marker used by current sigmoid gate
    def _draw_object_center_cross(self, half_extent=0.2, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)

        centers = self.states["object_center_pos"]
        grasp_targets = self.states["object_grasp_target_pos"]
        object_quat = self.states["object_quat"]

        local_z = torch.zeros((self.num_envs, 3), dtype=object_quat.dtype, device=object_quat.device)
        local_z[:, 2] = 1.0
        object_z_axis_world = quat_apply(object_quat, local_z)
        object_z_axis_world = object_z_axis_world / torch.norm(object_z_axis_world, dim=-1, keepdim=True).clamp_min(1e-8)
        # offseted gate target for sigmoid-gate visualization.
        gate_offset = self.reward_settings["grasp_on_object_z_height_offset"]
        gate_targets = grasp_targets + gate_offset * object_z_axis_world

        finger_positions = torch.stack(
            [
                self.states["eef_finger1_pos"],
                self.states["eef_finger2_pos"],
                self.states["eef_finger3_pos"],
                self.states["eef_finger4_pos"],
            ],
            dim=1,
        )  # (N,4,3)
        fingertip_to_target = finger_positions - gate_targets.unsqueeze(1)
        finger_signed_height_err = torch.sum(fingertip_to_target * object_z_axis_world.unsqueeze(1), dim=-1)  # (N,4)
        # keep visualization aligned with reward gate metric.
        # use the exact gate input from reward: x = abs(max(signed_height_err)).
        top_finger_signed_height = torch.max(finger_signed_height_err, dim=1)[0]  # (N,)
        d_finger_height_align = torch.abs(top_finger_signed_height)

        # visualize all envs (same style as base debug helper).
        x_low = float(self.reward_settings["grasp_on_object_z_height_tolerance"][0].item())
        x_high = float(self.reward_settings["grasp_on_object_z_height_tolerance"][1].item())
        err_marker_half = 0.2
        band_cross_half = 0.2
        thick_eps = 0.006

        centers_np = centers.detach().cpu().numpy()
        grasp_targets_np = grasp_targets.detach().cpu().numpy()
        gate_targets_np = gate_targets.detach().cpu().numpy()
        z_axis_np = object_z_axis_world.detach().cpu().numpy()
        finger_pos_np = finger_positions.detach().cpu().numpy()
        finger_signed_err_np = finger_signed_height_err.detach().cpu().numpy()
        d_align_np = d_finger_height_align.detach().cpu().numpy()

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
            g_gate = gate_targets_np[i]
            z_axis = z_axis_np[i]

            verts_flat = []
            colors_flat = []

            # object center cross (red)
            # add_thick_cross(verts_flat, colors_flat, c, half_extent, [1.0, 0.0, 0.0])

            # grasp target cross (cyan)
            add_thick_cross(verts_flat, colors_flat, g, half_extent, [0.0, 0.8, 1.0])
            # offseted gate-target cross (blue).
            add_thick_cross(verts_flat, colors_flat, g_gate, half_extent, [0.0, 0.2, 1.0])

            # draw gate bands at +x_low (green) and +x_high (yellow) from gate target.
            for dist, col in ((x_low, [0.0, 1.0, 0.0]), (x_high, [1.0, 1.0, 0.0])):
                p = g_gate + dist * z_axis
                add_thick_cross(verts_flat, colors_flat, p, band_cross_half, col)

            # draw projected fingertip points onto object z-axis (magenta).
            # for f_idx in range(4):
            #     signed_err = float(finger_signed_err_np[i, f_idx])
            #     p_proj = g + signed_err * z_axis
            #     add_thick_cross(verts_flat, colors_flat, p_proj, band_cross_half, [1.0, 0.0, 1.0])

            #     # Link fingertip to projection for easier visual sanity check.
            #     fp = finger_pos_np[i, f_idx]
            #     verts_flat.extend([fp[0], fp[1], fp[2], p_proj[0], p_proj[1], p_proj[2]])
            #     colors_flat.extend([0.7, 0.2, 1.0])

            p_gate_x = g_gate + float(d_align_np[i]) * z_axis
            add_thick_cross(verts_flat, colors_flat, p_gate_x, err_marker_half, [1.0, 0.5, 0.0])  # orange

            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                len(verts_flat) // 6,
                verts_flat,
                colors_flat,
            )
    
    
    # @ray: visualize workspace XYZ limits as a wireframe box (env 0)
    def _draw_workspace_limits_box(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        # CODEX
        # Match debug box with actual left/right sampling bounds.
        limits_min_rel = torch.tensor(self.eef_init["limits_xyz_min"], device=self.device, dtype=torch.float32)
        limits_max_rel = torch.tensor(self.eef_init["limits_xyz_max"], device=self.device, dtype=torch.float32)
        center = [
            float(self.cuboid_pos[env_id, 0, 0].item()),
            float(self.cuboid_pos[env_id, 0, 1].item()),
            float(self.table_surface_height[env_id].item()),
        ]
        # CODEX
        limits_min = torch.tensor(center, device=self.device, dtype=torch.float32) + limits_min_rel
        limits_max = torch.tensor(center, device=self.device, dtype=torch.float32) + limits_max_rel
        if bool(self.side_is_left[env_id].item()):
            center_y = float(center[1])
            y_min_r = float(limits_min[1].item())
            y_max_r = float(limits_max[1].item())
            limits_min[1] = 2.0 * center_y - y_max_r
            limits_max[1] = 2.0 * center_y - y_min_r
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
        colors_flat = [1.0, 1.0, 0.0] * (len(verts_flat) // 6)
        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts_flat) // 6, verts_flat, colors_flat)

    def _draw_right_section_workspace(self, clear_lines=True):
        # CODEX: draw currently active right-section workspace box for env 0.
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        num_sections = int(self.eef_init["right_section_num"])
        active_sections = self._get_active_right_sections()
        x_center = float(self.cuboid_pos[env_id, 0, 0].item())
        y_center = float(self.cuboid_pos[env_id, 0, 1].item())
        z_center = float(self.table_surface_height[env_id].item())
        limits_min_rel = self.eef_init["limits_xyz_min"]
        limits_max_rel = self.eef_init["limits_xyz_max"]
        x_min = x_center + float(limits_min_rel[0])
        x_max = x_center + float(limits_max_rel[0])
        # derive Y span from actual active right-bank samples, so sign/flip is always correct
        obj_id = int(self.env_object_ids[env_id].item())
        right_group_id = obj_id * 2 + 0
        bank_sections = self.pose_bank_section[right_group_id, :]
        valid_bank = (bank_sections >= 0) & (bank_sections < active_sections)
        valid_ids = valid_bank.nonzero(as_tuple=False).squeeze(-1)
        if valid_ids.numel() > 0:
            y_samples = self.pose_bank_eef_pos[right_group_id, valid_ids, 1]
            y_min = float(torch.min(y_samples).item())
            y_max = float(torch.max(y_samples).item())
        else:
            y_min = y_center
            y_max = y_center
        z_min = z_center + float(limits_min_rel[2])
        z_max = z_center + float(limits_max_rel[2])

        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max],
        ], dtype=np.float32)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        verts_flat = []
        colors_flat = []
        color = [1.0, 0.5, 0.0]  # orange
        for a, b in edges:
            pa = corners[a]
            pb = corners[b]
            verts_flat.extend([pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]])
            colors_flat.extend(color)
        self.gym.add_lines(self.viewer, self.envs[env_id], len(edges), verts_flat, colors_flat)
    
    # @ray: visualize IK reachability grid (env 0)
    def _draw_ik_reachability_grid(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        dtype = self._object_center_init_state.dtype
        center = torch.tensor(
            [
                self.cuboid_pos[env_id, 0, 0].item(),
                self.cuboid_pos[env_id, 0, 1].item(),
                self.table_surface_height[env_id].item(),
            ],
            device=self.device,
            dtype=dtype,
        )
        offsets = [[-0.3, 0.3, 15], [-0.5, 0.5, 25], [0.0, 0.6, 15]]
        xs = torch.linspace(offsets[0][0], offsets[0][1], int(offsets[0][2]), device=self.device)
        ys = torch.linspace(offsets[1][0], offsets[1][1], int(offsets[1][2]), device=self.device)
        zs = torch.linspace(offsets[2][0], offsets[2][1], int(offsets[2][2]), device=self.device)
        grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1).reshape(-1, 3)
        points = center.unsqueeze(0) + grid
        target_quat = self.reward_settings["target_quat"][env_id:env_id + 1].repeat(points.shape[0], 1)
        eef_pose = torch.cat([points, target_quat], dim=-1)
        with torch.enable_grad():
            _, success = self.get_joint_from_ee(eef_pose, return_success=True, use_debug=True)
        success = success.bool().reshape(-1).detach().cpu().numpy()
        pts = points.detach().cpu().numpy()
        cross_size = float(self.eef_init["ik_debug_cross_size"])
        verts_flat = []
        colors_flat = []
        for p, ok in zip(pts, success):
            color = [0.0, 1.0, 0.0] if ok else [1.0, 0.0, 0.0]
            verts_flat.extend([p[0] - cross_size, p[1], p[2], p[0] + cross_size, p[1], p[2]])
            colors_flat.extend(color)
            verts_flat.extend([p[0], p[1] - cross_size, p[2], p[0], p[1] + cross_size, p[2]])
            colors_flat.extend(color)
            verts_flat.extend([p[0], p[1], p[2] - cross_size, p[0], p[1], p[2] + cross_size])
            colors_flat.extend(color)
        if len(verts_flat) == 0:
            return
        self.gym.add_lines(self.viewer, self.envs[env_id], len(verts_flat) // 6, verts_flat, colors_flat)

    def compute_observations(self):
        self._refresh() # @ray checks table collision and updates states

        obs_components = ["q_hand",
                          "grasp_side_binary",
                          "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                          "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                          "object_to_eef", "object_to_eef_rot_6d",
                          "object_grasp_target_to_eef",
                          "target_to_eef", "target_to_eef_rot_6d"]

        states_components = ["q", "qd",
                             "grasp_side_binary",
                             "eef_pos", "eef_rot_6d", "eef_vel",
                             "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                             "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                             "object_to_eef", "object_to_eef_rot_6d",
                             "object_grasp_target_to_eef",
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
        reward_dict = compute_franka_leap_reward(self.states, self.reward_settings)

        self.rew_buf[:] = reward_dict["r_total"]
        self.extras["sep_reward/r_hand_obj"] = torch.mean(reward_dict["r_hand_obj"]).item()
        self.extras["sep_reward/r_obj_goal"] = torch.mean(reward_dict["r_obj_goal"]).item()
        self.extras["sep_reward/r_obj_goal_rot"] = torch.mean(reward_dict["r_obj_goal_rot"]).item()
        self.extras["sep_reward/lift_height_gate"] = torch.mean(reward_dict["lift_height_gate"]).item()
        self.extras["sep_reward/lift_height_gate_soft"] = torch.mean(reward_dict["lift_height_gate_soft"]).item()
        self.extras["sep_reward/r_lift"] = torch.mean(reward_dict["r_lift"]).item()
        self.extras["sep_reward/r_curl"] = torch.mean(reward_dict["r_curl"]).item()
        self.extras["sep_reward/r_actionreg"] = torch.mean(reward_dict["r_actionreg"]).item()
        self.extras["dis/d_hand_obj"] = torch.mean(reward_dict["d_hand_obj"]).item()
        self.extras["dis/d_lift"] = torch.mean(reward_dict["d_lift"]).item()
        self.extras["dis/d_eef_point_goal"] = torch.mean(reward_dict["d_eef_point_goal"]).item()
        self.extras["dis/d_eef_point_goal_rot"] = torch.mean(reward_dict["d_eef_point_goal_rot"]).item()

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
        self.success_long_enough = self.success_duration >= self.reward_settings["success_timeout"]
        self.lifting_long_enough = self.lifting_duration >= self.reward_settings["lifting_timeout"]
        done_envs = self.reset_buf > 0

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

        self.extras["metrics/success_rate_5cm_per_ep"] = torch.mean(self.success_flags.float()).item()
        self.extras["metrics/success_rate_5cm_per_ep_instant"] = torch.mean(self.success_flags_instant).item()
        self.extras["metrics/success_rate_5cm_per_step"] = torch.mean(self.success_5cm_per_step.float()).item()
        self.extras["metrics/lifting_rate_5cm_per_ep"] = torch.mean(self.lifting_flags.float()).item()
        self.extras["metrics/lifting_rate_5cm_per_ep_instant"] = torch.mean(self.lifting_flags_instant).item()
        self.extras["metrics/lifting_rate_5cm_per_step"] = torch.mean(self.lifting_5cm_per_step.float()).item()

        # CODEX: success-based right-section curriculum update + logging.
        self._right_section_last_success_rate = float(self.extras["metrics/success_rate_5cm_per_ep"])
        self._update_right_section_curriculum_from_success()
        self.extras["curriculum/right_section_active_sections"] = float(self._get_active_right_sections())
        self.extras["curriculum/right_section_success_hold_counter"] = float(self._right_section_success_hold_counter)
        self.extras["curriculum/right_section_success_threshold"] = float(
            self.eef_init["right_section_curriculum_success_threshold"]
        )
        
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

    # CODEX: Optional hand-object distance branch that uses XY distance at the midpoint of (max,min) finger heights.
    use_midheight_xy = bool(reward_settings["hand_obj_use_midheight_xy"])
    if use_midheight_xy:
        object_target_pos = states["object_grasp_target_pos"]
        finger_to_target = finger_positions - object_target_pos.unsqueeze(1)
        finger_signed_height = torch.sum(finger_to_target * object_z_axis_world.unsqueeze(1), dim=-1)
        mid_finger_height = 0.5 * (
            torch.max(finger_signed_height, dim=1)[0] + torch.min(finger_signed_height, dim=1)[0]
        )
        # CODEX: tangential (radial-to-axis) component in object frame.
        finger_radial = finger_to_target - torch.sum(
            finger_to_target * object_z_axis_world.unsqueeze(1), dim=-1, keepdim=True
        ) * object_z_axis_world.unsqueeze(1)
        d_xy = torch.mean(torch.norm(finger_radial, dim=-1), dim=1)
        # CODEX: separate height-consistency term so midpoint height actually affects reward.
        d_height = torch.mean(torch.abs(finger_signed_height - mid_finger_height.unsqueeze(1)), dim=1)
        d_hand_obj = d_xy + reward_settings["hand_obj_midheight_height_weight"] * d_height
    else:
        d_hand_obj = d_hand_obj_max

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
    # print(
    #     "[reward_debug] x0=", x[0],
    #     "x_low=", x_low,
    #     "x_high=", x_high,
    #     "x_mid=", x_mid,
    #     "s=", s,
    #     "gate0=", lift_height_gate[0],
    #     "r_lift_before0=", r_lift_before_gate[0],
    #     "r_lift_after0=", r_lift[0],
    # )

    # R4: Hand orientation reward (based on average point matching distance)
    d_eef_point_goal_hand = states["point_matching_err_hand"]
    beta_hand_orientation = reward_settings["beta_hand_orientation"]
    r_hand_orientation = torch.exp(-beta_hand_orientation * d_eef_point_goal_hand)

    # R5: Finger curl
    hand_dof_pos = states["q"][:, 7:] # hand joint angles
    near_object = (d_hand_obj <= reward_settings["curl_reaching_threshold"])
    finger_pos_diff = torch.sum((hand_dof_pos - reward_settings["grasp_finger_dof_pos"]) ** 2, dim=1)

    beta_curl = reward_settings["beta_curl"]
    r_curl= torch.exp(-beta_curl * finger_pos_diff)
    r_curl = torch.where(near_object, r_curl, 0.0)

    # R6: Velocity Regularization/Penalty
    actionreg = states["actionreg"]
    r_actionreg = torch.sum(actionreg**2, dim=-1)

    w_hand_obj = reward_settings["w_hand_obj"]
    w_obj_goal = reward_settings["w_obj_goal"]
    w_hand_orientation = reward_settings["w_hand_orientation"]
    w_lift = reward_settings["w_lift"]
    w_curl = reward_settings["w_curl"]
    w_actionreg = reward_settings["w_actionreg"]


    use_curl = bool(reward_settings["use_curl"])
    # @ray 
    # use activated rewards only
    # but compute all rewards anyways for logging
    # @ray using the same weight for obj_goal position and rotation, can be changed later if needed
    r_total =w_hand_obj*r_hand_obj + w_obj_goal*r_obj_goal + w_hand_orientation*r_hand_orientation + \
              w_curl*r_curl * float(use_curl) + \
              w_lift*r_lift + w_actionreg*r_actionreg
    
    rewards = {
        "r_hand_obj": w_hand_obj*r_hand_obj,
        "r_lift": w_lift*r_lift,
        "r_obj_goal": w_obj_goal*r_obj_goal,
        "r_obj_goal_rot": w_hand_orientation*r_hand_orientation,
        "r_curl": w_curl*r_curl,
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
