"""
Franka + LEAP Hand Pick Env
TODO:1. clean up
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
from isaacgymenvs.tasks import FrankaLEAPMobileDistillation
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from tqdm import tqdm
import wandb

class FrankaLEAPMobileDistillationPickSide(FrankaLEAPMobileDistillation):
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
        # Use teacher side-distance noise as switching tolerance band for side grasp handover.
        self.switch_tol = float(cfg["env"]["eef_init"]["side_distance_noise"])
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )
        

    def reset_idx(self, env_ids=None):
        super().reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return

        pending_reset_mask = self.object_reset_pending_mask  # CODEX

        # Retarget XY for all object resets in this step (regular reset + teleport).
        if torch.any(pending_reset_mask):
            self.reward_settings["target_pos"][pending_reset_mask, :2] = self._object_center_init_state[pending_reset_mask, :2]
            pending_reset_env_ids = pending_reset_mask.nonzero(as_tuple=False).squeeze(-1)  # CODEX
            reset_related_env_ids = torch.unique(torch.cat([env_ids, pending_reset_env_ids], dim=0))  # CODEX
        else:
            reset_related_env_ids = env_ids  # CODEX

        self.pre_teacher_stage_reached[reset_related_env_ids] = False  # CODEX
        self.switch_target_quat_latched[reset_related_env_ids] = False  # CODEX: force switching target refresh after teleport/reset.
        self.switching_eef_init_pos[reset_related_env_ids] = self._eef_state[reset_related_env_ids, :3]  # CODEX
        if self.use_center_tracking_switch_target:  # CODEX
            self.side_is_left[reset_related_env_ids] = True  # CODEX: center-tracking mode is left-only by design.
        self._set_reward_target_quat_from_side_mask(reset_related_env_ids)  # CODEX: refresh teacher target quat on teleport/reset.

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
            center_radius = float(self.eef_init["side_distance"])  # CODEX
            self.switch_activate_radius = torch.full((self.num_envs,), center_radius, device=self.device, dtype=object_center_pos.dtype)  # CODEX
            return  # CODEX
        
        # CODEX: two-stage switching target for non-center mode.
        # If object starts on hand-left: directly use left target.
        # If object starts on hand-right: stage-1 to right target, then stage-2 to left target.
        side_distance = float(self.eef_init["side_distance"])
        self.switching_target_z_offset = 0.3
        left_offset = torch.zeros((self.num_envs, 3), device=self.device, dtype=object_center_pos.dtype)  # CODEX
        right_offset = torch.zeros((self.num_envs, 3), device=self.device, dtype=object_center_pos.dtype)  # CODEX
        left_offset[:, 1] = side_distance  # CODEX
        right_offset[:, 1] = -side_distance  # CODEX
        left_offset[:, 2] = self.switching_target_z_offset  # CODEX
        right_offset[:, 2] = self.switching_target_z_offset  # CODEX

        left_target_pos = object_center_pos + left_offset  # CODEX
        right_target_pos = object_center_pos + right_offset  # CODEX
        left_target_quat = self.target_quat_left.repeat(self.num_envs, 1)  # CODEX
        right_target_quat = self.target_quat_left.repeat(self.num_envs, 1)  # CODEX: keep same orientation for stage-1 and stage-2.

        # Decide left/right by initial hand-object relation to avoid per-step side flapping.
        object_is_left = object_center_pos[:, 1] >= self.switching_eef_init_pos[:, 1]  # CODEX
        object_is_right = ~object_is_left  # CODEX
        self.pre_teacher_stage_reached[object_is_left] = True  # CODEX: left side goes directly to stage-2.

        # Stage-1 completion for right-start envs: match right-side target first.
        if torch.any(object_is_right):  # CODEX
            right_matching_err = self._get_eef_point_matching_err(  # CODEX
                curent_eef_pos7=self._eef_state[:, :7],
                target_eef_pos7=torch.cat([right_target_pos, right_target_quat], dim=-1),
            )
            reached_right_stage = object_is_right & (right_matching_err <= self.pre_teacher_stage1_tol)  # CODEX
            self.pre_teacher_stage_reached[reached_right_stage] = True  # CODEX

        use_left_stage = object_is_left | self.pre_teacher_stage_reached  # CODEX
        self.switching_target_pos = torch.where(  # CODEX
            use_left_stage.unsqueeze(-1), left_target_pos, right_target_pos
        )
        self.switching_target_quat = torch.where(  # CODEX
            use_left_stage.unsqueeze(-1), left_target_quat, right_target_quat
        )

        # Keep the larger switching hysteresis radii.
        switch_radius = torch.norm(self.switching_target_pos - object_center_pos, dim=-1)
        self.switch_activate_radius = switch_radius + float(self.eef_init["side_distance_noise"])

    def init_data(self, actor_num):
        super().init_data(actor_num)
        self.side_is_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.pre_teacher_stage_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.switch_target_quat_latched = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  # CODEX
        self.switching_eef_init_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)  # CODEX
        self.switching_target_quat_latched_value = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)  # CODEX
        self.switching_target_pos_latched_value = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)  # CODEX
        self.ep_hist_window_sizes = [max(1, self.num_envs // 2), self.num_envs, self.num_envs * 2, self.num_envs * 4]  # CODEX
        self.ep_hist_max = int(self.ep_hist_window_sizes[-1])  # CODEX
        self.ep_hist_success = torch.full((self.ep_hist_max,), -1, device=self.device, dtype=torch.int8)  # CODEX
        self.ep_hist_lifting = torch.full((self.ep_hist_max,), -1, device=self.device, dtype=torch.int8)  # CODEX
        self.ep_hist_ptr = 0  # CODEX
        self.ep_hist_count = 0  # CODEX
        self.pre_teacher_stage1_tol = self.switch_tol*2

        # Teacher-consistent target position semantics:
        # use fixed cfg reward target_pos instead of parent dynamic obj_pos_target.
        static_target_pos = to_torch(
            self.cfg["reward"]["params"]["target_pos"], device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)
        static_target_pos[:, 2] = self.table_surface_height + static_target_pos[:, 2]
        self.reward_settings["target_pos"] = static_target_pos


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
        self.reward_settings["success_tolerance"] = to_torch(self.cfg["reward"]["params"]["success_tolerance"], device=self.device)

        # Initialize right-side-only target quaternions for all envs.
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.side_is_left[:] = True
        self._set_reward_target_quat_from_side_mask(all_env_ids)


    def post_physics_step(self):
        super().post_physics_step()
        # Visualize switching target frame for debugging in viewer mode.
        if self.debug_viz:
            self.gym.clear_lines(self.viewer)
            debug_env_id = 1  # CODEX
            # self._draw_base_init_pose_grid(env_id=debug_env_id, clear_lines=False)
            self._draw_switching_target_pose(env_id=debug_env_id, axis_len=0.10, clear_lines=False)
            self._draw_eef_quat_at_teacher_target_pose(env_id=debug_env_id, axis_len=0.095, clear_lines=False)  # CODEX
            self._draw_teacher_target_pose(env_id=debug_env_id, axis_len=0.08, clear_lines=False)
            #self._draw_object_grasp_center(env_id=0, cross_len=0.2, clear_lines=False)
            # self._draw_observation_rays_from_eef(env_id=debug_env_id, clear_lines=False)  # CODEX
            pass

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
        # Side-grasp switching override:
        # 1) Activate teacher when EEF pose is within switch_tol of switching target pose.
        # 2) Keep teacher active while EEF remains within side_distance + side_distance_noise of object center.
        # 3) Reset-step envs always stay on fabric.
        activated_now = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        if self.enable_fabric:
            eef_pos = self._eef_state[:, :3]
            obj_center = self.states["object_center_pos"]
            radial_dist = torch.norm(eef_pos - obj_center, dim=-1)

            # CODEX: full pose matching for switch gate (includes position and orientation).
            switching_matching_err = self._get_eef_point_matching_err(
                curent_eef_pos7=self._eef_state[:, :7],
                target_eef_pos7=torch.cat([self.switching_target_pos, self.switching_target_quat], dim=-1),  # CODEX
            )

            teacher_active_prev = ~self.fabric_switch_enable
            if self.use_center_tracking_switch_target:  # CODEX
                # CODEX: require both proximity and pose alignment before teacher handover.
                activate_now = (radial_dist <= self.switch_activate_radius) & (switching_matching_err <= self.switch_tol)  # CODEX
            else:  # CODEX
                activate_now = self.pre_teacher_stage_reached & (switching_matching_err <= self.switch_tol)  # CODEX
            keep_radius = float(self.eef_init["side_distance"]) + float(self.eef_init["side_distance_noise"])
            keep_active = teacher_active_prev & (radial_dist <= keep_radius)
            teacher_active = (activate_now | keep_active) & (self.progress_buf > 0)

            # CODEX: after teleport/reset this step, if object is on hand-right, force re-entry to fabric two-stage.
            if not self.use_center_tracking_switch_target:
                reset_event_mask = self.object_reset_pending_mask.clone()
                if torch.any(reset_event_mask):
                    object_on_right = obj_center[:, 1] < eef_pos[:, 1]
                    force_fabric_mask = reset_event_mask & object_on_right
                    teacher_active[force_fabric_mask] = False


            self.fabric_switch_enable[:] = ~teacher_active
            self.fabric_switch_enable[self.progress_buf == 0] = True
            activated_now = (~teacher_active_prev) & teacher_active

        # Keep reward target updates exactly as before: update target pose on activation edge only.
        active_ids = activated_now.nonzero(as_tuple=False).squeeze(-1)
        if active_ids.numel() > 0:
            obj_center_active = self.states["object_center_pos"][active_ids]
            eef_pos_active = self._eef_state[active_ids, :3]
            desired_dir = obj_center_active - eef_pos_active
            desired_dir = desired_dir / torch.norm(desired_dir, dim=-1, keepdim=True).clamp_min(1e-6)  # CODEX

            self.reward_settings["target_pos"][active_ids, :2] = obj_center_active[:, :2]

            left_mask = self.side_is_left[active_ids]
            num_active = active_ids.numel()
            base_quat_right = self.target_quat_right.repeat(num_active, 1)
            base_quat_left = self.target_quat_left.repeat(num_active, 1)
            if self.use_center_tracking_switch_target:  # CODEX
                # CODEX: center-tracking mode is strictly left-canonical for teacher target orientation.
                self.side_is_left[active_ids] = True
                left_mask = torch.ones_like(left_mask, dtype=torch.bool)
                base_quat = base_quat_left
            else:
                base_quat = torch.where(left_mask.unsqueeze(-1), base_quat_left, base_quat_right)

            ref_dir = torch.zeros((num_active, 3), device=self.device, dtype=desired_dir.dtype)
            ref_dir[:, 1] = torch.where(
                left_mask,
                -torch.ones_like(left_mask, dtype=desired_dir.dtype),
                torch.ones_like(left_mask, dtype=desired_dir.dtype),
            )
            # CODEX: full directional alignment (teacher checkpoint convention).
            cross = torch.cross(ref_dir, desired_dir, dim=-1)  # CODEX
            dot = torch.sum(ref_dir * desired_dir, dim=-1, keepdim=True)  # CODEX
            q_align = torch.cat([cross, 1.0 + dot], dim=-1)  # CODEX
            q_align = q_align / torch.norm(q_align, dim=-1, keepdim=True).clamp_min(1e-6)  # CODEX
            target_quat_active = quat_mul(q_align, base_quat)  # CODEX
            target_quat_active = target_quat_active / torch.norm(target_quat_active, dim=-1, keepdim=True).clamp_min(1e-8)

            self.reward_settings["target_quat"][active_ids] = target_quat_active
            self.reward_settings["target_rot_6d"][active_ids] = matrix_to_rotation_6d(
                quaternion_to_matrix_ig(target_quat_active)
            )
            eef_pos_active = self._eef_state[active_ids, :3]
            target_to_eef_world_active = self.reward_settings["target_pos"][active_ids] - eef_pos_active
            eef_rot_mat_active = quaternion_to_matrix_ig(self._eef_state[active_ids, 3:7])
            eef_rot_mat_t_active = eef_rot_mat_active.transpose(1, 2)
            if self.teacher_use_eef_frame:
                target_to_eef_active = torch.matmul(
                    eef_rot_mat_t_active, target_to_eef_world_active.unsqueeze(-1)
                ).squeeze(-1)
            else:
                target_to_eef_active = target_to_eef_world_active

            target_rot_mat_active = quaternion_to_matrix_ig(target_quat_active)
            target_rot_mat_in_eef_active = torch.matmul(eef_rot_mat_t_active, target_rot_mat_active)
            target_to_eef_rot_6d_active = matrix_to_rotation_6d(target_rot_mat_in_eef_active)

            self.states["target_to_eef"][active_ids] = target_to_eef_active
            self.states["target_to_eef_rot_6d"][active_ids] = target_to_eef_rot_6d_active

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

        # @ray not just update but also create new keys here
        # Binary grasp-side indicator for policy observation: 1=left, 0=right.
        grasp_side = self.side_is_left.to(dtype=object_grasp_target_to_eef.dtype).unsqueeze(-1)
        self.states.update({
            "object_grasp_target_pos": object_grasp_target_pos, # @ray reward-only grasp target
            "object_grasp_target_to_eef":  object_grasp_target_to_eef, # @ray for policy observation
            "grasp_side": grasp_side,
        })

    
    def compute_observations(self):
        self._refresh() # @ray checks table collision and updates states

        obs_components = ["q_hand",
                          "grasp_side",
                          "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                          "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                          "object_to_eef", "object_to_eef_rot_6d",
                          "object_grasp_target_to_eef",
                          "target_to_eef", "target_to_eef_rot_6d"]

        states_components = ["q", "qd",
                             "grasp_side",
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
        # @ray states used are updated in compute_observations(), called right before compute_reward()

        self.reset_buf[:] = torch.where((self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf[self.states['object_center_pos'][:, 2] < self.table_surface_height-0.1] = 1

        reward_dict = compute_franka_leap_reward(self.states, self.reward_settings)

        self.rew_buf[:] = reward_dict["r_total"]
        self.extras["sep_reward/r_hand_obj"] = torch.mean(reward_dict["r_hand_obj"]).item()
        self.extras["sep_reward/r_obj_goal"] = torch.mean(reward_dict["r_obj_goal"]).item()
        self.extras["sep_reward/r_obj_goal_rot"] = torch.mean(reward_dict["r_obj_goal_rot"]).item()
        self.extras["sep_reward/lift_height_gate"] = torch.mean(reward_dict["lift_height_gate"]).item()
        self.extras["sep_reward/lift_height_gate_soft"] = torch.mean(reward_dict["lift_height_gate_soft"]).item()
        self.extras["sep_reward/r_lift"] = torch.mean(reward_dict["r_lift"]).item()
        self.extras["sep_reward/r_curl"] = torch.mean(reward_dict["r_curl"]).item()
        self.extras["sep_reward/r_colli"] = torch.mean(reward_dict["r_colli"]).item()
        self.extras["sep_reward/r_actionreg"] = torch.mean(reward_dict["r_actionreg"]).item()
        self.extras["dis/d_hand_obj"] = torch.mean(reward_dict["d_hand_obj"]).item()
        self.extras["dis/d_lift"] = torch.mean(reward_dict["d_lift"]).item()
        self.extras["dis/d_eef_point_goal"] = torch.mean(reward_dict["d_eef_point_goal"]).item()
        self.extras["dis/d_eef_point_goal_rot"] = torch.mean(reward_dict["d_eef_point_goal_rot"]).item()

        # log metrics
        # CODEX: use object-table contact for lifting metric (debug/metrics), not pure height.
        self.lifting_5cm_per_step = ~self.table_collision
        self.lifting_flags_instant[self.lifting_5cm_per_step] = 1
        self.success_5cm_per_step = (reward_dict["d_eef_point_goal"] < self.reward_settings["success_tolerance"]) & self.lifting_5cm_per_step
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
            n_done = int(done_env_ids.numel())
            if n_done > 0:
                write_idx = (torch.arange(n_done, device=self.device) + self.ep_hist_ptr) % self.ep_hist_max
                self.ep_hist_success[write_idx] = done_success_bits
                self.ep_hist_lifting[write_idx] = done_lifting_bits
                self.ep_hist_ptr = (self.ep_hist_ptr + n_done) % self.ep_hist_max
                self.ep_hist_count = min(self.ep_hist_count + n_done, self.ep_hist_max)

        # @ray log per-object per-interval success rates locally and a histograom to wandb
        if self.sim_steps > 0 and (self.sim_steps % self.log_per_object_success_freq == 0):
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
        self.extras["metrics/success_rate_5cm_per_ep"] = float(success_eps) / float(total_eps)
        self.extras["metrics/success_rate_5cm_per_ep_instant"] = torch.mean(self.success_flags_instant).item()
        self.extras["metrics/success_rate_5cm_per_step"] = torch.mean(self.success_5cm_per_step.float()).item()
        self.extras["metrics/lifting_rate_5cm_per_ep"] = float(lifting_eps) / float(total_eps)
        self.extras["metrics/lifting_rate_5cm_per_ep_instant"] = torch.mean(self.lifting_flags_instant).item()
        self.extras["metrics/lifting_rate_5cm_per_step"] = torch.mean(self.lifting_5cm_per_step.float()).item()

        # CODEX: windowed episode metrics (empty slots are ignored, never counted as failures).
        window_key_suffixes = ["envs_div2", "envs", "envs_x2", "envs_x4"]  # CODEX
        for win_size, key_suffix in zip(self.ep_hist_window_sizes, window_key_suffixes):  # CODEX
            win_len = min(int(win_size), int(self.ep_hist_count))  # CODEX
            if win_len > 0:  # CODEX
                idx = (torch.arange(win_len, device=self.device) + (self.ep_hist_ptr - win_len)) % self.ep_hist_max  # CODEX
                succ_vals = self.ep_hist_success[idx]  # CODEX
                lift_vals = self.ep_hist_lifting[idx]  # CODEX
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
