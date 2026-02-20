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
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )
        # Use teacher side-distance noise as switching tolerance band for side grasp handover.
        self.switch_tol = float(self.eef_init["side_distance_noise"])

    def reset_idx(self, env_ids=None):
        input_env_ids = env_ids
        super().reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return
        
        # Retarget XY for all object resets in this step (regular reset + teleport).
        object_reset_mask = self.object_reset_pending_mask
        if torch.any(object_reset_mask):
            self.reward_settings["target_pos"][object_reset_mask, :2] = self._object_center_init_state[object_reset_mask, :2]

        # TODO: @ray right-side-only behavior.
        self.side_is_left[env_ids] = False

        self._set_reward_target_quat_from_side_mask(env_ids)

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
 
    def _update_fabric_switching_target(self, object_center_pos):
        # Fixed right-side switching target:
        # position = object_center + [0, -side_distance, 0], quat = target_quat_right.
        self.side_is_left[:] = False
        side_distance = float(self.eef_init["side_distance"])
        side_offset = torch.zeros((self.num_envs, 3), device=self.device, dtype=object_center_pos.dtype)
        side_offset[:, 1] = -side_distance
        self.switching_target_pos = object_center_pos + side_offset
        self.switching_target_pos[:, 2] += 0.1
        self.switching_target_quat = self.target_quat_right.repeat(self.num_envs, 1)

        # rot_local_x_180 = torch.tensor(
        #     [[1.0, 0.0, 0.0, 0.0]] * self.num_envs,
        #     device=self.device,
        #     dtype=direction.dtype,
        # )
        # self.switching_target_quat = rot_local_x_180


    def init_data(self, actor_num):
        super().init_data(actor_num)
        self.side_is_left = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

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

        # Initialize right-side-only target quaternions for all envs.
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.side_is_left[:] = False
        self._set_reward_target_quat_from_side_mask(all_env_ids)


    def post_physics_step(self):
        super().post_physics_step()
        # Visualize switching target frame for debugging in viewer mode.
        if self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self._draw_switching_target_pose(env_id=0, axis_len=0.10, clear_lines=False)
            self._draw_object_grasp_center(env_id=0, cross_len=0.2, clear_lines=False)
            self._draw_observation_rays_from_eef(env_id=0, clear_lines=False)

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

        # These two vectors are in world axes and point from EEF to object center / grasp target.
        object_center_pos = eef_pos + object_to_eef
        object_grasp_target_pos = eef_pos + object_grasp_target_to_eef

        # Reward target position reconstructed from observation vector.
        target_pos = eef_pos + target_to_eef

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

        self.gym.add_lines(self.viewer, self.envs[env_id], 2, verts, colors)

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
        # 1) activate teacher when EEF is within tolerance of fixed switching target position;
        # 2) keep teacher active while EEF remains within side_distance+tol from object center;
        # 3) if outside keep radius, re-enable fabric (except freshly reset envs).
        if self.enable_fabric:
            eef_pos = self._eef_state[:, :3]
            obj_center = self.states["object_center_pos"]
            radial_dist = torch.norm(eef_pos - obj_center, dim=-1)
            dist_to_switch_target = torch.norm(eef_pos - self.switching_target_pos, dim=-1)
            tol = float(self.eef_init["side_distance_noise"])
            keep_radius = float(self.eef_init["side_distance"]) + tol + 0.05

            teacher_active_prev = ~self.fabric_switch_enable
            activate_now = dist_to_switch_target <= tol
            keep_active = teacher_active_prev & (radial_dist <= keep_radius)
            teacher_active = (activate_now | keep_active) & (self.progress_buf > 0)
            self.fabric_switch_enable[:] = ~teacher_active
            self.fabric_switch_enable[self.progress_buf == 0] = True
        
        object_grasp_target_pos = self._object_state[:, :3].clone()
        local_offset = torch.zeros([self.num_envs, 3], dtype=torch.float, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[:, 2] * self.object_grasp_target_z_scale
        object_rot_mat = quaternion_to_matrix_ig(self._object_state[:, 3:7])
        rotated_offset = torch.matmul(object_rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)
        object_grasp_target_pos += rotated_offset


        # @ray not just update but also create new keys here
        self.states.update({
            "object_grasp_target_pos": object_grasp_target_pos, # @ray reward-only grasp target
            "object_grasp_target_to_eef": object_grasp_target_pos - self._eef_state[:, :3], # @ray for policy observation
        })
    
    def compute_observations(self):
        self._refresh() # @ray checks table collision and updates states

        obs_components = ["q_hand",
                          "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                          "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                          "object_to_eef", "object_to_eef_rot_6d",
                          "object_grasp_target_to_eef",
                          "target_to_eef", "target_to_eef_rot_6d"]

        states_components = ["q", "qd",
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
        t1 = time.time()
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