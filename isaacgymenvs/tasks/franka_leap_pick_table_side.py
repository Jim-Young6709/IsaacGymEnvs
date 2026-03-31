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
        # @ray precompute a reset pose bank to reuse
        # this is better than resampling during each reset, far less compute
        # however, need to potentially modify bank size for different sampling regions
        self._create_reset_pose_bank()
    
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

    def _create_reset_pose_bank(self):
        self.pose_bank_size = int(self.eef_init["pose_bank_size"])
        bank_batch = self.pose_bank_size
        max_rounds = int(self.eef_init["pose_bank_max_rounds"])
        num_groups = int(self.num_objects * 2)

        # @ray build left/right/both pose banks per object
        # This assumes that all instances of the same object shares the same scale
        side_mode = self.eef_init["side_mode"]
        side_build = (
            [True] if side_mode == "left"
            else [False] if side_mode == "right"
            else [False, True]
        )

        # @ray initialize left+right banks, we may only use half if side_mode!=both
        # but no need to optimize for memory as this is pretty cheap 
        # (~30mb) for 4096 bank size of 69 objects
        self.pose_bank_joint = torch.zeros((num_groups, self.pose_bank_size, self.num_dofs), device=self.device)
        self.pose_bank_quat = torch.zeros((num_groups, self.pose_bank_size, 4), device=self.device)
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
                limits_min = torch.tensor(self.eef_init["limits_xyz_min"], device=self.device)
                limits_max = torch.tensor(self.eef_init["limits_xyz_max"], device=self.device)
                # @ray HACK assumes fixed object start position at table center
                table_center = torch.tensor(
                    [
                        self.cuboid_pos[rep_env_id, 0, 0].item(),
                        self.cuboid_pos[rep_env_id, 0, 1].item(),
                        self.table_surface_height[rep_env_id].item(),
                    ],
                    device=self.device)
                limits_min = table_center + limits_min
                limits_max = table_center + limits_max
                side_distance = float(self.eef_init["side_distance"])
                side_distance_noise = float(self.eef_init["side_distance_noise"])
                target_quat_noise_deg = float(self.eef_init["target_quat_noise_deg"])
                target_quat_noise_rad = float(np.deg2rad(target_quat_noise_deg))
                center = self._object_center_init_state[rep_env_id]
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
                        if left_side:
                            # left fan: [-overlap, +pi/2]
                            phi = torch.rand(batch_size, device=self.device) * torch.pi
                        else:
                            # right fan: [-pi/2, +overlap]
                            phi = -torch.pi + torch.rand(batch_size, device=self.device) * torch.pi
                            
                        dirs = torch.stack(
                            [torch.sin(theta) * torch.cos(phi), torch.sin(theta) * torch.sin(phi), torch.cos(theta)],
                            dim=-1,
                        )
                        points = center + dirs * radius.unsqueeze(-1)
                        valid = ((points >= limits_min) & (points <= limits_max)).all(dim=-1)
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
                    quat_ok = target_quat[ok]
                    num_ok = int(arm_ok.shape[0])
                    count = int(self.pose_bank_count[group_id].item())
                    take = min(self.pose_bank_size - count, num_ok)
                    joint_block = torch.zeros((take, self.num_dofs), device=self.device)
                    joint_block[:, :7] = arm_ok[:take]
                    self.pose_bank_joint[group_id, count:count + take] = joint_block
                    self.pose_bank_quat[group_id, count:count + take] = quat_ok[:take]
                    self.pose_bank_count[group_id] = count + take
                pbar.update(1)
        pbar.close()
    
    def reset_idx(self, env_ids=None):
        super().reset_idx(env_ids)
        if env_ids is None: # @ray already computed in super()
            env_ids = torch.arange(self.num_envs, device=self.device)
        num_envs = len(env_ids)
        side_mode = self.eef_init["side_mode"]
        if side_mode == "left":
            left_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        elif side_mode == "right":
            left_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        else:
            left_mask = torch.rand(num_envs, device=self.device) < 0.5

        # update the target_quat of envs based on mode ("left/right/both")
        self.side_is_left[env_ids] = left_mask
        target_quat = self.target_quat_right.repeat(num_envs, 1)
        if int(left_mask.sum().item()) > 0:
            target_quat[left_mask] = self.target_quat_left.repeat(int(left_mask.sum().item()), 1)
        self.reward_settings["target_quat"][env_ids] = target_quat
        self.reward_settings["target_rot_6d"][env_ids] = matrix_to_rotation_6d(quaternion_to_matrix_ig(target_quat))

        obj_ids = self.env_object_ids[env_ids].long()
        group_ids = obj_ids * 2 + left_mask.long()
        chosen = torch.randint(0, self.pose_bank_size, (num_envs,), device=self.device)
        joint_config = self.pose_bank_joint[group_ids, chosen]
        target_quat = self.pose_bank_quat[group_ids, chosen]

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
            # self._draw_workspace_limits_box(clear_lines=False)
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

        # @ray not just update but also create new keys here
        self.states.update({
            # Table Contact Status, check whether the object is lifted
            "lift": ~self.table_collision,
            "object_grasp_target_pos": object_grasp_target_pos, # @ray reward-only grasp target
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
        # visualize stored pose-bank entries for env 0's current object-side group.
        left_side = bool(self.side_is_left[env_id].item()) if hasattr(self, "side_is_left") else False
        obj_id = int(self.env_object_ids[env_id].item())
        group_id = obj_id * 2 + (1 if left_side else 0)
        count = int(self.pose_bank_count[group_id].item())
        if count <= 0:
            # this should not happen if bank construction succeeded.
            return
        draw_n = min(max_draw, count)
        sample_idx = torch.randperm(count, device=self.device)[:draw_n]
        arm_joint = self.pose_bank_joint[group_id, sample_idx, :7]
        eef_pose = self.get_ee_from_joint(arm_joint)
        points = eef_pose[:, :3].detach().cpu().numpy()

        verts_flat = []
        colors_flat = []
        color = [0.0, 1.0, 0.0]
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
    
    # @ray: visualize workspace XYZ limits as a wireframe box (env 0)
    def _draw_workspace_limits_box(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        env_id = 0
        limits_min = self.eef_init["limits_xyz_min"]
        limits_max = self.eef_init["limits_xyz_max"]
        center = [
            float(self.cuboid_pos[env_id, 0, 0].item()),
            float(self.cuboid_pos[env_id, 0, 1].item()),
            float(self.table_surface_height[env_id].item()),
        ]
        x_min, y_min, z_min = [center[0] + limits_min[0], center[1] + limits_min[1], center[2] + limits_min[2]]
        x_max, y_max, z_max = [center[0] + limits_max[0], center[1] + limits_max[1], center[2] + limits_max[2]]
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
                          "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                          "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                          "object_to_eef", "object_to_eef_rot_6d",
                          "target_to_eef", "target_to_eef_rot_6d"]

        states_components = ["q", "qd",
                             "eef_pos", "eef_rot_6d", "eef_vel",
                             "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                             "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                             "object_to_eef", "object_to_eef_rot_6d",
                             "target_to_eef", "target_to_eef_rot_6d"]

        obs_buf = torch.cat([self.states[ob] for ob in obs_components], dim=-1)
        states_buf = torch.cat([self.states[st] for st in states_components], dim=-1)

        # @ray optionally append object bbox info
        if self.cfg["observation"]["usage"]["use_z"]:
            obj_height = self.mesh_aabb_extents[:, 2:3]
            obs_buf = torch.cat([obs_buf, obj_height], dim=-1)
            states_buf = torch.cat([states_buf, obj_height], dim=-1)
        if self.cfg["observation"]["usage"]["use_xy"]:
            obj_xy_bbox = self.mesh_aabb_extents[:, :2]
            obs_buf = torch.cat([obs_buf, obj_xy_bbox], dim=-1)
            states_buf = torch.cat([states_buf, obj_xy_bbox], dim=-1)

        obs_buf = torch.cat([obs_buf, self.mesh_aabb_extents], dim=-1)
        states_buf = torch.cat([states_buf, self.mesh_aabb_extents], dim=-1)

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

    # R3: Object goal distance reward (based on average point matching distance)
    d_eef_point_goal_target = states["point_matching_err_target"]
    beta_object_goal = reward_settings["beta_object_goal"]
    r_obj_goal = torch.exp(-beta_object_goal * d_eef_point_goal_target)
    r_obj_goal = torch.where(states["lift"], r_obj_goal, 0.0)

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
