"""
This code is kinda messy for compatibility between Dagger and residual RL, TODO: cleanup later
TODO: try to let everything goes with tensor, not np.array
"""

import time

import hydra
import isaacgym
import numpy as np
import torch
from typing import Tuple
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.franka_mp import FrankaMP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.demo_loader import DemoLoader
from isaacgymenvs.utils.torch_jit_utils import *

from geometrout.primitive import Cuboid, Cylinder, Sphere
from neural_mp.utils.pcd_utils import decompose_scene_pcd_params_obs, compute_scene_oracle_pcd
from neural_mp.utils.geometry import construct_mixed_point_cloud
from neural_mp.real_utils.real_world_collision_checker import FrankaCollisionChecker
from collections import OrderedDict
from omegaconf import DictConfig
from tqdm import tqdm
from fabrics_sim.worlds.voxels import VoxelCounter


def random_quaternion_xyzw():
    u1, u2, u3 = np.random.uniform(0, 1, 3)

    qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    return np.array([qx, qy, qz, qw])


class FrankaMPRRL(FrankaMP):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, num_env_per_env=1):
        self.device = sim_device
        self.no_base_action = False
        self.base_policy_only = cfg["env"]["base_policy_only"]

        # Demo loading
        hdf5_path = cfg["env"]["hdf5_path"]
        self.demo_loader = DemoLoader(hdf5_path, cfg["env"]["numEnvs"])
        self.batch_idx = cfg["env"]["batch_idx"]

        # need to change the logic here (2 layers of reset ; multiple start & goal in one env ; relaunch IG)
        self.batch = self.demo_loader.get_next_batch(batch_idx=self.batch_idx)
        # self.batch = [self.batch[0]] * 16

        self.start_config = torch.zeros((cfg["env"]["numEnvs"], 7), device=self.device)
        self.goal_config = torch.zeros((cfg["env"]["numEnvs"], 7), device=self.device)
        self.obstacle_configs = []
        self.obstacle_handles = []
        self.blocking_obj_handles = []
        self.dynamic_obj_handles = []
        self.max_obstacles = 0
        self.frankacc = FrankaCollisionChecker()

        for env_idx, demo in enumerate(self.batch):
            self.start_config[env_idx] = torch.tensor(demo['states'][0][:7], device=self.device)
            self.goal_config[env_idx] = torch.tensor(demo['states'][0][7:14], device=self.device)

            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)
            self.max_obstacles = max(len(obstacle_config[0]), self.max_obstacles)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.progress_buf = torch.randint(0, self.max_episode_length, (self.num_envs,)).to(self.device)

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.cuboid_dims = []  # xyz
        self.capsule_dims = []  # r, l
        self.sphere_radii = []  # r

        self.x_reset_flag = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)

        # setup franka
        franka_dof_props = self._create_franka()
        franka_asset = self.franka_asset
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # setup moving obstacles
        self.num_blocking_objs = self.cfg["blocking_obj"]["num_obj"]
        blocking_objs_dim_range = self.cfg["blocking_obj"]["dim_range"]
        blocking_dist_range = self.cfg["blocking_obj"]["dist_range"]
        self.blk_vel = np.random.uniform(self.cfg["blocking_obj"]["vel_range"][0], self.cfg["blocking_obj"]["vel_range"][1], (self.num_envs, self.num_blocking_objs))
        self.blk_vel = torch.from_numpy(self.blk_vel).to(self.device, dtype=torch.float32) * self.cfg["sim"]["dt"]
        self.blk_pos = torch.zeros((self.num_envs, self.num_blocking_objs, 3), device=self.device, dtype=torch.float32)
        self.blk_vel_direction = torch.zeros((self.num_envs, self.num_blocking_objs, 3), device=self.device, dtype=torch.float32) # to store the direction of velocity for each blocking object
        self.blocking_dist = np.random.uniform(blocking_dist_range[0], blocking_dist_range[1], (self.num_envs, self.num_blocking_objs))
        self.blocking_dist = torch.from_numpy(self.blocking_dist).to(self.device, dtype=torch.float32)
        self.blk_update_freq = self.cfg["blocking_obj"]["update_freq"]
        self.rand_sphere_idx = torch.zeros((self.num_envs, self.num_blocking_objs, 1), dtype=torch.int64, device=self.device)

        num_dynamic_objs = self.cfg["dynamic_obj"]["num_obj"]
        dynamic_objs_dim_range = self.cfg["dynamic_obj"]["dim_range"]
        dynamic_center_range = self.cfg["dynamic_obj"]["center_range"]
        self.dynamic_objs_init_radius = self.cfg["dynamic_obj"]["init_radius"]
        dynamic_objs_init_pos = torch.tensor(self.cfg["dynamic_obj"]["init_pos"], device=self.device, dtype=torch.float32)
        self.dyn_centers = np.random.uniform(dynamic_center_range[0], dynamic_center_range[1], (self.num_envs, num_dynamic_objs, 3))
        self.dyn_vel = np.random.uniform(self.cfg["dynamic_obj"]["vel_range"][0], self.cfg["dynamic_obj"]["vel_range"][1], (self.num_envs, num_dynamic_objs))
        self.dyn_centers = torch.from_numpy(self.dyn_centers).to(self.device, dtype=torch.float32)
        self.dyn_vel = torch.from_numpy(self.dyn_vel).to(self.device, dtype=torch.float32) * self.cfg["sim"]["dt"]
        self.dyn_vel_direction = torch.zeros((self.num_envs, num_dynamic_objs, 3), device=self.device, dtype=torch.float32) # to store the direction of velocity for each dynamic object

        self.xy_threshold = self.cfg["xy_threshold"]
        self.x_blocking = self.cfg["x_blocking"]
        self.sdf = torch.zeros(self.num_envs, device=self.device)
        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + self.max_obstacles + self.num_blocking_objs + num_dynamic_objs # franka + obstacles
        max_agg_shapes = num_franka_shapes + self.max_obstacles + self.num_blocking_objs + num_dynamic_objs
        self.frankas = []
        self.env_ptrs = []

        self.num_robot_points = self.pcd_spec_dict['num_robot_points']
        self.num_scene_points = self.pcd_spec_dict['num_obstacle_points']
        self.num_moving_points_per_obj = self.pcd_spec_dict['num_moving_obstacle_points_per_obj']
        self.num_moving_points = (self.num_blocking_objs + num_dynamic_objs) * self.num_moving_points_per_obj
        self.num_static_points = self.num_scene_points - self.num_moving_points
        num_target_points = self.pcd_spec_dict['num_target_points']
        self.static_pcds = torch.zeros(self.num_envs, self.num_static_points, 3, device=self.device)
        self.combined_pcds = torch.cat(
            (
                torch.zeros(self.num_robot_points, 4, device=self.device),
                torch.ones(self.num_scene_points, 4, device=self.device),
                2 * torch.ones(num_target_points, 4, device=self.device),
            ),
            dim=0,
        ).repeat(self.num_envs, 1, 1)
        self.moving_pcds = []

        self.obstacle_count = 0
        self.max_objects_per_env = 20

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            
            self.objects_per_env = 0

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            franka_actor = self.gym.create_actor(
                env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create obstacles using initial demo data
            env_obstacles = []
            block_obstacles = []
            dyn_obstacles = []

            (
                cuboid_dims, 
                cuboid_centers, 
                cuboid_quats,
                cylinder_radii, 
                cylinder_heights,
                cylinder_centers,
                cylinder_quats,
                *_
            ) = self.obstacle_configs[i]

            cuboid_dims = cuboid_dims[[0]]
            cuboid_centers = cuboid_centers[[0]]
            cuboid_quats = cuboid_quats[[0]]

            # num_cylinders = len(cylinder_radii) #pausing cylinders due to incorrect spawning. Likely an actor indexing issue.

            num_cubes = len(cuboid_dims)

            # Create actual obstacles with proper sizes
            for j in range(self.max_obstacles):
                if j < num_cubes:
                    # Create obstacle with actual size and position
                    obstacle_asset, obstacle_pose = self._create_cube(
                        pos=cuboid_centers[j].tolist(),
                        size=cuboid_dims[j].tolist(),
                        quat=cuboid_quats[j].tolist()
                    )
                else:
                    # Create minimal placeholder obstacles far away
                    obstacle_asset, obstacle_pose = self._create_cube(
                        pos=[0., 0., -100.0],
                        size=[0.001, 0.001, 0.001],
                        quat=[0, 0, 0, 1]
                    )

                obstacle_actor = self.gym.create_actor(
                    env_ptr,
                    obstacle_asset,
                    obstacle_pose,
                    f"obstacle_{j}",
                    i,
                    1,
                    0
                )
                env_obstacles.append(obstacle_actor)

            # update max_objects_per_envs
            if self.objects_per_env > self.max_objects_per_env:
                self.max_objects_per_env = self.objects_per_env
                
            self.obstacle_handles.append(env_obstacles)

            # init moving obstacles
            moving_cuboids = []
            for j in range(self.num_blocking_objs):
                blocking_objs_dim = np.random.uniform(blocking_objs_dim_range[0], blocking_objs_dim_range[1])
                blocking_objs_pos = [0.5, 0., 0.5]
                blocking_objs_xyzw = random_quaternion_xyzw()
                blocking_asset, blocking_pose = self._create_cube(
                    pos=blocking_objs_pos,
                    size=blocking_objs_dim.tolist(),
                    quat=blocking_objs_xyzw.tolist(),
                )
                blocking_actor = self.gym.create_actor(
                    env_ptr,
                    blocking_asset,
                    blocking_pose,
                    f"blocking_{j}",
                    i,
                    1,
                    0
                )
                if not self.headless:
                    self.gym.set_rigid_body_color(env_ptr, blocking_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 1.0))
                block_obstacles.append(blocking_actor)
                moving_cuboids.append(Cuboid(np.array([0.0, 0.0, 0.0]), blocking_objs_dim, blocking_objs_xyzw[[3, 0, 1, 2]]))

            self.blocking_obj_handles.append(block_obstacles)

            for j in range(num_dynamic_objs):
                dynamic_objs_dim = np.random.uniform(dynamic_objs_dim_range[0], dynamic_objs_dim_range[1])
                vec = torch.randn(3, device=self.device)
                vec = vec / vec.norm() * self.dynamic_objs_init_radius
                dynamic_objs_start_pos = vec + dynamic_objs_init_pos
                if (dynamic_objs_start_pos[0] < self.xy_threshold) and (dynamic_objs_start_pos[1] < self.xy_threshold):
                    dynamic_objs_start_pos[0] = self.xy_threshold
                center_direction = self.dyn_centers[i, j] - dynamic_objs_start_pos
                self.dyn_vel_direction[i, j] = center_direction / center_direction.norm() # randomize direction of velocity
                dynamic_objs_xyzw = random_quaternion_xyzw()
                dynamic_asset, dynamic_pose = self._create_cube(
                    pos=dynamic_objs_start_pos.tolist(),
                    size=dynamic_objs_dim.tolist(),
                    quat=dynamic_objs_xyzw.tolist(),
                )
                dynamic_actor = self.gym.create_actor(
                    env_ptr,
                    dynamic_asset,
                    dynamic_pose,
                    f"dynamic_{j}",
                    i,
                    1,
                    0
                )
                if not self.headless:
                    self.gym.set_rigid_body_color(env_ptr, dynamic_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 1.0))
                dyn_obstacles.append(dynamic_actor)
                moving_cuboids.append(Cuboid(np.array([0.0, 0.0, 0.0]), dynamic_objs_dim, dynamic_objs_xyzw[[3, 0, 1, 2]]))

            self.dynamic_obj_handles.append(dyn_obstacles)

            moving_pcds_i = np.array(construct_mixed_point_cloud(moving_cuboids, num_points=self.num_moving_points_per_obj*len(moving_cuboids), return_point_list=True, even=True))[..., :3]

            # for vectorization, we unified the dim of the pcd of each objects
            self.moving_pcds.append(moving_pcds_i)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.env_ptrs.append(env_ptr)
            self.frankas.append(franka_actor)

            # compute the static scene pcd (currently only consider static scenes)
            self.static_pcds[i] = torch.from_numpy(compute_scene_oracle_pcd(
                num_obstacle_points=self.num_static_points,
                cuboid_dims=cuboid_dims,
                cuboid_centers=cuboid_centers,
                cuboid_quats=cuboid_quats,
            )).to(self.device)
            self.combined_pcds[i, self.num_robot_points:self.num_robot_points+self.num_static_points, :3] = self.static_pcds[i]

        self.moving_pcds = torch.from_numpy(np.array(self.moving_pcds)).to(self.device, dtype=torch.float32)

        # Setting up voxel counter
        voxel_size = 0.15
        num_voxels_x = 20
        num_voxels_y = 20
        num_voxels_z = 20
        x_min = -0.5
        y_min = -0.75
        z_min = -0.25
        # basis_coord_limits = np.array([[-0.75, -1., -0.1], [1.5, 1., 1.25]])

        self.voxel_counter = VoxelCounter(batch_size=self.num_envs,
                                        device=self.device,
                                        voxel_size=voxel_size,
                                        num_voxels_x=num_voxels_x,
                                        num_voxels_y=num_voxels_y,
                                        num_voxels_z=num_voxels_z,
                                        x_min=x_min,
                                        y_min=y_min,
                                        z_min=z_min)

        self.voxel_visit_binary = torch.zeros((self.num_envs, num_voxels_x*num_voxels_y*num_voxels_z), device=self.device)
        self.num_visited_voxels_t0 = torch.zeros((self.num_envs), device=self.device)

        # Setup data
        actor_num = 1 + self.max_obstacles + self.num_blocking_objs + num_dynamic_objs # franka  + obstacles
        self.blocking_obj_indices = torch.tensor(self.blocking_obj_handles, device=self.device, dtype=torch.int32)
        self.dynamic_obj_indices = torch.tensor(self.dynamic_obj_handles, device=self.device, dtype=torch.int32)
        for i in range(self.num_envs):
            self.blocking_obj_indices[i] += actor_num * i
            self.dynamic_obj_indices[i] += actor_num * i
        self.init_data(actor_num=actor_num)

    def _debug_viz_draw(self, pcd=False):
        draw_obstacle_vectors = False

        self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        for i in range(self.num_envs):
            if pcd:
                # draw point clouds
                points = self.combined_pcds[i][2048:6144, :3].cpu().numpy()

                # Parameters
                offset = np.array([0.005, 0.0, 0.0], dtype=np.float32)  # small x-direction offset for line
                num_points = points.shape[0]

                # Prepare flattened vertices list: [x1,y1,z1,x2,y2,z2,...]
                verts_flat = []
                for p in points:
                    p0 = p - offset
                    p1 = p + offset
                    verts_flat.extend([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]])

                # Colors: same RGB for each line
                color = [1.0, 0.0, 0.0]  # red
                colors_flat = color * num_points  # repeat for each line

                # Add lines to viewer
                self.gym.add_lines(
                    self.viewer,
                    self.env_ptrs[i],
                    num_points,     # num_lines = num points
                    verts_flat,     # flat list of start/end points
                    colors_flat     # flat list of RGB triples
                )

            # draw hand frame
            fabric_ee_pose = self.get_ee_from_joint(self.states['q'][:, :7])
            px = (fabric_ee_pose[:, 0:3][i] 
                + quat_apply(fabric_ee_pose[:, 3:7][i], torch.tensor([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()

            py = (fabric_ee_pose[:, 0:3][i] 
                + quat_apply(fabric_ee_pose[:, 3:7][i], torch.tensor([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()

            pz = (fabric_ee_pose[:, 0:3][i] 
                + quat_apply(fabric_ee_pose[:, 3:7][i], torch.tensor([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

            p0 = fabric_ee_pose[:, 0:3][i].cpu().numpy()
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], px[0], px[1], px[2]], 
                [0.85, 0.1, 0.1]
            )
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], py[0], py[1], py[2]], 
                [0.1, 0.85, 0.1]
            )
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], 
                [0.1, 0.1, 0.85]
            )

            # draw goal frame
            fabric_goal_pose = self.get_ee_from_joint(self.goal_config)
            px = (fabric_goal_pose[:, 0:3][i] 
                + quat_apply(fabric_goal_pose[:, 3:7][i], torch.tensor([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()

            py = (fabric_goal_pose[:, 0:3][i] 
                + quat_apply(fabric_goal_pose[:, 3:7][i], torch.tensor([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()

            pz = (fabric_goal_pose[:, 0:3][i] 
                + quat_apply(fabric_goal_pose[:, 3:7][i], torch.tensor([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

            p0 = fabric_goal_pose[:, 0:3][i].cpu().numpy()
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], px[0], px[1], px[2]], 
                [0.85, 0.1, 0.1]
            )
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], py[0], py[1], py[2]], 
                [0.1, 0.85, 0.1]
            )
            self.gym.add_lines(
                self.viewer, self.env_ptrs[i], 1, 
                [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], 
                [0.1, 0.1, 0.85]
            )

    def update_obstacle_configs_from_batch(self, batch_data):
        """Update obstacle configurations from a new batch of demos."""
        self.obstacle_configs = []
        for demo in batch_data:
            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)

    def blk_flashing(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # blk_update_ids = env_ids[self.progress_buf[env_ids] % self.blk_update_freq == 0]
        # update blocking obstacles
        current_configs = self.get_joint_angles()[env_ids]
        torch_spheres = self.frankacc.torch_spheres(current_configs)
        centers = torch_spheres.centers[:, 28:-10, :] # link4 - gripper
        radii = torch_spheres.radii[:, 28:-10]

        self.rand_sphere_idx[env_ids] = torch.randint(low=0, high=radii[:, 8:].shape[1], size=(len(env_ids),self.num_blocking_objs, 1), dtype=torch.int64).to(self.device)

        centers = torch.gather(centers, dim=1, index=self.rand_sphere_idx[env_ids].expand(-1, -1, centers.shape[-1])) # (num_envs, num_obj, 3)
        radii = torch.gather(radii, dim=1, index=self.rand_sphere_idx[env_ids].expand(-1, -1, radii.shape[-1])) # (num_envs, num_obj, 1)

        # flashing position
        center_direction = torch.randn_like(centers)
        center_shift = center_direction / center_direction.norm(dim=-1, keepdim=True) * (radii + self.blocking_dist[env_ids].unsqueeze(-1))
        blocking_objs_pos = centers + center_shift

        flat_blk_indices = self.blocking_obj_indices[env_ids].view(-1)
        flat_root_state = self._root_state.view(-1, 13)
        flat_root_state[flat_blk_indices, 0:3] = blocking_objs_pos.view(-1, 3)

        self.blk_pos = flat_root_state[self.blocking_obj_indices.view(-1), 0:3].view(self.num_envs, self.num_blocking_objs, 3)
        self.blk_vel_direction[env_ids] = 0.0

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(flat_root_state),
            gymtorch.unwrap_tensor(flat_blk_indices),
            flat_blk_indices.numel()
        )

    def blk_chasing(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        # update blocking obstacle vels (chasing phase)
        ischasing = self.progress_buf[env_ids] <= self.blk_update_freq

        current_configs = self.get_joint_angles()[env_ids]
        torch_spheres = self.frankacc.torch_spheres(current_configs)
        centers = torch_spheres.centers[:, 28:, :] # link4 - gripper
        centers = torch.gather(centers, dim=1, index=self.rand_sphere_idx.expand(-1, -1, centers.shape[-1])) # (num_envs, num_obj, 3)

        flat_blk_indices = self.blocking_obj_indices[env_ids].view(-1)
        flat_root_state = self._root_state.view(-1, 13)

        displacement = centers.view(-1, 3) - flat_root_state[flat_blk_indices, 0:3]
        updated_vel_direction = displacement / displacement.norm(dim=-1, keepdim=True)
        self.blk_vel_direction[env_ids[ischasing]] = updated_vel_direction.view(-1, self.num_blocking_objs, 3)[ischasing]

        flat_root_state[flat_blk_indices, 0:3] += self.blk_vel.view(-1).unsqueeze(-1) * self.blk_vel_direction.view(-1, 3)

        self.blk_pos = flat_root_state[flat_blk_indices, 0:3].view(self.num_envs, self.num_blocking_objs, 3)
        # TODO: temporarily save it here, but cleanup later
        if self.x_blocking:
            x_pos = flat_root_state[flat_blk_indices, 0]
            # y_pos = flat_root_state[flat_blk_indices, 1]
            # z_pos = flat_root_state[flat_blk_indices, 2]
            # safety_corr = (x_pos < self.xy_threshold) & (y_pos < self.xy_threshold / 2) & (y_pos > - self.xy_threshold / 2) & (z_pos < self.xy_threshold) & (z_pos > 0.0)
            x_corr = x_pos < self.xy_threshold

            # x_corr = (x_pos > self.xy_threshold - 0.01) & (safety_corr)
            # y_corr_p = (x_pos <= self.xy_threshold - 0.01) & (y_pos > 0) & (safety_corr)
            # y_corr_n = (x_pos <= self.xy_threshold - 0.01) & (y_pos < 0) & (safety_corr)

            flat_root_state[flat_blk_indices[x_corr], 0] = self.xy_threshold
            # flat_root_state[flat_blk_indices[x_corr], 0] = self.xy_threshold
            # flat_root_state[flat_blk_indices[y_corr_p], 1] = self.xy_threshold / 2
            # flat_root_state[flat_blk_indices[y_corr_n], 1] = -self.xy_threshold / 2

            # self.x_reset_flag = flat_root_state[flat_blk_indices, 0] < self.xy_threshold

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(flat_root_state),
            gymtorch.unwrap_tensor(flat_blk_indices),
            flat_blk_indices.numel()
        )

    def update_moving_obstacles_state(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs).to(self.device)

        # update dynamic obstacles
        flat_dyn_indices = self.dynamic_obj_indices[env_ids].view(-1)
        flat_root_state = self._root_state.view(-1, 13)
        flat_root_state[flat_dyn_indices, 0:3] += self.dyn_vel_direction[env_ids].view(-1, 3) * self.dyn_vel[env_ids].view(-1).unsqueeze(1)

        center_dist = (flat_root_state[self.dynamic_obj_indices.view(-1), 0:3].view(self.num_envs, -1, 3) - self.dyn_centers).norm(dim=-1)

        xy_pos = flat_root_state[self.dynamic_obj_indices.view(-1), 0:3].view(self.num_envs, -1, 3)[:, :, 0:2]
        isreturn =  (center_dist > self.dynamic_objs_init_radius * 1.2) | ((xy_pos[:, :, 0] <= self.xy_threshold) & (xy_pos[:, :, 1] <= self.xy_threshold))

        self.dyn_vel_direction[isreturn] = self.dyn_vel_direction[isreturn] * (-1)

        # update all pcds for moving obstacles
        flat_indices_all = torch.cat([self.blocking_obj_indices.view(-1), self.dynamic_obj_indices.view(-1)])
        moving_obj_pos = flat_root_state[flat_indices_all, 0:3].view(self.num_envs, -1, 3)
        self.current_moving_obs_pcds = self.moving_pcds + moving_obj_pos.unsqueeze(2)

        self.combined_pcds[:, self.num_robot_points+self.num_static_points:self.num_robot_points+self.num_static_points+self.num_moving_points, :3] = self.current_moving_obs_pcds.view(self.num_envs, -1, 3)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(flat_root_state),
            gymtorch.unwrap_tensor(flat_indices_all),
            flat_indices_all.numel()
        )

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self.start_config = tensor_clamp(self.start_config, self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

        self.goal_config = tensor_clamp(self.goal_config, self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

        self.goal_ee = self.get_ee_from_joint(self.goal_config)

        self.set_robot_joint_state(self.start_config[env_ids], env_ids=env_ids, debug=False)

        self.voxel_counter.zero_voxels(env_ids)
        self.voxel_visit_binary[env_ids] = torch.zeros_like(self.voxel_visit_binary[env_ids])
        self.num_visited_voxels_t0[env_ids] = torch.zeros_like(self.num_visited_voxels_t0[env_ids])

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.x_reset_flag[:] = 0
        self.blk_flashing(env_ids=env_ids)
        self.compute_observations()

    def compute_reward(self, actions):
        self.check_robot_collision()
        current_angles = self.get_joint_angles()
        current_ee = self.get_ee_from_joint(current_angles)

        joint_err = torch.norm(current_angles - self.goal_config, dim=1)
        pos_err = torch.norm(current_ee[:, :3] - self.goal_ee[:, :3], dim=1)
        quat_err = orientation_error(self.goal_ee[:, 3:], current_ee[:, 3:])
        self.goal_reaching = (pos_err < 0.05) & (quat_err < 15.0) # making it slightly more tolerant atm
        # self.goal_reaching = (pos_err < 0.01) & (quat_err < 15.0) # TODO: should apply this metric later

        num_visited_voxels_t1 = torch.sum(self.voxel_visit_binary, dim=1)

        # TODO: 
        self.sdf = self.frankacc.check_scene_sdf_batch(current_angles, self.current_moving_obs_pcds.view(self.num_envs, -1, 3), debug=False, sphere_repr_only=True) # (num_envs, num_points)
        self.sdf = torch.min(self.sdf, dim=1)[0] # (num_envs, )

        self.rew_buf[:], self.reset_buf[:], reaching_rewards, intrinsic_rewards, sdf_rewards, flag_rewards = compute_franka_reward(
            self.reset_buf, self.progress_buf,
            joint_err, pos_err, quat_err,
            self.num_visited_voxels_t0, num_visited_voxels_t1,
            self.collision, self.sdf, self.residual_flag,
            self.max_episode_length
        )

        self.num_visited_voxels_t0 = num_visited_voxels_t1

        self.extras['reaching_rewards'] = torch.mean(reaching_rewards).item()
        self.extras['sdf_rewards'] = torch.mean(sdf_rewards).item()
        self.extras['flag_rewards'] = torch.mean(flag_rewards).item()

        self.success_flags[self.goal_reaching & (self.reset_buf == 1) & (self.collision_flags == 0)] = 1
        self.success_flags[(~self.goal_reaching) & (self.reset_buf == 1)] = 0
        self.reaching_flags[self.goal_reaching & (self.reset_buf == 1)] = 1 # this records reaching rate at the last step, while goal_reaching will be updated each step
        self.reaching_flags[(~self.goal_reaching) & (self.reset_buf == 1)] = 0

        self.extras['success_rate'] = torch.mean(self.success_flags.float()).item()
        self.extras['collision_rate'] = torch.mean(self.collision_flags.float()).item()
        self.extras['reaching_rate'] = torch.mean(self.reaching_flags.float()).item()

        self.collision_flags[self.reset_buf == 1] = 0 # reset collision rate after logging

        self.extras['actions/residual_action_magnitude'] = actions.norm(dim=1).mean()
        self.extras['actions/base_action_magnitude'] = self.base_delta_action.norm(dim=1).mean()

    def pre_physics_step(self, actions):
        self.residual_flag = actions[:, -1]
        is_residual_disabled = self.residual_flag > 0
        delta_actions = actions.clone()[:, :7] * torch.abs(self.residual_flag.unsqueeze(-1))
        delta_actions[is_residual_disabled] = 0.0
        current_joint_state = self.get_joint_angles()
        delta_actions = delta_actions * self.action_scale
        self.actions = delta_actions
        # since the below is commonly used for debugging, let's temporarily keep it here
        # self.base_policy_only = True
        gripper_state = torch.Tensor([[0.035, 0.035]] * self.num_envs).to(self.device)
        if self.base_policy_only:
            abs_actions = current_joint_state + self.base_delta_action
        elif self.no_base_action:
            abs_actions = current_joint_state + delta_actions
        else:
            abs_actions = current_joint_state + delta_actions + self.base_delta_action
        if abs_actions.shape[-1] == 7:
            abs_actions = torch.cat((abs_actions, gripper_state), dim=1)

        self.update_moving_obstacles_state()
        self.blk_chasing()
        if (not self.headless) and self.vis_goal:
            self._debug_viz_draw(self.pcd_spec_dict['debug'])
        # vel_targets = torch.zeros_like(abs_actions, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))
        # self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(vel_targets))

    def post_physics_step(self):
        gripper_pos = self.get_eef_pose()[:, :3]

        super().post_physics_step()
        self.reset_buf[self.x_reset_flag] = 1


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
    env = FrankaMPRRL(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    total_error = 0
    num_failed_plans = 0
    num_plans = 1000
    for i in tqdm(range(num_plans)):
        import ipdb ; ipdb.set_trace()
        t1 = time.time()
        env.reset_idx()
        t2 = time.time()

        env.render()

    print(f"Average Error: {total_error / num_plans}")
    print(f"Percentage of failed plans: {num_failed_plans / num_plans * 100} ")

def orientation_error(q1, q2):
    """
    batched orientation error computation
    input shape [B, 4(xyzw)], input ordering doesn't matter
    return absolute difference in degrees
    """
    assert q1.shape == q2.shape, "Desired and current orientations must have the same shape"

    cc = quat_conjugate(q2)
    q_r = quat_mul(q1, cc)

    # Compute the angle difference using the scalar part (w) of q_r
    w = torch.abs(q_r[:, 3])
    err = 2 * torch.acos(torch.clamp(w, -1.0, 1.0)) / torch.pi * 180  # Clamp for numerical stability, return in degrees
    return err

#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def compute_franka_reward(
    reset_buf: torch.Tensor, progress_buf: torch.Tensor,
    joint_err: torch.Tensor, pos_err: torch.Tensor, quat_err: torch.Tensor,
    num_visited_voxels_t0: torch.Tensor, num_visited_voxels_t1: torch.Tensor,
    collision_status: torch.Tensor, sdf: torch.Tensor, residual_flag: torch.Tensor,
    max_episode_length: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # Sparse reaching reward (TODO: change to use key points)
    reaching_rewards = 50*torch.exp(-10*joint_err)

    # intrinsic reward
    intrinsic_rewards = num_visited_voxels_t1 - num_visited_voxels_t0

    # sdf reward
    sdf_rewards = torch.clamp(100*sdf, -1, 20)

    # lazy reward (reward for being 'lazy' so not affect the reaching of the base policy)
    # sdf_threshold = 0.1

    # flag_diff = torch.where(sdf > sdf_threshold, 1 - residual_flag, residual_flag + 1)

    flag_diff = torch.abs(residual_flag - torch.clamp(10 * (sdf - 0.2), -1, 1))

    # print("flag: ", residual_flag)
    # print("sdf: ", sdf)
    # print("isflag correct: ", residual_flag * (sdf - 0.1) > 0)
    # print("diff: ", flag_diff)

    flag_rewards = 1 / (flag_diff + 0.1)

    # rewards = reaching_rewards + intrinsic_rewards

    rewards = sdf_rewards + flag_rewards # + reaching_rewards

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    # reset_buf[(collision_status == 1) & (progress_buf > 30)] = 1

    return rewards, reset_buf, reaching_rewards, intrinsic_rewards, sdf_rewards, flag_rewards


if __name__ == "__main__":
    launch_test()
