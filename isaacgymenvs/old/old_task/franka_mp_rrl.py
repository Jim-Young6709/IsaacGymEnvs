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
from neural_mp.utils.franka_utils import unnormalize_franka_joints
from collections import OrderedDict, deque
from omegaconf import DictConfig
from tqdm import tqdm


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
        self.is_rrl = cfg["env"]["is_rrl"]
        self.no_base_action = cfg["env"]["no_base_action"]
        if cfg["env"]["rl_as_goal"]:
            cfg["env"]["base_policy_only"] = True
        self.base_policy_only = cfg["env"]["base_policy_only"]

        # Demo loading
        hdf5_path = cfg["env"]["hdf5_path"]
        self.demo_loader = DemoLoader(hdf5_path, cfg["env"]["numEnvs"])
        self.batch_idx = cfg["env"]["batch_idx"]

        # need to change the logic here (2 layers of reset ; multiple start & goal in one env ; relaunch IG)
        self.batch, self.vec_states = self.demo_loader.get_next_batch(batch_idx=self.batch_idx)

        self.obstacle_configs = []
        self.obstacle_handles = []
        self.dyn_obj_handles = []
        self.max_obstacles = 0
        self.step_counter = 0
        self.frankacc = FrankaCollisionChecker()

        for env_idx, demo in enumerate(self.batch):
            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)
            self.max_obstacles = max(len(obstacle_config[0]), self.max_obstacles)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        # this is very ugly coded now, cleanup this later with multi start & goal scenes
        for env_idx, demo in enumerate(self.batch):
            self.start_config[env_idx] = torch.tensor(demo['states'][0][:7], device=self.device)
            self.goal_config[env_idx] = torch.tensor(demo['states'][0][7:14], device=self.device)

        self.reset_idx()

        assert "numObservations" in self.cfg["env"], "numObservations must be specified in the config"
        assert "numStates" in self.cfg["env"], "numStates must be specified in the config"
        assert "numActions" in self.cfg["env"], "numActions must be specified in the config"
        self.progress_buf = torch.randint(0, self.max_episode_length, (self.num_envs,)).to(self.device)

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.cuboid_dims = []  # xyz
        self.capsule_dims = []  # r, l
        self.sphere_radii = []  # r

        # setup franka
        franka_dof_props = self._create_franka()
        franka_asset = self.franka_asset
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # setup moving obstacles
        self.num_dyn_objs = self.cfg["dyn_obj"]["num_obj"]
        dyn_objs_dim_range = self.cfg["dyn_obj"]["dim_range"]
        dyn_dist_range = self.cfg["dyn_obj"]["dist_range"]
        self.dyn_vel = np.random.uniform(self.cfg["dyn_obj"]["vel_range"][0], self.cfg["dyn_obj"]["vel_range"][1], (self.num_envs, self.num_dyn_objs))
        self.dyn_vel = torch.from_numpy(self.dyn_vel).to(self.device, dtype=torch.float32) * self.cfg["sim"]["dt"]
        self.dyn_pos = torch.zeros((self.num_envs, self.num_dyn_objs, 3), device=self.device, dtype=torch.float32)
        self.dyn_vel_direction = torch.zeros((self.num_envs, self.num_dyn_objs, 3), device=self.device, dtype=torch.float32) # to store the direction of velocity for each dynamic object
        self.dyn_dist = np.random.uniform(dyn_dist_range[0], dyn_dist_range[1], (self.num_envs, self.num_dyn_objs))
        self.dyn_dist = torch.from_numpy(self.dyn_dist).to(self.device, dtype=torch.float32)
        self.dyn_chasing_flag = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool) # to store the chasing flag for each dynamic object
        self.rand_sphere_idx = torch.zeros((self.num_envs, self.num_dyn_objs, 1), dtype=torch.int64, device=self.device)

        self.bounceback_enable = self.cfg["dyn_obj"]["bounce_back"]["enable"]
        self.bounceback_thres = self.cfg["dyn_obj"]["bounce_back"]["thres"]

        self.sdf_rw_max_curri = torch.tensor(self.cfg["reward"]["sdf_rw_max"], device=self.device, dtype=torch.float32)
        self.flag_rw_max_curri = torch.tensor(self.cfg["reward"]["flag_rw_max"], device=self.device, dtype=torch.float32)
        self.sdf_threshold = self.cfg["reward"]["sdf_threshold"]
        self.curri_freq = self.cfg["reward"]["curri_freq"]
        self.curri_idx = 0

        self.prob_start_at_goal = self.cfg["env"]["prob_start_at_goal"]
        self.xyz_threshold = torch.tensor(self.cfg["dyn_obj"]["xyz_threshold"], device=self.device, dtype=torch.float32)
        self.ground_truth_flag = self.cfg["env"]["ground_truth_flag"]
        self.rl_as_goal = self.cfg["env"]["rl_as_goal"]
        self.rl_abs = self.cfg["env"]["rl_abs"]

        self.dyn_sdf = torch.zeros(self.num_envs, device=self.device)
        self.static_sdf = torch.zeros(self.num_envs, device=self.device)
        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + self.max_obstacles + self.num_dyn_objs # franka + obstacles
        max_agg_shapes = num_franka_shapes + self.max_obstacles + self.num_dyn_objs
        self.frankas = []
        self.env_ptrs = []

        self.num_robot_points = self.pcd_spec_dict['num_robot_points']
        self.num_scene_points = self.pcd_spec_dict['num_obstacle_points']
        self.num_moving_points_per_obj = self.pcd_spec_dict['num_moving_obstacle_points_per_obj']
        self.num_moving_points = self.num_dyn_objs * self.num_moving_points_per_obj
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

            # cuboid_dims = cuboid_dims[[0]]
            # cuboid_centers = cuboid_centers[[0]]
            # cuboid_quats = cuboid_quats[[0]]

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
            for j in range(self.num_dyn_objs):
                dyn_objs_dim = np.random.uniform(dyn_objs_dim_range[0], dyn_objs_dim_range[1])
                dyn_objs_pos = [0.5, 0., 0.5]
                dyn_objs_xyzw = random_quaternion_xyzw()
                dyn_asset, dyn_pose = self._create_cube(
                    pos=dyn_objs_pos,
                    size=dyn_objs_dim.tolist(),
                    quat=dyn_objs_xyzw.tolist(),
                )
                dyn_actor = self.gym.create_actor(
                    env_ptr,
                    dyn_asset,
                    dyn_pose,
                    f"dyn_{j}",
                    i,
                    1,
                    0
                )
                if not self.headless:
                    self.gym.set_rigid_body_color(env_ptr, dyn_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 1.0))
                block_obstacles.append(dyn_actor)
                moving_cuboids.append(Cuboid(np.array([0.0, 0.0, 0.0]), dyn_objs_dim, dyn_objs_xyzw[[3, 0, 1, 2]]))

            self.dyn_obj_handles.append(block_obstacles)
            
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

        # Setup data
        actor_num = 1 + self.max_obstacles + self.num_dyn_objs # franka  + obstacles
        self.dyn_obj_indices = torch.tensor(self.dyn_obj_handles, device=self.device, dtype=torch.int32)
        for i in range(self.num_envs):
            self.dyn_obj_indices[i] += actor_num * i
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
            fabric_goal_pose = self.get_ee_from_joint(self.updated_goal)
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

    def dyn_flashing(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # update dynamic obstacles
        current_configs = self.get_joint_angles()[env_ids]
        torch_spheres = self.frankacc.torch_spheres(current_configs)
        centers = torch_spheres.centers[:, 28:-10, :] # link4 - gripper
        radii = torch_spheres.radii[:, 28:-10]

        self.rand_sphere_idx[env_ids] = torch.randint(low=0, high=radii[:, 8:].shape[1], size=(len(env_ids),self.num_dyn_objs, 1), dtype=torch.int64).to(self.device)

        centers = torch.gather(centers, dim=1, index=self.rand_sphere_idx[env_ids].expand(-1, -1, centers.shape[-1])) # (num_envs, num_obj, 3)
        radii = torch.gather(radii, dim=1, index=self.rand_sphere_idx[env_ids].expand(-1, -1, radii.shape[-1])) # (num_envs, num_obj, 1)

        # flashing position
        center_direction = torch.randn_like(centers)
        center_shift = center_direction / center_direction.norm(dim=-1, keepdim=True) * (radii + self.dyn_dist[env_ids].unsqueeze(-1))
        dyn_objs_pos = centers + center_shift

        flat_dyn_indices = self.dyn_obj_indices[env_ids].view(-1)
        flat_root_state = self._root_state.view(-1, 13)
        flat_root_state[flat_dyn_indices, 0:3] = dyn_objs_pos.view(-1, 3)

        self.dyn_pos = flat_root_state[self.dyn_obj_indices.view(-1), 0:3].view(self.num_envs, self.num_dyn_objs, 3)

        displacement = centers.view(-1, 3) - flat_root_state[flat_dyn_indices, 0:3]
        updated_vel_direction = displacement / displacement.norm(dim=-1, keepdim=True)
        self.dyn_vel_direction[env_ids] = updated_vel_direction.view(-1, self.num_dyn_objs, 3)
        self.dyn_chasing_flag[env_ids] = True

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(flat_root_state),
            gymtorch.unwrap_tensor(flat_dyn_indices),
            flat_dyn_indices.numel()
        )

    def dyn_chasing(self):
        # update dynamic obstacle chasing flags
        if self.bounceback_enable:
            # bounce back mode
            floating_ids = self.dyn_sdf < self.bounceback_thres[0]
            chasing_ids = self.dyn_sdf > self.bounceback_thres[1]
            self.dyn_chasing_flag[floating_ids] = False
            self.dyn_chasing_flag[chasing_ids] = True
        else:
            raise NotImplementedError("Other mode no longer supported.")

        # update robot & dyn obj states
        current_configs = self.get_joint_angles()
        torch_spheres = self.frankacc.torch_spheres(current_configs)
        centers = torch_spheres.centers[:, 28:, :] # link5 - gripper
        centers = torch.gather(centers, dim=1, index=self.rand_sphere_idx.expand(-1, -1, centers.shape[-1])) # (num_envs, num_obj, 3)

        flat_dyn_indices = self.dyn_obj_indices.view(-1)
        flat_root_state = self._root_state.view(-1, 13)

        # safety region correction
        # TODO: this only work with one dynamic obstacle at the moment
        x_pos = flat_root_state[flat_dyn_indices, 0]
        y_pos = flat_root_state[flat_dyn_indices, 1]
        z_pos = flat_root_state[flat_dyn_indices, 2]

        safety_err = torch.zeros((self.num_envs, 3, 2), dtype=torch.float32, device=self.device)
        safety_err[:, 0, 0] = self.xyz_threshold[0, 0] - x_pos
        safety_err[:, 0, 1] = x_pos - self.xyz_threshold[0, 1]
        safety_err[:, 1, 0] = self.xyz_threshold[1, 0] - y_pos
        safety_err[:, 1, 1] = y_pos - self.xyz_threshold[1, 1]
        safety_err[:, 2, 0] = self.xyz_threshold[2, 0] - z_pos
        safety_err[:, 2, 1] = z_pos - self.xyz_threshold[2, 1]

        safety_vio = torch.max(safety_err.view(self.num_envs, -1), dim=-1)[0] <= 0.0
        safety_vio_xyz = torch.min(torch.abs(safety_err.view(self.num_envs, -1)), dim=-1)[1] # check which axis' motion is near the boundary

        # update velocity
        displacement = centers.view(-1, 3) - flat_root_state[flat_dyn_indices, 0:3]
        self.dyn_vel_direction[self.dyn_chasing_flag] = displacement.view(-1, self.num_dyn_objs, 3)[self.dyn_chasing_flag]
        self.dyn_vel_direction[safety_vio, :, (safety_vio_xyz[safety_vio] // 2)] = 0.0
        self.dyn_vel_direction = self.dyn_vel_direction / self.dyn_vel_direction.norm(dim=-1, keepdim=True)

        if self.bounceback_enable:
            flat_root_state[flat_dyn_indices, 0:3] += self.dyn_vel.view(-1).unsqueeze(-1) * self.dyn_vel_direction.view(-1, 3)

        self.dyn_pos = flat_root_state[flat_dyn_indices, 0:3].view(self.num_envs, self.num_dyn_objs, 3)
        flat_root_state[flat_dyn_indices[safety_vio], (safety_vio_xyz[safety_vio] // 2)] = self.xyz_threshold.view(-1)[safety_vio_xyz[safety_vio]]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(flat_root_state),
            gymtorch.unwrap_tensor(flat_dyn_indices),
            flat_dyn_indices.numel()
        )

    def update_dynamic_obstacles_pcd(self):
        # update all pcds for moving obstacles
        flat_root_state = self._root_state.view(-1, 13)
        flat_indices_all = self.dyn_obj_indices.view(-1)
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

        start_config = tensor_clamp(self.start_config[env_ids], self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

        goal_config = tensor_clamp(self.goal_config[env_ids], self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

        start_at_goal = torch.rand(len(env_ids), device=self.device) < self.prob_start_at_goal
        start_config[start_at_goal] = goal_config[start_at_goal]

        self.goal_ee = self.get_ee_from_joint(self.goal_config)

        self.set_robot_joint_state(start_config, env_ids=env_ids, debug=False)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.dyn_flashing(env_ids=env_ids)
        self.update_dynamic_obstacles_pcd()
        self.compute_observations()
        if (self.pcd_his_len > 0) & (self.pcd_feat_buffer is not None):
            for i in range(self.pcd_his_len - 1):
                # after reset, set all history to the latest observation
                self.pcd_feat_buffer[i][env_ids] = self.pcd_feat_buffer[-1][env_ids]

    def compute_observations(self):
        self._refresh()

        robot_config = self.states['q'][:, :7].clone()
        self.update_robot_pcds(robot_config)

        obs_base = OrderedDict()
        obs_base["current_angles"] = robot_config
        obs_base["goal_angles"] = self.updated_goal.clone()
        obs_base["compute_pcd_params"] = self.combined_pcds.clone()

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                pcd_latent = self.pcd_encoder(obs_base)

        if self.pcd_his_len > 0:
            if self.pcd_feat_buffer is None:
                # self.pcd_feat_buffer[0] is the oldest pcd feature ; self.pcd_feat_buffer[-1] is the latest pcd feature
                self.pcd_feat_buffer = deque([pcd_latent[:, :-14].clone() for _ in range(self.pcd_his_len)], maxlen=self.pcd_his_len)
            elif self.pcd_his_len > 0:
                self.pcd_feat_buffer.append(pcd_latent[:, :-14].clone())

            if self.pcd_his_delta:
                his_pcd_feats = self.pcd_feat_buffer[-1] - self.pcd_feat_buffer[0]
            else:
                his_pcd_feats = self.pcd_feat_buffer[0]
            obs = torch.cat((his_pcd_feats, pcd_latent), dim=1)
        else:
            num_scene_points = self.num_static_points + self.num_moving_points
            dyn_pcd = self.combined_pcds[:, self.num_robot_points+self.num_static_points:self.num_robot_points+num_scene_points, :3].clone()
            idx = torch.randint(0, self.num_moving_points, (dyn_pcd.shape[0], num_scene_points,), device=dyn_pcd.device)
            idx_expanded = idx.unsqueeze(-1).expand(-1, -1, 3)
            upsampled_dyn_pcd = torch.gather(dyn_pcd, dim=1, index=idx_expanded)
            obs_base["compute_pcd_params"][:, self.num_robot_points:self.num_robot_points+num_scene_points, :3] = upsampled_dyn_pcd
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    dyn_pcd_latent = self.pcd_encoder(obs_base)
            obs = torch.cat((dyn_pcd_latent[:, :-14], pcd_latent), dim=1)

        assert self.obs_buf.size(1) == 2055

        obs = obs[:, :-7]
        self.obs_buf = obs

        # update sdf values
        dyn_sdf = self.frankacc.check_scene_sdf_batch(robot_config, self.current_moving_obs_pcds.view(self.num_envs, -1, 3), debug=False, sphere_repr_only=True) # (num_envs, num_dyn_points)
        static_sdf = self.frankacc.check_scene_sdf_batch(robot_config, self.static_pcds, debug=False, sphere_repr_only=True) # (num_envs, num_static_points)

        self.dyn_sdf = torch.min(dyn_sdf, dim=1)[0] # (num_envs, )
        self.static_sdf = torch.min(static_sdf, dim=1)[0] # (num_envs, )

        self.states_buf[:, 0:self.num_obs] = obs.clone()
        self.states_buf[:, self.num_obs:self.num_obs +self.num_dyn_objs*3] = self.dyn_pos.reshape(self.num_envs, -1)
        self.states_buf[:, self.num_obs +self.num_dyn_objs*3:self.num_obs +2*self.num_dyn_objs*3] = self.dyn_vel_direction.reshape(self.num_envs, -1)
        self.states_buf[:, self.num_obs +2*self.num_dyn_objs*3:-1] = self.dyn_sdf.unsqueeze(-1)
        self.states_buf[:, -1:] = self.static_sdf.unsqueeze(-1)

        return self.obs_buf

    def compute_reward(self, actions):
        self.check_robot_collision()
        current_angles = self.get_joint_angles()
        current_ee = self.get_ee_from_joint(current_angles)

        joint_err = torch.norm(current_angles - self.goal_config, dim=1)
        pos_err = torch.norm(current_ee[:, :3] - self.goal_ee[:, :3], dim=1)
        quat_err = orientation_error(self.goal_ee[:, 3:], current_ee[:, 3:])
        self.goal_reaching = (pos_err < 0.05) & (quat_err < 15.0) # making it slightly more tolerant atm
        # self.goal_reaching = (pos_err < 0.01) & (quat_err < 15.0) # TODO: should apply this metric later

        sdf_curri_idx = min(self.curri_idx, len(self.sdf_rw_max_curri)-1)
        flag_curri_idx = min(self.curri_idx, len(self.flag_rw_max_curri)-1)

        self.rew_buf[:], self.reset_buf[:], sdf_rewards, flag_rewards = compute_franka_reward(
            self.reset_buf, self.progress_buf,
            joint_err, pos_err, quat_err,
            self.collision, self.dyn_sdf, self.static_sdf,
            self.sdf_rw_max_curri[sdf_curri_idx], self.flag_rw_max_curri[flag_curri_idx], self.sdf_threshold,
            self.residual_flag, self.max_episode_length
        )

        self.extras['num_sim_steps'] = self.step_counter

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
        self.step_counter += 1
        if self.step_counter % self.curri_freq == 0:
            self.curri_idx += 1

        self.residual_flag = actions[:, -1]
        if self.ground_truth_flag > self.step_counter:
            is_residual_disabled = self.dyn_sdf > self.sdf_threshold
        else:
            is_residual_disabled = self.residual_flag > 0

        current_joint_state = self.get_joint_angles()

        if self.rl_abs:
            abs_actions = actions.clone()[:, :7]
            abs_actions = unnormalize_franka_joints(abs_actions)
            self.actions = abs_actions - current_joint_state
        else:
            if self.rl_as_goal:
                delta_actions = actions.clone()[:, :7]
            else:
                delta_actions = actions.clone()[:, :7] * torch.abs(self.residual_flag.unsqueeze(-1))
            delta_actions[is_residual_disabled] = 0.0
            delta_actions = delta_actions * self.action_scale
            self.actions = delta_actions

        if self.rl_as_goal:
            abs_goal = self.actions + current_joint_state
            self.updated_goal = abs_goal.clone()
            self.updated_goal[is_residual_disabled] = self.goal_config[is_residual_disabled]
        else:
            self.updated_goal = self.goal_config.clone()

        self.update_robot_pcds(current_joint_state)
        obs_base = OrderedDict()
        obs_base["current_angles"] = current_joint_state
        obs_base["goal_angles"] = self.updated_goal.clone()
        obs_base["compute_pcd_params"] = self.combined_pcds.clone()
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                self.base_delta_action = self.base_model.policy.get_action(obs_dict=obs_base, mean_actions=self.use_mean_actions)

        if not self.is_rrl:
            self.base_delta_action[~is_residual_disabled] = 0.0

        gripper_state = torch.Tensor([[0.035, 0.035]] * self.num_envs).to(self.device)
        if self.base_policy_only:
            abs_actions = current_joint_state + self.base_delta_action
        elif self.no_base_action:
            abs_actions = current_joint_state + delta_actions
        else:
            abs_actions = current_joint_state + delta_actions + self.base_delta_action
        if abs_actions.shape[-1] == 7:
            abs_actions = torch.cat((abs_actions, gripper_state), dim=1)

        self.dyn_chasing()
        self.update_dynamic_obstacles_pcd()
        if (not self.headless) and self.vis_goal:
            self._debug_viz_draw(self.pcd_spec_dict['debug'])
        # vel_targets = torch.zeros_like(abs_actions, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))
        # self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(vel_targets))

    def post_physics_step(self):
        super().post_physics_step()


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
    collision_status: torch.Tensor, dyn_sdf: torch.Tensor, static_sdf: torch.Tensor,
    sdf_rw_max: torch.Tensor, flag_rw_max: torch.Tensor, sdf_threshold: torch.Tensor,
    residual_flag: torch.Tensor, max_episode_length: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # sdf reward
    sdf = torch.min(dyn_sdf, static_sdf)

    sdf_rewards = torch.clamp( (sdf_rw_max / sdf_threshold)*sdf, -1, sdf_rw_max)
    # we don't want to penalize the robot for being close to the static obstacles during reaching phase when dynamic obstacles are far away
    sdf_rewards[(dyn_sdf > sdf_threshold) & (residual_flag > 0)] = sdf_rw_max
    # sdf_rewards[(dyn_sdf > sdf_threshold) & (residual_flag <= 0)] = 0

    flag_diff = torch.abs(residual_flag - torch.clamp(10000 * (dyn_sdf - sdf_threshold), -1, 1))
    if flag_rw_max == 0:
        flag_rewards = torch.zeros_like(flag_diff).to(flag_diff.device)
    else:
        flag_rewards = 1 / (flag_diff + (1 / flag_rw_max) )

    # print(flag_rewards)
    # print("flag: ", residual_flag)
    # print("sdf: ", dyn_sdf)
    # print("isflag correct: ", residual_flag * (dyn_sdf - 0.1) > 0)
    # print("diff: ", flag_diff)

    rewards = sdf_rewards + flag_rewards

    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    # reset_buf[(collision_status == 1) & (progress_buf > 30)] = 1

    return rewards, reset_buf, sdf_rewards, flag_rewards


if __name__ == "__main__":
    launch_test()