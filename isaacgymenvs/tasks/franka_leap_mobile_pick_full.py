"""
Franka + LEAP Hand Pick Env
TODO:1. clean up
"""

import time

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.utils.pcd_utils import *
from isaacgymenvs.utils.rotation_conversions import *
from isaacgymenvs.tasks import FrankaLEAPMobile
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.demo_loader import DemoLoader
from omegaconf import DictConfig
from tqdm import tqdm



class FrankaLEAPMobilePickFull(FrankaLEAPMobile):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg_override = cfg["cfg_override"]
        cfg = self._config_override(cfg)
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

        if not self.headless:
            for i in range(self.num_envs):
                self.draw_box_lines(i, self.box_pos[i].clone(), self.box_quats[i].clone(), self.box_dims[i].clone())

    def _config_override(self, cfg):
        # TopdownTable, TopdownConstrained, SideTable, SideConstrained
        if self.cfg_override == "TopdownTable":
            cfg["env"]["numObservations"] = 49
            cfg["env"]["numStates"] = 94
        elif self.cfg_override == "TopdownConstrained":
            cfg["env"]["numObservations"] = 61
            cfg["env"]["numStates"] = 106
        elif self.cfg_override == "SideTable":
            cfg["env"]["numObservations"] = 49
            cfg["env"]["numStates"] = 94
        elif self.cfg_override == "SideConstrained":
            cfg["env"]["numObservations"] = 61
            cfg["env"]["numStates"] = 106
            cfg["env"]["robot_init"]["switch_pos_offset"] = [-0.1,0.0,0.15] # TODO: the x offset should be along box_quat's x-axis, but now its the global x axis, update this later
            cfg["env"]["robot_init"]["switch_tol"] = 0.2
            cfg["reward"]["params"]["target_quat"] = [0.5, -0.5, 0.5, -0.5]

        return cfg

    def _init_env_config(self):
        # hdf5 scene loading
        hdf5_path = self.cfg["env"]["scene"]["hdf5_path"]
        self.demo_loader = DemoLoader(hdf5_path, self.cfg["env"]["numEnvs"])
        self.batch_idx = self.cfg["env"]["scene"]["batch_idx"]
        self.batch = self.demo_loader.get_next_batch(batch_idx=self.batch_idx)
        self.obstacle_configs = []
        self.max_obstacles = 0
        self.compartments = []
        self.init_robot_states = []
        self.saved_mesh_indices = []

        for env_idx, demo in enumerate(self.batch):
            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)
            self.max_obstacles = max(len(obstacle_config[0]), self.max_obstacles)
            self.compartments.append(demo['compartment_states'][0])
            if 'init_robot_states' in demo:
                self.init_robot_states.append(demo['init_robot_states'])
            if 'mesh_idx' in demo:
                self.saved_mesh_indices.append(int(np.asarray(demo['mesh_idx']).reshape(-1)[0]))
            else:
                self.saved_mesh_indices.append(None)
        replay_mesh_indices = [idx for idx in self.saved_mesh_indices if idx is not None]
        if replay_mesh_indices:
            self.required_mesh_preload_count = max(replay_mesh_indices) + 1  # CODEX: small debug runs must still load replayed HDF5 mesh indices.

        self.compartments = torch.tensor(self.compartments, device=self.device) # (num_envs, 10), 10 = 3 (xyz dims) + 3 (xyz pos) + 4 (xyzw quat)

        # if no init_robot_states provided, will use randomized init states later
        if len(self.init_robot_states) > 0:
            self.init_robot_states = torch.tensor(self.init_robot_states, device=self.device) # (num_envs, num_dofs)

        print("-----------------------------------------------------------")
        print(f"Loaded scene from {hdf5_path}, batch_idx: {self.batch_idx}, num_demos_in_batch: {len(self.batch)}")
        print("-----------------------------------------------------------")

    def _post_init_buffers(self):
        super()._post_init_buffers()
        # overwrite the canonical joint config with init_robot_states
        if len(self.init_robot_states) > 0:
            self.default_reset_joint_config = self.init_robot_states.clone()

    def _setup_fabric_switching_target(self):
        self.switching_target_pos = self._object_state[:, :3].clone()
        # self.switching_target_pos[:, 2] += self.mesh_aabb_extents[:, 2] / 2
        if self.cfg_override == "SideConstrained":
            self.switching_target_pos[:, 0] = self.box_pos[:, 0] - self.box_dims[:, 0] / 2 # override x to be at the box edge
            self._switching_target_quat_precomputed = torch.tensor([[0.5, -0.5, 0.5, -0.5]] * self.num_envs, device=self.device)
        else:
            self._switching_target_quat_precomputed = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * self.num_envs, device=self.device)  # 180 degrees around local x-axis
        self.switching_target_pos += self.switch_pos_offset
        self.switching_target_quat = quat_mul(self.box_quats, self._switching_target_quat_precomputed)

    def _create_envs(self, spacing, num_per_row):
        """
        loading Franka + LEAP + a table in the environment, this is for debugging purposes only
        """
        self._init_env_config()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # setup params
        self.table_pos = []
        self.table_size = []

        self.cuboid_dims = []  # xyz
        self.cuboid_pos = []
        self.cuboid_quats = [] # xyzw

        self.box_dims = self.compartments[:, :3]
        self.box_pos = self.compartments[:, 3:6]
        self.box_pos[:, 2] -= self.box_dims[:, 2] / 2  # box z pos is the bottom of the box, not the center
        self.box_quats = self.compartments[:, 6:]

        self.mesh_aabb_extents = None  # xyz, axis-aligned bounding box full extents
        self.table_surface_height = torch.zeros((self.num_envs,), device=self.device)
        self.obj_reset_pos_range = torch.zeros((self.num_envs, 4), device=self.device) # x-min, x-max, y-min, y-max
        self.obj_pos_target = torch.zeros((self.num_envs, 3), device=self.device) # x, y, z
        self.box_tol = self.cfg["env"]["scene"]["safety_box_tol"]

        self.obj_reset_pos_range[:, 0] = - self.box_dims[:, 0] / 2 + self.box_tol
        self.obj_reset_pos_range[:, 1] =   self.box_dims[:, 0] / 2 - self.box_tol
        self.obj_reset_pos_range[:, 2] = - self.box_dims[:, 1] / 2 + self.box_tol
        self.obj_reset_pos_range[:, 3] =   self.box_dims[:, 1] / 2 - self.box_tol
        self.table_surface_height = self.box_pos[:, 2]

        # setup robot (franka + leap)
        robot_dof_props = self._create_franka_leap()
        robot_asset = self.robot_asset
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # make sure robot spawns at the origin, this matches the IK setting with cuRobo
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        max_agg_bodies = num_robot_bodies + self.max_obstacles + 1  # 1 for object
        max_agg_shapes = num_robot_shapes + self.max_obstacles + 1  # 1 for object

        self.robots = []
        self.objects = []
        self.envs = []
        self.mesh_indices = []
        self._object_center_init_state = torch.zeros((self.num_envs, 3), device=self.device)

        # load all meshes first
        all_meshes_list = self.create_all_meshes()

        # Create environments
        for i in tqdm(range(self.num_envs), desc="Creating Envs"):
            # grasp object
            mesh_idx = self.saved_mesh_indices[i]
            if mesh_idx is None:
                mesh_idx = i % len(all_meshes_list)
            self.mesh_indices.append(mesh_idx)
            object_asset, object_start_pose, object_scale, object_id, mesh_id = all_meshes_list[mesh_idx]

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

            # create static scene from hdf5 loading, only cuboids for now,
            env_obstacles = []
            (
                cuboid_dims, 
                cuboid_centers, 
                cuboid_quats,
                *_
            ) = self.obstacle_configs[i]

            num_cubes = len(cuboid_dims)
            # Create obstacles
            for j in range(self.max_obstacles):
                if j == 0:
                    # first cuboid is always the table, create actor and store table params
                    obstacle_asset, obstacle_pose = self._create_cube(
                        pos=cuboid_centers[j].tolist(),
                        size=cuboid_dims[j].tolist(),
                        quat=cuboid_quats[j].tolist()
                    )

                    obstacle_actor = self.gym.create_actor(
                        env_ptr, obstacle_asset, obstacle_pose, "table", i, 1, 0
                    )
                    self.table_pos.append(cuboid_centers[j].tolist())
                    self.table_size.append(cuboid_dims[j].tolist())
                    if self.enable_fabric:
                        surface_height = cuboid_centers[j][2] + cuboid_dims[j][2] / 2
                        fabric_table_center = cuboid_centers[j].copy()
                        fabric_table_center[2] = surface_height / 2
                        fabric_table_dims = cuboid_dims[j].copy()
                        fabric_table_dims[1] *= 2 # double the table in y direction, so fabric doesn't move side ways
                        fabric_table_dims[2] = surface_height
                        self._create_fabric_cube(
                            pos=fabric_table_center.tolist(),
                            size=fabric_table_dims.tolist(),
                            quat=cuboid_quats[j].tolist(),
                            env_id=i,
                        )
                else:
                    if j < num_cubes:
                        # Create obstacle with actual size and position
                        obstacle_asset, obstacle_pose = self._create_cube(
                            pos=cuboid_centers[j].tolist(),
                            size=cuboid_dims[j].tolist(),
                            quat=cuboid_quats[j].tolist()
                        )
                        if self.enable_fabric:
                            self._create_fabric_cube(
                                pos=cuboid_centers[j].tolist(),
                                size=cuboid_dims[j].tolist(),
                                quat=cuboid_quats[j].tolist(),
                                env_id=i,
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
            self.max_objects_per_env = self.max_obstacles

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
            self.objects.append(self._object_id)

            # Precompute static and object point cloud
            static_pcd_i = torch.from_numpy(compute_scene_oracle_pcd(
                num_obstacle_points=self.pcd_spec_dict["num_static_points"],
                cuboid_dims=cuboid_dims,
                cuboid_centers=cuboid_centers,
                cuboid_quats=cuboid_quats,
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

        self.table_pos = torch.tensor(self.table_pos, device=self.device)
        self.table_size = torch.tensor(self.table_size, device=self.device)

        self.cuboid_dims = torch.tensor(self.cuboid_dims, device=self.device)
        self.cuboid_pos = torch.tensor(self.cuboid_pos, device=self.device)
        self.cuboid_quats = torch.tensor(self.cuboid_quats, device=self.device)

        # brief post processing of generated pcds
        self.static_pcds = torch.stack(self.static_pcds, dim=0).to(self.device).to(torch.float32) # (num_envs, num_points, 3)
        self.object_pcds = torch.stack(self.object_pcds, dim=0).to(self.device).to(torch.float32)
        self.combined_pcds = torch.cat([self.static_pcds, self.object_pcds], dim=1).to(self.device) # (num_envs, num_static_points + num_object_points, 3)
        if self.distractor_settings["enable"]:
            self._create_distractor_pcd()
            self.combined_pcds = torch.cat([self.combined_pcds, self.distractor_pcds], dim=1).to(self.device) # (num_envs, num_static_points + num_object_points + num_distractor_points, 3)

        # get mesh AABB (axis-aligned bounding box) extents
        min_xyz = self.object_pcds.min(axis=1).values
        max_xyz = self.object_pcds.max(axis=1).values
        self.mesh_aabb_extents = max_xyz - min_xyz
        self._object_center_init_state[:, 2] += self.mesh_aabb_extents[:, 2] / 2

        # refine obj_reset_pos_range based on mesh AABB
        self.obj_reset_pos_range[:, 0] += self.mesh_aabb_extents[:, 0] / 2
        self.obj_reset_pos_range[:, 1] -= self.mesh_aabb_extents[:, 0] / 2
        self.obj_reset_pos_range[:, 2] += self.mesh_aabb_extents[:, 1] / 2
        self.obj_reset_pos_range[:, 3] -= self.mesh_aabb_extents[:, 1] / 2
        self.obj_reset_center_xy = self.box_pos[:, :2].clone()
        self.obj_reset_center_quat = self.box_quats.clone()

        # Setup data
        actor_num = 1 + self.max_obstacles + 1  # robot, obstacles, object
        self.init_data(actor_num=actor_num)

        if self.enable_fabric:
            self._init_fabric()

    def init_data(self, actor_num):
        super().init_data(actor_num=actor_num)
        self.reward_settings["target_pos"] = self.obj_pos_target
        self.reward_settings["beta_object_drag"] = to_torch(self.cfg["reward"]["exp"]["beta_object_drag"], device=self.device)
        self.reward_settings["w_obj_drag"] = to_torch(self.cfg["reward"]["weights"]["w_obj_drag"], device=self.device)
        self.reward_settings["w_colli"] = to_torch(self.cfg["reward"]["weights"]["w_colli"], device=self.device)

    def draw_box_lines(self, env_idx, pos_xyz, quat_xyzw, dims_xyz, color=(1.0, 0.2, 0.2)):
        px, py, pz = pos_xyz
        qx, qy, qz, qw = quat_xyzw
        sx, sy, sz = dims_xyz
        sx = sx - 0.0001
        sy = sy - 0.0001
        sz = sz - 0.0001
        pz = pz + sz / 2

        center = gymapi.Vec3(px, py, pz)
        q = gymapi.Quat(qx, qy, qz, qw)

        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        corners_local = [
            gymapi.Vec3(-hx, -hy, -hz),
            gymapi.Vec3( hx, -hy, -hz),
            gymapi.Vec3( hx,  hy, -hz),
            gymapi.Vec3(-hx,  hy, -hz),
            gymapi.Vec3(-hx, -hy,  hz),
            gymapi.Vec3( hx, -hy,  hz),
            gymapi.Vec3( hx,  hy,  hz),
            gymapi.Vec3(-hx,  hy,  hz),
        ]

        corners_world = [gymapi.Quat.rotate(q, c) for c in corners_local]
        corners_world = [gymapi.Vec3(c.x + center.x, c.y + center.y, c.z + center.z) for c in corners_world]

        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]

        # Collect line endpoints
        lines = []
        for i, j in edges:
            lines.append(corners_world[i])
            lines.append(corners_world[j])

        # Convert to numpy
        line_points = np.array([[p.x, p.y, p.z] for p in lines], dtype=np.float32)
        line_colors = np.array([list(color)] * len(edges), dtype=np.float32)

        self.gym.add_lines(self.viewer, self.envs[env_idx], len(edges), line_points, line_colors)

        axis_len = min(hx, hy, hz) * 0.6
        axis_endpoints = []
        axis_colors = []
        for axis_local, axis_color in (
            (gymapi.Vec3(axis_len, 0.0, 0.0), (1.0, 0.0, 0.0)),
            (gymapi.Vec3(0.0, axis_len, 0.0), (0.0, 1.0, 0.0)),
            (gymapi.Vec3(0.0, 0.0, axis_len), (0.0, 0.0, 1.0)),
        ):
            axis_world = gymapi.Quat.rotate(q, axis_local)
            axis_endpoints.append(center)
            axis_endpoints.append(gymapi.Vec3(center.x + axis_world.x, center.y + axis_world.y, center.z + axis_world.z))
            axis_colors.append(axis_color)

        axis_points = np.array([[p.x, p.y, p.z] for p in axis_endpoints], dtype=np.float32)
        axis_colors = np.array(axis_colors, dtype=np.float32)

        self.gym.add_lines(self.viewer, self.envs[env_idx], len(axis_colors), axis_points, axis_colors)

    def draw_switching_target_pose(self, axis_len=0.08):
        pos = self.switching_target_pos.detach()
        axis_dirs = quaternion_to_matrix_ig(self.switching_target_quat.detach()).transpose(1, 2) * axis_len
        axis_starts = pos[:, None, :].expand(-1, 3, -1)
        axis_ends = pos[:, None, :] + axis_dirs
        axis_points = torch.stack((axis_starts, axis_ends), dim=2).reshape(self.num_envs, 6, 3).cpu().numpy()
        axis_colors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        for env_idx in range(self.num_envs):
            self.gym.add_lines(self.viewer, self.envs[env_idx], 3, axis_points[env_idx], axis_colors)

    def _update_states(self):
        super()._update_states()
        eef_rot_mat = quaternion_to_matrix_ig(self._eef_state[:, 3:7])
        eef_rot_mat_T = eef_rot_mat.transpose(1, 2)
        box_rot_mat = quaternion_to_matrix_ig(self.box_quats)
        box_to_eef_rot_mat = torch.matmul(eef_rot_mat_T, box_rot_mat)
        box_to_eef_rot_6d = matrix_to_rotation_6d(box_to_eef_rot_mat)

        box_to_eef_world = self.box_pos - self._eef_state[:, :3] # note box_pos here is box bottom center not box center
        box_to_eef = torch.matmul(eef_rot_mat_T, box_to_eef_world.unsqueeze(-1)).squeeze(-1)

        lift_5cm = self.states["object_center_pos"][:, 2] - self._object_center_init_state[:, 2] > 0.05

        # need to do in place assignment for self.obj_pos_target, since it affects the reward computation
        if self.cfg_override == "TopdownTable":
            self.obj_pos_target[~lift_5cm, :2] = self.states["object_center_pos"][~lift_5cm, :2]  # x, y
            self.obj_pos_target[:, 2] = self.table_surface_height + self.reward_settings['target_lift_dis']
        elif self.cfg_override == "TopdownConstrained":
            self.obj_pos_target[:, :2] = self.box_pos[:, :2]
            self.obj_pos_target[:, 2] = self.box_pos[:, 2] + self.box_dims[:, 2] + 0.2
        elif self.cfg_override == "SideTable":
            pass
        elif self.cfg_override == "SideConstrained":
            self.obj_pos_target[:] = self.box_pos.clone()
            self.obj_pos_target[:, 0] -= (self.box_dims[:, 0] / 2 + 0.1)
            self.obj_pos_target[:, 2] += 0.15 # self.box_dims[:, 2] / 2

        self.switching_target_pos = self.states['object_center_pos'].clone()
        self.switching_target_pos[:, 2] -= self.mesh_aabb_extents[:, 2] / 2
        if self.cfg_override == "SideConstrained":
            self.switching_target_pos[:, 0] = self.box_pos[:, 0] - self.box_dims[:, 0] / 2
        self.switching_target_pos += self.switch_pos_offset
        if self.viewer is not None:
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                self.draw_box_lines(i, self.box_pos[i], self.box_quats[i], self.box_dims[i])
            self.draw_switching_target_pose()

        self.states.update({
            # Box region
            "box_bottom_to_eef": box_to_eef,
            "box_dims": self.box_dims,
            "box_to_eef_rot_6d": box_to_eef_rot_6d,
            "obj_to_box_center_xy": self._object_state[:, :2] - self.box_pos[:, :2],
            # check whether the object is lifted based on bottom board force contact info
            "lift": lift_5cm,
            "collision": self.scene_collision,
        })

    def check_robot_collision(self):
        super().check_robot_collision()
        # TODO: from previous notes it seems first 58 element belongs to base + franka + leap, 59 is the table, object is the last
        # but need to double check later this is still the case
        self.scene_collision = torch.any(self.contact_forces[:, 59:-1].view(self.num_envs, -1) != 0, dim=1)
        self.table_collision = torch.any(self.contact_forces[:, 58].view(self.num_envs, -1) != 0, dim=1)

    def compute_observations(self):
        self._refresh()

        if self.num_observations == 49:
            obs_components = ["q_hand",
                            "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                            "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                            "object_to_eef", "object_to_eef_rot_6d",
                            "target_to_eef", "target_to_eef_rot_6d"]
        elif self.num_observations == 61:
            obs_components = ["q_hand",
                            "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                            "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                            "box_bottom_to_eef", "box_dims", "box_to_eef_rot_6d",
                            "object_to_eef", "object_to_eef_rot_6d",
                            "target_to_eef", "target_to_eef_rot_6d"]

        if self.num_states == 94:
            states_components = ["q", "qd",
                                "eef_pos", "eef_rot_6d", "eef_vel",
                                "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                                "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                                "object_to_eef", "object_to_eef_rot_6d",
                                "target_to_eef", "target_to_eef_rot_6d"]
        elif self.num_states == 106:
            states_components = ["q", "qd",
                                "eef_pos", "eef_rot_6d", "eef_vel",
                                "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                                "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                                "box_bottom_to_eef", "box_dims", "box_to_eef_rot_6d",
                                "object_to_eef", "object_to_eef_rot_6d",
                                "target_to_eef", "target_to_eef_rot_6d"]

        obs_buf = torch.cat([self.states[ob] for ob in obs_components], dim=-1)
        states_buf = torch.cat([self.states[st] for st in states_components], dim=-1)

        obs_buf = torch.cat([obs_buf, self.mesh_aabb_extents], dim=-1)
        states_buf = torch.cat([states_buf, self.mesh_aabb_extents], dim=-1)

        self.obs_buf = obs_buf
        self.states_buf = states_buf

        return self.obs_buf

    def compute_reward(self):
        self.reset_buf[:] = torch.where((self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf[self.states['object_center_pos'][:, 2] < self.table_surface_height-0.1] = 1

        reward_dict = compute_franka_leap_reward(self.states, self.reward_settings)

        self.rew_buf[:] = reward_dict["r_total"]
        self.extras["sep_reward/r_hand_obj"] = torch.mean(reward_dict["r_hand_obj"]).item()
        self.extras["sep_reward/r_obj_goal"] = torch.mean(reward_dict["r_obj_goal"]).item()
        self.extras["sep_reward/r_obj_drag"] = torch.mean(reward_dict["r_obj_drag"]).item()
        self.extras["sep_reward/r_lift"] = torch.mean(reward_dict["r_lift"]).item()
        self.extras["sep_reward/r_curl"] = torch.mean(reward_dict["r_curl"]).item()
        self.extras["sep_reward/r_colli"] = torch.mean(reward_dict["r_colli"]).item()
        self.extras["sep_reward/r_actionreg"] = torch.mean(reward_dict["r_actionreg"]).item()
        self.extras["dis/d_hand_obj"] = torch.mean(reward_dict["d_hand_obj"]).item()
        self.extras["dis/d_lift"] = torch.mean(reward_dict["d_lift"]).item()
        self.extras["dis/d_eef_point_goal"] = torch.mean(reward_dict["d_eef_point_goal"]).item()
        self.extras["dis/d_obj_drag"] = torch.mean(reward_dict["d_obj_drag"]).item()

        # log metrics
        self.lifting_5cm_per_step = self.states["lift"]
        self.lifting_flags[self.lifting_5cm_per_step] = 1
        self.success_5cm_per_step = (reward_dict["d_eef_point_goal"] < 0.05) & self.lifting_5cm_per_step
        self.success_flags[self.success_5cm_per_step] = 1

        self.extras["metrics/success_rate_5cm_per_step"] = torch.mean(self.success_5cm_per_step.float()).item()
        self.extras["metrics/lifting_rate_5cm_per_step"] = torch.mean(self.lifting_5cm_per_step.float()).item()
        self.extras["metrics/success_rate_5cm_per_ep"] = torch.mean(self.success_flags).item()
        self.extras["metrics/lifting_rate_5cm_per_ep"] = torch.mean(self.lifting_flags).item()
        self.extras["metrics/collision_rate_per_step"] = torch.mean(self.states["collision"].float()).item()

    def set_viewer(self):
        super().set_viewer(
            pos=[-1.2, -1.2, 1.0],
            target=[0.5, 0.0, 0.5],
        )


@torch.jit.script
def compute_franka_leap_reward(states, reward_settings):
    # type: (Dict[str, Tensor], Dict[str, Tensor]) -> Dict[str, Tensor]

    # R1: Hand (palm, fingers) to object distance
    d_palm = torch.norm(states["object_center_pos"] - states["eef_pos"], dim=-1)
    d_finger1 = torch.norm(states["object_center_pos"] - states["eef_finger1_pos"], dim=-1)
    d_finger2 = torch.norm(states["object_center_pos"] - states["eef_finger2_pos"], dim=-1)
    d_finger3 = torch.norm(states["object_center_pos"] - states["eef_finger3_pos"], dim=-1)
    d_finger4 = torch.norm(states["object_center_pos"] - states["eef_finger4_pos"], dim=-1)

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
    d_eef_point_goal = states["point_matching_err"]
    beta_object_goal = reward_settings["beta_object_goal"]
    r_obj_goal = torch.exp(-beta_object_goal * d_eef_point_goal)
    r_obj_goal = torch.where(states["lift"], r_obj_goal, 0.0)

    # R4: Drag reward
    beta_drag = reward_settings["beta_object_drag"]
    d_obj_drag = torch.norm(states["obj_to_box_center_xy"], dim=1)
    r_obj_drag = torch.exp(-beta_drag * d_obj_drag)

    # R5: Finger curl
    hand_dof_pos = states["q"][:, 10:26] # hand joint angles
    near_object = (d_hand_obj <= reward_settings["curl_reaching_threshold"])
    finger_pos_diff = torch.sum((hand_dof_pos - reward_settings["grasp_finger_dof_pos"]) ** 2, dim=1)

    beta_curl = reward_settings["beta_curl"]
    r_curl= torch.exp(-beta_curl * finger_pos_diff)
    r_curl = torch.where(near_object, r_curl, 0.0)

    # R6: Colli Penalty
    # import ipdb ; ipdb.set_trace()
    r_colli = torch.where(states["collision"], 1.0, 0.0)

    # R7: Velocity Regularization/Penalty
    actionreg = states["actionreg"]
    r_actionreg = torch.sum(actionreg**2, dim=-1)

    w_hand_obj = reward_settings["w_hand_obj"]
    w_obj_goal = reward_settings["w_obj_goal"]
    w_obj_drag = reward_settings["w_obj_drag"]
    w_lift = reward_settings["w_lift"]
    w_curl = reward_settings["w_curl"]
    w_colli = reward_settings["w_colli"]
    w_actionreg = reward_settings["w_actionreg"]

    r_total = w_hand_obj*r_hand_obj + w_obj_goal*r_obj_goal + \
              w_obj_drag*r_obj_drag + w_lift*r_lift + w_curl*r_curl + \
              w_colli*r_colli + w_actionreg*r_actionreg

    rewards = {
        "r_hand_obj": w_hand_obj*r_hand_obj,
        "r_lift": w_lift*r_lift,
        "r_obj_goal": w_obj_goal*r_obj_goal,
        "r_obj_drag": w_obj_drag*r_obj_drag,
        "r_curl": w_curl*r_curl,
        "r_colli": w_colli*r_colli,
        "r_actionreg": w_actionreg*r_actionreg,
        "r_total": r_total,
        "d_hand_obj": d_hand_obj,
        "d_lift": object_height,
        "d_eef_point_goal": d_eef_point_goal,
        "d_obj_drag": d_obj_drag,
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
    env = FrankaLEAPMobilePickFull(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
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
