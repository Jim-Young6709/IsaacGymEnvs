import os
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from isaacgym.torch_utils import *
from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.demo_loader import DemoLoader
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.tasks.utils.pcd_utils import decompose_scene_pcd_params_obs, compute_scene_oracle_pcd
from isaacgymenvs.tasks.utils.geometry import construct_mixed_point_cloud
from isaacgymenvs.tasks.utils.drp_evals_utils import orientation_error, random_quaternion_xyzw, transform_pcds_to_world
from isaacgymenvs.obstacle_spawner import ObstacleSpawner

from robofin.pointcloud.torch import FrankaSampler
from geometrout.primitive import Cuboid



class DRPEvals(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.headless = headless
        self.device = sim_device
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        if self.headless:
            self.debug_viz = False
        
        self.use_dynamic_obstacles = True

        self.max_num_static_obstacles = 0
        self.max_num_dynamic_obstacles = 0
        self.load_data_set()
        self.gpu_fk_sampler = FrankaSampler(sim_device, use_cache=True)

        self.obstacle_spawner = ObstacleSpawner(self)

        super().__init__(
            config=self.cfg, rl_device=sim_device, sim_device=sim_device, graphics_device_id=graphics_device_id, 
            headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )
        self._refresh()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
    

    def load_data_set(self):
        # Loading environment dataset
        hdf5_path = self.cfg["env"]["asset"]["data_set_path"]
        self.demo_loader = DemoLoader(hdf5_path, self.cfg["env"]["numEnvs"])
        # len(data_batch) = self.num_envs
        data_batch = self.demo_loader.get_next_batch()

        self.start_joint_pos = torch.zeros((self.cfg["env"]["numEnvs"], 7), device=self.device)
        self.goal_joint_pos = torch.zeros((self.cfg["env"]["numEnvs"], 7), device=self.device)
        
        self.obstacle_configs = list()
        self.max_num_static_obstacles = 0

        for env_idx, demo in enumerate(data_batch):
            self.start_joint_pos[env_idx] = torch.tensor(demo['states'][0][0:7], device=self.device)
            self.goal_joint_pos[env_idx] = torch.tensor(demo['states'][0][7:14], device=self.device)
            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)
            self.max_num_static_obstacles = max(len(obstacle_config[0]), self.max_num_static_obstacles)
          

    def create_sim(self):
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_franka(self, ):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"
        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        self.franka_asset = franka_asset

        self.franka_default_joint_pos = torch.tensor(
            [0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4], device=self.device
        ).repeat(self.num_envs, 1)
        franka_dof_stiffness = torch.tensor([1000.0]*7 + [800., 800.], dtype=torch.float, device=self.device)
        franka_dof_damping = torch.tensor([50]* 7 + [40., 40.], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
            franka_dof_props['damping'][i] = franka_dof_damping[i]
            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
        self.franka_dof_lower_limits = torch.tensor(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = torch.tensor(self.franka_dof_upper_limits, device=self.device)
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        return franka_dof_props, franka_start_pose

    def _create_cube(self, pos, size, quat=[0, 0, 0, 1]):
        # create cube asset
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        asset = self.gym.create_box(self.sim, *size, opts)
        # define start pose
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(*quat)
        return asset, start_pose
    

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # set up franka
        franka_dof_props, franka_start_pose = self._create_franka()
        
        # set up handle buffers
        self.franka_handles = list()
        self.env_handles = list()
        self.static_obstacle_handles = list()
        self.dynamic_obstacle_handles = list()

        # (num_envs, num_dynamic_obstacles, num_moving_points_per_obj, 3)
        self.dynamic_obstacle_pcd = list()

        # create environments
        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.env_handles.append(env_ptr)

            # ----- create Franka ----- 
            franka_actor = self.gym.create_actor(
                env_ptr, self.franka_asset, franka_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)
            self.franka_handles.append(franka_actor)

            # ----- create static obstacles ----- 
            static_obstacles_handles = list()
            cuboid_dims, cuboid_centers, cuboid_quats, *_ = self.obstacle_configs[i]
            num_cubes = len(cuboid_dims)
            for j in range(self.max_num_static_obstacles):
                if j < num_cubes:
                    # create obstacle with actual size and position
                    obstacle_asset, obstacle_pose = self._create_cube(
                        pos=cuboid_centers[j].tolist(),
                        size=cuboid_dims[j].tolist(),
                        quat=cuboid_quats[j].tolist()
                    )
                else:
                    # create minimal placeholder obstacles far away
                    obstacle_asset, obstacle_pose = self._create_cube(
                        pos=[0., 0., -100.0],
                        size=[0.001, 0.001, 0.001],
                        quat=[0, 0, 0, 1]
                    )
                obstacle_actor = self.gym.create_actor(env_ptr, obstacle_asset, obstacle_pose, f"obstacle_{j}", i, 1, 0)
                static_obstacles_handles.append(obstacle_actor)
            self.static_obstacle_handles.append(static_obstacles_handles)


            # ----- create dynamic obstacles -----
            if self.use_dynamic_obstacles:
                self.max_num_dynamic_obstacles = 2 #10
                self.num_points_per_dynamic_obstacle = 500
                self.dynamic_obstacle_pcd_combined = torch.zeros((self.num_envs, self.max_num_dynamic_obstacles*self.num_points_per_dynamic_obstacle, 3), device=self.device)

                dynamic_obstacle_handles = list()
                dynamic_obstacles = list()
                for j in range(self.max_num_dynamic_obstacles):
                    dyn_objs_dim = np.random.uniform([0.1, 0.1, 0.1], [0.3, 0.3, 0.3])
                    dyn_objs_pos = np.array([0.5, 0., 0.5])
                    dyn_objs_xyzw = random_quaternion_xyzw()
                    dyn_asset, dyn_pose = self._create_cube(
                        pos=dyn_objs_pos,
                        size=dyn_objs_dim.tolist(),
                        quat=dyn_objs_xyzw.tolist(),
                    )
                    dynamic_obstacle_actor = self.gym.create_actor(env_ptr, dyn_asset, dyn_pose, f"dyn_{j}", i, 1, 0)
                    self.gym.set_rigid_body_color(env_ptr, dynamic_obstacle_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 1.0))
                    dynamic_obstacle_handles.append(dynamic_obstacle_actor)

                    dynamic_obstacles.append(Cuboid(np.array([0.0, 0.0, 0.0]), dyn_objs_dim, np.array([1.0, 0.0, 0.0, 0.0])))

                self.dynamic_obstacle_handles.append(dynamic_obstacle_handles)

                # (num_dynamic_obstacles, num_moving_points_per_obj, 3)
                dynamic_pcd = torch.tensor(
                    construct_mixed_point_cloud(
                        dynamic_obstacles, num_points=self.num_points_per_dynamic_obstacle*len(dynamic_obstacles), return_point_list=True, even=True
                    ), device=self.device
                )[..., 0:3]
                self.dynamic_obstacle_pcd.append(dynamic_pcd)
        
        actor_num = 1 + self.max_num_static_obstacles + self.max_num_dynamic_obstacles
        self._init_data(actor_num=actor_num)
    

    def _init_data(self, actor_num):
        # setup sim handles
        env_ptr = self.env_handles[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "right_gripper"),
        }

        # get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # set up tensor buffers
        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(_net_contact_forces).view(self.num_envs, -1, 3)
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]

        # initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        # initialize indices
        self._global_indices = torch.arange(
            self.num_envs * actor_num, dtype=torch.int32, device=self.device
        ).view(self.num_envs, -1)

        # global indices for dynamic obstacles
        if self.use_dynamic_obstacles:
            # (num_envs, max_num_dynamic_obstacles)
            self.dynamic_obstacle_indices = torch.tensor(self.dynamic_obstacle_handles, device=self.device, dtype=torch.int32)
            for i in range(self.num_envs):
                self.dynamic_obstacle_indices[i] += actor_num * i
            self.dynamic_obstacle_pcd = torch.stack(self.dynamic_obstacle_pcd, dim=0)

        # initialize useful buffers
        self.states = dict()
        self.scene_collision = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self.collision = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self.scene_collision_counter = torch.zeros(self.num_envs, dtype=int, device=self.device)
        self.ee_goal_pose = self.get_ee_from_joint(self.goal_joint_pos)
        self.start_joint_pos = tensor_clamp(self.start_joint_pos, self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])
        self.goal_joint_pos = tensor_clamp(self.goal_joint_pos, self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

    
    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # refresh states
        self.states.update({
            "q": self._q[:, :],
            "qd": self._qd[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
        })
        self.check_robot_collision()


    def check_robot_collision(self):
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.scene_collision = torch.where(
            torch.norm(torch.sum(self.contact_forces[:, :16, :], dim=1), dim=1) > 1.0, 1.0, 0.0
        )  # the first 16 elements belong to franka robot
        self.collision = torch.where(
            torch.sum(torch.norm(self.contact_forces[:, :16, :], dim=2), dim=1) > 1.0, 1.0, 0.0
        )  # the first 16 elements belong to franka robot, this includes self collision
    

    def get_robot_pcds(self, joint_pos):
        robot_pcd = self.gpu_fk_sampler.sample(joint_pos, self.num_robot_points)
        return robot_pcd
    

    def generate_scene_pcd(self, num_robot_points, num_goal_robot_points, num_obstacle_points):
        # set up pcd buffers
        self.num_robot_points = num_robot_points
        self.num_goal_robot_points = num_goal_robot_points
        self.num_obstacle_points = num_obstacle_points
        self.static_obstacle_pcd = torch.zeros((self.num_envs, self.num_obstacle_points, 3), device=self.device)
        self.robot_pcd = torch.zeros((self.num_envs, self.num_robot_points, 3), device=self.device)
        self.goal_robot_pcd = torch.zeros((self.num_envs, self.num_goal_robot_points, 3), device=self.device)

        # generate static obstacle pcd
        for i in range(self.num_envs):
            cuboid_dims, cuboid_centers, cuboid_quats, *_ = self.obstacle_configs[i]
            # (num_obstacle_points, 3)
            static_obstacle_pcd = compute_scene_oracle_pcd(
                num_obstacle_points=self.num_obstacle_points,
                cuboid_dims=cuboid_dims,
                cuboid_centers=cuboid_centers,
                cuboid_quats=cuboid_quats,
            )
            self.static_obstacle_pcd[i,:, :] = torch.tensor(static_obstacle_pcd, device=self.device)
        
        # create target robot pcd
        self.goal_robot_pcd = self.get_robot_pcds(self.goal_joint_pos)
    

    def get_ee_from_joint(self, joint_pos, frame="right_gripper"):
        """
        Get the end effector pose from the joint angles.
        Args:
            joint_pos (torch.Tensor): 7-dof joint angles. (B, 7)
        Returns:
            ee_pose (torch.Tensor)): 7D end effector pose. xyz, xyzw
        """
        eef_tranforms = self.gpu_fk_sampler.end_effector_pose(joint_pos, frame)
        eef_xyz = eef_tranforms[:, :3, 3]
        eef_rotations = eef_tranforms[:, :3, :3].cpu().numpy()
        eef_xyzw = Rotation.from_matrix(eef_rotations).as_quat()
        eef_xyzw = torch.Tensor(eef_xyzw).to(self.device)
        return torch.cat((eef_xyz, eef_xyzw), dim=1)


    def reset_idx(self, env_ids=None):
        # will refresh tensors here via set_robot_joint_state
        self.set_robot_joint_state(
            joint_pos=self.start_joint_pos[env_ids], env_ids=env_ids,
        )
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def apply_joint_pos_targets(self, joint_pos_targets):
        gripper_targets = torch.tensor([0.04, 0.04], device=self.device).repeat(self.num_envs, 1)
        franka_actions = torch.cat([joint_pos_targets, gripper_targets], dim=1)
        self._pos_control[:, :] = franka_actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(franka_actions))

    
    def set_robot_joint_state(self, joint_pos, joint_vel=None, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        assert len(joint_pos) == len(env_ids)
        
        gripper_pos = torch.tensor([0.04, 0.04], device=self.device).repeat(len(env_ids), 1)
        arm_gripper_joint_pos = torch.cat([joint_pos, gripper_pos], dim=1)

        # prepare full joint state tensor (len(env_ids), 9, 2)
        state_tensor = arm_gripper_joint_pos.unsqueeze(2)
        state_tensor = torch.cat((state_tensor, torch.zeros_like(state_tensor)), dim=2)
        # fill in joint velocity if given
        if joint_vel is not None:
            state_tensor[:, 0:7, 1] = joint_vel

        # reset the internal obs accordingly
        pos = state_tensor[:, :, 0].contiguous()
        vel = state_tensor[:, :, 1].contiguous()
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = vel
        self._dof_state[env_ids, :] = state_tensor
        self._pos_control[env_ids, :] = pos

        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

        # update simulation
        self.gym.simulate(self.sim)
        self._refresh()
        if not self.headless:
            self.render()
    

    def set_dynamic_obstacle_pose(self, dynamic_obstacle_poses, env_ids=None):
        # dynamic_obstacle_poses: (num_envs, num_dynamic_obstacles, 7)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        flat_dyn_indices = self.dynamic_obstacle_indices[env_ids].view(-1)
        flat_root_state = self._root_state.view(-1, 13)
        flat_root_state[flat_dyn_indices, 0:7] = dynamic_obstacle_poses[env_ids, :, :].view(-1, 7) #(num_envs*num_dyn_obs, 7)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(flat_root_state), gymtorch.unwrap_tensor(flat_dyn_indices), flat_dyn_indices.numel(),
        )

        # need to update dynamic pcd
        dynamic_obstacle_pcd_world = transform_pcds_to_world(self.dynamic_obstacle_pcd, dynamic_obstacle_poses)
        self.dynamic_obstacle_pcd_combined = dynamic_obstacle_pcd_world.view(dynamic_obstacle_pcd_world.shape[0], -1, 3)


    def get_observations(self):
        self._refresh()
        joint_pos = self.states['q'][:, 0:7].clone()
        self.robot_pcd[:, :, :] = self.get_robot_pcds(joint_pos)

        env_obs_dict = dict()
        env_obs_dict["joint_pos"] = joint_pos
        env_obs_dict["goal_joint_pos"] = self.goal_joint_pos
        env_obs_dict["total_collision_status"] = self.collision
        env_obs_dict["scene_collision_status"] = self.scene_collision
        env_obs_dict["robot_pcd"] = self.robot_pcd
        env_obs_dict["goal_robot_pcd"] = self.goal_robot_pcd
        env_obs_dict["static_obstacle_pcd"] = self.static_obstacle_pcd
        env_obs_dict["dynamic_obstacle_pcd"] = self.dynamic_obstacle_pcd_combined    
        return env_obs_dict


    def pre_physics_step(self, actions, dynamic_obstacle_poses=None):
        joint_position_targets = actions
        self.apply_joint_pos_targets(joint_position_targets)

        if self.use_dynamic_obstacles:
            dynamic_obstacle_poses = torch.zeros((self.num_envs, self.max_num_dynamic_obstacles, 7), device=self.device)
            dynamic_obstacle_poses[:, :, 0:3] = torch.tensor([1.0, 3, 2.0], device=self.device) #torch.rand((self.num_envs, self.max_num_dynamic_obstacles, 3), device=self.device)
            dynamic_obstacle_poses[:, :, 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

            if dynamic_obstacle_poses is not None:
                self.set_dynamic_obstacle_pose(dynamic_obstacle_poses)
        

    def post_physics_step(self):
        self._refresh()

        self.check_robot_collision()
        self.scene_collision_counter += self.scene_collision.int()
        self.progress_buf += 1

        if self.debug_viz:
            self.gym.clear_lines(self.viewer)
            for i in range(self.num_envs):
                # visualize goal
                px = (self.ee_goal_pose[:, 0:3][i] + quat_apply(self.ee_goal_pose[:, 3:7][i], torch.tensor([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.ee_goal_pose[:, 0:3][i] + quat_apply(self.ee_goal_pose[:, 3:7][i], torch.tensor([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.ee_goal_pose[:, 0:3][i] + quat_apply(self.ee_goal_pose[:, 3:7][i], torch.tensor([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                p0 = self.ee_goal_pose[:, 0:3][i].cpu().numpy()
                self.gym.add_lines(
                    self.viewer, self.env_handles[i], 1, 
                    [p0[0], p0[1], p0[2], px[0], px[1], px[2]], 
                    [0.85, 0.1, 0.1]
                )
                self.gym.add_lines(
                    self.viewer, self.env_handles[i], 1, 
                    [p0[0], p0[1], p0[2], py[0], py[1], py[2]], 
                    [0.1, 0.85, 0.1]
                )
                self.gym.add_lines(
                    self.viewer, self.env_handles[i], 1, 
                    [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], 
                    [0.1, 0.1, 0.85]
                )

                # visualize current ee
                current_ee_pose = torch.cat((self.states["eef_pos"], self.states["eef_quat"]), dim=1)
                px = (current_ee_pose[:, 0:3][i] + quat_apply(current_ee_pose[:, 3:7][i], torch.tensor([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (current_ee_pose[:, 0:3][i] + quat_apply(current_ee_pose[:, 3:7][i], torch.tensor([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (current_ee_pose[:, 0:3][i] + quat_apply(current_ee_pose[:, 3:7][i], torch.tensor([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                p0 = current_ee_pose[:, 0:3][i].cpu().numpy()
                self.gym.add_lines(
                    self.viewer, self.env_handles[i], 1, 
                    [p0[0], p0[1], p0[2], px[0], px[1], px[2]], 
                    [0.85, 0.1, 0.1]
                )
                self.gym.add_lines(
                    self.viewer, self.env_handles[i], 1, 
                    [p0[0], p0[1], p0[2], py[0], py[1], py[2]], 
                    [0.1, 0.85, 0.1]
                )
                self.gym.add_lines(
                    self.viewer, self.env_handles[i], 1, 
                    [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], 
                    [0.1, 0.1, 0.85]
                )


    def get_eval_info(self):
        # reaching rate calculation
        ee_pose = torch.cat((self.states["eef_pos"], self.states["eef_quat"]), dim=1)
        pos_err = torch.norm(ee_pose[:, 0:3] - self.ee_goal_pose[:, 0:3], dim=1)
        quat_err = orientation_error(self.ee_goal_pose[:, 3:], ee_pose[:, 3:])

        # import ipdb; ipdb.set_trace()

        has_reached = (pos_err < 0.05) & (quat_err < 15.0) # 5.0
        reach_rate = torch.sum(has_reached) / self.num_envs

        # collision rate calculation
        total_scene_collision_num = self.scene_collision_counter
        has_collided = self.scene_collision_counter > 0
        collision_rate = torch.sum(has_collided) / self.num_envs

        # success rate calculation
        has_succeeded = has_reached & (~has_collided)
        success_rate = torch.sum(has_succeeded) / self.num_envs

        eval_info_dict = dict()
        eval_info_dict["has_reached"] = has_reached
        eval_info_dict["reach_rate"] = reach_rate
        eval_info_dict["total_scene_collision_num"] = total_scene_collision_num
        eval_info_dict["collision_rate"] = collision_rate
        eval_info_dict["has_succeeded"] = has_succeeded
        eval_info_dict["success_rate"] = success_rate
        return eval_info_dict






        



