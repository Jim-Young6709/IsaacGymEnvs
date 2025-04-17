import numpy as np
import os
import torch

from .base.vec_task import VecTask

from isaacgym.torch_utils import *
from isaacgym import gymutil, gymtorch, gymapi

from isaacgymenvs.utils.demo_loader import DemoLoader
# from isaacgymenvs.tasks.utils.pcd_utils import decompose_scene_pcd_params_obs, compute_scene_oracle_pcd


class DRPEvals(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.device = sim_device
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_obstacles = 0
        self.num_dyn_objs = 0
        # self.load_data_set()

        super().__init__(
            config=self.cfg, rl_device=sim_device, sim_device=sim_device, graphics_device_id=graphics_device_id, 
            headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        
    
        self._refresh()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
    

    # def load_data_set(self):
    #     # Loading environment dataset
    #     hdf5_path = self.cfg["env"]["asset"]["data_set_path"]
    #     self.demo_loader = DemoLoader(hdf5_path, self.num_envs)
    #     # need to change the logic here (2 layers of reset ; multiple start & goal in one env ; relaunch IG)
    #     data_batch = self.demo_loader.get_next_batch()

    #     self.start_config = torch.zeros((self.num_envs, 7), device=self.device)
    #     self.goal_config = torch.zeros((self.num_envs, 7), device=self.device)
    #     self.obstacle_configs = []
    #     self.obstacle_handles = []
    #     self.max_obstacles = 0

    #     for env_idx, demo in enumerate(data_batch):
    #         self.start_config[env_idx] = torch.tensor(demo['states'][0][0:7], device=self.device)
    #         self.goal_config[env_idx] = torch.tensor(demo['states'][0][7:14], device=self.device)

    #         pcd_params = demo['states'][0][15:]
    #         obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
    #         self.obstacle_configs.append(obstacle_config)
    #         self.max_obstacles = max(len(obstacle_config[0]), self.max_obstacles)


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
        return franka_dof_props
    

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        # setup franka
        franka_dof_props = self._create_franka()
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.frankas = []
        self.env_ptrs = []

        # create environments
        for i in range(num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            # create franka
            franka_actor = self.gym.create_actor(
                env_ptr, self.franka_asset, franka_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)





            # store the created env pointers
            self.env_ptrs.append(env_ptr)
            self.frankas.append(franka_actor)


        actor_num = 1 + self.max_obstacles + self.num_dyn_objs
        self._init_data(actor_num=actor_num)
    

    def _init_data(self, actor_num):
        # setup sim handles
        env_ptr = self.env_ptrs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
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

    

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)



    def apply_joint_pos_targets(self, joint_pos_targets):
        gripper_targets = torch.tensor([0.04, 0.04], device=self.device).repeat(self.num_envs, 1)
        franka_actions = torch.cat([joint_pos_targets, gripper_targets], dim=1)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(franka_actions))


    def reset_idx(self, env_ids=None):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        


    def pre_physics_step(self, actions):
        joint_position_targets = actions
        self.apply_joint_pos_targets(joint_position_targets)
        

        
        

    def post_physics_step(self):
        self._refresh()

        self.progress_buf += 1

