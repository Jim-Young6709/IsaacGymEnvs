import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask

from isaacgymenvs.utils.demo_loader import DemoLoader
from isaacgymenvs.tasks.utils.pcd_utils import decompose_scene_pcd_params_obs, compute_scene_oracle_pcd


class DRPEvals(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.device = sim_device
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        super().__init__(
            config=self.cfg, rl_device=sim_device, sim_device=sim_device, graphics_device_id=graphics_device_id, 
            headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        # self.load_data_set()
    
        self._refresh()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
    

    def load_data_set(self):
        # Loading environment dataset
        hdf5_path = self.cfg["env"]["asset"]["data_set_path"]
        self.demo_loader = DemoLoader(hdf5_path, self.num_envs)
        # need to change the logic here (2 layers of reset ; multiple start & goal in one env ; relaunch IG)
        data_batch = self.demo_loader.get_next_batch()

        self.start_config = torch.zeros((self.num_envs, 7), device=self.device)
        self.goal_config = torch.zeros((self.num_envs, 7), device=self.device)
        self.obstacle_configs = []
        self.obstacle_handles = []
        self.max_obstacles = 0

        for env_idx, demo in enumerate(data_batch):
            self.start_config[env_idx] = torch.tensor(demo['states'][0][0:7], device=self.device)
            self.goal_config[env_idx] = torch.tensor(demo['states'][0][7:14], device=self.device)

            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.obstacle_configs.append(obstacle_config)
            self.max_obstacles = max(len(obstacle_config[0]), self.max_obstacles)



    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 2.0
            # asset is rotated z-up by default, no additional rotations needed
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.cartpole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)

            self.envs.append(env_ptr)
            self.cartpole_handles.append(cartpole_handle)
    

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)


    def reset_idx(self, env_ids=None):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def pre_physics_step(self, actions):
        pass



    def post_physics_step(self):
        self.progress_buf += 1

