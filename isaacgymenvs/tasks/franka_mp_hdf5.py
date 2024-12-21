import time

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.franka_mp import FrankaMP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.demo_loader import DemoLoader
from omegaconf import DictConfig
from tqdm import tqdm

def decompose_scene_pcd_params_obs(scene_pcd_params):
    """
    Decompose the pcd params observation.
    """
    M = int(scene_pcd_params[0])

    cuboid_params = scene_pcd_params[1 : 1 + 10 * M]
    cuboid_dims = cuboid_params[: 3 * M].reshape(-1, 3)
    cuboid_centers = cuboid_params[3 * M : 6 * M].reshape(-1, 3)
    cuboid_quats = cuboid_params[6 * M : 10 * M].reshape(-1, 4)

    cylinder_params = scene_pcd_params[1 + 10 * M : 1 + 10 * M + 9 * M]
    cylinder_radii = cylinder_params[: 1 * M].reshape(-1)
    cylinder_heights = cylinder_params[1 * M : 2 * M].reshape(-1)
    cylinder_centers = cylinder_params[2 * M : 5 * M].reshape(-1, 3)
    cylinder_quats = cylinder_params[5 * M : 9 * M].reshape(-1, 4)

    sphere_params = scene_pcd_params[1 + 10 * M + 9 * M : 1 + 10 * M + 9 * M + 4 * M]
    sphere_centers = sphere_params[: 3 * M].reshape(-1, 3)
    sphere_radii = sphere_params[3 * M : 4 * M].reshape(-1)

    mesh_params = scene_pcd_params[1 + 10 * M + 9 * M + 4 * M :]
    mesh_positions = mesh_params[: 3 * M].reshape(-1, 3)
    mesh_scales = mesh_params[3 * M : 4 * M].reshape(-1)
    mesh_quaternions = mesh_params[4 * M : 8 * M].reshape(-1, 4)
    obj_ids = mesh_params[8 * M : 9 * M].reshape(-1)
    mesh_ids = mesh_params[9 * M : 10 * M].reshape(-1)

    return (
        np.array(cuboid_dims).astype(np.float32),
        np.array(cuboid_centers).astype(np.float32),
        np.array(cuboid_quats).astype(np.float32),
        np.array(cylinder_radii).astype(np.float32),
        np.array(cylinder_heights).astype(np.float32),
        np.array(cylinder_centers).astype(np.float32),
        np.array(cylinder_quats).astype(np.float32),
        np.array(sphere_centers).astype(np.float32),
        np.array(sphere_radii).astype(np.float32),
        np.array(
            mesh_positions,
        ).astype(np.float32),
        np.array(
            mesh_scales,
        ).astype(np.float32),
        np.array(
            mesh_quaternions,
        ).astype(np.float32),
        np.array(
            obj_ids,
        ).astype(np.float32),
        np.array(
            mesh_ids,
        ).astype(np.float32),
    )

class FrankaMPSimple(FrankaMP):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # self.MAX_CUBES = 20  
        # self.MAX_CYLINDERS = 20
        self.MAX_OBSTACLES = 40
        self.device = sim_device
        print("Before super init:")
        print("self.device:", self.device)
        print("sim_device:", sim_device)
        print("rl_device:", rl_device)
        
        # Demo loading
        hdf5_path = '/home/jimyoung/Neural_MP_Proj/neural_mp/datasets/lerobot_debug100.hdf5'
        self.demo_loader = DemoLoader(hdf5_path, cfg["env"]["numEnvs"])
        self.initial_batch = self.demo_loader.get_next_batch()
        
        # Initialize configs
        self.start_config = torch.zeros((cfg["env"]["numEnvs"], 7), device=self.device)
        self.goal_config = torch.zeros((cfg["env"]["numEnvs"], 7), device=self.device)
        
        self.obstacle_handles = []
        
        self.initial_obstacle_configs = []
        for demo in self.initial_batch:
            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.initial_obstacle_configs.append(obstacle_config)
        
        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        
        print("After super init:")
        print("self.device:", self.device)

    def _create_envs(self, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self.cuboid_dims = []  # xyz
        self.capsule_dims = []  # r, l
        self.sphere_radii = []  # r

        # Setup franka
        franka_dof_props = self._create_franka()
        franka_asset = self.franka_asset
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.01)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + self.MAX_OBSTACLES  # franka + obstacles
        max_agg_shapes = num_franka_shapes + self.MAX_OBSTACLES

        self.frankas = []
        self.envs = []
        
        # Create environments
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

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
            (
                cuboid_dims, 
                cuboid_centers, 
                cuboid_quats,
                cylinder_radii, 
                cylinder_heights,
                cylinder_centers,
                cylinder_quats,
                *_
            ) = self.initial_obstacle_configs[i]
            
            num_cubes = len(cuboid_dims)
            # num_cylinders = len(cylinder_radii) #pausing cylinders due to incorrect spawning. Likely an actor indexing issue.
            
            # Create actual obstacles with proper sizes
            for j in range(self.MAX_OBSTACLES):
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
                        pos=[100.0, 100.0, 100.0],
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
                
            self.obstacle_handles.append(env_obstacles)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup data
        actor_num = 1 + self.MAX_OBSTACLES  # franka  + obstacles
        self.init_data(actor_num=actor_num)
    
    def update_obstacle_configs_from_batch(self, batch_data):
        """Update obstacle configurations from a new batch of demos."""
        self.initial_obstacle_configs = []
        for demo in batch_data:
            pcd_params = demo['states'][0][15:]
            obstacle_config = decompose_scene_pcd_params_obs(pcd_params)
            self.initial_obstacle_configs.append(obstacle_config)
            
    def reset_idx(self, env_ids=None):
        if env_ids is None:
            # print("env ids passed as none")
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        if len(env_ids) == self.num_envs and self.demo_loader is not None and self.demo_loader.has_more_data():
            print("initiating reset")
            batch_data = self.demo_loader.get_next_batch()
            if batch_data is not None:
                # TODO After first update the cubes get messed up. 
                # self.update_obstacle_configs_from_batch(batch_data)
                for env_idx, demo in enumerate(batch_data):
                    self.start_config[env_idx] = torch.tensor(demo['states'][0][:7], device=self.device)
                    self.goal_config[env_idx] = torch.tensor(demo['states'][0][7:14], device=self.device)
                    
                    obstacle_config = self.initial_obstacle_configs[env_idx]
                    (
                        cuboid_dims, 
                        cuboid_centers, 
                        cuboid_quats,
                        *_
                    ) = obstacle_config
                    
                    obstacle_start_idx = 1
                    num_actors = self._root_state.shape[1]  # Get total available actor slots
                    # print(f"Shape of _root_state: {self._root_state.shape}")
                    # print(f"Number of cubes: {len(cuboid_centers)}")
                    # print(f"Number of cylinders: {len(cylinder_centers)}")


                    for i in range(self.MAX_OBSTACLES):
                        
                        actor_idx = obstacle_start_idx + i
                        
                        if actor_idx >= num_actors:
                            # print(f"Warning: Trying to access actor_idx {actor_idx} but only have {num_actors} slots")
                            break
    
                        if i < len(cuboid_centers):
                            self._root_state[env_idx, actor_idx, 0:3] = torch.tensor(cuboid_centers[i], device=self.device)
                            self._root_state[env_idx, actor_idx, 3:7] = torch.tensor(cuboid_quats[i], device=self.device)
                            self._root_state[env_idx, actor_idx, 7:13] = 0
                        else:
                            if actor_idx < num_actors:
                                self._root_state[env_idx, actor_idx] = torch.tensor([100.0, 100.0, 100.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], device=self.device)
                
                env_indices = self._global_indices[env_ids].flatten()
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self._root_state),
                    gymtorch.unwrap_tensor(env_indices),
                    len(env_indices)
                )
        else:
            print("reset not successful")
        reset_noise = torch.rand((self.num_envs, 7), device=self.device)
        pos = tensor_clamp(
            self.start_config + 
            self.franka_dof_noise*0,
            self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])
        self.set_robot_joint_state(pos, debug=True)
        self.progress_buf[env_ids] = 0 
        self.reset_buf[env_ids] = 0

    def compute_reward(self, actions):
        self.check_robot_collision()
        self.rew_buf[:], self.reset_buf[:], d = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.states, self.goal_config, self.collision, self.max_episode_length
        )

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        if is_last_step:
            self.extras['successes'] = torch.mean(torch.where(d < 0.01, 1.0, 0.0)).item()

    def pre_physics_step(self, actions):
        print("pre")
        delta_actions = actions.clone().to(self.device)
        gripper_state = torch.Tensor([[0.035, 0.035]] * self.num_envs).to(self.device)
        delta_actions = delta_actions * self.cmd_limit / self.action_scale
        abs_actions = self.get_joint_angles() + delta_actions
        if abs_actions.shape[-1] == 7:
            abs_actions = torch.cat((abs_actions, gripper_state), dim=1)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))
        
    # def post_physics_step(self):
    #     super().post_physics_step()
        
        # Only print environment #0 every 10 ticks
        env_to_track = 0  # Change this to track a different environment
        # if self.progress_buf[0] % 10 == 0:
        joint_angles = self.get_joint_angles()
        current_angles = [f"{a:.3f}" for a in joint_angles[env_to_track].cpu().numpy()]
        start_angles = [f"{a:.3f}" for a in self.start_config[env_to_track].cpu().numpy()]
        print(f"\nEnv {env_to_track} - Tick {self.progress_buf[0]}:")
        print(f"Current angles: [{', '.join(current_angles)}]")
        print(f"Start config:   [{', '.join(start_angles)}]")
        diff = torch.norm(joint_angles[env_to_track] - self.start_config[env_to_track])
        print(f"Difference magnitude: {diff:.3f}")

@hydra.main(config_name="config", config_path="../cfg/")
def launch_test(cfg: DictConfig):
    import isaacgymenvs
    from isaacgymenvs.learning import amp_continuous, amp_models, amp_network_builder, amp_players
    from isaacgymenvs.utils.rlgames_utils import (
        RLGPUAlgoObserver,
        RLGPUEnv,
        get_rlgames_env_creator,
    )
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

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
    env = FrankaMPSimple(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    total_error = 0
    num_failed_plans = 0
    num_plans = 1000
    for i in tqdm(range(num_plans)):
        t1 = time.time()
        env.reset_idx()
        t2 = time.time()
        current_angles = env.get_joint_angles()
        diff = current_angles - env.start_config
        
        # Print start configs for first few envs
        en = min(20, env.num_envs)
        # print(f"\nStart configs for first {en} environments:")
        # for env_idx in range(min(20, env.num_envs)):
            # print(f"Env {env_idx} start config: [{', '.join(f'{x:.6f}' for x in env.start_config[env_idx])}]")
            # print(f"Env {env_idx} current angles: [{', '.join(f'{x:.6f}' for x in current_angles[env_idx])}]")
            # print(f"Env {env_idx} differences: [{', '.join(f'{x:.6f}' for x in diff[env_idx])}]")
            # print()
        # current_angles = env.get_joint_angles()
        # diff = current_angles - env.start_config
        # print(f"current Angles: [{', '.join(f'{d:.6f}' for d in current_angles[0])}]")  # Format each joint difference
        # print(f"start diff: [{', '.join(f'{d:.6f}' for d in env.start_config[0])}]")  # Format each joint difference
        # print(f"Joint differences: [{', '.join(f'{d:.6f}' for d in diff[0])}]")  # Format each joint difference

        # env.set_robot_joint_state(env.start_config)
        # env.print_resampling_info(env.start_config)
        # env.render()
        # print(f"Reset time: {t2 - t1}")
        # print("\nvalidation checking:")
        # env.set_robot_joint_state(env.start_config)
        # env.print_resampling_info(env.start_config)

        env.render()

    print(f"Average Error: {total_error / num_plans}")
    print(f"Percentage of failed plans: {num_failed_plans / num_plans * 100} ")

def orientation_error(desired, current):
    batch_diff = int(current.shape[0] / desired.shape[0])
    desired = desired.repeat(batch_diff, 1)
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return torch.abs((q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)).mean(dim=1))
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, states, goal, collision_status, max_episode_length
):
    # type: (Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    target_pos = torch.tensor([0.3858, 0.3525, 0.2422]).to(states["eef_pos"].device).unsqueeze(0)
    target_quat = torch.tensor([0.6524, 0.7514, -0.0753, 0.0641]).to(states["eef_quat"].device).unsqueeze(0)
    pos_err = torch.norm(states["eef_pos"] - target_pos, dim=1)
    quat_err = orientation_error(target_quat, states["eef_quat"])
    joint_err = torch.norm(states["q"][:, :7] - goal, dim=1)

    exp_r = True
    if exp_r:
        exp_eef = torch.exp(-pos_err) + torch.exp(-10*pos_err) + torch.exp(-100*pos_err) + torch.exp(-quat_err) + torch.exp(-10*quat_err) + torch.exp(-100*quat_err)
        exp_joint = torch.exp(-joint_err) + torch.exp(-10*joint_err) + torch.exp(-100*joint_err)
        exp_colli = 3*torch.exp(-100*collision_status)
        rewards = exp_eef + exp_joint + exp_colli
    else:
        eef_reward = 1.0 - (torch.tanh(10*pos_err)+torch.tanh(10*quat_err))/2.0
        joint_reward = 1.0 - torch.tanh(10*joint_err)
        collision_reward = 1.0 - collision_status
        rewards = eef_reward + joint_reward + collision_reward
    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf, joint_err


if __name__ == "__main__":
    launch_test()
