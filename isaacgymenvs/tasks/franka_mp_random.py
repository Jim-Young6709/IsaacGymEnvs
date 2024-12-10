import time

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.franka_mp import FrankaMP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from tqdm import tqdm


class FrankaMPRandom(FrankaMP):
    def add_random_objects(self, num_objects, size_sampler, add_func, kwargs={}):
        """
        Add random objects to the environment.

        Args:
            num_objects (int): Number of objects to add.
            add_func (function): Function to add the object.
            is_cylinder (bool): Whether to add a cylinder or a capsule.
        """
        objs = []
        zshifts = []
        for _ in range(num_objects):
            size = size_sampler().astype(np.float32)
            xypos = self.sample_random_xypos()
            position = np.concatenate([xypos, [0]]).astype(np.float32)
            asset, start_pose = add_func(position, size, **kwargs)
            objs.append((asset, start_pose))
            zshifts.append(start_pose.p.z)
        return objs, zshifts

    def sample_random_xypos(self):
        inner_box_size = self.inner_box_size
        regions = ["left", "right", "top", "bottom"]
        left_right_prob = (-inner_box_size - (-1)) * (inner_box_size - (-inner_box_size))
        top_bottom_prob = (1 - (-1)) * (1 - inner_box_size)
        probabilities = [left_right_prob, left_right_prob, top_bottom_prob, top_bottom_prob]
        region = np.random.choice(regions, p=probabilities / np.sum(probabilities))
        if region == "left":
            xypos = np.random.uniform(
                low=[-1, -inner_box_size], high=[-inner_box_size, inner_box_size]
            )
        elif region == "right":
            xypos = np.random.uniform(
                low=[inner_box_size, -inner_box_size], high=[1, inner_box_size]
            )
        elif region == "top":
            xypos = np.random.uniform(low=[-1, inner_box_size], high=[1, 1])
        elif region == "bottom":
            xypos = np.random.uniform(low=[-1, -1], high=[1, -inner_box_size])
        return xypos

    def _create_randomized_obstacles(self, num_obstacles: int):
        # sample and place num_obstacles/3 boxes
        size_sampler = lambda: np.random.uniform(low=0.1, high=0.3, size=(3,))
        cubes, zshift_cubes = self.add_random_objects(
            int(num_obstacles / 3), size_sampler, self._create_cube
        )

        # sample and place num_obstacles/3 capsules
        size_sampler = lambda: np.random.uniform(low=0.1, high=0.3, size=(2,))
        capsules, zshift_capsules = self.add_random_objects(
            int(num_obstacles / 3), size_sampler, self._create_capsule
        )

        # sample and place num_obstacles/3 spheres
        size_sampler = lambda: np.random.uniform(low=0.1, high=0.3, size=(1,))
        sphere, zshift_spheres = self.add_random_objects(
            int(num_obstacles / 3), size_sampler, self._create_sphere
        )

        obstacles = cubes + capsules + sphere  # List[(asset, start_pose)]
        zshifts = zshift_cubes + zshift_capsules + zshift_spheres  # List[float]
        return obstacles, zshifts

    def _reset_obstacle(self, env_ids=None):
        if env_ids is None:
            env_ids = range(self.num_envs)
        self._obstacle_state[env_ids, :, :2] = torch.zeros_like(
            self._obstacle_state[env_ids, :, :2]
        )
        # set obstacles below ground plane
        self._obstacle_state[env_ids, :, 2] = -self.zshifts[env_ids] - 0.01
        for env_id in env_ids:
            num_cuboids = np.random.randint(1, self.max_num_cuboids)
            num_spheres = np.random.randint(1, self.max_num_spheres)
            num_capsules = np.random.randint(1, self.max_num_capsules)

            cache_num_each = int(self.max_num_obstacles / 3)
            selected_idx = []
            cuboids_idx = np.random.choice(range(cache_num_each), size=num_cuboids, replace=False)
            spheres_idx = np.random.choice(
                range(cache_num_each, cache_num_each * 2), size=num_spheres, replace=False
            )
            capsules_idx = np.random.choice(
                range(cache_num_each * 2, cache_num_each * 3), size=num_capsules, replace=False
            )
            selected_idx = np.concatenate((cuboids_idx, spheres_idx, capsules_idx))
            self._obstacle_state[env_id, selected_idx, 2] = self.zshifts[env_id, selected_idx]

            for idx in selected_idx:
                self._obstacle_state[env_id, idx, :2] = torch.tensor(
                    self.sample_random_xypos(), device=self.device
                )

        obs_idx = self._global_indices[env_ids, 2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(obs_idx),
            len(obs_idx),
        )

    def _create_envs(self, spacing, num_per_row):
        # setup params
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.max_num_obstacles = 30
        self.max_num_cuboids = 5
        self.max_num_capsules = 5
        self.max_num_spheres = 5
        self.inner_box_size = 0.3
        self.cuboid_dims = []  # xyz
        self.capsule_dims = []  # r, l
        self.sphere_radii = []  # r

        # setup franka
        franka_dof_props = self._create_franka()
        franka_asset = self.franka_asset
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.01)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # setup plane (TODO: or do we call it table?)
        plane_size = [2, 2, 0.01]
        plane_asset, plane_start_pose = self._create_cube(
            pos=[0.0, 0.0, 0.0],
            size=plane_size,
        )

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = (
            num_franka_bodies + 1 + self.max_num_obstacles
        )  # 1 for plane, num_obstacles
        max_agg_shapes = (
            num_franka_shapes + 1 + self.max_num_obstacles
        )  # 1 for plane, num_obstacles

        self.frankas = []
        self.envs = []
        self.zshifts = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

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

            # Create table
            self.cuboid_dims.append(plane_size)
            self.plane_actor = self.gym.create_actor(
                env_ptr, plane_asset, plane_start_pose, "plane", i, 1, 0
            )

            # Create obstacles
            obstacles, zshifts = self._create_randomized_obstacles(self.max_num_obstacles)
            self.zshifts.append(zshifts)
            for obstacle in obstacles:
                obs_asset, obs_start_pose = obstacle
                self.gym.create_actor(env_ptr, obs_asset, obs_start_pose, "obstacle", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup data
        self.zshifts = torch.tensor(self.zshifts, device=self.device)
        actor_num = 1 + 1 + self.max_num_obstacles
        self.init_data(actor_num=actor_num)

    def init_data(self, actor_num):
        super().init_data(actor_num=actor_num)
        # format of _root_state: (num_envs, actor_num<franka_actor, plane_actor, obstacle_actor>, 13)
        # format of _obstacle_state: (num_envs, max_num_obstacles<max_num_cuboids, max_num_capsules, max_num_spheres>, 13)
        self._obstacle_state = self._root_state[:, 2:, :]

    def compute_reward(self, actions):
        self.check_robot_collision()
        self.rew_buf[:], self.reset_buf[:], successes = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.states, self.goal_config, self.goal_pose, self.collision, self.max_episode_length
        )

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        self.successes += successes
        self.num_collisions += self.collision
        if is_last_step:
            self.extras['successes'] = torch.mean(torch.where(self.successes >= 1, 1.0, 0.0)).item()
            self.extras['has_collisions'] = torch.mean(torch.where(self.num_collisions >= 1, 1.0, 0.0)).item()
            self.successes = torch.zeros_like(self.successes)
            self.num_collisions = torch.zeros_like(self.num_collisions)

    def pre_physics_step(self, actions):
        """
            takes in delta actions
        """
        delta_actions = actions.clone().to(self.device)
        gripper_state = torch.Tensor([[0.035, 0.035]] * self.num_envs).to(self.device)
        delta_actions = torch.clamp(delta_actions, -self.cmd_limit, self.cmd_limit) / self.action_scale
        abs_actions = self.get_joint_angles() + delta_actions
        if abs_actions.shape[-1] == 7:
            abs_actions = torch.cat((abs_actions, gripper_state), dim=1)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))

def orientation_error(desired, current):
    batch_diff = int(current.shape[0] / desired.shape[0])
    desired = desired.repeat(batch_diff, 1)
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return torch.abs((q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)).mean(dim=1))

@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, states, goal_config, goal_pose, collision_status, max_episode_length
):
    # type: (Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    pos_err = torch.norm(goal_pose[:,:3] - states["eef_pos"], dim=1)
    quat_err = orientation_error(goal_pose[:, 3:], states["eef_quat"])
    joint_err = torch.norm(goal_config - states["q"][:, :7], dim=1)

    exp_r = True
    if exp_r:
        exp_eef = 3*torch.exp(-100*pos_err) + 3*torch.exp(-100*quat_err)
        exp_joint = 3*torch.exp(-10*joint_err)
        exp_colli = 3*torch.exp(-100*collision_status)
        rewards = 3*exp_eef + exp_joint + exp_colli
        print(f"rewards: {exp_eef.mean()}|{exp_joint.mean()}|{exp_colli.mean()}|{rewards.mean()}")
    else:
        eef_reward = 0#1.0 - (torch.tanh(10*pos_err)+torch.tanh(10*quat_err))/2.0
        joint_reward = 1.0 - torch.tanh(10*joint_err)
        collision_reward = 1.0 - collision_status
        rewards = eef_reward + joint_reward + collision_reward
    # Compute resets
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    # Compute successes
    successes = torch.where( (pos_err*100 <= 1) * (quat_err*180./torch.pi <=15) , 1.0, 0.0)

    return rewards, reset_buf, successes

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
    env = FrankaMPRandom(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    total_error = 0
    num_failed_plans = 0
    num_plans = 1000
    for i in tqdm(range(num_plans)):
        t1 = time.time()
        env.reset_env()
        t2 = time.time()
        print(f"Reset time: {t2 - t1}")
        print("\nvalidation checking:")
        env.set_robot_joint_state(env.start_config)
        env.print_resampling_info(env.start_config)

        env.render()

    print(f"Average Error: {total_error / num_plans}")
    print(f"Percentage of failed plans: {num_failed_plans / num_plans * 100} ")


if __name__ == "__main__":
    launch_test()
