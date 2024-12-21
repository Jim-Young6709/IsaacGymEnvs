import time

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.franka_mp import FrankaMP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from collections import OrderedDict
from omegaconf import DictConfig
from tqdm import tqdm
from geometrout.primitive import Cuboid
from neural_mp.utils.geometry import construct_mixed_point_cloud


class FrankaMPSimple(FrankaMP):
    def _create_envs(self, spacing, num_per_row):
        # setup params
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

        # setup plane (TODO: or do we call it table? need to shift able down to avoid collision?)
        plane_size = [1.0, 1.6, 0.01]
        plane_asset, plane_start_pose = self._create_cube(
            pos=[0.7, 0.0, 0.0],
            size=plane_size,
        )

        obs_asset, obs_start_pose = self._create_cube(
            pos=[0.6, 0.0, 0.0],
            size=[0.6, 0.2, 0.5],
        )

        # initialize combined pcd with mask ids
        num_robot_points = self.pcd_spec_dict['num_robot_points']
        num_scene_points = self.pcd_spec_dict['num_obstacle_points']
        num_target_points = self.pcd_spec_dict['num_target_points']
        self.combined_pcds = torch.cat(
            (
                torch.zeros(num_robot_points, 4, device=self.device),
                torch.ones(num_scene_points, 4, device=self.device),
                2 * torch.ones(num_target_points, 4, device=self.device),
            ),
            dim=0,
        ).repeat(self.num_envs, 1, 1)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = (
            num_franka_bodies + 1 + 1
        )  # 1 for plane, 1 for obstacle
        max_agg_shapes = (
            num_franka_shapes + 1 + 1
        )  # 1 for plane, 1 for obstacle

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

            # Create obstacle
            self.obstacle_actor = self.gym.create_actor(
                env_ptr, obs_asset, obs_start_pose, "obstacle", i, 1, 0
            )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

            # compute the static scene pcd (currently only consider static scenes)
            static_scene_pcd = self.compute_scene_pcds()
            self.combined_pcds[i, num_robot_points:num_robot_points+num_scene_points, :3] = torch.from_numpy(static_scene_pcd).to(self.device)

        # Setup data
        self.zshifts = torch.tensor(self.zshifts, device=self.device)
        actor_num = 1 + 1 + 1
        self.init_data(actor_num=actor_num)

    def plan_open_loop(self, use_controller=True):
        """
        debug base policy
        """
        obs_base = OrderedDict()

        for i in range(150):
            self.update_robot_pcds()
            obs_base["current_angles"] = self.states['q'][:, :7].clone()
            obs_base["goal_angles"] = self.goal_config.clone()
            obs_base["compute_pcd_params"] = self.combined_pcds.clone()
            base_action = self.base_model.policy.get_action(obs_dict=obs_base)
            abs_action = base_action + self.get_joint_angles()
            if use_controller:
                self.step(base_action)
                self._refresh()
            else:
                self.set_robot_joint_state(abs_action)
                self.gym.simulate(self.sim)
                self.render()
                self._refresh()

    def visualize_pcd_meshcat(self):
        "for debug purposes"
        import meshcat
        import urchin
        from robofin.robots import FrankaRobot
        self.viz = meshcat.Visualizer()
        self.urdf = urchin.URDF.load(FrankaRobot.urdf)
        for idx, (k, v) in enumerate(self.urdf.visual_trimesh_fk(np.zeros(8)).items()):
            self.viz[f"robot/{idx}"].set_object(
                meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
                meshcat.geometry.MeshLambertMaterial(wireframe=False),
            )
            self.viz[f"robot/{idx}"].set_transform(v)
        pcd_rgb = np.zeros((3, 8192))
        pcd_rgb[0, :2048] = 1
        pcd_rgb[1, 2048:6144] = 1
        pcd_rgb[2, 6144:] = 1
        
        self.viz['pcd'].set_object(
            meshcat.geometry.PointCloud(
                position=self.combined_pcds[0, :, :3].cpu().numpy().T,
                color=pcd_rgb,
                size=0.005,
            )
        )

    def reset_idx(self, env_ids: np.ndarray = None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        self.start_config = to_torch(
            [[1.5315238, -0.52429098, -2.3127201, -1.6001221, -0.35851285, 1.9580338, -0.97504485]]*self.num_envs, device=self.device
        )
        self.goal_config = to_torch(
            [[0.84051591, 0.11769871, -0.060447611, -2.1588054, -0.15564187, 2.1291728, -0.05379761]]*self.num_envs, device=self.device
        )
        reset_noise = torch.rand((self.num_envs, 7), device=self.device)
        self.start_config = tensor_clamp(
            self.start_config +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])
        reset_noise = torch.rand((self.num_envs, 7), device=self.device)
        self.goal_config = tensor_clamp(
            self.goal_config +
            self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])

        self.goal_ee = self.get_ee_from_joint(self.goal_config)

        self.set_robot_joint_state(self.start_config, debug=False)
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def compute_reward(self, actions):
        self.check_robot_collision()
        current_angles = self.get_joint_angles()
        current_ee = self.get_ee_from_joint(current_angles)

        joint_err = torch.norm(current_angles - self.goal_config, dim=1)
        pos_err = torch.norm(current_ee[:, :3] - self.goal_ee[:, :3], dim=1)
        quat_err = orientation_error(self.goal_ee[:, 3:], current_ee[:, 3:])

        self.rew_buf[:], self.reset_buf[:], d = compute_franka_reward(
            self.reset_buf, self.progress_buf, joint_err, pos_err, quat_err, self.collision, self.max_episode_length
        )

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        if is_last_step:
            self.extras['successes'] = torch.mean(torch.where(d < 0.1, 1.0, 0.0)).item()

    def pre_physics_step(self, actions):
        delta_actions = actions.clone().to(self.device)
        gripper_state = torch.Tensor([[0.035, 0.035]] * self.num_envs).to(self.device)

        delta_actions = torch.clamp(delta_actions, -self.cmd_limit, self.cmd_limit) / self.action_scale
        # delta_actions = delta_actions * self.cmd_limit / self.action_scale
        abs_actions = self.get_joint_angles() + delta_actions
        if abs_actions.shape[-1] == 7:
            abs_actions = torch.cat((abs_actions, gripper_state), dim=1)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))

    def post_physics_step(self):
        super().post_physics_step()
        # reset the robot to start if collision occurs
        if sum(self.collision) > 0:
            collision_env_idx = self.collision.nonzero(as_tuple=False).flatten()
            self.set_robot_joint_state(self.start_config[collision_env_idx], env_ids=collision_env_idx)

    def compute_scene_pcds(self):
        # hardcoded for simple env
        cuboids = [
            Cuboid([0.7, 0.0, 0.005], [1.0, 1.6, 0.01], [1, 0, 0, 0]),
            Cuboid([0.6, 0.0, 0.25], [0.6, 0.2, 0.5], [1, 0, 0, 0]),
        ]
        return construct_mixed_point_cloud(cuboids, self.pcd_spec_dict['num_obstacle_points'])[:, :3]

    def update_robot_pcds(self):
        num_robot_points = self.pcd_spec_dict['num_robot_points']
        num_target_points = self.pcd_spec_dict['num_target_points']
        robot_pcd = self.gpu_fk_sampler.sample(self.get_joint_angles(), num_robot_points)
        target_pcd = self.gpu_fk_sampler.sample(self.goal_config, num_target_points)
        self.combined_pcds[:, :num_robot_points, :3] = robot_pcd
        self.combined_pcds[:, -num_target_points:, :3] = target_pcd

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
    reset_buf, progress_buf, joint_err, pos_err, quat_err, collision_status, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, Tensor]
    exp_r = True
    if exp_r:
        exp_eef = torch.exp(-100*pos_err) + torch.exp(-100*quat_err)
        exp_joint = torch.exp(-100*joint_err)
        # exp_colli = 3*torch.exp(-100*collision_status)
        rewards = exp_eef + exp_joint # + exp_colli
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