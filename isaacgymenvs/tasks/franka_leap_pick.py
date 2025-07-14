"""
Franka + LEAP Hand Pick Env
"""

import time

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks import FrankaLEAP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.pcd_utils import compute_scene_oracle_pcd
from omegaconf import DictConfig
from tqdm import tqdm
from collections import OrderedDict
from neural_mp.real_utils.model import NeuralMPModel



class FrankaLEAPPick(FrankaLEAP):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # load pretrained encoder TODO: note this is the drp_neural_mp encoder, should use IMPACT one
        self.base_model = NeuralMPModel.from_pretrained("jimyoung6709/DRP_Dagger")
        self.pcd_encoder = self.base_model.policy.nets['policy'].model.nets['encoder'].nets['obs']
        self.pcd_encoder.eval()

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

        # TODO: add full env loading here

    def _create_envs(self, spacing, num_per_row):
        """
        loading Franka + LEAP + a table in the environment, this is for debugging purposes only
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # setup params
        table_thickness = 0.05
        self.cuboid_dims = []  # xyz
        self.capsule_dims = []  # r, l
        self.sphere_radii = []  # r

        # setup robot (franka + leap)
        robot_dof_props = self._create_franka_leap()
        robot_asset = self.robot_asset
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0 + table_thickness / 2)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # setup table
        table_asset, table_start_pose = self._create_cube(
            pos=[0.5, 0.0, 0.0],
            size=[0.7, 1.2, table_thickness],
        )

        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        max_agg_bodies = num_robot_bodies + 1 + 1  # 1 for table, 1 for object
        max_agg_shapes = num_robot_shapes + 1 + 1  # 1 for table, 1 for object

        self.robots = []
        self.objects = []
        self.env_ptrs = []

        # Create environments
        for i in tqdm(range(self.num_envs)):
            # grasp object
            object_asset, object_start_pose, object_scale, object_id, mesh_id = self.create_rand_mesh()

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
            self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, 1, 0
            )

            # Create object
            self._object_id = self.gym.create_actor(
                env_ptr, object_asset, object_start_pose, "object", i, 2, 0
            )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.env_ptrs.append(env_ptr)
            self.robots.append(robot_actor)
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
            )).to(self.device)
            self.object_pcds.append(object_pcd_i)

        self.static_pcds = torch.stack(self.static_pcds, dim=0).to(self.device) # (num_envs, num_points, 3)
        self.object_pcds = torch.stack(self.object_pcds, dim=0).to(self.device)
        self.combined_pcds = torch.cat([self.static_pcds, self.object_pcds], dim=1).to(self.device) # (num_envs, num_static_points + num_object_points, 3)

        # Setup data
        actor_num = 1 + 1 + 1  # robot, table, object
        self.init_data(actor_num=actor_num)

    def _reset_obstacle(self): # TODO: add reset logic here, randomize pos, ori
        pass

    def compute_observations(self):
        self._refresh()

        obs_base = OrderedDict()
        dummy_config = torch.ones((self.num_envs, 7), device=self.device, dtype=torch.float32)
        zero_padding = torch.zeros(self.num_envs, self.combined_pcds.shape[1], 1, device=self.device, dtype=torch.float32)
        input_pcd = torch.cat([self.combined_pcds, zero_padding], dim=-1).to(torch.float32)
        obs_base["current_angles"] = dummy_config.clone()
        obs_base["goal_angles"] = dummy_config.clone()
        obs_base["compute_pcd_params"] = input_pcd
        pcd_latent = self.pcd_encoder(obs_base)
        pcd_latent = pcd_latent[:, :1024]

        obs_components = ["q", "eef_pos", "eef_rot_6d",
                          "eef_finger1_pos", "eef_finger2_pos", "eef_finger3_pos", "eef_finger4_pos",
                          "object_pos", "object_rot_6d", "hand_to_object",
                          "object_to_target", "object_target_6d_diff"]

        states_components = ["q", "eef_pos", "eef_rot_6d",
                          "eef_finger1_pos", "eef_finger2_pos", "eef_finger3_pos", "eef_finger4_pos",
                          "object_pos", "object_rot_6d", "hand_to_object",
                          "object_to_target", "object_target_6d_diff"]

        obs_buf = torch.cat([self.states[ob] for ob in obs_components] + [pcd_latent], dim=-1)
        states_buf = torch.cat([self.states[st] for st in states_components] + [pcd_latent], dim=-1)

        self.obs_buf = obs_buf
        self.states_buf = states_buf

        return self.obs_buf

    def reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        reset_noise = torch.rand((len(env_ids), 23), device=self.device)
        reset_config = tensor_clamp(
            self.canonical_joint_config[env_ids] +
            0.1 * 2.0 * (reset_noise - 0.5),
            self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        self.set_robot_joint_state(reset_config, env_ids=env_ids)

        self._reset_obstacle()
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.compute_observations()

    def pre_physics_step(self, actions):
        """
        Args:
            actions (torch.Tensor): delta unnormalized joint angles (num_selected_envs, 7+4*4)
        """
        delta_actions = actions * self.action_scale
        self.actions = delta_actions
        abs_actions = self.states['q'] + delta_actions # TODO: not sure if should directly use self.states, need to really make sure its always up to date
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.reset_buf[:] = torch.where((self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.rew_buf[:] = compute_franka_leap_reward(self.states, self.reward_settings)

@torch.jit.script
def compute_franka_leap_reward(states, reward_settings):
    # type: (Dict[str, Tensor], Dict[str, Tensor]) -> Tensor

    # Hand (palm, fingers) to object distance
    d_palm = torch.norm(states["object_pos"] - states["eef_pos"], dim=-1)
    d_finger1 = torch.norm(states["object_pos"] - states["eef_finger1_pos"], dim=-1)
    d_finger2 = torch.norm(states["object_pos"] - states["eef_finger2_pos"], dim=-1)
    d_finger3 = torch.norm(states["object_pos"] - states["eef_finger3_pos"], dim=-1)
    d_finger4 = torch.norm(states["object_pos"] - states["eef_finger4_pos"], dim=-1)
    
    # Max dist component to object: max_i∈{palm_pos,fingertips} ||x^i - x^obj||
    d_hand_obj = torch.stack([d_palm, d_finger1, d_finger2, d_finger3, d_finger4], dim=1)
    d_hand_obj = torch.max(d_hand_obj, dim=1)[0]
    
    # Hand object distance reward
    r_hand_obj = torch.exp(-d_hand_obj)

    # Goal Reward
    target_pos = reward_settings["target_pos"].squeeze(-1)
    d_obj_goal = torch.norm(states["object_pos"] - target_pos, dim=-1)
    r_obj_goal = torch.exp(-d_obj_goal)

    rewards = r_hand_obj + r_obj_goal

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
    env = FrankaLEAPPick(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    for i in tqdm(range(1000)):
        t1 = time.time()
        env.reset_idx()
        import ipdb ; ipdb.set_trace()
        t2 = time.time()
        print(f"Reset time: {t2 - t1}")
        env.render()


if __name__ == "__main__":
    launch_test()
