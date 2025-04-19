import yaml
import hydra
import isaacgym # must import isaacgym before pytorch
import torch
import numpy as np

from tqdm import tqdm
from omegaconf import DictConfig
from isaacgymenvs.tasks import DRPEvals
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.utils.media_utils import camera_shot
from isaacgymenvs.motion_planners import DRPNeuralMP
from isaacgymenvs.motion_planners import Curobo


class Eval:
    def __init__(self, cfg):
        self.sim_device = 'cuda:0'
        self.seed = 10 #42
        self.env_cfg = cfg
        set_seed(self.seed)

        self.set_up_env()
        self.set_up_motion_planner()

    def set_up_env(self):
        headless = self.env_cfg['headless']
        force_render = True
        if headless:
            force_render = False
        graphics_device_id = 0
        virtual_screen_capture = False

        self.env = DRPEvals(
            self.env_cfg, self.sim_device, graphics_device_id, headless, virtual_screen_capture, force_render
        )
        self.use_controller = self.env_cfg.env.useController
        self.interpolated_substeps = self.env_cfg.env.interpolated_substeps

    def set_up_motion_planner(self):
        planner = self.env_cfg.task.planner
        self.action_chunking = False
        if planner == "Curobo":
            self.motion_planner = Curobo(self.env)
            self.action_chunking = True
        elif planner == "DRP":
            self.motion_planner = DRPNeuralMP(self.env)

    def reset_envs(self):
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.env.reset_idx(env_ids)
        self.motion_planner.reset()

    @torch.no_grad()
    def test_closed_loop(self):
        self.env.generate_scene_pcd(
            num_robot_points=self.motion_planner.num_robot_points,
            num_goal_robot_points=self.motion_planner.num_goal_robot_points,
            num_obstacle_points=self.motion_planner.num_obstacle_points,
        )
        testing_epoch_num = 1
        for _ in range(testing_epoch_num):
            self.reset_envs()
            # for test_step in range(self.env.max_episode_length):
            while self.env.progress_buf[0] < self.env.max_episode_length:
                env_obs_dict = self.env.get_observations()
                joint_pos_targets = self.motion_planner.get_actions(env_obs_dict)
                if self.action_chunking:
                    # action chunking is for curobo, where joint_pos_targets is of shape (15, num_envs, 7)
                    for i in range(15):
                        test_step = self.env.progress_buf[0]
                        print(test_step)
                        self.env.step(joint_pos_targets[i])
                else:
                    test_step = self.env.progress_buf[0]
                    print(test_step)
                    self.env.step(joint_pos_targets)

            
            # get the eval information
            eval_info_dict = self.env.get_eval_info()

            print("Reach Rate:",     eval_info_dict["reach_rate"])
            print("Collision Rate:", eval_info_dict["collision_rate"])
            print("Success Rate:",   eval_info_dict["success_rate"])
            print("Mean Scene Collision Timestep Percentage:", eval_info_dict["mean_scene_collision_timestep_percentage"])
            print("Mean Scene Contact Force Norm Sum:", eval_info_dict["mean_scene_contact_force_norm_sum"])
            print(f"{eval_info_dict['reach_rate'].item():.4f}, {eval_info_dict['collision_rate'].item():.4f}, {eval_info_dict['success_rate'].item():.4f}, {eval_info_dict['mean_scene_collision_timestep_percentage'].item():.4f}, {eval_info_dict['mean_scene_contact_force_norm_sum'].item():.4f}")


    @torch.no_grad()
    def test_open_loop(self):
        self.env.generate_scene_pcd(
            num_robot_points=self.motion_planner.num_robot_points,
            num_goal_robot_points=self.motion_planner.num_goal_robot_points, 
            num_obstacle_points=self.motion_planner.num_obstacle_points,
        )
        testing_epoch_num = 1
        for _ in tqdm(range(testing_epoch_num), desc="Eval epoch"):
            self.reset_envs()
            env_obs_dict = self.env.get_observations()

            # (max_episode_length, num_envs, 7)
            joint_pos_targets_buffer = self.motion_planner.get_actions_open_loop(env_obs_dict)

            # roll out open loop
            for i in tqdm(range(self.env.max_episode_length), desc="Env step"):
                env_obs_dict = self.env.get_observations()
                current_joint_pos = env_obs_dict["joint_pos"]
                joint_pos_targets = joint_pos_targets_buffer[i]
                for i in range(self.interpolated_substeps):
                    sub_joint_pos_targets = (
                        current_joint_pos + (joint_pos_targets - current_joint_pos) * (i + 1) / self.interpolated_substeps
                    )
                    if self.use_controller:
                        self.env.step(sub_joint_pos_targets)
                    else:
                        self.env.set_robot_joint_state(sub_joint_pos_targets)
                        self.env.check_robot_collision()
                        self.env.scene_collision_counter += self.env.scene_collision.int()

            # get the eval information
            eval_info_dict = self.env.get_eval_info()

            print("Reach Rate:",     eval_info_dict["reach_rate"])
            print("Collision Rate:", eval_info_dict["collision_rate"])
            print("Success Rate:",   eval_info_dict["success_rate"])
            print("Mean Scene Collision Timestep Percentage:", eval_info_dict["mean_scene_collision_timestep_percentage"])
            print("Mean Scene Contact Force Norm Sum:", eval_info_dict["mean_scene_contact_force_norm_sum"])


@hydra.main(version_base="1.1", config_name="DRPEvals", config_path="./cfg")
def main(cfg: DictConfig):
    agent = Eval(cfg)
    if cfg.task.close_loop:
        agent.test_closed_loop()
    else:
        agent.test_open_loop()

if __name__ == "__main__":
    main()
