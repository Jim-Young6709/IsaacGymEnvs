import yaml

import isaacgym # must import isaacgym before pytorch
import torch
import numpy as np

from isaacgymenvs.tasks import DRPEvals
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.utils.media_utils import camera_shot
from isaacgymenvs.motion_planners import DRPNeuralMP


class Eval:
    def __init__(self):
        self.sim_device = 'cuda:0'
        self.seed = 42
        set_seed(self.seed)

        self.set_up_env()
        self.set_up_motion_planner()


    def set_up_env(self):
        config_file_path = "./cfg/DRPEvals.yaml"
        with open(config_file_path, 'r') as file:
            self.env_cfg = yaml.safe_load(file)

        headless = self.env_cfg['headless']
        force_render = True
        if headless:
            force_render = False
        graphics_device_id = 0
        virtual_screen_capture = False

        self.env = DRPEvals(
            self.env_cfg, self.sim_device, graphics_device_id, headless, virtual_screen_capture, force_render
        )

    def set_up_motion_planner(self):
        self.motion_planner = DRPNeuralMP(self.env)

    def reset_envs(self):
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.env.reset_idx(env_ids)


    @torch.no_grad()
    def test(self):
        self.env.generate_scene_pcd(
            num_robot_points=self.motion_planner.num_robot_points,
            num_goal_robot_points=self.motion_planner.num_goal_robot_points, 
            num_obstacle_points=self.motion_planner.num_obstacle_points,
        )
        testing_epoch_num = 1
        for _ in range(testing_epoch_num):
            self.reset_envs()
            for test_step in range(self.env.max_episode_length):
                print(test_step)
                env_obs_dict = self.env.get_observations()
                joint_pos_targets = self.motion_planner.get_actions(env_obs_dict)
                self.env.step(joint_pos_targets)
            
            # get the eval information
            eval_info_dict = self.env.get_eval_info()

            print("Reach Rate:",     eval_info_dict["reach_rate"])
            print("Collision Rate:", eval_info_dict["collision_rate"])
            print("Success Rate:",   eval_info_dict["success_rate"])
    

    @torch.no_grad()
    def test_open_loop(self):
        self.env.generate_scene_pcd(
            num_robot_points=self.motion_planner.num_robot_points,
            num_goal_robot_points=self.motion_planner.num_goal_robot_points, 
            num_obstacle_points=self.motion_planner.num_obstacle_points,
        )
        testing_epoch_num = 1
        for _ in range(testing_epoch_num):
            self.reset_envs()
            env_obs_dict = self.env.get_observations()

            # (max_episode_length, num_envs, 7)
            joint_pos_targets_buffer = self.motion_planner.get_actions_open_loop(env_obs_dict)

            # roll out open loop
            for i in range(self.env.max_episode_length):
                joint_pos_targets = joint_pos_targets_buffer[i]
                self.env.step(joint_pos_targets)
            
            # get the eval information
            eval_info_dict = self.env.get_eval_info()

            print("Reach Rate:",     eval_info_dict["reach_rate"])
            print("Collision Rate:", eval_info_dict["collision_rate"])
            print("Success Rate:",   eval_info_dict["success_rate"])



            

            
def main():
    agent = Eval()
    agent.test()

if __name__ == "__main__":
    main()
