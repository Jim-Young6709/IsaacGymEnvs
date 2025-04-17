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
        self.set_up_motion_planner()
        self.set_up_env()


    def set_up_motion_planner(self):
        self.motion_planner = DRPNeuralMP()

    def set_up_env(self):
        config_file_path = "./cfg/DRPEvals.yaml"
        with open(config_file_path, 'r') as file:
            self.cfg = yaml.safe_load(file)
        headless = self.cfg['headless']
        force_render = True
        if headless:
            force_render = False
        graphics_device_id = 0
        virtual_screen_capture = False

        self.cfg["pcd_spec"]["num_robot_points"] = self.motion_planner.num_robot_points
        self.cfg["pcd_spec"]["num_goal_robot_points"] = self.motion_planner.num_goal_robot_points
        self.cfg["pcd_spec"]["num_obstacle_points"] = self.motion_planner.num_obstacle_points

        self.env = DRPEvals(
            self.cfg, self.sim_device, graphics_device_id, headless, virtual_screen_capture, force_render
        )
       
    def reset_envs(self):
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.env.reset_idx(env_ids)


    @torch.no_grad()
    def test(self):
        while True:
            self.reset_envs()
            for test_step in range(self.env.max_episode_length - 1):
                env_obs_dict = self.env.get_observations()
                joint_pos_targets = self.motion_planner.get_actions(env_obs_dict)

                joint_pos_targets = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.sim_device)
                # joint_pos_targets = torch.rand((self.env.num_envs, self.env.num_actions), device=self.sim_device)


                obs_dict, rews, dones, infos = self.env.step(joint_pos_targets)
                pass

            
def main():
    # torch._dynamo.config.disable = True
    # torch.backends.cudnn.benchmark = True
    # torch.set_float32_matmul_precision("medium")
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True

    agent = Eval()
    agent.test()




if __name__ == "__main__":
    main()
