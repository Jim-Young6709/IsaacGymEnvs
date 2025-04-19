import os
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
        set_seed(self.seed)
        self.cfg = cfg

        self.set_up_problem_configs()
        self.set_up_env()
        self.set_up_motion_planner()
    

    def set_up_problem_configs(self):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        problem_config_path = os.path.join(
            current_file_dir, "cfg/eval_problems", self.cfg.task.task_type, f"{self.cfg.task.task_name}.yaml"
        )
        with open(problem_config_path, 'r') as file:
            self.problem_config = yaml.safe_load(file)
        self.problem_config["static_scene_path"] = os.path.join(
            current_file_dir, "static_scenes", f"{self.problem_config['static_scene']}.hdf5"
        )
        self.testing_epoch_num = 1
        if self.cfg.task.task_type == "quasi_dynamic":
            self.testing_epoch_num = 2
        

    def set_up_env(self):
        headless = self.cfg.headless
        force_render = True
        if headless:
            force_render = False
        else:
            self.cfg.env.numEnvs = 32
        graphics_device_id = 0
        virtual_screen_capture = False

        self.env = DRPEvals(
            self.cfg, self.sim_device, graphics_device_id, headless, virtual_screen_capture, force_render, self.problem_config
        )
        self.interpolated_substeps = 10


    def set_up_motion_planner(self):
        planner = self.cfg.task.planner
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
        for current_epoch_num in range(self.testing_epoch_num):
            self.env.test_epoch = current_epoch_num
            self.reset_envs()
            
            if self.action_chunking:
                iteration = self.env.max_episode_length // 15 + 1
            else:
                iteration = self.env.max_episode_length

            step_size = 15 if self.action_chunking else 1
            for t in tqdm(range(0, self.env.max_episode_length, step_size), desc=f"Eval Epoch Num {current_epoch_num}"):
                env_obs_dict = self.env.get_observations()
                joint_pos_targets = self.motion_planner.get_actions(env_obs_dict)
                if self.action_chunking:
                    # action chunking is for curobo, where joint_pos_targets is of shape (15, num_envs, 7)
                    for i in range(15):
                        self.env.step(joint_pos_targets[i])
                else:
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
        for _ in tqdm(range(self.testing_epoch_num), desc="Eval epoch"):
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
                    self.env.step(sub_joint_pos_targets)

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
