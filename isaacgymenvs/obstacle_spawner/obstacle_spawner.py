
import yaml
import torch
import numpy as np


class ObstacleSpawner:
    def __init__(self, env, num_envs, config_path):
        self.env = env
        self.num_envs = num_envs
        self.device = self.env.device
        self.max_episode_length = self.env.max_episode_length
        
        with open(config_path, 'r') as file:
            self.spawner_cfg = yaml.safe_load(file)
        
        self.use_goal_blocker = self.spawner_cfg["goal_blocker"]["enable"]
        self.use_quasi_dynamic = self.spawner_cfg["quasi_dynamic"]["enable"]
        self.use_dynamic = self.spawner_cfg["dynamic"]["enable"]


        # [(num_envs, n, 3), (num_envs, m, 3), ....]
        self.combined_obstacle_dim_list = list()
        self.obstacle_index_dict = dict()
        self.num_obstacles = 0
        if self.use_goal_blocker:
            self.initialize_goal_blocker()
        
        

        # (num_envs, num_obstacles, 3)
        self.combined_obstacle_dim_tensor = torch.cat(self.combined_obstacle_dim_list, dim=1).to(self.device)
        # (num_envs, num_obstacles, 7)
        self.obstacle_poses = torch.zeros((self.num_envs, self.num_obstacles, 7), device=self.device)

        self.disable_pose = self.obstacle_poses.clone()
        self.disable_pose[:, :, 2] = -2.0 # z
        self.disable_pose[:, :, 6] = 1.0 # qw



    def initialize_goal_blocker(self):
        size_low_range = torch.tensor(self.spawner_cfg["goal_blocker"]["size"]["low"]).to(self.device)
        size_high_range = torch.tensor(self.spawner_cfg["goal_blocker"]["size"]["high"]).to(self.device)
        rand_vals = torch.rand(self.num_envs, 1, 3, device=self.device)

        # (num_envs, 1, 3)
        goal_blocker_dim = size_low_range + (size_high_range - size_low_range) * rand_vals
        self.combined_obstacle_dim_list.append(goal_blocker_dim)

        self.obstacle_index_dict["goal_blocker"] = [self.num_obstacles]
        self.num_obstacles += 1

    

    def update_obstacle_poses(self, timestep):
        current_ee_pose = torch.cat((self.env.states["eef_pos"], self.env.states["eef_quat"]), dim=1)
        goal_ee_pose = self.env.ee_goal_pose

        if self.use_goal_blocker:
            retract_timestep = self.max_episode_length - 100
            ee_error = torch.norm(current_ee_pose[:, 0:3] - goal_ee_pose[:, 0:3], dim=1)

            # enable the goal blocker if the ee is close to the goal. retract the goal blocker
            # if the timestep is 100 steps before the max episode length
            enable_idx = (ee_error < 0.2) & (timestep < retract_timestep)
            goal_blocker_idx = self.obstacle_index_dict["goal_blocker"]
            self.obstacle_poses[enable_idx, goal_blocker_idx, :] = goal_ee_pose[enable_idx, :]
            self.obstacle_poses[~enable_idx, goal_blocker_idx, :] = self.disable_pose[~enable_idx, goal_blocker_idx, :]
        

        return self.obstacle_poses
            










    

