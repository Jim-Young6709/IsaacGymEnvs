
import yaml
import torch
import numpy as np


class ObstacleSpawner:
    def __init__(self, env, num_envs, config_path):
        self.env = env
        self.num_envs = num_envs
        self.device = self.env.device
        self.max_episode_length = self.env.max_episode_length
        
        # --------------- loading obstacle spawner configs ---------------
        with open(config_path, 'r') as file:
            self.spawner_cfg = yaml.safe_load(file)
        self.use_goal_blocker = self.spawner_cfg["goal_blocker"]["enable"]
        self.use_quasi_dynamic = self.spawner_cfg["quasi_dynamic"]["enable"]
        self.use_floating = self.spawner_cfg["floating"]["enable"]

        # --------------- initializing obstacle sizes ---------------
        # [(num_envs, n, 3), (num_envs, m, 3), ....]
        self.combined_obstacle_dim_list = list()
        self.obstacle_index_dict = dict()
        self.obstacle_counter = 0
        if self.use_goal_blocker:
            self.initialize_obstacles("goal_blocker")
        if self.use_quasi_dynamic:
            self.initialize_obstacles("quasi_dynamic")
        if self.use_floating:
            self.initialize_obstacles("floating")

        # ------------------- initializing buffers -------------------
        # record the total number of obstacles
        self.num_obstacles = self.obstacle_counter
        # obstacle dimensions (num_envs, num_obstacles, 3)
        self.combined_obstacle_dim_tensor = torch.cat(self.combined_obstacle_dim_list, dim=1).to(self.device)
        # object poses for effectively disabling a dynamic obstacle (num_envs, num_obstacles, 7)
        self.disable_pose = torch.zeros((self.num_envs, self.num_obstacles, 7), device=self.device)
        self.disable_pose[:, :, 2] = -2.0 # z
        self.disable_pose[:, :, 6] = 1.0 # qw
        # initialize all obstacle poses to effectively disable them (num_envs, num_obstacles, 7)
        self.obstacle_poses = self.disable_pose.clone()
        # flag indicating which dynamic obstacles are constantly moving (num_obstacles,)
        self.moving_obstacle_flag = torch.zeros(self.num_obstacles, dtype=bool, device=self.device)
        if self.use_goal_blocker:
            self.moving_obstacle_flag[self.obstacle_index_dict["goal_blocker"]] = False
        if self.use_quasi_dynamic:
            self.moving_obstacle_flag[self.obstacle_index_dict["quasi_dynamic"]] = False
        if self.use_floating:
            self.moving_obstacle_flag[self.obstacle_index_dict["floating"]] = True
        self.num_moving_obstacles = self.moving_obstacle_flag.sum().item()


    @property
    def obstacle_dims(self):
        return self.combined_obstacle_dim_tensor

    def initialize_obstacles(self, obstacle_type):
        # ----------- creating obstacle dims -----------
        num_obstacles = self.spawner_cfg[obstacle_type]["num"]
        size_low_range = torch.tensor(self.spawner_cfg[obstacle_type]["size"]["low"]).to(self.device)
        size_high_range = torch.tensor(self.spawner_cfg[obstacle_type]["size"]["high"]).to(self.device)
        rand_vals = torch.rand(self.num_envs, num_obstacles, 3, device=self.device)
        # (num_envs, num_obstacles, 3)
        obstacles_dims = size_low_range + (size_high_range - size_low_range) * rand_vals
        self.combined_obstacle_dim_list.append(obstacles_dims)
        # ---------- setting obstacle indices ----------
        self.obstacle_index_dict[obstacle_type] = list(
            range(self.obstacle_counter, self.obstacle_counter + num_obstacles)
        )
        # ---------- updating obstacle counter ----------
        self.obstacle_counter += num_obstacles
    
    
    def update_obstacle_poses(self, timestep):
        current_ee_pose = torch.cat((self.env.states["eef_pos"], self.env.states["eef_quat"]), dim=1)
        goal_ee_pose = self.env.ee_goal_pose

        if self.use_goal_blocker:
            retract_timestep = self.max_episode_length - 200
            ee_error = torch.norm(current_ee_pose[:, 0:3] - goal_ee_pose[:, 0:3], dim=1)

            # enable the goal blocker if the ee is close to the goal. retract the goal blocker
            # if the timestep is 100 steps before the max episode length
            enable_idx = (ee_error < 0.4) & (timestep < retract_timestep)
            goal_blocker_idx = self.obstacle_index_dict["goal_blocker"]
            self.obstacle_poses[enable_idx, goal_blocker_idx, :] = goal_ee_pose[enable_idx, :]
            self.obstacle_poses[~enable_idx, goal_blocker_idx, :] = self.disable_pose[~enable_idx, goal_blocker_idx, :]
        

        return self.obstacle_poses
            










    

