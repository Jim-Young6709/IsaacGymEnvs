
import yaml
import torch
import numpy as np

class ObstacleSpawner:
    def __init__(self, env, num_envs, config):
        self.env = env
        self.num_envs = num_envs
        self.device = self.env.device
        self.max_episode_length = self.env.max_episode_length
        
        # --------------- loading obstacle spawner configs ---------------
        self.spawner_cfg = config
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
            self.quasi_dynamic_obstacle_enable_num = 0
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

    

    def generate_obstacle_gt(self):
        obstacle_configs = list()
        for i in range(self.num_envs):
            obstacle_config_i = (
                self.combined_obstacle_dim_tensor[i].cpu().numpy().astype(np.float32),
                self.obstacle_poses[i, :, 0:3].cpu().numpy().astype(np.float32),
                self.obstacle_poses[i, :, 3:].cpu().numpy().astype(np.float32),
                None,
            )
            obstacle_configs.append(obstacle_config_i)
        return obstacle_configs



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
    

    def reset(self):
        self.obstacle_poses = self.disable_pose.clone()
        self.quasi_dynamic_obstacle_enable_num = 0
    
    
    def update_obstacle_poses(self, timestep):
        current_ee_pose = torch.cat((self.env.states["eef_pos"], self.env.states["eef_quat"]), dim=1)
        current_ee_trans_vel = self.env.states["eef_vel"][:, 0:3]
        goal_ee_pose = self.env.ee_goal_pose
        set_quasi_dynamic_obs = False

        if self.use_goal_blocker:
            retract_timestep = self.max_episode_length - 200
            ee_error = torch.norm(current_ee_pose[:, 0:3] - goal_ee_pose[:, 0:3], dim=1)
            # enable the goal blocker if the ee is close to the goal. retract the goal blocker
            # if the timestep is 100 steps before the max episode length
            enable_idx = (ee_error < 0.4) & (timestep < retract_timestep)
            goal_blocker_idx = self.obstacle_index_dict["goal_blocker"]
            self.obstacle_poses[enable_idx, goal_blocker_idx, :] = goal_ee_pose[enable_idx, :]
            self.obstacle_poses[~enable_idx, goal_blocker_idx, :] = self.disable_pose[~enable_idx, goal_blocker_idx, :]
        
        if self.use_quasi_dynamic:
            if self.env.test_epoch > 0:
                start_time = self.spawner_cfg["quasi_dynamic"]["start_time"]
                safe_buffer_dist = self.spawner_cfg["quasi_dynamic"]["safe_buffer_dist"]
                time_interval = 200
                last_spawning_timestep = time_interval * self.spawner_cfg["quasi_dynamic"]["num"] + start_time



                if ((timestep[0] - start_time) % time_interval == 0) and (timestep[0] < last_spawning_timestep):
                    quasi_dynamic_obstacle_id = self.obstacle_index_dict["quasi_dynamic"][self.quasi_dynamic_obstacle_enable_num]
                    self.quasi_dynamic_obstacle_enable_num += 1
                    # future_time_step = min(timestep[0].item()+60, self.max_episode_length-100)
                    safe_radius = torch.norm(self.combined_obstacle_dim_tensor[:, quasi_dynamic_obstacle_id, 0:3]/2, dim=1)
                    # for i in range(self.num_envs):
                    #     future_time_step = min(timestep[0].item()+60, self.max_episode_length)
                    #     for j in range(future_time_step, self.max_episode_length-100):
                    #         future_ee_pose = self.env.ee_pose_trajectory[i, j, :]
                    #         if torch.norm(future_ee_pose[0:3] - current_ee_pose[i, 0:3]) > (safe_radius[i]+0.15):
                    #             if torch.norm(future_ee_pose[0:3] - self.env.ee_goal_pose[i, 0:3]) > (safe_radius[i]+0.15):
                    #                 self.obstacle_poses[i, quasi_dynamic_obstacle_id, :] = future_ee_pose.clone()
                    #                 break

                    # Get start and end time indices for each env
                    start_ts = torch.clamp(timestep[0] + 0, max=self.max_episode_length - 100)
                    j_range = torch.arange(start_ts, self.max_episode_length - 100, device=self.device)
                    # Create expanded versions for broadcasting
                    future_ee_pose = self.env.ee_pose_trajectory[:, j_range, :]                    # (num_envs, T, 7)
                    future_pos = future_ee_pose[:, :, 0:3]                                         # (num_envs, T, 3)
                    current_pos = current_ee_pose[:, 0:3].unsqueeze(1)                             # (num_envs, 1, 3)
                    goal_pos = self.env.ee_goal_pose[:, 0:3].unsqueeze(1)                          # (num_envs, 1, 3)
                    safe_radius_expand = safe_radius.view(-1, 1)                                   # (num_envs, 1)
                    # Calculate distance from future EE to current and goal
                    dist_to_current = torch.norm(future_pos - current_pos, dim=2)                 # (num_envs, T)
                    dist_to_goal = torch.norm(future_pos - goal_pos, dim=2)                       # (num_envs, T)
                    # Find where both distances exceed threshold
                    mask = (dist_to_current > safe_radius_expand + safe_buffer_dist) #& (dist_to_goal > safe_radius_expand + 0.15)  # (num_envs, T)
                    # Find first j index where condition is satisfied for each env
                    valid_mask_any = mask.any(dim=1)
                    first_valid_indices = mask.float().argmax(dim=1)  # if no valid, this will be 0, need to handle
                    # Only update for environments that found a valid index
                    valid_envs = torch.nonzero(valid_mask_any).squeeze(-1)  # shape (num_valid_envs,)
                    valid_js = first_valid_indices[valid_envs]              # shape (num_valid_envs,)
                    # Gather corresponding future poses
                    selected_future_pose = self.env.ee_pose_trajectory[valid_envs, j_range[valid_js], :]  # (num_valid_envs, 7)
                    # Update obstacle poses
                    self.obstacle_poses[valid_envs, quasi_dynamic_obstacle_id, :] = selected_future_pose
                    set_quasi_dynamic_obs = True

                # disable quasi-dynamic obstacles
                if timestep[0] > (self.max_episode_length - 100):
                    quasi_dynamic_idx = self.obstacle_index_dict["quasi_dynamic"]
                    self.obstacle_poses[:, quasi_dynamic_idx, :] = self.disable_pose[:, quasi_dynamic_idx, :]
        

        if self.use_floating:
            floating_obstacle_id = self.obstacle_index_dict["floating"]
            num_floating_obstacles = len(floating_obstacle_id)
            sphere_center = torch.tensor([0.15, 0.0, 0.5], device=self.device)
            #((self.env.ee_goal_pose[:, 0:3] + self.env.ee_start_pose[:, 0:3])/2).unsqueeze(1).repeat(1, num_floating_obstacles, 1) 
            if timestep[0].item() == 0:
                # # set initial floating obstacles pose
                # total_num_floating_obstacles = self.num_envs * num_floating_obstacles
                # rand_dirs = torch.randn((total_num_floating_obstacles, 3), device=self.device)
                # rand_dirs = rand_dirs / rand_dirs.norm(dim=1, keepdim=True)  # normalize
                # # explicit low and high radius range
                # radius_low = self.spawner_cfg["floating"]["init_radius"]["low"]
                # radius_high = self.spawner_cfg["floating"]["init_radius"]["high"]
                # radii = torch.rand((total_num_floating_obstacles, 1), device=self.device) * (radius_high - radius_low) + radius_low
                # positions = radii * rand_dirs
                # rand_quats = torch.randn((total_num_floating_obstacles, 4), device=self.device)
                # rand_quats = rand_quats / rand_quats.norm(dim=1, keepdim=True)
                # poses = torch.cat([positions, rand_quats], dim=1)
                # poses = poses.view(self.num_envs, num_floating_obstacles, 7)
                # # poses[:, :, 0:3] += sphere_center
                # self.obstacle_poses[:, floating_obstacle_id, :] = poses

                # set initial floating obstacles pose
                total_num_floating_obstacles = self.num_envs * num_floating_obstacles
                rand_dirs = torch.randn((total_num_floating_obstacles, 3), device=self.device)
                rand_dirs = rand_dirs / rand_dirs.norm(dim=1, keepdim=True)  # normalize

                # Ensure x > 0 for half-sphere sampling
                rand_dirs[:, 0] = rand_dirs[:, 0].abs()

                # explicit low and high radius range
                radius_low = self.spawner_cfg["floating"]["init_radius"]["low"]
                radius_high = self.spawner_cfg["floating"]["init_radius"]["high"]
                radii = torch.rand((total_num_floating_obstacles, 1), device=self.device) * (radius_high - radius_low) + radius_low
                positions = radii * rand_dirs

                rand_quats = torch.randn((total_num_floating_obstacles, 4), device=self.device)
                rand_quats = rand_quats / rand_quats.norm(dim=1, keepdim=True)

                poses = torch.cat([positions, rand_quats], dim=1)
                poses = poses.view(self.num_envs, num_floating_obstacles, 7)
                poses[:, :, 0:3] += sphere_center
                self.obstacle_poses[:, floating_obstacle_id, :] = poses

                # generate random movement direction
                noise_std = 0.2
                self.floating_obstacle_moving_dir = -1 * rand_dirs.reshape(self.num_envs, num_floating_obstacles, 3)
                noisy_dirs = self.floating_obstacle_moving_dir + torch.randn_like(self.floating_obstacle_moving_dir) * noise_std
                self.floating_obstacle_moving_dir = noisy_dirs / noisy_dirs.norm(dim=2, keepdim=True)

                # generate random velocity
                velocity_low = self.spawner_cfg["floating"]["velocity"]["low"]#0.0001
                velocity_high = self.spawner_cfg["floating"]["velocity"]["high"] #0.003
                self.floating_obstacle_velocity = torch.rand(
                    (self.num_envs, num_floating_obstacles), device=self.device
                ) * (velocity_high - velocity_low) + velocity_low

            else:
                self.obstacle_poses[:, floating_obstacle_id, 0:3] += self.floating_obstacle_velocity.unsqueeze(-1) * self.floating_obstacle_moving_dir
                # flip moving direction if obstacles are too far
                self.floating_obstacle_moving_dir = torch.where(
                    torch.norm(self.obstacle_poses[:, floating_obstacle_id, 0:3] - sphere_center, dim=2, keepdim=True) > 1.2,
                    self.floating_obstacle_moving_dir * -1,
                    self.floating_obstacle_moving_dir,
                )
                # flip moving direction if obstacles are approaching the base of the robot
                xy_positions = self.obstacle_poses[:, floating_obstacle_id, :2]  # shape: (num_envs, num_selected_obstacles, 2)
                dist_squared = torch.sum(xy_positions ** 2, dim=-1)  # shape: (num_envs, num_selected_obstacles)
                safe_zone_radius = 0.25
                is_in_safe_zone = dist_squared <= safe_zone_radius ** 2  # shape: (num_envs, num_selected_obstacles)
                is_in_safe_zone = is_in_safe_zone.unsqueeze(-1).expand(-1, -1, 3) #(num_envs, num_selected_obstacles, 3)

                self.floating_obstacle_moving_dir[is_in_safe_zone] *= -1
                # noise_std = 0.02
                # self.floating_obstacle_moving_dir[is_in_safe_zone] += torch.randn_like(self.floating_obstacle_moving_dir[is_in_safe_zone]) * noise_std
                self.floating_obstacle_moving_dir = self.floating_obstacle_moving_dir / self.floating_obstacle_moving_dir.norm(dim=2, keepdim=True)

                # Flip moving direction if x < 0.1
                x_below_threshold = self.obstacle_poses[:, floating_obstacle_id, 0] < 0.1  # shape: (num_envs, num_selected_obstacles)
                x_below_threshold = x_below_threshold.unsqueeze(-1).expand(-1, -1, 3)  # shape: (num_envs, num_selected_obstacles, 3)
                self.floating_obstacle_moving_dir[x_below_threshold] *= -1


            # disable quasi-dynamic obstacles
            if timestep[0] > (self.max_episode_length - 300):
                floating_obstacle_id = self.obstacle_index_dict["floating"]
                self.obstacle_poses[:, floating_obstacle_id, :] = self.disable_pose[:, floating_obstacle_id, :]







                            

        return self.obstacle_poses, set_quasi_dynamic_obs
            










    

