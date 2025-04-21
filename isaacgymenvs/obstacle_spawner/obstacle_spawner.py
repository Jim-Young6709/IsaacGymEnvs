
import yaml
import torch
import numpy as np

from isaacgymenvs.tasks.utils.drp_evals_utils import transform_pcds_to_world
from isaacgymenvs.reactive_artificial_potential.utils.franka_collision_checker import FrankaCollisionChecker


class ObstacleSpawner:
    def __init__(self, env, num_envs, config):
        self.env = env
        self.num_envs = num_envs
        self.device = self.env.device
        self.max_episode_length = self.env.max_episode_length
        self.collision_checker = FrankaCollisionChecker()
        
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
        current_joint_pos = self.env.states["q"][:, 0:7].clone()
        goal_ee_pose = self.env.ee_goal_pose
        set_quasi_dynamic_obs = False

        if self.use_goal_blocker:
            goal_blocker_idx = self.obstacle_index_dict["goal_blocker"]
            if timestep[0].item() == 0:
                self.set_goal_blocking_flag = torch.zeros((self.num_envs), device=self.device, dtype=bool)
                self.obstacle_poses[:, goal_blocker_idx, :] = self.disable_pose[:, goal_blocker_idx, :]

            retract_timestep = self.max_episode_length - 200
            ee_error = torch.norm(current_ee_pose[:, 0:3] - goal_ee_pose[:, 0:3], dim=1)
            # enable the goal blocker if the ee is close to the goal. retract the goal blocker
            # if the timestep is 100 steps before the max episode length
            # enable_idx = (ee_error < 0.4) & (timestep < retract_timestep)
            # goal_blocker_idx = self.obstacle_index_dict["goal_blocker"]

            # env_ids = torch.arange(self.num_envs, device=self.device)[enable_idx]
            # if len(env_ids) > 0:
            #     try:
            #         is_safe = self.check_collisions(current_joint_pos[enable_idx, :], goal_ee_pose[enable_idx, :], goal_blocker_idx, 0.3, env_ids)
            #         print(is_safe)
            #     except:
            #         import ipdb; ipdb.set_trace()
            # self.obstacle_poses[enable_idx, goal_blocker_idx, :] = goal_ee_pose[enable_idx, :]
            # self.obstacle_poses[~enable_idx, goal_blocker_idx, :] = self.disable_pose[~enable_idx, goal_blocker_idx, :]

            
            set_candidate_flag = (ee_error < 0.4) & (timestep < retract_timestep) & (~self.set_goal_blocking_flag)
            set_candidate_ids = torch.arange(self.num_envs, device=self.device)[set_candidate_flag]


            if len(set_candidate_ids) > 0:
                # (len(set_candidate_ids),)
                is_safe = self.check_collisions(current_joint_pos[set_candidate_ids, :], goal_ee_pose[set_candidate_ids, :], goal_blocker_idx, 0.01, set_candidate_ids)
                safe_set_ids = set_candidate_ids[is_safe]
                self.obstacle_poses[safe_set_ids, goal_blocker_idx, :] = goal_ee_pose[safe_set_ids, :]
                self.set_goal_blocking_flag[safe_set_ids] = True
            

            if timestep[0] > (self.max_episode_length - 100):
                self.obstacle_poses[:, goal_blocker_idx, :] = self.disable_pose[:, goal_blocker_idx, :]



        
        if self.use_quasi_dynamic:
            # self.moving_obstacle_flag[self.obstacle_index_dict["quasi_dynamic"]] = False
            if self.env.test_epoch > 0:
                start_time = self.spawner_cfg["quasi_dynamic"]["start_time"]
                safe_buffer_dist = self.spawner_cfg["quasi_dynamic"]["safe_buffer_dist"]
                time_interval = self.spawner_cfg["quasi_dynamic"]["time_interval"]
                last_spawning_timestep = time_interval * self.spawner_cfg["quasi_dynamic"]["num"] + start_time
                
                if ((timestep[0] - start_time) % time_interval == 0) and (timestep[0] < last_spawning_timestep):
                    # Get the obstacle ID for the current quasi-dynamic obstacle
                    quasi_dynamic_obstacle_id = self.obstacle_index_dict["quasi_dynamic"][self.quasi_dynamic_obstacle_enable_num]
                    self.quasi_dynamic_obstacle_enable_num += 1
                    # Initialize mask to track which envs have successfully placed the obstacle
                    obstacle_set_idx = torch.zeros(self.num_envs, dtype=bool, device=self.device)
                    # Determine the range of future timesteps to check
                    start_idx = min(timestep[0].item(), self.max_episode_length - 200)
                    for i in range(start_idx, self.max_episode_length , 5):
                        # Get future end-effector pose at current timestep
                        future_ee_pose = self.env.ee_pose_trajectory[:, i, :]  # (num_envs, 7)
                        # Check collisions at that pose
                        is_safe = self.check_collisions(
                            joint_pos=current_joint_pos,
                            obstacle_poses=future_ee_pose,
                            obstacle_id=quasi_dynamic_obstacle_id,
                            threshold=safe_buffer_dist,
                        )
                        # Determine which envs are still unset and collision-free
                        is_free_to_set = (~obstacle_set_idx) & (is_safe)
                        if is_free_to_set.any():
                            # Get a slightly more future pose to assign the obstacle to
                            future_pose_target = self.env.ee_pose_trajectory[:, i, :]  # safe: i+10 < max_episode_length - 190
                            self.obstacle_poses[is_free_to_set, quasi_dynamic_obstacle_id, :] = future_pose_target[is_free_to_set, :]
                            obstacle_set_idx[is_free_to_set] = True
                        if obstacle_set_idx.all():
                            break
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
                velocity_low = self.spawner_cfg["floating"]["velocity"]["low"]
                velocity_high = self.spawner_cfg["floating"]["velocity"]["high"]
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



    def check_collisions(self, joint_pos, obstacle_poses, obstacle_id, threshold, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        # (num_envs, 1, P, 3)
        obstacle_pcd = self.env.dynamic_obstacle_pcd[env_ids, [obstacle_id], :, :]
        # obstacle_poses (num_envs, 1, 7)
        obstacle_poses = obstacle_poses[:, None, :]
        # (num_envs, 1, num_points_per_obstacle, 3)
        dynamic_obstacle_pcd_world = transform_pcds_to_world(obstacle_pcd, obstacle_poses)
        # all potentially moving obstacle pcd (num_envs, num_dynamic_pcd, 3)
        dynamic_obstacle_pcd_world = dynamic_obstacle_pcd_world.view(len(env_ids), -1, 3)
        # (num_envs, num_points)
        sdf = self.collision_checker.check_scene_sdf_batch(
            joint_pos, dynamic_obstacle_pcd_world.float(), debug=False, sphere_repr_only=True
        )
        min_sdf_per_env = torch.min(sdf, dim=1).values
        is_safe = min_sdf_per_env > threshold
        return is_safe 
    



