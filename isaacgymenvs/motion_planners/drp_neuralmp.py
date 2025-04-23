
import yaml
import torch
import numpy as np
from pathlib import Path
from hydra.utils import instantiate
from tqdm import tqdm
from isaacgymenvs.motion_planners import MotionPlannerBase


class DRPNeuralMP(MotionPlannerBase):
    """
    Class for:
    1.Neural MP
    2.Neural MP + TTO
    3.Neural MP model with DRP framework
    """

    def __init__(self, env):
        super().__init__(env)
        self._num_robot_points = 2048
        self._num_goal_robot_points = 2048
        self._num_obstacle_points = 4096
        self.set_up_policy()


    @property
    def num_robot_points(self):
        return self._num_robot_points

    @property
    def num_goal_robot_points(self):
        return self._num_goal_robot_points

    @property
    def num_obstacle_points(self):
        return self._num_obstacle_points


    def set_up_policy(self):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("medium")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch._dynamo.config.suppress_errors = True

        checkpoint_folder = "/home/avenger/Projects/drp/nmp_refactor/checkpoints/legacy"
        # checkpoint_folder = "/home/jimyoung/Neural_MP_Proj/IsaacGymEnvs/ckpts"
        config_file = Path(checkpoint_folder) / "model_config.yaml"

        # checkpoint_name = "checkpoint_2ep_step50000_success_0.6914.pth.pth"
        # checkpoint_name = 'checkpoint_step60000_success_0.3132.pth.pth' # my second favourite so far (really good obstacle avoiding capability, but less good for goal reaching)
        # checkpoint_name = 'checkpoint_step60000_success_0.5142.pth.pth' # my favourite so far. slightly better at goal reaching and slightly worse at obstacle avoiding
        # checkpoint_name = 'checkpoint_gblk0.3_step170000_success_0.5400.pth.pth'
        checkpoint_name = "aitstar_18500ep.pth" # neural mp checkpoint

        checkpoint_file = Path(checkpoint_folder) / checkpoint_name
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
        self.model = instantiate(model_config)
        self.model.load_checkpoint(checkpoint_file)
        self.model = self.model.to("cuda").float()

        self.model = torch.compile(self.model)
        self.model.policy.reset()
        self.model.policy.set_eval()


    def get_actions(self, env_obs_dict, mean_actions=True):
        joint_pos = env_obs_dict["joint_pos"]
        goal_joint_pos = env_obs_dict["goal_joint_pos"]
        current_robot_pcd = env_obs_dict["robot_pcd"]
        goal_robot_pcd = env_obs_dict["goal_robot_pcd"]

        # [(n, 3), (m, 3), ... ] -> length is num_envs
        obstacle_pcd_list = env_obs_dict["combined_obstacle_pcd"]
        subsampled_pcd_list = []
        for pcd in obstacle_pcd_list:
            num_points = pcd.shape[0]
            if num_points >= self.num_obstacle_points:
                indices = torch.randperm(num_points)[0:self.num_obstacle_points]
                sampled = pcd[indices]
            else:
                indices = torch.randint(0, num_points, (self.num_obstacle_points,), device=self.device)
                sampled = pcd[indices]
            subsampled_pcd_list.append(sampled)

        # Stack into final tensor of shape (num_envs, num_obstacle_points, 3)
        subsampled_pcd = torch.stack(subsampled_pcd_list, dim=0)

        nn_pcd_obs = self._prepare_neuralmp_observation(obstacle_pcd=subsampled_pcd, goal_robot_pcd=goal_robot_pcd, current_robot_pcd=None)
        # roll out open loop
        open_loop_steps = 1
        open_loop_joint_pos = joint_pos.clone()
        for i in range(open_loop_steps):
            current_robot_pcd = self.env.get_robot_pcds(open_loop_joint_pos)
            nn_pcd_obs = self._update_neuralmp_robot_pcd_observation(nn_pcd_obs, current_robot_pcd)
            obs_dict = {
                "compute_pcd_params": nn_pcd_obs,       # (num_envs, NUM_TOTAL_POINTS, 4)
                "current_angles": open_loop_joint_pos,  # (num_envs, 7)
                "goal_angles": goal_joint_pos,          # (num_envs, 7)
            }
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    delta_joint_pos_action = self.model.get_action_robomimic(obs_dict, mean_actions)
            open_loop_joint_pos += delta_joint_pos_action * 1.0 #* 1.0
        joint_pos_target = open_loop_joint_pos
        return joint_pos_target

    def get_actions_open_loop(self, env_obs_dict, tto_batch_size=100):
        """
        in original NeuralMP paper, tto_batch_size = 100
        """
        self.model.policy.reset()
        self.model.policy.set_train()

        obstacle_pcd_list = env_obs_dict["combined_obstacle_pcd"]
        subsampled_pcd_list = []
        for pcd in obstacle_pcd_list:
            num_points = pcd.shape[0]
            if num_points >= self.num_obstacle_points:
                indices = torch.randperm(num_points)[0:self.num_obstacle_points]
                sampled = pcd[indices]
            else:
                indices = torch.randint(0, num_points, (self.num_obstacle_points,), device=self.device)
                sampled = pcd[indices]
            subsampled_pcd_list.append(sampled)

        # Stack into final tensor of shape (num_envs, num_obstacle_points, 3)
        subsampled_pcd = torch.stack(subsampled_pcd_list, dim=0)

        # ideally we can get things fully vectorized, but batch size will be: B_tto * B_envs, hard to fit into gpu memory
        planning_actions_abs = []
        for i in tqdm(range(self.num_envs), desc="Open Loop Rollout"):
            # add tto batch dim
            env_obs_dict_idx = {
                "joint_pos": env_obs_dict['joint_pos'][i].repeat(tto_batch_size, 1),
                "goal_joint_pos": env_obs_dict['goal_joint_pos'][i].repeat(tto_batch_size, 1),
                "robot_pcd": env_obs_dict['robot_pcd'][i].repeat(tto_batch_size, 1, 1),
                "goal_robot_pcd": env_obs_dict['goal_robot_pcd'][i].repeat(tto_batch_size, 1, 1),
                "combined_obstacle_pcd": [env_obs_dict['combined_obstacle_pcd'][i]] * tto_batch_size,
            }

            # rollout
            trajectory = []
            for _ in tqdm(range(self.env.max_episode_length), desc="Open Loop Rollout"):
                current_joint_angles = self.get_actions(env_obs_dict_idx, mean_actions=False)
                trajectory.append(current_joint_angles)
                env_obs_dict_idx["joint_pos"] = current_joint_angles
                env_obs_dict_idx["robot_pcd"] = self.env.get_robot_pcds(current_joint_angles)

            # TTO
            output_traj = torch.stack(trajectory).permute(1, 0, 2)  # [tto_batch_size, max_rollout_len, 7]
            goal_reaching = torch.norm(output_traj[:, -1] - env_obs_dict_idx["goal_joint_pos"], dim=1) < 0.1
            if goal_reaching.sum() == 0:
                print("No valid trajectory found.")
                continue

            output_traj = output_traj[goal_reaching]
            num_valid_traj = output_traj.shape[0]
            scene_pcd_i = subsampled_pcd[i].repeat(self.env.max_episode_length, 1, 1)

            traj_c_nums = []
            for j in range(num_valid_traj):
                output_traj_idx = output_traj[j]
                waypoint_c_num = self.env.collision_checker.check_scene_collision_batch(
                    output_traj_idx, scene_pcd_i, thred=0.01, sphere_repr_only=True
                )
                traj_c_num = waypoint_c_num.sum()
                traj_c_nums.append(traj_c_num.item())

            # print(traj_c_nums)
            best_traj_idx = torch.argmin(torch.tensor(traj_c_nums, device=self.device))
            output_traj = output_traj[best_traj_idx].unsqueeze(1)
            planning_actions_abs.append(output_traj)

        planning_actions_abs = torch.cat(planning_actions_abs, dim=1).to(self.device)
        return planning_actions_abs

    def _prepare_neuralmp_observation(self, obstacle_pcd, goal_robot_pcd, current_robot_pcd=None): 
        # (num_envs, num_points, 4)
        nn_pcd_obs = torch.cat((
                torch.zeros(self.num_robot_points, 4), # mask robot pcd with 0
                torch.ones(self.num_obstacle_points, 4), # mask obstacle pcd with 1
                2 * torch.ones(self.num_robot_points, 4), # mask goal obstacle pcd with 2
        ), dim=0).unsqueeze(0).repeat(obstacle_pcd.shape[0], 1, 1).cuda()


        # add robot points
        if current_robot_pcd is not None:
            nn_pcd_obs[:, 0:self.num_robot_points, 0:3] = current_robot_pcd
        
        # add obstacle points
        nn_pcd_obs[:, self.num_robot_points:self.num_robot_points+self.num_obstacle_points, 0:3] = obstacle_pcd

        # add goal robot points
        nn_pcd_obs[:, self.num_robot_points+self.num_obstacle_points:, 0:3] = goal_robot_pcd
        return nn_pcd_obs


    def _update_neuralmp_robot_pcd_observation(self, nn_pcd_obs, current_robot_pcd):
        nn_pcd_obs[:, 0:self.num_robot_points, 0:3] = current_robot_pcd
        return nn_pcd_obs
    

    def reset(self):
        self.model.policy.reset()



