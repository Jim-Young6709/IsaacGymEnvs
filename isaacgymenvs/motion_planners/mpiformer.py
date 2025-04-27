import yaml
import torch
import numpy as np
from pathlib import Path
from hydra.utils import instantiate
from tqdm import tqdm
from avoid_everything.pretraining import PretrainingMotionPolicyTransformer
from mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from isaacgymenvs.motion_planners import MotionPlannerBase


class MPiFormer(MotionPlannerBase):
    def __init__(self, env):
        super().__init__(env)
        self._num_robot_points = 2048
        self._num_goal_robot_points = 128
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
        checkpoint_folder = self.env.ckpt_cfg.dir + "/mpiformer"
        checkpoint_name = self.env.ckpt_cfg.filename
        checkpoint_file = Path(checkpoint_folder) / checkpoint_name
        cfg_path = Path(checkpoint_folder) / "mpiformer_eval.yaml"

        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        mdl_class = PretrainingMotionPolicyTransformer
        self.model = mdl_class.load_from_checkpoint(
            checkpoint_file,
            disable_viz=True,
            **cfg["training_model_parameters"],
            **cfg["shared_parameters"],
        ).to(self.device)
        self.model.setup()
        self.model.eval()

    @torch.no_grad()
    def get_actions(self, env_obs_dict):
        start_config = env_obs_dict['joint_pos']
        goal_config = env_obs_dict['goal_joint_pos']

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

        nn_pcd_obs = self._prepare_mpiformer_observation(
            start_config, goal_config, subsampled_pcd
        )

        q_norm = normalize_franka_joints(start_config)

        pcd_labels = nn_pcd_obs[:, :, 3]
        pcd = nn_pcd_obs[:, :, 0:3]

        q_norm = torch.clamp(q_norm + self.model.mpiformer(pcd_labels, pcd, q_norm, self.model.pc_bounds).squeeze(1), min=-1, max=1)
        joint_pos_target = unnormalize_franka_joints(q_norm)

        return joint_pos_target

    @torch.no_grad()
    def get_actions_open_loop(self, env_obs_dict):
        """
        motion planning with MPiNets.

        Args:
            start_config (np.ndarray): Joint angles of the robot at the start of the task.
            goal_config (np.ndarray): Joint angles of the robot at the goal of the task.
            points (np.ndarray): xyz information of the point cloud.

        Returns:
            Tuple[list, bool, float]: output trajectory, planning success flag, and average rollout time.
        """

        start_config = env_obs_dict['joint_pos']
        goal_config = env_obs_dict['goal_joint_pos']

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

        nn_pcd_obs = self._prepare_mpiformer_observation(
            start_config, goal_config, subsampled_pcd
        )

        q = start_config

        planning_actions_abs = []
        q_norm = normalize_franka_joints(q)
        assert isinstance(q_norm, torch.Tensor)

        for _ in tqdm(range(self.env.max_episode_length), desc="Open Loop Rollout"):
            q_norm = torch.clamp(q_norm + self.model(nn_pcd_obs, q_norm), min=-1, max=1)
            qt = unnormalize_franka_joints(q_norm)
            assert isinstance(qt, torch.Tensor)
            planning_actions_abs.append(qt.unsqueeze(0))

            current_robot_pcd = self.env.get_robot_pcds(qt)
            nn_pcd_obs = self._update_mpiformer_robot_pcd_observation(nn_pcd_obs, current_robot_pcd)

        planning_actions_abs = torch.cat(planning_actions_abs, dim=0).to(self.device)

        # (max_episode_length, num_envs, 7)

        return planning_actions_abs

    def _prepare_mpiformer_observation(
        self,
        start_config: torch.Tensor,
        goal_config: torch.Tensor,
        obstacle_pcd: torch.Tensor
    ):
        # goal_pose = FrankaRobot.fk(goal_config, eff_frame="right_gripper")
        eef_tranforms = self.env.gpu_fk_sampler.end_effector_pose(goal_config, "right_gripper")

        gripper_width = 0.04
        current_robot_pcd = self.env.get_robot_pcds(start_config)
        goal_eef_pcd = self.env.gpu_fk_sampler.sample_end_effector(
            torch.as_tensor(eef_tranforms).type_as(current_robot_pcd),
            num_points=self.num_goal_robot_points,
            gripper_width=gripper_width,
        )

        nn_pcd_obs = torch.cat(
            (
                torch.zeros(self.num_robot_points, 4),
                torch.ones(self.num_obstacle_points, 4),
                2 * torch.ones(self.num_goal_robot_points, 4),
            ),
            dim=0,
        ).unsqueeze(0).repeat(obstacle_pcd.shape[0], 1, 1).cuda()

        nn_pcd_obs[:, 0:self.num_robot_points, 0:3] = current_robot_pcd
        nn_pcd_obs[:, self.num_robot_points:self.num_robot_points+self.num_obstacle_points, 0:3] = obstacle_pcd
        nn_pcd_obs[:, self.num_robot_points+self.num_obstacle_points:, 0:3] = goal_eef_pcd

        return nn_pcd_obs

    def _update_mpiformer_robot_pcd_observation(self, nn_pcd_obs, current_robot_pcd):
        nn_pcd_obs[:, 0:self.num_robot_points, 0:3] = current_robot_pcd
        return nn_pcd_obs

    def reset(self):
        pass
