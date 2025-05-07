
import yaml
import torch
import numpy as np
from pathlib import Path
from hydra.utils import instantiate

from isaacgymenvs.motion_planners import MotionPlannerBase


class DRPACT(MotionPlannerBase):
    def __init__(self, env):
        super().__init__(env)
        self._num_robot_points = 256
        self._num_goal_robot_points = 256
        self._num_obstacle_points = 2048
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

        # checkpoint_folder = "/home/avenger/Projects/drp/nmp_refactor/checkpoints/dagger_multi"
        # checkpoint_folder = "/home/avenger/Projects/drp/nmp_refactor/checkpoints/dagger_multi_blk"
        # checkpoint_folder = "/home/avenger/Projects/drp/nmp_refactor/checkpoints/dagger_multi_ac2_iter3_fabricsphere0_blk0.2"
        # checkpoint_folder = "/home/avenger/Projects/drp/nmp_refactor/checkpoints/dagger_multi_blk_ac2"

        # checkpoint_folder = "/home/avenger/Projects/drp/nmp_refactor/checkpoints/dagger_ttf"
        # base without dagger
        checkpoint_folder = "/home/avenger/Projects/drp/nmp_refactor/checkpoints/nmp_transformer_10M_mesh_blk"

        checkpoint_path = "best.pt"

        model_config_file = Path(checkpoint_folder) / "model_config.yaml"
        if not model_config_file.exists():
            print(f"Model config file {model_config_file} does not exist")
            exit()

        with open(model_config_file, "r") as f:
            model_config = yaml.safe_load(f)
        model = instantiate(model_config)

        checkpoint_path = Path(checkpoint_folder) / checkpoint_path
        if not checkpoint_path.exists():
            print(f"Checkpoint path {checkpoint_path} does not exist")
            exit()

        epoch, eval_success_rate, val_loss = model.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path} at epoch {epoch}")
        print(f"\tEval success rate: {eval_success_rate:}")
        print(f"\tVal loss: {val_loss:}")
        self.model = torch.compile(model).to("cuda").float()


    def get_actions(self, env_obs_dict):
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

        obs_dict = self.prepare_act_observation(
            joint_pos=joint_pos, 
            goal_joint_pos=goal_joint_pos, 
            obstacle_pcd=subsampled_pcd, 
            goal_robot_pcd=goal_robot_pcd, 
            current_robot_pcd=current_robot_pcd, 
        )
        # max is 10
        n_action = 0 #5
        # (num_envs, S, 7)
        action_chunk = self.model.get_action(obs_dict)
        # absolute joint position target (num_envs, 7)
        absolute_joint_pos_action = action_chunk[:, min(n_action, action_chunk.shape[1])]
        joint_pos_target = absolute_joint_pos_action
        # delta_action = joint_pos_target - joint_pos
        # joint_pos_target = joint_pos + delta_action*4
        return joint_pos_target
        

    def prepare_act_observation(self, joint_pos, goal_joint_pos, obstacle_pcd, goal_robot_pcd, current_robot_pcd):
        obs_dict = dict()
        obs_dict["current_angles"] = joint_pos.unsqueeze(1).float()
        obs_dict["goal_angles"] = goal_joint_pos.unsqueeze(1).float()
        obs_dict["scene_pcd"] = obstacle_pcd.unsqueeze(1).float()
        obs_dict["robot_pcd"] = current_robot_pcd.unsqueeze(1).float()
        obs_dict["target_pcd"] = goal_robot_pcd.unsqueeze(1).float()
        return obs_dict


    def reset(self):
        pass



