
import yaml
import torch
import numpy as np
from pathlib import Path
from hydra.utils import instantiate

from isaacgymenvs.motion_planners import MotionPlannerBase





class DRPNeuralMP(MotionPlannerBase):
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
        config_file = Path(checkpoint_folder) / "model_config.yaml"

        # checkpoint_name = "checkpoint_2ep_step50000_success_0.6914.pth.pth"
        # checkpoint_name = 'checkpoint_step60000_success_0.3132.pth.pth' # my second favourite so far (really good obstacle avoiding capability, but less good for goal reaching)
        checkpoint_name = 'checkpoint_step60000_success_0.5142.pth.pth' # my favourite so far. slightly better at goal reaching and slightly worse at obstacle avoiding

        checkpoint_file = Path(checkpoint_folder) / checkpoint_name
        with open(config_file, "r") as f:
            model_config = yaml.safe_load(f)
        self.model = instantiate(model_config)
        self.model.load_checkpoint(checkpoint_file)
        self.model = self.model.to("cuda").float()

        self.model = torch.compile(self.model)
        self.model.policy.reset()
        self.model.policy.set_eval()


    def get_actions(self, env_obs_dict):
        joint_pos = env_obs_dict["joint_pos"]
        goal_joint_pos = env_obs_dict["goal_joint_pos"]
        current_robot_pcd = env_obs_dict["robot_pcd"]
        goal_robot_pcd = env_obs_dict["goal_robot_pcd"]
        static_obstacle_pcd = env_obs_dict["static_obstacle_pcd"]
        
        nn_pcd_obs = self._prepare_neuralmp_observation(obstacle_pcd=static_obstacle_pcd, goal_robot_pcd=goal_robot_pcd, current_robot_pcd=None)
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
                    delta_joint_pos_action = self.model.get_action_robomimic(obs_dict)
            open_loop_joint_pos += delta_joint_pos_action
        joint_pos_target = open_loop_joint_pos
        return joint_pos_target
    

    def _prepare_neuralmp_observation(self, obstacle_pcd, goal_robot_pcd, current_robot_pcd=None): 
        # (num_envs, num_points, 4)
        nn_pcd_obs = torch.cat((
                torch.zeros(self.num_robot_points, 4), # mask robot pcd with 0
                torch.ones(self.num_obstacle_points, 4), # mask obstacle pcd with 1
                2 * torch.ones(self.num_robot_points, 4), # mask goal obstacle pcd with 2
        ), dim=0).unsqueeze(0).repeat(self.num_envs, 1, 1).cuda()


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



