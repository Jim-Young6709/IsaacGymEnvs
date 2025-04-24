
import yaml
import torch
import numpy as np
from isaacgymenvs.motion_planners import MotionPlannerBase
from isaacgymenvs.reactive_artificial_potential import ReactiveArtificialPotential


class RMPOnly(MotionPlannerBase):
    def __init__(self, env):
        super().__init__(env)
        self._num_robot_points = 2048
        self._num_goal_robot_points = 2048
        self._num_obstacle_points = 4096
        self.rmp = ReactiveArtificialPotential(self.env)


    @property
    def num_robot_points(self):
        return self._num_robot_points

    @property
    def num_goal_robot_points(self):
        return self._num_goal_robot_points

    @property
    def num_obstacle_points(self):
        return self._num_obstacle_points


    def get_actions(self, env_obs_dict):
        goal_joint_pos = env_obs_dict["goal_joint_pos"]
        env_obs_dict = self.rmp.apply_rmp_vectorized(env_obs_dict, use_full_pcd=True, use_integrator=True)
        return goal_joint_pos
    

    def get_actions_open_loop(self, env_obs_dict):
        return None
   

    def reset(self):
        pass



