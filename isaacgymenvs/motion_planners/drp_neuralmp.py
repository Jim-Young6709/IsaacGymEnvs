import torch
import numpy as np

from isaacgymenvs.motion_planners import MotionPlannerBase


class DRPNeuralMP(MotionPlannerBase):
    def __init__(self):
        self._num_robot_points = 2048
        self._num_goal_robot_points = 2048
        self._num_obstacle_points = 4096


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
        pass
    


