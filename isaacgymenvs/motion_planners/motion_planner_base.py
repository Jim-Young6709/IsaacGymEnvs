
import torch
import numpy as np
from isaacgymenvs.tasks import DRPEvals
from abc import ABC, abstractmethod


class MotionPlannerBase(ABC):
    def __init__(self, env: DRPEvals):
        self.env = env
        self.num_envs = self.env.num_envs
        self.device = self.env.device

    @property
    @abstractmethod
    def num_robot_points(self):
        pass

    @property
    @abstractmethod
    def num_goal_robot_points(self):
        pass

    @property
    @abstractmethod
    def num_obstacle_points(self):
        pass

    @abstractmethod
    def get_actions(self, env_obs_dict):
        pass

    @abstractmethod
    def reset(self):
        pass


