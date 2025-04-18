
import torch
import numpy as np


class ObstacleSpawner:
    def __init__(self, env):
        self.env = env
        self.num_envs = self.env.num_envs

    

