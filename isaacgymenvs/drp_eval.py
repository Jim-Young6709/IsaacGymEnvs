import os
import shutil
import signal
import time
import cv2
import imageio
import wandb
from collections import OrderedDict, Counter, deque

import isaacgym # must import isaacgym before pytorch
import numpy as np
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import isaacgymenvs.utils.robomimic_utils as RMUtils
from isaacgymenvs.utils.media_utils import camera_shot
from isaacgymenvs.utils.utils import set_seed
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.tasks import Ant

import hydra
from omegaconf import DictConfig



class Eval:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sim_device = 'cuda:0'
        self.seed = 42
        set_seed(self.seed)

        self.set_up_env()

     
    def set_up_env(self):
        # cfg_task = cfg_dict["task"]

        # print(self.cfg)
        # assert 1==2

        cfg_dict = omegaconf_to_dict(self.cfg)
        cfg_task = cfg_dict["task"]


        print(cfg_dict)
        rl_device = self.sim_device
        sim_device = self.sim_device
        graphics_device_id = 0
        virtual_screen_capture = False
        headless = False

        force_render = True
        if headless:
            force_render = False
        
        self.env = Ant(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
       

    def reset_envs(self):
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.env.reset_idx(env_ids)
        # self.env.base_model.policy.reset()


    @torch.no_grad()
    def test(self):

        while True:
            self.reset_envs()
            # state_obs = self.env.compute_observations()
            for test_step in range(self.env.max_episode_length - 1):
                actions = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.sim_device)
                obs_dict, rews, dones, infos = self.env.step(actions)
                pass


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def main(cfg: DictConfig):
    torch._dynamo.config.disable = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    agent = Eval(cfg)
    agent.test()


if __name__ == "__main__":
    main()
