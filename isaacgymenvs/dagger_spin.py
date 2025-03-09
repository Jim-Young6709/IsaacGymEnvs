from copy import deepcopy
import datetime
from nntplib import NNTPPermanentError
from attr import has
from imageio import RETURN_BYTES
import isaacgym
import torch
import xml.etree.ElementTree as ET
import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import to_absolute_path
import gym
import ml_runlog
import sys
from rl_games.algos_torch import torch_ext
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner, _override_sigma, _restore
from rl_games.algos_torch import model_builder
from isaacgymenvs.learning import amp_continuous
from isaacgymenvs.learning import amp_players
from isaacgymenvs.learning import amp_models
from isaacgymenvs.learning import amp_network_builder
import numpy as np
from gym import spaces
import isaacgymenvs
import matplotlib.pyplot as plt
from collections import deque
import math
import shutil
from isaacgym import gymtorch, gymapi
import random
import time
from tqdm import tqdm
from copy import deepcopy
from depth_backbone import ConvBackbone58x87, ConvBackbone
import torchvision.transforms.functional as fn
import gc
import time
import torch.distributed as dist
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)

class Dagger(object):
    def __init__(self, config):
        self.cfg = config
        self.bc = False #True #False
        self.use_camera_gt = False #True 
        self.use_base_gt = False #True
        if(self.cfg.multi_gpu):
            # TODO 
            self.truncate_grads = self.cfg.get('truncate_grads', True) #False # TODO 
            self.mixed_precision = self.cfg.get('mixed_precision', False)
            self.grad_norm = self.cfg.get('grad_norm', 1.0)
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.cfg_dict = omegaconf_to_dict(config)
        print_dict(self.cfg)
        gc.enable()

        self.rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_size = int(os.getenv("WORLD_SIZE", "1"))
        
        if self.cfg.multi_gpu:
            # torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py

            os.environ['VK_ICD_FILENAMES'] = '/etc/vulkan/icd.d/nvidia_icd.json'
            os.environ['DISABLE_LAYER_NV_OPTIMUS_1'] = '1'
            os.environ['DISABLE_LAYER_AMD_SWITCHABLE_GRAPHICS_1'] = '1'
            os.environ['DISPLAY'] = str(self.rank)
            os.environ['ENABLE_DEVICE_CHOOSER_LAYER'] = '1'
            os.environ['VULKAN_DEVICE_INDEX'] = str(self.rank)
            
            self.cfg.graphics_device_id = self.rank
            self.cfg.sim_device = f'cuda:{self.rank}'
            self.cfg.rl_device = f'cuda:{self.rank}'

            dist.init_process_group("nccl", rank=self.rank, world_size=self.rank_size)
    
    # TODO: not using for now, add with scheduler
    def update_lr(self, lr):
        if self.cfg.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def trancate_gradients_and_step(self):
        ### student player model
        if self.cfg.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.student_player.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.student_player.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.rank_size
                    )
                    offset += param.numel()

        if(self.cfg.multi_gpu):
            if self.truncate_grads:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.student_player.model.parameters(), self.grad_norm)

        ### depth backbone model
        if self.cfg.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.depth_backbone.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.depth_backbone.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.rank_size
                    )
                    offset += param.numel()
        
        if(self.cfg.multi_gpu):
            if self.truncate_grads:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.depth_backbone.parameters(), self.grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

    def play(self, **kwargs):
        # rank = int(os.getenv("LOCAL_RANK", "0"))

        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{self.cfg.wandb_name}_{time_str}"

        # sets seed. if seed is -1 will pick a random one
        self.cfg.seed += self.rank
        self.cfg.seed = set_seed(self.cfg.seed, torch_deterministic=self.cfg.torch_deterministic, rank=self.rank)

        if self.cfg.wandb_activate and self.rank == 0:
        # Make sure to install WandB if you actually use this.
            import wandb

            run = wandb.init(
                project=self.cfg.wandb_project,
                # group=cfg.wandb_group,
                entity=self.cfg.wandb_entity,
                config=self.cfg_dict,
                sync_tensorboard=True,
                name=run_name,
                # id="3548w109", ### for resuming a particular id run in wandb
                resume="allow", #"must"
                monitor_gym=True,
            )

        envs = isaacgymenvs.make(
            self.cfg.seed, 
            self.cfg.task_name, 
            self.cfg.task.env.numEnvs, 
            self.cfg.sim_device,
            self.cfg.rl_device,
            self.cfg.graphics_device_id,
            self.cfg.headless,
            self.cfg.multi_gpu,
            self.cfg.capture_video,
            self.cfg.force_render,
            self.cfg,
            **kwargs,
        )
        
        phase2_train_cfg = self.cfg.train.params.config.phase2
        pbar = tqdm(range(phase2_train_cfg.max_epochs))
        
        self.player.model = self.player.model.cpu().to(self.cfg.rl_device)
        
        self.student_player = deepcopy(self.player)
        self.student_player.model.to(self.cfg.rl_device)
        self.student_player.model.train()
        
        self.depth_backbone = eval(phase2_train_cfg.backbone_class)(*phase2_train_cfg.depth_backbone_args).to(self.cfg.rl_device)
        self.depth_backbone.train()

        if(self.cfg.train.params.config.phase2.resume):
            checkpoint = torch.load(self.cfg.train.params.config.phase2.distillation_checkpoint)
            self.student_player.model.load_state_dict(checkpoint['model_state_dict'])
            self.depth_backbone.load_state_dict(checkpoint['depth_backbone_state_dict'])

        params = [
            {'params': self.student_player.model.parameters(), 'lr': 1e-5},
            {'params': self.depth_backbone.parameters(), 'lr': 1e-3},
            # {'params': gripper_depth_backbone.parameters(), 'lr': 2e-3},
        ]

        # # define the optimizer with different learning rates for each parameter group
        self.optimizer = torch.optim.Adam(params)

        # self.optimizer = torch.optim.Adam(
        #     list(self.student_player.model.parameters()) + list(self.depth_backbone.parameters()),
        #     lr=phase2_train_cfg.learning_rate
        # )

        criterion = torch.nn.MSELoss().to(self.cfg.rl_device)
        distillation_version = 58
        distillation_path = './runs/distillation_v{}_multi_gpu_learnt_camera_movement_pointnet_occlusion_aware_18-04-51-25'.format(str(distillation_version))

        if not os.path.exists(distillation_path):
            os.makedirs(distillation_path)

        obs_dict = envs.reset()
        obs_buf = obs_dict['obs']
        avg_episode_length = 0.

        if self.cfg.multi_gpu:
            torch.cuda.set_device(self.rank)
            print("====================broadcasting parameters")
            student_player_model_params = [self.student_player.model.state_dict()]
            dist.broadcast_object_list(student_player_model_params, 0)
            self.student_player.model.load_state_dict(student_player_model_params[0])

            depth_backbone_model_params = [self.depth_backbone.state_dict()] #[self.model.state_dict()]
            dist.broadcast_object_list(depth_backbone_model_params, 0)
            self.depth_backbone.load_state_dict(depth_backbone_model_params[0])
        
        prev_action = torch.zeros((obs_buf.shape[0], self.num_actions)).to(self.cfg.rl_device)
        camera_used = torch.zeros(obs_buf.shape[0]).to(self.cfg.rl_device)
        base_used = torch.zeros(obs_buf.shape[0]).to(self.cfg.rl_device)


        # actions_buffer = deque(maxlen=10000)
        # teacher_actions_buffer = deque(maxlen=10000)
        # depth_latent_buffer = deque(maxlen=10000)

        for iter in pbar:
            actions_buffer = deque(maxlen=10000)
            teacher_actions_buffer = deque(maxlen=10000)
            depth_latent_buffer = deque(maxlen=10000)
            # actions_buffer = []
            # teacher_actions_buffer = []
            # depth_latent_buffer = []

            # scandots_teacher_buffer = []
            # scandots_student_buffer = []
            # scandots_buffer = []
            avg_loss = []

            detach_hidden_states(self.student_player)
            depth_image = None
            
            # t1 = time.time()
            for step in range(phase2_train_cfg.horizon_length):
                teacher_actions, teacher_obs_compressed = self.player.get_action(obs_buf, is_deterministic=True, allow_grad=False)
                
                if depth_image is None or envs.new_depth_image: ##### check for depth cam frequency later
                    front_depth_buffer = envs.get_visual_observations()
                    front_depth_buffer = front_depth_buffer.clone().cpu().to(self.cfg.rl_device)

                    front_depth_buffer = front_depth_buffer.unsqueeze(1)
                    depth_latent = self.depth_backbone(torch.cat([front_depth_buffer], dim=1)) # (num_envs, 70) -> for dex_wrist without obstacles (state except proprioception)
                    
                    self.student_player.depth_latent = depth_latent
                    depth_latent_buffer.append(depth_latent)
                
                # visual_obs_buf = torch.cat([obs_buf[:, :34], prev_action, depth_latent, obs_buf[:, -2:]], dim=-1) # proprioception, noisy obj center and goal pos relative to base are given
                visual_obs_buf = torch.cat([obs_buf[:, :34], prev_action, obs_buf[:, 39:42], depth_latent, obs_buf[:, -2:]], dim=-1) # proprioception, noisy obj center and goal pos relative to base are given
                
                actions, _ = self.student_player.get_action(visual_obs_buf, is_deterministic=True, allow_grad=True)
                
                modified_action = actions.clone()
                if(self.use_camera_gt):
                    camera_used = torch.where(torch.norm(teacher_actions[:, 3:] - actions[:, 3:], dim=-1) < 0.2, 1., 0.) # 1 for student camera and 0 for gt camera 
                    modified_action[:, 3:] = torch.where(torch.norm(teacher_actions[:, 3:] - actions[:, 3:], dim=-1).unsqueeze(1).repeat(1, 2) < 0.2, actions[:, 3:], teacher_actions[:, 3:])

                if(self.use_base_gt):
                    base_used = torch.where(torch.norm(teacher_actions[:, :3] - actions[:, :3], dim=-1) < 0.2, 1., 0.) # 1 for student camera and 0 for gt camera
                    modified_action[:, :3] = torch.where(torch.norm(teacher_actions[:, :3] - actions[:, :3], dim=-1).unsqueeze(1).repeat(1, 3) < 0.2, actions[:, :3], teacher_actions[:, :3])

                teacher_actions_buffer.append(teacher_actions)
                actions_buffer.append(actions)

                # scandots_teacher_buffer.append(teacher_obs_compressed)
                # scandots_student_buffer.append(depth_latent)

                prev_action = actions.clone().detach()

                if(self.bc):
                    obs_dict, rews, dones, infos = envs.step(teacher_actions.clone().detach())
                else:
                    obs_dict, rews, dones, infos = envs.step(modified_action.clone().detach())

                obs_buf = obs_dict['obs'].clone()

            depth_latent_buffer_tensor = torch.stack(tuple(depth_latent_buffer), dim=0)
            actions_buffer_tensor = torch.stack(tuple(actions_buffer), dim=0)
            teacher_actions_buffer_tensor = torch.stack(tuple(teacher_actions_buffer), dim=0)

            action_loss = criterion(actions_buffer_tensor, teacher_actions_buffer_tensor)

            loss = action_loss #+ 0.5 * scandots_loss
            avg_loss.append(loss)
            if(self.rank==0):
                print('loss:', loss.item(), iter)

            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.multi_gpu:
                self.trancate_gradients_and_step()
            self.optimizer.step()

            if self.cfg.multi_gpu:
                ep_loss = torch_ext.mean_list(avg_loss)
                dist.all_reduce(ep_loss, op=dist.ReduceOp.SUM)
                ep_loss /= self.rank_size

            if self.cfg.wandb_activate and self.rank == 0:
                run.log({"distillation/action_loss": action_loss})
                # run.log({"distillation/scandots_loss": scandots_loss})
                # run.log({"distillation/total_loss": loss})

                if(self.use_base_gt):
                    run.log({"distillation/base_used": torch.sum(base_used).item()/base_used.shape[0]})
                if(self.use_camera_gt):
                    run.log({"distillation/camera_used": torch.sum(camera_used).item()/camera_used.shape[0]})
                # run.log({"distillation/loss": loss})

            if(iter%20==0):
                distillation_ckpt_path = distillation_path + '/distilation_policy_learnt_camera_movement_{}.pth'.format(str(iter))
                
                torch.save({
                    'epoch': iter,
                    'model_state_dict': self.student_player.model.state_dict(),
                    'depth_backbone_state_dict': self.depth_backbone.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                    }, distillation_ckpt_path)

                print('saved distillation ckpt at:', distillation_ckpt_path)

            gc.collect()

            new_ids = envs.reset_buf.nonzero(as_tuple=False).squeeze(-1).to(self.cfg.rl_device)
            # import pdb; pdb.set_trace()
            if(new_ids.shape[0] > 0.):
                avg_episode_length = torch.mean(envs.progress_buf[new_ids.cpu()].float())
            
            if self.cfg.wandb_activate and self.rank == 0:
                run.log({"episode_length": avg_episode_length})

            self.reset_hidden_states(self.player, new_ids)
            self.reset_hidden_states(self.student_player, new_ids)

        if self.cfg.wandb_activate and self.rank == 0:
            wandb.finish()

        if self.cfg.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.bool().item()

    def reset_hidden_states(self, player, new_ids):
        if player.is_rnn:
            for s in player.states:
                s[:, new_ids, :] = s[:,new_ids, :] * 0.0

    def restore(self):
        # import pdb; pdb.set_trace()
        rlg_config_dict = self.cfg_dict["train"]
        rlg_config_dict["params"]["config"]["env_info"] = {}
        self.num_sample_points = self.cfg_dict["task"]["env"]["numSamplePoints"]
        self.fps_ratio = self.cfg_dict["task"]["env"]["fpsRatio"]
        self.scan_dots = self.cfg_dict["task"]["env"]["scanDots"]  
        rlg_config_dict["params"]["config"]["rl_device"] = self.cfg.rl_device

        if(self.scan_dots): 
            self.num_obs = 39 + 5 + 5 * 24
            self.num_actions = 5
        else:
            self.num_obs = 39 + 5 + 5 * 24
            self.num_actions = 5
        observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
        action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        #lift
        action_space.low[0] = 0.0
        action_space.high[0] = 1.1

        # base rotation
        action_space.low[1] = -0.4
        action_space.high[1] = 0.4 

        # base translation x
        action_space.low[2] = -0.4
        action_space.high[2] = 0.4

        # camera pan
        action_space.low[3] = -3.9
        action_space.high[3] = 1.5

        # camera tilt
        action_space.low[4] = -1.53
        action_space.high[4] = 0.79

        rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
        rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1

        def build_runner(algo_observer):
            runner = Runner(algo_observer)
            runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
            runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
            model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
            model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

            return runner

        runner = build_runner(RLGPUAlgoObserver())
        runner.load(rlg_config_dict)
        runner.reset()

        args = {
            'train': False,
            'play': True,
            'checkpoint' : self.cfg['checkpoint'],
            'sigma' : None
        }

        self.player = runner.create_player()
        _restore(self.player, args)


        _override_sigma(self.player, args)

        self.player.batch_size = self.cfg["task"]["env"]["numEnvs"]
        self.player.has_batch_dimension = True

        if self.player.is_rnn:
            self.player.init_rnn()

@hydra.main(config_name='config', config_path='./cfg')
def main(cfg: DictConfig):
    agent = Dagger(cfg)
    agent.restore()
    agent.play()

def detach_hidden_states(player):
    if not player.is_rnn:
        return
    
    player.states = [s.detach().clone() for s in player.states]

if __name__ == '__main__':
    main()

# torchrun --standalone --nnodes=1 --nproc_per_node=4 train_distillation.py multi_gpu=True task=StretchPickPlaceRandomized num_envs=16 wandb_activate=False experiment='distillation_multi_gpu_learnt_camera_movement_after_rescaling_new_dof_noise_randomization_13-15-17-59' checkpoint=./checkpoints/multi_gpu_learnt_camera_movement_after_rescaling_new_dof_noise_randomization_13-15-17-59/multi_gpu_learnt_camera_movement_after_rescaling_new_dof_noise_randomization.pth
# DISPLAY=:1 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json ENABLE_DEVICE_CHOOSER_LAYER=1 VULKAN_DEVICE_INDEX=3 python train_distillation.py sim_device='cuda:3' rl_device='cuda:3' graphics_device_id=3 task=StretchPickPlaceRandomized num_envs=512 wandb_activate=True experiment='distillation_v1' checkpoint=./checkpoints/multi_gpu_learnt_camera_movement_l2_dist_plus_vel_dir_reward_w_stall_penalty_w_increased_obs_height_14-21-08-09/multi_gpu_learnt_camera_movement_l2_dist_plus_vel_dir_reward_w_stall_penalty_w_increased_obs_height.pth