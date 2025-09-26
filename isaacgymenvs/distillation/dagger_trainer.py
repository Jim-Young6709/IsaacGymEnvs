import isaacgym
import isaacgymenvs, gym
from datetime import datetime, timedelta

import torch
import torch.optim as optim
from hydra.utils import instantiate
from tqdm import tqdm
from collections import OrderedDict
from isaacgymenvs.utils.rotation_conversions import quaternion_to_matrix_ig
from isaacgymenvs.utils.pcd_utils import crop_local_pcd, visualize_pcd
from isaacgymenvs.utils.training_utils import *
from isaacgymenvs.utils.simulate_depth_cam import simulate_depth_cam_render

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import yaml
import os
import time

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder
import wandb

from typing import Dict
from pathlib import Path
from isaacgymenvs.tasks import FrankaLEAP


class Dagger:
    def __init__(self, cfg):
        # load configs
        self.multi_gpu = cfg.multi_gpu
        if self.multi_gpu:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.global_rank = int(os.getenv("RANK", "0"))
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))   

            cfg.sim_device = f"cuda:{self.local_rank}"
            cfg.rl_device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.local_rank)

            if self.local_rank == 0:
                cfg.graphics_device_id = self.local_rank
            else:
                cfg.task.env.video_logging.capture = False # note the actual video logging flag is in task env, not in general cfg.capture_video
                cfg.graphics_device_id = -1

        self.cfg = cfg
        self.total_episodes = cfg.dagger.total_episodes
        self.steps_per_episode = cfg.dagger.steps_per_episode
        self.warmup_episodes = cfg.dagger.warmup_episodes
        self.max_grad_norm = cfg.dagger.max_grad_norm
        self.local_pcd_range = cfg.dagger.local_pcd_range
        self.reaching_reset_threshold = cfg.dagger.reaching_reset_threshold
        self.device = cfg['sim_device']
        self.seed = cfg.seed
        self.exp_name = cfg.experiment
        set_seed_and_precision(self.seed)

        self.learning_rate = cfg.dagger.learning_rate
        self.weight_decay = cfg.dagger.weight_decay
        # load env
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.wandb_name}_{time_str}"
        def create_isaacgym_env(**kwargs) -> FrankaLEAP:
            envs = isaacgymenvs.make(
                cfg.seed,
                cfg.task_name,
                cfg.task.env.numEnvs,
                cfg.sim_device,
                cfg.rl_device,
                cfg.graphics_device_id,
                cfg.headless,
                cfg.multi_gpu,
                cfg.capture_video,
                cfg.force_render,
                cfg,
                **kwargs,
            )
            if cfg.capture_video:
                envs.is_vector_env = True
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                    video_length=cfg.capture_video_len,
                )
            return envs
        self.env = create_isaacgym_env()

        # load teacher
        self.value_size = 1 # not sure why, but seems most cases its 1
        self.num_seqs = 1 # this is mainly for RNNs, the number equals to the number of robot agents per env, which is 1 here
        self.normalize_value = self.cfg.train.params.config.normalize_value
        self.normalize_input = self.cfg.train.params.config.normalize_input
        self.teacher_model_config = {
            "actions_num": self.env.num_actions,
            "input_shape": (self.env.num_observations,),
            "num_seqs": self.num_seqs,
            "value_size": self.value_size,
            'normalize_value': self.normalize_value, 
            'normalize_input': self.normalize_input,
        }
        self.teacher_network_params = self.load_param_dict(self.cfg["teacher"]["cfg"])["params"]
        self.teacher_network = self.load_networks(self.teacher_network_params)
        self.teacher_model = self.teacher_network.build(self.teacher_model_config).to(self.device)
        self.set_weights(self.cfg["teacher"]["ckpt"])
        self.teacher_model.eval()
        self.is_teacher_rnn = self.teacher_model.is_rnn()

        # load student network
        self.student_model = instantiate(self.cfg.model).to(self.device)
        self.optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_episodes * self.steps_per_episode,
            num_training_steps=self.total_episodes * self.steps_per_episode
        )

        # dagger
        self.episode = 0
        self.total_steps = 0
        self.batch_idx = 0
        self.batch_size = self.cfg.dagger.batch_size

        self.use_wandb = self.cfg.wandb_activate
        self.wandb_project = self.cfg.wandb_project
        self.wandb_name = self.cfg.wandb_name
        self.wandb_id = None

        self.pcd_encoders_keys = self.cfg.model.pcd_encoders_cfg.keys()
        self.num_local_points = self.env.pcd_spec_dict["num_local_points"]

        self.save_dir = Path("dagger_ckpts") / self.exp_name
        self.save_freq = self.cfg.dagger.save_freq
        os.makedirs(self.save_dir, exist_ok=True)

        if self.use_wandb and (not self.multi_gpu or self.global_rank == 0):
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_name,
                id=self.wandb_id,
                resume="must" if self.wandb_id else None,
                config={
                    "batch_size": self.batch_size,
                    "num_episodes": self.total_episodes,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                }
            )

        if self.multi_gpu:            
            self.use_wandb = (self.cfg.wandb_activate and self.global_rank == 0)

            self.student_model = self.student_model.to(self.device)
            self.student_model = DDP(
                self.student_model,
                device_ids=[self.local_rank],
                static_graph=True,
            )

        # Load stats if provided
        # TODO: give options for resume training / retrain from scratch
        load_checkpoint_path = self.cfg.dagger.load_ckpt_path
        if load_checkpoint_path is not None:
            success_rate_ep = self.load_checkpoint(load_checkpoint_path)
            colorprint(f"Resumed training from {load_checkpoint_path}: steps={self.total_steps}, success_rate_ep={success_rate_ep}", color="magenta")


    # TODO: teacher loading utils, shall I just simply merge them?
    def load_param_dict(self, cfg_path) -> Dict:
        base_dir = os.path.dirname(__file__)
        full_path = os.path.join(base_dir, cfg_path)

        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def load_networks(self, params):
        """Loads the teacher rl network"""
        builder = ModelBuilder()
        return builder.load(params)

    def set_weights(self, ckpt):
        """Set the weights of the model."""
        weights = torch_ext.load_checkpoint(ckpt)
        model = self.teacher_model
        model.load_state_dict(weights["model"])
        if self.normalize_input and 'running_mean_std' in weights:
            model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def preprocess_inputs(self, obs):
        # eef states
        eef_pos = self.env.states['eef_pos'] # (num_envs, 3)
        eef_quat = self.env.states['eef_quat'] # (num_envs, 4)
        eef_rot_mat = quaternion_to_matrix_ig(eef_quat)
        rot_global2eef = eef_rot_mat.transpose(1, 2) # (num_envs, 3, 3)

        obs['full_pcd_t'] = torch.cat([obs["full_scene_pcd_t"], obs["robot_pcd_t"]], dim=1)

        if self.env.pcd_spec_dict['simulate_depth_cam']:
            num_full_pcd_points = self.env.pcd_spec_dict['num_static_points'] + \
                                  self.env.pcd_spec_dict['num_robot_points'] + \
                                  self.env.pcd_spec_dict['num_object_points']

            sim_depth_pcd, logs = simulate_depth_cam_render(
                obs['full_pcd_t'],
                self.env._object_center_init_state,
                num_full_pcd_points,
            )

            if self.use_wandb:
                wandb.log(logs, step=self.total_steps)

            obs['full_pcd_t'] = sim_depth_pcd

        # convert all pcd to eef frame
        for key in obs.keys():
            if "pcd" in key:
                pcd_shifted = obs[key] - eef_pos.unsqueeze(1) # (num_envs, N, 3)
                pcd_eef_frame = torch.bmm(pcd_shifted, rot_global2eef) # (num_envs, N, 3), bmm is like matmul but specifically made for batches of 2D matrices, faster than matmul
                obs[key] = pcd_eef_frame

        obs_student = OrderedDict()
        obs_student["q_hand"] = obs["q_hand"]

        if "static_scene_pcd_t0" in self.pcd_encoders_keys:
            obs_student["static_scene_pcd_t0"] = obs["static_scene_pcd_t0"]
        if "object_pcd_t0" in self.pcd_encoders_keys:
            obs_student["object_pcd_t0"] = obs["object_pcd_t0"]
        if "full_scene_pcd_t0" in self.pcd_encoders_keys:
            obs_student["full_scene_pcd_t0"] = torch.cat([obs["static_scene_pcd_t0"], obs["object_pcd_t0"]], dim=1)

        if "robot_pcd_t" in self.pcd_encoders_keys:
            obs_student["robot_pcd_t"] = obs["robot_pcd_t"]
        if "hand_pcd_t" in self.pcd_encoders_keys:
            obs_student["hand_pcd_t"] = obs["hand_pcd_t"]

        if "full_scene_pcd_t" in self.pcd_encoders_keys:
            obs_student["full_scene_pcd_t"] = obs["full_scene_pcd_t"]

        if "local_pcd_t" in self.pcd_encoders_keys:
            obs_student["local_pcd_t"], crop_logs = crop_local_pcd(obs['full_pcd_t'], self.local_pcd_range, self.num_local_points) # (num_envs, num_local_points, 3)
            if self.use_wandb:
                wandb.log(crop_logs, step=self.total_steps)
        elif "local_scene_pcd_t" in self.pcd_encoders_keys:
            obs_student["local_scene_pcd_t"], crop_logs = crop_local_pcd(obs["full_scene_pcd_t"], self.local_pcd_range, self.num_local_points)
            if self.use_wandb:
                wandb.log(crop_logs, step=self.total_steps)

        return obs_student

    def save_checkpoint(self, episode, success_rate_ep=None, top_k=3): # TODO
        checkpoint = {
            "episode": episode,
            "model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "success_rate_ep": success_rate_ep,
            "batch_idx": self.batch_idx,
            "total_steps": self.total_steps,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.use_wandb:
            checkpoint["wandb_id"] = wandb.run.id
            checkpoint["wandb_name"] = wandb.run.name
            checkpoint["wandb_project"] = wandb.run.project
        torch.save(checkpoint, self.save_dir / "latest.pt")
        if episode % self.save_freq == 0:
            torch.save(checkpoint, self.save_dir / f"episode_{episode:06d}.pt")
        
        checkpoint_files = sorted([f for f in os.listdir(self.save_dir) if f.startswith("episode_")])
        if len(checkpoint_files) > top_k:
            for old_checkpoint in checkpoint_files[:-top_k]:
                os.remove(os.path.join(self.save_dir, old_checkpoint))
        
        best_path = os.path.join(self.save_dir, "best.pt")
        if not os.path.exists(best_path) or success_rate_ep > torch.load(best_path, weights_only=True)["success_rate_ep"]:
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.student_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.episode = (checkpoint["episode"] + 1) % self.total_episodes
        self.total_steps = checkpoint["total_steps"]
        self.batch_idx = checkpoint["batch_idx"]
        if "wandb_id" in checkpoint:
            self.wandb_id = checkpoint["wandb_id"]
            self.wandb_name = checkpoint["wandb_name"]
            self.wandb_project = checkpoint["wandb_project"]
        return checkpoint["success_rate_ep"]

    def train_episode(self):
        self.env.reset() # TODO: necessary?
        count_reaching = torch.zeros(self.env.num_envs, device=self.device).int()

        for _ in tqdm(range(self.steps_per_episode), desc=f"Training {self.episode+1}/{self.total_episodes}", \
            ncols=None, dynamic_ncols=True, disable=(self.multi_gpu and self.global_rank != 0) ):

            # get teacher action
            teacher_obs = self.env.obs_buf.clone()
            batch_dict = {
                "is_train": False,
                "obs": teacher_obs,
                "prev_actions": None,
            }

            is_deterministic = True

            with torch.no_grad():
                res_dict = self.teacher_model(batch_dict)

            mu = res_dict['mus']
            action = res_dict['actions']
            self.states = res_dict['rnn_states']
            if is_deterministic:
                teacher_actions = mu
            else:
                teacher_actions = action
            teacher_actions = torch.clamp(teacher_actions, -self.env.clip_actions, self.env.clip_actions)

            # student obs, q_hand, rel_pcd
            q_robot = self.env.states['q'] # (num_envs, 23)
            q_hand = self.env.states['q_hand'] # (num_envs, 16)

            static_scene_pcd_t0 = self.env.static_scene_pcd_t0
            object_pcd_t0 = self.env.object_pcd_t0
            full_scene_pcd_t = self.env.combined_pcds
            robot_pcd_t = self.env.robot_pcd_sampler.sample(q_robot, self.env.torchurdf_to_isaac_idx)
            hand_pcd_t = self.env.robot_pcd_sampler.sample(q_robot, self.env.torchurdf_to_isaac_idx, hand_only=True)

            obs_dict = OrderedDict([
                ("static_scene_pcd_t0", static_scene_pcd_t0),
                ("object_pcd_t0", object_pcd_t0),
                ("full_scene_pcd_t", full_scene_pcd_t),
                ("robot_pcd_t", robot_pcd_t),
                ("hand_pcd_t", hand_pcd_t),
                ("q_hand", q_hand),
            ])
            obs_input = self.preprocess_inputs(obs_dict)

            with torch.no_grad():
                student_model = self.student_model.module if self.multi_gpu else self.student_model
                student_model.eval()
                student_actions_chunk = student_model(obs_input)

            student_actions = student_actions_chunk[:, 0, :] # NOTE: supervise student model on first step, there might be a way to still do ACT

            # step with student actions
            step_actions = torch.clamp(student_actions, -self.env.clip_actions, self.env.clip_actions)
            self.env.step(step_actions)

            # reset envs to start config if reached
            count_reaching += self.env.success_5cm_per_step
            count_reaching *= self.env.success_5cm_per_step
            if (count_reaching >= self.reaching_reset_threshold).any():
                reached_reset_flags = (count_reaching >= self.reaching_reset_threshold)
                reset_ids = torch.where(reached_reset_flags)[0]
                self.env.reset_buf[reset_ids] = 1
                count_reaching[reached_reset_flags] = 0
            
            self.student_model.train()
            n_batches = self.env.num_envs // self.batch_size # now this is 1
            indices = torch.randperm(self.env.num_envs, device=self.device)
            total_loss = 0
            for i in range(n_batches):
                batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                batch_obs = {k: v[batch_indices] for k, v in obs_input.items()}
                batch_actions = teacher_actions[batch_indices]
                # NOTE: supervise student model on first step
                loss = self.student_model.forward(batch_obs, batch_actions, action_chunk_idx=0)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=self.max_grad_norm) 
                self.optimizer.step()
                total_loss += loss.item()
            total_loss /= n_batches
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            if self.use_wandb:
                wandb.log({
                    "train/loss": total_loss,
                    "train/lr": self.optimizer.param_groups[0]["lr"]
                }, step=self.total_steps)
                
            self.total_steps += 1

        return total_loss

    def train(self):
        while self.episode < self.total_episodes:
            metrics = {}

            start_time = time.time()
            remaining_episodes = self.total_episodes - self.episode

            train_loss = self.train_episode()

            if (not self.multi_gpu) or (self.global_rank == 0):
                episode_time = time.time() - start_time
                estimated_finish_time = start_time + episode_time * remaining_episodes

                metrics["train/loss_episode"] = train_loss
                metrics["time/episode_time"] = episode_time
                metrics["episode"] = self.episode
                metrics.update(self.env.extras)
                if self.use_wandb:
                    wandb.log(metrics, step=self.total_steps)

                # TODO: add args: save ckpt? frequency?
                self.save_checkpoint(self.episode, metrics["metrics/success_rate_5cm_per_ep"])

                colorprint(f"Episode {self.episode + 1}/{self.total_episodes} completed in {timedelta(seconds=int(episode_time))}", color="magenta")
                for metric, value in metrics.items():
                    if type(value) == float:
                        colorprint(f"{metric}: {value:.4f}", color="green")
                colorprint(f"Average episodes per hour: {1/episode_time*3600:.2f}")
                colorprint(f"Estimated completion: {datetime.fromtimestamp(estimated_finish_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print("\n")
            
            self.episode += 1

