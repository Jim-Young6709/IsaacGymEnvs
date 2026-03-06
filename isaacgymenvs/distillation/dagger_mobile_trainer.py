import isaacgym
import isaacgymenvs, gym
from datetime import datetime, timedelta

import torch
import torch.optim as optim
from hydra.utils import instantiate
from tqdm import tqdm
from collections import OrderedDict
from isaacgymenvs.utils.rotation_conversions import quaternion_to_matrix_ig
from isaacgymenvs.utils.pcd_utils import downsample_pcd_batched, crop_local_pcd, visualize_pcd
from isaacgymenvs.utils.training_utils import *
from isaacgymenvs.utils.simulate_depth_cam_compile import simulate_depth_cam_render_from_pose
from isaacgymenvs.utils.simulate_lidar_compile import simulate_lidar_render_from_pose

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
from isaacgymenvs.tasks import FrankaLEAPMobile


class DaggerMobile:
    def __init__(self, cfg):
        # overwrite the video logging freq so it aligns well with the eval pattern
        cfg.task.env.video_logging.freq = max((cfg.dagger.eval_freq + 1), 10) * cfg.task.env.episodeLength

        # load configs
        self.multi_gpu = cfg.multi_gpu
        if self.multi_gpu:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.global_rank = int(os.getenv("RANK", "0"))
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))   

            cfg.task.env.scene.batch_idx = self.global_rank
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
        self.steps_per_episode = int(cfg.dagger.steps_per_episode / cfg.chunk_size)
        self.warmup_episodes = cfg.dagger.warmup_episodes
        self.max_grad_norm = cfg.dagger.max_grad_norm
        self.local_pcd_range = cfg.dagger.local_pcd_range
        self.reaching_reset_threshold = cfg.dagger.reaching_reset_threshold
        self.teacher_forcing_cfg = cfg.dagger.teacher_forcing
        self.chunk_size = cfg.chunk_size
        self.device = cfg['sim_device']
        self.seed = cfg.seed
        self.exp_name = cfg.experiment
        set_seed_and_precision(self.seed)

        self.learning_rate = cfg.dagger.learning_rate
        self.weight_decay = cfg.dagger.weight_decay
        # load env
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{cfg.wandb_name}_{time_str}"
        def create_isaacgym_env(**kwargs) -> FrankaLEAPMobile:
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
        self.env.reset() # Is called only once when environment starts to provide the first observations as place holder. Doesn't calculate the actual observations.

        # env cfg overrides
        self.env.distillation_mode = True
        self.env.delta_franka_action = self.cfg.action_space.delta_franka_action
        self.env.delta_leap_action = self.cfg.action_space.delta_leap_action
        self.env.delta_arx_action = self.cfg.action_space.delta_arx_action

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

        self.state_encoders_keys = self.cfg.model.state_encoders_cfg.keys()
        self.pcd_encoders_keys = self.cfg.model.pcd_encoders_cfg.keys()

        self.save_dir = Path("dagger_ckpts") / self.exp_name
        self.save_freq = self.cfg.dagger.save_freq
        os.makedirs(self.save_dir, exist_ok=True)

        self.eval_freq = self.cfg.dagger.eval_freq

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

        if self.use_wandb:
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

        # aux
        aux_weight = float(self.cfg.model.get("aux_weight", 0.0))
        self.has_aux_input = "aux_object_state" in self.state_encoders_keys
        self.has_aux_prediction = aux_weight > 0.0
        self.aux_prediction_mode = str(self.cfg.model.get("aux_prediction_mode", "absolute")).lower()
        self.aux_delta_scale = float(self.cfg.model.get("aux_delta_scale", 0.01))
        if self.aux_prediction_mode not in ["absolute", "delta"]:
            raise ValueError(f"aux_prediction_mode must be 'absolute' or 'delta', got {self.aux_prediction_mode}")
        self.aux_feedback_to_policy = bool(self.cfg.dagger.get("aux_feedback_to_policy", True))
        self.aux_init_only = bool(self.cfg.dagger.get("aux_init_only", False)) and (not self.aux_feedback_to_policy) # if not feeding aux feedback to policy, then aux is effectively only used for initialization
        self.aux_switch_steps = int(self.cfg.dagger.get("aux_feedback_start_steps", 30000))
        self.aux_buffer = torch.zeros(self.env.num_envs, 1, 3, device=self.device)

    # teacher loading utils
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

    def save_checkpoint(self, episode, train_success_rate_ep=None, eval_success_rate_ep=None, top_k=3):
        checkpoint = {
            "episode": episode,
            "model_state_dict": self.student_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_success_rate_ep": train_success_rate_ep,
            "eval_success_rate_ep": eval_success_rate_ep,
            "batch_idx": self.batch_idx,
            "total_steps": self.total_steps,
            "cfg": self.cfg,
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

        best_train_path = os.path.join(self.save_dir, "best_train_success.pt")
        if not os.path.exists(best_train_path) or train_success_rate_ep > torch.load(best_train_path, map_location="cpu")["train_success_rate_ep"]:
            torch.save(checkpoint, best_train_path)

        if eval_success_rate_ep is not None:
            best_eval_path = os.path.join(self.save_dir, "best_eval_success.pt")
            if not os.path.exists(best_eval_path) or eval_success_rate_ep > torch.load(best_eval_path, map_location="cpu")["eval_success_rate_ep"]:
                torch.save(checkpoint, best_eval_path)

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
        return checkpoint["train_success_rate_ep"]

    def preprocess_inputs(self, obs):
        # TODO(Jim): need a deep cleanup here
        wandb_logs = {}

        # franka base states
        franka_base_pos = self.env.states['franka_base_pose7'][:, :3] # (num_envs, 3)
        franka_base_quat = self.env.states['franka_base_pose7'][:, 3:] # (num_envs, 4)
        franka_base_rot_mat = quaternion_to_matrix_ig(franka_base_quat)
        rot_global2base = franka_base_rot_mat.transpose(1, 2) # (num_envs, 3, 3)

        obs['gt_pcd_t'] = torch.cat([obs["full_scene_pcd_t"], obs["robot_pcd_t"]], dim=1) # ground truth pcd, sampled from mesh surfaces

        if self.env.pcd_spec_dict['simulate_sensor_pcd']:
            num_full_pcd_points = self.env.pcd_spec_dict['num_static_points'] + \
                                  self.env.pcd_spec_dict['num_robot_points'] + \
                                  self.env.pcd_spec_dict['num_object_points'] + \
                                  self.env.pcd_spec_dict['num_distractor_points']
            num_full_pcd_points = min(10000, num_full_pcd_points)

            camera_pose7 = self.env.states['camera_pose7'].clone() # (num_envs, 7)
            sim_depth_pcd, sim_depth_render_logs = simulate_depth_cam_render_from_pose(
                pcd=obs['gt_pcd_t'],
                camera_pose=camera_pose7,
                num_points=num_full_pcd_points,
            )

            # lidar pcd
            lidar_pose7 = self.env.states["lidar_pose7"].clone() # (num_envs, 7)
            sim_lidar_pcd_raw, sim_lidar_render_logs = simulate_lidar_render_from_pose(
                pcd=obs['gt_pcd_t'],
                lidar_pose=lidar_pose7,
                num_points=num_full_pcd_points,
                num_azimuth=512,
                num_polar=128,
                suppress_bins=2,
                jitter_std_m=0.001,
            )

            # exclude robot pcd in the lidar pcd (cause in real world the robot lidar pcd is super messy and we are removing it)
            sim_lidar_pcd = self.env.robot_spherical_representation.filter_pointcloud_outside_spheres(
                pointclouds=sim_lidar_pcd_raw,
                joint_angles=self.env.states['q'].clone(),
            )

            if self.use_wandb:
                wandb_logs.update(sim_depth_render_logs)
                wandb_logs.update(sim_lidar_render_logs)

            obs['full_pcd_t'] = torch.cat([sim_depth_pcd, sim_lidar_pcd], dim=1)
        else:
            obs['full_pcd_t'] = obs['gt_pcd_t']

        if "local_pcd_t" in self.pcd_encoders_keys:
            # Codex
            num_points = self.cfg.model.pcd_encoders_cfg["local_pcd_t"]["num_points"] # [num cylindrical points, num spherical eef points, num spherical aux points]
            local_ranges = self.local_pcd_range
            local_eef_spherical_range = local_ranges[1]
            local_aux_spherical_range = local_ranges[2]

            # get full pcd in eef frame (only xyz shifted, not rotated)
            eef_pos = self.env.states['eef_pos'] # (num_envs, 3)
            full_pcd_shifted = obs['full_pcd_t'] - eef_pos.unsqueeze(1) # (num_envs, N, 3)
            eef_spherical_local_pcd_t, eef_spherical_crop_logs = crop_local_pcd(full_pcd_shifted, local_eef_spherical_range, num_points[1], is_cylindrical=False) # (num_envs, num_local_points, 3)
            # local eef pcd in global frame
            obs["local_eef_pcd_t"] = eef_spherical_local_pcd_t + eef_pos.unsqueeze(1) # back to global frame for now, will be converted to franka base frame later

            # Codex: aux-centered local pcd in global frame
            aux_crop_origin = eef_pos
            if "aux_object_state" in self.state_encoders_keys:
                noisy_object_center_pos = self.env.states["object_center_pos"].clone()
                # add noise (-0.05m ~ 0.05m)
                noisy_object_center_pos = noisy_object_center_pos + 0.1 * ( torch.rand(self.env.num_envs, 3, device=self.device) - 0.5 )
                if not self._use_aux_feedback():
                    aux_crop_origin = noisy_object_center_pos
                else:
                    aux_crop_origin = self.aux_buffer.clone()
                    if aux_crop_origin.ndim == 3:
                        aux_crop_origin = aux_crop_origin[:, 0, :]
                    if hasattr(self.env, "object_reset_mask") and torch.any(self.env.object_reset_mask):
                        aux_crop_origin[self.env.object_reset_mask] = noisy_object_center_pos[self.env.object_reset_mask]
            aux_full_pcd_shifted = obs['full_pcd_t'] - aux_crop_origin.unsqueeze(1) # (num_envs, N, 3)
            aux_spherical_local_pcd_t, aux_spherical_crop_logs = crop_local_pcd(aux_full_pcd_shifted, local_aux_spherical_range, num_points[2], is_cylindrical=False) # (num_envs, num_local_points, 3)
            obs["local_aux_pcd_t"] = aux_spherical_local_pcd_t + aux_crop_origin.unsqueeze(1)

        # convert all pcd to franka base frame
        for key in obs.keys():
            if "pcd" in key:
                pcd_shifted = obs[key] - franka_base_pos.unsqueeze(1) # (num_envs, N, 3)
                pcd_base_frame = torch.bmm(pcd_shifted, rot_global2base) # (num_envs, N, 3), bmm is like matmul but specifically made for batches of 2D matrices, faster than matmul
                obs[key] = pcd_base_frame

        obs_student = OrderedDict()

        if "full_scene_pcd_t0" in self.pcd_encoders_keys:
            obs["full_scene_pcd_t0"] = torch.cat([obs["static_scene_pcd_t0"], obs["object_pcd_t0"]], dim=1)

        for key in self.pcd_encoders_keys:
            if key in ["static_scene_pcd_t0", "object_pcd_t0", "full_scene_pcd_t0", "full_scene_pcd_t", "robot_pcd_t", "hand_pcd_t"]:
                num_points_key = self.cfg.model.pcd_encoders_cfg[key]["num_points"]
                obs_student[key] = downsample_pcd_batched(obs[key], num_points_key)

        if "full_pcd_t" in self.pcd_encoders_keys:
            num_points_full_pcd_t = self.cfg.model.pcd_encoders_cfg["full_pcd_t"]["num_points"]
            if self.env.pcd_spec_dict['simulate_sensor_pcd']:
                full_pcd_t = obs["full_pcd_t"][:, :num_points_full_pcd_t]
                # replace nan values as 0s
                full_pcd_t_zero_padding = torch.nan_to_num(full_pcd_t, nan=0.0)
                obs_student["full_pcd_t"] = full_pcd_t_zero_padding
            else:
                obs_student["full_pcd_t"] = downsample_pcd_batched(obs["full_pcd_t"], num_points_full_pcd_t)

        if "local_pcd_t" in self.pcd_encoders_keys:
            # Codex
            num_points = self.cfg.model.pcd_encoders_cfg["local_pcd_t"]["num_points"] # [num cylindrical points, num spherical eef points, num spherical aux points]
            cylindrical_local_pcd_t, cylindrical_crop_logs = crop_local_pcd(obs['full_pcd_t'], self.local_pcd_range[0], num_points[0], is_cylindrical=True) # (num_envs, num_local_points, 3)
            obs_student["local_pcd_t"] = torch.cat([cylindrical_local_pcd_t, obs["local_eef_pcd_t"], obs["local_aux_pcd_t"]], dim=1)

            if self.use_wandb:
                wandb_logs.update(cylindrical_crop_logs)
                wandb_logs.update(eef_spherical_crop_logs)
                wandb_logs.update({
                    # Codex
                    "local_spherical_crop_aux/avg_num_valid_points": aux_spherical_crop_logs["local_spherical_crop/avg_num_valid_points"],
                    # Codex
                    "local_spherical_crop_aux/min_num_valid_points": aux_spherical_crop_logs["local_spherical_crop/min_num_valid_points"],
                })

        elif "local_scene_pcd_t" in self.pcd_encoders_keys: # TODO: this is kinda outdated
            obs_student["local_scene_pcd_t"], crop_logs = crop_local_pcd(obs["full_scene_pcd_t"], self.local_pcd_range[0], self.cfg.model.pcd_encoders_cfg["local_pcd_t"]["num_points"][0], is_cylindrical=True)
            if self.use_wandb:
                wandb_logs.update(crop_logs)

        # for viser visualization
        # env_id = self.env.viser_visualizer.env_id
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="full_points", 
        #     point_cloud=obs['gt_pcd_t'][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="rendered_full_points",
        #     point_cloud=obs['full_pcd_t'][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="rendered_cam_points",
        #     point_cloud=sim_depth_pcd[env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="rendered_lidar_points",
        #     point_cloud=sim_lidar_pcd[env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="policy_input_points",
        #     point_cloud=obs_student["local_pcd_t"][env_id].cpu().numpy()
        # )

        return obs_student, wandb_logs

    # Codex
    def _get_object_center_pos_in_base_frame(self, use_initial_frame=False):
        if use_initial_frame:
            object_center_pos = self.env._object_center_init_state.clone()
        else:
            object_center_pos = self.env.states["object_center_pos"].clone()

        franka_base_pos = self.env.states['franka_base_pose7'][:, :3] # (num_envs, 3)
        franka_base_quat = self.env.states['franka_base_pose7'][:, 3:] # (num_envs, 4)
        franka_base_rot_mat = quaternion_to_matrix_ig(franka_base_quat)
        rot_global2base = franka_base_rot_mat.transpose(1, 2) # (num_envs, 3, 3)

        point_shifted = (object_center_pos - franka_base_pos).unsqueeze(1) # (num_envs, 1, 3)
        point_base_frame = torch.bmm(point_shifted, rot_global2base) # (num_envs, N, 3)
        return point_base_frame[:, 0, :] # (num_envs, 3)

    def _aux_to_2d(self, aux_tensor):
        if aux_tensor.ndim == 3:
            return aux_tensor[:, 0, :]
        return aux_tensor

    def _decode_aux_prediction(self, aux_pred, prev_abs_aux):
        prev_abs_aux_2d = self._aux_to_2d(prev_abs_aux)
        if self.aux_prediction_mode == "delta":
            aux_delta = torch.clamp(aux_pred, -1.0, 1.0)
            return prev_abs_aux_2d.unsqueeze(1) + self.aux_delta_scale * aux_delta
        return self._aux_to_2d(aux_pred).unsqueeze(1)

    def _use_aux_feedback(self):
        return self.has_aux_input and self.has_aux_prediction and self.aux_feedback_to_policy and (self.total_steps >= self.aux_switch_steps)

    def _get_aux_target(self, object_center_pos, prev_abs_aux):
        prev_abs_aux_2d = self._aux_to_2d(prev_abs_aux)
        if self.aux_prediction_mode == "delta":
            target_delta = (object_center_pos - prev_abs_aux_2d) / self.aux_delta_scale
            return torch.clamp(target_delta, -1.0, 1.0)
        return object_center_pos

    def train_episode(self):
        count_reaching = torch.zeros(self.env.num_envs, device=self.device).int()

        # get teacher forcing envs
        teacher_forcing_prop = 0.0
        if self.teacher_forcing_cfg.enable:
            if self.episode < self.teacher_forcing_cfg.warmup_episodes:
                teacher_forcing_prop = 1.0
            elif self.episode < self.teacher_forcing_cfg.warmup_episodes + self.teacher_forcing_cfg.scheduling_episodes:
                teacher_forcing_prop = 1.0 - (self.episode - self.teacher_forcing_cfg.warmup_episodes) / self.teacher_forcing_cfg.scheduling_episodes
        num_teacher_forcing_envs = int(teacher_forcing_prop * self.env.num_envs)
        teacher_forcing_env_idx = np.random.choice(self.env.num_envs, size=num_teacher_forcing_envs, replace=False)
        self.teacher_forcing_prop = teacher_forcing_prop # for logging purposes

        for _ in tqdm(range(self.steps_per_episode), desc=f"Training {self.episode+1}/{self.total_episodes}", \
            ncols=None, dynamic_ncols=True, disable=(self.multi_gpu and self.global_rank != 0) ):
            self.total_steps += 1

            # get obs t_a0 for student, q_hand, rel_pcd
            q_robot = self.env.states['q'].clone() # (num_envs, 32)

            if self.env.sim_steps == 0:
                self.env.abs_actions[:] = q_robot

            static_scene_pcd_t0 = self.env.static_scene_pcd_t0 # scene pcd doesn't include object
            object_pcd_t0 = self.env.object_pcd_t0
            full_scene_pcd_t = self.env.combined_pcds # scene pcd + object pcd
            robot_pcd_t = self.env.robot_pcd_sampler.sample(q_robot, self.env.torchurdf_to_isaac_idx)
            hand_pcd_t = self.env.robot_pcd_sampler.sample(q_robot, self.env.torchurdf_to_isaac_idx, hand_only=True)

            # prepare pcd inputs
            obs_dict_a0 = OrderedDict([
                ("static_scene_pcd_t0", static_scene_pcd_t0),
                ("object_pcd_t0", object_pcd_t0),
                ("full_scene_pcd_t", full_scene_pcd_t),
                ("robot_pcd_t", robot_pcd_t),
                ("hand_pcd_t", hand_pcd_t),
            ])
            obs_input_a0, input_wandb_logs = self.preprocess_inputs(obs_dict_a0)

            # prepare state inputs
            q_arm_manip = self.env.states['q'][:, 3:10].clone() # (num_envs, 7)
            q_arm_vision = self.env.states['q'][:, 26:].clone() # (num_envs, 6)
            q_hand = self.env.states['q'][:, 10:26].clone() # (num_envs, 16)

            obs_input_a0["q_arm_manip"] = self.env.normalize_robot_joints(q_arm_manip, robot="franka", delta=False)
            obs_input_a0["q_arm_vision"] = self.env.normalize_robot_joints(q_arm_vision, robot="arx", delta=False)
            obs_input_a0["q_hand"] = self.env.normalize_robot_joints(q_hand, robot="leap", delta=False)
            if "q_hand_ctrl_delta" in self.state_encoders_keys:
                obs_input_a0["q_hand_ctrl_delta"] = self.env.normalize_robot_joints(q_hand - self.env.abs_actions[:, 10:26], robot="leap", delta=True)
            if "objxyz_t0" in self.state_encoders_keys:
                obs_input_a0["objxyz_t0"] = self.env._object_center_init_state.clone()

                franka_base_pos = self.env.states['franka_base_pose7'][:, :3] # (num_envs, 3)
                franka_base_quat = self.env.states['franka_base_pose7'][:, 3:] # (num_envs, 4)
                franka_base_rot_mat = quaternion_to_matrix_ig(franka_base_quat)
                rot_global2base = franka_base_rot_mat.transpose(1, 2) # (num_envs, 3, 3)

                point_shifted = (obs_input_a0["objxyz_t0"] - franka_base_pos).unsqueeze(1) # (num_envs, 1, 3)
                point_base_frame = torch.bmm(point_shifted, rot_global2base) # (num_envs, N, 3), bmm is like matmul but specifically made for batches of 2D matrices (input has to be 3D), faster than matmul
                obs_input_a0["objxyz_t0"] = point_base_frame[:, 0, :] # (num_envs, 3)
            if "aux_object_state" in self.state_encoders_keys:
                # Codex
                noisy_object_center_pos = self._get_object_center_pos_in_base_frame(use_initial_frame=self.aux_init_only)
                # add noise (-0.05m ~ 0.05m)
                noisy_object_center_pos = noisy_object_center_pos + 0.1 * ( torch.rand(self.env.num_envs, 3, device=self.device) - 0.5 )

                if not self._use_aux_feedback():
                    obs_input_a0["aux_object_state"] = noisy_object_center_pos
                else:
                    aux_object_state = self.aux_buffer.clone()
                    # Codex
                    if hasattr(self.env, "object_reset_mask") and torch.any(self.env.object_reset_mask):
                        if aux_object_state.ndim == 3:
                            aux_object_state[self.env.object_reset_mask, 0, :] = noisy_object_center_pos[self.env.object_reset_mask]
                        else:
                            aux_object_state[self.env.object_reset_mask, :] = noisy_object_center_pos[self.env.object_reset_mask]
                    obs_input_a0["aux_object_state"] = aux_object_state

                # Codex
                if hasattr(self.env, "object_reset_mask"):
                    self.env.object_reset_mask[:] = False

            # Viser debug utils
            # env_id = self.env.viser_visualizer.env_id
            # self.env.viser_visualizer.update_point_cloud(
            #     point_cloud_type="obj_point_t",
            #     point_cloud=obs_input_a0["objxyz_t0"][env_id].reshape(1, 3).cpu().numpy()
            # )

            with torch.no_grad():
                student_model = self.student_model.module if self.multi_gpu else self.student_model
                student_model.eval()
                output = student_model(obs_input_a0)
                student_actions_chunk = output["action"]
                if self.has_aux_prediction:
                    self.aux_buffer[:] = self._decode_aux_prediction(output["aux"], obs_input_a0["aux_object_state"])

            teacher_preds_buffer = []
            aux_ref_state = obs_input_a0["aux_object_state"] if self.has_aux_prediction else None

            for action_idx in range(self.chunk_size):
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
                self.env._pre_physics_step_teacher(teacher_actions)
                teacher_actions = self.env.teacher_actions_converted.clone()
                if self.has_aux_prediction:
                    object_center_pos = self._get_object_center_pos_in_base_frame()
                    aux_target = self._get_aux_target(object_center_pos, aux_ref_state)
                    teacher_pred = torch.cat([teacher_actions, aux_target], dim=1) # add aux info, object_xyz_pos
                else:
                    teacher_pred = teacher_actions

                teacher_preds_buffer.append(teacher_pred)

                student_actions = student_actions_chunk[:, action_idx, :]
                step_actions = student_actions
                step_actions[teacher_forcing_env_idx] = teacher_actions[teacher_forcing_env_idx]
                # step with student actions
                step_actions = torch.clamp(step_actions, -self.env.clip_actions, self.env.clip_actions)

                self.env.progress_buf -= 1 # to avoid automatic resets during the chunk steps, only update progress_buf at the end of the chunk
                if action_idx == self.chunk_size - 1:
                    self.env.progress_buf += self.chunk_size

                    # early reset: reset envs to start config if reached (and stay reached for a while)
                    if (count_reaching >= self.reaching_reset_threshold).any():
                        reached_reset_flags = (count_reaching >= self.reaching_reset_threshold)
                        reset_ids = torch.where(reached_reset_flags)[0]
                        self.env.reset_buf[reset_ids] = 1
                        count_reaching[reached_reset_flags] = 0

                # sync distillation steps for wandb video logging
                self.env.distillation_steps = self.total_steps
                self.env.step(step_actions)

                # count continuous reaching success
                count_reaching += self.env.success_5cm_per_step
                count_reaching *= self.env.success_5cm_per_step

            teacher_preds_buffer = torch.stack(teacher_preds_buffer, dim=1) # (num_envs, chunk_size, action_dim)

            self.student_model.train()
            n_batches = self.env.num_envs // self.batch_size # now this is 1
            indices = torch.randperm(self.env.num_envs, device=self.device)
            ave_loss = {
                "action": 0.0,
                "aux": 0.0,
                "total": 0.0,
            }
            for i in range(n_batches):
                batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                batch_obs = {k: v[batch_indices] for k, v in obs_input_a0.items()}
                batch_actions = teacher_preds_buffer[batch_indices]
                # NOTE: supervise student model on first step
                loss = self.student_model.forward(batch_obs, batch_actions, action_chunk_idx=0)
                self.optimizer.zero_grad()
                loss["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=self.max_grad_norm) 
                self.optimizer.step()

                for key in loss.keys():
                    ave_loss[key] += loss[key].item()

            for key in ave_loss.keys():
                ave_loss[key] /= n_batches

            if self.scheduler is not None:
                self.scheduler.step()

            mem_allocated_GB = float(torch.cuda.memory_allocated() / 1024**3)
            mem_reserved_GB = float(torch.cuda.memory_reserved() / 1024**3)

            if self.use_wandb:
                wandb_logs = {
                    "train/loss_total": ave_loss["total"],
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "mem/allocated_GB": mem_allocated_GB,
                    "mem/reserved_GB": mem_reserved_GB,
                }
                wandb_logs.update(input_wandb_logs)
                if self.has_aux_prediction:
                    aux_wandb_logs = {
                        "train/loss_aux": ave_loss["aux"],
                        "train/loss_action": ave_loss["action"],
                    }
                    wandb_logs.update(aux_wandb_logs)

                wandb.log(wandb_logs, step=self.total_steps)

        return ave_loss

    def eval(self):
        self.env.reset_idx()
        self.env.compute_observations()
        self.env.abs_actions[:] = self.env.states['q'].clone()

        for _ in tqdm(range(self.steps_per_episode), desc="Evaluating", \
            ncols=None, dynamic_ncols=True, disable=(self.multi_gpu and self.global_rank != 0) ):

            # get obs t_a0 for student, q_hand, rel_pcd
            q_robot = self.env.states['q'].clone() # (num_envs, 32)

            static_scene_pcd_t0 = self.env.static_scene_pcd_t0 # scene pcd doesn't include object
            object_pcd_t0 = self.env.object_pcd_t0
            full_scene_pcd_t = self.env.combined_pcds # scene pcd + object pcd
            robot_pcd_t = self.env.robot_pcd_sampler.sample(q_robot, self.env.torchurdf_to_isaac_idx)
            hand_pcd_t = self.env.robot_pcd_sampler.sample(q_robot, self.env.torchurdf_to_isaac_idx, hand_only=True)

            # prepare pcd inputs
            obs_dict_a0 = OrderedDict([
                ("static_scene_pcd_t0", static_scene_pcd_t0),
                ("object_pcd_t0", object_pcd_t0),
                ("full_scene_pcd_t", full_scene_pcd_t),
                ("robot_pcd_t", robot_pcd_t),
                ("hand_pcd_t", hand_pcd_t),
            ])
            obs_input_a0, _ = self.preprocess_inputs(obs_dict_a0)

            # prepare state inputs
            q_arm_manip = self.env.states['q'][:, 3:10].clone() # (num_envs, 7)
            q_arm_vision = self.env.states['q'][:, 26:].clone() # (num_envs, 6)
            q_hand = self.env.states['q'][:, 10:26].clone() # (num_envs, 16)

            obs_input_a0["q_arm_manip"] = self.env.normalize_robot_joints(q_arm_manip, robot="franka", delta=False)
            obs_input_a0["q_arm_vision"] = self.env.normalize_robot_joints(q_arm_vision, robot="arx", delta=False)
            obs_input_a0["q_hand"] = self.env.normalize_robot_joints(q_hand, robot="leap", delta=False)
            if "q_hand_ctrl_delta" in self.state_encoders_keys:
                obs_input_a0["q_hand_ctrl_delta"] = self.env.normalize_robot_joints(q_hand - self.env.abs_actions[:, 10:26], robot="leap", delta=True)
            if "objxyz_t0" in self.state_encoders_keys:
                obs_input_a0["objxyz_t0"] = self.env._object_center_init_state.clone()

                franka_base_pos = self.env.states['franka_base_pose7'][:, :3] # (num_envs, 3)
                franka_base_quat = self.env.states['franka_base_pose7'][:, 3:] # (num_envs, 4)
                franka_base_rot_mat = quaternion_to_matrix_ig(franka_base_quat)
                rot_global2base = franka_base_rot_mat.transpose(1, 2) # (num_envs, 3, 3)

                point_shifted = (obs_input_a0["objxyz_t0"] - franka_base_pos).unsqueeze(1) # (num_envs, 1, 3)
                point_base_frame = torch.bmm(point_shifted, rot_global2base) # (num_envs, N, 3), bmm is like matmul but specifically made for batches of 2D matrices (input has to be 3D), faster than matmul
                obs_input_a0["objxyz_t0"] = point_base_frame[:, 0, :] # (num_envs, 3)
            if "aux_object_state" in self.state_encoders_keys:
                if self._use_aux_feedback():
                    obs_input_a0["aux_object_state"] = self.aux_buffer.clone()
                else:
                    obs_input_a0["aux_object_state"] = self._get_object_center_pos_in_base_frame(use_initial_frame=self.aux_init_only)

            # Viser debug utils
            # env_id = self.env.viser_visualizer.env_id
            # self.env.viser_visualizer.update_point_cloud(
            #     point_cloud_type="obj_point_t",
            #     point_cloud=obs_input_a0["objxyz_t0"][env_id].reshape(1, 3).cpu().numpy()
            # )

            with torch.no_grad():
                student_model = self.student_model.module if self.multi_gpu else self.student_model
                student_model.eval()
                output = student_model(obs_input_a0)
                student_actions_chunk = output["action"]
                if self.has_aux_prediction:
                    self.aux_buffer[:] = self._decode_aux_prediction(output["aux"], obs_input_a0["aux_object_state"])

            for action_idx in range(self.chunk_size):
                # # get teacher action
                # teacher_obs = self.env.obs_buf.clone()
                # batch_dict = {
                #     "is_train": False,
                #     "obs": teacher_obs,
                #     "prev_actions": None,
                # }

                # is_deterministic = True

                # with torch.no_grad():
                #     res_dict = self.teacher_model(batch_dict)

                # mu = res_dict['mus']
                # action = res_dict['actions']
                # self.states = res_dict['rnn_states']
                # if is_deterministic:
                #     teacher_actions = mu
                # else:
                #     teacher_actions = action
                # teacher_actions = torch.clamp(teacher_actions, -self.env.clip_actions, self.env.clip_actions)
                # self.env._pre_physics_step_teacher(teacher_actions)
                # teacher_actions = self.env.teacher_actions_converted.clone()

                student_actions = student_actions_chunk[:, action_idx, :]
                step_actions = student_actions # for debugging purposes, this can be changed to teacher_actions to see teacher performance
                # step with student actions
                step_actions = torch.clamp(step_actions, -self.env.clip_actions, self.env.clip_actions)

                self.env.progress_buf[:] = 0 # since we are only evaling one episode, just disable env resets, it might mess up loggings a little bit

                self.env.step(step_actions)

        # set env state back for training
        self.env.reset_idx()
        self.env.compute_observations()
        self.env.progress_buf = torch.randint(
            0, self.env.max_episode_length,
            (self.env.num_envs,),
            device=self.device,
            dtype=self.env.progress_buf.dtype,
        )
        self.env.abs_actions[:] = self.env.states['q'].clone()

        eval_wandb_logs = {
            "metrics/eval_success_rate_5cm_final_step": self.env.extras["metrics/success_rate_5cm_per_step"],
            "metrics/eval_success_rate_5cm_per_ep": self.env.extras["metrics/success_rate_5cm_per_ep"],
            "metrics/eval_lifting_rate_5cm_final_step": self.env.extras["metrics/lifting_rate_5cm_per_step"],
            "metrics/eval_lifting_rate_5cm_per_ep": self.env.extras["metrics/lifting_rate_5cm_per_ep"],
        }

        return eval_wandb_logs

    def train(self):
        while self.episode < self.total_episodes:
            metrics = {}

            start_time = time.time()
            remaining_episodes = self.total_episodes - self.episode

            train_loss = self.train_episode()

            eval_policy = (self.eval_freq > 0) and (self.episode % self.eval_freq == 0)
            if eval_policy:
                eval_wandb_logs = self.eval()

            if (not self.multi_gpu) or (self.global_rank == 0):
                episode_time = time.time() - start_time
                estimated_finish_time = start_time + episode_time * remaining_episodes

                metrics["train/loss_episode"] = train_loss
                metrics["time/episode_time"] = episode_time
                metrics["episode"] = self.episode
                metrics["train/teacher_forcing_prop"] = self.teacher_forcing_prop
                metrics.update(self.env.extras)
                if eval_policy:
                    metrics.update(eval_wandb_logs)

                if self.use_wandb:
                    wandb.log(metrics, step=self.total_steps)

                # TODO: add args: save ckpt? frequency?
                self.save_checkpoint(self.episode, metrics["metrics/success_rate_5cm_per_ep"], metrics.get("metrics/eval_lifting_rate_5cm_per_ep", None))

                colorprint(f"Episode {self.episode + 1}/{self.total_episodes} completed in {timedelta(seconds=int(episode_time))}", color="magenta")
                for metric, value in metrics.items():
                    if type(value) == float:
                        colorprint(f"{metric}: {value:.4f}", color="green")
                colorprint(f"Average episodes per hour: {1/episode_time*3600:.2f}")
                colorprint(f"Estimated completion: {datetime.fromtimestamp(estimated_finish_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print("\n")

            self.episode += 1
