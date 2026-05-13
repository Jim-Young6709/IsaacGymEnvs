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
        self.local_rank = 0  # CODEX: keep rank fields available for optional env hooks in single-GPU runs.
        self.global_rank = 0  # CODEX: verified-bank loading uses this shard rank when present.
        self.world_size = 1  # CODEX
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

            cfg.seed = max(cfg.seed, 1) * (self.global_rank + 1)

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
        self._maybe_load_verified_teacher_bank()  # CODEX: env-owned optional reset-bank loading.

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
        self.use_bf16 = True # hardcoded to true for now
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
        self.grad_updates_per_step = self.cfg.dagger.grad_updates_per_step

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
        self.teacher_eval_only = bool(self.cfg.dagger.get("teacher_eval_only", False))  # CODEX
        self.profile_timing = bool(self.cfg.dagger.get("profile_timing", True))
        self.profile_print_freq = int(self.cfg.dagger.get("profile_print_freq", 1))
        self.profile_cuda_sync = bool(self.cfg.dagger.get("profile_cuda_sync", True))

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
                    "grad_updates_per_step": self.grad_updates_per_step,
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
        self.aux_anchor_state = torch.zeros(self.env.num_envs, 3, device=self.device)  # CODEX: aux input in current base frame.
        self.aux_anchor_base_pose7 = torch.zeros(self.env.num_envs, 7, device=self.device)  # CODEX
        self.aux_anchor_base_pose7[:, 6] = 1.0  # CODEX: identity quaternion until first refresh.
        self.last_aux_state_from_prev_step_world = torch.zeros(self.env.num_envs, 3, device=self.device)  # CODEX

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
        if (episode + 1) % 50 == 0:
            torch.save(checkpoint, self.save_dir / f"distillation_episode_{episode + 1:06d}.pt")

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

    def _maybe_load_verified_teacher_bank(self):
        if not bool(getattr(self.env, "_verified_teacher_bank_enable", False)):
            return False
        if not hasattr(self.env, "load_verified_teacher_bank_hdf5"):
            raise RuntimeError("Env enables verified_teacher_bank but does not implement load_verified_teacher_bank_hdf5")

        loaded_verified_bank = self.env.load_verified_teacher_bank_hdf5(rank=self.global_rank, strict=False)
        if (not self.multi_gpu) or (self.global_rank == 0):
            shard_path = (
                self.env._get_verified_teacher_bank_shard_path(rank=self.global_rank)
                if hasattr(self.env, "_get_verified_teacher_bank_shard_path")
                else "<unknown>"
            )
            print(
                "[DaggerMobile/verified_teacher_bank] "
                f"enable=True loaded={loaded_verified_bank} path={shard_path}"
            )

        if loaded_verified_bank:
            # CODEX: after loading the env-owned bank, force one reset so the first rollout uses bank states.
            all_env_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
            self.env.reset_idx(all_env_ids)
            self.env.compute_observations()
            if hasattr(self.env, "abs_actions") and "q" in self.env.states:
                self.env.abs_actions[:] = self.env.states["q"].clone()
        return loaded_verified_bank

    def _get_teacher_step_actions(self):
        # CODEX: teacher-only eval uses the same teacher conversion path as DAgger supervision.
        teacher_obs = self.env.obs_buf.clone()
        batch_dict = {
            "is_train": False,
            "obs": teacher_obs,
            "prev_actions": None,
        }
        with torch.no_grad():
            res_dict = self.teacher_model(batch_dict)
        teacher_actions = res_dict["mus"]
        self.states = res_dict["rnn_states"]
        teacher_actions = torch.clamp(teacher_actions, -self.env.clip_actions, self.env.clip_actions)
        self.env._pre_physics_step_teacher(teacher_actions)
        return self.env.teacher_actions_converted.clone()

    # profiling utils
    def _sync_for_timing(self):
        if (not self.profile_timing) or (not self.profile_cuda_sync) or (not torch.cuda.is_available()):
            return
        device = torch.device(self.device) if str(self.device).startswith("cuda") else None
        torch.cuda.synchronize(device=device)

    def _profile_start(self):
        if not self.profile_timing:
            return None
        self._sync_for_timing()
        return time.perf_counter()

    def _profile_end(self, profile_stats, name, start_time):
        if start_time is None:
            return
        self._sync_for_timing()
        if profile_stats is None:
            return
        profile_stats[name] = profile_stats.get(name, 0.0) + (time.perf_counter() - start_time)

    def _summarize_profile_stats(self, profile_stats, total_key=None, top_k=6):
        if not profile_stats:
            return "no timing data"

        total_time = profile_stats.get(total_key, 0.0) if total_key is not None else sum(profile_stats.values())
        ranked_items = sorted(
            [(key, value) for key, value in profile_stats.items() if key != total_key and value > 0.0],
            key=lambda item: item[1],
            reverse=True,
        )
        if not ranked_items:
            return "no timing data"

        summary_parts = []
        for key, value in ranked_items[:]:
            ratio = (100.0 * value / total_time) if total_time > 0.0 else 0.0
            summary_parts.append(f"{key}={value:.3f}s ({ratio:.1f}%)")
        return ", ".join(summary_parts)

    def preprocess_inputs(self, obs, profile_stats=None):
        # TODO(Jim): need a deep cleanup here
        wandb_logs = {}

        # franka base states
        profile_start = self._profile_start()
        franka_base_pos = self.env.states['franka_base_pose7'][:, :3] # (num_envs, 3)
        franka_base_quat = self.env.states['franka_base_pose7'][:, 3:] # (num_envs, 4)
        franka_base_rot_mat = quaternion_to_matrix_ig(franka_base_quat)
        rot_global2base = franka_base_rot_mat.transpose(1, 2) # (num_envs, 3, 3)

        obs['gt_pcd_t'] = torch.cat([obs["full_scene_pcd_t"], obs["robot_pcd_t"]], dim=1) # ground truth pcd, sampled from mesh surfaces
        self._profile_end(profile_stats, "preprocess/base_frame", profile_start)

        if self.env.pcd_spec_dict['simulate_sensor_pcd']:
            profile_start = self._profile_start()
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

            self._profile_end(profile_stats, "preprocess/depth_render", profile_start)
            profile_start = self._profile_start()

            # lidar pcd
            lidar_pose7 = self.env.states["lidar_pose7"].clone() # (num_envs, 7)
            sim_lidar_pcd_raw, sim_lidar_render_logs = simulate_lidar_render_from_pose(
                pcd=obs['gt_pcd_t'],
                lidar_pose=lidar_pose7,
                num_points=num_full_pcd_points,
                num_azimuth=128,
                num_polar=256,
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

            obs['depth_pcd_t'] = sim_depth_pcd
            obs['lidar_pcd_t'] = sim_lidar_pcd
            obs['full_pcd_t'] = torch.cat([sim_depth_pcd, sim_lidar_pcd], dim=1)
            self._profile_end(profile_stats, "preprocess/lidar_render", profile_start)
        else:
            obs['full_pcd_t'] = obs['gt_pcd_t']

        if "local_pcd_t" in self.pcd_encoders_keys:
            profile_start = self._profile_start()
            # get cropping params
            num_points = torch.tensor(self.cfg.model.pcd_encoders_cfg["local_pcd_t"]["num_points"], device=self.device, dtype=torch.int) # [num cylindrical points, num spherical eef points, num spherical aux points]
            depth_pcd_ratio = self.cfg["task"]["pcd_spec"].get("depth_pcd_ratio", 1.0)
            num_points_dict = {}
            num_points_dict['depth'] = (num_points * depth_pcd_ratio).to(torch.int)
            num_points_dict['lidar'] = num_points - num_points_dict['depth']

            local_ranges = self.local_pcd_range # [base cylindrical crop range, eef spherical crop range, aux spherical crop range]
            eef_pos = self.env.states['eef_pos'] # (num_envs, 3)
            # get aux origin
            aux_crop_origin = eef_pos
            if "aux_object_state" in self.state_encoders_keys:
                # CODEX: crop around the same aux anchor passed to the policy, converted back to world frame.
                aux_crop_origin = self._base_points_to_world_frame(
                    self.aux_anchor_state.clone(),
                    self.aux_anchor_base_pose7,
                )

            for key in ['depth', 'lidar']:
                if num_points_dict[key][1] > 0:
                    eef_spherical_local_pcd_t, eef_spherical_crop_logs = crop_local_pcd(
                        pcd=obs[f'{key}_pcd_t'],
                        local_range=local_ranges[1],
                        num_local_points=num_points_dict[key][1],
                        is_cylindrical=False,
                        crop_center=eef_pos,
                        log_name=f"eef{key}",
                    ) # (num_envs, num_local_points, 3)
                else:
                    eef_spherical_local_pcd_t = torch.zeros((self.env.num_envs, 0, 3), device=self.device)
                obs[f"local_eef{key}_pcd_t"] = eef_spherical_local_pcd_t # local eef pcd in global frame

                aux_spherical_local_pcd_t, aux_spherical_crop_logs = crop_local_pcd(
                    pcd=obs[f'{key}_pcd_t'],
                    local_range=local_ranges[2],
                    num_local_points=num_points_dict[key][2],
                    is_cylindrical=False,
                    crop_center=aux_crop_origin,
                    log_name=f"aux{key}",
                ) # (num_envs, num_local_points, 3)
                obs[f"local_aux{key}_pcd_t"] = aux_spherical_local_pcd_t # local aux pcd in global frame

                base_cylindrical_local_pcd_t, base_cylindrical_crop_logs = crop_local_pcd(
                    pcd=obs[f'{key}_pcd_t'],
                    local_range=local_ranges[0],
                    num_local_points=num_points_dict[key][0],
                    is_cylindrical=True,
                    crop_center=franka_base_pos,
                    log_name=f"base{key}",
                ) # (num_envs, num_local_points, 3)
                obs[f"local_base{key}_pcd_t"] = base_cylindrical_local_pcd_t # local base pcd in global frame

                if self.use_wandb:
                    if num_points_dict[key][1] > 0:
                        wandb_logs.update(eef_spherical_crop_logs)
                    wandb_logs.update(aux_spherical_crop_logs)
                    wandb_logs.update(base_cylindrical_crop_logs)

            obs["local_pcd_t"] = torch.cat(
                [
                    obs["local_basedepth_pcd_t"],
                    obs["local_baselidar_pcd_t"],
                    obs["local_eefdepth_pcd_t"],
                    obs["local_eeflidar_pcd_t"],
                    obs["local_auxdepth_pcd_t"],
                    obs["local_auxlidar_pcd_t"],
                ],
                dim=1,
            )
            self._profile_end(profile_stats, "preprocess/local_crop", profile_start)

        # for viser visualization
        if self.env.enable_viser:
            env_id = self.env.viser_visualizer.env_id
            self.env.viser_visualizer.update_point_cloud(
                point_cloud_type="full_points", 
                point_cloud=obs['gt_pcd_t'][env_id].cpu().numpy()
            )
            self.env.viser_visualizer.update_point_cloud(
                point_cloud_type="rendered_full_points",
                point_cloud=obs['full_pcd_t'][env_id].cpu().numpy()
            )
            self.env.viser_visualizer.update_point_cloud(
                point_cloud_type="rendered_cam_points",
                point_cloud=sim_depth_pcd[env_id].cpu().numpy()
            )
            self.env.viser_visualizer.update_point_cloud(
                point_cloud_type="rendered_lidar_points",
                point_cloud=sim_lidar_pcd[env_id].cpu().numpy()
            )
            self.env.viser_visualizer.update_point_cloud(
                point_cloud_type="policy_input_points",
                point_cloud=obs["local_pcd_t"][env_id].cpu().numpy()
            )

        obs_student = OrderedDict()

        if "local_pcd_t" in self.pcd_encoders_keys:
            obs_student['local_pcd_t'] = obs['local_pcd_t']

        # convert all pcd to franka base frame
        profile_start = self._profile_start()
        for key in obs_student.keys():
            if "pcd" in key:
                pcd_shifted = obs_student[key] - franka_base_pos.unsqueeze(1) # (num_envs, N, 3)
                pcd_base_frame = torch.bmm(pcd_shifted, rot_global2base) # (num_envs, N, 3), bmm is like matmul but specifically made for batches of 2D matrices, faster than matmul
                obs_student[key] = pcd_base_frame
        self._profile_end(profile_stats, "preprocess/pcd_to_base", profile_start)

        return obs_student, wandb_logs

    # CODEX: aux feedback is stored in world frame, then re-expressed in the current base frame.
    def _world_points_to_base_frame(self, points_world, base_pose7):
        base_pos = base_pose7[:, :3]
        base_quat = base_pose7[:, 3:]
        base_rot_mat = quaternion_to_matrix_ig(base_quat)
        rot_global2base = base_rot_mat.transpose(1, 2)
        point_shifted = (points_world - base_pos).unsqueeze(1)
        point_base_frame = torch.bmm(point_shifted, rot_global2base)
        return point_base_frame[:, 0, :]

    def _base_points_to_world_frame(self, points_base, base_pose7):
        base_pos = base_pose7[:, :3]
        base_quat = base_pose7[:, 3:]
        base_rot_mat = quaternion_to_matrix_ig(base_quat)
        return torch.bmm(points_base.unsqueeze(1), base_rot_mat)[:, 0, :] + base_pos

    def _get_object_center_pos_in_base_frame(self, use_initial_frame=False, base_pose7=None):
        if use_initial_frame:
            object_center_pos = self.env._object_center_init_state.clone()
        else:
            object_center_pos = self.env.states["object_center_pos"].clone()
        if base_pose7 is None:
            base_pose7 = self.env.states['franka_base_pose7']
        return self._world_points_to_base_frame(object_center_pos, base_pose7)

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

    def _refresh_aux_anchor_state(self, add_noise, use_initial_frame=False):
        if "aux_object_state" not in self.state_encoders_keys:
            return
        fallback_aux_base = self._get_object_center_pos_in_base_frame(
            use_initial_frame=use_initial_frame,
            base_pose7=self.aux_anchor_base_pose7,
        )
        if add_noise:
            fallback_aux_base = fallback_aux_base + 0.1 * (
                torch.rand(self.env.num_envs, 3, device=self.device) - 0.5
            )

        if not self._use_aux_feedback():
            self.aux_anchor_state[:] = fallback_aux_base
        else:
            self.aux_anchor_state[:] = self._world_points_to_base_frame(
                self.last_aux_state_from_prev_step_world,
                self.aux_anchor_base_pose7,
            )
            if hasattr(self.env, "object_reset_mask") and torch.any(self.env.object_reset_mask):
                reset_mask = self.env.object_reset_mask
                self.aux_anchor_state[reset_mask] = fallback_aux_base[reset_mask]
                self.last_aux_state_from_prev_step_world[reset_mask] = self._base_points_to_world_frame(
                    fallback_aux_base[reset_mask],
                    self.aux_anchor_base_pose7[reset_mask],
                )
        self.aux_buffer[:] = self.aux_anchor_state.unsqueeze(1)

    def _get_aux_target(self, object_center_pos, prev_abs_aux):
        prev_abs_aux_2d = self._aux_to_2d(prev_abs_aux)
        if self.aux_prediction_mode == "delta":
            target_delta = (object_center_pos - prev_abs_aux_2d) / self.aux_delta_scale
            return torch.clamp(target_delta, -1.0, 1.0)
        return object_center_pos

    def _plot_aux_prediction_in_viewer(self):
        if (not self.has_aux_prediction) or (not hasattr(self.env, "viewer")) or (self.env.viewer is None):
            return
        if (not hasattr(self.env, "draw_box_lines")) or (not hasattr(self.env, "gym")):
            return

        franka_base_pos = self.env.states['franka_base_pose7'][:, :3]
        franka_base_quat = self.env.states['franka_base_pose7'][:, 3:]
        franka_base_rot_mat = quaternion_to_matrix_ig(franka_base_quat)

        # CODEX: draw the latest aux prediction in world frame; aux_buffer itself is tied to its anchor base frame.
        aux_world = self.last_aux_state_from_prev_step_world.clone()

        aux_dims = self.env.mesh_aabb_extents
        aux_box_pos = aux_world.clone()
        aux_box_pos[:, 2] -= 0.5 * aux_dims[:, 2]

        self.env.gym.clear_lines(self.env.viewer)

        if hasattr(self.env, "box_pos") and hasattr(self.env, "box_quats") and hasattr(self.env, "box_dims"):
            for env_idx in range(self.env.num_envs):
                self.env.draw_box_lines(
                    env_idx,
                    self.env.box_pos[env_idx].clone(),
                    self.env.box_quats[env_idx].clone(),
                    self.env.box_dims[env_idx].clone(),
                )

        for env_idx in range(self.env.num_envs):
            self.env.draw_box_lines(
                env_idx,
                aux_box_pos[env_idx].clone(),
                (0.0, 0.0, 0.0, 1.0),
                aux_dims[env_idx].clone(),
                color=(0.2, 1.0, 0.2),
            )

    def train_episode(self):
        count_reaching = torch.zeros(self.env.num_envs, device=self.device).int()
        episode_profile_stats = {}

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
            step_start = self._profile_start()

            # get obs t_a0 for student, q_hand, rel_pcd
            profile_start = self._profile_start()
            q_robot = self.env.states['q'].clone() # (num_envs, 32)
            self.aux_anchor_base_pose7[:] = self.env.states['franka_base_pose7'].clone()  # CODEX
            self._refresh_aux_anchor_state(add_noise=True, use_initial_frame=self.aux_init_only)  # CODEX

            if self.env.sim_steps == 0:
                self.env.abs_actions[:] = q_robot

            full_scene_pcd_t = self.env.combined_pcds # scene pcd + object pcd
            robot_pcd_t = self.env.robot_pcd_sampler.sample(q_robot, self.env.torchurdf_to_isaac_idx)

            # prepare pcd inputs
            obs_dict_a0 = OrderedDict([
                ("full_scene_pcd_t", full_scene_pcd_t),
                ("robot_pcd_t", robot_pcd_t),
            ])
            self._profile_end(episode_profile_stats, "train/obs_collection", profile_start)
            obs_input_a0, input_wandb_logs = self.preprocess_inputs(obs_dict_a0, profile_stats=episode_profile_stats)

            # prepare state inputs
            profile_start = self._profile_start()
            q_arm_manip = self.env.states['q'][:, 3:10].clone() # (num_envs, 7)
            q_arm_vision = self.env.states['q'][:, 26:].clone() # (num_envs, 6)
            q_hand = self.env.states['q'][:, 10:26].clone() # (num_envs, 16)

            obs_input_a0["q_arm_manip"] = self.env.normalize_robot_joints(q_arm_manip, robot="franka", delta=False)
            obs_input_a0["q_arm_vision"] = self.env.normalize_robot_joints(q_arm_vision, robot="arx", delta=False)
            obs_input_a0["q_hand"] = self.env.normalize_robot_joints(q_hand, robot="leap", delta=False)
            if "q_hand_ctrl_delta" in self.state_encoders_keys:
                obs_input_a0["q_hand_ctrl_delta"] = self.env.normalize_robot_joints(q_hand - self.env.abs_actions[:, 10:26], robot="leap", delta=True)
            if "action_history" in self.state_encoders_keys:
                obs_input_a0["action_history"] = self.env.action_history_buf.clone()

            if "aux_object_state" in self.state_encoders_keys:
                # CODEX: aux scalar input and aux local crop share the same current-base-frame anchor.
                obs_input_a0["aux_object_state"] = self.aux_anchor_state.clone()
                if hasattr(self.env, "object_reset_mask"):
                    self.env.object_reset_mask[:] = False
            self._profile_end(episode_profile_stats, "train/state_inputs", profile_start)

            profile_start = self._profile_start()
            with torch.no_grad():
                student_model = self.student_model.module if self.multi_gpu else self.student_model
                student_model.eval()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    output = student_model(obs_input_a0)
                student_actions_chunk = output["action"].float()
                if self.has_aux_prediction:
                    decoded_aux = self._decode_aux_prediction(output["aux"], obs_input_a0["aux_object_state"])  # CODEX
                    self.aux_buffer[:] = decoded_aux[:, :1, :]  # CODEX
                    self.last_aux_state_from_prev_step_world[:] = self._base_points_to_world_frame(  # CODEX
                        self.aux_buffer[:, 0, :],
                        self.aux_anchor_base_pose7,
                    )
            self._profile_end(episode_profile_stats, "train/student_inference", profile_start)

            teacher_preds_buffer = []
            aux_ref_state = self.aux_anchor_state.clone() if self.has_aux_prediction else None  # CODEX

            for action_idx in range(self.chunk_size):
                # get teacher action
                teacher_obs = self.env.obs_buf.clone()
                batch_dict = {
                    "is_train": False,
                    "obs": teacher_obs,
                    "prev_actions": None,
                }

                is_deterministic = True

                profile_start = self._profile_start()
                with torch.no_grad():
                    res_dict = self.teacher_model(batch_dict)
                self._profile_end(episode_profile_stats, "train/teacher_inference", profile_start)

                profile_start = self._profile_start()
                mu = res_dict['mus']
                action = res_dict['actions']
                self.states = res_dict['rnn_states']
                if is_deterministic:
                    teacher_actions = mu
                else:
                    teacher_actions = action
                teacher_actions = torch.clamp(teacher_actions, -self.env.clip_actions, self.env.clip_actions)
                self.env._pre_physics_step_teacher(teacher_actions)

                self._profile_end(episode_profile_stats, "train/teacher_fabric", profile_start)

                profile_start = self._profile_start()

                teacher_actions = self.env.teacher_actions_converted.clone()
                if self.has_aux_prediction:
                    object_center_pos = self._get_object_center_pos_in_base_frame(  # CODEX
                        base_pose7=self.aux_anchor_base_pose7,
                    )
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
                self._profile_end(episode_profile_stats, "train/teacher_postprocess", profile_start)

                # sync distillation steps for wandb video logging
                self.env.distillation_steps = self.total_steps
                profile_start = self._profile_start()
                self.env.step(step_actions)
                self._profile_end(episode_profile_stats, "train/env_step", profile_start)

                # count continuous reaching success
                count_reaching += self.env.success_5cm_per_step
                count_reaching *= self.env.success_5cm_per_step

            profile_start = self._profile_start()
            teacher_preds_buffer = torch.stack(teacher_preds_buffer, dim=1) # (num_envs, chunk_size, action_dim)
            self._profile_end(episode_profile_stats, "train/teacher_stack", profile_start)

            self.student_model.train()
            n_batches = self.env.num_envs // self.batch_size # now this is 1
            indices = torch.randperm(self.env.num_envs, device=self.device)
            ave_loss = {
                "action": 0.0,
                "aux": 0.0,
                "total": 0.0,
            }
            profile_start = self._profile_start()
            for _ in range(self.grad_updates_per_step):
                for i in range(n_batches):
                    batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
                    batch_obs = {k: v[batch_indices] for k, v in obs_input_a0.items()}
                    batch_actions = teacher_preds_buffer[batch_indices]
                    # NOTE: supervise student model on first step
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                        loss = self.student_model.forward(batch_obs, batch_actions)
                    self.optimizer.zero_grad()
                    loss["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=self.max_grad_norm) 
                    self.optimizer.step()

                    for key in loss.keys():
                        ave_loss[key] += loss[key].item()
            self._profile_end(episode_profile_stats, "train/optimization", profile_start)

            for key in ave_loss.keys():
                ave_loss[key] /= n_batches

            if self.scheduler is not None:
                profile_start = self._profile_start()
                self.scheduler.step()
                self._profile_end(episode_profile_stats, "train/scheduler", profile_start)

            mem_allocated_GB = float(torch.cuda.memory_allocated(self.device) / 1024**3)
            mem_reserved_GB = float(torch.cuda.memory_reserved(self.device) / 1024**3)

            if self.use_wandb:
                profile_start = self._profile_start()
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
                self._profile_end(episode_profile_stats, "train/wandb_log", profile_start)

            self._profile_end(episode_profile_stats, "train/step_total", step_start)

        return ave_loss, episode_profile_stats

    def eval(self, policy_source="student"):
        if policy_source not in ("student", "teacher"):
            raise ValueError(f"policy_source must be 'student' or 'teacher', got {policy_source}")
        self.env.reset_idx()
        self.env.compute_observations()
        self.env.abs_actions[:] = self.env.states['q'].clone()

        for _ in tqdm(range(self.steps_per_episode), desc="Evaluating", \
            ncols=None, dynamic_ncols=True, disable=(self.multi_gpu and self.global_rank != 0) ):

            if policy_source == "teacher":
                # CODEX: teacher-only eval should not depend on student inputs or student forward.
                for _ in range(self.chunk_size):
                    step_actions = self._get_teacher_step_actions()
                    step_actions = torch.clamp(step_actions, -self.env.clip_actions, self.env.clip_actions)
                    self.env.progress_buf[:] = 0
                    self.env.step(step_actions)
                continue

            # get obs t_a0 for student, q_hand, rel_pcd
            q_robot = self.env.states['q'].clone() # (num_envs, 32)
            self.aux_anchor_base_pose7[:] = self.env.states['franka_base_pose7'].clone()  # CODEX
            self._refresh_aux_anchor_state(add_noise=False, use_initial_frame=self.aux_init_only)  # CODEX

            full_scene_pcd_t = self.env.combined_pcds # scene pcd + object pcd
            robot_pcd_t = self.env.robot_pcd_sampler.sample(q_robot, self.env.torchurdf_to_isaac_idx)

            # prepare pcd inputs
            obs_dict_a0 = OrderedDict([
                ("full_scene_pcd_t", full_scene_pcd_t),
                ("robot_pcd_t", robot_pcd_t),
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
            if "action_history" in self.state_encoders_keys:
                obs_input_a0["action_history"] = self.env.action_history_buf.clone()

            if "aux_object_state" in self.state_encoders_keys:
                # CODEX: eval uses the same frame-consistent aux anchor, without training noise.
                obs_input_a0["aux_object_state"] = self.aux_anchor_state.clone()
                if hasattr(self.env, "object_reset_mask"):
                    self.env.object_reset_mask[:] = False

            with torch.no_grad():
                student_model = self.student_model.module if self.multi_gpu else self.student_model
                student_model.eval()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    output = student_model(obs_input_a0)
                student_actions_chunk = output["action"].float()
                if self.has_aux_prediction:
                    decoded_aux = self._decode_aux_prediction(output["aux"], obs_input_a0["aux_object_state"])  # CODEX
                    self.aux_buffer[:] = decoded_aux[:, :1, :]  # CODEX
                    self.last_aux_state_from_prev_step_world[:] = self._base_points_to_world_frame(  # CODEX
                        self.aux_buffer[:, 0, :],
                        self.aux_anchor_base_pose7,
                    )

            # plot aux prediction in gui
            if not self.env.headless:
                self._plot_aux_prediction_in_viewer()

            for action_idx in range(self.chunk_size):
                student_actions = student_actions_chunk[:, action_idx, :]
                step_actions = student_actions
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

        if policy_source == "student":
            eval_wandb_logs = {
                "metrics/eval_success_rate_5cm_final_step": self.env.extras["metrics/success_rate_5cm_per_step"],
                "metrics/eval_success_rate_5cm_per_ep": self.env.extras["metrics/success_rate_5cm_per_ep"],
                "metrics/eval_lifting_rate_5cm_final_step": self.env.extras["metrics/lifting_rate_5cm_per_step"],
                "metrics/eval_lifting_rate_5cm_per_ep": self.env.extras["metrics/lifting_rate_5cm_per_ep"],
            }
        else:
            eval_wandb_logs = {
                "teacher_eval/success_rate_5cm_final_step": self.env.extras["metrics/success_rate_5cm_per_step"],
                "teacher_eval/success_rate_5cm_per_ep": self.env.extras["metrics/success_rate_5cm_per_ep"],
                "teacher_eval/lifting_rate_5cm_final_step": self.env.extras["metrics/lifting_rate_5cm_per_step"],
                "teacher_eval/lifting_rate_5cm_per_ep": self.env.extras["metrics/lifting_rate_5cm_per_ep"],
            }
            for key, value in self.env.extras.items():
                if key.startswith("metrics/verified_bank_"):
                    eval_wandb_logs[f"teacher_eval/{key[len('metrics/') :]}"] = value  # CODEX
            if (not self.multi_gpu) or (self.global_rank == 0):
                print(
                    "[teacher_eval] "
                    f"episode={self.episode} "
                    f"success_rate_5cm_per_ep={eval_wandb_logs['teacher_eval/success_rate_5cm_per_ep']:.4f} "
                    f"lifting_rate_5cm_per_ep={eval_wandb_logs['teacher_eval/lifting_rate_5cm_per_ep']:.4f}"
                )

        return eval_wandb_logs

    def train(self):
        if self.teacher_eval_only:
            while True:
                eval_wandb_logs = self.eval(policy_source="teacher")  # CODEX
                self.total_steps += int(self.env.max_episode_length)
                if ((not self.multi_gpu) or (self.global_rank == 0)) and self.use_wandb:
                    wandb.log(eval_wandb_logs, step=self.total_steps)
                self.episode += 1
            return

        while self.episode < self.total_episodes:
            metrics = {}

            start_time = time.time()
            remaining_episodes = self.total_episodes - self.episode

            train_loss, train_profile_stats = self.train_episode()

            eval_policy = (self.eval_freq > 0) and (self.episode % self.eval_freq == 0) and (self.episode > 0) # skip eval at episode 0
            if eval_policy:
                eval_wandb_logs = self.eval()

            if (not self.multi_gpu) or (self.global_rank == 0):
                episode_time = time.time() - start_time
                estimated_finish_time = start_time + episode_time * remaining_episodes

                metrics["train/loss_episode"] = train_loss
                metrics["time/episode_time"] = episode_time
                metrics["episode"] = self.episode
                metrics["train/teacher_forcing_prop"] = self.teacher_forcing_prop
                if self.profile_timing and train_profile_stats:
                    avg_step_time = train_profile_stats.get("train/step_total", 0.0) / max(self.steps_per_episode, 1)
                    ranked_profile_items = sorted(
                        [(key, value) for key, value in train_profile_stats.items() if key != "train/step_total"],
                        key=lambda item: item[1],
                        reverse=True,
                    )
                    if ranked_profile_items:
                        top_name, top_value = ranked_profile_items[0]
                        metrics["profile/train_top_section_seconds"] = top_value
                        metrics["profile/train_top_section_pct"] = 100.0 * top_value / max(train_profile_stats.get("train/step_total", 1e-8), 1e-8)
                        metrics["profile/train_avg_step_seconds"] = avg_step_time
                metrics.update(self.env.extras)
                if eval_policy:
                    metrics.update(eval_wandb_logs)

                if self.use_wandb:
                    wandb.log(metrics, step=self.total_steps)

                self.save_checkpoint(self.episode, metrics["metrics/success_rate_5cm_per_ep"], metrics.get("metrics/eval_lifting_rate_5cm_per_ep", None))

                colorprint(f"Episode {self.episode + 1}/{self.total_episodes} completed in {timedelta(seconds=int(episode_time))}", color="magenta")
                for metric, value in metrics.items():
                    if type(value) == float:
                        colorprint(f"{metric}: {value:.4f}", color="green")
                if self.profile_timing and train_profile_stats and ((self.episode % self.profile_print_freq) == 0):
                    colorprint(
                        f"Timing profile: {self._summarize_profile_stats(train_profile_stats, total_key='train/step_total')}",
                        color="cyan",
                    )
                colorprint(f"Average episodes per hour: {1/episode_time*3600:.2f}")
                colorprint(f"Estimated completion: {datetime.fromtimestamp(estimated_finish_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print("\n")

            self.episode += 1
