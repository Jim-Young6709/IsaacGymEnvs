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
from isaacgymenvs.utils.simulate_depth_cam import simulate_depth_cam_render_from_pose

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import yaml
import os
import time
import numbers

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder
import wandb

from typing import Dict
from pathlib import Path
from isaacgymenvs.tasks import FrankaLEAPMobileDistillation


class DaggerMobile:
    def _validate_teacher_obs_dim(self, stage):
        self.env.compute_observations()
        obs_shape = tuple(self.env.obs_buf.shape)
        actual_obs_dim = int(obs_shape[-1])
        configured_obs_dim = int(self.env.num_observations)

        teacher_rms_dim = None
        if hasattr(self.teacher_model, "running_mean_std") and hasattr(self.teacher_model.running_mean_std, "running_mean"):
            teacher_rms_dim = int(self.teacher_model.running_mean_std.running_mean.numel())

        if actual_obs_dim != configured_obs_dim or (
            teacher_rms_dim is not None and actual_obs_dim != teacher_rms_dim
        ):
            raise RuntimeError(
                f"[teacher_obs_dim_mismatch:{stage}] "
                f"task={self.cfg.task.name} "
                f"env_class={type(self.env).__module__}.{type(self.env).__name__} "
                f"obs_buf_shape={obs_shape} "
                f"actual_obs_dim={actual_obs_dim} "
                f"configured_num_observations={configured_obs_dim} "
                f"teacher_rms_dim={teacher_rms_dim}"
            )

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

            cfg.sim_device = f"cuda:{self.local_rank}"
            cfg.rl_device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.local_rank)

            if self.local_rank == 0:
                cfg.graphics_device_id = self.local_rank
            else:
                cfg.task.env.video_logging.capture = False # note the actual video logging flag is in task env, not in general cfg.capture_video
                cfg.graphics_device_id = -1
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1

        torch_current_device = "cpu"
        if torch.cuda.is_available():
            torch_current_device = f"cuda:{torch.cuda.current_device()}"
        print(
            "[DaggerMobile/startup] "
            f"global_rank={self.global_rank} "
            f"local_rank={self.local_rank} "
            f"world_size={self.world_size} "
            f"sim_device={cfg.sim_device} "
            f"rl_device={cfg.rl_device} "
            f"graphics_device_id={cfg.graphics_device_id} "
            f"torch_current_device={torch_current_device}"
        )

        self.cfg = cfg
        self.total_episodes = cfg.dagger.total_episodes
        self.steps_per_episode = int(cfg.dagger.steps_per_episode / cfg.chunk_size)
        self.warmup_episodes = cfg.dagger.warmup_episodes
        self.max_grad_norm = cfg.dagger.max_grad_norm
        self.local_pcd_range = cfg.dagger.local_pcd_range
        self.reaching_reset_threshold = cfg.dagger.reaching_reset_threshold
        self.teacher_forcing_cfg = cfg.dagger.teacher_forcing
        self.teacher_state_bank_cfg = cfg.dagger.teacher_state_bank
        self.teacher_state_bank_collect_before_train = bool(self.teacher_state_bank_cfg["collect_before_train"])
        self.teacher_state_bank_collect_only = bool(self.teacher_state_bank_cfg["collect_only"])
        max_success_episodes_per_category_cfg = self.teacher_state_bank_cfg.get(
            "max_success_episodes_per_category",
            self.teacher_state_bank_cfg.get("max_done_episodes_per_category", 50000),
        )
        self.teacher_state_bank_max_success_episodes_per_category = int(
            max_success_episodes_per_category_cfg
        )
        self.teacher_state_bank_save_each_category = bool(self.teacher_state_bank_cfg["save_each_category"])
        self.teacher_state_bank_save_interval_seconds = float(
            self.teacher_state_bank_cfg.get("save_interval_seconds", 300.0)
        )
        self.chunk_size = cfg.chunk_size
        self.device = cfg['sim_device']
        self.seed = cfg.seed
        self.delayed_update_steps = int(cfg.dagger["delayed_update_steps"])
        self.update_epochs_per_rollout = int(cfg.dagger["update_epochs_per_rollout"])
        self.disable_updates_when_debug_viz = bool(cfg.dagger.get("disable_updates_when_debug_viz", True))
        self.updates_enabled = not (
            bool(cfg.task.env.get("enableDebugVis", False)) and self.disable_updates_when_debug_viz
        )
        self.rank_seed = int(self.seed) + (int(self.global_rank) if self.multi_gpu else 0)
        base_exp_name = str(cfg.experiment) if str(cfg.experiment) != "" else str(cfg.train.params.config.name)
        if self.multi_gpu:
            # ensure all ranks use exactly one shared run name/checkpoint dir.
            shared_time_suffix = ""
            if self.global_rank == 0:
                shared_time_suffix = "{date:%d-%H-%M-%S}".format(date=datetime.now())
            shared_name_list = [shared_time_suffix]
            dist.broadcast_object_list(shared_name_list, src=0)
            self.exp_name = base_exp_name + "_" + shared_name_list[0]
        else:
            self.exp_name = base_exp_name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now())
        set_seed_and_precision(self.seed, (int(self.global_rank) if self.multi_gpu else 0))

        self.learning_rate = cfg.dagger.learning_rate
        self.weight_decay = cfg.dagger.weight_decay
        # load env
        run_name = self.exp_name  # keep local video run folder consistent with experiment+timestamp naming.
        def create_isaacgym_env(**kwargs) -> FrankaLEAPMobileDistillation:
            envs = isaacgymenvs.make(
                self.rank_seed,
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
        if getattr(self.env, "_verified_teacher_bank_enable", False):
            loaded_verified_bank = self.env.load_verified_teacher_bank_hdf5(rank=self.global_rank, strict=False)
            if (not self.multi_gpu) or (self.global_rank == 0):
                print(
                    "[DaggerMobile/verified_teacher_bank] "
                    f"enable={self.env._verified_teacher_bank_enable} "
                    f"loaded={loaded_verified_bank} "
                    f"path={self.env._get_verified_teacher_bank_shard_path(rank=self.global_rank)}"
                )
                variation_json_cfg = self.cfg.task.env.scene.get("teacher_bank_variation_json", None)
                legacy_height_json_cfg = self.cfg.task.env.scene.get("teacher_bank_height_assignment_json", None)
                if (
                    self.teacher_state_bank_collect_before_train
                    and int(self.cfg.task.env.scene["teacher_bank_height_bins"]) <= 0
                    and not variation_json_cfg
                    and not legacy_height_json_cfg
                ):
                    print(
                        "[DaggerMobile/verified_teacher_bank] "
                        "collection is enabled but no deterministic variation assignment is configured. "
                        "Set task.env.scene.teacher_bank_variation_json or use teacher_bank_height_bins > 0."
                    )
            if loaded_verified_bank:
                all_env_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
                self.env.reset_idx(all_env_ids)
                self.env.compute_observations()

        # env cfg overrides
        self.env.teleport_boundary_chunk_size = int(self.chunk_size)
        self.env.distillation_mode = True
        self.env.delta_franka_action = self.cfg.action_space.delta_franka_action
        self.env.delta_leap_action = self.cfg.action_space.delta_leap_action
        self.env.delta_arx_action = self.cfg.action_space.delta_arx_action
        self.student_action_dim = int(self.env.teacher_actions_converted.shape[1]) # @ray need to update student action dim in runtime
        # @ray we override success/lifting with early termination logic
        reset_window_steps = int(self.reaching_reset_threshold)
        self.env.reward_settings["success_timeout_steps"] = torch.full(
            (self.env.num_envs,), reset_window_steps, dtype=torch.int32, device=self.device
        )
        self.env.reward_settings["lifting_timeout_steps"] = torch.full(
            (self.env.num_envs,), reset_window_steps, dtype=torch.int32, device=self.device
        )

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
        self._validate_teacher_obs_dim(stage="post_teacher_load")

        # load student network
        if "q_hand_ctrl_delta" in cfg.model.state_encoders_cfg:
            cfg.model.state_encoders_cfg.q_hand_ctrl_delta.input_dim = 16 # @ray used to compute the difference between predicted q_hand and actual q_hand from proprioception
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
        if (not self.multi_gpu) or (self.global_rank == 0):
            print(
                "[DaggerMobile/train_mode] "
                f"enableDebugVis={bool(cfg.task.env.get('enableDebugVis', False))} "
                f"disable_updates_when_debug_viz={self.disable_updates_when_debug_viz} "
                f"updates_enabled={self.updates_enabled}"
            )

        # dagger
        self.episode = 0
        self.total_steps = 0
        self.batch_idx = 0
        self.batch_size = self.cfg.dagger.batch_size

        self.use_wandb = self.cfg.wandb_activate
        self.wandb_project = self.cfg.wandb_project
        self.wandb_name = self.exp_name  # enforce display name consistency (experiment + timestamp).
        self.wandb_id = None

        self.state_encoders_keys = self.cfg.model.state_encoders_cfg.keys()
        self.pcd_encoders_keys = self.cfg.model.pcd_encoders_cfg.keys()

        self.save_dir = Path("dagger_ckpts") / self.exp_name
        self.save_freq = self.cfg.dagger.save_freq
        os.makedirs(self.save_dir, exist_ok=True)
        if (not self.multi_gpu) or (self.global_rank == 0):
            colorprint(f"Checkpoint dir: {self.save_dir.resolve()}", color="magenta")
        # TEMP DEBUG DISABLED:
        self.eval_freq = self.cfg.dagger.eval_freq
        self.eval_only_teacher = bool(self.cfg.dagger.get("eval_only_teacher", False))
        self.debug_teacher_eval = bool(self.cfg.dagger.get("debug_teacher_eval", False))
        self.teacher_eval_only = bool(self.cfg.dagger.get("teacher_eval_only", False))
        self.latest_eval_wandb_logs = {}
        self.last_eval_episode = -1

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

        # action
        # @ray Following UMI, for delta actions, in an action chunk of length n, each delta action is based on the state when the model generates the chunk, not based on the previous action in the chunk.
        # The state when the model generates the chunk is called the anchor state. We rollout actions open loop, so every nth state is the anchor state
        # For observation, we additionally pass in the difference between the predicted joint state and the proprioception of the anchor state as implicit force feedback
        self.last_student_actions = torch.zeros( # @ray normalized (delta) actions
            (self.env.num_envs, self.student_action_dim), device=self.device
        )
        self.action_chunk_anchor_q = torch.zeros( # @ray unnormalized joint positions at the anchor state
            (self.env.num_envs, 32), device=self.device
        )
        self.chunk_anchor_base_pose7 = torch.zeros( # @ray base pose at the anchor state, used to transform base actions and for aux prediction frame conversions
            (self.env.num_envs, 7), device=self.device
        )

        # aux
        # @ray Again using UMI's style, student also predicts open loop aux states for every step in the chunk
        # For delta aux prediction, all deltas in the action chunk is based on the prediction at the anchor state
        # Model outputs aux predictions in the current base frame, so the anchor state -> world frame to be decoded back to current base frame for each step in the chunk
        # For observation, the last aux prediction in the previous chunk is converted to current anchor base frame as input
        self.has_aux_input = "aux_object_state" in self.state_encoders_keys
        self.aux_prediction_mode = str(self.cfg.model["aux_prediction_mode"]).lower()
        self.aux_delta_scale = float(self.cfg.model["aux_delta_scale"])
        if self.aux_prediction_mode not in ["absolute", "delta"]:
            raise ValueError(f"aux_prediction_mode must be 'absolute' or 'delta', got {self.aux_prediction_mode}")
        self.aux_feedback_to_policy = bool(self.cfg.dagger["aux_feedback_to_policy"]) # @ray use teacher forcing or not
        self.aux_init_only = bool(self.cfg.dagger["aux_init_only"]) and (not self.aux_feedback_to_policy) # @ray use only aux at the start or teacher forcing 
        self.aux_switch_steps = int(self.cfg.dagger["aux_feedback_start_steps"]) # @ray steps to switch from teacher forcing back to autoregressive
        self.aux_anchor_state = torch.zeros(self.env.num_envs, 3, device=self.device)  # @ray absolute aux prediction at the current robot base frame
        self.last_aux_state_from_prev_chunk_world = torch.zeros(self.env.num_envs, 3, device=self.device) # @ray absolute aux prediction in world frame, equivalent to anchor state in the world frame (last aux state from prev chunk is the anchor state for a new chunk)

    def _sanitize_wandb_logs(self, logs: Dict) -> Dict:
        out = {}
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    out[k] = float(v.detach().item())
            elif isinstance(v, np.ndarray):
                if v.size == 1:
                    out[k] = float(v.reshape(-1)[0])
            elif isinstance(v, numbers.Number):
                out[k] = float(v)
        return out

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
        wandb_logs = {}

        # franka base states
        franka_base_pos = self.env.states['franka_base_pose7'][:, :3] # (num_envs, 3)
        franka_base_quat = self.env.states['franka_base_pose7'][:, 3:] # (num_envs, 4)
        franka_base_rot_mat = quaternion_to_matrix_ig(franka_base_quat)
        rot_global2base = franka_base_rot_mat.transpose(1, 2) # (num_envs, 3, 3)

        obs['full_pcd_t'] = torch.cat([obs["full_scene_pcd_t"], obs["robot_pcd_t"]], dim=1)

        if self.env.pcd_spec_dict['simulate_depth_cam']:
            # based on scene pointcloud, simulate a partial pointcloud from the robot camera
            num_full_pcd_points = self.env.pcd_spec_dict['num_static_points'] + \
                                  self.env.pcd_spec_dict['num_robot_points'] + \
                                  self.env.pcd_spec_dict['num_object_points']

            camera_pose7 = self.env.states['camera_pose7'].clone() # (num_envs, 7)
            sim_depth_pcd, sim_depth_render_logs = simulate_depth_cam_render_from_pose(
                pcd=obs['full_pcd_t'],
                camera_pose=camera_pose7,
                num_points=num_full_pcd_points,
            )

            if self.use_wandb:
                wandb_logs.update(sim_depth_render_logs)

            obs['full_pcd_t'] = sim_depth_pcd

        # Viser debug utils
        # env_id = self.env.viser_visualizer.env_id
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="rendered_points",
        #     point_cloud=obs['full_pcd_t'][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="hand_pcd_t",
        #     point_cloud=obs["hand_pcd_t"][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="seg_static_obsacles_t0",
        #     point_cloud=obs["static_scene_pcd_t0"][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="seg_static_object_t0",
        #     point_cloud=obs["object_pcd_t0"][env_id].cpu().numpy()
        # )

        if "local_pcd_t" in self.pcd_encoders_keys:
            # sample local pointcloud around the eef and (optionally object)
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

            # aux-centered local pcd in global frame
            aux_crop_origin = eef_pos
            if "aux_object_state" in self.state_encoders_keys:
                aux_crop_origin = self._base_points_to_world_frame(
                    self.aux_anchor_state.clone(),
                    self.env.states["franka_base_pose7"].clone(),
                )
                if aux_crop_origin.ndim == 3:
                    aux_crop_origin = aux_crop_origin[:, 0, :]
            aux_full_pcd_shifted = obs['full_pcd_t'] - aux_crop_origin.unsqueeze(1) # (num_envs, N, 3)
            aux_spherical_local_pcd_t, aux_spherical_crop_logs = crop_local_pcd(aux_full_pcd_shifted, local_aux_spherical_range, num_points[2], is_cylindrical=False) # (num_envs, num_local_points, 3)
            obs["local_aux_pcd_t"] = aux_spherical_local_pcd_t + aux_crop_origin.unsqueeze(1)

        # env_id = self.env.viser_visualizer.env_id
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="rendered_points",
        #     point_cloud=obs['full_pcd_t'][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="hand_pcd_t",
        #     point_cloud=obs["local_eef_pcd_t"][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="seg_static_obsacles_t0",
        #     point_cloud=obs["local_aux_pcd_t"][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="obj_point_t",
        #     point_cloud=aux_crop_origin[env_id].reshape(1, 3).cpu().numpy()
        # )

        # convert all pcd to franka base frame
        for key in obs.keys():
            if "pcd" in key:
                pcd_shifted = obs[key] - franka_base_pos.unsqueeze(1) # (num_envs, N, 3)
                pcd_base_frame = torch.bmm(pcd_shifted, rot_global2base) # (num_envs, N, 3), bmm is like matmul but specifically made for batches of 2D matrices, faster than matmul
                obs[key] = pcd_base_frame

        # Viser debug utils
        # env_id = self.env.viser_visualizer.env_id
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="rendered_points",
        #     point_cloud=obs['full_pcd_t'][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="hand_pcd_t",
        #     point_cloud=obs["hand_pcd_t"][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="seg_static_obsacles_t0",
        #     point_cloud=obs["static_scene_pcd_t0"][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="seg_static_object_t0",
        #     point_cloud=obs["object_pcd_t0"][env_id].cpu().numpy()
        # )
        # self.env.viser_visualizer.wheel_odom_frame.position = franka_base_pos[env_id].cpu().numpy()
        # self.env.viser_visualizer.wheel_odom_frame.wxyz = franka_base_quat[env_id, [3, 0, 1, 2]].cpu().numpy()

        obs_student = OrderedDict()

        if "full_scene_pcd_t0" in self.pcd_encoders_keys:
            obs["full_scene_pcd_t0"] = torch.cat([obs["static_scene_pcd_t0"], obs["object_pcd_t0"]], dim=1)

        for key in self.pcd_encoders_keys:
            if key in ["static_scene_pcd_t0", "object_pcd_t0", "full_scene_pcd_t0", "full_scene_pcd_t", "robot_pcd_t", "hand_pcd_t"]:
                num_points_key = self.cfg.model.pcd_encoders_cfg[key]["num_points"]
                obs_student[key] = downsample_pcd_batched(obs[key], num_points_key)

        if "full_pcd_t" in self.pcd_encoders_keys:
            num_points_full_pcd_t = self.cfg.model.pcd_encoders_cfg["full_pcd_t"]["num_points"]
            if self.env.pcd_spec_dict['simulate_depth_cam']:
                full_pcd_t = obs["full_pcd_t"][:, :num_points_full_pcd_t]
                # replace nan values as 0s
                full_pcd_t_zero_padding = torch.nan_to_num(full_pcd_t, nan=0.0)
                obs_student["full_pcd_t"] = full_pcd_t_zero_padding
            else:
                obs_student["full_pcd_t"] = downsample_pcd_batched(obs["full_pcd_t"], num_points_full_pcd_t)

        if "local_pcd_t" in self.pcd_encoders_keys:
            num_points = self.cfg.model.pcd_encoders_cfg["local_pcd_t"]["num_points"] # [num cylindrical points, num spherical eef points, num spherical aux points]
            cylindrical_local_pcd_t, cylindrical_crop_logs = crop_local_pcd(obs['full_pcd_t'], self.local_pcd_range[0], num_points[0], is_cylindrical=True) # (num_envs, num_local_points, 3)
            obs_student["local_pcd_t"] = torch.cat([cylindrical_local_pcd_t, obs["local_eef_pcd_t"], obs["local_aux_pcd_t"]], dim=1)

            if self.use_wandb:
                wandb_logs.update(cylindrical_crop_logs)
                wandb_logs.update(eef_spherical_crop_logs)
                wandb_logs.update({
                    "local_spherical_crop_aux/avg_num_valid_points": aux_spherical_crop_logs["local_spherical_crop/avg_num_valid_points"],
                    "local_spherical_crop_aux/min_num_valid_points": aux_spherical_crop_logs["local_spherical_crop/min_num_valid_points"],
                })

        elif "local_scene_pcd_t" in self.pcd_encoders_keys: # TODO: this is kinda outdated
            obs_student["local_scene_pcd_t"], crop_logs = crop_local_pcd(obs["full_scene_pcd_t"], self.local_pcd_range[0], self.cfg.model.pcd_encoders_cfg["local_pcd_t"]["num_points"][0], is_cylindrical=True)
            if self.use_wandb:
                wandb_logs.update(crop_logs)

        # Viser debug utils
        # vis_local_pcd_t = obs_student['local_pcd_t'].clone()
        # vis_local_pcd_t = torch.bmm(vis_local_pcd_t, rot_global2base.transpose(1, 2)) # (num_envs, N, 3), bmm is like matmul but specifically made for batches of 2D matrices, faster than matmul
        # vis_local_pcd_t = vis_local_pcd_t + franka_base_pos.unsqueeze(1) # (num_envs, N, 3)

        # self.env.viser_visualizer.update_point_cloud(
        #     point_cloud_type="local_point_t",
        #     point_cloud=vis_local_pcd_t[env_id].cpu().numpy()
        # )

        return obs_student, wandb_logs
    
    def _world_points_to_base_frame(self, points_world, base_pose7):
        base_pos = base_pose7[:, :3]
        base_quat = base_pose7[:, 3:]
        base_rot_mat = quaternion_to_matrix_ig(base_quat)
        rot_global2base = base_rot_mat.transpose(1, 2)
        shifted = (points_world - base_pos).unsqueeze(1)
        points_base = torch.bmm(shifted, rot_global2base)
        return points_base[:, 0, :]
    
    def _base_points_to_world_frame(self, points_base, base_pose7):
        base_pos = base_pose7[:, :3]
        base_quat = base_pose7[:, 3:]
        base_rot_mat = quaternion_to_matrix_ig(base_quat)
        points_world = torch.bmm(points_base.unsqueeze(1), base_rot_mat)[:, 0, :] + base_pos
        return points_world
    
    def _get_aux_object_pos_in_base_frame(self, base_pose7=None):
        object_pos = self.env.states["object_center_pos"].clone()
        if base_pose7 is None:
            base_pose7 = self.env.states['franka_base_pose7']
        return self._world_points_to_base_frame(object_pos, base_pose7)

    def _expand_aux_to_chunk(self, aux_tensor):
        if aux_tensor.ndim == 2:
            return aux_tensor.unsqueeze(1).expand(-1, self.chunk_size, -1)
        if aux_tensor.ndim == 3 and aux_tensor.shape[1] == 1 and self.chunk_size > 1:
            return aux_tensor.expand(-1, self.chunk_size, -1)
        return aux_tensor
    
    def _decode_aux_prediction(self, aux_pred, prev_abs_aux):
        prev_abs_aux_chunk = self._expand_aux_to_chunk(prev_abs_aux)
        if self.aux_prediction_mode == "delta":
            aux_delta = torch.clamp(aux_pred, -1.0, 1.0)
            return prev_abs_aux_chunk + self.aux_delta_scale * aux_delta
        return self._expand_aux_to_chunk(aux_pred)

    def _use_aux_feedback(self):
        return self.has_aux_input and self.aux_feedback_to_policy and (self.total_steps >= self.aux_switch_steps)

    def _refresh_current_aux_anchor_state(self, add_noise: bool):
        if "aux_object_state" not in self.state_encoders_keys:
            return
        object_pos_base = self._get_aux_object_pos_in_base_frame(base_pose7=self.chunk_anchor_base_pose7)
        fallback_aux_base = object_pos_base.clone()
        if add_noise:
            fallback_aux_base = fallback_aux_base + 0.1 * (
                torch.rand(self.env.num_envs, 3, device=self.device) - 0.5
            )

        if not self._use_aux_feedback():
            self.aux_anchor_state[:] = fallback_aux_base
        else:
            self.aux_anchor_state[:] = self._world_points_to_base_frame(
                self.last_aux_state_from_prev_chunk_world,
                self.chunk_anchor_base_pose7,
            )
            if hasattr(self.env, "object_reset_mask") and torch.any(self.env.object_reset_mask):
                reset_mask = self.env.object_reset_mask
                self.aux_anchor_state[reset_mask] = fallback_aux_base[reset_mask]
                self.last_aux_state_from_prev_chunk_world[reset_mask] = self._base_points_to_world_frame(
                    fallback_aux_base[reset_mask],
                    self.chunk_anchor_base_pose7[reset_mask],
                )

    def _get_aux_target(self, object_pos, prev_abs_aux):
        prev_abs_aux_chunk = self._expand_aux_to_chunk(prev_abs_aux)
        object_pos_chunk = self._expand_aux_to_chunk(object_pos)
        if self.aux_prediction_mode == "delta":
            target_delta = (object_pos_chunk - prev_abs_aux_chunk) / self.aux_delta_scale
            return torch.clamp(target_delta, -1.0, 1.0)
        return object_pos_chunk

    def _compute_aux_metrics(self, aux_pred, prev_abs_aux, object_pos):
        aux_pred_abs = self._decode_aux_prediction(aux_pred, prev_abs_aux)
        aux_gt_abs = self._expand_aux_to_chunk(object_pos)
        aux_target = self._get_aux_target(object_pos, prev_abs_aux)
        aux_pred_target = self._expand_aux_to_chunk(aux_pred)
        aux_diff_l2 = torch.linalg.norm(aux_pred_abs - aux_gt_abs, dim=-1).mean()
        aux_loss = torch.mean((aux_pred_target - aux_target) ** 2)
        return {
            "aux_diff_l2": float(aux_diff_l2.item()),
            "aux_loss": float(aux_loss.item()),
        }

    def train_episode(self):
        count_reaching = torch.zeros(self.env.num_envs, device=self.device).int()
        rollout_obs_buffer = []
        rollout_target_buffer = []
        episode_loss_sums = {"action": 0.0, "aux": 0.0, "total": 0.0}
        episode_update_count = 0

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

        for step_idx in tqdm(range(self.steps_per_episode), desc=f"Training {self.episode+1}/{self.total_episodes}", \
            ncols=None, dynamic_ncols=True, disable=(self.multi_gpu and self.global_rank != 0) ):
            self.total_steps += 1

            # get obs t_a0 for student, q_hand, rel_pcd
            q_robot = self.env.states['q'].clone() # (num_envs, 32)
            self.action_chunk_anchor_q[:] = q_robot
            self.chunk_anchor_base_pose7[:] = self.env.states['franka_base_pose7'].clone() # (num_envs, 7)
            self._refresh_current_aux_anchor_state(add_noise=True)

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
                obs_input_a0["aux_object_state"] = self._expand_aux_to_chunk(self.aux_anchor_state)

                if hasattr(self.env, "object_reset_mask"):
                    self.env.object_reset_mask[:] = False

            # Viser debug utils
            # env_id = self.env.viser_visualizer.env_id
            # self.env.viser_visualizer.update_point_cloud(
            #     point_cloud_type="obj_point_t",
            #     point_cloud=obs_input_a0["objxyz_t0"][env_id].reshape(1, 3).cpu().numpy()
            # )

            # @ray sample student model for the next action chunk and aux prediction chunk
            with torch.no_grad():
                student_model = self.student_model.module if self.multi_gpu else self.student_model
                student_model.eval()
                output = student_model(obs_input_a0)
                student_actions_chunk = output["action"]
                train_aux_metrics = None
                if self.has_aux_input:
                    object_pos = self._get_aux_object_pos_in_base_frame()
                    train_aux_metrics = self._compute_aux_metrics(
                        output["aux"],
                        obs_input_a0["aux_object_state"],
                        object_pos,
                    )
                    aux_chunk_abs = self._decode_aux_prediction(output["aux"], obs_input_a0["aux_object_state"])
                    self.last_aux_state_from_prev_chunk_world[:] = self._base_points_to_world_frame(
                        aux_chunk_abs[:, -1, :],
                        self.chunk_anchor_base_pose7,
                    )
            
            teacher_preds_buffer = []
            aux_ref_state = self.aux_anchor_state.clone() if self.has_aux_input else None
            prev_teleport_enable = bool(self.env.object_teleport_args["enable"]) # @ray shouldn't teleport object during an action chunk, temporarily disable teleport during the chunk steps if it's enabled

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

                # Compute teacher absolute targets and converted student-space targets.
                teacher_actions_abs = self.env._pre_physics_step_teacher(teacher_actions)
                teacher_actions_exec = self.env.teacher_actions_converted.clone() # normalized delta actions if delta is being used
                teacher_actions = self.env._encode_abs_targets_to_student_space( # @ray convert teacher actions to UMI action format for loss computation against student
                    teacher_actions_abs,
                    self.action_chunk_anchor_q,
                )
                if "aux_object_state" in self.state_encoders_keys:
                    object_pos = self._get_aux_object_pos_in_base_frame(base_pose7=self.chunk_anchor_base_pose7)
                    if self.aux_prediction_mode == "delta":
                        aux_target = torch.clamp(
                            (object_pos - aux_ref_state) / self.aux_delta_scale, -1.0, 1.0
                        )
                    else:
                        aux_target = object_pos
                    teacher_pred = torch.cat([teacher_actions, aux_target], dim=1) # add aux info, object_xyz_pos
                else:
                    teacher_pred = teacher_actions

                teacher_preds_buffer.append(teacher_pred)

                student_actions = student_actions_chunk[:, action_idx, :]
                student_abs_targets = self.env._decode_student_actions_with_anchor( # from UMI action format to absolute actions
                    student_actions,
                    self.action_chunk_anchor_q
                )
                step_actions = self.env._encode_abs_targets_to_student_space( # from absolute actions to per step delta actions
                    student_abs_targets,
                    self.env.states['q'].clone(),
                )
                if num_teacher_forcing_envs > 0:
                    tf_idx_t = torch.as_tensor(teacher_forcing_env_idx, device=step_actions.device, dtype=torch.long)
                    step_actions[tf_idx_t] = teacher_actions_exec[tf_idx_t]
                step_actions = torch.clamp(step_actions, -self.env.clip_actions, self.env.clip_actions)

                self.env.object_teleport_args["enable"] = prev_teleport_enable and (action_idx == self.chunk_size - 1)

                self.env.progress_buf -= 1 # to avoid automatic resets during the chunk steps, only update progress_buf at the end of the chunk
                if action_idx == self.chunk_size - 1:
                    self.env.progress_buf += self.chunk_size

                    # early reset: reset envs to start config if reached (and stay reached for a while)
                    if (count_reaching >= self.reaching_reset_threshold).any():
                        reached_reset_flags = (count_reaching >= self.reaching_reset_threshold)
                        reset_ids = torch.where(reached_reset_flags)[0]
                        # Finalize episode counters before env post_physics_step reset clears duration/latch states.
                        self.env.finalize_episode_metrics_before_reset(
                            reset_ids,
                            mark_success=True,
                            mark_lifting=True,
                        )
                        self.env.reset_buf[reset_ids] = 1
                        count_reaching[reached_reset_flags] = 0

                # sync distillation steps for wandb video logging
                self.env.distillation_steps = self.total_steps
                self.env.step(step_actions)
                self.last_student_actions[:] = step_actions.detach()


                # count continuous reaching success
                count_reaching += self.env.success_5cm_per_step
                count_reaching *= self.env.success_5cm_per_step

            self.env.object_teleport_args["enable"] = prev_teleport_enable

            teacher_preds_buffer = torch.stack(teacher_preds_buffer, dim=1) # (num_envs, chunk_size, action_dim)
            rollout_obs_buffer.append({k: v.detach().clone() for k, v in obs_input_a0.items()})
            rollout_target_buffer.append(teacher_preds_buffer.detach().clone())

            # @ray delayed updates after each chunk
            should_update = (
                ((step_idx + 1) % self.delayed_update_steps == 0)
                or (step_idx == self.steps_per_episode - 1)
            )

            ave_loss = {"action": 0.0, "aux": 0.0, "total": 0.0}
            num_update_batches = 0
            if should_update:
                if self.updates_enabled:
                    self.student_model.train()
                    rollout_obs = {
                        k: torch.cat([step_obs[k] for step_obs in rollout_obs_buffer], dim=0)
                        for k in rollout_obs_buffer[0].keys()
                    }
                    rollout_actions = torch.cat(rollout_target_buffer, dim=0)
                    n_samples = rollout_actions.shape[0]
                    for _ in range(self.update_epochs_per_rollout):
                        indices = torch.randperm(n_samples, device=self.device)
                        for batch_start in range(0, n_samples, self.batch_size):
                            batch_indices = indices[batch_start:batch_start + self.batch_size]
                            batch_obs = {k: v[batch_indices] for k, v in rollout_obs.items()}
                            batch_actions = rollout_actions[batch_indices]
                            loss = self.student_model.forward(batch_obs, batch_actions, action_chunk_idx=0)
                            self.optimizer.zero_grad()
                            loss["total"].backward()
                            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=self.max_grad_norm)
                            self.optimizer.step()
                            num_update_batches += 1
                            episode_update_count += 1
                            for key in ave_loss.keys():
                                ave_loss[key] += loss[key].item()
                                episode_loss_sums[key] += loss[key].item()

                    for key in ave_loss.keys():
                        ave_loss[key] /= max(num_update_batches, 1)
                rollout_obs_buffer.clear()
                rollout_target_buffer.clear()
                if self.updates_enabled and self.scheduler is not None:
                    self.scheduler.step()

            if self.use_wandb:
                wandb_logs = {
                    "train/loss_total": ave_loss["total"],
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "train/updates_enabled": float(self.updates_enabled),
                }
                wandb_logs.update(self._sanitize_wandb_logs(input_wandb_logs))
                if self.has_aux_input:
                    aux_wandb_logs = {
                        "train/loss_aux": ave_loss["aux"],
                        "train/loss_action": ave_loss["action"],
                        "train/aux_diff_l2": train_aux_metrics["aux_diff_l2"],
                    }
                    wandb_logs.update(aux_wandb_logs)

                train_log_step = int(self.total_steps)
                if wandb.run is not None and wandb.run.step is not None:
                    train_log_step = max(train_log_step, int(wandb.run.step))
                wandb.log(self._sanitize_wandb_logs(wandb_logs), step=train_log_step)

        if episode_update_count == 0:
            return {"action": 0.0, "aux": 0.0, "total": 0.0}
        return {k: v / episode_update_count for k, v in episode_loss_sums.items()}
    
    def eval(
        self,
        policy_source: str = "student",
        capture_video: bool = True,
        restore_training_state: bool = True,
        reset_all_envs_at_start: bool = True,
    ):
        if policy_source not in ("student", "teacher"):
            raise ValueError(f"Unsupported policy_source={policy_source}. Expected 'student' or 'teacher'.")
        metric_prefix = "eval" if policy_source == "student" else "teacher_eval"
        if policy_source == "student":
            student_model = self.student_model.module if self.multi_gpu else self.student_model

        prev_teleport_enable = bool(self.env.object_teleport_args["enable"])
        self.env.object_teleport_args["enable"] = False  # CODEX: disable object teleport during eval rollout.
        # snapshot cumulative episode counters so eval per-episode rates are computed
        # from this eval rollout only (delta), not from historical training totals.
        pre_total_eps = int(self.env.per_object_episode_counts.sum().item()) if hasattr(self.env, "per_object_episode_counts") else None
        pre_total_succ = int(self.env.per_object_success_counts.sum().item()) if hasattr(self.env, "per_object_success_counts") else None
        pre_total_lift = int(self.env.per_object_lifting_counts.sum().item()) if hasattr(self.env, "per_object_lifting_counts") else None
        pre_episode_length_sum = (
            int(self.env._episode_length_sum_total.item())
            if hasattr(self.env, "_episode_length_sum_total")
            else None
        )
        pre_success_episode_length_sum = (
            int(self.env._success_episode_length_sum_total.item())
            if hasattr(self.env, "_success_episode_length_sum_total")
            else None
        )
        pre_failure_episode_length_sum = (
            int(self.env._failure_episode_length_sum_total.item())
            if hasattr(self.env, "_failure_episode_length_sum_total")
            else None
        )
        pre_episode_length_max = (
            int(self.env._episode_length_max_total.item())
            if hasattr(self.env, "_episode_length_max_total")
            else None
        )
        category_eval_enabled = bool(getattr(self.env, "_verified_teacher_bank_loaded", False))
        pre_category_eps = (
            self.env.per_verified_bank_episode_counts.clone()
            if category_eval_enabled and hasattr(self.env, "per_verified_bank_episode_counts")
            else None
        )
        pre_category_succ = (
            self.env.per_verified_bank_success_counts.clone()
            if category_eval_enabled and hasattr(self.env, "per_verified_bank_success_counts")
            else None
        )

        if reset_all_envs_at_start:
            self.env.reset_idx()
            self.env.compute_observations()
            self.env.progress_buf[:] = 0  # CODEX: start eval from the beginning of an episode horizon.
            self.env.reset_buf[:] = 0  # CODEX: clear any pending resets before the first eval rollout window.
        else:
            self.env.compute_observations()
        self.env.abs_actions[:] = self.env.states['q'].clone()
        self.chunk_anchor_base_pose7[:] = self.env.states['franka_base_pose7'].clone()
        if self.has_aux_input:
            noisy_object_pos = self._get_aux_object_pos_in_base_frame()
            noisy_object_pos = noisy_object_pos + 0.1 * (torch.rand(self.env.num_envs, 3, device=self.device) - 0.5)
            self.aux_anchor_state[:] = noisy_object_pos
            self.last_aux_state_from_prev_chunk_world[:] = self._base_points_to_world_frame(
                noisy_object_pos,
                self.chunk_anchor_base_pose7,
            )
        total_eval_sim_steps = int(self.env.max_episode_length)  # CODEX: evaluate for exactly one env episode horizon.
        if capture_video and self.env.video_logging["capture"] and policy_source == "student":
            # hard-reset video mode flags before enabling eval capture to avoid phase leakage.
            self.env.train_video_active = False
            self.env.train_video_step_idx = 0
            self.env.train_video_active = False
            self.env.eval_video_active = True
            self.env.eval_video_total_steps = total_eval_sim_steps
            self.env.eval_video_step_idx = 0

        eval_chunks = (total_eval_sim_steps + self.chunk_size - 1) // self.chunk_size 
        eval_sim_step = 0
        eval_aux_diff_l2_sum = 0.0
        eval_aux_loss_sum = 0.0
        eval_aux_metric_count = 0
        eval_completed_episode_lengths = []
        for _ in tqdm(range(eval_chunks), desc="Evaluating", \
            ncols=None, dynamic_ncols=True, disable=(self.multi_gpu and self.global_rank != 0) ):

            # get obs t_a0 for student, q_hand, rel_pcd
            q_robot = self.env.states['q'].clone() # (num_envs, 32)
            self.action_chunk_anchor_q[:] = q_robot
            self.chunk_anchor_base_pose7[:] = self.env.states['franka_base_pose7'].clone() # (num_envs, 7)
            self._refresh_current_aux_anchor_state(add_noise=False)

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
            q_hand = self.env.states['q'][:, 10:26].clone() # (num_envs, 16)'
            obs_inputs = [obs_input_a0]

            q_arm_manip_norm = self.env.normalize_robot_joints(q_arm_manip, robot="franka", delta=False)
            q_arm_vision_norm = self.env.normalize_robot_joints(q_arm_vision, robot="arx", delta=False)
            q_hand_norm = self.env.normalize_robot_joints(q_hand, robot="leap", delta=False)
            q_hand_ctrl_delta = None
            if "q_hand_ctrl_delta" in self.state_encoders_keys:
                q_hand_ctrl_delta = self.env.normalize_robot_joints(
                    q_hand - self.env.abs_actions[:, 10:26], robot="leap", delta=True
                )
            objxyz_t0_base = None
            if "objxyz_t0" in self.state_encoders_keys:
                object_center_init_world = self.env._object_center_init_state.clone()
                franka_base_pos = self.env.states['franka_base_pose7'][:, :3] # (num_envs, 3)
                franka_base_quat = self.env.states['franka_base_pose7'][:, 3:] # (num_envs, 4)
                franka_base_rot_mat = quaternion_to_matrix_ig(franka_base_quat)
                rot_global2base = franka_base_rot_mat.transpose(1, 2) # (num_envs, 3, 3)

                point_shifted = (object_center_init_world - franka_base_pos).unsqueeze(1) # (num_envs, 1, 3)
                point_base_frame = torch.bmm(point_shifted, rot_global2base) # (num_envs, N, 3), bmm is like matmul but specifically made for batches of 2D matrices (input has to be 3D), faster than matmul
                objxyz_t0_base = point_base_frame[:, 0, :] # (num_envs, 3)
            aux_object_state = None
            if "aux_object_state" in self.state_encoders_keys:
                aux_object_state = self._expand_aux_to_chunk(self.aux_anchor_state)

            for obs_input in obs_inputs:
                obs_input["q_arm_manip"] = q_arm_manip_norm.clone()
                obs_input["q_arm_vision"] = q_arm_vision_norm.clone()
                obs_input["q_hand"] = q_hand_norm.clone()
                if q_hand_ctrl_delta is not None:
                    obs_input["q_hand_ctrl_delta"] = q_hand_ctrl_delta.clone()
                if objxyz_t0_base is not None:
                    obs_input["objxyz_t0"] = objxyz_t0_base.clone()
                if aux_object_state is not None:
                    obs_input["aux_object_state"] = aux_object_state.clone()

            with torch.no_grad():
                student_model = self.student_model.module if self.multi_gpu else self.student_model
                student_model.eval()
                output = student_model(obs_input_a0)
                student_actions_chunk = output["action"]
                if self.has_aux_input:
                    object_pos = self._get_aux_object_pos_in_base_frame()
                    eval_aux_metrics = self._compute_aux_metrics(
                        output["aux"],
                        obs_input_a0["aux_object_state"],
                        object_pos,
                    )
                    eval_aux_diff_l2_sum += eval_aux_metrics["aux_diff_l2"]
                    eval_aux_loss_sum += eval_aux_metrics["aux_loss"]
                    eval_aux_metric_count += 1
                    aux_chunk_abs = self._decode_aux_prediction(output["aux"], obs_input_a0["aux_object_state"])  # CODEX NEW
                    self.last_aux_state_from_prev_chunk_world[:] = self._base_points_to_world_frame(  # CODEX NEW
                        aux_chunk_abs[:, -1, :],
                        self.chunk_anchor_base_pose7,
                    )

            for action_idx in range(self.chunk_size):
                if eval_sim_step >= total_eval_sim_steps:  # support partial tail chunk.
                    break

                if policy_source == "student":
                    student_actions = student_actions_chunk[:, action_idx, :]
                    student_abs_targets = self.env._decode_student_actions_with_anchor(
                        student_actions,
                        self.action_chunk_anchor_q
                    )
                    step_actions = self.env._encode_abs_targets_to_student_space(
                        student_abs_targets,
                        self.env.states['q'].clone(),
                    )
                    step_actions = torch.clamp(step_actions, -self.env.clip_actions, self.env.clip_actions)
                else:
                    # teacher-only eval under the same eval rollout conditions.
                    teacher_obs = self.env.obs_buf.clone()
                    batch_dict = {
                        "is_train": False,
                        "obs": teacher_obs,
                        "prev_actions": None,
                    }
                    with torch.no_grad():
                        res_dict = self.teacher_model(batch_dict)
                    teacher_actions_raw = res_dict["mus"]
                    self.states = res_dict["rnn_states"]
                    teacher_actions_raw = torch.clamp(teacher_actions_raw, -self.env.clip_actions, self.env.clip_actions)
                    self.env._pre_physics_step_teacher(teacher_actions_raw)
                    step_actions = self.env.teacher_actions_converted.clone()

                # CODEX: keep distillation step synced during eval so wandb video logs carry the current trainer step.
                self.env.distillation_steps = self.total_steps
                # Let eval use the normal reset path. Forcing reset_buf to zero keeps bad states
                # alive far past their safety termination and can destabilize GPU kernels.
                self.env.step(step_actions)
                done_mask = self.env.reset_buf > 0
                if torch.any(done_mask):
                    done_lengths = (self.env.progress_buf[done_mask].to(dtype=torch.long) + 1).detach().cpu().tolist()
                    eval_completed_episode_lengths.extend(int(x) for x in done_lengths)
                self.last_student_actions[:] = step_actions.detach()
                eval_sim_step += 1  # CODEX

        if restore_training_state:
            # Restore a fresh post-reset state for the training loop.
            self.env.reset_idx()
            self.env.compute_observations()
            self.last_student_actions.zero_()
            self.env.progress_buf = torch.randint(
                0, self.env.max_episode_length,
                (self.env.num_envs,),
                device=self.device,
                dtype=self.env.progress_buf.dtype,
            )
            self.env.abs_actions[:] = self.env.states['q'].clone()
        else:
            self.last_student_actions.zero_()
        self.env.object_teleport_args["enable"] = prev_teleport_enable  # CODEX

        eval_success_rate_per_ep = self.env.extras["metrics/success_rate_5cm_per_ep"]
        eval_lifting_rate_per_ep = self.env.extras["metrics/lifting_rate_5cm_per_ep"]
        eval_episode_length_mean_per_ep = self.env.extras.get("metrics/episode_length_mean_per_ep", 0.0)
        eval_success_episode_length_mean_per_ep = self.env.extras.get("metrics/success_episode_length_mean_per_ep", 0.0)
        eval_failure_episode_length_mean_per_ep = self.env.extras.get("metrics/failure_episode_length_mean_per_ep", 0.0)
        eval_episode_length_max = self.env.extras.get("metrics/episode_length_max", 0)
        if pre_total_eps is not None and pre_total_succ is not None and pre_total_lift is not None:
            post_total_eps = int(self.env.per_object_episode_counts.sum().item())
            post_total_succ = int(self.env.per_object_success_counts.sum().item())
            post_total_lift = int(self.env.per_object_lifting_counts.sum().item())
            delta_eps = post_total_eps - pre_total_eps
            delta_succ = post_total_succ - pre_total_succ
            delta_lift = post_total_lift - pre_total_lift
            delta_episode_length_sum = (
                int(self.env._episode_length_sum_total.item()) - pre_episode_length_sum
            ) if pre_episode_length_sum is not None else None
            delta_success_episode_length_sum = (
                int(self.env._success_episode_length_sum_total.item()) - pre_success_episode_length_sum
            ) if pre_success_episode_length_sum is not None else None
            delta_failure_episode_length_sum = (
                int(self.env._failure_episode_length_sum_total.item()) - pre_failure_episode_length_sum
            ) if pre_failure_episode_length_sum is not None else None
            post_episode_length_max = (
                int(self.env._episode_length_max_total.item())
                if pre_episode_length_max is not None else None
            )
            if self.multi_gpu:
                totals = torch.tensor(
                    [delta_eps, delta_succ, delta_lift],
                    device=self.device,
                    dtype=torch.long,
                )
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                delta_eps = int(totals[0].item())
                delta_succ = int(totals[1].item())
                delta_lift = int(totals[2].item())
                if delta_episode_length_sum is not None:
                    length_totals = torch.tensor(
                        [
                            delta_episode_length_sum,
                            delta_success_episode_length_sum,
                            delta_failure_episode_length_sum,
                            post_episode_length_max,
                        ],
                        device=self.device,
                        dtype=torch.long,
                    )
                    dist.all_reduce(length_totals[:3], op=dist.ReduceOp.SUM)
                    dist.all_reduce(length_totals[3:4], op=dist.ReduceOp.MAX)
                    delta_episode_length_sum = int(length_totals[0].item())
                    delta_success_episode_length_sum = int(length_totals[1].item())
                    delta_failure_episode_length_sum = int(length_totals[2].item())
                    post_episode_length_max = int(length_totals[3].item())
            if delta_eps > 0:
                eval_success_rate_per_ep = float(delta_succ) / float(delta_eps)
                eval_lifting_rate_per_ep = float(delta_lift) / float(delta_eps)
                if delta_episode_length_sum is not None:
                    eval_episode_length_mean_per_ep = float(delta_episode_length_sum) / float(delta_eps)
                    eval_success_episode_length_mean_per_ep = (
                        float(delta_success_episode_length_sum) / float(delta_succ)
                    ) if delta_succ > 0 else 0.0
                    delta_failure_eps = max(delta_eps - delta_succ, 0)
                    eval_failure_episode_length_mean_per_ep = (
                        float(delta_failure_episode_length_sum) / float(delta_failure_eps)
                    ) if delta_failure_eps > 0 else 0.0
                    eval_episode_length_max = post_episode_length_max

        eval_wandb_logs = {
            f"{metric_prefix}/eval_success_rate_5cm_per_step": self.env.extras["metrics/success_rate_5cm_per_step"],
            f"{metric_prefix}/eval_success_rate_5cm_per_ep_instant": self.env.extras["metrics/success_rate_5cm_per_ep_instant"],
            f"{metric_prefix}/eval_success_rate_5cm_per_ep": eval_success_rate_per_ep,
            f"{metric_prefix}/eval_lifting_rate_5cm_per_step": self.env.extras["metrics/lifting_rate_5cm_per_step"],
            f"{metric_prefix}/eval_lifting_rate_5cm_per_ep_instant": self.env.extras["metrics/lifting_rate_5cm_per_ep_instant"],
            f"{metric_prefix}/eval_lifting_rate_5cm_per_ep": eval_lifting_rate_per_ep,
            f"{metric_prefix}/eval_episode_length_mean_per_ep": eval_episode_length_mean_per_ep,
            f"{metric_prefix}/eval_success_episode_length_mean_per_ep": eval_success_episode_length_mean_per_ep,
            f"{metric_prefix}/eval_failure_episode_length_mean_per_ep": eval_failure_episode_length_mean_per_ep,
            f"{metric_prefix}/eval_episode_length_max": eval_episode_length_max,
        }
        if self.multi_gpu:
            gathered_episode_lengths = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_episode_lengths, eval_completed_episode_lengths)
            eval_completed_episode_lengths = []
            for lengths in gathered_episode_lengths:
                if lengths:
                    eval_completed_episode_lengths.extend(int(x) for x in lengths)
        eval_wandb_logs[f"{metric_prefix}/eval_episode_length_hist_count"] = len(eval_completed_episode_lengths)
        if (
            len(eval_completed_episode_lengths) > 0
            and ((not self.multi_gpu) or (self.global_rank == 0))
        ):
            eval_wandb_logs[f"{metric_prefix}/eval_episode_length_hist"] = wandb.Histogram(
                torch.tensor(eval_completed_episode_lengths, dtype=torch.float32).cpu().numpy()
            )
        if pre_category_eps is not None and pre_category_succ is not None:
            post_category_eps = self.env.per_verified_bank_episode_counts.clone()
            post_category_succ = self.env.per_verified_bank_success_counts.clone()
            delta_category_eps = (post_category_eps - pre_category_eps).to(dtype=torch.long)
            delta_category_succ = (post_category_succ - pre_category_succ).to(dtype=torch.long)
            if self.multi_gpu:
                dist.all_reduce(delta_category_eps, op=dist.ReduceOp.SUM)
                dist.all_reduce(delta_category_succ, op=dist.ReduceOp.SUM)
            for category_id, category_name in enumerate(self.env.VERIFIED_BANK_CATEGORY_NAMES):
                category_total_eps = int(delta_category_eps[category_id].item())
                category_success_eps = int(delta_category_succ[category_id].item())
                category_success_rate = (
                    float(category_success_eps) / float(category_total_eps)
                ) if category_total_eps > 0 else 0.0
                eval_wandb_logs[
                    f"{metric_prefix}/eval_success_rate_5cm_per_ep_{category_name}"
                ] = category_success_rate
                eval_wandb_logs[
                    f"{metric_prefix}/eval_success_rate_5cm_per_ep_{category_name}_count"
                ] = category_total_eps
        if policy_source == "student" and self.has_aux_input and eval_aux_metric_count > 0:
            eval_wandb_logs[f"{metric_prefix}/aux_diff_l2"] = eval_aux_diff_l2_sum / eval_aux_metric_count
            eval_wandb_logs[f"{metric_prefix}/aux_loss"] = eval_aux_loss_sum / eval_aux_metric_count

        success_key = f"{metric_prefix}/eval_success_rate_5cm_per_ep"
        lift_key = f"{metric_prefix}/eval_lifting_rate_5cm_per_ep"
        if (not self.multi_gpu) or (self.global_rank == 0):
            print(
                f"[{metric_prefix}] episode={self.episode} "
                f"success_rate_5cm_per_ep={float(eval_wandb_logs[success_key]):.4f} "
                f"lifting_rate_5cm_per_ep={float(eval_wandb_logs[lift_key]):.4f} "
                f"episode_length_mean={float(eval_wandb_logs[f'{metric_prefix}/eval_episode_length_mean_per_ep']):.1f} "
                f"success_length_mean={float(eval_wandb_logs[f'{metric_prefix}/eval_success_episode_length_mean_per_ep']):.1f} "
                f"failure_length_mean={float(eval_wandb_logs[f'{metric_prefix}/eval_failure_episode_length_mean_per_ep']):.1f} "
                f"episode_length_max={int(eval_wandb_logs[f'{metric_prefix}/eval_episode_length_max'])}"
            )  # CODEX
            if pre_category_eps is not None and pre_category_succ is not None:
                category_parts = []
                for category_name in self.env.VERIFIED_BANK_CATEGORY_NAMES:
                    rate_key = f"{metric_prefix}/eval_success_rate_5cm_per_ep_{category_name}"
                    count_key = f"{metric_prefix}/eval_success_rate_5cm_per_ep_{category_name}_count"
                    category_parts.append(
                        f"{category_name}={float(eval_wandb_logs.get(rate_key, 0.0)):.4f}"
                        f" (n={int(eval_wandb_logs.get(count_key, 0))})"
                    )
                print(f"[{metric_prefix}] per_type " + " ".join(category_parts))

        return eval_wandb_logs

    def _get_global_verified_teacher_bank_category_stats(self, category_name):
        category_id = int(self.env.VERIFIED_BANK_CATEGORY_NAMES.index(category_name))
        global_counts = self.env.get_verified_teacher_bank_category_counts_tensor(category_id).clone()
        global_correct_counts = self.env.get_verified_teacher_bank_category_correct_counts_tensor(category_id).clone()
        if self.multi_gpu:
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_correct_counts, op=dist.ReduceOp.SUM)

        local_summary = self.env.get_verified_teacher_bank_count_summary()[category_name]
        attempt_total = int(local_summary["attempt_total"])
        success_total = int(local_summary["success_total"])
        attempt_mean_steps = float(local_summary.get("attempt_mean_steps", 0.0))
        success_mean_steps = float(local_summary.get("success_mean_steps", 0.0))
        failure_mean_steps = float(local_summary.get("failure_mean_steps", 0.0))
        attempt_max_steps = int(local_summary.get("attempt_max_steps", 0))
        if self.multi_gpu:
            attempt_total_tensor = torch.tensor([attempt_total], device=self.device, dtype=torch.long)
            success_total_tensor = torch.tensor([success_total], device=self.device, dtype=torch.long)
            attempt_step_sum_tensor = torch.tensor(
                [attempt_mean_steps * float(max(attempt_total, 0))],
                device=self.device,
                dtype=torch.float64,
            )
            success_step_sum_tensor = torch.tensor(
                [success_mean_steps * float(max(success_total, 0))],
                device=self.device,
                dtype=torch.float64,
            )
            failure_total_local = max(attempt_total - success_total, 0)
            failure_step_sum_tensor = torch.tensor(
                [failure_mean_steps * float(failure_total_local)],
                device=self.device,
                dtype=torch.float64,
            )
            attempt_max_steps_tensor = torch.tensor([attempt_max_steps], device=self.device, dtype=torch.long)
            dist.all_reduce(attempt_total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(success_total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(attempt_step_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(success_step_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(failure_step_sum_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(attempt_max_steps_tensor, op=dist.ReduceOp.MAX)
            attempt_total = int(attempt_total_tensor.item())
            success_total = int(success_total_tensor.item())
            attempt_mean_steps = (
                float(attempt_step_sum_tensor.item()) / float(attempt_total)
            ) if attempt_total > 0 else 0.0
            success_mean_steps = (
                float(success_step_sum_tensor.item()) / float(success_total)
            ) if success_total > 0 else 0.0
            failure_total = max(attempt_total - success_total, 0)
            failure_mean_steps = (
                float(failure_step_sum_tensor.item()) / float(failure_total)
            ) if failure_total > 0 else 0.0
            attempt_max_steps = int(attempt_max_steps_tensor.item())

        return {
            "counts": global_counts,
            "correct_counts": global_correct_counts,
            "stored_total": int(global_counts.sum().item()),
            "stored_min_per_variation": int(global_counts.min().item()) if global_counts.numel() > 0 else 0,
            "stored_max_per_variation": int(global_counts.max().item()) if global_counts.numel() > 0 else 0,
            "correct_stored_total": int(global_correct_counts.sum().item()),
            "correct_stored_min_per_variation": int(global_correct_counts.min().item()) if global_correct_counts.numel() > 0 else 0,
            "correct_stored_max_per_variation": int(global_correct_counts.max().item()) if global_correct_counts.numel() > 0 else 0,
            "attempt_total": attempt_total,
            "success_total": success_total,
            "success_rate": (float(success_total) / float(attempt_total)) if attempt_total > 0 else 0.0,
            "attempt_mean_steps": attempt_mean_steps,
            "success_mean_steps": success_mean_steps,
            "failure_mean_steps": failure_mean_steps,
            "attempt_max_steps": attempt_max_steps,
        }

    def collect_verified_teacher_bank(self):
        if not getattr(self.env, "_verified_teacher_bank_enable", False):
            return

        prev_teleport_enable = bool(self.env.object_teleport_args["enable"])
        self.env.object_teleport_args["enable"] = False
        all_env_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long)
        target_count_per_variation = int(getattr(self.env, "_verified_teacher_bank_capacity_per_env", 0))
        num_variants_global = int(getattr(self.env, "teacher_bank_num_variants_global", int(self.env.num_envs)))
        target_total_per_category = max(target_count_per_variation, 0) * num_variants_global
        success_budget_cap = int(self.teacher_state_bank_max_success_episodes_per_category)
        periodic_save_interval = max(self.teacher_state_bank_save_interval_seconds, 0.0)
        last_periodic_save_time = time.time()
        last_periodic_saved_local_total = 0
        progress = None
        requested_category = self.cfg.dagger.teacher_state_bank.get("collect_category", None)
        if requested_category is not None:
            requested_category = str(requested_category)
            if requested_category not in self.env.VERIFIED_BANK_CATEGORY_NAMES:
                raise ValueError(
                    f"Unsupported dagger.teacher_state_bank.collect_category={requested_category}. "
                    f"Expected one of {self.env.VERIFIED_BANK_CATEGORY_NAMES}."
                )
            category_names = (requested_category,)
        else:
            category_names = self.env.VERIFIED_BANK_CATEGORY_NAMES

        try:
            for category_name in category_names:
                category_id = int(self.env.VERIFIED_BANK_CATEGORY_NAMES.index(category_name))
                initial_stats = self._get_global_verified_teacher_bank_category_stats(category_name)
                if (
                    target_count_per_variation > 0
                    and initial_stats["stored_min_per_variation"] >= target_count_per_variation
                ):
                    continue

                self.env.configure_verified_teacher_bank_collection(True, category_name)
                self.env.reset_idx(all_env_ids)
                self.env.compute_observations()

                success_episode_count_global = 0
                prev_stored_total = int(self.env.get_verified_teacher_bank_count_summary()[category_name]["stored_total"])
                prev_success_total = int(self.env.get_verified_teacher_bank_count_summary()[category_name]["success_total"])
                progress_total = target_total_per_category
                if progress_total <= 0:
                    progress_total = max(success_budget_cap, 1)
                progress = tqdm(
                    total=progress_total,
                    desc=f"Collect Verified {category_name}",
                    ncols=None,
                    dynamic_ncols=True,
                    disable=(self.multi_gpu and self.global_rank != 0),
                )
                last_periodic_saved_local_total = prev_stored_total

                while True:
                    pre_step_stats = self._get_global_verified_teacher_bank_category_stats(category_name)
                    if (
                        target_count_per_variation > 0
                        and pre_step_stats["stored_min_per_variation"] >= target_count_per_variation
                    ):
                        break
                    if success_budget_cap > 0 and success_episode_count_global >= success_budget_cap:
                        break

                    gained_success_local = 0
                    gained_local = 0
                    teacher_obs = self.env.obs_buf.clone()
                    batch_dict = {
                        "is_train": False,
                        "obs": teacher_obs,
                        "prev_actions": None,
                    }
                    with torch.no_grad():
                        res_dict = self.teacher_model(batch_dict)
                    teacher_actions_raw = torch.clamp(
                        res_dict["mus"], -self.env.clip_actions, self.env.clip_actions
                    )
                    self.env._pre_physics_step_teacher(teacher_actions_raw)
                    step_actions = self.env.teacher_actions_converted.clone()

                    self.env.distillation_steps = self.total_steps
                    self.env.step(step_actions)

                    success_reset_ids = (
                        self.env.success_long_enough & (self.env.reset_buf == 0)
                    ).nonzero(as_tuple=False).squeeze(-1)
                    if success_reset_ids.numel() > 0:
                        self.env.finalize_episode_metrics_before_reset(
                            success_reset_ids,
                            mark_success=True,
                            mark_lifting=True,
                        )
                        self.env.reset_buf[success_reset_ids] = 1

                    done_env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                    if done_env_ids.numel() > 0:
                        self.env.record_verified_teacher_episode_outcomes(done_env_ids)
                        current_summary = self.env.get_verified_teacher_bank_count_summary()[category_name]
                        current_success_total = int(current_summary["success_total"])
                        gained_success_local = max(current_success_total - prev_success_total, 0)
                        prev_success_total = current_success_total
                        self.env.reset_idx(done_env_ids)
                        self.env.compute_observations()

                        current_total = int(current_summary["stored_total"])
                        gained_local = current_total - prev_stored_total
                        prev_stored_total = current_total

                    gained_success_global = gained_success_local
                    if self.multi_gpu:
                        gained_success_tensor = torch.tensor([gained_success_local], device=self.device, dtype=torch.long)
                        dist.all_reduce(gained_success_tensor, op=dist.ReduceOp.SUM)
                        gained_success_global = int(gained_success_tensor.item())

                    success_episode_count_global += max(gained_success_global, 0)

                    current_stats = self._get_global_verified_teacher_bank_category_stats(category_name)
                    capped_progress_value = min(current_stats["stored_total"], progress_total)
                    if capped_progress_value > progress.n:
                        progress.update(capped_progress_value - progress.n)

                    if (gained_local > 0 or gained_success_global > 0) and ((not self.multi_gpu) or (self.global_rank == 0)):
                        progress.set_postfix(
                            stored_total=current_stats["stored_total"],
                            stored_min_per_variation=current_stats["stored_min_per_variation"],
                            correct_stored_total=current_stats["correct_stored_total"],
                            correct_min_per_variation=current_stats["correct_stored_min_per_variation"],
                            attempt_total=current_stats["attempt_total"],
                            success_total=current_stats["success_total"],
                            success_rate=f"{current_stats['success_rate']:.3f}",
                            attempt_mean_steps=f"{current_stats['attempt_mean_steps']:.1f}",
                            failure_mean_steps=f"{current_stats['failure_mean_steps']:.1f}",
                            attempt_max_steps=current_stats["attempt_max_steps"],
                            gained=gained_local,
                            success_global=success_episode_count_global,
                        )

                    if (
                        periodic_save_interval > 0.0
                        and prev_stored_total > last_periodic_saved_local_total
                        and (time.time() - last_periodic_save_time) >= periodic_save_interval
                    ):
                        self.env.save_verified_teacher_bank_hdf5(rank=self.global_rank)
                        if hasattr(self.env, "save_activation_snapshot_bank_hdf5"):
                            self.env.save_activation_snapshot_bank_hdf5(rank=self.global_rank)
                        last_periodic_save_time = time.time()
                        last_periodic_saved_local_total = prev_stored_total
                        if (not self.multi_gpu) or (self.global_rank == 0):
                            print(
                                "[DaggerMobile/verified_teacher_bank] "
                                f"periodic_save category={category_name} "
                                f"local_stored_total={prev_stored_total}"
                            )

                progress.close()
                progress = None
                if self.teacher_state_bank_save_each_category:
                    self.env.save_verified_teacher_bank_hdf5(rank=self.global_rank)
                    if hasattr(self.env, "save_activation_snapshot_bank_hdf5"):
                        self.env.save_activation_snapshot_bank_hdf5(rank=self.global_rank)
                summary = self._get_global_verified_teacher_bank_category_stats(category_name)
                stored_total = int(summary["stored_total"])
                stored_min_per_variation = int(summary["stored_min_per_variation"])
                correct_stored_total = int(summary["correct_stored_total"])
                correct_stored_min_per_variation = int(summary["correct_stored_min_per_variation"])
                attempt_total = int(summary["attempt_total"])
                success_total = int(summary["success_total"])
                success_rate = float(summary["success_rate"])
                attempt_mean_steps = float(summary["attempt_mean_steps"])
                success_mean_steps = float(summary["success_mean_steps"])
                failure_mean_steps = float(summary["failure_mean_steps"])
                attempt_max_steps = int(summary["attempt_max_steps"])
                if (not self.multi_gpu) or (self.global_rank == 0):
                    print(
                        "[DaggerMobile/verified_teacher_bank] "
                        f"category={category_name} "
                        f"stored_total={stored_total} "
                        f"stored_min_per_variation={stored_min_per_variation} "
                        f"correct_stored_total={correct_stored_total} "
                        f"correct_stored_min_per_variation={correct_stored_min_per_variation} "
                        f"attempt_total={attempt_total} "
                        f"success_total={success_total} "
                        f"success_rate={success_rate:.4f} "
                        f"attempt_mean_steps={attempt_mean_steps:.2f} "
                        f"success_mean_steps={success_mean_steps:.2f} "
                        f"failure_mean_steps={failure_mean_steps:.2f} "
                        f"attempt_max_steps={attempt_max_steps} "
                        f"success_budget_global={success_episode_count_global} "
                        f"target_per_variation={target_count_per_variation}"
                    )
        finally:
            if progress is not None:
                progress.close()
            self.env.configure_verified_teacher_bank_collection(False, None)
            self.env.object_teleport_args["enable"] = prev_teleport_enable
            self.env.save_verified_teacher_bank_hdf5(rank=self.global_rank)
            if hasattr(self.env, "save_activation_snapshot_bank_hdf5"):
                self.env.save_activation_snapshot_bank_hdf5(rank=self.global_rank)
            if self.multi_gpu and not self.teacher_state_bank_collect_only:
                dist.barrier()

    def train(self):
        if self.teacher_state_bank_collect_before_train or self.teacher_state_bank_collect_only:
            self.collect_verified_teacher_bank()
            self.env._verified_teacher_bank_loaded = True
            self.env.reset_idx(torch.arange(self.env.num_envs, device=self.device, dtype=torch.long))
            self.env.compute_observations()
            if self.teacher_state_bank_collect_only:
                return
        if self.teacher_eval_only:
            teacher_eval_first_window = True
            while True:
                eval_wandb_logs = self.eval(
                    policy_source="teacher",
                    capture_video=False,
                    restore_training_state=False,
                    reset_all_envs_at_start=teacher_eval_first_window,
                )
                teacher_eval_first_window = False
                self.latest_eval_wandb_logs = eval_wandb_logs.copy()
                self.last_eval_episode = int(self.episode)
                self.total_steps += int(self.env.max_episode_length)
                if (not self.multi_gpu) or (self.global_rank == 0):
                    wandb_payload = dict(eval_wandb_logs)
                    wandb_payload["episode"] = self.episode
                    wandb_payload["eval/triggered"] = 1.0
                    if self.use_wandb:
                        merged_log_step = int(self.total_steps)
                        if wandb.run is not None and wandb.run.step is not None:
                            merged_log_step = max(merged_log_step, int(wandb.run.step))
                        wandb.log(self._sanitize_wandb_logs(wandb_payload), step=merged_log_step)
                self.episode += 1
        while self.episode < self.total_episodes:
            train_metrics = {}
            eval_metrics = None

            start_time = time.time()
            remaining_episodes = self.total_episodes - self.episode
            eval_policy = (self.eval_freq > 0) and (self.episode % self.eval_freq == 0)

            # Start training-video capture before training rollout begins.
            if self.env.video_logging["capture"] and eval_policy:
                self.env.eval_video_active = False
                self.env.train_video_active = True
                self.env.train_video_total_steps = int(self.steps_per_episode * self.chunk_size)
                self.env.train_video_step_idx = 0

            # while True:
            #     self.eval(policy_source="student", capture_video=False)

            train_loss = self.train_episode()
            # CODEX: snapshot train extras before eval so eval cannot contaminate train metrics.
            train_env_extras = {}
            for k, v in self.env.extras.items():
                if torch.is_tensor(v):
                    train_env_extras[k] = v.detach().clone()
                else:
                    train_env_extras[k] = v
            if eval_policy:
                if self.eval_only_teacher:
                    eval_wandb_logs = self.eval(policy_source="teacher", capture_video=False)  # CODEX
                    self.latest_eval_wandb_logs = eval_wandb_logs.copy()  # CODEX
                    self.last_eval_episode = int(self.episode)  # CODEX
                else:
                    eval_wandb_logs = self.eval(policy_source="student", capture_video=True)
                    self.latest_eval_wandb_logs = eval_wandb_logs.copy()  # CODEX
                    self.last_eval_episode = int(self.episode)  # CODEX
                    if self.debug_teacher_eval:
                        teacher_eval_wandb_logs = self.eval(policy_source="teacher", capture_video=False)  # CODEX
                        self.latest_eval_wandb_logs.update(teacher_eval_wandb_logs)  # CODEX

            if (not self.multi_gpu) or (self.global_rank == 0):
                episode_time = time.time() - start_time
                estimated_finish_time = start_time + episode_time * remaining_episodes

                train_metrics["train/loss_episode"] = train_loss
                train_metrics["time/episode_time"] = episode_time
                train_metrics["episode"] = self.episode
                train_metrics["train/teacher_forcing_prop"] = self.teacher_forcing_prop
                train_metrics.update(self._sanitize_wandb_logs(train_env_extras))
                if eval_policy and self.latest_eval_wandb_logs:
                    eval_metrics = self.latest_eval_wandb_logs.copy()  # CODEX
                    eval_metrics["eval/triggered"] = 1.0 if eval_policy else 0.0  # CODEX
                    eval_metrics["eval/episodes_since_last_eval"] = float(self.episode - self.last_eval_episode)  # CODEX
                    eval_metrics["episode"] = self.episode  # CODEX

                if self.use_wandb:
                    wandb_payload = dict(train_metrics)
                    if eval_metrics is not None:
                        wandb_payload.update(eval_metrics)
                    merged_log_step = int(self.total_steps)  # CODEX
                    if wandb.run is not None and wandb.run.step is not None:  # CODEX
                        merged_log_step = max(merged_log_step, int(wandb.run.step))  # CODEX
                    wandb.log(self._sanitize_wandb_logs(wandb_payload), step=merged_log_step)  # CODEX
                # TODO: add args: save ckpt? frequency?
                if eval_policy:
                    if self.eval_only_teacher:
                        eval_lifting = eval_wandb_logs.get("teacher_eval/eval_lifting_rate_5cm_per_ep", None)  # CODEX
                    else:
                        eval_lifting = eval_wandb_logs.get("eval/eval_lifting_rate_5cm_per_ep", None)  # CODEX
                else:
                    eval_lifting = None  # CODEX
                self.save_checkpoint(self.episode, train_metrics["metrics/success_rate_5cm_per_ep"], eval_lifting)

                log_metrics = dict(train_metrics)
                if eval_metrics is not None:
                    log_metrics.update(eval_metrics)

                colorprint(f"Episode {self.episode + 1}/{self.total_episodes} completed in {timedelta(seconds=int(episode_time))}", color="magenta")
                for metric, value in log_metrics.items():
                    if type(value) == float:
                        colorprint(f"{metric}: {value:.4f}", color="green")
                colorprint(f"Average episodes per hour: {1/episode_time*3600:.2f}")
                colorprint(f"Estimated completion: {datetime.fromtimestamp(estimated_finish_time).strftime('%Y-%m-%d %H:%M:%S')}")
                print("\n")

            self.episode += 1
