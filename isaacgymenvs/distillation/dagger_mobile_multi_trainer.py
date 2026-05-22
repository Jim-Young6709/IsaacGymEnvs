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
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict  # CODEX: rank-local task config composition.
from isaacgymenvs.tasks import FrankaLEAPMobile


class DaggerMobileMultiExp:
    def __init__(self, cfg):
        # load configs
        self.multi_gpu = cfg.multi_gpu
        self.local_rank = 0  # CODEX: keep rank fields available for optional env hooks in single-GPU runs.
        self.global_rank = 0  # CODEX: verified-bank loading uses this shard rank when present.
        self.world_size = 1  # CODEX
        expert_idx = 0  # CODEX: used by optional per-expert task/env config selection.
        teacher_ckpts = cfg.teacher.ckpt  # CODEX: per-expert rank mapping must be known before env creation.
        num_experts = len(teacher_ckpts) if isinstance(teacher_ckpts, (list, tuple, ListConfig)) else 1
        if self.multi_gpu:
            dist.init_process_group(backend="nccl")
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.global_rank = int(os.getenv("RANK", "0"))
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))   

            cfg.task.env.scene.batch_idx = self.global_rank
            cfg.sim_device = f"cuda:{self.local_rank}"
            cfg.rl_device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.local_rank)

            cfg.seed = max(cfg.seed, 1) * (self.global_rank + 1)

            expert_idx = self.local_rank % num_experts
        self.num_experts = num_experts  # CODEX
        # CODEX: keep the launcher-level rollout length before rank-local task swaps.
        # Env reset length remains task-local via cfg.task.env.episodeLength, so side can
        # reset at 600 while the multi-teacher training iteration still runs 1000 steps.
        launcher_steps_per_episode = int(cfg.dagger.steps_per_episode / cfg.chunk_size)
        self.expert_rank, self.expert_world_size = self._compute_expert_rank_info(expert_idx, num_experts)  # CODEX
        self._expert_override_values = self._snapshot_expert_overrides(cfg)  # CODEX
        # CODEX: WBCMultiExp can mix task lineages by composing a rank-local task config before env creation.
        self._apply_expert_task_config(cfg, expert_idx)  # CODEX
        self._apply_expert_overrides(cfg, expert_idx)  # CODEX
        self._apply_expert_rank_config(cfg)  # CODEX
        if self.multi_gpu and self._has_cfg_path(cfg, "task.env.scene.batch_idx"):
            self._set_cfg_path(cfg, "task.env.scene.batch_idx", self.global_rank)  # CODEX
        self.expert_idx = expert_idx  # CODEX: identify per-rank expert for W&B/task logs.
        self.expert_task_name = str(cfg.task.get("name", cfg.task_name))  # CODEX
        self._configure_rank_video_logging(cfg)  # CODEX: one video rank per expert/task.
        cfg.task.env.video_logging.freq = max((cfg.dagger.eval_freq + 1), 10) * cfg.task.env.episodeLength

        self.cfg = cfg
        self.total_episodes = cfg.dagger.total_episodes
        self.steps_per_episode = launcher_steps_per_episode  # CODEX: decoupled from rank-local env episodeLength.
        self.env_episode_length = int(cfg.task.env.episodeLength)  # CODEX: task-local timeout/reset length.
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
        self.resume_wandb = bool(self.cfg.dagger.get("resume_wandb", False))  # CODEX: checkpoint resume starts fresh W&B runs by default.

        self.use_wandb = self.cfg.wandb_activate
        self.wandb_project = self.cfg.wandb_project
        self.wandb_name = self.cfg.wandb_name
        self.wandb_group = self.cfg.wandb_group or self.exp_name  # CODEX: group per-task expert runs.
        self.wandb_id = None

        self.state_encoders_keys = self.cfg.model.state_encoders_cfg.keys()
        self.pcd_encoders_keys = self.cfg.model.pcd_encoders_cfg.keys()

        self.save_dir = Path("dagger_ckpts") / self.exp_name
        self.save_freq = self.cfg.dagger.save_freq
        os.makedirs(self.save_dir, exist_ok=True)

        self.eval_freq = self.cfg.dagger.eval_freq
        self.teacher_eval_only = bool(self.cfg.dagger.get("teacher_eval_only", False))  # CODEX
        self.eval_only = bool(self.cfg.dagger.get("eval_only", False))  # CODEX: skip training and repeatedly run student eval.
        self.profile_timing = bool(self.cfg.dagger.get("profile_timing", True))
        self.profile_print_freq = int(self.cfg.dagger.get("profile_print_freq", 1))
        self.profile_cuda_sync = bool(self.cfg.dagger.get("profile_cuda_sync", True))
        self.debug_eval_sync = bool(self.cfg.dagger.get("debug_eval_sync", False))  # CODEX: diagnose eval rollout vs post-eval all_gather stalls.
        self.debug_eval_steps = int(self.cfg.dagger.get("debug_eval_steps", 2))  # CODEX: only print detailed eval checkpoints for the first few steps.

        if self.multi_gpu:
            self.use_wandb = bool(self.cfg.wandb_activate) and (int(self.expert_rank) == 0)  # CODEX: one W&B run per task/expert.
            if self.use_wandb:
                base_wandb_name = str(self.cfg.wandb_name)
                self.wandb_name = (
                    f"{base_wandb_name}_expert{self.expert_idx}_{self.expert_task_name}"
                )  # CODEX
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
            wandb_run_id = self.wandb_id
            wandb_resume = "must" if self.wandb_id else "never"
            if wandb_run_id is None:
                wandb_run_id = (
                    f"{wandb.util.generate_id()}-"
                    f"e{int(self.expert_idx)}-r{int(self.global_rank)}"
                )  # CODEX: avoid shared WANDB_RUN_ID/run-file collisions across expert owner ranks.
            wandb_dir = Path("wandb") / f"expert_{int(self.expert_idx)}_rank_{int(self.global_rank)}"  # CODEX
            os.makedirs(wandb_dir, exist_ok=True)  # CODEX
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_name,
                group=self.wandb_group,  # CODEX: keep all per-expert runs under one experiment group.
                job_type=f"expert_{self.expert_idx}",  # CODEX
                id=wandb_run_id,  # CODEX
                resume=wandb_resume,  # CODEX
                dir=str(wandb_dir),  # CODEX: isolate local W&B datastore by expert owner rank.
                config={
                    "batch_size": self.batch_size,
                    "grad_updates_per_step": self.grad_updates_per_step,
                    "num_episodes": self.total_episodes,
                    "steps_per_episode": self.steps_per_episode,  # CODEX
                    "env_episode_length": self.env_episode_length,  # CODEX
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "rank": self.global_rank,  # CODEX
                    "local_rank": self.local_rank,  # CODEX
                    "expert_idx": self.expert_idx,  # CODEX
                    "expert_rank": self.expert_rank,  # CODEX
                    "expert_world_size": self.expert_world_size,  # CODEX
                    "expert_task_name": self.expert_task_name,  # CODEX
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
    def _select_expert_value(self, value, expert_idx):
        # CODEX: support optional per-expert side-bank config lists without affecting scalar configs.
        if isinstance(value, (list, tuple, ListConfig)):
            if expert_idx >= len(value):
                raise ValueError(
                    f"Per-expert override has {len(value)} entries, but rank maps to expert_idx={expert_idx}"
                )
            return value[expert_idx]
        return value

    def _compute_expert_rank_info(self, expert_idx, num_experts):
        # CODEX: side reset banks are sharded by expert-local ranks, not by all WBCMultiExp launcher ranks.
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        world_size = max(int(self.world_size), 1)
        matching_ranks = [rank for rank in range(world_size) if rank % num_experts == int(expert_idx)]
        if not matching_ranks:
            return 0, 1
        expert_rank = sum(1 for rank in matching_ranks if rank < int(self.global_rank))
        return expert_rank, len(matching_ranks)

    def _configure_rank_video_logging(self, cfg):
        # CODEX: log videos from one GPU per task/expert instead of only launcher rank 0.
        # For 8 GPUs and experts [shelf, side, top], this enables video on ranks 0, 1, 2.
        if not self._has_cfg_path(cfg, "task.env.video_logging.capture"):
            return

        requested_capture = bool(cfg.task.env.video_logging.capture)
        capture_this_rank = requested_capture and ((not self.multi_gpu) or int(self.expert_rank) == 0)
        self._set_cfg_path(cfg, "task.env.video_logging.capture", capture_this_rank)

        if self.multi_gpu:
            cfg.graphics_device_id = int(self.local_rank) if capture_this_rank else -1

        if capture_this_rank:
            print(
                "[DaggerMobileMultiExp/CODEX] "
                f"video_capture rank={self.global_rank} expert_idx={self.expert_idx} "
                f"expert_rank={self.expert_rank} graphics_device_id={cfg.graphics_device_id}"
            )

    def _maybe_start_env_video(self, mode, total_steps, periodic=False):
        # CODEX: copied side-distillation env uses explicit train/eval video capture windows,
        # while dex envs use their own modulo video_logger. This hook only affects envs
        # that expose the explicit side-style flags.
        if not bool(getattr(self.env, "video_logging", {}).get("capture", False)):
            return

        active_attr = f"{mode}_video_active"
        total_attr = f"{mode}_video_total_steps"
        step_attr = f"{mode}_video_step_idx"
        if not all(hasattr(self.env, attr) for attr in (active_attr, total_attr, step_attr)):
            return
        if bool(getattr(self.env, active_attr)):
            return

        if periodic:
            video_freq = int(self.env.video_logging.get("freq", 0))
            if video_freq <= 0 or (int(self.env.sim_steps) % video_freq) != 0:
                return

        setattr(self.env, active_attr, True)
        setattr(self.env, total_attr, int(total_steps))
        setattr(self.env, step_attr, 0)

    def _env_has_explicit_video_window(self, mode):
        # CODEX: side isolated envs expose explicit video windows; legacy dex envs do not.
        active_attr = f"{mode}_video_active"
        total_attr = f"{mode}_video_total_steps"
        step_attr = f"{mode}_video_step_idx"
        return all(hasattr(self.env, attr) for attr in (active_attr, total_attr, step_attr))

    def _disable_legacy_eval_video_if_needed(self):
        # CODEX: legacy dex video_logger captures/render-buffers every eval step and can
        # make eval appear stuck. Keep explicit side-style eval videos enabled.
        video_logging = getattr(self.env, "video_logging", None)
        if not isinstance(video_logging, (dict, DictConfig)):
            return None
        if not bool(video_logging.get("capture", False)):
            return None
        if self._env_has_explicit_video_window("eval"):
            return None

        prev_capture = video_logging["capture"]
        video_logging["capture"] = False
        if (not self.multi_gpu) or int(self.expert_rank) == 0:
            print(
                "[DaggerMobileMultiExp/CODEX eval] disabled legacy env video_logger during eval "
                f"rank={self.global_rank} expert_idx={self.expert_idx} task={self.expert_task_name}",
                flush=True,
            )
        return prev_capture

    def _restore_legacy_eval_video(self, prev_capture):
        # CODEX: restore train-time capture after eval if we temporarily disabled it.
        if prev_capture is None:
            return
        video_logging = getattr(self.env, "video_logging", None)
        if isinstance(video_logging, (dict, DictConfig)):
            video_logging["capture"] = prev_capture

    def _debug_eval_checkpoint(self, eval_step, stage):
        # CODEX: optional per-rank breadcrumbs to identify the exact eval section that stalls.
        if not self.debug_eval_sync:
            return
        if int(eval_step) >= int(self.debug_eval_steps):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=torch.device(self.device))
        print(
            "[DaggerMobileMultiExp/CODEX eval_step] "
            f"rank={self.global_rank} expert_idx={self.expert_idx} expert_rank={self.expert_rank} "
            f"step={int(eval_step)} stage={stage}",
            flush=True,
        )

    def _debug_eval_action_snapshot(self, eval_step, step_actions):
        # CODEX: temporary eval-only crash diagnostic. This prints immediately
        # before PhysX sees the action, so we can separate bad policy outputs
        # from bad reset/scene state.
        if not (self.debug_eval_sync or self.eval_only):
            return
        if int(eval_step) >= int(self.debug_eval_steps):
            return

        def _tensor_summary(name, tensor):
            if tensor is None:
                return f"{name}=missing"
            with torch.no_grad():
                t = tensor.detach()
                finite = bool(torch.isfinite(t).all().item())
                tf = t.float()
                return (
                    f"{name}:finite={finite} "
                    f"min={float(tf.min().item()):.4g} "
                    f"max={float(tf.max().item()):.4g} "
                    f"mean={float(tf.mean().item()):.4g}"
                )

        def _action_slice_summary(name, start, end):
            # CODEX: aggregate action stats can hide one saturated subsystem.
            if step_actions.shape[-1] <= start:
                return f"{name}=missing"
            return _tensor_summary(name, step_actions[:, start:min(end, step_actions.shape[-1])])

        states = getattr(self.env, "states", {})
        progress = getattr(self.env, "progress_buf", None)
        reset_buf = getattr(self.env, "reset_buf", None)
        progress_text = "progress=missing"
        if progress is not None:
            progress_text = f"progress_min={int(progress.min().item())} progress_max={int(progress.max().item())}"
        reset_text = "reset_buf=missing"
        if reset_buf is not None:
            reset_text = f"reset_sum={int(reset_buf.sum().item())}"

        print(
            "[DaggerMobileMultiExp/CODEX eval_action] "
            f"rank={self.global_rank} expert_idx={self.expert_idx} expert_rank={self.expert_rank} "
            f"task={self.expert_task_name} step={int(eval_step)} "
            f"{_tensor_summary('action', step_actions)} "
            f"{_action_slice_summary('base_action', 0, 3)} "
            f"{_action_slice_summary('franka_action', 3, 10)} "
            f"{_action_slice_summary('hand_action', 10, 26)} "
            f"{_action_slice_summary('camera_action', 26, 32)} "
            f"{_tensor_summary('q', states.get('q'))} "
            f"{_tensor_summary('object_pos', states.get('object_pos'))} "
            f"{progress_text} {reset_text}",
            flush=True,
        )

    def _task_cfg_dir(self):
        # CODEX: load copied task config stacks from this checkout, not from an installed package path.
        return Path(__file__).resolve().parents[1] / "cfg" / "task"

    def _load_task_cfg_with_defaults(self, task_name, seen=None):
        # CODEX: small Hydra-defaults loader for rank-local task swaps inside an already-composed Hydra job.
        if seen is None:
            seen = set()
        task_name = str(task_name)
        if task_name in seen:
            raise ValueError(f"Recursive task defaults while loading {task_name}: {sorted(seen)}")
        seen.add(task_name)

        task_path = self._task_cfg_dir() / f"{task_name}.yaml"
        if not task_path.is_file():
            raise FileNotFoundError(f"Per-expert task config not found: {task_path}")

        task_cfg = OmegaConf.load(task_path)
        merged_cfg = OmegaConf.create()
        for entry in task_cfg.get("defaults", []):
            if isinstance(entry, str):
                if entry == "_self_":
                    continue
                default_name = entry.split("@", 1)[0]
                merged_cfg = OmegaConf.merge(
                    merged_cfg,
                    self._load_task_cfg_with_defaults(default_name, seen=seen),
                )
            elif isinstance(entry, dict):
                for _, default_name in entry.items():
                    if default_name in (None, "_self_"):
                        continue
                    merged_cfg = OmegaConf.merge(
                        merged_cfg,
                        self._load_task_cfg_with_defaults(default_name, seen=seen),
                    )

        if "defaults" in task_cfg:
            with open_dict(task_cfg):
                del task_cfg["defaults"]
        seen.remove(task_name)
        return OmegaConf.merge(merged_cfg, task_cfg)

    def _has_cfg_path(self, cfg, path):
        # CODEX: safe path probes let optional side-only config keys coexist with dex tasks.
        node = cfg
        for key in path.split("."):
            if not isinstance(node, DictConfig) or key not in node:
                return False
            node = node[key]
        return True

    def _get_cfg_path(self, cfg, path):
        # CODEX: read whitelisted per-expert override paths before task config replacement.
        node = cfg
        for key in path.split("."):
            node = node[key]
        return node

    def _set_cfg_path(self, cfg, path, value):
        # CODEX: write rank-local override values after the selected task config is loaded.
        node = cfg
        parts = path.split(".")
        for key in parts[:-1]:
            node = node[key]
        with open_dict(node):
            node[parts[-1]] = value

    def _snapshot_expert_overrides(self, cfg):
        # CODEX: only these paths are interpreted as per-expert lists; ordinary task list fields stay untouched.
        paths = (
            "teacher.cfg",
            "teacher.ckpt",
            "task.task.randomize",
            "task.cfg_override",
            "task.env.video_logging.capture",
            "task.env.video_logging.envs",
            "task.env.enableDebugVis",
            "task.env.enable_viser",
            "task.env.teacher_obs_action_frame",
            "task.env.mesh.mesh_dir",
            "task.env.mesh.variant_manifest_json",
            "task.env.grasp_guide_idx",
            "task.env.object_settings.mass_range",
            "task.env.object_teleport.enable",
            "task.env.object_wrench.enable",
            "task.env.scene.hdf5_path",
            "task.env.scene.teacher_bank_variation_json",
            "task.env.scene.teacher_bank_height_assignment_json",
            "task.env.scene.teacher_bank_height_bins",
            "task.env.verified_teacher_bank.enable",
            "task.env.verified_teacher_bank.hdf5_path",
            "task.env.verified_teacher_bank.capacity_per_env",
            "task.env.verified_teacher_bank.collection_region_filter",
            "task.env.verified_teacher_bank.sampling_probs.afar",
            "task.env.verified_teacher_bank.sampling_probs.near_recovery",
            "task.env.verified_teacher_bank.sampling_probs.far_recovery",
            "task.env.verified_teacher_bank.sampling_probs.failure_recovery",
        )
        return {path: self._get_cfg_path(cfg, path) for path in paths if self._has_cfg_path(cfg, path)}

    def _apply_expert_task_config(self, cfg, expert_idx):
        task_names = cfg.dagger.get("multi_task_names", None)
        if task_names is None:
            return
        selected_task_name = self._select_expert_value(task_names, expert_idx)
        if selected_task_name in (None, "", "null"):
            return

        selected_task_cfg = self._load_task_cfg_with_defaults(selected_task_name)
        if self._has_cfg_path(cfg, "task.type"):
            with open_dict(selected_task_cfg):
                selected_task_cfg.type = cfg.task.type
        with open_dict(cfg):
            cfg.task = selected_task_cfg
        print(
            f"[DaggerMobileMultiExp/CODEX] rank={self.global_rank} local_rank={self.local_rank} "
            f"expert_idx={expert_idx} task={selected_task_name}"
        )

    def _apply_expert_overrides(self, cfg, expert_idx):
        for path, value in self._expert_override_values.items():
            if not self._has_cfg_path(cfg, path):
                continue
            self._set_cfg_path(cfg, path, self._select_expert_value(value, expert_idx))

    def _apply_expert_rank_config(self, cfg):
        # CODEX: isolated side env reads this to slice variation/HDF5 shards per side expert only.
        if not self._has_cfg_path(cfg, "task.env"):
            return
        self._set_cfg_path(
            cfg,
            "task.env.multi_teacher_rank",
            {
                "local_rank": int(self.expert_rank),
                "global_rank": int(self.expert_rank),
                "world_size": int(self.expert_world_size),
                "launcher_local_rank": int(self.local_rank),
                "launcher_global_rank": int(self.global_rank),
                "launcher_world_size": int(self.world_size),
            },
        )

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
        if self.resume_wandb and "wandb_id" in checkpoint:
            self.wandb_id = checkpoint["wandb_id"]
            self.wandb_name = checkpoint["wandb_name"]
            self.wandb_project = checkpoint["wandb_project"]
        elif "wandb_id" in checkpoint:
            colorprint(  # CODEX
                "Checkpoint contains W&B metadata, but dagger.resume_wandb=False; starting a fresh W&B run.",
                color="yellow",
            )
        return checkpoint["train_success_rate_ep"]

    def _maybe_load_verified_teacher_bank(self):
        if not bool(getattr(self.env, "_verified_teacher_bank_enable", False)):
            return False
        if not hasattr(self.env, "load_verified_teacher_bank_hdf5"):
            raise RuntimeError("Env enables verified_teacher_bank but does not implement load_verified_teacher_bank_hdf5")

        bank_rank = int(getattr(self, "expert_rank", self.global_rank))  # CODEX: side banks use expert-local shards.
        loaded_verified_bank = self.env.load_verified_teacher_bank_hdf5(rank=bank_rank, strict=False)
        if (not self.multi_gpu) or (self.global_rank == 0) or loaded_verified_bank:
            shard_path = (
                self.env._get_verified_teacher_bank_shard_path(rank=bank_rank)
                if hasattr(self.env, "_get_verified_teacher_bank_shard_path")
                else "<unknown>"
            )
            print(
                "[DaggerMobile/verified_teacher_bank] "
                f"enable=True loaded={loaded_verified_bank} rank={bank_rank} path={shard_path}"
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

        for rollout_step in tqdm(range(self.steps_per_episode), desc=f"Training {self.episode+1}/{self.total_episodes}", \
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
                self._maybe_start_env_video(  # CODEX: side-style train video window, one rank per expert.
                    mode="train",
                    total_steps=min(int(self.env.max_episode_length * 2), int(self.steps_per_episode)),
                    periodic=True,
                )
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
        eval_steps = min(int(self.env.max_episode_length), int(self.steps_per_episode))  # CODEX: eval should respect task-local episode length.
        if self.debug_eval_sync:
            print(
                "[DaggerMobileMultiExp/CODEX eval] "
                f"rank={self.global_rank} expert_idx={self.expert_idx} expert_rank={self.expert_rank} "
                f"policy={policy_source} eval_steps={eval_steps} env_max={int(self.env.max_episode_length)} "
                f"launcher_steps={int(self.steps_per_episode)} start",
                flush=True,
            )
        legacy_eval_video_capture = self._disable_legacy_eval_video_if_needed()  # CODEX
        self._maybe_start_env_video(  # CODEX: side-style eval video window, one rank per expert.
            mode="eval",
            total_steps=eval_steps,
            periodic=False,
        )

        for eval_step in tqdm(range(eval_steps), desc="Evaluating", \
            ncols=None, dynamic_ncols=True, disable=(self.multi_gpu and self.global_rank != 0) ):

            if policy_source == "teacher":
                # CODEX: teacher-only eval should not depend on student inputs or student forward.
                for _ in range(self.chunk_size):
                    self._debug_eval_checkpoint(eval_step, "teacher_action_start")
                    step_actions = self._get_teacher_step_actions()
                    step_actions = torch.clamp(step_actions, -self.env.clip_actions, self.env.clip_actions)
                    self.env.progress_buf[:] = 0
                    self._debug_eval_checkpoint(eval_step, "teacher_env_step_start")
                    self.env.step(step_actions)
                    self._debug_eval_checkpoint(eval_step, "teacher_env_step_done")
                continue

            # get obs t_a0 for student, q_hand, rel_pcd
            self._debug_eval_checkpoint(eval_step, "obs_collection_start")
            q_robot = self.env.states['q'].clone() # (num_envs, 32)
            self.aux_anchor_base_pose7[:] = self.env.states['franka_base_pose7'].clone()  # CODEX
            self._refresh_aux_anchor_state(add_noise=False, use_initial_frame=self.aux_init_only)  # CODEX

            full_scene_pcd_t = self.env.combined_pcds # scene pcd + object pcd
            robot_pcd_t = self.env.robot_pcd_sampler.sample(q_robot, self.env.torchurdf_to_isaac_idx)
            self._debug_eval_checkpoint(eval_step, "obs_collection_done")

            # prepare pcd inputs
            obs_dict_a0 = OrderedDict([
                ("full_scene_pcd_t", full_scene_pcd_t),
                ("robot_pcd_t", robot_pcd_t),
            ])
            self._debug_eval_checkpoint(eval_step, "preprocess_start")
            obs_input_a0, _ = self.preprocess_inputs(obs_dict_a0)
            self._debug_eval_checkpoint(eval_step, "preprocess_done")

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

            self._debug_eval_checkpoint(eval_step, "student_forward_start")
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
            self._debug_eval_checkpoint(eval_step, "student_forward_done")

            # plot aux prediction in gui
            if not self.env.headless:
                self._plot_aux_prediction_in_viewer()

            for action_idx in range(self.chunk_size):
                student_actions = student_actions_chunk[:, action_idx, :]
                step_actions = student_actions
                # step with student actions
                step_actions = torch.clamp(step_actions, -self.env.clip_actions, self.env.clip_actions)

                self.env.progress_buf[:] = 0 # since we are only evaling one episode, just disable env resets, it might mess up loggings a little bit

                self._debug_eval_checkpoint(eval_step, "env_step_start")
                self._debug_eval_action_snapshot(eval_step, step_actions)  # CODEX
                self.env.step(step_actions)
                self._debug_eval_checkpoint(eval_step, "env_step_done")

        self._restore_legacy_eval_video(legacy_eval_video_capture)  # CODEX
        if self.debug_eval_sync:
            print(
                "[DaggerMobileMultiExp/CODEX eval] "
                f"rank={self.global_rank} expert_idx={self.expert_idx} expert_rank={self.expert_rank} "
                f"policy={policy_source} rollout_done",
                flush=True,
            )

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

    def _get_per_gpu_metric_logs(self):
        metric_keys = (
            "metrics/success_rate_5cm_per_step",
            "metrics/lifting_rate_5cm_per_step",
            "metrics/success_rate_5cm_per_ep",
            "metrics/lifting_rate_5cm_per_ep",
        )
        if not self.multi_gpu:
            return {}

        local_metrics = torch.tensor(
            [float(self.env.extras[key]) for key in metric_keys],
            device=self.device,
            dtype=torch.float32,
        )
        gathered_metrics = [torch.empty_like(local_metrics) for _ in range(self.world_size)]
        dist.all_gather(gathered_metrics, local_metrics)

        if int(self.expert_rank) != 0:
            return {}

        expert_global_ranks = [
            rank for rank in range(int(self.world_size))
            if (rank % int(self.num_experts)) == int(self.expert_idx)
        ]
        expert_metrics = torch.stack([gathered_metrics[rank] for rank in expert_global_ranks], dim=0)
        expert_mean_metrics = torch.mean(expert_metrics, dim=0)

        per_gpu_logs = {}
        for key, value in zip(metric_keys, expert_mean_metrics):
            # CODEX: make the main success/lift curves task-level means for the one W&B run per task.
            per_gpu_logs[key] = value.item()
            per_gpu_logs[f"{key}/expert_mean"] = value.item()
        for rank in expert_global_ranks:
            rank_metrics = gathered_metrics[rank]
            for key, value in zip(metric_keys, rank_metrics):
                per_gpu_logs[f"{key}/gpu_{rank}"] = value.item()
        return per_gpu_logs

    def _aggregate_expert_scalar_logs(self, logs):
        # CODEX: one W&B run per task should show task-level eval means, not only
        # the local logging GPU. All ranks still call all_gather to avoid DDP hangs.
        if not self.multi_gpu:
            return logs

        scalar_items = [
            (key, float(value))
            for key, value in logs.items()
            if isinstance(value, (int, float))
        ]
        if not scalar_items:
            return logs if int(self.expert_rank) == 0 else {}

        keys = [key for key, _ in scalar_items]
        local_values = torch.tensor(
            [value for _, value in scalar_items],
            device=self.device,
            dtype=torch.float32,
        )
        gathered_values = [torch.empty_like(local_values) for _ in range(self.world_size)]
        if self.debug_eval_sync:
            print(
                "[DaggerMobileMultiExp/CODEX eval] "
                f"rank={self.global_rank} expert_idx={self.expert_idx} expert_rank={self.expert_rank} "
                f"enter_eval_all_gather num_scalars={len(scalar_items)}",
                flush=True,
            )
        dist.all_gather(gathered_values, local_values)
        if self.debug_eval_sync:
            print(
                "[DaggerMobileMultiExp/CODEX eval] "
                f"rank={self.global_rank} expert_idx={self.expert_idx} expert_rank={self.expert_rank} "
                "exit_eval_all_gather",
                flush=True,
            )

        if int(self.expert_rank) != 0:
            return {}

        expert_global_ranks = [
            rank for rank in range(int(self.world_size))
            if (rank % int(self.num_experts)) == int(self.expert_idx)
        ]
        expert_values = torch.stack([gathered_values[rank] for rank in expert_global_ranks], dim=0)
        expert_mean_values = torch.mean(expert_values, dim=0)

        aggregated_logs = {}
        for key, value in zip(keys, expert_mean_values):
            aggregated_logs[key] = value.item()
            aggregated_logs[f"{key}/expert_mean"] = value.item()
        for rank in expert_global_ranks:
            rank_values = gathered_values[rank]
            for key, value in zip(keys, rank_values):
                aggregated_logs[f"{key}/gpu_{rank}"] = value.item()
        return aggregated_logs

    def train(self):
        if self.teacher_eval_only:
            while True:
                eval_wandb_logs = self.eval(policy_source="teacher")  # CODEX
                eval_wandb_logs = self._aggregate_expert_scalar_logs(eval_wandb_logs)  # CODEX
                self.total_steps += int(self.env.max_episode_length)
                if self.use_wandb:  # CODEX: only expert_rank 0 owns the task W&B run.
                    wandb.log(eval_wandb_logs, step=self.total_steps)
                self.episode += 1
            return
        if self.eval_only:
            while True:
                eval_wandb_logs = self.eval(policy_source="student")  # CODEX
                eval_wandb_logs = self._aggregate_expert_scalar_logs(eval_wandb_logs)  # CODEX
                self.total_steps += int(self.env.max_episode_length)
                if self.use_wandb:  # CODEX
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
                eval_wandb_logs = self._aggregate_expert_scalar_logs(eval_wandb_logs)  # CODEX

            per_gpu_metric_logs = self._get_per_gpu_metric_logs()

            episode_time = time.time() - start_time  # CODEX: log local timing for every expert run.
            estimated_finish_time = start_time + episode_time * remaining_episodes

            metrics["train/loss_episode"] = train_loss
            metrics["time/episode_time"] = episode_time
            metrics["episode"] = self.episode
            metrics["train/teacher_forcing_prop"] = self.teacher_forcing_prop
            metrics["expert/rank"] = self.global_rank  # CODEX
            metrics["expert/local_rank"] = self.local_rank  # CODEX
            metrics["expert/index"] = self.expert_idx  # CODEX
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
            if (not self.multi_gpu) or (int(self.expert_rank) == 0):
                metrics.update(per_gpu_metric_logs)
            if eval_policy:
                metrics.update(eval_wandb_logs)

            if self.use_wandb:
                wandb.log(metrics, step=self.total_steps)

            if (not self.multi_gpu) or (self.global_rank == 0):
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
