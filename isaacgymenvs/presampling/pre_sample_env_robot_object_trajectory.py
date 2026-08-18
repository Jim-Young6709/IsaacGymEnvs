import copy
import json
import os
import time
from typing import Dict

import h5py
import hydra
import isaacgym
import isaacgymenvs
import numpy as np
import torch
import torch.distributed as dist
import yaml
from isaacgymenvs.tasks import FrankaLEAPMobile
from isaacgymenvs.utils.rotation_conversions import quaternion_to_matrix_ig
from isaacgymenvs.utils.training_utils import *
from omegaconf import DictConfig, OmegaConf
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder
from tqdm import tqdm


def _to_numpy(value, dtype=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype)
    return value


def _jsonify(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _write_string(group, name, value):
    dtype = h5py.string_dtype(encoding="utf-8")
    dataset = group.create_dataset(name, shape=(), dtype=dtype)
    dataset[()] = "" if value is None else str(value)


def _write_json(group, name, value):
    _write_string(group, name, json.dumps(_jsonify(value), sort_keys=True))


def _write_file_text_if_exists(group, name, path):
    if not path or not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        _write_string(group, name, f.read())


def _write_dataset(group, name, value, compression=True):
    data = _to_numpy(value)
    kwargs = {}
    if compression and data.size > 32:
        kwargs["compression"] = "gzip"
    group.create_dataset(name, data=data, **kwargs)


def install_mesh_metadata_hooks():
    if getattr(FrankaLEAPMobile, "_trajectory_metadata_hooks_installed", False):
        return

    original_create_mesh_urdf = FrankaLEAPMobile._create_mesh_urdf
    original_create_mesh = FrankaLEAPMobile._create_mesh

    def _create_mesh_urdf_with_metadata(self, mesh_path, scale=[1.0, 1.0, 1.0], mass=0.5):
        urdf_rel, asset_root = original_create_mesh_urdf(
            self,
            mesh_path,
            scale=scale,
            mass=mass,
        )

        scale_array = np.asarray(scale, dtype=np.float32).reshape(-1)
        if scale_array.shape[0] == 1:
            scale_array = np.repeat(scale_array, 3)

        self._trajectory_last_mesh_urdf_info = {
            "mesh_path": os.path.abspath(mesh_path),
            "urdf_relative_path": urdf_rel,
            "asset_root": os.path.abspath(asset_root),
            "urdf_path": os.path.abspath(os.path.join(asset_root, urdf_rel)),
            "scale_xyz": scale_array.astype(np.float32),
            "mass": np.nan if mass is None else float(mass),
        }
        return urdf_rel, asset_root

    def _create_mesh_with_metadata(
        self,
        mesh_path,
        pos,
        scale,
        quat=[0, 0, 0, 1],
        fix_base_link=True,
        obj_str2int=None,
        asset_obj_id=None,
        asset_mesh_id=None,
    ):
        asset, start_pose, returned_scale, returned_obj_id, returned_mesh_id = original_create_mesh(
            self,
            mesh_path,
            pos,
            scale,
            quat=quat,
            fix_base_link=fix_base_link,
            obj_str2int=obj_str2int,
            asset_obj_id=asset_obj_id,
            asset_mesh_id=asset_mesh_id,
        )

        if not hasattr(self, "_trajectory_mesh_asset_metadata"):
            self._trajectory_mesh_asset_metadata = []

        metadata = copy.deepcopy(getattr(self, "_trajectory_last_mesh_urdf_info", {}))
        metadata.update(
            {
                "asset_obj_id": returned_obj_id,
                "asset_mesh_id": returned_mesh_id,
                "start_position": np.asarray(pos, dtype=np.float32),
                "start_quaternion_xyzw": np.asarray(quat, dtype=np.float32),
                "returned_scale": np.asarray(returned_scale, dtype=np.float32),
                "fix_base_link": bool(fix_base_link),
            }
        )
        self._trajectory_mesh_asset_metadata.append(metadata)
        return asset, start_pose, returned_scale, returned_obj_id, returned_mesh_id

    FrankaLEAPMobile._create_mesh_urdf = _create_mesh_urdf_with_metadata
    FrankaLEAPMobile._create_mesh = _create_mesh_with_metadata
    FrankaLEAPMobile._trajectory_metadata_hooks_installed = True


class EpisodeTrajectory:
    def __init__(self, env):
        self.env = env
        self.frames = []
        self.steps = []

    def append_frame(self):
        env = self.env
        eef_pose = torch.cat([env.states["eef_pos"], env.states["eef_quat"]], dim=-1)
        object_pose = torch.cat([env.states["object_pos"], env.states["object_quat"]], dim=-1)

        frame = {
            "robot_joint_pos": _to_numpy(env.states["q"], np.float32),
            "robot_joint_vel": _to_numpy(env.states["qd"], np.float32),
            "robot_eef_pose7_xyzw": _to_numpy(eef_pose, np.float32),
            "object_root_state": _to_numpy(env._object_state, np.float32),
            "object_pose7_xyzw": _to_numpy(object_pose, np.float32),
            "object_center_pos": _to_numpy(env.states["object_center_pos"], np.float32),
            "success_flags": _to_numpy(env.success_flags, np.uint8),
            "lifting_flags": _to_numpy(env.lifting_flags, np.uint8),
            "success_5cm_per_step": _to_numpy(env.success_5cm_per_step, np.uint8),
            "lifting_5cm_per_step": _to_numpy(env.lifting_5cm_per_step, np.uint8),
            "reset_buf": _to_numpy(env.reset_buf, np.uint8),
            "progress_buf": _to_numpy(env.progress_buf, np.int32),
        }

        if "camera_pose7" in env.states:
            frame["camera_pose7_xyzw"] = _to_numpy(env.states["camera_pose7"], np.float32)
        if "lidar_pose7" in env.states:
            frame["lidar_pose7_xyzw"] = _to_numpy(env.states["lidar_pose7"], np.float32)
        if "franka_base_pose7" in env.states:
            frame["franka_base_pose7_xyzw"] = _to_numpy(env.states["franka_base_pose7"], np.float32)

        self.frames.append(frame)

    def append_step(self, policy_action, converted_teacher_action, applied_action, abs_dof_target):
        self.steps.append(
            {
                "policy_action": _to_numpy(policy_action, np.float32),
                "converted_teacher_action": _to_numpy(converted_teacher_action, np.float32),
                "applied_action": _to_numpy(applied_action, np.float32),
                "abs_dof_position_target": _to_numpy(abs_dof_target, np.float32),
            }
        )

    def stack_frames(self, key, env_idx):
        return np.stack([frame[key][env_idx] for frame in self.frames], axis=0)

    def stack_steps(self, key, env_idx):
        if not self.steps:
            return np.empty((0,), dtype=np.float32)
        return np.stack([step[key][env_idx] for step in self.steps], axis=0)


class PresampleEnvRobotObjectTrajectory:
    def __init__(self, cfg):
        install_mesh_metadata_hooks()

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
                cfg.task.env.video_logging.capture = False
                cfg.graphics_device_id = -1
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1

        self.cfg = cfg
        self.episode = 0
        self.total_steps = 0
        self.total_episodes = 1
        self.steps_per_episode = cfg.presample.steps_per_episode
        self.output_hdf5_dir = cfg.presample.output_hdf5_dir
        legacy_output_hdf5_name = cfg.presample.get("output_hdf5_name", "presampled_init_poses.hdf5")
        default_output_hdf5_name = (
            "presampled_env_robot_object_trajectories.hdf5"
            if legacy_output_hdf5_name == "presampled_init_poses.hdf5"
            else legacy_output_hdf5_name
        )
        self.output_hdf5_name = cfg.presample.get(
            "output_trajectory_hdf5_name",
            default_output_hdf5_name,
        )
        self.device = cfg["sim_device"]
        self.seed = cfg.seed
        self.exp_name = cfg.experiment
        self.demo_count = 0
        self.total_possible_trajectories = 0
        self.output_file = None
        self.data_group = None
        self.temp_output_path = None

        set_seed_and_precision(self.seed)

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
            return envs

        self.env = create_isaacgym_env()
        self.env.reset()

        self.env.distillation_mode = True
        self.env.delta_franka_action = self.cfg.action_space.delta_franka_action
        self.env.delta_leap_action = self.cfg.action_space.delta_leap_action
        self.env.delta_arx_action = self.cfg.action_space.delta_arx_action

        self.value_size = 1
        self.num_seqs = 1
        self.normalize_value = self.cfg.train.params.config.normalize_value
        self.normalize_input = self.cfg.train.params.config.normalize_input
        self.teacher_model_config = {
            "actions_num": self.env.num_actions,
            "input_shape": (self.env.num_observations,),
            "num_seqs": self.num_seqs,
            "value_size": self.value_size,
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }
        self.teacher_network_params = self.load_param_dict(self.cfg["teacher"]["cfg"])["params"]
        self.teacher_network = self.load_networks(self.teacher_network_params)
        self.teacher_model = self.teacher_network.build(self.teacher_model_config).to(self.device)
        self.set_weights(self.cfg["teacher"]["ckpt"])
        self.teacher_model.eval()
        self.is_teacher_rnn = self.teacher_model.is_rnn()

    def load_param_dict(self, cfg_path) -> Dict:
        base_dir = os.path.dirname(__file__)
        full_path = os.path.join(base_dir, cfg_path)

        with open(full_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def load_networks(self, params):
        builder = ModelBuilder()
        return builder.load(params)

    def set_weights(self, ckpt):
        weights = torch_ext.load_checkpoint(ckpt)
        model = self.teacher_model
        model.load_state_dict(weights["model"])
        if self.normalize_input and "running_mean_std" in weights:
            model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def sample_rand_init_pose(self):
        self.cfg_base_init_range = self.cfg.presample.rand_cfg.base_init_range
        self.cfg_franka_canonical = [0.0, -0.25 * np.pi, 0.0, -0.75 * np.pi, 0.0, 0.5 * np.pi, 0.0]
        self.cfg_franka_noise = 0.5
        self.cfg_leap_canonical = [
            0.7, -0.2, 0.7, 0.7,
            0.8, 1.57, 0.77, 0.9,
            0.65, 0.0, 0.65, 0.65,
            0.7, 0.2, 0.7, 0.7,
        ]
        self.cfg_leap_noise = 0.5
        self.cfg_arx_canonical = [0.0, 1.0, 2.0, -1.0, 0.0, 0.0]
        self.cfg_arx_noise = 0.5
        self.cfg_franka_full = False
        self.cfg_leap_full = False
        self.cfg_arx_full = False

        base_init_range = torch.tensor(self.cfg_base_init_range, device=self.device)
        base_init_pose = torch.rand((self.env.num_envs, 3), device=self.device)
        base_init_pose = base_init_pose * (base_init_range[1] - base_init_range[0]) + base_init_range[0]
        base_init_pose[:, 1] += getattr(self.env, "box_pos", torch.zeros_like(base_init_pose))[:, 1]

        if self.cfg_franka_full:
            franka_init_config_normalized = 2 * (torch.rand((self.env.num_envs, 7), device=self.device) - 0.5)
            franka_init_config = self.env.unnormalize_robot_joints(franka_init_config_normalized, "franka", delta=False)
        else:
            franka_init_config = torch.tensor([self.cfg_franka_canonical] * self.env.num_envs, device=self.device)
            franka_init_config += (torch.rand_like(franka_init_config, device=self.device) - 0.5) * 2 * self.cfg_franka_noise

        if self.cfg_leap_full:
            leap_init_config_normalized = 2 * (torch.rand((self.env.num_envs, 16), device=self.device) - 0.5)
            leap_init_config = self.env.unnormalize_robot_joints(leap_init_config_normalized, "leap", delta=False)
        else:
            leap_init_config = torch.tensor([self.cfg_leap_canonical] * self.env.num_envs, device=self.device)
            leap_init_config += (torch.rand_like(leap_init_config, device=self.device) - 0.5) * 2 * self.cfg_leap_noise

        if self.cfg.presample.assume_obj_in_view_t0:
            arx_init_config = torch.tensor([self.cfg_arx_canonical] * self.env.num_envs, device=self.device)
        else:
            if self.cfg_arx_full:
                arx_init_config_normalized = 2 * (torch.rand((self.env.num_envs, 6), device=self.device) - 0.5)
                arx_init_config = self.env.unnormalize_robot_joints(arx_init_config_normalized, "arx", delta=False)
            else:
                arx_init_config = torch.tensor([self.cfg_arx_canonical] * self.env.num_envs, device=self.device)
                arx_init_config += (torch.rand_like(arx_init_config, device=self.device) - 0.5) * 2 * self.cfg_arx_noise

        sampled_init_pose = torch.cat([base_init_pose, franka_init_config, leap_init_config, arx_init_config], dim=-1)
        return sampled_init_pose

    def sample_valid_init_pose(self):
        final_init_pose = torch.zeros((self.env.num_envs, 32), device=self.device)
        validation_mask = torch.zeros((self.env.num_envs,), dtype=torch.bool, device=self.device)

        self.env.reset_idx()

        while True:
            sampled_init_pose = self.sample_rand_init_pose()
            final_init_pose[~validation_mask] = sampled_init_pose[~validation_mask]

            self.env.set_robot_joint_state(final_init_pose)
            self.env.step_sim_multi(1, False)
            self.env.compute_observations()

            validation_mask[:] = ~self.env.env_collision[:].bool()

            if torch.all(validation_mask):
                print("All envs have collision free initial pose!")
                self.env.default_reset_joint_config[:] = final_init_pose.clone()
                break

        if self.cfg.presample.assume_obj_in_view_t0:
            self.adjust_vision_arm_pose()
            self.env.default_reset_joint_config[:, 26:32] = self.env.states["q"][:, 26:32].clone()
            self.env.reset_idx()
            self.env.compute_observations()

    def adjust_vision_arm_pose(self, gaze_err_tol_deg=5.0):
        gaze_err_tol_rad = torch.deg2rad(torch.tensor(gaze_err_tol_deg, device=self.device))
        dummy_teacher_actions = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device)

        for step_idx in range(self.steps_per_episode):
            gaze_target = self.env.box_pos.clone()
            self.env._pre_physics_step_teacher(dummy_teacher_actions, gaze_target)
            teacher_actions = self.env.teacher_actions_converted.clone()
            step_actions = torch.clamp(teacher_actions, -self.env.clip_actions, self.env.clip_actions)
            step_actions[:, :26] = 0.0
            self.env.step(step_actions)

            camera_pose7 = self.env.states["camera_pose7"]
            camera_pos = camera_pose7[:, :3]
            camera_quat = camera_pose7[:, 3:7]

            camera_rot_mat = quaternion_to_matrix_ig(camera_quat)
            camera_forward_world = camera_rot_mat[:, :, 0]

            cam_to_target = gaze_target - camera_pos
            cam_to_target = cam_to_target / cam_to_target.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            cos_gaze_err = (camera_forward_world * cam_to_target).sum(dim=-1).clamp(-1.0, 1.0)
            gaze_err_rad = torch.acos(cos_gaze_err)

            if torch.all(gaze_err_rad <= gaze_err_tol_rad):
                print(
                    f"adjust_vision_arm_pose succeeded in {step_idx} steps "
                    f"with max gaze error {torch.rad2deg(gaze_err_rad).max().item():.2f} deg"
                )
                break

            if step_idx == self.steps_per_episode - 1:
                max_err_deg = torch.rad2deg(gaze_err_rad).max().item()
                colorprint(
                    f"adjust_vision_arm_pose reached max steps ({self.steps_per_episode}) "
                    f"with max gaze error {max_err_deg:.2f} deg",
                    color="yellow",
                )

    def _open_output_hdf5(self):
        os.makedirs(self.output_hdf5_dir, exist_ok=True)
        name, ext = os.path.splitext(self.output_hdf5_name)
        rank_suffix = f"_rank_{self.global_rank}" if self.multi_gpu else ""
        self.temp_output_path = os.path.join(
            self.output_hdf5_dir,
            f"{name}{rank_suffix}_tmp_{os.getpid()}{ext}",
        )
        self.output_file = h5py.File(self.temp_output_path, "w")
        self.data_group = self.output_file.create_group("data")
        metadata_group = self.output_file.create_group("metadata")
        _write_string(metadata_group, "script", os.path.basename(__file__))
        _write_string(metadata_group, "schema_version", "env_robot_object_trajectory_v1")
        _write_string(metadata_group, "created_time", time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            _write_string(metadata_group, "hydra_config_yaml", OmegaConf.to_yaml(self.cfg, resolve=True))
        except Exception as exc:
            _write_string(metadata_group, "hydra_config_yaml_error", str(exc))

    def _final_output_path(self, final_success_rate):
        dir_name, file_name = os.path.split(os.path.join(self.output_hdf5_dir, self.output_hdf5_name))
        name, ext = os.path.splitext(file_name)
        rank_suffix = f"_rank_{self.global_rank}" if self.multi_gpu else ""
        output_path = os.path.join(
            dir_name,
            f"{name}{rank_suffix}_sr_{final_success_rate:.4f}_num_{self.demo_count}{ext}",
        )
        if not os.path.exists(output_path):
            return output_path

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(
            dir_name,
            f"{name}{rank_suffix}_sr_{final_success_rate:.4f}_num_{self.demo_count}_{timestamp}{ext}",
        )

    def _close_output_hdf5(self, final_success_rate, success_rate_per_env):
        self.data_group.attrs["total_samples"] = self.demo_count
        self.data_group.attrs["env_args"] = json.dumps(
            {
                "task_name": str(self.cfg.task_name),
                "num_envs": int(self.env.num_envs),
                "steps_per_episode": int(self.steps_per_episode),
                "total_episodes": int(self.total_episodes),
            },
            sort_keys=True,
        )
        self.output_file.attrs["presample_success_rate"] = final_success_rate
        self.output_file.attrs["num_valid_samples"] = self.demo_count
        self.output_file.attrs["total_possible_trajectories"] = self.total_possible_trajectories
        self.output_file.attrs["postprocessed"] = True
        self.output_file.create_dataset(
            "success_rate_per_env",
            data=_to_numpy(success_rate_per_env, np.float32),
            compression="gzip",
        )
        self.output_file.close()

        output_path = self._final_output_path(final_success_rate)
        os.replace(self.temp_output_path, output_path)
        self.temp_output_path = output_path
        print(
            f"Saved {self.demo_count} successful trajectories to {output_path} "
            f"with sampling success rate {final_success_rate:.4f}"
        )

    def sample_episode(self):
        episode_traj = EpisodeTrajectory(self.env)
        episode_traj.append_frame()

        iterator = tqdm(
            range(self.steps_per_episode),
            desc=f"Presampling {self.episode + 1}/{self.total_episodes}",
            ncols=None,
            dynamic_ncols=True,
            disable=(self.multi_gpu and self.global_rank != 0),
        )
        for _ in iterator:
            self.total_steps += 1

            teacher_obs = self.env.obs_buf.clone()
            batch_dict = {
                "is_train": False,
                "obs": teacher_obs,
                "prev_actions": None,
            }

            with torch.no_grad():
                res_dict = self.teacher_model(batch_dict)

            mu = res_dict["mus"]
            self.states = res_dict["rnn_states"]
            teacher_policy_action = mu
            teacher_policy_action = torch.clamp(
                teacher_policy_action,
                -self.env.clip_actions,
                self.env.clip_actions,
            )

            self.env._pre_physics_step_teacher(teacher_policy_action)
            converted_teacher_action = self.env.teacher_actions_converted.clone()
            step_actions = torch.clamp(
                converted_teacher_action,
                -self.env.clip_actions,
                self.env.clip_actions,
            )

            self.env.distillation_steps = self.total_steps
            self.env.step(step_actions)
            episode_traj.append_step(
                policy_action=teacher_policy_action,
                converted_teacher_action=converted_teacher_action,
                applied_action=step_actions,
                abs_dof_target=self.env.abs_actions.clone(),
            )
            episode_traj.append_frame()

        return episode_traj

    def _write_obstacles(self, demo_group, env_idx):
        obstacles_group = demo_group.create_group("obstacles")
        cuboids_group = obstacles_group.create_group("cuboids")

        cuboid_dims, cuboid_centers, cuboid_quats, *_ = self.env.obstacle_configs[env_idx]
        cuboid_dims = np.asarray(cuboid_dims, dtype=np.float32)
        cuboid_centers = np.asarray(cuboid_centers, dtype=np.float32)
        cuboid_quats = np.asarray(cuboid_quats, dtype=np.float32)

        _write_dataset(cuboids_group, "dimensions_xyz", cuboid_dims)
        _write_dataset(cuboids_group, "position_xyz", cuboid_centers)
        _write_dataset(cuboids_group, "quaternion_xyzw", cuboid_quats)
        _write_dataset(cuboids_group, "pose_xyz_quat_xyzw", np.concatenate([cuboid_centers, cuboid_quats], axis=-1))
        cuboids_group.attrs["num_cuboids"] = int(cuboid_dims.shape[0])

    def _write_environment(self, demo_group, env_idx):
        env_group = demo_group.create_group("environment")
        env_group.attrs["source_env_idx"] = int(env_idx)
        env_group.attrs["source_batch_idx"] = int(self.cfg.task.env.scene.batch_idx)
        _write_string(env_group, "source_scene_hdf5", self.cfg.task.env.scene.hdf5_path)

        source_demo = dict(self.env.batch[env_idx])
        for key in ["states", "compartment_states", "init_robot_states", "mesh_idx"]:
            if key in source_demo:
                _write_dataset(env_group, key, source_demo[key])
                if key in ["states", "compartment_states", "init_robot_states", "mesh_idx"]:
                    _write_dataset(demo_group, key, source_demo[key])

        if hasattr(self.env, "box_dims"):
            _write_dataset(env_group, "box_dimensions_xyz", self.env.box_dims[env_idx].detach().cpu().numpy())
        if hasattr(self.env, "box_pos"):
            _write_dataset(env_group, "box_bottom_position_xyz", self.env.box_pos[env_idx].detach().cpu().numpy())
        if hasattr(self.env, "box_quats"):
            _write_dataset(env_group, "box_quaternion_xyzw", self.env.box_quats[env_idx].detach().cpu().numpy())
        if hasattr(self.env, "table_pos"):
            _write_dataset(env_group, "table_position_xyz", self.env.table_pos[env_idx].detach().cpu().numpy())
        if hasattr(self.env, "table_size"):
            _write_dataset(env_group, "table_dimensions_xyz", self.env.table_size[env_idx].detach().cpu().numpy())

    def _get_compartment_state(self, env_idx):
        source_demo = dict(self.env.batch[env_idx])
        if "compartment_states" in source_demo:
            compartment = np.asarray(source_demo["compartment_states"], dtype=np.float32)
            if compartment.ndim > 1:
                compartment = compartment[0]
            compartment = compartment.reshape(-1)
            if compartment.shape[0] >= 10:
                size = compartment[:3]
                position = compartment[3:6]
                quaternion = compartment[6:10]
                bottom_position = position.copy()
                bottom_position[2] -= size[2] / 2.0
                return size, position, quaternion, bottom_position

        if not all(hasattr(self.env, attr) for attr in ("box_dims", "box_pos", "box_quats")):
            return None

        size = self.env.box_dims[env_idx].detach().cpu().numpy().astype(np.float32)
        bottom_position = self.env.box_pos[env_idx].detach().cpu().numpy().astype(np.float32)
        quaternion = self.env.box_quats[env_idx].detach().cpu().numpy().astype(np.float32)
        position = bottom_position.copy()
        position[2] += size[2] / 2.0
        return size, position, quaternion, bottom_position

    def _write_compartment_state(self, demo_group, env_idx):
        compartment_state = self._get_compartment_state(env_idx)
        if compartment_state is None:
            return

        size, position, quaternion, bottom_position = compartment_state
        compartment_group = demo_group.create_group("compartment_state")
        _write_dataset(compartment_group, "size_xyz", size)
        _write_dataset(compartment_group, "position_xyz", position)
        _write_dataset(compartment_group, "quaternion_xyzw", quaternion)
        _write_dataset(compartment_group, "bottom_position_xyz", bottom_position)
        _write_dataset(compartment_group, "pose_xyz_quat_xyzw", np.concatenate([position, quaternion], axis=0))
        _write_json(
            compartment_group,
            "metadata_json",
            {
                "size_xyz": size,
                "position_xyz": position,
                "quaternion_xyzw": quaternion,
                "bottom_position_xyz": bottom_position,
                "position_convention": "center",
                "quaternion_convention": "xyzw",
            },
        )

    def _write_robot_info(self, demo_group):
        robot_group = demo_group.create_group("robot_info")
        _write_string(robot_group, "asset_root", getattr(self.env, "asset_root", ""))
        _write_string(robot_group, "asset_file", getattr(self.env, "robot_asset_file", ""))
        robot_urdf_path = getattr(self.env, "full_robot_asset_path", "")
        _write_string(robot_group, "urdf_path", robot_urdf_path)
        _write_file_text_if_exists(robot_group, "urdf_text", robot_urdf_path)

        dof_names = self.env.gym.get_actor_dof_names(self.env.envs[0], self.env.robots[0])
        body_names = self.env.gym.get_actor_rigid_body_names(self.env.envs[0], self.env.robots[0])
        string_dtype = h5py.string_dtype(encoding="utf-8")
        robot_group.create_dataset("dof_names", data=np.asarray(dof_names, dtype=object), dtype=string_dtype)
        robot_group.create_dataset("rigid_body_names", data=np.asarray(body_names, dtype=object), dtype=string_dtype)
        _write_dataset(robot_group, "dof_lower_limits", self.env.robot_dof_lower_limits.detach().cpu().numpy())
        _write_dataset(robot_group, "dof_upper_limits", self.env.robot_dof_upper_limits.detach().cpu().numpy())

    def _object_metadata_for_env(self, env_idx):
        mesh_idx = int(self.env.mesh_indices[env_idx])
        metadata = {
            "mesh_idx": mesh_idx,
            "mesh_args": OmegaConf.to_container(self.cfg.task.env.mesh, resolve=True),
        }

        mesh_metadata = getattr(self.env, "_trajectory_mesh_asset_metadata", [])
        if 0 <= mesh_idx < len(mesh_metadata):
            metadata.update(mesh_metadata[mesh_idx])

        if hasattr(self.env, "mesh_aabb_extents"):
            metadata["mesh_aabb_extents"] = self.env.mesh_aabb_extents[env_idx].detach().cpu().numpy()
        if hasattr(self.env, "object_id_to_name"):
            metadata["object_id_to_name"] = getattr(self.env, "object_id_to_name")

        return metadata

    def _write_object_info(self, demo_group, env_idx):
        object_group = demo_group.create_group("object_info")
        metadata = self._object_metadata_for_env(env_idx)

        numeric_keys = {
            "mesh_idx",
            "asset_obj_id",
            "scale_xyz",
            "returned_scale",
            "start_position",
            "start_quaternion_xyzw",
            "mesh_aabb_extents",
            "mass",
        }
        string_keys = {
            "mesh_path",
            "urdf_relative_path",
            "asset_root",
            "urdf_path",
            "asset_mesh_id",
        }

        for key in sorted(numeric_keys):
            if key in metadata:
                _write_dataset(object_group, key, metadata[key])

        for key in sorted(string_keys):
            if key in metadata:
                _write_string(object_group, key, metadata[key])

        _write_file_text_if_exists(object_group, "urdf_text", metadata.get("urdf_path", ""))
        _write_json(object_group, "metadata_json", metadata)
        if hasattr(self.env, "object_pcds"):
            _write_dataset(object_group, "object_pcd_local", self.env.object_pcds[env_idx].detach().cpu().numpy())

    def _write_robot_trajectory(self, demo_group, episode_traj, env_idx):
        robot_group = demo_group.create_group("robot_trajectory")
        _write_dataset(robot_group, "joint_position", episode_traj.stack_frames("robot_joint_pos", env_idx))
        _write_dataset(robot_group, "joint_velocity", episode_traj.stack_frames("robot_joint_vel", env_idx))
        _write_dataset(robot_group, "eef_pose7_xyzw", episode_traj.stack_frames("robot_eef_pose7_xyzw", env_idx))
        _write_dataset(robot_group, "base_pose_xyyaw", episode_traj.stack_frames("robot_joint_pos", env_idx)[:, :3])
        _write_dataset(robot_group, "policy_action", episode_traj.stack_steps("policy_action", env_idx))
        _write_dataset(robot_group, "converted_teacher_action", episode_traj.stack_steps("converted_teacher_action", env_idx))
        _write_dataset(robot_group, "applied_action", episode_traj.stack_steps("applied_action", env_idx))
        _write_dataset(robot_group, "abs_dof_position_target", episode_traj.stack_steps("abs_dof_position_target", env_idx))

        for optional_key in ["camera_pose7_xyzw", "lidar_pose7_xyzw", "franka_base_pose7_xyzw"]:
            if optional_key in episode_traj.frames[0]:
                _write_dataset(robot_group, optional_key, episode_traj.stack_frames(optional_key, env_idx))

    def _write_object_trajectory(self, demo_group, episode_traj, env_idx):
        object_group = demo_group.create_group("object_trajectory")
        root_state = episode_traj.stack_frames("object_root_state", env_idx)
        _write_dataset(object_group, "root_state", root_state)
        _write_dataset(object_group, "pose7_xyzw", episode_traj.stack_frames("object_pose7_xyzw", env_idx))
        _write_dataset(object_group, "position_xyz", root_state[:, :3])
        _write_dataset(object_group, "quaternion_xyzw", root_state[:, 3:7])
        _write_dataset(object_group, "linear_velocity", root_state[:, 7:10])
        _write_dataset(object_group, "angular_velocity", root_state[:, 10:13])
        _write_dataset(object_group, "center_position_xyz", episode_traj.stack_frames("object_center_pos", env_idx))

    def _write_episode_flags(self, demo_group, episode_traj, env_idx):
        flags_group = demo_group.create_group("episode_flags")
        success_per_step = episode_traj.stack_frames("success_5cm_per_step", env_idx).reshape(-1)
        _write_dataset(flags_group, "success_flags", episode_traj.stack_frames("success_flags", env_idx))
        _write_dataset(flags_group, "lifting_flags", episode_traj.stack_frames("lifting_flags", env_idx))
        _write_dataset(flags_group, "success_5cm_per_step", success_per_step)
        _write_dataset(flags_group, "lifting_5cm_per_step", episode_traj.stack_frames("lifting_5cm_per_step", env_idx))
        _write_dataset(flags_group, "reset_buf", episode_traj.stack_frames("reset_buf", env_idx))
        _write_dataset(flags_group, "progress_buf", episode_traj.stack_frames("progress_buf", env_idx))

        success_indices = np.flatnonzero(success_per_step)
        demo_group.attrs["success_step"] = int(success_indices[0]) if success_indices.size > 0 else -1

    def save_successful_env_trajectory(self, episode_traj, episode_idx, env_idx):
        demo_name = f"demo_{self.demo_count}"
        demo_group = self.data_group.create_group(demo_name)
        demo_group.attrs["episode_idx"] = int(episode_idx)
        demo_group.attrs["source_env_idx"] = int(env_idx)
        demo_group.attrs["num_steps"] = int(len(episode_traj.steps))
        demo_group.attrs["num_frames"] = int(len(episode_traj.frames))
        demo_group.attrs["success"] = True

        self._write_obstacles(demo_group, env_idx)
        self._write_environment(demo_group, env_idx)
        self._write_compartment_state(demo_group, env_idx)
        self._write_robot_info(demo_group)
        self._write_object_info(demo_group, env_idx)
        self._write_robot_trajectory(demo_group, episode_traj, env_idx)
        self._write_object_trajectory(demo_group, episode_traj, env_idx)
        self._write_episode_flags(demo_group, episode_traj, env_idx)

        self.demo_count += 1

    def sample(self):
        self.sample_valid_init_pose()
        success_rate_per_env = torch.zeros((self.env.num_envs,), device=self.device)
        self.total_possible_trajectories = int(self.env.num_envs)

        self._open_output_hdf5()
        try:
            episode_traj = self.sample_episode()
            episode_success = self.env.success_flags.clone()
            success_rate_per_env[:] = episode_success

            successful_env_ids = torch.nonzero(episode_success.bool(), as_tuple=False).flatten().cpu().numpy()
            for env_idx in successful_env_ids:
                self.save_successful_env_trajectory(
                    episode_traj=episode_traj,
                    episode_idx=self.episode,
                    env_idx=int(env_idx),
                )

            self.episode = 1
            final_success_rate = success_rate_per_env.mean().item()
            if not self.multi_gpu or self.global_rank == 0:
                print(
                    f"Sampling success rate after one episode: {final_success_rate:.4f}; "
                    f"saved_successful_trajectories={self.demo_count}"
                )

            self._close_output_hdf5(final_success_rate, success_rate_per_env)
        except Exception:
            if self.output_file is not None:
                self.output_file.close()
            raise


@hydra.main(config_name="pre_sample_robot_init_pose.yaml", config_path="../cfg")
def main(cfg: DictConfig):
    sampler = PresampleEnvRobotObjectTrajectory(cfg=cfg)
    sampler.sample()


if __name__ == "__main__":
    import torch._dynamo

    torch._dynamo.config.disable = True
    main()
