import isaacgym
import isaacgymenvs, gym
import hydra
import h5py

from omegaconf import DictConfig
import torch
import torch.optim as optim
from hydra.utils import instantiate
from tqdm import tqdm
from collections import OrderedDict
from isaacgymenvs.utils.rotation_conversions import quaternion_to_matrix_ig
from isaacgymenvs.utils.training_utils import *

import torch.distributed as dist

import yaml
import os
import time

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder

from typing import Dict
from isaacgymenvs.tasks import FrankaLEAPMobile


class PresampleInitPose:
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
        self.episode = 0
        self.total_steps = 0
        self.total_episodes = cfg.presample.total_episodes
        self.steps_per_episode = cfg.presample.steps_per_episode
        self.output_hdf5_dir = cfg.presample.output_hdf5_dir
        self.output_hdf5_name = cfg.presample.output_hdf5_name
        self.device = cfg['sim_device']
        self.seed = cfg.seed
        self.exp_name = cfg.experiment
        set_seed_and_precision(self.seed)

        # load env
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

    def sample_rand_init_pose(self):
        # cfgs
        # cfg_base_init_range = [[-1.0, -0.25, -0.3], [-0.2, 0.25, 0.3]]
        self.cfg_base_init_range = [[-0.2, -0.25, -0.3], [-0.2, 0.25, 0.3]]
        self.cfg_franka_canonical = [0.0, -0.25*np.pi, 0.0, -0.75*np.pi, 0.0, 0.5*np.pi, 0.0]
        self.cfg_franka_noise = 0.2
        self.cfg_leap_canonical = [0.0,]*16
        self.cfg_leap_noise = 0.2
        self.cfg_arx_canonical = [0.0, 1.0, 2.0, -1.0, 0.0, 0.0]
        self.cfg_arx_noise = 0.3
        self.cfg_franka_full = False
        self.cfg_leap_full = False
        self.cfg_arx_full = False

        # sample base init pose
        base_init_range = torch.tensor(self.cfg_base_init_range, device=self.device)
        base_init_pose = torch.rand((self.env.num_envs, 3), device=self.device) * (base_init_range[1] - base_init_range[0]) + base_init_range[0]
        base_init_pose[:, 1] += getattr(self.env, "box_pos", torch.zeros_like(base_init_pose))[:, 1]

        # sample franka init pose
        if self.cfg_franka_full:
            franke_init_config_normalized = 2 * (torch.rand((self.env.num_envs, 7), device=self.device) - 0.5 ) # [-1, 1]
            franke_init_config = self.env.unnormalize_robot_joints(franke_init_config_normalized, "franka", delta=False)
        else:
            franke_init_config = torch.tensor([self.cfg_franka_canonical] * self.env.num_envs, device=self.device) # (num_envs, 7)
            franke_init_config += (torch.rand_like(franke_init_config, device=self.device) - 0.5) * 2 * self.cfg_franka_noise

        # sample leap init pose
        if self.cfg_leap_full:
            leap_init_config_normalized = 2 * (torch.rand((self.env.num_envs, 16), device=self.device) - 0.5 ) # [-1, 1]
            leap_init_config = self.env.unnormalize_robot_joints(leap_init_config_normalized, "leap", delta=False)
        else:
            leap_init_config = torch.tensor([self.cfg_leap_canonical] * self.env.num_envs, device=self.device) # (num_envs, 16)
            leap_init_config += (torch.rand_like(leap_init_config, device=self.device) - 0.5) * 2 * self.cfg_leap_noise

        # sample arx init pose
        if self.cfg.presample.assume_obj_in_view_t0:
            arx_init_config = torch.tensor([self.cfg_arx_canonical] * self.env.num_envs, device=self.device) # (num_envs, 6)
        else:
            if self.cfg_arx_full:
                arx_init_config_normalized = 2 * (torch.rand((self.env.num_envs, 6), device=self.device) - 0.5 ) # [-1, 1]
                arx_init_config = self.env.unnormalize_robot_joints(arx_init_config_normalized, "arx", delta=False)
            else:
                arx_init_config = torch.tensor([self.cfg_arx_canonical] * self.env.num_envs, device=self.device) # (num_envs, 6)
                arx_init_config += (torch.rand_like(arx_init_config, device=self.device) - 0.5) * 2 * self.cfg_arx_noise

        sampled_init_pose = torch.cat([base_init_pose, franke_init_config, leap_init_config, arx_init_config], dim=-1)
        return sampled_init_pose

    def sample_valid_init_pose(self):
        final_init_pose = torch.zeros((self.env.num_envs, 32), device=self.device)
        validation_mask = torch.zeros((self.env.num_envs,), dtype=torch.bool, device=self.device)

        self.env.reset_idx()

        while True:
            sampled_init_pose = self.sample_rand_init_pose()
            final_init_pose[~validation_mask] = sampled_init_pose[~validation_mask]

            # reset all env to the new canonical pose
            self.env.set_robot_joint_state(final_init_pose)
            self.env.step_sim_multi(1, False)
            self.env.compute_observations()

            validation_mask[:] = ~self.env.env_collision[:].bool()

            if torch.all(validation_mask):
                print("All envs have collision free initial pose!")
                self.env.canonical_joint_config[:] = final_init_pose.clone()
                break

        if self.cfg.presample.assume_obj_in_view_t0:
            self.adjust_vision_arm_pose()
            self.env.canonical_joint_config[:, 26:32] = self.env.states['q'][:, 26:32].clone()
            self.env.canonical_joint_config[:, 26:32] += (torch.rand((self.env.num_envs, 6), device=self.device) - 0.5) * 2 * self.cfg_arx_noise
            self.env.reset_idx()

    def adjust_vision_arm_pose(self, gaze_err_tol_deg=5.0):
        gaze_err_tol_rad = torch.deg2rad(torch.tensor(gaze_err_tol_deg, device=self.device))
        dummy_teacher_actions = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.device)

        for step_idx in range(self.steps_per_episode):
            gaze_target = self.env.box_pos.clone() # xyz
            self.env._pre_physics_step_teacher(dummy_teacher_actions, gaze_target)
            teacher_actions = self.env.teacher_actions_converted.clone()
            step_actions = torch.clamp(teacher_actions, -self.env.clip_actions, self.env.clip_actions)
            step_actions[:, :26] = 0.0 # only update arx joints to adjust the arm pose for better vision
            self.env.step(step_actions)

            camera_pose7 = self.env.states['camera_pose7'] # xyz + xyzw
            camera_pos = camera_pose7[:, :3]
            camera_quat = camera_pose7[:, 3:7]

            # camera forward axis is +X in camera frame; convert it into world frame.
            camera_rot_mat = quaternion_to_matrix_ig(camera_quat)
            camera_forward_world = camera_rot_mat[:, :, 0]

            cam_to_target = gaze_target - camera_pos
            cam_to_target = cam_to_target / cam_to_target.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            cos_gaze_err = (camera_forward_world * cam_to_target).sum(dim=-1).clamp(-1.0, 1.0)
            gaze_err_rad = torch.acos(cos_gaze_err)

            if torch.all(gaze_err_rad <= gaze_err_tol_rad):
                print(f"adjust_vision_arm_pose succeeded in {step_idx} steps with max gaze error {torch.rad2deg(gaze_err_rad).max().item():.2f} deg")
                break

            if step_idx == self.steps_per_episode - 1:
                max_err_deg = torch.rad2deg(gaze_err_rad).max().item()
                colorprint(
                    f"adjust_vision_arm_pose reached max steps ({self.steps_per_episode}) "
                    f"with max gaze error {max_err_deg:.2f} deg",
                    color="yellow",
                )

    def sample_episode(self):
        for _ in tqdm(range(self.steps_per_episode), desc=f"Presampling {self.episode+1}/{self.total_episodes}", \
            ncols=None, dynamic_ncols=True, disable=(self.multi_gpu and self.global_rank != 0) ):
            self.total_steps += 1

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

            # step with teacher actions
            step_actions = torch.clamp(teacher_actions, -self.env.clip_actions, self.env.clip_actions)

            # sync distillation steps for video logging
            self.env.distillation_steps = self.total_steps
            self.env.step(step_actions)

    def sample(self):
        self.sample_valid_init_pose()
        success_rate_per_env = torch.zeros((self.env.num_envs,), device=self.device)

        while self.episode < self.total_episodes:
            self.sample_episode()
            success_rate_per_env += self.env.success_flags

            self.episode += 1

        final_success_rate = (success_rate_per_env / self.total_episodes).mean().item()
        final_success_flag = ((success_rate_per_env / self.total_episodes) == 1.0)

        init_states = []
        for i in range(self.env.num_envs):
            if final_success_flag[i]:
                env_i_states = self.env.batch[i]
                env_i_states['init_robot_states'] = self.env.canonical_joint_config[i].cpu().numpy()
                init_states.append(env_i_states)

        output_path = os.path.join(self.output_hdf5_dir, self.output_hdf5_name)
        os.makedirs(self.output_hdf5_dir, exist_ok=True)
        self.save_batch_as_hdf5(init_states, output_path, final_success_rate)

    def save_batch_as_hdf5(self, batch_data, output_path, final_success_rate):
        """Save a batch of demonstrations to a new HDF5 file"""
        with h5py.File(output_path, 'w') as f:
            for idx, data in enumerate(batch_data):
                demo_group = f.create_group(f"demo_{idx}")
                for key in data.keys():
                    demo_group.create_dataset(key, data=data[key])
            f.attrs['presample_success_rate'] = final_success_rate
        print(f"Batch saved to {output_path} with sampling success rate {final_success_rate:.2f}")

@hydra.main(config_name="pre_sample_robot_init_pose.yaml", config_path="../cfg")
def main(cfg: DictConfig):
    sampler = PresampleInitPose(
        cfg=cfg,
    )
    sampler.sample()

if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.disable = True
    main()
