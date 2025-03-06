from collections import Counter, deque
import os
import shutil
import signal
import time
import cv2
import imageio
import wandb
from collections import OrderedDict
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
from isaacgymenvs.tasks import FrankaMPFull

import hydra
from omegaconf import DictConfig

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.base_nets import DDPModelWrapper
from robomimic.utils.log_utils import custom_tqdm as tqdm  # use robomimic tqdm that prints to stdout
from robomimic.config import config_factory
from robomimic.algo import algo_factory


def make_video(frames, logdir, steps, name=None):
    filename = os.path.join(logdir, f"viz_step{steps}.mp4" if name is None else name)
    frames = np.asarray(frames)
    with imageio.get_writer(filename, fps=20) as writer:
        for frame in frames:
            writer.append_data(frame)


def get_obs_shape_meta():
    obs_shape_meta = {
        'ac_dim': 7,
        'all_shapes': OrderedDict([('compute_pcd_params', [1]), ('current_angles', [7]), ('goal_angles', [7])]),
        'all_obs_keys': ['compute_pcd_params', 'current_angles', 'goal_angles'],
        'use_images': False,
        'use_depths': False,
    }
    return obs_shape_meta


class Dagger(object):
    def __init__(self, config):
        # multi-gpu setup & if single gpu, default to cuda:0
        if config.multi_gpu:
            dist.init_process_group(backend="nccl")
        # local rank of the GPU in a node
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        # global rank of the GPU
        self.global_rank = int(os.getenv("RANK", "0"))
        # total number of GPUs across all nodes
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        config.device = f'cuda:{self.local_rank}'
        _sim_device = f'cuda:{self.local_rank}'
        _rl_device = f'cuda:{self.local_rank}'
        torch.cuda.set_device(self.local_rank)  # Explicitly set device

        config.rl_device = _rl_device
        config.sim_device = _sim_device
        config.task.env.batch_idx = config.task.env.batch_idx * self.world_size + self.global_rank
        print(f"global_rank = {self.global_rank} local_rank = {self.local_rank} world_size = {self.world_size} assigned batch_idx = {config.task.env.batch_idx}")

        self.cfg = config
        self.cfg_dict = omegaconf_to_dict(config)
        self.device = self.cfg.device
        self.cfg.seed = set_seed(self.cfg.seed)

        # robomimic init
        ext_cfg = json.load(open("../robomimic/robomimic/exps/mp/neural_mp_rnn.json", 'r'))
        robomimic_cfg = config_factory(ext_cfg["algo_name"])
        with robomimic_cfg.values_unlocked():
            robomimic_cfg.update(ext_cfg)
        robomimic_cfg.experiment.name = "debug"
        robomimic_cfg.lock()
        ObsUtils.initialize_obs_utils_with_config(robomimic_cfg)
        self.seq_length = robomimic_cfg.train.seq_length
        self.frame_stack = 0 # robomimic_cfg.train.frame_stack, TODO: hardcode to 0 for now

        # logging
        run_name = f"{self.cfg.wandb_run_name}"
        self.log_dir = os.path.join(self.cfg.train_dir, run_name)
        self.checkpoint_dir = os.path.join(self.log_dir, "nn")
        self.video_dir = os.path.join(self.log_dir, "videos")

        if self.global_rank == 0:
            if os.path.exists(self.log_dir) and not self.cfg.resume:
                ans = input("WARNING: training directory ({}) already exists! \noverwrite? (y/n)\n".format(self.log_dir))
                if ans == "y":
                    print("REMOVING")
                    shutil.rmtree(self.log_dir)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.video_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.setup_rl_env_and_expert()
        obs_shape_meta = get_obs_shape_meta()

        # set up training
        self.setup_training(robomimic_cfg, obs_shape_meta, self.cfg.ckpt_path)
        self.total_steps = 0
        self.total_episodes = 0
        self.total_epochs = 0
        self.log_history = {}

        assert self.env.num_envs % self.cfg.dagger.batch_size == 0, "Number of environments must be divisible by mini batch size"
        # restore checkpoint
        if self.cfg.resume:
            self.resume = True
            self.load_checkpoint(self.cfg.resume)
        else:
            self.resume = False
            if not (self.cfg.eval_mode or self.cfg.debug_training):
                self.eval_expert()

    def log(self, run, log_dict):
        if run is not None:
            expanded_log_dict = {}
            for k, v in log_dict.items():
                v = v.cpu().numpy()
                if k not in self.log_history:
                    self.log_history[k] = []
                self.log_history[k].append(v)
                expanded_log_dict.update({
                    k: v,
                    k + '/mean': np.mean(self.log_history[k]),
                    k + '/std': np.std(self.log_history[k]),
                    k + '/min': np.min(self.log_history[k]),
                    k + '/max': np.max(self.log_history[k]),
                })
            run.log(expanded_log_dict)

    def setup_rl_env_and_expert(self):
        """Set up the environment & expert model."""
        cfg_dict = omegaconf_to_dict(self.cfg)
        cfg_task = cfg_dict["task"]
        rl_device = cfg_dict["rl_device"]
        sim_device = cfg_dict["sim_device"]
        headless = cfg_dict["headless"]
        graphics_device_id = 0
        virtual_screen_capture = False
        force_render = cfg_dict["force_render"]
        self.env = FrankaMPFull(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
        self.env.action_scale = 1.0
        self.env.force_no_fabric = False
        self.env.no_base_action = False
        self.reaching_reset_threshold = cfg_task['env']['reaching_reset_threshold']
        self.reset_on_collision = cfg_task['env']['reset_on_collision']
        self.step_back_on_collision = cfg_task['env']['step_back_on_collision']
        assert not (self.reset_on_collision and self.step_back_on_collision), "Cannot have both reset_on_collision and step_back_on_collision"
        self.abs_angles_his = deque([self.env.start_config.clone() for _ in range(self.step_back_on_collision)], maxlen=self.step_back_on_collision)

    def setup_training(self, robomimic_cfg, obs_shape_meta, ckpt_path=None):
        """Set up student model and training parameters."""
        self.student_player = self.build_model(
            config=robomimic_cfg,
            shape_meta=obs_shape_meta,
            ckpt_path=ckpt_path,
        )

        # print student model details
        def format_parameters(num):
            if num < 1e6:
                return f"{num / 1e3:.2f}K"  # Thousands
            elif num < 1e9:
                return f"{num / 1e6:.2f}M"  # Millions
            elif num < 1e12:
                return f"{num / 1e9:.2f}G"  # Billions
            else:
                return f"{num / 1e12:.2f}T"  # Trillions

        if self.cfg.multi_gpu:
            self.student_player.nets['policy'] = DDP(self.student_player.nets['policy'], device_ids=[self.local_rank], static_graph=True, find_unused_parameters=True)

        # print model info
        print("\n============= Model Summary =============")
        print(self.student_player)  # print model summary
        if self.cfg.multi_gpu:
            num_policy_params =sum(p.numel() for p in self.student_player.nets['policy'].module.parameters())
            num_enc_params = sum(p.numel() for p in self.student_player.nets['policy'].module.model.nets['encoder'].parameters())
        else:
            num_policy_params = sum(p.numel() for p in self.student_player.nets['policy'].parameters())
            num_enc_params = sum(p.numel() for p in self.student_player.nets['policy'].model.nets['encoder'].parameters())
        print("Policy params:", format_parameters(num_policy_params))
        print("Encoder params:", format_parameters(num_enc_params))
        print("")

        # parameters
        self.num_learning_iterations = self.cfg.dagger.num_learning_iterations
        self.num_learning_epochs = self.cfg.dagger.num_learning_epochs

    def build_model(self, config, shape_meta, ckpt_path=None):
        if ckpt_path is not None and ckpt_path != 'None':
            model, _ = FileUtils.model_from_checkpoint(
                ckpt_path=ckpt_path,
                device=self.cfg.rl_device,
                verbose=True,
                config=config,
            )
        else:
            model = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=shape_meta["all_shapes"],
                ac_dim=shape_meta["ac_dim"],
                device=self.cfg.rl_device,
            )
            model.nets['policy'] = DDPModelWrapper(model.nets['policy'])
        return model

    def reset_envs(self):
        env_ids = torch.arange(self.env.num_envs, device=self.env.device)
        self.env.reset_idx(env_ids)
        self.env.base_model.policy.reset()

    @torch.no_grad()
    def eval_expert(self, eval_iter=1):
        self.expert_success_rate = torch.tensor(0, dtype=torch.float32, device=self.device)
        self.expert_collision_rate = torch.tensor(0, dtype=torch.float32, device=self.device)
        self.expert_reaching_rate = torch.tensor(0, dtype=torch.float32, device=self.device)

        for iter_id in tqdm(range(eval_iter * self.env.max_episode_length), desc=f"Evaling expert"):
            # take a step
            self.env.force_no_fabric = False
            self.env.no_base_action = False
            dummy_actions = torch.zeros((self.env.num_envs, self.env.num_actions), device=self.env.device)
            obs_dict, rews, dones, infos = self.env.step(dummy_actions)

            if (iter_id + 1) % self.env.max_episode_length == 0:
                self.expert_success_rate += infos['success_rate']
                self.expert_collision_rate += infos['collision_rate']
                self.expert_reaching_rate += infos['reaching_rate']

                self.reset_envs()

        if self.cfg.multi_gpu:
            dist.all_reduce(self.expert_success_rate, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.expert_collision_rate, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.expert_reaching_rate, op=dist.ReduceOp.SUM)

        self.expert_success_rate /= (eval_iter * self.world_size)
        self.expert_collision_rate /= (eval_iter * self.world_size)
        self.expert_reaching_rate /= (eval_iter * self.world_size)

    def reset_student_rnn(self, reset_idx):
        if reset_idx.any():
            # this is very specific to LSTM policies
            hidden_state = RMUtils.get_hidden_state(self.student_player)
            hidden_state[0][0][:, reset_idx, ...] = 0 # reset hidden state
            hidden_state[0][1][:, reset_idx, ...] = 0 # reset cell state
            RMUtils.set_hidden_state(self.student_player, hidden_state)

    def train(self):
        """Train the student policy using DAgger."""
        if self.cfg.wandb_activate and self.global_rank == 0:
            # log / load wandb run id
            run_id_file = os.path.join(self.log_dir, "wandb_run_id.json")
            if self.resume:
                with open(run_id_file, "r") as f:
                    run_data = json.load(f)
                    run_id = run_data.get("run_id")
            else:
                run_id = f"dagger_{int(time.time())}"
                with open(run_id_file, "w") as f:
                    json.dump({"run_id": run_id}, f)

            # init wandb
            run = wandb.init(
                project=self.cfg.wandb_project,
                config=self.cfg_dict,
                sync_tensorboard=True,
                name=self.cfg.wandb_run_name,
                resume="allow",
                id=run_id,
                dir=self.log_dir,
            )
        else:
            run = None

        print("Training DAgger...")
        start_step = self.total_steps % self.num_learning_iterations
        print(f"Starting from step {start_step}")
        with tqdm(range(start_step, self.num_learning_iterations), desc='DAgger Training') as pbar:
            init_training = not (self.resume or self.cfg.debug_training)
            if init_training:
                test_success = self.test(num_test_iterations=self.cfg.test_episodes, run=run) # just to prime the dict
                if self.cfg.multi_gpu:
                    for k, value in test_success.items():
                        value_tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
                        dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
                        test_success[k] = value_tensor / self.world_size
                self.reset_envs()
            else:
                test_success = {"success_rate": torch.tensor(0, dtype=torch.float32, device=self.device)}

            pbar.set_postfix(
                ep=self.total_episodes,
                test_success=f"{test_success['success_rate']:.4f}",
            )

            count_reaching = torch.zeros(self.env.num_envs, dtype=torch.int, device=self.env.device)
            reset_envs_bool = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
            init_buffer = True

            for iter_id in pbar:
                if iter_id % self.env.max_episode_length == 0:
                    self.reset_envs()
                    self.student_player.reset()
                    init_buffer = True
                    count_reaching = torch.zeros(self.env.num_envs, dtype=torch.int, device=self.env.device)
                    reset_envs_bool = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

                if self.global_rank == 0:
                    if init_training:
                        # log expert success rate and initial test success rate
                        wandb_log_dict = {
                            "expert/expert_success_rate": self.expert_success_rate,
                            "expert/expert_collision_rate": self.expert_collision_rate,
                            "expert/expert_reaching_rate": self.expert_reaching_rate,
                            f"eval/success_rate": test_success['success_rate'],
                            f"eval/collision_rate": test_success['collision_rate'],
                            f"eval/reaching_rate": test_success['reaching_rate'],
                        }
                        init_training = False
                    else:
                        wandb_log_dict = {}

                # rollout student
                t1 = time.time()
                with torch.no_grad():
                    self.student_player.set_eval()

                    if (self.total_steps) % (self.env.max_episode_length * self.cfg.test_frequency) == 0:
                        test_success = self.test(num_test_iterations=self.cfg.test_episodes, run=run)

                        if self.cfg.multi_gpu:
                            for k, value in test_success.items():
                                value_tensor = torch.tensor(value, dtype=torch.float32, device=self.device)
                                dist.all_reduce(value_tensor, op=dist.ReduceOp.SUM)
                                test_success[k] = value_tensor / self.world_size

                        if self.global_rank == 0:
                            for k in test_success:
                                wandb_log_dict[f"eval/{k}"] = test_success[k]

                        pbar.set_postfix(
                            ep=self.total_episodes,
                            test_success=f"{test_success['success_rate']:.4f}",
                        )

                        self.save_checkpoint(f"checkpoint_step{self.total_steps}_success_{test_success['success_rate']:.4f}.pth")

                    # TODO: get action from (base_policy + fabric), kinda messy, cleanup later
                    abs_base_policy_action = self.env.base_delta_action + self.env.get_joint_angles()
                    self.env.compute_fabric_action(abs_base_policy_action)
                    actions_expert = self.env.delta_fabric_actions.clone()

                    current_angles = self.env.get_joint_angles().clone()
                    goal_angles = self.env.goal_config.clone()
                    compute_pcd_params = self.env.combined_pcds.clone()

                    obs_student = OrderedDict()
                    obs_student["current_angles"] = current_angles
                    obs_student["goal_angles"] = goal_angles
                    obs_student["compute_pcd_params"] = compute_pcd_params
                    actions = self.student_player.get_action(obs_dict=obs_student, mean_actions=self.env.use_mean_actions)

                    self.abs_angles_his.append(current_angles.clone())
                    # take a step
                    self.env.force_no_fabric = True
                    self.env.no_base_action = True
                    obs_dict, rews, dones, infos = self.env.step(actions)
                    self.env.force_no_fabric = False
                    self.env.no_base_action = False

                    if self.seq_length > 0:
                        if init_buffer:
                            current_angles_buffer = deque([current_angles.clone().unsqueeze(1) for _ in range(self.seq_length)], maxlen=self.seq_length)
                            goal_angles_buffer = deque([goal_angles.clone().unsqueeze(1) for _ in range(self.seq_length)], maxlen=self.seq_length)
                            pcd_buffer = deque([compute_pcd_params.clone().unsqueeze(1) for _ in range(self.seq_length)], maxlen=self.seq_length)
                            actions_expert_buffer = deque([actions_expert.clone().unsqueeze(1) for _ in range(self.seq_length)], maxlen=self.seq_length)
                            init_buffer = False

                        if reset_envs_bool.any():
                            for i in range(self.seq_length):
                                current_angles_buffer[i][reset_envs_bool, :, :] = current_angles[reset_envs_bool].clone().unsqueeze(1)
                                goal_angles_buffer[i][reset_envs_bool, :, :] = goal_angles[reset_envs_bool].clone().unsqueeze(1)
                                pcd_buffer[i][reset_envs_bool, :, :] = compute_pcd_params[reset_envs_bool].clone().unsqueeze(1)
                                actions_expert_buffer[i][reset_envs_bool, :, :] = actions_expert[reset_envs_bool].clone().unsqueeze(1)
                            reset_envs_bool = torch.zeros(self.env.num_envs, dtype=torch.bool)
                        else:
                            current_angles_buffer.append(current_angles.clone().unsqueeze(1))
                            goal_angles_buffer.append(goal_angles.clone().unsqueeze(1))
                            pcd_buffer.append(compute_pcd_params.clone().unsqueeze(1))
                            actions_expert_buffer.append(actions_expert.clone().unsqueeze(1))

                        concat_current_angles = torch.cat(tuple(current_angles_buffer), dim=1)
                        concat_goal_angles = torch.cat(tuple(goal_angles_buffer), dim=1)
                        concat_pcd = torch.cat(tuple(pcd_buffer), dim=1)
                        concat_actions_expert = torch.cat(tuple(actions_expert_buffer), dim=1)

                    if self.env.scene_collision.any() and (self.step_back_on_collision or self.reset_on_collision):
                        if self.step_back_on_collision:
                            reset_angles = self.abs_angles_his[0].clone()
                            self.abs_angles_his = deque([reset_angles.clone() for _ in range(self.step_back_on_collision)], maxlen=self.step_back_on_collision)
                        elif self.reset_on_collision:
                            reset_angles = self.env.start_config.clone()

                        # this part feels junky, maybe just use reset_idx?
                        reset_envs_bool = self.env.scene_collision.bool().clone()
                        self.env.set_robot_joint_state(reset_angles[reset_envs_bool], torch.where(reset_envs_bool)[0])
                        self.reset_student_rnn(reset_envs_bool)
                        count_reaching[reset_envs_bool] = 0
                        self.env.compute_observations()
                        self.env.lock_in[reset_envs_bool] = False

                    count_reaching += self.env.goal_reaching
                    if (count_reaching >= self.reaching_reset_threshold).any():
                        reset_angles = self.env.start_config.clone()
                        reset_envs_bool = (count_reaching >= self.reaching_reset_threshold).clone()
                        self.env.set_robot_joint_state(reset_angles[reset_envs_bool], torch.where(reset_envs_bool)[0])
                        self.reset_student_rnn(reset_envs_bool)
                        count_reaching[reset_envs_bool] = 0
                        self.env.compute_observations()
                        self.env.lock_in[reset_envs_bool] = False

                    self.total_steps += 1

                t2 = time.time()
                # learning step
                t2 = time.time()
                hidden_state = RMUtils.get_hidden_state(self.student_player)

                training_batch = {
                    "actions": concat_actions_expert,
                    "obs": {
                        "current_angles": concat_current_angles,
                        "goal_angles": concat_goal_angles,
                        "compute_pcd_params": concat_pcd,
                    }
                }
                avg_loss, loss_dict = self.update(training_batch)
                t3 = time.time()

                if self.cfg.multi_gpu:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss /= self.world_size

                    for k in loss_dict.keys():
                        dist.all_reduce(loss_dict[k], op=dist.ReduceOp.SUM)
                        loss_dict[k] /= self.world_size

                if self.global_rank == 0:
                    wandb_log_dict["distillation/training_loss"] = avg_loss
                    for k in loss_dict.keys():
                        wandb_log_dict[f"distillation/{k}"] = loss_dict[k]
                RMUtils.set_hidden_state(self.student_player, hidden_state)

                # log to wandb
                if self.cfg.wandb_activate and self.global_rank == 0:
                    self.log(run, wandb_log_dict)

                if not self.cfg.logging.suppress_timing:
                    print(f"Running time: rollout: {t2 - t1:.2f}s, update: {t3 - t2:.2f}s")
                
                if self.total_steps % self.env.max_episode_length == 0:
                    # save checkpoint every 10 epochs, its expensive to save the buffer (30s)
                    self.save_checkpoint(prefix='checkpoint_latest')
                    self.total_episodes += 1
                    pbar.set_postfix(
                        ep=self.total_episodes,
                        test_success=f"{test_success['success_rate']:.4f}",
                    )

        self.save_checkpoint(prefix='checkpoint_latest')

    def update(self, training_batch=None):
        model = self.student_player
        model.set_train()

        mini_batch_size = self.cfg.dagger.batch_size
        tot_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        num_mini_batches = self.env.num_envs // mini_batch_size

        loss_dict = {
            "action_loss": torch.tensor(0, dtype=torch.float32, device=self.device),
            "dists_means_l1_loss": torch.tensor(0, dtype=torch.float32, device=self.device),
            "dists_means_l2_loss": torch.tensor(0, dtype=torch.float32, device=self.device),
        }

        if self.cfg.dagger.loss_type == "gmm":
            loss_type = "action_loss"
        elif self.cfg.dagger.loss_type == "l1":
            loss_type = "dists_means_l1_loss"
        elif self.cfg.dagger.loss_type == "l2":
            loss_type = "dists_means_l2_loss"

        if training_batch is not None:
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size

                # create mini batch
                mini_batch = {
                    "actions": training_batch["actions"][start:end],
                    "obs": {
                        "current_angles": training_batch["obs"]["current_angles"][start:end],
                        "goal_angles": training_batch["obs"]["goal_angles"][start:end],
                        "compute_pcd_params": training_batch["obs"]["compute_pcd_params"][start:end],
                    }
                }

                # process batch for training
                input_batch = model.process_batch_for_training(mini_batch)
                input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

                # forward pass
                predictons = model._forward_training(input_batch)
                losses = model._compute_losses(predictons, input_batch)

                # backward pass
                model.optimizers["policy"].zero_grad()

                losses[loss_type].backward()
                model.optimizers["policy"].step()

                for k in loss_dict.keys():
                    loss_dict[k] += losses[k].detach().item()
                tot_loss += losses[loss_type].detach().item()

            for k in loss_dict.keys():
                loss_dict[k] /= num_mini_batches
            tot_loss /= num_mini_batches
            return tot_loss, loss_dict

        # TODO: setup num_learning_epochs correctly and cleanup the below
        batch_indices = self.storage.mini_batch_generator(mini_batch_size)
        num_batches = len(batch_indices)

        for epoch in range(self.num_learning_epochs):
            for indices in batch_indices:
                obs_batch, visual_batch, actions_expert_batch = self.storage.get_batch(indices)
                batch = {
                    "actions": actions_expert_batch,
                    "obs": {
                        "current_angles": obs_batch[..., :7],
                        "goal_angles": obs_batch[..., 7:],
                        "compute_pcd_params": self.env.combined_pcds[visual_batch.int()],
                    }
                }

                # process batch for training
                input_batch = model.process_batch_for_training(batch)
                input_batch = model.postprocess_batch_for_training(input_batch, obs_normalization_stats=None)

                # forward pass
                predictons = model._forward_training(input_batch)
                losses = model._compute_losses(predictons, input_batch)

                # backward pass
                model.optimizers["policy"].zero_grad()
                losses["action_loss"].backward()
                model.optimizers["policy"].step()

                tot_loss += losses["action_loss"].detach().item()

            model.on_epoch_end(self.total_epochs)
            self.total_epochs += 1

        tot_loss /= num_batches * self.num_learning_epochs
        return tot_loss

    @torch.no_grad()
    def test(self, num_test_iterations=5, run=None):
        """Test the student policy."""
        self.student_player.set_eval()

        num_success = Counter()
        total_iters_per_key = Counter()
        tik = time.time()

        total_runs = 0
        video_ims = []
        with tqdm(total=num_test_iterations * self.env.max_episode_length, desc='Testing student policy') as pbar:
            for iter_id in range(num_test_iterations):
                self.reset_envs()
                self.student_player.reset()
                
                state_obs = self.env.compute_observations()
                visual_obs = torch.arange(self.env.num_envs, device=self.device)

                for test_step in range(self.env.max_episode_length - 1):
                    obs_student = OrderedDict()
                    obs_student["current_angles"] = self.env.get_joint_angles().clone()
                    obs_student["goal_angles"] = self.env.goal_config.clone()
                    obs_student["compute_pcd_params"] = self.env.combined_pcds.clone()
                    actions = self.student_player.get_action(obs_dict=obs_student, mean_actions=self.env.use_mean_actions)

                    self.env.force_no_fabric = True
                    self.env.no_base_action = True
                    obs_dict, rews, dones, infos = self.env.step(actions)
                    self.env.force_no_fabric = False
                    self.env.no_base_action = False
                    prev_state_obs = state_obs.clone()
                    state_obs = obs_dict["obs"].clone()
                    visual_obs = torch.arange(self.env.num_envs, device=self.device)

                    if self.env.capture_video and iter_id < self.env.capture_iter_max and test_step % self.env.capture_freq == 0:
                        if "hardcode_images" not in infos or len(infos["hardcode_images"]) == 0:
                            cs = camera_shot(self.env, env_ids=range(self.env.capture_envs), camera_ids=[0])
                            ims = np.array(cs[0])[:, 0, :, :, :3]

                            for env_idx in range(ims.shape[0]):
                                # Convert to uint8 and correct color format for OpenCV
                                img = ims[env_idx].astype(np.uint8).copy()
                                
                                # Create a separate overlay image for the semi-transparent rectangle
                                overlay = img.copy()
                                # Draw grey rectangle on overlay (RGB: 128,128,128)
                                cv2.rectangle(overlay, (10, 10), (300, 80), (128, 128, 128), -1)
                                # Apply the overlay with transparency (alpha = 0.7)
                                alpha = 0.7
                                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                                # Add black text
                                cv2.putText(img, f'Env: {env_idx}  Iter: {iter_id}  Step: {test_step}', (20, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                                cv2.putText(img, f'Has Collided: {self.env.collision_flags[env_idx].bool()}', (20, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                                cv2.putText(img, f'Goal Reaching: {self.env.goal_reaching[env_idx].bool()}', (20, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                                ims[env_idx] = img
                                
                            video_ims.append(ims)
                        else:
                            ims = np.array(infos["hardcode_images"])[:, :, 0, :, :, :3]
                            ims = [ims[i] for i in range(ims.shape[0]) if i % 3 == 0]
                            video_ims.extend(ims)
                    pbar.update()

                for k in infos:
                    if k.endswith('rate'):
                        if self.cfg.eval_mode:
                            print(f"iter{iter_id} {k}: {infos[k]}")
                        num_success[k] += infos[k]
                        total_iters_per_key[k] += 1
                total_runs += self.env.num_envs

        if self.env.capture_video and self.global_rank == 0:
            ims = []
            for env_idx in range(self.env.capture_envs):
                for im in video_ims:
                    ims.append(im[env_idx])
            make_video(ims, self.video_dir, steps=self.total_steps)
            # log video to wandb:
            if self.cfg.wandb_activate and run is not None:
                run.log({"visualization/video": wandb.Video(os.path.join(self.video_dir, f"viz_step{self.total_steps}.mp4"))}, commit=False)

        if not self.cfg.logging.suppress_timing:
            print(f"Finished testing in {time.time() - tik:.2f}s")

        self.reset_envs()

        return {k: v / total_iters_per_key[k] for k, v in num_success.items()}

    def save_checkpoint(self, prefix='checkpoint_latest'):
        if self.global_rank == 0:
            start_time = time.time()
            distillation_ckpt_path = os.path.join(self.checkpoint_dir, f"{prefix}.pth")
            checkpoint = {
                "student_state_dict": self.student_player.serialize(),
                "total_steps": self.total_steps,
                "total_episodes": self.total_episodes,
                "total_epochs": self.total_epochs,
            }
            torch.save(
                checkpoint,
                distillation_ckpt_path,
            )
            if not self.cfg.logging.suppress_timing:
                print("Time to save checkpoint:", time.time() - start_time, "s")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.student_player.deserialize(checkpoint["student_state_dict"], ddp=self.cfg.multi_gpu)
        self.total_steps = checkpoint["total_steps"]
        self.total_episodes = checkpoint["total_episodes"]
        self.total_epochs = checkpoint["total_epochs"]


@hydra.main(version_base="1.1", config_name="config_dagger", config_path="./cfg")
def main(cfg: DictConfig):
    import torch
    
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    agent = Dagger(cfg)
    def handle(signum, frame):
        print("Signal handler called with signal", signum)
        print("Saving checkpoint before exiting...")
        agent.save_checkpoint()
        exit()
    signal.signal(signal.SIGUSR1, handle)
    if cfg.eval_mode:
        if cfg.export_rigid_body_poses_dir:             # export rigid body poses for rendering
            if os.path.exists(cfg.export_rigid_body_poses_dir):
                shutil.rmtree(cfg.export_rigid_body_poses_dir)
            agent.env.export_rigid_body_poses_dir = cfg.export_rigid_body_poses_dir
        test_success = agent.test(num_test_iterations=cfg.test_episodes)
        print(f"Test success rate: {test_success['success_rate']:.4f}")
        print(f"Test collision rate: {test_success['collision_rate']:.4f}")
        print(f"Test reaching rate: {test_success['reaching_rate']:.4f}")

        if cfg.export_rigid_body_poses_dir:
            meta_data = {"task": cfg.task_name}
            if cfg.task_name in ("pick", "place"):
                meta_data.update({
                    "object_code": agent.env.object_code,
                    "object_scale": agent.env.object_scale,
                    "clutter": agent.env.clutter_object_codes_and_scales,
                })
            elif cfg.task_name in ("grasp_handle", "open", "close", "open_nograsp", "close_nograsp"):
                meta_data.update({
                    "object_code": agent.env.object_code,
                })
            np.save(os.path.join(cfg.export_rigid_body_poses_dir, "meta_data.npy"), meta_data)
    else:
        agent.train()


if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.disable = True

    main()
