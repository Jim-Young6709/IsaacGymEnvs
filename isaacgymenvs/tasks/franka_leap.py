"""
Franka + LEAP Hand Env
"""

import os
import time
from datetime import datetime
from pathlib import Path
import json
from abc import abstractmethod

import cv2
import imageio
import trimesh
import wandb
import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import to_torch, tensor_clamp, quat_from_angle_axis, quat_mul, quat_apply
from isaacgymenvs.tasks.base.vec_task import VecTask
import isaacgymenvs.utils.eef_ctrl as eef_ctrl
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from isaacgymenvs.utils.rotation_conversions import quaternion_to_matrix_ig, matrix_to_rotation_6d, sample_spherical_shell, A2B_quaternion
from isaacgymenvs.utils.pcd_utils import transform_pcds_to_world, FrankaLeapSampler
from omegaconf import DictConfig
from tqdm import tqdm
import random
from scipy.spatial.transform import Rotation as R
from curobo.types.math import Pose



class FrankaLEAP(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.device = sim_device
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.reset_noise_scale = self.cfg["env"]["resetNoiseScale"]
        self.eef_actions = True if self.cfg["env"]["numActions"] == 22 else False
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.mesh_args = self.cfg["env"]["mesh"]
        self.object_center_z_scale = float(self.cfg["env"]["object_settings"]["object_center_z_scale"])
        self.object_wrench_args = self.cfg["env"]["object_wrench"]
        self.object_teleport_args = self.cfg["env"]["object_teleport"]
        self.eef_init = self.cfg["env"]["eef_init"]
        self.distractor_settings = self.cfg["env"]["distractor_settings"]
        self.video_logging = self.cfg["env"]["video_logging"]
        self.video_dir = os.path.join('videos', self.cfg["name"] + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        os.makedirs(self.video_dir, exist_ok=True)
        
        self.log_per_object_success = self.cfg["env"]["log_per_object_success"]["capture"]
        self.log_per_object_success_freq = int(self.cfg["env"]["log_per_object_success"]["freq"])
        run_name = None
        if wandb.run is not None and wandb.run.name is not None:
            run_name = str(wandb.run.name)
        elif "experiment" in self.cfg:
            run_name = str(self.cfg["experiment"])
        else:
            run_name = "run"
        run_stamp = time.strftime("%m-%d-%H-%M-%S")
        run_dir = f"{run_name}_{run_stamp}"
        self.log_per_object_success_dir = os.path.join("logs", "per_object_success", run_dir)
        self.log_per_object_success_artifact = f"per_object_success_{run_dir}"
        os.makedirs(self.log_per_object_success_dir, exist_ok=True)

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type == "joint_position", "currently only support joint position control"
        assert "numObservations" in self.cfg["env"], "numObservations must be specified in the config"
        assert "numActions" in self.cfg["env"], "numActions must be specified in the config"

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.pcd_spec_dict = cfg['pcd_spec']

        self.up_axis = "z"
        self.up_axis_idx = 2

        self._init_buffers()
        self._init_cuRobo_ik_solver()

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

        self._post_init_buffers()
        self._build_joint_mapping()

        # Reset all environments
        self._refresh() # TODO: what is this for?
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.step_sim_multi(1, False)
        self.compute_observations()

        # randomize progress buffer
        self.progress_buf = torch.randint(0, self.max_episode_length, (self.num_envs,)).to(self.device)

    def _init_buffers(self):
        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self._object_state = None               # Current state of object for the current env
        self._object_center_init_state = None   # Initial state of object for the current env
        self._object_id = None                  # Actor ID corresponding to object for a given env
        self._add_on_obstacle_ids = []          # Actor ID corresponding to add on obstacles for a given env

        # Tensor placeholders
        self._root_state = None                 # State of root body        (n_envs, 13)
        self._dof_state = None                  # State of all joints       (n_envs, n_dof)
        self._q = None                          # Joint positions           (n_envs, n_dof)
        self._qd = None                         # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None           # State of all rigid bodies (n_envs, n_bodies, 13)
        self._contact_forces = None             # Contact forces in sim
        self._eef_state = None                  # end effector state        (at grasping point)
        self._eef_finger1_state = None          # End effector state (at finger 1)
        self._eef_finger2_state = None          # End effector state (at finger 2)
        self._eef_finger3_state = None          # End effector state (at finger 3)
        self._eef_finger4_state = None          # End effector state (at finger 4)
        self._j_eef = None                      # Jacobian for end effector
        self._mm = None                         # Mass matrix
        self._pos_control = None                # Position actions
        self._effort_control = None             # Torque actions
        self._robot_effort_limits = None        # Actuator effort limits for the robot (franka 7 + leap 4*4)
        self._global_indices = None             # Unique indices corresponding to all envs in flattened array

        self._qd_prev = None                    # Previous joint velocities (n_envs, n_dof)

        # pcd
        self.static_pcds = []
        self.object_pcds = []
        self.combined_pcds = []
        self.static_scene_pcd_t0 = None
        self.object_pcd_t0 = None

        # @ray per object success rate logging
        self.env_object_ids = None

    def _post_init_buffers(self):
        if not hasattr(self, 'canonical_joint_config'):
            # @ray during reset hand is randomized around these joints
            hand_default_1 = [
                0.5,  0.0,  0.5,  0.5,
                1.57,  0.0, -0.3,  0.3,
                0.5,  0.0,  0.5,  0.5,
                0.5,  0.0,  0.5,  0.5,
            ]
            hand_default_2 = [
                0.7, -0.2,  0.7,  0.7,
                0.8,   1.57, 0.77,  0.9,
                0.65,  0.0,  0.65,  0.65,
                0.7,  0.2,  0.7,  0.7,
            ]
            # @ray default 2 is being used
            self.hand_default = ([hand_default_1] + [hand_default_2])[self.cfg['env']['grasp_guide_idx']]

            self.canonical_joint_config = torch.tensor(
                [
                    [0, 0, 0, -3*torch.pi/4, 0, 3*torch.pi/4, 0] + \
                    self.hand_default
                ] * self.num_envs
            ).to(self.device)
        self.ik_regularization_config = self.canonical_joint_config[:, :7]

        self.delta_joint_actions = torch.zeros((self.num_envs, self.num_robot_dofs), device=self.device, dtype=torch.float) # Current delta actions to be deployed
        self.delta_eef_actions = torch.zeros((self.num_envs, self.num_robot_dofs-1), device=self.device, dtype=torch.float) # Current delta actions to be deployed at the end effector

        # @ray per step success tracking
        self.success_5cm_per_step = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device) # success within 5cm threshold
        self.lifting_5cm_per_step = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # @ray instantaneous success tracking, true if condition met at current step
        self.success_flags_instant = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device) # 1 if success condition has been achieved at any step, 0 otherwise
        self.lifting_flags_instant = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        # @ray time-based success tracking, true if condition met for a duration
        self.success_duration = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device) # number of seconds in success region, resets to 0 if object drops
        self.lifting_duration = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device) 
        self.success_long_enough = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device) # true if success duration > threshold in any part of an episode, note that this does not reset until episode end
        self.lifting_long_enough = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.success_flags = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device) # 1 if success condition has been achieved during the episode, 0 otherwise
        self.lifting_flags = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        # @ray 
        # per object success rate tracking
        # need to be post init to get num_objects
        self.per_object_episode_counts = torch.zeros((self.num_objects,), dtype=torch.int64, device=self.device)
        self.per_object_success_counts = torch.zeros((self.num_objects,), dtype=torch.int64, device=self.device)
        self.per_object_episode_counts_interval = torch.zeros((self.num_objects,), dtype=torch.int64, device=self.device)
        self.per_object_success_counts_interval = torch.zeros((self.num_objects,), dtype=torch.int64, device=self.device)

        self.static_scene_pcd_t0 = self.static_pcds.clone()

        self.abs_actions = torch.zeros(self.num_envs, 23, device=self.device)

        self.rigid_body_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.rigid_body_torques = torch.zeros_like(self.rigid_body_forces)
        self.object_applied_forces = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_applied_torques = torch.zeros_like(self.object_applied_forces)

        self.sim_steps = 0.0 # keep track on the number of simulation steps

        # teleport init
        self.num_teleport_envs = int(round(self.object_teleport_args['env_proportion'] * self.num_envs))
        self.teleport_env_ids = torch.randperm(self.num_envs, device=self.device)[:self.num_teleport_envs]
        tele_n0 = self.object_teleport_args['n0']
        tele_n1 = self.object_teleport_args['n1']
        tele_n2 = self.object_teleport_args['n2']
        self.teleport_probs = torch.zeros(tele_n2, dtype=torch.float32, device=self.device)
        self.teleport_probs[tele_n0:tele_n1] = 0.5 / (tele_n1 - tele_n0) # until n1 steps, the probability of teleporting sum up to 0.5
        # for n1~n2 steps, increase the teleport probability quadratically from 0.5 / (tele_n1 - tele_n0) to 1.0
        indexing = torch.arange(tele_n1, tele_n2, dtype=torch.float32, device=self.device)
        quad_c = 0.5 / (tele_n1 - tele_n0)
        quad_b = tele_n1
        quad_a = (1 - quad_c) / ( (tele_n2 - quad_b)**2 )
        self.teleport_probs[tele_n1:] = quad_a*(indexing + 1 - quad_b)**2 + quad_c
        self.teleport_buf = torch.zeros((self.num_envs,), dtype=torch.int, device=self.device)

    def _build_joint_mapping(self):
        env_ptr = self.envs[0]
        robot_handle = self.robots[0]

        isaacgym_dof_list = self.gym.get_actor_dof_names(env_ptr, robot_handle)
        torch_urdf_dof_list = self.robot_pcd_sampler.robot.actuated_joint_names

        assert len(isaacgym_dof_list) == len(torch_urdf_dof_list), \
            f"Mismatch: IsaacGym({len(isaacgym_dof_list)} DOFs) vs TorchURDF({len(torch_urdf_dof_list)} DOFs)"

        # Build mapping lists
        self.torchurdf_to_isaac_idx = []
        self.isaac_to_torchurdf_idx = []

        for i, name in enumerate(torch_urdf_dof_list):
            self.torchurdf_to_isaac_idx.append(isaacgym_dof_list.index(name))
        for i, name in enumerate(isaacgym_dof_list):
            self.isaac_to_torchurdf_idx.append(torch_urdf_dof_list.index(name))

    def _init_cuRobo_ik_solver(self):
        """
        IK is solved with respect to Franka link "panda_link7"
        """
        from curobo.types.base import TensorDeviceType
        from curobo.types.robot import RobotConfig
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        tensor_args = TensorDeviceType()
        config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        urdf_file = config_file["robot_cfg"]["kinematics"][
            "urdf_path"
        ]  # Send global path starting with "/"
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = "panda_link7"
        robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=10,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=True,
            regularization=True,
            grad_iters=None
        )
        self.ik_solver = IKSolver(ik_config)

        if self.debug_viz:
            ik_config_debug = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                None,
                rotation_threshold=0.05,
                position_threshold=0.005,
                num_seeds=10,
                self_collision_check=False,
                self_collision_opt=False,
                tensor_args=tensor_args,
                use_cuda_graph=False,
                regularization=True,
                grad_iters=None
            )
            self.ik_solver_debug = IKSolver(ik_config_debug)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # Domain randomization, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 1.0 # according to current randomization params, -0.275 would be the lowest surface from the env
        self.gym.add_ground(self.sim, plane_params)

    def _create_franka_leap(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        robot_asset_file = "urdf/franka_hand/robots/franka_leap_right.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            robot_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", robot_asset_file)

        full_robot_asset_path = os.path.join(asset_root, robot_asset_file)
        self.robot_pcd_sampler = FrankaLeapSampler(
            urdf_path=full_robot_asset_path,
            device=self.device,
            num_points=self.pcd_spec_dict["num_robot_points"],
        )

        # load FrankaLEAP asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # NOTE: setting it to False allows Leap hand to be black
        asset_options.use_mesh_materials = False
        # NOTE: convex decomposition: disable this for now due to penetration of meshes
        asset_options.vhacd_enabled = False

        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)
        self.robot_asset = robot_asset

        # currently only support joint position control
        robot_dof_stiffness = to_torch([1000.0]*7 + [800.0]*16, dtype=torch.float, device=self.device)
        robot_dof_damping = to_torch([50]*7 + [40.0]*16, dtype=torch.float, device=self.device)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)

        print("num FrankaLEAP bodies: ", self.num_robot_bodies)
        print("num FrankaLEAP dofs: ", self.num_robot_dofs)

        # set FrankaLEAP dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        self._robot_effort_limits = []
        for i in range(self.num_robot_dofs):
            if self.control_type == "joint_position":
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            else:
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                robot_dof_props['stiffness'][i] = robot_dof_stiffness[i]
                robot_dof_props['damping'][i] = robot_dof_damping[i]
            else:
                robot_dof_props['stiffness'][i] = 7000.0
                robot_dof_props['damping'][i] = 50.0

            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])
            self._robot_effort_limits.append(robot_dof_props['effort'][i])

        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)

        if self.cfg["env"]["eef_init"]["limit_abad"]:
            # limit abduction to [-0.1, 0.1] for leap hand
            self.robot_dof_lower_limits[[8, 16, 20]] = -0.15
            self.robot_dof_upper_limits[[8, 16, 20]] = 0.15
        self._robot_effort_limits = to_torch(self._robot_effort_limits, device=self.device)
        return robot_dof_props

    def init_data(self, actor_num):
        # Setup sim handles
        env_ptr = self.envs[0]
        robot_handle = 0
        self.handles = {
            # FrankaLEAP
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "palm_center"),
            "finger1_tip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "index_tip_head"),
            "finger2_tip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "middle_tip_head"),
            "finger3_tip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "ring_tip_head"),
            "finger4_tip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "thumb_tip_head"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.contact_forces = gymtorch.wrap_tensor(_net_contact_forces).view(self.num_envs, -1, 3)
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self._rigid_body_state.shape[1]
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["hand"], :]
        self._eef_finger1_state = self._rigid_body_state[:, self.handles["finger1_tip"], :]
        self._eef_finger2_state = self._rigid_body_state[:, self.handles["finger2_tip"], :]
        self._eef_finger3_state = self._rigid_body_state[:, self.handles["finger3_tip"], :]
        self._eef_finger4_state = self._rigid_body_state[:, self.handles["finger4_tip"], :]
        self._object_state = self._root_state[:, self._object_id, :]

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, robot_handle)['palm_center_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * actor_num, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1) # 3 actors, franka, table, table_stand
        
        target_pos = to_torch(self.cfg["reward"]["params"]["target_pos"], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        target_lift_dis = to_torch(self.cfg["reward"]["params"]["target_lift_dis"], device=self.device)

        # @ray for side picks we need to switch between left and right rot targets
        target_quat = to_torch(self.cfg["reward"]["params"]["target_quat"], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        target_quat_norm = torch.norm(target_quat, dim=1, keepdim=True)  # normalize quaternion
        target_quat = target_quat / (target_quat_norm + 1e-10)

        # finger indexing: 0-3:index ; 4-7:thumb ; 8-11:middle ; 12-15:ring
        # @ray curl config
        grasp_default_1 = [
            0.65,  0.0,  0.65,  0.65,
            1.57,  0.0,  0.10,  0.40,
            0.65,  0.0,  0.65,  0.65,
            0.65,  0.0,  0.65,  0.65,
        ]
        # @ray default is 2
        grasp_default_2 = [
            0.95, -0.2,  0.95,  0.95,
            1.0,   1.57, 1.0,   1.14,
            0.9,  0.0,  0.9,  0.9,
            0.95,  0.2,  0.95,  0.95,
        ]

        self.grasp_default = ([grasp_default_1] + [grasp_default_2])[self.cfg['env']['grasp_guide_idx']]

        self.grasp_finger_dof_pos = torch.tensor(self.grasp_default, device=self.device)

        # for visualization purposes
        self.canonical_grasp_config = torch.tensor(
            [[0, 0, 0, -3*torch.pi/4, 0, 3*torch.pi/4, 0] + self.grasp_finger_dof_pos.tolist()] * self.num_envs
        ).to(self.device)

        self.reward_settings = {
            "target_pos": target_pos,
            "target_lift_dis": target_lift_dis,
            "target_quat": target_quat,
            "target_rot_6d": matrix_to_rotation_6d(quaternion_to_matrix_ig(target_quat)),

            "curl_reaching_threshold": to_torch(self.cfg["reward"]["params"]["curl_reaching_threshold"], device=self.device),
            "success_timeout": to_torch(self.cfg["reward"]["params"]["success_timeout"], device=self.device),
            "lifting_timeout": to_torch(self.cfg["reward"]["params"]["lifting_timeout"], device=self.device),
            "object_init_height": self.mesh_aabb_extents[:, 2] * self.object_center_z_scale + self.table_surface_height, # @ray default center is 1/2 z height
            "grasp_finger_dof_pos": self.grasp_finger_dof_pos,

            "beta_hand_object": to_torch(self.cfg["reward"]["exp"]["beta_hand_object"], device=self.device),
            "beta_object_goal": to_torch(self.cfg["reward"]["exp"]["beta_object_goal"], device=self.device),
            "beta_lift": to_torch(self.cfg["reward"]["exp"]["beta_lift"], device=self.device),
            "beta_curl": to_torch(self.cfg["reward"]["exp"]["beta_curl"], device=self.device),

            "w_hand_obj": to_torch(self.cfg["reward"]["weights"]["w_hand_obj"], device=self.device),
            "w_obj_goal": to_torch(self.cfg["reward"]["weights"]["w_obj_goal"], device=self.device),
            "w_lift": to_torch(self.cfg["reward"]["weights"]["w_lift"], device=self.device),
            "w_curl": to_torch(self.cfg["reward"]["weights"]["w_curl"], device=self.device),
            "w_actionreg": to_torch(self.cfg["reward"]["weights"]["w_actionreg"], device=self.device),

            # @ray what rewards to use, for running ablations
            "use_curl": to_torch(self.cfg["reward"]["usage"]["use_curl"], device=self.device),
        }

    def _create_cube(self, pos, size, quat=[0, 0, 0, 1]):
        """
        Args:
            position (np.ndarray): (3,) xyz position of the cube center
            size (np.ndarray): (3,) length along xyz direction of the cube
            quat (np.ndarray): (4,), [x, y, z, w]
        Returns:
            asset (gymapi.Asset): asset handle of the cube
            start_pose (gymapi.Transform): start pose of the cube
        """
        # Create cube asset
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        asset = self.gym.create_box(self.sim, *size, opts)
        # Define start pose
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(*quat)  # quat in xyzw order
        self.cuboid_dims.append(size)
        self.cuboid_pos.append(pos)
        self.cuboid_quats.append(quat)
        return asset, start_pose

    def _create_sphere(self, pos, size):
        """
        Args:
            position (np.ndarray): (3,) xyz position of the sphere center
            size (np.ndarray): (1, ) radius of the sphere, note this should have a dim of 1
        Returns:
            asset (gymapi.Asset): asset handle of the sphere
            start_pose (gymapi.Transform): start pose of the sphere
        """
        # Create cube asset
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        asset = self.gym.create_sphere(self.sim, size, opts)
        # Define start pose
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        self.sphere_radii.append(size)
        self.sphere_pos.append(pos)
        return asset, start_pose

    def _create_capsule(self, pos, size):
        """
        Args:
            position (np.ndarray): (3,) xyz position of the capsule center
            size (np.ndarray): (2,) radius and length of the capsule
                radius (float): radius of the sphere
                length (float): semi-length of the cylindrical part
        Returns:
            asset (gymapi.Asset): asset handle of the capsule
            start_pose (gymapi.Transform): start pose of the capsule
        """
        # Create cube asset
        opts = gymapi.AssetOptions()
        opts.fix_base_link = True
        asset = self.gym.create_capsule(self.sim, size[0], size[1], opts)
        # Define start pose
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(*[0.0, -0.707, 0.0, 0.707])  # quat in xyzw order
        self.capsule_dims.append(size)
        self.capsule_pos.append(pos)
        return asset, start_pose

    def _create_mesh_urdf(self, mesh_path, scale=[1.0, 1.0, 1.0], mass=0.5):
        mesh_dir = os.path.dirname(mesh_path)
        mesh_filename = os.path.basename(mesh_path)
        mesh_name, _ = os.path.splitext(mesh_filename)

        urdf_rel = mesh_name + ".urdf"

        # TODO: change logic in the future, recreating urdf might not be a good idea
        # urdf_path = os.path.join(mesh_dir, urdf_rel)

        # mesh = trimesh.load(mesh_path)
        # z_com = mesh.extents[2] * scale[2] / 2

        # # URDF content
        # urdf_str = f"""<?xml version="1.0" ?>
        #     <robot name="mesh_object">
        #     <link name="base">
        #         <visual>
        #             <geometry>
        #                 <mesh filename="{mesh_filename}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
        #             </geometry>
        #         </visual>
        #         <collision>
        #             <geometry>
        #                 <mesh filename="{mesh_filename}" scale="{scale[0]} {scale[1]} {scale[2]}"/>
        #             </geometry>
        #         </collision>
        #         <inertial>
        #             <origin xyz="0 0 {z_com}" rpy="0 0 0"/>
        #             <mass value="{mass}"/>
        #             <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
        #         </inertial>
        #     </link>
        #     </robot>
        # """

        # # Save URDF
        # with open(urdf_path, 'w') as f:
        #     f.write(urdf_str)
        return urdf_rel, mesh_dir

    def _create_mesh(self, mesh_path, pos, scale, quat=[0, 0, 0, 1], fix_base_link=True, obj_str2int=None):
        """
        Args:
            position (np.ndarray): (3,) xyz position of the mesh center
            scale (float): (1,) scale of the mesh
            quat (np.ndarray): (4,), [x, y, z, w]
        Returns:
            asset (gymapi.Asset): asset handle of the mesh
            start_pose (gymapi.Transform): start pose of the mesh
        """
        # convert .obj into .urdf file
        mesh_scale = [scale, scale, scale]
        urdf_path, asset_root = self._create_mesh_urdf(mesh_path, scale=mesh_scale)


        # @ray urdf format
        # ├── apple_1
        # │   ├── apple_1.glb
        # │   ├── apple_1.json
        # │   ├── apple_1.npy
        # │   ├── apple_1.obj
        # │   └── apple_1.urdf
        # └── type_mapping.json
        # object specs including id is in apple_1.json
        # asset_specs_path = Path(mesh_path).with_suffix(".json")

        asset_mesh_id = Path(mesh_path).parts[-2]
        asset_obj_id = int(obj_str2int[asset_mesh_id])

        # Create mesh asset
        opts = gymapi.AssetOptions()
        opts.fix_base_link = fix_base_link
        asset = self.gym.load_asset(self.sim, asset_root, urdf_path, opts) # TODO: this step seems to take a lot of time, try to optimize it
        # Define start pose
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(*quat)  # quat in xyzw order
        return asset, start_pose, scale, asset_obj_id, asset_mesh_id

    def create_rand_mesh(self, fix_base_link=False):
        # get randomly sampled mesh path
        mesh_dir = self.mesh_args["mesh_dir"]
        object_list = self.mesh_args["obj_list"]

        if object_list == ["all"]:
            object_list = [
                obj
                for obj in os.listdir(mesh_dir)
                if obj != "type_mapping.json"
            ]

        mesh_files = [
            os.path.join(mesh_dir, obj, file)
            for obj in object_list
            for file in os.listdir(os.path.join(mesh_dir, obj))
            if file.endswith(".obj")
        ]
        mesh_sampler = lambda: random.choice(mesh_files)
        sampled_mesh_path = mesh_sampler()

        # sample random size, pos and ori
        scale_range = self.cfg["env"]["object_settings"]["scale_range"]
        pos_range = self.cfg["env"]["object_settings"]["xyz_range"]

        mesh_scale = np.random.uniform(scale_range[0], scale_range[1])
        mesh_pos = np.random.uniform(pos_range[0], pos_range[1])
        mesh_quat = R.random().as_quat()  # [x, y, z, w]

        return self._create_mesh(sampled_mesh_path, mesh_pos, mesh_scale, mesh_quat, fix_base_link)

    def create_all_meshes(self, fix_base_link=False):
        """
        Create all meshes in the mesh directory.
        Args:
            fix_base_link (bool): whether to fix the base link of the mesh
            on_table (bool): whether to place the mesh on the table surface
        Returns:
            meshes (list): list of tuples containing asset, start_pose, scale, asset_obj_id, asset_mesh_id
        """
        mesh_dir = self.mesh_args["mesh_dir"]
        object_list = self.mesh_args["obj_list"]

        if object_list == ["all"]:
            object_list = [
                obj
                for obj in os.listdir(mesh_dir)
                if obj != "type_mapping.json"
            ]

        obj_mapping_path = os.path.join(mesh_dir, "type_mapping.json")
        with open(obj_mapping_path, "r") as f:
            obj_str2int = json.load(f)
        object_list = sorted(
            object_list,
            key=lambda obj: (0, obj_str2int[obj]) if obj in obj_str2int else (1, obj),
        )

        mesh_files = []
        for obj in object_list:
            obj_dir = os.path.join(mesh_dir, obj)
            if not os.path.isdir(obj_dir):
                continue
            for file in sorted(os.listdir(obj_dir)):
                if file.endswith(".obj"):
                    mesh_files.append(os.path.join(obj_dir, file))

        meshes = []
        for mesh_file_path in tqdm(mesh_files, desc="Preparing Meshes"):
            # sample random size, pos and ori
            scale_range = self.cfg["env"]["object_settings"]["scale_range"]
            pos_range = self.cfg["env"]["object_settings"]["xyz_range"]

            mesh_scale = np.random.uniform(scale_range[0], scale_range[1])
            mesh_pos = np.random.uniform(pos_range[0], pos_range[1])
            mesh_quat = R.random().as_quat()
            asset, start_pose, scale, asset_obj_id, asset_mesh_id = self._create_mesh(
                mesh_file_path, mesh_pos, mesh_scale, mesh_quat, fix_base_link, obj_str2int
            )
            meshes.append((asset, start_pose, scale, asset_obj_id, asset_mesh_id))

        return meshes

    def _refresh(self):
        self._qd_prev = self._qd.clone()

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self.check_robot_collision()
        self._update_states()

    def _update_states(self):
        # update arm eef state
        eef_rot_mat = quaternion_to_matrix_ig(self._eef_state[:, 3:7])
        eef_rot_6d = matrix_to_rotation_6d(eef_rot_mat)

        # update object state
        object_center_pos = self._object_state[:, :3].clone()
        local_offset = torch.zeros([self.num_envs, 3], dtype=torch.float, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[:, 2] * self.object_center_z_scale
        object_rot_mat = quaternion_to_matrix_ig(self._object_state[:, 3:7])
        rotated_offset = torch.matmul(object_rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)
        object_center_pos += rotated_offset

        object_rot_6d = matrix_to_rotation_6d(object_rot_mat)

        object_rot_mat_in_eef_frame = torch.matmul(eef_rot_mat.transpose(1, 2), object_rot_mat)
        object_to_eef_rot_6d = matrix_to_rotation_6d(object_rot_mat_in_eef_frame)

        # update target state
        target_rot_mat = quaternion_to_matrix_ig(self.reward_settings["target_quat"])
        target_rot_mat_in_eef_frame = torch.matmul(eef_rot_mat.transpose(1, 2), target_rot_mat)
        target_to_eef_rot_6d = matrix_to_rotation_6d(target_rot_mat_in_eef_frame)

        # @ray we don't need pos and rot error except for side grasp table
        # probably should refactor to use separate update states later
        point_matching_err_target = self._get_eef_point_matching_err(
            curent_eef_pos7=self._eef_state[:, :7],
            target_eef_pos7=torch.cat([self.reward_settings["target_pos"], self.reward_settings["target_quat"]], dim=-1),
        )
        hand_eef_pos7_rot = torch.cat([self._eef_state[:, :3], self.reward_settings["target_quat"]], dim=-1)
        point_matching_err_hand = self._get_eef_point_matching_err(
            curent_eef_pos7=self._eef_state[:, :7],
            target_eef_pos7=hand_eef_pos7_rot,
        )

        # update point clouds
        object_pcds_world = transform_pcds_to_world(self.object_pcds, self._object_state[:, :7])
        self.combined_pcds[:, -self.pcd_spec_dict["num_object_points"]:] = object_pcds_world

        # update initial frame pcd
        if self.object_pcd_t0 is None:
            self.object_pcd_t0 = object_pcds_world.clone()
        else:
            init_flag = (self.progress_buf == 0)
            self.object_pcd_t0[init_flag] = object_pcds_world[init_flag].clone()

        if self.cfg["reward"]["actionreg_type"] == "delta_joint_action":
            actionreg = self.delta_joint_actions
        elif self.cfg["reward"]["actionreg_type"] == "delta_eef_action":
            actionreg = self.delta_eef_actions
        elif self.cfg["reward"]["actionreg_type"] == "delta_qd":
            actionreg = self._qd - self._qd_prev
        else:
            actionreg = torch.zeros_like(self._qd)

        # update states
        # @ray not just to update states, but this is where the keys are created
        self.states.update({
            # Robot
            "q": self._q[:, :],
            "q_hand": self._q[:, 7:],
            "qd": self._qd[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_rot_6d": eef_rot_6d,
            "eef_vel": self._eef_state[:, 7:],
            "eef_finger1_pos": self._eef_finger1_state[:, :3],
            "eef_finger2_pos": self._eef_finger2_state[:, :3],
            "eef_finger3_pos": self._eef_finger3_state[:, :3],
            "eef_finger4_pos": self._eef_finger4_state[:, :3],

            # Fingertip positions relative to hand base (palm_center)
            "eef_finger1_pos_relative": self._eef_finger1_state[:, :3] - self._eef_state[:, :3],
            "eef_finger2_pos_relative": self._eef_finger2_state[:, :3] - self._eef_state[:, :3],
            "eef_finger3_pos_relative": self._eef_finger3_state[:, :3] - self._eef_state[:, :3],
            "eef_finger4_pos_relative": self._eef_finger4_state[:, :3] - self._eef_state[:, :3],

            # Object
            "object_quat": self._object_state[:, 3:7],
            "object_rot_6d": object_rot_6d,
            "object_center_pos": object_center_pos,
            "object_pos": self._object_state[:, :3],

            # Task related
            "object_to_eef": object_center_pos - self._eef_state[:, :3],
            "object_to_eef_rot_6d": object_to_eef_rot_6d,
            "target_to_eef": self.reward_settings["target_pos"] - self._eef_state[:, :3],
            "target_to_eef_rot_6d": target_to_eef_rot_6d,
            "point_matching_err_target": point_matching_err_target,
            "point_matching_err_hand": point_matching_err_hand, # @ray separates pos and rot error

            # recorded actions
            "actionreg": actionreg,
        })

    def _get_eef_point_matching_err(self, curent_eef_pos7: torch.Tensor, target_eef_pos7: torch.Tensor):
        """
        Get the point matching error between current end effector position 
        and target end effector position. (based on 5 points on eef)
        @ray optionally can return orientation and position error separately

        Args:
            curent_eef_pos7: (B, 7) xyz + xyzw
            target_eef_pos7: (B, 7) xyz + xyzw
        """
        B = curent_eef_pos7.shape[0]

        pos_c = curent_eef_pos7[:, :3]  # (B, 3)
        quat_c = curent_eef_pos7[:, 3:] # (B, 4)
        pos_t = target_eef_pos7[:, :3]  # (B, 3)
        quat_t = target_eef_pos7[:, 3:] # (B, 4)

        matching_dis = self.cfg["reward"]["params"]["matching_dis"]
        local_pts = torch.tensor(
            [[matching_dis, 0., 0.],
            [-matching_dis, 0., 0.],
            [0., 0., 0.],
            [0., matching_dis, 0.],
            [0., -matching_dis, 0.]],
            dtype=curent_eef_pos7.dtype,
            device=curent_eef_pos7.device
        )
        P = local_pts.shape[0]

        # Repeat points and quats for batch
        pts_a_flat = local_pts.unsqueeze(0).expand(B, P, 3).reshape(B*P, 3)
        pts_b_flat = local_pts.unsqueeze(0).expand(B, P, 3).reshape(B*P, 3)
        qc_rep = quat_c.repeat_interleave(P, dim=0)
        qt_rep = quat_t.repeat_interleave(P, dim=0)

        # Rotate and translate to world frame
        world_c = quat_apply(qc_rep, pts_a_flat).view(B, P, 3) + pos_c.unsqueeze(1)
        world_t = quat_apply(qt_rep, pts_b_flat).view(B, P, 3) + pos_t.unsqueeze(1)

        # Mean squared error per sample
        avg_point_dis_error = torch.norm(world_c - world_t, dim=-1).mean(dim=-1)

        return avg_point_dis_error

    def check_robot_collision(self):
        # TODO: figure out arm & hand collision
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.scene_collision = torch.where(
            torch.norm(torch.sum(self.contact_forces[:, :30, :], dim=1), dim=1) > 1.0, 1.0, 0.0
        )  # the first 30 elements belong to franka + leap
        self.collision = torch.where(
            torch.sum(torch.norm(self.contact_forces[:, :30, :], dim=2), dim=1) > 1.0, 1.0, 0.0
        )  # the first 30 elements belong to franka + leap, this includes self collision

    def normalize_robot_joints(self, joint_angles: torch.Tensor, robot: bool, delta: bool = False) -> torch.Tensor:
        """
        Normalize joint angles to be within the joint limits.
        Args:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        Returns:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        """
        if robot=="arm":
            assert joint_angles.shape[-1] == 7
            lower_limits, upper_limits = self.get_joint_limits_franka()
        elif robot=="hand":
            assert joint_angles.shape[-1] == 16
            lower_limits, upper_limits = self.get_joint_limits_leap()
        else:
            raise ValueError("robot must be either 'arm' or 'hand'")

        franka_limit_range = upper_limits - lower_limits

        if delta:
            normalized = joint_angles / franka_limit_range
        else:
            desired_lower_limits = -1 * torch.ones_like(joint_angles)
            desired_upper_limits = 1 * torch.ones_like(joint_angles)
            normalized = (joint_angles - lower_limits) / franka_limit_range * (
                desired_upper_limits - desired_lower_limits
            ) + desired_lower_limits
        return normalized

    def unnormalize_robot_joints(self, joint_angles: torch.Tensor, robot: bool, delta: bool = False) -> torch.Tensor:
        """
        Unnormalize joint angles.
        Args:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        Returns:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        """
        if robot=="arm":
            assert joint_angles.shape[-1] == 7
            lower_limits, upper_limits = self.get_joint_limits_franka()
        elif robot=="hand":
            assert joint_angles.shape[-1] == 16
            lower_limits, upper_limits = self.get_joint_limits_leap()
        else:
            raise ValueError("robot must be either 'arm' or 'hand'")

        franka_limit_range = upper_limits - lower_limits

        if delta:
            unnormalized = joint_angles * franka_limit_range
        else:
            desired_lower_limits = -1 * torch.ones_like(joint_angles)
            desired_upper_limits = 1 * torch.ones_like(joint_angles)
            unnormalized = (joint_angles - desired_lower_limits) * franka_limit_range / (
                desired_upper_limits - desired_lower_limits
            ) + lower_limits
        return unnormalized

    def get_joint_from_ee(self, eef_pose, return_success=False, use_debug=False):
        """
        Get the joint angles from the end effector pose. This func is well tested
        Args:
            eef_pose (np.ndarray): 7D end effector pose. (B, xyz xyzw)
        Returns:
            joint_angles (np.ndarray): 7-dof joint angles.  (B, 7)
        """
        eef_pose = eef_pose.clone().contiguous()
        eef_pos = eef_pose[:, :3]
        eef_quat_xyzw = eef_pose[:, 3:]
        eef_quat_wxyz = eef_quat_xyzw[:, [3, 0, 1, 2]]

        B = eef_pose.shape[0]
        B_pad = self.num_envs - B

        if B_pad > 0:
            eef_pos_dummy = torch.tensor([[0.3, 0.0, 0.3]]*B_pad, dtype=torch.float, device=self.device)
            eef_quat_wxyz_dummy = torch.tensor([[1.0, 0.0, 0.0, 0.0]]*B_pad, dtype=torch.float, device=self.device)

            eef_pos = torch.cat((eef_pos, eef_pos_dummy), dim=0)
            eef_quat_wxyz = torch.cat((eef_quat_wxyz, eef_quat_wxyz_dummy), dim=0)

        goal = Pose(eef_pos, eef_quat_wxyz) # Pose need quat in wxyz format
        solver = self.ik_solver_debug if use_debug else self.ik_solver
        result = solver.solve_batch(
            goal_pose=goal,
            retract_config=self.ik_regularization_config,
        )
        if torch.any(result.success[:B] == False):
            # @ray report only the real envs
            failed = (~result.success[:B]).nonzero(as_tuple=False).squeeze(-1)
            # print(f"IK solver failed for some environments: {failed}/{B}")
            # TODO: need to think a bit how to handle such cases

        q_solution = result.solution[:B, 0]
        if return_success:
            return q_solution, result.success[:B]
        return q_solution

    def get_ee_from_joint(self, joint_angles):
        """
        Get the end effector pose from the joint angles. This func is well tested
        Args:
            joint_angles (torch.Tensor): 7-dof joint angles. (B, 7)
        Returns:
            ee_pose (torch.Tensor)): 7D end effector pose. xyz, xyzw
        """
        joint_angles = joint_angles.clone().contiguous()
        kin_state = self.ik_solver.fk(joint_angles)
        eef_pose = kin_state.ee_position
        eef_wxyz = kin_state.ee_quaternion
        eef_xyzw = eef_wxyz[:, [1, 2, 3, 0]]

        return torch.cat((eef_pose, eef_xyzw), dim=-1)  # (B, 7) with xyz and xyzw

    def set_joint_pos_from_ee_pos(self, target_ee_pose): # TODO: implement
        """
        Set the joint angles from the end effector pose.
        Args:
            target_ee_pose (np.ndarray): 7D end effector pose.
        """
        raise NotImplementedError("not implemented yet")

    def set_robot_joint_state(self, joint_state: torch.Tensor, joint_vel=None, env_ids=None):
        """
        Set the joint state of the robot. (set the dof state (pos/vel) of each joint,
        joint_vel (torch.Tensor): (num_selected_envs, 7+4*4) joint velocity

        Args:
            joint_state (torch.Tensor): (num_selected_envs, 7)
        """
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        assert joint_state.shape[0] == len(env_ids)
        assert joint_state.shape[1] == self.num_robot_dofs

        state_tensor = joint_state.clone().unsqueeze(2)  # (num_selected_envs, self.num_robot_dofs, 1)
        state_tensor = torch.cat((state_tensor, torch.zeros_like(state_tensor)), dim=2)

        if joint_vel is not None:
            state_tensor[:, :23, 1] = joint_vel

        pos = state_tensor[:, :, 0].contiguous()
        vel = state_tensor[:, :, 1].contiguous()

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = vel
        self._dof_state[env_ids, :] = state_tensor
        self._pos_control[env_ids, :] = pos

        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def _reset_object_state(self, object_reset_env_ids):
        if object_reset_env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = object_reset_env_ids.clone()

        if self.object_teleport_args["enable"]:
            if self.sim_steps % (self.max_episode_length * self.object_teleport_args['swap_freq']) == 0:
                self.teleport_env_ids = torch.randperm(self.num_envs, device=self.device)[:self.num_teleport_envs]

            curri_factor = min(self.sim_steps / self.object_teleport_args['curri_steps'], 1.0)
            # teleport object (note this is in addition to normal reset)
            _teleport_buf = self.teleport_buf[self.teleport_env_ids] # get the corresponding teleport buffer
            apply_teleport = (self.teleport_probs[_teleport_buf] * curri_factor) > torch.rand(len(_teleport_buf), device=self.device)
            apply_teleport_env_ids = self.teleport_env_ids[apply_teleport]

            env_ids = torch.unique(torch.cat([env_ids, apply_teleport_env_ids], dim=0))

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_object_state = torch.zeros(num_resets, 13, device=self.device)

        # Sampling is "centered" around middle of table
        reset_pos = torch.zeros(num_resets, 3, device=self.device)
        reset_pos[:, :2] = torch.rand(num_resets, 2, device=self.device) * (self.obj_pos_range[env_ids][:, [1,3]] - self.obj_pos_range[env_ids][:, [0,2]]) + self.obj_pos_range[env_ids][:, [0,2]]
        reset_pos[:, 2] = self.table_surface_height[env_ids]

        sampled_object_state[:, 6] = 1.0
        # theta = torch.rand(num_resets, device=self.device) * 2 * torch.pi  # random angle [0, 2π)
        # # quat = [0.0, 0.0, torch.sin(theta/2), torch.cos(theta/2)]
        # sampled_object_state[:, 5] = torch.sin(theta/2)
        # sampled_object_state[:, 6] = torch.cos(theta/2)
        sampled_object_state[:, :3] = reset_pos
        self._object_state[env_ids] = sampled_object_state
        self._object_center_init_state[env_ids] = reset_pos
        self._object_center_init_state[env_ids, 2] += self.mesh_aabb_extents[env_ids, 2] * self.object_center_z_scale

        multi_env_ids_obj_int32 = self._global_indices[env_ids, self._object_id].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_obj_int32), len(multi_env_ids_obj_int32),
        )

        self.teleport_buf[env_ids] = 0

    def get_joint_limits_franka(self):
        """
        Get the joint limits of the Franka arm. Franka (7) + LEAP (4*4), 23 DOF in total

        Returns:
            lower_limits (torch.Tensor): (7,)
            upper_limits (torch.Tensor): (7,)
        """
        lower_limits = self.robot_dof_lower_limits[:7]
        upper_limits = self.robot_dof_upper_limits[:7]
        return lower_limits, upper_limits

    def get_joint_limits_leap(self):
        """
        Get the joint limits of the LEAP hand. Franka (7) + LEAP (4*4), 23 DOF in total

        Returns:
            lower_limits (torch.Tensor): (16,)
            upper_limits (torch.Tensor): (16,)
        """
        lower_limits = self.robot_dof_lower_limits[7:]
        upper_limits = self.robot_dof_upper_limits[7:]
        return lower_limits, upper_limits

    # visualization
    def set_viewer(self, pos=[1.5, 1.0, 0.7], target=[0.5, 0.0, 0.1]):
        """
        Create the viewer.
        """

        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

            # set the camera position based on up axis
            centre = self.cfg["env"]['envSpacing'] + int(np.sqrt(self.num_envs))
            
            # cam_pos = gymapi.Vec3(0, 0, 5)
            # cam_target = gymapi.Vec3(centre, centre, 0)
            # let camera look at env 0
            cam_pos = gymapi.Vec3(pos[0], pos[1], pos[2])
            cam_target = gymapi.Vec3(target[0], target[1], target[2])

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        if self.video_logging["capture"]:
            assert self.video_logging["envs"] <= self.num_envs, "Number of environments for video logging exceeds total number of environments."
            self.camera_handles = []
            self.obs_camera_handles = []
            camera_props = gymapi.CameraProperties()
            camera_props.width = 640
            camera_props.height = 480
            camera_props.horizontal_fov = 90.0
            camera_props.enable_tensors = False # disable gpu tensors, so cameras won't have automatic updates
            for i in range(self.video_logging["envs"]):
                self.camera_handles.append([])
                self.obs_camera_handles.append([])
                # global
                camera_handle = self.gym.create_camera_sensor(
                    self.envs[i], camera_props
                )
                if camera_handle == -1:
                    print(f"Failed to create camera sensor for env {i}")
                    continue  # Skip this camera if creation failed

                camera_position = gymapi.Vec3(pos[0], pos[1], pos[2])
                camera_target = gymapi.Vec3(target[0], target[1], target[2])
                self.gym.set_camera_location(
                    camera_handle, self.envs[i], camera_position, camera_target
                )
                self.camera_handles[i].append(camera_handle)

    def get_camera_render(self):
        """
        Returns:
            images: List[List[np.ndarray]], RGB images from all specified environments and cameras
        """

        assert self.video_logging["capture"], "Camera is not enabled."
        env_ids = range(self.video_logging["envs"])

        if self.device != "cpu":
            self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        images = []
        for env_id in env_ids:
            images.append([])

            camera_handle = self.camera_handles[env_id][0]
            camera_image = self.gym.get_camera_image(
                self.sim, self.envs[env_id], camera_handle, gymapi.IMAGE_COLOR
            )
            shape = camera_image.shape
            camera_image = camera_image.reshape(shape[0], -1, 4)
            images[-1].append(camera_image)

        return images

    def video_logger(self):
        render_step = self.sim_steps % self.video_logging["freq"]
        if render_step == 0:
            self.video_writers = []
            self.video_step_start = self.sim_steps + 1 - self.max_episode_length
            self.video_env_ids = list(range(self.video_logging["envs"]))
            for env_idx in self.video_env_ids:
                filename = os.path.join(
                    self.video_dir,
                    f"viz_step{self.video_step_start}_env{env_idx}.mp4"
                )
                try:
                    writer = imageio.get_writer(filename, fps=60, format="ffmpeg")
                except Exception as exc:
                    print(f"[video_logger] ffmpeg writer unavailable, skipping video: {exc}")
                    writer = None
                self.video_writers.append(writer)

        if render_step < self.max_episode_length * 2:
            camera_renders = self.get_camera_render()
            ims = np.array(camera_renders)[:, 0, :, :, :3]

            for idx, env_idx in enumerate(self.video_env_ids):
                writer = self.video_writers[idx]
                if writer is None:
                    continue
                # Convert to uint8 and correct color format for OpenCV
                img = ims[env_idx].astype(np.uint8).copy()

                # Create a separate overlay image for the semi-transparent rectangle
                overlay = img.copy()
                # Draw grey rectangle on overlay (RGB: 128,128,128)
                cv2.rectangle(overlay, (10, 10), (220, 50), (128, 128, 128), -1)
                # Apply the overlay with transparency (alpha = 0.7)
                alpha = 0.7
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                # Add black text
                cv2.putText(
                    img, f'Env: {env_idx}  Step: {render_step}', (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2
                )
                writer.append_data(img)

        if render_step == 2*self.max_episode_length - 1:
            for writer in getattr(self, "video_writers", []):
                if writer is not None:
                    writer.close()
            self.video_writers = []
            if wandb.run is not None:
                for idx, env_idx in enumerate(self.video_env_ids):
                    path = os.path.join(
                        self.video_dir,
                        f"viz_step{self.video_step_start}_env{env_idx}.mp4"
                    )
                    wandb.log(
                        {f"visualization/video_env_{env_idx}": wandb.Video(path)},
                        commit=(idx == len(self.video_env_ids) - 1),
                    )

    # debugging utils
    def step_sim_multi(self, num_steps=1, render_pcd=True):
        """
        Step the simulation. (for debugging purposes)
        """
        for _ in range(num_steps):
            self.gym.simulate(self.sim)
            self._refresh()
            if render_pcd:
                self.vis_pcd()
            self.render()

    def render_multi(self, num_steps=1):
        """
        Render the simulation. (for debugging purposes)
        """
        for _ in range(num_steps):
            self.render()

    def vis_pcd(self, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        for i in range(self.num_envs):
            # draw point clouds
            points = self.combined_pcds[i].cpu().numpy()

            # Parameters
            offset = np.array([0.005, 0.0, 0.0], dtype=np.float32)  # small x-direction offset for line
            num_points = points.shape[0]

            # Prepare flattened vertices list: [x1,y1,z1,x2,y2,z2,...]
            verts_flat = []
            for p in points:
                p0 = p - offset
                p1 = p + offset
                verts_flat.extend([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]])

            # Colors: same RGB for each line
            color = [1.0, 0.0, 0.0]  # red
            colors_flat = color * num_points  # repeat for each line

            # Add lines to viewer
            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                num_points,     # num_lines = num points
                verts_flat,     # flat list of start/end points
                colors_flat     # flat list of RGB triples
            )

    def _draw_object_center_cross(self, half_extent=0.2, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        centers = self.states["object_center_pos"].detach().cpu().numpy()
        grasp_targets = None
        if "object_grasp_target_pos" in self.states:
            grasp_targets = self.states["object_grasp_target_pos"].detach().cpu().numpy()
        for i in range(self.num_envs):
            c = centers[i]
            verts_flat = [
                c[0] - half_extent, c[1], c[2],  c[0] + half_extent, c[1], c[2],
                c[0], c[1] - half_extent, c[2],  c[0], c[1] + half_extent, c[2],
                c[0], c[1], c[2] - half_extent,  c[0], c[1], c[2] + half_extent,
            ]
            colors_flat = [1.0, 0.0, 0.0] * 3  # red: object center
            if grasp_targets is not None:
                g = grasp_targets[i]
                verts_flat.extend([
                    g[0] - half_extent, g[1], g[2],  g[0] + half_extent, g[1], g[2],
                    g[0], g[1] - half_extent, g[2],  g[0], g[1] + half_extent, g[2],
                    g[0], g[1], g[2] - half_extent,  g[0], g[1], g[2] + half_extent,
                ])
                colors_flat.extend([0.0, 0.8, 1.0] * 3)  # cyan: grasp target
            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                len(verts_flat) // 6,
                verts_flat,
                colors_flat
            )

    def _draw_world_x_axis(self, half_extent=0.08, clear_lines=True):
        if clear_lines:
            self.gym.clear_lines(self.viewer)
        centers = self.states["object_center_pos"].detach().cpu().numpy()
        grasp_targets = None
        if "object_grasp_target_pos" in self.states:
            grasp_targets = self.states["object_grasp_target_pos"].detach().cpu().numpy()
        for i in range(self.num_envs):
            c = centers[i]
            verts_flat = [
                c[0] - half_extent, c[1], c[2],  c[0] + half_extent, c[1], c[2],
            ]
            colors_flat = [1.0, 0.0, 0.0]  # red: object center x-axis
            if grasp_targets is not None:
                g = grasp_targets[i]
                verts_flat.extend([
                    g[0] - half_extent, g[1], g[2],  g[0] + half_extent, g[1], g[2],
                ])
                colors_flat.extend([0.0, 0.8, 1.0])  # cyan: grasp target x-axis
            self.gym.add_lines(
                self.viewer,
                self.envs[i],
                len(verts_flat) // 6,
                verts_flat,
                colors_flat
            )

    def reset_idx(self, env_ids=None):
        # Domain randomization, can happen only at reset time since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._reset_object_state(env_ids) # reset object state

        if env_ids.numel() == 0:
            return

        reset_noise = torch.rand((len(env_ids), 23), device=self.device) # [0, 1]
        reset_noise = 2.0 * (reset_noise - 0.5) # [-1, 1]
        reset_noise[:, :7] *= self.reset_noise_scale["franka"]

        if self.reset_noise_scale["leap"] is None:
            reset_noise[:, 7:] = self.unnormalize_robot_joints(reset_noise[:, 7:], robot="hand", delta=False)
            reset_noise[:, 7:] -= self.canonical_joint_config[env_ids, 7:]
        else:
            reset_noise[:, 7:] *= self.reset_noise_scale["leap"]

        reset_joint_config = tensor_clamp(
            self.canonical_joint_config[env_ids] + reset_noise,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        self.set_robot_joint_state(reset_joint_config, env_ids=env_ids)
        self.success_flags_instant[env_ids] = 0
        self.lifting_flags_instant[env_ids] = 0
        # @ray have to update in reset_idx instead of compute_reward otherwise the duration will be overwritten to 0
        self.success_flags[env_ids] = self.success_long_enough[env_ids].float()
        self.lifting_flags[env_ids] = self.lifting_long_enough[env_ids].float()
        self.success_duration[env_ids] = 0
        self.lifting_duration[env_ids] = 0
        self.success_long_enough[env_ids] = False
        self.lifting_long_enough[env_ids] = False
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        if self.object_wrench_args["enable"]:
            self.object_applied_forces[env_ids] = 0.0
            self.object_applied_torques[env_ids] = 0.0
            self.rigid_body_forces[env_ids] = 0
            self.rigid_body_torques[env_ids] = 0

    def _apply_object_wrench(self):
        curriculum_factor = min(self.sim_steps / self.object_wrench_args["curri_steps"], 1)
        max_linear_force = self.object_wrench_args["max_linear_force"] * curriculum_factor
        linear_force_mag = max_linear_force * torch.rand(self.num_envs, 1, device=self.device)
        torque_mag = (linear_force_mag * self.object_wrench_args["torsional_radius"])
        rand_forces =\
            linear_force_mag * torch.nn.functional.normalize(
                torch.randn(self.num_envs, 3, device=self.device),
                dim=-1
            )
        rand_torques =\
            torque_mag * torch.nn.functional.normalize(
                torch.randn(self.num_envs, 3, device=self.device),
                dim=-1
            )

        num_trigger_steps = int(round(self.object_wrench_args["trigger_duration"] / self.dt))
        activation_dis = self.object_wrench_args["activation_dis"]
        apply_wrench = ( self.states["object_to_eef"].norm(dim=-1) < activation_dis ) | self.lifting_5cm_per_step

        self.object_applied_forces = torch.where(
            ((self.progress_buf % num_trigger_steps) == 0).unsqueeze(-1),
            rand_forces,
            self.object_applied_forces
        )

        self.object_applied_forces = torch.where(
            apply_wrench.unsqueeze(-1),
            self.object_applied_forces,
            torch.zeros_like(self.object_applied_forces)
        )

        self.object_applied_torques = torch.where(
            ((self.progress_buf % num_trigger_steps) == 0).unsqueeze(-1),
            rand_torques,
            self.object_applied_torques
        )

        self.object_applied_torques = torch.where(
            apply_wrench.unsqueeze(-1),
            self.object_applied_torques,
            torch.zeros_like(self.object_applied_torques)
        )

        # NOTE: this assumes object body is always the last rigid body in the env, which mean object actor must be created last in _create_envs
        self.rigid_body_forces[:, -1, :] = self.object_applied_forces
        self.rigid_body_torques[:, -1, :] = self.object_applied_torques

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.rigid_body_forces),
            gymtorch.unwrap_tensor(self.rigid_body_torques),
            gymapi.ENV_SPACE,  # ENV_SPACE (world) or LOCAL_SPACE
        )

    def pre_physics_step(self, actions):
        """
        Args:
            actions (torch.Tensor): normalized delta joint angles (num_selected_envs, 7+4*4)
        """
        if self.eef_actions:
            self.delta_eef_actions = actions.clone()
            pos_actions = actions[:, 0:3] * self.action_scale["eef_pos"] * self.dt
            ctrl_target_eef_pos = self.states['eef_pos'] + pos_actions

            # Interpret actions as target rot (axis-angle) displacements
            rot_actions = actions[:, 3:6] * self.action_scale["eef_rot"] * self.dt
            angle = torch.norm(rot_actions, p=2, dim=-1)
            axis = rot_actions / angle.unsqueeze(-1)
            rot_actions_quat = quat_from_angle_axis(angle, axis)

            # clamp tiny rotations to avoid numerical issues
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
                rot_actions_quat,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(
                    self.num_envs, 1
                ),
            )
            ctrl_target_eef_quat = quat_mul(
                rot_actions_quat, self.states['eef_quat'] # xyzw format
            )

            delta_arm_joint_actions_unnormalized = eef_ctrl.compute_dof_pos_delta(
                arm_dof_pos=self.states['q'][:, :7],
                current_eef_pos=self.states['eef_pos'],
                current_eef_quat=self.states['eef_quat'],
                jacobian=self._j_eef,
                ctrl_target_eef_pos=ctrl_target_eef_pos,
                ctrl_target_eef_quat=ctrl_target_eef_quat,
            )

            hand_actions = actions[:, 6:] * self.action_scale["leap"] * self.dt
            delta_hand_joint_actions_unnormalized = self.unnormalize_robot_joints(hand_actions, robot="hand", delta=True)
        else:
            arm_actions = actions[:, :7] * self.action_scale["franka"] * self.dt
            hand_actions = actions[:, 7:] * self.action_scale["leap"] * self.dt
            delta_arm_joint_actions_unnormalized = self.unnormalize_robot_joints(arm_actions, robot="arm", delta=True)
            delta_hand_joint_actions_unnormalized = self.unnormalize_robot_joints(hand_actions, robot="hand", delta=True)

        self.delta_joint_actions[:, :7] = delta_arm_joint_actions_unnormalized
        self.delta_joint_actions[:, 7:] = delta_hand_joint_actions_unnormalized

        self.abs_actions[:] = self.states['q'] + self.delta_joint_actions # need to really make sure states['q'] is always up to date
        self.abs_actions[:] = tensor_clamp(
            self.abs_actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.abs_actions))

        # Add F/T wrench to object
        if self.object_wrench_args["enable"]:
            self._apply_object_wrench()

    def post_physics_step(self):
        self.progress_buf += 1
        self.teleport_buf += 1
        self.teleport_buf = self.teleport_buf % self.object_teleport_args['n2']

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.reset_idx(env_ids)

        self.compute_observations()
        if self.debug_viz and self.viewer is not None:
            self.vis_pcd(clear_lines=True)
            # self._draw_object_center_cross(clear_lines=False)
            self._draw_world_x_axis(clear_lines=False)
        self.compute_reward()

        # video logging
        if self.video_logging["capture"]:
            self.video_logger()
        self.sim_steps += 1

    @abstractmethod
    def _create_envs(self, spacing, num_per_row):
        self.table_surface_height = ...
        self.mesh_aabb_extents = ...
        self.obj_pos_range = ...
        self.envs = ...
        self.num_objects = ...

    @abstractmethod
    def compute_reward(self):
        pass

    @abstractmethod
    def compute_observations(self):
        self._refresh()
        return self.obs_buf


@hydra.main(config_name="config", config_path="../cfg/")
def launch_test(cfg: DictConfig):
    np.random.seed(0)
    torch.manual_seed(0)
    cfg_dict = omegaconf_to_dict(cfg)
    cfg_task = cfg_dict["task"]
    rl_device = cfg_dict["rl_device"]
    sim_device = cfg_dict["sim_device"]
    headless = cfg_dict["headless"]
    graphics_device_id = 0
    virtual_screen_capture = False
    force_render = False
    env = FrankaLEAP(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    for i in tqdm(range(1000)):
        t1 = time.time()
        env.reset_idx()
        import ipdb ; ipdb.set_trace()
        t2 = time.time()
        print(f"Reset time: {t2 - t1}")
        env.render()


if __name__ == "__main__":
    launch_test()
