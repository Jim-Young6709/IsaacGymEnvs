"""
Franka + LEAP Hand Env

['base_x_joint', 'base_y_joint', 'base_rotation_joint',
 'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7',
 'finger_joint_1', 'finger_joint_0', 'finger_joint_2', 'finger_joint_3',
 'finger_joint_12', 'finger_joint_13', 'finger_joint_14', 'finger_joint_15',
 'finger_joint_5', 'finger_joint_4', 'finger_joint_6', 'finger_joint_7',
 'finger_joint_9', 'finger_joint_8', 'finger_joint_10', 'finger_joint_11',
 'x5_joint1', 'x5_joint2', 'x5_joint3', 'x5_joint4', 'x5_joint5', 'x5_joint6',

TODO:
1. setup fabric open loop + local policy distillation
2. tune obstacle rand for the mobile base
3. tune switching part

"""

import os
import re
import shutil
import hashlib
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
from isaacgymenvs.utils.rotation_conversions import quaternion_to_matrix_ig, matrix_to_rotation_6d, se2_transform
from isaacgymenvs.utils.pcd_utils import transform_pcds_to_world, compute_scene_oracle_pcd, FrankaLeapSampler, GlorbotSampler
from isaacgymenvs.utils.viser_visualizer import ViserVisualizer
from isaacgymenvs.utils.simulate_depth_cam_compile import simulate_depth_cam_render_from_pose
from isaacgymenvs.utils.simulate_lidar_compile import simulate_lidar_render_from_pose
from isaacgymenvs.utils.glorbot_collision_checker import GlorbotCollisionChecker
from omegaconf import DictConfig
from tqdm import tqdm
import random
from scipy.spatial.transform import Rotation as R
from curobo.types.math import Pose

from fabrics_sim.fabrics.glorbot_vision_fabric import GlorbotVisionFabric
from fabrics_sim.integrator.integrators import DisplacementIntegrator
from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel
from fabrics_sim.utils.utils import initialize_warp



class FrankaLEAPMobile(VecTask):
    # class inits
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.device = sim_device
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.reset_noise_scale = self.cfg["env"]["resetNoiseScale"]
        self.eef_actions = True if self.cfg["env"]["numActions"] == 22 else False
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.mesh_args = self.cfg["env"]["mesh"]
        self.object_wrench_args = self.cfg["env"]["object_wrench"]
        self.object_teleport_args = self.cfg["env"]["object_teleport"]
        self.eef_init = self.cfg["env"]["eef_init"]
        self.distractor_settings = self.cfg["env"]["distractor_settings"]
        self.action_history_len = int(self.cfg["env"].get("action_history_len", 0))
        self.enable_fabric = self.cfg['fabric']['enable']
        self.video_logging = self.cfg["env"]["video_logging"]
        self.video_dir = os.path.join('videos', self.cfg["name"] + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        os.makedirs(self.video_dir, exist_ok=True)

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
        self.enable_viser = self.cfg['env']['enable_viser'] and (not self.headless)
        if self.enable_viser:
            self._init_viser_visualizer()
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

        self._q_prev = None                     # Previous joint positions (n_envs, n_dof)
        self._qd_prev = None                    # Previous joint velocities (n_envs, n_dof)

        # pcd
        self.static_pcds = []
        self.distractor_pcds = []
        self.object_pcds = []
        self.combined_pcds = []
        self.static_scene_pcd_t0 = None
        self.object_pcd_t0 = None

        # load franka asset path
        self.asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"]["assetRoot"])
        self.robot_asset_file = self.cfg["env"]["asset"]["assetFileNameFranka"]
        self.full_robot_asset_path = os.path.join(self.asset_root, self.robot_asset_file)

        # init fabric
        if self.enable_fabric:
            self.obstacle_count = 0
            self.max_objects_per_env = 20 # its like allocating a buffer? need to check inside create env and update
            self.fabrics_world_dict = dict()

            # Set GPU device
            device_int = 0
            # Set the warp cache directory based on device int
            initialize_warp(str(device_int))

    def _post_init_buffers(self):
        if not hasattr(self, 'canonical_joint_config'):
            base_init_range = torch.tensor(self.cfg['env']['robot_init']['base_init_range'], device=self.device)
            base_init_pose = torch.rand((self.num_envs, 3), device=self.device) * (base_init_range[1] - base_init_range[0]) + base_init_range[0]
            base_init_pose[:, 1] += getattr(self, "box_pos", torch.zeros_like(base_init_pose))[:, 1]

            self.canonical_joint_config = torch.tensor(
                [
                    [0.0, 0.0, 0.0] + \
                    [0.0, -0.25*np.pi, 0.0, -0.75*np.pi, 0.0, 0.5*np.pi, 0.0] + \
                    # [0.0, -0.25*np.pi, 0.0, -0.75*np.pi, 0.0, 0.5*np.pi, -np.pi/2] + \
                    # [-0.5*np.pi, -0.25*np.pi, 0.0, -0.75*np.pi, 0.5*np.pi, 0.5*np.pi, 0.0] + \
                    # [-0.25*np.pi, -0.25*np.pi, 0.0, -0.75*np.pi, 0.25*np.pi, 0.5*np.pi, 0.0] + \
                    # [0.0, 0.0, 0.0, 0.0,
                    #  0.0, 0.0, 1.0, 0.57,
                    #  0.0, 0.0, 0.0, 0.0,
                    #  0.0, 0.0, 0.0, 0.0,] + \
                    [1.57, 0.0, 1.57, 0.0,
                     0.0, 0.0, 1.0,  0.57,
                     1.57, 0.0, 1.57, 0.0,
                     1.57, 0.0, 1.57, 0.0,] + \
                    [0.0, 1.0, 2.0, -1.0, 0.0, 0.0]
                ] * self.num_envs
            ).to(self.device)
            self.canonical_joint_config[:, :3] = base_init_pose

        if not hasattr(self, 'default_reset_joint_config'):
            self.default_reset_joint_config = self.canonical_joint_config.clone()

        self.ik_regularization_config = self.canonical_joint_config[:, 3:10]
        self.delta_joint_actions = torch.zeros((self.num_envs, self.num_robot_dofs), device=self.device, dtype=torch.float) # Current delta actions to be deployed
        self.delta_eef_actions = torch.zeros((self.num_envs, self.num_robot_dofs-1), device=self.device, dtype=torch.float) # Current delta actions to be deployed at the end effector
        self.success_flags = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device) # 1 if success condition has been achieved at any step, 0 otherwise
        self.success_5cm_per_step = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device) # success within 5cm threshold
        self.lifting_flags = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self.lifting_5cm_per_step = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.static_scene_pcd_t0 = self.static_pcds.clone()

        # for distillation purposes
        self.distillation_mode = False
        self.distillation_steps = 0 # if use distillation mode, this should get tracked in the distillation script
        self.abs_actions = torch.zeros(self.num_envs, 32, device=self.device)
        self.teacher_actions_converted = torch.zeros(self.num_envs, 32, device=self.device)
        self.action_history_buf = torch.zeros(
            (self.num_envs, self.action_history_len, self.abs_actions.shape[1]),
            device=self.device,
            dtype=torch.float,
        )
        # student policy actions space (should get overridden in the distillation class)
        self.delta_franka_action = True
        self.delta_leap_action = True
        self.delta_arx_action = True

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
        # Codex
        # Latched until consumed by distillation logic, so resets across chunked steps are preserved.
        self.object_reset_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # Codex
        # Pending resets are promoted after one post-physics pass so downstream logic reads post-reset state.
        self.object_reset_pending_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

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
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

        tensor_args = TensorDeviceType()
        base_link = "panda_link0"
        ee_link = "palm_center"
        robot_cfg = RobotConfig.from_basic(self.full_robot_asset_path, base_link, ee_link, tensor_args)

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
        plane_params.distance = 1.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_franka_leap(self):
        self.robot_pcd_sampler = GlorbotSampler(
            urdf_path=self.full_robot_asset_path,
            device=self.device,
            num_points=self.pcd_spec_dict["num_robot_points"],
        )
        self.robot_spherical_representation = GlorbotCollisionChecker(
            urdf_path=self.full_robot_asset_path,
            device=self.device,
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

        robot_asset = self.gym.load_asset(self.sim, self.asset_root, self.robot_asset_file, asset_options)
        self.robot_asset = robot_asset

        # currently only support joint position control
        robot_dof_stiffness = to_torch([800.0*100]*2 + [800.0*10] + [1000.0]*7 + [800.0]*16 + [800.0]*6, dtype=torch.float, device=self.device)
        robot_dof_damping = to_torch([40.0*100]*2 + [40.0*10] + [50.0]*7 + [40.0]*16 + [40.0]*6, dtype=torch.float, device=self.device)
        
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

        self._robot_effort_limits = to_torch(self._robot_effort_limits, device=self.device)
        return robot_dof_props

    def _init_fabric(self):
        self.fabrics_world_model = WorldMeshesModel(
            batch_size=self.num_envs,
            max_objects_per_env=self.max_objects_per_env,
            device=self.device,
            world_dict=self.fabrics_world_dict,
        )
        self.fabrics_object_ids, self.fabrics_object_indicator = self.fabrics_world_model.get_object_ids()

        # Create franka fabric
        self.franka_fabric = GlorbotVisionFabric(self.num_envs, self.device)

        # Create integrator for the fabric dynamics.
        self.franka_integrator = DisplacementIntegrator(self.franka_fabric)

        cspace_dim = 3 + 7 + 6
        self.fabric_q = torch.zeros((self.num_envs, cspace_dim), dtype=torch.float, device=self.device)
        self.fabric_qd = torch.zeros((self.num_envs, cspace_dim), dtype=torch.float, device=self.device)
        self.fabric_qdd = torch.zeros((self.num_envs, cspace_dim), dtype=torch.float, device=self.device)

        self.fabric_switch_enable = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device) # 0 -- disable ; 1 -- enable
        self.switch_pos_offset = torch.tensor(self.cfg['env']['robot_init']['switch_pos_offset'], device=self.device)
        self.switch_tol = self.cfg['env']['robot_init']['switch_tol']

        self._setup_fabric_switching_target()

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
        self._j_eef = jacobian[:, hand_joint_index, :, :10]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, 3:10, 3:10]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * actor_num, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1) # 3 actors, franka, table, table_stand

        target_pos = to_torch(self.cfg["reward"]["params"]["target_pos"], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        target_lift_dis = to_torch(self.cfg["reward"]["params"]["target_lift_dis"], device=self.device)
        target_quat = to_torch(self.cfg["reward"]["params"]["target_quat"], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        target_quat_norm = torch.norm(target_quat, dim=1, keepdim=True)  # normalize quaternion
        target_quat = target_quat / (target_quat_norm + 1e-10)

        # finger indexing: 0-3:index ; 4-7:thumb ; 8-11:middle ; 12-15:ring
        # v0
        # self.grasp_finger_dof_pos = torch.tensor([
        #     1.0176, -0.8376,  0.9564,  0.9632,
        #     1.5700,  0.0000,  0.3100,  1.2880,
        #     1.0176,  0.0000,  0.9564,  0.9632,
        #     1.0176,  0.8376,  0.9564,  0.9632
        # ], device=self.device)

        # v1
        self.grasp_finger_dof_pos = torch.tensor([
            0.65,  0.0,  0.65,  0.65,
            1.57,  0.0,  0.10,  0.40,
            0.65,  0.0,  0.65,  0.65,
            0.65,  0.0,  0.65,  0.65,
        ], device=self.device)

        self.reward_settings = {
            "target_pos": target_pos,
            "target_lift_dis": target_lift_dis,
            "target_quat": target_quat,
            "target_rot_6d": matrix_to_rotation_6d(quaternion_to_matrix_ig(target_quat)),
            "curl_reaching_threshold": to_torch(self.cfg["reward"]["params"]["curl_reaching_threshold"], device=self.device),
            "object_init_height": self.mesh_aabb_extents[:, 2] / 2 + self.table_surface_height,
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
        }

    # object spawning utils
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
            size (float): radius of the sphere, scalar value
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
        if mass is None:
            return urdf_rel, mesh_dir

        # Create a temp cached copy of the mesh folder and patch URDF mass there.
        mass_value = float(mass)
        scale_vec = np.asarray(scale, dtype=np.float32).reshape(-1)
        if scale_vec.shape[0] == 1:
            scale_vec = np.repeat(scale_vec, 3)
        assert scale_vec.shape[0] == 3, "URDF scale must be scalar or shape (3,)"
        mass_tag = f"{mass_value:.8g}".replace(".", "p").replace("-", "m")
        scale_tag = "_".join([f"{float(s):.6g}".replace(".", "p").replace("-", "m") for s in scale_vec])
        mesh_dir_hash = hashlib.sha1(mesh_dir.encode("utf-8")).hexdigest()[:10]
        cache_root = os.environ.get(
            "ISAACGYM_URDF_CACHE_ROOT",
            os.environ.get("WARP_CACHE_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "isaacgym_urdf_overrides")),
        )
        cache_mesh_dir = os.path.join(cache_root, f"{mesh_name}_{mesh_dir_hash}_mass_{mass_tag}_scale_{scale_tag}")
        os.makedirs(cache_root, exist_ok=True)
        if not os.path.exists(cache_mesh_dir):
            shutil.copytree(mesh_dir, cache_mesh_dir)

        cache_urdf_path = os.path.join(cache_mesh_dir, urdf_rel)
        with open(cache_urdf_path, "r") as f:
            urdf_text = f.read()
        patched_urdf_text, n_sub = re.subn(
            r'(<mass\s+value\s*=\s*")[^"]+("\s*/?>)',
            rf'\g<1>{mass_value:.8g}\2',
            urdf_text,
        )
        if n_sub == 0:
            raise ValueError(f"No <mass value=...> tag found in URDF: {cache_urdf_path}")

        scale_str = f"{float(scale_vec[0]):.8g} {float(scale_vec[1]):.8g} {float(scale_vec[2]):.8g}"
        patched_urdf_text, n_scale_sub = re.subn(
            r'(<mesh\b[^>]*\bscale\s*=\s*")[^"]+(")',
            rf'\g<1>{scale_str}\2',
            patched_urdf_text,
        )
        if n_scale_sub == 0:
            raise ValueError(f"No <mesh ... scale=\"...\"> tag found in URDF: {cache_urdf_path}")

        if patched_urdf_text != urdf_text:
            with open(cache_urdf_path, "w") as f:
                f.write(patched_urdf_text)
        return urdf_rel, cache_mesh_dir

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

    def _create_mesh(
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
        if np.isscalar(scale):
            mesh_scale = [float(scale), float(scale), float(scale)]
        else:
            mesh_scale_arr = np.asarray(scale, dtype=np.float32).reshape(-1)
            assert mesh_scale_arr.shape[0] == 3, "Mesh scale must be scalar or shape (3,)"
            mesh_scale = mesh_scale_arr.tolist()
        mesh_mass_range = self.cfg["env"]["object_settings"]["mass_range"]
        sampled_mesh_mass = None
        if mesh_mass_range is not None:
            mass_lo = float(mesh_mass_range[0])
            mass_hi = float(mesh_mass_range[1])
            sampled_mesh_mass = float(np.random.uniform(mass_lo, mass_hi))
        urdf_path, asset_root = self._create_mesh_urdf(
            mesh_path,
            scale=mesh_scale,
            mass=sampled_mesh_mass,
        )  # CODEX


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

        if asset_mesh_id is None:
            asset_mesh_id = Path(mesh_path).parts[-2]
        if asset_obj_id is None:
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

    def _discover_variant_mesh_entries(self, mesh_dir, object_list):
        """
        Discover meshes for both legacy and variant layouts.
        Returns:
            List[dict] with keys:
                mesh_path, asset_obj_id, asset_mesh_id, display_name
        """
        type_mapping_path = os.path.join(mesh_dir, "type_mapping.json")
        entries = []

        if os.path.isfile(type_mapping_path):
            # Legacy layout: <mesh_dir>/<object>/<mesh>.obj with type_mapping.json at root.
            with open(type_mapping_path, "r") as f:
                obj_str2int = json.load(f)
            if object_list == ["all"]:
                object_list = [obj for obj in os.listdir(mesh_dir) if obj != "type_mapping.json"]
            object_list = sorted(
                object_list,
                key=lambda obj: (0, obj_str2int[obj]) if obj in obj_str2int else (1, obj),
            )
            for obj in object_list:
                obj_dir = os.path.join(mesh_dir, obj)
                if not os.path.isdir(obj_dir):
                    continue
                for file in sorted(os.listdir(obj_dir)):
                    if not file.endswith(".obj"):
                        continue
                    mesh_path = os.path.join(obj_dir, file)
                    asset_mesh_id = Path(mesh_path).parts[-2]
                    entries.append(
                        {
                            "mesh_path": mesh_path,
                            "asset_obj_id": int(obj_str2int[asset_mesh_id]),
                            "asset_mesh_id": asset_mesh_id,
                            "display_name": asset_mesh_id,
                        }
                    )
            return entries

        # Variant layout:
        # - <mesh_dir>/<category>/<object>/<variant>/<object>_<variant>.obj
        # - <mesh_dir>/<object>/<variant>/<object>_<variant>.obj
        if object_list == ["all"]:
            selected_objects = None
        else:
            selected_objects = set(object_list)

        variant_entries = []
        for root, _, files in os.walk(mesh_dir):
            obj_files = sorted([f for f in files if f.endswith(".obj")])
            if not obj_files:
                continue
            rel_dir = os.path.relpath(root, mesh_dir)
            parts = rel_dir.split(os.sep)
            if len(parts) == 3:
                category, object_name, variant = parts
                mesh_rel_prefix = os.path.join(category, object_name, variant)
            elif len(parts) == 2:
                object_name, variant = parts
                category = Path(mesh_dir).name
                mesh_rel_prefix = os.path.join(object_name, variant)
            else:
                continue
            if selected_objects is not None and object_name not in selected_objects:
                continue
            for obj_file in obj_files:
                if not obj_file.startswith(f"{object_name}_{variant}"):
                    continue
                mesh_path = os.path.join(root, obj_file)
                mesh_stem = os.path.splitext(obj_file)[0]
                json_path = os.path.join(root, mesh_stem + ".json")
                if not os.path.isfile(json_path):
                    continue
                with open(json_path, "r") as jf:
                    meta = json.load(jf)
                if meta["transform_model"] != "dilation_only_v1":
                    continue
                # mesh_id can be a nested relative path; ObjaMesh handles this directly.
                mesh_id_rel = os.path.join(mesh_rel_prefix, mesh_stem)
                display_name = f"{object_name}_{variant}"
                variant_entries.append((display_name, mesh_path, mesh_id_rel, meta["dilation_range"]))

        variant_entries = sorted(variant_entries, key=lambda x: (x[0], x[1]))
        for idx, (display_name, mesh_path, mesh_id_rel, dilation_range) in enumerate(variant_entries):
            entries.append(
                {
                    "mesh_path": mesh_path,
                    "asset_obj_id": idx + 1,  # keep >0 for ObjaMesh filtering
                    "asset_mesh_id": mesh_id_rel,
                    "display_name": display_name,
                    "dilation_range": dilation_range,
                }
            )
        return entries

    def create_rand_mesh(self, fix_base_link=False):
        # get randomly sampled mesh path
        mesh_dir = self.mesh_args["mesh_dir"]
        object_list = self.mesh_args["obj_list"]
        mesh_entries = self._discover_variant_mesh_entries(mesh_dir, object_list)
        sampled = random.choice(mesh_entries)
        sampled_mesh_path = sampled["mesh_path"]

        # sample random size, pos and ori
        scale_range = self.cfg["env"]["object_settings"]["scale_range"]
        pos_range = self.cfg["env"]["object_settings"]["xyz_range"]

        if "dilation_range" in sampled:
            d = sampled["dilation_range"]
            mesh_scale = np.array(
                [
                    np.random.uniform(float(d["x"]["min"]), float(d["x"]["max"])),
                    np.random.uniform(float(d["y"]["min"]), float(d["y"]["max"])),
                    np.random.uniform(float(d["z"]["min"]), float(d["z"]["max"])),
                ],
                dtype=np.float32,
            )
        else:
            mesh_scale = np.random.uniform(scale_range[0], scale_range[1])
        mesh_pos = np.random.uniform(pos_range[0], pos_range[1])
        mesh_quat = R.random().as_quat()  # [x, y, z, w]

        return self._create_mesh(
            sampled_mesh_path,
            mesh_pos,
            mesh_scale,
            mesh_quat,
            fix_base_link,
            asset_obj_id=sampled["asset_obj_id"],
            asset_mesh_id=sampled["asset_mesh_id"],
        )

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

        mesh_entries = self._discover_variant_mesh_entries(mesh_dir, object_list)
        if len(mesh_entries) == 0:
            raise RuntimeError(
                f"No mesh entries discovered under mesh_dir={mesh_dir} "
                f"with obj_list={object_list}. "
                "Expected legacy layout with type_mapping.json or variant layout "
                "(mesh_dir/object/variant/*.obj or mesh_dir/category/object/variant/*.obj)."
            )
        self.mesh_variant_mode = ("dilation_range" in mesh_entries[0])
        if self.mesh_variant_mode:
            base_entries = mesh_entries
            # CODEX: keep per-object metrics keyed by true variants (not expanded preload copies).
            self.object_id_to_name = [e["display_name"] for e in base_entries]

            variant_count = len(base_entries)
            preload_multiplier = int(self.mesh_args.get("variant_preload_multiplier", 1))
            if preload_multiplier < 1:
                raise ValueError("env.mesh.variant_preload_multiplier must be >= 1")

            target_preload = variant_count * preload_multiplier
            max_preload = int(self.mesh_args.get("variant_preload_max", self.num_envs))
            if max_preload > 0:
                target_preload = min(target_preload, max_preload)
            target_preload = min(target_preload, self.num_envs)
            target_preload = max(1, target_preload)

            # Build a pooled preload list by repeating variants as needed, then shuffle.
            idx = np.arange(target_preload, dtype=np.int64) % variant_count
            np.random.shuffle(idx)
            mesh_entries = [base_entries[int(i)] for i in idx.tolist()]
        else:
            self.object_id_to_name = [e["display_name"] for e in mesh_entries]

        meshes = []
        for entry in tqdm(mesh_entries, desc="Preparing Meshes"):
            # sample random size, pos and ori
            scale_range = self.cfg["env"]["object_settings"]["scale_range"]
            pos_range = self.cfg["env"]["object_settings"]["xyz_range"]

            mesh_scale = np.random.uniform(scale_range[0], scale_range[1])
            mesh_pos = np.random.uniform(pos_range[0], pos_range[1])
            mesh_quat = R.random().as_quat()
            asset, start_pose, scale, asset_obj_id, asset_mesh_id = self._create_mesh(
                entry["mesh_path"],
                mesh_pos,
                mesh_scale,
                mesh_quat,
                fix_base_link,
                asset_obj_id=entry["asset_obj_id"],
                asset_mesh_id=entry["asset_mesh_id"],
            )
            meshes.append((asset, start_pose, scale, asset_obj_id, asset_mesh_id))

        return meshes

    def _create_distractor_pcd(self):
        """
        create distractor objects under/behind/side the table to approximate real world setting
        since the robot will never interact with these objects, we only create pcd for them rather than actually spawning them in sim
        """

        def _sample_random_distractors(pos_range):
            """
            sample random distractor objects within the given pos region

            Args:
                pos_range: List[[x_min, y_min, z_min], [x_max, y_max, z_max]], note this is the boundary range not the object center pos range
            """
            if pos_range[1][2] <= 0:
                return

            _params = self.distractor_settings["params"]
            rand01 = np.random.uniform(0.0, 1.0)
            if rand01 < _params["skip_prob"]:
                return
            elif rand01 < (_params["skip_prob"] + _params["full_prob"]):
                _pos_range = np.array(pos_range)
                _cuboid_dim = _pos_range[1] - _pos_range[0]
                _cuboid_pos = (_pos_range[0] + _pos_range[1]) / 2
                _cuboid_quat = np.array([0.0, 0.0, 0.0, 1.0])
                cuboid_dims.append(_cuboid_dim)
                cuboid_pos.append(_cuboid_pos)
                cuboid_quats.append(_cuboid_quat)
                return

            _num_range = _params["num_distractors_per_region_range"]
            _cuboid_size_range = _params["cuboid_size_range"]
            _cylinder_size_range = _params["cylinder_size_range"]
            _sphere_size_range = _params["sphere_size_range"]

            _num = np.random.randint(_num_range[0], _num_range[1]+1)
            for _ in range(_num):
                _type = random.choice([0, 0, 0, 1, 1, 2]) # biased sampling
                _pos_range = np.array(pos_range)
                height_limit = _pos_range[1][2] - _pos_range[0][2]
                if _type == 0: # cuboid
                    _cuboid_dim = np.random.uniform(_cuboid_size_range[0], _cuboid_size_range[1])
                    if _cuboid_dim[2] > height_limit:
                        _cuboid_dim[2] = height_limit
                    _pos_range[0] += _cuboid_dim / 2
                    _pos_range[1] -= _cuboid_dim / 2
                    _cuboid_pos = np.random.uniform(_pos_range[0], _pos_range[1])
                    _cuboid_quat = np.array([0.0, 0.0, 0.0, 1.0])
                    cuboid_dims.append(_cuboid_dim)
                    cuboid_pos.append(_cuboid_pos)
                    cuboid_quats.append(_cuboid_quat)
                elif _type == 1: # cylinder
                    _cylinder_dim = np.random.uniform(_cylinder_size_range[0], _cylinder_size_range[1])
                    _cylinder_radius = _cylinder_dim[0]
                    _cylinder_height = _cylinder_dim[1]
                    if _cylinder_height > height_limit:
                        _cylinder_height = height_limit
                    _pos_range_offset = np.array([_cylinder_radius, _cylinder_radius, _cylinder_height / 2])
                    _pos_range[0] += _pos_range_offset
                    _pos_range[1] -= _pos_range_offset
                    _cylinder_pos = np.random.uniform(_pos_range[0], _pos_range[1])
                    _cylinder_quat = np.array([0.0, 0.0, 0.0, 1.0])
                    cylinder_radii.append(_cylinder_radius)
                    cylinder_heights.append(_cylinder_height)
                    cylinder_pos.append(_cylinder_pos)
                    cylinder_quats.append(_cylinder_quat)
                elif _type == 2: # sphere
                    _sphere_dim = np.random.uniform(_sphere_size_range[0], _sphere_size_range[1])
                    _sphere_radius = _sphere_dim
                    if _sphere_radius > height_limit / 2:
                        _sphere_radius = height_limit / 2
                    _pos_range[0] += _sphere_radius
                    _pos_range[1] -= _sphere_radius
                    _sphere_pos = np.random.uniform(_pos_range[0], _pos_range[1])
                    sphere_radii.append(_sphere_radius)
                    sphere_pos.append(_sphere_pos)

        table_pos = self.table_pos.cpu().numpy()
        table_size = self.table_size.cpu().numpy()
        table_extend = self.distractor_settings["params"]["table_extend"]
        max_z_height = self.distractor_settings["params"]["free_space_distractor_max_height"]
        for i in range(self.num_envs):
            # init lists
            cuboid_dims = []  # xyz
            cuboid_pos = []
            cuboid_quats = [] # xyzw

            cylinder_radii = []
            cylinder_heights = []
            cylinder_pos = []
            cylinder_quats = []

            sphere_radii = []
            sphere_pos = []

            # ground plane
            cuboid_dims.append([4.0, 6.0, 0.001])
            cuboid_pos.append([-1.0, 0.0, -0.0005])
            cuboid_quats.append([0.0, 0.0, 0.0, 1.0])

            # adding distractor pos range when: side/under/behind the table
            table_x_min = table_pos[i][0] - table_size[i][0] / 2
            table_x_max = table_pos[i][0] + table_size[i][0] / 2
            table_y_min = table_pos[i][1] - table_size[i][1] / 2
            table_y_max = table_pos[i][1] + table_size[i][1] / 2
            table_z_min = table_pos[i][2] - table_size[i][2] / 2

            distractor_pos_range_list = [
                [ # side 1
                    [table_x_min, table_y_min - table_extend, 0.0],
                    [table_x_max + table_extend, table_y_min, max_z_height],
                ],
                [ # side 2
                    [table_x_min, table_y_max, 0.0],
                    [table_x_max + table_extend, table_y_max + table_extend, max_z_height],
                ],
                [ # behind
                    [table_x_max, table_y_min, 0.0],
                    [table_x_max + table_extend, table_y_max, max_z_height],
                ],
                [ # under
                    [table_x_min, table_y_min, 0.0],
                    [table_x_min + table_extend, table_y_max, table_z_min], # bias towards the front part of the table
                ],
            ]

            # under the table
            for subregion_distractor_pos_range in distractor_pos_range_list:
                _sample_random_distractors(subregion_distractor_pos_range)

            # get distractor pcd
            distractor_pcd_i = torch.from_numpy(compute_scene_oracle_pcd(
                num_obstacle_points=self.pcd_spec_dict["num_distractor_points"],
                cuboid_dims=np.array(cuboid_dims),
                cuboid_centers=np.array(cuboid_pos),
                cuboid_quats=np.array(cuboid_quats),
                cylinder_radii=np.array(cylinder_radii),
                cylinder_heights=np.array(cylinder_heights),
                cylinder_centers=np.array(cylinder_pos),
                cylinder_quats=np.array(cylinder_quats),
                sphere_centers=np.array(sphere_pos),
                sphere_radii=np.array(sphere_radii),
            )).to(self.device)
            self.distractor_pcds.append(distractor_pcd_i)

        self.distractor_pcds = torch.stack(self.distractor_pcds, dim=0).to(self.device).to(torch.float32) # (num_envs, num_distractor_points, 3)

    # fabric utils
    def _create_fabric_cube(self, pos, size, quat, env_id):
        """
        Args:
            pos  (list): (3,) xyz position of the cube center
            size (list): (3,) length along xyz direction of the cube
            quat (list): (4,) [x, y, z, w]
            env_id (int): environment index
        """
        self.obstacle_count += 1

        transform = list(pos) + list(quat)
        self.fabrics_world_dict[f"cube_{self.obstacle_count}"] = {
            "env_index": env_id,
            "type": "box",
            "scaling": " ".join(map(str, size)),
            "transform": " ".join(map(str, transform)),
        }
        return

    def _create_fabric_cylinder(self, pos, size, quat, env_id):
        """
        Args:
            pos  (list): (3,) xyz position of the cube center
            size (list): (2,) radius and height of the cylinder
            quat (list): (4,) [x, y, z, w]
            env_id (int): environment index
        """
        self.obstacle_count += 1

        transform = list(pos) + list(quat)
        self.fabrics_world_dict[f"cylinder_{self.obstacle_count}"] = {
            "env_index": env_id,
            "type": "cylinder",
            "scaling": " ".join(map(str, [2*size[0], 2*size[0], size[1]])), # default is 0.5 for radius and 1 for height
            "transform": " ".join(map(str, transform)),
        }
        return

    def _create_fabric_sphere(self, pos, radius, quat, env_id):
        """
        Args:
            pos  (list): (3,) xyz position of the cube center
            radius (float): (scalar) radius of the sphere
            quat (list): (4,) [x, y, z, w]
            env_id (int): environment index
        """
        self.obstacle_count += 1

        transform = list(pos) + list(quat)
        self.fabrics_world_dict[f"sphere_{self.obstacle_count}"] = {
            "env_index": env_id,
            "type": "sphere", # cylinder
            "scaling": " ".join(map(str, [radius, radius, radius])),
            "transform": " ".join(map(str, transform)),
        }
        return

    def compute_fabric_action(self, eef_target, gaze_target=None):
        # timestep: ideally 1/60 but something as low as 1/20 may work. The larger the dt, the more
        # unstable fabric may become.
        # speed_scalar: Anything over 3.5 seems to make the fabric unstable. 
        # Acceleration Limits in the Yaml: for the first 3 joints (base), can tune
        # Go into GlorbotVisionFabric class. In the set_features function, tune parameters that 
        # determine how close the base gets to the table.
        # damping_radius in forcing_base_position_attractor determines how close the base tends to stop in front
        # of the target (||base_center - ee_target[0:2]||^2)

        timestep = 1/30. # 1/60.

        self.fabric_q[:, :10] = self.states['q'][:, :10].clone()
        self.fabric_q[:, 10:] = self.states['q'][:, 26:].clone()

        qd_delta = (self.states['q'] - self.states['q_prev']) / timestep
        self.fabric_qd[:, :10] = qd_delta[:, :10].clone()
        self.fabric_qd[:, 10:] = qd_delta[:, 26:].clone()

        if gaze_target is None:
            gaze_target = self.states['object_center_pos'].clone()

        self.franka_fabric.set_features(
            eef_target,
            gaze_target,
            self.fabric_q.detach(),
            self.fabric_qd.detach(),
            self.fabrics_object_ids,
            self.fabrics_object_indicator,
        )

        self.fabric_q, self.fabric_qd, self.fabric_qdd = self.franka_integrator.step(
            self.fabric_q.detach(), self.fabric_qd.detach(), timestep, speed_scalar=1.5,
        )

        return self.fabric_q

    @abstractmethod
    def _setup_fabric_switching_target(self):
        self.switching_target_pos = ...
        self.switching_target_quat = ...

    # sim state update
    def _refresh(self):
        self._q_prev = self._q.clone()
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
        if self.enable_viser:
            self._update_viser_visualizer()

    def _update_states(self):
        # update arm eef state
        eef_rot_mat = quaternion_to_matrix_ig(self._eef_state[:, 3:7])
        eef_rot_mat_T = eef_rot_mat.transpose(1, 2)
        eef_rot_6d = matrix_to_rotation_6d(eef_rot_mat)

        # update object state
        object_center_pos = self._object_state[:, :3].clone()
        local_offset = torch.zeros([self.num_envs, 3], dtype=torch.float, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[:, 2] / 2
        object_rot_mat = quaternion_to_matrix_ig(self._object_state[:, 3:7])
        rotated_offset = torch.matmul(object_rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)
        object_center_pos += rotated_offset

        object_rot_6d = matrix_to_rotation_6d(object_rot_mat)

        object_rot_mat_in_eef_frame = torch.matmul(eef_rot_mat_T, object_rot_mat)
        object_to_eef_rot_6d = matrix_to_rotation_6d(object_rot_mat_in_eef_frame)

        # update target state
        target_rot_mat = quaternion_to_matrix_ig(self.reward_settings["target_quat"])
        target_rot_mat_in_eef_frame = torch.matmul(eef_rot_mat_T, target_rot_mat)
        target_to_eef_rot_6d = matrix_to_rotation_6d(target_rot_mat_in_eef_frame)

        # get world frame pos / delta pos, and convert them to eef frame
        eef_pos = self._eef_state[:, :3]
        eef_finger1_pos_relative_world = self._eef_finger1_state[:, :3] - eef_pos
        eef_finger2_pos_relative_world = self._eef_finger2_state[:, :3] - eef_pos
        eef_finger3_pos_relative_world = self._eef_finger3_state[:, :3] - eef_pos
        eef_finger4_pos_relative_world = self._eef_finger4_state[:, :3] - eef_pos
        object_to_eef_world = object_center_pos - eef_pos
        target_to_eef_world = self.reward_settings["target_pos"] - eef_pos

        eef_finger1_pos_relative = torch.matmul(eef_rot_mat_T, eef_finger1_pos_relative_world.unsqueeze(-1)).squeeze(-1)
        eef_finger2_pos_relative = torch.matmul(eef_rot_mat_T, eef_finger2_pos_relative_world.unsqueeze(-1)).squeeze(-1)
        eef_finger3_pos_relative = torch.matmul(eef_rot_mat_T, eef_finger3_pos_relative_world.unsqueeze(-1)).squeeze(-1)
        eef_finger4_pos_relative = torch.matmul(eef_rot_mat_T, eef_finger4_pos_relative_world.unsqueeze(-1)).squeeze(-1)
        object_to_eef = torch.matmul(eef_rot_mat_T, object_to_eef_world.unsqueeze(-1)).squeeze(-1)
        target_to_eef = torch.matmul(eef_rot_mat_T, target_to_eef_world.unsqueeze(-1)).squeeze(-1)

        point_matching_err = self._get_eef_point_matching_err(
            curent_eef_pos7=self._eef_state[:, :7],
            target_eef_pos7=torch.cat([self.reward_settings["target_pos"], self.reward_settings["target_quat"]], dim=-1)
        )

        if self.enable_fabric:
            # update fabric switching state
            switching_matching_err = self._get_eef_point_matching_err(
                curent_eef_pos7=self._eef_state[:, :7],
                target_eef_pos7=torch.cat([self.switching_target_pos, self.switching_target_quat], dim=-1)
            )
            self.fabric_switch_enable[switching_matching_err < self.switch_tol] = False
            self.fabric_switch_enable[self.progress_buf == 0] = True

            # update camera pose and franka base pose
            current_joint_pos_fabric = torch.zeros_like(self.fabric_q, device=self.device)
            current_joint_pos_fabric[:, :10] = self._q[:, :10].clone()
            current_joint_pos_fabric[:, 10:] = self._q[:, 26:].clone()
            glorbot_fk = self.franka_fabric.forward_kinematics(["camera_link", "lidar", "panda_link0"], current_joint_pos_fabric) # (num_envs, num_links, xyz+xyzw)

        # update point clouds
        object_pcds_world = transform_pcds_to_world(self.object_pcds, self._object_state[:, :7])
        self.combined_pcds[:, self.pcd_spec_dict["num_static_points"]: \
            self.pcd_spec_dict["num_static_points"]+self.pcd_spec_dict["num_object_points"]] = object_pcds_world

        if self.cfg["reward"]["actionreg_type"] == "delta_joint_action":
            actionreg = self.delta_joint_actions
        elif self.cfg["reward"]["actionreg_type"] == "delta_eef_action":
            actionreg = self.delta_eef_actions
        elif self.cfg["reward"]["actionreg_type"] == "delta_qd":
            actionreg = self._qd - self._qd_prev
        else:
            actionreg = torch.zeros_like(self._qd)

        # update states
        self.states.update({
            # Robot
            "base": self._q[:, :3],
            "q": self._q[:, :],
            "q_prev": self._q_prev,
            "q_hand": self._q[:, 10:26],
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
            "eef_finger1_pos_relative": eef_finger1_pos_relative,
            "eef_finger2_pos_relative": eef_finger2_pos_relative,
            "eef_finger3_pos_relative": eef_finger3_pos_relative,
            "eef_finger4_pos_relative": eef_finger4_pos_relative,

            # Object
            "object_quat": self._object_state[:, 3:7],
            "object_rot_6d": object_rot_6d,
            "object_center_pos": object_center_pos,
            "object_pos": self._object_state[:, :3],

            # Task related
            "object_to_eef": object_to_eef,
            "object_to_eef_rot_6d": object_to_eef_rot_6d,
            "target_to_eef": target_to_eef,
            "target_to_eef_rot_6d": target_to_eef_rot_6d,
            "point_matching_err": point_matching_err,

            # recorded actions
            "actionreg": actionreg,
        })

        if self.enable_fabric:
            self.states.update({
                "camera_pose7": glorbot_fk[:, 0, :],  # camera_link, xyz + xyzw
                "lidar_pose7": glorbot_fk[:, 1, :],  # lidar, xyz + xyzw
                "franka_base_pose7": glorbot_fk[:, 2, :],  # panda_link0, xyz + xyzw
            })

    def _get_eef_point_matching_err(self, curent_eef_pos7: torch.Tensor, target_eef_pos7: torch.Tensor):
        """
        Get the point matching error between current end effector position 
        and target end effector position. (based on 5 points on eef)

        Args:
            curent_eef_pos7: (B, 7) xyz + xyzw
            target_eef_pos7: (B, 7) xyz + xyzw
        """
        B = curent_eef_pos7.shape[0]

        pos_c = curent_eef_pos7[:, :3]  # (B, 3)
        quat_c = curent_eef_pos7[:, 3:] # (B, 4)
        pos_t = target_eef_pos7[:, :3]  # (B, 3)
        quat_t = target_eef_pos7[:, 3:] # (B, 4)

        local_pts = torch.tensor(
            [[0.1, 0., 0.],
            [-0.1, 0., 0.],
            [0., 0., 0.],
            [0., 0.1, 0.],
            [0., -0.1, 0.]],
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
        self.env_collision = torch.where(
            torch.norm(torch.sum(self.contact_forces[:, :58, :], dim=1), dim=1) > 1.0, 1.0, 0.0
        )  # the first 58 elements belong to base + franka + leap + arx
        self.collision = torch.where(
            torch.sum(torch.norm(self.contact_forces[:, :58, :], dim=2), dim=1) > 1.0, 1.0, 0.0
        )  # the first 58 elements belong to base + franka + leap + arx, this includes self collision

    # robot kinematics related
    def normalize_robot_joints(self, joint_angles: torch.Tensor, robot: bool, delta: bool = False) -> torch.Tensor:
        """
        Normalize joint angles to be within [-1, 1].
        Args:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        Returns:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        """
        if robot=="franka":
            assert joint_angles.shape[-1] == 7
            lower_limits, upper_limits = self.get_joint_limits_franka()
        elif robot=="leap":
            assert joint_angles.shape[-1] == 16
            lower_limits, upper_limits = self.get_joint_limits_leap()
        elif robot=="arx":
            assert joint_angles.shape[-1] == 6
            lower_limits, upper_limits = self.get_joint_limits_arx()
        else:
            raise ValueError("robot must be in ['franka', 'leap', 'arx']")

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
        if robot=="franka":
            assert joint_angles.shape[-1] == 7
            lower_limits, upper_limits = self.get_joint_limits_franka()
        elif robot=="leap":
            assert joint_angles.shape[-1] == 16
            lower_limits, upper_limits = self.get_joint_limits_leap()
        elif robot=="arx":
            assert joint_angles.shape[-1] == 6
            lower_limits, upper_limits = self.get_joint_limits_arx()
        else:
            raise ValueError("robot must be in ['franka', 'leap', 'arx']")

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
        # TODO: update this
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
            state_tensor[:, :32, 1] = joint_vel

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

        if self.enable_fabric:
            self.fabric_q[env_ids, :10] = joint_state[:, :10]
            self.fabric_q[env_ids, 10:] = joint_state[:, 26:]
            self.fabric_qd[env_ids, :] = torch.zeros_like(self.fabric_q[env_ids])
            if joint_vel is not None:
                self.fabric_qd[env_ids, :10] = joint_vel[:, :10]
                self.fabric_qd[env_ids, 10:] = joint_vel[:, 26:]
            self.fabric_qdd[env_ids, :] = torch.zeros_like(self.fabric_q[env_ids])

    def get_joint_limits_franka(self):
        """
        Get the joint limits of the ARX hand. Base (3) + Franka (7) + LEAP (4*4) + ARX (6), 32 DOF in total

        Returns:
            lower_limits (torch.Tensor): (7,)
            upper_limits (torch.Tensor): (7,)
        """
        lower_limits = self.robot_dof_lower_limits[3:10]
        upper_limits = self.robot_dof_upper_limits[3:10]
        return lower_limits, upper_limits

    def get_joint_limits_leap(self):
        """
        Get the joint limits of the ARX hand. Base (3) + Franka (7) + LEAP (4*4) + ARX (6), 32 DOF in total

        Returns:
            lower_limits (torch.Tensor): (16,)
            upper_limits (torch.Tensor): (16,)
        """
        lower_limits = self.robot_dof_lower_limits[10:26]
        upper_limits = self.robot_dof_upper_limits[10:26]
        return lower_limits, upper_limits

    def get_joint_limits_arx(self):
        """
        Get the joint limits of the ARX hand. Base (3) + Franka (7) + LEAP (4*4) + ARX (6), 32 DOF in total

        Returns:
            lower_limits (torch.Tensor): (6,)
            upper_limits (torch.Tensor): (6,)
        """
        lower_limits = self.robot_dof_lower_limits[26:]
        upper_limits = self.robot_dof_upper_limits[26:]
        return lower_limits, upper_limits

    # sim basics
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

    def _pre_physics_step_teacher(self, actions, gaze_target=None):
        """
        Args:
            actions (torch.Tensor): normalized delta joint angles (num_selected_envs, 7+4*4)
        """
        if self.eef_actions:
            self.delta_eef_actions = actions.clone()
            # Interpret position deltas in EEF-local frame and rotate to world.
            pos_actions_local = actions[:, 0:3] * self.action_scale["eef_pos"] * self.dt
            eef_rot_mat = quaternion_to_matrix_ig(self.states["eef_quat"])
            pos_actions_world = torch.matmul(eef_rot_mat, pos_actions_local.unsqueeze(-1)).squeeze(-1)
            ctrl_target_eef_pos = self.states["eef_pos"] + pos_actions_world

            # Interpret rotation deltas as target rot (axis-angle) displacements in EEF-local frame.
            rot_actions_local = actions[:, 3:6] * self.action_scale["eef_rot"] * self.dt
            angle = torch.norm(rot_actions_local, p=2, dim=-1)
            axis = rot_actions_local / angle.unsqueeze(-1).clamp_min(1.0e-8)
            rot_actions_quat_local = quat_from_angle_axis(angle, axis)

            # clamp tiny rotations to avoid numerical issues
            rot_actions_quat_local = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
                rot_actions_quat_local,
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
            )
            # Local-frame composition: q_target = q_current * q_delta_local.
            ctrl_target_eef_quat = quat_mul(self.states["eef_quat"], rot_actions_quat_local) # xyzw format

            if self.enable_fabric:
                fabric_target_eef_pos = self.switching_target_pos
                fabric_target_eef_quat = self.switching_target_quat
                fabric_eef_target = torch.cat((fabric_target_eef_pos, fabric_target_eef_quat), dim=-1)
                abs_full_joint_actions_fabric = self.compute_fabric_action(fabric_eef_target, gaze_target)

            delta_arm_joint_actions_unnormalized = torch.zeros((self.num_envs, 10), device=self.device)
            delta_arm_joint_actions_unnormalized[:, 3:10] = eef_ctrl.compute_dof_pos_delta(
                arm_dof_pos=self.states['q'][:, 3:10],
                current_eef_pos=self.states['eef_pos'],
                current_eef_quat=self.states['eef_quat'],
                jacobian=self._j_eef[:, :, 3:10],
                ctrl_target_eef_pos=ctrl_target_eef_pos,
                ctrl_target_eef_quat=ctrl_target_eef_quat,
            )

            hand_actions = actions[:, 6:] * self.action_scale["leap"] * self.dt
            delta_hand_joint_actions_unnormalized = self.unnormalize_robot_joints(hand_actions, robot="leap", delta=True)
        else:
            arm_actions = actions[:, 3:10] * self.action_scale["franka"] * self.dt
            hand_actions = actions[:, 10:26] * self.action_scale["leap"] * self.dt
            delta_arm_joint_actions_unnormalized = self.unnormalize_robot_joints(arm_actions, robot="franka", delta=True)
            delta_hand_joint_actions_unnormalized = self.unnormalize_robot_joints(hand_actions, robot="leap", delta=True)

        self.delta_joint_actions[:, :10] = delta_arm_joint_actions_unnormalized[:, :10]
        self.delta_joint_actions[:, 10:26] = delta_hand_joint_actions_unnormalized

        teacher_actions_abs = self.states['q'] + self.delta_joint_actions # need to really make sure states['q'] is always up to date
        teacher_actions_abs = tensor_clamp(
            teacher_actions_abs, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

        if self.enable_fabric:
            teacher_actions_abs[self.fabric_switch_enable, :10] = abs_full_joint_actions_fabric[self.fabric_switch_enable, :10]
            teacher_actions_abs[self.fabric_switch_enable, 10:26] = self.canonical_joint_config[self.fabric_switch_enable, 10:26]
            teacher_actions_abs[:, 26:] = abs_full_joint_actions_fabric[:, 10:]
            teacher_actions_abs[:, :2] = abs_full_joint_actions_fabric[:, :2] # always use fabric's base action regardless of the switching status # TODO: this caused the rotation issue?

        if self.distillation_mode:
            # get teacher actions for student to regress on
            delta_actions = teacher_actions_abs - self.states['q']

            base_delta_actions_worldframe = delta_actions[:, :3]
            base_delta_actions_baseframe = se2_transform(base_delta_actions_worldframe, -self.states['q'][:, 2])
            base_actions_vel_baseframe = base_delta_actions_baseframe / self.dt # numerical difference for joint velocity

            # NOTE: here we want to keep everything ranging in [-1, 1], only leap part is guaranteed, franka & arx is an empirical approximation cause their actions are from eef_converted & fabrics
            if self.delta_franka_action:
                franka_actions_normalized = self.normalize_robot_joints(delta_actions[:, 3:10], robot="franka", delta=True) / self.action_scale["franka"] / self.dt
            else:
                franka_actions_normalized = self.normalize_robot_joints(teacher_actions_abs[:, 3:10], robot="franka", delta=False)

            if self.delta_leap_action:
                leap_actions_normalized = self.normalize_robot_joints(delta_actions[:, 10:26], robot="leap", delta=True) / self.action_scale["leap"] / self.dt
            else:
                leap_actions_normalized = self.normalize_robot_joints(teacher_actions_abs[:, 10:26], robot="leap", delta=False)

            if self.delta_arx_action:
                arx_actions_normalized = self.normalize_robot_joints(delta_actions[:, 26:], robot="arx", delta=True) / self.action_scale["arx"] / self.dt
            else:
                arx_actions_normalized = self.normalize_robot_joints(teacher_actions_abs[:, 26:], robot="arx", delta=False)

            self.teacher_actions_converted[:, :3] = base_actions_vel_baseframe
            self.teacher_actions_converted[:, 3:10] = franka_actions_normalized
            self.teacher_actions_converted[:, 10:26] = leap_actions_normalized
            self.teacher_actions_converted[:, 26:] = arx_actions_normalized

        return teacher_actions_abs

    def _pre_physics_step_student(self, actions):
        """
        Args:
            actions (torch.Tensor): student actions (num_selected_envs, 3+7+4*4+6)
        """
        student_actions_abs = actions.clone()

        # base abs action
        base_pos_worldframe = self.states['q'][:, :3]
        base_action_baseframe = actions[:, :3] * self.dt
        base_action_worldframe = se2_transform(base_action_baseframe, base_pos_worldframe[:, 2])
        student_actions_abs[:, :3] = base_action_worldframe + base_pos_worldframe  # base abs action

        if self.delta_franka_action:
            student_actions_abs[:, 3:10] = self.unnormalize_robot_joints(student_actions_abs[:, 3:10], robot="franka", delta=True) * self.action_scale["franka"] * self.dt + self.states['q'][:, 3:10]
        else:
            student_actions_abs[:, 3:10] = self.unnormalize_robot_joints(student_actions_abs[:, 3:10], robot="franka", delta=False)

        if self.delta_leap_action:
            student_actions_abs[:, 10:26] = self.unnormalize_robot_joints(student_actions_abs[:, 10:26], robot="leap", delta=True) * self.action_scale["leap"] * self.dt + self.states['q'][:, 10:26]
        else:
            student_actions_abs[:, 10:26] = self.unnormalize_robot_joints(student_actions_abs[:, 10:26], robot="leap", delta=False)

        if self.delta_arx_action:
            student_actions_abs[:, 26:] = self.unnormalize_robot_joints(student_actions_abs[:, 26:], robot="arx", delta=True) * self.action_scale["arx"] * self.dt + self.states['q'][:, 26:]
        else:
            student_actions_abs[:, 26:] = self.unnormalize_robot_joints(student_actions_abs[:, 26:], robot="arx", delta=False)

        return student_actions_abs

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

        # Codex
        if env_ids.numel() > 0:
            self.object_reset_pending_mask[env_ids] = True

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_object_state = torch.zeros(num_resets, 13, device=self.device)

        # obj_reset_pos_range is always interpreted as XY range relative to box center, in box local frame.
        reset_pos = torch.zeros(num_resets, 3, device=self.device)
        relative_xy = torch.rand(num_resets, 2, device=self.device) * (
            self.obj_reset_pos_range[env_ids][:, [1, 3]] - self.obj_reset_pos_range[env_ids][:, [0, 2]]
        ) + self.obj_reset_pos_range[env_ids][:, [0, 2]]
        relative_pos_local = torch.zeros(num_resets, 3, device=self.device)
        relative_pos_local[:, :2] = relative_xy
        relative_pos_world = quat_apply(self.obj_reset_center_quat[env_ids], relative_pos_local)
        reset_pos[:, :2] = relative_pos_world[:, :2] + self.obj_reset_center_xy[env_ids]
        reset_pos[:, 2] = self.table_surface_height[env_ids]

        sampled_object_state[:, :3] = reset_pos
        sampled_object_state[:, 3:7] = self.obj_reset_center_quat[env_ids]
        self._object_state[env_ids] = sampled_object_state
        self._object_center_init_state[env_ids] = reset_pos
        self._object_center_init_state[env_ids, 2] += self.mesh_aabb_extents[env_ids, 2] / 2

        multi_env_ids_obj_int32 = self._global_indices[env_ids, self._object_id].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_obj_int32), len(multi_env_ids_obj_int32),
        )

        # update initial frame pcd
        object_pcds_world = transform_pcds_to_world(self.object_pcds, self._object_state[:, :7])
        if self.object_pcd_t0 is None:
            self.object_pcd_t0 = object_pcds_world.clone()
        self.object_pcd_t0[env_ids] = object_pcds_world[env_ids].clone()

        self.teleport_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        """
        Args:
            actions (torch.Tensor): if teacher action: normalized delta joint angles (num_selected_envs, 7+4*4)
                                    if student action: (num_selected_envs, 3+7+4*4+6)
        """
        if self.distillation_mode and self.action_history_len > 0:
            self.action_history_buf = torch.roll(self.action_history_buf, shifts=-1, dims=1)
            self.action_history_buf[:, -1, :] = actions

        if self.distillation_mode:
            self.abs_actions[:] = self._pre_physics_step_student(actions)
        else:
            self.abs_actions[:] = self._pre_physics_step_teacher(actions)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.abs_actions))

        # Add F/T wrench to object
        if self.object_wrench_args["enable"]:
            self._apply_object_wrench()

    def post_physics_step(self):
        # Codex
        # Promote pending reset events one sim step later, matching when set_*_tensor_indexed changes are observed.
        if torch.any(self.object_reset_pending_mask):
            self.object_reset_mask |= self.object_reset_pending_mask
            self.object_reset_pending_mask[:] = False

        self.progress_buf += 1
        self.teleport_buf += 1
        self.teleport_buf = self.teleport_buf % self.object_teleport_args['n2']

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        # video logging
        if self.video_logging["capture"]:
            self.video_logger()
        self.sim_steps += 1

    def step(self, actions: torch.Tensor):
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions[~self.fabric_switch_enable][:, :26] = self.dr_randomizations['actions']['noise_lambda'](actions)[~self.fabric_switch_enable][:, :26]

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def reset_idx(self, env_ids=None):
        # Domain randomization, can happen only at reset time since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._reset_object_state(env_ids) # reset object state

        if env_ids.numel() == 0:
            return

        reset_noise = torch.rand((len(env_ids), 32), device=self.device) # [0, 1]
        reset_noise = 2.0 * (reset_noise - 0.5) # [-1, 1]
        reset_noise[:, :3] *= 0 # no base reset noise for now
        reset_noise[:, 3:10] *= self.reset_noise_scale["franka"]

        if self.reset_noise_scale["leap"] is None:
            reset_noise[:, 10:26] = self.unnormalize_robot_joints(reset_noise[:, 10:26], robot="leap", delta=False)
            reset_noise[:, 10:26] -= self.default_reset_joint_config[env_ids, 10:26]
        else:
            reset_noise[:, 10:26] *= self.reset_noise_scale["leap"]

        reset_noise[:, 26:32] *= self.reset_noise_scale["arx"]

        reset_joint_config = tensor_clamp(
            self.default_reset_joint_config[env_ids] + reset_noise,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        self.set_robot_joint_state(reset_joint_config, env_ids=env_ids)

        if self.enable_fabric:
            self.fabric_q[env_ids, :10] = reset_joint_config[:, :10]
            self.fabric_q[env_ids, 10:] = reset_joint_config[:, 26:]
            self.fabric_qd[env_ids, :] = torch.zeros_like(self.fabric_q[env_ids])
            self.fabric_qdd[env_ids, :] = torch.zeros_like(self.fabric_q[env_ids])

        self.success_flags[env_ids] = 0
        self.lifting_flags[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        if self.action_history_len > 0:
            self.action_history_buf[env_ids] = 0.0

        if self.object_wrench_args["enable"]:
            self.object_applied_forces[env_ids] = 0.0
            self.object_applied_torques[env_ids] = 0.0
            self.rigid_body_forces[env_ids] = 0
            self.rigid_body_torques[env_ids] = 0

    # visualization
    def set_viewer(self, pos=[1.5, 0.0, 0.7], target=[0.5, 0.0, 0.1]):
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
            self.video_ims = []

        if render_step < self.max_episode_length * 2:
            camera_renders = self.get_camera_render()
            ims = np.array(camera_renders)[:, 0, :, :, :3]

            for env_idx in range(ims.shape[0]):
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
                cv2.putText(img, f'Env: {env_idx}  Step: {render_step}', (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                ims[env_idx] = img
            self.video_ims.append(ims)

        if render_step == 2*self.max_episode_length - 1:
            render_step_start = self.sim_steps + 1 - self.max_episode_length
            filename = os.path.join(self.video_dir, f"viz_step{render_step_start}.mp4")
            frames = np.asarray(self.video_ims) # (num_frames, num_envs, height, width, channels)
            frames = frames.transpose(1, 0, 2, 3, 4) # (num_envs, num_frames, height, width, channels)
            frames = frames.reshape(-1, frames.shape[2], frames.shape[3], frames.shape[4])  # (num_envs * num_frames, height, width, channels)
            with imageio.get_writer(filename, fps=60) as writer:
                for frame in frames:
                    writer.append_data(frame)

            if wandb.run is not None:
                # note this wandb log has to coordinate with the distillation logs, commit=True should be set there
                if self.distillation_mode:
                    wandb.log({"visualization/video": wandb.Video(os.path.join(self.video_dir, f"viz_step{render_step_start}.mp4"))}, step=self.distillation_steps)
                else:
                    wandb.log({"visualization/video": wandb.Video(os.path.join(self.video_dir, f"viz_step{render_step_start}.mp4"))}, commit=True)

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

    def vis_pcd(self):
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

    def _init_viser_visualizer(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot"))
        robot_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka")

        full_robot_asset_path = os.path.join(asset_root, robot_asset_file)
        self.viser_visualizer = ViserVisualizer(
            urdf_path=self.full_robot_asset_path,
            num_envs=self.num_envs,
        )

    def _update_viser_visualizer(self):
        # only render the selected environment
        env_id = self.viser_visualizer.env_id
        self.viser_visualizer.set_joint_positions(
            self.states['q'][env_id].cpu().numpy(),
        )

        # pcds are now getting updated in distillation part
        # # (1, N, 3)
        # pcd_full_scene = self.combined_pcds[env_id:env_id+1] 
        # # (1, M, 3)
        # robot_pcd_t = self.robot_pcd_sampler.sample(self.states['q'][env_id:env_id+1], self.torchurdf_to_isaac_idx)
        # pcd_full = torch.cat([pcd_full_scene, robot_pcd_t], dim=1)

        # # (1, 7)
        # current_camera_pose = self.states["camera_pose7"][env_id:env_id+1]
        # sim_depth_pcd, logs = simulate_depth_cam_render_from_pose(
        #     pcd=pcd_full,
        #     camera_pose=current_camera_pose,
        #     num_points=4096,
        # )

        # lidar_link_pose = self.states["lidar_pose7"][env_id:env_id+1]
        # sim_lidar_pcd, logs = simulate_lidar_render_from_pose(
        #     pcd=pcd_full,
        #     lidar_pose=lidar_link_pose,
        #     num_points=5000,
        #     num_azimuth=512,
        #     num_polar=128,
        #     suppress_bins=2,
        #     jitter_std_m=0.001,
        # )

        # rendered_full_pcd = torch.cat([sim_depth_pcd, sim_lidar_pcd], dim=1)

        # self.viser_visualizer.update_point_cloud(
        #     point_cloud_type="full_points", 
        #     point_cloud=pcd_full[0].cpu().numpy()
        # )
        # self.viser_visualizer.update_point_cloud(
        #     point_cloud_type="rendered_full_points",
        #     point_cloud=rendered_full_pcd[0].cpu().numpy()
        # )
        # self.viser_visualizer.update_point_cloud(
        #     point_cloud_type="rendered_cam_points",
        #     point_cloud=sim_depth_pcd[0].cpu().numpy()
        # )
        # self.viser_visualizer.update_point_cloud(
        #     point_cloud_type="rendered_lidar_points",
        #     point_cloud=sim_lidar_pcd[0].cpu().numpy()
        # )
        # self.viser_visualizer.update_point_cloud(
        #     point_cloud_type="obj_point_t",
        #     point_cloud=self.states['object_pos'][env_id].reshape(1, 3).cpu().numpy()
        # )
        # if self.object_pcd_t0 is not None:
        #     self.viser_visualizer.update_point_cloud(
        #         point_cloud_type="seg_static_object_t0",
        #         point_cloud=self.object_pcd_t0[env_id].cpu().numpy()
        #     )
        # if self.distractor_settings["enable"]:
        #     self.viser_visualizer.update_point_cloud(
        #         point_cloud_type="seg_distractor_t0",
        #         point_cloud=self.distractor_pcds[env_id].cpu().numpy()
        #     )

    @abstractmethod
    def _create_envs(self, spacing, num_per_row):
        self.table_pos = ...
        self.table_size = ...
        self.table_surface_height = ...
        self.mesh_aabb_extents = ...
        self.obj_reset_pos_range = ...
        self.obj_reset_center_xy = ...
        self.obj_reset_center_quat = ...
        self.envs = ...

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
