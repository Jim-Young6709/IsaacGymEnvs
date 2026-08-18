#!/usr/bin/env python3

"""FrankaLeapPickCubePixels task."""

import numpy as np
import os
import torch
import imageio
import random
import wandb

import torchvision
from torchvision.transforms import Resize, Compose, Normalize, GaussianBlur, Lambda

from collections import defaultdict
from typing import Tuple
from torch import Tensor

from evi.utils.torch_jit_utils import *
from evi.utils.rotation_conversions import quaternion_to_matrix_ig, matrix_to_quaternion_ig, matrix_to_rotation_6d
from evi.tasks.base.vec_task import VecTask

from isaacgym import gymtorch
from isaacgym import gymapi

# import matplotlib.pyplot as plt


ENCODER_OUTPUT_SIZES = {
    "dinov2-vits": 384,
    "dinov2-vitb": 768,
    "dinov2-vitl": 1024,
    "dinov2-vitg": 1536,
}

class FrankaLeapPickCubePixels2(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.action_scale = self.cfg["control"]["action_scale"]
        self.object_init_pos_noise = self.cfg["env"]["object_init_pos_noise"]
        self.object_init_rot_noise = self.cfg["env"]["object_init_rot_noise"]
        self.robot_init_pos_noise = self.cfg["env"]["robot_init_pos_noise"]
        self.robot_init_rot_noise = self.cfg["env"]["robot_init_rot_noise"]
        self.franka_init_dof_noise = self.cfg["env"]["franka_init_dof_noise"]
        self.hand_init_dof_noise = self.cfg["env"]["hand_init_dof_noise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.enable_debug_vis = self.cfg["env"]["debug_vis"]

        self.decimation = self.cfg["control"]["decimation"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        self.cfg["env"]["controlFrequencyInv"] = self.cfg["control"]["decimation"]
        self.max_episode_length_s = self.cfg["env"]["episode_length_s"]
        self.max_episode_length = int(np.ceil(self.max_episode_length_s / self.dt))
        
        # Controller type
        self.control_type = self.cfg["control"]["control_type"]
        assert self.control_type in {"joint_pos_relative_prev"}, \
            "Invalid control type specified. Must be one of: {osc, joint_tor, joint_pos_relative_prev}"

        # Domain randomization
        self.randomize = self.cfg["task"]["randomize"]
        self.randomize_lights = self.randomize and self.cfg["task"]["randomize_lights"]
        self.randomize_camera = self.randomize and self.cfg["task"]["randomize_camera"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.misc_buf = 0

        # Random impulse
        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        # Sensor camera settings
        self.pixel_vision = self.cfg["obs_camera"]["pixel_vision"]
        self.enable_encoder = self.cfg["obs_camera"]["enable_encoder"]
        if self.pixel_vision:
            self._setup_obs_camera()

        # Sensor camera visualization settings
        self.visualize_camera_obs = self.cfg["obs_camera"].get("visualize_camera_obs", False)
        self.viz_mode = self.cfg["obs_camera"].get("viz_mode", "save")  # 'save' or 'live'
        self.viz_save_frequency = self.cfg["obs_camera"].get("viz_save_frequency", 10)
        self.viz_save_dir = self.cfg["obs_camera"].get("viz_save_dir", "logs/camera_viz")
        if self.visualize_camera_obs and self.pixel_vision:
            if self.viz_mode == 'save':
                import os
                os.makedirs(self.viz_save_dir, exist_ok=True)
                print(f"Camera visualization enabled - saving to {self.viz_save_dir}")
                self.viz_step_count = 0
            elif self.viz_mode == 'live':
                self._setup_live_camera_viewer()
                print("Camera visualization enabled - live matplotlib view")
            else:
                print(f"Warning: Unknown viz_mode '{self.viz_mode}', defaulting to 'save'")
                self.viz_mode = 'save'

        # Visual encoder settings
        if self.enable_encoder:
            self.encoder_type_full = self.cfg["obs_camera"]["encoder_model"] + "-" + self.cfg["obs_camera"]["encoder_size"]
            self.encoded_obs_size_per_cam = ENCODER_OUTPUT_SIZES[self.encoder_type_full]
            self.encoded_obs_size = self.encoded_obs_size_per_cam * self.num_obs_cameras
        else:
            self.encoded_obs_size = 0
        self.cfg["env"]["numObservations"] = 117 + self.encoded_obs_size # Real-world obs + vision obs: 96 + self.encoded_obs_size, Full obs + vision obs: 117 + self.encoded_obs_size
        self.cfg["env"]["numActions"] = 23
        print("observation size:", self.cfg["env"]["numObservations"])

        self._init_buffers_pre()
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        self._init_buffers_post()

        self._setup_rewards()
        if self.pixel_vision:
            self._setup_visual_encoder()

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()

    def _init_buffers_pre(self):
        # Values to be filled in at runtime
        self.states = {}                      # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                     # will be dict mapping names to relevant sim handles
        self.num_dofs = None                  # Total number of DOFs per env
        self.actions = None                   # Current actions to be deployed
        self._init_object_state = None        # Initial state of object for the current env
        self._object_state = None             # Current state of object for the current env
        self._object_id = None                # Actor ID corresponding to object for a given env

        # Tensor placeholders
        self._root_state = None               # State of root body                       (n_envs, 13)
        self._dof_state = None                # State of all joints                      (n_envs, n_dof)
        self._q = None                        # Joint positions                          (n_envs, n_dof)
        self._qd = None                       # Joint velocities                         (n_envs, n_dof)
        self._rigid_body_state = None         # State of all rigid bodies                (n_envs, n_bodies, 13)
        self._contact_forces = None           # Contact forces in sim
        self._eef_state = None                # End effector state (at grasping point)
        self._eef_finger1_state = None        # End effector state (at finger 1)
        self._eef_finger2_state = None        # End effector state (at finger 2)
        self._eef_finger3_state = None        # End effector state (at finger 3)
        self._eef_finger4_state = None        # End effector state (at finger 4)
        self._j_eef = None                    # Jacobian for end effector
        self._mm = None                       # Mass matrix
        self._franka_effort_limits = None     # Actuator effort limits for franka
        self._hand_effort_limits = None       # Actuator effort limits for hand
        self._global_indices = None           # Unique indices corresponding to all envs in flattened array

        # Misc
        self.fingertips = self.cfg["env"]["fingertips"]

    def _init_buffers_post(self):
        # Default init dof pos
        self.franka_default_dof_pos = to_torch(
            self.cfg["init"]["franka"]["dof_pos"], device=self.device
        )
        self.hand_default_dof_pos = to_torch(
            self.cfg["init"]["hand"]["dof_pos"], device=self.device
        )

        # Success counts
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Random impulse
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # Misc
        self.object_picked = torch.zeros_like(self.successes)
        self.curr_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.prev_prev_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
    
        self.franka_dof_speed_scale = self.cfg["control"]["franka_dof_speed_scale"]
        self.hand_dof_speed_scale = self.cfg["control"]["hand_dof_speed_scale"]
        self.dof_speed_scale = torch.tensor(
            [self.franka_dof_speed_scale] * self.num_arm_dofs + [self.hand_dof_speed_scale] * self.num_hand_dofs,
            device=self.device
        )

        self.grasp_finger_dof_pos = self.robot_dof_upper_limits[self.num_arm_dofs:] - self.robot_dof_lower_limits[self.num_arm_dofs:]
        self.grasp_finger_dof_pos *= 0.5
        self.grasp_finger_dof_pos[0] = 0.0
        self.grasp_finger_dof_pos[4] = 0.0
        self.grasp_finger_dof_pos[8] = 0.0
        self.grasp_finger_dof_pos[13] = 1.6 #TODO: not sure if this is right

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # Domain randomization, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # # DEBUG: cube masses
        # masses = []
        # scales = []
        # ref = np.linspace(0.95, 1.05, 100)
        # for i in range(self.num_envs):
        #     scales.append(ref[i % len(ref)])
        #     mass = sum((self.gym.get_actor_rigid_body_properties(self.envs[i], self.objects[i])[l].mass for l in range(1)))
        #     print(mass)
        #     masses.append(mass)
        # print('MIN', np.min(masses))
        # print('MAX', np.max(masses))
        # print('MEAN', np.mean(masses))
        # print('STD', np.std(masses))
        # exit()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # Create robot asset
        asset_root = "./"
        robot_asset_file = "./"
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
        robot_asset_file = self.cfg["env"]["asset"].get("assetFileName", robot_asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, asset_options)

        self.num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        self.num_robot_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_arm_dofs = 7
        self.num_hand_dofs = self.num_robot_dofs - self.num_arm_dofs
        print("Num dofs: ", self.num_robot_dofs)
        
        self.num_robot_actuators = self.num_robot_dofs
        self.actuated_dof_indices = [i for i in range(self.num_robot_dofs)]

        robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)

        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        # self.robot_dof_default_pos = [] # TODO: set this up?
        # self.robot_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        # create fingertip force sensors
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(robot_asset, name) for name in self.fingertips]
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(robot_asset, ft_handle, sensor_pose)

        self.franka_dof_stiffness = to_torch(self.cfg["control"]["franka_dof_stiffness"], dtype=torch.float, device=self.device)
        self.franka_dof_damping = to_torch(self.cfg["control"]["franka_dof_damping"], dtype=torch.float, device=self.device)
        self.franka_effort_limits = [87, 87, 87, 87, 12, 12, 12]
        self.franka_velocity_limits = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]

        self.hand_dof_stiffness = self.cfg["control"]["hand_p_gain"]
        self.hand_dof_damping = self.cfg["control"]["hand_d_gain"]
        # self.hand_effort_limits = [0.95, 0.95, 0.95, 0.95,
        #                           0.95, 0.95, 0.95, 0.95,
        #                           0.95, 0.95, 0.95, 0.95,
        #                           0.95, 0.95, 0.95, 0.95]
        # TODO: try higher effort limits for fingers
        self.hand_effort_limits = [5.0, 5.0, 5.0, 5.0,
                                  5.0, 5.0, 5.0, 5.0,
                                  5.0, 5.0, 5.0, 5.0,
                                  5.0, 5.0, 5.0, 5.0]
        self.hand_velocity_limits = [8.48, 8.48, 8.48, 8.48,
                                    8.48, 8.48, 8.48, 8.48,
                                    8.48, 8.48, 8.48, 8.48,
                                    8.48, 8.48, 8.48, 8.48]

        # Arm dof props
        for i in range(self.num_arm_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])

            robot_dof_props['effort'][i] = self.franka_effort_limits[i]
            robot_dof_props['stiffness'][i] = self.franka_dof_stiffness[i]
            robot_dof_props['damping'][i] = self.franka_dof_damping[i]
            robot_dof_props['friction'][i] = 0.01
            robot_dof_props['armature'][i] = 0.002

        # Hand dof props
        for i in range(self.num_arm_dofs, self.num_robot_dofs):
            self.robot_dof_lower_limits.append(robot_dof_props['lower'][i])
            self.robot_dof_upper_limits.append(robot_dof_props['upper'][i])

            idx = i - self.num_arm_dofs
            robot_dof_props['effort'][i] = self.hand_effort_limits[idx]
            robot_dof_props['stiffness'][i] = self.hand_dof_stiffness
            robot_dof_props['damping'][i] = self.hand_dof_damping
            robot_dof_props['friction'][i] = 0.01
            robot_dof_props['armature'][i] = 0.002

        self.franka_effort_limits = to_torch(self.franka_effort_limits, device=self.device)
        self.hand_effort_limits = to_torch(self.hand_effort_limits, device=self.device)
        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits, device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits, device=self.device)
        # Arm
        self.franka_dof_lower_limits = self.robot_dof_lower_limits[:self.num_arm_dofs]
        self.franka_dof_upper_limits = self.robot_dof_upper_limits[:self.num_arm_dofs]
        # Hand
        self.hand_dof_lower_limits = self.robot_dof_lower_limits[self.num_arm_dofs:]
        self.hand_dof_upper_limits = self.robot_dof_upper_limits[self.num_arm_dofs:]

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.6
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create object asset
        self.object_size = self.cfg["env"]["object"]["object_size"] #0.08 #0.045
        object_opts = gymapi.AssetOptions()
        object_opts.density = self.cfg["env"]["object"]["density"]  #100.0 # default is 1000
        object_asset = self.gym.create_box(self.sim, *([self.object_size] * 3), object_opts)
        object_color = gymapi.Vec3(0.0, 0.0, 1.0)

        # Define start pose for the robot
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2)
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Define start pose for table
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = to_torch(np.array(table_pos) + np.array([0, 0, table_thickness / 2]), device=self.device)
        # self.reward_settings["table_height"] = self._table_surface_pos[2]

        # Define start pose for objects (doesn't really matter since they're get overridden during reset() anyways)
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.object_z_init = object_start_pose.p.z

        # compute aggregate size
        num_robot_bodies = self.num_robot_bodies
        num_robot_shapes = self.num_robot_shapes
        extra_assets_per_env = 2 # 1 for table, object
        max_agg_bodies = num_robot_bodies + extra_assets_per_env
        max_agg_shapes = num_robot_shapes + extra_assets_per_env

        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        # https://github.com/isaac-sim/IsaacGymEnvs/blob/aeed298638a1f7b5421b38f5f3cc2d1079b6d9c3/isaacgymenvs/tasks/allegro_kuka/allegro_kuka_base.py#L655
        self.object_rb_handles = list(range(self.num_robot_bodies, self.num_robot_bodies + num_table_bodies + num_object_bodies))

        self.robots = []
        self.objects = []
        self.envs = []
        self.vis_cam_handles = []
        if self.pixel_vision:
            self.cams = []
            self.cam_tensors = []

            if self.num_obs_cameras > 1:
                self.cams = [[] for _ in range(self.num_obs_cameras)]
                self.cam_tensors = [[] for _ in range(self.num_obs_cameras)]

        # Create environments
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: robot should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create robot
            # Potentially randomize start pose
            if self.robot_init_pos_noise > 0:
                rand_xy = self.robot_init_pos_noise * (-1. + np.random.rand(2) * 2.0)
                robot_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2)
            if self.robot_init_rot_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.robot_init_rot_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                robot_start_pose.r = gymapi.Quat(*new_quat)
            robot_actor = self.gym.create_actor(env_ptr, robot_asset, robot_start_pose, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create object
            self._object_id = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 2, 0)
            # Set colors
            self.gym.set_rigid_body_color(env_ptr, self._object_id, 0, gymapi.MESH_VISUAL, object_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.objects.append(self._object_id)
            self.robots.append(robot_actor)

            # Add camera for visualization
            self.attach_camera(i, env_ptr, self.robots[i]) 

            # Add camera for observation
            if self.pixel_vision:
                for cam_idx, cam_name in enumerate(self.enabled_cams):
                    cam_config = self.cam_configs[cam_name]

                    cam_props = gymapi.CameraProperties()
                    cam_props.width = cam_config["w"]
                    cam_props.height = cam_config["h"]
                    cam_props.horizontal_fov = cam_config["fov"] + np.random.randn() * self.cam_randomization_params[cam_name]["fov_noise"]
                    cam_props.supersampling_horizontal = cam_config["ss"]
                    cam_props.supersampling_vertical = cam_config["ss"]
                    cam_props.near_plane = cam_config["near_plane"]
                    cam_props.far_plane = cam_config["far_plane"]
                    cam_props.enable_tensors = True
                    
                    cam_handle = self.gym.create_camera_sensor(env_ptr, cam_props)

                    if cam_config["attachment"] == "floating":
                        local_transform = gymapi.Transform()
                        local_transform.p = gymapi.Vec3(*cam_config["loc_p"])
                        xyz_angle_rad = [np.radians(a) for a in cam_config["loc_r"]]
                        local_transform.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
                        
                        root_handle = self.gym.get_actor_root_rigid_body_handle(env_ptr, table_actor)
                        self.gym.attach_camera_to_body(
                            cam_handle, env_ptr, root_handle,
                            local_transform, gymapi.FOLLOW_POSITION
                        )
                    else:
                        rigid_body_handle = self.gym.find_actor_rigid_body_handle(
                            env_ptr, robot_actor, cam_config["attachment"])

                        local_t = gymapi.Transform()
                        local_t.p = gymapi.Vec3(*cam_config["loc_p"])
                        xyz_angle_rad = [np.radians(a) for a in cam_config["loc_r"]]
                        local_t.r = gymapi.Quat.from_euler_zyx(*xyz_angle_rad)
                        self.gym.attach_camera_to_body(
                            cam_handle, env_ptr, rigid_body_handle,
                            local_t, gymapi.FOLLOW_TRANSFORM
                        )        

                    if self.num_obs_cameras > 1:
                        self.cams[cam_idx].append(cam_handle)
                        image_type = gymapi.IMAGE_COLOR if cam_config["type"] == "rgb" else gymapi.IMAGE_DEPTH
                        cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, image_type)
                        cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
                        self.cam_tensors[cam_idx].append(cam_tensor_th)
                    else:
                        self.cams.append(cam_handle)
                        image_type = gymapi.IMAGE_COLOR if cam_config["type"] == "rgb" else gymapi.IMAGE_DEPTH
                        cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, cam_handle, image_type)
                        cam_tensor_th = gymtorch.wrap_tensor(cam_tensor)
                        self.cam_tensors.append(cam_tensor_th)
        object_rb_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self._object_id)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        # Setup init state buffer
        self._init_object_state = torch.zeros(self.num_envs, 13, device=self.device)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        robot_handle = 0
        self.handles = {
            # Hand
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "palm_center"),
            "finger1_tip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "realtip_1"),
            "finger2_tip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "realtip_2"),
            "finger3_tip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "realtip_3"),
            "finger4_tip": self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, "realtip_4"),
            # object
            "object_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._object_id, "box"),
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
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
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, robot_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :, :]
        self._object_state = self._root_state[:, self._object_id, :]

        # Initialize states
        self.states.update({
            "object_size": torch.ones_like(self._eef_state[:, 0]) * self.object_size,
            "target_height": self._table_surface_pos[2] + 0.4 
        })

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        eef_rot_6d = matrix_to_rotation_6d(quaternion_to_matrix_ig(self._eef_state[:, 3:7]))
        object_rot_6d = matrix_to_rotation_6d(quaternion_to_matrix_ig(self._object_state[:, 3:7]))
        ctrl_diff = self.curr_targets[:, :] - self._q[:, :]
        hand_base_pos = self._eef_state[:, :3]

        # Pixel obs
        if self.pixel_vision:
            self.compute_pixel_obs()
            if self.enable_encoder:
                self.compute_encoded_obs()

        self.states.update({
            # Robot
            "q": self._q[:, :],
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
            "eef_finger1_pos_relative": self._eef_finger1_state[:, :3] - hand_base_pos,
            "eef_finger2_pos_relative": self._eef_finger2_state[:, :3] - hand_base_pos,
            "eef_finger3_pos_relative": self._eef_finger3_state[:, :3] - hand_base_pos,
            "eef_finger4_pos_relative": self._eef_finger4_state[:, :3] - hand_base_pos,

            # Object
            "object_quat": self._object_state[:, 3:7],
            "object_rot_6d": object_rot_6d,
            "object_pos": self._object_state[:, :3],

            # Other
            # "actions": self.actions[:, :],
            "ctrl_diff": ctrl_diff,
            "hand_to_object": self._object_state[:, :3] - self._eef_state[:, :3],
            "object_to_target": self.reward_settings["target_pos"] - self._object_state[:, :3],
            "object_to_target_height": self.reward_settings["target_height"] - self._object_state[:, 3].unsqueeze(-1),
            "object_target_6d_diff": self.reward_settings["target_rot_6d"] - object_rot_6d,

            # Pixel obs
            "encoded_obs": self.encoded_obs if self.enable_encoder else None,
        })

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self):
        """
        Reward: hand-object proximity, object-goal position, lifting, and finger regularizaion.
        """
        # Get rewards and components from the jit function

        if self.cfg["env"]["reward_mode"] == 1:
            total_reward, resets, \
            r_hand_obj, r_obj_goal, r_lift, r_curl, r_vel_penalty, \
            r_hand_obj_scaled, r_obj_goal_scaled, r_lift_scaled, r_curl_scaled, r_vel_penalty_scaled, \
            object_picked = compute_franka_leap_reward1(
                self.reset_buf, self.progress_buf, self.actions, self.states, 
                self.reward_settings, self.reward_config, self.max_episode_length,
                self.num_arm_dofs
            )
        else:
            total_reward, resets, \
            r_hand_obj, r_obj_goal, r_lift, r_curl, r_vel_penalty, \
            r_hand_obj_scaled, r_obj_goal_scaled, r_lift_scaled, r_curl_scaled, r_vel_penalty_scaled, \
            object_picked = compute_franka_leap_reward2(
                self.reset_buf, self.progress_buf, self.actions, self.states, 
                self.reward_settings, self.reward_config, self.max_episode_length,
                self.num_arm_dofs
            )

        # Update buffers
        self.rew_buf[:] = total_reward
        self.object_picked[:] = object_picked
        
        # Logging
        self.extras.update({
            "reward_total": total_reward.mean().item(),

            # Raw
            # "reward_r_hand_obj": r_hand_obj.mean().item(),
            # "rewards_r_obj_goal": r_obj_goal.mean().item(),
            # "rewards_r_lift": r_lift.mean().item(),
            # "rewards_r_curl": r_curl.mean().item(),
            # "rewards_r_vel_penalty": r_vel_penalty.mean().item(),

            # Scaled
            "rewards_r_hand_obj_scaled": r_hand_obj_scaled.mean().item(),
            "rewards_r_obj_goal_scaled": r_obj_goal_scaled.mean().item(),
            "rewards_r_lift_scaled": r_lift_scaled.mean().item(),
            "rewards_r_curl_scaled": r_curl_scaled.mean().item(),
            "rewards_r_vel_penalty_scaled": r_vel_penalty_scaled.mean().item(),

            # Successes
            "successes": object_picked.mean()
        })

    def compute_observations(self):
        self._refresh()

        obs_components = ["q", "qd", "eef_pos", "eef_rot_6d", "eef_vel",
                          "eef_finger1_pos", "eef_finger2_pos", "eef_finger3_pos", "eef_finger4_pos",
                          "object_pos", "object_rot_6d", "hand_to_object", 
                          "object_to_target", "object_target_6d_diff", "ctrl_diff"]

        # Pixel obs
        if self.pixel_vision:
            self.compute_pixel_obs()
            if self.enable_encoder:
                self.compute_encoded_obs()
                obs_components.append("encoded_obs")

        self.obs_buf = torch.cat([self.states[ob] for ob in obs_components], dim=-1)

        # maxs = {ob: torch.max(self.states[ob]).item() for ob in obs}

        return self.obs_buf
    
    # For use in DAgger
    def compute_observations_student(self):
        # obs_components = ["q", "qd", "eef_pos", "eef_rot_6d", "eef_vel",
        #                   "eef_finger1_pos", "eef_finger2_pos", "eef_finger3_pos", "eef_finger4_pos",
        #                   "ctrl_diff"]

        obs_components = ["q", "eef_pos", "eef_rot_6d",
                          "eef_finger1_pos_relative", "eef_finger1_pos_relative", "eef_finger1_pos_relative", "eef_finger1_pos_relative",
                          "ctrl_diff"]

        if self.pixel_vision and self.enable_encoder:
            obs_components.append("encoded_obs")

        self.obs_buf = torch.cat([self.states[ob] for ob in obs_components], dim=-1)
        return self.obs_buf

    # For use in DAgger
    def compute_observations_teacher(self):
        obs_components = ["q", "qd", "eef_pos", "eef_rot_6d", "eef_vel",
                          "eef_finger1_pos", "eef_finger2_pos", "eef_finger3_pos", "eef_finger4_pos",
                          "object_pos", "object_rot_6d", "hand_to_object", 
                          "object_to_target", "object_target_6d_diff", "ctrl_diff"]

        self.obs_buf = torch.cat([self.states[ob] for ob in obs_components], dim=-1)
        return self.obs_buf

    # For use in DAgger
    def _compute_observations(self, obs_type="student"):
        if obs_type == "teacher":
            self._refresh()
            obs = self.compute_observations_teacher()
        else:
            obs = self.compute_observations_student()
        return obs

    def check_termination(self):
        # Object below table height
        object_table_delta = self.states["object_pos"][:, 2] - self.reward_settings["table_height"]
        object_below = object_table_delta <= -0.1
        self.reset_buf[:] = torch.where(object_below, torch.ones_like(self.reset_buf), self.reset_buf)

        # # Terminate on contact
        # terminate_contact = torch.any(torch.norm(self.contact_forces[:, self.terminate_contact_inds, :], dim=-1) > 1.0, dim=1)
        # self.reset_buf[:] = torch.where(terminate_contact, torch.ones_like(self.reset_buf), self.reset_buf)

        # Max episode length exceeded
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

    def compute_pixel_obs(self):
        if self.cfg["env"]["test_mode"] == 1:

            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            
            for i in range(self.num_envs):
                for cam_idx, cam_name in enumerate(self.enabled_cams):
                    cam_config = self.cam_configs[cam_name]

                    if self.num_obs_cameras > 1:
                        cam_tensor = self.cam_tensors[cam_idx][i]
                    else:
                        cam_tensor = self.cam_tensors[i]
                    
                    # Crop to square if needed
                    k = min(cam_config["w"], cam_config["h"])
                    crop_l = (cam_config["w"] - k) // 2
                    crop_r = crop_l + k
                    crop_u = (cam_config["h"] - k) // 2
                    crop_d = crop_u + k

                    if self.randomize_camera:
                        crop_shift = self.cam_randomization_params[cam_name]["crop_shift"]
                        shift = random.randint(-crop_shift, crop_shift)
                        if k == cam_config["w"]:
                            crop_u += shift
                            crop_d += shift
                            assert crop_u >= 0 and crop_d <= cam_config["h"]
                        else:
                            crop_l += shift
                            crop_r += shift
                            assert crop_l >= 0 and crop_r <= cam_config["w"]

                    # Process the image
                    cropped_image = cam_tensor[crop_u:crop_d, crop_l:crop_r, :3].permute(2, 0, 1).float() / 255.
                    processed_image = self.im_transform(cropped_image)
                    
                    if self.num_obs_cameras > 1:
                        self.pixel_obs_buf[i][cam_idx] = processed_image
                    else:
                        self.pixel_obs_buf[i] = processed_image

            # # DEBUG: save camera obs for env 0
            # os.makedirs("logs/pixel_obs", exist_ok=True)
            # img_to_save = self.pixel_obs_buf[0].cpu()
            # torchvision.utils.save_image(
            #     img_to_save, 
            #     f"logs/pixel_obs/env0_step_{self.progress_buf[0].item()}.png", 
            #     normalize=True
            # )
            # print("saved non vectorized")

            self.gym.end_access_image_tensors(self.sim)
        else:
            self.compute_pixel_obs_vectorized()

            # # DEBUG: save camera obs for env 0
            # os.makedirs("logs/pixel_obs", exist_ok=True)
            # img_to_save = self.pixel_obs_buf[0].cpu()
            # torchvision.utils.save_image(
            #     img_to_save, 
            #     f"logs/pixel_obs/env0_step_{self.progress_buf[0].item()}.png", 
            #     normalize=True
            # )
            # print("saved vectorized")

        # Camera visualization for rollout
        if self.visualize_camera_obs:
            if self.viz_mode == 'live':
                self._update_live_camera_viewer()
            elif self.viz_step_count % self.viz_save_frequency == 0:
                self._save_camera_visualization()
            
            if self.viz_mode == 'save':
                self.viz_step_count += 1

    def compute_pixel_obs_vectorized(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for cam_idx, cam_name in enumerate(self.enabled_cams):
            cam_config = self.cam_configs[cam_name]
            
            # Get camera tensors for all environments
            if self.num_obs_cameras > 1:
                cam_tensors = torch.stack(self.cam_tensors[cam_idx], dim=0)  # (N, H, W, 3)
            else:
                cam_tensors = torch.stack(self.cam_tensors, dim=0)  # (N, H, W, 3)
            
            # Pre-compute crop parameters
            k = min(cam_config["w"], cam_config["h"])
            crop_l_base = (cam_config["w"] - k) // 2
            crop_r_base = crop_l_base + k
            crop_u_base = (cam_config["h"] - k) // 2
            crop_d_base = crop_u_base + k
            
            # Handle randomization
            if self.randomize_camera:
                crop_shift = self.cam_randomization_params[cam_name]["crop_shift"]
                # Generate random shifts for all environments (matching unvectorized logic)
                shifts = torch.randint(-crop_shift, crop_shift + 1, (self.num_envs,), device=self.device)
                
                if k == cam_config["w"]:  # Square is limited by width, so shift vertically
                    crop_u = torch.clamp(crop_u_base + shifts, 0, cam_config["h"] - k)
                    crop_d = crop_u + k
                    crop_l = torch.full((self.num_envs,), crop_l_base, device=self.device)
                    crop_r = torch.full((self.num_envs,), crop_r_base, device=self.device)
                else:  # Square is limited by height, so shift horizontally
                    crop_l = torch.clamp(crop_l_base + shifts, 0, cam_config["w"] - k)
                    crop_r = crop_l + k
                    crop_u = torch.full((self.num_envs,), crop_u_base, device=self.device)
                    crop_d = torch.full((self.num_envs,), crop_d_base, device=self.device)
            else:
                # No randomization - use base crops for all environments
                crop_l = torch.full((self.num_envs,), crop_l_base, device=self.device)
                crop_r = torch.full((self.num_envs,), crop_r_base, device=self.device)
                crop_u = torch.full((self.num_envs,), crop_u_base, device=self.device)
                crop_d = torch.full((self.num_envs,), crop_d_base, device=self.device)
            
            # Vectorized cropping using advanced indexing
            # Create batch indices
            batch_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1).unsqueeze(1)
            
            # Create coordinate grids for each environment
            processed_images = []
            for i in range(self.num_envs):
                # Extract crop coordinates for this environment
                u, d, l, r = crop_u[i].item(), crop_d[i].item(), crop_l[i].item(), crop_r[i].item()
                
                # Crop and process the image (matching unvectorized logic exactly)
                cropped_image = cam_tensors[i, u:d, l:r, :3].permute(2, 0, 1).float() / 255.0
                processed_images.append(cropped_image)
            
            # Stack all processed images
            batch_images = torch.stack(processed_images, dim=0)  # (N, 3, k, k)
            
            # Apply the transform (this was missing in the original vectorized version)
            # Note: self.im_transform should handle batch processing
            final_images = self.im_transform(batch_images)  # (N, 3, im_size, im_size)
            
            # Store results
            if self.num_obs_cameras > 1:
                self.pixel_obs_buf[:, cam_idx] = final_images
            else:
                self.pixel_obs_buf[:] = final_images

        self.gym.end_access_image_tensors(self.sim)

    def _process_camera_batch(self, cam_idx, cam_config, shifts):
        """Process a batch of camera images for all environments."""
        # Get camera tensors for all environments
        if self.num_obs_cameras > 1:
            cam_tensors = [self.cam_tensors[cam_idx][i] for i in range(self.num_envs)]
        else:
            cam_tensors = [self.cam_tensors[i] for i in range(self.num_envs)]
        
        # Pre-compute crop parameters
        k = min(cam_config["w"], cam_config["h"])
        crop_l_base = (cam_config["w"] - k) // 2
        crop_u_base = (cam_config["h"] - k) // 2
        
        # Process all images in a list comprehension (much faster than nested loops)
        processed_images = []
        for i, (cam_tensor, shift) in enumerate(zip(cam_tensors, shifts)):
            # Calculate crop coordinates with randomization
            if k == cam_config["w"]:  # Horizontal crop
                crop_l, crop_r = crop_l_base, crop_l_base + k
                crop_u = max(0, min(cam_config["h"] - k, crop_u_base + shift))
                crop_d = crop_u + k
            else:  # Vertical crop  
                crop_u, crop_d = crop_u_base, crop_u_base + k
                crop_l = max(0, min(cam_config["w"] - k, crop_l_base + shift))
                crop_r = crop_l + k
            
            # Crop, convert, and permute in one go
            cropped = cam_tensor[crop_u:crop_d, crop_l:crop_r, :3].permute(2, 0, 1).float() / 255.0
            processed_images.append(cropped)
        
        # Stack all processed images and apply transform to the batch
        batch_images = torch.stack(processed_images, dim=0)
        return self.im_transform(batch_images)

    def compute_encoded_obs(self):
        if self.cfg["env"]["test_mode"] == 1:
            with torch.no_grad():
                if self.num_obs_cameras > 1:
                    encoded_obs_list = []
                    for cam_idx in range(self.num_obs_cameras):
                        encoded_obs_list.append(self.encoder(self.encoded_obs_transform(self.pixel_obs_buf[:, cam_idx])))
                    self.encoded_obs[:] = torch.cat(encoded_obs_list, dim=-1)
                else:
                    self.encoded_obs[:] = self.encoder(self.encoded_obs_transform(self.pixel_obs_buf))
        else:
            self.compute_encoded_obs_vectorized()

    def compute_encoded_obs_vectorized(self):
        """Efficient vectorized version of compute_encoded_obs."""
        with torch.no_grad():
            if self.num_obs_cameras > 1:
                # Reshape pixel observations for batch processing
                # From (num_envs, num_cameras, 3, H, W) to (num_envs * num_cameras, 3, H, W)
                original_shape = self.pixel_obs_buf.shape
                batch_pixel_obs = self.pixel_obs_buf.view(-1, *original_shape[2:])
                
                # Apply normalization to entire batch
                transformed_obs = self.encoded_obs_transform(batch_pixel_obs)
                
                # Encode entire batch at once
                encoded_batch = self.encoder(transformed_obs)
                
                # Reshape back to (num_envs, num_cameras, encoded_size)
                encoded_per_cam = encoded_batch.view(self.num_envs, self.num_obs_cameras, -1)
                
                # Concatenate across cameras: (num_envs, num_cameras * encoded_size_per_cam)
                self.encoded_obs[:] = encoded_per_cam.contiguous().view(self.num_envs, -1)
                
            else:
                # Single camera case - process all environments at once
                transformed_obs = self.encoded_obs_transform(self.pixel_obs_buf)
                self.encoded_obs[:] = self.encoder(transformed_obs)

    # def compute_successes(self):
    #     success_env_mask = self.object_picked.bool()
    #     curr_success = torch.where(self.successes < 10.0, torch.zeros_like(self.successes), self.successes)
    #     self.successes[:] = torch.where(success_env_mask, self.successes + 1, curr_success)

    #     # 10 steps at goal for success
    #     binary_curr_success = torch.where(self.successes >= 10, torch.ones_like(self.successes), torch.zeros_like(self.successes))
    #     self.successes[:] = torch.where(self.reset_buf > 0, binary_curr_success, self.successes)
    #     success_rate = (self.successes >= 10).float().mean()
    #     self.extras["success_rate"] = success_rate.item()

    def compute_successes(self):
        """
        Track consecutive steps where object is at goal.
        Episode is successful if object stays at goal for 10+ consecutive steps.
        """
        success_env_mask = self.object_picked.bool()

        self.successes[:] = torch.where(
            success_env_mask, 
            self.successes + 1, 
            torch.zeros_like(self.successes)
        )
        achieved_threshold = (self.successes >= 10).float()
        episode_success = achieved_threshold.clone()
        self.successes[:] = torch.where(
            self.reset_buf > 0, 
            torch.zeros_like(self.successes), 
            self.successes
        )
        current_success_rate = achieved_threshold.mean()
        self.extras["success_rate"] = current_success_rate.item()
        # self.extras["consecutive_successes_mean"] = self.successes.float().mean().item()
        # self.extras["max_consecutive_successes"] = self.successes.float().max().item()

    def reset_idx(self, env_ids):
        # env_ids_int32 = env_ids.to(dtype=torch.int32)

        # Domain randomization, can happen only at reset time since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # Reset object
        self._reset_init_object_state(env_ids=env_ids)
        self._object_state[env_ids] = self._init_object_state[env_ids]

        # Reset agent
        arm_reset_noise = torch.rand((len(env_ids), self.num_arm_dofs), device=self.device)
        hand_reset_noise = torch.rand((len(env_ids), self.num_hand_dofs), device=self.device)
        pos_arm = tensor_clamp(
            self.franka_default_dof_pos +
            self.franka_init_dof_noise * 2.0 * (arm_reset_noise - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        pos_hand = tensor_clamp(
            self.hand_default_dof_pos +
            self.hand_init_dof_noise * 2.0 * (hand_reset_noise - 0.5),
            self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        pos = torch.cat([pos_arm, pos_hand], dim=-1)

        # Reset internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Reset prevs
        self.prev_targets[env_ids, :] = pos
        self.curr_targets[env_ids, :] = pos
        self.prev_prev_targets[env_ids, :] = pos

        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self._dof_state),
                                            gymtorch.unwrap_tensor(multi_env_ids_int32),
                                            len(multi_env_ids_int32))     


        # Object multi env ids
        # object_multi_env_ids_int32 = self.global_indices[env_ids, self.env_object_ind].flatten()
        multi_env_ids_objects_int32 = self._global_indices[env_ids, -1:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_objects_int32), len(multi_env_ids_objects_int32))

        self.object_picked[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def _reset_init_object_state(self, env_ids):
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)
        sampled_object_state = torch.zeros(num_resets, 13, device=self.device)

        object_state_all = self._init_object_state
        object_heights = self.states["object_size"]

        # Sampling is "centered" around middle of table
        centered_object_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        sampled_object_state[:, 2] = self._table_surface_pos[2] + object_heights[env_ids] / 2
        sampled_object_state[:, 6] = 1.0
        sampled_object_state[:, :2] = centered_object_xy_state.unsqueeze(0) + \
                                            2.0 * self.object_init_pos_noise * (
                                                    torch.rand(num_resets, 2, device=self.device) - 0.5)

        if self.object_init_rot_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.object_init_rot_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_object_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_object_state[:, 3:7])

        object_state_all[env_ids, :] = sampled_object_state

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        # For joint position control (mujoco playground frankapickpixels)
        if self.control_type == "joint_pos_relative_prev":
            self._apply_actions_joint_pos_relative_prev()

        # Random impulse
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn(self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device)
                * self.object_rb_masses
                * self.force_scale
            )
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE
            )

    def _apply_actions_joint_pos_relative_prev(self):
        actions = self.actions[:]
        
        targets = self.prev_targets[:] + self.dof_speed_scale * self.dt * actions
        self.curr_targets[:] = targets

        # self.curr_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.curr_targets[:,self.actuated_dof_indices] + \
        #                                                     (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]

        # self.curr_targets[:, self.actuated_dof_indices] = tensor_clamp(self.curr_targets[:, self.actuated_dof_indices],
        #                                                                   self.hand_dof_lower_limits[self.actuated_dof_indices], self.hand_dof_upper_limits[self.actuated_dof_indices])

        self.dof_delta = self.curr_targets[:] - self.prev_targets[:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.curr_targets))
        self.prev_actions[:] = self.actions.clone()

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations() # Refresh tensors and then compute observations
        self.compute_reward()
        self.check_termination()
        self.compute_successes()

        self.compute_prevs()

        if self.enable_debug_vis and self.viewer:
            self.debug_vis()

    def compute_prevs(self):
        self.prev_targets[:, :7] = self.curr_targets[:, :7]

    def _setup_rewards(self):
        target_pos = to_torch(self.cfg["env"]["target_pos"], device=self.device)
        target_rot = to_torch(self.cfg["env"]["target_rot"], device=self.device)
        target_height_scalar = self._table_surface_pos[2] + 0.4
        target_height = torch.full((self.num_envs, 1), target_height_scalar.item(), device=self.device)
        lift_threshold = torch.full((self.num_envs,), 0.04, device=self.device)
        grasp_finger_dof_pos = self.grasp_finger_dof_pos
        self.reward_settings = {
            # Store table height for reward calculations
            "table_height": self._table_surface_pos[2],

            # Object init height
            "object_init_height": self._table_surface_pos[2] + self.states["object_size"] / 2,

            # Target
            "target_pos": target_pos,
            "target_rot": target_rot,
            "target_rot_6d": matrix_to_rotation_6d(quaternion_to_matrix_ig(target_rot)),
            # Only target height specified
            "target_height": target_height,

            # How much lifted off table to be counted as lifted
            "lift_threshold": lift_threshold,

            # Hand "grasp" dof pos
            "grasp_finger_dof_pos": grasp_finger_dof_pos
        }

        # Float reward scales for logging or non-script use
        self.reward_config = {
            "w_hand_obj": self.cfg["rewards"]["scales"]["w_hand_obj"],
            "w_lift": self.cfg["rewards"]["scales"]["w_lift"],
            "w_obj_goal": self.cfg["rewards"]["scales"]["w_obj_goal"],
            "w_curl": self.cfg["rewards"]["scales"]["w_curl"],
            "w_vel_penalty": self.cfg["rewards"]["penalty"]["w_vel_penalty"],

            "beta_hand_obj": self.cfg["rewards"]["params"]["beta_hand_obj"],
            "beta_obj_goal": self.cfg["rewards"]["params"]["beta_obj_goal"],
            "beta_lift": self.cfg["rewards"]["params"]["beta_lift"],
            "beta_curl": self.cfg["rewards"]["params"]["beta_curl"],
        }

    def _setup_obs_camera(self):
        self.enabled_cams = self.cfg["obs_camera"]["enabled_cams"]
        if not self.enabled_cams:
            return
        self.im_size = self.cfg["obs_camera"]["im_size"]

        self.cam_configs = {}
        self.cam_randomization_params = {}
        for cam_name in self.enabled_cams:
            self.cam_configs[cam_name] = {
                "type": self.cfg["obs_camera"][cam_name]["cam"]["type"],
                "w": self.cfg["obs_camera"][cam_name]["cam"]["w"],
                "h": self.cfg["obs_camera"][cam_name]["cam"]["h"],
                "fov": self.cfg["obs_camera"][cam_name]["cam"]["fov"],
                "ss": self.cfg["obs_camera"][cam_name]["cam"]["ss"],
                "near_plane": self.cfg["obs_camera"][cam_name]["cam"]["near_plane"],
                "far_plane": self.cfg["obs_camera"][cam_name]["cam"]["far_plane"],
                "loc_p": self.cfg["obs_camera"][cam_name]["cam"]["loc_p"],
                "loc_r": self.cfg["obs_camera"][cam_name]["cam"]["loc_r"],
                "attachment": self.cfg["obs_camera"][cam_name]["cam"].get("attachment", "palm_center")
            }
            assert self.cfg["obs_camera"][cam_name]["cam"]["w"] % 2 == 0
            
            cam_params = self.cfg["task"]["randomization_params"]["misc"]["camera"]   
            if self.randomize_camera:  
                self.cam_randomization_params[cam_name] = {
                    "pos_noise": cam_params["pos_sd"], 
                    "theta_noise": np.radians(cam_params["theta_sd"]),
                    "phi_noise": np.radians(cam_params["phi_sd"]),
                    "roll_noise": np.radians(cam_params["roll_sd"]),
                    "fov_noise": cam_params["fov_sd"],
                    "crop_shift": cam_params["crop_shift"]
                }
            else:
                self.cam_randomization_params[cam_name] = {
                    "pos_noise": 0.0, 
                    "theta_noise": 0.0,
                    "phi_noise": 0.0,
                    "roll_noise": 0.0,
                    "fov_noise": 0.0,
                    "crop_shift": 0.0
                }

        # Total number of cameras per environment
        self.num_obs_cameras = len(self.enabled_cams)
        if self.num_obs_cameras > 1:
            self.pixel_obs_size = (self.num_obs_cameras, 3, self.im_size, self.im_size)
        else:
            self.pixel_obs_size = (3, self.im_size, self.im_size)

    def _setup_visual_encoder(self):
        # Image mean and std
        if self.enable_encoder:

            # Placeholder encoder
            # model = {
            #     "vits": vit_s16,
            #     "vitb": vit_b16,
            #     "vitl": vit_l16,
            # }[self.cfg["vision"]["mvp_model"]]
            # self.encoder = model(pretrained=self.cfg["vision"]["mvp_weights"], img_size=self.im_size)[0].to(self.device)
            # print('EMBED DIM', self.encoder.embed_dim)
            # self.encoder.eval()

            # class SimplePlaceholderEncoder(torch.nn.Module):
            #     def __init__(self, output_size):
            #         super(SimplePlaceholderEncoder, self).__init__()
            #         self.output_size = output_size
                    
            #     def forward(self, x):
            #         flat = x.flatten(start_dim=1)
            #         return flat[:, :self.output_size]

            # self.encoder = SimplePlaceholderEncoder(self.encoded_obs_size).to(self.device)
            # self.encoder.eval()

            # Use TorchHub
            torch.hub.set_dir(self.cfg["obs_camera"]["encoder_weights"])
            model = torch.hub.load(
                self.cfg["obs_camera"]["encoder_path"],
                self.encoder_type_full.replace("-", "_") + "14",           # model entrypoint, patch size 14
                source='local',                                            # use the local hubconf
                pretrained=True,                                           # load weights automatically
            )
            model.eval()  
            
            # Load locally without TorchHub
            # from dinov2.models.vision_transformer import vit_small  # for ViT-S/14
            # model = vit_small(
            #     patch_size=14,   # must match the pretrained config
            #     img_size=526,    # default crop size used during pretraining
            #     init_values=1.0,
            #     block_chunks=0
            # )
            # state_dict = torch.load('/home/andrew/models/dinov2_vits14_pretrain.pth')
            # model.load_state_dict(state_dict, strict=True)
            # # for param in model.parameters():
            # #     param.requires_grad = False
            # model.eval()

            self.encoder = model.to(self.device)

        self.im_mean = [0.485, 0.456, 0.406]
        self.im_std = [0.229, 0.224, 0.225]
        self.pixel_obs_buf = torch.zeros((self.num_envs, *self.pixel_obs_size), dtype=torch.float32, device=self.device)
        self.encoded_obs = torch.zeros((self.num_envs, self.encoded_obs_size), dtype=torch.float32, device=self.device)
        self.im_transform = Resize((self.im_size, self.im_size))
        self.encoded_obs_transform = Compose([
            # Lambda(lambda img: adjust_sharpness(img, random.random() * 2) if random.random() > 0.5 else img),
            # GaussianBlur(kernel_size=(3, 3), sigma=(0.0001, 0.02)),
            Normalize(mean=self.im_mean, std=self.im_std)])
        pass

    def _setup_live_camera_viewer(self):
        """Setup matplotlib for live camera visualization"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # Use non-blocking backend
            matplotlib.use('TkAgg')  # or 'Qt5Agg' if TkAgg is not available
            
            self.plt = plt
            
            # Setup figure and axes
            if self.num_obs_cameras > 1:
                # Multi-camera: create subplots
                rows = 1
                cols = min(self.num_obs_cameras, 4)  # Max 4 cameras per row
                if self.num_obs_cameras > 4:
                    rows = 2
                    cols = (self.num_obs_cameras + 1) // 2
                    
                self.fig, self.axes = self.plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
                if rows == 1 and cols == 1:
                    self.axes = [self.axes]
                elif rows == 1:
                    self.axes = list(self.axes)
                else:
                    self.axes = self.axes.flatten()
                    
                # Setup each subplot
                self.im_artists = []
                for i, cam_name in enumerate(self.enabled_cams):
                    if i < len(self.axes):
                        ax = self.axes[i]
                        ax.set_title(f'{cam_name}', fontsize=12, fontweight='bold')
                        ax.axis('off')
                        # Create placeholder image
                        placeholder = np.zeros((224, 224, 3))
                        im = ax.imshow(placeholder, interpolation='bilinear')
                        self.im_artists.append(im)
                        
                # Hide unused subplots
                for i in range(len(self.enabled_cams), len(self.axes)):
                    self.axes[i].axis('off')
                    
            else:
                # Single camera
                self.fig, self.ax = self.plt.subplots(1, 1, figsize=(8, 8))
                cam_name = self.enabled_cams[0]
                self.ax.set_title(f'{cam_name}', fontsize=14, fontweight='bold')
                self.ax.axis('off')
                # Create placeholder image
                placeholder = np.zeros((224, 224, 3))
                self.im_artist = self.ax.imshow(placeholder, interpolation='bilinear')
            
            # Add main title
            self.fig.suptitle('Camera View', fontsize=14, fontweight='bold')
            self.plt.tight_layout()
            self.plt.ion()  # Turn on interactive mode
            self.plt.show(block=False)
            
            # Store viewer state
            self.viewer_active = True
            print("Live camera viewer initialized successfully")
            
        except Exception as e:
            print(f"Error setting up live camera viewer: {e}")
            print("Make sure you have a GUI backend available (TkAgg, Qt5Agg, etc.)")
            self.visualize_camera_obs = False
            self.viewer_active = False

    def _update_live_camera_viewer(self):
        """Update the live camera viewer with current observations"""
        if not hasattr(self, 'viewer_active') or not self.viewer_active:
            return
            
        try:
            # Check if matplotlib window is still open
            if not self.plt.get_fignums():
                print("Camera viewer window closed, disabling visualization")
                self.visualize_camera_obs = False
                self.viewer_active = False
                return
                
            env_idx = 0  # Visualize first environment
            
            if self.num_obs_cameras > 1:
                # Multi-camera case
                for cam_idx in range(min(len(self.im_artists), self.num_obs_cameras)):
                    # Convert tensor to numpy and transpose for matplotlib (H, W, C)
                    img_data = self.pixel_obs_buf[env_idx, cam_idx].cpu().numpy().transpose(1, 2, 0)
                    # Clip to valid range [0, 1]
                    img_data = np.clip(img_data, 0, 1)
                    
                    # Update image
                    self.im_artists[cam_idx].set_array(img_data)
                    
                    # Update title with current step info
                    cam_name = self.enabled_cams[cam_idx]
                    step_info = getattr(self, 'viz_step_count', 0) if hasattr(self, 'viz_step_count') else 0
                    self.axes[cam_idx].set_title(f'{cam_name} (Step: {step_info})', 
                                            fontsize=12, fontweight='bold')
                    
            else:
                # Single camera case
                # Convert tensor to numpy and transpose for matplotlib (H, W, C)
                img_data = self.pixel_obs_buf[env_idx].cpu().numpy().transpose(1, 2, 0)
                # Clip to valid range [0, 1]
                img_data = np.clip(img_data, 0, 1)
                
                # Update image
                self.im_artist.set_array(img_data)
                
                # Update title with current step info
                cam_name = self.enabled_cams[0]
                step_info = getattr(self, 'viz_step_count', 0) if hasattr(self, 'viz_step_count') else 0
                self.ax.set_title(f'{cam_name} (Step: {step_info})', 
                                fontsize=14, fontweight='bold')
            
            # Update main title with episode info if available
            if hasattr(self, 'progress_buf'):
                episode_step = self.progress_buf[env_idx].item()
                self.fig.suptitle(f'Camera View - Step: {episode_step}', 
                                fontsize=14, fontweight='bold')
            
            # Redraw efficiently
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            # Small delay to prevent overwhelming the display
            import time
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error updating live camera viewer: {e}")
            self.visualize_camera_obs = False
            self.viewer_active = False

    def _cleanup_live_camera_viewer(self):
        """Clean up the live camera viewer"""
        if hasattr(self, 'viewer_active') and self.viewer_active:
            try:
                if hasattr(self, 'plt'):
                    self.plt.close('all')
                    self.plt.ioff()  # Turn off interactive mode
                self.viewer_active = False
                print("Live camera viewer cleaned up")
            except Exception as e:
                print(f"Error cleaning up live camera viewer: {e}")

    def _save_camera_visualization(self):
        """Save camera observations for visualization during rollout"""
        try:
            import torchvision
            
            env_idx = 0  # Visualize first environment
            
            if self.num_obs_cameras > 1:
                # Multi-camera case
                for cam_idx, cam_name in enumerate(self.enabled_cams):
                    # Get raw pixel observation
                    raw_img = self.pixel_obs_buf[env_idx, cam_idx].cpu()
                    
                    # Save raw image
                    raw_filename = f"{self.viz_save_dir}/env{env_idx}_{cam_name}_step_{self.viz_step_count:06d}_raw.png"
                    torchvision.utils.save_image(raw_img, raw_filename, normalize=False)
                    
                    # Save normalized image (as fed to encoder)
                    if self.enable_encoder:
                        normalized_img = self.encoded_obs_transform(raw_img.unsqueeze(0)).squeeze(0)
                        norm_filename = f"{self.viz_save_dir}/env{env_idx}_{cam_name}_step_{self.viz_step_count:06d}_normalized.png"
                        torchvision.utils.save_image(normalized_img, norm_filename, normalize=True)
            else:
                # Single camera case
                raw_img = self.pixel_obs_buf[env_idx].cpu()
                cam_name = self.enabled_cams[0] if self.enabled_cams else "cam0"
                
                # Save raw image
                raw_filename = f"{self.viz_save_dir}/env{env_idx}_{cam_name}_step_{self.viz_step_count:06d}_raw.png"
                torchvision.utils.save_image(raw_img, raw_filename, normalize=False)
                
                # Save normalized image (as fed to encoder)
                if self.enable_encoder:
                    normalized_img = self.encoded_obs_transform(raw_img.unsqueeze(0)).squeeze(0)
                    norm_filename = f"{self.viz_save_dir}/env{env_idx}_{cam_name}_step_{self.viz_step_count:06d}_normalized.png"
                    torchvision.utils.save_image(normalized_img, norm_filename, normalize=True)
            
            # Print status occasionally
            if self.viz_step_count % (self.viz_save_frequency * 10) == 0:
                print(f"Saved camera visualization at step {self.viz_step_count}")
                
        except Exception as e:
            print(f"Error saving camera visualization: {e}")
            # Disable visualization to prevent spam
            self.visualize_camera_obs = False

    def debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Grab relevant states to visualize
        eef_pos = self.states["eef_pos"]
        eef_rot = self.states["eef_quat"]
        object_pos = self.states["object_pos"]
        object_rot = self.states["object_quat"]
        
        # Fingertip positions
        finger1_pos = self.states["eef_finger1_pos"]
        finger2_pos = self.states["eef_finger2_pos"]
        finger3_pos = self.states["eef_finger3_pos"]
        finger4_pos = self.states["eef_finger4_pos"]
        
        # Calculate goal position (same x,y as object but target height for z)
        goal_pos = self.reward_settings["target_pos"]

        # Other
        object_height = self.states["object_pos"][:, 2] - self.reward_settings["object_init_height"].squeeze(-1)
        lift_threshold = self.reward_settings["lift_threshold"]

        # Plot visualizations
        for i in range(self.num_envs):
            # # Draw coordinate frames for end-effector and object
            # for pos, rot in zip((eef_pos, object_pos), (eef_rot, object_rot)):
            #     px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
            #     py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
            #     pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

            #     p0 = pos[i].cpu().numpy()
            #     self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
            #     self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
            #     self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])
            
            # Visualize goal position
            goal_p0 = goal_pos.cpu().numpy()
            object_p0 = object_pos[i].cpu().numpy()
            
            # Draw a vertical line from the current object position to the goal height
            self.gym.add_lines(self.viewer, self.envs[i], 1, 
                            [object_p0[0], object_p0[1], object_p0[2], goal_p0[0], goal_p0[1], goal_p0[2]], 
                            [0.2, 0.8, 0.2])  # Green line to goal
            
            # # Draw a horizontal "circle" at the goal height to make it more visible
            # radius = 0.05
            # segments = 16
            # for seg in range(segments):
            #     theta1 = seg * 2 * np.pi / segments
            #     theta2 = (seg + 1) * 2 * np.pi / segments
            #     x1 = goal_p0[0] + radius * np.cos(theta1)
            #     y1 = goal_p0[1] + radius * np.sin(theta1)
            #     x2 = goal_p0[0] + radius * np.cos(theta2)
            #     y2 = goal_p0[1] + radius * np.sin(theta2)
            #     self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                     [x1, y1, goal_p0[2], x2, y2, goal_p0[2]],
            #                     [0.2, 0.8, 0.2])  # Green circle at goal
            
            # # Draw lines from each fingertip to the object
            # # Finger 1
            # f1_p0 = finger1_pos[i].cpu().numpy()
            # self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                 [f1_p0[0], f1_p0[1], f1_p0[2], object_p0[0], object_p0[1], object_p0[2]],
            #                 [1.0, 0.5, 0.0])  # Orange line from finger1 to object
            
            # # Finger 2
            # f2_p0 = finger2_pos[i].cpu().numpy()
            # self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                 [f2_p0[0], f2_p0[1], f2_p0[2], object_p0[0], object_p0[1], object_p0[2]],
            #                 [0.0, 0.5, 1.0])  # Blue line from finger2 to object
            
            # # Finger 3
            # f3_p0 = finger3_pos[i].cpu().numpy()
            # self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                 [f3_p0[0], f3_p0[1], f3_p0[2], object_p0[0], object_p0[1], object_p0[2]],
            #                 [1.0, 0.0, 1.0])  # Purple line from finger3 to object
            
            # # Finger 4
            # f4_p0 = finger4_pos[i].cpu().numpy()
            # self.gym.add_lines(self.viewer, self.envs[i], 1,
            #                 [f4_p0[0], f4_p0[1], f4_p0[2], object_p0[0], object_p0[1], object_p0[2]],
            #                 [1.0, 1.0, 0.0])  # Yellow line from finger4 to object

            # # Change object color when lifted, only when no DR
            # if not self.randomize:
            #     if object_height[i] > lift_threshold[i]: # lifted
            #         self.gym.set_rigid_body_color(self.envs[i], self._object_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 1.0, 0.0))
            #     else: # not lifted
            #         self.gym.set_rigid_body_color(self.envs[i], self._object_id, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.0, 0.0))

            # Camera visualization for obs_camera cam1
            if self.pixel_vision and "cam1" in self.enabled_cams:
                self._debug_vis_camera(i)

    def _debug_vis_camera(self, env_idx=0):
        """Debug visualization for camera pose and frustum"""
        if "cam1" not in self.enabled_cams:
            return
            
        cam_config = self.cam_configs["cam1"]
        
        # Get the attachment body's world pose
        attachment_name = cam_config["attachment"]
        if attachment_name == "floating":
            # If floating, attached to table - get table pose (assuming it's at origin)
            attachment_pos = torch.tensor([0.0, 0.0, 1.0 + 0.3], device=self.device)  # table surface
            attachment_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        else:
            # Get robot body pose
            if attachment_name in self.handles:
                body_idx = self.handles[attachment_name]
            else:
                # Find the body index for the attachment
                env_ptr = self.envs[env_idx]
                robot_handle = self.robots[env_idx]
                try:
                    body_idx = self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, attachment_name)
                except:
                    return  # Can't find attachment body
            
            attachment_pos = self._rigid_body_state[env_idx, body_idx, :3]
            attachment_quat = self._rigid_body_state[env_idx, body_idx, 3:7]
        
        # Convert camera local transform to tensors
        cam_local_pos = torch.tensor(cam_config["loc_p"], device=self.device, dtype=torch.float32)
        
        # Convert camera rotation from degrees to radians to quaternion
        xyz_angle_rad = [np.radians(a) for a in cam_config["loc_r"]]
        # Convert euler to quaternion (ZYX order as in the original code)
        cam_local_quat = torch.tensor([
            np.sin(xyz_angle_rad[0]/2) * np.cos(xyz_angle_rad[1]/2) * np.cos(xyz_angle_rad[2]/2) - np.cos(xyz_angle_rad[0]/2) * np.sin(xyz_angle_rad[1]/2) * np.sin(xyz_angle_rad[2]/2),
            np.cos(xyz_angle_rad[0]/2) * np.sin(xyz_angle_rad[1]/2) * np.cos(xyz_angle_rad[2]/2) + np.sin(xyz_angle_rad[0]/2) * np.cos(xyz_angle_rad[1]/2) * np.sin(xyz_angle_rad[2]/2),
            np.cos(xyz_angle_rad[0]/2) * np.cos(xyz_angle_rad[1]/2) * np.sin(xyz_angle_rad[2]/2) - np.sin(xyz_angle_rad[0]/2) * np.sin(xyz_angle_rad[1]/2) * np.cos(xyz_angle_rad[2]/2),
            np.cos(xyz_angle_rad[0]/2) * np.cos(xyz_angle_rad[1]/2) * np.cos(xyz_angle_rad[2]/2) + np.sin(xyz_angle_rad[0]/2) * np.sin(xyz_angle_rad[1]/2) * np.sin(xyz_angle_rad[2]/2)
        ], device=self.device, dtype=torch.float32)
        
        # Transform camera local position to world coordinates
        cam_world_pos = attachment_pos + quat_apply(attachment_quat, cam_local_pos)
        
        # Combine rotations: world_quat = attachment_quat * local_quat
        cam_world_quat = quat_mul(attachment_quat, cam_local_quat)
        
        # Create coordinate axes for camera (transform from +X to +Z pointing forward)
        axis_length = 0.1
        
        # Apply rotation to transform camera from looking at +X to looking at +Z
        # This is a +90 degree rotation around Y-axis: X->Z, Y->Y, Z->-X
        transform_quat = torch.tensor([0.0, 0.7071068, 0.0, 0.7071068], device=self.device, dtype=torch.float32)  # 90 deg around Y
        cam_transformed_quat = quat_mul(cam_world_quat, transform_quat)
        
        # Camera coordinate system: +X right, +Y up, +Z forward (into scene)
        cam_x_axis = quat_apply(cam_transformed_quat, torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32)) * axis_length
        cam_y_axis = quat_apply(cam_transformed_quat, torch.tensor([0, 1, 0], device=self.device, dtype=torch.float32)) * axis_length  
        cam_z_axis = quat_apply(cam_transformed_quat, torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32)) * axis_length
        
        # Convert to numpy for visualization
        cam_pos_np = cam_world_pos.cpu().numpy()
        cam_x_end = (cam_world_pos + cam_x_axis).cpu().numpy()
        cam_y_end = (cam_world_pos + cam_y_axis).cpu().numpy() 
        cam_z_end = (cam_world_pos + cam_z_axis).cpu().numpy()
        
        # Draw coordinate axes (RGB = XYZ)
        # X-axis (red)
        self.gym.add_lines(self.viewer, self.envs[env_idx], 1, 
                        [cam_pos_np[0], cam_pos_np[1], cam_pos_np[2], 
                        cam_x_end[0], cam_x_end[1], cam_x_end[2]], 
                        [1.0, 0.0, 0.0])
        
        # Y-axis (green) 
        self.gym.add_lines(self.viewer, self.envs[env_idx], 1,
                        [cam_pos_np[0], cam_pos_np[1], cam_pos_np[2],
                        cam_y_end[0], cam_y_end[1], cam_y_end[2]],
                        [0.0, 1.0, 0.0])
        
        # Z-axis (blue) - this points in camera's view direction
        self.gym.add_lines(self.viewer, self.envs[env_idx], 1,
                        [cam_pos_np[0], cam_pos_np[1], cam_pos_np[2],
                        cam_z_end[0], cam_z_end[1], cam_z_end[2]],
                        [0.0, 0.0, 1.0])
        
        # Draw frustum circles at different distances
        near_dist = 0.05  # Near circle distance
        far_dist = 0.2    # Far circle distance
        
        # Calculate circle radii based on FOV
        fov_rad = np.radians(cam_config["fov"])
        near_radius = near_dist * np.tan(fov_rad / 2)
        far_radius = far_dist * np.tan(fov_rad / 2)
        
        # Function to draw a circle perpendicular to camera Z-axis
        def draw_frustum_circle(center_pos, radius, color, segments=16):
            center_np = center_pos.cpu().numpy()
            
            # Create circle in camera's local X-Y plane
            for i in range(segments):
                theta1 = i * 2 * np.pi / segments
                theta2 = (i + 1) * 2 * np.pi / segments
                
                # Points in camera local coordinates
                local_p1 = torch.tensor([radius * np.cos(theta1), radius * np.sin(theta1), 0], 
                                    device=self.device, dtype=torch.float32)
                local_p2 = torch.tensor([radius * np.cos(theta2), radius * np.sin(theta2), 0], 
                                    device=self.device, dtype=torch.float32)
                
                # Transform to world coordinates (using transformed quaternion)
                world_p1 = center_pos + quat_apply(cam_transformed_quat, local_p1)
                world_p2 = center_pos + quat_apply(cam_transformed_quat, local_p2)
                
                world_p1_np = world_p1.cpu().numpy()
                world_p2_np = world_p2.cpu().numpy()
                
                # Draw circle segment
                self.gym.add_lines(self.viewer, self.envs[env_idx], 1,
                                [world_p1_np[0], world_p1_np[1], world_p1_np[2],
                                world_p2_np[0], world_p2_np[1], world_p2_np[2]],
                                color)
        
        # Draw near circle (smaller, brighter)
        near_center = cam_world_pos + cam_z_axis * (near_dist / axis_length)
        draw_frustum_circle(near_center, near_radius, [1.0, 1.0, 0.0])  # Yellow
        
        # Draw far circle (larger, dimmer) 
        far_center = cam_world_pos + cam_z_axis * (far_dist / axis_length)
        draw_frustum_circle(far_center, far_radius, [0.8, 0.8, 0.0])   # Dim yellow
        
        # Draw frustum edges connecting the circles
        segments = 8  # Fewer lines for frustum edges
        for i in range(0, segments, 2):  # Every other segment to avoid clutter
            theta = i * 2 * np.pi / segments
            
            # Points on near circle
            local_near = torch.tensor([near_radius * np.cos(theta), near_radius * np.sin(theta), 0], 
                                    device=self.device, dtype=torch.float32)
            world_near = near_center + quat_apply(cam_transformed_quat, local_near)
            
            # Points on far circle  
            local_far = torch.tensor([far_radius * np.cos(theta), far_radius * np.sin(theta), 0],
                                    device=self.device, dtype=torch.float32) 
            world_far = far_center + quat_apply(cam_transformed_quat, local_far)
            
            world_near_np = world_near.cpu().numpy()
            world_far_np = world_far.cpu().numpy()
            
            # Draw edge line
            self.gym.add_lines(self.viewer, self.envs[env_idx], 1,
                            [world_near_np[0], world_near_np[1], world_near_np[2],
                            world_far_np[0], world_far_np[1], world_far_np[2]], 
                            [0.6, 0.6, 0.0])  # Darker yellow


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat

@torch.jit.script
def compute_franka_leap_reward1(
    reset_buf, progress_buf, actions, states, reward_settings, reward_config, max_episode_length, num_arm_dofs
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Dict[str, float], int, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    
    # Hand (palm, fingers) to object distance
    d_palm = torch.norm(states["object_pos"] - states["eef_pos"], dim=-1)
    d_finger1 = torch.norm(states["object_pos"] - states["eef_finger1_pos"], dim=-1)
    d_finger2 = torch.norm(states["object_pos"] - states["eef_finger2_pos"], dim=-1)
    d_finger3 = torch.norm(states["object_pos"] - states["eef_finger3_pos"], dim=-1)
    d_finger4 = torch.norm(states["object_pos"] - states["eef_finger4_pos"], dim=-1)
    
    # Max dist component to object: max_i∈{palm_pos,fingertips} ||x^i - x^obj||
    d_hand_obj = torch.max(torch.stack([d_palm, d_finger1, d_finger2, d_finger3, d_finger4], dim=1), dim=1)[0]
    closest_dist = torch.min(torch.stack([d_palm, d_finger1, d_finger2, d_finger3, d_finger4], dim=1), dim=1)[0]
    
    # Hand object distance reward: r_hand_obj = exp(-beta_hand_obj * d_hand_obj)
    beta_hand_obj = reward_config["beta_hand_obj"]
    r_hand_obj = torch.exp(-beta_hand_obj * d_hand_obj)
    


    # Goal height
    target_height = reward_settings["target_height"].squeeze(-1)
    goal_pos = torch.clone(states["object_pos"])
    goal_pos[:, 2] = target_height
    
    # Lifting bonus: r_lift = 1.0 * w_lift if object is lifted
    object_height = states["object_pos"][:, 2] - reward_settings["object_init_height"].squeeze(-1)
    r_lift = torch.where(object_height > reward_settings["lift_threshold"], 1.0, torch.zeros_like(object_height))



    # Object goal height distance reward: r_obj_goal = exp(-beta_obj_goal ||x^obj - x^goal||)
    object_init_height = reward_settings["object_init_height"].squeeze(-1)
    height_diff = torch.abs(states["object_pos"][:, 2] - target_height)
    init_to_target_dist = torch.abs(object_init_height - target_height)
    height_progress = torch.clamp((init_to_target_dist - height_diff) / init_to_target_dist, 0.0, 1.0)
    beta_obj_goal = reward_config["beta_obj_goal"]
    r_obj_goal = height_progress * torch.exp(-beta_obj_goal * height_diff)



   # Finger curl
    hand_dof_pos = states["q"][:, num_arm_dofs:]
    near_object = d_hand_obj <= 0.15
    finger_pos_diff = torch.sum((hand_dof_pos - reward_settings["grasp_finger_dof_pos"]) ** 2, dim=1)

    # Finger curl reward: r_curl = TODO: testing
    beta_curl = reward_config["beta_curl"]
    r_curl= torch.exp(-beta_curl * finger_pos_diff)
    r_curl = torch.where(near_object, r_curl, 0.0)



    # Velocity penalty
    q_vel = states["qd"]
    vel_penalty = torch.sum(q_vel**2, dim=-1)
    r_vel_penalty = vel_penalty

    # Calculate reward
    w_hand_obj = reward_config["w_hand_obj"]
    w_obj_goal = reward_config["w_obj_goal"]
    w_lift = reward_config["w_lift"]
    w_curl = reward_config["w_curl"]
    w_vel_penalty = reward_config["w_vel_penalty"]

    r_hand_obj_scaled = w_hand_obj * r_hand_obj
    r_obj_goal_scaled = w_obj_goal * r_obj_goal
    r_lift_scaled = w_lift * r_lift
    r_curl_scaled = w_curl * r_curl
    r_vel_penalty_scaled = w_vel_penalty * r_vel_penalty
    
    rewards = r_hand_obj_scaled + r_obj_goal_scaled + r_lift_scaled + r_curl_scaled + r_vel_penalty_scaled



    # Success
    # Object lifted and within threshold of target height
    object_picked = torch.where(torch.abs(object_height - target_height) <= 0.04, 1.0, 0.0)

    return (rewards, reset_buf, 
            r_hand_obj, r_obj_goal, r_lift, r_curl, r_vel_penalty,
            r_hand_obj_scaled, r_obj_goal_scaled, r_lift_scaled, r_curl_scaled, r_vel_penalty_scaled,
            object_picked)

@torch.jit.script
def compute_franka_leap_reward2(
    reset_buf, progress_buf, actions, states, reward_settings, reward_config, max_episode_length, num_arm_dofs
):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Dict[str, float], int, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    
    # Hand (palm, fingers) to object distance
    d_palm = torch.norm(states["object_pos"] - states["eef_pos"], dim=-1)
    d_finger1 = torch.norm(states["object_pos"] - states["eef_finger1_pos"], dim=-1)
    d_finger2 = torch.norm(states["object_pos"] - states["eef_finger2_pos"], dim=-1)
    d_finger3 = torch.norm(states["object_pos"] - states["eef_finger3_pos"], dim=-1)
    d_finger4 = torch.norm(states["object_pos"] - states["eef_finger4_pos"], dim=-1)
    
    # Max dist component to object: max_i∈{palm_pos,fingertips} ||x^i - x^obj||
    d_hand_obj = torch.stack([d_palm, d_finger1, d_finger2, d_finger3, d_finger4], dim=1)
    d_hand_obj = torch.max(d_hand_obj, dim=1)[0]
    
    # Hand object distance reward: r_hand_obj = exp(-beta_hand_obj * d_hand_obj)
    beta_hand_obj = reward_config["beta_hand_obj"]
    r_hand_obj = torch.exp(-beta_hand_obj * d_hand_obj)
    


    # Goal height
    target_height = reward_settings["target_height"].squeeze(-1)
    target_pos = reward_settings["target_pos"].squeeze(-1)
    goal_pos = target_pos
    
    # Lifting bonus: r_lift = 1.0 * w_lift if object is lifted
    object_height = states["object_pos"][:, 2] - reward_settings["object_init_height"].squeeze(-1)
    r_lift = torch.where(object_height > reward_settings["lift_threshold"], 1.0, torch.zeros_like(object_height))

    # Object goal distance reward: r_obj_goal = exp(-beta_obj_goal ||x^obj - x^goal||)
    # Only activate when object lifted
    d_obj_goal = torch.norm(states["object_pos"] - goal_pos, dim=-1)
    beta_obj_goal = reward_config["beta_obj_goal"]
    r_obj_goal = torch.exp(-beta_obj_goal * d_obj_goal)
    r_obj_goal = torch.where(object_height > reward_settings["lift_threshold"], r_obj_goal, 0.0)



    # Finger curl
    hand_dof_pos = states["q"][:, num_arm_dofs:]
    near_object = d_hand_obj <= 0.15
    finger_pos_diff = torch.sum((hand_dof_pos - reward_settings["grasp_finger_dof_pos"]) ** 2, dim=1)

    # Finger curl reward: r_curl = TODO: testing
    beta_curl = reward_config["beta_curl"]
    r_curl= torch.exp(-beta_curl * finger_pos_diff)
    r_curl = torch.where(near_object, r_curl, 0.0)



    # Velocity penalty
    q_vel = states["qd"]
    vel_penalty = torch.sum(q_vel**2, dim=-1)
    r_vel_penalty = vel_penalty

    # Calculate reward
    w_hand_obj = reward_config["w_hand_obj"]
    w_obj_goal = reward_config["w_obj_goal"]
    w_lift = reward_config["w_lift"]
    w_curl = reward_config["w_curl"]
    w_vel_penalty = reward_config["w_vel_penalty"]

    r_hand_obj_scaled = w_hand_obj * r_hand_obj
    r_obj_goal_scaled = w_obj_goal * r_obj_goal
    r_lift_scaled = w_lift * r_lift
    r_curl_scaled = w_curl * r_curl
    r_vel_penalty_scaled = w_vel_penalty * r_vel_penalty
    
    rewards = r_hand_obj_scaled + r_obj_goal_scaled + r_lift_scaled + r_curl_scaled + r_vel_penalty_scaled



    # Success
    # Object lifted and within threshold of target height
    object_picked = torch.where(d_obj_goal <= 0.04, 1.0, 0.0)

    return (rewards, reset_buf, 
            r_hand_obj, r_obj_goal, r_lift, r_curl, r_vel_penalty,
            r_hand_obj_scaled, r_obj_goal_scaled, r_lift_scaled, r_curl_scaled, r_vel_penalty_scaled,
            object_picked)
