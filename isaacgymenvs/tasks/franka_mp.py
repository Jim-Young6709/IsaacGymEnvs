"""
Residual RL for motion planning
"""
import os
import time

import cv2
import hydra
import imageio
import isaacgym
import numpy as np
import torch
import abc
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from collections import OrderedDict
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from neural_mp.real_utils.model import NeuralMPModel
from robofin.pointcloud.torch import FrankaSampler

IMAGE_TYPES = {
    "rgb": gymapi.IMAGE_COLOR,
    "depth": gymapi.IMAGE_DEPTH,
    "segmentation": gymapi.IMAGE_SEGMENTATION,
}


def compute_ik(damping, j_eef, num_envs, dpose, device):
    """
    Compute the inverse kinematics for the end effector.
    Args:
        damping (float): Damping factor.
        j_eef (torch.Tensor): Jacobian of the end effector.
        num_envs (int): Number of environments.
        dpose (torch.Tensor): delta pose: position error, orientation error. (6D)
        device (torch.device): Device to use.
    """
    # TODO (mdalal): fix this, currently the IK is quite far off
    # solve damped least squares
    dpose = np.expand_dims(dpose, axis=-1)
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping**2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(
        num_envs, 7
    )  # J^T (J J^T + lambda I)^-1 dpose
    return u


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


class FrankaMP(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        self.device = sim_device
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # Controller type
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_position"},\
            "Invalid control type specified. Must be one of: {osc, joint_position}"

        assert "numObservations" in self.cfg["env"], "numObservations must be specified in the config"
        assert "numActions" in self.cfg["env"], "numActions must be specified in the config"

        # Values to be filled in at runtime
        self.states = {}                        # will be dict filled with relevant states to use for reward calculation
        self.handles = {}                       # will be dict mapping names to relevant sim handles
        self.num_dofs = None                    # Total number of DOFs per env
        self.actions = None                     # Current actions to be deployed

        # Tensor placeholders
        self._root_state = None                 # State of root body        (n_envs, 13)
        self._dof_state = None                  # State of all joints       (n_envs, n_dof)
        self._q = None                          # Joint positions           (n_envs, n_dof)
        self._qd = None                         # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None           # State of all rigid bodies (n_envs, n_bodies, 13)
        self._contact_forces = None             # Contact forces in sim
        self._eef_state = None                  # end effector state        (at grasping point)
        self._eef_lf_state = None               # end effector state        (at left fingertip)
        self._eef_rf_state = None               # end effector state        (at left fingertip)
        self._j_eef = None                      # Jacobian for end effector
        self._mm = None                         # Mass matrix
        self._arm_control = None                # Tensor buffer for controlling arm
        self._gripper_control = None            # Tensor buffer for controlling gripper
        self._pos_control = None                # Position actions
        self._effort_control = None             # Torque actions
        self._franka_effort_limits = None       # Actuator effort limits for franka
        self._global_indices = None             # Unique indices corresponding to all envs in flattened array

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        # Set control limits
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else to_torch([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0)

        self.pcd_spec_dict = cfg['pcd_spec']
        self.gpu_fk_sampler = FrankaSampler(sim_device, use_cache=True)
        self.goal_tolerance = cfg["env"].get("goal_tolerance", 0.05)

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

        if not hasattr(self, 'canonical_joint_config'):
            self.canonical_joint_config = torch.tensor(
                [[0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854]] * self.num_envs
            ).to(self.device)
        self.seed_joint_angles = self.canonical_joint_config.clone()
        self.num_collisions = torch.zeros(self.num_envs, device=self.device)
        self.success_flags = torch.zeros(cfg["env"]["numEnvs"], device=self.device) # 0 for failure, 1 for success
        self.base_model = NeuralMPModel.from_pretrained("jimyoung6709/DRP")
        self.base_model.eval()

        # Refresh tensors & Reset all environments
        self._refresh()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.3 # according to current randomization params, -0.275 would be the lowest surface from the env
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, spacing, num_per_row):
        """
        loading obstacles and franka robot in the environment

        Args:
            spacing (_type_): _description_
            num_per_row (_type_): _description_
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # setup params
        table_thickness = 0.05
        table_stand_height = 0.1
        self.cuboid_dims = []  # xyz
        self.capsule_dims = []  # r, l
        self.sphere_radii = []  # r
        self.combined_pcds = None

        # setup franka
        franka_dof_props = self._create_franka()
        franka_asset = self.franka_asset
        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness + table_stand_height)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # setup table
        table_asset, table_start_pose = self._create_cube(
            pos=[0.0, 0.0, 1.0],
            size=[1.2, 1.2, table_thickness],
        )

        # setup table stand
        table_stand_asset, table_stand_start_pose = self._create_cube(
            pos=[-0.5, 0.0, 1.0 + table_thickness],
            size=[0.2, 0.2, table_stand_height],
        )

        # setup sphere
        sphere_asset, sphere_start_pose = self._create_sphere(pos=[0, 0, 0], size=0.3)

        # setup capsule
        capsule_asset, capsule_start_pose = self._create_capsule(pos=[-0.1, 0, 1], size=[0.15, 0.3])

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = num_franka_bodies + 4  # 1 for table, table stand
        max_agg_shapes = num_franka_shapes + 4  # 1 for table, table stand

        self.frankas = []
        self.envs = []

        # Create environments
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            franka_actor = self.gym.create_actor(
                env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table
            self.table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, 1, 0
            )
            self.table_stand_actor = self.gym.create_actor(
                env_ptr, table_stand_asset, table_stand_start_pose, "table_stand", i, 1, 0
            )
            self.sphere_actor = self.gym.create_actor(
                env_ptr, sphere_asset, sphere_start_pose, "sphere", i, 1, 0
            )
            self.capsule_actor = self.gym.create_actor(
                env_ptr, capsule_asset, capsule_start_pose, "capsule", i, 1, 0
            )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        # Setup data
        actor_num = 1 + 1 + 3
        self.init_data(actor_num=actor_num)

    def _create_franka(self, ):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_gripper.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        self.franka_asset = franka_asset

        # todo modify this for joint position control vs. osc
        if self.control_type == "osc":
            franka_dof_stiffness = to_torch([0, 0, 0, 0, 0, 0, 0, 5000., 5000.], dtype=torch.float, device=self.device)
            franka_dof_damping = to_torch([0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
        elif self.control_type == "joint_position":
            franka_dof_stiffness = to_torch([1000.0]*7 + [800., 800.], dtype=torch.float, device=self.device)
            franka_dof_damping = to_torch([50]* 7 + [40., 40.], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            if self.control_type == "joint_position":
                franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            else:
                franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            self._franka_effort_limits.append(franka_dof_props['effort'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self._franka_effort_limits = to_torch(self._franka_effort_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200
        return franka_dof_props

    def init_data(self, actor_num):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_hand"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_leftfinger_tip"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_rightfinger_tip"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, franka_handle, "panda_grip_site"),
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
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, franka_handle)['panda_hand_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :7]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_indices = torch.arange(self.num_envs * actor_num, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1) # 3 actors, franka, table, table_stand

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
        return asset, start_pose

    def _create_sphere(self, pos, size):
        """
        Args:
            position (np.ndarray): (3,) xyz position of the sphere center
            size (float): radius of the sphere
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
        return asset, start_pose

    def _create_capsule(self, pos, size):
        """
        Args:
            position (np.ndarray): (3,) xyz position of the capsule center
            size (np.ndarray): (2,) radius and length of the capsule
                radius (float): radius of the sphere
                length (float): length of the capsule
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
        return asset, start_pose

    def _reset_obstacle(self):
        pass

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self.states.update({
            # Franka
            "q": self._q[:, :],
            "q_gripper": self._q[:, -2:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": self._eef_lf_state[:, :3],
            "eef_rf_pos": self._eef_rf_state[:, :3],
        })
        self.check_robot_collision()

    def compute_observations(self):
        self._refresh()
        # Note this will only work for robomimic checkpoints, the encoded pcd_feats dim is (pcd_feat+current_angles+goal_angles = 1024+7+7)
        self.update_robot_pcds()
        obs_base = OrderedDict()
        obs_base["current_angles"] = self.states['q'][:, :7].clone()
        obs_base["goal_angles"] = self.goal_config.clone()
        obs_base["compute_pcd_params"] = self.combined_pcds.clone()
        with torch.no_grad():
            self.base_delta_action = self.base_model.policy.get_action(obs_dict=obs_base)
        pcd_feats = self.base_model.policy.nets['policy'].model.encoded_feats.clone()
        pcd_feats = pcd_feats.contiguous().view(pcd_feats.size(0), -1) # (num_envs, 1038) , 1038 = 1024 (pointnet++_feat) + 7 (current) + 7 (goal)

        obs_residual = pcd_feats
        obs_residual[:, -14:-7] += self.base_delta_action

        obs_residual = torch.cat((obs_residual, self.base_delta_action), dim=1)

        self.obs_buf = obs_residual

        return self.obs_buf

    def check_robot_collision(self):
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.scene_collision = torch.where(
            torch.norm(torch.sum(self.contact_forces[:, :16, :], dim=1), dim=1) > 1.0, 1.0, 0.0
        )  # the first 16 elements belong to franka robot
        self.collision = torch.where(
            torch.sum(torch.norm(self.contact_forces[:, :16, :], dim=2), dim=1) > 1.0, 1.0, 0.0
        )  # the first 16 elements belong to franka robot, this includes self collision

    def setup_configs(self):
        self.goal_config, valid_scene = self.sample_valid_joint_configs(
            initial_configs=self.goal_config, check_not_in_collision=True
        )
        self.goal_pose = self.get_eef_pose() # (:, 7) xyz, wxyz
        if not valid_scene:
            print("Failed to sample valid goal config")
            return False
        print("Sampled valid goal config")
        self.start_config, valid_scene = self.sample_valid_joint_configs(
            initial_configs=self.start_config, check_not_in_collision=True
        )
        if not valid_scene:
            print("Failed to sample valid start config")
            return False
        print("Sampled valid start config")
        return True

    def normalize_franka_joints(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Normalize joint angles to be within the joint limits.
        Args:
            joint_angles (torch.Tensor): (num_envs, 7)
        Returns:
            joint_angles (torch.Tensor): (num_envs, 7)
        """
        lower_limits, upper_limits = self.get_joint_limits()
        desired_lower_limits = -1 * torch.ones_like(joint_angles)
        desired_upper_limits = 1 * torch.ones_like(joint_angles)
        normalized = (joint_angles - lower_limits) / (upper_limits - lower_limits) * (
            desired_upper_limits - desired_lower_limits
        ) + desired_lower_limits
        return normalized

    def unnormalize_franka_joints(self, joint_angles: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize joint angles.
        Args:
            joint_angles (torch.Tensor): (num_envs, 7)
        Returns:
            joint_angles (torch.Tensor): (num_envs, 7)
        """
        lower_limits, upper_limits = self.get_joint_limits()
        franka_limit_range = upper_limits - lower_limits
        desired_lower_limits = -1 * torch.ones_like(joint_angles)
        desired_upper_limits = 1 * torch.ones_like(joint_angles)
        unnormalized = (joint_angles - desired_lower_limits) * franka_limit_range / (
            desired_upper_limits - desired_lower_limits
        ) + lower_limits
        return unnormalized

    def get_joint_angles(self) -> torch.Tensor:
        """
        Get the joint angles of the robot.
        Returns:
            joint_angles (torch.Tensor): (num_envs, 7) 7-dof joint angles.
        """
        return self.get_proprio()[2]

    def get_eef_pose(self) -> torch.Tensor:
        """
        Get the end effector pose of the robot.
        Returns:
            eef_pose (torch.Tensor): (num_envs, 7) 7-dof end effector pose. quaternion in xyzw format
        """
        return torch.cat((self.get_proprio()[0], self.get_proprio()[1]), dim=1)

    def get_proprio(self):
        """
        Get the proprioceptive states of the robot: ee_pos, ee_quat, joint_angles
        Returns:
            ee_pos (Tensor): 3D end effector position.
            ee_quat (Tensor): 4D end effector quaternion.
            joint_angles (Tensor): (num_envs, 7) 7-dof joint angles.
        TODO: maybe add in gripper position support as well
        """
        self.gym.fetch_results(self.sim, True)
        ee_pos = self.states['eef_pos']
        ee_quat = self.states['eef_quat']
        joint_angles = self.states['q'][:, :7]
        return ee_pos, ee_quat, joint_angles

    def get_joint_from_ee(self, target_ee_pose):
        """
        Get the joint angles from the end effector pose.
        Args:
            target_ee_pose (np.ndarray): 7D end effector pose.
        Returns:
            joint_angles (np.ndarray): 7-dof joint angles.
        """
        # TODO (mdalal): IK seems to be systematically off, need to fix this
        start_angles = self.get_proprio()[2].copy()
        self.set_robot_joint_state(self.seed_joint_angles)
        for _ in range(100):
            num_envs = self.num_envs
            damping = 0.05
            ee_pos, ee_quat, joint_angles = self.get_proprio()
            dpos = target_ee_pose[:, :3] - ee_pos
            dori = orientation_error(target_ee_pose[:, 3:], ee_quat)

            dpose = torch.cat((dpos, dori), dim=1)
            delta_joint_angles = compute_ik(damping, self._j_eef, num_envs, dpose, self.device)
            joint_angles = joint_angles + delta_joint_angles
            self.set_robot_joint_state(joint_angles)
            error = torch.norm(dpose, dim=1)
            print(error)
        self.set_robot_joint_state(start_angles)
        return joint_angles

    def get_ee_from_joint(self, joint_angles,  frame="right_gripper"):
        """
        Get the end effector pose from the joint angles.
        Args:
            joint_angles (torch.Tensor): 7-dof joint angles. (B, 7)
        Returns:
            ee_pose (torch.Tensor)): 7D end effector pose. xyz, xyzw
        """
        eef_tranforms = self.gpu_fk_sampler.end_effector_pose(joint_angles, frame)
        eef_xyz = eef_tranforms[:, :3, 3]

        eef_rotations = eef_tranforms[:, :3, :3].cpu().numpy()
        eef_xyzw = Rotation.from_matrix(eef_rotations).as_quat()
        eef_xyzw = torch.Tensor(eef_xyzw).to(self.device)

        return torch.cat((eef_xyz, eef_xyzw), dim=1)

    def set_joint_pos_from_ee_pos(self, target_ee_pose):
        """
        Set the joint angles from the end effector pose.
        Args:
            target_ee_pose (np.ndarray): 7D end effector pose.
        """
        joint_angles = self.get_joint_from_ee(target_ee_pose)
        self.set_robot_joint_state(joint_angles)
        achieved_ee_pose = self.get_proprio()[0]
        assert np.allclose(achieved_ee_pose, target_ee_pose[:, :3], atol=1e-4), np.linalg.norm(
            achieved_ee_pose - target_ee_pose[:, :3]
        )

    def get_success(self, goal_angles, check_not_in_collision=False):
        """
        Compute successes for each env.
        Args:
            goal_angles (np.ndarray): (num_envs, 7)
            check_not_in_collision (bool): If True, also check that the robot is in a collision free state.
        Returns:
            successes (np.ndarray): (num_envs,)
        """
        # TODO: modify success metric to use position and orientation error
        _, _, joint_angles = self.get_proprio()
        goal_dists = np.linalg.norm(goal_angles - joint_angles, axis=1)
        success = goal_dists < self.goal_tolerance
        if check_not_in_collision:
            success = success and self.no_collisions
        return success

    def sample_valid_joint_configs(
        self, initial_configs=None, check_not_in_collision=False, max_attempts=50, debug=False
    ):
        """
        Sample valid joint configurations. Must be collision free.
        Args:
            check_not_in_collision (bool): If True, also check that the robot is in a collision free state.
        Returns:
            joint_configs (np.ndarray): (num_envs, 7)
            (bool): whether sampled configs are valid
        """
        joint_configs = (
            self.sample_joint_configs(num_envs=self.num_envs)
            if initial_configs is None
            else initial_configs.clone()
        )

        if check_not_in_collision:
            self.set_robot_joint_state(joint_configs)
            count = 0
            while True:
                count += 1
                self.check_robot_collision()
                resampling_idx = torch.nonzero(
                    (torch.sum(self._q[:, :7] - joint_configs, axis=1) != 0) + self.collision
                )[:, 0]
                num_resampling = len(resampling_idx)
                if (not num_resampling) or (count > max_attempts):
                    break

                resampling = self.sample_joint_configs(num_envs=num_resampling)
                joint_configs[resampling_idx] = resampling
                self.set_robot_joint_state(joint_configs)
                if debug:
                    self.print_resampling_info(joint_configs)
            if num_resampling:
                if debug:
                    print("------------")
                    self.print_resampling_info(joint_configs)
                    print("------------")
                self.invalid_scene_idx = resampling_idx
                return joint_configs, False
        return joint_configs, True

    def print_collision_info(self):
        self.check_robot_collision()
        num_collision = int(sum(self.collision))
        if num_collision:
            print(f"num_collision: {num_collision}/{self.num_envs}")
            if num_collision <= 5:
                print(f"env_incollision: {torch.nonzero(self.collision)[:, 0]}")

    def print_resampling_info(self, joint_configs, env_ids):
        self.check_robot_collision()
        resampling_idx = torch.nonzero(
            (torch.sum(self._q[env_ids, :7] - joint_configs, axis=1) != 0) + self.scene_collision[env_ids]
        )[:, 0]
        num_resampling = len(resampling_idx)
        print(f"num_resampling: {num_resampling}/{self.num_envs}")
        if num_resampling <= 5:
            print(f"env_inresampling: {resampling_idx}")

    def set_robot_joint_state(self, joint_state: torch.Tensor, env_ids=None, debug=False):
        """
        Set the joint state of the robot. (set the dof state (pos/vel) of each joint,
        for MP we don't care about vel, so make it 0 and the gripper joints can be fully open (.035, 0.035))

        Args:
            joint_state (torch.Tensor): (num_selected_envs, 7)
        """
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        assert len(joint_state) == len(env_ids)

        if joint_state.size(1) == 7:
            gripper_state = torch.Tensor([[0.035, 0.035]] * len(env_ids)).to(self.device)
            state_tensor = torch.cat((joint_state, gripper_state), dim=1).unsqueeze(2)
        state_tensor = torch.cat((state_tensor, torch.zeros_like(state_tensor)), dim=2)
        pos = state_tensor[:, :, 0].contiguous()

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
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

        self.gym.simulate(self.sim)
        self._refresh()

        if debug:
            self.render()

            if not torch.allclose(joint_state, self.get_proprio()[2][env_ids]):
                print("------------")
                print("set state failed due to collision")
                err = torch.norm(joint_state - self.get_proprio()[2][env_ids], dim=1)
                num_err = int(sum(torch.where(err > 1e-2, 1.0, 0.0)))
                if num_err <= 5:
                    print(torch.nonzero(err)[:, 0])
                else:
                    print(f"num_err: {num_err}/{self.num_envs}")
                self.print_resampling_info(joint_state, env_ids)
                num_scene_collision = int(sum(self.scene_collision))
                print(f"num_scene_collision: {num_scene_collision}/{self.num_envs}")
                print("------------")
            else:
                print("set state success")

    def get_joint_limits(self):
        """
        Get the joint limits of the robot.

        Returns:
            lower_limits (torch.Tensor): (7,)
            upper_limits (torch.Tensor): (7,)
        """
        lower_limits = self.franka_dof_lower_limits[:7]
        upper_limits = self.franka_dof_upper_limits[:7]
        return lower_limits, upper_limits

    def sample_joint_configs(self, num_envs: int) -> torch.Tensor:
        """
        Sample a valid joint configuration within the joint limits of the robot.

        Inputs:
            num_envs (int): number of isaac gym environments
        Returns:
            joint_config (torch.Tensor): (num_envs, 7)
        """
        lower_limits, upper_limits = self.get_joint_limits()
        joint_config = (upper_limits - lower_limits) * torch.rand(
            (num_envs, lower_limits.shape[0]), device=self.device
        ) + lower_limits
        return joint_config

    def set_viewer(self):
        """
        Create the viewer.
        NOTE: hardcoded for single env setup.
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
            
            cam_pos = gymapi.Vec3(-1.5, -1.5, 5)
            cam_target = gymapi.Vec3(centre, centre, 0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def reset_idx(self, env_ids=None):
        """
        Reset the environment.
        """
        self.start_config = None
        self.goal_config = None
        self.invalid_scene_idx = []
        self._reset_obstacle()
        while True:
            valid_scene = self.setup_configs()
            if valid_scene:
                break
            self._reset_obstacle(self.invalid_scene_idx)
            print("setup_configs failed, reset obstacles and retry")

        # currently doesn't support reset selected envs
        self.progress_buf[:] = 0
        self.reset_buf[:] = 0
        self.base_model.start_episode()
        self.compute_observations()

    def pre_physics_step(self, actions):
        delta_actions = actions.clone().to(self.device)
        gripper_state = torch.Tensor([[0.035, 0.035]] * self.num_envs).to(self.device)

        delta_actions = torch.clamp(delta_actions, -self.cmd_limit, self.cmd_limit) / self.action_scale
        abs_actions = self.get_joint_angles() + delta_actions + self.base_delta_action
        if abs_actions.shape[-1] == 7:
            abs_actions = torch.cat((abs_actions, gripper_state), dim=1)

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(abs_actions))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # reset the robot to start if it collides with the obstacles
        if sum(self.scene_collision) > 0:
            self.reset_buf = torch.where(self.scene_collision > 0, torch.ones_like(self.reset_buf), self.reset_buf)
            self.success_flags[self.scene_collision.bool()] = 0

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Grab relevant states to visualize
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]

            # Plot visualizations
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos), (eef_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

    def update_robot_pcds(self):
        num_robot_points = self.pcd_spec_dict['num_robot_points']
        num_target_points = self.pcd_spec_dict['num_target_points']
        robot_pcd = self.gpu_fk_sampler.sample(self.get_joint_angles(), num_robot_points)
        target_pcd = self.gpu_fk_sampler.sample(self.goal_config, num_target_points)
        self.combined_pcds[:, :num_robot_points, :3] = robot_pcd
        self.combined_pcds[:, -num_target_points:, :3] = target_pcd

    # debug utils
    def plan_open_loop(self, use_controller=True):
        """
        debug base policy
        """
        obs_base = OrderedDict()

        for i in range(150):
            self.update_robot_pcds()
            obs_base["current_angles"] = self.states['q'][:, :7].clone()
            obs_base["goal_angles"] = self.goal_config.clone()
            obs_base["compute_pcd_params"] = self.combined_pcds.clone()
            base_action = self.base_model.policy.get_action(obs_dict=obs_base)
            abs_action = base_action + self.get_joint_angles()
            if use_controller:
                self.step(base_action)
                self._refresh()
            else:
                self.set_robot_joint_state(abs_action)
                self.gym.simulate(self.sim)
                self.render()
                self._refresh()

    def visualize_pcd_meshcat(self, env_idx: int=0):
        "for debug purposes"
        import meshcat
        import urchin
        from robofin.robots import FrankaRobot
        self.viz = meshcat.Visualizer()
        self.urdf = urchin.URDF.load(FrankaRobot.urdf)
        for idx, (k, v) in enumerate(self.urdf.visual_trimesh_fk(np.zeros(8)).items()):
            self.viz[f"robot/{idx}"].set_object(
                meshcat.geometry.TriangularMeshGeometry(k.vertices, k.faces),
                meshcat.geometry.MeshLambertMaterial(wireframe=False),
            )
            self.viz[f"robot/{idx}"].set_transform(v)
        pcd_rgb = np.zeros((3, 8192))
        pcd_rgb[0, :2048] = 1
        pcd_rgb[1, 2048:6144] = 1
        pcd_rgb[2, 6144:] = 1
        
        self.viz['pcd'].set_object(
            meshcat.geometry.PointCloud(
                position=self.combined_pcds[env_idx, :, :3].cpu().numpy().T,
                color=pcd_rgb,
                size=0.005,
            )
        )

    def step_sim_multi(self, num_steps=1):
        """
        Step the simulation. (for debugging purposes)
        """
        for _ in range(num_steps):
            self.gym.simulate(self.sim)
            self.render()
            self._refresh()

    def render_multi(self, num_steps=1):
        """
        Render the simulation. (for debugging purposes)
        """
        for _ in range(num_steps):
            self.render()

    @abc.abstractmethod
    def compute_reward(self, actions):
        pass


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
    env = FrankaMP(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    total_error = 0
    num_failed_plans = 0
    num_plans = 1000
    for i in tqdm(range(num_plans)):
        t1 = time.time()
        env.reset_idx()
        t2 = time.time()
        print(f"Reset time: {t2 - t1}")
        env.set_robot_joint_state(env.start_config)
        env.print_resampling_info(env.start_config)

        env.render()

    print(f"Average Error: {total_error / num_plans}")
    print(f"Percentage of failed plans: {num_failed_plans / num_plans * 100} ")


if __name__ == "__main__":
    launch_test()
