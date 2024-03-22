"""
Try to incorporate this file with FrankaReach later
"""
import os
import time

import cv2
import hydra
import imageio
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.franka_reach import FrankaReach
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


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


def test_ik(env):
    ik_failures = 0
    error_bound = 1e-4
    for _ in tqdm(range(25)):
        for _ in range(1000):
            env.execute_joint_action(np.random.uniform(-10, 10, 7).reshape(1, -1))
        ee_pos, ee_quat, target_angles = env.get_proprio()
        target_ee_pose = np.concatenate((ee_pos, ee_quat), axis=1).copy()
        env.reset_idx(torch.zeros(1).long())
        new_joint_angles = env.get_joint_from_ee(torch.from_numpy(target_ee_pose))
        env.set_robot_joint_state(new_joint_angles)
        new_ee_pos, new_ee_quat, joint_angles = env.get_proprio()
        new_ee_pose = np.concatenate((new_ee_pos, new_ee_quat), axis=1)
        error = np.linalg.norm(new_ee_pose - target_ee_pose)
        print("delta error: ", new_ee_pose - target_ee_pose)
        if error > error_bound:
            ik_failures += 1
    env.reset_idx(torch.zeros(1).long())
    print("percentage of IK failures: ", ik_failures / 25)


class FrankaMP(FrankaReach):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        graphics_device_id = 0
        virtual_screen_capture = False
        force_render = False
        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        self.goal_tolerance = cfg["env"].get("goal_tolerance", 0.05)
        self.canonical_joint_config = torch.Tensor(
            [[0, 0.1963, 0, -2.6180, 0, 2.9416, 0.7854]] * self.num_envs
        ).to(self.device)
        self.seed_joint_angles = self.get_proprio()[2].clone()
        self.num_collisions = torch.zeros(self.num_envs, device=self.device)

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
        pos[2] = pos[2] + size[2] / 2
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
        pos[2] = pos[2] + size
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
        pos[2] = pos[2] + size[0] + size[1]
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*pos)
        start_pose.r = gymapi.Quat(*[0.0, -0.707, 0.0, 0.707])  # quat in xyzw order
        self.capsule_dims.append(size)
        return asset, start_pose

    def _reset_obstacle(self):
        pass

    def setup_configs(self):
        self.goal_config, valid_scene = self.sample_valid_joint_configs(
            initial_configs=self.goal_config, check_not_in_collision=True
        )
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

    def step(
        self,
        action: torch.Tensor,
        isDeta=True,
        isInterpolation=False,
        exe_steps=10,
        detect_collision=True,
        render=False,
        debug=False,
    ):
        """
        Step in the environment with an action.
        Args:
            action (np.array): (num_envs, 7) action to take
        Returns:
            observation (dict): new observation dictionary
            reward (float): reward for this step
            done (bool): whether the task is done
            info (dict): extra information
        """

        gripper_state = torch.Tensor([[0.035, 0.035]] * self.num_envs).to(self.device)
        done = self.is_done()
        if isDeta:
            action = self.get_joint_angles() + action

        start_angles = self.get_joint_angles()
        start_state = torch.cat((start_angles, gripper_state), dim=1)
        target_state = torch.cat((action, gripper_state), dim=1)

        for step in range(exe_steps):
            action = start_state + (target_state - start_state) * (step + 1) / exe_steps
            if isInterpolation:
                self.set_robot_joint_state(action)
            else:
                super().step(action)
                if detect_collision:
                    self.check_robot_collision()
                    self.num_collisions += self.collision
                if debug:
                    self.print_collision_info()
            if render:
                self.render()

        if debug:
            print("step errors: ", torch.norm(target_state[:, :7] - self.get_joint_angles(), dim=1))

    def get_info(self):
        return None

    def is_done(self):
        """
        Compute done for env.
        Returns:
            done (bool): whether the task is done
        """
        return False

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
            eef_pose (torch.Tensor): (num_envs, 7) 7-dof end effector pose. quaternion in wxyz format
        """
        return torch.cat((self.get_proprio()[0], self.get_proprio()[1]), dim=1)

    def setup_camera(
        self,
        image_size=(512, 512),
        fov=75,
        camera_position=(1.5, 0.0, 2.0),
        camera_target=(0.0, 0.0, 1.5),
    ):
        """
        Setup the camera for all environments.
        Args:
            image_size (tuple): (height, width) of the image
            fov (float): field of view of the camera
            camera_position (tuple): (x, y, z) position of the camera
            camera_target (tuple): (x, y, z) position of the camera target
        Returns:
            list: list of camera handles
        """
        camera_properties = gymapi.CameraProperties()
        camera_properties.height, camera_properties.width = image_size
        camera_properties.horizontal_fov = fov
        self.camera_properties = camera_properties

        camera_handles = []
        for i in range(self.num_envs):
            camera_handle = self.gym.create_camera_sensor(self.envs[i], camera_properties)
            # TODO (mdalal): set the camera location appropriately for each vec env, currently just hardcoded to the best spot for env 0
            self.gym.set_camera_location(
                camera_handle,
                self.envs[i],
                gymapi.Vec3(*camera_position),
                gymapi.Vec3(*camera_target),
            )
            camera_handles.append(camera_handle)
        return camera_handles

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
        obs = self.compute_observations()
        ee_pos, ee_quat, joint_angles = obs[:, :3], obs[:, 3:7], obs[:, 7:14]
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

    def check_robot_collision(self):
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.scene_collision = torch.where(
            torch.norm(torch.sum(self.contact_forces[:, :16, :], dim=1), dim=1) > 1.0, 1.0, 0.0
        )  # the first 16 elements belong to franka robot
        self.collision = torch.where(
            torch.sum(torch.norm(self.contact_forces[:, :16, :], dim=2), dim=1) > 1.0, 1.0, 0.0
        )  # the first 16 elements belong to franka robot

    def print_collision_info(self):
        self.check_robot_collision()
        num_collision = int(sum(self.collision))
        if num_collision:
            print(f"num_collision: {num_collision}/{self.num_envs}")
            if num_collision <= 5:
                print(f"env_incollision: {torch.nonzero(self.collision)[:, 0]}")

    def print_resampling_info(self, joint_configs):
        self.check_robot_collision()
        resampling_idx = torch.nonzero(
            (torch.sum(self._q[:, :7] - joint_configs, axis=1) != 0) + self.collision
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

        if not torch.allclose(joint_state, self.get_proprio()[2][env_ids]) and debug:
            print("------------")
            print("set state failed due to collision")
            err = torch.norm(joint_state - self.get_proprio()[2][env_ids], dim=1)
            num_err = int(sum(torch.where(err > 1e-2, 1.0, 0.0)))
            if num_err <= 5:
                print(torch.nonzero(err)[:, 0])
            else:
                print(f"num_err: {num_err}/{self.num_envs}")
            self.print_resampling_info(joint_state)
            print("------------")

    def step_sim_multi(self, num_steps=1):
        """
        Step the simulation. (for debugging purposes)
        """
        for _ in range(num_steps):
            self.gym.simulate(self.sim)
            self.render()
            self._refresh()

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
            cam_pos = gymapi.Vec3(1.5, 0.0, 2)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.5)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def render(self, image_type="rgb"):
        """
        Render the scene.
        """
        if self.headless:
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)

            images = []
            for i in range(self.num_envs):
                image = self.gym.get_camera_image(
                    self.sim,
                    self.envs[i],
                    self.camera_handles[i],
                    IMAGE_TYPES[image_type],
                )

                if image_type == "depth":
                    max_depth = 5
                    image = np.clip(image * -1, a_min=0, a_max=max_depth) / max_depth
                    image = (image * 255).astype(np.uint8)
                image = image.reshape(
                    (self.camera_properties.height, self.camera_properties.width, -1)
                )[..., :3]
                images.append(np.expand_dims(image, 0))
            image = np.concatenate(images, axis=0)
            return image
        else:
            super().render()

    def execute_plan(
        self, plan, exe_steps, set_intermediate_states=False, render=False, debug=False
    ):
        """
        Args:
            plan (List[torch.Tensor]): list of joint angles, including initial starting state
        Returns:
            error (float): error in final joint angles
        """
        print("Executing plan...")
        print(f"Plan length: {len(plan)}")
        self.check_robot_collision()
        assert not sum(self.collision) > 0, "Robot in collision!"

        # now take path and execute
        self.num_collisions = torch.zeros(self.num_envs, device=self.device)
        for plan_idx, action in enumerate(plan):
            self.step(
                action=action,
                isDeta=False,
                isInterpolation=set_intermediate_states,
                exe_steps=exe_steps,
                detect_collision=True,
                render=render,
                debug=debug,
            )

        achieved_joint_angles = self.get_joint_angles()
        joint_error = torch.norm(achieved_joint_angles - plan[-1], dim=1)
        if debug:
            print(f"execution results:\njoint_err: {joint_error} deg")
            print(f"{torch.sum(self.num_collisions!=0)} / {self.num_envs} envs has collided")
        return joint_error

    def reset_env(self):
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

    def decompose_pcd_params(self):
        num_obs = int(self.max_num_obstacles / 3)
        # need to exclude the extra counted ground plane at the beginning
        cuboid_all_dims = torch.Tensor(
            np.asarray(self.cuboid_dims[1:]).reshape(self.num_envs, num_obs + 1, 3)
        ).to(self.device)
        # format of _root_state: (num_envs, actor_num<franka_actor, plane_actor, obstacle_actor>, 13)
        # format of _obstacle_state: (num_envs, max_num_obstacles<max_num_cuboids, max_num_capsules, max_num_spheres>, 13)
        gp_dim = cuboid_all_dims[:, 0:1, :]
        gp_center = self._root_state[:, 1:2, :3]
        gp_quaternion = self._root_state[:, 1:2, 3:7]

        cuboid_dims = cuboid_all_dims[:, 1:, :]
        cuboid_centers = self._root_state[:, 2 : num_obs + 2, :3]
        cuboid_quaternions = self._root_state[:, 2 : num_obs + 2, 3:7]

        capsule_dims = torch.Tensor(
            np.asarray(self.capsule_dims).reshape(self.num_envs, num_obs, 2)
        ).to(self.device)
        capsule_radii = capsule_dims[:, :, 0:1]
        capsule_heights = capsule_dims[:, :, 1:2] * 2
        capsule_centers = self._root_state[:, num_obs + 2 : num_obs * 2 + 2, :3]
        capsule_quaternions = self._root_state[:, num_obs + 2 : num_obs * 2 + 2, 3:7]

        sphere_radii = torch.Tensor(
            np.asarray(self.sphere_radii).reshape(self.num_envs, num_obs, 1)
        ).to(self.device)
        sphere_centers = self._root_state[:, num_obs * 2 + 2 :, :3]
        return (
            gp_dim,
            gp_center,
            gp_quaternion,
            cuboid_dims,
            cuboid_centers,
            cuboid_quaternions,
            capsule_radii,
            capsule_heights,
            capsule_centers,
            capsule_quaternions,
            sphere_centers,
            sphere_radii,
        )


@hydra.main(config_name="config", config_path="../cfg/")
def launch_test(cfg: DictConfig):
    import isaacgymenvs
    from isaacgymenvs.learning import amp_continuous, amp_models, amp_network_builder, amp_players
    from isaacgymenvs.utils.rlgames_utils import (
        RLGPUAlgoObserver,
        RLGPUEnv,
        get_rlgames_env_creator,
    )
    from rl_games.algos_torch import model_builder
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

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
        env.reset_env()
        t2 = time.time()
        print(f"Reset time: {t2 - t1}")
        env.set_robot_joint_state(env.start_config)
        env.print_resampling_info(env.start_config)

        env.render()

    print(f"Average Error: {total_error / num_plans}")
    print(f"Percentage of failed plans: {num_failed_plans / num_plans * 100} ")


if __name__ == "__main__":
    launch_test()
