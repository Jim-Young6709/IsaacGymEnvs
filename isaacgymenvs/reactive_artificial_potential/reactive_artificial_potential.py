

import torch
import numpy as np
import pinocchio as pin

from .utils.franka_collision_checker import FrankaCollisionChecker, SELF_COLLISION_SPHERES
from .utils.franka_kinematics import FrankaKinematics


class ReactiveArtificialPotential:
    def __init__(self, env):
        self.env = env
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.gripper_width = 0.04
        self.collision_checker = FrankaCollisionChecker()
        self.franka_kinematics = FrankaKinematics()


    def get_closest_surface_point_jacobian(self, point_cloud: np.ndarray, joint_pos: np.ndarray):
        """
        Finds the point on the robot surface (sphere-based) closest to the input point cloud
        and returns the Jacobian at that point.

        Returns:
            Tuple[np.ndarray, str, np.ndarray, np.ndarray]: 
                - Jacobian at the surface point (6, dof),
                - link name,
                - surface point on the robot in world frame,
                - closest point in the point cloud.
        """

        q = np.ones(8)
        q[:7] = joint_pos
        q[-1] = self.gripper_width
        fk = self.collision_checker.robot.link_fk(q, use_names=True)

        min_dist = float('inf')
        closest_link = None

        for link_name, center, radius in SELF_COLLISION_SPHERES:
            sphere_center = (fk[link_name] @ np.array([*center, 1]))[:3]
            dists = np.linalg.norm(point_cloud - sphere_center, axis=1)
            closest_idx = np.argmin(dists)
            dist_to_sphere_surface = dists[closest_idx] - radius
            if dist_to_sphere_surface < min_dist:
                min_dist = dist_to_sphere_surface
                closest_link = link_name
                closest_point = point_cloud[closest_idx]
                closest_sphere_center = sphere_center
                closest_sphere_radius = radius

        if closest_link is None:
            raise RuntimeError("Could not find a closest link.")

        if closest_link == "panda_hand":
            closest_link = "franka_ee"

        # Compute surface point
        direction = closest_point - closest_sphere_center
        direction /= np.linalg.norm(direction)
        surface_point = closest_sphere_center + closest_sphere_radius * direction

        # Pinocchio Jacobian
        pin_joint_pos = np.asarray(joint_pos)
        pin.framesForwardKinematics(self.franka_kinematics.pin_model, self.franka_kinematics.pin_data, pin_joint_pos)
        frame_id = self.franka_kinematics.pin_model.getFrameId(closest_link)
        J_frame = pin.computeFrameJacobian(
            self.franka_kinematics.pin_model, self.franka_kinematics.pin_data, pin_joint_pos, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        oMf = self.franka_kinematics.pin_data.oMf[frame_id]
        r = surface_point - oMf.translation

        def skew(v):
            return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])

        adj = np.block([
            [np.eye(3), np.zeros((3, 3))],
            [-skew(r), np.eye(3)]
        ])
        J_surface = adj @ J_frame

        return J_surface, closest_link, surface_point, closest_point



    def compute_repulsive_joint_torque(self, J_surface, link_name, surface_point, closest_point, d0=0.1, eta=5.0):
        """
        Compute joint torques caused by a repulsive potential field from the closest
        point cloud point, exerted at the closest point on the robot's surface.

        Args:
            joint_pos (np.ndarray): Robot joint configuration (7,).
            d0 (float): Repulsion cutoff distance.
            eta (float): Gain for the potential field.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Joint torques (dof,), and the repulsive force vector (3,)
        """
        
        if link_name in ["panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "franka_ee"]:
            # Compute repulsive force direction
            d_vec = surface_point - closest_point
            dist = np.linalg.norm(d_vec)

            if dist < 1e-5 or dist > d0:
                return np.zeros(J_surface.shape[1]), np.zeros(3), None, None

            direction = d_vec / dist
            f_repulse = eta * (d0 - dist) * direction  # 3D force vector

            # Spatial force: [f; 0]
            F = np.zeros(6)
            F[0:3] = f_repulse

            # tau = J^T F
            tau = J_surface.T @ F

            return tau, f_repulse, surface_point, closest_point
        
        else:
            return np.zeros(J_surface.shape[1]), np.zeros(3), None, None
    

    def apply_reactive_artificial_potential(self, env_obs_dict):
        # [(n, 3), (m, 3), ... ] -> length is num_envs
        joint_pos_array = env_obs_dict["joint_pos"].cpu().numpy()
        dynamic_obstacles_list = env_obs_dict["moving_dynamic_obstacle_pcd"]
    

        for i, dynamic_pcd_torch in enumerate(dynamic_obstacles_list):
            joint_pos = joint_pos_array[i]
            dynamic_pcd_np = dynamic_pcd_torch.cpu().numpy()

            if len(dynamic_pcd_np) > 0:
                J_surface, link_name, surface_point, closest_point = self.get_closest_surface_point_jacobian(dynamic_pcd_np, joint_pos)
                tau_repulsion, f_repulse, closest_surface_point, closest_pcd_point = self.compute_repulsive_joint_torque(
                    J_surface, link_name, surface_point, closest_point, d0=0.1, eta=3.0, # d0=0.2
                )
                if closest_surface_point is not None:
                    # should run through forward dynamics, but in practice, this works too
                    new_joint_pos_target = torch.tensor(joint_pos + tau_repulsion, device=self.device)
                    env_obs_dict["goal_joint_pos"][i] = new_joint_pos_target.clone()
        
        env_obs_dict["goal_robot_pcd"] = self.env.get_robot_pcds(env_obs_dict["goal_joint_pos"])
        return env_obs_dict




