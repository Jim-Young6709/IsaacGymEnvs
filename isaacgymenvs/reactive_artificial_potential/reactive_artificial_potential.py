

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


    def compute_repulsive_joint_torque(self, J_surface, link_name, surface_point, closest_point, d0=0.2, eta=3.0):
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
                # add obstacle points
                num_obstacle_points = 1000 #500
                if len(dynamic_pcd_np) > num_obstacle_points:
                    random_obstacle_indices = np.random.choice(len(dynamic_pcd_np), size=num_obstacle_points, replace=False)
                else:
                    random_obstacle_indices = np.random.choice(len(dynamic_pcd_np), size=num_obstacle_points, replace=True)
              
                dynamic_pcd_np_downsample = dynamic_pcd_np[random_obstacle_indices, 0:3]

                J_surface, link_name, surface_point, closest_point = self.get_closest_surface_point_jacobian(dynamic_pcd_np_downsample, joint_pos)
                tau_repulsion, f_repulse, closest_surface_point, closest_pcd_point = self.compute_repulsive_joint_torque(
                    J_surface, link_name, surface_point, closest_point, d0=0.2, eta=3.0, # d0=0.
                )
                if closest_surface_point is not None:
                    # should run through forward dynamics, but in practice, this works too
                    new_joint_pos_target = torch.tensor(joint_pos + tau_repulsion, device=self.device)
                    env_obs_dict["goal_joint_pos"][i] = new_joint_pos_target.clone()
        
        env_obs_dict["goal_robot_pcd"] = self.env.get_robot_pcds(env_obs_dict["goal_joint_pos"])
        return env_obs_dict



    def apply_reactive_artificial_potential_vectorized(self, env_obs_dict):
        joint_pos_tensor = env_obs_dict["joint_pos"]
        dynamic_obstacles_list = env_obs_dict["moving_dynamic_obstacle_pcd"]
        num_obstacle_points = 1000

        subsampled_pcd_list = list()
        has_dynamic_obstacles_flag = torch.ones(self.num_envs, device=self.device, dtype=bool)

        for i, pcd in enumerate(dynamic_obstacles_list):
            num_points = pcd.shape[0]
            if num_points == 0:
                # Avoid sampling from empty tensor
                sampled = torch.zeros((num_obstacle_points, 3), device=self.device)
                has_dynamic_obstacles_flag[i] = False
            elif num_points >= num_obstacle_points:
                indices = torch.randperm(num_points, device=self.device)[:num_obstacle_points]
                sampled = pcd[indices]
            else:
                indices = torch.randint(0, num_points, (num_obstacle_points,), device=self.device)
                sampled = pcd[indices]
            subsampled_pcd_list.append(sampled)
        # (num_envs, num_obstacle_points, 3)
        dynamic_pcd_downsample = torch.stack(subsampled_pcd_list, dim=0)
        # dynamic_pcd_downsample = torch.ones_like(dynamic_pcd_downsample, device=self.device)

        # ([num_envs, 9, 4, 4]) | 9 links = link 0-7 + panda hand
        link_transforms = self.collision_checker.compute_transformations(joint_pos_tensor)#[:, 0:9, :, :]

        link_name_to_index = {
            "panda_link0": 0,
            "panda_link1": 1,
            "panda_link2": 2,
            "panda_link3": 3,
            "panda_link4": 4,
            "panda_link5": 5,
            "panda_link6": 6,
            "panda_link7": 7,
            "panda_hand": 8,
        }

        # Prepare all sphere centers in local frame
        sphere_offsets = torch.tensor([center for _, center, _ in SELF_COLLISION_SPHERES], device='cuda')  # (num_spheres, 3)
        sphere_offsets_homo = torch.cat([sphere_offsets, torch.ones((len(SELF_COLLISION_SPHERES), 1), device='cuda')], dim=1)  # (num_spheres, 4)
        sphere_radii = torch.tensor([r for _, _, r in SELF_COLLISION_SPHERES], device='cuda')  # (num_spheres,)
        sphere_link_indices = torch.tensor([link_name_to_index[name] for name, _, _ in SELF_COLLISION_SPHERES], device='cuda')  # (num_spheres,)

        # Expand transforms for each sphere
        selected_transforms = link_transforms[:, sphere_link_indices]  # (num_envs, num_spheres, 4, 4)
        # sphere_offsets_homo: (num_spheres, 4)
        # expand it to (1, num_spheres, 4, 1) and broadcast
        sphere_offsets_expanded = sphere_offsets_homo[None, :, :, None]  # (1, num_spheres, 4, 1)
        sphere_offsets_expanded = sphere_offsets_expanded.expand(self.num_envs, -1, -1, -1)  # (num_envs, num_spheres, 4, 1)
        # now matmul works: (num_envs, num_spheres, 4, 4) @ (num_envs, num_spheres, 4, 1)
        sphere_offsets_world = torch.matmul(selected_transforms, sphere_offsets_expanded)  # (num_envs, num_spheres, 4, 1)
        sphere_centers_world = sphere_offsets_world[:, :, 0:3, 0]  # (num_envs, num_spheres, 3)


        # Compute distances between each sphere center and all points in the pointcloud
        # (num_envs, num_spheres, num_obstacle_points)
        dists = torch.norm(
            dynamic_pcd_downsample[:, None, :, :] - sphere_centers_world[:, :, None, :],
            dim=3
        )

        # Get closest point index and corresponding distance for each env and sphere
        min_dist_vals, min_point_indices = torch.min(dists, dim=2)  # (num_envs, num_spheres)

        # Adjust for surface distance
        dist_to_surface = min_dist_vals - sphere_radii[None, :]  # (num_envs, num_spheres)

        # Get the closest sphere per env
        closest_sphere_dists, closest_sphere_idx = torch.min(dist_to_surface, dim=1)  # (num_envs,)
        closest_point_idx = min_point_indices[torch.arange(self.num_envs, device='cuda'), closest_sphere_idx]  # (num_envs,)

        # Extract results
        closest_link_indices = sphere_link_indices[closest_sphere_idx]  # (num_envs,)
        closest_sphere_centers = sphere_centers_world[torch.arange(self.num_envs, device='cuda'), closest_sphere_idx]  # (num_envs, 3)
        closest_pcd_points = dynamic_pcd_downsample[torch.arange(self.num_envs, device='cuda'), closest_point_idx]  # (num_envs, 3)
        closest_sphere_radii = sphere_radii[closest_sphere_idx]  # (num_envs,)

        # Surface point calculation
        direction = closest_pcd_points - closest_sphere_centers
        direction = direction / torch.norm(direction, dim=1, keepdim=True)
        surface_points = closest_sphere_centers + closest_sphere_radii.unsqueeze(1) * direction  # (num_envs, 3)

        link_index_to_name = [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_hand",
        ]
        # closest_link_names = [link_index_to_name[i] for i in closest_link_indices.tolist()]


        # (num_envs, 9, 6, 7)
        jacobians = self.env.get_link_jacobians()
        batch_indices = torch.arange(self.num_envs, device=jacobians.device)
        # (num_envs, 6, 7)
        closest_link_jacobians = jacobians[batch_indices, closest_link_indices]


        # ------ compute torque ------
        # Define valid repulsive links
        repulsive_link_names = {"panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7", "panda_hand"}
        # (num_envs,) bool mask
        is_repulsive_link = torch.tensor(
            [link_index_to_name[i] in repulsive_link_names for i in closest_link_indices.tolist()],
            device=self.device
        )

        # Compute distance vectors and norms
        d_vec = surface_points - closest_pcd_points  # (num_envs, 3)
        d_norm = torch.norm(d_vec, dim=1)  # (num_envs,)

        # Conditions: non-zero, less than d0, and valid repulsive link
        d0 = 0.1 
        eta = 10.0
        valid_mask = (d_norm > 1e-5) & (d_norm < d0) & is_repulsive_link & has_dynamic_obstacles_flag # (num_envs,)

        # Normalize direction vectors safely
        direction = torch.zeros_like(d_vec)
        direction[valid_mask] = d_vec[valid_mask] / d_norm[valid_mask].unsqueeze(1)

        # Compute repulsive force: (num_envs, 3)
        f_repulse = torch.zeros_like(d_vec)
        f_repulse[valid_mask] = eta * (d0 - d_norm[valid_mask]).unsqueeze(1) * direction[valid_mask]

        # Construct spatial force: (num_envs, 6)
        F = torch.zeros((d_vec.shape[0], 6), device=jacobians.device)
        F[:, 0:3] = f_repulse

        # tau = J^T @ F
        # closest_link_jacobians: (num_envs, 6, 7), F: (num_envs, 6) → unsqueeze F to (num_envs, 6, 1)
        tau = torch.bmm(closest_link_jacobians.transpose(1, 2), F.unsqueeze(2)).squeeze(2)  # (num_envs, 7)

        env_obs_dict["goal_joint_pos"][valid_mask] = (joint_pos_tensor[valid_mask] + tau[valid_mask]).clone()
        env_obs_dict["goal_robot_pcd"] = self.env.get_robot_pcds(env_obs_dict["goal_joint_pos"])
        return env_obs_dict