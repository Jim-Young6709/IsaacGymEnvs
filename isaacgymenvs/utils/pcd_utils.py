import numpy as np
import torch
import random
import trimesh
from pathlib import Path
import open3d as o3d
from typing import Sequence, Union
from geometrout.primitive import Cuboid, Cylinder, Sphere
from isaacgymenvs.utils.geometry import ObjaMesh
from isaacgymenvs.utils.torch_urdf import TorchURDF


def construct_mixed_point_cloud(
    obstacles: Sequence[Union[Sphere, Cuboid, Cylinder, ObjaMesh]],
    num_points: int,
    return_point_list: bool = False,
    even: bool = False,
) -> np.ndarray:
    """
    Creates a random point cloud from a collection of obstacles. The points in
    the point cloud should be fairly(-ish) distributed amongst the obstacles based
    on their surface area.

    :param obstacles Sequence[Union[Sphere, Cuboid, Cylinder]]: The obstacles in the scene
    :param num_points int: The total number of points in the samples scene (not
                           the number of points per obstacle)
    :rtype np.ndarray: Has dim [N, 3] where N is num_points
    """
    point_set = []
    total_obstacles = len(obstacles)
    if total_obstacles == 0:
        return np.array([[]])

    # Allocate points based on obstacle surface area for even sampling
    surface_areas = np.array([o.surface_area for o in obstacles])
    total_area = np.sum(surface_areas)
    if even:
        proportions = np.ones(total_obstacles) / total_obstacles
    else:
        proportions = (surface_areas / total_area).tolist()

    indices = list(range(1, total_obstacles + 1))
    random.shuffle(indices)
    idx = 0

    for o, prop in zip(obstacles, proportions):
        sample_number = int(prop * num_points) + 500
        samples = o.sample_surface(sample_number)
        _points = indices[idx] * np.ones((sample_number, 4))
        _points[:, :3] = samples
        point_set.append(_points)
        idx += 1

    if return_point_list:
        lengths = torch.tensor([ps.shape[0] for ps in point_set], dtype=torch.float32)
        total = lengths.sum()
        ratios = lengths / total
        num_samples = (ratios * num_points).floor().to(torch.int32)
        num_samples[-1] += num_points - num_samples.sum()

        downsampled_point_set = []
        for point_subset, n in zip(point_set, num_samples):
            indices = torch.randperm(point_subset.shape[0])[:n]
            downsampled_point_set.append(point_subset[indices])

        assert (
            torch.tensor([ps.shape[0] for ps in downsampled_point_set], dtype=torch.float32).sum()
            == num_points
        )
        return downsampled_point_set

    points = np.concatenate(point_set, axis=0)

    # Downsample to the desired number of points
    return points[np.random.choice(points.shape[0], num_points, replace=False), :]


def compute_scene_oracle_pcd(
    num_obstacle_points: int,
    cuboid_dims: np.ndarray = [],
    cuboid_centers: np.ndarray = [],
    cuboid_quats: np.ndarray = [],
    cylinder_radii: np.ndarray = [],
    cylinder_heights: np.ndarray = [],
    cylinder_centers: np.ndarray = [],
    cylinder_quats: np.ndarray = [],
    sphere_centers: np.ndarray = [],
    sphere_radii: np.ndarray = [],
    mesh_position: np.ndarray = [],
    mesh_scale: np.ndarray = [],
    mesh_quaternion: np.ndarray = [],
    obj_id: np.ndarray = [],
    mesh_id: np.ndarray = [],
    meshes_dir: str = None,
    return_point_list: bool = False,
    even: bool = False,
):
    """
    Compute the oracle point cloud. Input quaternions are in xyzw format
    """

    def quaternions_xyzw_to_wxyz(quaternions_xyzw):
        quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
        return quaternions_wxyz

    cuboids = []
    cylinders = []
    spheres = []
    meshes = []

    if len(cuboid_dims) > 0:
        cuboids = [
            Cuboid(c, d, q)
            for c, d, q in zip(cuboid_centers, cuboid_dims, quaternions_xyzw_to_wxyz(cuboid_quats))
        ]
        cuboids = [c for c in cuboids if not c.is_zero_volume()]

    if len(cylinder_radii) > 0:
        cylinders = [
            Cylinder(c, r, h, q)
            for c, r, h, q in zip(
                cylinder_centers,
                cylinder_radii,
                cylinder_heights,
                quaternions_xyzw_to_wxyz(cylinder_quats),
            )
        ]
        cylinders = [c for c in cylinders if not c.is_zero_volume()]

    if len(sphere_centers) > 0:
        spheres = [Sphere(c, r) for c, r in zip(sphere_centers, sphere_radii)]
        spheres = [s for s in spheres if not s.is_zero_volume()]

    if len(mesh_position) > 0:
        meshes = [
            ObjaMesh(pos, scale, quat, obj_id, str(mesh_id), meshes_dir=meshes_dir)
            for pos, scale, quat, obj_id, mesh_id in zip(
                mesh_position,
                mesh_scale,
                quaternions_xyzw_to_wxyz(mesh_quaternion),
                obj_id,
                mesh_id,
            )
            if obj_id != 0.0 and scale > 0
        ]
        meshes = [m for m in meshes if not m.is_zero_volume()]

    obstacle_points = construct_mixed_point_cloud(
        cuboids + cylinders + spheres + meshes,
        num_obstacle_points,
        return_point_list=return_point_list,
        even=even,
    )
    obstacle_points = np.array(obstacle_points)[..., :3]
    return obstacle_points


def decompose_pcd_params_obs(pcd_params_obs):
    """
    Decompose the pcd params observation.
    """
    current_joint_angles, goal_joint_angles, scene_pcd_params = np.split(pcd_params_obs, [7, 14])
    goal_joint_angles, gripper_state = np.split(goal_joint_angles, [7])
    return (
        current_joint_angles,
        goal_joint_angles,
        gripper_state,
        *decompose_scene_pcd_params_obs(scene_pcd_params),
    )


def decompose_scene_pcd_params_obs(scene_pcd_params):
    """
    Decompose the pcd params observation.
    """
    M = int(scene_pcd_params[0])

    cuboid_params = scene_pcd_params[1 : 1 + 10 * M]
    cuboid_dims = cuboid_params[: 3 * M].reshape(-1, 3)
    cuboid_centers = cuboid_params[3 * M : 6 * M].reshape(-1, 3)
    cuboid_quats = cuboid_params[6 * M : 10 * M].reshape(-1, 4)

    cylinder_params = scene_pcd_params[1 + 10 * M : 1 + 10 * M + 9 * M]
    cylinder_radii = cylinder_params[: 1 * M].reshape(-1)
    cylinder_heights = cylinder_params[1 * M : 2 * M].reshape(-1)
    cylinder_centers = cylinder_params[2 * M : 5 * M].reshape(-1, 3)
    cylinder_quats = cylinder_params[5 * M : 9 * M].reshape(-1, 4)

    sphere_params = scene_pcd_params[1 + 10 * M + 9 * M : 1 + 10 * M + 9 * M + 4 * M]
    sphere_centers = sphere_params[: 3 * M].reshape(-1, 3)
    sphere_radii = sphere_params[3 * M : 4 * M].reshape(-1)

    mesh_params = scene_pcd_params[1 + 10 * M + 9 * M + 4 * M :]
    mesh_positions = mesh_params[: 3 * M].reshape(-1, 3)
    mesh_scales = mesh_params[3 * M : 4 * M].reshape(-1)
    mesh_quaternions = mesh_params[4 * M : 8 * M].reshape(-1, 4)
    obj_ids = mesh_params[8 * M : 9 * M].reshape(-1)
    mesh_ids = mesh_params[9 * M : 10 * M].reshape(-1)

    return (
        np.array(cuboid_dims).astype(np.float32),
        np.array(cuboid_centers).astype(np.float32),
        np.array(cuboid_quats).astype(np.float32),
        np.array(cylinder_radii).astype(np.float32),
        np.array(cylinder_heights).astype(np.float32),
        np.array(cylinder_centers).astype(np.float32),
        np.array(cylinder_quats).astype(np.float32),
        np.array(sphere_centers).astype(np.float32),
        np.array(sphere_radii).astype(np.float32),
        np.array(
            mesh_positions,
        ).astype(np.float32),
        np.array(
            mesh_scales,
        ).astype(np.float32),
        np.array(
            mesh_quaternions,
        ).astype(np.float32),
        np.array(
            obj_ids,
        ).astype(np.float32),
        np.array(
            mesh_ids,
        ).astype(np.float32),
    )


def quaternion_to_rotation_matrix(q):
    # q: (..., 4) -> (..., 3, 3)
    # (qx, qy, qz, qw)
    x, y, z, w = q.unbind(-1)

    B = q.shape[:-1]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = torch.stack([
        ww + xx - yy - zz, 2 * (xy - wz),       2 * (xz + wy),
        2 * (xy + wz),     ww - xx + yy - zz,   2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),       ww - xx - yy + zz
    ], dim=-1).reshape(*B, 3, 3)
    return rot


def transform_pcds_to_world(pcds_local, poses):
    # pcds_local: (N, D, P, 3)
    # poses: (N, D, 7) -> (x, y, z, qx, qy, qz, qw)
    trans = poses[..., :3]  # (..., 3)
    quat = poses[..., 3:]   # (..., 4)

    rot = quaternion_to_rotation_matrix(quat).to(pcds_local.dtype)  # (..., 3, 3)

    # Transform pointclouds
    pcds_local = pcds_local.unsqueeze(-1)  # (..., P, 3, 1)
    pcds_rotated = torch.matmul(rot.unsqueeze(-3), pcds_local).squeeze(-1)  # (..., P, 3)
    pcds_world = pcds_rotated + trans.unsqueeze(-2)  # (N, D, P, 3)
    return pcds_world


def shuffle_pcd(pcd: torch.Tensor) -> torch.Tensor:
    """
    Randomize ordering of points in a point cloud.
    
    Args:
        pcd: (B, N, 3) tensor
    Returns:
        shuffled: (B, N, 3) tensor with points randomly permuted
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # Random permutations (different per batch)
    idx = torch.argsort(torch.rand(B, N, device=device), dim=-1)  # (B, N)

    # Build batch index for gather
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)

    # Gather shuffled points
    return pcd[batch_idx, idx]  # (B, N, 3)


def downsample_pcd_batched(pcd: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    pcd: (B, N, 3) point cloud tensor
    num_points:   target number of points
    returns: (B, num_points, 3)
    """
    B, N, _ = pcd.shape
    M = num_points
    idx = torch.rand(B, N, device=pcd.device).argsort(dim=1)[:, :M]  # (B, M)
    return torch.gather(pcd, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def crop_local_pcd(
    pcd: torch.Tensor,
    local_range: torch.float,
    num_local_points: torch.int,
    is_cylindrical: bool = False,
    x_direction_cutoff: torch.float = -0.5,
):
    """
    Crop the point cloud to a local region around the origin with 0 padding.
    Args:
        pcd: (B, N, 3) tensor
        range: float, the radius of the local region
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # get local pcd
    masked_pcds = shuffle_pcd(pcd)
    if is_cylindrical:
        dist = torch.norm(masked_pcds[..., :2], dim=-1)
    else:
        dist = torch.norm(masked_pcds, dim=-1)
    mask = dist < local_range # nan < X always returns false, so if there are nan values in pcd input, it get automatically filtered out
    masked_pcds[~mask] = float("nan")

    if x_direction_cutoff is not None:
        x_dir_mask = masked_pcds[:, :, 0] > x_direction_cutoff
        masked_pcds[~x_dir_mask] = float("nan")

    # sort to get all the valid points
    is_valid = mask.int()
    sort_idx = torch.argsort(is_valid, dim=-1, descending=True)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    sorted_pcds = masked_pcds[batch_idx, sort_idx]  # (B, N, 3)
    pcd_local_nan_padding = sorted_pcds[:, :num_local_points]

    avg_num_valid_points = is_valid.sum() / B
    min_num_valid_points = is_valid.sum(dim=-1).min()

    crop_type = "cylindrical" if is_cylindrical else "spherical"
    logs = {
        f"local_{crop_type}_crop/avg_num_valid_points": avg_num_valid_points.item(),
        f"local_{crop_type}_crop/min_num_valid_points": min_num_valid_points.item(),
    }

    # replace nan values as 0s
    local_pcd_zero_padding = torch.nan_to_num(pcd_local_nan_padding, nan=0.0)

    return local_pcd_zero_padding, logs


def transform_pointcloud(pc, T):
    """pc: (B,N,3), T: (B,4,4) -> (B,N,3)"""
    B, N, _ = pc.shape
    homo = torch.cat([pc, torch.ones(B, N, 1, device=pc.device)], dim=-1)  # (B,N,4)
    out = torch.matmul(T, homo.transpose(1,2))  # (B,4,N)
    return out[:, :3].transpose(1,2)  # (B,N,3)


def visualize_pcd(points):
    """
    points: (N,3) torch or numpy
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.2, 0.6, 0.9])  # light blue
    o3d.visualization.draw_geometries([pcd])


class FrankaLeapSampler:
    def __init__(self, urdf_path, device, num_points=4096):
        self.device = device
        self.robot = TorchURDF.load(urdf_path, lazy_load_meshes=True, device=device)
        # Load meshes for all links with visuals
        self.links = [l for l in self.robot.links if len(l.visuals)]
        self.hand_links = [l for l in self.links if ("panda" not in l.name)]
        meshes = [
            trimesh.load(Path(urdf_path).parent / l.visuals[0].geometry.mesh.filename, force="mesh")
            for l in self.links
        ]
        areas = np.array([m.bounding_box_oriented.area for m in meshes])
        n_pts = np.round(num_points * areas / areas.sum()).astype(int)
        n_pts[0] += num_points - n_pts.sum()  # fix rounding
        self.points = {
            l.name: torch.as_tensor(
                trimesh.sample.sample_surface(meshes[i], n_pts[i])[0],
                device=device, dtype=torch.float32
            ).unsqueeze(0)  # (1,Ni,3)
            for i, l in enumerate(self.links)
        }

    def sample(self, joint_angles, joint_mapping_list=None, num_points=None, hand_only=False):
        """
        joint_angles: (B, 23) joint config
        joint_mapping_list: list[int], optional mapping to torch_urdf ordering
        hand_only: if True, only sample from hand_links
        returns: (B, num_points, 3) world-frame pointcloud
        """
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.unsqueeze(0)

        if joint_mapping_list is not None:
            joint_angles = joint_angles[:, joint_mapping_list]

        fk = self.robot.visual_geometry_fk_batch(joint_angles)  # dict[geom] -> (B,4,4)
        pcs = []

        link_set = self.hand_links if hand_only else self.links
        for l in link_set:
            T = fk[l.visuals[0].geometry]  # (B,4,4)
            pc = self.points[l.name].repeat(joint_angles.shape[0], 1, 1)  # (B,Ni,3)
            pcs.append(transform_pointcloud(pc, T))

        pc = torch.cat(pcs, dim=1)  # (B, totalN, 3)
        if num_points is None:
            return pc
        idx = np.random.choice(pc.shape[1], num_points, replace=False)
        return pc[:, idx, :]


class GlorbotSampler:
    def __init__(self, urdf_path, device, num_points=4096):
        self.device = device
        self.tidybot_links = [
            'front_panel', 'back_panel', 'left_panel', 'right_panel', 'top_panel',
            'tidybot2_base_link', 'franka_control_box', #'lidar', 'imu', 
            # 'front_right_steer_link', 'front_right_drive_link', 'front_left_steer_link', 'front_left_drive_link', 
            # 'back_left_steer_link', 'back_left_drive_link', 'back_right_steer_link', 'back_right_drive_link',
        ]
        self.franka_links = [
            'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 
        ]
        self.leap_hand_links = [
            'palm_lower', 'mcp_1', 'pip_1', 'dip_1', 'fingertip_1', 
            'mcp_2', 'pip_2', 'dip_2', 'fingertip_2', 'mcp_3', 
            'pip_3', 'dip_3', 'fingertip_3', 
            'thumb_temp_base', 'pip_4', 'dip_4', 'fingertip_4', 
        ]
        self.arx_links = [
            'x5_base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'x5_camera_link',
        ]

        # Allowed link names
        self.allowed_link_names = set(
            self.tidybot_links
            + self.franka_links
            + self.leap_hand_links
            + self.arx_links
        )

        # Load URDF with torch-aware kinematics
        self.robot = TorchURDF.load(urdf_path, lazy_load_meshes=True, device=device)

        mesh_links = []
        mesh_geoms = []  # list[trimesh.Trimesh]

        for l in self.robot.links:
            # Only consider links that are in our allowed sets
            if l.name not in self.allowed_link_names:
                continue

            if not l.visuals:
                continue

            # We'll only use the first visual per link (same assumption as before)
            geom = l.visuals[0].geometry

            # ---- Case 1: mesh geometry ----
            mesh = getattr(geom, "mesh", None)
            if mesh is not None and getattr(mesh, "filename", None) is not None:
                mesh_path = Path(urdf_path).parent / mesh.filename
                tm = trimesh.load(mesh_path, force="mesh")
                mesh_links.append(l)
                mesh_geoms.append(tm)
                continue

            # ---- Case 2: box primitive ----
            box = getattr(geom, "box", None)
            # URDF: <box size="sx sy sz">  (full extents)
            if box is not None and getattr(box, "size", None) is not None:
                size = np.asarray(box.size, dtype=float)  # (3,)
                tm = trimesh.creation.box(extents=size)
                mesh_links.append(l)
                mesh_geoms.append(tm)
                continue

            # ---- Case 3: cylinder primitive ----
            cyl = getattr(geom, "cylinder", None)
            # URDF: <cylinder radius="r" length="L">
            if (
                cyl is not None
                and getattr(cyl, "radius", None) is not None
                and getattr(cyl, "length", None) is not None
            ):
                radius = float(cyl.radius)
                length = float(cyl.length)
                tm = trimesh.creation.cylinder(radius=radius, height=length)
                mesh_links.append(l)
                mesh_geoms.append(tm)
                continue

            # ---- Case 4: sphere primitive ----
            sph = getattr(geom, "sphere", None)
            # URDF: <sphere radius="r">
            if sph is not None and getattr(sph, "radius", None) is not None:
                radius = float(sph.radius)
                tm = trimesh.creation.icosphere(radius=radius)
                mesh_links.append(l)
                mesh_geoms.append(tm)
                continue

            # Anything else gets skipped

        if len(mesh_links) == 0:
            raise RuntimeError(
                "No mesh/box/cylinder/sphere visuals found in URDF for GlorbotSampler "
                "among the allowed links."
            )

        # Store only links we actually built geometry for
        self.links = mesh_links
        self.hand_links = [l for l in self.links if (l.name in self.leap_hand_links)]

        # Compute areas and allocate point counts
        areas = np.array([m.area for m in mesh_geoms])
        areas_sum = areas.sum()
        if areas_sum <= 0:
            raise RuntimeError("Total mesh area is zero; cannot sample robot surface.")
        n_pts = np.round(num_points * areas / areas_sum).astype(int)
        # Fix rounding so total points == num_points
        n_pts[0] += num_points - n_pts.sum()

        # Sample point clouds in geometry frame for each link
        self.points = {}
        for link, mesh_obj, n in zip(self.links, mesh_geoms, n_pts):
            if n <= 0:
                continue
            pts = trimesh.sample.sample_surface(mesh_obj, int(n))[0]
            self.points[link.name] = torch.as_tensor(
                pts, device=device, dtype=torch.float32
            ).unsqueeze(0)  # (1, Ni, 3)

    def sample(self, joint_angles, joint_mapping_list=None, num_points=None, hand_only=False):
        """
        joint_angles: (B, 32) joint config
        joint_mapping_list: list[int], optional mapping to torch_urdf ordering
        returns: (B, num_points, 3) world-frame pointcloud
        """
        if joint_angles.ndim == 1:
            joint_angles = joint_angles.unsqueeze(0)

        if joint_mapping_list is not None:
            joint_angles = joint_angles[:, joint_mapping_list]

        # dict[Geometry] -> (B,4,4), geometry frame in world
        fk = self.robot.visual_geometry_fk_batch(joint_angles)

        pcs = []
        B = joint_angles.shape[0]

        link_set = self.hand_links if hand_only else self.links
        for l in link_set:
            # We only used the first visual to build the mesh/primitive
            geom = l.visuals[0].geometry
            T = fk[geom]  # (B, 4, 4)

            pc = self.points[l.name].repeat(B, 1, 1)  # (B, Ni, 3)
            pcs.append(transform_pointcloud(pc, T))

        pc = torch.cat(pcs, dim=1)  # (B, totalN, 3)

        if num_points is None:
            return pc

        idx = np.random.choice(pc.shape[1], num_points, replace=False)
        return pc[:, idx, :]
