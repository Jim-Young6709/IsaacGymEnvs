import numpy as np
import torch
import random
from typing import Sequence, Union
from geometrout.primitive import Cuboid, Cylinder, Sphere
from isaacgymenvs.utils.geometry import ObjaMesh


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
            ObjaMesh(pos, scale, quat, obj_id, str(int(mesh_id)), meshes_dir=meshes_dir)
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


def crop_local_pcd(pcd: torch.Tensor, local_range: torch.float, num_local_points: torch.int) -> torch.Tensor:
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
    dist = torch.norm(masked_pcds, dim=-1)
    mask = dist < local_range
    masked_pcds[~mask] = float("nan")

    # sort to get all the valid points
    is_valid = mask.int()
    sort_idx = torch.argsort(is_valid, dim=-1, descending=True)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    sorted_pcds = masked_pcds[batch_idx, sort_idx]  # (B, N, 3)
    pcd_local_nan_padding = sorted_pcds[:, :num_local_points]

    # replace nan values as 0s
    local_pcd_zero_padding = torch.nan_to_num(pcd_local_nan_padding, nan=0.0)
    return local_pcd_zero_padding
