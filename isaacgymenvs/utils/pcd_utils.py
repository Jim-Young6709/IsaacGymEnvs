import cv2
import numpy as np
import torch
from geometrout.primitive import Cuboid, Cylinder, Sphere

from isaacgymenvs.utils.geometry import ObjaMesh, construct_mixed_point_cloud


def compute_scene_oracle_pcd(
    num_obstacle_points,
    cuboid_dims=[],
    cuboid_centers=[],
    cuboid_quats=[],
    cylinder_radii=[],
    cylinder_heights=[],
    cylinder_centers=[],
    cylinder_quats=[],
    sphere_centers=[],
    sphere_radii=[],
    mesh_position=[],
    mesh_scale=[],
    mesh_quaternion=[],
    obj_id=[],
    mesh_id=[],
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
            ObjaMesh(pos, scale, quat, obj_id, str(int(mesh_id)))
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

