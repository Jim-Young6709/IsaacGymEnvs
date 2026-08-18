import json
import os
import random
import time
from copy import copy
from pathlib import Path

import hydra
import numpy as np
import pybullet as p
from omegaconf import DictConfig

import neural_mp
from neural_mp.envs.franka_pybullet_table_env import FrankaTableTopEnv, run_planner_on_env
from neural_mp.envs.procedural_assets import *
from neural_mp.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def is_pc_primitive_collision(pc, support_vol):
    def get_sdf_value(point):
        sdf_value = support_vol.sdf(point)
        if isinstance(sdf_value, tuple):
            sdf_value = sdf_value[0]
        return sdf_value

    # Use np.apply_along_axis to get SDF values for all points
    sdf_values = np.apply_along_axis(get_sdf_value, 1, pc)

    # Calculate the percentage of points outside the volume
    num_points_outside = np.sum(sdf_values > 0)
    percent_points_outside = (num_points_outside / len(pc)) * 100

    # Find the maximum SDF value and the corresponding point
    max_sdf_value = np.max(sdf_values)
    max_sdf_point = pc[np.argmax(sdf_values)]
    return percent_points_outside, max_sdf_value, max_sdf_point


def adjust_mesh_position(position, max_sdf_point, max_sdf_value, support_vol):
    """
    Adjust the mesh position to shift it inside the support volume.

    :param position: Original position of the mesh.
    :param max_sdf_point: The point with the maximum SDF value.
    :param max_sdf_value: The maximum SDF value.
    :param support_vol: Support volume object with an sdf method.
    :return: New position for the mesh.
    """
    distance_from_center = max_sdf_point - support_vol.center
    distance_from_center[2] = 0
    shift_vector = -distance_from_center * max_sdf_value / np.linalg.norm(distance_from_center)
    new_position = position + shift_vector

    return new_position


def is_mesh_inside_volume(mesh, support_vol, scale, position, orientation):
    # Replace extension in mesh path with .npy
    point_cloud_path = mesh.replace(".obj", ".npy")

    # Load the .npy sampled point cloud
    points = np.load(point_cloud_path)

    # Apply the scale to the points
    scaled_points = points * scale

    # Apply the orientation to the points
    rotation_matrix = p.getMatrixFromQuaternion(orientation)
    rotation_matrix = np.array(rotation_matrix).reshape(3, 3)
    rotated_points = np.dot(scaled_points, rotation_matrix.T)

    # Translate the points to the new position
    transformed_points = rotated_points + position

    def get_sdf_value(point):
        sdf_value = support_vol.sdf(point)
        if isinstance(sdf_value, tuple):
            sdf_value = sdf_value[0]
        return sdf_value

    # Use np.apply_along_axis to get SDF values for all points
    sdf_values = np.apply_along_axis(get_sdf_value, 1, transformed_points)

    # Calculate the percentage of points outside the volume
    num_points_outside = np.sum(sdf_values > 0)
    percent_points_outside = (num_points_outside / len(points)) * 100

    # Find the maximum SDF value and the corresponding point
    max_sdf_value = np.max(sdf_values)
    max_sdf_point = transformed_points[np.argmax(sdf_values)]

    return percent_points_outside, max_sdf_value, max_sdf_point


class FrankaProceduralTableTop(FrankaTableTopEnv):
    """
    Env that sets up a table with procedural assets.
    """

    def setup_obstacles(self):
        """
        Set up the table using cuboids returned from the Table class and integrate into the PyBullet environment.
        """
        # add a table which will be a box
        table_dim_ranges = self.cfg.task.obstacle_kwargs.table_dim_ranges
        table_dims = [
            np.random.uniform(table_dim_ranges[i][0], table_dim_ranges[i][1]) for i in range(3)
        ]
        table_height = np.random.uniform(*self.cfg.task.obstacle_kwargs.table_height_range)
        # move forward by half the depth of the table
        table_offset = 0.1
        table_x = table_dims[0] / 2 + table_offset
        table_pos = [table_x, 0, table_height]

        table_id = self.add_box(position=table_pos, size=table_dims, orientation=[0, 0, 0, 1])
        num_shelves = np.random.randint(*self.cfg.task.obstacle_kwargs.num_shelves_range)
        num_open_boxes = np.random.randint(*self.cfg.task.obstacle_kwargs.num_open_boxes_range)
        num_cubbys = np.random.randint(*self.cfg.task.obstacle_kwargs.num_cubbys_range)
        num_cages = np.random.randint(
            *(
                self.cfg.task.obstacle_kwargs.num_cages_range
                if hasattr(self.cfg.task.obstacle_kwargs, "num_cages_range")
                else (0, 1)
            )
        )
        num_microwaves = np.random.randint(
            *(
                self.cfg.task.obstacle_kwargs.num_microwaves_range
                if hasattr(self.cfg.task.obstacle_kwargs, "num_microwaves_range")
                else (0, 1)
            )
        )
        num_dishwashers = np.random.randint(
            *(
                self.cfg.task.obstacle_kwargs.num_dishwashers_range
                if hasattr(self.cfg.task.obstacle_kwargs, "num_dishwashers_range")
                else (0, 1)
            )
        )
        num_wallcabinets = np.random.randint(
            *(
                self.cfg.task.obstacle_kwargs.num_wallcabinets_range
                if hasattr(self.cfg.task.obstacle_kwargs, "num_wallcabinets_range")
                else (0, 1)
            )
        )

        num_table_meshes = np.random.randint(
            *(
                self.cfg.task.obstacle_kwargs.mesh_env_kwargs.mesh_object_on_table
                if hasattr(self.cfg.task.obstacle_kwargs.mesh_env_kwargs, "mesh_object_on_table")
                else (2, 4)
            )
        )

        # Variables to store physical parameters for scene PCD computation
        cuboid_dims = [table_dims]
        cuboid_centers = [table_pos]
        cuboid_quats = [[0, 0, 0, 1]]
        self.table_info = (table_pos, table_dims, [0, 0, 0, 1])

        cylinder_radii = []
        cylinder_heights = []
        cylinder_centers = []
        cylinder_quats = []

        sphere_centers = []
        sphere_radii = []

        mesh_positions = []
        mesh_scales = []
        mesh_quaternions = []
        objaverse_ids = []
        mesh_ids = []

        # make a list with Shelf, OpenBox, and Cubby strings based on the number of each
        if self.assets_candidate is None:
            self.assets_candidate = (
                ["OpenBox"] * num_open_boxes
                + ["Shelf"] * num_shelves
                + ["Cubby"] * num_cubbys
                + ["Cage"] * num_cages
                + ["Microwave"] * num_microwaves
                + ["Dishwasher"] * num_dishwashers
                + ["WallCabinet"] * num_wallcabinets
                + ["Mesh"] * num_table_meshes
            )
            np.random.shuffle(self.assets_candidate)
            # Sort the list to push "Mesh" entries to the end
            self.assets_candidate.sort(
                key=lambda x: x == "Mesh"
            )  # fill the table with objects last

        skip_asset = False
        self.assets = []

        asset_cuboid_dims = []
        asset_cuboid_centers = []
        asset_cuboid_quats = []
        asset_cylinder_radii = []
        asset_cylinder_heights = []
        asset_cylinder_centers = []
        asset_cylinder_quats = []
        asset_sphere_centers = []
        asset_sphere_radii = []
        asset_mesh_position = []
        asset_mesh_scale = []
        asset_mesh_quaternion = []
        asset_obj_id = []
        asset_mesh_id = []

        obj_mapping_path = Path(__file__).parent.parent.parent / "meshes/type_mapping.json"
        obj_str2int = {}
        try:
            with open(obj_mapping_path, "r") as f:
                obj_str2int = json.load(f)
            if not obj_str2int:
                raise ValueError("Object type mapping is empty")
        except FileNotFoundError:
            print("Object mapping file not found.")
        if not obj_str2int:
            raise ValueError("Object type mapping is empty")
        for asset_type in self.assets_candidate:
            if asset_type == "Shelf":
                asset = Shelf(
                    **self.cfg.task.obstacle_kwargs.shelf_kwargs,
                    base_height=table_height + table_dims[2] / 2
                )
            elif asset_type == "OpenBox":
                asset = OpenBox(
                    **self.cfg.task.obstacle_kwargs.open_box_kwargs,
                    base_height=table_height + table_dims[2] / 2
                )
            elif asset_type == "Cubby":
                self.cfg.task.obstacle_kwargs.cubby_kwargs.cubby_bottom_range[0] = (
                    table_height + table_dims[2] / 2
                )
                self.cfg.task.obstacle_kwargs.cubby_kwargs.cubby_bottom_range[1] = 0
                asset = Cubby(**self.cfg.task.obstacle_kwargs.cubby_kwargs)
            elif asset_type == "Cage":
                asset = Cage(
                    **self.cfg.task.obstacle_kwargs.cage_kwargs,
                    base_height=table_height + table_dims[2] / 2
                )
            elif asset_type == "Microwave":
                asset = Microwave(
                    **self.cfg.task.obstacle_kwargs.microwave_kwargs,
                    base_height=table_height + table_dims[2] / 2
                )
            elif asset_type == "Dishwasher":
                asset = Dishwasher(
                    **self.cfg.task.obstacle_kwargs.dishwasher_kwargs,
                    base_height=table_height + table_dims[2] / 2
                )
            elif asset_type == "WallCabinet":
                asset = WallCabinet(
                    **self.cfg.task.obstacle_kwargs.wallcabinet_kwargs,
                    base_height=table_height + table_dims[2] / 2
                )
            elif asset_type == "Mesh":
                # Spawning one mesh directly on the table
                obstacle_list = copy(self.obstacles)
                mesh_dir = self.cfg.task.obstacle_kwargs.mesh_env_kwargs.mesh_dir
                object_list = self.cfg.task.obstacle_kwargs.mesh_env_kwargs.mesh_objects
                abs_path_dir = neural_mp.__file__[: -len("neural_mp/__init__.py")]
                if object_list == ["all"]:
                    object_list = [
                        obj
                        for obj in os.listdir(os.path.join(abs_path_dir, mesh_dir))
                        if obj != "type_mapping.json"
                    ]
                mesh_files = [
                    os.path.join(abs_path_dir, mesh_dir, obj, file)
                    for obj in object_list
                    for file in os.listdir(os.path.join(abs_path_dir, mesh_dir, obj))
                    if file.endswith(".obj")
                ]
                rpy_range = self.cfg.task.obstacle_kwargs.mesh_env_kwargs.rpy_range_deg
                mesh_sampler = lambda: random.choice(mesh_files)
                quat_sampler = (
                    lambda: np.random.uniform(low=rpy_range[0], high=rpy_range[1]) * np.pi / 180
                )

                scale_range = self.cfg.task.obstacle_kwargs.mesh_env_kwargs.scale_range
                size_sampler = lambda: np.random.uniform(
                    low=scale_range[0], high=scale_range[1], size=(1,)
                )
                sampled_size = size_sampler()

                spawn_position = [
                    np.random.uniform(table_offset, table_dims[0] + table_offset),
                    np.random.uniform(-table_dims[1] / 2, table_dims[1] / 2),
                    table_height + table_dims[2] / 2,
                ]

                sampled_mesh = mesh_sampler()
                mesh_orientation = np.array(p.getQuaternionFromEuler(quat_sampler()))

                use_simplified_mesh = getattr(
                    self.cfg.task.obstacle_kwargs.mesh_env_kwargs, "use_simplified_mesh", False
                )
                if not use_simplified_mesh:
                    asset_id = self.add_mesh(
                        spawn_position, sampled_size, mesh_orientation, mesh_path=sampled_mesh
                    )
                    asset_mesh_scale.append(sampled_size)
                    asset_mesh_position.append(spawn_position)
                    asset_mesh_quaternion.append(mesh_orientation)
                    asset_obj_id.append(int(obj_str2int[Path(sampled_mesh).parts[-2]]))
                    asset_mesh_id.append(int(Path(sampled_mesh).parts[-1].split(".")[-2]))
                else:
                    # multiply by certain ratio, so cube is not too big
                    sampled_size[0] = sampled_size[0] * 0.8
                    spawn_position[2] = spawn_position[2] + sampled_size[0] / 2
                    asset_id = self.add_box(spawn_position, [sampled_size[0]] * 3, mesh_orientation)
                    asset_cuboid_dims.append([sampled_size[0]] * 3)
                    asset_cuboid_centers.append(spawn_position)
                    asset_cuboid_quats.append(mesh_orientation)

                    cuboid_dims.extend(asset_cuboid_dims)
                    cuboid_centers.extend(asset_cuboid_centers)
                    cuboid_quats.extend(asset_cuboid_quats)

                # Resolve collision for the mesh
                skip_asset = not self.resolve_collision(
                    asset_id, obstacle_list, table_dims, table_pos
                )
                if skip_asset:
                    p.removeBody(asset_id)
                    self.obstacles.remove(asset_id)
                    continue

                final_pos = np.array(p.getBasePositionAndOrientation(asset_id)[0])
                out_of_bounds_x = (
                    final_pos[0] < table_offset or final_pos[0] > table_dims[0] + table_offset
                )
                out_of_bounds_y = (
                    final_pos[1] < -table_dims[1] / 2 or final_pos[1] > table_dims[1] / 2
                )
                if out_of_bounds_x or out_of_bounds_y:
                    p.removeBody(asset_id)
                    self.obstacles.remove(asset_id)
                    continue

                mesh_positions.extend(asset_mesh_position)
                mesh_scales.extend(asset_mesh_scale)
                mesh_quaternions.extend(asset_mesh_quaternion)
                objaverse_ids.extend(asset_obj_id)
                mesh_ids.extend(asset_mesh_id)
                continue

            obstacle_list = copy(self.obstacles)
            current_cuboids, current_cylinders, _ = asset.get_asset_shapes(global_space=False)

            asset_id = self.add_asset(
                current_cuboids,
                current_cylinders,
                asset.position,
                asset.orientation,
                rgbaColor=[1, 0, 0, 1],
            )
            skip_asset = not self.resolve_collision(asset_id, obstacle_list, table_dims, table_pos)
            asset.position = np.array(p.getBasePositionAndOrientation(asset_id)[0])
            asset.orientation = np.array(p.getBasePositionAndOrientation(asset_id)[1])

            if asset.__class__.__name__ == "Cubby":
                asset.update_params_post_pos_update()

            out_of_bounds_x = (
                asset.position[0] < table_offset or asset.position[0] > table_dims[0] + table_offset
            )
            out_of_bounds_y = (
                asset.position[1] < -table_dims[1] / 2 or asset.position[1] > table_dims[1] / 2
            )
            if skip_asset or out_of_bounds_x or out_of_bounds_y:
                p.removeBody(asset_id)
                self.obstacles.remove(asset_id)
                continue

            self.assets.append(asset)

            cubs, cyls, sphs = asset.get_asset_shapes(global_space=True)
            for pos, dim, quat in cubs:
                asset_cuboid_dims.append(dim)
                asset_cuboid_centers.append(pos)
                asset_cuboid_quats.append(quat)

            for pos, dim, quat in cyls:
                pos = np.array(pos)
                quat = np.array(quat)
                dim = np.array(dim)
                asset_cylinder_radii.append(dim[0])
                asset_cylinder_heights.append(dim[1])
                asset_cylinder_centers.append(pos)
                asset_cylinder_quats.append(quat)

            mesh_dir = self.cfg.task.obstacle_kwargs.mesh_env_kwargs.mesh_dir
            object_list = self.cfg.task.obstacle_kwargs.mesh_env_kwargs.mesh_objects
            abs_path_dir = neural_mp.__file__[: -len("neural_mp/__init__.py")]
            if object_list == ["all"]:
                object_list = [
                    obj
                    for obj in os.listdir(os.path.join(abs_path_dir, mesh_dir))
                    if obj != "type_mapping.json"
                ]
            mesh_files = [
                os.path.join(abs_path_dir, mesh_dir, obj, file)
                for obj in object_list
                for file in os.listdir(os.path.join(abs_path_dir, mesh_dir, obj))
                if file.endswith(".obj")
            ]
            rpy_range = self.cfg.task.obstacle_kwargs.mesh_env_kwargs.rpy_range_deg
            mesh_sampler = lambda: random.choice(mesh_files)
            quat_sampler = (
                lambda: np.random.uniform(low=rpy_range[0], high=rpy_range[0]) * np.pi / 180
            )

            region_of_interest = asset.compartments
            for region in region_of_interest:
                if self.cfg.task.obstacle_kwargs.debug_compartments:
                    self.add_visual_box(
                        region.center,
                        region.dims,
                        region._pose.so3.xyzw,
                        rgbaColor=[0, 1, 0, 0.5],
                    )
                if np.all(np.array(region.dims) < 0.2):
                    logger.warning("Support volume is too small")
                    return False
                    # Just skip asset and continue

            avg_mesh_object_per_support_volume_range = (
                self.cfg.task.obstacle_kwargs.mesh_env_kwargs.mesh_object_per_support_volume
            )
            avg_mesh_supp_vol = np.random.randint(
                avg_mesh_object_per_support_volume_range[0],
                avg_mesh_object_per_support_volume_range[1],
            )
            for _ in range(len(asset.compartments) * avg_mesh_supp_vol):
                support_vol = random.choice(asset.compartments)
                scale_range = self.cfg.task.obstacle_kwargs.mesh_env_kwargs.scale_range
                size_sampler = lambda: np.random.uniform(
                    low=scale_range[0], high=scale_range[1], size=(1,)
                )
                sampled_size = size_sampler()
                spawn_position = support_vol.sample_volume(1)[0]
                spawn_position[2] = support_vol.center[2] - support_vol.dims[2] / 2

                sampled_mesh = mesh_sampler()
                mesh_orientation = np.array(p.getQuaternionFromEuler(quat_sampler()))
                percent_outside, max_sdf_value, max_sdf_point = is_mesh_inside_volume(
                    sampled_mesh, support_vol, sampled_size, spawn_position, mesh_orientation
                )

                if percent_outside > 0.1:
                    spawn_position = adjust_mesh_position(
                        spawn_position, max_sdf_point, max_sdf_value, support_vol
                    )
                percent_outside, max_sdf_value, max_sdf_point = is_mesh_inside_volume(
                    sampled_mesh, support_vol, sampled_size, spawn_position, mesh_orientation
                )
                if percent_outside > 0.1:
                    continue

                if self.cfg.task.obstacle_kwargs.debug_spawn_locations:
                    self.add_visual_box(
                        spawn_position + np.array([0, 0, 0.1 / 2]),
                        np.array([0.1, 0.1, 0.1]),
                        rgbaColor=[0, 1, 0, 0.6],
                    )

                use_simplified_mesh = getattr(
                    self.cfg.task.obstacle_kwargs.mesh_env_kwargs, "use_simplified_mesh", False
                )
                if not use_simplified_mesh:
                    self.add_mesh(
                        spawn_position, sampled_size, mesh_orientation, mesh_path=sampled_mesh
                    )
                    asset_mesh_scale.append(sampled_size)
                    asset_mesh_position.append(spawn_position)
                    asset_mesh_quaternion.append(mesh_orientation)
                    asset_obj_id.append(int(obj_str2int[Path(sampled_mesh).parts[-2]]))
                    asset_mesh_id.append(int(Path(sampled_mesh).parts[-1].split(".")[-2]))
                else:
                    # multiply by certain ratio, so cube is not too big
                    sampled_size[0] = sampled_size[0] * 0.5
                    spawn_position[2] = spawn_position[2] + sampled_size[0] / 2
                    asset_id = self.add_box(spawn_position, [sampled_size[0]] * 3, mesh_orientation)
                    asset_cuboid_dims.append([sampled_size[0]] * 3)
                    asset_cuboid_centers.append(spawn_position)
                    asset_cuboid_quats.append(mesh_orientation)

                # add the asset's physical parameters to the scene PCD computation
            cuboid_dims.extend(asset_cuboid_dims)
            cuboid_centers.extend(asset_cuboid_centers)
            cuboid_quats.extend(asset_cuboid_quats)
            cylinder_radii.extend(asset_cylinder_radii)
            cylinder_heights.extend(asset_cylinder_heights)
            cylinder_centers.extend(asset_cylinder_centers)
            cylinder_quats.extend(asset_cylinder_quats)
            sphere_centers.extend(asset_sphere_centers)
            sphere_radii.extend(asset_sphere_radii)
            mesh_positions.extend(asset_mesh_position)
            mesh_scales.extend(asset_mesh_scale)
            mesh_quaternions.extend(asset_mesh_quaternion)
            objaverse_ids.extend(asset_obj_id)
            mesh_ids.extend(asset_mesh_id)

        if len(self.obstacles) <= 1:
            logger.warning("No assets loaded, skipping scene")
            return False
        # Compute scene PCD parameters
        self.compute_scene_pcd_params(
            max_num_objs_per_type=max(
                [len(cuboid_dims), len(cylinder_radii), len(sphere_radii), len(mesh_positions)]
            ),
            cuboid_dims=np.array(cuboid_dims).flatten().astype(np.float32),
            cuboid_centers=np.array(cuboid_centers).flatten().astype(np.float32),
            cuboid_quats=np.array(cuboid_quats).flatten().astype(np.float32),
            cylinder_radii=np.array(cylinder_radii).astype(np.float32),
            cylinder_heights=np.array(cylinder_heights).astype(np.float32),
            cylinder_centers=np.array(cylinder_centers).flatten().astype(np.float32),
            cylinder_quats=np.array(cylinder_quats).flatten().astype(np.float32),
            sphere_centers=np.array(sphere_centers).flatten().astype(np.float32),
            sphere_radii=np.array(sphere_radii).astype(np.float32),
            mesh_position=np.array(mesh_positions, dtype=np.float32).flatten(),
            mesh_scale=np.array(mesh_scales, dtype=np.float32).flatten(),
            mesh_quaternion=np.array(mesh_quaternions, dtype=np.float32).flatten(),
            obj_id=np.array(objaverse_ids, dtype=np.float32).flatten(),
            mesh_id=np.array(mesh_ids, dtype=np.float32).flatten(),
        )
        return True


@hydra.main(config_name="config", config_path="../configs/", version_base=None)
def main(cfg: DictConfig):
    env = eval(cfg.task.env_name)(cfg)

    # used for debugging start and goal config
    for i in range(100):
        input("show start...")
        env.set_robot_joint_state(env.start_config)
        input("show goal...")
        env.set_robot_joint_state(env.goal_angles)
        input("reset...")
        env.reset()
    frames = run_planner_on_env(env, cfg.num_plans, True, cfg.task.mp_kwargs, cfg.video_name)


if __name__ == "__main__":
    main()
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
