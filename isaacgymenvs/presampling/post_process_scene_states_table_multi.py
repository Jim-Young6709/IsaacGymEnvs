import argparse
import os
import shutil

import h5py
import hydra
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from isaacgymenvs.utils.pcd_utils import decompose_scene_pcd_params_obs

STATE_PREFIX_DIMS = 15
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
COMPARTMENT_CLEARANCE = 1e-3
NUMERICAL_TOL = 1e-6
REGULAR_COMPARTMENT_MIN_XY_DIM = np.array([0.25, 0.25], dtype=np.float32)
SHELF_COMPARTMENT_MIN_XY_DIM = np.array([0.2, 0.25], dtype=np.float32)


def load_task_env_cfg(task_name: str):
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cfg"))
    hydra_state = GlobalHydra.instance()
    if hydra_state.is_initialized():
        hydra_state.clear()
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(config_name=None, overrides=[f"+task={task_name}"])
    return cfg.task.env


def yaw_quat(rng) -> np.ndarray:
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    return np.array([0.0, 0.0, np.sin(theta * 0.5), np.cos(theta * 0.5)], dtype=np.float32)


def empty_cuboids():
    return (
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
    )


def append_cuboids_to_scene(
    cuboid_dims,
    cuboid_centers,
    cuboid_quats,
    cylinder_radii,
    cylinder_heights,
    cylinder_centers,
    cylinder_quats,
    sphere_centers,
    sphere_radii,
    mesh_positions,
    mesh_scales,
    mesh_quats,
    obj_ids,
    mesh_ids,
    extra_dims,
    extra_centers,
    extra_quats,
):
    added = int(extra_dims.shape[0])
    if added == 0:
        return (
            cuboid_dims,
            cuboid_centers,
            cuboid_quats,
            cylinder_radii,
            cylinder_heights,
            cylinder_centers,
            cylinder_quats,
            sphere_centers,
            sphere_radii,
            mesh_positions,
            mesh_scales,
            mesh_quats,
            obj_ids,
            mesh_ids,
        )

    zero_vec = np.zeros((added,), dtype=np.float32)
    zero_xyz = np.zeros((added, 3), dtype=np.float32)
    identity_quats = np.broadcast_to(IDENTITY_QUAT, (added, 4))
    return (
        np.concatenate((cuboid_dims, extra_dims), axis=0),
        np.concatenate((cuboid_centers, extra_centers), axis=0),
        np.concatenate((cuboid_quats, extra_quats), axis=0),
        np.concatenate((cylinder_radii, zero_vec)),
        np.concatenate((cylinder_heights, zero_vec)),
        np.concatenate((cylinder_centers, zero_xyz), axis=0),
        np.concatenate((cylinder_quats, identity_quats), axis=0),
        np.concatenate((sphere_centers, zero_xyz), axis=0),
        np.concatenate((sphere_radii, zero_vec)),
        np.concatenate((mesh_positions, zero_xyz), axis=0),
        np.concatenate((mesh_scales, zero_vec)),
        np.concatenate((mesh_quats, identity_quats), axis=0),
        np.concatenate((obj_ids, zero_vec)),
        np.concatenate((mesh_ids, zero_vec)),
    )


def create_distractor_cuboids(table_pos: np.ndarray, table_size: np.ndarray, params, seed: int):
    rng = np.random.default_rng(seed)
    table_extend = float(params["table_extend"])
    max_z = float(params["free_space_distractor_max_height"])

    table_x_min = table_pos[0] - table_size[0] * 0.5
    table_x_max = table_pos[0] + table_size[0] * 0.5
    table_y_min = table_pos[1] - table_size[1] * 0.5
    table_y_max = table_pos[1] + table_size[1] * 0.5
    table_z_min = table_pos[2] - table_size[2] * 0.5

    regions = [
        [[table_x_min, table_y_min - table_extend, 0.0], [table_x_max + table_extend, table_y_min, max_z]],
        [[table_x_min, table_y_max, 0.0], [table_x_max + table_extend, table_y_max + table_extend, max_z]],
        [[table_x_max, table_y_min, 0.0], [table_x_max + table_extend, table_y_max, max_z]],
        [[table_x_min, table_y_min, 0.0], [table_x_min + table_extend, table_y_max, table_z_min]],
    ]

    dims = []
    centers = []
    quats = []
    size_range = np.asarray(params["cuboid_size_range"], dtype=np.float32)
    count_range = params["num_distractors_per_region_range"]
    skip_prob = float(params["skip_prob"])
    full_prob = float(params["full_prob"])
    for region in regions:
        pos_range = np.asarray(region, dtype=np.float32)
        if pos_range[1, 2] <= 0:
            continue

        rand01 = float(rng.uniform(0.0, 1.0))
        if rand01 < skip_prob:
            continue
        if rand01 < skip_prob + full_prob:
            cuboid_dim = pos_range[1] - pos_range[0]
            if np.all(cuboid_dim > 0):
                dims.append(cuboid_dim.astype(np.float32))
                centers.append(((pos_range[0] + pos_range[1]) * 0.5).astype(np.float32))
                quats.append(IDENTITY_QUAT)
            continue

        count = int(rng.integers(count_range[0], count_range[1] + 1))
        for _ in range(count):
            cuboid_dim = rng.uniform(size_range[0], size_range[1]).astype(np.float32)
            cuboid_dim[2] = min(cuboid_dim[2], float(pos_range[1, 2] - pos_range[0, 2]))
            if np.any(cuboid_dim <= 0):
                continue
            center_min = pos_range[0] + cuboid_dim * 0.5
            center_max = pos_range[1] - cuboid_dim * 0.5
            if np.any(center_max < center_min):
                continue
            dims.append(cuboid_dim)
            centers.append(rng.uniform(center_min, center_max).astype(np.float32))
            quats.append(IDENTITY_QUAT)

    if not dims:
        return empty_cuboids()
    return (
        np.asarray(dims, dtype=np.float32).reshape(-1, 3),
        np.asarray(centers, dtype=np.float32).reshape(-1, 3),
        np.asarray(quats, dtype=np.float32).reshape(-1, 4),
    )


def create_mobile_obstacle_cuboids(compartment_dims, compartment_center, compartment_quat, params, rng):
    cuboid_cfg = params["cuboids"]
    count_range = cuboid_cfg["num_range"]
    if int(count_range[1]) <= 0:
        return empty_cuboids()
    count = int(rng.integers(int(count_range[0]), int(count_range[1]) + 1))

    size_range = np.asarray(cuboid_cfg["size_range"], dtype=np.float32)
    x_offset_range = params["x_offset_range"]
    y_offset_range = params["y_offset_range"]
    compartment_rot = quaternion_to_matrix_xyzw(compartment_quat)

    dims = []
    centers = []
    quats = []
    for _ in range(count):
        cuboid_dim = rng.uniform(size_range[0], size_range[1]).astype(np.float32)
        local_center = np.array(
            [
                -0.5 * compartment_dims[0] + rng.uniform(*x_offset_range),
                (0.5 * compartment_dims[1] + rng.uniform(*y_offset_range)) * rng.choice([-1.0, 1.0]),
                cuboid_dim[2] * 0.5,
            ],
            dtype=np.float32,
        )
        world_center = compartment_center + compartment_rot @ local_center
        world_center[2] = local_center[2]
        dims.append(cuboid_dim)
        centers.append(world_center.astype(np.float32))
        quats.append(yaw_quat(rng))

    return (
        np.asarray(dims, dtype=np.float32).reshape(-1, 3),
        np.asarray(centers, dtype=np.float32).reshape(-1, 3),
        np.asarray(quats, dtype=np.float32).reshape(-1, 4),
    )


def quaternion_to_matrix_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= NUMERICAL_TOL:
        return np.eye(3, dtype=np.float32)

    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def cuboid_xy_half_extent(cuboid_dim: np.ndarray, cuboid_quat: np.ndarray, frame_rot: np.ndarray) -> np.ndarray:
    half_dim = np.asarray(cuboid_dim, dtype=np.float32) * 0.5
    cuboid_rot = quaternion_to_matrix_xyzw(cuboid_quat)
    cuboid_rot_in_frame = frame_rot.T @ cuboid_rot
    return np.abs(cuboid_rot_in_frame[:2, :]) @ half_dim


def cuboid_intersects_compartment(
    center_local: np.ndarray,
    cuboid_dim: np.ndarray,
    cuboid_quat: np.ndarray,
    compartment_dims: np.ndarray,
    compartment_rot: np.ndarray,
) -> bool:
    cuboid_half_dim = np.asarray(cuboid_dim, dtype=np.float32) * 0.5
    compartment_half_dim = np.asarray(compartment_dims, dtype=np.float32) * 0.5

    # All boxes are assumed upright, so z overlap is just interval overlap.
    if abs(center_local[2]) > compartment_half_dim[2] + cuboid_half_dim[2] - NUMERICAL_TOL:
        return False

    cuboid_rot = quaternion_to_matrix_xyzw(cuboid_quat)
    cuboid_rot_local = compartment_rot.T @ cuboid_rot
    cuboid_axes_xy = cuboid_rot_local[:2, :2]
    center_xy = center_local[:2]

    axes = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        cuboid_axes_xy[:, 0],
        cuboid_axes_xy[:, 1],
    ]

    for axis in axes:
        if np.dot(axis, axis) <= NUMERICAL_TOL:
            continue
        axis = axis / np.linalg.norm(axis)
        compartment_radius = np.dot(compartment_half_dim[:2], np.abs(axis))
        cuboid_radius = np.dot(cuboid_half_dim[:2], np.abs(cuboid_axes_xy.T @ axis))
        center_distance = abs(np.dot(center_xy, axis))
        if center_distance > compartment_radius + cuboid_radius:
            return False

    return True


def sample_shrunk_boundary(rng, low: float, high: float):
    if high - low <= NUMERICAL_TOL:
        return None
    return float(rng.uniform(low, high))


def shrink_compartment_boundaries(demo_group, rng, shrink_prob: float, shelf: bool = False):
    if shrink_prob <= 0.0:
        return 0, []

    shrink_prob = min(float(shrink_prob), 1.0)
    compartment_dataset = demo_group["compartment_states"]
    compartment_states = np.asarray(compartment_dataset[:], dtype=np.float32).reshape(-1, 10)
    min_xy_dim = SHELF_COMPARTMENT_MIN_XY_DIM if shelf else REGULAR_COMPARTMENT_MIN_XY_DIM

    compartment_dims = compartment_states[0, :3].copy()
    compartment_center = compartment_states[0, 3:6].copy()
    compartment_quat = compartment_states[0, 6:10]
    compartment_rot = quaternion_to_matrix_xyzw(compartment_quat)

    x_min = -0.5 * float(compartment_dims[0])
    x_max = 0.5 * float(compartment_dims[0])
    y_min = -0.5 * float(compartment_dims[1])
    y_max = 0.5 * float(compartment_dims[1])
    shrunk_boundaries = []

    selected_boundaries = {
        "front": (not shelf) and rng.uniform(0.0, 1.0) < shrink_prob,
        "back": rng.uniform(0.0, 1.0) < shrink_prob,
        "right": rng.uniform(0.0, 1.0) < shrink_prob,
        "left": rng.uniform(0.0, 1.0) < shrink_prob,
    }

    changed = False
    if selected_boundaries["front"]:
        new_x_min = sample_shrunk_boundary(rng, x_min, x_max - float(min_xy_dim[0]))
        if new_x_min is not None:
            shrunk_boundaries.append("front")
            x_min = new_x_min
            changed = True
    if selected_boundaries["back"]:
        new_x_max = sample_shrunk_boundary(rng, x_min + float(min_xy_dim[0]), x_max)
        if new_x_max is not None:
            shrunk_boundaries.append("back")
            x_max = new_x_max
            changed = True
    if selected_boundaries["right"]:
        new_y_min = sample_shrunk_boundary(rng, y_min, y_max - float(min_xy_dim[1]))
        if new_y_min is not None:
            shrunk_boundaries.append("right")
            y_min = new_y_min
            changed = True
    if selected_boundaries["left"]:
        new_y_max = sample_shrunk_boundary(rng, y_min + float(min_xy_dim[1]), y_max)
        if new_y_max is not None:
            shrunk_boundaries.append("left")
            y_max = new_y_max
            changed = True

    if not changed:
        return 0, []

    compartment_dims[:2] = np.array([x_max - x_min, y_max - y_min], dtype=np.float32)
    center_offset_local = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5, 0.0], dtype=np.float32)
    moved_center = compartment_center + compartment_rot @ center_offset_local
    compartment_states[0, :2] = compartment_dims[:2]
    compartment_states[0, 3:5] = moved_center[:2]
    compartment_dataset[...] = compartment_states.reshape(compartment_dataset.shape)

    new_bounds = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }
    shrink_events = [
        {
            "boundary": boundary,
            "new_bounds": new_bounds,
            "center": moved_center,
            "dims": compartment_dims.copy(),
            "quat": compartment_quat.copy(),
        }
        for boundary in shrunk_boundaries
    ]

    return 1, shrink_events


def create_add_on_cuboids_from_shrink_events(shrink_events, params, rng):
    if not shrink_events:
        return empty_cuboids()

    cuboid_cfg = params["cuboids"]
    count_range = cuboid_cfg["num_per_boundary_range"]
    if int(count_range[1]) <= 0:
        return empty_cuboids()

    size_xy = np.asarray(cuboid_cfg["size_xy"], dtype=np.float32)
    size_z_min = float(cuboid_cfg["size_z_min"])

    dims = []
    centers = []
    quats = []
    for event in shrink_events:
        boundary = event["boundary"]
        bounds = event["new_bounds"]
        count = int(rng.integers(int(count_range[0]), int(count_range[1]) + 1))

        for _ in range(count):
            cuboid_dim = np.empty((3,), dtype=np.float32)
            cuboid_dim[:2] = rng.uniform(size_xy[0], size_xy[1]).astype(np.float32)
            cuboid_dim[2] = float(rng.uniform(size_z_min, max(size_z_min, float(event["dims"][2]))))

            x_low = bounds["x_min"] + cuboid_dim[0] * 0.5
            x_high = bounds["x_max"] - cuboid_dim[0] * 0.5
            y_low = bounds["y_min"] + cuboid_dim[1] * 0.5
            y_high = bounds["y_max"] - cuboid_dim[1] * 0.5
            if x_high < x_low or y_high < y_low:
                continue

            center_local = np.array([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high), 0.0], dtype=np.float32)
            boundary_offset = COMPARTMENT_CLEARANCE
            if boundary == "front":
                center_local[0] = bounds["x_min"] + boundary_offset
            elif boundary == "back":
                center_local[0] = bounds["x_max"] - boundary_offset
            elif boundary == "right":
                center_local[1] = bounds["y_min"] + boundary_offset
            else:
                center_local[1] = bounds["y_max"] - boundary_offset

            compartment_rot = quaternion_to_matrix_xyzw(event["quat"])
            world_center = event["center"] + compartment_rot @ center_local
            compartment_bottom_height = float(event["center"][2] - event["dims"][2] * 0.5)
            world_center[2] = compartment_bottom_height + cuboid_dim[2] * 0.5

            dims.append(cuboid_dim)
            centers.append(world_center.astype(np.float32))
            quats.append(yaw_quat(rng))

    if not dims:
        return empty_cuboids()
    return (
        np.asarray(dims, dtype=np.float32).reshape(-1, 3),
        np.asarray(centers, dtype=np.float32).reshape(-1, 3),
        np.asarray(quats, dtype=np.float32).reshape(-1, 4),
    )


def move_cuboids_outside_first_compartment(
    demo_group,
    cuboid_dims,
    cuboid_centers,
    cuboid_quats,
    shelf: bool = False,
) -> int:
    compartment_states = np.asarray(demo_group["compartment_states"][:], dtype=np.float32).reshape(-1, 10)
    first_compartment = compartment_states[0]
    compartment_dims = first_compartment[:3]
    compartment_center = first_compartment[3:6]
    compartment_quat = first_compartment[6:10]
    compartment_rot = quaternion_to_matrix_xyzw(compartment_quat)

    x_min = -0.5 * compartment_dims[0]
    x_max = 0.5 * compartment_dims[0]
    y_min = -0.5 * compartment_dims[1]
    y_max = 0.5 * compartment_dims[1]

    moved = 0
    # The first cuboid is the table in this scene encoding; only later cuboid
    # entries are movable cuboid objects that should be pushed out of the box.
    for idx in range(1, int(cuboid_centers.shape[0])):
        if np.any(cuboid_dims[idx] <= NUMERICAL_TOL):
            continue
        center = cuboid_centers[idx]
        center_local = compartment_rot.T @ (center - compartment_center)
        # Move cuboids if any part of their upright volume intersects the
        # compartment volume, not only when the cuboid center is inside.
        if not cuboid_intersects_compartment(
            center_local,
            cuboid_dims[idx],
            cuboid_quats[idx],
            compartment_dims,
            compartment_rot,
        ):
            continue

        # Compare distance in the compartment's local xy frame, so rotated
        # compartments use their own rectangular boundaries instead of world axes.
        boundary_distances = {
            "front": abs(center_local[0] - x_min),
            "back": abs(x_max - center_local[0]),
            "right": abs(center_local[1] - y_min),
            "left": abs(y_max - center_local[1]),
        }
        if shelf:
            del boundary_distances["front"]
        closest_boundary = min(boundary_distances, key=boundary_distances.get)
        half_extent_xy = cuboid_xy_half_extent(cuboid_dims[idx], cuboid_quats[idx], compartment_rot)

        # Move to the closest local wall by exactly the cuboid's projected
        # half-extent plus clearance, keeping it as close as possible while
        # placing the whole cuboid outside the compartment.
        if closest_boundary == "front":
            center_local[0] = x_min - half_extent_xy[0] - COMPARTMENT_CLEARANCE
        elif closest_boundary == "back":
            center_local[0] = x_max + half_extent_xy[0] + COMPARTMENT_CLEARANCE
        elif closest_boundary == "right":
            center_local[1] = y_min - half_extent_xy[1] - COMPARTMENT_CLEARANCE
        else:
            center_local[1] = y_max + half_extent_xy[1] + COMPARTMENT_CLEARANCE

        moved_center = compartment_center + compartment_rot @ center_local
        cuboid_centers[idx, :2] = moved_center[:2]
        moved += 1

    return moved


def build_scene_pcd_params(
    cuboid_dims,
    cuboid_centers,
    cuboid_quats,
    cylinder_radii,
    cylinder_heights,
    cylinder_centers,
    cylinder_quats,
    sphere_centers,
    sphere_radii,
    mesh_positions,
    mesh_scales,
    mesh_quats,
    obj_ids,
    mesh_ids,
) -> np.ndarray:
    m = int(cuboid_dims.shape[0])
    updated_scene = np.empty((1 + 33 * m,), dtype=np.float32)
    updated_scene[0] = m
    offset = 1
    for block in (
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quats,
        obj_ids,
        mesh_ids,
    ):
        flat = np.asarray(block, dtype=np.float32).reshape(-1)
        next_offset = offset + flat.size
        updated_scene[offset:next_offset] = flat
        offset = next_offset

    return updated_scene


def process_demo(
    demo_group,
    demo_idx: int,
    base_seed: int,
    env_cfg,
    shelf: bool = False,
    compartment_shrink_prob: float = 0.3,
):
    states_dataset = demo_group["states"]
    states = np.asarray(states_dataset[:], dtype=np.float32)
    scene_pcd_params = states[0, 15:]
    (
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quats,
        obj_ids,
        mesh_ids,
    ) = decompose_scene_pcd_params_obs(scene_pcd_params)

    table_pos = cuboid_centers[0].astype(np.float32)
    table_size = cuboid_dims[0].astype(np.float32)
    scene_cfg = env_cfg.scene
    distractor_params = env_cfg.distractor_settings.params

    extra_dims, extra_centers, extra_quats = create_distractor_cuboids(
        table_pos, table_size, distractor_params, base_seed + demo_idx
    )
    distractor_added = int(extra_dims.shape[0])
    (
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quats,
        obj_ids,
        mesh_ids,
    ) = append_cuboids_to_scene(
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quats,
        obj_ids,
        mesh_ids,
        extra_dims,
        extra_centers,
        extra_quats,
    )

    rng = np.random.default_rng(base_seed + demo_idx + 1000003)  # separate stream from distractor sampling
    shrunk, shrink_events = shrink_compartment_boundaries(
        demo_group,
        rng,
        compartment_shrink_prob,
        shelf=shelf,
    )

    add_on_dims, add_on_centers, add_on_quats = create_add_on_cuboids_from_shrink_events(
        shrink_events,
        scene_cfg.add_on_obstacles,
        rng,
    )
    add_on_added = int(add_on_dims.shape[0])
    (
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quats,
        obj_ids,
        mesh_ids,
    ) = append_cuboids_to_scene(
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quats,
        obj_ids,
        mesh_ids,
        add_on_dims,
        add_on_centers,
        add_on_quats,
    )

    first_compartment = np.asarray(demo_group["compartment_states"][:], dtype=np.float32).reshape(-1, 10)[0]
    mobile_dims, mobile_centers, mobile_quats = create_mobile_obstacle_cuboids(
        first_compartment[:3],
        first_compartment[3:6],
        first_compartment[6:10],
        scene_cfg.mobile_obstacles,
        rng,
    )
    mobile_added = int(mobile_dims.shape[0])
    (
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quats,
        obj_ids,
        mesh_ids,
    ) = append_cuboids_to_scene(
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quats,
        obj_ids,
        mesh_ids,
        mobile_dims,
        mobile_centers,
        mobile_quats,
    )

    moved = move_cuboids_outside_first_compartment(
        demo_group,
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        shelf=shelf,
    )
    added = distractor_added + add_on_added + mobile_added
    if added == 0 and moved == 0 and shrunk == 0:
        return 0, 0, 0, 0, 0

    updated_scene = build_scene_pcd_params(
        cuboid_dims,
        cuboid_centers,
        cuboid_quats,
        cylinder_radii,
        cylinder_heights,
        cylinder_centers,
        cylinder_quats,
        sphere_centers,
        sphere_radii,
        mesh_positions,
        mesh_scales,
        mesh_quats,
        obj_ids,
        mesh_ids,
    )

    updated_states = np.empty((states.shape[0], STATE_PREFIX_DIMS + updated_scene.shape[0]), dtype=np.float32)
    updated_states[:, :STATE_PREFIX_DIMS] = states[:, :STATE_PREFIX_DIMS]
    updated_states[:, STATE_PREFIX_DIMS:] = updated_scene

    # Write the post-processed cuboid states back to the original HDF5 key. If
    # distractors changed the state width, recreate the dataset; otherwise write
    # in place so unchanged HDF5 storage details and attrs remain intact.
    if updated_states.shape == states_dataset.shape:
        states_dataset[...] = updated_states
    else:
        attrs = dict(states_dataset.attrs.items())
        del demo_group["states"]
        dataset = demo_group.create_dataset("states", data=updated_states)
        for key, value in attrs.items():
            dataset.attrs[key] = value

    return distractor_added, add_on_added, mobile_added, moved, shrunk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_name", default="DexMobileExpPostProcessing")
    parser.add_argument(
        "--shelf",
        action="store_true",
        help="Do not move intersecting cuboids to the shelf opening side, defined as negative local x.",
    )
    parser.add_argument(
        "--compartment_shrink_prob",
        type=float,
        default=0.3,
        help="Independent probability of shrinking each compartment xy boundary during post-processing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output.strip() or input_path)
    inplace = input_path == output_path

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    with h5py.File(input_path, "r") as hdf5_file:
        already_postprocessed = bool(hdf5_file.attrs.get("postprocessed", False))

    if already_postprocessed:
        raise RuntimeError(f"HDF5 is already post-processed: {input_path}")

    if not inplace:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.copy2(input_path, output_path)

    env_cfg = load_task_env_cfg(args.task_name)

    with h5py.File(output_path, "r+") as hdf5_file:
        demo_root = hdf5_file["data"] if "data" in hdf5_file else hdf5_file
        demo_keys = sorted(
            (key for key in demo_root.keys() if key.startswith("demo_")),
            key=lambda key: int(key.split("_")[-1]),
        )

        total_distractor_added = 0
        total_add_on_added = 0
        total_mobile_added = 0
        total_moved = 0
        total_shrunk = 0
        for demo_key in tqdm(demo_keys, desc="Post-processing demos"):
            demo_idx = int(demo_key.split("_")[-1])
            distractor_added, add_on_added, mobile_added, moved, shrunk = process_demo(
                demo_root[demo_key],
                demo_idx,
                args.seed,
                env_cfg,
                shelf=args.shelf,
                compartment_shrink_prob=args.compartment_shrink_prob,
            )
            total_distractor_added += distractor_added
            total_add_on_added += add_on_added
            total_mobile_added += mobile_added
            total_moved += moved
            total_shrunk += shrunk

        hdf5_file.attrs["postprocessed"] = True

    print(f"Processed HDF5: {output_path}")
    print(f"Updated demos: {len(demo_keys)}")
    print(f"Added distractor cuboids: {total_distractor_added}")
    print(f"Added boundary add-on cuboids: {total_add_on_added}")
    print(f"Added mobile obstacles: {total_mobile_added}")
    print(f"Moved cuboids outside first compartment: {total_moved}")
    print(f"Shrunk compartments: {total_shrunk}")


if __name__ == "__main__":
    main()
