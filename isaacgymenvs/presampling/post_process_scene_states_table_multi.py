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


def load_distractor_params(task_name: str):
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cfg"))
    hydra_state = GlobalHydra.instance()
    if hydra_state.is_initialized():
        hydra_state.clear()
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(config_name=None, overrides=[f"+task={task_name}"])
    return cfg.task.env.distractor_settings.params


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
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
    return (
        np.asarray(dims, dtype=np.float32).reshape(-1, 3),
        np.asarray(centers, dtype=np.float32).reshape(-1, 3),
        np.asarray(quats, dtype=np.float32).reshape(-1, 4),
    )


def process_demo(demo_group, demo_idx: int, base_seed: int, distractor_params):
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
    extra_dims, extra_centers, extra_quats = create_distractor_cuboids(
        table_pos, table_size, distractor_params, base_seed + demo_idx
    )
    added = int(extra_dims.shape[0])
    if added == 0:
        return 0

    cuboid_dims = np.concatenate((cuboid_dims, extra_dims), axis=0)
    cuboid_centers = np.concatenate((cuboid_centers, extra_centers), axis=0)
    cuboid_quats = np.concatenate((cuboid_quats, extra_quats), axis=0)

    zero_vec = np.zeros((added,), dtype=np.float32)
    zero_xyz = np.zeros((added, 3), dtype=np.float32)
    identity_quats = np.broadcast_to(IDENTITY_QUAT, (added, 4))
    cylinder_radii = np.concatenate((cylinder_radii, zero_vec))
    cylinder_heights = np.concatenate((cylinder_heights, zero_vec))
    cylinder_centers = np.concatenate((cylinder_centers, zero_xyz), axis=0)
    cylinder_quats = np.concatenate((cylinder_quats, identity_quats), axis=0)
    sphere_centers = np.concatenate((sphere_centers, zero_xyz), axis=0)
    sphere_radii = np.concatenate((sphere_radii, zero_vec))
    mesh_positions = np.concatenate((mesh_positions, zero_xyz), axis=0)
    mesh_scales = np.concatenate((mesh_scales, zero_vec))
    mesh_quats = np.concatenate((mesh_quats, identity_quats), axis=0)
    obj_ids = np.concatenate((obj_ids, zero_vec))
    mesh_ids = np.concatenate((mesh_ids, zero_vec))

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

    updated_states = np.empty((states.shape[0], STATE_PREFIX_DIMS + updated_scene.shape[0]), dtype=np.float32)
    updated_states[:, :STATE_PREFIX_DIMS] = states[:, :STATE_PREFIX_DIMS]
    updated_states[:, STATE_PREFIX_DIMS:] = updated_scene

    attrs = dict(states_dataset.attrs.items())
    del demo_group["states"]
    dataset = demo_group.create_dataset("states", data=updated_states)
    for key, value in attrs.items():
        dataset.attrs[key] = value

    return added


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_name", default="DexMobileExpBase")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path.strip() or input_path)
    inplace = input_path == output_path

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    with h5py.File(input_path, "r") as hdf5_file:
        already_postprocessed = bool(hdf5_file.attrs.get("distractor_postprocessed", False))

    if already_postprocessed:
        raise RuntimeError(f"HDF5 is already post-processed: {input_path}")

    if not inplace:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.copy2(input_path, output_path)

    distractor_params = load_distractor_params(args.task_name)

    with h5py.File(output_path, "r+") as hdf5_file:
        demo_root = hdf5_file["data"] if "data" in hdf5_file else hdf5_file
        demo_keys = sorted(
            (key for key in demo_root.keys() if key.startswith("demo_")),
            key=lambda key: int(key.split("_")[-1]),
        )

        total_added = 0
        for demo_key in tqdm(demo_keys, desc="Post-processing demos"):
            demo_idx = int(demo_key.split("_")[-1])
            total_added += process_demo(demo_root[demo_key], demo_idx, args.seed, distractor_params)

        hdf5_file.attrs["distractor_postprocessed"] = True

    print(f"Processed HDF5: {output_path}")
    print(f"Updated demos: {len(demo_keys)}")
    print(f"Added distractor cuboids: {total_added}")


if __name__ == "__main__":
    main()
