#!/usr/bin/env python3
"""Make the last two cuboids in each trajectory demo the compartment side walls."""

import argparse
import os
import shutil

import h5py
import numpy as np


WALL_THICKNESS = 0.01
STATE_PREFIX_DIMS = 15


def demo_sort_key(name):
    if name.startswith("demo_") and name[5:].isdigit():
        return int(name[5:])
    return name


def wall_dims_and_positions(compartment_state):
    compartment_dims = np.asarray(compartment_state[:3], dtype=np.float32)
    compartment_pos = np.asarray(compartment_state[3:6], dtype=np.float32)

    wall_dim = np.array(
        [
            WALL_THICKNESS,
            compartment_dims[0],
            compartment_dims[2],
        ],
        dtype=np.float32,
    )
    wall_dims = np.stack([wall_dim, wall_dim], axis=0)

    y_offset = compartment_dims[1] * 0.5
    wall_positions = np.stack(
        [
            compartment_pos + np.array([0.0, y_offset, 0.0], dtype=np.float32),
            compartment_pos + np.array([0.0, -y_offset, 0.0], dtype=np.float32),
        ],
        axis=0,
    )
    return wall_dims, wall_positions


def update_obstacle_cuboids(demo, wall_dims, wall_positions):
    if "obstacles" not in demo or "cuboids" not in demo["obstacles"]:
        return

    cuboids = demo["obstacles/cuboids"]
    dims = cuboids["dimensions_xyz"][:]
    positions = cuboids["position_xyz"][:]
    if len(dims) < 2:
        raise ValueError("Need at least two cuboids to overwrite the last two as side walls.")

    dims[-2:] = wall_dims
    positions[-2:] = wall_positions
    cuboids["dimensions_xyz"][...] = dims
    cuboids["position_xyz"][...] = positions

    if "pose_xyz_quat_xyzw" in cuboids:
        poses = cuboids["pose_xyz_quat_xyzw"][:]
        poses[-2:, :3] = wall_positions
        cuboids["pose_xyz_quat_xyzw"][...] = poses


def update_states_dataset(dataset, wall_dims, wall_positions):
    states = dataset[:]
    flat_states = states.reshape(-1, states.shape[-1])

    for state in flat_states:
        scene = state[STATE_PREFIX_DIMS:]
        num_cuboids = int(scene[0])
        if num_cuboids < 2:
            raise ValueError("Need at least two cuboids in states to overwrite the last two.")

        dims_start = STATE_PREFIX_DIMS + 1
        positions_start = dims_start + 3 * num_cuboids

        dims = state[dims_start : dims_start + 3 * num_cuboids].reshape(num_cuboids, 3)
        positions = state[positions_start : positions_start + 3 * num_cuboids].reshape(num_cuboids, 3)
        dims[-2:] = wall_dims
        positions[-2:] = wall_positions

    dataset[...] = states


def process_demo(demo):
    if "compartment_states" in demo:
        compartment_state = demo["compartment_states"][0]
    else:
        compartment_state = demo["environment/compartment_states"][0]

    wall_dims, wall_positions = wall_dims_and_positions(compartment_state)

    update_obstacle_cuboids(demo, wall_dims, wall_positions)
    if "states" in demo:
        update_states_dataset(demo["states"], wall_dims, wall_positions)
    if "environment" in demo and "states" in demo["environment"]:
        update_states_dataset(demo["environment/states"], wall_dims, wall_positions)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="",
        help="Output HDF5 path. If omitted, the input file is modified in place.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output) if args.output else input_path

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    if output_path != input_path:
        if os.path.exists(output_path):
            raise FileExistsError(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(input_path, output_path)

    with h5py.File(output_path, "r+") as hdf5_file:
        root = hdf5_file["data"] if "data" in hdf5_file else hdf5_file
        demo_keys = sorted(
            [key for key in root.keys() if key.startswith("demo_")],
            key=demo_sort_key,
        )

        for demo_key in demo_keys:
            process_demo(root[demo_key])

        hdf5_file.attrs["side_wall_postprocessed"] = True

    print(f"Processed {len(demo_keys)} demos: {output_path}")


if __name__ == "__main__":
    main()
