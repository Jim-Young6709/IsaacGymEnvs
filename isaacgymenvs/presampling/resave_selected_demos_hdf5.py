#!/usr/bin/env python3
"""Copy selected demos from one or more trajectory HDF5 files into one file.

Example:
    python IsaacGymEnvs/isaacgymenvs/presampling/resave_selected_demos_hdf5.py \
        --select traj_a.hdf5 3 5 8 \
        --select traj_b.hdf5 0 2 \
        --output selected_demos.hdf5

The output contains:
    /data/demo_0  <- traj_a.hdf5 demo index 3
    /data/demo_1  <- traj_a.hdf5 demo index 5
    /data/demo_2  <- traj_a.hdf5 demo index 8
    /data/demo_3  <- traj_b.hdf5 demo index 0
    /data/demo_4  <- traj_b.hdf5 demo index 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


def demo_sort_key(name: str):
    if name.startswith("demo_") and name[5:].isdigit():
        return int(name[5:])
    return name


def list_demo_keys(hdf5_file: h5py.File) -> list[str]:
    root = hdf5_file["data"] if "data" in hdf5_file else hdf5_file
    demo_keys = sorted(
        (name for name in root.keys() if name.startswith("demo_")),
        key=demo_sort_key,
    )
    if not demo_keys:
        raise RuntimeError("No demo_* groups found in HDF5.")
    return demo_keys


def resolve_demo_key(hdf5_file: h5py.File, demo: str) -> tuple[h5py.Group, str]:
    root = hdf5_file["data"] if "data" in hdf5_file else hdf5_file
    demo_keys = list_demo_keys(hdf5_file)

    if demo.startswith("demo_"):
        demo_key = demo
    else:
        demo_idx = int(demo)
        if demo_idx < 0 or demo_idx >= len(demo_keys):
            raise IndexError(f"Demo index {demo_idx} out of range [0, {len(demo_keys) - 1}].")
        demo_key = demo_keys[demo_idx]

    if demo_key not in root:
        raise KeyError(f"Demo '{demo_key}' not found. Available examples: {demo_keys[:5]}")
    return root[demo_key], demo_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-save selected demo groups from one or more trajectory HDF5 files."
    )
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("HDF5_PATH", "DEMO_IDX_OR_KEY"),
        help="Input HDF5 path and demo index/key. Repeat this flag for multiple demos.",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        action="append",
        metavar=("HDF5_PATH", "DEMO_IDX_OR_KEY"),
        help="Input HDF5 path followed by one or more demo indices/keys. Repeat this flag for multiple files.",
    )
    parser.add_argument(
        "--hdf5-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Input HDF5 paths. Must be paired with --demo-idxs.",
    )
    parser.add_argument(
        "--demo-idxs",
        nargs="+",
        default=None,
        help="Demo indices/keys corresponding one-to-one with --hdf5-paths.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output HDF5 path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists.")
    return parser.parse_args()


def get_sources(args: argparse.Namespace) -> list[tuple[Path, str]]:
    pair_sources = args.input or []
    grouped_sources = args.select or []
    list_paths = args.hdf5_paths or []
    list_demos = args.demo_idxs or []

    num_modes = sum(bool(mode) for mode in (pair_sources, grouped_sources, list_paths or list_demos))
    if num_modes > 1:
        raise ValueError("Use only one input style: --select, repeated --input, or --hdf5-paths/--demo-idxs.")
    if grouped_sources:
        sources = []
        for group in grouped_sources:
            if len(group) < 2:
                raise ValueError("--select requires one HDF5 path followed by at least one demo index/key.")
            source_path = Path(group[0]).expanduser().resolve()
            sources.extend((source_path, str(demo)) for demo in group[1:])
        return sources
    if pair_sources:
        return [(Path(path).expanduser().resolve(), str(demo)) for path, demo in pair_sources]
    if list_paths or list_demos:
        if len(list_paths) != len(list_demos):
            raise ValueError(
                f"--hdf5-paths and --demo-idxs must have the same length, got "
                f"{len(list_paths)} and {len(list_demos)}."
            )
        return [(path.expanduser().resolve(), str(demo)) for path, demo in zip(list_paths, list_demos)]
    raise ValueError("Provide demos with either --input HDF5 DEMO or --hdf5-paths/--demo-idxs.")


def main() -> None:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sources = get_sources(args)

    with h5py.File(output_path, "w") as dst_file:
        dst_data = dst_file.create_group("data")
        dst_file.attrs["source_selection_json"] = json.dumps(
            [{"hdf5_path": str(path), "demo": demo} for path, demo in sources],
            sort_keys=True,
        )

        for output_idx, (source_path, demo) in enumerate(sources):
            if not source_path.exists():
                raise FileNotFoundError(f"Input HDF5 does not exist: {source_path}")

            with h5py.File(source_path, "r") as src_file:
                src_demo_group, src_demo_key = resolve_demo_key(src_file, demo)
                dst_demo_key = f"demo_{output_idx}"
                src_file.copy(src_demo_group, dst_data, name=dst_demo_key)

                dst_demo_group = dst_data[dst_demo_key]
                dst_demo_group.attrs["source_hdf5_path"] = str(source_path)
                dst_demo_group.attrs["source_demo_key"] = src_demo_key
                dst_demo_group.attrs["source_demo_arg"] = demo

                print(f"{source_path}:{src_demo_key} -> /data/{dst_demo_key}")

    print(f"Wrote {len(sources)} demos to {output_path}")


if __name__ == "__main__":
    main()
