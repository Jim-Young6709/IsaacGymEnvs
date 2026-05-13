import argparse
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


def demo_sort_key(name):
    if name.startswith("demo_"):
        suffix = name[5:]
        if suffix.isdigit():
            return int(suffix)
    return name


def get_demo_names(group):
    return sorted(
        (name for name in group.keys() if name.startswith("demo_")),
        key=demo_sort_key,
    )


def get_demo_root(hdf5_file, demo_root):
    if demo_root is not None:
        root_name = demo_root.strip("/")
        if root_name == "":
            return hdf5_file, "/"
        return hdf5_file[root_name], f"/{root_name}"

    if "data" in hdf5_file and isinstance(hdf5_file["data"], h5py.Group):
        return hdf5_file["data"], "/data"
    return hdf5_file, "/"


def read_mesh_idx(dataset):
    value = np.asarray(dataset[()]).reshape(-1)
    if value.size == 0:
        raise ValueError("mesh_idx dataset is empty")
    return int(value[0])


def discover_mesh_entries(mesh_dir, object_list):
    mesh_dir = os.path.abspath(mesh_dir)
    type_mapping_path = os.path.join(mesh_dir, "type_mapping.json")
    entries = []

    if os.path.isfile(type_mapping_path):
        with open(type_mapping_path, "r", encoding="utf-8") as f:
            obj_str2int = json.load(f)

        if object_list == ["all"]:
            object_list = [
                obj for obj in os.listdir(mesh_dir)
                if obj != "type_mapping.json" and os.path.isdir(os.path.join(mesh_dir, obj))
            ]

        object_list = sorted(
            object_list,
            key=lambda obj: (0, obj_str2int[obj]) if obj in obj_str2int else (1, obj),
        )

        for obj in object_list:
            obj_dir = os.path.join(mesh_dir, obj)
            if not os.path.isdir(obj_dir):
                continue
            for file_name in sorted(os.listdir(obj_dir)):
                if not file_name.endswith(".obj"):
                    continue
                mesh_path = os.path.join(obj_dir, file_name)
                asset_mesh_id = Path(mesh_path).parts[-2]
                entries.append(
                    {
                        "mesh_path": mesh_path,
                        "asset_mesh_id": asset_mesh_id,
                        "display_name": asset_mesh_id,
                        "variant_mode": False,
                    }
                )
        return entries

    selected_objects = None if object_list == ["all"] else set(object_list)
    variant_entries = []

    for root, _, files in os.walk(mesh_dir):
        obj_files = sorted(file_name for file_name in files if file_name.endswith(".obj"))
        if not obj_files:
            continue

        rel_dir = os.path.relpath(root, mesh_dir)
        parts = rel_dir.split(os.sep)
        if len(parts) == 3:
            category, object_name, variant = parts
            mesh_rel_prefix = os.path.join(category, object_name, variant)
        elif len(parts) == 2:
            object_name, variant = parts
            mesh_rel_prefix = os.path.join(object_name, variant)
        else:
            continue

        if selected_objects is not None and object_name not in selected_objects:
            continue

        for obj_file in obj_files:
            if not obj_file.startswith(f"{object_name}_{variant}"):
                continue

            mesh_path = os.path.join(root, obj_file)
            mesh_stem = os.path.splitext(obj_file)[0]
            json_path = os.path.join(root, mesh_stem + ".json")
            if not os.path.isfile(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("transform_model") != "dilation_only_v1":
                continue

            variant_entries.append(
                (
                    f"{object_name}_{variant}",
                    mesh_path,
                    os.path.join(mesh_rel_prefix, mesh_stem),
                )
            )

    for display_name, mesh_path, asset_mesh_id in sorted(variant_entries, key=lambda item: (item[0], item[1])):
        entries.append(
            {
                "mesh_path": mesh_path,
                "asset_mesh_id": asset_mesh_id,
                "display_name": display_name,
                "variant_mode": True,
            }
        )

    return entries


def mesh_label(mesh_idx, mesh_entries):
    if not mesh_entries:
        return ""

    if 0 <= mesh_idx < len(mesh_entries):
        entry = mesh_entries[mesh_idx]
    elif mesh_entries[0].get("variant_mode", False) and mesh_idx >= 0:
        entry = mesh_entries[mesh_idx % len(mesh_entries)]
    else:
        return "<out of range>"

    return entry["display_name"]


def print_counts(hdf5_path, key, demo_root, mesh_entries, show_missing):
    with h5py.File(hdf5_path, "r") as hdf5_file:
        demos, demo_root_path = get_demo_root(hdf5_file, demo_root)
        demo_names = get_demo_names(demos)

        counts = Counter()
        missing = []
        bad = []

        for demo_name in demo_names:
            demo = demos[demo_name]
            if key not in demo:
                missing.append(demo_name)
                continue

            try:
                mesh_idx = read_mesh_idx(demo[key])
            except Exception as exc:
                bad.append((demo_name, str(exc)))
                continue

            counts[mesh_idx] += 1

    total_counted = sum(counts.values())
    total_demos = len(demo_names)

    print(f"File: {hdf5_path}")
    print(f"Demo root: {demo_root_path}")
    print(f"Total demos: {total_demos}")
    print(f"Demos with {key}: {total_counted}")
    print(f"Demos missing {key}: {len(missing)}")
    print(f"Demos with unreadable {key}: {len(bad)}")

    if not counts:
        return

    rows = []
    for mesh_idx, count in sorted(counts.items()):
        pct = (100.0 * count / total_counted) if total_counted else 0.0
        rows.append(
            (
                str(mesh_idx),
                str(count),
                f"{pct:.2f}",
                mesh_label(mesh_idx, mesh_entries),
            )
        )

    headers = ("mesh_idx", "count", "percent", "mesh")
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rows))
        for col in range(len(headers))
    ]

    print("")
    print("  ".join(headers[col].ljust(widths[col]) for col in range(len(headers))))
    print("  ".join("-" * widths[col] for col in range(len(headers))))
    for row in rows:
        print("  ".join(row[col].ljust(widths[col]) for col in range(len(row))))

    if show_missing and missing:
        print("")
        print("Missing demos:")
        for demo_name in missing:
            print(demo_name)

    if bad:
        print("")
        print("Unreadable demos:")
        for demo_name, reason in bad:
            print(f"{demo_name}: {reason}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count saved mesh_idx values in a presampled HDF5 file."
    )
    parser.add_argument(
        "hdf5_path",
        help="Path to the presampled HDF5 file.",
    )
    parser.add_argument(
        "--key",
        default="mesh_idx",
        help="Dataset name to count inside each demo group. Default: mesh_idx.",
    )
    parser.add_argument(
        "--demo-root",
        default=None,
        help="Group containing demo_* entries. Default: auto-detect /data, otherwise root.",
    )
    parser.add_argument(
        "--mesh-dir",
        default=None,
        help="Optional mesh directory used to resolve indices to mesh display names.",
    )
    parser.add_argument(
        "--obj-list",
        nargs="+",
        default=["all"],
        help="Objects to include when resolving --mesh-dir. Default: all.",
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Print demo names that do not contain the mesh index dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mesh_entries = []
    if args.mesh_dir is not None:
        mesh_entries = discover_mesh_entries(args.mesh_dir, args.obj_list)
        if not mesh_entries:
            raise RuntimeError(
                f"No mesh entries discovered under {args.mesh_dir} with obj_list={args.obj_list}"
            )

    print_counts(
        hdf5_path=args.hdf5_path,
        key=args.key,
        demo_root=args.demo_root,
        mesh_entries=mesh_entries,
        show_missing=args.show_missing,
    )


if __name__ == "__main__":
    main()
