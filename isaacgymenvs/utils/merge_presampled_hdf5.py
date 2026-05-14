import argparse
import os
import random

import h5py
import numpy as np


REQUIRED_DEMO_KEYS = ("states", "compartment_states", "init_robot_states", "mesh_idx")


def _demo_sort_key(name: str):
    if name.startswith("demo_"):
        suffix = name[5:]
        if suffix.isdigit():
            return int(suffix)
    return name


def parse_input_spec(spec: str):
    parts = spec.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit():
        path = parts[0]
        num_demos = int(parts[1])
        if num_demos < 0:
            raise ValueError(f"Invalid num_demos in '{spec}'. It must be >= 0.")
        return path, num_demos
    return spec, None


def load_demos(src_path, num_demos=None):
    demos = []
    with h5py.File(src_path, "r") as src_file:
        demo_names = sorted(
            (k for k in src_file.keys() if k.startswith("demo_")),
            key=_demo_sort_key,
        )

        if num_demos is not None:
            if num_demos > len(demo_names):
                raise ValueError(
                    f"Requested {num_demos} demos from {src_path}, but only {len(demo_names)} exist."
                )
            demo_names = demo_names[:num_demos]

        for demo_name in demo_names:
            src_demo = src_file[demo_name]
            demo_data = {}
            for required_key in REQUIRED_DEMO_KEYS:
                if required_key not in src_demo:
                    raise KeyError(
                        f"Missing key '{required_key}' in {src_path}:{demo_name}"
                    )
                demo_data[required_key] = src_demo[required_key][()]
            demos.append(demo_data)

        src_success_rate = src_file.attrs.get("presample_success_rate", None)
        selected_count = len(demo_names)

    return demos, src_success_rate, selected_count


def check_all_hdf5_postprocessed(input_paths):
    not_postprocessed_paths = []
    for input_path in input_paths:
        with h5py.File(input_path, "r") as hdf5_file:
            if not bool(hdf5_file.attrs.get("postprocessed", False)):
                not_postprocessed_paths.append(input_path)

    all_postprocessed = not not_postprocessed_paths
    if not all_postprocessed:
        print(
            "Warning: not all HDF5 files are postprocessed; "
            "the merged HDF5 will not be marked as postprocessed."
        )
        print("HDF5 files that are not postprocessed:")
        for input_path in not_postprocessed_paths:
            print(f"  {input_path}")

    return all_postprocessed


def merge_hdf5_files(input_specs, output_path):
    if not input_specs:
        raise ValueError("No input HDF5 files provided.")

    output_dir, file_name = os.path.split(output_path)
    name, ext = os.path.splitext(file_name)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_weighted_success = 0.0
    total_weight = 0
    all_demos = []
    input_paths = []

    for spec in input_specs:
        src_path, num_demos = parse_input_spec(spec)
        input_paths.append(src_path)
        demos, src_success_rate, selected_count = load_demos(src_path, num_demos)
        all_demos.extend(demos)

        if src_success_rate is not None:
            total_weighted_success += float(src_success_rate) * selected_count
            total_weight += selected_count

    all_postprocessed = check_all_hdf5_postprocessed(input_paths)

    random.shuffle(all_demos)

    merged_demo_count = len(all_demos)
    merged_success_rate = total_weighted_success / total_weight if total_weight > 0 else float("nan")

    output_path = os.path.join(
        output_dir,
        f"{name}_sr_{merged_success_rate:.4f}_num_{merged_demo_count}{ext}",
    )

    with h5py.File(output_path, "w") as dst_file:
        for idx, demo_data in enumerate(all_demos):
            dst_demo = dst_file.create_group(f"demo_{idx}")
            for key, value in demo_data.items():
                dst_demo.create_dataset(key, data=value)

        dst_file.attrs["num_valid_samples"] = merged_demo_count
        if total_weight > 0:
            dst_file.attrs["presample_success_rate"] = total_weighted_success / total_weight
        else:
            dst_file.attrs["presample_success_rate"] = np.nan
        dst_file.attrs["num_source_files"] = len(input_paths)
        dst_file.attrs["source_files"] = np.array(input_paths, dtype=h5py.string_dtype("utf-8"))
        if all_postprocessed:
            dst_file.attrs["postprocessed"] = True

    print(
        f"Merged {len(input_paths)} files with {merged_demo_count} demos into {output_path}. "
        f"presample_success_rate={merged_success_rate:.6f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple presampled HDF5 files into one file."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input specs: /path/to/file.hdf5 or /path/to/file.hdf5:num_demos",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output merged HDF5 file path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    merge_hdf5_files(args.input, args.output)


if __name__ == "__main__":
    main()
