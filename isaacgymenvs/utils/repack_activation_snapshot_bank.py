import argparse
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np


def _resolve_path(path_str):
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _glob_shard_paths(base_path):
    root, ext = os.path.splitext(str(base_path))
    if ext == "":
        ext = ".hdf5"
    shard_paths = sorted(
        Path(base_path).parent.glob(f"{Path(root).name}_rank*{ext}"),
        key=lambda p: int(p.stem.split("_rank")[-1]),
    )
    if not shard_paths:
        raise FileNotFoundError(f"No activation snapshot bank shards found for base={base_path}")
    return shard_paths


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-rank activation snapshot bank shards into one global bank "
            "indexed by [mode, variation, slot] for the copied side RL env."
        )
    )
    parser.add_argument("--input-base", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--skip-bad-shards", action="store_true")
    args = parser.parse_args()

    if args.target_count <= 0:
        raise ValueError("target-count must be positive")

    input_base = _resolve_path(args.input_base)
    output_path = _resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shard_paths = _glob_shard_paths(input_base)
    rank_data = []
    category_names = None
    num_dofs = None
    num_variants = None
    has_target_quat = True
    has_failure_branch = True
    has_episode_steps = True
    has_table_size = True
    num_skipped_shards = 0

    for shard_path in shard_paths:
        try:
            with h5py.File(shard_path, "r") as f:
                shard_category_names = json.loads(f.attrs["category_names_json"])
                shard_num_dofs = int(f.attrs["num_dofs"])
                shard_num_variants = int(f.attrs.get("num_variants_global", f["variation_id"][...].max() + 1))
                if category_names is None:
                    category_names = shard_category_names
                    num_dofs = shard_num_dofs
                    num_variants = shard_num_variants
                else:
                    if shard_category_names != category_names:
                        raise RuntimeError(f"Category-name mismatch in shard {shard_path}")
                    if shard_num_dofs != num_dofs:
                        raise RuntimeError(f"num_dofs mismatch in shard {shard_path}")
                    if shard_num_variants != num_variants:
                        raise RuntimeError(f"num_variants mismatch in shard {shard_path}")

                shard_payload = {
                    "counts": f["counts"][...],
                    "activation_counts": f["activation_counts"][...],
                    "joint_config": f["joint_config"][...],
                    "object_center_world": f["object_center_world"][...],
                    "object_quat_world": f["object_quat_world"][...],
                    "side_is_left": f["side_is_left"][...].astype(np.bool_),
                    "table_pos_world": f["table_pos_world"][...],
                    "table_quat_world": f["table_quat_world"][...],
                    "variation_id": f["variation_id"][...].astype(np.int64),
                }
                if "table_size" in f:
                    shard_payload["table_size"] = f["table_size"][...]
                else:
                    has_table_size = False
                if "episode_steps" in f:
                    shard_payload["episode_steps"] = f["episode_steps"][...]
                else:
                    has_episode_steps = False
                if "target_quat_world" in f:
                    shard_payload["target_quat_world"] = f["target_quat_world"][...]
                else:
                    has_target_quat = False
                if "failure_counts" in f:
                    shard_payload["failure_counts"] = f["failure_counts"][...]
                    shard_payload["failure_joint_config"] = f["failure_joint_config"][...]
                    shard_payload["failure_object_center_world"] = f["failure_object_center_world"][...]
                    shard_payload["failure_object_quat_world"] = f["failure_object_quat_world"][...]
                    shard_payload["failure_side_is_left"] = f["failure_side_is_left"][...].astype(np.bool_)
                    shard_payload["failure_table_pos_world"] = f["failure_table_pos_world"][...]
                    shard_payload["failure_table_quat_world"] = f["failure_table_quat_world"][...]
                    if "failure_episode_steps" in f:
                        shard_payload["failure_episode_steps"] = f["failure_episode_steps"][...]
                    else:
                        has_episode_steps = False
                    if "failure_target_quat_world" in f:
                        shard_payload["failure_target_quat_world"] = f["failure_target_quat_world"][...]
                    else:
                        has_target_quat = False
                else:
                    has_failure_branch = False
                rank_data.append(shard_payload)
        except Exception as exc:
            if not args.skip_bad_shards:
                raise
            num_skipped_shards += 1
            print(f"[repack_activation_snapshot_bank] skipping unreadable shard={shard_path} err={exc!r}")

    if not rank_data:
        raise RuntimeError("No readable activation snapshot shards were found")

    num_categories = len(category_names)
    sample_refs = [[[] for _ in range(num_variants)] for _ in range(num_categories)]
    activation_event_counts = np.zeros((num_categories, num_variants), dtype=np.int64)
    failure_sample_refs = [[[] for _ in range(num_variants)] for _ in range(num_categories)]
    variant_table_size = None
    if has_table_size:
        variant_table_size = np.full((num_variants, 3), np.nan, dtype=np.float32)

    for shard_idx, shard in enumerate(rank_data):
        local_num_envs = int(shard["variation_id"].shape[0])
        for local_env_id in range(local_num_envs):
            variation_id = int(shard["variation_id"][local_env_id])
            if has_table_size:
                table_size_i = np.asarray(shard["table_size"][local_env_id], dtype=np.float32)
                if np.isnan(variant_table_size[variation_id]).any():
                    variant_table_size[variation_id] = table_size_i
                elif not np.allclose(variant_table_size[variation_id], table_size_i, atol=1.0e-6, rtol=0.0):
                    raise RuntimeError(
                        f"Inconsistent table_size for variation_id={variation_id}: "
                        f"{variant_table_size[variation_id].tolist()} vs {table_size_i.tolist()}"
                    )
            for category_id in range(num_categories):
                activation_event_counts[category_id, variation_id] += int(
                    shard["activation_counts"][category_id, local_env_id]
                )
                stored_count = int(shard["counts"][category_id, local_env_id])
                for sample_idx in range(stored_count):
                    sample_refs[category_id][variation_id].append((shard_idx, local_env_id, sample_idx))
                if has_failure_branch:
                    failure_stored_count = int(shard["failure_counts"][category_id, local_env_id])
                    for sample_idx in range(failure_stored_count):
                        failure_sample_refs[category_id][variation_id].append((shard_idx, local_env_id, sample_idx))

    rng = random.Random(int(args.seed))
    selected_refs = [[[] for _ in range(num_variants)] for _ in range(num_categories)]
    selected_counts = np.zeros((num_categories, num_variants), dtype=np.int32)
    failure_selected_refs = [[[] for _ in range(num_variants)] for _ in range(num_categories)]
    failure_selected_counts = np.zeros((num_categories, num_variants), dtype=np.int32)
    for category_id in range(num_categories):
        for variation_id in range(num_variants):
            refs = list(sample_refs[category_id][variation_id])
            rng.shuffle(refs)
            chosen = refs[: args.target_count]
            selected_refs[category_id][variation_id] = chosen
            selected_counts[category_id, variation_id] = len(chosen)
            if has_failure_branch:
                failure_refs = list(failure_sample_refs[category_id][variation_id])
                rng.shuffle(failure_refs)
                chosen_failure = failure_refs[: args.target_count]
                failure_selected_refs[category_id][variation_id] = chosen_failure
                failure_selected_counts[category_id, variation_id] = len(chosen_failure)

    if not args.allow_short:
        min_selected_count = int(selected_counts.min()) if selected_counts.size > 0 else 0
        if min_selected_count < int(args.target_count):
            short_rows = np.argwhere(selected_counts < int(args.target_count))
            preview = short_rows[:10].tolist()
            raise RuntimeError(
                f"Not enough activation snapshots to repack target_count={args.target_count}. "
                f"global_min_selected={min_selected_count} first_short_rows={preview}"
            )

    joint_config = np.zeros((num_categories, num_variants, args.target_count, num_dofs), dtype=np.float32)
    object_center_world = np.zeros((num_categories, num_variants, args.target_count, 3), dtype=np.float32)
    object_quat_world = np.zeros((num_categories, num_variants, args.target_count, 4), dtype=np.float32)
    side_is_left = np.zeros((num_categories, num_variants, args.target_count), dtype=np.uint8)
    table_pos_world = np.zeros((num_categories, num_variants, args.target_count, 3), dtype=np.float32)
    table_quat_world = np.zeros((num_categories, num_variants, args.target_count, 4), dtype=np.float32)
    target_quat_world = None
    episode_steps = None
    if has_target_quat:
        target_quat_world = np.zeros((num_categories, num_variants, args.target_count, 4), dtype=np.float32)
    if has_episode_steps:
        episode_steps = np.zeros((num_categories, num_variants, args.target_count), dtype=np.int32)
    failure_joint_config = None
    failure_object_center_world = None
    failure_object_quat_world = None
    failure_side_is_left = None
    failure_table_pos_world = None
    failure_table_quat_world = None
    failure_target_quat_world = None
    failure_episode_steps = None
    if has_failure_branch:
        failure_joint_config = np.zeros((num_categories, num_variants, args.target_count, num_dofs), dtype=np.float32)
        failure_object_center_world = np.zeros((num_categories, num_variants, args.target_count, 3), dtype=np.float32)
        failure_object_quat_world = np.zeros((num_categories, num_variants, args.target_count, 4), dtype=np.float32)
        failure_side_is_left = np.zeros((num_categories, num_variants, args.target_count), dtype=np.uint8)
        failure_table_pos_world = np.zeros((num_categories, num_variants, args.target_count, 3), dtype=np.float32)
        failure_table_quat_world = np.zeros((num_categories, num_variants, args.target_count, 4), dtype=np.float32)
        if has_episode_steps:
            failure_episode_steps = np.zeros((num_categories, num_variants, args.target_count), dtype=np.int32)
        if has_target_quat:
            failure_target_quat_world = np.zeros((num_categories, num_variants, args.target_count, 4), dtype=np.float32)

    for category_id in range(num_categories):
        for variation_id in range(num_variants):
            for write_idx, (src_rank, src_env_id, src_sample_idx) in enumerate(selected_refs[category_id][variation_id]):
                shard = rank_data[src_rank]
                joint_config[category_id, variation_id, write_idx] = shard["joint_config"][category_id, src_env_id, src_sample_idx]
                object_center_world[category_id, variation_id, write_idx] = shard["object_center_world"][category_id, src_env_id, src_sample_idx]
                object_quat_world[category_id, variation_id, write_idx] = shard["object_quat_world"][category_id, src_env_id, src_sample_idx]
                side_is_left[category_id, variation_id, write_idx] = np.uint8(
                    shard["side_is_left"][category_id, src_env_id, src_sample_idx]
                )
                table_pos_world[category_id, variation_id, write_idx] = shard["table_pos_world"][category_id, src_env_id, src_sample_idx]
                table_quat_world[category_id, variation_id, write_idx] = shard["table_quat_world"][category_id, src_env_id, src_sample_idx]
                if has_episode_steps:
                    episode_steps[category_id, variation_id, write_idx] = shard["episode_steps"][
                        category_id, src_env_id, src_sample_idx
                    ]
                if has_target_quat:
                    target_quat_world[category_id, variation_id, write_idx] = shard["target_quat_world"][
                        category_id, src_env_id, src_sample_idx
                    ]
            if has_failure_branch:
                for write_idx, (src_rank, src_env_id, src_sample_idx) in enumerate(
                    failure_selected_refs[category_id][variation_id]
                ):
                    shard = rank_data[src_rank]
                    failure_joint_config[category_id, variation_id, write_idx] = shard["failure_joint_config"][
                        category_id, src_env_id, src_sample_idx
                    ]
                    failure_object_center_world[category_id, variation_id, write_idx] = shard[
                        "failure_object_center_world"
                    ][category_id, src_env_id, src_sample_idx]
                    failure_object_quat_world[category_id, variation_id, write_idx] = shard[
                        "failure_object_quat_world"
                    ][category_id, src_env_id, src_sample_idx]
                    failure_side_is_left[category_id, variation_id, write_idx] = np.uint8(
                        shard["failure_side_is_left"][category_id, src_env_id, src_sample_idx]
                    )
                    failure_table_pos_world[category_id, variation_id, write_idx] = shard[
                        "failure_table_pos_world"
                    ][category_id, src_env_id, src_sample_idx]
                    failure_table_quat_world[category_id, variation_id, write_idx] = shard[
                        "failure_table_quat_world"
                    ][category_id, src_env_id, src_sample_idx]
                    if has_episode_steps:
                        failure_episode_steps[category_id, variation_id, write_idx] = shard[
                            "failure_episode_steps"
                        ][category_id, src_env_id, src_sample_idx]
                    if has_target_quat:
                        failure_target_quat_world[category_id, variation_id, write_idx] = shard[
                            "failure_target_quat_world"
                        ][category_id, src_env_id, src_sample_idx]

    with h5py.File(output_path, "w") as f:
        f.attrs["num_dofs"] = int(num_dofs)
        f.attrs["capacity_per_variation"] = int(args.target_count)
        f.attrs["category_names_json"] = json.dumps(list(category_names))
        f.attrs["num_variants_global"] = int(num_variants)
        f.attrs["num_input_shards_read"] = int(len(rank_data))
        f.attrs["num_input_shards_skipped"] = int(num_skipped_shards)
        f.create_dataset("counts", data=selected_counts)
        f.create_dataset("activation_counts", data=activation_event_counts)
        f.create_dataset("joint_config", data=joint_config)
        f.create_dataset("object_center_world", data=object_center_world)
        f.create_dataset("object_quat_world", data=object_quat_world)
        f.create_dataset("side_is_left", data=side_is_left)
        f.create_dataset("table_pos_world", data=table_pos_world)
        f.create_dataset("table_quat_world", data=table_quat_world)
        if has_table_size and variant_table_size is not None and not np.isnan(variant_table_size).any():
            f.create_dataset("table_size", data=variant_table_size)
        if has_episode_steps:
            f.create_dataset("episode_steps", data=episode_steps)
        if has_target_quat:
            f.create_dataset("target_quat_world", data=target_quat_world)
        if has_failure_branch:
            f.create_dataset("failure_counts", data=failure_selected_counts)
            f.create_dataset("failure_joint_config", data=failure_joint_config)
            f.create_dataset("failure_object_center_world", data=failure_object_center_world)
            f.create_dataset("failure_object_quat_world", data=failure_object_quat_world)
            f.create_dataset("failure_side_is_left", data=failure_side_is_left)
            f.create_dataset("failure_table_pos_world", data=failure_table_pos_world)
            f.create_dataset("failure_table_quat_world", data=failure_table_quat_world)
            if has_episode_steps:
                f.create_dataset("failure_episode_steps", data=failure_episode_steps)
            if has_target_quat:
                f.create_dataset("failure_target_quat_world", data=failure_target_quat_world)

    for category_id, category_name in enumerate(category_names):
        category_min = int(selected_counts[category_id].min()) if selected_counts.shape[1] > 0 else 0
        category_max = int(selected_counts[category_id].max()) if selected_counts.shape[1] > 0 else 0
        category_total = int(selected_counts[category_id].sum())
        event_total = int(activation_event_counts[category_id].sum())
        print(
            f"[repack_activation_snapshot_bank] category={category_name} "
            f"selected_total={category_total} selected_min_per_variant={category_min} "
            f"selected_max_per_variant={category_max} activation_event_total={event_total} "
            f"target_count={args.target_count}"
        )
        if has_failure_branch:
            failure_total = int(failure_selected_counts[category_id].sum())
            failure_min = int(failure_selected_counts[category_id].min()) if failure_selected_counts.shape[1] > 0 else 0
            failure_max = int(failure_selected_counts[category_id].max()) if failure_selected_counts.shape[1] > 0 else 0
            print(
                f"[repack_activation_snapshot_bank] category={category_name} "
                f"failure_selected_total={failure_total} failure_selected_min_per_variant={failure_min} "
                f"failure_selected_max_per_variant={failure_max}"
            )

    print(f"Wrote activation snapshot bank: {output_path}")


if __name__ == "__main__":
    main()
