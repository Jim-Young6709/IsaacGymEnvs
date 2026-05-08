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


def _get_shard_path(base_path, rank):
    root, ext = os.path.splitext(str(base_path))
    if ext == "":
        ext = ".hdf5"
    return Path(f"{root}_rank{int(rank)}{ext}")


def _load_assignment_payload(path):
    with _resolve_path(path).open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "entries" not in payload:
        raise ValueError(f"Unsupported variation assignment payload: {path}")
    return payload


def _validate_assignment_payload(payload, expected_mode):
    mode = str(payload.get("mode", ""))
    if mode != expected_mode:
        raise ValueError(f"Expected assignment mode={expected_mode}, got mode={mode}")
    world_size = int(payload["world_size"])
    envs_per_rank = int(payload["envs_per_rank"])
    entries = payload["entries"]
    expected_entries = world_size * envs_per_rank
    if len(entries) != expected_entries:
        raise ValueError(
            f"Assignment entry count mismatch for mode={mode}: "
            f"expected {expected_entries}, got {len(entries)}"
        )
    return world_size, envs_per_rank


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Repack generation-time verified teacher-bank shards into training-time shards "
            "with one slot per variant."
        )
    )
    parser.add_argument("--input-base", type=str, required=True)
    parser.add_argument("--generation-assignment", type=str, required=True)
    parser.add_argument("--training-assignment", type=str, required=True)
    parser.add_argument("--output-base", type=str, required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--region-filter",
        type=str,
        default="all",
        choices=["all", "correct_only", "wrong_only"],
    )
    parser.add_argument("--allow-short", action="store_true")
    args = parser.parse_args()

    if args.target_count <= 0:
        raise ValueError("target-count must be positive")

    generation_payload = _load_assignment_payload(args.generation_assignment)
    training_payload = _load_assignment_payload(args.training_assignment)
    generation_world_size, generation_envs_per_rank = _validate_assignment_payload(generation_payload, "generation")
    training_world_size, training_envs_per_rank = _validate_assignment_payload(training_payload, "training")

    num_variants = int(generation_payload["num_variants"])
    if int(training_payload["num_variants"]) != num_variants:
        raise ValueError(
            f"num_variants mismatch: generation={num_variants} training={training_payload['num_variants']}"
        )

    input_base = _resolve_path(args.input_base)
    output_base = _resolve_path(args.output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    rank_data = []
    category_names = None
    num_dofs = None
    local_capacity = None
    for rank in range(generation_world_size):
        shard_path = _get_shard_path(input_base, rank)
        if not shard_path.exists():
            raise FileNotFoundError(f"Generation shard not found: {shard_path}")
        with h5py.File(shard_path, "r") as f:
            shard_category_names = json.loads(f.attrs["category_names_json"])
            if category_names is None:
                category_names = shard_category_names
                num_dofs = int(f.attrs["num_dofs"])
                local_capacity = int(f.attrs["capacity_per_env"])
            else:
                if shard_category_names != category_names:
                    raise RuntimeError(f"Category-name mismatch in shard {shard_path}")
                if int(f.attrs["num_dofs"]) != num_dofs:
                    raise RuntimeError(f"num_dofs mismatch in shard {shard_path}")
            rank_data.append(
                {
                    "counts": f["counts"][...],
                    "attempt_counts": f["attempt_counts"][...],
                    "success_counts": f["success_counts"][...],
                    "joint_config": f["joint_config"][...],
                    "object_center_world": f["object_center_world"][...],
                    "object_quat_world": f["object_quat_world"][...],
                    "side_is_left": f["side_is_left"][...].astype(np.bool_),
                    "correct_region": (
                        f["correct_region"][...].astype(np.bool_)
                        if "correct_region" in f
                        else None
                    ),
                }
            )

    num_categories = len(category_names)
    sample_refs = [[[] for _ in range(num_variants)] for _ in range(num_categories)]
    variant_attempt_counts = np.zeros((num_categories, num_variants), dtype=np.int64)
    variant_success_counts = np.zeros((num_categories, num_variants), dtype=np.int64)

    for global_slot_id, entry in enumerate(generation_payload["entries"]):
        rank = int(entry.get("rank", global_slot_id // generation_envs_per_rank))
        local_env_id = int(entry.get("local_env_id", global_slot_id % generation_envs_per_rank))
        variant_id = int(entry["variant_id"])
        shard = rank_data[rank]
        for category_id in range(num_categories):
            variant_attempt_counts[category_id, variant_id] += int(shard["attempt_counts"][category_id, local_env_id])
            variant_success_counts[category_id, variant_id] += int(shard["success_counts"][category_id, local_env_id])
            stored_count = int(shard["counts"][category_id, local_env_id])
            if stored_count > local_capacity:
                raise RuntimeError(
                    f"Shard count exceeds capacity: rank={rank} local_env_id={local_env_id} "
                    f"category={category_id} stored_count={stored_count} capacity={local_capacity}"
                )
            for sample_idx in range(stored_count):
                if args.region_filter != "all":
                    correct_region = shard["correct_region"]
                    if correct_region is None:
                        raise RuntimeError(
                            "Requested region-filtered repack but shard is missing correct_region dataset. "
                            f"rank={rank} path={shard_path}"
                        )
                    sample_is_correct = bool(correct_region[category_id, local_env_id, sample_idx])
                    if args.region_filter == "correct_only" and (not sample_is_correct):
                        continue
                    if args.region_filter == "wrong_only" and sample_is_correct:
                        continue
                sample_refs[category_id][variant_id].append((rank, local_env_id, sample_idx))

    rng = random.Random(int(args.seed))
    selected_refs = [[[] for _ in range(num_variants)] for _ in range(num_categories)]
    selected_counts = np.zeros((num_categories, num_variants), dtype=np.int32)
    for category_id in range(num_categories):
        for variant_id in range(num_variants):
            refs = list(sample_refs[category_id][variant_id])
            rng.shuffle(refs)
            chosen = refs[: args.target_count]
            selected_refs[category_id][variant_id] = chosen
            selected_counts[category_id, variant_id] = len(chosen)

    if not args.allow_short:
        min_selected_count = int(selected_counts.min()) if selected_counts.size > 0 else 0
        if min_selected_count < int(args.target_count):
            short_rows = np.argwhere(selected_counts < int(args.target_count))
            preview = short_rows[:10].tolist()
            raise RuntimeError(
                f"Not enough collected states to repack target_count={args.target_count}. "
                f"global_min_selected={min_selected_count} first_short_rows={preview}"
            )

    training_entries = training_payload["entries"]
    if len(training_entries) != num_variants:
        raise ValueError(
            f"Training entries length mismatch: expected {num_variants}, got {len(training_entries)}"
        )

    seen_variants = sorted(int(entry["variant_id"]) for entry in training_entries)
    if seen_variants != list(range(num_variants)):
        raise ValueError("Training assignment must contain each variant_id exactly once")

    variant_meta = {
        int(variant["variant_id"]): variant
        for variant in training_payload.get("variants", generation_payload.get("variants", []))
    }

    for rank in range(training_world_size):
        counts = np.zeros((num_categories, training_envs_per_rank), dtype=np.int32)
        attempt_counts = np.zeros((num_categories, training_envs_per_rank), dtype=np.int32)
        success_counts = np.zeros((num_categories, training_envs_per_rank), dtype=np.int32)
        joint_config = np.zeros((num_categories, training_envs_per_rank, args.target_count, num_dofs), dtype=np.float32)
        object_center_world = np.zeros((num_categories, training_envs_per_rank, args.target_count, 3), dtype=np.float32)
        object_quat_world = np.zeros((num_categories, training_envs_per_rank, args.target_count, 4), dtype=np.float32)
        side_is_left = np.zeros((num_categories, training_envs_per_rank, args.target_count), dtype=np.uint8)
        env_object_ids = np.zeros((training_envs_per_rank,), dtype=np.int64)
        variation_ids = np.zeros((training_envs_per_rank,), dtype=np.int64)
        table_surface_height = np.zeros((training_envs_per_rank,), dtype=np.float32)
        table_size = np.zeros((training_envs_per_rank, 3), dtype=np.float32)
        object_mass = np.zeros((training_envs_per_rank,), dtype=np.float32)
        has_object_mass = False

        for local_env_id in range(training_envs_per_rank):
            global_slot_id = rank * training_envs_per_rank + local_env_id
            entry = training_entries[global_slot_id]
            variant_id = int(entry["variant_id"])
            variant = variant_meta.get(
                variant_id,
                {
                    "object_id": int(entry["object_id"]),
                    "z_shift": float(entry["z_shift"]),
                    **({"table_size": list(entry["table_size"])} if "table_size" in entry else {}),
                    **({"object_mass": float(entry["object_mass"])} if "object_mass" in entry else {}),
                },
            )
            env_object_ids[local_env_id] = int(variant["object_id"])
            variation_ids[local_env_id] = variant_id
            table_surface_height[local_env_id] = float(variant["z_shift"])
            if "table_size" in variant:
                table_size[local_env_id] = np.asarray(variant["table_size"], dtype=np.float32)
            if "object_mass" in variant:
                object_mass[local_env_id] = float(variant["object_mass"])
                has_object_mass = True
            for category_id in range(num_categories):
                chosen_refs = selected_refs[category_id][variant_id]
                counts[category_id, local_env_id] = len(chosen_refs)
                attempt_counts[category_id, local_env_id] = int(variant_attempt_counts[category_id, variant_id])
                success_counts[category_id, local_env_id] = int(variant_success_counts[category_id, variant_id])
                for write_idx, (src_rank, src_env_id, src_sample_idx) in enumerate(chosen_refs):
                    shard = rank_data[src_rank]
                    joint_config[category_id, local_env_id, write_idx] = shard["joint_config"][category_id, src_env_id, src_sample_idx]
                    object_center_world[category_id, local_env_id, write_idx] = shard["object_center_world"][category_id, src_env_id, src_sample_idx]
                    object_quat_world[category_id, local_env_id, write_idx] = shard["object_quat_world"][category_id, src_env_id, src_sample_idx]
                    side_is_left[category_id, local_env_id, write_idx] = np.uint8(shard["side_is_left"][category_id, src_env_id, src_sample_idx])

        shard_path = _get_shard_path(output_base, rank)
        with h5py.File(shard_path, "w") as f:
            f.attrs["num_envs"] = int(training_envs_per_rank)
            f.attrs["num_dofs"] = int(num_dofs)
            f.attrs["capacity_per_env"] = int(args.target_count)
            f.attrs["category_names_json"] = json.dumps(list(category_names))
            f.attrs["num_variants_global"] = int(num_variants)
            f.create_dataset("counts", data=counts)
            f.create_dataset("attempt_counts", data=attempt_counts)
            f.create_dataset("success_counts", data=success_counts)
            f.create_dataset("joint_config", data=joint_config)
            f.create_dataset("object_center_world", data=object_center_world)
            f.create_dataset("object_quat_world", data=object_quat_world)
            f.create_dataset("side_is_left", data=side_is_left)
            f.create_dataset("env_object_ids", data=env_object_ids)
            f.create_dataset("variation_id", data=variation_ids)
            f.create_dataset("height_bin_idx", data=variation_ids)
            f.create_dataset("table_surface_height", data=table_surface_height)
            if np.any(table_size):
                f.create_dataset("table_size", data=table_size)
            if has_object_mass:
                f.create_dataset("object_mass", data=object_mass)

    for category_id, category_name in enumerate(category_names):
        category_min = int(selected_counts[category_id].min()) if selected_counts.shape[1] > 0 else 0
        category_max = int(selected_counts[category_id].max()) if selected_counts.shape[1] > 0 else 0
        category_total = int(selected_counts[category_id].sum())
        print(
            f"[repack_verified_teacher_bank] category={category_name} "
            f"selected_total={category_total} selected_min_per_variant={category_min} "
            f"selected_max_per_variant={category_max} target_count={args.target_count} "
            f"region_filter={args.region_filter}"
        )

    print(
        f"Wrote repacked training shards base={output_base} "
        f"world_size={training_world_size} envs_per_rank={training_envs_per_rank}"
    )


if __name__ == "__main__":
    main()
