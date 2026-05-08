import argparse
import json
from pathlib import Path

import torch


def _resolve_out_path(path_str):
    out_path = Path(path_str).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _write_payload(path, payload):
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate shared variation-assignment JSON files for verified teacher-bank "
            "generation and training."
        )
    )
    parser.add_argument("--num-variants", type=int, required=True)
    parser.add_argument("--num-objects", type=int, required=True)
    parser.add_argument("--z-min", type=float, required=True)
    parser.add_argument("--z-max", type=float, required=True)
    parser.add_argument("--table-size-min", type=float, nargs=3, default=None)
    parser.add_argument("--table-size-max", type=float, nargs=3, default=None)
    parser.add_argument("--object-mass-min", type=float, default=None)
    parser.add_argument("--object-mass-max", type=float, default=None)
    parser.add_argument("--replication-factor", type=int, required=True)
    parser.add_argument("--generation-world-size", type=int, required=True)
    parser.add_argument("--generation-envs-per-rank", type=int, required=True)
    parser.add_argument("--training-world-size", type=int, required=True)
    parser.add_argument("--training-envs-per-rank", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--generation-out", type=str, required=True)
    parser.add_argument("--training-out", type=str, required=True)
    args = parser.parse_args()

    if args.num_variants <= 0:
        raise ValueError("num-variants must be positive")
    if args.num_objects <= 0:
        raise ValueError("num-objects must be positive")
    if args.z_max < args.z_min:
        raise ValueError("z-max must be >= z-min")
    if (args.table_size_min is None) != (args.table_size_max is None):
        raise ValueError("table-size-min and table-size-max must be provided together")
    if args.table_size_min is not None:
        for i, (mn, mx) in enumerate(zip(args.table_size_min, args.table_size_max)):
            if mx < mn:
                raise ValueError(f"table-size-max[{i}] must be >= table-size-min[{i}]")
    if (args.object_mass_min is None) != (args.object_mass_max is None):
        raise ValueError("object-mass-min and object-mass-max must be provided together")
    if args.object_mass_min is not None and args.object_mass_max < args.object_mass_min:
        raise ValueError("object-mass-max must be >= object-mass-min")
    if args.replication_factor <= 0:
        raise ValueError("replication-factor must be positive")
    if args.generation_world_size <= 0 or args.generation_envs_per_rank <= 0:
        raise ValueError("generation world size and envs per rank must be positive")
    if args.training_world_size <= 0 or args.training_envs_per_rank <= 0:
        raise ValueError("training world size and envs per rank must be positive")

    generation_total_slots = args.generation_world_size * args.generation_envs_per_rank
    expected_generation_slots = args.num_variants * args.replication_factor
    if generation_total_slots != expected_generation_slots:
        raise ValueError(
            f"generation slots mismatch: world_size*envs_per_rank={generation_total_slots} "
            f"but num_variants*replication_factor={expected_generation_slots}"
        )

    training_total_slots = args.training_world_size * args.training_envs_per_rank
    if training_total_slots != args.num_variants:
        raise ValueError(
            f"training slots mismatch: world_size*envs_per_rank={training_total_slots} "
            f"but num_variants={args.num_variants}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))

    z_shift = torch.rand(args.num_variants, generator=generator, dtype=torch.float32)
    z_shift = z_shift * (args.z_max - args.z_min) + args.z_min
    table_size = None
    if args.table_size_min is not None:
        table_size_min = torch.tensor(args.table_size_min, dtype=torch.float32)
        table_size_max = torch.tensor(args.table_size_max, dtype=torch.float32)
        table_size = torch.rand(args.num_variants, 3, generator=generator, dtype=torch.float32)
        table_size = table_size * (table_size_max - table_size_min) + table_size_min
    object_mass = None
    if args.object_mass_min is not None:
        object_mass = torch.rand(args.num_variants, generator=generator, dtype=torch.float32)
        object_mass = object_mass * (args.object_mass_max - args.object_mass_min) + args.object_mass_min

    object_id_values = torch.arange(args.num_variants, dtype=torch.long) % args.num_objects
    object_id_values = object_id_values[torch.randperm(args.num_variants, generator=generator)]

    variants = []
    for variant_id in range(args.num_variants):
        variants.append(
            {
                "variant_id": variant_id,
                "object_id": int(object_id_values[variant_id].item()),
                "z_shift": float(z_shift[variant_id].item()),
                **(
                    {
                        "table_size": [float(v) for v in table_size[variant_id].tolist()],
                    }
                    if table_size is not None
                    else {}
                ),
                **(
                    {
                        "object_mass": float(object_mass[variant_id].item()),
                    }
                    if object_mass is not None
                    else {}
                ),
            }
        )

    generation_rank_entries = [[] for _ in range(args.generation_world_size)]
    for variant in variants:
        variant_id = int(variant["variant_id"])
        start_rank = variant_id % args.generation_world_size
        for copy_idx in range(args.replication_factor):
            rank = (start_rank + copy_idx) % args.generation_world_size
            generation_rank_entries[rank].append(
                {
                    "variant_id": variant_id,
                    "object_id": int(variant["object_id"]),
                    "z_shift": float(variant["z_shift"]),
                    **({"table_size": list(variant["table_size"])} if "table_size" in variant else {}),
                    **({"object_mass": float(variant["object_mass"])} if "object_mass" in variant else {}),
                    "copy_idx": copy_idx,
                }
            )

    for rank in range(args.generation_world_size):
        perm = torch.randperm(len(generation_rank_entries[rank]), generator=generator).tolist()
        generation_rank_entries[rank] = [generation_rank_entries[rank][idx] for idx in perm]
        if len(generation_rank_entries[rank]) != args.generation_envs_per_rank:
            raise RuntimeError(
                f"generation rank {rank} entry count mismatch: expected {args.generation_envs_per_rank}, "
                f"got {len(generation_rank_entries[rank])}"
            )

    generation_entries = []
    for rank in range(args.generation_world_size):
        for local_env_id, entry in enumerate(generation_rank_entries[rank]):
            entry = dict(entry)
            entry["rank"] = rank
            entry["local_env_id"] = local_env_id
            entry["global_slot_id"] = rank * args.generation_envs_per_rank + local_env_id
            generation_entries.append(entry)

    train_variant_order = torch.randperm(args.num_variants, generator=generator).tolist()
    training_entries = []
    for global_slot_id, variant_id in enumerate(train_variant_order):
        rank = global_slot_id // args.training_envs_per_rank
        local_env_id = global_slot_id % args.training_envs_per_rank
        variant = variants[int(variant_id)]
        training_entries.append(
            {
                "variant_id": int(variant_id),
                "object_id": int(variant["object_id"]),
                "z_shift": float(variant["z_shift"]),
                **({"table_size": list(variant["table_size"])} if "table_size" in variant else {}),
                **({"object_mass": float(variant["object_mass"])} if "object_mass" in variant else {}),
                "rank": rank,
                "local_env_id": local_env_id,
                "global_slot_id": global_slot_id,
            }
        )

    generation_payload = {
        "format": "teacher_bank_variations_v1",
        "mode": "generation",
        "seed": int(args.seed),
        "num_variants": int(args.num_variants),
        "num_objects": int(args.num_objects),
        "replication_factor": int(args.replication_factor),
        "world_size": int(args.generation_world_size),
        "envs_per_rank": int(args.generation_envs_per_rank),
        "z_shift_range": [float(args.z_min), float(args.z_max)],
        **(
            {
                "table_size_range": [
                    [float(v) for v in args.table_size_min],
                    [float(v) for v in args.table_size_max],
                ]
            }
            if args.table_size_min is not None
            else {}
        ),
        **(
            {
                "object_mass_range": [float(args.object_mass_min), float(args.object_mass_max)],
            }
            if args.object_mass_min is not None
            else {}
        ),
        "variants": variants,
        "entries": generation_entries,
    }
    training_payload = {
        "format": "teacher_bank_variations_v1",
        "mode": "training",
        "seed": int(args.seed),
        "num_variants": int(args.num_variants),
        "num_objects": int(args.num_objects),
        "world_size": int(args.training_world_size),
        "envs_per_rank": int(args.training_envs_per_rank),
        "z_shift_range": [float(args.z_min), float(args.z_max)],
        **(
            {
                "table_size_range": [
                    [float(v) for v in args.table_size_min],
                    [float(v) for v in args.table_size_max],
                ]
            }
            if args.table_size_min is not None
            else {}
        ),
        **(
            {
                "object_mass_range": [float(args.object_mass_min), float(args.object_mass_max)],
            }
            if args.object_mass_min is not None
            else {}
        ),
        "variants": variants,
        "entries": training_entries,
    }

    generation_out_path = _resolve_out_path(args.generation_out)
    training_out_path = _resolve_out_path(args.training_out)
    _write_payload(generation_out_path, generation_payload)
    _write_payload(training_out_path, training_payload)

    print(
        "Wrote generation variation assignment "
        f"path={generation_out_path} num_variants={args.num_variants} "
        f"replication_factor={args.replication_factor} total_slots={generation_total_slots}"
    )
    print(
        "Wrote training variation assignment "
        f"path={training_out_path} num_variants={args.num_variants} total_slots={training_total_slots}"
    )


if __name__ == "__main__":
    main()
