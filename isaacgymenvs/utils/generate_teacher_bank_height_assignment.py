import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Generate a shared teacher-bank variation file with per-slot object_id and z_shift."
    )
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--num-objects", type=int, required=True)
    parser.add_argument("--z-min", type=float, required=True)
    parser.add_argument("--z-max", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    if args.num_envs <= 0:
        raise ValueError("num-envs must be positive")
    if args.num_objects <= 0:
        raise ValueError("num-objects must be positive")
    if args.z_max < args.z_min:
        raise ValueError("z-max must be >= z-min")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(args.seed))
    z_shift = torch.rand(args.num_envs, generator=generator, dtype=torch.float32)
    z_shift = z_shift * (args.z_max - args.z_min) + args.z_min

    entries = []
    for env_id in range(args.num_envs):
        object_id = env_id % args.num_objects
        repeat_idx = env_id // args.num_objects
        entries.append(
            {
                "env_id": env_id,
                "object_id": object_id,
                "repeat_idx": repeat_idx,
                "z_shift": float(z_shift[env_id].item()),
            }
        )

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "num_envs": int(args.num_envs),
        "num_objects": int(args.num_objects),
        "z_shift_range": [float(args.z_min), float(args.z_max)],
        "seed": int(args.seed),
        "entries": entries,
        "z_shift": [entry["z_shift"] for entry in entries],
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)

    repeats_per_object_floor = args.num_envs // args.num_objects
    print(
        "Wrote teacher bank variation assignment JSON "
        f"path={out_path} num_envs={args.num_envs} num_objects={args.num_objects} "
        f"repeats_per_object_floor={repeats_per_object_floor}"
    )


if __name__ == "__main__":
    main()
