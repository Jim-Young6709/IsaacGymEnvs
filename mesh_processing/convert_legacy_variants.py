#!/usr/bin/env python3
import argparse
import os
import shutil
from pathlib import Path


def is_variant_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit()


def has_legacy_object_files(object_dir: Path) -> bool:
    for child in object_dir.iterdir():
        if child.is_file() and child.suffix.lower() in {".json", ".obj", ".glb", ".npy", ".urdf"}:
            return True
    return False


def convert_object_dir(object_dir: Path, dry_run: bool) -> str:
    if not object_dir.is_dir():
        return "skip:not-dir"

    variant_dirs = [child for child in object_dir.iterdir() if is_variant_dir(child)]
    if variant_dirs:
        return "skip:already-variant"

    if not has_legacy_object_files(object_dir):
        return "skip:no-legacy-files"

    target_dir = object_dir / "v000"
    if target_dir.exists():
        return "skip:v000-exists"

    legacy_files = [child for child in object_dir.iterdir() if child.is_file()]
    if not legacy_files:
        return "skip:no-files"

    if dry_run:
        return "would-convert"

    target_dir.mkdir()
    for child in legacy_files:
        shutil.move(str(child), str(target_dir / child.name))
    return "converted"


def convert_root(root: Path, dry_run: bool) -> dict:
    summary = {
        "converted": 0,
        "would_convert": 0,
        "skip_already_variant": 0,
        "skip_no_legacy_files": 0,
        "skip_v000_exists": 0,
        "skip_not_dir": 0,
        "skip_no_files": 0,
    }

    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root}")

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        for object_dir in sorted(category_dir.iterdir()):
            if not object_dir.is_dir():
                continue
            result = convert_object_dir(object_dir, dry_run)
            if result == "converted":
                summary["converted"] += 1
            elif result == "would-convert":
                summary["would_convert"] += 1
            elif result == "skip:already-variant":
                summary["skip_already_variant"] += 1
            elif result == "skip:no-legacy-files":
                summary["skip_no_legacy_files"] += 1
            elif result == "skip:v000-exists":
                summary["skip_v000_exists"] += 1
            elif result == "skip:not-dir":
                summary["skip_not_dir"] += 1
            elif result == "skip:no-files":
                summary["skip_no_files"] += 1
            print(f"{category_dir.name}/{object_dir.name}: {result}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert legacy mesh output layout into v000 variant layout."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default="/home/rayliu/grogu/IsaacGymEnvs/meshes_small",
        help="Legacy output root to convert in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without moving files.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    summary = convert_root(root, dry_run=bool(args.dry_run))
    print("\nSummary")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
