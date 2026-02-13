#!/usr/bin/env python3
"""Convert legacy mesh layout to the per-object-folder layout.

python isaacgymenvs/utils/convert_legacy_meshes.py \
--legacy-root /home/rayliu/grogu/IsaacGymEnvs/meshes_69 \
--output-root /home/rayliu/grogu/IsaacGymEnvs/meshes_69_new \
--overwrite

Legacy layout example:
    <legacy_root>/alarm_clock/2.obj
    <legacy_root>/alarm_clock/2.urdf
    <legacy_root>/alarm_clock/2.npy
    <legacy_root>/alarm_clock/2.glb

New layout example:
    <output_root>/alarm_clock_2/alarm_clock_2.obj
    <output_root>/alarm_clock_2/alarm_clock_2.urdf
    <output_root>/alarm_clock_2/alarm_clock_2.npy
    <output_root>/alarm_clock_2/alarm_clock_2.glb
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert legacy meshes to new layout and scale.")
    parser.add_argument("--legacy-root", type=Path, required=True, help="Root dir of legacy categories.")
    parser.add_argument("--output-root", type=Path, required=True, help="Root dir for converted objects.")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.1,
        help="Scale factor applied to OBJ vertices and NPY point clouds (default: 0.1).",
    )
    parser.add_argument(
        "--glb-mode",
        choices=["regenerate", "rescale", "copy", "skip"],
        default="regenerate",
        help=(
            "How to output GLB: regenerate from scaled OBJ, rescale legacy GLB, copy as-is, or skip. "
            "Default: regenerate."
        ),
    )
    parser.add_argument(
        "--npy-dtype",
        choices=["float32", "keep"],
        default="float32",
        help="Output dtype for NPY point cloud (default: float32).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output object folders.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected file is missing. By default, missing files are skipped with warnings.",
    )
    return parser.parse_args()


def iter_legacy_objects(category_dir: Path):
    """Yield (index_stem, obj_path) for numeric OBJ stems in one category dir."""
    for obj_path in sorted(category_dir.glob("*.obj")):
        stem = obj_path.stem
        if stem.isdigit():
            yield stem, obj_path


def scale_obj(src: Path, dst: Path, scale: float) -> None:
    lines_out = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x = float(parts[1]) * scale
                    y = float(parts[2]) * scale
                    z = float(parts[3]) * scale
                    rest = ""
                    if len(parts) > 4:
                        rest = " " + " ".join(parts[4:])
                    lines_out.append(f"v {x:.8f} {y:.8f} {z:.8f}{rest}\n")
                    continue
            lines_out.append(line)

    with dst.open("w", encoding="utf-8") as f:
        f.writelines(lines_out)


def scale_npy(src: Path, dst: Path, scale: float, dtype_mode: str) -> None:
    arr = np.load(src)
    out = np.array(arr, copy=True)

    # Scale xyz coordinates; supports Nx3 or arrays where last dim starts with xyz.
    if out.ndim == 1 and out.shape[0] >= 3:
        out[:3] *= scale
    elif out.ndim >= 2 and out.shape[-1] >= 3:
        out[..., :3] *= scale
    else:
        raise ValueError(f"Unsupported NPY shape for xyz scaling: {out.shape}")

    if dtype_mode == "float32":
        out = out.astype(np.float32)

    np.save(dst, out)


def rewrite_urdf(src: Path, dst: Path, obj_filename: str) -> None:
    tree = ET.parse(src)
    root = tree.getroot()

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # Handles XML namespaces if present.
        if tag == "mesh":
            elem.set("filename", obj_filename)
            elem.set("scale", "1 1 1")

    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass

    tree.write(dst, encoding="utf-8", xml_declaration=False)


def require_trimesh():
    try:
        import trimesh  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "trimesh is required for --glb-mode regenerate/rescale. "
            "Install trimesh or use --glb-mode copy/skip."
        ) from exc
    return trimesh


def convert_glb(
    src_glb: Path,
    dst_glb: Path,
    dst_obj: Path,
    mode: str,
    scale: float,
) -> None:
    if mode == "skip":
        return
    if mode == "copy":
        shutil.copy2(src_glb, dst_glb)
        return

    trimesh = require_trimesh()

    if mode == "regenerate":
        mesh = trimesh.load(dst_obj, force="mesh")
        mesh.export(dst_glb)
        return

    if mode == "rescale":
        scene_or_mesh = trimesh.load(src_glb, force="scene")
        scene_or_mesh.apply_scale(scale)
        scene_or_mesh.export(dst_glb)
        return

    raise ValueError(f"Unknown GLB mode: {mode}")


def main() -> int:
    args = parse_args()

    legacy_root = args.legacy_root.resolve()
    output_root = args.output_root.resolve()

    if not legacy_root.exists() or not legacy_root.is_dir():
        print(f"[ERROR] Invalid --legacy-root: {legacy_root}")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    category_dirs = sorted([d for d in legacy_root.iterdir() if d.is_dir()])
    if not category_dirs:
        print(f"[WARN] No category directories found in {legacy_root}")

    for category_dir in category_dirs:
        category = category_dir.name
        objects = list(iter_legacy_objects(category_dir))
        if not objects:
            continue

        for idx, obj_src in objects:
            legacy_base = category_dir / idx
            urdf_src = legacy_base.with_suffix(".urdf")
            npy_src = legacy_base.with_suffix(".npy")
            glb_src = legacy_base.with_suffix(".glb")

            missing = [
                str(p)
                for p in [obj_src, urdf_src, npy_src]
                if not p.exists()
            ]
            if missing:
                msg = f"[WARN] Missing required files for {category}/{idx}: {missing}"
                if args.strict:
                    print(msg.replace("[WARN]", "[ERROR]"))
                    return 1
                print(msg)
                skipped += 1
                continue

            full_name = f"{category}_{idx}"
            out_dir = output_root / full_name

            if out_dir.exists():
                if not args.overwrite:
                    print(f"[WARN] Output exists, skipping (use --overwrite): {out_dir}")
                    skipped += 1
                    continue
                shutil.rmtree(out_dir)

            out_dir.mkdir(parents=True, exist_ok=False)

            obj_dst = out_dir / f"{full_name}.obj"
            urdf_dst = out_dir / f"{full_name}.urdf"
            npy_dst = out_dir / f"{full_name}.npy"
            glb_dst = out_dir / f"{full_name}.glb"

            scale_obj(obj_src, obj_dst, args.scale)
            rewrite_urdf(urdf_src, urdf_dst, obj_dst.name)
            scale_npy(npy_src, npy_dst, args.scale, args.npy_dtype)

            if glb_src.exists():
                convert_glb(glb_src, glb_dst, obj_dst, args.glb_mode, args.scale)
            else:
                if args.strict and args.glb_mode != "skip":
                    print(f"[ERROR] Missing GLB: {glb_src}")
                    return 1
                print(f"[WARN] GLB not found for {category}/{idx}: {glb_src}")

            converted += 1
            print(f"[OK] Converted {category}/{idx} -> {out_dir}")

    print(f"\nDone. converted={converted}, skipped={skipped}, output_root={output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
