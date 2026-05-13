import hashlib
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np


def _ensure_conda_libpython_visible():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    conda_lib = os.path.join(conda_prefix, "lib")
    libpython = os.path.join(conda_lib, "libpython3.8.so.1.0")
    if not os.path.isfile(libpython):
        return

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_entries = [entry for entry in ld_library_path.split(":") if entry]
    if conda_lib in ld_entries:
        return

    if os.environ.get("_ISAACGYM_DEMO_REEXEC") == "1":
        return

    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = conda_lib if not ld_library_path else f"{conda_lib}:{ld_library_path}"
    new_env["_ISAACGYM_DEMO_REEXEC"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_ROOT = REPO_ROOT / "meshes_long" / "default"
gymapi = None
gymutil = None


def _load_isaacgym():
    global gymapi, gymutil
    if gymapi is not None and gymutil is not None:
        return
    _ensure_conda_libpython_visible()
    from isaacgym import gymapi as _gymapi
    from isaacgym import gymutil as _gymutil

    gymapi = _gymapi
    gymutil = _gymutil


def _rotation_abs_from_meta(meta):
    transform = meta.get("transform", {})
    rx = np.deg2rad(float(transform.get("rot_x", 0.0)))
    ry = np.deg2rad(float(transform.get("rot_y", 0.0)))
    rz = np.deg2rad(float(transform.get("rot_z", 0.0)))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.asarray([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot_y = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_z = np.asarray([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return np.abs(rot_z @ rot_y @ rot_x)


def _baked_dilation_from_meta(meta, json_path):
    transform = meta.get("transform", {})
    keys = ("dilate_x", "dilate_y", "dilate_z")
    if all(k in transform for k in keys):
        return np.asarray([float(transform[k]) for k in keys], dtype=np.float32)

    stats = meta.get("stats", {})
    original_extents = np.asarray(stats.get("original", {}).get("bbox_extents", []), dtype=np.float32)
    scaled_extents = np.asarray(stats.get("scaled", {}).get("bbox_extents", []), dtype=np.float32)
    if original_extents.shape[0] == 3 and scaled_extents.shape[0] == 3:
        if np.all(np.abs(original_extents) > 1.0e-8):
            return scaled_extents / original_extents

    dilation_range = meta.get("dilation_range", {})
    try:
        return np.asarray(
            [
                0.5 * (float(dilation_range["x"]["min"]) + float(dilation_range["x"]["max"])),
                0.5 * (float(dilation_range["y"]["min"]) + float(dilation_range["y"]["max"])),
                0.5 * (float(dilation_range["z"]["min"]) + float(dilation_range["z"]["max"])),
            ],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise ValueError(f"Cannot infer baked dilation from variant metadata: {json_path}") from exc


def _discover_variant_entries(mesh_root):
    entries = []
    for root, _, files in os.walk(mesh_root):
        obj_files = sorted([f for f in files if f.endswith(".obj")])
        if not obj_files:
            continue

        rel_dir = os.path.relpath(root, mesh_root)
        parts = rel_dir.split(os.sep)
        if len(parts) == 2:
            object_name, variant = parts
            mesh_rel_prefix = os.path.join(object_name, variant)
        elif len(parts) == 3:
            _, object_name, variant = parts
            mesh_rel_prefix = os.path.join(*parts)
        else:
            continue

        for obj_file in obj_files:
            if not obj_file.startswith(f"{object_name}_{variant}"):
                continue
            mesh_stem = os.path.splitext(obj_file)[0]
            json_path = Path(root) / f"{mesh_stem}.json"
            urdf_path = Path(root) / f"{mesh_stem}.urdf"
            obj_path = Path(root) / obj_file
            if not json_path.is_file() or not urdf_path.is_file():
                continue

            meta = json.loads(json_path.read_text())
            if meta.get("transform_model") != "dilation_only_v1":
                continue

            entries.append(
                {
                    "display_name": f"{object_name}_{variant}",
                    "object_name": object_name,
                    "variant": variant,
                    "mesh_rel_prefix": mesh_rel_prefix,
                    "obj_path": obj_path,
                    "urdf_path": urdf_path,
                    "json_path": json_path,
                    "dilation_range": meta["dilation_range"],
                    "baked_dilation": _baked_dilation_from_meta(meta, json_path),
                    "baked_rotation_abs": _rotation_abs_from_meta(meta),
                }
            )
    return sorted(entries, key=lambda e: (e["display_name"], str(e["obj_path"])))


def _sample_dilation_and_relative_scale(entry, rng):
    dilation_range = entry["dilation_range"]
    target_dilation = np.asarray(
        [
            rng.uniform(float(dilation_range["x"]["min"]), float(dilation_range["x"]["max"])),
            rng.uniform(float(dilation_range["y"]["min"]), float(dilation_range["y"]["max"])),
            rng.uniform(float(dilation_range["z"]["min"]), float(dilation_range["z"]["max"])),
        ],
        dtype=np.float32,
    )
    baked_dilation = np.asarray(entry["baked_dilation"], dtype=np.float32)
    raw_relative_scale = target_dilation / baked_dilation
    relative_scale = np.asarray(entry["baked_rotation_abs"], dtype=np.float32) @ raw_relative_scale
    return target_dilation, relative_scale


def _obj_bounds(obj_path, scale_xyz):
    mins = np.full(3, np.inf, dtype=np.float32)
    maxs = np.full(3, -np.inf, dtype=np.float32)
    num_vertices = 0
    scale_xyz = np.asarray(scale_xyz, dtype=np.float32).reshape(3)
    with obj_path.open("r") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32) * scale_xyz
            mins = np.minimum(mins, xyz)
            maxs = np.maximum(maxs, xyz)
            num_vertices += 1
    if num_vertices == 0:
        raise RuntimeError(f"No OBJ vertices found in {obj_path}.")
    return mins, maxs


def _patch_urdf_to_cache(source_urdf_path, scale_xyz, cache_root):
    source_dir = source_urdf_path.parent
    mesh_name = source_urdf_path.stem
    scale_xyz = np.asarray(scale_xyz, dtype=np.float32).reshape(3)
    scale_tag = "_".join([f"{float(v):.6g}".replace(".", "p").replace("-", "m") for v in scale_xyz])
    dir_hash = hashlib.sha1(str(source_dir).encode("utf-8")).hexdigest()[:10]
    cache_dir = Path(cache_root).expanduser().resolve() / f"{mesh_name}_{dir_hash}_scale_{scale_tag}"
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    if not cache_dir.exists():
        shutil.copytree(source_dir, cache_dir)

    urdf_rel = source_urdf_path.name
    cache_urdf_path = cache_dir / urdf_rel
    urdf_text = source_urdf_path.read_text()
    scale_str = f"{float(scale_xyz[0]):.8g} {float(scale_xyz[1]):.8g} {float(scale_xyz[2]):.8g}"
    patched, n_scale = re.subn(
        r'(<mesh\b[^>]*\bscale\s*=\s*")[^"]+(")',
        rf'\g<1>{scale_str}\2',
        urdf_text,
    )
    if n_scale == 0:
        raise ValueError(f"No <mesh ... scale=\"...\"> tag found in {source_urdf_path}")

    def _scale_origin(match):
        xyz = np.fromstring(match.group(2), sep=" ", dtype=np.float32)
        if xyz.shape[0] != 3:
            raise ValueError(f"Expected origin xyz to have 3 values in URDF: {source_urdf_path}")
        scaled_xyz = xyz * scale_xyz
        xyz_str = f"{float(scaled_xyz[0]):.8g} {float(scaled_xyz[1]):.8g} {float(scaled_xyz[2]):.8g}"
        return match.group(1) + xyz_str + match.group(3)

    patched, _ = re.subn(r'(<origin\b[^>]*\bxyz\s*=\s*")([^"]+)(")', _scale_origin, patched)
    cache_urdf_path.write_text(patched)
    return cache_dir, urdf_rel


def _create_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim.")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)
    return sim


def _create_table_asset(gym, sim, table_size):
    opts = gymapi.AssetOptions()
    opts.fix_base_link = True
    return gym.create_box(sim, *table_size, opts)


def _load_mesh_asset(gym, sim, asset_root, asset_file):
    opts = gymapi.AssetOptions()
    opts.fix_base_link = True
    opts.disable_gravity = True
    return gym.load_asset(sim, str(asset_root), asset_file, opts)


def _select_entries(entries, num_variants, seed):
    rng = random.Random(seed)
    if num_variants >= len(entries):
        return entries
    indices = sorted(rng.sample(range(len(entries)), num_variants))
    return [entries[i] for i in indices]


def main():
    _load_isaacgym()
    args = gymutil.parse_arguments(
        description="Display sampled dilated versions from meshes_long/default on a table.",
        custom_parameters=[
            {"name": "--mesh-root", "type": str, "default": str(DEFAULT_MESH_ROOT), "help": "Variant mesh root, e.g. meshes_long/default."},
            {"name": "--num-variants", "type": int, "default": 6, "help": "Number of source variants to sample."},
            {"name": "--versions-per-variant", "type": int, "default": 4, "help": "Number of sampled dilation versions per selected variant."},
            {"name": "--seed", "type": int, "default": 0, "help": "Random seed for variant and dilation sampling."},
            {"name": "--cache-root", "type": str, "default": "", "help": "URDF cache root. Defaults to ISAACGYM_URDF_CACHE_ROOT or ~/.cache/isaacgym_urdf_overrides/variant_table_demo."},
            {"name": "--table-length", "type": float, "default": 1.8, "help": "Table size along x."},
            {"name": "--table-width", "type": float, "default": 1.2, "help": "Table size along y."},
            {"name": "--table-height", "type": float, "default": 0.75, "help": "Table surface height."},
        ],
    )

    mesh_root = Path(args.mesh_root).expanduser().resolve()
    entries = _discover_variant_entries(mesh_root)
    if not entries:
        raise RuntimeError(f"No dilation variant entries found under {mesh_root}.")
    selected = _select_entries(entries, max(1, int(args.num_variants)), int(args.seed))

    cache_root = args.cache_root
    if not cache_root:
        cache_root = os.environ.get(
            "ISAACGYM_URDF_CACHE_ROOT",
            os.path.join(os.path.expanduser("~"), ".cache", "isaacgym_urdf_overrides", "variant_table_demo"),
        )

    gym = gymapi.acquire_gym()
    sim = _create_sim(gym, args)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer.")

    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)
    table_thickness = 0.08
    table_surface_height = float(args.table_height)
    table_size = [float(args.table_length), float(args.table_width), table_thickness]
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, table_surface_height - table_thickness / 2.0)
    table_asset = _create_table_asset(gym, sim, table_size)
    gym.create_actor(env, table_asset, table_pose, "display_table", 0, 0, 0)

    rng = np.random.default_rng(int(args.seed))
    num_rows = len(selected)
    num_cols = max(1, int(args.versions_per_variant))
    x_margin = 0.16
    y_margin = 0.14
    usable_x = float(args.table_length) - 2.0 * x_margin
    usable_y = float(args.table_width) - 2.0 * y_margin
    row_xs = [-usable_x / 2.0 + (i + 0.5) * (usable_x / num_rows) for i in range(num_rows)]
    col_ys = [-usable_y / 2.0 + (j + 0.5) * (usable_y / num_cols) for j in range(num_cols)]

    manifest = []
    print(f"Displaying {len(selected)} variants from {mesh_root}")
    for row_idx, entry in enumerate(selected):
        print(f"  row {row_idx}: {entry['display_name']}")
        for col_idx in range(num_cols):
            target_dilation, relative_scale = _sample_dilation_and_relative_scale(entry, rng)
            cache_dir, urdf_rel = _patch_urdf_to_cache(entry["urdf_path"], relative_scale, cache_root)
            bbox_min, bbox_max = _obj_bounds(entry["obj_path"], relative_scale)
            bbox_extents = bbox_max - bbox_min

            asset = _load_mesh_asset(gym, sim, cache_dir, urdf_rel)
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                row_xs[row_idx],
                col_ys[col_idx],
                table_surface_height - float(bbox_min[2]) + 0.002,
            )
            pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            actor_name = f"{entry['object_name']}_{entry['variant']}_sample{col_idx:02d}"
            actor = gym.create_actor(env, asset, pose, actor_name, 0, 1, 0)

            color = [
                0.35 + 0.10 * (row_idx % 4),
                0.45 + 0.10 * (col_idx % 4),
                0.85 - 0.08 * (row_idx % 5),
            ]
            for body_idx in range(gym.get_actor_rigid_body_count(env, actor)):
                gym.set_rigid_body_color(env, actor, body_idx, gymapi.MESH_VISUAL, gymapi.Vec3(*color))

            print(
                f"    v{col_idx}: dilation={target_dilation.round(4).tolist()} "
                f"scale={relative_scale.round(4).tolist()} bbox={bbox_extents.round(4).tolist()}"
            )
            manifest.append(
                {
                    "actor": actor_name,
                    "display_name": entry["display_name"],
                    "source_obj": str(entry["obj_path"]),
                    "cache_urdf": str(cache_dir / urdf_rel),
                    "target_dilation": target_dilation.tolist(),
                    "relative_scale": relative_scale.tolist(),
                    "bbox_extents": bbox_extents.tolist(),
                    "table_xy": [row_xs[row_idx], col_ys[col_idx]],
                }
            )

    manifest_path = Path(cache_root).expanduser().resolve() / "variant_table_demo_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest: {manifest_path}")

    cam_pos = gymapi.Vec3(-1.9, 0.0, 1.25)
    cam_target = gymapi.Vec3(0.0, 0.0, table_surface_height)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    gym.prepare_sim(sim)

    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
