import os
import sys
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


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


_ensure_conda_libpython_visible()

from isaacgym import gymapi, gymutil


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH_ROOT = REPO_ROOT / "meshes_side" / "long"


def _load_obj_bbox_extents(obj_path: Path, scale_xyz):
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    num_vertices = 0

    with obj_path.open("r") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            xyz = [float(parts[1]) * scale_xyz[0], float(parts[2]) * scale_xyz[1], float(parts[3]) * scale_xyz[2]]
            for i in range(3):
                mins[i] = min(mins[i], xyz[i])
                maxs[i] = max(maxs[i], xyz[i])
            num_vertices += 1

    if num_vertices == 0:
        raise RuntimeError(f"No OBJ vertices found in {obj_path}.")

    return [maxs[i] - mins[i] for i in range(3)]


def _load_bbox_extents(mesh_dir: Path):
    json_path = mesh_dir / f"{mesh_dir.name}.json"
    if json_path.is_file():
        meta = json.loads(json_path.read_text())
        return meta["scaled_stats"]["bbox_extents"]

    urdf_path = mesh_dir / f"{mesh_dir.name}.urdf"
    tree = ET.parse(urdf_path)
    mesh_elem = tree.find(".//mesh")
    if mesh_elem is None:
        raise RuntimeError(f"No <mesh> element found in {urdf_path}.")

    mesh_filename = mesh_elem.attrib["filename"]
    scale_str = mesh_elem.attrib.get("scale", "1 1 1")
    scale_xyz = [float(v) for v in scale_str.split()]
    obj_path = mesh_dir / mesh_filename
    return _load_obj_bbox_extents(obj_path, scale_xyz)


def _load_mesh_entries(mesh_root: Path):
    entries = []
    for mesh_dir in sorted(p for p in mesh_root.iterdir() if p.is_dir()):
        urdf_path = mesh_dir / f"{mesh_dir.name}.urdf"
        if not urdf_path.is_file():
            continue

        bbox_extents = _load_bbox_extents(mesh_dir)
        entries.append(
            {
                "name": mesh_dir.name,
                "asset_root": str(mesh_dir),
                "asset_file": urdf_path.name,
                "bbox_extents": bbox_extents,
                "height": float(bbox_extents[2]),
            }
        )
    return entries


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
    return gym.load_asset(sim, asset_root, asset_file, opts)


def main():
    args = gymutil.parse_arguments(
        description="Display all meshes in a mesh directory on a large table.",
        custom_parameters=[
            {"name": "--mesh-root", "type": str, "default": str(DEFAULT_MESH_ROOT), "help": "Mesh directory containing one subdirectory per object with URDF/OBJ."},
            {"name": "--rows", "type": int, "default": 4, "help": "Number of table rows from front to back."},
            {"name": "--table-length", "type": float, "default": 1.8, "help": "Table size along x."},
            {"name": "--table-width", "type": float, "default": 1.1, "help": "Table size along y."},
            {"name": "--table-height", "type": float, "default": 0.75, "help": "Table surface height."},
        ],
    )

    gym = gymapi.acquire_gym()
    sim = _create_sim(gym, args)

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer.")

    mesh_root = Path(args.mesh_root).expanduser().resolve()
    mesh_entries = _load_mesh_entries(mesh_root)
    if not mesh_entries:
        raise RuntimeError(f"No mesh URDFs found under {mesh_root}.")

    mesh_entries.sort(key=lambda x: x["height"])

    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)

    table_thickness = 0.08
    table_size = [args.table_length, args.table_width, table_thickness]
    table_surface_height = args.table_height
    table_center = [0.0, 0.0, table_surface_height - table_thickness / 2.0]

    table_asset = _create_table_asset(gym, sim, table_size)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(*table_center)
    gym.create_actor(env, table_asset, table_pose, "display_table", 0, 0, 0)

    num_rows = max(1, int(args.rows))
    items_per_row = math.ceil(len(mesh_entries) / num_rows)

    x_margin = 0.16
    y_margin = 0.14
    usable_x = args.table_length - 2.0 * x_margin
    usable_y = args.table_width - 2.0 * y_margin

    row_xs = [
        -usable_x / 2.0 + (i + 0.5) * (usable_x / num_rows)
        for i in range(num_rows)
    ]

    print(f"Displaying meshes from {mesh_root} front-to-back by bbox z height:")
    for row_idx in range(num_rows):
        row_entries = mesh_entries[row_idx * items_per_row : (row_idx + 1) * items_per_row]
        if not row_entries:
            continue
        n_cols = len(row_entries)
        col_ys = [
            -usable_y / 2.0 + (j + 0.5) * (usable_y / n_cols)
            for j in range(n_cols)
        ]
        print(f"  row {row_idx} x={row_xs[row_idx]:+.3f}: " + ", ".join(f"{e['name']}({e['height']:.3f}m)" for e in row_entries))
        for col_idx, entry in enumerate(row_entries):
            asset = _load_mesh_asset(gym, sim, entry["asset_root"], entry["asset_file"])
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(row_xs[row_idx], col_ys[col_idx], table_surface_height + 0.002)
            actor = gym.create_actor(env, asset, pose, entry["name"], 0, 1, 0)

            # Give each row a different tint to make the front/back grouping obvious.
            row_color = [
                0.35 + 0.15 * row_idx,
                0.55,
                0.85 - 0.12 * row_idx,
            ]
            num_bodies = gym.get_actor_rigid_body_count(env, actor)
            for body_idx in range(num_bodies):
                gym.set_rigid_body_color(
                    env,
                    actor,
                    body_idx,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(*row_color),
                )

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
