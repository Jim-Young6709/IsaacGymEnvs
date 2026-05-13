import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import demo_long_variant_versions_table as variant_table_demo
from demo_long_variant_versions_table import (
    DEFAULT_MESH_ROOT,
    _create_sim,
    _create_table_asset,
    _discover_variant_entries,
    _ensure_conda_libpython_visible,
    _obj_bounds,
    _patch_urdf_to_cache,
    _sample_dilation_and_relative_scale,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_isaacgym():
    _ensure_conda_libpython_visible()
    from isaacgym import gymapi as _gymapi
    from isaacgym import gymtorch as _gymtorch
    from isaacgym import gymutil as _gymutil
    import torch as _torch

    return _gymapi, _gymtorch, _gymutil, _torch


def _load_dynamic_mesh_asset(gym, sim, asset_root, asset_file):
    opts = gymapi.AssetOptions()
    opts.fix_base_link = False
    opts.disable_gravity = False
    return gym.load_asset(sim, str(asset_root), asset_file, opts)


def _quat_xyzw_up_z(quat):
    x = quat[:, 0]
    y = quat[:, 1]
    return 1.0 - 2.0 * (x * x + y * y)


def _make_candidate(entry, object_idx, candidate_idx, rng):
    target_dilation, relative_scale = _sample_dilation_and_relative_scale(entry, rng)
    return {
        "object_idx": int(object_idx),
        "candidate_idx": int(candidate_idx),
        "object_name": entry["object_name"],
        "variant": entry["variant"],
        "display_name": entry["display_name"],
        "mesh_rel_prefix": entry["mesh_rel_prefix"],
        "obj_path": str(entry["obj_path"]),
        "urdf_path": str(entry["urdf_path"]),
        "target_dilation": target_dilation.tolist(),
        "relative_scale": relative_scale.tolist(),
        "baked_dilation": np.asarray(entry["baked_dilation"], dtype=np.float32).tolist(),
        "baked_rotation_abs": np.asarray(entry["baked_rotation_abs"], dtype=np.float32).tolist(),
    }


def _write_json(path, payload):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def main():
    global gymapi
    gymapi, gymtorch, gymutil, torch = _load_isaacgym()
    variant_table_demo.gymapi = gymapi
    variant_table_demo.gymutil = gymutil
    args = gymutil.parse_arguments(
        description=(
            "Sample dilation candidates from meshes_long/default, test whether they "
            "stand on a table in IsaacGym, and write a fixed 4x/object manifest."
        ),
        custom_parameters=[
            {"name": "--mesh-root", "type": str, "default": str(DEFAULT_MESH_ROOT), "help": "Variant mesh root, e.g. meshes_long/default."},
            {"name": "--candidate-count", "type": int, "default": 10, "help": "Dilation samples to test per source object."},
            {"name": "--select-count", "type": int, "default": 4, "help": "Stable variants to keep per source object."},
            {"name": "--seed", "type": int, "default": 1, "help": "Random seed for dilation sampling."},
            {"name": "--settle-steps", "type": int, "default": 300, "help": "Physics steps to wait before checking stability."},
            {"name": "--max-tilt-deg", "type": float, "default": 12.0, "help": "Maximum final tilt from upright."},
            {"name": "--max-z-drop", "type": float, "default": 0.05, "help": "Maximum allowed root z drop."},
            {"name": "--max-xy-drift", "type": float, "default": 0.10, "help": "Maximum allowed XY drift."},
            {"name": "--max-linear-vel", "type": float, "default": 0.05, "help": "Maximum final linear speed."},
            {"name": "--max-angular-vel", "type": float, "default": 0.50, "help": "Maximum final angular speed."},
            {"name": "--allow-best-fill", "action": "store_true", "help": "If fewer than select-count are stable, fill from least-tilted unstable candidates."},
            {"name": "--cache-root", "type": str, "default": "", "help": "URDF cache root. Defaults under ISAACGYM_URDF_CACHE_ROOT."},
            {"name": "--manifest-out", "type": str, "default": "teacher_state_bank_3/side_long_default_stable4_manifest.json", "help": "Output fixed variant manifest."},
            {"name": "--report-out", "type": str, "default": "teacher_state_bank_3/side_long_default_stable4_report.json", "help": "Output full stability report."},
            {"name": "--table-length", "type": float, "default": 0.7, "help": "Table x size."},
            {"name": "--table-width", "type": float, "default": 1.2, "help": "Table y size."},
            {"name": "--table-thickness", "type": float, "default": 0.05, "help": "Table thickness."},
            {"name": "--table-height", "type": float, "default": 0.4, "help": "Table surface height."},
            {"name": "--env-spacing", "type": float, "default": 1.5, "help": "IsaacGym env spacing."},
            {"name": "--headless", "action": "store_true", "help": "Accepted for launch-script compatibility; this script never opens a viewer."},
        ],
    )

    mesh_root = Path(args.mesh_root).expanduser()
    if not mesh_root.is_absolute():
        mesh_root = REPO_ROOT / mesh_root
    mesh_root = mesh_root.resolve()
    entries = _discover_variant_entries(mesh_root)
    if len(entries) == 0:
        raise RuntimeError(f"No dilation variant entries found under {mesh_root}")

    rng = np.random.default_rng(int(args.seed))
    candidates = []
    for object_idx, entry in enumerate(entries):
        for candidate_idx in range(int(args.candidate_count)):
            candidates.append(_make_candidate(entry, object_idx, candidate_idx, rng))

    cache_root = args.cache_root
    if not cache_root:
        cache_root = os.environ.get(
            "ISAACGYM_URDF_CACHE_ROOT",
            os.path.join(os.path.expanduser("~"), ".cache", "isaacgym_urdf_overrides"),
        )
        cache_root = os.path.join(cache_root, "stable_long_mesh_variants")

    gym = gymapi.acquire_gym()
    sim = _create_sim(gym, args)
    table_asset = _create_table_asset(
        gym,
        sim,
        [float(args.table_length), float(args.table_width), float(args.table_thickness)],
    )

    object_actor_indices = []
    init_root_pos = []
    lower = gymapi.Vec3(-float(args.env_spacing), -float(args.env_spacing), 0.0)
    upper = gymapi.Vec3(float(args.env_spacing), float(args.env_spacing), float(args.env_spacing))
    num_per_row = int(math.ceil(math.sqrt(len(candidates))))
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(0.0, 0.0, float(args.table_height) - 0.5 * float(args.table_thickness))

    print(f"Testing {len(candidates)} candidates from {len(entries)} source objects under {mesh_root}")
    for candidate_id, candidate in enumerate(candidates):
        env = gym.create_env(sim, lower, upper, num_per_row)
        gym.create_actor(env, table_asset, table_pose, "table", candidate_id, 1, 0)

        obj_path = Path(candidate["obj_path"])
        urdf_path = Path(candidate["urdf_path"])
        relative_scale = np.asarray(candidate["relative_scale"], dtype=np.float32)
        cache_dir, urdf_rel = _patch_urdf_to_cache(urdf_path, relative_scale, cache_root)
        bbox_min, _ = _obj_bounds(obj_path, relative_scale)

        asset = _load_dynamic_mesh_asset(gym, sim, cache_dir, urdf_rel)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, float(args.table_height) - float(bbox_min[2]) + 0.002)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        actor = gym.create_actor(env, asset, pose, f"object_{candidate_id:04d}", candidate_id, 2, 0)
        object_actor_indices.append(gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM))
        init_root_pos.append([pose.p.x, pose.p.y, pose.p.z])

    gym.prepare_sim(sim)
    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_state = gymtorch.wrap_tensor(root_state_tensor)
    object_actor_indices_t = torch.tensor(object_actor_indices, dtype=torch.long, device=root_state.device)
    init_root_pos_t = torch.tensor(init_root_pos, dtype=torch.float32, device=root_state.device)

    for _ in range(int(args.settle_steps)):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    gym.refresh_actor_root_state_tensor(sim)
    states = root_state[object_actor_indices_t]
    pos = states[:, :3]
    quat = states[:, 3:7]
    lin_vel = torch.linalg.norm(states[:, 7:10], dim=-1)
    ang_vel = torch.linalg.norm(states[:, 10:13], dim=-1)
    up_z = _quat_xyzw_up_z(quat).clamp(-1.0, 1.0)
    tilt_deg = torch.rad2deg(torch.acos(up_z))
    z_drop = init_root_pos_t[:, 2] - pos[:, 2]
    xy_drift = torch.linalg.norm(pos[:, :2] - init_root_pos_t[:, :2], dim=-1)
    stable = (
        (tilt_deg <= float(args.max_tilt_deg))
        & (z_drop <= float(args.max_z_drop))
        & (xy_drift <= float(args.max_xy_drift))
        & (lin_vel <= float(args.max_linear_vel))
        & (ang_vel <= float(args.max_angular_vel))
    )

    report_by_object = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        metrics = {
            "stable": bool(stable[idx].item()),
            "tilt_deg": float(tilt_deg[idx].item()),
            "z_drop": float(z_drop[idx].item()),
            "xy_drift": float(xy_drift[idx].item()),
            "linear_vel": float(lin_vel[idx].item()),
            "angular_vel": float(ang_vel[idx].item()),
            "final_pos": [float(v) for v in pos[idx].detach().cpu().tolist()],
            "final_quat_xyzw": [float(v) for v in quat[idx].detach().cpu().tolist()],
        }
        candidate = dict(candidate)
        candidate["metrics"] = metrics
        report_by_object[candidate["object_name"]].append(candidate)

    selected_entries = []
    insufficient = []
    for object_name in sorted(report_by_object):
        object_candidates = report_by_object[object_name]
        stable_candidates = [c for c in object_candidates if c["metrics"]["stable"]]
        stable_candidates = sorted(
            stable_candidates,
            key=lambda c: (
                c["metrics"]["tilt_deg"],
                c["metrics"]["xy_drift"],
                c["metrics"]["z_drop"],
                c["candidate_idx"],
            ),
        )
        selected = stable_candidates[: int(args.select_count)]
        if len(selected) < int(args.select_count):
            insufficient.append(
                {
                    "object_name": object_name,
                    "stable_count": len(stable_candidates),
                    "required": int(args.select_count),
                }
            )
            if args.allow_best_fill:
                fallback = sorted(
                    object_candidates,
                    key=lambda c: (
                        c["metrics"]["tilt_deg"],
                        c["metrics"]["xy_drift"],
                        c["metrics"]["z_drop"],
                        c["candidate_idx"],
                    ),
                )
                seen = {(c["object_name"], c["candidate_idx"]) for c in selected}
                for candidate in fallback:
                    key = (candidate["object_name"], candidate["candidate_idx"])
                    if key in seen:
                        continue
                    selected.append(candidate)
                    seen.add(key)
                    if len(selected) >= int(args.select_count):
                        break
        for candidate in selected:
            object_id = len(selected_entries)
            asset_mesh_id = os.path.join(candidate["mesh_rel_prefix"], Path(candidate["obj_path"]).stem)
            selected_entries.append(
                {
                    "object_id": object_id,
                    "asset_obj_id": object_id + 1,
                    "asset_mesh_id": asset_mesh_id,
                    "display_name": f"{candidate['object_name']}_stable{candidate['candidate_idx']:02d}",
                    "base_object_name": candidate["object_name"],
                    "base_variant": candidate["variant"],
                    "candidate_idx": candidate["candidate_idx"],
                    "mesh_path": os.path.relpath(candidate["obj_path"], mesh_root),
                    "target_dilation": candidate["target_dilation"],
                    "relative_scale": candidate["relative_scale"],
                    "baked_dilation": candidate["baked_dilation"],
                    "baked_rotation_abs": candidate["baked_rotation_abs"],
                    "stability": candidate["metrics"],
                }
            )

    report_payload = {
        "format": "teacher_bank_mesh_variant_stability_report_v1",
        "mesh_dir": str(mesh_root),
        "seed": int(args.seed),
        "candidate_count": int(args.candidate_count),
        "select_count": int(args.select_count),
        "settle_steps": int(args.settle_steps),
        "thresholds": {
            "max_tilt_deg": float(args.max_tilt_deg),
            "max_z_drop": float(args.max_z_drop),
            "max_xy_drift": float(args.max_xy_drift),
            "max_linear_vel": float(args.max_linear_vel),
            "max_angular_vel": float(args.max_angular_vel),
        },
        "insufficient": insufficient,
        "objects": {name: vals for name, vals in sorted(report_by_object.items())},
    }
    report_path = _write_json(args.report_out, report_payload)

    if insufficient and not args.allow_best_fill:
        gym.destroy_sim(sim)
        print(f"Wrote report: {report_path}")
        raise RuntimeError(
            "Not enough stable candidates for some objects. "
            "Inspect report or rerun with larger --candidate-count / --allow-best-fill. "
            f"First failures: {insufficient[:5]}"
        )

    manifest_payload = {
        "format": "teacher_bank_mesh_variant_manifest_v1",
        "mesh_dir": str(mesh_root),
        "seed": int(args.seed),
        "source_object_count": len(entries),
        "selected_per_object": int(args.select_count),
        "num_objects": len(selected_entries),
        "entries": selected_entries,
    }
    manifest_path = _write_json(args.manifest_out, manifest_payload)
    gym.destroy_sim(sim)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote report: {report_path}")
    print(f"selected_objects={len(selected_entries)} source_objects={len(entries)}")


if __name__ == "__main__":
    main()
