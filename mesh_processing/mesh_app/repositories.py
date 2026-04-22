import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh

from .config import AppConfig
from .models import MeshEntry, TransformState


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_com_z_midpoint(mesh: trimesh.Trimesh) -> np.ndarray:
    bounds = mesh.bounds
    mid = 0.5 * (bounds[0] + bounds[1])
    return mid.astype(np.float64)


class MeshRepository:
    def __init__(self, config: AppConfig):
        self.config = config
        self._mesh_cache: Dict[str, trimesh.Trimesh] = {}

    def scan(self, root: str) -> List[MeshEntry]:
        entries: List[MeshEntry] = []
        if not os.path.isdir(root):
            return entries
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            obj_files = sorted(f for f in filenames if f.lower().endswith(".obj"))
            for fname in obj_files:
                obj_path = os.path.join(dirpath, fname)
                base_name = os.path.splitext(fname)[0]
                parent_dir = os.path.basename(dirpath)

                label = parent_dir

                def maybe(path):
                    return path if os.path.isfile(path) else None

                entries.append(
                    MeshEntry(
                        label=label,
                        base_name=base_name,
                        obj_path=obj_path,
                        npy_path=maybe(os.path.join(dirpath, base_name + ".npy")),
                        glb_path=maybe(os.path.join(dirpath, base_name + ".glb")),
                        urdf_path=maybe(os.path.join(dirpath, base_name + ".urdf")),
                    )
                )
        entries.sort(key=lambda e: e.full_name)
        return entries

    def scan_category_names(self, roots: List[str]) -> List[str]:
        names = set()
        for root_idx, root in enumerate(roots):
            if not root or not os.path.isdir(root):
                continue
            for name in os.listdir(root):
                path = os.path.join(root, name)
                if not os.path.isdir(path):
                    continue
                if root_idx == 0:
                    # Source hierarchy: source_root/category_dir/object_dir
                    # Only treat the first-level folder as a category if it contains
                    # child directories, not just mesh files.
                    has_child_dir = any(
                        os.path.isdir(os.path.join(path, child))
                        for child in os.listdir(path)
                    )
                    if not has_child_dir:
                        continue
                names.add(name)
        return sorted(names)

    def load_mesh(self, entry: MeshEntry) -> Optional[trimesh.Trimesh]:
        cache_key = entry.obj_path
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]
        try:
            mesh = trimesh.load(entry.obj_path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump().sum()
            initial_scale = self.get_urdf_initial_scale(entry.urdf_path)
            if not np.allclose(initial_scale, 1.0):
                S = np.eye(4)
                S[0, 0] = initial_scale[0]
                S[1, 1] = initial_scale[1]
                S[2, 2] = initial_scale[2]
                mesh.apply_transform(S)
            self._mesh_cache[cache_key] = mesh
            return mesh
        except Exception:
            return None

    def load_pointcloud(self, entry: MeshEntry) -> Optional[np.ndarray]:
        if entry.npy_path and os.path.isfile(entry.npy_path):
            try:
                pts = np.load(entry.npy_path)
                return pts
            except Exception:
                return None
        return None

    def get_urdf_initial_scale(self, urdf_path: Optional[str]) -> np.ndarray:
        if not urdf_path or not os.path.isfile(urdf_path):
            return np.asarray(self.config.default_urdf_scale, dtype=np.float64)
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for mesh_tag in root.findall(".//visual/geometry/mesh"):
                scale_str = mesh_tag.get("scale")
                if scale_str:
                    vec = np.fromstring(scale_str, sep=" ")
                    if vec.shape == (3,):
                        return vec.astype(np.float64)
        except Exception:
            pass
        return np.asarray(self.config.default_urdf_scale, dtype=np.float64)


class HistoryRepository:
    def __init__(self, config: AppConfig):
        self.config = config

    def _object_dir(self, category: str, object_name: str) -> str:
        return os.path.join(self.config.output_root, category, object_name)

    def _variant_dir(self, category: str, object_name: str, variant: str) -> str:
        return os.path.join(self._object_dir(category, object_name), variant)

    def list_variants(self, entry: MeshEntry, category: str) -> List[str]:
        base_dir = self._object_dir(category, entry.full_name)
        if not os.path.isdir(base_dir):
            return []
        variants = [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name)) and name.startswith("v") and name[1:].isdigit()
        ]
        return sorted(variants)

    def next_variant_name(self, entry: MeshEntry, category: str) -> str:
        variants = self.list_variants(entry, category)
        if not variants:
            return "v000"
        max_idx = max(int(name[1:]) for name in variants if name.startswith("v") and name[1:].isdigit())
        return f"v{max_idx + 1:03d}"

    def has_variant(self, entry: MeshEntry, category: str, variant: str) -> bool:
        stem = f"{entry.full_name}_{variant}"
        json_path = os.path.join(self._variant_dir(category, entry.full_name, variant), f"{stem}.json")
        return os.path.isfile(json_path)

    def list_versions(self, entry: MeshEntry, category: str) -> List[str]:
        return self.list_variants(entry, category)

    def next_version(self, entry: MeshEntry, category: str) -> str:
        return self.next_variant_name(entry, category)

    def prune_versions(self, entry: MeshEntry, category: str) -> None:
        return

    def read_metadata(self, entry: MeshEntry, category: str, version: str) -> Optional[Dict]:
        stem = f"{entry.full_name}_{version}"
        json_path = os.path.join(self._variant_dir(category, entry.full_name, version), f"{stem}.json")
        if not os.path.isfile(json_path):
            return None
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def active_path(self, entry: MeshEntry, category: str) -> str:
        return os.path.join(self.config.output_root, category, entry.full_name, "active.json")

    def get_active_version(self, entry: MeshEntry, category: str) -> Optional[str]:
        variants = self.list_variants(entry, category)
        return variants[0] if variants else None

    def set_active_version(self, entry: MeshEntry, category: str, version: Optional[str]) -> None:
        path = self.active_path(entry, category)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass

    def list_saved_object_names(self, category: str) -> List[str]:
        category_dir = os.path.join(self.config.output_root, category)
        if not os.path.isdir(category_dir):
            return []
        names = [
            name
            for name in os.listdir(category_dir)
            if os.path.isdir(os.path.join(category_dir, name))
        ]
        names.sort()
        return names

    def resolve_version_for_name(self, category: str, object_name: str) -> Optional[str]:
        base_dir = self._object_dir(category, object_name)
        if not os.path.isdir(base_dir):
            return None
        variants = [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name)) and name.startswith("v") and name[1:].isdigit()
        ]
        variants.sort()
        return variants[0] if variants else None

    def read_metadata_for_name(self, category: str, object_name: str, version: str) -> Optional[Dict]:
        stem = f"{object_name}_{version}"
        json_path = os.path.join(self._variant_dir(category, object_name, version), f"{stem}.json")
        if not os.path.isfile(json_path):
            return None
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def category_reference_config_path(self, category: str) -> str:
        return os.path.join(self.config.output_root, category, "_reference_config.json")

    def load_category_reference_config(self, category: str) -> Optional[Dict]:
        path = self.category_reference_config_path(category)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def save_category_reference_config(self, category: str, data: Dict) -> str:
        path = self.category_reference_config_path(category)
        ensure_dir(os.path.dirname(path))
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path


class TransformRepository:
    def __init__(self, config: AppConfig, mesh_repo: MeshRepository, history_repo: HistoryRepository):
        self.config = config
        self.mesh_repo = mesh_repo
        self.history_repo = history_repo

    def _saved_object_dir(self, category: str, object_name: str, variant: str) -> str:
        return os.path.join(self.config.output_root, category, object_name, variant)

    def _variant_stem(self, object_name: str, variant: str) -> str:
        return f"{object_name}_{variant}"

    def compute_stats(self, mesh: trimesh.Trimesh) -> Dict[str, float]:
        ex = mesh.bounding_box.extents
        if ex is None or ex.size == 0:
            ex = np.array([0.0, 0.0, 0.0])
        try:
            vol = float(mesh.volume) if mesh.is_volume else float(mesh.convex_hull.volume)
        except Exception:
            vol = float("nan")
        max_dim = float(np.max(ex)) if ex.size > 0 else float("nan")
        return {
            "bbox_extents": ex.tolist(),
            "volume": vol,
            "max_dim": max_dim,
        }

    def prepare_pointcloud(self, entry: MeshEntry, mesh: trimesh.Trimesh, transform: TransformState) -> Optional[np.ndarray]:
        pts = self.mesh_repo.load_pointcloud(entry)
        if pts is None:
            return None
        if pts.ndim != 2 or pts.shape[1] != 3:
            return pts

        pts = pts.astype(np.float64)
        initial_scale = self.mesh_repo.get_urdf_initial_scale(entry.urdf_path)

        # Heuristic to avoid double-applying URDF scale.
        mesh_ext = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
        pc_ext = np.asarray(pts.max(axis=0) - pts.min(axis=0), dtype=np.float64)
        mesh_max = float(np.max(mesh_ext)) if mesh_ext.size else 0.0
        pc_max = float(np.max(pc_ext)) if pc_ext.size else 0.0
        apply_init = True
        if mesh_max > 1e-9 and pc_max > 1e-9:
            ratio = mesh_max / pc_max
            if np.isclose(ratio, 1.0, rtol=0.25):
                apply_init = False
            else:
                s = float(np.mean(initial_scale))
                if np.isclose(ratio, s, rtol=0.25):
                    apply_init = True
                elif np.isclose(ratio, 1.0 / max(s, 1e-9), rtol=0.25):
                    apply_init = False
                else:
                    apply_init = pc_max > mesh_max
        if apply_init:
            pts = pts * initial_scale

        pts = transform.apply_to_points(pts)
        return pts

    def save(self, entry: MeshEntry, transform: TransformState, categories: List[str], confidence: str, variant: str) -> Dict[str, Dict[str, str]]:
        transform.normalized()
        mesh = self.mesh_repo.load_mesh(entry)
        if mesh is None:
            return {}

        results: Dict[str, Dict[str, str]] = {}
        for category in categories:
            version = variant
            out_dir = self._saved_object_dir(category, entry.full_name, version)
            stem = self._variant_stem(entry.full_name, version)
            ensure_dir(out_dir)

            mesh_scaled = transform.apply_to_mesh(mesh)
            out_obj = os.path.join(out_dir, f"{stem}.obj")
            mesh_scaled.export(out_obj)

            out_glb = None
            if entry.glb_path:
                try:
                    glb_mesh = trimesh.load(entry.glb_path, force="mesh")
                    if not isinstance(glb_mesh, trimesh.Trimesh):
                        glb_mesh = glb_mesh.dump().sum()
                    init_scale = self.mesh_repo.get_urdf_initial_scale(entry.urdf_path)
                    S = np.eye(4)
                    S[0, 0] = init_scale[0]
                    S[1, 1] = init_scale[1]
                    S[2, 2] = init_scale[2]
                    glb_mesh.apply_transform(S)
                    glb_mesh.apply_transform(transform.transform_matrix())
                    out_glb = os.path.join(out_dir, f"{stem}.glb")
                    glb_mesh.export(out_glb)
                except Exception:
                    out_glb = None

            out_npy = None
            if entry.npy_path:
                pts = self.prepare_pointcloud(entry, mesh, transform)
                if pts is not None:
                    out_npy = os.path.join(out_dir, f"{stem}.npy")
                    np.save(out_npy, pts.astype(np.float32))

            out_urdf = os.path.join(out_dir, f"{stem}.urdf")
            self._write_urdf(entry, transform, mesh_scaled, out_urdf, mesh_filename=f"{stem}.obj")

            stats_orig = self.compute_stats(mesh)
            stats_scaled = self.compute_stats(mesh_scaled)
            transform_payload = {
                "dilate_x": float(transform.dilate_x),
                "dilate_y": float(transform.dilate_y),
                "dilate_z": float(transform.dilate_z),
                "dilate_x_min": float(transform.dilate_x_min),
                "dilate_x_max": float(transform.dilate_x_max),
                "dilate_y_min": float(transform.dilate_y_min),
                "dilate_y_max": float(transform.dilate_y_max),
                "dilate_z_min": float(transform.dilate_z_min),
                "dilate_z_max": float(transform.dilate_z_max),
                "rot_x": float(transform.rot_x),
                "rot_y": float(transform.rot_y),
                "rot_z": float(transform.rot_z),
            }

            metadata = {
                "transform_model": "dilation_only_v1",
                "label": entry.label,
                "base_name": entry.base_name,
                "full_name": entry.full_name,
                "category": category,
                "confidence": confidence,
                "version": version,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "transform": transform_payload,
                "dilation_range": {
                    "x": {"min": float(transform.dilate_x_min), "max": float(transform.dilate_x_max)},
                    "y": {"min": float(transform.dilate_y_min), "max": float(transform.dilate_y_max)},
                    "z": {"min": float(transform.dilate_z_min), "max": float(transform.dilate_z_max)},
                },
                "stats": {
                    "original": stats_orig,
                    "scaled": stats_scaled,
                },
                "paths": {
                    "output": {
                        "obj": out_obj,
                        "glb": out_glb,
                        "npy": out_npy,
                        "urdf": out_urdf,
                    },
                    "original": {
                        "obj": entry.obj_path,
                        "glb": entry.glb_path,
                        "npy": entry.npy_path,
                        "urdf": entry.urdf_path,
                    },
                },
            }
            json_path = os.path.join(out_dir, f"{stem}.json")
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)

            results[category] = {
                "version": version,
                "dir": out_dir,
            }

        return results

    def discard(self, entry: MeshEntry, transform: TransformState, variant: str = "v000") -> Dict[str, str]:
        transform.normalized()
        mesh = self.mesh_repo.load_mesh(entry)
        if mesh is None:
            return {}
        category = self.config.discard_dir
        version = variant
        out_dir = self._saved_object_dir(category, entry.full_name, version)
        stem = self._variant_stem(entry.full_name, version)
        ensure_dir(out_dir)

        mesh_scaled = transform.apply_to_mesh(mesh)
        out_obj = os.path.join(out_dir, f"{stem}.obj")
        mesh_scaled.export(out_obj)

        out_glb = None
        if entry.glb_path:
            try:
                glb_mesh = trimesh.load(entry.glb_path, force="mesh")
                if not isinstance(glb_mesh, trimesh.Trimesh):
                    glb_mesh = glb_mesh.dump().sum()
                init_scale = self.mesh_repo.get_urdf_initial_scale(entry.urdf_path)
                S = np.eye(4)
                S[0, 0] = init_scale[0]
                S[1, 1] = init_scale[1]
                S[2, 2] = init_scale[2]
                glb_mesh.apply_transform(S)
                glb_mesh.apply_transform(transform.transform_matrix())
                out_glb = os.path.join(out_dir, f"{stem}.glb")
                glb_mesh.export(out_glb)
            except Exception:
                out_glb = None

        out_npy = None
        if entry.npy_path:
            pts = self.prepare_pointcloud(entry, mesh, transform)
            if pts is not None:
                out_npy = os.path.join(out_dir, f"{stem}.npy")
                np.save(out_npy, pts.astype(np.float32))

        out_urdf = os.path.join(out_dir, f"{stem}.urdf")
        self._write_urdf(entry, transform, mesh_scaled, out_urdf, mesh_filename=f"{stem}.obj")

        stats_orig = self.compute_stats(mesh)
        stats_scaled = self.compute_stats(mesh_scaled)
        transform_payload = {
            "dilate_x": float(transform.dilate_x),
            "dilate_y": float(transform.dilate_y),
            "dilate_z": float(transform.dilate_z),
            "dilate_x_min": float(transform.dilate_x_min),
            "dilate_x_max": float(transform.dilate_x_max),
            "dilate_y_min": float(transform.dilate_y_min),
            "dilate_y_max": float(transform.dilate_y_max),
            "dilate_z_min": float(transform.dilate_z_min),
            "dilate_z_max": float(transform.dilate_z_max),
            "rot_x": float(transform.rot_x),
            "rot_y": float(transform.rot_y),
            "rot_z": float(transform.rot_z),
        }

        metadata = {
            "transform_model": "dilation_only_v1",
            "label": entry.label,
            "base_name": entry.base_name,
            "full_name": entry.full_name,
            "category": category,
            "state": "discarded",
            "version": version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transform": transform_payload,
            "dilation_range": {
                "x": {"min": float(transform.dilate_x_min), "max": float(transform.dilate_x_max)},
                "y": {"min": float(transform.dilate_y_min), "max": float(transform.dilate_y_max)},
                "z": {"min": float(transform.dilate_z_min), "max": float(transform.dilate_z_max)},
            },
            "stats": {
                "original": stats_orig,
                "scaled": stats_scaled,
            },
            "paths": {
                "output": {
                    "obj": out_obj,
                    "glb": out_glb,
                    "npy": out_npy,
                    "urdf": out_urdf,
                },
                "original": {
                    "obj": entry.obj_path,
                    "glb": entry.glb_path,
                    "npy": entry.npy_path,
                    "urdf": entry.urdf_path,
                },
            },
        }
        json_path = os.path.join(out_dir, f"{stem}.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {"version": version, "dir": out_dir}

    def deactivate_category(self, entry: MeshEntry, category: str) -> None:
        variant = self.history_repo.get_active_version(entry, category)
        if variant is None:
            return
        out_dir = self._saved_object_dir(category, entry.full_name, variant)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

    def deactivate_all(self, entry: MeshEntry, categories: List[str]) -> None:
        for category in categories:
            object_dir = os.path.join(self.config.output_root, category, entry.full_name)
            if os.path.isdir(object_dir):
                shutil.rmtree(object_dir, ignore_errors=True)

    def load_transform(self, entry: MeshEntry, category: str, version: str) -> Optional[TransformState]:
        metadata = self.history_repo.read_metadata(entry, category, version)
        if not metadata:
            return None
        transform = metadata.get("transform", {})
        dilation_range = metadata.get("dilation_range", {})
        transform_model = str(metadata.get("transform_model", "")).strip()
        if transform_model != "dilation_only_v1":
            return None
        try:
            dilate_x = float(transform.get("dilate_x", 1.0))
            dilate_y = float(transform.get("dilate_y", 1.0))
            dilate_z = float(transform.get("dilate_z", 1.0))
            dx_min = float(dilation_range.get("x", {}).get("min", transform.get("dilate_x_min", dilate_x)))
            dx_max = float(dilation_range.get("x", {}).get("max", transform.get("dilate_x_max", dilate_x)))
            dy_min = float(dilation_range.get("y", {}).get("min", transform.get("dilate_y_min", dilate_y)))
            dy_max = float(dilation_range.get("y", {}).get("max", transform.get("dilate_y_max", dilate_y)))
            dz_min = float(dilation_range.get("z", {}).get("min", transform.get("dilate_z_min", dilate_z)))
            dz_max = float(dilation_range.get("z", {}).get("max", transform.get("dilate_z_max", dilate_z)))
            return TransformState(
                scale=1.0,
                scale_min=float(np.mean([dx_min, dy_min, dz_min])),
                scale_max=float(np.mean([dx_max, dy_max, dz_max])),
                dilate_x=dilate_x,
                dilate_y=dilate_y,
                dilate_z=dilate_z,
                dilate_x_min=dx_min,
                dilate_x_max=dx_max,
                dilate_y_min=dy_min,
                dilate_y_max=dy_max,
                dilate_z_min=dz_min,
                dilate_z_max=dz_max,
                rot_x=float(transform.get("rot_x", 0.0)),
                rot_y=float(transform.get("rot_y", 0.0)),
                rot_z=float(transform.get("rot_z", 0.0)),
            ).normalized()
        except Exception:
            return None

    def load_saved_mesh(self, entry: MeshEntry, category: str, version: str) -> Optional[trimesh.Trimesh]:
        stem = self._variant_stem(entry.full_name, version)
        obj_path = os.path.join(self._saved_object_dir(category, entry.full_name, version), f"{stem}.obj")
        if not os.path.isfile(obj_path):
            return None
        try:
            mesh = trimesh.load(obj_path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump().sum()
            return mesh
        except Exception:
            return None

    def bulk_scale_saved(self, category: str, object_names: List[str], scale_factor: float) -> Dict[str, str]:
        results: Dict[str, str] = {}
        if scale_factor <= 0.0:
            return {name: "invalid scale factor" for name in object_names}
        for object_name in object_names:
            version = self.history_repo.resolve_version_for_name(category, object_name)
            if not version:
                results[object_name] = "no saved version"
                continue
            metadata = self.history_repo.read_metadata_for_name(category, object_name, version)
            if not metadata:
                results[object_name] = "missing metadata"
                continue
            try:
                transform_data = metadata.get("transform", {})
                scale_range = metadata.get("scale_range", {})
                dilation_range = metadata.get("dilation_range", {})
                scale = float(transform_data.get("scale", 1.0))
                scale_min = float(scale_range.get("min", transform_data.get("scale_min", scale)))
                scale_max = float(scale_range.get("max", transform_data.get("scale_max", scale)))
                dilate_x = float(transform_data.get("dilate_x", 1.0))
                dilate_y = float(transform_data.get("dilate_y", 1.0))
                dilate_z = float(transform_data.get("dilate_z", 1.0))
                transform = TransformState(
                    scale=scale * scale_factor,
                    scale_min=scale_min * scale_factor,
                    scale_max=scale_max * scale_factor,
                    dilate_x=dilate_x,
                    dilate_y=dilate_y,
                    dilate_z=dilate_z,
                    dilate_x_min=float(dilation_range.get("x", {}).get("min", transform_data.get("dilate_x_min", dilate_x))),
                    dilate_x_max=float(dilation_range.get("x", {}).get("max", transform_data.get("dilate_x_max", dilate_x))),
                    dilate_y_min=float(dilation_range.get("y", {}).get("min", transform_data.get("dilate_y_min", dilate_y))),
                    dilate_y_max=float(dilation_range.get("y", {}).get("max", transform_data.get("dilate_y_max", dilate_y))),
                    dilate_z_min=float(dilation_range.get("z", {}).get("min", transform_data.get("dilate_z_min", dilate_z))),
                    dilate_z_max=float(dilation_range.get("z", {}).get("max", transform_data.get("dilate_z_max", dilate_z))),
                    rot_x=float(transform_data.get("rot_x", 0.0)),
                    rot_y=float(transform_data.get("rot_y", 0.0)),
                    rot_z=float(transform_data.get("rot_z", 0.0)),
                ).normalized()
                original_paths = metadata.get("paths", {}).get("original", {})
                entry = MeshEntry(
                    label=str(metadata.get("label", "")),
                    base_name=str(metadata.get("base_name", object_name)),
                    obj_path=str(original_paths.get("obj", "")),
                    npy_path=original_paths.get("npy"),
                    glb_path=original_paths.get("glb"),
                    urdf_path=original_paths.get("urdf"),
                )
                mesh = self.mesh_repo.load_mesh(entry)
                if mesh is None:
                    results[object_name] = "failed to load source mesh"
                    continue
                version_dir = self._saved_object_dir(category, object_name, version)
                stem = self._variant_stem(object_name, version)
                mesh_scaled = transform.apply_to_mesh(mesh)
                out_obj = os.path.join(version_dir, f"{stem}.obj")
                mesh_scaled.export(out_obj)

                out_glb = os.path.join(version_dir, f"{stem}.glb")
                if entry.glb_path and os.path.isfile(entry.glb_path):
                    try:
                        glb_mesh = trimesh.load(entry.glb_path, force="mesh")
                        if not isinstance(glb_mesh, trimesh.Trimesh):
                            glb_mesh = glb_mesh.dump().sum()
                        init_scale = self.mesh_repo.get_urdf_initial_scale(entry.urdf_path)
                        S = np.eye(4)
                        S[0, 0] = init_scale[0]
                        S[1, 1] = init_scale[1]
                        S[2, 2] = init_scale[2]
                        glb_mesh.apply_transform(S)
                        glb_mesh.apply_transform(transform.transform_matrix())
                        glb_mesh.export(out_glb)
                    except Exception:
                        pass

                out_npy = os.path.join(version_dir, f"{stem}.npy")
                if entry.npy_path and os.path.isfile(entry.npy_path):
                    pts = self.prepare_pointcloud(entry, mesh, transform)
                    if pts is not None:
                        np.save(out_npy, pts.astype(np.float32))

                out_urdf = os.path.join(version_dir, f"{stem}.urdf")
                mass_value = self._read_mass_value(out_urdf)
                self._write_urdf(entry, transform, mesh_scaled, out_urdf, mass_value=mass_value, mesh_filename=f"{stem}.obj")

                metadata["transform"] = asdict(transform)
                metadata["scale_range"] = {
                    "min": float(transform.scale_min),
                    "max": float(transform.scale_max),
                }
                metadata["dilation_range"] = {
                    "x": {"min": float(transform.dilate_x_min), "max": float(transform.dilate_x_max)},
                    "y": {"min": float(transform.dilate_y_min), "max": float(transform.dilate_y_max)},
                    "z": {"min": float(transform.dilate_z_min), "max": float(transform.dilate_z_max)},
                }
                metadata["stats"] = {
                    "original": self.compute_stats(mesh),
                    "scaled": self.compute_stats(mesh_scaled),
                }
                metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                json_path = os.path.join(version_dir, f"{stem}.json")
                with open(json_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                results[object_name] = "scaled"
            except Exception as exc:
                results[object_name] = f"error: {exc}"
        return results

    def bulk_set_mass(self, category: str, object_names: List[str], mass_value: float) -> Dict[str, str]:
        results: Dict[str, str] = {}
        if mass_value <= 0.0:
            return {name: "invalid mass" for name in object_names}
        for object_name in object_names:
            version = self.history_repo.resolve_version_for_name(category, object_name)
            if not version:
                results[object_name] = "no saved version"
                continue
            version_dir = self._saved_object_dir(category, object_name, version)
            stem = self._variant_stem(object_name, version)
            urdf_path = os.path.join(version_dir, f"{stem}.urdf")
            json_path = os.path.join(version_dir, f"{stem}.json")
            if not self._rewrite_urdf_mass(urdf_path, mass_value):
                results[object_name] = "failed to update urdf"
                continue
            if os.path.isfile(json_path):
                try:
                    with open(json_path, "r") as f:
                        metadata = json.load(f)
                    metadata["mass"] = float(mass_value)
                    metadata["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(json_path, "w") as f:
                        json.dump(metadata, f, indent=2)
                except Exception:
                    pass
            results[object_name] = "mass updated"
        return results

    def _read_mass_value(self, urdf_path: str) -> Optional[float]:
        if not os.path.isfile(urdf_path):
            return None
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for mass_tag in root.findall(".//inertial/mass"):
                value = mass_tag.get("value")
                if value is not None:
                    return float(value)
        except Exception:
            return None
        return None

    def _rewrite_urdf_mass(self, urdf_path: str, mass_value: float) -> bool:
        if not os.path.isfile(urdf_path):
            return False
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            updated = False
            for mass_tag in root.findall(".//inertial/mass"):
                mass_tag.set("value", f"{float(mass_value):.6f}")
                updated = True
            if not updated:
                return False
            tree.write(urdf_path)
            return True
        except Exception:
            return False

    def _write_urdf(
        self,
        entry: MeshEntry,
        transform: TransformState,
        mesh_scaled: trimesh.Trimesh,
        out_urdf: str,
        mass_value: Optional[float] = None,
        mesh_filename: Optional[str] = None,
    ) -> None:
        init_scale = self.mesh_repo.get_urdf_initial_scale(entry.urdf_path)
        R = transform.rotation_matrix()
        scale_vec = float(transform.scale) * transform.dilation_vector()
        mass_value = 0.5 if mass_value is None else float(mass_value)
        mesh_filename = entry.full_name + ".obj" if mesh_filename is None else str(mesh_filename)

        if entry.urdf_path and os.path.isfile(entry.urdf_path):
            try:
                tree = ET.parse(entry.urdf_path)
                root = tree.getroot()

                def update_origin(tag, apply_rotation: bool, apply_init_scale: bool = True) -> None:
                    if tag is None:
                        return
                    xyz_str = tag.get("xyz")
                    if not xyz_str:
                        return
                    vec = np.fromstring(xyz_str, sep=" ")
                    if vec.shape != (3,):
                        return
                    if apply_init_scale:
                        vec = vec * init_scale
                    vec = vec * scale_vec
                    if apply_rotation:
                        vec = R @ vec
                    tag.set("xyz", f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}")

                def update_mesh_tag(geometry_tag) -> None:
                    if geometry_tag is None:
                        return
                    mesh_tag = geometry_tag.find("mesh")
                    if mesh_tag is not None:
                        mesh_tag.set("filename", mesh_filename)
                        mesh_tag.set("scale", "1 1 1")

                for link in root.findall("link"):
                    for visual in link.findall("visual"):
                        update_origin(visual.find("origin"), apply_rotation=True)
                        update_mesh_tag(visual.find("geometry"))
                    for collision in link.findall("collision"):
                        update_origin(collision.find("origin"), apply_rotation=True)
                        update_mesh_tag(collision.find("geometry"))
                    for inertial in link.findall("inertial"):
                        update_origin(inertial.find("origin"), apply_rotation=False, apply_init_scale=False)
                        mass_tag = inertial.find("mass")
                        if mass_tag is not None:
                            mass_tag.set("value", f"{mass_value:.6f}")

                tree.write(out_urdf)
                return
            except Exception:
                pass

        com = infer_com_z_midpoint(mesh_scaled)
        cx, cy, cz = float(com[0]), float(com[1]), float(com[2])
        urdf_content = f"""<?xml version=\"1.0\" ?>
<robot name=\"mesh_object\">
  <link name=\"base\">
    <visual>
      <geometry>
        <mesh filename=\"{mesh_filename}\" scale=\"1 1 1\"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename=\"{mesh_filename}\" scale=\"1 1 1\"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz=\"{cx:.6f} {cy:.6f} {cz:.6f}\" rpy=\"0 0 0\"/>
      <mass value=\"{mass_value:.6f}\"/>
      <inertia ixx=\"0.01\" iyy=\"0.01\" izz=\"0.01\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/>
    </inertial>
  </link>
</robot>
"""
        with open(out_urdf, "w") as f:
            f.write(urdf_content)
