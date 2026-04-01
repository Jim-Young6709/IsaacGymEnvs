import os
import sys
import time
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.transformations as tt
import yourdfpy

from .config import AppConfig, CylinderSpec, default_cylinder_specs
from .models import MeshEntry, TransformState, ShellTransform, combine_shell_transforms, split_transform_state
from .repositories import MeshRepository, HistoryRepository, TransformRepository, infer_com_z_midpoint

if TYPE_CHECKING:
    from .viewer import MeshViewer


class Navigator:
    def __init__(self, entries: List[MeshEntry]):
        self.entries = entries
        self.filtered = entries[:]
        self.index = 0

    def set_filter(self, predicate):
        self.filtered = [e for e in self.entries if predicate(e)]
        self.index = 0

    def current(self) -> Optional[MeshEntry]:
        if not self.filtered:
            return None
        self.index = max(0, min(self.index, len(self.filtered) - 1))
        return self.filtered[self.index]

    def next(self) -> Optional[MeshEntry]:
        if self.index < len(self.filtered) - 1:
            self.index += 1
        return self.current()

    def prev(self) -> Optional[MeshEntry]:
        if self.index > 0:
            self.index -= 1
        return self.current()

    def jump_to_index(self, idx: int) -> Optional[MeshEntry]:
        if not self.filtered:
            return None
        idx = max(0, min(idx, len(self.filtered) - 1))
        self.index = idx
        return self.current()

    def jump_to_name(self, query: str) -> Optional[MeshEntry]:
        q = query.lower().strip()
        if not q:
            return self.current()
        for i, entry in enumerate(self.filtered):
            if q in entry.full_name.lower():
                self.index = i
                return entry
        return None


class MeshController:
    VIEW_SOURCE = "source"
    VIEW_SAVED_OR_SOURCE = "saved_or_source"
    VIEW_SAVED_OR_CURRENT = "saved_or_current"
    DEFAULT_LEAP_POSE = [
        0.0000, 0.0000, 0.0000, 0.0000,
        0.6900, 1.0100, 0.0000, -0.0300,
        0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000,
    ]
    DEFAULT_ARM_JOINTS = [
        -1.41111064, -1.20421876, 1.11514925, -2.30643184,
        0.97677832, 1.59482316, -0.73056353,
    ]

    @staticmethod
    def _fmt_extent(value: float) -> str:
        value = float(value)
        if abs(value) >= 1e-3:
            return f"{value:.4f}"
        return f"{value:.2e}"

    def __init__(self, config: AppConfig, viewer: "MeshViewer"):
        self.config = config
        self.viewer = viewer
        self.mesh_repo = MeshRepository(config)
        self.history_repo = HistoryRepository(config)
        self.transform_repo = TransformRepository(config, self.mesh_repo, self.history_repo)

        self.entries = self.mesh_repo.scan(config.mesh_root)
        self.navigator = Navigator(self.entries)
        self.min_shell_state = ShellTransform()
        self.max_shell_state = ShellTransform()
        self.transform_state = combine_shell_transforms(self.min_shell_state, self.max_shell_state)
        self.transform_states = {}
        self.category_names = self.mesh_repo.scan_category_names(
            [config.mesh_root, config.output_root]
        )
        if not self.category_names:
            self.category_names = ["default"]
        self.edit_category = self.category_names[0]
        self.confidence = "confident"

        self.leap_mesh: Optional[trimesh.Trimesh] = None
        self.leap_path: Optional[str] = None
        self.leap_scale = 1.0
        self.show_leap = False

        self.show_pointcloud = True
        self.show_com = True
        self.show_grid_labels = False
        self.show_bbox = False
        self.show_bbox_dims = False
        self.auto_scale_unsaved = True
        self.use_mesh_analysis = False
        self.show_convex_hull = False
        self.fix_watertight_preview = False
        self.mesh_info = {}
        self.hand_mesh = None
        self.show_hand = False
        self.hand_x = 0.0
        self.hand_y = 0.0
        self.hand_z = 0.1
        self.hand_scale = 1.0
        self.hand_rot_deg = [0.0, 0.0, 0.0]
        self.hand_urdf = None
        self.hand_filename_resolver = None
        self.hand_link_names: List[str] = []
        self.hand_arm_joint_angles: Dict[str, float] = {}
        self.hand_mesh_cache: Dict[str, trimesh.Trimesh] = {}
        self.hand_joint_order = []
        self.hand_joint_angles = []
        self.hand_joint_limits = {}
        self.auto_scaled = {}
        self.pending_focus_mode: Optional[str] = None
        self.visible_categories = set(self.category_names)
        self.current_scene_stats = {}
        self.display_mesh_mode = self.VIEW_SOURCE
        self.active_shell = "min"
        self.selected_variant = "v000"
        self._refresh_lock = threading.RLock()

        self._feature_registry = None

    def register_features(self, registry):
        self._feature_registry = registry

    def notify_entry_loaded(self, entry: MeshEntry):
        if self._feature_registry:
            self._feature_registry.notify_entry_loaded(entry)

    def notify_transform_changed(self):
        if self._feature_registry:
            self._feature_registry.notify_transform_changed(self._sync_transform_snapshot())

    def _sync_transform_snapshot(self) -> TransformState:
        self.transform_state = combine_shell_transforms(self.min_shell_state, self.max_shell_state)
        return self.transform_state

    def _load_shell_states_from_transform(self, state: TransformState) -> None:
        self.min_shell_state, self.max_shell_state = split_transform_state(state)
        self._sync_transform_snapshot()

    def _reset_shell_states(self) -> None:
        self.min_shell_state = ShellTransform()
        self.max_shell_state = ShellTransform()
        self._sync_transform_snapshot()

    def current_entry(self) -> Optional[MeshEntry]:
        return self.navigator.current()

    def load_entry(self, entry: Optional[MeshEntry]):
        if entry is None:
            return
        self.auto_scaled = {}
        self.transform_states = {}
        self.selected_variant = self._default_variant_for(entry, self.edit_category)
        self._reload_transform_for_current_context(entry, self.edit_category, self.selected_variant)
        self.refresh_view()
        self.notify_entry_loaded(entry)

    def refresh_view(self):
        with self._refresh_lock:
            entry = self.current_entry()
            if entry is None:
                return
            source_mesh = self.mesh_repo.load_mesh(entry)
            if source_mesh is None:
                return

            self._sync_transform_snapshot()
            self.mesh_info = {}
            display = self._resolve_display_mesh(entry, source_mesh)
            if display is None:
                return
            base_mesh, is_saved, mesh_state, apply_current_transform = display
            saved_state = None
            if is_saved:
                saved_state = self.transform_repo.load_transform(
                    entry, self.edit_category, self.selected_variant
                )
            save_status = self._save_status_text(entry, is_saved)
            fixed_preview = False
            fixed_overlay = None
            if self.fix_watertight_preview:
                try:
                    original = base_mesh.copy()
                    trimesh.repair.fill_holes(base_mesh)
                    base_mesh.remove_degenerate_faces()
                    base_mesh.remove_unreferenced_vertices()
                    fixed_preview = True
                    fixed_overlay = self._compute_repair_overlay(original, base_mesh)
                except Exception:
                    pass

            if (
                self.display_mesh_mode == self.VIEW_SAVED_OR_SOURCE
                and is_saved
                and saved_state is not None
            ):
                obj_mesh = base_mesh.copy()
                min_cov_mesh = self._apply_saved_endpoint_to_baked_mesh(base_mesh, saved_state, "min")
                max_cov_mesh = self._apply_saved_endpoint_to_baked_mesh(base_mesh, saved_state, "max")
            elif (
                self.display_mesh_mode == self.VIEW_SAVED_OR_CURRENT
                and is_saved
                and saved_state is not None
            ):
                base_min_mesh = self._apply_saved_endpoint_to_baked_mesh(base_mesh, saved_state, "min")
                base_max_mesh = self._apply_saved_endpoint_to_baked_mesh(base_mesh, saved_state, "max")
                min_cov_mesh = self.min_shell_state.apply_to_mesh(base_min_mesh)
                max_cov_mesh = self.max_shell_state.apply_to_mesh(base_max_mesh)
                obj_mesh = min_cov_mesh if self.active_shell == "min" else max_cov_mesh
            else:
                min_cov_mesh = self.min_shell_state.apply_to_mesh(base_mesh)
                max_cov_mesh = self.max_shell_state.apply_to_mesh(base_mesh)
                obj_mesh = min_cov_mesh if self.active_shell == "min" else max_cov_mesh
            hull_mesh = None
            if self.use_mesh_analysis:
                try:
                    watertight = bool(obj_mesh.is_watertight)
                except Exception:
                    watertight = False
                hull_volume = None
                if self.show_convex_hull or self.use_mesh_analysis:
                    try:
                        hull_mesh = obj_mesh.convex_hull
                        hull_volume = float(hull_mesh.volume)
                    except Exception:
                        hull_mesh = None
                self.mesh_info = {
                    "watertight": watertight,
                    "hull_volume": hull_volume,
                    "fixed_preview": fixed_preview,
                    "mesh_state": mesh_state,
                }

            if self.show_hand:
                min_ref, max_ref = None, None
            else:
                min_ref, max_ref = self._get_reference_cylinders(self.edit_category)
            self.current_scene_stats = self._build_scene_stats(
                obj_mesh,
                min_ref,
                max_ref,
                hull_mesh,
                min_cov_mesh,
                max_cov_mesh,
            )
            if hasattr(self.viewer, "show_single"):
                self.viewer.show_single(
                    category=self.edit_category,
                    obj_mesh=obj_mesh,
                    min_cov_mesh=min_cov_mesh,
                    max_cov_mesh=max_cov_mesh,
                    active_shell=self.active_shell,
                    min_ref_mesh=min_ref,
                    max_ref_mesh=max_ref,
                    is_saved=is_saved,
                    scene_stats=self.current_scene_stats,
                    show_com=self.show_com,
                    show_bbox=self.show_bbox,
                    show_bbox_dims=self.show_bbox_dims,
                    show_hull=self.show_convex_hull,
                    hull_mesh=hull_mesh,
                    hand_mesh=self.hand_mesh if self.show_hand else None,
                    hand_z=self.hand_z,
                    hand_xy=(self.hand_x, self.hand_y),
                    title_text=f"{self.edit_category}\n{entry.full_name} / {self.selected_variant} [{save_status}] [{self.active_shell}]",
                    stats_text=self._viewer_stats_text(min_cov_mesh, max_cov_mesh),
                )
            else:
                self.viewer.show_grid(
                    [(self.edit_category, obj_mesh, max_ref, is_saved, hull_mesh, fixed_preview, fixed_overlay)],
                    cols=1,
                    selected_category=self.edit_category,
                    show_labels=self.show_grid_labels,
                    show_com=self.show_com,
                    show_bbox=self.show_bbox,
                    show_bbox_dims=self.show_bbox_dims,
                    show_hull=self.show_convex_hull,
                    focus_mode=None,
                    hand_mesh=self.hand_mesh if self.show_hand else None,
                    hand_z=self.hand_z,
                    hand_xy=(self.hand_x, self.hand_y),
                )
            self.pending_focus_mode = None
            self.notify_transform_changed()

    def set_transform(self, state: TransformState):
        self._load_shell_states_from_transform(state.normalized())
        self.transform_states[self.edit_category] = self.transform_state
        self.refresh_view()

    def _set_transform_raw(self, state: TransformState) -> None:
        self._load_shell_states_from_transform(state)
        self.transform_states[self.edit_category] = self.transform_state
        self.refresh_view()

    def clear_transform(self) -> None:
        self._reset_shell_states()
        self.transform_states[self.edit_category] = self.transform_state
        self.refresh_view()

    def set_transform_state(self, state: TransformState):
        self._load_shell_states_from_transform(state.normalized())
        self.transform_states[self.edit_category] = self.transform_state

    def set_scale_range(self, scale_min: float, scale_max: float):
        self.set_dilation_ranges(
            float(scale_min),
            float(scale_max),
            float(scale_min),
            float(scale_max),
            float(scale_min),
            float(scale_max),
        )

    def set_active_shell(self, shell: str) -> None:
        if shell not in {"min", "max"}:
            return
        self.active_shell = shell
        self.refresh_view()

    def toggle_active_shell(self) -> None:
        self.set_active_shell("max" if self.active_shell == "min" else "min")

    def set_dilation_ranges(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        z_min: float,
        z_max: float,
    ) -> None:
        max_scale = float(getattr(self.config, "max_scale", 10.0))
        self.min_shell_state.dilate_x = min(max_scale, max(0.05, float(x_min)))
        self.max_shell_state.dilate_x = min(max_scale, max(0.05, float(x_max)))
        self.min_shell_state.dilate_y = min(max_scale, max(0.05, float(y_min)))
        self.max_shell_state.dilate_y = min(max_scale, max(0.05, float(y_max)))
        self.min_shell_state.dilate_z = min(max_scale, max(0.05, float(z_min)))
        self.max_shell_state.dilate_z = min(max_scale, max(0.05, float(z_max)))
        self._sync_transform_snapshot()
        self.transform_states[self.edit_category] = self.transform_state
        self.refresh_view()

    def nudge_active_shell_scale(self, delta: float) -> None:
        self.nudge_uniform_dilation_ranges(delta)

    def nudge_uniform_dilation_ranges(self, delta: float) -> None:
        max_scale = float(getattr(self.config, "max_scale", 10.0))
        shell = self.min_shell_state if self.active_shell == "min" else self.max_shell_state
        shell.dilate_x = min(max_scale, max(0.05, float(shell.dilate_x) + delta))
        shell.dilate_y = min(max_scale, max(0.05, float(shell.dilate_y) + delta))
        shell.dilate_z = min(max_scale, max(0.05, float(shell.dilate_z) + delta))
        self._sync_transform_snapshot()
        self.transform_states[self.edit_category] = self.transform_state
        self.refresh_view()

    def nudge_axis_dilation_ranges(self, axis: str, delta: float) -> None:
        max_scale = float(getattr(self.config, "max_scale", 10.0))
        shell = self.min_shell_state if self.active_shell == "min" else self.max_shell_state
        name = f"dilate_{axis}"
        setattr(shell, name, min(max_scale, max(0.05, float(getattr(shell, name)) + delta)))
        self._sync_transform_snapshot()
        self.transform_states[self.edit_category] = self.transform_state
        self.refresh_view()

    def _using_saved_edit_base(self) -> bool:
        if self.display_mesh_mode != self.VIEW_SAVED_OR_CURRENT:
            return False
        entry = self.current_entry()
        if entry is None:
            return False
        return self.history_repo.has_variant(entry, self.edit_category, self.selected_variant)


    def set_display_mesh_mode(self, mode: str) -> None:
        valid = {
            self.VIEW_SOURCE,
            self.VIEW_SAVED_OR_SOURCE,
            self.VIEW_SAVED_OR_CURRENT,
        }
        if mode not in valid:
            return
        self.display_mesh_mode = mode
        entry = self.current_entry()
        if entry is not None:
            self._reload_transform_for_current_context(entry, self.edit_category, self.selected_variant)
        self.refresh_view()

    def load_hand(self, path: str):
        try:
            urdf_path = path
            if os.path.isdir(path):
                cand = os.path.join(path, "franka_leap.urdf")
                if os.path.isfile(cand):
                    urdf_path = cand
            if urdf_path.endswith(".urdf") and os.path.isfile(urdf_path):
                urdf_file = os.path.abspath(urdf_path)
                base_dir = os.path.dirname(urdf_file)
                assets_root = os.path.abspath(getattr(self.config, "leap_hand_assets_root", "")) or None

                def _filename_handler(fname: str, **_: object) -> str:
                    candidates = [
                        os.path.join(base_dir, fname),
                    ]
                    if assets_root:
                        candidates.append(os.path.join(assets_root, fname))
                    for candidate in candidates:
                        if os.path.isfile(candidate):
                            return candidate
                    return candidates[0]

                self.hand_urdf = yourdfpy.URDF.load(
                    urdf_file,
                    build_scene_graph=True,
                    load_meshes=True,
                    filename_handler=_filename_handler,
                )
                self.hand_filename_resolver = _filename_handler
                self.hand_mesh_cache = {}

                available_joint_names = tuple(self.hand_urdf.actuated_joint_names)
                joint_limits = {}
                for joint_name, joint in zip(available_joint_names, self.hand_urdf.actuated_joints):
                    limit = joint.limit
                    joint_limits[str(joint_name)] = (
                        None if limit is None else limit.lower,
                        None if limit is None else limit.upper,
                    )

                joint_order = []
                config_order = getattr(self.config, "leap_hand_joint_order", None) or []
                for name in config_order:
                    if name in joint_limits:
                        joint_order.append(name)
                if not joint_order:
                    joint_order = list(available_joint_names)

                self.hand_link_names = [
                    name
                    for name in self.hand_urdf.link_map.keys()
                    if not str(name).startswith("panda_link")
                ]
                self.hand_arm_joint_angles = {}
                for idx, name in enumerate(available_joint_names[:7]):
                    if idx < len(self.DEFAULT_ARM_JOINTS):
                        self.hand_arm_joint_angles[str(name)] = float(self.DEFAULT_ARM_JOINTS[idx])
                self.hand_joint_order = joint_order
                self.hand_joint_limits = joint_limits
                if len(self.hand_joint_angles) != len(joint_order):
                    self.hand_joint_angles = self._default_hand_joint_angles()
                self.hand_z = 0.1
                self.hand_mesh = self._build_hand_mesh()
                return
            # Fallback: load a single mesh file.
            mesh = trimesh.load(path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump().sum()
            self.hand_mesh = mesh
        except Exception:
            self.hand_urdf = None
            self.hand_mesh = None

    def _build_hand_mesh(self) -> Optional[trimesh.Trimesh]:
        if self.hand_urdf is None:
            return None
        actuated_names = tuple(self.hand_urdf.actuated_joint_names)
        cfg = np.zeros(len(actuated_names), dtype=np.float64)
        angle_map = dict(self.hand_arm_joint_angles)
        angle_map.update(
            {
                name: float(angle)
                for name, angle in zip(self.hand_joint_order, self.hand_joint_angles)
            }
        )
        for idx, name in enumerate(actuated_names):
            cfg[idx] = angle_map.get(name, 0.0)
        self.hand_urdf.update_cfg(cfg)
        meshes: List[trimesh.Trimesh] = []
        for link_name in self.hand_link_names:
            link = self.hand_urdf.link_map.get(link_name)
            if link is None or not link.visuals:
                continue
            link_transform = self.hand_urdf.get_transform(link_name, self.hand_urdf.base_link)
            for visual in link.visuals:
                geometry = getattr(visual, "geometry", None)
                mesh_spec = getattr(geometry, "mesh", None) if geometry is not None else None
                filename = getattr(mesh_spec, "filename", None)
                if not filename or self.hand_filename_resolver is None:
                    continue
                resolved = self.hand_filename_resolver(filename)
                if not os.path.isfile(resolved):
                    continue
                cached = self.hand_mesh_cache.get(resolved)
                if cached is None:
                    loaded = trimesh.load(resolved, force="mesh")
                    if not isinstance(loaded, trimesh.Trimesh):
                        loaded = loaded.dump().sum()
                    cached = loaded
                    self.hand_mesh_cache[resolved] = cached
                mesh = cached.copy()
                scale = getattr(mesh_spec, "scale", None)
                if scale is not None:
                    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
                    if scale.shape == (3,):
                        S = np.eye(4)
                        S[0, 0] = scale[0]
                        S[1, 1] = scale[1]
                        S[2, 2] = scale[2]
                        mesh.apply_transform(S)
                visual_origin = getattr(visual, "origin", None)
                if visual_origin is None:
                    visual_origin = np.eye(4)
                mesh.apply_transform(link_transform @ visual_origin)
                meshes.append(mesh)
        if len(meshes) == 0:
            return None
        combined = trimesh.util.concatenate(meshes)
        rx, ry, rz = self.hand_rot_deg
        if any(abs(v) > 1e-6 for v in (rx, ry, rz)):
            R = tt.euler_matrix(
                np.deg2rad(rx),
                np.deg2rad(ry),
                np.deg2rad(rz),
                axes="sxyz",
            )
            combined.apply_transform(R)
        return combined

    def _default_hand_joint_angles(self) -> List[float]:
        if len(self.hand_joint_order) == len(self.DEFAULT_LEAP_POSE):
            return list(self.DEFAULT_LEAP_POSE)
        return [0.0 for _ in self.hand_joint_order]

    def set_hand_joint_angles(self, angles: List[float]):
        if not self.hand_joint_order:
            return
        if len(angles) != len(self.hand_joint_order):
            return
        self.hand_joint_angles = list(angles)
        self.hand_mesh = self._build_hand_mesh()

    def set_hand_pose(
        self,
        x: float,
        y: float,
        z: float,
        rot_x: float,
        rot_y: float,
        rot_z: float,
    ) -> None:
        self.hand_x = float(x)
        self.hand_y = float(y)
        self.hand_z = float(z)
        self.hand_rot_deg = [float(rot_x), float(rot_y), float(rot_z)]
        self.hand_mesh = self._build_hand_mesh()

    def reset_hand_canonical(self) -> None:
        self.hand_x = 0.0
        self.hand_y = 0.0
        self.hand_z = 0.1
        self.hand_rot_deg = [0.0, 0.0, 0.0]
        if self.hand_joint_order:
            self.hand_joint_angles = self._default_hand_joint_angles()
        self.hand_mesh = self._build_hand_mesh()

    def hand_joint_report(self) -> str:
        if not self.hand_joint_order:
            return "No hand joints loaded."
        lines = ["LEAP joint angles:"]
        lines.extend(
            f"{name}: {angle:.6f}"
            for name, angle in zip(self.hand_joint_order, self.hand_joint_angles)
        )
        lines.append("")
        lines.append("LEAP logical order:")
        values = list(self.hand_joint_angles)
        for i in range(0, len(values), 4):
            chunk = ", ".join(f"{values[j]:.4f}" for j in range(i, min(i + 4, len(values))))
            lines.append(chunk)
        return "\n".join(lines)

    def rotate_hand(self, axis: str, degrees: float = 90.0):
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            return
        idx = axis_map[axis]
        self.hand_rot_deg[idx] = (self.hand_rot_deg[idx] + degrees) % 360.0
        self.hand_mesh = self._build_hand_mesh()

    def _get_cylinder_spec(self, category_key: str, which: str) -> CylinderSpec:
        spec_map = self.config.cylinder_min_specs if which == "min" else self.config.cylinder_max_specs
        return spec_map.get(category_key, CylinderSpec(x_dim=0.06, y_dim=0.06, height=0.10))

    def set_cylinder_spec(self, category_key: str, which: str, x_dim: float, y_dim: float, height: float) -> None:
        spec_map = self.config.cylinder_min_specs if which == "min" else self.config.cylinder_max_specs
        spec_map[category_key] = CylinderSpec(
            x_dim=float(x_dim),
            y_dim=float(y_dim),
            height=float(height),
        )
        self.refresh_view()

    def reset_cylinder_spec(self, category_key: str) -> None:
        defaults_min = default_cylinder_specs(self.config.cylinder_range_min_factor)
        defaults_max = default_cylinder_specs(self.config.cylinder_range_max_factor)
        self.config.cylinder_min_specs[category_key] = defaults_min.get(category_key, CylinderSpec(x_dim=0.06, y_dim=0.06, height=0.10))
        self.config.cylinder_max_specs[category_key] = defaults_max.get(category_key, CylinderSpec(x_dim=0.06, y_dim=0.06, height=0.10))
        self.refresh_view()

    def _build_cylinder_mesh(self, category_key: str, which: str) -> trimesh.Trimesh:
        spec = self._get_cylinder_spec(category_key, which)
        mesh = trimesh.creation.cylinder(radius=0.5, height=float(spec.height), sections=16)
        scale = np.eye(4)
        scale[0, 0] = float(spec.x_dim)
        scale[1, 1] = float(spec.y_dim)
        mesh.apply_transform(scale)
        return mesh

    def _get_reference_cylinders(self, category_key: str) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        min_mesh = self._build_cylinder_mesh(category_key, "min")
        max_mesh = self._build_cylinder_mesh(category_key, "max")
        return min_mesh, max_mesh

    def _build_scene_stats(
        self,
        obj_mesh: trimesh.Trimesh,
        min_ref: Optional[trimesh.Trimesh],
        max_ref: Optional[trimesh.Trimesh],
        hull_mesh: Optional[trimesh.Trimesh],
        min_cov_mesh: Optional[trimesh.Trimesh] = None,
        max_cov_mesh: Optional[trimesh.Trimesh] = None,
    ) -> dict:
        def safe_extents(mesh: Optional[trimesh.Trimesh]) -> Optional[np.ndarray]:
            if mesh is None:
                return None
            try:
                bounds = mesh.bounds
                if bounds is None:
                    return None
                ext = np.asarray(bounds[1] - bounds[0], dtype=np.float64)
                if ext.shape != (3,):
                    return None
                return ext
            except Exception:
                return None

        meshes = [
            m
            for m in (obj_mesh, min_ref, max_ref, hull_mesh, min_cov_mesh, max_cov_mesh)
            if m is not None
        ]
        max_extent = 0.1
        max_height = 0.1
        for mesh in meshes:
            ext = safe_extents(mesh)
            if ext is not None:
                max_extent = max(max_extent, float(np.max(ext)))
                max_height = max(max_height, float(ext[2]))
        obj_ext = safe_extents(obj_mesh)
        if obj_ext is None:
            obj_ext = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        return {
            "object_extents": obj_ext.tolist(),
            "frame_extent": max_extent,
            "frame_height": max_height,
        }

    def _viewer_stats_text(self, min_mesh: trimesh.Trimesh, max_mesh: trimesh.Trimesh) -> str:
        min_stats = self.transform_repo.compute_stats(min_mesh)
        max_stats = self.transform_repo.compute_stats(max_mesh)
        min_bbox = min_stats["bbox_extents"]
        max_bbox = max_stats["bbox_extents"]
        state = self._sync_transform_snapshot()
        lines = [
            f"shell {self.active_shell}",
            f"min x {self._fmt_extent(min_bbox[0])}",
            f"min y {self._fmt_extent(min_bbox[1])}",
            f"min z {self._fmt_extent(min_bbox[2])}",
            f"max x {self._fmt_extent(max_bbox[0])}",
            f"max y {self._fmt_extent(max_bbox[1])}",
            f"max z {self._fmt_extent(max_bbox[2])}",
            f"dx {state.dilate_x_min:.2f}-{state.dilate_x_max:.2f}",
            f"dy {state.dilate_y_min:.2f}-{state.dilate_y_max:.2f}",
            f"dz {state.dilate_z_min:.2f}-{state.dilate_z_max:.2f}",
        ]
        min_volume = min_stats.get("volume")
        max_volume = max_stats.get("volume")
        if min_volume is not None:
            lines.append(f"vmin {min_volume:.4f}")
        if max_volume is not None:
            lines.append(f"vmax {max_volume:.4f}")
        return "\n".join(lines)

    def set_leap_mesh(self, path: str, scale: float):
        try:
            mesh = trimesh.load(path, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump().sum()
            if scale != 1.0:
                mesh.apply_scale(scale)
            self.leap_mesh = mesh
            self.leap_path = path
            self.leap_scale = scale
        except Exception:
            self.leap_mesh = None
            self.leap_path = None

    def save_current(self, categories: List[str], confidence: str):
        entry = self.current_entry()
        if entry is None:
            return {}
        return self.transform_repo.save(entry, self._sync_transform_snapshot(), categories, confidence, self.selected_variant)

    def discard_current(self):
        entry = self.current_entry()
        if entry is None:
            return {}
        return self.transform_repo.discard(entry, self._sync_transform_snapshot(), self.selected_variant)

    def deactivate_category(self, category: str):
        entry = self.current_entry()
        if entry is None:
            return
        out_dir = os.path.join(self.config.output_root, category, entry.full_name, self.selected_variant)
        if os.path.isdir(out_dir):
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
        if not self.get_variant_names():
            self.selected_variant = "v000"

    def deactivate_all(self):
        entry = self.current_entry()
        if entry is None:
            return
        categories = self.get_category_names()
        self.transform_repo.deactivate_all(entry, categories)

    def get_category_names(self) -> List[str]:
        return list(self.category_names)

    def refresh_categories(self) -> List[str]:
        names = self.mesh_repo.scan_category_names(
            [self.config.mesh_root, self.config.output_root]
        )
        if not names:
            names = ["default"]
        self.category_names = sorted(set(names))
        if self.edit_category not in self.category_names:
            self.edit_category = self.category_names[0]
        self.visible_categories = set(self.category_names)
        return self.get_category_names()

    def create_category(self, category: str) -> bool:
        category_name = str(category).strip()
        if not category_name:
            return False
        os.makedirs(os.path.join(self.config.output_root, category_name), exist_ok=True)
        if category_name not in self.category_names:
            self.category_names.append(category_name)
            self.category_names.sort()
        self.visible_categories.add(category_name)
        return True

    def _default_variant_for(self, entry: MeshEntry, category: str) -> str:
        variants = self.history_repo.list_variants(entry, category)
        return variants[0] if variants else "v000"

    def get_variant_names(self) -> List[str]:
        entry = self.current_entry()
        if entry is None:
            return ["v000"]
        variants = self.history_repo.list_variants(entry, self.edit_category)
        return variants or ["v000"]

    def create_next_variant(self, copy_current: bool = False) -> str:
        entry = self.current_entry()
        if entry is None:
            self.selected_variant = "v000"
            return self.selected_variant
        prev_variant = self.selected_variant
        prev_min_shell = ShellTransform(**vars(self.min_shell_state))
        prev_max_shell = ShellTransform(**vars(self.max_shell_state))
        self.selected_variant = self.history_repo.next_variant_name(entry, self.edit_category)
        if copy_current:
            saved_state = None
            if (
                self.display_mesh_mode in {self.VIEW_SAVED_OR_SOURCE, self.VIEW_SAVED_OR_CURRENT}
                and self.history_repo.has_variant(entry, self.edit_category, prev_variant)
            ):
                saved_state = self.transform_repo.load_transform(entry, self.edit_category, prev_variant)
            if saved_state is not None:
                saved_min, saved_max = split_transform_state(saved_state)
                self.min_shell_state = ShellTransform(
                    dilate_x=float(saved_min.dilate_x) * float(prev_min_shell.dilate_x),
                    dilate_y=float(saved_min.dilate_y) * float(prev_min_shell.dilate_y),
                    dilate_z=float(saved_min.dilate_z) * float(prev_min_shell.dilate_z),
                    rot_x=float(saved_min.rot_x) + float(prev_min_shell.rot_x),
                    rot_y=float(saved_min.rot_y) + float(prev_min_shell.rot_y),
                    rot_z=float(saved_min.rot_z) + float(prev_min_shell.rot_z),
                )
                self.max_shell_state = ShellTransform(
                    dilate_x=float(saved_max.dilate_x) * float(prev_max_shell.dilate_x),
                    dilate_y=float(saved_max.dilate_y) * float(prev_max_shell.dilate_y),
                    dilate_z=float(saved_max.dilate_z) * float(prev_max_shell.dilate_z),
                    rot_x=float(saved_max.rot_x) + float(prev_max_shell.rot_x),
                    rot_y=float(saved_max.rot_y) + float(prev_max_shell.rot_y),
                    rot_z=float(saved_max.rot_z) + float(prev_max_shell.rot_z),
                )
            else:
                self.min_shell_state = prev_min_shell
                self.max_shell_state = prev_max_shell
        else:
            self._reset_shell_states()
        self._sync_transform_snapshot()
        self.transform_states[self.edit_category] = self.transform_state
        self.refresh_view()
        return self.selected_variant

    def _apply_saved_endpoint_to_baked_mesh(
        self,
        baked_mesh: trimesh.Trimesh,
        saved_state: TransformState,
        which: str,
    ) -> trimesh.Trimesh:
        saved_state = saved_state.normalized()
        mid_vec = np.array([saved_state.dilate_x, saved_state.dilate_y, saved_state.dilate_z], dtype=np.float64)
        end_vec = saved_state.endpoint_dilation_vector(which)
        rel = end_vec / np.maximum(mid_vec, 1e-9)
        mesh = baked_mesh.copy()
        T = np.eye(4)
        T[:3, :3] = np.diag(rel)
        mesh.apply_transform(T)
        return mesh

    def _reload_transform_for_current_context(
        self,
        entry: MeshEntry,
        category: str,
        variant: str,
    ) -> None:
        if (
            self.display_mesh_mode in {self.VIEW_SAVED_OR_SOURCE, self.VIEW_SAVED_OR_CURRENT}
            and self.history_repo.has_variant(entry, category, variant)
        ):
            self._reset_shell_states()
        else:
            state = self.transform_repo.load_transform(entry, category, variant)
            self._load_shell_states_from_transform(state if state is not None else TransformState())
        self.transform_states[category] = self.transform_state

    def set_selected_variant(self, variant: str) -> None:
        entry = self.current_entry()
        self.selected_variant = str(variant).strip() or "v000"
        if entry is None:
            return
        self._reload_transform_for_current_context(entry, self.edit_category, self.selected_variant)
        self.refresh_view()

    def _load_category_reference_config(self, category: str) -> None:
        defaults_min = default_cylinder_specs(self.config.cylinder_range_min_factor)
        defaults_max = default_cylinder_specs(self.config.cylinder_range_max_factor)
        self.config.cylinder_min_specs[category] = defaults_min.get(
            category,
            CylinderSpec(x_dim=0.06, y_dim=0.06, height=0.10),
        )
        self.config.cylinder_max_specs[category] = defaults_max.get(
            category,
            CylinderSpec(x_dim=0.06, y_dim=0.06, height=0.10),
        )
        data = self.history_repo.load_category_reference_config(category)
        if not isinstance(data, dict):
            return
        min_data = data.get("min", {})
        max_data = data.get("max", {})
        try:
            if isinstance(min_data, dict):
                min_fallback = self._get_cylinder_spec(category, "min")
                min_radius = float(min_data.get("radius", 0.5 * min_fallback.x_dim))
                self.config.cylinder_min_specs[category] = CylinderSpec(
                    x_dim=float(min_data.get("x_dim", 2.0 * min_radius)),
                    y_dim=float(min_data.get("y_dim", 2.0 * min_radius)),
                    height=float(min_data.get("height", min_fallback.height)),
                )
            if isinstance(max_data, dict):
                max_fallback = self._get_cylinder_spec(category, "max")
                max_radius = float(max_data.get("radius", 0.5 * max_fallback.x_dim))
                self.config.cylinder_max_specs[category] = CylinderSpec(
                    x_dim=float(max_data.get("x_dim", 2.0 * max_radius)),
                    y_dim=float(max_data.get("y_dim", 2.0 * max_radius)),
                    height=float(max_data.get("height", max_fallback.height)),
                )
        except Exception:
            return

    def save_category_reference_config(self, category: str) -> Optional[str]:
        min_spec = self._get_cylinder_spec(category, "min")
        max_spec = self._get_cylinder_spec(category, "max")
        payload = {
            "category": category,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "min": {
                "x_dim": float(min_spec.x_dim),
                "y_dim": float(min_spec.y_dim),
                "height": float(min_spec.height),
            },
            "max": {
                "x_dim": float(max_spec.x_dim),
                "y_dim": float(max_spec.y_dim),
                "height": float(max_spec.height),
            },
        }
        return self.history_repo.save_category_reference_config(category, payload)

    def set_edit_category(self, category: str):
        self.edit_category = category
        self._load_category_reference_config(category)
        entry = self.current_entry()
        if entry is None:
            return
        self.selected_variant = self._default_variant_for(entry, category)
        self._reload_transform_for_current_context(entry, category, self.selected_variant)
        self.refresh_view()

    def set_visible_categories(self, categories):
        self.visible_categories = set(categories)
        self.refresh_view()

    def _compute_repair_overlay(self, original: trimesh.Trimesh, repaired: trimesh.Trimesh) -> Optional[trimesh.Trimesh]:
        try:
            def tri_hashes(mesh: trimesh.Trimesh):
                verts = mesh.vertices
                faces = mesh.faces
                hashes = set()
                for f in faces:
                    tri = verts[f]
                    tri = np.round(tri, 4)
                    tri = tri[np.lexsort(tri.T)]
                    hashes.add(tuple(tri.flatten()))
                return hashes

            orig_hashes = tri_hashes(original)
            rep_verts = repaired.vertices
            rep_faces = repaired.faces
            changed = []
            for i, f in enumerate(rep_faces):
                tri = rep_verts[f]
                tri = np.round(tri, 4)
                tri = tri[np.lexsort(tri.T)]
                if tuple(tri.flatten()) not in orig_hashes:
                    changed.append(i)
            if not changed:
                return repaired
            overlay = repaired.submesh([changed], append=True, repair=False)
            return overlay
        except Exception:
            return None

    def focus_selected(self):
        self.pending_focus_mode = "selected"
        self.refresh_view()

    def focus_all(self):
        self.pending_focus_mode = "all"
        self.refresh_view()

    def load_version(self, category: str, version: str):
        entry = self.current_entry()
        if entry is None:
            return
        self.selected_variant = version
        state = self.transform_repo.load_transform(entry, category, version)
        if state is not None:
            self.set_transform(state)

    def jump_to(self, query: str):
        entry = None
        if query.isdigit():
            entry = self.navigator.jump_to_index(int(query) - 1)
        else:
            entry = self.navigator.jump_to_name(query)
        self.load_entry(entry)

    def next(self):
        self.load_entry(self.navigator.next())

    def prev(self):
        self.load_entry(self.navigator.prev())

    def get_stats(self):
        entry = self.current_entry()
        if entry is None:
            return None, None
        source_mesh = self.mesh_repo.load_mesh(entry)
        if source_mesh is None:
            return None, None
        display = self._resolve_display_mesh(entry, source_mesh)
        if display is None:
            return None, None
        base_mesh, _, _, apply_current_transform = display
        obj_mesh = self._sync_transform_snapshot().apply_to_mesh(base_mesh) if apply_current_transform else base_mesh
        obj_stats = self.transform_repo.compute_stats(obj_mesh)
        min_ref, max_ref = self._get_reference_cylinders(self.edit_category)
        return obj_stats, {
            "min": self.transform_repo.compute_stats(min_ref),
            "max": self.transform_repo.compute_stats(max_ref),
        }

    def _resolve_display_mesh(
        self,
        entry: MeshEntry,
        source_mesh: trimesh.Trimesh,
    ) -> Optional[Tuple[trimesh.Trimesh, bool, str, bool]]:
        saved_mesh = None
        if self.history_repo.has_variant(entry, self.edit_category, self.selected_variant):
            saved_mesh = self.transform_repo.load_saved_mesh(entry, self.edit_category, self.selected_variant)

        if self.display_mesh_mode == self.VIEW_SOURCE:
            return source_mesh.copy(), False, "source", False
        if self.display_mesh_mode == self.VIEW_SAVED_OR_SOURCE:
            if saved_mesh is not None:
                return saved_mesh.copy(), True, "saved", False
            return source_mesh.copy(), False, "source", False
        if self.display_mesh_mode == self.VIEW_SAVED_OR_CURRENT:
            if saved_mesh is not None:
                return saved_mesh.copy(), True, "saved+current", True
            return source_mesh.copy(), False, "current", True
        return None

    def _save_status_text(self, entry: MeshEntry, is_saved_loaded: bool) -> str:
        saved_exists = self.history_repo.has_variant(entry, self.edit_category, self.selected_variant)
        if is_saved_loaded:
            return "saved (loaded)"
        if saved_exists:
            return "saved (not loaded)"
        return "not saved"

    def list_saved_object_names(self, category: str) -> List[str]:
        return self.history_repo.list_saved_object_names(category)

    def bulk_scale_saved(self, category: str, object_names: List[str], scale_factor: float) -> Dict[str, str]:
        return self.transform_repo.bulk_scale_saved(category, object_names, scale_factor)

    def bulk_set_mass(self, category: str, object_names: List[str], mass_value: float) -> Dict[str, str]:
        return self.transform_repo.bulk_set_mass(category, object_names, mass_value)

    def generate_saved_mapping(self, category: str) -> Tuple[bool, str]:
        target_dir = os.path.join(self.config.output_root, category)
        if not os.path.isdir(target_dir):
            return False, f"missing directory: {target_dir}"
        try:
            utils_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "isaacgymenvs", "utils")
            )
            if utils_dir not in sys.path:
                sys.path.insert(0, utils_dir)
            from generate_mapping import generate_mapping

            generate_mapping(target_dir)
            return True, os.path.join(target_dir, "type_mapping.json")
        except Exception as exc:
            return False, str(exc)
