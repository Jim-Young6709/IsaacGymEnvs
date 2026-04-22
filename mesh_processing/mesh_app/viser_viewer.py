from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

_LOCAL_VISER_SRC = Path(__file__).resolve().parents[3] / "viser" / "src"
if _LOCAL_VISER_SRC.exists():
    sys.path.insert(0, str(_LOCAL_VISER_SRC))

import viser


class ViserMeshViewer:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        self._handles = []
        self._hand_handle = None
        self._object_handle = None
        self._min_cov_handle = None
        self._max_cov_handle = None
        self._ref_min_handle = None
        self._ref_max_handle = None
        self._bbox_handle = None
        self._bbox_dims_handle = None
        self._title_category_handle = None
        self._title_object_handle = None
        self._stats_handle = None
        self._selected_text = ""
        self._scene_stats = None
        self.server.scene.add_grid(
            "/mesh_processing/grid",
            width=0.6,
            height=0.6,
            cell_size=0.05,
            section_size=0.1,
            cell_thickness=0.5,
            section_thickness=1.0,
            position=(0.0, 0.0, 0.0),
            shadow_opacity=0.15,
        )
        @self.server.on_client_connect
        def _on_client_connect(client: viser.ClientHandle) -> None:
            armed = {"done": False}

            @client.camera.on_update
            def _(_: object) -> None:
                if armed["done"]:
                    return
                armed["done"] = True
                self._frame_client_camera(client)

    def toggle_spin(self) -> None:
        # Viser owns camera interaction in the browser; the Qt auto-spin behavior has no direct equivalent here.
        return

    def set_selected_text(self, text: str) -> None:
        self._selected_text = text

    def _clear(self) -> None:
        handles = self._handles
        self._handles = []
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    def _add_handle(self, handle) -> None:
        self._handles.append(handle)

    def _set_label(
        self,
        attr_name: str,
        name: str,
        text: str,
        color: tuple[int, int, int],
        position: np.ndarray,
        anchor: str,
        font_screen_scale: float,
    ) -> None:
        handle = getattr(self, attr_name)
        if not text:
            if handle is not None:
                handle.visible = False
            return
        if handle is None:
            handle = self.server.scene.add_label(
                name,
                text=text,
                color=color,
                position=position,
                anchor=anchor,
                font_size_mode="screen",
                font_screen_scale=font_screen_scale,
                depth_test=False,
            )
            setattr(self, attr_name, handle)
            return
        handle.text = text
        handle.color = color
        handle.position = position
        handle.anchor = anchor
        handle.font_screen_scale = font_screen_scale
        handle.visible = True

    def _set_mesh_handle(
        self,
        attr_name: str,
        name: str,
        mesh: Optional[trimesh.Trimesh],
        color: tuple[int, int, int],
        opacity: float,
        position: np.ndarray,
        cast_shadow: bool,
        receive_shadow: bool,
        wireframe: bool = False,
    ) -> None:
        handle = getattr(self, attr_name)
        if mesh is None:
            if handle is not None:
                handle.visible = False
            return
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        if handle is None:
            handle = self.server.scene.add_mesh_simple(
                name,
                vertices=vertices,
                faces=faces,
                color=color,
                opacity=opacity,
                position=position,
                cast_shadow=cast_shadow,
                receive_shadow=receive_shadow,
                wireframe=wireframe,
                flat_shading=False,
                side="double",
            )
            setattr(self, attr_name, handle)
            return
        handle.vertices = vertices
        handle.faces = faces
        handle.position = position
        handle.color = color
        handle.opacity = opacity
        handle.cast_shadow = cast_shadow
        handle.receive_shadow = receive_shadow
        handle.wireframe = wireframe
        handle.visible = True

    def _bbox_geometry(
        self,
        mesh: trimesh.Trimesh,
        shift: np.ndarray,
        color: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        bounds = mesh.bounds
        if bounds is None:
            return None, None
        corners = np.array(
            [
                [bounds[0][0], bounds[0][1], bounds[0][2]],
                [bounds[1][0], bounds[0][1], bounds[0][2]],
                [bounds[1][0], bounds[1][1], bounds[0][2]],
                [bounds[0][0], bounds[1][1], bounds[0][2]],
                [bounds[0][0], bounds[0][1], bounds[1][2]],
                [bounds[1][0], bounds[0][1], bounds[1][2]],
                [bounds[1][0], bounds[1][1], bounds[1][2]],
                [bounds[0][0], bounds[1][1], bounds[1][2]],
            ],
            dtype=np.float32,
        )
        corners = corners + shift
        edges = np.array(
            [
                [corners[0], corners[1]],
                [corners[1], corners[2]],
                [corners[2], corners[3]],
                [corners[3], corners[0]],
                [corners[4], corners[5]],
                [corners[5], corners[6]],
                [corners[6], corners[7]],
                [corners[7], corners[4]],
                [corners[0], corners[4]],
                [corners[1], corners[5]],
                [corners[2], corners[6]],
                [corners[3], corners[7]],
            ],
            dtype=np.float32,
        )
        edge_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(edges), 2, 1))
        return edges, edge_colors

    def _set_bbox_handle(
        self,
        mesh: Optional[trimesh.Trimesh],
        shift: np.ndarray,
        color: tuple[int, int, int],
    ) -> None:
        handle = self._bbox_handle
        if mesh is None:
            if handle is not None:
                handle.visible = False
            return
        edges, edge_colors = self._bbox_geometry(mesh, shift, color)
        if edges is None or edge_colors is None:
            if handle is not None:
                handle.visible = False
            return
        if handle is None:
            self._bbox_handle = self.server.scene.add_line_segments(
                "/mesh_processing/bbox",
                points=edges,
                colors=edge_colors,
                line_width=1.5,
            )
            return
        handle.points = edges
        handle.colors = edge_colors
        handle.visible = True

    def _replace_hand_handle(
        self,
        hand_mesh: Optional[trimesh.Trimesh],
        hand_z: float,
        hand_xy: tuple[float, float],
    ) -> None:
        if hand_mesh is None:
            if self._hand_handle is not None:
                try:
                    self._hand_handle.remove()
                except Exception:
                    pass
                self._hand_handle = None
            return
        grounded_hand, _ = self._grounded_mesh(hand_mesh)
        vertices = np.asarray(grounded_hand.vertices, dtype=np.float32)
        faces = np.asarray(grounded_hand.faces, dtype=np.uint32)
        position = np.array([hand_xy[0], hand_xy[1], hand_z], dtype=np.float32)
        if self._hand_handle is None:
            self._hand_handle = self.server.scene.add_mesh_simple(
                "/mesh_processing/hand",
                vertices=vertices,
                faces=faces,
                color=(220, 220, 220),
                opacity=0.35,
                position=position,
                cast_shadow=False,
                receive_shadow=False,
                wireframe=False,
                flat_shading=False,
                side="double",
            )
            return
        self._hand_handle.vertices = vertices
        self._hand_handle.faces = faces
        self._hand_handle.position = position

    def _split_title_text(self, title_text: str) -> tuple[str, str]:
        lines = [line.strip() for line in title_text.splitlines() if line.strip()]
        if not lines:
            return "", ""
        if len(lines) == 1:
            return lines[0], ""
        return lines[0], " ".join(lines[1:])

    def _format_stats_text(self, stats_text: str) -> str:
        rows: list[tuple[str, str]] = []
        for raw_line in stats_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) == 2:
                label, value = parts
            else:
                label, value = line, ""
            rows.append((label, value))
        if not rows:
            return ""
        width = max(len(label) for label, _ in rows)
        formatted_lines: list[str] = []
        for label, value in rows:
            extra_gap = 0
            if label == "max":
                extra_gap = 1
            elif label == "vol":
                extra_gap = 2
            formatted_lines.append(f"{label:<{width + extra_gap}}      {value}")
        return "\n".join(formatted_lines)

    def _fmt_extent(self, value: float) -> str:
        value = float(value)
        if abs(value) >= 1e-3:
            return f"{value:.4f}"
        return f"{value:.2e}"

    def _frame_client_camera(self, client: viser.ClientHandle) -> None:
        if not self._scene_stats:
            return
        extent = float(self._scene_stats.get("frame_extent", 0.1))
        height = float(self._scene_stats.get("frame_height", extent))
        look_at = np.array([0.0, 0.0, 0.45 * height], dtype=np.float64)
        distance = max(0.18, 1.6 * extent)
        position = np.array([1.15 * distance, -1.15 * distance, 0.75 * distance + 0.35 * height], dtype=np.float64)
        try:
            with client.atomic():
                client.camera.position = position
                client.camera.look_at = look_at
                client.camera.up_direction = (0.0, 0.0, 1.0)
        except AssertionError:
            return

    def _frame_all_clients(self) -> None:
        for client in self.server.get_clients().values():
            self._frame_client_camera(client)

    def adjust_view(self) -> None:
        self._frame_all_clients()

    def _category_color(self, category: str) -> tuple[int, int, int, int]:
        h = abs(hash(category)) % 360
        c = 180
        x = int(c * (1 - abs((h / 60.0) % 2 - 1)))
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        return (r + 40, g + 40, b + 40, 255)

    def _centered_mesh(self, mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
        out = mesh.copy()
        verts = out.vertices
        center = verts.mean(axis=0) if len(verts) > 0 else np.zeros(3, dtype=np.float32)
        out.vertices = verts - center
        return out, center

    def _grounded_mesh(self, mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, np.ndarray]:
        out = mesh.copy()
        verts = np.asarray(out.vertices, dtype=np.float32)
        if len(verts) == 0:
            return out, np.zeros(3, dtype=np.float32)
        center_xy = verts[:, :2].mean(axis=0)
        min_z = float(np.min(verts[:, 2]))
        offset = np.array([center_xy[0], center_xy[1], min_z], dtype=np.float32)
        out.vertices = verts - offset
        return out, offset

    def _safe_extents(self, mesh: Optional[trimesh.Trimesh]) -> Optional[np.ndarray]:
        if mesh is None:
            return None
        try:
            bounds = mesh.bounds
            if bounds is None:
                return None
            ext = np.asarray(bounds[1] - bounds[0], dtype=np.float32)
            if ext.shape != (3,):
                return None
            return ext
        except Exception:
            return None

    def _add_mesh(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        color: tuple[int, int, int],
        opacity: float,
        position: np.ndarray,
        cast_shadow: bool,
        receive_shadow: bool,
        wireframe: bool = False,
    ) -> None:
        self._add_handle(
            self.server.scene.add_mesh_simple(
                name,
                vertices=np.asarray(mesh.vertices, dtype=np.float32),
                faces=np.asarray(mesh.faces, dtype=np.uint32),
                color=color,
                opacity=opacity,
                position=position,
                cast_shadow=cast_shadow,
                receive_shadow=receive_shadow,
                wireframe=wireframe,
                flat_shading=False,
                side="double",
            )
        )

    def _add_com_axes(self, name: str, mesh: trimesh.Trimesh, shift: np.ndarray) -> None:
        bounds = mesh.bounds
        if bounds is None:
            return
        verts = mesh.vertices
        center = verts.mean(axis=0) if len(verts) > 0 else np.zeros(3, dtype=np.float32)
        com = 0.5 * (bounds[0] + bounds[1])
        ext = self._safe_extents(mesh)
        axis_scale = float(max(1e-4, 0.08 * np.max(ext))) if ext is not None else 0.01
        colors = np.array(
            [
                [255, 60, 60],
                [255, 60, 60],
                [60, 255, 60],
                [60, 255, 60],
                [60, 60, 255],
                [60, 60, 255],
            ],
            dtype=np.uint8,
        )
        points = np.array(
            [
                [-axis_scale, 0.0, 0.0],
                [axis_scale, 0.0, 0.0],
                [0.0, -axis_scale, 0.0],
                [0.0, axis_scale, 0.0],
                [0.0, 0.0, -axis_scale],
                [0.0, 0.0, axis_scale],
            ],
            dtype=np.float32,
        )
        points += (com - center + shift).astype(np.float32)
        self._add_handle(
            self.server.scene.add_line_segments(
                name,
                points=points.reshape(-1, 2, 3),
                colors=colors.reshape(-1, 2, 3),
                line_width=2.0,
            )
        )

    def show_single(
        self,
        category: str,
        obj_mesh: trimesh.Trimesh,
        min_cov_mesh: Optional[trimesh.Trimesh],
        max_cov_mesh: Optional[trimesh.Trimesh],
        active_shell: str,
        min_ref_mesh: Optional[trimesh.Trimesh],
        max_ref_mesh: Optional[trimesh.Trimesh],
        is_saved: bool,
        scene_stats: Optional[dict] = None,
        show_com: bool = True,
        show_bbox: bool = False,
        show_bbox_dims: bool = False,
        show_hull: bool = False,
        hull_mesh: Optional[trimesh.Trimesh] = None,
        hand_mesh: Optional[trimesh.Trimesh] = None,
        hand_z: float = 0.0,
        hand_xy: Optional[tuple[float, float]] = None,
        title_text: Optional[str] = None,
        stats_text: Optional[str] = None,
    ) -> None:
        self._clear()
        self._scene_stats = scene_stats or {}
        hand_xy = hand_xy or (0.0, 0.0)
        shift = np.zeros(3, dtype=np.float32)
        cell_extent = 0.25
        for mesh in (obj_mesh, min_ref_mesh, max_ref_mesh, hull_mesh):
            if mesh is not None:
                ext = self._safe_extents(mesh)
                if ext is not None:
                    cell_extent = max(cell_extent, float(np.max(ext)))

        category_rgb = self._category_color(category)[:3]
        grounded_obj, _ = self._grounded_mesh(obj_mesh)
        if min_cov_mesh is not None:
            grounded_cov_min, _ = self._grounded_mesh(min_cov_mesh)
            self._set_mesh_handle(
                "_min_cov_handle",
                "/mesh_processing/object_cov_min",
                grounded_cov_min,
                (240, 140, 30),
                0.42 if active_shell == "min" else 0.24,
                shift,
                False,
                False,
                False,
            )
        else:
            self._set_mesh_handle(
                "_min_cov_handle",
                "/mesh_processing/object_cov_min",
                None,
                (240, 140, 30),
                0.24,
                shift,
                False,
                False,
                False,
            )
        if max_cov_mesh is not None:
            grounded_cov_max, _ = self._grounded_mesh(max_cov_mesh)
            self._set_mesh_handle(
                "_max_cov_handle",
                "/mesh_processing/object_cov_max",
                grounded_cov_max,
                (50, 160, 70),
                0.42 if active_shell == "max" else 0.20,
                shift,
                False,
                False,
                False,
            )
        else:
            self._set_mesh_handle(
                "_max_cov_handle",
                "/mesh_processing/object_cov_max",
                None,
                (50, 160, 70),
                0.20,
                shift,
                False,
                False,
                False,
            )
        if min_cov_mesh is None and max_cov_mesh is None:
            self._set_mesh_handle(
                "_object_handle",
                "/mesh_processing/object",
                grounded_obj,
                (80, 180, 255),
                0.82,
                shift,
                True,
                True,
            )
        else:
            self._set_mesh_handle(
                "_object_handle",
                "/mesh_processing/object",
                None,
                (80, 180, 255),
                0.82,
                shift,
                True,
                True,
            )
        if min_ref_mesh is not None:
            grounded_ref, _ = self._grounded_mesh(min_ref_mesh)
            self._set_mesh_handle(
                "_ref_min_handle",
                "/mesh_processing/ref_min",
                grounded_ref,
                (120, 78, 18),
                0.55,
                shift,
                False,
                False,
                True,
            )
        else:
            self._set_mesh_handle(
                "_ref_min_handle",
                "/mesh_processing/ref_min",
                None,
                (120, 78, 18),
                0.55,
                shift,
                False,
                False,
                True,
            )
        if max_ref_mesh is not None:
            grounded_ref, _ = self._grounded_mesh(max_ref_mesh)
            self._set_mesh_handle(
                "_ref_max_handle",
                "/mesh_processing/ref_max",
                grounded_ref,
                (110, 28, 28),
                0.55,
                shift,
                False,
                False,
                True,
            )
        else:
            self._set_mesh_handle(
                "_ref_max_handle",
                "/mesh_processing/ref_max",
                None,
                (110, 28, 28),
                0.55,
                shift,
                False,
                False,
                True,
            )
        if show_bbox:
            self._set_bbox_handle(
                grounded_obj,
                shift,
                (40, 70, 160) if active_shell == "min" else (20, 120, 60),
            )
        else:
            self._set_bbox_handle(None, shift, category_rgb)
        if show_bbox_dims:
            if min_cov_mesh is not None and max_cov_mesh is not None:
                min_ext = self._safe_extents(grounded_cov_min)
                max_ext = self._safe_extents(grounded_cov_max)
                if min_ext is None or max_ext is None:
                    dims_text = ""
                else:
                    dims_text = (
                        f"min {self._fmt_extent(min_ext[0])}, {self._fmt_extent(min_ext[1])}, {self._fmt_extent(min_ext[2])}\n"
                        f"max {self._fmt_extent(max_ext[0])}, {self._fmt_extent(max_ext[1])}, {self._fmt_extent(max_ext[2])}"
                    )
            else:
                ext = self._safe_extents(grounded_obj)
                dims_text = "" if ext is None else f"{self._fmt_extent(ext[0])}, {self._fmt_extent(ext[1])}, {self._fmt_extent(ext[2])}"
            self._set_label(
                "_bbox_dims_handle",
                "/mesh_processing/bbox_dims",
                dims_text,
                (18, 42, 92),
                shift + np.array([0.0, 0.0, cell_extent * 0.95], dtype=np.float32),
                "bottom-center",
                1.5,
            )
        else:
            self._set_label(
                "_bbox_dims_handle",
                "/mesh_processing/bbox_dims",
                "",
                (18, 42, 92),
                shift,
                "bottom-center",
                1.5,
            )
        if show_hull and hull_mesh is not None:
            grounded_hull, _ = self._grounded_mesh(hull_mesh)
            self._add_mesh(
                "/mesh_processing/hull",
                grounded_hull,
                color=(240, 200, 80),
                opacity=0.25,
                position=shift,
                cast_shadow=False,
                receive_shadow=False,
            )
        self._replace_hand_handle(hand_mesh, hand_z, hand_xy)
        if title_text:
            category_title, object_title = self._split_title_text(title_text)
            title_base = shift + np.array([0.0, 0.0, cell_extent * 0.80], dtype=np.float32)
            if category_title:
                self._set_label(
                    "_title_category_handle",
                    "/mesh_processing/viewer_title_category",
                    category_title,
                    (72, 32, 110),
                    title_base + np.array([0.0, 0.0, cell_extent * 0.035], dtype=np.float32),
                    "bottom-center",
                    3.10,
                )
            else:
                self._set_label(
                    "_title_category_handle",
                    "/mesh_processing/viewer_title_category",
                    "",
                    (72, 32, 110),
                    title_base,
                    "bottom-center",
                    3.10,
                )
            self._set_label(
                "_title_object_handle",
                "/mesh_processing/viewer_title_object",
                object_title,
                (20, 48, 118),
                title_base,
                "top-center",
                1.75,
            )
        else:
            self._set_label(
                "_title_category_handle",
                "/mesh_processing/viewer_title_category",
                "",
                (72, 32, 110),
                shift,
                "bottom-center",
                3.10,
            )
            self._set_label(
                "_title_object_handle",
                "/mesh_processing/viewer_title_object",
                "",
                (20, 48, 118),
                shift,
                "top-center",
                1.75,
            )
        if stats_text:
            formatted_stats = self._format_stats_text(stats_text)
            if formatted_stats:
                self._set_label(
                    "_stats_handle",
                    "/mesh_processing/viewer_stats",
                    formatted_stats,
                    (18, 76, 28),
                    shift + np.array([cell_extent * 0.68, 0.0, cell_extent * 0.50], dtype=np.float32),
                    "center-left",
                    1.65,
                )
            else:
                self._set_label(
                    "_stats_handle",
                    "/mesh_processing/viewer_stats",
                    "",
                    (18, 76, 28),
                    shift,
                    "center-left",
                    1.65,
                )
        else:
            self._set_label(
                "_stats_handle",
                "/mesh_processing/viewer_stats",
                "",
                (18, 76, 28),
                shift,
                "center-left",
                1.65,
            )

    def update_hand(
        self,
        hand_mesh: Optional[trimesh.Trimesh],
        hand_z: float = 0.0,
        hand_xy: Optional[tuple[float, float]] = None,
    ) -> None:
        self._replace_hand_handle(hand_mesh, hand_z, hand_xy or (0.0, 0.0))
