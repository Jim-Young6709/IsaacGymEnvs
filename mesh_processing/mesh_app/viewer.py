from typing import Dict, Optional, Tuple
import numpy as np
import trimesh
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
try:
    from pyqtgraph.opengl import GLTextItem
except Exception:
    GLTextItem = None


class MeshViewer(gl.GLViewWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setCameraPosition(distance=1.6)
        self.setBackgroundColor((32, 33, 36, 255))

        # Background grid handled per-cell in grid view.

        self.obj_item: Optional[gl.GLMeshItem] = None
        self.ref_item: Optional[gl.GLMeshItem] = None
        self.hand_item: Optional[gl.GLMeshItem] = None
        self.pc_item: Optional[gl.GLScatterPlotItem] = None
        self.com_items = []
        self.grid_items: list[gl.GLGraphicsItem.GLGraphicsItem] = []
        self.grid_labels = []
        self.bbox_items = []
        self.overlay = QtWidgets.QWidget(self)
        self.overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.overlay.setStyleSheet("background: transparent;")
        self.overlay_layout = QtWidgets.QGridLayout(self.overlay)
        self.overlay_layout.setContentsMargins(8, 8, 8, 8)
        self.overlay_layout.setSpacing(4)
        self.overlay_labels = []
        self.selected_label = QtWidgets.QLabel("", self.overlay)
        self.selected_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.selected_label.setStyleSheet("color: #8ab4f8; font-weight: 700; font-size: 16pt;")
        self.selected_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self.spin_timer = QtCore.QTimer()
        self.spin_timer.timeout.connect(self._spin_tick)

    def _spin_tick(self):
        self.orbit(1.0, 0.0)

    def toggle_spin(self):
        if self.spin_timer.isActive():
            self.spin_timer.stop()
        else:
            self.spin_timer.start(15)

    def _clear_item(self, item):
        if item is not None:
            self.removeItem(item)

    def _clear_com(self):
        for item in self.com_items:
            self.removeItem(item)
        self.com_items = []

    def _clear_bbox(self):
        for item in self.bbox_items:
            self.removeItem(item)
        self.bbox_items = []

    def _clear_grid(self):
        for item in self.grid_items:
            self.removeItem(item)
        self.grid_items = []
        self._clear_bbox()
        for item in self.grid_labels:
            self.removeItem(item)
        self.grid_labels = []
        for label in self.overlay_labels:
            label.deleteLater()
        self.overlay_labels = []

    def set_selected_text(self, text: str) -> None:
        self.selected_label.setText(text)

    def show_scene(
        self,
        obj_mesh: trimesh.Trimesh,
        ref_mesh: Optional[trimesh.Trimesh] = None,
        obj_pointcloud: Optional[np.ndarray] = None,
        com_point: Optional[np.ndarray] = None,
        hand_mesh: Optional[trimesh.Trimesh] = None,
        show_pointcloud: bool = True,
        show_com: bool = True,
    ) -> None:
        self._clear_item(self.obj_item)
        self._clear_item(self.ref_item)
        self._clear_item(self.hand_item)
        self._clear_item(self.pc_item)
        self._clear_com()
        self._clear_grid()
        self._clear_bbox()
        self.obj_item = None
        self.ref_item = None
        self.hand_item = None
        self.pc_item = None

        obj_extents = obj_mesh.bounding_box.extents
        obj_width = obj_extents[0] if obj_extents is not None else 1.0
        ref_width = 1.0
        if ref_mesh is not None:
            ref_extents = ref_mesh.bounding_box.extents
            ref_width = ref_extents[0] if ref_extents is not None else 1.0

        padding = ref_width * 1.0
        ref_center_pos = - (ref_width / 2.0 + padding / 2.0)
        obj_center_pos = + (obj_width / 2.0 + padding / 2.0)

        def prepare_mesh(mesh: trimesh.Trimesh, shift_x: float) -> Tuple[np.ndarray, np.ndarray]:
            verts = mesh.vertices
            c_local = verts.mean(axis=0) if len(verts) > 0 else np.zeros(3)
            verts_centered = verts - c_local
            verts_final = verts_centered + np.array([shift_x, 0.0, 0.0])
            return verts_final, c_local

        # Object mesh
        obj_verts, obj_center = prepare_mesh(obj_mesh, obj_center_pos)
        obj_item = gl.GLMeshItem(vertexes=obj_verts, faces=obj_mesh.faces, drawEdges=True, drawFaces=True, smooth=False)
        obj_item.setColor((0.3, 0.8, 1.0, 0.35))
        obj_item.setGLOptions("translucent")
        self.addItem(obj_item)
        self.obj_item = obj_item

        # Reference mesh
        if ref_mesh is not None:
            ref_verts, _ = prepare_mesh(ref_mesh, ref_center_pos)
            ref_item = gl.GLMeshItem(vertexes=ref_verts, faces=ref_mesh.faces, drawEdges=True, drawFaces=True, smooth=False)
            ref_item.setColor((1.0, 0.3, 0.3, 1.0))
            ref_item.setGLOptions("opaque")
            self.addItem(ref_item)
            self.ref_item = ref_item

        # LEAP hand mesh
        if hand_mesh is not None:
            hand_verts, _ = prepare_mesh(hand_mesh, obj_center_pos)
            hand_item = gl.GLMeshItem(vertexes=hand_verts, faces=hand_mesh.faces, drawEdges=True, drawFaces=True, smooth=False)
            hand_item.setColor((0.8, 0.8, 0.8, 0.4))
            hand_item.setGLOptions("translucent")
            self.addItem(hand_item)
            self.hand_item = hand_item

        # Pointcloud
        if show_pointcloud and obj_pointcloud is not None:
            pc = np.asarray(obj_pointcloud, dtype=np.float32)
            if pc.ndim == 2 and pc.shape[1] == 3 and pc.shape[0] > 0:
                pc_final = (pc - obj_center) + np.array([obj_center_pos, 0.0, 0.0])
                pc_item = gl.GLScatterPlotItem(
                    pos=pc_final,
                    size=2.0,
                    color=(1.0, 1.0, 0.2, 0.9),
                    pxMode=True,
                )
                self.addItem(pc_item)
                self.pc_item = pc_item

        # COM axes
        if show_com and com_point is not None:
            com = np.asarray(com_point, dtype=np.float32).reshape(3,)
            com_final = (com - obj_center) + np.array([obj_center_pos, 0.0, 0.0])
            s = float(max(1e-4, 0.08 * float(np.max(obj_extents)))) if obj_extents is not None else 0.01
            axes = [
                np.array([[-s, 0, 0], [s, 0, 0]]) + com_final,
                np.array([[0, -s, 0], [0, s, 0]]) + com_final,
                np.array([[0, 0, -s], [0, 0, s]]) + com_final,
            ]
            for pts in axes:
                item = gl.GLLinePlotItem(
                    pos=pts,
                    color=(0.2, 1.0, 0.2, 1.0),
                    width=4,
                    antialias=True,
                    mode="lines",
                )
                self.addItem(item)
                self.com_items.append(item)

    def _category_color(self, category: str) -> Tuple[float, float, float, float]:
        h = abs(hash(category)) % 360
        c = 0.7
        x = c * (1 - abs((h / 60.0) % 2 - 1))
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
        m = 0.2
        return (r + m, g + m, b + m, 1.0)

    def show_grid(self, items, cols: int = 3, selected_category: Optional[str] = None, show_labels: bool = True, show_com: bool = True, show_bbox: bool = False, show_bbox_dims: bool = False, show_hull: bool = False, focus_mode: Optional[str] = None, hand_mesh: Optional[trimesh.Trimesh] = None, hand_z: float = 0.0, hand_xy: Optional[tuple[float, float]] = None) -> None:
        self._clear_item(self.obj_item)
        self._clear_item(self.ref_item)
        self._clear_item(self.hand_item)
        self._clear_item(self.pc_item)
        self._clear_com()
        self._clear_grid()
        self._clear_bbox()

        if not items:
            return

        cols = max(1, cols)
        rows = int(np.ceil(len(items) / float(cols)))

        # Precompute cell sizes
        cell_width = 0.2
        cell_height = 0.2
        ref_width = 0.1
        for _, obj_mesh, ref_mesh, _, _, _, _ in items:
            if obj_mesh is not None:
                ex = obj_mesh.bounding_box.extents
                cell_width = max(cell_width, float(ex[0]))
                cell_height = max(cell_height, float(ex[1]))
            if ref_mesh is not None:
                ex = ref_mesh.bounding_box.extents
                ref_width = max(ref_width, float(ex[0]))
                cell_height = max(cell_height, float(ex[1]))

        scale_factor = 0.1
        obj_ref_gap_x = ref_width * 0.8 * scale_factor
        obj_ref_gap_y = cell_height * 0.8 * scale_factor
        cell_gap = 0.8 * scale_factor
        cell_span = max(cell_width + ref_width + obj_ref_gap_x, cell_height + obj_ref_gap_y) + cell_gap
        cell_span_x = cell_span
        cell_span_y = cell_span

        def add_mesh(mesh, shift, color, translucent=False):
            if mesh is None:
                return
            verts = mesh.vertices
            c_local = verts.mean(axis=0) if len(verts) > 0 else np.zeros(3)
            verts_final = (verts - c_local) + shift
            item = gl.GLMeshItem(vertexes=verts_final, faces=mesh.faces, drawEdges=True, drawFaces=True, smooth=False)
            item.setColor(color)
            item.setGLOptions("translucent" if translucent else "opaque")
            self.addItem(item)
            self.grid_items.append(item)

        def add_com(mesh, shift):
            if mesh is None:
                return
            verts = mesh.vertices
            if verts.size == 0:
                return
            c_local = verts.mean(axis=0)
            bounds = mesh.bounds
            com = 0.5 * (bounds[0] + bounds[1])
            com_final = (com - c_local) + shift
            ext = mesh.bounding_box.extents
            s = float(max(1e-4, 0.08 * np.max(ext))) if ext is not None else 0.01
            axes = [
                np.array([[-s, 0, 0], [s, 0, 0]]) + com_final,
                np.array([[0, -s, 0], [0, s, 0]]) + com_final,
                np.array([[0, 0, -s], [0, 0, s]]) + com_final,
            ]
            for pts in axes:
                item = gl.GLLinePlotItem(
                    pos=pts,
                    color=(0.2, 1.0, 0.2, 1.0),
                    width=3,
                    antialias=True,
                    mode="lines",
                )
                self.addItem(item)
                self.grid_items.append(item)

        def add_bbox(mesh, shift, color):
            if mesh is None:
                return
            bounds = mesh.bounds
            if bounds is None:
                return
            corners = np.array([
                [bounds[0][0], bounds[0][1], bounds[0][2]],
                [bounds[1][0], bounds[0][1], bounds[0][2]],
                [bounds[1][0], bounds[1][1], bounds[0][2]],
                [bounds[0][0], bounds[1][1], bounds[0][2]],
                [bounds[0][0], bounds[0][1], bounds[1][2]],
                [bounds[1][0], bounds[0][1], bounds[1][2]],
                [bounds[1][0], bounds[1][1], bounds[1][2]],
                [bounds[0][0], bounds[1][1], bounds[1][2]],
            ], dtype=np.float32)
            verts = mesh.vertices
            c_local = verts.mean(axis=0) if len(verts) > 0 else np.zeros(3)
            corners = (corners - c_local) + shift
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            lines = []
            for a, b in edges:
                lines.append(corners[a])
                lines.append(corners[b])
                lines.append([np.nan, np.nan, np.nan])
            pts = np.array(lines, dtype=np.float32)
            item = gl.GLLinePlotItem(pos=pts, color=color, width=2, antialias=True, mode="lines")
            self.addItem(item)
            self.bbox_items.append(item)

            # Optional translucent box faces for full box visibility.
            try:
                box = trimesh.creation.box(extents=bounds[1] - bounds[0])
                box.vertices = (box.vertices - box.vertices.mean(axis=0)) + shift
                face_item = gl.GLMeshItem(vertexes=box.vertices, faces=box.faces, drawEdges=False, drawFaces=True, smooth=False)
                face_item.setColor((color[0], color[1], color[2], 0.15))
                face_item.setGLOptions("translucent")
                self.addItem(face_item)
                self.bbox_items.append(face_item)
            except Exception:
                pass

        cell_centers = {}
        hand_xy = hand_xy or (0.0, 0.0)
        for idx, (category, obj_mesh, ref_mesh, is_saved, hull_mesh, fixed_preview, fixed_overlay) in enumerate(items):
            row = idx // cols
            col = idx % cols
            base_x = col * cell_span_x
            base_y = -row * cell_span_y
            cell_centers[category] = (base_x, base_y, 0.0)
            ref_shift = np.array([base_x - (ref_width / 2.0 + obj_ref_gap_x / 2.0), base_y, 0.0])
            obj_shift = np.array([base_x + (cell_width / 2.0 + obj_ref_gap_x / 2.0), base_y, 0.0])
            hand_shift = np.array([base_x + hand_xy[0], base_y + hand_xy[1], hand_z])

            is_selected = category == selected_category
            base_color = self._category_color(category)
            obj_color = (0.3, 0.7, 0.9, 0.4)
            ref_color = (1.0, 0.3, 0.3, 0.9)

            grid_size = max(0.05, cell_span * 0.9)
            if category == selected_category:
                grid_color = (0.4, 0.9, 0.4, 0.9)
            else:
                grid_color = (0.4, 0.7, 1.0, 0.9) if is_saved else (0.9, 0.2, 0.2, 0.9)
            spacing = 0.02
            half = grid_size / 2.0
            xs = np.arange(-half, half + 1e-6, spacing)
            ys = np.arange(-half, half + 1e-6, spacing)

            points = []
            for x in xs:
                points.append([base_x + x, base_y - half, -0.02])
                points.append([base_x + x, base_y + half, -0.02])
                points.append([np.nan, np.nan, np.nan])
            for y in ys:
                points.append([base_x - half, base_y + y, -0.02])
                points.append([base_x + half, base_y + y, -0.02])
                points.append([np.nan, np.nan, np.nan])

            pts = np.array(points, dtype=np.float32)
            grid_lines = gl.GLLinePlotItem(pos=pts, color=grid_color, width=1, antialias=True, mode="lines")
            self.addItem(grid_lines)
            self.grid_items.append(grid_lines)

            if show_labels and GLTextItem is not None:
                try:
                    font = QtGui.QFont("Sans Serif", 16)
                    font.setBold(True)
                    label = GLTextItem(
                        pos=(base_x, base_y + cell_span_y * 0.35, 0.1),
                        text=category,
                        color=(1.0, 1.0, 1.0, 1.0),
                        font=font,
                    )
                    try:
                        label.setDepthValue(10)
                    except Exception:
                        pass
                    self.addItem(label)
                    self.grid_labels.append(label)
                except Exception:
                    pass

            add_mesh(ref_mesh, ref_shift, ref_color, translucent=False)
            add_mesh(obj_mesh, obj_shift, obj_color, translucent=True)
            if show_hull and hull_mesh is not None and category == selected_category:
                add_mesh(hull_mesh, obj_shift, (1.0, 0.4, 0.0, 0.35), translucent=True)
            if fixed_overlay is not None and category == selected_category:
                add_mesh(fixed_overlay, obj_shift, (0.9, 0.2, 0.9, 0.9), translucent=True)
            if show_com and category == selected_category:
                add_com(obj_mesh, obj_shift)
            if show_bbox:
                bbox_color = (1.0, 0.75, 0.2, 0.9)
                add_bbox(obj_mesh, obj_shift, bbox_color)
                if show_bbox_dims and category == selected_category:
                    ext = obj_mesh.bounding_box.extents
                    if ext is not None:
                        dim_label = QtWidgets.QLabel(
                            f"X:{ext[0]:.3f} Y:{ext[1]:.3f} Z:{ext[2]:.3f}",
                            self.overlay,
                        )
                        dim_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                        dim_label.setStyleSheet("color: #ffcc66; font-weight: 600; font-size: 12pt;")
                        self.overlay_layout.addWidget(dim_label, row, col, QtCore.Qt.AlignBottom)
                        self.overlay_labels.append(dim_label)

            if show_labels:
                label = QtWidgets.QLabel(category, self.overlay)
                label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                label.setStyleSheet("color: white; font-weight: 600; font-size: 14pt;")
                self.overlay_layout.addWidget(label, row, col, QtCore.Qt.AlignTop)
                self.overlay_labels.append(label)

            if hand_mesh is not None and category == selected_category:
                add_mesh(hand_mesh, hand_shift, (0.8, 0.8, 0.8, 0.4), translucent=True)

        self.overlay.setGeometry(0, 0, self.width(), self.height())
        self.overlay.raise_()
        self.selected_label.setGeometry(0, 0, self.width(), 28)

        # Camera focus: only apply when requested.
        if focus_mode:
            max_span = max(cell_span_x * cols, cell_span_y * rows)
            dist = max(0.05, max_span * 1.2)
            center = QtGui.QVector3D(0.0, 0.0, 0.0)
            if focus_mode == "selected":
                target = selected_category
                if target == "all_categories" and items:
                    target = items[0][0]
                if target:
                    if target in cell_centers:
                        cx, cy, cz = cell_centers[target]
                        center = QtGui.QVector3D(cx, cy, cz)
                    for category, obj_mesh, ref_mesh, _, _, _, _ in items:
                        if category != target:
                            continue
                        extents = []
                        if obj_mesh is not None:
                            extents.append(obj_mesh.bounding_box.extents)
                        if ref_mesh is not None:
                            extents.append(ref_mesh.bounding_box.extents)
                        if extents:
                            span = float(np.max(np.vstack(extents)))
                            dist = max(0.05, span * 4.5)
                        break
            else:
                center = QtGui.QVector3D((cols - 1) * cell_span_x / 2.0, -(rows - 1) * cell_span_y / 2.0, 0.0)
            self.opts["center"] = center
            self.setCameraPosition(distance=dist)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.overlay.setGeometry(0, 0, self.width(), self.height())
        self.selected_label.setGeometry(0, 0, self.width(), 28)

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            ev = QtGui.QMouseEvent(
                ev.type(),
                ev.localPos(),
                ev.windowPos(),
                ev.screenPos(),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                ev.modifiers(),
            )
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & QtCore.Qt.RightButton:
            ev = QtGui.QMouseEvent(
                ev.type(),
                ev.localPos(),
                ev.windowPos(),
                ev.screenPos(),
                QtCore.Qt.LeftButton,
                QtCore.Qt.LeftButton,
                ev.modifiers(),
            )
        super().mouseMoveEvent(ev)
