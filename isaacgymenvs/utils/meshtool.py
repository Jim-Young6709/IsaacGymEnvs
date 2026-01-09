import os
import sys
import shutil
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import trimesh

from PyQt5 import QtCore, QtWidgets, QtGui
import pyqtgraph.opengl as gl


# ========= CONFIG ========= #
MESH_ROOT = "/home/rayliu/IsaacGymEnvs/meshes"
OUTPUT_ROOT = "/home/rayliu/IsaacGymEnvs/meshes_sorted"
REF_PATH = "/home/rayliu/IsaacGymEnvs/meshes_debug/apple"

CATEGORY_DIRS = {
    "regular": "regular",
    "long": "long",
    "small": "small",
}

DISCARD_DIR = "discard"
# ========================== #


@dataclass
class MeshEntry:
    """Represents one object variant, e.g. alarm_clock/2.obj."""
    label: str          # e.g. 'alarm_clock'
    base_name: str      # e.g. '2'
    obj_path: str
    npy_path: Optional[str]
    glb_path: Optional[str]
    urdf_path: Optional[str]


def scan_meshes(root: str) -> List[MeshEntry]:
    entries: List[MeshEntry] = []
    for label in sorted(os.listdir(root)):
        label_dir = os.path.join(root, label)
        if not os.path.isdir(label_dir):
            continue

        for fname in sorted(os.listdir(label_dir)):
            if not fname.endswith(".obj"):
                continue
            base_name = os.path.splitext(fname)[0]  # '1', '2', '3', ...
            obj_path = os.path.join(label_dir, fname)

            def maybe(path):
                return path if os.path.isfile(path) else None

            npy_path = maybe(os.path.join(label_dir, base_name + ".npy"))
            glb_path = maybe(os.path.join(label_dir, base_name + ".glb"))
            urdf_path = maybe(os.path.join(label_dir, base_name + ".urdf"))

            entries.append(
                MeshEntry(
                    label=label,
                    base_name=base_name,
                    obj_path=obj_path,
                    npy_path=npy_path,
                    glb_path=glb_path,
                    urdf_path=urdf_path,
                )
            )
    return entries

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class MeshViewer(gl.GLViewWidget):
    """3D mesh viewer using pyqtgraph.opengl with a reference mesh."""
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setCameraPosition(distance=0.8)
        
        # Dark grey background
        self.setBackgroundColor((32, 33, 36, 255))

        self.obj_item: Optional[gl.GLMeshItem] = None
        self.ref_item: Optional[gl.GLMeshItem] = None
        self.ref_mesh: Optional[trimesh.Trimesh] = None

        # Grid in the X–Y plane, Z up
        g = gl.GLGridItem()
        g.setSize(10, 10)
        g.scale(1, 1, 1)
        self.addItem(g)

        self._load_reference_mesh()

        # Animation Timer
        self.spin_timer = QtCore.QTimer()
        self.spin_timer.timeout.connect(self._spin_tick)

        self.pc_item: Optional[gl.GLScatterPlotItem] = None
        self.com_geom_items: List[gl.GLLinePlotItem] = []
        self.com_urdf_items: List[gl.GLLinePlotItem] = []

    def _clear_overlays(self):
        if self.pc_item is not None:
            self.removeItem(self.pc_item)
            self.pc_item = None

        for it in self.com_geom_items:
            self.removeItem(it)
        self.com_geom_items.clear()

        for it in self.com_urdf_items:
            self.removeItem(it)
        self.com_urdf_items.clear()

    def _spin_tick(self):
        self.orbit(1.0, 0.0)

    def toggle_spin(self):
        if self.spin_timer.isActive():
            self.spin_timer.stop()
        else:
            self.spin_timer.start(15)

    def _load_reference_mesh(self):
        """Load the apple reference OBJ."""
        # Use global config if possible, else hardcode fallback
        ref_path = "/home/rayliu/IsaacGymEnvs/meshes_debug/apple" 
        # (Note: In the full app, we update this via update_reference_mesh, 
        # so this default is just for startup safety)
        
        try:
            if os.path.isdir(ref_path):
                obj_files = sorted(f for f in os.listdir(ref_path) if f.lower().endswith(".obj"))
                if obj_files:
                    mesh_path = os.path.join(ref_path, obj_files[0])
                    m = trimesh.load(mesh_path, force='mesh')
                    if not isinstance(m, trimesh.Trimesh): m = m.dump().sum()
                    self.ref_mesh = m
        except Exception as e:
            print(f"Ref Load Error: {e}")
            self.ref_mesh = None

    def update_reference_mesh(self, mesh_path: str, scale: float = 1.0):
        try:
            # print(f"[MeshViewer] Loading ref: {mesh_path} (Scale={scale})")
            m = trimesh.load(mesh_path, force='mesh')
            if not isinstance(m, trimesh.Trimesh):
                m = m.dump().sum()
            if scale != 1.0:
                m.apply_scale(scale)
            self.ref_mesh = m
            if self.ref_item is not None:
                self.removeItem(self.ref_item)
                self.ref_item = None
        except Exception as e:
            print(f"Ref Update Error: {e}")

    def show_pair(
        self,
        obj_mesh: trimesh.Trimesh,
        scale: float = 1.0,
        rot_z_degrees: float = 0.0,
        rot_x_degrees: float = 0.0,
        rot_y_degrees: float = 0.0,
        reset_camera: bool = False,
        obj_pointcloud: Optional[np.ndarray] = None,   # Nx3 in same frame as obj_mesh
        obj_com_geom: Optional[np.ndarray] = None,          # (3,)
        obj_com_urdf: Optional[np.ndarray] = None,
        show_pointcloud: bool = True,
        show_com: bool = True,
    ):
        
        # 1. Cleanup old items
        if self.obj_item is not None:
            self.removeItem(self.obj_item)
            self.obj_item = None
        if self.ref_item is not None:
            self.removeItem(self.ref_item)
            self.ref_item = None
        self._clear_overlays()

        # 2. Helper to transform mesh
        def apply_transforms(mesh, sx, rx, ry, rz):
            m = mesh.copy()
            m.apply_scale(sx)
            if rx % 360 != 0:
                ax = np.deg2rad(rx)
                Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
                m.vertices = m.vertices @ Rx.T
            if ry % 360 != 0:
                ay = np.deg2rad(ry)
                Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
                m.vertices = m.vertices @ Ry.T
            if rz % 360 != 0:
                az = np.deg2rad(rz)
                Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
                m.vertices = m.vertices @ Rz.T
            return m

        # 3. Prepare Meshes
        meshes = []
        obj = apply_transforms(obj_mesh, scale, rot_x_degrees, rot_y_degrees, rot_z_degrees)
        meshes.append(("obj", obj))
        
        if self.ref_mesh is not None:
            # Note: ref_mesh is already scaled by update_reference_mesh
            meshes.append(("ref", self.ref_mesh.copy()))

        if not meshes: return

        # 4. [FIXED] Spacing Logic (Anchor Reference)
        
        # A. Measure Reference Width (Static)
        if self.ref_mesh is not None:
            # Measure width along X axis
            ref_extents = self.ref_mesh.bounding_box.extents
            ref_width = ref_extents[0] if ref_extents is not None else 1.0
        else:
            ref_width = 1.0
            
        # B. Measure Object Width (Dynamic)
        obj_extents = obj.bounding_box.extents
        obj_width = obj_extents[0] if obj_extents is not None else 1.0
        
        # C. Define Padding relative to Reference (so it doesn't jump when Obj grows)
        padding = ref_width * 1.0

        # D. Calculate Positions
        # Reference is anchored to the LEFT of the origin
        # Object is anchored to the RIGHT of the origin
        
        # Ref Center = - (Half Width + Half Padding)
        ref_center_pos = - (ref_width / 2.0 + padding / 2.0)
        
        # Obj Center = + (Half Width + Half Padding)
        obj_center_pos = + (obj_width / 2.0 + padding / 2.0)

        # 5. Add to Scene
        # Need to gather all extents for camera calculation later
        all_extents = []
        
        for name, m in meshes:
            verts = m.vertices
            # Local center of geometry (to normalize mesh to 0,0,0 first)
            c_local = verts.mean(axis=0) if len(verts) > 0 else np.zeros(3)
            verts_centered = verts - c_local
            
            # Apply Layout Shift
            if name == "obj":
                shift = np.array([obj_center_pos, 0, 0])
                all_extents.append(obj_extents if obj_extents is not None else np.zeros(3))
                # --- Pointcloud overlay (object only) ---
                if show_pointcloud and obj_pointcloud is not None:
                    pc = np.asarray(obj_pointcloud, dtype=np.float32)
                    if pc.ndim == 2 and pc.shape[1] == 3 and pc.shape[0] > 0:
                        pc_final = (pc - c_local) + shift
                        self.pc_item = gl.GLScatterPlotItem(
                            pos=pc_final,
                            size=2.0,
                            color=(1.0, 1.0, 0.2, 0.9),  # yellow-ish
                            pxMode=True
                        )
                        self.addItem(self.pc_item)

                # --- COM overlay (object only) ---
                def draw_com(com, color, target_list):
                    com = np.asarray(com, dtype=np.float32).reshape(3,)
                    com_final = (com - c_local) + shift

                    ex = obj_extents if obj_extents is not None else np.array([0.1, 0.1, 0.1])
                    s = float(max(1e-4, 0.08 * np.max(ex)))

                    axes = [
                        np.array([[-s, 0, 0], [s, 0, 0]]) + com_final,
                        np.array([[0, -s, 0], [0, s, 0]]) + com_final,
                        np.array([[0, 0, -s], [0, 0, s]]) + com_final,
                    ]

                    for pts in axes:
                        it = gl.GLLinePlotItem(
                            pos=pts,
                            color=color,
                            width=4,
                            antialias=True,
                            mode="lines"
                        )
                        self.addItem(it)
                        target_list.append(it)

                if show_com:
                    if obj_com_geom is not None:
                        draw_com(obj_com_geom, (0.2, 1.0, 0.2, 1.0), self.com_geom_items)

                    if obj_com_urdf is not None:
                        # print("[MeshViewer] Drawing URDF COM, ", obj_com_urdf)
                        draw_com(obj_com_urdf, (1.0, 0.2, 1.0, 1.0), self.com_urdf_items)
            else:
                shift = np.array([ref_center_pos, 0, 0])
                all_extents.append(ref_extents if ref_extents is not None else np.zeros(3))
                
            verts_final = verts_centered + shift

            item = gl.GLMeshItem(vertexes=verts_final, faces=m.faces, drawEdges=True, drawFaces=True, smooth=False)
            if name == "obj":
                item.setColor((0.3, 0.8, 1.0, 0.4))     # <- alpha 0.25 (tweak 0.15–0.4)
                item.setGLOptions("translucent")
            else:
                item.setColor((1.0, 0.3, 0.3, 1.0))
                item.setGLOptions("opaque")
            self.addItem(item)
            
            if name == "obj": self.obj_item = item
            else: self.ref_item = item

        # # 6. Smart Camera Zoom
        # if reset_camera:
        #     # Calculate total width of the scene for zooming
        #     if all_extents:
        #         max_extent = float(np.max(np.vstack(all_extents)))
        #     else:
        #         max_extent = 1.0
            
        #     dist = max(0.1, 1. * max_extent)
        #     self.setCameraPosition(distance=dist)

    # Mouse Events (Right click = Rotate)
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            ev = QtGui.QMouseEvent(ev.type(), ev.localPos(), ev.windowPos(), ev.screenPos(),
                                   QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, ev.modifiers())
        super().mousePressEvent(ev)
    def mouseMoveEvent(self, ev):
        if ev.buttons() & QtCore.Qt.RightButton:
            ev = QtGui.QMouseEvent(ev.type(), ev.localPos(), ev.windowPos(), ev.screenPos(),
                                   QtCore.Qt.LeftButton, QtCore.Qt.LeftButton, ev.modifiers())
        super().mouseMoveEvent(ev)


class MeshToolApp(QtWidgets.QWidget):
    def __init__(self, entries: List[MeshEntry]):
        super().__init__()
        self.entries = entries
        self.index = 0

        # Cache loaded meshes to avoid reloading when going back
        self.mesh_cache: Dict[int, trimesh.Trimesh] = {}

        # Undo stack: list of (index, category)
        self.undo_stack: List[tuple[int, str]] = []

        # Per-object rotation (degrees) and scale
        self.rotation_z_degrees: float = 0.0
        self.rotation_x_degrees: float = 0.0
        self.rotation_y_degrees: float = 0.0

        # Per-object state: idx -> {scale, rot_x, rot_y, rot_z}
        self.object_state: Dict[int, Dict[str, float]] = {}
        self.last_state: Optional[Dict[str, float]] = None


        self._build_ui()
        self._load_current()

    # ---------- UI setup ---------- #
    def _build_ui(self):
        self.setWindowTitle("Mesh Sorting Tool")

        # Global style
        self.setStyleSheet("""
        QWidget { background-color: #202124; color: #e8eaed; font-family: "Segoe UI", Arial; font-size: 11pt; }
        QGroupBox { border: 1px solid #3c4043; margin-top: 8px; border-radius: 6px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa0a6; font-weight: 500; }
        QLabel#indexLabel { font-weight: 600; color: #8ab4f8; }
        QLabel#nameLabel { font-weight: 500; color: #e8eaed; }
        QLabel#stateLabel { color: #fbbc05; font-size: 9.5pt; }
        QPushButton { background-color: #303134; border: 1px solid #5f6368; border-radius: 6px; padding: 6px 10px; }
        QPushButton:hover { background-color: #3c4043; }
        QPushButton#accentButton { background-color: #1a73e8; border-color: #1a73e8; color: white; }
        QPushButton#accentButton:hover { background-color: #4285f4; }
        QDoubleSpinBox { background-color: #303134; border: 1px solid #5f6368; border-radius: 4px; padding: 2px 4px; }
        """)

        main_layout = QtWidgets.QHBoxLayout(self)

        # 1. Left: Viewer
        self.viewer = MeshViewer()
        main_layout.addWidget(self.viewer, stretch=2)

        # 2. Right: Controls
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setSpacing(6)
        right_panel.setContentsMargins(5, 5, 5, 5)

        # -- Header --
        header_layout = QtWidgets.QHBoxLayout()
        title_layout = QtWidgets.QVBoxLayout()
        title_layout.setSpacing(2)
        
        self.title_label = QtWidgets.QLabel("Mesh Sorting Tool")
        self.title_label.setStyleSheet("font-size: 18pt; font-weight: 600;")
        
        self.subtitle_label = QtWidgets.QLabel("Scale/Sort for Isaac Gym")
        self.subtitle_label.setStyleSheet("color: #9aa0a6; font-size: 10pt;")
        
        title_layout.addWidget(self.title_label)
        title_layout.addWidget(self.subtitle_label)
        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        help_btn = QtWidgets.QPushButton("?")
        help_btn.setFixedSize(24, 24)
        help_btn.clicked.connect(self._show_help)
        header_layout.addWidget(help_btn, alignment=QtCore.Qt.AlignTop)

        right_panel.addLayout(header_layout)
        right_panel.addSpacing(10)

        # -- Group 1: Object Info --
        obj_group = QtWidgets.QGroupBox("Object")
        obj_layout = QtWidgets.QVBoxLayout()
        obj_layout.setSpacing(4)
        
        self.lbl_index = QtWidgets.QLabel(); self.lbl_index.setObjectName("indexLabel")
        self.lbl_name = QtWidgets.QLabel(); self.lbl_name.setObjectName("nameLabel")
        self.lbl_paths = QtWidgets.QLabel(); self.lbl_paths.setWordWrap(True)
        self.lbl_state = QtWidgets.QLabel(); self.lbl_state.setObjectName("stateLabel"); self.lbl_state.setWordWrap(True)
        
        obj_layout.addWidget(self.lbl_index)
        obj_layout.addWidget(self.lbl_name)
        obj_layout.addWidget(self.lbl_paths)
        obj_layout.addWidget(self.lbl_state)
        obj_group.setLayout(obj_layout)
        right_panel.addWidget(obj_group)

        # -- Group 2: Reference & Stats --
        geom_group = QtWidgets.QGroupBox("Reference & Geometry")
        geom_layout = QtWidgets.QVBoxLayout()
        geom_layout.setSpacing(4)

        # Ref Controls
        ref_controls = QtWidgets.QHBoxLayout()
        self.btn_load_ref = QtWidgets.QPushButton("Ref...")
        self.btn_load_ref.clicked.connect(self._on_change_reference)
        self.btn_load_ref.setMaximumWidth(60)
        
        self.ref_scale_spin = QtWidgets.QDoubleSpinBox()
        self.ref_scale_spin.setRange(0.001, 1000.0); self.ref_scale_spin.setDecimals(4)
        self.ref_scale_spin.setValue(1.0); self.ref_scale_spin.setSingleStep(0.1)
        self.ref_scale_spin.valueChanged.connect(self._on_ref_scale_changed)

        self.chk_sync_ref = QtWidgets.QCheckBox("Sync URDF")
        self.chk_sync_ref.setChecked(True)
        self.chk_sync_ref.stateChanged.connect(lambda: self._load_current())

        ref_controls.addWidget(self.btn_load_ref)
        ref_controls.addWidget(QtWidgets.QLabel("Scale:"))
        ref_controls.addWidget(self.ref_scale_spin)
        ref_controls.addWidget(self.chk_sync_ref)
        geom_layout.addLayout(ref_controls)

        # Stats Label
        self.lbl_geom = QtWidgets.QLabel()
        self.lbl_geom.setObjectName("geomLabel")
        self.lbl_geom.setWordWrap(True)
        geom_layout.addWidget(self.lbl_geom)

        # --- Diagnostics toggles (object only) ---
        diag_row = QtWidgets.QHBoxLayout()

        self.chk_show_pc = QtWidgets.QCheckBox("Show pointcloud")
        self.chk_show_pc.setChecked(True)
        self.chk_show_pc.stateChanged.connect(lambda: self._refresh_view_scale())

        self.chk_show_com = QtWidgets.QCheckBox("Show COM")
        self.chk_show_com.setChecked(True)
        self.chk_show_com.stateChanged.connect(lambda: self._refresh_view_scale())

        diag_row.addWidget(self.chk_show_pc)
        diag_row.addWidget(self.chk_show_com)
        diag_row.addStretch()

        geom_layout.addLayout(diag_row)
        
        geom_group.setLayout(geom_layout)
        right_panel.addWidget(geom_group)

        # -- Group 3: Scaling & Category --
        sc_group = QtWidgets.QGroupBox("Scaling & Category")
        sc_layout = QtWidgets.QVBoxLayout()
        sc_layout.setSpacing(6)

        # Slider/Spin
        scale_row = QtWidgets.QHBoxLayout()
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 3.0); self.scale_spin.setSingleStep(0.025); self.scale_spin.setValue(1.0)
        
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(10, 300); self.scale_slider.setValue(100)
        
        scale_row.addWidget(QtWidgets.QLabel("Scale:"))
        scale_row.addWidget(self.scale_spin)
        sc_layout.addLayout(scale_row)
        sc_layout.addWidget(self.scale_slider)

        # Load Mode Dropdown
        mode_row = QtWidgets.QHBoxLayout()
        self.load_mode_combo = QtWidgets.QComboBox()
        self.load_mode_combo.addItems(["1. Always Reset", "2. Keep Settings", "3. Load Saved"])
        self.load_mode_combo.setCurrentIndex(2)
        mode_row.addWidget(QtWidgets.QLabel("On Load:"))
        mode_row.addWidget(self.load_mode_combo)
        sc_layout.addLayout(mode_row)

        # Cat Buttons
        btn_reg = QtWidgets.QPushButton("Regular (1)"); btn_reg.setObjectName("accentButton")
        btn_lng = QtWidgets.QPushButton("Long (2)"); btn_lng.setObjectName("accentButton")
        btn_sml = QtWidgets.QPushButton("Small (3)"); btn_sml.setObjectName("accentButton")
        btn_dsc = QtWidgets.QPushButton("Discard (D)")

        btn_reg.clicked.connect(lambda: self._assign_category("regular"))
        btn_lng.clicked.connect(lambda: self._assign_category("long"))
        btn_sml.clicked.connect(lambda: self._assign_category("small"))
        btn_dsc.clicked.connect(self._discard_current)

        sc_layout.addWidget(btn_reg)
        sc_layout.addWidget(btn_lng)
        sc_layout.addWidget(btn_sml)
        sc_layout.addWidget(btn_dsc)
        
        sc_group.setLayout(sc_layout)
        right_panel.addWidget(sc_group)

        # -- Group 4: Navigation --
        nav_group = QtWidgets.QGroupBox("Navigation")
        nav_layout = QtWidgets.QVBoxLayout()
        
        r1 = QtWidgets.QHBoxLayout()
        b_prev = QtWidgets.QPushButton("Prev (K)"); b_prev.clicked.connect(self._prev)
        b_next = QtWidgets.QPushButton("Next (J)"); b_next.clicked.connect(self._next)
        r1.addWidget(b_prev); r1.addWidget(b_next)
        
        # [UPDATED BUTTON LABELS]
        r2 = QtWidgets.QHBoxLayout()
        b_rx = QtWidgets.QPushButton("X (E)"); b_rx.clicked.connect(self._rotate_x)
        b_ry = QtWidgets.QPushButton("Y (R)"); b_ry.clicked.connect(self._rotate_y) # Y mapped to R
        b_rz = QtWidgets.QPushButton("Z (T)"); b_rz.clicked.connect(self._rotate_z) # Z mapped to T
        r2.addWidget(b_rx); r2.addWidget(b_ry); r2.addWidget(b_rz)

        r3 = QtWidgets.QHBoxLayout()
        b_load = QtWidgets.QPushButton("Load Saved (L)"); b_load.clicked.connect(self._apply_saved_transform)
        b_spin = QtWidgets.QPushButton("Spin (S)"); b_spin.clicked.connect(self.viewer.toggle_spin)
        r3.addWidget(b_load); r3.addWidget(b_spin)

        nav_layout.addLayout(r1)
        nav_layout.addLayout(r2)
        nav_layout.addLayout(r3)
        nav_group.setLayout(nav_layout)
        right_panel.addWidget(nav_group)

        # -- Status & Stretch --
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #9aa0a6;")
        right_panel.addWidget(self.status)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, stretch=1)

        # [UPDATED SHORTCUTS]
        QtWidgets.QShortcut(QtCore.Qt.Key_1, self, activated=lambda: self._assign_category("regular"))
        QtWidgets.QShortcut(QtCore.Qt.Key_2, self, activated=lambda: self._assign_category("long"))
        QtWidgets.QShortcut(QtCore.Qt.Key_3, self, activated=lambda: self._assign_category("small"))
        QtWidgets.QShortcut(QtCore.Qt.Key_D, self, activated=self._discard_current)
        QtWidgets.QShortcut(QtCore.Qt.Key_J, self, activated=self._next)
        QtWidgets.QShortcut(QtCore.Qt.Key_K, self, activated=self._prev)
        
        # New Mapping: E->X, R->Y, T->Z
        QtWidgets.QShortcut(QtCore.Qt.Key_E, self, activated=self._rotate_x)
        QtWidgets.QShortcut(QtCore.Qt.Key_R, self, activated=self._rotate_y) 
        QtWidgets.QShortcut(QtCore.Qt.Key_T, self, activated=self._rotate_z)
        
        QtWidgets.QShortcut(QtCore.Qt.Key_L, self, activated=self._apply_saved_transform)
        QtWidgets.QShortcut(QtCore.Qt.Key_S, self, activated=self.viewer.toggle_spin)

        self.scale_slider.valueChanged.connect(self._on_scale_slider_changed)
        self.scale_spin.valueChanged.connect(self._on_scale_spin_changed)

    # ---------- Pointcloud & COM transforms ---------- #
    def _infer_com_z_midpoint(self, mesh_final: trimesh.Trimesh) -> np.ndarray:
        """
        Infer COM as the center of the bounding box (x_mid, y_mid, z_mid)
        of the (already transformed) mesh. This places the marker visually
        centered instead of forcing x=y=0.
        """
        b = mesh_final.bounds  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
        mid = 0.5 * (b[0] + b[1])  # [x_mid, y_mid, z_mid]
        return mid.astype(np.float64)

    def _get_urdf_com(self, entry: MeshEntry) -> Optional[np.ndarray]:
        if not entry.urdf_path or not os.path.isfile(entry.urdf_path):
            return None

        try:
            tree = ET.parse(entry.urdf_path)
            root = tree.getroot()
            inertial = root.find(".//link/inertial/origin")
            if inertial is None:
                return None

            xyz = inertial.get("xyz")
            if xyz is None:
                return None

            com = np.fromstring(xyz, sep=" ")
            if com.shape != (3,):
                return None

            return com.astype(np.float64)
        except Exception:
            return None
        
    def _transform_urdf_com_for_view(self, com: np.ndarray, entry: MeshEntry) -> np.ndarray:
        com = com.astype(np.float64)
        # User scale
        com *= float(self.scale_spin.value())

        # User rotations (same order as viewer)
        if self.rotation_x_degrees % 360 != 0:
            ax = np.deg2rad(self.rotation_x_degrees)
            Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
            com = Rx @ com
        if self.rotation_y_degrees % 360 != 0:
            ay = np.deg2rad(self.rotation_y_degrees)
            Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
            com = Ry @ com
        if self.rotation_z_degrees % 360 != 0:
            az = np.deg2rad(self.rotation_z_degrees)
            Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
            com = Rz @ com

        return com.astype(np.float32)

    def _transform_points_for_view(self, pts: np.ndarray, entry: MeshEntry) -> np.ndarray:
        """
        pts: raw Nx3 from NPY in the *raw* object frame.
        Returns points in the same coordinates as the mesh you pass to viewer.show_pair().
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return pts

        # Apply initial URDF scale ONLY if URDF exists
        if entry.urdf_path and os.path.isfile(entry.urdf_path):
            init_scale = self._get_urdf_initial_scale(entry.urdf_path)
        else:
            init_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        pts = pts * init_scale

        # Apply user scale
        user_scale = float(self.scale_spin.value())
        pts = pts * user_scale

        # Apply user rotations in same order as viewer: X then Y then Z (with .T usage)
        if self.rotation_x_degrees % 360 != 0:
            ax = np.deg2rad(self.rotation_x_degrees)
            Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
            pts = pts @ Rx.T
        if self.rotation_y_degrees % 360 != 0:
            ay = np.deg2rad(self.rotation_y_degrees)
            Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
            pts = pts @ Ry.T
        if self.rotation_z_degrees % 360 != 0:
            az = np.deg2rad(self.rotation_z_degrees)
            Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
            pts = pts @ Rz.T

        return pts.astype(np.float32)

    def _compute_com_for_view(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute a COM-like point in the same frame as the mesh passed to viewer.show_pair().
        Uses trimesh.center_mass when available; otherwise centroid.
        mesh is already URDF-scaled by _load_mesh() before it gets here.
        """
        m = mesh.copy()

        # Apply user scale
        m.apply_scale(float(self.scale_spin.value()))

        # Apply user rotations (same convention)
        def rot_apply(msh, deg, axis):
            if deg % 360 == 0:
                return
            a = np.deg2rad(deg)
            if axis == "x":
                R = np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
            elif axis == "y":
                R = np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
            else:
                R = np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
            msh.vertices = msh.vertices @ R.T

        rot_apply(m, self.rotation_x_degrees, "x")
        rot_apply(m, self.rotation_y_degrees, "y")
        rot_apply(m, self.rotation_z_degrees, "z")

        try:
            if m.is_volume:
                return np.asarray(m.center_mass, dtype=np.float32)
        except Exception:
            pass

        # fallback: centroid of vertices
        return np.asarray(m.vertices.mean(axis=0), dtype=np.float32)

    # ---------- Help popup ---------- #
    def _show_help(self):
        text = (
            "<b>Keyboard Shortcuts</b><br><br>"
            "<b>1</b> – Regular<br>"
            "<b>2</b> – Long<br>"
            "<b>3</b> – Small<br>"
            "<b>D</b> – Discard object<br>"
            "<b>J</b> – Next object<br>"
            "<b>K</b> – Previous object<br>"
            "<b>S</b> – Toggle Auto-Spin<br>"
            "<b>L</b> – Load saved scale/rot<br><br>"
            "<b>Rotation Controls:</b><br>"
            "<b>E</b> – Rotate X axis (+90°)<br>"
            "<b>R</b> – Rotate Y axis (+90°)<br>"
            "<b>T</b> – Rotate Z axis (+90°)<br><br>"
            "Use the scale factor to uniformly resize the mesh before saving.\n"
            "The Reference Apple helps judge absolute size (Meters)."
        )
        QtWidgets.QMessageBox.information(self, "Keyboard Shortcuts", text)

    # ---------- Scale & rotation handling ---------- #
    def _on_scale_slider_changed(self, value: int):
        """Slider -> spinbox, then refresh view."""
        factor = value / 100.0
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(factor)
        self.scale_spin.blockSignals(False)
        self._refresh_view_scale()

    def _on_scale_spin_changed(self, value: float):
        """Spinbox -> slider, then refresh view."""
        slider_val = int(round(value * 100))
        slider_val = max(self.scale_slider.minimum(),
                         min(self.scale_slider.maximum(), slider_val))
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(slider_val)
        self.scale_slider.blockSignals(False)
        self._refresh_view_scale()

    def _on_change_reference(self):
        """Open file dialog to pick a new reference OBJ."""
        global REF_PATH  # <--- FIX: Must be declared before ANY usage of REF_PATH
        
        # Now it is safe to read REF_PATH to find the starting directory
        start_dir = os.path.dirname(REF_PATH) if os.path.exists(REF_PATH) else MESH_ROOT
        
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Reference Mesh", start_dir, "OBJ Files (*.obj)"
        )
        
        if path:
            REF_PATH = path
            
            # Update Viewer
            scale = self.ref_scale_spin.value()
            self.viewer.update_reference_mesh(path, scale)
            
            # Refresh View (redraw scene + update stats text)
            self._load_current()

    def _on_ref_scale_changed(self, val):
        """Re-load the reference with the new scale."""
        # We need to reload from file to avoid compounding scale errors (1.1 * 1.1 * 1.1...)
        # Assume REF_PATH is valid (or fallback to viewer logic)
        
        # Determine path (Use global REF_PATH, or fallback to Apple)
        path = REF_PATH
        # If REF_PATH was a directory (the default config), find the obj inside
        if os.path.isdir(path):
             objs = [f for f in os.listdir(path) if f.endswith('.obj')]
             if objs:
                 path = os.path.join(path, objs[0])
        
        if os.path.isfile(path):
            self.viewer.update_reference_mesh(path, val)
            self._load_current()

    def _store_state(self):
        """Store current scale & rotations for this index and as last global state."""
        if not (0 <= self.index < len(self.entries)):
            return

        state = {
            "scale": self.scale_spin.value(),
            "rot_x": self.rotation_x_degrees,
            "rot_y": self.rotation_y_degrees,
            "rot_z": self.rotation_z_degrees,
        }
        self.object_state[self.index] = state
        self.last_state = state

    def _set_rotation_status(self, highlight: bool):
        """Update the status label to reflect current rotation (and scale)."""
        color = "#fbbc05" if highlight else "#9aa0a6"
        self.status.setStyleSheet(f"color:{color};")
        self.status.setText(
            f"Rotation: X={self.rotation_x_degrees:.0f}°, "
            f"Y={self.rotation_y_degrees:.0f}°, "
            f"Z={self.rotation_z_degrees:.0f}° | "
            f"scale={self.scale_spin.value():.2f}"
        )

    def _refresh_view_scale(self):
        """Update the viewer to reflect the current scale and rotation."""
        if not (0 <= self.index < len(self.entries)):
            return
        mesh = self._load_mesh(self.index)
        if mesh is None:
            return
        scale = self.scale_spin.value()

        entry = self.entries[self.index]

        # Pointcloud
        pc = None
        if entry.npy_path is not None and os.path.isfile(entry.npy_path):
            try:
                raw = np.load(entry.npy_path)
                if raw.ndim == 2 and raw.shape[1] == 3:
                    pc = self._transform_points_for_view(raw, entry)
            except Exception:
                pc = None

        # geometry-derived COM (what you already had)
        mesh_final = mesh.copy()
        mesh_final.apply_scale(float(scale))

        # Apply the same rotations as MeshViewer.show_pair(): X -> Y -> Z using verts @ R.T
        def _apply_rot(msh, deg, axis):
            if deg % 360 == 0:
                return
            a = np.deg2rad(deg)
            if axis == "x":
                R = np.array([[1,0,0],
                            [0,np.cos(a),-np.sin(a)],
                            [0,np.sin(a), np.cos(a)]], dtype=np.float64)
            elif axis == "y":
                R = np.array([[ np.cos(a),0,np.sin(a)],
                            [0,1,0],
                            [-np.sin(a),0,np.cos(a)]], dtype=np.float64)
            else:  # "z"
                R = np.array([[np.cos(a),-np.sin(a),0],
                            [np.sin(a), np.cos(a),0],
                            [0,0,1]], dtype=np.float64)
            msh.vertices = msh.vertices @ R.T

        _apply_rot(mesh_final, self.rotation_x_degrees, "x")
        _apply_rot(mesh_final, self.rotation_y_degrees, "y")
        _apply_rot(mesh_final, self.rotation_z_degrees, "z")

        geom_com = self._infer_com_z_midpoint(mesh_final).astype(np.float32)

        # URDF COM
        urdf_com = None
        raw_urdf_com = self._get_urdf_com(entry)   # parses <inertial><origin xyz=...>
        if raw_urdf_com is not None:
            urdf_com = self._transform_urdf_com_for_view(raw_urdf_com, entry)

        # Update viewer
        self.viewer.show_pair(
            mesh,
            scale,
            self.rotation_z_degrees,
            self.rotation_x_degrees,
            self.rotation_y_degrees,
            reset_camera=False,  # or whatever you use
            obj_pointcloud=pc,
            obj_com_geom=geom_com,
            obj_com_urdf=urdf_com,
            show_pointcloud=self.chk_show_pc.isChecked(),
            show_com=self.chk_show_com.isChecked(),
        )

        # Update geometry stats live
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_scale(scale)
        ex_obj, vol_obj, max_obj = self._compute_stats(mesh_scaled)
        if self.viewer.ref_mesh is not None:
            ex_ref, vol_ref, max_ref = self._compute_stats(self.viewer.ref_mesh)
        else:
            ex_ref = np.array([0.0, 0.0, 0.0])
            vol_ref = float("nan")
            max_ref = float("nan")
        self._update_geom_label(ex_obj, vol_obj, max_obj,
                                ex_ref, vol_ref, max_ref)

        # Persist state
        self._store_state()

    def _rotate_z(self):
        """Rotate current object by +90° around Z (both view and saved mesh)."""
        if not (0 <= self.index < len(self.entries)):
            return
        self.rotation_z_degrees = (self.rotation_z_degrees + 90.0) % 360.0
        self._refresh_view_scale()
        self._set_rotation_status(highlight=True)

    def _rotate_x(self):
        """Rotate current object by +90° around X (both view and saved mesh)."""
        if not (0 <= self.index < len(self.entries)):
            return
        self.rotation_x_degrees = (self.rotation_x_degrees + 90.0) % 360.0
        self._refresh_view_scale()
        self._set_rotation_status(highlight=True)

    def _rotate_y(self):
        """Rotate current object by +90° around Y (both view and saved mesh)."""
        if not (0 <= self.index < len(self.entries)):
            return
        self.rotation_y_degrees = (self.rotation_y_degrees + 90.0) % 360.0
        self._refresh_view_scale()
        self._set_rotation_status(highlight=True)

    # ---------- Helpers ---------- #
    def _load_mesh(self, idx: int) -> Optional[trimesh.Trimesh]:
        if idx in self.mesh_cache:
            return self.mesh_cache[idx]

        entry = self.entries[idx]
        try:
            # 1. Load the Raw OBJ
            mesh = trimesh.load(entry.obj_path, force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = mesh.dump().sum()

            # 2. [NEW] Apply the Initial URDF Scale immediately
            # This ensures the Viewer and Stats show the "Real" simulation size
            if entry.urdf_path:
                initial_scale = self._get_urdf_initial_scale(entry.urdf_path)
                
                # Check if it's essentially 1.0 (to save compute)
                if not np.allclose(initial_scale, 1.0):
                    # Create a transform matrix for non-uniform scaling
                    S = np.eye(4)
                    S[0,0] = initial_scale[0]
                    S[1,1] = initial_scale[1]
                    S[2,2] = initial_scale[2]
                    mesh.apply_transform(S)
                    # print(f"Applied URDF initial scale {initial_scale} to {entry.base_name}")

            self.mesh_cache[idx] = mesh
            return mesh
        except Exception as e:
            self.status.setText(f"Failed to load OBJ: {e}")
            return None
        
    def _get_transform_matrix(self) -> np.ndarray:
        """
        Constructs the 4x4 transformation matrix based on 
        current GUI scale and rotation settings.
        """
        scale = self.scale_spin.value()
        
        # 1. Scale Matrix
        S = np.eye(4)
        S[:3, :3] *= scale

        # 2. Rotation Matrices
        # Order matters! We apply X, then Y, then Z (matches visualizer logic)
        R_total = np.eye(3)

        if self.rotation_x_degrees % 360 != 0:
            ax = np.deg2rad(self.rotation_x_degrees)
            cx, sx = np.cos(ax), np.sin(ax)
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            R_total = Rx @ R_total

        if self.rotation_y_degrees % 360 != 0:
            ay = np.deg2rad(self.rotation_y_degrees)
            cy, sy = np.cos(ay), np.sin(ay)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            R_total = Ry @ R_total

        if self.rotation_z_degrees % 360 != 0:
            az = np.deg2rad(self.rotation_z_degrees)
            cz, sz = np.cos(az), np.sin(az)
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R_total = Rz @ R_total

        # Combine: T = R * S
        # Note: In standard linear algebra T = R @ S. 
        # But since we often operate on row vectors (v @ M.T), we just build the 4x4 here.
        T = np.eye(4)
        T[:3, :3] = R_total @ S[:3, :3]
        
        return T, scale, R_total

    def _compute_stats(self, mesh: trimesh.Trimesh):
        ex = mesh.bounding_box.extents
        if ex is None or ex.size == 0:
            ex = np.array([0.0, 0.0, 0.0])
        try:
            if mesh.is_volume:
                vol = float(mesh.volume)
            else:
                vol = float(mesh.convex_hull.volume)
        except Exception:
            vol = float("nan")
        max_dim = float(np.max(ex)) if ex.size > 0 else float("nan")
        return ex, vol, max_dim
    
    def _update_geom_label(self, ex_obj, vol_obj, max_obj,
                           ex_ref, vol_ref, max_ref):
        geom_html = f"""
<b>Object</b>
<div style="margin-left:18px;">
<table cellspacing="0" cellpadding="0">
  <tr>
    <td style="padding-left:10px;color:#8ab4f8;">BBox (x, y, z):</td>
    <td style="padding-left:24px;color:#e8eaed;">
      {ex_obj[0]:.5f}, {ex_obj[1]:.5f}, {ex_obj[2]:.5f}
    </td>
  </tr>
  <tr>
    <td style="padding-left:10px;color:#34a853;">Volume:</td>
    <td style="padding-left:24px;color:#e8eaed;">{vol_obj:.10f}</td>
  </tr>
  <tr>
    <td style="padding-left:10px;color:#fbbc05;">Max dim:</td>
    <td style="padding-left:24px;color:#e8eaed;">{max_obj:.5f}</td>
  </tr>
</table>
</div>
<br>
<b>Reference (apple)</b>
<div style="margin-left:18px;">
<table cellspacing="0" cellpadding="0">
  <tr>
    <td style="padding-left:10px;color:#8ab4f8;">BBox (x, y, z):</td>
    <td style="padding-left:24px;color:#e8eaed;">
      {ex_ref[0]:.5f}, {ex_ref[1]:.5f}, {ex_ref[2]:.5f}
    </td>
  </tr>
  <tr>
    <td style="padding-left:10px;color:#34a853;">Volume:</td>
    <td style="padding-left:24px;color:#e8eaed;">{vol_ref:.10f}</td>
  </tr>
  <tr>
    <td style="padding-left:10px;color:#fbbc05;">Max dim:</td>
    <td style="padding-left:24px;color:#e8eaed;">{max_ref:.5f}</td>
  </tr>
</table>
</div>
"""
        self.lbl_geom.setText(geom_html)

    def _apply_saved_transform(self):
        """
        If this object has been saved before (in any category),
        load its scale/rotation from the JSON on disk and apply it
        to the viewer + UI controls.
        """
        if not (0 <= self.index < len(self.entries)):
            return

        entry = self.entries[self.index]
        info = self._get_fs_status(entry)

        if info["state"] != "saved" or info["scale"] is None:
            self.status.setStyleSheet("color:#ea4335;")
            self.status.setText("No saved transform found for this object.")
            return

        scale = float(info["scale"])
        self.rotation_x_degrees = float(info["rot_x"] or 0.0)
        self.rotation_y_degrees = float(info["rot_y"] or 0.0)
        self.rotation_z_degrees = float(info["rot_z"] or 0.0)

        # Update UI controls
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(scale)
        self.scale_spin.blockSignals(False)

        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(int(round(scale * 100)))
        self.scale_slider.blockSignals(False)

        # Refresh view + stats and persist state
        self._refresh_view_scale()
        self._set_rotation_status(highlight=True)
        self._update_save_state_label()

        self.status.setStyleSheet("color:#34a853;")
        self.status.setText(
            f"Applied saved transform: "
            f"scale={scale:.3f}, "
            f"X={self.rotation_x_degrees:.0f}°, "
            f"Y={self.rotation_y_degrees:.0f}°, "
            f"Z={self.rotation_z_degrees:.0f}°"
        )

    def _get_fs_status(self, entry: MeshEntry) -> Dict:
        """
        Look on disk to see if this object is:
          - not processed
          - saved to a category
          - discarded

        Returns a dict:
          {
            "state": "none" | "saved" | "discarded",
            "category": str or None,
            "scale": float or None,
            "rot_x": float or None,
            "rot_y": float or None,
            "rot_z": float or None,
        }
        """
        full_name = f"{entry.label}_{entry.base_name}"

        # 1) check categories
        for cat_key, subdir in CATEGORY_DIRS.items():
            cat_dir = os.path.join(OUTPUT_ROOT, subdir, full_name)
            if os.path.isdir(cat_dir):
                json_path = os.path.join(cat_dir, f"{full_name}.json")
                scale = rot_x = rot_y = rot_z = None
                if os.path.isfile(json_path):
                    try:
                        with open(json_path, "r") as f:
                            data = json.load(f)
                        scale = float(data.get("scale", 1.0))
                        r = data.get("rotation_degrees", {})
                        rot_x = float(r.get("x", 0.0))
                        rot_y = float(r.get("y", 0.0))
                        rot_z = float(r.get("z", 0.0))
                    except Exception:
                        pass
                return {
                    "state": "saved",
                    "category": cat_key,
                    "scale": scale,
                    "rot_x": rot_x,
                    "rot_y": rot_y,
                    "rot_z": rot_z,
                }

        # 2) check discarded
        disc_dir = os.path.join(OUTPUT_ROOT, DISCARD_DIR, full_name)
        if os.path.isdir(disc_dir):
            json_path = os.path.join(disc_dir, f"{full_name}.json")
            scale = rot_x = rot_y = rot_z = None
            if os.path.isfile(json_path):
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    scale = float(data.get("scale", 1.0))
                    r = data.get("rotation_degrees", {})
                    rot_x = float(r.get("x", 0.0))
                    rot_y = float(r.get("y", 0.0))
                    rot_z = float(r.get("z", 0.0))
                except Exception:
                    pass
            return {
                "state": "discarded",
                "category": None,
                "scale": scale,
                "rot_x": rot_x,
                "rot_y": rot_y,
                "rot_z": rot_z,
            }

        # 3) no dir at all
        return {
            "state": "none",
            "category": None,
            "scale": None,
            "rot_x": None,
            "rot_y": None,
            "rot_z": None,
        }

    def _update_save_state_label(self):
        """Show whether current object is saved / discarded / untouched."""
        if not (0 <= self.index < len(self.entries)):
            self.lbl_state.setText("Status: n/a")
            self.lbl_state.setStyleSheet("color:#9aa0a6;")
            return

        entry = self.entries[self.index]
        info = self._get_fs_status(entry)
        state = info["state"]

        if state == "none":
            self.lbl_state.setText("Status: not saved / discarded")
            self.lbl_state.setStyleSheet("color:#9aa0a6;")
        elif state == "discarded":
            self.lbl_state.setText("Status: discarded")
            self.lbl_state.setStyleSheet("color:#ea4335;")  # red-ish
        elif state == "saved":
            cat = info["category"] or "?"
            sc = info["scale"]
            rx = info["rot_x"]
            ry = info["rot_y"]
            rz = info["rot_z"]
            # some JSONs might be missing rotation, so guard with defaults
            if sc is None:
                sc = 1.0
            if rx is None:
                rx = 0.0
            if ry is None:
                ry = 0.0
            if rz is None:
                rz = 0.0
            self.lbl_state.setText(
                f"Status: saved to '{cat}' "
                f"(scale={sc:.3f}, X={rx:.0f}°, Y={ry:.0f}°, Z={rz:.0f}°)"
            )
            self.lbl_state.setStyleSheet("color:#34a853;")  # green
        else:
            self.lbl_state.setText("Status: not saved / discarded")
            self.lbl_state.setStyleSheet("color:#9aa0a6;")

    def _get_urdf_initial_scale(self, urdf_path: Optional[str]) -> np.ndarray:
        """
        Reads URDF visual mesh scale. If no URDF or no scale tag, returns [1,1,1].
        """
        if not urdf_path or not os.path.isfile(urdf_path):
            return np.array([0.1, 0.1, 0.1], dtype=np.float64)

        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for mesh_tag in root.findall(".//visual/geometry/mesh"):
                scale_str = mesh_tag.get("scale")
                if scale_str:
                    vec = np.fromstring(scale_str, sep=" ")
                    if vec.shape == (3,):
                        return vec.astype(np.float64)
            return np.array([0.1, 0.1, 0.1], dtype=np.float64)
        except Exception as e:
            print(f"Error reading URDF scale: {e}")
            return np.array([0.1, 0.1, 0.1], dtype=np.float64)
    
    def _load_current(self):
        if not (0 <= self.index < len(self.entries)):
            self.status.setText("No meshes or index out of range.")
            return

        entry = self.entries[self.index]
        mesh = self._load_mesh(self.index)
        if mesh is None:
            return
        
        # --- Handle Reference Scaling Logic ---
        ref_scale = self.ref_scale_spin.value()
        
        if self.chk_sync_ref.isChecked():
            urdf_scale_vec = self._get_urdf_initial_scale(entry.urdf_path)
            # choose a scalar; most robust is geometric mean or max
            urdf_scale_factor = float(np.mean(urdf_scale_vec))
            ref_scale *= urdf_scale_factor


        ref_path = REF_PATH
        if os.path.isdir(ref_path):
             objs = [f for f in os.listdir(ref_path) if f.endswith('.obj')]
             if objs: ref_path = os.path.join(ref_path, objs[0])
        
        if os.path.isfile(ref_path):
            self.viewer.update_reference_mesh(ref_path, ref_scale)
        # -------------------------------------

        fs_info = self._get_fs_status(entry)
        is_saved = fs_info["state"] in ("saved", "discarded")

        mode = self.load_mode_combo.currentIndex()
        
        target_scale = 1.0
        target_rx = 0.0
        target_ry = 0.0
        target_rz = 0.0

        if mode == 0: 
            pass 
        elif mode == 1:
            target_scale = self.scale_spin.value()
            target_rx = self.rotation_x_degrees
            target_ry = self.rotation_y_degrees
            target_rz = self.rotation_z_degrees
        elif mode == 2:
            if is_saved and fs_info["scale"] is not None:
                target_scale = float(fs_info["scale"])
                target_rx = float(fs_info["rot_x"] or 0.0)
                target_ry = float(fs_info["rot_y"] or 0.0)
                target_rz = float(fs_info["rot_z"] or 0.0)

        self.rotation_x_degrees = target_rx
        self.rotation_y_degrees = target_ry
        self.rotation_z_degrees = target_rz
        
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(target_scale)
        self.scale_spin.blockSignals(False)

        self.scale_slider.blockSignals(True)
        slider_val = int(round(target_scale * 100))
        self.scale_slider.setValue(slider_val)
        self.scale_slider.blockSignals(False)

        # Update Viewer [reset_camera=True for new object]
        self._refresh_view_scale()

        # Geometry Stats & Label Updates
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_scale(target_scale)
        ex_obj, vol_obj, max_obj = self._compute_stats(mesh_scaled)

        if self.viewer.ref_mesh is not None:
            ex_ref, vol_ref, max_ref = self._compute_stats(self.viewer.ref_mesh)
        else:
            ex_ref = np.array([0.0, 0.0, 0.0])
            vol_ref = float("nan")
            max_ref = float("nan")

        obj_dir = os.path.dirname(entry.obj_path)
        full_name = f"{entry.label}_{entry.base_name}"

        self.lbl_index.setText(f"Object {self.index + 1} / {len(self.entries)}")
        self.lbl_name.setText(f"Name: {full_name}")
        self.lbl_paths.setText(f"Object dir: {obj_dir}")

        self._update_geom_label(ex_obj, vol_obj, max_obj,
                                ex_ref, vol_ref, max_ref)

        self._update_save_state_label()
        self._set_rotation_status(highlight=(target_rx!=0 or target_ry!=0 or target_rz!=0))
        self._store_state()

    def _discard_current(self):
        """Mark current object as discarded and optionally copy it to a discarded folder."""
        if not (0 <= self.index < len(self.entries)):
            return

        entry = self.entries[self.index]
        mesh = self._load_mesh(self.index)
        if mesh is None:
            return

        scale = self.scale_spin.value()

        full_name = f"{entry.label}_{entry.base_name}"

        # Remove any category versions (if previously saved)
        for cat_key2, subdir in CATEGORY_DIRS.items():
            existing_dir = os.path.join(OUTPUT_ROOT, subdir, full_name)
            if os.path.isdir(existing_dir):
                shutil.rmtree(existing_dir, ignore_errors=True)

        # Prepare discarded dir (fresh)
        out_dir = os.path.join(OUTPUT_ROOT, DISCARD_DIR, full_name)
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)
        ensure_dir(out_dir)


        # Apply same transform pipeline as for saving (so discarded geometry is consistent)
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_scale(scale)

        if self.rotation_x_degrees % 360 != 0:
            ax = np.deg2rad(self.rotation_x_degrees)
            cx, sx = np.cos(ax), np.sin(ax)
            Rx = np.array([
                [1.0, 0.0,  0.0],
                [0.0, cx,  -sx],
                [0.0, sx,   cx],
            ])
            mesh_scaled.vertices = mesh_scaled.vertices @ Rx.T

        if self.rotation_y_degrees % 360 != 0:
            ay = np.deg2rad(self.rotation_y_degrees)
            cy, sy = np.cos(ay), np.sin(ay)
            Ry = np.array([
                [ cy, 0.0, sy],
                [0.0, 1.0, 0.0],
                [-sy, 0.0, cy],
            ])
            mesh_scaled.vertices = mesh_scaled.vertices @ Ry.T

        if self.rotation_z_degrees % 360 != 0:
            az = np.deg2rad(self.rotation_z_degrees)
            cz, sz = np.cos(az), np.sin(az)
            Rz = np.array([
                [cz, -sz, 0.0],
                [sz,  cz, 0.0],
                [0.0, 0.0, 1.0],
            ])
            mesh_scaled.vertices = mesh_scaled.vertices @ Rz.T

        ex_orig, vol_orig, max_orig = self._compute_stats(mesh)
        ex_scaled, vol_scaled, max_scaled = self._compute_stats(mesh_scaled)

        full_name = f"{entry.label}_{entry.base_name}"

        # Discard folder
        out_dir = os.path.join(OUTPUT_ROOT, DISCARD_DIR, full_name)
        ensure_dir(out_dir)

        # Export a scaled OBJ for reference (even though it's "discarded")
        out_obj = os.path.join(out_dir, f"{full_name}.obj")
        try:
            mesh_scaled.export(out_obj)
        except Exception as e:
            self.status.setText(f"Failed to export discarded OBJ: {e}")
            return

        # Copy auxiliary files unchanged
        out_npy = None
        out_urdf = None
        if entry.npy_path is not None:
            out_npy = os.path.join(out_dir, f"{full_name}.npy")
            try:
                shutil.copy2(entry.npy_path, out_npy)
            except Exception as e:
                self.status.setText(f"Failed to copy discarded npy: {e}")
        if entry.urdf_path is not None:
            out_urdf = os.path.join(out_dir, f"{full_name}.urdf")
            try:
                shutil.copy2(entry.urdf_path, out_urdf)
            except Exception as e:
                self.status.setText(f"Failed to copy discarded urdf: {e}")

        # JSON log for discarded object
        log_data = {
            "label": entry.label,
            "base_name": entry.base_name,
            "full_name": full_name,
            "state": "discarded",
            "scale": scale,
            "rotation_degrees": {
                "x": self.rotation_x_degrees,
                "y": self.rotation_y_degrees,
                "z": self.rotation_z_degrees,
            },
            "original_stats": {
                "bbox_extents": ex_orig.tolist(),
                "volume": vol_orig,
                "max_dim": max_orig,
            },
            "scaled_stats": {
                "bbox_extents": ex_scaled.tolist(),
                "volume": vol_scaled,
                "max_dim": max_scaled,
            },
            "paths": {
                "original": {
                    "obj": entry.obj_path,
                    "glb": None,
                    "npy": entry.npy_path,
                    "urdf": entry.urdf_path,
                },
                "discarded_output": {
                    "obj": out_obj,
                    "npy": out_npy,
                    "urdf": out_urdf,
                },
            },
        }
        json_path = os.path.join(out_dir, f"{full_name}.json")
        try:
            with open(json_path, "w") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            self.status.setText(f"Failed to write discarded JSON log: {e}")

        # Update status map
        self._update_save_state_label()

        # Status text at bottom
        self.status.setStyleSheet("color:#ea4335;")
        self.status.setText(
            f"🚫 Discarded | scale={scale:.3f}, "
            f"X={self.rotation_x_degrees:.0f}°, "
            f"Y={self.rotation_y_degrees:.0f}°, "
            f"Z={self.rotation_z_degrees:.0f}° | {out_dir}"
        )

        # Move to next object
        self._store_state()
        self.index += 1
        if self.index >= len(self.entries):
            self.index = len(self.entries) - 1
        self._load_current()

    # ---------- Category / saving ---------- #
    def _assign_category(self, category_key: str):
        if category_key not in CATEGORY_DIRS:
            return
        if not (0 <= self.index < len(self.entries)):
            return

        entry = self.entries[self.index]
        mesh = self._load_mesh(self.index)
        if mesh is None:
            return

        # 1. Get Transformation Data
        # T_matrix includes User Scale + User Rotation
        T_matrix, user_scale, R_matrix = self._get_transform_matrix()
        
        # 2. Get Initial URDF Scale (The "0.1" you mentioned)
        # We need this to normalize raw files (GLB/NPY) to match the OBJ
        initial_scale_vec = self._get_urdf_initial_scale(entry.urdf_path)

        # --- 3. PREPARE OBJ ---
        # Note: 'mesh' is already scaled by initial_scale_vec (done in _load_mesh)
        # So we only apply the User Transform here.
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_transform(T_matrix)

        # Compute stats for logging
        ex_orig, vol_orig, max_orig = self._compute_stats(mesh)
        ex_scaled, vol_scaled, max_scaled = self._compute_stats(mesh_scaled)

        # --- 4. PREPARE DIRECTORIES ---
        full_name = f"{entry.label}_{entry.base_name}"
        
        disc_dir = os.path.join(OUTPUT_ROOT, DISCARD_DIR, full_name)
        if os.path.isdir(disc_dir):
            shutil.rmtree(disc_dir, ignore_errors=True)

        for cat_key2, subdir in CATEGORY_DIRS.items():
            existing_dir = os.path.join(OUTPUT_ROOT, subdir, full_name)
            if os.path.isdir(existing_dir):
                shutil.rmtree(existing_dir, ignore_errors=True)

        cat_dir = CATEGORY_DIRS[category_key]
        out_dir = os.path.join(OUTPUT_ROOT, cat_dir, full_name)
        ensure_dir(out_dir)

        # --- 5. EXPORT OBJ ---
        out_obj = os.path.join(out_dir, f"{full_name}.obj")
        try:
            mesh_scaled.export(out_obj)
        except Exception as e:
            self.status.setText(f"Failed to export OBJ: {e}")
            return

        # --- 6. EXPORT GLB (Fixed) ---
        out_glb = None
        if entry.glb_path is not None:
            try:
                glb_mesh = trimesh.load(entry.glb_path, force='mesh')
                if not isinstance(glb_mesh, trimesh.Trimesh):
                    glb_mesh = glb_mesh.dump().sum()
                
                # A. Apply Initial URDF Scale (Fix the "Applied Twice" mismatch)
                S_init = np.eye(4)
                S_init[0,0] = initial_scale_vec[0]
                S_init[1,1] = initial_scale_vec[1]
                S_init[2,2] = initial_scale_vec[2]
                glb_mesh.apply_transform(S_init)

                # B. Apply User Transform
                glb_mesh.apply_transform(T_matrix)
                
                out_glb = os.path.join(out_dir, f"{full_name}.glb")
                glb_mesh.export(out_glb)
            except Exception as e:
                print(f"GLB Processing Error: {e}")
                out_glb = None

        # --- 7. EXPORT NPY (Robust: avoid double-applying 0.1) ---
        out_npy = None
        if entry.npy_path is not None:
            try:
                arr = np.load(entry.npy_path)
                if arr.ndim == 2 and arr.shape[1] == 3:
                    arr = arr.astype(np.float64)

                    # mesh is already initial-scale adjusted by _load_mesh()
                    mesh_init = mesh  # (URDF-scaled mesh in meters)

                    # Compare extents: decide whether NPY needs the URDF initial scale
                    mesh_ext = np.asarray(mesh_init.bounding_box.extents, dtype=np.float64)
                    pc_ext = np.asarray(arr.max(axis=0) - arr.min(axis=0), dtype=np.float64)

                    mesh_max = float(np.max(mesh_ext))
                    pc_max = float(np.max(pc_ext))
                    apply_init = True
                    if mesh_max > 1e-9 and pc_max > 1e-9:
                        ratio = mesh_max / pc_max
                        # If pc already matches mesh units, don't apply init scale again
                        if np.isclose(ratio, 1.0, rtol=0.25):
                            apply_init = False
                        else:
                            # If pc is ~10x larger than mesh (ratio ~0.1), apply 0.1
                            s = float(np.mean(initial_scale_vec))
                            if np.isclose(ratio, s, rtol=0.25):
                                apply_init = True
                            # If pc is ~10x smaller than mesh (ratio ~10), do NOT apply again
                            elif np.isclose(ratio, 1.0 / max(s, 1e-9), rtol=0.25):
                                apply_init = False
                            else:
                                # fallback heuristic: if pc seems bigger, apply init
                                apply_init = pc_max > mesh_max

                    if apply_init:
                        arr = arr * initial_scale_vec

                    # Then apply user transform (same as mesh_scaled)
                    arr = arr * user_scale
                    arr = arr @ R_matrix.T

                    out_npy = os.path.join(out_dir, f"{full_name}.npy")
                    np.save(out_npy, arr.astype(np.float32))
                else:
                    out_npy = os.path.join(out_dir, f"{full_name}.npy")
                    shutil.copy2(entry.npy_path, out_npy)
            except Exception as e:
                print(f"NPY Processing Error: {e}")
                out_npy = None

        # --- 8. EXPORT URDF ---
        out_urdf = os.path.join(out_dir, f"{full_name}.urdf")
        
        if entry.urdf_path is not None and os.path.isfile(entry.urdf_path):
            # === CASE A: URDF Exists (Update it) ===
            try:
                tree = ET.parse(entry.urdf_path)
                root = tree.getroot()
                
                def update_origin_tag(tag):
                    if tag is None: return
                    xyz_str = tag.get('xyz')
                    if xyz_str:
                        try:
                            vec = np.fromstring(xyz_str, sep=' ')
                            if vec.shape == (3,):
                                vec = vec * initial_scale_vec # Init scale
                                vec *= user_scale             # User scale
                                vec = R_matrix @ vec          # Rotation
                                tag.set('xyz', f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}")
                        except ValueError:
                            pass

                def update_mesh_tag(geometry_tag):
                    if geometry_tag is None: return
                    mesh_tag = geometry_tag.find('mesh')
                    if mesh_tag is not None:
                        mesh_tag.set('filename', f"{full_name}.obj")
                        mesh_tag.set('scale', "1 1 1") # Reset scale

                for link in root.findall('link'):
                    for visual in link.findall('visual'):
                        update_origin_tag(visual.find('origin'))
                        update_mesh_tag(visual.find('geometry'))
                    for collision in link.findall('collision'):
                        update_origin_tag(collision.find('origin'))
                        update_mesh_tag(collision.find('geometry'))
                    for inertial in link.findall('inertial'):
                        org = inertial.find('origin')
                        if org is None:
                            continue
                        xyz_str = org.get('xyz')
                        if not xyz_str:
                            continue
                        try:
                            vec = np.fromstring(xyz_str, sep=' ')
                            if vec.shape == (3,):
                                # Only scale the original COM by user scale (NO rotation, NO initial URDF mesh scale)
                                vec = vec * user_scale
                                org.set('xyz', f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}")
                        except ValueError:
                            pass
                                    
                tree.write(out_urdf)
            except Exception as e:
                print(f"URDF Update Error: {e}")
                shutil.copy2(entry.urdf_path, out_urdf)
        
        else:
            # === CASE B: No URDF (Generate New) ===
            try:
                # Calculate approximate Center of Mass (Geometric Center)
                # mesh_scaled is the final transformed mesh
                com = self._infer_com_z_midpoint(mesh_scaled)
                cx, cy, cz = float(com[0]), float(com[1]), float(com[2])

                urdf_content = f"""<?xml version="1.0" ?>
<robot name="mesh_object">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="{full_name}.obj" scale="1 1 1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{full_name}.obj" scale="1 1 1"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
                with open(out_urdf, "w") as f:
                    f.write(urdf_content)
                print(f"Generated new URDF: {out_urdf}")
                
            except Exception as e:
                self.status.setText(f"Failed to generate URDF: {e}")

        # --- 9. LOGGING JSON ---
        log_data = {
            "label": entry.label,
            "base_name": entry.base_name,
            "full_name": full_name,
            "category": category_key,
            "scale": user_scale,
            "initial_urdf_scale": initial_scale_vec.tolist(),
            "rotation_degrees": {
                "x": self.rotation_x_degrees,
                "y": self.rotation_y_degrees,
                "z": self.rotation_z_degrees,
            },
            "original_stats": {
                "bbox_extents": ex_orig.tolist(),
                "volume": vol_orig,
                "max_dim": max_orig,
            },
            "scaled_stats": {
                "bbox_extents": ex_scaled.tolist(),
                "volume": vol_scaled,
                "max_dim": max_scaled,
            },
            "paths": {
                "original": {
                    "obj": entry.obj_path,
                    "glb": entry.glb_path,
                    "npy": entry.npy_path,
                    "urdf": entry.urdf_path,
                },
                "output": {
                    "obj": out_obj,
                    "glb": out_glb,
                    "npy": out_npy,
                    "urdf": out_urdf,
                },
            },
        }

        json_path = os.path.join(out_dir, f"{full_name}.json")
        try:
            with open(json_path, "w") as f:
                json.dump(log_data, f, indent=2)
        except Exception as e:
            self.status.setText(f"Failed to write JSON log: {e}")

        # Update UI & State
        self._update_save_state_label()
        msg = (
            f"✅ Saved to '{category_key}' | "
            f"scale={user_scale:.3f}, Files Synced | {out_dir}"
        )
        self.status.setStyleSheet("color:#34a853;")
        self.status.setText(msg)

        self._store_state()
        self.undo_stack.append((self.index, category_key))

        self.index += 1
        if self.index >= len(self.entries):
            self.index = len(self.entries) - 1
        self._load_current()

    # ---------- Navigation ---------- #
    def _next(self):
        if self.index < len(self.entries) - 1:
            self.index += 1
            self._load_current()

    def _prev(self):
        if self.index > 0:
            self.index -= 1
            self._load_current()

def main():
    entries = scan_meshes(MESH_ROOT)
    if not entries:
        print(f"No meshes found in {MESH_ROOT}")
        return

    app = QtWidgets.QApplication(sys.argv)
    w = MeshToolApp(entries)
    w.resize(2100, 1200)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()