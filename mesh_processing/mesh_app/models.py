from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import trimesh


@dataclass
class MeshEntry:
    label: str
    base_name: str
    obj_path: str
    npy_path: Optional[str]
    glb_path: Optional[str]
    urdf_path: Optional[str]

    @property
    def full_name(self) -> str:
        if self.label == self.base_name:
            return self.label
        return f"{self.label}_{self.base_name}"


@dataclass
class TransformState:
    scale: float = 1.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    dilate_x: float = 1.0
    dilate_y: float = 1.0
    dilate_z: float = 1.0
    dilate_x_min: float = 1.0
    dilate_x_max: float = 1.0
    dilate_y_min: float = 1.0
    dilate_y_max: float = 1.0
    dilate_z_min: float = 1.0
    dilate_z_max: float = 1.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0

    def normalized(self) -> "TransformState":
        x_low = float(min(self.dilate_x_min, self.dilate_x_max))
        x_high = float(max(self.dilate_x_min, self.dilate_x_max))
        y_low = float(min(self.dilate_y_min, self.dilate_y_max))
        y_high = float(max(self.dilate_y_min, self.dilate_y_max))
        z_low = float(min(self.dilate_z_min, self.dilate_z_max))
        z_high = float(max(self.dilate_z_min, self.dilate_z_max))
        self.dilate_x_min = x_low
        self.dilate_x_max = x_high
        self.dilate_x = 0.5 * (x_low + x_high)
        self.dilate_y_min = y_low
        self.dilate_y_max = y_high
        self.dilate_y = 0.5 * (y_low + y_high)
        self.dilate_z_min = z_low
        self.dilate_z_max = z_high
        self.dilate_z = 0.5 * (z_low + z_high)
        self.scale_min = float(np.mean([x_low, y_low, z_low]))
        self.scale_max = float(np.mean([x_high, y_high, z_high]))
        self.scale = float(np.mean([self.dilate_x, self.dilate_y, self.dilate_z]))
        return self

    def dilation_vector(self) -> np.ndarray:
        self.normalized()
        return np.array([self.dilate_x, self.dilate_y, self.dilate_z], dtype=np.float64)

    def endpoint_dilation_vector(self, which: str) -> np.ndarray:
        self.normalized()
        if which == "min":
            return np.array(
                [self.dilate_x_min, self.dilate_y_min, self.dilate_z_min],
                dtype=np.float64,
            )
        if which == "max":
            return np.array(
                [self.dilate_x_max, self.dilate_y_max, self.dilate_z_max],
                dtype=np.float64,
            )
        raise ValueError(f"Unknown endpoint: {which}")

    def endpoint_scale(self, which: str) -> float:
        self.normalized()
        if which == "min":
            return float(self.scale_min)
        if which == "max":
            return float(self.scale_max)
        raise ValueError(f"Unknown endpoint: {which}")

    def endpoint_transform_matrix(self, which: str) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix() @ np.diag(self.endpoint_dilation_vector(which))
        return T

    def endpoint_dilation_vector_raw(self, which: str) -> np.ndarray:
        if which == "min":
            return np.array(
                [self.dilate_x_min, self.dilate_y_min, self.dilate_z_min],
                dtype=np.float64,
            )
        if which == "max":
            return np.array(
                [self.dilate_x_max, self.dilate_y_max, self.dilate_z_max],
                dtype=np.float64,
            )
        raise ValueError(f"Unknown endpoint: {which}")

    def endpoint_scale_raw(self, which: str) -> float:
        if which == "min":
            return float(self.scale_min)
        if which == "max":
            return float(self.scale_max)
        raise ValueError(f"Unknown endpoint: {which}")

    def endpoint_transform_matrix_raw(self, which: str) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix() @ np.diag(self.endpoint_dilation_vector_raw(which))
        return T

    def apply_endpoint_to_mesh(self, mesh: trimesh.Trimesh, which: str) -> trimesh.Trimesh:
        m = mesh.copy()
        m.apply_transform(self.endpoint_transform_matrix(which))
        return m

    def apply_endpoint_to_mesh_raw(self, mesh: trimesh.Trimesh, which: str) -> trimesh.Trimesh:
        m = mesh.copy()
        m.apply_transform(self.endpoint_transform_matrix_raw(which))
        return m

    def rotation_matrix(self) -> np.ndarray:
        rx = np.deg2rad(self.rot_x)
        ry = np.deg2rad(self.rot_y)
        rz = np.deg2rad(self.rot_z)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ Ry @ Rx

    def transform_matrix(self) -> np.ndarray:
        self.normalized()
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix() @ np.diag(self.dilation_vector())
        return T

    def apply_to_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        self.normalized()
        m = mesh.copy()
        m.apply_transform(self.transform_matrix())
        return m

    def apply_to_points(self, points: np.ndarray) -> np.ndarray:
        self.normalized()
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return pts
        pts = pts * self.dilation_vector()
        pts = pts @ self.rotation_matrix().T
        return pts.astype(np.float32)


@dataclass
class ShellTransform:
    dilate_x: float = 1.0
    dilate_y: float = 1.0
    dilate_z: float = 1.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0

    def rotation_matrix(self) -> np.ndarray:
        rx = np.deg2rad(self.rot_x)
        ry = np.deg2rad(self.rot_y)
        rz = np.deg2rad(self.rot_z)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ Ry @ Rx

    def transform_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation_matrix() @ np.diag(
            np.array([self.dilate_x, self.dilate_y, self.dilate_z], dtype=np.float64)
        )
        return T

    def apply_to_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        m = mesh.copy()
        m.apply_transform(self.transform_matrix())
        return m


def combine_shell_transforms(min_shell: ShellTransform, max_shell: ShellTransform) -> TransformState:
    mid_x = 0.5 * (float(min_shell.dilate_x) + float(max_shell.dilate_x))
    mid_y = 0.5 * (float(min_shell.dilate_y) + float(max_shell.dilate_y))
    mid_z = 0.5 * (float(min_shell.dilate_z) + float(max_shell.dilate_z))
    return TransformState(
        scale=float(np.mean([mid_x, mid_y, mid_z])),
        scale_min=float(np.mean([min_shell.dilate_x, min_shell.dilate_y, min_shell.dilate_z])),
        scale_max=float(np.mean([max_shell.dilate_x, max_shell.dilate_y, max_shell.dilate_z])),
        dilate_x=mid_x,
        dilate_y=mid_y,
        dilate_z=mid_z,
        dilate_x_min=float(min_shell.dilate_x),
        dilate_x_max=float(max_shell.dilate_x),
        dilate_y_min=float(min_shell.dilate_y),
        dilate_y_max=float(max_shell.dilate_y),
        dilate_z_min=float(min_shell.dilate_z),
        dilate_z_max=float(max_shell.dilate_z),
        rot_x=0.5 * (float(min_shell.rot_x) + float(max_shell.rot_x)),
        rot_y=0.5 * (float(min_shell.rot_y) + float(max_shell.rot_y)),
        rot_z=0.5 * (float(min_shell.rot_z) + float(max_shell.rot_z)),
    )


def split_transform_state(state: TransformState) -> Tuple[ShellTransform, ShellTransform]:
    return (
        ShellTransform(
            dilate_x=float(state.dilate_x_min),
            dilate_y=float(state.dilate_y_min),
            dilate_z=float(state.dilate_z_min),
            rot_x=float(state.rot_x),
            rot_y=float(state.rot_y),
            rot_z=float(state.rot_z),
        ),
        ShellTransform(
            dilate_x=float(state.dilate_x_max),
            dilate_y=float(state.dilate_y_max),
            dilate_z=float(state.dilate_z_max),
            rot_x=float(state.rot_x),
            rot_y=float(state.rot_y),
            rot_z=float(state.rot_z),
        ),
    )


@dataclass
class CategoryAssignment:
    categories: List[str]
    confidence: str


@dataclass
class HistoryRecord:
    version: str
    timestamp: str
    transform: TransformState
    categories: List[str]
    confidence: str
    paths: Dict[str, str]
    stats: Dict[str, Dict]
