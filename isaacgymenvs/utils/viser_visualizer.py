from __future__ import annotations

import time
import numpy as np
from pathlib import Path

import viser
from viser.extras import ViserUrdf

import threading


class ViserVisualizer:
    def __init__(self, urdf_path):
        self.urdf_path = Path(urdf_path)

        # Create server and a parent frame for the robot; this frame lets us move the whole robot.
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(control_width="medium")  # small | medium | large
        # Visualize world frame
        self.server.scene.add_frame("/WorldAxes", show_axes=True, axes_length=0.3, axes_radius=0.02, visible=True,)

        # Get the robot base frame (taking in fused state information)
        self.robot_base_frame = self.server.scene.add_frame("/robot", show_axes=False)

        # Build URDF visualizer.
        self._urdf_vis = ViserUrdf(
            self.server, urdf_or_path=self.urdf_path, root_node_name="/robot", load_meshes=True, load_collision_meshes=False
        )

        self.camera_frame_viz = self.server.scene.add_frame("/cam_frame", show_axes=True, axes_length=0.2, axes_radius=0.01, visible=True)

        # Create grid.
        self.server.scene.add_grid("/grid", width=10, height=10, position=(0.0, 0.0, 0.0), shadow_opacity=0.1)

        # Cache joint ordering and limits.
        self.joint_names = self._urdf_vis.get_actuated_joint_names()
        self._name_to_idx = {name: i for i, name in enumerate(self.joint_names)}
        # Default joint information
        self.base_default_joint_pos = np.zeros(3)
        self.franka_default_joint_pos = np.array([0.0, -0.25 * np.pi, 0.0, -0.75 * np.pi, 0.0, 0.5 * np.pi, 0.0])
        self.leap_v1_default_joint_pos = np.zeros(16)
        self.arx_default_joint_pos = np.array([0.0, 0.785, 0.785, 0.0, 0.0, 0.0])
        self.initial_config = np.concatenate(
            (self.base_default_joint_pos, self.franka_default_joint_pos, self.leap_v1_default_joint_pos, self.arx_default_joint_pos)
        )
        self._q = self.initial_config.copy()
        self._urdf_vis.update_cfg(self._q)

        self._q_lock = threading.Lock()
        self._stop_event = threading.Event()
        update_hz = 30
        self._update_period = 1.0 / float(update_hz)
        self._cfg_thread = threading.Thread(
            target=self._cfg_pusher_loop, name="ViserCfgPusher", daemon=True
        )
        self._cfg_thread.start()

        # ------------------------- storage for scene obstacle handles -------------------------
        self.scene_handles = {
            "box": {},
            "mesh": {},
            "sphere": {},
            "cyl": {},
        }

        # ---------------------- predefined point clouds ----------------------
        self._point_cloud_handle = dict()
        self._point_cloud_handle["camera_los_points"] = self.server.scene.add_point_cloud(
            name="/camera_los_points",
            points=np.zeros((0, 3), dtype=np.float16),
            colors=(200, 0, 0),
            point_size=0.01/2,
            precision="float16",
            visible=True,
        )

        self._point_cloud_handle["rendered_points"] = self.server.scene.add_point_cloud(
            name="/rendered_points",
            points=np.zeros((0, 3), dtype=np.float16),
            colors=(0, 0, 200),
            point_size=0.01/2,
            precision="float16",
            visible=True,
        )

        self._point_cloud_handle["full_points"] = self.server.scene.add_point_cloud(
            name="/full_points",
            points=np.zeros((0, 3), dtype=np.float16),
            colors=(0, 0, 0),
            point_size=0.01/2,
            precision="float16",
            visible=True,
        )
        # ---------------------- camera line of sight ----------------------
        self._camera_los_lines_handle = None

    # ----------- joint pos update thread -----------
    def _cfg_pusher_loop(self):
        """Continuously push the latest joint cfg to the URDF viewer at fixed rate."""
        next_t = time.perf_counter()
        last_pushed = None  # cache to avoid redundant websocket traffic
        while not self._stop_event.is_set():
            # Sleep to maintain ~30 Hz
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += self._update_period

            with self._q_lock:
                q_snapshot = self._q.copy()

            # self._urdf_vis.update_cfg(q_snapshot)
            # # Optional: cheap redundancy filter
            if last_pushed is None or not np.array_equal(q_snapshot, last_pushed):
                self._urdf_vis.update_cfg(q_snapshot)
                last_pushed = q_snapshot

    def close(self, join_timeout: float | None = 1.0):
        self._stop_event.set()
        self._cfg_thread.join(timeout=join_timeout)

    def update_joint_pos(self, joint_pos_map):
        q = np.zeros_like(self._q)
        for name, v in joint_pos_map.items():
            idx = self._name_to_idx.get(name)
            if idx is None:
                raise KeyError(f"Unknown joint '{name}'. Known joints: {list(self.joint_names)}")
            # self._q[idx] = v
            q[idx] = v

        with self._q_lock:
            self._q[:] = q
        # self._urdf_vis.update_cfg(self._q)

    def set_joint_positions(self, q: np.ndarray):
        if q.shape != self._q.shape:
            raise ValueError(f"Expected q shape {self._q.shape}, got {q.shape}.")
        q = np.asarray(q, dtype=float)
        with self._q_lock:
            self._q[:] = q
        # self._urdf_vis.update_cfg(self._q)    

    def update_point_cloud(self, point_cloud_type, point_cloud, colors=None, point_size=None, precision="float16"):
        # point_cloud_type is either "local" or "scene"
        point_cloud_handle = self._point_cloud_handle[point_cloud_type]

        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        point_cloud = point_cloud.astype(np.float16 if precision == "float16" else np.float32, copy=False)
        point_cloud_handle.points = point_cloud

        if hasattr(point_cloud_handle, "precision"):
            point_cloud_handle.precision = precision  # type: ignore[attr-defined]
        if point_size is not None:
            point_cloud_handle.point_size = float(point_size)
        if colors is not None:
            point_cloud_handle.colors = colors # type: ignore[assignment]

    def update_camera_los_lines(
        self,
        camera_los_sample_points: np.ndarray,
        camera_los_accel_dir: np.ndarray,
        camera_los_distance: np.ndarray,
        name: str = "/camera_los_lines",
        line_width: float = 2.0,
    ):
        """
        Visualize N line segments for camera LoS repulsion.

        Each line:
            start = camera_los_sample_points[i]
            end   = start + camera_los_accel_dir[i] * |camera_los_distance[i]|
        Colors:
            green if distance >= 0
            red   if distance < 0
        """
        # Ensure shapes
        p = np.asarray(camera_los_sample_points, dtype=np.float32).reshape(-1, 3)
        d = np.asarray(camera_los_accel_dir, dtype=np.float32).reshape(-1, 3)
        dist = np.asarray(camera_los_distance, dtype=np.float32).reshape(-1)

        assert p.shape == d.shape, f"sample_points ({p.shape}) and accel_dir ({d.shape}) must match"
        assert p.shape[0] == dist.shape[0], "distance must have length N"

        N = p.shape[0]
        if N == 0:
            # Nothing to draw
            if self._camera_los_lines_handle is not None:
                self._camera_los_lines_handle.points = np.zeros((0, 2, 3), dtype=np.float32)
            return

        # Use magnitude of signed distance as length; change to `dist` if you want signed length.
        lengths = np.abs(dist)  # (N,)
        end_points = p + d * lengths[:, None]  # (N, 3)

        # (N, 2, 3): [start, end] for each segment
        points = np.stack([p, end_points], axis=1)  # (N, 2, 3)

        # Color segments by sign of distance: green = free, red = colliding
        colors_per_segment = np.zeros((N, 3), dtype=np.uint8)
        mask_free = dist >= 0.0
        colors_per_segment[mask_free] = np.array([0, 0, 255], dtype=np.uint8)   # green
        colors_per_segment[~mask_free] = np.array([255, 0, 0], dtype=np.uint8)  # red
        colors = np.repeat(colors_per_segment[:, None, :], 2, axis=1)  # (N, 2, 3)

        if self._camera_los_lines_handle is None:
            # First time: create node
            self._camera_los_lines_handle = self.server.scene.add_line_segments(
                name=name,
                points=points,
                colors=colors,
                line_width=line_width,
            )
        else:
            # Subsequent calls: just update geometry/colors
            self._camera_los_lines_handle.points = points
            self._camera_los_lines_handle.colors = colors
