from __future__ import annotations

import time
import numpy as np
from pathlib import Path

import viser
from viser.extras import ViserUrdf

import threading


class ViserVisualizer:
    def __init__(self, urdf_path, num_envs):
        self.urdf_path = Path(urdf_path)

        # Create server and a parent frame for the robot; this frame lets us move the whole robot.
        self.server = viser.ViserServer()
        self.server.gui.configure_theme(control_width="medium")  # small | medium | large
        # Visualize world frame
        self.server.scene.add_frame("/WorldAxes", show_axes=True, axes_length=0.15, axes_radius=0.01, visible=True,)

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

        # ---------------------- predefined point clouds ----------------------
        self._point_cloud_handle = dict()
        self._point_cloud_handle["rendered_points"] = self.server.scene.add_point_cloud(
            name="/rendered_points",
            points=np.zeros((0, 3), dtype=np.float16),
            colors=(79, 195, 247),
            point_size=0.01/3,
            precision="float16",
            visible=True,
        )
        self._point_cloud_handle["full_points"] = self.server.scene.add_point_cloud(
            name="/full_points",
            points=np.zeros((0, 3), dtype=np.float16),
            colors=(100, 100, 100),
            point_size=0.01/4,
            precision="float16",
            visible=True,
        )

        # ---------------------- env_id ----------------------
        self.env_id = 0
        self._env_id_handle = self.server.gui.add_number(
            label="Env ID",
            initial_value=0,
            min=0,
            max=num_envs-1,
            step=1, # force integer steps
            hint="Select environment index",
        )
        @self._env_id_handle.on_update
        def _on_env_id_update(event):
            # event.value will already be clamped to [0, num_envs-1]
            self.env_id = int(self._env_id_handle.value)
        
        # ---------------------- env_id ----------------------
        self.isaac_to_viser_idx = [
            0,  # base_x_joint
            1,  # base_y_joint
            2,  # base_rotation_joint
            3,  # panda_joint1
            4,  # panda_joint2
            5,  # panda_joint3
            6,  # panda_joint4
            7,  # panda_joint5
            8,  # panda_joint6
            9,  # panda_joint7

            11, # finger_joint_0  (Isaac has 1 then 0)
            10, # finger_joint_1
            12, # finger_joint_2
            13, # finger_joint_3

            19, # finger_joint_4
            18, # finger_joint_5
            20, # finger_joint_6
            21, # finger_joint_7

            23, # finger_joint_8
            22, # finger_joint_9
            24, # finger_joint_10
            25, # finger_joint_11

            14, # finger_joint_12
            15, # finger_joint_13
            16, # finger_joint_14
            17, # finger_joint_15

            26, # x5_joint1
            27, # x5_joint2
            28, # x5_joint3
            29, # x5_joint4
            30, # x5_joint5
            31, # x5_joint6
        ]


    

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

    def set_joint_positions(self, q):
        # q is assumed to be in the same ordering as Isaac Gym
        q = q[self.isaac_to_viser_idx]
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
