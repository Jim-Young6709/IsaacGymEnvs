from PyQt5 import QtCore, QtWidgets

from .base import FeatureBase


class LeapHandFeature(FeatureBase):
    name = "leap_hand"

    def init(self, ui, controller, registry=None):
        super().init(ui, controller, registry)
        self.controller.hand_scale = 1.0
        self.controller.hand_z = 0.1
        self.controller.load_hand(self.controller.config.leap_hand_path)
        self._default_pose = [
            0.95, -0.2, 0.95, 0.95,
            1.0, 1.57, 1.0, 1.14,
            0.9, 0.0, 0.9, 0.9,
            0.95, 0.2, 0.95, 0.95,
        ]
        if len(self.controller.hand_joint_order) == len(self._default_pose):
            self.controller.set_hand_joint_angles(self._pose_to_joint_angles(self._default_pose))
        self._widgets_ready = False
        self.chk_show = None
        self.z_spin = None
        self.x_spin = None
        self.y_spin = None
        self.joint_sliders = []
        self.offset_sliders = {}
        self.print_btn = None
        self.rot_btns = {}

    def on_entry_loaded(self, entry):
        if getattr(self, "_widgets_ready", False):
            self.controller.hand_z = float(self.z_spin.value())

    def _update(self):
        if not self.chk_show or not self.z_spin or not self.x_spin or not self.y_spin:
            return
        self.controller.show_hand = self.chk_show.isChecked()
        self.controller.hand_x = float(self.x_spin.value())
        self.controller.hand_y = float(self.y_spin.value())
        self.controller.hand_z = float(self.z_spin.value())
        self.controller.refresh_view()

    def _rotate_hand(self, axis):
        self.controller.rotate_hand(axis, 90.0)
        self.controller.refresh_view()

    def _pose_to_joint_angles(self, pose):
        if not self.controller.hand_joint_order:
            return pose
        finger_groups = {
            "index": [f"finger_joint_{i}" for i in range(0, 4)],
            "thumb": [f"finger_joint_{i}" for i in range(4, 8)],
            "middle": [f"finger_joint_{i}" for i in range(8, 12)],
            "ring": [f"finger_joint_{i}" for i in range(12, 16)],
        }
        order = getattr(self.controller.config, "leap_hand_pose_order", ["index", "thumb", "middle", "ring"])
        values = list(pose)
        mapping = {}
        idx = 0
        for finger in order:
            joints = finger_groups.get(finger, [])
            for joint in joints:
                if idx < len(values):
                    mapping[joint] = values[idx]
                idx += 1
        angles = []
        for name in self.controller.hand_joint_order:
            angles.append(mapping.get(name, 0.0))
        return angles

    def _joint_angles_to_pose(self, angles):
        finger_groups = {
            "index": [f"finger_joint_{i}" for i in range(0, 4)],
            "thumb": [f"finger_joint_{i}" for i in range(4, 8)],
            "middle": [f"finger_joint_{i}" for i in range(8, 12)],
            "ring": [f"finger_joint_{i}" for i in range(12, 16)],
        }
        order = getattr(self.controller.config, "leap_hand_pose_order", ["index", "thumb", "middle", "ring"])
        angle_map = {}
        for name, angle in zip(self.controller.hand_joint_order, angles):
            angle_map[name] = angle
        pose = []
        for finger in order:
            for joint in finger_groups.get(finger, []):
                pose.append(angle_map.get(joint, 0.0))
        return pose

    def _print_pose(self):
        angles = list(self.controller.hand_joint_angles)
        if not angles:
            return
        pose = self._joint_angles_to_pose(angles)
        print("leap_hand_joint_order:", self.controller.hand_joint_order)
        print("leap_hand_joint_angles:", [round(v, 4) for v in angles])
        print("leap_hand_pose_order:", getattr(self.controller.config, "leap_hand_pose_order", []))
        print("grasp_default = [")
        for i in range(0, len(pose), 4):
            chunk = ", ".join(f"{pose[j]:.4f}" for j in range(i, min(i + 4, len(pose))))
            print(f"    {chunk},")
        print("]")

    def _on_joint_change(self, idx, value):
        if not self.controller.hand_joint_order:
            return
        if idx < 0 or idx >= len(self.controller.hand_joint_order):
            return
        lower, upper = self.controller.hand_joint_limits.get(
            self.controller.hand_joint_order[idx], (-3.14, 3.14)
        )
        if lower is None:
            lower = -3.14
        if upper is None:
            upper = 3.14
        t = float(value) / 10000.0
        angle = lower + t * (upper - lower)
        angles = list(self.controller.hand_joint_angles)
        while len(angles) < len(self.controller.hand_joint_order):
            angles.append(0.0)
        angles[idx] = angle
        self.controller.set_hand_joint_angles(angles)
        self.controller.refresh_view()

    def _set_slider_values(self):
        if not self.controller.hand_joint_order:
            return
        for idx, slider in enumerate(self.joint_sliders):
            lower, upper = self.controller.hand_joint_limits.get(
                self.controller.hand_joint_order[idx], (-3.14, 3.14)
            )
            if lower is None:
                lower = -3.14
            if upper is None:
                upper = 3.14
            angle = 0.0
            if idx < len(self.controller.hand_joint_angles):
                angle = self.controller.hand_joint_angles[idx]
            if upper == lower:
                slider.setValue(0)
            else:
                t = (angle - lower) / (upper - lower)
                slider.setValue(int(max(0.0, min(1.0, t)) * 10000))

    def widget(self):
        group = QtWidgets.QGroupBox("Hand")
        layout = QtWidgets.QVBoxLayout()

        self.chk_show = QtWidgets.QCheckBox("Show hand")
        self.chk_show.setChecked(False)
        layout.addWidget(self.chk_show)

        offset_row = QtWidgets.QGridLayout()
        offset_row.addWidget(QtWidgets.QLabel("X offset"), 0, 0)
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setRange(-1.0, 1.0)
        self.x_spin.setDecimals(3)
        self.x_spin.setSingleStep(0.01)
        self.x_spin.setValue(0.0)
        offset_row.addWidget(self.x_spin, 0, 1)
        x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        x_slider.setRange(-10000, 10000)
        x_slider.setValue(0)
        offset_row.addWidget(x_slider, 0, 2)
        self.offset_sliders["x"] = x_slider

        offset_row.addWidget(QtWidgets.QLabel("Y offset"), 1, 0)
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(-1.0, 1.0)
        self.y_spin.setDecimals(3)
        self.y_spin.setSingleStep(0.01)
        self.y_spin.setValue(0.0)
        offset_row.addWidget(self.y_spin, 1, 1)
        y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        y_slider.setRange(-10000, 10000)
        y_slider.setValue(0)
        offset_row.addWidget(y_slider, 1, 2)
        self.offset_sliders["y"] = y_slider

        offset_row.addWidget(QtWidgets.QLabel("Z offset"), 2, 0)
        self.z_spin = QtWidgets.QDoubleSpinBox()
        self.z_spin.setRange(-1.0, 1.0)
        self.z_spin.setDecimals(3)
        self.z_spin.setSingleStep(0.01)
        self.z_spin.setValue(0.1)
        offset_row.addWidget(self.z_spin, 2, 1)
        z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        z_slider.setRange(-10000, 10000)
        z_slider.setValue(1000)
        offset_row.addWidget(z_slider, 2, 2)
        self.offset_sliders["z"] = z_slider

        layout.addLayout(offset_row)

        rot_row = QtWidgets.QHBoxLayout()
        rot_row.addWidget(QtWidgets.QLabel("Rotate +90"))
        for axis in ("x", "y", "z"):
            btn = QtWidgets.QPushButton(axis.upper())
            btn.clicked.connect(lambda _, a=axis: self._rotate_hand(a))
            self.rot_btns[axis] = btn
            rot_row.addWidget(btn)
        layout.addLayout(rot_row)

        if self.controller.hand_joint_order:
            joints_group = QtWidgets.QGroupBox("Joints")
            joints_layout = QtWidgets.QGridLayout()
            self.joint_sliders = []
            for idx, name in enumerate(self.controller.hand_joint_order):
                label = QtWidgets.QLabel(name.replace("finger_joint_", "J"))
                slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
                slider.setRange(0, 10000)
                slider.valueChanged.connect(lambda val, i=idx: self._on_joint_change(i, val))
                joints_layout.addWidget(label, idx, 0)
                joints_layout.addWidget(slider, idx, 1)
                self.joint_sliders.append(slider)
            joints_group.setLayout(joints_layout)
            layout.addWidget(joints_group)
            self._set_slider_values()

        self.print_btn = QtWidgets.QPushButton("Print joint angles")
        self.print_btn.clicked.connect(self._print_pose)
        layout.addWidget(self.print_btn)

        group.setLayout(layout)

        def sync_spin_from_slider(axis, slider):
            def _handler(value):
                spin = {"x": self.x_spin, "y": self.y_spin, "z": self.z_spin}[axis]
                spin.blockSignals(True)
                spin.setValue(float(value) / 10000.0)
                spin.blockSignals(False)
                self._update()
            return _handler

        def sync_slider_from_spin(axis, spin):
            def _handler(value):
                slider = self.offset_sliders[axis]
                slider.blockSignals(True)
                slider.setValue(int(round(float(value) * 10000)))
                slider.blockSignals(False)
                self._update()
            return _handler

        self.chk_show.stateChanged.connect(self._update)
        self.x_spin.valueChanged.connect(sync_slider_from_spin("x", self.x_spin))
        self.y_spin.valueChanged.connect(sync_slider_from_spin("y", self.y_spin))
        self.z_spin.valueChanged.connect(sync_slider_from_spin("z", self.z_spin))
        self.offset_sliders["x"].valueChanged.connect(sync_spin_from_slider("x", self.offset_sliders["x"]))
        self.offset_sliders["y"].valueChanged.connect(sync_spin_from_slider("y", self.offset_sliders["y"]))
        self.offset_sliders["z"].valueChanged.connect(sync_spin_from_slider("z", self.offset_sliders["z"]))
        self._widgets_ready = True

        return group

    def on_transform_changed(self, state):
        if not getattr(self, "_widgets_ready", False):
            if hasattr(self, "z_spin") and self.z_spin is not None:
                self._widgets_ready = True
