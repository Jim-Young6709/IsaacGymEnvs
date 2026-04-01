from PyQt5 import QtCore, QtWidgets

from .base import FeatureBase
from ..models import TransformState


class TransformFeature(FeatureBase):
    name = "transform"

    def __init__(self):
        self.scale_spin = None
        self.scale_slider = None
        self.lbl_state = None
        self._debounce_timer = QtCore.QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(80)

    def widget(self):
        group = QtWidgets.QGroupBox("Transform")
        layout = QtWidgets.QVBoxLayout()

        self.scale_spin = QtWidgets.QDoubleSpinBox()
        max_scale = getattr(self.controller.config, "max_scale", 10.0)
        self.scale_spin.setRange(0.05, max_scale)
        self.scale_spin.setDecimals(3)
        self.scale_spin.setSingleStep(0.01)
        self.scale_spin.setValue(1.0)

        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(50, int(max_scale * 1000))
        self.scale_slider.setValue(1000)

        scale_row = QtWidgets.QHBoxLayout()
        scale_row.addWidget(QtWidgets.QLabel("Scale"))
        scale_row.addWidget(self.scale_spin)
        layout.addLayout(scale_row)
        layout.addWidget(self.scale_slider)

        rot_row = QtWidgets.QHBoxLayout()
        btn_rx = QtWidgets.QPushButton("X +90")
        btn_ry = QtWidgets.QPushButton("Y +90")
        btn_rz = QtWidgets.QPushButton("Z +90")
        rot_row.addWidget(btn_rx)
        rot_row.addWidget(btn_ry)
        rot_row.addWidget(btn_rz)
        layout.addLayout(rot_row)

        self.lbl_state = QtWidgets.QLabel("Rotation: X=0 Y=0 Z=0")
        layout.addWidget(self.lbl_state)

        btn_rx.clicked.connect(lambda: self._rotate("x"))
        btn_ry.clicked.connect(lambda: self._rotate("y"))
        btn_rz.clicked.connect(lambda: self._rotate("z"))

        self.scale_spin.valueChanged.connect(self._on_spin)
        self.scale_slider.valueChanged.connect(self._on_slider)

        QtWidgets.QShortcut(QtCore.Qt.Key_Z, self.ui, activated=lambda: self._rotate("x"))
        QtWidgets.QShortcut(QtCore.Qt.Key_X, self.ui, activated=lambda: self._rotate("y"))
        QtWidgets.QShortcut(QtCore.Qt.Key_C, self.ui, activated=lambda: self._rotate("z"))

        self._debounce_timer.timeout.connect(self.controller.refresh_view)

        group.setLayout(layout)
        return group

    def _update_state_label(self, state: TransformState):
        self.lbl_state.setText(
            f"Rotation: X={state.rot_x:.0f} Y={state.rot_y:.0f} Z={state.rot_z:.0f} | scale={state.scale:.2f}"
        )

    def _on_spin(self, value: float):
        slider_val = int(round(value * 1000))
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(slider_val)
        self.scale_slider.blockSignals(False)
        state = self.controller.transform_state
        state.scale = value
        self.controller.set_transform_state(state)
        self._debounce_timer.start()

    def _on_slider(self, value: int):
        scale = value / 1000.0
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(scale)
        self.scale_spin.blockSignals(False)
        state = self.controller.transform_state
        state.scale = scale
        self.controller.set_transform_state(state)
        self._debounce_timer.start()

    def _rotate(self, axis: str):
        state = self.controller.transform_state
        if axis == "x":
            state.rot_x = (state.rot_x + 90.0) % 360.0
        elif axis == "y":
            state.rot_y = (state.rot_y + 90.0) % 360.0
        else:
            state.rot_z = (state.rot_z + 90.0) % 360.0
        self.controller.set_transform(state)

    def on_transform_changed(self, state: TransformState):
        self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(state.scale)
        self.scale_spin.blockSignals(False)

        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(int(round(state.scale * 1000)))
        self.scale_slider.blockSignals(False)

        self._update_state_label(state)
