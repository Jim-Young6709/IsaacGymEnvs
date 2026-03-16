from PyQt5 import QtCore, QtWidgets

from .base import FeatureBase


class OverlayFeature(FeatureBase):
    name = "overlay"

    def widget(self):
        group = QtWidgets.QGroupBox("Viewer")
        layout = QtWidgets.QVBoxLayout()

        self.chk_pc = QtWidgets.QCheckBox("Show pointcloud")
        self.chk_pc.setChecked(True)
        self.chk_com = QtWidgets.QCheckBox("Show COM")
        self.chk_com.setChecked(True)
        self.chk_labels = QtWidgets.QCheckBox("Show grid labels")
        self.chk_labels.setChecked(False)
        self.chk_bbox = QtWidgets.QCheckBox("Show BBox")
        self.chk_bbox.setChecked(False)
        self.chk_bbox_dims = QtWidgets.QCheckBox("Show BBox dims")
        self.chk_bbox_dims.setChecked(False)
        self.chk_autoscale = QtWidgets.QCheckBox("Auto-scale unsaved")
        self.chk_autoscale.setChecked(True)
        self.chk_analysis = QtWidgets.QCheckBox("Compute heavy stats")
        self.chk_analysis.setChecked(False)
        self.chk_hull = QtWidgets.QCheckBox("Show convex hull")
        self.chk_hull.setChecked(False)
        self.chk_fix = QtWidgets.QCheckBox("Preview watertight fix")
        self.chk_fix.setChecked(False)
        self.btn_spin = QtWidgets.QPushButton("Toggle spin")
        self.btn_focus_selected = QtWidgets.QPushButton("Focus selected")
        self.btn_focus_all = QtWidgets.QPushButton("Focus all")

        layout.addWidget(self.chk_pc)
        layout.addWidget(self.chk_com)
        layout.addWidget(self.chk_labels)
        layout.addWidget(self.chk_bbox)
        layout.addWidget(self.chk_bbox_dims)
        layout.addWidget(self.chk_autoscale)
        layout.addWidget(self.chk_analysis)
        layout.addWidget(self.chk_hull)
        layout.addWidget(self.chk_fix)
        layout.addWidget(self.btn_spin)
        layout.addWidget(self.btn_focus_selected)
        layout.addWidget(self.btn_focus_all)

        ref_row = QtWidgets.QHBoxLayout()
        self.btn_ref = QtWidgets.QPushButton("Ref...")
        self.ref_scale = QtWidgets.QDoubleSpinBox()
        self.ref_scale.setRange(0.001, 1000.0)
        self.ref_scale.setDecimals(4)
        self.ref_scale.setValue(1.0)
        ref_row.addWidget(self.btn_ref)
        ref_row.addWidget(QtWidgets.QLabel("Scale"))
        ref_row.addWidget(self.ref_scale)
        layout.addLayout(ref_row)

        self.chk_pc.stateChanged.connect(self._update)
        self.chk_com.stateChanged.connect(self._update)
        self.chk_labels.stateChanged.connect(self._update)
        self.chk_bbox.stateChanged.connect(self._update)
        self.chk_bbox_dims.stateChanged.connect(self._update)
        self.chk_autoscale.stateChanged.connect(self._update)
        self.chk_analysis.stateChanged.connect(self._update)
        self.chk_hull.stateChanged.connect(self._update)
        self.chk_fix.stateChanged.connect(self._update)
        self.btn_spin.clicked.connect(self.controller.viewer.toggle_spin)
        self.btn_focus_selected.clicked.connect(self.controller.focus_selected)
        self.btn_focus_all.clicked.connect(self.controller.focus_all)
        self.btn_ref.clicked.connect(self._select_ref)
        self.ref_scale.valueChanged.connect(self._update_ref_scale)
        QtWidgets.QShortcut(QtCore.Qt.Key_F, self.ui, activated=self.controller.focus_selected)
        QtWidgets.QShortcut(QtCore.Qt.Key_G, self.ui, activated=self.controller.focus_all)

        group.setLayout(layout)
        return group

    def _update(self):
        self.controller.show_pointcloud = self.chk_pc.isChecked()
        self.controller.show_com = self.chk_com.isChecked()
        self.controller.show_grid_labels = self.chk_labels.isChecked()
        self.controller.show_bbox = self.chk_bbox.isChecked()
        self.controller.show_bbox_dims = self.chk_bbox_dims.isChecked()
        self.controller.auto_scale_unsaved = self.chk_autoscale.isChecked()
        self.controller.use_mesh_analysis = self.chk_analysis.isChecked()
        self.controller.show_convex_hull = self.chk_hull.isChecked()
        self.controller.fix_watertight_preview = self.chk_fix.isChecked()
        self.controller.refresh_view()

    def _select_ref(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.ui, "Select Reference OBJ", "", "OBJ Files (*.obj)")
        if not path:
            return
        self.controller.set_reference(path, self.ref_scale.value())

    def _update_ref_scale(self):
        self.controller.set_reference(self.controller.reference_path, self.ref_scale.value())
