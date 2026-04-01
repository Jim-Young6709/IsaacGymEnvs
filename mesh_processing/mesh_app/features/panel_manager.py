from PyQt5 import QtCore, QtWidgets

from .base import FeatureBase


class PanelManagerFeature(FeatureBase):
    name = "panel_manager"
    title = "Panels"

    def widget(self):
        group = QtWidgets.QGroupBox("Panels")
        layout = QtWidgets.QVBoxLayout()
        group.setStyleSheet(
            "QGroupBox { font-size: 16pt; } "
            "QListWidget, QPushButton { font-size: 16pt; }"
        )

        body_row = QtWidgets.QHBoxLayout()

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_widget.setFlow(QtWidgets.QListView.LeftToRight)
        self.list_widget.setWrapping(True)
        self.list_widget.setResizeMode(QtWidgets.QListView.Adjust)
        self.list_widget.setSpacing(6)
        body_row.addWidget(self.list_widget, stretch=1)

        btn_col = QtWidgets.QVBoxLayout()
        self.btn_up = QtWidgets.QPushButton("Up")
        self.btn_down = QtWidgets.QPushButton("Down")
        btn_col.addWidget(self.btn_up)
        btn_col.addWidget(self.btn_down)
        btn_col.addStretch()
        body_row.addLayout(btn_col)

        layout.addLayout(body_row)

        self.btn_up.clicked.connect(lambda: self._move(-1))
        self.btn_down.clicked.connect(lambda: self._move(1))
        self.list_widget.itemChanged.connect(self._toggle_visibility)

        group.setLayout(layout)
        self._populate()
        return group

    def refresh_panels(self):
        self._populate()

    def _populate(self):
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for feature in self.registry.features:
            if feature.name == self.name:
                continue
            widget = self.ui.feature_widgets.get(feature.name)
            if widget is None:
                continue
            item = QtWidgets.QListWidgetItem(feature.name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            item.setCheckState(QtCore.Qt.Checked)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            self.ui.set_feature_visible(item.text(), True)

    def _toggle_visibility(self, item):
        feature_name = item.text()
        visible = item.checkState() == QtCore.Qt.Checked
        self.ui.set_feature_visible(feature_name, visible)

    def _move(self, delta: int):
        row = self.list_widget.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if new_row < 0 or new_row >= self.list_widget.count():
            return
        item = self.list_widget.takeItem(row)
        self.list_widget.insertItem(new_row, item)
        self.list_widget.setCurrentRow(new_row)
        feature_name = item.text()
        self.ui.reorder_feature(feature_name, new_row + 1)

    
