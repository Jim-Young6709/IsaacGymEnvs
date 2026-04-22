from PyQt5 import QtCore, QtWidgets

from .base import FeatureBase


class HistoryFeature(FeatureBase):
    name = "history"

    def __init__(self, categories):
        self.categories = categories
        self.list_widget = None
        self.category_combo = None
        self.status = None

    def widget(self):
        group = QtWidgets.QGroupBox("History")
        layout = QtWidgets.QVBoxLayout()

        self.category_combo = QtWidgets.QComboBox()
        self.category_combo.addItem("all")
        for cat in self.categories:
            if cat.key not in [self.category_combo.itemText(i) for i in range(self.category_combo.count())]:
                self.category_combo.addItem(cat.key)
        layout.addWidget(self.category_combo)

        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.list_widget)

        btn_row = QtWidgets.QHBoxLayout()
        btn_load = QtWidgets.QPushButton("Load")
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_row.addWidget(btn_load)
        btn_row.addWidget(btn_refresh)
        layout.addLayout(btn_row)

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

        self.category_combo.currentIndexChanged.connect(self._refresh)
        btn_refresh.clicked.connect(self._refresh)
        btn_load.clicked.connect(self._load_selected)

        group.setLayout(layout)
        return group

    def on_entry_loaded(self, entry):
        self._refresh()

    def _refresh(self):
        self.list_widget.clear()
        entry = self.controller.current_entry()
        if entry is None:
            return
        selected = self.category_combo.currentText()
        categories = [selected] if selected != "all" else [c.key for c in self.categories]
        for cat in categories:
            versions = self.controller.history_repo.list_versions(entry, cat)
            for version in versions[::-1]:
                item = QtWidgets.QListWidgetItem(f"{cat} / {version}")
                item.setData(QtCore.Qt.UserRole, (cat, version))
                self.list_widget.addItem(item)

    def _load_selected(self):
        item = self.list_widget.currentItem()
        if item is None:
            self.status.setText("No version selected.")
            return
        cat, version = item.data(QtCore.Qt.UserRole)
        self.controller.load_version(cat, version)
        self.status.setText(f"Loaded {cat} / {version}")
