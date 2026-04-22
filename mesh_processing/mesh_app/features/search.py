from PyQt5 import QtCore, QtWidgets

from .base import FeatureBase


class SearchFeature(FeatureBase):
    name = "search"

    def widget(self):
        group = QtWidgets.QGroupBox("Navigate")
        layout = QtWidgets.QVBoxLayout()

        row = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("index or name")
        btn_go = QtWidgets.QPushButton("Go")
        row.addWidget(self.search)
        row.addWidget(btn_go)
        layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        btn_prev = QtWidgets.QPushButton("Prev")
        btn_next = QtWidgets.QPushButton("Next")
        row2.addWidget(btn_prev)
        row2.addWidget(btn_next)
        layout.addLayout(row2)

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

        btn_go.clicked.connect(self._go)
        btn_prev.clicked.connect(self.controller.prev)
        btn_next.clicked.connect(self.controller.next)
        self.search.returnPressed.connect(self._go)
        QtWidgets.QShortcut(QtCore.Qt.Key_Left, self.ui, activated=self.controller.prev)
        QtWidgets.QShortcut(QtCore.Qt.Key_Right, self.ui, activated=self.controller.next)

        group.setLayout(layout)
        return group

    def on_entry_loaded(self, entry):
        idx = self.controller.navigator.index + 1
        total = len(self.controller.navigator.filtered)
        self.status.setText(
            f"<span style='color:#9aa0a6;'>{idx} / {total}</span> | "
            f"<span style='color:#8ab4f8;font-weight:600;'>{entry.full_name}</span>"
        )

    def _go(self):
        query = self.search.text().strip()
        if not query:
            return
        self.controller.jump_to(query)
