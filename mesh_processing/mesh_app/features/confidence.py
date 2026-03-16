from PyQt5 import QtCore, QtWidgets

from .base import FeatureBase


class ConfidenceFeature(FeatureBase):
    name = "confidence"

    def init(self, ui, controller, registry=None):
        super().init(ui, controller, registry)
        QtWidgets.QShortcut(QtCore.Qt.Key_C, ui, activated=self._toggle_confidence)

    def widget(self):
        group = QtWidgets.QGroupBox("Confidence")
        layout = QtWidgets.QVBoxLayout()
        self.btn_conf = QtWidgets.QPushButton("confident")
        self.btn_not = QtWidgets.QPushButton("not confident")
        layout.addWidget(self.btn_conf)
        layout.addWidget(self.btn_not)
        group.setLayout(layout)

        self.btn_conf.clicked.connect(lambda: self._set_confidence("confident"))
        self.btn_not.clicked.connect(lambda: self._set_confidence("not_confident"))
        self._set_confidence("confident")
        return group

    def _set_confidence(self, value: str):
        self.controller.confidence = value
        if value == "confident":
            self.btn_conf.setStyleSheet("background-color: #1a73e8; border-color: #1a73e8; color: white;")
            self.btn_not.setStyleSheet("background-color: #3c4043; border-color: #5f6368; color: #e8eaed;")
        else:
            self.btn_not.setStyleSheet("background-color: #d93025; border-color: #d93025; color: white;")
            self.btn_conf.setStyleSheet("background-color: #3c4043; border-color: #5f6368; color: #e8eaed;")

    def _toggle_confidence(self):
        current = getattr(self.controller, "confidence", "confident")
        self._set_confidence("not_confident" if current == "confident" else "confident")
