from PyQt5 import QtWidgets

from .base import FeatureBase


class ShortcutsFeature(FeatureBase):
    name = "shortcuts"
    title = "Shortcuts"

    def widget(self):
        group = QtWidgets.QGroupBox("Shortcuts")
        layout = QtWidgets.QVBoxLayout()
        self.lbl = QtWidgets.QLabel(
            """
<b>Category Groups</b><br>
Q: regular group<br>
W: long group<br>
E: irregular group<br><br>
<b>Category Selection</b><br>
1-9: pick category in group<br><br>
<b>Navigation</b><br>
Left/Right: prev/next object<br><br>
<b>Actions</b><br>
S: save selected<br>
D: remove selected<br>
A: save all categories<br>
C: toggle confidence<br><br>
<b>Focus</b><br>
F: focus selected<br>
G: focus all<br><br>
<b>Transforms</b><br>
X/Y/Z: rotate +90 deg<br>
Shift + wheel: scale
"""
        )
        self.lbl.setWordWrap(True)
        layout.addWidget(self.lbl)
        group.setLayout(layout)
        return group
