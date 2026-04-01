from PyQt5 import QtWidgets

from .base import FeatureBase


class StatsFeature(FeatureBase):
    name = "stats"

    def widget(self):
        group = QtWidgets.QGroupBox("Stats")
        layout = QtWidgets.QVBoxLayout()
        self.lbl = QtWidgets.QLabel("")
        self.lbl.setWordWrap(True)
        layout.addWidget(self.lbl)
        group.setLayout(layout)
        return group

    def on_entry_loaded(self, entry):
        self._update()

    def on_transform_changed(self, state):
        self._update()

    def _update(self):
        obj_stats, ref_stats = self.controller.get_stats()
        if obj_stats is None:
            self.lbl.setText("No stats")
            return
        def fmt_stats(title, stats, color):
            bbox = stats["bbox_extents"]
            return (
                f"<b style='color:{color};'>{title}</b><br>"
                f"<table>"
                f"<tr><td style='padding-right:8px;color:#8ab4f8;'>bbox:</td>"
                f"<td><span style='font-weight:700;font-size:12pt;'>"
                f"{bbox[0]:.5f}, {bbox[1]:.5f}, {bbox[2]:.5f}"
                f"</span></td></tr>"
                f"<tr><td style='padding-right:8px;color:#34a853;'>volume:</td>"
                f"<td><span style='font-weight:700;font-size:12pt;'>"
                f"{stats['volume']:.8f}"
                f"</span></td></tr>"
                f"<tr><td style='padding-right:8px;color:#fbbc05;'>max:</td>"
                f"<td><span style='font-weight:700;font-size:12pt;'>"
                f"{stats['max_dim']:.5f}"
                f"</span></td></tr>"
                f"</table>"
            )

        text = fmt_stats("Object", obj_stats, "#e8eaed")
        info = getattr(self.controller, "mesh_info", {})
        if info:
            watertight = info.get("watertight")
            hull_volume = info.get("hull_volume")
            fixed = info.get("fixed_preview")
            text += "<br><b style='color:#8ab4f8;'>Analysis</b><br>"
            if watertight is not None:
                status = "watertight" if watertight else "not watertight"
                color = "#34a853" if watertight else "#ea4335"
                text += f"<span style='color:{color};'>mesh:</span> {status}<br>"
            if hull_volume is not None:
                text += f"<span style='color:#fbbc05;'>hull volume:</span> {hull_volume:.8f}<br>"
            if fixed:
                text += "<span style='color:#c58af9;'>preview fix:</span> on<br>"
        if ref_stats is not None:
            text += "<br>" + fmt_stats("Reference", ref_stats, "#ff8a80")
        self.lbl.setText(text)
