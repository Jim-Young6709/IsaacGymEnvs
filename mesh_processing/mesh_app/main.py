import os
import sys
import argparse

from PyQt5 import QtWidgets

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from mesh_app.config import AppConfig
from mesh_app.controller import MeshController
from mesh_app.features import FeatureRegistry
from mesh_app.features.category import CategoryFeature
from mesh_app.features.confidence import ConfidenceFeature
from mesh_app.features.history import HistoryFeature
from mesh_app.features.overlay import OverlayFeature
from mesh_app.features.panel_manager import PanelManagerFeature
from mesh_app.features.search import SearchFeature
from mesh_app.features.shortcuts import ShortcutsFeature
from mesh_app.features.stats import StatsFeature
from mesh_app.features.transform import TransformFeature
from mesh_app.features.leap_hand import LeapHandFeature
from mesh_app.ui import AppWindow
from mesh_app.viewer import MeshViewer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scale", type=float, default=10.0, help="Maximum scale value.")
    args, _ = parser.parse_known_args()

    app = QtWidgets.QApplication(sys.argv)
    config = AppConfig()
    config.max_scale = float(args.max_scale)

    viewer = MeshViewer()
    controller = MeshController(config, viewer)

    registry = FeatureRegistry()
    registry.register(PanelManagerFeature())
    registry.register(SearchFeature())          # navigate
    registry.register(StatsFeature())           # states
    registry.register(TransformFeature())
    registry.register(ShortcutsFeature())
    registry.register(ConfidenceFeature())
    registry.register(CategoryFeature(config.categories))
    registry.register(OverlayFeature())         # viewer
    registry.register(HistoryFeature(config.categories))
    registry.register(LeapHandFeature())

    window = AppWindow(controller, registry, viewer)
    controller.register_features(registry)

    if controller.entries:
        controller.load_entry(controller.current_entry())

    window.resize(2100, 1200)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
