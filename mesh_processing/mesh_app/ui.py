from PyQt5 import QtCore, QtWidgets

from .viewer import MeshViewer


class ResizablePanel(QtWidgets.QWidget):
    def __init__(self, inner: QtWidgets.QWidget, parent=None):
        super().__init__(parent=parent)
        self.inner = inner

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(inner)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)


class AppWindow(QtWidgets.QWidget):
    def __init__(self, controller, registry, viewer: MeshViewer):
        super().__init__()
        self.controller = controller
        self.registry = registry
        self.viewer = viewer

        self.setWindowTitle("Mesh Processing Tool")
        self.setStyleSheet(
            """
            QWidget { background-color: #202124; color: #e8eaed; font-family: \"Segoe UI\", Arial; font-size: 10pt; }
            QGroupBox { border: 1px solid #3c4043; margin-top: 8px; border-radius: 6px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa0a6; font-weight: 500; }
            QPushButton { background-color: #303134; border: 1px solid #5f6368; border-radius: 6px; padding: 6px 10px; }
            QPushButton:hover { background-color: #3c4043; }
            QDoubleSpinBox, QLineEdit, QListWidget, QComboBox { background-color: #303134; border: 1px solid #5f6368; border-radius: 4px; padding: 2px 4px; }
            QScrollBar:vertical { width: 16px; background: #202124; }
            QScrollBar::handle:vertical { background: #5f6368; min-height: 30px; border-radius: 6px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            """
        )

        self.registry.init_all(self, controller)

        outer_layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        outer_layout.addWidget(splitter)

        top_container = QtWidgets.QWidget()
        top_bar = QtWidgets.QHBoxLayout(top_container)
        top_bar.setContentsMargins(5, 5, 5, 5)
        splitter.addWidget(top_container)

        main_container = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(main_container)
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(8)
        main_splitter.setOpaqueResize(True)
        main_splitter.addWidget(viewer)
        splitter.addWidget(main_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([120, 900])

        self.feature_widgets = {}
        self.panel_wrappers = {}
        self.left_panel_wrappers = {}

        panel_manager_widget = None
        ordered_features = []
        for feature in self.registry.features:
            if feature.name == "panel_manager":
                panel_manager_widget = feature.widget()
                if panel_manager_widget is not None:
                    self.feature_widgets[feature.name] = panel_manager_widget
            else:
                ordered_features.append(feature)

        if panel_manager_widget is not None:
            panel_manager_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            top_bar.addWidget(panel_manager_widget, stretch=1)

        for feature in ordered_features:
            widget = feature.widget()
            if widget is None:
                continue
            widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            wrapper = ResizablePanel(widget)
            wrapper.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
            self.feature_widgets[feature.name] = widget
            self.panel_wrappers[feature.name] = wrapper

        left_container = QtWidgets.QWidget()
        self.left_layout = QtWidgets.QVBoxLayout(left_container)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(6)

        right_container = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout(right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(6)

        left_names = {"stats", "search", "transform", "shortcuts"}
        for feature in ordered_features:
            wrapper = self.panel_wrappers.get(feature.name)
            if wrapper is None:
                continue
            if feature.name in left_names:
                self.left_layout.addWidget(wrapper)
                self.left_panel_wrappers[feature.name] = wrapper
            else:
                self.right_layout.addWidget(wrapper)
        self.left_layout.addStretch()
        self.right_layout.addStretch()

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        left_scroll.setWidget(left_container)

        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        right_scroll.setWidget(right_container)

        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(right_scroll)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 1)
        main_layout.addWidget(main_splitter)

        for feature in self.registry.features:
            if hasattr(feature, "refresh_panels"):
                feature.refresh_panels()

        for name, wrapper in self.panel_wrappers.items():
            wrapper.adjustSize()

        self.viewer.installEventFilter(self)

    def set_feature_visible(self, feature_name: str, visible: bool) -> None:
        wrapper = self.panel_wrappers.get(feature_name)
        if wrapper is None:
            return
        wrapper.setVisible(visible)

    def reorder_feature(self, feature_name: str, new_index: int) -> None:
        wrapper = self.panel_wrappers.get(feature_name)
        if wrapper is None:
            return
        if feature_name in self.left_panel_wrappers:
            layout = self.left_layout
        else:
            layout = self.right_layout
        current_index = layout.indexOf(wrapper)
        if current_index < 0:
            return
        count = layout.count() - 1
        new_index = max(0, min(new_index, count - 1))
        layout.removeWidget(wrapper)
        layout.insertWidget(new_index, wrapper)

    def feature_order(self):
        ordered = []
        for layout in (self.left_layout, self.right_layout):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                wrapper = item.widget() if item is not None else None
                if wrapper is None:
                    continue
                for name, w in self.panel_wrappers.items():
                    if w is wrapper:
                        ordered.append(name)
                        break
        return ordered

    def eventFilter(self, obj, event):
        if obj is self.viewer and event.type() == QtCore.QEvent.Wheel:
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                delta = event.angleDelta().y() / 120.0
                state = self.controller.transform_state
                max_scale = getattr(self.controller.config, "max_scale", 10.0)
                new_scale = max(0.05, min(max_scale, state.scale + delta * 0.06))
                state.scale = new_scale
                self.controller.set_transform(state)
                return True
        return super().eventFilter(obj, event)
