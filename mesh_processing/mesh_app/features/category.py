from PyQt5 import QtCore, QtWidgets

from .base import FeatureBase


class CategoryFeature(FeatureBase):
    name = "categories"
    title = "Categories"

    def __init__(self, categories):
        self.categories = categories
        self.status = None
        self.recent = None
        self.selected_label = None
        self.visibility_list = None
        self.buttons = {}
        self.group_map = {}
        self.group_keys = []
        self.current_group = None

    def widget(self):
        group = QtWidgets.QGroupBox("Categories")
        layout = QtWidgets.QVBoxLayout()

        selected_row = QtWidgets.QHBoxLayout()
        selected_row.addWidget(QtWidgets.QLabel("Selected:"))
        self.selected_label = QtWidgets.QLabel("(none)")
        selected_row.addWidget(self.selected_label)
        selected_row.addStretch()
        layout.addLayout(selected_row)

        self.recent = QtWidgets.QLabel("")
        self.recent.setWordWrap(True)
        layout.addWidget(self.recent)

        self._build_group_map()
        group_shortcuts = ["Q", "W", "E"]

        all_btn = QtWidgets.QPushButton("all_categories")
        all_btn.setToolTip("all_categories")
        all_btn.clicked.connect(lambda _=False: self._select_category("all_categories"))
        layout.addWidget(all_btn)
        self.buttons["all_categories"] = all_btn

        current_group = None
        for cat in self.categories:
            if current_group != cat.group:
                current_group = cat.group
                group_idx = self.group_keys.index(current_group) if current_group in self.group_keys else -1
                group_key = group_shortcuts[group_idx] if 0 <= group_idx < len(group_shortcuts) else ""
                suffix = f" ({group_key})" if group_key else ""
                lbl = QtWidgets.QLabel(current_group + suffix)
                lbl.setStyleSheet("color: #8ab4f8; font-weight: 600;")
                layout.addWidget(lbl)
            idx = self.group_map[current_group].index(cat) + 1
            btn = QtWidgets.QPushButton(f"{idx}. {cat.label}")
            btn.setToolTip(cat.key)
            btn.clicked.connect(lambda _=False, key=cat.key: self._select_category(key))
            layout.addWidget(btn)
            self.buttons[cat.key] = btn

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)

        action_row = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("Save selected")
        btn_remove = QtWidgets.QPushButton("Remove selected")
        btn_remove_all = QtWidgets.QPushButton("Remove all")
        action_row.addWidget(btn_save)
        action_row.addWidget(btn_remove)
        action_row.addWidget(btn_remove_all)
        layout.addLayout(action_row)

        vis_label = QtWidgets.QLabel("Grid visibility")
        layout.addWidget(vis_label)

        self.visibility_list = QtWidgets.QListWidget()
        for cat in self.categories:
            item = QtWidgets.QListWidgetItem(cat.key)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.visibility_list.addItem(item)
        layout.addWidget(self.visibility_list)

        group.setLayout(layout)

        btn_save.clicked.connect(self._save_selected)
        btn_remove.clicked.connect(self._remove_selected)
        btn_remove_all.clicked.connect(self._remove_all)
        self.visibility_list.itemChanged.connect(self._on_visibility_changed)

        self._bind_shortcuts()
        return group

    def _select_category(self, category_key: str):
        self.controller.set_edit_category(category_key)
        self.selected_label.setText(category_key)
        if hasattr(self.controller.viewer, "set_selected_text"):
            self.controller.viewer.set_selected_text(f"Selected: {category_key}")
        self._update_button_styles(category_key)
        self._update_recent()

    def _save_selected(self):
        category_key = self.controller.edit_category
        confidence = getattr(self.controller, "confidence", "confident")
        if category_key == "all_categories":
            categories = [c.key for c in self.categories]
        else:
            categories = [category_key]
        results = self.controller.save_current(categories, confidence)
        if not results:
            self.status.setText("Save failed.")
            return
        if category_key == "all_categories":
            self.status.setText("Saved: all_categories")
        else:
            version = results[category_key]["version"]
            self.status.setText(f"Saved: {category_key} ({version})")
        self._update_recent()
        self.controller.set_edit_category(category_key)
        self.controller.refresh_view()

    def on_entry_loaded(self, entry):
        if self.selected_label is not None:
            self.selected_label.setText(self.controller.edit_category)
            self._update_button_styles(self.controller.edit_category)
        if hasattr(self.controller.viewer, "set_selected_text"):
            self.controller.viewer.set_selected_text(f"Selected: {self.controller.edit_category}")
        self._update_recent()

    def _update_recent(self):
        entry = self.controller.current_entry()
        if entry is None:
            self.recent.setText("")
            return
        lines = []
        for cat in self.categories:
            versions = self.controller.history_repo.list_versions(entry, cat.key)
            if not versions:
                continue
            active = self.controller.history_repo.get_active_version(entry, cat.key)
            if not active:
                continue
            recent = ", ".join(versions[-3:])
            lines.append(f"{cat.key}: active={active} | recent={recent}")
        self.recent.setText("\n".join(lines))

    def _remove_selected(self):
        category = self.controller.edit_category
        if category == "all_categories":
            self.controller.deactivate_all()
            self.status.setText("Removed from all categories")
            self._update_recent()
            self.controller.set_edit_category(category)
            self.controller.refresh_view()
            return
        self.controller.deactivate_category(category)
        self.status.setText(f"Removed from category: {category}")
        self._update_recent()
        self.controller.set_edit_category(category)
        self.controller.refresh_view()

    def _remove_all(self):
        self.controller.deactivate_all()
        self.status.setText("Removed from all categories")
        self._update_recent()
        self.controller.set_edit_category(self.controller.edit_category)
        self.controller.refresh_view()

    def _on_visibility_changed(self, _item):
        active = []
        for i in range(self.visibility_list.count()):
            item = self.visibility_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                active.append(item.text())
        self.controller.set_visible_categories(active)

    def _update_button_styles(self, selected_key: str):
        for key, btn in self.buttons.items():
            if key == selected_key:
                btn.setStyleSheet("background-color: #1a73e8; border-color: #1a73e8; color: white;")
            else:
                btn.setStyleSheet("")

    def _build_group_map(self):
        self.group_map = {}
        for cat in self.categories:
            self.group_map.setdefault(cat.group, []).append(cat)
        self.group_keys = list(self.group_map.keys())
        if self.group_keys:
            self.current_group = self.group_keys[0]

    def _select_group_by_index(self, idx: int):
        if idx < 0 or idx >= len(self.group_keys):
            return
        self.current_group = self.group_keys[idx]

    def _select_by_number(self, number: int):
        if self.current_group not in self.group_map:
            return
        group_cats = self.group_map[self.current_group]
        if number < 1 or number > len(group_cats):
            return
        cat = group_cats[number - 1]
        self._select_category(cat.key)

    def _bind_shortcuts(self):
        QtWidgets.QShortcut(QtCore.Qt.Key_Q, self.ui, activated=lambda: self._select_group_by_index(0))
        QtWidgets.QShortcut(QtCore.Qt.Key_W, self.ui, activated=lambda: self._select_group_by_index(1))
        QtWidgets.QShortcut(QtCore.Qt.Key_E, self.ui, activated=lambda: self._select_group_by_index(2))
        for i in range(1, 10):
            QtWidgets.QShortcut(getattr(QtCore.Qt, f"Key_{i}"), self.ui, activated=lambda i=i: self._select_by_number(i))
        QtWidgets.QShortcut(QtCore.Qt.Key_S, self.ui, activated=self._save_selected)
        QtWidgets.QShortcut(QtCore.Qt.Key_D, self.ui, activated=self._remove_selected)
        QtWidgets.QShortcut(QtCore.Qt.Key_A, self.ui, activated=self._save_all_shortcut)

    def _save_all_shortcut(self):
        self.controller.set_edit_category("all_categories")
        self.selected_label.setText("all_categories")
        self._update_button_styles("all_categories")
        self._save_selected()
