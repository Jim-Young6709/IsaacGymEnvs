# Mesh Processing App API Reference

This document summarizes core classes, their roles, and how they interact.
It is intended for engineers extending the tool.

## Config
**AppConfig** (`mesh_app/config.py`)
- Purpose: Central configuration for paths and defaults.
- Key fields: `mesh_root`, `output_root`, `reference_path`, `history_keep`, `categories`.
- `default_categories()` returns the grasp-based category list.

## Models
**MeshEntry** (`mesh_app/models.py`)
- Fields: `label`, `base_name`, `obj_path`, `npy_path`, `glb_path`, `urdf_path`.
- Property: `full_name` for output naming.

**TransformState** (`mesh_app/models.py`)
- Fields: `scale`, `rot_x`, `rot_y`, `rot_z`.
- Methods:
  - `rotation_matrix()` -> 3x3 rotation matrix.
  - `transform_matrix()` -> 4x4 scale+rotation.
  - `apply_to_mesh(mesh)` -> transformed mesh copy.
  - `apply_to_points(points)` -> transformed pointcloud.

**CategoryAssignment** (`mesh_app/models.py`)
- Fields: `categories`, `confidence`.

**HistoryRecord** (`mesh_app/models.py`)
- Fields: `version`, `timestamp`, `transform`, `categories`, `confidence`, `paths`, `stats`.

## Repositories
**MeshRepository** (`mesh_app/repositories.py`)
- `scan(root)` -> list of MeshEntry.
- `load_mesh(entry)` -> mesh loaded with initial URDF scale applied.
- `load_pointcloud(entry)` -> NPY array.
- `get_urdf_initial_scale(urdf_path)` -> scale vector.

**HistoryRepository** (`mesh_app/repositories.py`)
- `list_versions(entry, category)` -> list of version folders.
- `next_version(entry, category)` -> next `vNNN` name.
- `prune_versions(entry, category)` -> keep last N versions.
- `read_metadata(entry, category, version)` -> JSON metadata.

**TransformRepository** (`mesh_app/repositories.py`)
- `save(entry, transform, categories, confidence)` -> writes OBJ/GLB/NPY/URDF + JSON.
- `load_transform(entry, category, version)` -> TransformState.
- `compute_stats(mesh)` -> bbox/volume/max dim.
- `prepare_pointcloud(...)` -> pointcloud with scaling/rotation heuristics.

## Controller + Navigation
**Navigator** (`mesh_app/controller.py`)
- Maintains filtered list + index.
- `next()`, `prev()`, `jump_to_index(i)`, `jump_to_name(query)`.

**MeshController** (`mesh_app/controller.py`)
- Orchestrates repositories, viewer, and UI.
- Key methods:
  - `load_entry(entry)`
  - `refresh_view()`
  - `set_transform(state)`
  - `save_current(categories, confidence)`
  - `load_version(category, version)`
  - `jump_to(query)`, `next()`, `prev()`
  - `set_reference(path, scale)`
  - `set_leap_mesh(path, scale)`
  - `get_stats()`

## Viewer
**MeshViewer** (`mesh_app/viewer.py`)
- Responsible for rendering object/reference/hand/overlays.
- `show_scene(...)` accepts meshes, pointcloud, and COM point.
- `toggle_spin()` animates view.

## Features System
**FeatureBase** (`mesh_app/features/base.py`)
- Base interface: `init(ui, controller)`, `widget()`, `on_entry_loaded()`, `on_transform_changed()`.

**FeatureRegistry** (`mesh_app/features/__init__.py`)
- Holds and initializes feature modules.
- Notifies features of entry/transform changes.

**Built-in Features**
- `SearchFeature`: search + jump + next/prev.
- `TransformFeature`: scale + rotation controls.
- `ConfidenceFeature`: confident/not confident.
- `CategoryFeature`: multi-category selection + save.
- `HistoryFeature`: version list + load.
- `OverlayFeature`: pointcloud/COM and reference selection.
- `LeapHandFeature`: load/show LEAP hand mesh.
- `StatsFeature`: bbox/volume/max dim display.

## Extension Pattern
To add a new capability:
1. Create a new Feature class in `mesh_app/features/`.
2. Implement `widget()` and hook UI events to controller methods.
3. Register the Feature in `mesh_app/main.py`.


## Example Call Flows

### Save Flow (multi-category + confidence)
1. User selects categories + confidence in UI.
2. `CategoryFeature._save()` calls `MeshController.save_current(categories, confidence)`.
3. `TransformRepository.save(...)` writes transformed OBJ/GLB/NPY/URDF into per-category version folders.
4. `HistoryRepository.prune_versions(...)` trims old versions.

### Load Previous Version
1. User selects `category / version` in History.
2. `HistoryFeature._load_selected()` calls `MeshController.load_version(category, version)`.
3. `TransformRepository.load_transform(...)` reads JSON and returns TransformState.
4. `MeshController.set_transform(...)` updates view.

### Search + Jump
1. User enters a query (index or name) and presses Enter.
2. `SearchFeature._go()` calls `MeshController.jump_to(query)`.
3. `Navigator.jump_to_index` or `Navigator.jump_to_name` resolves entry.
4. `MeshController.load_entry(...)` refreshes view and notifies features.

## How to Add a Feature (Step-by-Step)

### Goal: Add a Convex Hull Overlay Toggle

1) Create the feature file:
`mesh_app/features/convex_hull.py`

```python
from PyQt5 import QtWidgets

from .base import FeatureBase


class ConvexHullFeature(FeatureBase):
    name = "convex_hull"

    def widget(self):
        group = QtWidgets.QGroupBox("Convex Hull")
        layout = QtWidgets.QVBoxLayout()
        self.chk = QtWidgets.QCheckBox("Show convex hull")
        layout.addWidget(self.chk)
        group.setLayout(layout)
        self.chk.stateChanged.connect(self._update)
        return group

    def on_entry_loaded(self, entry):
        self._update()

    def on_transform_changed(self, state):
        self._update()

    def _update(self):
        if not self.chk.isChecked():
            self.controller.viewer._clear_item(self.controller.viewer.hand_item)
            return
        entry = self.controller.current_entry()
        if entry is None:
            return
        mesh = self.controller.mesh_repo.load_mesh(entry)
        if mesh is None:
            return
        mesh_scaled = self.controller.transform_state.apply_to_mesh(mesh)
        hull = mesh_scaled.convex_hull
        # Reuse the hand slot for now or add a new layer for hull rendering.
        self.controller.viewer.show_scene(
            obj_mesh=mesh_scaled,
            ref_mesh=self.controller.reference_mesh,
            obj_pointcloud=None,
            com_point=None,
            hand_mesh=hull,
            show_pointcloud=False,
            show_com=False,
        )
```

2) Register the feature in `mesh_app/main.py`:
```python
from .features.convex_hull import ConvexHullFeature

registry.register(ConvexHullFeature())
```

3) (Optional) Create a dedicated render layer in `MeshViewer` for hulls.

### Notes
- The example reuses the hand mesh slot to keep the patch small.
- For production, add a proper `hull_item` to `MeshViewer` and update `show_scene`.
- Features are isolated; you shouldn’t need to modify controller logic.

