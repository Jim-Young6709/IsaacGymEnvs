# Mesh Processing App

This folder contains a modular, feature-based mesh processing GUI built on top of the old `meshtool.py` workflow.
It is designed for grasping dataset preparation: transform meshes, view overlays, and save outputs into multiple
categories with confidence labels and version history.

## What This Tool Does
- Load mesh assets (OBJ/GLB/NPY/URDF) from a root directory and visualize them.
- Apply scale + 90-degree rotations and preview results.
- Save outputs into multiple categories at once with a confidence tag.
- Keep version history (default 10) per object/category to support revert.
- Show overlays: pointcloud, COM estimate, reference mesh, and optional LEAP hand.
- Jump to any object by index/name via search.

## Typical Workflow
1. Launch the app.
2. Search or step through objects.
3. Adjust scale/rotation to match expected grasp pose.
4. Select one or more categories and a confidence label.
5. Save (this writes a new version per category).
6. Use History to load a previous version if needed.

## Running
```bash
python /home/rayliu/grogu/IsaacGymEnvs/mesh_processing/mesh_app/main.py
```

## Running The Viser Port
```bash
python /home/rayliu/grogu/IsaacGymEnvs/mesh_processing/mesh_app/viser_app.py --port 8081
```

Current `viser` migration scope:
- Reuses the existing `AppConfig`, `MeshController`, and repositories.
- Runs in the browser instead of PyQt5.
- Uses a single-object editing view instead of the multi-category grid.
- Supports navigation, scale-range editing, 90-degree rotations, category selection, save/remove actions, history loading, stats, and basic hand overlay controls.
- Keeps the same output/versioning pipeline.

Current gaps vs Qt app:
- No keyboard shortcuts.
- No file picker or arbitrary reference mesh loading in the browser UI.
- No LEAP joint sliders yet; only show/hide + XYZ offset.
- The `viser` path now uses cylindrical silhouette references instead of mesh references.

## Key Design Ideas
- **Modular features**: Each UI function is a Feature class that can be added/removed without changing the core app.
- **Controller + repositories**: UI is decoupled from file I/O and processing logic.
- **Versioned outputs**: Each save produces a new version folder for easy rollback.

## File Map
- `mesh_app/main.py`: App bootstrap; wires viewer, controller, and features.
- `mesh_app/ui.py`: Main window layout and styling.
- `mesh_app/config.py`: Paths, category specs, and app defaults.
- `mesh_app/models.py`: Data classes (mesh entries, transforms, history records).
- `mesh_app/controller.py`: MeshController + Navigator (core orchestration).
- `mesh_app/repositories.py`: Mesh I/O, transforms, URDF updates, history/versioning.
- `mesh_app/viewer.py`: 3D viewer and render layers.
- `mesh_app/features/`: Individual UI/logic modules.

## Features Directory
- `mesh_app/features/base.py`: Feature interface.
- `mesh_app/features/category.py`: Multi-category selection and save.
- `mesh_app/features/confidence.py`: Confidence toggle.
- `mesh_app/features/history.py`: Version list + manual load.
- `mesh_app/features/search.py`: Search/jump navigation.
- `mesh_app/features/transform.py`: Scale/rotation controls.
- `mesh_app/features/overlay.py`: Pointcloud/COM + reference mesh controls.
- `mesh_app/features/leap_hand.py`: Optional LEAP hand view.
- `mesh_app/features/stats.py`: Basic geometry stats display.

## API Reference (High Level)
See `docs/api.md` for class responsibilities and method contracts.
