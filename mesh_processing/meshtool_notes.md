# MeshTool Notes

## User Instructions / Requirements
- Extend mesh sorting beyond {regular, long, small} into grasp-oriented categories.
- Grasp-oriented categories (from user spec):
  - regular / top-down: large (special/difficult), tall large (envelop), medium (regular), thin (scoop), small (pinch), tiny (2–3 finger pinch)
  - long / side: large (clamp/tea-container-like), medium (regular bottle), small (pinch from below)
  - irregular: rim (edge grasp, bowl-like), handle (handle or side grasp)
- Add a confidence label ("confident" vs "not confident") for each classification; train on confident first, then evaluate not-confident.
- Allow assigning an object to multiple categories at once (with separate output copies per category).
- New tool should allow manually selecting which saved transform to load when multiple exist.
- Transform point clouds alongside meshes; generate new URDFs when needed.
- Estimate COM using inferred Z-center and scaled original (match current tool behavior).
- Preserve good UI options (shortcuts, viewer controls, stats, overlays).
- Add viewer option to load/display the LEAP hand model alongside objects.
- Add search/jump UI to index objects and go directly to a specific one.
- Add version history (keep last ~10 saves per object) with easy revert.

## Meshtool.py Key Functionality Summary
- Scans mesh folders for OBJ variants (and optional NPY/GLB/URDF) into MeshEntry records.
- GUI: PyQt5 app with 3D viewer (pyqtgraph) showing object + reference mesh with grid, overlays, and optional auto-spin.
- View transforms: per-object scale + 90-degree rotations (X/Y/Z); load modes: reset/keep/load-saved; state caching.
- Viewer features: point cloud overlay, COM overlay (geometry + URDF), reference sync with URDF scale.
- Stats: bbox extents, volume, max dimension for object and reference; displayed in UI.
- Save pipeline per category:
  - Applies initial URDF scale to OBJ on load.
  - Applies user scale + rotations to OBJ/GLB/NPY.
  - Heuristics to avoid double-applying initial scale to NPY.
  - Updates or generates URDFs; normalizes mesh scale and updates origins.
  - Writes JSON logs with transforms, stats, and paths.
- Discard pipeline: exports transformed OBJ + copies aux files + JSON log.
- Saved/discarded status inferred by output directory scanning; UI reflects state.
- Undo stack (recorded, not yet exposed in UI).
- Reference mesh selection + scaling controls.
