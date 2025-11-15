# README — Converting the Panda URDF for Isaac Sim 5.0.0

## 1) Prevent fixed-joint collapsing
Add the attribute `dont_collapse="true"` (some builds accept `dontcollapse="true"`) to the fixed joints you want to keep as separate links:
- `panda_hand_joint`
- `palm_center_joint`
- `thumb_tip`, `index_tip`, `middle_tip`, `ring_tip`

Example:
<joint name="palm_center_joint" type="fixed" dont_collapse="true">

## 2) Import from the Isaac Sim 5.0.0 viewport
- Open the **URDF Importer** (Search/Window → URDF Importer).
- Select your `.urdf`, choose a prim path (e.g., `/World/panda`), and **Import**.
  - (Optional but helpful) Import **in stage** and avoid instanceable assets so visuals are easy to edit.

## 3) Fix visual orientation (Franka links only)
For each `panda_link0` … `panda_link7`:
- Expand to `…/visuals` and add a child **Xform** (e.g., `visuals_rot`) or edit the visuals’ existing Xform.
- Set **RotateXYZ** to **(-90, 0, 0)** on that visuals-only Xform.
- Do **not** rotate the link prim itself—only the visuals—so the attached hand (palm/fingers) doesn’t move.

Done. Save the stage (optionally “Save Flattened As…” to bake references) and you’re ready to use the links (including the tip heads) directly in code.
