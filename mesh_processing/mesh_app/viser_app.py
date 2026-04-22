from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

_LOCAL_VISER_SRC = Path(__file__).resolve().parents[3] / "viser" / "src"
if _LOCAL_VISER_SRC.exists():
    sys.path.insert(0, str(_LOCAL_VISER_SRC))

import viser

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mesh_app.config import AppConfig
    from mesh_app.controller import MeshController
    from mesh_app.viser_viewer import ViserMeshViewer
else:
    from .config import AppConfig
    from .controller import MeshController
    from .viser_viewer import ViserMeshViewer


class ViserMeshApp:
    def __init__(self, server: viser.ViserServer, controller: MeshController):
        self.server = server
        self.controller = controller
        self._syncing = False
        self._history_versions: List[str] = []
        self._current_entry_name: Optional[str] = None
        self._ops_object_names: List[str] = []
        self._ops_checkboxes: List[Tuple[str, object]] = []
        self._hand_joint_handles: List[Tuple[str, object]] = []
        self._advance_after_save = False
        self._syncing_hand_target = False
        self._held_dilation_axis: Optional[str] = None
        self._held_category_mode = False
        self._held_variant_mode = False
        self._held_variant_shift = False
        self._variant_navigation_used = False
        self.hand_target = self.server.scene.add_transform_controls(
            "/mesh_processing/hand_target",
            scale=0.18,
            line_width=3.0,
            fixed=False,
            disable_sliders=False,
            disable_rotations=False,
            depth_test=False,
            opacity=0.9,
            visible=False,
        )
        if getattr(self.controller.config, "leap_hand_path", None):
            self.controller.load_hand(self.controller.config.leap_hand_path)
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="large",
            camera_keyboard_shortcuts=False,
            brand_color=(136, 171, 218),
        )
        self._build_gui()
        self._bind_shortcuts()
        if self.controller.entries:
            self.controller.load_entry(self.controller.current_entry())
        self._sync_from_controller()

    def _build_gui(self) -> None:
        tabs = self.server.gui.add_tab_group()
        category_options = self.controller.get_category_names()
        with tabs.add_tab("Main"):
            with self.server.gui.add_folder("Stats"):
                self.summary_md = self.server.gui.add_markdown("Mesh processing viewer")
                self.stats_md = self.server.gui.add_markdown("No stats")

            with self.server.gui.add_folder("Navigate"):
                self.query_text = self.server.gui.add_text("Query", "")
                self.go_btn = self.server.gui.add_button("Go")
                self.prev_btn = self.server.gui.add_button("Prev")
                self.next_btn = self.server.gui.add_button("Next")

            with self.server.gui.add_folder("Transform"):
                self.active_shell = self.server.gui.add_dropdown("Active shell", ["min", "max"], initial_value="min")
                max_scale = float(getattr(self.controller.config, "max_scale", 10.0))
                self.scale_min = self.server.gui.add_number("Uniform min", 1.0, min=0.05, max=max_scale, step=0.01)
                self.scale_max = self.server.gui.add_number("Uniform max", 1.0, min=0.05, max=max_scale, step=0.01)
                self.dilate_x_min = self.server.gui.add_number("Dilate X min", 1.0, min=0.05, max=max_scale, step=0.01)
                self.dilate_x_max = self.server.gui.add_number("Dilate X max", 1.0, min=0.05, max=max_scale, step=0.01)
                self.dilate_y_min = self.server.gui.add_number("Dilate Y min", 1.0, min=0.05, max=max_scale, step=0.01)
                self.dilate_y_max = self.server.gui.add_number("Dilate Y max", 1.0, min=0.05, max=max_scale, step=0.01)
                self.dilate_z_min = self.server.gui.add_number("Dilate Z min", 1.0, min=0.05, max=max_scale, step=0.01)
                self.dilate_z_max = self.server.gui.add_number("Dilate Z max", 1.0, min=0.05, max=max_scale, step=0.01)
                self.rot_x_btn = self.server.gui.add_button("Rotate X +90")
                self.rot_y_btn = self.server.gui.add_button("Rotate Y +90")
                self.rot_z_btn = self.server.gui.add_button("Rotate Z +90")

            with self.server.gui.add_folder("Category"):
                self.edit_category = self.server.gui.add_dropdown("Edit category", category_options, initial_value=category_options[0])
                self.variant = self.server.gui.add_dropdown("Variant", ["v000"], initial_value="v000")
                self.new_variant_btn = self.server.gui.add_button("New variant")
                self.clone_variant_btn = self.server.gui.add_button("Clone variant")
                self.confidence = self.server.gui.add_dropdown(
                    "Confidence",
                    ["confident", "not_confident"],
                    initial_value="confident",
                )
                self.new_category_text = self.server.gui.add_text("New category", "")
                self.create_category_btn = self.server.gui.add_button("Create category")
                self.advance_after_save = self.server.gui.add_checkbox("Next after save", initial_value=False)
                self.save_btn = self.server.gui.add_button("Save selected", color="green")
                self.remove_btn = self.server.gui.add_button("Remove selected", color="red")
                self.remove_all_btn = self.server.gui.add_button("Remove all")

            with self.server.gui.add_folder("Reference"):
                self.ref_min_x = self.server.gui.add_number("Min X dim", 0.06, min=0.001, max=1.0, step=0.001)
                self.ref_min_y = self.server.gui.add_number("Min Y dim", 0.06, min=0.001, max=1.0, step=0.001)
                self.ref_min_height = self.server.gui.add_number("Min height", 0.10, min=0.001, max=2.0, step=0.001)
                self.ref_max_x = self.server.gui.add_number("Max X dim", 0.06, min=0.001, max=1.0, step=0.001)
                self.ref_max_y = self.server.gui.add_number("Max Y dim", 0.06, min=0.001, max=1.0, step=0.001)
                self.ref_max_height = self.server.gui.add_number("Max height", 0.10, min=0.001, max=2.0, step=0.001)
                self.ref_save_btn = self.server.gui.add_button("Save category reference config")
                self.ref_reset_btn = self.server.gui.add_button("Reset cylinder defaults")

            with self.server.gui.add_folder("History"):
                history_options = ["all"] + category_options
                self.history_category = self.server.gui.add_dropdown("Category", history_options, initial_value="all")
                self.history_version = self.server.gui.add_dropdown("Version", ["(none)"], initial_value="(none)")
                self.refresh_history_btn = self.server.gui.add_button("Refresh")
                self.load_history_btn = self.server.gui.add_button("Load selected")
                self.history_md = self.server.gui.add_markdown("")

        with tabs.add_tab("Viewer"):
            with self.server.gui.add_folder("Viewer"):
                self.adjust_view_btn = self.server.gui.add_button("Adjust viewer")
                self.display_mode = self.server.gui.add_dropdown(
                    "View mode",
                    [
                        "Source mesh",
                        "Saved baked mesh",
                        "Edit from saved mesh",
                    ],
                    initial_value="Source mesh",
                )
                self.show_com = self.server.gui.add_checkbox("Show COM", initial_value=True)
                self.show_bbox = self.server.gui.add_checkbox("Show BBox", initial_value=False)
                self.show_bbox_dims = self.server.gui.add_checkbox("Show BBox dims", initial_value=False)
                self.heavy_stats = self.server.gui.add_checkbox("Compute heavy stats", initial_value=False)
                self.show_hull = self.server.gui.add_checkbox("Show convex hull", initial_value=False)
                self.preview_fix = self.server.gui.add_checkbox("Preview watertight fix", initial_value=False)

        with tabs.add_tab("Shortcuts"):
            self.shortcuts_md = self.server.gui.add_markdown(
                "\n".join(
                    [
                        "**Keyboard shortcuts**",
                        "",
                        "- `A`: rotate X +90",
                        "- `S`: rotate Y +90",
                        "- `D`: rotate Z +90",
                        "- `Q`: previous category",
                        "- `E`: next category",
                        "- `J`: previous object",
                        "- `K`: next object",
                        "- `Hold C + Left/Right`: previous/next category",
                        "- `Tap V`: new blank variant",
                        "- `Tap Shift + V`: clone current into new variant",
                        "- `Hold V + Left/Right`: previous/next variant",
                        "- `Tab`: switch active shell",
                        "- `Shift + Scroll`: uniform shell dilation",
                        "- `Shift + Z + Scroll`: X dilation",
                        "- `Shift + X + Scroll`: Y dilation",
                        "- `Shift + C + Scroll`: Z dilation",
                        "- `Space`: save selected",
                        "- `Delete`: remove selected",
                    ]
                )
            )

        with tabs.add_tab("Operations"):
            with self.server.gui.add_folder("Selection"):
                self.ops_category = self.server.gui.add_dropdown(
                    "Saved category",
                    category_options,
                    initial_value=category_options[0],
                )
                self.ops_idx_text = self.server.gui.add_text("Indices", "")
                self.ops_refresh_btn = self.server.gui.add_button("Refresh saved list")
                self.ops_select_idx_btn = self.server.gui.add_button("Select from indices")
                self.ops_select_all_btn = self.server.gui.add_button("Select all")
                self.ops_clear_btn = self.server.gui.add_button("Clear selection")
                self.ops_selection_md = self.server.gui.add_markdown("No saved objects loaded")
                self.ops_checkbox_folder = self.server.gui.add_folder("Object names", expand_by_default=False)

            with self.server.gui.add_folder("Scale"):
                self.ops_scale_factor = self.server.gui.add_number("Scale factor", 1.0, min=0.01, max=100.0, step=0.01)
                self.ops_scale_btn = self.server.gui.add_button("Scale selected objects")

            with self.server.gui.add_folder("Mass"):
                self.ops_mass_value = self.server.gui.add_number("Mass value", 0.5, min=0.0001, max=1000.0, step=0.01)
                self.ops_mass_btn = self.server.gui.add_button("Set mass for selected")

            with self.server.gui.add_folder("Mapping"):
                self.ops_mapping_btn = self.server.gui.add_button("Generate mapping for category dir")
                self.ops_mapping_md = self.server.gui.add_markdown("")

            self.ops_result_md = self.server.gui.add_markdown("")

        with tabs.add_tab("LEAP"):
            with self.server.gui.add_folder("Hand"):
                self.show_hand = self.server.gui.add_checkbox("Show LEAP hand", initial_value=False)
                self.hand_x = self.server.gui.add_number("EEF X", 0.0, min=-1.0, max=1.0, step=0.005)
                self.hand_y = self.server.gui.add_number("EEF Y", 0.0, min=-1.0, max=1.0, step=0.005)
                self.hand_z = self.server.gui.add_number("EEF Z", 0.0, min=-1.0, max=1.0, step=0.005)
                self.hand_rx = self.server.gui.add_number("Rot X", 0.0, min=-360.0, max=360.0, step=1.0)
                self.hand_ry = self.server.gui.add_number("Rot Y", 0.0, min=-360.0, max=360.0, step=1.0)
                self.hand_rz = self.server.gui.add_number("Rot Z", 0.0, min=-360.0, max=360.0, step=1.0)
                hand_preset_names = list((self.controller.config.leap_hand_presets or {}).keys()) or ["(none)"]
                self.hand_preset = self.server.gui.add_dropdown("Joint preset", hand_preset_names, initial_value=hand_preset_names[0])
                self.hand_preset_load_btn = self.server.gui.add_button("Load preset")
                self.hand_reset_btn = self.server.gui.add_button("Reset canonical")
                self.hand_print_btn = self.server.gui.add_button("Print joint angles")
                self.hand_status_md = self.server.gui.add_markdown("")
                self.hand_joint_folder = self.server.gui.add_folder("Joint angles", expand_by_default=False)

        self._bind_events()
        self._populate_hand_joint_controls()
        self._refresh_ops_objects()

    def _bind_events(self) -> None:
        @self.hand_target.on_update
        def _(_: object) -> None:
            if self._syncing or self._syncing_hand_target:
                return
            pos = np.asarray(self.hand_target.position, dtype=np.float64)
            rot = self._wxyz_to_euler_deg(self.hand_target.wxyz)
            self._syncing = True
            try:
                self.hand_x.value = float(pos[0])
                self.hand_y.value = float(pos[1])
                self.hand_z.value = float(pos[2])
                self.hand_rx.value = float(rot[0])
                self.hand_ry.value = float(rot[1])
                self.hand_rz.value = float(rot[2])
            finally:
                self._syncing = False
            self._apply_hand_pose()

        @self.go_btn.on_click
        def _(_: object) -> None:
            query = self.query_text.value.strip()
            if not query:
                return
            self.controller.jump_to(query)
            self._sync_from_controller()

        @self.prev_btn.on_click
        def _(_: object) -> None:
            self.controller.prev()
            self._sync_from_controller()

        @self.next_btn.on_click
        def _(_: object) -> None:
            self.controller.next()
            self._sync_from_controller()

        @self.scale_min.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_scale_range(float(self.scale_min.value), float(self.scale_max.value))
            self._sync_from_controller()

        @self.scale_max.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_scale_range(float(self.scale_min.value), float(self.scale_max.value))
            self._sync_from_controller()

        @self.active_shell.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_active_shell(str(self.active_shell.value))
            self._sync_from_controller()

        for handle in (
            self.dilate_x_min,
            self.dilate_x_max,
            self.dilate_y_min,
            self.dilate_y_max,
            self.dilate_z_min,
            self.dilate_z_max,
        ):
            @handle.on_update
            def _(_: object) -> None:
                if self._syncing:
                    return
                self.controller.set_dilation_ranges(
                    float(self.dilate_x_min.value),
                    float(self.dilate_x_max.value),
                    float(self.dilate_y_min.value),
                    float(self.dilate_y_max.value),
                    float(self.dilate_z_min.value),
                    float(self.dilate_z_max.value),
                )
                self._sync_from_controller()

        @self.rot_x_btn.on_click
        def _(_: object) -> None:
            self._rotate("x")

        @self.rot_y_btn.on_click
        def _(_: object) -> None:
            self._rotate("y")

        @self.rot_z_btn.on_click
        def _(_: object) -> None:
            self._rotate("z")

        @self.edit_category.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_edit_category(self.edit_category.value)
            self._sync_from_controller()

        @self.variant.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_selected_variant(self.variant.value)
            self._sync_from_controller()

        @self.new_variant_btn.on_click
        def _(_: object) -> None:
            self.controller.create_next_variant(copy_current=False)
            self._sync_from_controller()

        @self.clone_variant_btn.on_click
        def _(_: object) -> None:
            self.controller.create_next_variant(copy_current=True)
            self._sync_from_controller()

        @self.create_category_btn.on_click
        def _(_: object) -> None:
            if self.controller.create_category(self.new_category_text.value):
                self.controller.refresh_categories()
                self.controller.set_edit_category(self.new_category_text.value.strip())
                self.new_category_text.value = ""
                self._sync_category_options()
                self._refresh_ops_objects()
                self._sync_from_controller()

        @self.confidence.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.confidence = self.confidence.value
            self._sync_from_controller()

        @self.ref_min_x.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_cylinder_spec(
                self.controller.edit_category,
                "min",
                float(self.ref_min_x.value),
                float(self.ref_min_y.value),
                float(self.ref_min_height.value),
            )
            self._sync_from_controller()

        @self.ref_min_y.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_cylinder_spec(
                self.controller.edit_category,
                "min",
                float(self.ref_min_x.value),
                float(self.ref_min_y.value),
                float(self.ref_min_height.value),
            )
            self._sync_from_controller()

        @self.ref_min_height.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_cylinder_spec(
                self.controller.edit_category,
                "min",
                float(self.ref_min_x.value),
                float(self.ref_min_y.value),
                float(self.ref_min_height.value),
            )
            self._sync_from_controller()

        @self.ref_max_x.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_cylinder_spec(
                self.controller.edit_category,
                "max",
                float(self.ref_max_x.value),
                float(self.ref_max_y.value),
                float(self.ref_max_height.value),
            )
            self._sync_from_controller()

        @self.ref_max_y.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_cylinder_spec(
                self.controller.edit_category,
                "max",
                float(self.ref_max_x.value),
                float(self.ref_max_y.value),
                float(self.ref_max_height.value),
            )
            self._sync_from_controller()

        @self.ref_max_height.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.set_cylinder_spec(
                self.controller.edit_category,
                "max",
                float(self.ref_max_x.value),
                float(self.ref_max_y.value),
                float(self.ref_max_height.value),
            )
            self._sync_from_controller()

        @self.ref_reset_btn.on_click
        def _(_: object) -> None:
            self.controller.reset_cylinder_spec(self.controller.edit_category)
            self._sync_from_controller()

        @self.ref_save_btn.on_click
        def _(_: object) -> None:
            self.controller.save_category_reference_config(self.controller.edit_category)
            self._sync_from_controller()

        @self.save_btn.on_click
        def _(_: object) -> None:
            self._save_selected()

        @self.remove_btn.on_click
        def _(_: object) -> None:
            category = self.controller.edit_category
            self.controller.deactivate_category(category)
            self.controller.refresh_view()
            self._sync_from_controller()

        @self.remove_all_btn.on_click
        def _(_: object) -> None:
            self.controller.deactivate_all()
            self.controller.refresh_view()
            self._sync_from_controller()

        @self.advance_after_save.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self._advance_after_save = bool(self.advance_after_save.value)

        for handle in (
            self.display_mode,
            self.show_com,
            self.show_bbox,
            self.show_bbox_dims,
            self.heavy_stats,
            self.show_hull,
            self.preview_fix,
        ):
            @handle.on_update
            def _(_: object) -> None:
                if self._syncing:
                    return
                self._apply_viewer_flags()

        @self.show_hand.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self.controller.show_hand = bool(self.show_hand.value)
            self._refresh_hand_view(force_scene=True)

        for handle in (
            self.hand_x,
            self.hand_y,
            self.hand_z,
            self.hand_rx,
            self.hand_ry,
            self.hand_rz,
        ):
            @handle.on_update
            def _(_: object) -> None:
                if self._syncing:
                    return
                self._apply_hand_pose()

        @self.hand_reset_btn.on_click
        def _(_: object) -> None:
            self.controller.reset_hand_canonical()
            self.controller.show_hand = bool(self.show_hand.value)
            self._refresh_hand_view(force_scene=not bool(self.controller.show_hand))
            self._sync_from_controller()

        @self.hand_preset_load_btn.on_click
        def _(_: object) -> None:
            presets = self.controller.config.leap_hand_presets or {}
            values = presets.get(str(self.hand_preset.value))
            if not values:
                self.hand_status_md.content = "No preset selected."
                return
            self.controller.set_hand_joint_angles(list(values))
            self.controller.show_hand = bool(self.show_hand.value)
            self._refresh_hand_view(force_scene=not bool(self.controller.show_hand))
            self._sync_hand_panel()

        @self.hand_print_btn.on_click
        def _(_: object) -> None:
            report = self.controller.hand_joint_report()
            print(report)
            self.hand_status_md.content = f"```text\n{report}\n```"

        @self.history_category.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self._refresh_history_versions()

        @self.refresh_history_btn.on_click
        def _(_: object) -> None:
            self._refresh_history_versions()

        @self.load_history_btn.on_click
        def _(_: object) -> None:
            value = self.history_version.value
            if not value or value == "(none)":
                return
            category, version = value.split(" / ", 1)
            self.controller.load_version(category, version)
            self._sync_from_controller()

        @self.adjust_view_btn.on_click
        def _(_: object) -> None:
            self._adjust_view()

        @self.ops_category.on_update
        def _(_: object) -> None:
            if self._syncing:
                return
            self._refresh_ops_objects()

        @self.ops_refresh_btn.on_click
        def _(_: object) -> None:
            self._refresh_ops_objects()

        @self.ops_select_idx_btn.on_click
        def _(_: object) -> None:
            self._select_ops_by_indices()

        @self.ops_select_all_btn.on_click
        def _(_: object) -> None:
            self._set_all_ops_checkboxes(True)

        @self.ops_clear_btn.on_click
        def _(_: object) -> None:
            self._set_all_ops_checkboxes(False)

        @self.ops_scale_btn.on_click
        def _(_: object) -> None:
            selected = self._selected_ops_names()
            if not selected:
                self.ops_result_md.content = "No selected objects."
                return
            results = self.controller.bulk_scale_saved(
                self.ops_category.value,
                selected,
                float(self.ops_scale_factor.value),
            )
            self.ops_result_md.content = self._format_ops_results(results)

        @self.ops_mass_btn.on_click
        def _(_: object) -> None:
            selected = self._selected_ops_names()
            if not selected:
                self.ops_result_md.content = "No selected objects."
                return
            results = self.controller.bulk_set_mass(
                self.ops_category.value,
                selected,
                float(self.ops_mass_value.value),
            )
            self.ops_result_md.content = self._format_ops_results(results)

        @self.ops_mapping_btn.on_click
        def _(_: object) -> None:
            ok, msg = self.controller.generate_saved_mapping(self.ops_category.value)
            if ok:
                self.ops_mapping_md.content = f"Generated: `{msg}`"
            else:
                self.ops_mapping_md.content = f"Failed: `{msg}`"

    def _bind_shortcuts(self) -> None:
        @self.server.on_key_event("keydown")
        def _(event: viser.KeyEvent) -> None:
            key = event.key.lower()
            if key == "v":
                self._held_variant_mode = True
                self._held_variant_shift = bool(event.shift)
                self._variant_navigation_used = False
                return
            if key == "z":
                self._held_dilation_axis = "x"
                return
            if key == "x":
                self._held_dilation_axis = "y"
                return
            if key == "c":
                self._held_category_mode = True
                self._held_dilation_axis = "z"
                return
            if event.repeat:
                return
            if key == "tab":
                self.controller.toggle_active_shell()
                self._sync_from_controller()
                return
            if key == "a":
                self._rotate("x")
            elif key == "s":
                self._rotate("y")
            elif key == "d":
                self._rotate("z")
            elif key == "q":
                self._step_category(-1)
            elif key == "e":
                self._step_category(1)
            elif key == "j":
                self.controller.prev()
                self._sync_from_controller()
            elif key == "k":
                self.controller.next()
                self._sync_from_controller()
            elif key == "arrowleft":
                if self._held_variant_mode:
                    self._step_variant(-1)
                    self._variant_navigation_used = True
                elif self._held_category_mode:
                    self._step_category(-1)
            elif key == "arrowright":
                if self._held_variant_mode:
                    self._step_variant(1)
                    self._variant_navigation_used = True
                elif self._held_category_mode:
                    self._step_category(1)
            elif key == " ":
                self._save_selected()
            elif key == "delete":
                category = self.controller.edit_category
                self.controller.deactivate_category(category)
                self.controller.refresh_view()
                self._sync_from_controller()

        @self.server.on_key_event("keyup")
        def _(event: viser.KeyEvent) -> None:
            key = event.key.lower()
            if key == "v":
                if not self._variant_navigation_used:
                    self.controller.create_next_variant(copy_current=self._held_variant_shift)
                    self._sync_from_controller()
                self._held_variant_mode = False
                self._held_variant_shift = False
                self._variant_navigation_used = False
                return
            if key == "z" and self._held_dilation_axis == "x":
                self._held_dilation_axis = None
            elif key == "x" and self._held_dilation_axis == "y":
                self._held_dilation_axis = None
            elif key == "c" and self._held_dilation_axis == "z":
                self._held_category_mode = False
                self._held_dilation_axis = None

        @self.server.on_wheel_event()
        def _(event: viser.WheelEvent) -> None:
            if not event.shift:
                return
            step = 0.02 if event.delta_y < 0 else -0.02
            if self._held_dilation_axis is None:
                self.controller.nudge_active_shell_scale(step)
            else:
                self.controller.nudge_axis_dilation_ranges(self._held_dilation_axis, step)
            self._sync_transform_panel()

    def _rotate(self, axis: str) -> None:
        state = self.controller.transform_state
        if axis == "x":
            state.rot_x = (state.rot_x + 90.0) % 360.0
        elif axis == "y":
            state.rot_y = (state.rot_y + 90.0) % 360.0
        elif axis == "z":
            state.rot_z = (state.rot_z + 90.0) % 360.0
        self.controller.set_transform(state)
        self._sync_transform_panel()

    def _step_category(self, delta: int) -> None:
        categories = self.controller.get_category_names()
        if not categories:
            return
        try:
            current_idx = categories.index(self.controller.edit_category)
        except ValueError:
            current_idx = 0
        new_idx = (current_idx + delta) % len(categories)
        self.controller.set_edit_category(categories[new_idx])
        self._sync_from_controller()

    def _step_variant(self, delta: int) -> None:
        variants = self.controller.get_variant_names()
        if not variants:
            return
        try:
            current_idx = variants.index(self.controller.selected_variant)
        except ValueError:
            current_idx = 0
        new_idx = (current_idx + delta) % len(variants)
        self.controller.set_selected_variant(variants[new_idx])
        self._sync_from_controller()

    def _sync_category_options(self) -> None:
        category_options = self.controller.refresh_categories()
        self.edit_category.options = category_options
        if self.edit_category.value not in category_options and category_options:
            self.edit_category.value = category_options[0]
        variant_options = self.controller.get_variant_names()
        self.variant.options = variant_options
        if self.variant.value not in variant_options and variant_options:
            self.variant.value = variant_options[0]
        history_options = ["all"] + category_options
        self.history_category.options = history_options
        if self.history_category.value not in history_options:
            self.history_category.value = "all"
        self.ops_category.options = category_options
        if self.ops_category.value not in category_options and category_options:
            self.ops_category.value = category_options[0]

    def _adjust_view(self) -> None:
        adjust_view = getattr(self.controller.viewer, "adjust_view", None)
        if callable(adjust_view):
            adjust_view()

    def _populate_hand_joint_controls(self) -> None:
        for _, handle in self._hand_joint_handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._hand_joint_handles = []
        with self.hand_joint_folder:
            if not self.controller.hand_joint_order:
                self.hand_status_md.content = "LEAP hand not loaded."
                return
            self.hand_status_md.content = f"Loaded `{len(self.controller.hand_joint_order)}` LEAP joints."
            for idx, name in enumerate(self.controller.hand_joint_order):
                lower, upper = self.controller.hand_joint_limits.get(name, (None, None))
                min_value = -3.14 if lower is None else float(lower)
                max_value = 3.14 if upper is None else float(upper)
                initial = 0.0
                if idx < len(self.controller.hand_joint_angles):
                    initial = float(self.controller.hand_joint_angles[idx])
                slider = self.server.gui.add_slider(
                    name,
                    min=min_value,
                    max=max_value,
                    step=0.01,
                    initial_value=initial,
                )
                self._hand_joint_handles.append((name, slider))

                @slider.on_update
                def _(_: object, joint_idx=idx) -> None:
                    if self._syncing:
                        return
                    angles = list(self.controller.hand_joint_angles)
                    if joint_idx >= len(angles):
                        return
                    angles[joint_idx] = float(self._hand_joint_handles[joint_idx][1].value)
                    self.controller.set_hand_joint_angles(angles)
                    self.controller.show_hand = bool(self.show_hand.value)
                    self._refresh_hand_view(force_scene=not bool(self.controller.show_hand))
                    self._sync_hand_panel()

    def _apply_hand_pose(self) -> None:
        self.controller.set_hand_pose(
            float(self.hand_x.value),
            float(self.hand_y.value),
            float(self.hand_z.value),
            float(self.hand_rx.value),
            float(self.hand_ry.value),
            float(self.hand_rz.value),
        )
        self.controller.show_hand = bool(self.show_hand.value)
        self._refresh_hand_view(force_scene=not bool(self.controller.show_hand))

    def _refresh_hand_view(self, force_scene: bool = False) -> None:
        self._set_hand_target_from_controller()
        update_hand = getattr(self.controller.viewer, "update_hand", None)
        if not force_scene and self.controller.show_hand and callable(update_hand):
            update_hand(
                self.controller.hand_mesh,
                hand_z=self.controller.hand_z,
                hand_xy=(self.controller.hand_x, self.controller.hand_y),
            )
        else:
            self.controller.refresh_view()

    def _sync_hand_panel(self) -> None:
        self._syncing = True
        try:
            self.show_hand.value = bool(self.controller.show_hand)
            self.hand_x.value = float(self.controller.hand_x)
            self.hand_y.value = float(self.controller.hand_y)
            self.hand_z.value = float(self.controller.hand_z)
            self.hand_rx.value = float(self.controller.hand_rot_deg[0])
            self.hand_ry.value = float(self.controller.hand_rot_deg[1])
            self.hand_rz.value = float(self.controller.hand_rot_deg[2])
            self._set_hand_target_from_controller()
            for idx, (_, handle) in enumerate(self._hand_joint_handles):
                if idx < len(self.controller.hand_joint_angles):
                    handle.value = float(self.controller.hand_joint_angles[idx])
            self.summary_md.content = self._summary_markdown()
            self.stats_md.content = self._stats_markdown()
        finally:
            self._syncing = False

    def _euler_deg_to_wxyz(self, rot_x: float, rot_y: float, rot_z: float) -> tuple[float, float, float, float]:
        quat_xyzw = R.from_euler("xyz", [rot_x, rot_y, rot_z], degrees=True).as_quat()
        return (
            float(quat_xyzw[3]),
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
        )

    def _wxyz_to_euler_deg(self, wxyz: object) -> tuple[float, float, float]:
        quat = np.asarray(wxyz, dtype=np.float64).reshape(4)
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
        euler = R.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
        return float(euler[0]), float(euler[1]), float(euler[2])

    def _set_hand_target_from_controller(self) -> None:
        self._syncing_hand_target = True
        try:
            self.hand_target.visible = bool(self.controller.show_hand)
            self.hand_target.position = (
                float(self.controller.hand_x),
                float(self.controller.hand_y),
                float(self.controller.hand_z),
            )
            self.hand_target.wxyz = self._euler_deg_to_wxyz(
                float(self.controller.hand_rot_deg[0]),
                float(self.controller.hand_rot_deg[1]),
                float(self.controller.hand_rot_deg[2]),
            )
        finally:
            self._syncing_hand_target = False

    def _display_mode_to_controller(self, value: str) -> str:
        mapping = {
            "Source mesh": self.controller.VIEW_SOURCE,
            "Saved baked mesh": self.controller.VIEW_SAVED_OR_SOURCE,
            "Edit from saved mesh": self.controller.VIEW_SAVED_OR_CURRENT,
        }
        return mapping.get(value, self.controller.VIEW_SOURCE)

    def _controller_to_display_mode(self, value: str) -> str:
        mapping = {
            self.controller.VIEW_SOURCE: "Source mesh",
            self.controller.VIEW_SAVED_OR_SOURCE: "Saved baked mesh",
            self.controller.VIEW_SAVED_OR_CURRENT: "Edit from saved mesh",
        }
        return mapping.get(value, "Source mesh")

    def _save_selected(self) -> None:
        category = self.controller.edit_category
        self.controller.save_current([category], self.controller.confidence)
        self.controller.clear_transform()
        if self._advance_after_save:
            self.controller.next()
        self._sync_from_controller()

    def _refresh_ops_objects(self) -> None:
        for _, handle in self._ops_checkboxes:
            try:
                handle.remove()
            except Exception:
                pass
        self._ops_checkboxes = []
        self._ops_object_names = self.controller.list_saved_object_names(self.ops_category.value)
        with self.ops_checkbox_folder:
            for idx, name in enumerate(self._ops_object_names, start=1):
                checkbox = self.server.gui.add_checkbox(f"{idx}. {name}", False)
                self._ops_checkboxes.append((name, checkbox))
                @checkbox.on_update
                def _(_: object) -> None:
                    self._update_ops_selection_md()
        self._update_ops_selection_md()

    def _parse_indices(self, raw: str) -> Set[int]:
        selected: Set[int] = set()
        for token in re.split(r"[,\s]+", raw.strip()):
            if not token:
                continue
            if "-" in token:
                parts = token.split("-", 1)
                if len(parts) != 2:
                    continue
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                except ValueError:
                    continue
                low, high = sorted((start, end))
                selected.update(range(low, high + 1))
            else:
                try:
                    selected.add(int(token))
                except ValueError:
                    continue
        return selected

    def _select_ops_by_indices(self) -> None:
        selected_indices = self._parse_indices(self.ops_idx_text.value)
        for idx, (_, checkbox) in enumerate(self._ops_checkboxes, start=1):
            checkbox.value = idx in selected_indices
        self._update_ops_selection_md()

    def _set_all_ops_checkboxes(self, value: bool) -> None:
        for _, checkbox in self._ops_checkboxes:
            checkbox.value = value
        self._update_ops_selection_md()

    def _selected_ops_names(self) -> List[str]:
        return [name for name, checkbox in self._ops_checkboxes if bool(checkbox.value)]

    def _update_ops_selection_md(self) -> None:
        selected = self._selected_ops_names()
        total = len(self._ops_object_names)
        if total == 0:
            self.ops_selection_md.content = "No saved objects found for this category."
            return
        preview = selected[:12]
        preview_md = "\n".join(f"- `{name}`" for name in preview)
        if len(selected) > 12:
            preview_md += f"\n- ... and {len(selected) - 12} more"
        self.ops_selection_md.content = (
            f"Loaded `{total}` saved objects. Selected `{len(selected)}`.\n\n{preview_md or 'No objects selected.'}"
        )

    def _format_ops_results(self, results: Dict[str, str]) -> str:
        if not results:
            return "No results."
        lines = [f"- `{name}`: {status}" for name, status in results.items()]
        return "\n".join(lines[:30]) + (f"\n- ... and {len(lines) - 30} more" if len(lines) > 30 else "")

    def _apply_viewer_flags(self) -> None:
        self.controller.display_mesh_mode = self._display_mode_to_controller(
            self.display_mode.value
        )
        self.controller.show_grid_labels = False
        self.controller.show_com = bool(self.show_com.value)
        self.controller.show_bbox = bool(self.show_bbox.value)
        self.controller.show_bbox_dims = bool(self.show_bbox_dims.value)
        self.controller.auto_scale_unsaved = False
        self.controller.use_mesh_analysis = bool(self.heavy_stats.value)
        self.controller.show_convex_hull = bool(self.show_hull.value)
        self.controller.fix_watertight_preview = bool(self.preview_fix.value)
        self.controller.refresh_view()
        self._sync_from_controller()

    def _refresh_history_versions(self) -> None:
        entry = self.controller.current_entry()
        versions = []
        if entry is not None:
            selected = self.history_category.value
            categories = [selected] if selected != "all" else self.controller.get_category_names()
            for category in categories:
                for version in reversed(self.controller.history_repo.list_versions(entry, category)):
                    versions.append(f"{category} / {version}")
        self._history_versions = versions
        self.history_version.options = versions or ["(none)"]
        self.history_version.value = self.history_version.options[0]
        self.history_md.content = "\n".join([f"- `{item}`" for item in (versions[:12] or ["(none)"])])

    def _stats_markdown(self) -> str:
        obj_stats, ref_stats = self.controller.get_stats()
        if obj_stats is None:
            return "No stats"

        def fmt(title: str, stats: dict) -> str:
            bbox = stats["bbox_extents"]
            volume = stats["volume"]
            max_dim = stats["max_dim"]
            return (
                f"**{title}**\n"
                f"- bbox: `{bbox[0]:.5f}, {bbox[1]:.5f}, {bbox[2]:.5f}`\n"
                f"- volume: `{volume:.8f}`\n"
                f"- max dim: `{max_dim:.5f}`"
            )

        parts = [fmt("Object", obj_stats)]
        info = getattr(self.controller, "mesh_info", {})
        if info:
            analysis = ["**Analysis**"]
            if info.get("watertight") is not None:
                analysis.append(f"- watertight: `{bool(info['watertight'])}`")
            if info.get("hull_volume") is not None:
                analysis.append(f"- hull volume: `{info['hull_volume']:.8f}`")
            if info.get("fixed_preview"):
                analysis.append("- preview fix: `on`")
            parts.append("\n".join(analysis))
        if ref_stats is not None:
            parts.append(fmt("Cylinder min", ref_stats["min"]))
            parts.append(fmt("Cylinder max", ref_stats["max"]))
        return "\n\n".join(parts)

    def _summary_markdown(self) -> str:
        entry = self.controller.current_entry()
        if entry is None:
            return "No entries found."
        idx = self.controller.navigator.index + 1
        total = len(self.controller.navigator.filtered)
        state = self.controller.transform_state
        return (
            f"**Entry** `{idx} / {total}`  \n"
            f"**Name** `{entry.full_name}`  \n"
            f"**Edit category** `{self.controller.edit_category}`  \n"
            f"**Variant** `{self.controller.selected_variant}`  \n"
            f"**Active shell** `{self.controller.active_shell}`  \n"
            f"**Confidence** `{self.controller.confidence}`  \n"
            f"**View mode** `{self._controller_to_display_mode(self.controller.display_mesh_mode)}`  \n"
            f"**Uniform range** `min={state.scale_min:.3f} mid={state.scale:.3f} max={state.scale_max:.3f}`  \n"
            f"**Dilation** `x={state.dilate_x:.3f} y={state.dilate_y:.3f} z={state.dilate_z:.3f}`  \n"
            f"**Dilation range** `x=[{state.dilate_x_min:.3f},{state.dilate_x_max:.3f}] y=[{state.dilate_y_min:.3f},{state.dilate_y_max:.3f}] z=[{state.dilate_z_min:.3f},{state.dilate_z_max:.3f}]`  \n"
            f"**Rotation** `rx={state.rot_x:.0f} ry={state.rot_y:.0f} rz={state.rot_z:.0f}`"
        )

    def _sync_from_controller(self) -> None:
        self._syncing = True
        try:
            self._sync_category_options()
            self._sync_transform_panel_locked()
            self.edit_category.value = self.controller.edit_category
            self.variant.value = self.controller.selected_variant
            self.confidence.value = self.controller.confidence
            self.advance_after_save.value = bool(self._advance_after_save)
            self.display_mode.value = self._controller_to_display_mode(self.controller.display_mesh_mode)
            self.show_com.value = bool(self.controller.show_com)
            self.show_bbox.value = bool(self.controller.show_bbox)
            self.show_bbox_dims.value = bool(self.controller.show_bbox_dims)
            self.heavy_stats.value = bool(self.controller.use_mesh_analysis)
            self.show_hull.value = bool(self.controller.show_convex_hull)
            self.preview_fix.value = bool(self.controller.fix_watertight_preview)
            self.show_hand.value = bool(self.controller.show_hand)
            self.hand_x.value = float(self.controller.hand_x)
            self.hand_y.value = float(self.controller.hand_y)
            self.hand_z.value = float(self.controller.hand_z)
            self.hand_rx.value = float(self.controller.hand_rot_deg[0])
            self.hand_ry.value = float(self.controller.hand_rot_deg[1])
            self.hand_rz.value = float(self.controller.hand_rot_deg[2])
            self._set_hand_target_from_controller()
            for idx, (_, handle) in enumerate(self._hand_joint_handles):
                if idx < len(self.controller.hand_joint_angles):
                    handle.value = float(self.controller.hand_joint_angles[idx])
            min_spec = self.controller._get_cylinder_spec(self.controller.edit_category, "min")
            max_spec = self.controller._get_cylinder_spec(self.controller.edit_category, "max")
            self.ref_min_x.value = float(min_spec.x_dim)
            self.ref_min_y.value = float(min_spec.y_dim)
            self.ref_min_height.value = float(min_spec.height)
            self.ref_max_x.value = float(max_spec.x_dim)
            self.ref_max_y.value = float(max_spec.y_dim)
            self.ref_max_height.value = float(max_spec.height)
            self.summary_md.content = self._summary_markdown()
            self.stats_md.content = self._stats_markdown()
            self._refresh_history_versions()
            entry = self.controller.current_entry()
            entry_name = entry.full_name if entry is not None else None
            if entry_name != self._current_entry_name:
                self._current_entry_name = entry_name
                self._adjust_view()
        finally:
            self._syncing = False

    def _sync_transform_panel_locked(self) -> None:
        self.active_shell.value = str(self.controller.active_shell)
        self.scale_min.value = float(self.controller.transform_state.scale_min)
        self.scale_max.value = float(self.controller.transform_state.scale_max)
        self.dilate_x_min.value = float(self.controller.transform_state.dilate_x_min)
        self.dilate_x_max.value = float(self.controller.transform_state.dilate_x_max)
        self.dilate_y_min.value = float(self.controller.transform_state.dilate_y_min)
        self.dilate_y_max.value = float(self.controller.transform_state.dilate_y_max)
        self.dilate_z_min.value = float(self.controller.transform_state.dilate_z_min)
        self.dilate_z_max.value = float(self.controller.transform_state.dilate_z_max)
        self.summary_md.content = self._summary_markdown()
        self.stats_md.content = self._stats_markdown()

    def _sync_transform_panel(self) -> None:
        self._syncing = True
        try:
            self._sync_transform_panel_locked()
        finally:
            self._syncing = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max-scale", type=float, default=10.0)
    args = parser.parse_args()

    server = viser.ViserServer(host=args.host, port=args.port)
    config = AppConfig()
    config.max_scale = float(args.max_scale)

    viewer = ViserMeshViewer(server)
    controller = MeshController(config, viewer)
    controller.auto_scale_unsaved = False

    controller.show_hand = False

    app = ViserMeshApp(server, controller)

    print(f"Viser mesh tool running at http://{args.host}:{args.port}")
    print("This is a browser-based MVP port of the Qt mesh processing tool.")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
