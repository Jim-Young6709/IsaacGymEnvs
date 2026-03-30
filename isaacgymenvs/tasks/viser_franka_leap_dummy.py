#!/usr/bin/env python3
# CODEX
"""Quick Viser viewer for Franka+LEAP with joint sliders and hand-angle print button."""

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import viser
from viser.extras import ViserUrdf


# CODEX
# Arm defaults requested by user.
DEFAULT_ARM = [-1.41111064, -1.20421876, 1.11514925, -2.30643184, 0.97677832, 1.59482316, -0.73056353]

# CODEX
# Hand defaults in the same raw convention used by IsaacGym tasks:
# [index(4), thumb(4), middle(4), ring(4)] with the thumb block in
# IsaacGym order [4, 3, 2, 1] when counted from fingertip-first.
DEFAULT_HAND_ISAAC = [
     0.0000,  0.0000,  0.0000,  0.0000,
 0.6900,  1.0100,  0.0000, -0.0300,
 0.0000,  0.0000,  0.0000,  0.0000,
 0.0000,  0.0000,  0.0000,  0.0000,
]

HAND_PRESETS_ISAAC = {
    "default": DEFAULT_HAND_ISAAC,
    "pinch2": [
        1.15, 0.38, 0.23, 0.77,
        0.70, 0.70, -0.20, 0.79,
        0.48, -0.09, -0.20, 0.62,
        0.50, -0.05, 0.02, 0.29,
    ],
    "pinch3": [
        0.93, 0.00, 0.53, 0.77,
      0.89, 0.73, 0.69, 0.77,
      0.69, 0.01, 0.81, 0.85,
     -0.27, -0.07, 0.13, 0.33,
    ],
}

# Hand joint names in the same raw IsaacGym block order.
LEAP_HAND_JOINT_NAMES_ISAAC_ORDER = [
    "finger_joint_1", "finger_joint_0", "finger_joint_2", "finger_joint_3",      # index
    "finger_joint_12", "finger_joint_13", "finger_joint_14", "finger_joint_15",  # thumb
    "finger_joint_5", "finger_joint_4", "finger_joint_6", "finger_joint_7",      # middle
    "finger_joint_9", "finger_joint_8", "finger_joint_10", "finger_joint_11",     # ring
]

ARM_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
]


def build_default_joint_map(joint_names: List[str], hand_preset: str) -> Dict[str, float]:
    # CODEX
    # Initialize all actuated joints to zero, then apply requested defaults.
    joint_map = {name: 0.0 for name in joint_names}

    for name, val in zip(ARM_JOINT_NAMES, DEFAULT_ARM):
        if name in joint_map:
            joint_map[name] = float(val)

    for name, val in zip(LEAP_HAND_JOINT_NAMES_ISAAC_ORDER, HAND_PRESETS_ISAAC[hand_preset]):
        if name in joint_map:
            joint_map[name] = float(val)

    return joint_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--urdf",
        type=str,
        default="assets/franka_hand/franka_leap.urdf",
        help="Path to Franka+LEAP URDF",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--slider-min", type=float, default=-3.14)
    parser.add_argument("--slider-max", type=float, default=3.14)
    parser.add_argument("--slider-step", type=float, default=0.01)
    parser.add_argument(
        "--hand-preset",
        type=str,
        choices=sorted(HAND_PRESETS_ISAAC.keys()),
        default="default",
        help="Initial LEAP-hand preset in IsaacGym joint order.",
    )
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        urdf_path = (repo_root / urdf_path).resolve()

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # CODEX
    # Start Viser and load URDF.
    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(control_width="large")
    server.scene.add_frame("/WorldAxes", show_axes=True, axes_length=0.15, axes_radius=0.01)
    server.scene.add_grid("/grid", width=3, height=3, position=(0.0, 0.0, 0.0), shadow_opacity=0.2)

    urdf_vis = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/robot",
        load_meshes=True,
        load_collision_meshes=False,
    )

    # CODEX
    # Rotate ARM link meshes by 90 deg clockwise about local +X axis.
    # LEAP hand meshes live under "/palm_center/" and are intentionally untouched.
    # Clockwise about +X corresponds to -pi/2 with right-hand convention.
    half = -0.25 * math.pi
    q_local_x_cw = (
        float(math.cos(half)),
        float(math.sin(half)),
        0.0,
        0.0,
    )
    for mesh_handle in urdf_vis._meshes:
        mesh_name = getattr(mesh_handle, "name", "")
        if "/palm_center/" in mesh_name:
            continue
        mesh_handle.wxyz = q_local_x_cw

    joint_names = urdf_vis.get_actuated_joint_names()
    joint_map = build_default_joint_map(joint_names, args.hand_preset)

    def push_cfg() -> None:
        q = np.array([joint_map[name] for name in joint_names], dtype=np.float64)
        urdf_vis.update_cfg(q)

    push_cfg()

    # CODEX
    # GUI controls.
    folder = server.gui.add_folder("Franka+LEAP Joints")
    slider_handles: dict[str, object] = {}

    def make_cb(joint_name: str):
        def _cb(_: object) -> None:
            handle = slider_handles[joint_name]
            joint_map[joint_name] = float(handle.value)
            push_cfg()

        return _cb

    with folder:
        for joint_name in joint_names:
            handle = server.gui.add_slider(
                label=joint_name,
                min=args.slider_min,
                max=args.slider_max,
                step=args.slider_step,
                initial_value=float(joint_map[joint_name]),
            )
            slider_handles[joint_name] = handle
            handle.on_update(make_cb(joint_name))

    actions = server.gui.add_folder("Actions")
    with actions:
        reset_btn = server.gui.add_button("Reset to Default Pose")
        preset_default_btn = server.gui.add_button("Load Hand Default")
        preset_pinch2_btn = server.gui.add_button("Load Hand Pinch2")
        preset_pinch3_btn = server.gui.add_button("Load Hand Pinch3")
        print_btn = server.gui.add_button("Print LEAP Hand Angles")
        print_arm_btn = server.gui.add_button("Print Franka Arm Angles")  # CODEX

    def apply_hand_preset(hand_preset: str) -> None:
        for name, val in zip(LEAP_HAND_JOINT_NAMES_ISAAC_ORDER, HAND_PRESETS_ISAAC[hand_preset]):
            joint_map[name] = float(val)
            slider_handles[name].value = float(val)
        push_cfg()

    @reset_btn.on_click
    def _(_: object) -> None:
        defaults = build_default_joint_map(joint_names, args.hand_preset)
        for name in joint_names:
            joint_map[name] = defaults[name]
            slider_handles[name].value = float(defaults[name])
        push_cfg()

    @preset_default_btn.on_click
    def _(_: object) -> None:
        apply_hand_preset("default")

    @preset_pinch2_btn.on_click
    def _(_: object) -> None:
        apply_hand_preset("pinch2")

    @preset_pinch3_btn.on_click
    def _(_: object) -> None:
        apply_hand_preset("pinch3")

    @print_btn.on_click
    def _(_: object) -> None:
        # CODEX
        values = [joint_map[name] for name in LEAP_HAND_JOINT_NAMES_ISAAC_ORDER]
        print("LEAP hand angles in IsaacGym order (index, thumb, middle, ring):")
        print(f"{values[0]: .4f}, {values[1]: .4f}, {values[2]: .4f}, {values[3]: .4f},")
        print(f"{values[4]: .4f}, {values[5]: .4f}, {values[6]: .4f}, {values[7]: .4f},")
        print(f"{values[8]: .4f}, {values[9]: .4f}, {values[10]: .4f}, {values[11]: .4f},")
        print(f"{values[12]: .4f}, {values[13]: .4f}, {values[14]: .4f}, {values[15]: .4f},")

    @print_arm_btn.on_click
    def _(_: object) -> None:
        # CODEX
        # Print Franka arm joint angles in canonical joint order.
        values = [joint_map[name] for name in ARM_JOINT_NAMES]
        print("Franka arm angles (j1..j7):")
        print(", ".join([f"{v: .6f}" for v in values]))

    print(f"Viser server running at http://{args.host}:{args.port}")
    print(f"Loaded URDF: {urdf_path}")
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
