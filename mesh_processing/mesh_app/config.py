from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CategorySpec:
    key: str
    label: str
    group: str


@dataclass(frozen=True)
class CylinderSpec:
    x_dim: float
    y_dim: float
    height: float


@dataclass
class AppConfig:
    mesh_root: str = "/home/rayliu/grogu/IsaacGymEnvs/meshes_side/long_new" # "/home/rayliu/grogu/IsaacGymEnvs/meshes_69_new"
    output_root: str = "/home/rayliu/grogu/IsaacGymEnvs/meshes_long"
    history_keep: int = 10
    discard_dir: str = "discard"
    max_scale: float = 10.0
    leap_hand_path: str = "/home/rayliu/grogu/IsaacGymEnvs/assets/franka_hand/franka_leap.urdf"
    leap_hand_assets_root: str = "/home/rayliu/grogu/IsaacGymEnvs/assets/franka_hand"
    leap_hand_joint_order: Optional[List[str]] = None
    leap_hand_presets: Optional[Dict[str, List[float]]] = None
    default_urdf_scale: List[float] = None
    categories: List[CategorySpec] = None
    cylinder_min_specs: Optional[Dict[str, CylinderSpec]] = None
    cylinder_max_specs: Optional[Dict[str, CylinderSpec]] = None
    cylinder_range_min_factor: float = 0.85
    cylinder_range_max_factor: float = 1.15

    def __post_init__(self):
        if self.default_urdf_scale is None:
            # Default to no scaling when URDF is missing.
            self.default_urdf_scale = [1.0, 1.0, 1.0]
        if self.categories is None:
            self.categories = default_categories()
        if self.leap_hand_joint_order is None:
            self.leap_hand_joint_order = [
                "finger_joint_1", "finger_joint_0", "finger_joint_2", "finger_joint_3",
                "finger_joint_12", "finger_joint_13", "finger_joint_14", "finger_joint_15",
                "finger_joint_5", "finger_joint_4", "finger_joint_6", "finger_joint_7",
                "finger_joint_9", "finger_joint_8", "finger_joint_10", "finger_joint_11",
            ]
        if self.leap_hand_presets is None:
            self.leap_hand_presets = {
                "default": [
                    0.0000, 0.0000, 0.0000, 0.0000,
                    0.6900, 1.0100, 0.0000, -0.0300,
                    0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000,
                ],
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
        else:
            self.leap_hand_presets = dict(self.leap_hand_presets)
        if self.cylinder_min_specs is None:
            self.cylinder_min_specs = default_cylinder_specs(self.cylinder_range_min_factor)
        else:
            self.cylinder_min_specs = dict(self.cylinder_min_specs)
        if self.cylinder_max_specs is None:
            self.cylinder_max_specs = default_cylinder_specs(self.cylinder_range_max_factor)
        else:
            self.cylinder_max_specs = dict(self.cylinder_max_specs)


def default_categories() -> List[CategorySpec]:
    return [
        CategorySpec("regular.large", "large", "regular / top-down"),
        CategorySpec("regular.tall_large", "tall large", "regular / top-down"),
        CategorySpec("regular.medium", "medium", "regular / top-down"),
        CategorySpec("regular.thin", "thin", "regular / top-down"),
        CategorySpec("regular.small", "small", "regular / top-down"),
        CategorySpec("regular.tiny", "tiny", "regular / top-down"),
        CategorySpec("long.large", "large", "long / side"),
        CategorySpec("long.medium", "medium", "long / side"),
        CategorySpec("long.small", "small", "long / side"),
        CategorySpec("irregular.rim", "rim", "irregular"),
        CategorySpec("irregular.handle", "handle", "irregular"),
    ]


def default_cylinder_specs(scale: float = 1.0) -> Dict[str, CylinderSpec]:
    return {
        "regular.large": CylinderSpec(x_dim=0.110 * scale, y_dim=0.110 * scale, height=0.120 * scale),
        "regular.tall_large": CylinderSpec(x_dim=0.100 * scale, y_dim=0.100 * scale, height=0.180 * scale),
        "regular.medium": CylinderSpec(x_dim=0.084 * scale, y_dim=0.084 * scale, height=0.100 * scale),
        "regular.thin": CylinderSpec(x_dim=0.120 * scale, y_dim=0.120 * scale, height=0.040 * scale),
        "regular.small": CylinderSpec(x_dim=0.060 * scale, y_dim=0.060 * scale, height=0.070 * scale),
        "regular.tiny": CylinderSpec(x_dim=0.036 * scale, y_dim=0.036 * scale, height=0.045 * scale),
        "long.large": CylinderSpec(x_dim=0.064 * scale, y_dim=0.064 * scale, height=0.220 * scale),
        "long.medium": CylinderSpec(x_dim=0.052 * scale, y_dim=0.052 * scale, height=0.180 * scale),
        "long.small": CylinderSpec(x_dim=0.040 * scale, y_dim=0.040 * scale, height=0.140 * scale),
        "irregular.rim": CylinderSpec(x_dim=0.110 * scale, y_dim=0.110 * scale, height=0.080 * scale),
        "irregular.handle": CylinderSpec(x_dim=0.044 * scale, y_dim=0.044 * scale, height=0.120 * scale),
    }
