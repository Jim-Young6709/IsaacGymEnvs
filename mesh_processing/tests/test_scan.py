import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

sys.path.append('/home/rayliu/grogu/IsaacGymEnvs/mesh_processing')

from mesh_app.config import AppConfig
from mesh_app.models import MeshEntry, TransformState
from mesh_app.repositories import HistoryRepository, MeshRepository, TransformRepository


class TestMeshScan(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.repo = MeshRepository(self.config)

    def test_legacy_layout(self):
        with tempfile.TemporaryDirectory() as root:
            label_dir = os.path.join(root, "cup")
            os.makedirs(label_dir, exist_ok=True)
            open(os.path.join(label_dir, "1.obj"), "w").close()
            open(os.path.join(label_dir, "1.npy"), "w").close()

            entries = self.repo.scan(root)
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.label, "cup")
            self.assertEqual(entry.base_name, "1")
            self.assertTrue(entry.npy_path.endswith("1.npy"))

    def test_nested_layout(self):
        with tempfile.TemporaryDirectory() as root:
            obj_dir = os.path.join(root, "regular", "bubble_gum_2")
            os.makedirs(obj_dir, exist_ok=True)
            open(os.path.join(obj_dir, "bubble_gum_2.obj"), "w").close()
            open(os.path.join(obj_dir, "bubble_gum_2.npy"), "w").close()

            entries = self.repo.scan(root)
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.label, "bubble_gum_2")
            self.assertEqual(entry.base_name, "bubble_gum_2")
            self.assertEqual(entry.full_name, "bubble_gum_2")
            self.assertTrue(entry.npy_path.endswith("bubble_gum_2.npy"))

    def test_label_base_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            label_dir = os.path.join(root, "spoon")
            os.makedirs(label_dir, exist_ok=True)
            open(os.path.join(label_dir, "v2.obj"), "w").close()

            entries = self.repo.scan(root)
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry.label, "spoon")
            self.assertEqual(entry.base_name, "v2")
            self.assertEqual(entry.full_name, "spoon_v2")


class TestUrdfRewrite(unittest.TestCase):
    def setUp(self):
        self.config = AppConfig()
        self.mesh_repo = MeshRepository(self.config)
        self.history_repo = HistoryRepository(self.config)
        self.transform_repo = TransformRepository(
            self.config, self.mesh_repo, self.history_repo
        )

    def test_existing_urdf_origin_follows_mesh_transform(self):
        with tempfile.TemporaryDirectory() as root:
            urdf_path = os.path.join(root, "object.urdf")
            out_urdf = os.path.join(root, "rewritten.urdf")
            with open(urdf_path, "w") as f:
                f.write(
                    """<?xml version="1.0" ?>
<robot name="obj">
  <link name="base">
    <visual>
      <origin xyz="1 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="object.obj" scale="3 3 3"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="1 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="object.obj" scale="3 3 3"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="1 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>
"""
                )

            entry = MeshEntry(
                label="object",
                base_name="object",
                obj_path=os.path.join(root, "object.obj"),
                npy_path=None,
                glb_path=None,
                urdf_path=urdf_path,
            )
            transform = TransformState(
                dilate_x_min=2.0,
                dilate_x_max=2.0,
                dilate_y_min=2.0,
                dilate_y_max=2.0,
                dilate_z_min=2.0,
                dilate_z_max=2.0,
                rot_z=90.0,
            ).normalized()

            mesh_scaled = transform.apply_to_mesh(trimesh.creation.box())
            self.transform_repo._write_urdf(
                entry,
                transform,
                mesh_scaled,
                out_urdf,
                mesh_filename="object.obj",
            )

            root_xml = ET.parse(out_urdf).getroot()
            visual_origin = np.fromstring(
                root_xml.find(".//visual/origin").get("xyz"), sep=" "
            )
            collision_origin = np.fromstring(
                root_xml.find(".//collision/origin").get("xyz"), sep=" "
            )
            inertial_origin = np.fromstring(
                root_xml.find(".//inertial/origin").get("xyz"), sep=" "
            )

            self.assertTrue(np.allclose(visual_origin, [0.0, 6.0, 0.0]))
            self.assertTrue(np.allclose(collision_origin, [0.0, 6.0, 0.0]))
            self.assertTrue(np.allclose(inertial_origin, [0.0, 2.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
