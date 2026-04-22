import os
import sys
import tempfile
import unittest

sys.path.append('/home/rayliu/grogu/IsaacGymEnvs/mesh_processing')

from mesh_app.config import AppConfig
from mesh_app.repositories import MeshRepository


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


if __name__ == "__main__":
    unittest.main()
