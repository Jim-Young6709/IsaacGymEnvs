import argparse
import os
import shutil

import h5py
import hydra
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from isaacgymenvs.utils.pcd_utils import decompose_scene_pcd_params_obs


class PostProcessTableMultiSceneStates:
    def __init__(
        self,
        input_hdf5_path: str,
        output_hdf5_path: str,
        seed: int,
        overwrite: bool,
        task_name: str,
    ):
        self.seed = int(seed)
        self.input_hdf5_path = os.path.abspath(input_hdf5_path)
        output_path = output_hdf5_path.strip()
        if output_path == "":
            output_path = self.input_hdf5_path
        self.output_hdf5_path = os.path.abspath(output_path)
        self.overwrite = bool(overwrite)
        self.inplace = self.input_hdf5_path == self.output_hdf5_path
        self.distractor_params = self._load_distractor_params(task_name)

        if not os.path.isfile(self.input_hdf5_path):
            raise FileNotFoundError(f"Input HDF5 file not found: {self.input_hdf5_path}")

        if not self.inplace:
            os.makedirs(os.path.dirname(self.output_hdf5_path), exist_ok=True)
            if os.path.exists(self.output_hdf5_path):
                if not self.overwrite:
                    raise FileExistsError(
                        f"Output HDF5 already exists: {self.output_hdf5_path}. "
                        "Set postprocess.overwrite=True to replace it."
                    )
                os.remove(self.output_hdf5_path)
            shutil.copy2(self.input_hdf5_path, self.output_hdf5_path)

    @staticmethod
    def _load_distractor_params(task_name: str):
        cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cfg"))
        hydra_state = GlobalHydra.instance()
        if hydra_state.is_initialized():
            hydra_state.clear()

        with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
            cfg = hydra.compose(config_name=None, overrides=[f"+task={task_name}"])

        return cfg.task.env.distractor_settings.params

    @staticmethod
    def _resolve_demo_root(hdf5_file):
        if "data" in hdf5_file:
            return hdf5_file["data"]
        return hdf5_file

    @staticmethod
    def _sort_demo_keys(demo_keys):
        def _key_fn(key: str):
            try:
                return int(key.split("_")[-1])
            except ValueError:
                return key

        return sorted(demo_keys, key=_key_fn)

    @staticmethod
    def _to_2d_float32(arr, width: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, width), dtype=np.float32)
        return arr.reshape(-1, width).astype(np.float32)

    @staticmethod
    def _to_1d_float32(arr) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return arr.reshape(-1).astype(np.float32)

    @staticmethod
    def _pad_rows(arr: np.ndarray, target_rows: int, width: int) -> np.ndarray:
        arr = PostProcessTableMultiSceneStates._to_2d_float32(arr, width)
        if arr.shape[0] == target_rows:
            return arr
        if arr.shape[0] > target_rows:
            return arr[:target_rows]

        padded = np.zeros((target_rows, width), dtype=np.float32)
        if arr.shape[0] > 0:
            padded[: arr.shape[0]] = arr
        return padded

    @staticmethod
    def _pad_vector(arr: np.ndarray, target_rows: int) -> np.ndarray:
        arr = PostProcessTableMultiSceneStates._to_1d_float32(arr)
        if arr.shape[0] == target_rows:
            return arr
        if arr.shape[0] > target_rows:
            return arr[:target_rows]

        padded = np.zeros((target_rows,), dtype=np.float32)
        if arr.shape[0] > 0:
            padded[: arr.shape[0]] = arr
        return padded

    @staticmethod
    def _concat_rows(base: np.ndarray, extra, width: int) -> np.ndarray:
        base = PostProcessTableMultiSceneStates._to_2d_float32(base, width)
        extra = PostProcessTableMultiSceneStates._to_2d_float32(extra, width)
        if extra.shape[0] == 0:
            return base
        if base.shape[0] == 0:
            return extra
        return np.concatenate([base, extra], axis=0).astype(np.float32)

    def _build_scene_pcd_params(
        self,
        cuboid_dims: np.ndarray,
        cuboid_centers: np.ndarray,
        cuboid_quats: np.ndarray,
        cylinder_radii: np.ndarray,
        cylinder_heights: np.ndarray,
        cylinder_centers: np.ndarray,
        cylinder_quats: np.ndarray,
        sphere_centers: np.ndarray,
        sphere_radii: np.ndarray,
        mesh_positions: np.ndarray,
        mesh_scales: np.ndarray,
        mesh_quats: np.ndarray,
        obj_ids: np.ndarray,
        mesh_ids: np.ndarray,
    ) -> np.ndarray:
        m = max(
            self._to_2d_float32(cuboid_dims, 3).shape[0],
            self._to_1d_float32(cylinder_radii).shape[0],
            self._to_1d_float32(sphere_radii).shape[0],
            self._to_1d_float32(mesh_scales).shape[0],
        )
        if m <= 0:
            raise ValueError("Scene must contain at least one obstacle cuboid for the table.")

        cuboid_dims = self._pad_rows(cuboid_dims, m, 3)
        cuboid_centers = self._pad_rows(cuboid_centers, m, 3)
        cuboid_quats = self._pad_rows(cuboid_quats, m, 4)

        cylinder_radii = self._pad_vector(cylinder_radii, m)
        cylinder_heights = self._pad_vector(cylinder_heights, m)
        cylinder_centers = self._pad_rows(cylinder_centers, m, 3)
        cylinder_quats = self._pad_rows(cylinder_quats, m, 4)

        sphere_centers = self._pad_rows(sphere_centers, m, 3)
        sphere_radii = self._pad_vector(sphere_radii, m)

        mesh_positions = self._pad_rows(mesh_positions, m, 3)
        mesh_scales = self._pad_vector(mesh_scales, m)
        mesh_quats = self._pad_rows(mesh_quats, m, 4)
        obj_ids = self._pad_vector(obj_ids, m)
        mesh_ids = self._pad_vector(mesh_ids, m)

        scene_pcd_params = np.concatenate(
            [
                np.array([m], dtype=np.float32),
                cuboid_dims.reshape(-1),
                cuboid_centers.reshape(-1),
                cuboid_quats.reshape(-1),
                cylinder_radii,
                cylinder_heights,
                cylinder_centers.reshape(-1),
                cylinder_quats.reshape(-1),
                sphere_centers.reshape(-1),
                sphere_radii,
                mesh_positions.reshape(-1),
                mesh_scales,
                mesh_quats.reshape(-1),
                obj_ids,
                mesh_ids,
            ],
            axis=0,
        ).astype(np.float32)

        return scene_pcd_params

    def _sample_random_distractors(self, pos_range, cuboid_dims, cuboid_pos, cuboid_quats, np_rng):
        if pos_range[1][2] <= 0:
            return

        params = self.distractor_params
        rand01 = float(np_rng.uniform(0.0, 1.0))
        if rand01 < params["skip_prob"]:
            return
        if rand01 < (params["skip_prob"] + params["full_prob"]):
            pos_range_np = np.asarray(pos_range, dtype=np.float32)
            full_dim = pos_range_np[1] - pos_range_np[0]
            if np.any(full_dim <= 0):
                return
            full_pos = (pos_range_np[0] + pos_range_np[1]) / 2.0
            cuboid_dims.append(full_dim.astype(np.float32))
            cuboid_pos.append(full_pos.astype(np.float32))
            cuboid_quats.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            return

        num_range = params["num_distractors_per_region_range"]
        cuboid_size_range = np.asarray(params["cuboid_size_range"], dtype=np.float32)
        num_distractors = int(np_rng.integers(num_range[0], num_range[1] + 1))

        for _ in range(num_distractors):
            pos_range_np = np.asarray(pos_range, dtype=np.float32)
            height_limit = float(pos_range_np[1][2] - pos_range_np[0][2])
            cuboid_dim = np_rng.uniform(cuboid_size_range[0], cuboid_size_range[1]).astype(np.float32)
            if cuboid_dim[2] > height_limit:
                cuboid_dim[2] = height_limit

            if np.any(cuboid_dim <= 0):
                continue

            center_min = pos_range_np[0] + cuboid_dim / 2.0
            center_max = pos_range_np[1] - cuboid_dim / 2.0
            if np.any(center_max < center_min):
                continue

            sampled_pos = np_rng.uniform(center_min, center_max).astype(np.float32)
            cuboid_dims.append(cuboid_dim)
            cuboid_pos.append(sampled_pos)
            cuboid_quats.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    def _create_distractor_objects(self, table_pos: np.ndarray, table_size: np.ndarray, demo_seed: int):
        np_rng = np.random.default_rng(demo_seed)

        table_extend = float(self.distractor_params["table_extend"])
        max_z_height = float(self.distractor_params["free_space_distractor_max_height"])

        cuboid_dims = []
        cuboid_pos = []
        cuboid_quats = []

        table_x_min = table_pos[0] - table_size[0] / 2.0
        table_x_max = table_pos[0] + table_size[0] / 2.0
        table_y_min = table_pos[1] - table_size[1] / 2.0
        table_y_max = table_pos[1] + table_size[1] / 2.0
        table_z_min = table_pos[2] - table_size[2] / 2.0

        distractor_pos_range_list = [
            [
                [table_x_min, table_y_min - table_extend, 0.0],
                [table_x_max + table_extend, table_y_min, max_z_height],
            ],
            [
                [table_x_min, table_y_max, 0.0],
                [table_x_max + table_extend, table_y_max + table_extend, max_z_height],
            ],
            [
                [table_x_max, table_y_min, 0.0],
                [table_x_max + table_extend, table_y_max, max_z_height],
            ],
            [
                [table_x_min, table_y_min, 0.0],
                [table_x_min + table_extend, table_y_max, table_z_min],
            ],
        ]

        for pos_range in distractor_pos_range_list:
            self._sample_random_distractors(
                pos_range=pos_range,
                cuboid_dims=cuboid_dims,
                cuboid_pos=cuboid_pos,
                cuboid_quats=cuboid_quats,
                np_rng=np_rng,
            )

        return (
            self._to_2d_float32(cuboid_dims, 3),
            self._to_2d_float32(cuboid_pos, 3),
            self._to_2d_float32(cuboid_quats, 4),
        )

    def _replace_dataset(self, group, dataset_name: str, data: np.ndarray):
        attrs = {}
        if dataset_name in group:
            attrs = dict(group[dataset_name].attrs.items())
            del group[dataset_name]

        dataset = group.create_dataset(dataset_name, data=data)
        for key, value in attrs.items():
            dataset.attrs[key] = value

    def _process_demo(self, demo_group, demo_idx: int):
        if "states" not in demo_group:
            raise KeyError(f"demo_{demo_idx} does not contain a states dataset")

        states = np.asarray(demo_group["states"][:], dtype=np.float32)
        if states.ndim != 2 or states.shape[1] <= 15:
            raise ValueError(f"demo_{demo_idx} has invalid states shape: {states.shape}")

        scene_pcd_params = states[0, 15:]
        (
            cuboid_dims,
            cuboid_centers,
            cuboid_quats,
            cylinder_radii,
            cylinder_heights,
            cylinder_centers,
            cylinder_quats,
            sphere_centers,
            sphere_radii,
            mesh_positions,
            mesh_scales,
            mesh_quats,
            obj_ids,
            mesh_ids,
        ) = decompose_scene_pcd_params_obs(scene_pcd_params)

        if cuboid_dims.shape[0] == 0:
            raise ValueError(f"demo_{demo_idx} does not contain the table cuboid")

        table_pos = cuboid_centers[0].astype(np.float32)
        table_size = cuboid_dims[0].astype(np.float32)
        distractor_seed = self.seed + demo_idx
        distractor_cuboid_dims, distractor_cuboid_pos, distractor_cuboid_quats = self._create_distractor_objects(
            table_pos=table_pos,
            table_size=table_size,
            demo_seed=distractor_seed,
        )

        updated_cuboid_dims = self._concat_rows(cuboid_dims, distractor_cuboid_dims, 3)
        updated_cuboid_centers = self._concat_rows(cuboid_centers, distractor_cuboid_pos, 3)
        updated_cuboid_quats = self._concat_rows(cuboid_quats, distractor_cuboid_quats, 4)

        updated_scene_pcd_params = self._build_scene_pcd_params(
            cuboid_dims=updated_cuboid_dims,
            cuboid_centers=updated_cuboid_centers,
            cuboid_quats=updated_cuboid_quats,
            cylinder_radii=cylinder_radii,
            cylinder_heights=cylinder_heights,
            cylinder_centers=cylinder_centers,
            cylinder_quats=cylinder_quats,
            sphere_centers=sphere_centers,
            sphere_radii=sphere_radii,
            mesh_positions=mesh_positions,
            mesh_scales=mesh_scales,
            mesh_quats=mesh_quats,
            obj_ids=obj_ids,
            mesh_ids=mesh_ids,
        )

        updated_states = np.zeros((states.shape[0], 15 + updated_scene_pcd_params.shape[0]), dtype=np.float32)
        updated_states[:, :15] = states[:, :15]
        updated_states[:, 15:] = updated_scene_pcd_params[None, :]

        self._replace_dataset(demo_group, "states", updated_states)
        demo_group.attrs["distractor_postprocessed"] = True
        demo_group.attrs["distractor_postprocess_seed"] = distractor_seed
        demo_group.attrs["num_added_distractor_cuboids"] = int(distractor_cuboid_dims.shape[0])

        return int(distractor_cuboid_dims.shape[0])

    def run(self):
        processed = 0
        total_added = 0

        with h5py.File(self.output_hdf5_path, "r+") as hdf5_file:
            demo_root = self._resolve_demo_root(hdf5_file)
            demo_keys = self._sort_demo_keys([key for key in demo_root.keys() if key.startswith("demo_")])

            for demo_key in tqdm(demo_keys, desc="Post-processing demos"):
                demo_idx = int(demo_key.split("_")[-1])
                demo_group = demo_root[demo_key]
                added = self._process_demo(demo_group, demo_idx)
                processed += 1
                total_added += added

            hdf5_file.attrs["distractor_postprocessed"] = True
            hdf5_file.attrs["distractor_postprocess_seed"] = self.seed
            hdf5_file.attrs["distractor_postprocess_input"] = self.input_hdf5_path

        print("-----------------------------------------------------------")
        print(f"Processed HDF5: {self.output_hdf5_path}")
        print(f"Updated demos: {processed}")
        print(f"Added distractor cuboids: {total_added}")
        print("-----------------------------------------------------------")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process scene-state HDF5 files by regenerating distractor cuboids from the table geometry."
    )
    parser.add_argument("--input_hdf5_path", required=True)
    parser.add_argument("--output_hdf5_path", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_name", default="DexMobileExpBase")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    processor = PostProcessTableMultiSceneStates(
        input_hdf5_path=args.input_hdf5_path,
        output_hdf5_path=args.output_hdf5_path,
        seed=args.seed,
        overwrite=args.overwrite,
        task_name=args.task_name,
    )
    processor.run()


if __name__ == "__main__":
    main()
