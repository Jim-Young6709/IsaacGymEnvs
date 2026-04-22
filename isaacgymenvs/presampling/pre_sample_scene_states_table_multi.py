# completely Codex generated

import os

import h5py
import hydra
import isaacgym
import isaacgymenvs
import numpy as np
import torch
import random
from omegaconf import DictConfig
from typing import Tuple
from tqdm import tqdm

from isaacgymenvs.tasks import FrankaLEAPMobile


class PresampleTableMultiEnvStates:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.seed = cfg.seed
        self.num_envs = self._resolve_num_envs()
        self.output_hdf5_dir = cfg.presample.output_hdf5_dir
        self.output_hdf5_name = cfg.presample.output_hdf5_name

        self.env = self._create_env()
        self.env.reset()

    def _resolve_num_envs(self) -> int:
        if isinstance(self.cfg.num_envs, str):
            if self.cfg.num_envs.strip() == "":
                return int(self.cfg.task.env.numEnvs)
            return int(self.cfg.num_envs)
        if self.cfg.num_envs is None:
            return int(self.cfg.task.env.numEnvs)
        return int(self.cfg.num_envs)

    def _create_env(self) -> FrankaLEAPMobile:
        self.cfg.task.env.numEnvs = self.num_envs
        return isaacgymenvs.make(
            self.cfg.seed,
            self.cfg.task_name,
            self.cfg.task.env.numEnvs,
            self.cfg.sim_device,
            self.cfg.rl_device,
            self.cfg.graphics_device_id,
            self.cfg.headless,
            self.cfg.multi_gpu,
            self.cfg.capture_video,
            self.cfg.force_render,
            self.cfg,
        )

    @staticmethod
    def _to_numpy(arr) -> np.ndarray:
        if torch.is_tensor(arr):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    @staticmethod
    def _build_scene_pcd_params(
        cuboid_dims: np.ndarray,
        cuboid_centers: np.ndarray,
        cuboid_quats: np.ndarray,
    ) -> np.ndarray:
        """
        Build scene_pcd_params compatible with decompose_scene_pcd_params_obs().
        M follows the cuboid count because FrankaLEAPMobilePickFull currently
        loads cuboids from this payload.
        """
        m = int(cuboid_dims.shape[0])

        # Keep the same full layout used by decompose_scene_pcd_params_obs:
        # [M | cuboid(10M) | cylinder(9M) | sphere(4M) | mesh(10M)]
        cylinder_radii = np.zeros((m,), dtype=np.float32)
        cylinder_heights = np.zeros((m,), dtype=np.float32)
        cylinder_centers = np.zeros((m, 3), dtype=np.float32)
        cylinder_quats = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (m, 1))

        sphere_centers = np.zeros((m, 3), dtype=np.float32)
        sphere_radii = np.zeros((m,), dtype=np.float32)

        mesh_positions = np.zeros((m, 3), dtype=np.float32)
        mesh_scales = np.zeros((m,), dtype=np.float32)
        mesh_quats = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (m, 1))
        obj_ids = np.zeros((m,), dtype=np.float32)
        mesh_ids = np.zeros((m,), dtype=np.float32)

        scene_pcd_params = np.concatenate(
            [
                np.array([m], dtype=np.float32),
                cuboid_dims.reshape(-1).astype(np.float32),
                cuboid_centers.reshape(-1).astype(np.float32),
                cuboid_quats.reshape(-1).astype(np.float32),
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

    def _create_distractor_objects(self, env_idx):
        """
        create distractor objects under/behind/side the table to approximate real world setting
        since the robot will never interact with these objects, we only create pcd for them rather than actually spawning them in sim
        """

        def _sample_random_distractors(pos_range):
            """
            sample random distractor objects within the given pos region

            Args:
                pos_range: List[[x_min, y_min, z_min], [x_max, y_max, z_max]], note this is the boundary range not the object center pos range
            """
            if pos_range[1][2] <= 0:
                return

            _params = self.env.distractor_settings["params"]
            rand01 = np.random.uniform(0.0, 1.0)
            if rand01 < _params["skip_prob"]:
                return
            elif rand01 < (_params["skip_prob"] + _params["full_prob"]):
                _pos_range = np.array(pos_range)
                _cuboid_dim = _pos_range[1] - _pos_range[0]
                _cuboid_pos = (_pos_range[0] + _pos_range[1]) / 2
                _cuboid_quat = np.array([0.0, 0.0, 0.0, 1.0])
                cuboid_dims.append(_cuboid_dim)
                cuboid_pos.append(_cuboid_pos)
                cuboid_quats.append(_cuboid_quat)
                return

            _num_range = _params["num_distractors_per_region_range"]
            _cuboid_size_range = _params["cuboid_size_range"]
            _cylinder_size_range = _params["cylinder_size_range"]
            _sphere_size_range = _params["sphere_size_range"]

            _num = np.random.randint(_num_range[0], _num_range[1]+1)
            for _ in range(_num):
                _type = random.choice([0, 0, 0]) # TODO: only considering cuboids for now, fix later?
                _pos_range = np.array(pos_range)
                height_limit = _pos_range[1][2] - _pos_range[0][2]
                if _type == 0: # cuboid
                    _cuboid_dim = np.random.uniform(_cuboid_size_range[0], _cuboid_size_range[1])
                    if _cuboid_dim[2] > height_limit:
                        _cuboid_dim[2] = height_limit
                    _pos_range[0] += _cuboid_dim / 2
                    _pos_range[1] -= _cuboid_dim / 2
                    _cuboid_pos = np.random.uniform(_pos_range[0], _pos_range[1])
                    _cuboid_quat = np.array([0.0, 0.0, 0.0, 1.0])
                    cuboid_dims.append(_cuboid_dim)
                    cuboid_pos.append(_cuboid_pos)
                    cuboid_quats.append(_cuboid_quat)
                elif _type == 1: # cylinder
                    _cylinder_dim = np.random.uniform(_cylinder_size_range[0], _cylinder_size_range[1])
                    _cylinder_radius = _cylinder_dim[0]
                    _cylinder_height = _cylinder_dim[1]
                    if _cylinder_height > height_limit:
                        _cylinder_height = height_limit
                    _pos_range_offset = np.array([_cylinder_radius, _cylinder_radius, _cylinder_height / 2])
                    _pos_range[0] += _pos_range_offset
                    _pos_range[1] -= _pos_range_offset
                    _cylinder_pos = np.random.uniform(_pos_range[0], _pos_range[1])
                    _cylinder_quat = np.array([0.0, 0.0, 0.0, 1.0])
                    cylinder_radii.append(_cylinder_radius)
                    cylinder_heights.append(_cylinder_height)
                    cylinder_pos.append(_cylinder_pos)
                    cylinder_quats.append(_cylinder_quat)
                elif _type == 2: # sphere
                    _sphere_dim = np.random.uniform(_sphere_size_range[0], _sphere_size_range[1])
                    _sphere_radius = _sphere_dim
                    if _sphere_radius > height_limit / 2:
                        _sphere_radius = height_limit / 2
                    _pos_range[0] += _sphere_radius
                    _pos_range[1] -= _sphere_radius
                    _sphere_pos = np.random.uniform(_pos_range[0], _pos_range[1])
                    sphere_radii.append(_sphere_radius)
                    sphere_pos.append(_sphere_pos)

        table_pos = self.env.table_pos.cpu().numpy()
        table_size = self.env.table_size.cpu().numpy()
        table_extend = self.env.distractor_settings["params"]["table_extend"]
        max_z_height = self.env.distractor_settings["params"]["free_space_distractor_max_height"]

        # init lists
        cuboid_dims = []  # xyz
        cuboid_pos = []
        cuboid_quats = [] # xyzw

        cylinder_radii = []
        cylinder_heights = []
        cylinder_pos = []
        cylinder_quats = []

        sphere_radii = []
        sphere_pos = []

        # adding distractor pos range when: side/under/behind the table
        table_x_min = table_pos[env_idx][0] - table_size[env_idx][0] / 2
        table_x_max = table_pos[env_idx][0] + table_size[env_idx][0] / 2
        table_y_min = table_pos[env_idx][1] - table_size[env_idx][1] / 2
        table_y_max = table_pos[env_idx][1] + table_size[env_idx][1] / 2
        table_z_min = table_pos[env_idx][2] - table_size[env_idx][2] / 2

        distractor_pos_range_list = [
            [ # side 1
                [table_x_min, table_y_min - table_extend, 0.0],
                [table_x_max + table_extend, table_y_min, max_z_height],
            ],
            [ # side 2
                [table_x_min, table_y_max, 0.0],
                [table_x_max + table_extend, table_y_max + table_extend, max_z_height],
            ],
            [ # behind
                [table_x_max, table_y_min, 0.0],
                [table_x_max + table_extend, table_y_max, max_z_height],
            ],
            [ # under
                [table_x_min, table_y_min, 0.0],
                [table_x_min + table_extend, table_y_max, table_z_min], # bias towards the front part of the table
            ],
        ]

        # under the table
        for subregion_distractor_pos_range in distractor_pos_range_list:
            _sample_random_distractors(subregion_distractor_pos_range)

        return (
            cuboid_dims,
            cuboid_pos,
            cuboid_quats,
            cylinder_radii,
            cylinder_heights,
            cylinder_pos,
            cylinder_quats,
            sphere_radii,
            sphere_pos,
        )

    def _build_demo(self, env_idx: int) -> dict:
        cuboid_dims = self._to_numpy(self.env.cuboid_dims[env_idx]).reshape(-1, 3).astype(np.float32)
        cuboid_pos = self._to_numpy(self.env.cuboid_pos[env_idx]).reshape(-1, 3).astype(np.float32)
        cuboid_quats = self._to_numpy(self.env.cuboid_quats[env_idx]).reshape(-1, 4).astype(np.float32)

        # TODO: comment out for now, maybe no longer needed
        # if cuboid_dims.shape[0] > 1:
        #     table_x_min = cuboid_pos[0, 0] - 0.5 * cuboid_dims[0, 0]
        #     other_x_min = cuboid_pos[1:, 0] - 0.5 * cuboid_dims[1:, 0]
        #     invalid = other_x_min < table_x_min
        #     if np.any(invalid):
        #         cuboid_dims[1:][invalid] = np.array([0.001, 0.001, 0.001], dtype=np.float32)
        #         cuboid_pos[1:][invalid] = np.array([0.0, 0.0, -100.0], dtype=np.float32)
        #         cuboid_quats[1:][invalid] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        (
            distractor_cuboid_dims,
            distractor_cuboid_pos,
            distractor_cuboid_quats,
            distractor_cylinder_radii,
            distractor_cylinder_heights,
            distractor_cylinder_pos,
            distractor_cylinder_quats,
            distractor_sphere_radii,
            distractor_sphere_pos,
        ) = self._create_distractor_objects(env_idx)

        if len(distractor_cuboid_dims) > 0:
            cuboid_dims = np.concatenate([cuboid_dims, distractor_cuboid_dims], axis=0)
            cuboid_pos = np.concatenate([cuboid_pos, distractor_cuboid_pos], axis=0)
            cuboid_quats = np.concatenate([cuboid_quats, distractor_cuboid_quats], axis=0)

        scene_pcd_params = self._build_scene_pcd_params(
            cuboid_dims=cuboid_dims,
            cuboid_centers=cuboid_pos,
            cuboid_quats=cuboid_quats,
        )

        # Loader convention: states[0][15:] is scene_pcd_params.
        states = np.zeros((1, 15 + scene_pcd_params.shape[0]), dtype=np.float32)
        states[0, 15:] = scene_pcd_params

        box_dims = self._to_numpy(self.env.box_dims[env_idx]).reshape(3).astype(np.float32)
        # In table-multi env, box_pos.z is the box bottom. Loader expects
        # compartment_states to store center z, so convert before saving.
        box_pos = self._to_numpy(self.env.box_pos[env_idx]).reshape(3).astype(np.float32)
        box_pos_compartment = box_pos.copy()
        box_pos_compartment[2] += box_dims[2] / 2.0
        box_quats = self._to_numpy(self.env.box_quats[env_idx]).reshape(4).astype(np.float32)
        compartment_states = np.concatenate([box_dims, box_pos_compartment, box_quats], axis=0).astype(np.float32)[None, :]

        demo = {
            "states": states,
            "compartment_states": compartment_states,
        }

        return demo

    def save(self):
        os.makedirs(self.output_hdf5_dir, exist_ok=True)
        output_path = os.path.join(self.output_hdf5_dir, self.output_hdf5_name)

        with h5py.File(output_path, "w") as f:
            for env_idx in tqdm(range(self.num_envs), desc="Saving env states"):
                demo_group = f.create_group(f"demo_{env_idx}")
                demo_data = self._build_demo(env_idx)
                for key, value in demo_data.items():
                    demo_group.create_dataset(key, data=value)

            f.attrs["task_name"] = str(self.cfg.task_name)
            f.attrs["num_envs"] = int(self.num_envs)

        print(f"Saved {self.num_envs} demos to {output_path}")


@hydra.main(config_name="pre_sample_scene_states_table_multi.yaml", config_path="../cfg")
def main(cfg: DictConfig):
    saver = PresampleTableMultiEnvStates(cfg=cfg)
    saver.save()


if __name__ == "__main__":
    main()
