# completely Codex generated

import os

import h5py
import hydra
import isaacgym
import isaacgymenvs
import numpy as np
import torch
from omegaconf import DictConfig


class PresampleTableMultiEnvStates:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.seed = cfg.seed
        self.num_envs = self._resolve_num_envs()
        self.output_hdf5_dir = cfg.presample.output_hdf5_dir
        self.output_hdf5_name = cfg.presample.output_hdf5_name
        self.include_init_robot_states = cfg.presample.include_init_robot_states

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

    def _create_env(self):
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

    def _build_demo(self, env_idx: int) -> dict:
        cuboid_dims = self._to_numpy(self.env.cuboid_dims[env_idx]).reshape(-1, 3).astype(np.float32)
        cuboid_pos = self._to_numpy(self.env.cuboid_pos[env_idx]).reshape(-1, 3).astype(np.float32)
        cuboid_quats = self._to_numpy(self.env.cuboid_quats[env_idx]).reshape(-1, 4).astype(np.float32)

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

        if self.include_init_robot_states and hasattr(self.env, "canonical_joint_config"):
            init_robot_states = self._to_numpy(self.env.canonical_joint_config[env_idx]).astype(np.float32)
            demo["init_robot_states"] = init_robot_states

        return demo

    def save(self):
        os.makedirs(self.output_hdf5_dir, exist_ok=True)
        output_path = os.path.join(self.output_hdf5_dir, self.output_hdf5_name)

        with h5py.File(output_path, "w") as f:
            for env_idx in range(self.num_envs):
                demo_group = f.create_group(f"demo_{env_idx}")
                demo_data = self._build_demo(env_idx)
                for key, value in demo_data.items():
                    demo_group.create_dataset(key, data=value)

            f.attrs["task_name"] = str(self.cfg.task_name)
            f.attrs["num_envs"] = int(self.num_envs)
            f.attrs["include_init_robot_states"] = bool(self.include_init_robot_states)

        print(f"Saved {self.num_envs} demos to {output_path}")


@hydra.main(config_name="pre_sample_env_states_table_multi.yaml", config_path="../cfg")
def main(cfg: DictConfig):
    saver = PresampleTableMultiEnvStates(cfg=cfg)
    saver.save()


if __name__ == "__main__":
    main()
