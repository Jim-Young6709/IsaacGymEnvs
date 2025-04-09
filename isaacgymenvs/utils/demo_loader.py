import h5py
import sys
import random
from dataclasses import dataclass
import numpy as np


class DemoLoader:
    
    def __init__(self, hdf5_path, batch_size):
        """
        Initialize the demo loader
        Args:
            hdf5_path: Path to HDF5 file
            batch_size: Number of demos to load at once (should match num_envs)
        """
        self.batch_size = batch_size
        self.current_batch = 0
        self.hdf5_file = None
        self.demos = None
        self.total_demos = 0
        self._load_hdf5_file(hdf5_path)

    @dataclass
    class Plan:
        start_config: list
        goal_config: list
        gripper_state: float
        plan: list
    

    def _load_hdf5_file(self, file_path):
        """Load the HDF5 file and get total number of demos"""
        try:
            self.hdf5_file = h5py.File(file_path, 'r')
            self.demos = self.hdf5_file['data']
            self.total_demos = len(self.demos)
            print(f"Loaded HDF5 file with {self.total_demos} demonstrations")
            return True
        except Exception as e:
            print(f"Error loading HDF5 file: {e}")
            sys.exit(1)
            return False

    def get_random_arm_cfg_pair(self, demo_idx):
        demo_key = f"demo_{demo_idx}"
        # solutions = len(self.demos[demo_key])
        all_keys = self.demos[demo_key].keys()

        # Filter keys that contain the word "solution"
        number_of_solutions = len([key for key in all_keys if "solution" in str(key)])

        solution_idx = random.randint(0, number_of_solutions-1)
        solution_key = f"solution_{solution_idx}"
        start_config = self.demos[f"{demo_key}/{solution_key}/start_config"][:]
        goal_config = self.demos[f"{demo_key}/{solution_key}/goal_config"][:]
        return (start_config, goal_config)

    def get_next_batch(self, batch_idx=None, num_task_per_env=3):
        """Get next batch of demonstrations"""
        if self.demos is None:
            return None

        if batch_idx is None:
            start_idx = self.current_batch * self.batch_size
        else:
            start_idx = batch_idx * self.batch_size

        if start_idx >= self.total_demos:
            print("All demonstrations processed")
            return None

        vec_states = np.zeros((self.batch_size, num_task_per_env, 3, 9)) # 3 for start & goal & middle waypoint of the traj, 9 for 7-dim jonit angle + gripper state

        end_idx = min(start_idx + self.batch_size, self.total_demos)
        batch_data = []
        for demo_idx in range(start_idx, end_idx):
            demo_key = f"demo_{demo_idx}"
            solutions = len(self.demos[demo_key]) - 1  # Exclude the "states" key
            plans = []

            if 'solution_0' in self.demos['demo_0'].keys():
                for sol in range(num_task_per_env):
                    sol_idx = sol % solutions
                    solution_key = f"solution_{sol_idx}"
                    start_config = self.demos[demo_key][solution_key]["start_config"][:]
                    goal_config = self.demos[demo_key][solution_key]["goal_config"][:]
                    plan_len = len(self.demos[demo_key][solution_key]["plan"][:])
                    middle_waypoint = self.demos[demo_key][solution_key]["plan"][plan_len//2]
                    gripper_state = self.demos[demo_key][solution_key]["gripper_state"][:]

                    gripper_double = np.tile(gripper_state, 2)  # shape (2,)
                    start_row = np.concatenate([start_config, gripper_double])
                    goal_row = np.concatenate([goal_config, gripper_double])
                    middle_row = np.concatenate([middle_waypoint, gripper_double])

                    vec_states[demo_idx-start_idx, sol] = np.stack([start_row, goal_row, middle_row], axis=0)

            # for sol in range(solutions):
            #     solution_key = f"solution_{sol}"
            #     plan = DemoLoader.Plan(
            #         start_config =self.demos[demo_key][solution_key]["start_config"][:],
            #         goal_config  =self.demos[demo_key][solution_key]["goal_config"][:],
            #         gripper_state=self.demos[demo_key][solution_key]["gripper_state"][:],
            #         plan         =self.demos[demo_key][solution_key]["plan"][:]
            #     )
            #     plans.append(plan)

            try:
                # TODO: Support multiple configs in one env, ideally have one valid config for each support volume, or can even just load cuboids
                # Get all necessary data from the demo
                demo_data = {
                    'states': self.demos[f"{demo_key}/states"][:],
                    # 'plan': plans
                }
                batch_data.append(demo_data)
            except Exception as e:
                print(f"Error loading demo {demo_idx}: {e}")
                continue

        if batch_idx is None:
            self.current_batch += 1
        return batch_data, vec_states

    def reset(self):
        """Reset to first batch"""
        self.current_batch = 0

    def has_more_data(self):
        """Check if there are more demonstrations to process"""
        return self.current_batch * self.batch_size < self.total_demos

    def __del__(self):
        """Clean up HDF5 file handle"""
        if self.hdf5_file is not None:
            self.hdf5_file.close()