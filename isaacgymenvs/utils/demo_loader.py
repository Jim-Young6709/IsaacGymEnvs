import h5py
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
            return False

    def get_next_batch(self, batch_idx=None):
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

        end_idx = min(start_idx + self.batch_size, self.total_demos)
        batch_data = []
        
        for demo_idx in range(start_idx, end_idx):
            demo_key = f"demo_{demo_idx}"
            try:
                # TODO: Support multiple configs in one env, ideally have one valid config for each support volume, or can even just load cuboids
                # Get all necessary data from the demo
                demo_data = {
                    'states': self.demos[f"{demo_key}/states"][:],
                }
                batch_data.append(demo_data)
            except Exception as e:
                print(f"Error loading demo {demo_idx}: {e}")
                continue

        if batch_idx is None:
            self.current_batch += 1
        return batch_data

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