import os
import subprocess
import traceback

import h5py
import numpy as np
from tqdm import tqdm

# tlist = [3,5,9,10,11,13,15,16,17,18] # gb1
# tlist = [0,1,3,4,5,6,8,9,10,11] # gb2
# tlist = [0,1,2,3,4,7,8,9,10,11,12,13,14,15,18,19,20,21,22,23] # sao hybrid
# tlist = [0,1,2,4,5,7,8,9,10,12,14,15,16,19,20,21,22,23,24] # sao box
# tlist = [1,2,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,21,23,25,26] # sao tabletop


def write_trajectory_to_dataset(env, traj, data_grp, demo_name):
    """
    Write the collected trajectory to hdf5 compatible with robomimic.
    """
    # create group for this trajectory
    ep_data_grp = data_grp.create_group(demo_name)

    if "config_idx" in traj:
        ep_data_grp.attrs["config_idx"] = traj["config_idx"]

    if "assets" in traj:
        ep_data_grp.attrs["assets"] = traj["assets"]

    if "states" in traj:
        data = np.array(traj["states"])
        ep_data_grp.create_dataset("states", data=data)
        ep_data_grp.attrs["num_samples"] = traj["states"].shape[0]
    if "obs" in traj:
        for k in traj["obs"]:
            # assert dtype is np.uint8 or np.float32
            assert traj["obs"][k].dtype in [np.uint8, np.float32], (traj["obs"][k].dtype, k)
            # TODO: figure out if we can get away with float16, this will save a lot of space
            data = np.array(traj["obs"][k])
            # if k == 'pcd':
            #     data = data.astype(np.float16)
            ep_data_grp.create_dataset("obs/{}".format(k), data=data, compression="gzip")

    if "tight_config" in traj:
        data = np.array(traj["tight_config"])
        ep_data_grp.create_dataset("tight_config", data=data, compression="gzip")

    if "open_config" in traj:
        data = np.array(traj["open_config"])
        ep_data_grp.create_dataset("open_config", data=data, compression="gzip")

    if "actions" in traj:
        # episode metadata: number of transitions in this episode
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
        return traj["actions"].shape[0]

    for k in traj.keys():
        if "solution" in k:
            sol_grp = ep_data_grp.create_group(k)
            for subk in traj[k].keys():
                data = np.array(traj[k][subk])
                sol_grp.create_dataset(subk, data=data)


def load_demo_info(hdf5_file, filter_key=None):
    """
    Args:
        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        demos (list): list of demonstration keys to load from the hdf5 file. If
            omitted, all demos in the file (or under the @filter_by_attribute
            filter key) are used.
    """
    if filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(hdf5_file["mask/{}".format(filter_key)])]
    else:
        demos = list(hdf5_file["data"].keys())

    # sort demo keys
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    return demos


def load_dataset_in_memory(
    demo_list,
    hdf5_file,
    dataset_keys,
    data_grp,
    demo_count=0,
    total_samples=0,
    hdf5_path=None,
    dataset_path=None,
):
    """
    Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
    differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
    `getitem` operation.

    Args:
        demo_list (list): list of demo keys, e.g., 'demo_0'
        hdf5_file (h5py.File): file handle to the hdf5 dataset.
        obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
        dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'
        load_next_obs (bool): whether to load next_obs from the dataset

    Returns:
        all_data (dict): dictionary of loaded data.
    """

    for ep in tqdm(range(len(hdf5_file['data']))):
        demo_name = f"demo_{demo_count}"

        if ep not in tlist:
            continue
        traj = {}
        traj["attrs"] = {}
        traj["attrs"]["num_samples"] = hdf5_file["data/demo_{}".format(ep)].attrs["num_samples"]
        if "config_idx" in hdf5_file["data/demo_{}/".format(ep)].attrs:
            traj["config_idx"] = hdf5_file["data/demo_{}/".format(ep)].attrs["config_idx"]
        if "assets" in hdf5_file["data/demo_{}/".format(ep)].attrs:
            traj["assets"] = hdf5_file["data/demo_{}/".format(ep)].attrs["assets"]

        if "solution_count" in hdf5_file["data/demo_{}".format(ep)].attrs.keys():
            if hdf5_file["data/demo_{}".format(ep)].attrs["solution_count"] < 2:
                continue

        for k in hdf5_file["data/demo_{}".format(ep)].keys():
            if "solution" in k:
                traj[k] = {
                    subk: hdf5_file["data/demo_{}/{}/{}".format(ep, k, subk)][()]
                    for subk in hdf5_file["data/demo_{}/{}".format(ep, k)]
                }
            else:
                traj[k] = np.array(hdf5_file["data/demo_{}/{}".format(ep, k)])
        if "obs" in dataset_keys:
            traj["obs"] = {
                k: hdf5_file["data/demo_{}/obs/{}".format(ep, k)][()]
                for k in hdf5_file["data/demo_{}/obs".format(ep)]
            }

        write_trajectory_to_dataset(None, traj, data_grp, demo_name=demo_name)
        try:
            if hdf5_path is not None:
                filename = os.path.basename(hdf5_path).replace(".hdf5", "")
                dataset_filename = os.path.basename(dataset_path).replace(".hdf5", "")
                # copy from `planners/filename/planner_{demo_count}.pkl` to `planners/filename/demo_{demo_count}.pkl`
                filename_demo_count = int(ep)
                os.rename(
                    f"planners/{filename}/demo_{filename_demo_count}.pkl",
                    f"planners/{dataset_filename}/demo_{demo_count}.pkl",
                )
        except:
            pass
        demo_count += 1
        total_samples += traj["attrs"]["num_samples"]
    try:
        env_args = hdf5_file["data/"].attrs["env_args"]
    except:
        env_args = None
    return env_args, demo_count, total_samples


def global_dataset_updates(data_grp, total_samples, env_args):
    """
    Update the global dataset attributes.
    """
    data_grp.attrs["total_samples"] = total_samples
    data_grp.attrs["env_args"] = env_args
    return data_grp


def combine_hdf5(hdf5_paths, hdf5_use_swmr, dataset_path, filter_key):
    # remove dataset path incase it exists
    subprocess.run(["rm", "-rfv", dataset_path])
    data_writer = h5py.File(dataset_path, "w")
    print("writing to: ", dataset_path)
    data_grp = data_writer.create_group("data")
    # get the keys from the first hdf5 file
    dataset_keys = list(
        h5py.File(hdf5_paths[0], "r", swmr=hdf5_use_swmr, libver="latest")["data/demo_0"].keys()
    )
    demo_count = 0
    total_samples = 0
    env_args = None
    planner_dataset_filename = os.path.basename(dataset_path).replace(".hdf5", "")
    # clear planners/{planner_dataset_filename}
    subprocess.run(["rm", "-rfv", f"planners/{planner_dataset_filename}"])
    os.makedirs(f"planners/{planner_dataset_filename}", exist_ok=True)
    for hdf5_path in tqdm(hdf5_paths):
        try:
            hdf5_file = h5py.File(hdf5_path, "r", swmr=hdf5_use_swmr, libver="latest")
            demo_list = load_demo_info(hdf5_file, filter_key=filter_key)
            env_args_new, demo_count, total_samples = load_dataset_in_memory(
                demo_list,
                hdf5_file,
                dataset_keys,
                data_grp=data_grp,
                total_samples=total_samples,
                demo_count=demo_count,
                hdf5_path=hdf5_path,
                dataset_path=dataset_path,
            )
            if env_args_new is not None:
                env_args = env_args_new
            print("loaded: ", hdf5_path)
            # close the hdf5 file
            hdf5_file.close()
        except:
            print("failed to load: ", hdf5_path)
            print(traceback.format_exc())
            pass
    global_dataset_updates(data_grp, total_samples, env_args)
    data_writer.close()
    return dataset_path, demo_count


def divide_hdf5(hdf5_path, num_divisions, hdf5_use_swmr=True):
    # Open the original HDF5 file
    hdf5_file = h5py.File(hdf5_path, "r", swmr=hdf5_use_swmr, libver="latest")
    hdf5_dir = os.path.dirname(hdf5_path)
    hdf5_name = os.path.splitext(os.path.basename(hdf5_path))[0]

    # Retrieve the dataset's key structure
    try:
        env_args = hdf5_file["data"].attrs["env_args"]
    except:
        env_args = None
    data_group = hdf5_file["data"]
    total_demos = len(data_group)

    # Determine how many demos to put into each divided file
    demos_per_file = total_demos // num_divisions
    extra_demos = total_demos % num_divisions

    demo_count = 0
    for i in tqdm(range(num_divisions)):
        # Create a new HDF5 file for each division
        division_path = os.path.join(hdf5_dir, f"{hdf5_name}_{i}.hdf5")
        data_writer = h5py.File(division_path, "w")
        data_grp = data_writer.create_group("data")

        # Number of demos to put in this file
        num_demos_in_file = demos_per_file + (1 if i < extra_demos else 0)

        # Copy demos from the original file to this divided file
        sub_demo_count = 0
        for j in tqdm(range(num_demos_in_file)):
            data_grp.copy(data_group[f"demo_{demo_count}"], f"demo_{sub_demo_count}")
            sub_demo_count += 1
            demo_count += 1

        global_dataset_updates(data_grp, num_demos_in_file, env_args)
        # Close the divided HDF5 file
        data_writer.close()
        print(f"Saved division {i} to {division_path}")

    # Close the original HDF5 file
    hdf5_file.close()
    print("Finished dividing the dataset.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5_paths", nargs="+")
    parser.add_argument("--output_path")
    parser.add_argument("--filter_key", type=str, default=None)
    parser.add_argument("--divide", type=int, default=0)

    args = parser.parse_args()
    if args.divide:
        divide_hdf5(args.hdf5_paths[0], args.divide)
    else:
        combine_hdf5(args.hdf5_paths, True, args.output_path, args.filter_key)
