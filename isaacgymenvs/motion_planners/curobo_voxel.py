import time
import torch
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# cuRobo
from scipy.spatial.transform import Rotation as R_sci
from curobo.geom.types import VoxelGrid, WorldConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType

def flip_quaternion(quat):
    """_summary_

    Args:
        quat (np.ndarray): in wxyz format
    """
    ori = R_sci.from_quat(quat[[1,2,3,0]])
    ori_euler = ori.as_euler("XYZ")
    ori_euler[2] -= np.pi
    flip = R_sci.from_euler("XYZ", ori_euler)
    flip_quat = flip.as_quat()[[3,0,1,2]]

    return flip_quat

def voxelgrid_from_point_cloud(point_cloud, pose, dims, voxel_size):
    voxel_grid = VoxelGrid(name='voxel_pcd', pose=pose, dims=dims, voxel_size=voxel_size)
    grid_shape, low, high = voxel_grid.get_grid_shape()
    num_voxels = np.prod(grid_shape)

    # Create an empty feature tensor
    feature_tensor = torch.ones(num_voxels) * -100
    # feature_tensor = torch.Tensor(range(num_voxels)) / 1000
    # Create a mapping from points to voxel indices
    indices = ((point_cloud - np.array(low)) / voxel_size).astype(int)

    # Filter indices that fall within the grid shape
    valid_indices = (indices >= 0) & (indices < np.array(grid_shape))
    valid_indices = np.all(valid_indices, axis=1)
    indices = indices[valid_indices]

    # Calculate the voxel index in the feature_tensor
    voxel_indices = (
        indices[:, 0] * (grid_shape[1] * grid_shape[2]) +
        indices[:, 1] * grid_shape[2] +
        indices[:, 2]
    )

    # Mark occupied voxels
    feature_tensor[voxel_indices] = 100  # or some other feature value
    xyzr_tensor = voxel_grid.create_xyzr_tensor()

    voxel_grid.feature_tensor = feature_tensor
    voxel_grid.xyzr_tensor = xyzr_tensor
    # Return the populated VoxelGrid object
    return voxel_grid

def plot_voxels(voxel_grid: VoxelGrid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get the indices of the occupied voxels
    indices = torch.nonzero(voxel_grid.feature_tensor > 0, as_tuple=False)[:, 0]
    indices = indices.cpu().numpy()

    xyz_occu = voxel_grid.xyzr_tensor[indices]
    xyz_occu = xyz_occu.cpu().numpy()
    # Plot the occupied voxels
    ax.scatter(xyz_occu[:, 0], xyz_occu[:, 1], xyz_occu[:, 2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

def plot_voxels_from_array(voxel_array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(voxel_array[:, 0], voxel_array[:, 1], voxel_array[:, 2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument(
    "-v",
    "--voxel-size",
    type=float,
    default=0.05,
)
parser.add_argument(
    "--task",
    type=str,
    default="test_curobo",
    help="Specify the config set for testing",
)
args = parser.parse_args()
tensor_args = TensorDeviceType()

task_set = np.load(f"./cabinet_scene1_panda.npy")
point_cloud = np.load(f"./cabinet_scene1_pcd.npy")

voxel_pcd = voxelgrid_from_point_cloud(point_cloud, [0.0, 0, 0.0, 1, 0, 0, 0],  [2.0, 2.0, 2.0], voxel_size=args.voxel_size)
voxels = [voxel_pcd]

print("Voxels created from the point cloud")
# plot_voxels(voxel_pcd)

world_config = WorldConfig(voxel=voxels)
motion_gen_config = MotionGenConfig.load_from_robot_config(
    "franka.yml",
    world_config,
    interpolation_dt=0.01,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()

# retract_cfg = motion_gen.get_retract_config()

# state = motion_gen.rollout_fn.compute_kinematics(
#     JointState.from_position(retract_cfg.view(1, -1))
# )

motion_gen.world_collision.update_voxel_features(features=voxel_pcd.feature_tensor.unsqueeze(1), name=voxel_pcd.name, env_idx=0)
voxel_in=motion_gen.world_collision.get_voxel_grid('voxel_pcd')
voxel_occu = motion_gen.world_collision.get_occupancy_in_bounding_box(voxel_size=args.voxel_size)

print("Voxels reload from the collision world")
# plot_voxels(voxel_occu)

# import ipdb ; ipdb.set_trace()

planning_time = 0

trajs = []

for i in range(4):
    ee_translation_goal = task_set[1, i+1, :3]#cube_position
    ee_orientation_teleop_goal = flip_quaternion(task_set[1, i+1, 3:])#cube_orientation
    start_js_pos = task_set[0, i]
    goal_pose = Pose(
        position=tensor_args.to_device(ee_translation_goal),
        quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
    )

    start_state = JointState.from_position(
        tensor_args.to_device(start_js_pos),
        joint_names=[
            'panda_joint1',
            'panda_joint2',
            'panda_joint3',
            'panda_joint4',
            'panda_joint5',
            'panda_joint6',
            'panda_joint7',
        ]
    )

    t1 = time.time()
    result = motion_gen.plan_single(start_state.unsqueeze(0), goal_pose, MotionGenPlanConfig(enable_graph=False, enable_graph_attempt=4, max_attempts=2, enable_finetune_trajopt=True))
    # import ipdb ; ipdb.set_trace()
    if result.path_buffer_last_tstep is None:
        print("failed to generate anything")
    elif len(result.path_buffer_last_tstep) > 1:
        cmd_plan = result.interpolated_plan[1]
        print("failed to generate plan")
    else:
        cmd_plan = result.get_interpolated_plan()  # result.interpolation_dt has the dt between timesteps
    final_plan = cmd_plan.position.cpu().numpy()
    t2 = time.time()
    print("Trajectory Generation Success: ", result.success)
    planning_time += (t2-t1)
    trajs.append(final_plan)

planning_time /= 4

with open(f'./planning_output/{args.task}.pkl', 'wb') as f:
    pickle.dump(trajs, f)

print("Average planning time: ", planning_time)