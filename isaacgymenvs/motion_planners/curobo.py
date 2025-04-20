import yaml
import time
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from robofin.robots import FrankaRobot
from scipy.spatial.transform import Rotation as R

from isaacgymenvs.motion_planners import MotionPlannerBase
try:
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.sphere_fit import SphereFitType
    from curobo.geom.types import Cuboid, Cylinder, Mesh, Sphere, VoxelGrid, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState, RobotConfig
    from curobo.util_file import (
        get_robot_configs_path,
        get_world_configs_path,
        join_path,
        load_yaml,
    )
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    curobo_logger = logging.getLogger("curobo")

    # Set its level to WARNING (or CRITICAL to silence it completely)
    curobo_logger.setLevel(logging.WARNING)
except:
    print("curobo not installed, skipping import")




class Curobo(MotionPlannerBase):
    def __init__(self, env, use_gt):
        super().__init__(env)
        self._num_robot_points = 2048
        self._num_goal_robot_points = 2048
        self._num_obstacle_points = 4096
        self._voxel_size = 0.05 # same as curobo's realsense example script
        self.in_hand = False
        self.use_gt = use_gt
        self.set_up_policy()
        self.profiling = {
            "formatting input": 0,
            "update world to curobo": 0,
            "setup in hand obj in curobo": 0,
            "curobo plan": 0,
            "pybullet execution": 0,
        }

    @property
    def num_robot_points(self):
        return self._num_robot_points

    @property
    def num_goal_robot_points(self):
        return self._num_goal_robot_points

    @property
    def num_obstacle_points(self):
        return self._num_obstacle_points

    def set_up_policy(
        self,
        n_cubes: int = 300,
        collision_spheres_for_in_hand: int = 300,
        collision_buffer: float = 0.0,
        parallel_finetune=True,
    ):
        """
        Modified from curobo benchmark example
        """
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        robot_cfg["kinematics"]["collision_sphere_buffer"] = collision_buffer
        robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_mesh.yml"
        robot_cfg["kinematics"]["ee_link"] = "panda_hand"

        if self.in_hand:
            robot_cfg["kinematics"]["extra_collision_spheres"] = {
                "attached_object": collision_spheres_for_in_hand
            }

        if self.use_gt:
            c_checker = CollisionCheckerType.PRIMITIVE
            c_cache = {"obb": n_cubes}
            world_cfg = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
            ).get_obb_world()
        else:
            c_checker = CollisionCheckerType.VOXEL
            c_cache = {"obb": self._num_obstacle_points}
            world_cfg = WorldConfig.from_dict(
                {
                    "voxel": {
                        "base": {
                            "dims": [2.0, 2.0, 2.0],
                            "pose": [0, 0, 0, 1, 0, 0, 0],
                            "voxel_size": self._voxel_size,
                            "feature_dtype": torch.bfloat16,
                        },
                    }
                }
            )

        robot_cfg_instance = RobotConfig.from_dict(robot_cfg, tensor_args=TensorDeviceType())

        K = robot_cfg_instance.kinematics.kinematics_config.joint_limits
        K.position[0, :] -= 0.2
        K.position[1, :] += 0.2
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg_instance,
            world_cfg,
            collision_checker_type=c_checker,
            collision_cache=c_cache,
        )
        mg = MotionGen(motion_gen_config)
        mg.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=parallel_finetune)
        self.curobo_planner = mg
        
        self.env_planning_success_flag = torch.ones(self.num_envs, dtype=bool)

    def get_actions_open_loop(self, env_obs_dict):
        joint_pos = env_obs_dict["joint_pos"]
        goal_joint_pos = env_obs_dict["goal_joint_pos"]
        current_robot_pcd = env_obs_dict["robot_pcd"]
        goal_robot_pcd = env_obs_dict["goal_robot_pcd"]
        static_obstacle_pcd = env_obs_dict["static_obstacle_pcd"]

        num_planning_success = 0
        planning_actions_abs = []

        gt_info = self.env.obstacle_configs
        if self.env.use_dynamic_obstacles:
            dynamic_gt_info = self.env.obstacle_spawner.generate_obstacle_gt()
        else:
            dynamic_gt_info = [None for i in range(self.num_envs)]

        # for i in tqdm(range(self.num_envs), desc="Curobo Planning"):
        for i in range(self.num_envs):
            if True: #self.env_planning_success_flag[i]:
                # only run environments where the plan succeeded
                if self.use_gt:
                    (
                        planning_actions,
                        plan_log,
                    ) = self.mp_curobo(joint_pos[i], goal_joint_pos[i], gt_info[i], dynamic_gt_info[i])
                else:
                    (
                        planning_actions,
                        plan_log,
                    ) = self.mp_curobo_pcd(joint_pos[i], goal_joint_pos[i], static_obstacle_pcd[i]) # TODO: add dynamic obstacle pcd
                if plan_log == "success":
                    planning_actions = torch.from_numpy(planning_actions).to(self.device)
                    num_planning_success += 1
                else:
                    self.env_planning_success_flag[i] = False
                    planning_actions = joint_pos[i].unsqueeze(0)

                padding_num = self.env.max_episode_length - planning_actions.shape[0]
                last_row = planning_actions[-1].unsqueeze(0).repeat(padding_num, 1)
                planning_actions = torch.cat([planning_actions, last_row], dim=0).unsqueeze(1)
                planning_actions_abs.append(planning_actions)
            else:
                planning_actions = joint_pos[i].unsqueeze(0)
                padding_num = self.env.max_episode_length - planning_actions.shape[0]
                last_row = planning_actions[-1].unsqueeze(0).repeat(padding_num, 1)
                planning_actions = torch.cat([planning_actions, last_row], dim=0).unsqueeze(1)
                planning_actions_abs.append(planning_actions)


        planning_actions_abs = torch.cat(planning_actions_abs, dim=1).to(self.device)
        print("Planning success rate: ", num_planning_success / self.num_envs)

        return planning_actions_abs

    def get_actions(self, env_obs_dict):
        # the first couple of steps generated by curobo is basically zero delta action
        start_id = 5
        return self.get_actions_open_loop(env_obs_dict)[start_id:start_id+15]

    def mp_curobo(
        self,
        start_angles,
        target_angles,
        gt_info,
        dynamic_gt_info=None,
    ):
        t00 = time.time()
        # formatting start and goal
        tensor_args = TensorDeviceType()
        start_state = JointState.from_position(
            tensor_args.to_device(start_angles).unsqueeze(0),
            joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
        )

        goal_SE3 = FrankaRobot.fk(target_angles.cpu().numpy(), eff_frame="panda_hand")
        goal_pose = Pose(
            position=tensor_args.to_device(np.array(goal_SE3._xyz)),
            quaternion=tensor_args.to_device(self.flip_quaternion(np.array(goal_SE3._so3.wxyz))),
        )

        # extract obstacle information and prepare world config (first testing without meshes)
        (
            cuboid_dims,
            cuboid_centers,
            cuboid_quats,
            *_,
        ) = gt_info


        cuboids = []
        cylinders = []
        spheres = []
        meshes = []

        for i in range(len(cuboid_dims)):
            # need to convert quaternions to wxyz format
            cuboids.append(
                Cuboid(
                    name=f"cuboid_{i}",
                    pose=[*cuboid_centers[i], *cuboid_quats[i, [3, 0, 1, 2]]],
                    dims=cuboid_dims[i].tolist(),
                )
            )
        static_cuboids_len = len(cuboids)

        if dynamic_gt_info is not None:
            # extract dynamic obstacle information and prepare world config (first testing without meshes)
            (
                cuboid_dims,
                cuboid_centers,
                cuboid_quats,
                *_,
            ) = dynamic_gt_info

            for i in range(len(cuboid_dims)):
                # need to convert quaternions to wxyz format
                cuboids.append(
                    Cuboid(
                        name=f"cuboid_{i+static_cuboids_len}",
                        pose=[*cuboid_centers[i], *cuboid_quats[i, [3, 0, 1, 2]]],
                        dims=cuboid_dims[i].tolist(),
                    )
                )


        t01 = time.time()

        # update world config
        self.curobo_planner.reset(reset_seed=False)
        world_config = WorldConfig(
            cuboid=cuboids,
            cylinder=cylinders,
            sphere=spheres,
            mesh=meshes,
        ).get_obb_world()
        self.curobo_planner.world_coll_checker.clear_cache()
        self.curobo_planner.update_world(world_config)

        t02 = time.time()

        # setup in hand object, worry about this later when implementing in hand
        # self.curobo_planner.detach_object_from_robot()

        t03 = time.time()
        plan_config = MotionGenPlanConfig(max_attempts=20)
        result = self.curobo_planner.plan_single(
            start_state, goal_pose, plan_config
        )

        if result.success:
            traj = result.get_interpolated_plan()
            planning_actions = traj.position.cpu().numpy()
            planning_success = (result.position_error < 0.01) and (
                result.rotation_error < 15
            )
            if planning_success:
                plan_log = "success"
            else:
                plan_log = "failed to reach the goal"
        else:
            plan_log = result.status.value
            planning_actions = None

        t04 = time.time()

        self.profiling["formatting input"] += t01 - t00
        self.profiling["update world to curobo"] += t02 - t01
        self.profiling["setup in hand obj in curobo"] += t03 - t02
        self.profiling["curobo plan"] += t04 - t03

        return (
            planning_actions,
            plan_log,
        )

    def mp_curobo_pcd(
        self,
        start_angles,
        target_angles,
        obstacle_pcd,
        debug=False,
    ):
        t00 = time.time()
        # formatting start and goal
        tensor_args = TensorDeviceType()
        start_state = JointState.from_position(
            tensor_args.to_device(start_angles).unsqueeze(0),
            joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
            ],
        )

        goal_SE3 = FrankaRobot.fk(target_angles.cpu().numpy(), eff_frame="panda_hand")
        goal_pose = Pose(
            position=tensor_args.to_device(np.array(goal_SE3._xyz)),
            quaternion=tensor_args.to_device(self.flip_quaternion(np.array(goal_SE3._so3.wxyz))),
        )

        t01 = time.time()

        # pcd to voxel & update world
        cuboids = []
        for i in range(len(obstacle_pcd)):
            cuboids.append(
                Cuboid(
                    name=f"cuboid_{i}",
                    pose=[*obstacle_pcd[i, :3].cpu().numpy(), 1, 0, 0, 0],
                    dims=[0.001, 0.001, 0.001],
                )
            )

        world_config = WorldConfig(
            cuboid=cuboids,
            cylinder=[],
            sphere=[],
            mesh=[],
        ).get_obb_world()
        self.curobo_planner.world_coll_checker.clear_cache()
        self.curobo_planner.update_world(world_config)

        if debug:
            voxel_occu = self.curobo_planner.world_collision.get_occupancy_in_bounding_box(
                voxel_size=self._voxel_size,
                cuboid=Cuboid(name="test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[2, 2, 2])
            )
            print("Voxels reload from the collision world")
            self.plot_voxels(voxel_occu)

        t02 = time.time()

        # setup in hand object, worry about this later when implementing in hand
        # self.curobo_planner.detach_object_from_robot()

        t03 = time.time()
        plan_config = MotionGenPlanConfig(max_attempts=20)
        result = self.curobo_planner.plan_single(
            start_state, goal_pose, plan_config
        )

        if result.success:
            traj = result.get_interpolated_plan()
            planning_actions = traj.position.cpu().numpy()
            planning_success = (result.position_error < 0.01) and (
                result.rotation_error < 15
            )
            if planning_success:
                plan_log = "success"
            else:
                plan_log = "failed to reach the goal"
        else:
            plan_log = result.status.value
            planning_actions = None

        t04 = time.time()

        self.profiling["formatting input"] += t01 - t00
        self.profiling["update world to curobo"] += t02 - t01
        self.profiling["setup in hand obj in curobo"] += t03 - t02
        self.profiling["curobo plan"] += t04 - t03

        return (
            planning_actions,
            plan_log,
        )

    def reset(self):
        self.env_planning_success_flag[:] = True

    @staticmethod
    def flip_quaternion(quat):
        """_summary_

        Args:
            quat (np.ndarray): in wxyz format
        """
        ori = R.from_quat(quat[[1, 2, 3, 0]])
        ori_euler = ori.as_euler("XYZ")
        ori_euler[2] -= np.pi
        flip = R.from_euler("XYZ", ori_euler)
        flip_quat = flip.as_quat()[[3, 0, 1, 2]]

        return flip_quat

    @staticmethod
    def voxelgrid_from_point_cloud(point_cloud, voxel_size, pose=[0.0, 0, 0.0, 1, 0, 0, 0], dims=[2.0, 2.0, 2.0]):
        voxel_grid = VoxelGrid(name='voxel_pcd', pose=pose, dims=dims, voxel_size=voxel_size)
        grid_shape, low, high = voxel_grid.get_grid_shape()
        num_voxels = np.prod(grid_shape)

        # Create an empty feature tensor
        feature_tensor = torch.ones(num_voxels, device=point_cloud.device) * -100
        # feature_tensor = torch.Tensor(range(num_voxels)) / 1000
        # Create a mapping from points to voxel indices
        indices = ((point_cloud - torch.tensor(low, device=point_cloud.device)) / voxel_size).int()

        # Filter indices that fall within the grid shape
        valid_indices = (indices >= 0) & (indices < torch.tensor(grid_shape, device=point_cloud.device))
        valid_indices = torch.all(valid_indices, axis=1)
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

    @staticmethod
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


