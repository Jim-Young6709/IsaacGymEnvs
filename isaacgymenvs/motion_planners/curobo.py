import yaml
import time
import torch
import numpy as np
import logging
from robofin.robots import FrankaRobot
from scipy.spatial.transform import Rotation as R

from isaacgymenvs.motion_planners import MotionPlannerBase
try:
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.sphere_fit import SphereFitType
    from curobo.geom.types import Cuboid, Cylinder, Mesh, Sphere, WorldConfig
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
    def __init__(self, env):
        super().__init__(env)
        self._num_robot_points = 2048
        self._num_goal_robot_points = 2048
        self._num_obstacle_points = 4096
        self.in_hand = False
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
        n_cubes: int = 100,
        collision_spheres_for_in_hand: int = 300,
        mesh_mode: bool = True,
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

        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        ).get_obb_world()
        c_checker = CollisionCheckerType.PRIMITIVE
        c_cache = {"obb": n_cubes}
        if mesh_mode:
            c_checker = CollisionCheckerType.MESH
            c_cache = {"mesh": n_cubes}
            world_cfg = world_cfg.get_mesh_world()

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

    def get_actions(self, env_obs_dict, gt_info=None):
        joint_pos = env_obs_dict["joint_pos"]
        goal_joint_pos = env_obs_dict["goal_joint_pos"]
        current_robot_pcd = env_obs_dict["robot_pcd"]
        goal_robot_pcd = env_obs_dict["goal_robot_pcd"]
        static_obstacle_pcd = env_obs_dict["static_obstacle_pcd"]

        planning_actions_abs = []

        for i in range(self.num_envs):
            if gt_info is not None:
                (
                    planning_actions,
                    plan_log,
                ) = self.mp_curobo(joint_pos[i], goal_joint_pos[i], gt_info[i])
            # import ipdb ; ipdb.set_trace()

        return planning_actions_abs

    def get_actions_open_loop(self, env_obs_dict, gt_info=None):
        return self.get_actions(env_obs_dict, gt_info=gt_info)

    def mp_curobo(
        self,
        start_angles,
        target_angles,
        gt_info,
        mesh_mode=False,
        debug=False,
    ):
        """
        now force execute plan to true. TODO: fix this later
        """

        # initialize curobo planner if not already initialized

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

        t01 = time.time()

        # update world config
        self.curobo_planner.reset(reset_seed=False)
        if mesh_mode:
            world_config = WorldConfig(
                cuboid=cuboids,
                cylinder=cylinders,
                sphere=spheres,
                mesh=meshes,
            ).get_mesh_world(merge_meshes=False)
        else:
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
        )  # TODO: seems the number of attempts is gonna affect the TrajOpt part a lot

        if result.success:
            traj = result.get_interpolated_plan()
            planning_actions = traj.position.cpu().numpy()
            print("Len(path): ", len(planning_actions))

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

        print(plan_log)
        return (
            planning_actions,
            plan_log,
        )

    def reset(self):
        pass

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

