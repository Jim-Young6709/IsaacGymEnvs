import yaml
import time
import torch
import numpy as np
from pathlib import Path
from hydra.utils import instantiate

from isaacgymenvs.motion_planners import MotionPlannerBase
try:
    import logging
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
        self.set_up_policy()

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
        robot_cfg["kinematics"]["extra_collision_spheres"] = {
            "attached_object": collision_spheres_for_in_hand
        }
        robot_cfg["kinematics"]["ee_link"] = "panda_hand"

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

        if gt_info is not None:
            (
                planning_actions,
                planning_success,
                failure_reason,
            ) = self.mp_curobo(joint_pos, goal_joint_pos)
            return (
                planning_actions,
                planning_success,
                failure_reason,
            )

        nn_pcd_obs = self._prepare_neuralmp_observation(obstacle_pcd=static_obstacle_pcd, goal_robot_pcd=goal_robot_pcd, current_robot_pcd=None)
        # roll out open loop
        open_loop_steps = 1
        open_loop_joint_pos = joint_pos.clone()
        for i in range(open_loop_steps):
            current_robot_pcd = self.env.get_robot_pcds(open_loop_joint_pos)
            nn_pcd_obs = self._update_neuralmp_robot_pcd_observation(nn_pcd_obs, current_robot_pcd)
            obs_dict = {
                "compute_pcd_params": nn_pcd_obs,       # (num_envs, NUM_TOTAL_POINTS, 4)
                "current_angles": open_loop_joint_pos,  # (num_envs, 7)
                "goal_angles": goal_joint_pos,          # (num_envs, 7)
            }
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.float16):
                    delta_joint_pos_action = self.model.get_action_robomimic(obs_dict)
            open_loop_joint_pos += delta_joint_pos_action * 1.0
        joint_pos_target = open_loop_joint_pos
        return joint_pos_target

    def mp_curobo(
        self,
        start_angles,
        target_angles,
        mesh_mode=True,
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

        goal_SE3 = FrankaRobot.fk(target_angles, eff_frame="panda_hand")
        goal_pose = Pose(
            position=tensor_args.to_device(np.array(goal_SE3._xyz)),
            quaternion=tensor_args.to_device(flip_quaternion(np.array(goal_SE3._so3.wxyz))),
        )

        # extract obstacle information and prepare world config (first testing without meshes)
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
            mesh_quaternions,
            obj_ids,
            mesh_ids,
        ) = decompose_scene_pcd_params_obs(self.scene_pcd_params)

        max_len = int(self.scene_pcd_params[0])
        cuboids = []
        cylinders = []
        spheres = []
        meshes = []

        for i in range(max_len):
            # need to convert quaternions to wxyz format
            if sum(cuboid_dims[i]) != 0:
                cuboids.append(
                    Cuboid(
                        name=f"cuboid_{i}",
                        pose=[*cuboid_centers[i], *cuboid_quats[i, [3, 0, 1, 2]]],
                        dims=cuboid_dims[i].tolist(),
                    )
                )
            if cylinder_radii[i] != 0:
                cylinders.append(
                    Cylinder(
                        name=f"cylinder_{i}",
                        pose=[*cylinder_centers[i], *cylinder_quats[i, [3, 0, 1, 2]]],
                        radius=cylinder_radii[i],
                        height=cylinder_heights[i],
                    )
                )
            if sphere_radii[i] != 0:
                spheres.append(
                    Sphere(
                        name=f"sphere_{i}",
                        pose=[*sphere_centers[i], 1, 0, 0, 0],
                        radius=sphere_radii[i],
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

        # setup in hand object
        if self.in_hand_obj:
            in_hand_obj_pose = self.get_in_hand_obj_state()  # xyz, xyzw
            in_hand_obj_pose = in_hand_obj_pose[[0, 1, 2, 6, 3, 4, 5]]  # xyz, wxyz

            in_hand_type = ["box", "cylinder", "sphere", "mesh"][int(self.in_hand_type_idx)]

            if in_hand_type == "box":
                in_hand_obj = Cuboid(
                    name="in_hand_obj", pose=[*in_hand_obj_pose], dims=self.in_hand_size
                )
            elif in_hand_type == "cylinder":
                in_hand_obj = Cylinder(
                    name="in_hand_obj",
                    pose=[*in_hand_obj_pose],
                    radius=self.in_hand_size[0] / 2,
                    height=self.in_hand_size[1],
                )
            elif in_hand_type == "sphere":
                in_hand_obj = Sphere(
                    name="in_hand_obj", pose=[*in_hand_obj_pose], radius=self.in_hand_size[0] / 2
                )

            self.curobo_planner.attach_external_objects_to_robot(
                start_state,
                [in_hand_obj],
                # surface_sphere_radius=0.01,
                sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
            )
        else:
            self.curobo_planner.detach_object_from_robot()
        t03 = time.time()
        plan_config = MotionGenPlanConfig(max_attempts=20)
        result = self.curobo_planner.plan_single(
            start_state, goal_pose, plan_config
        )  # TODO: seems the number of attempts is gonna affect the TrajOpt part a lot
        solved_prob = result.success
        fail_info = None
        if solved_prob:
            traj = result.get_interpolated_plan()
            converted_path = traj.position.cpu().numpy()
            print("Len(path): ", len(converted_path))
        else:
            fail_info = result.status.value
            converted_path = []

        t04 = time.time()

        self.profiling["formatting input"] += t01 - t00
        self.profiling["update world to curobo"] += t02 - t01
        self.profiling["setup in hand obj in curobo"] += t03 - t02
        self.profiling["curobo plan"] += t04 - t03

        if not result.success:
            success = "failed to plan"
            failure_reason = "planner"
            planning_actions = None
        else:
            planning_success = (result.position_error < 0.01) and (
                result.rotation_error < 5
            )  # check if errors are within 1cm and 15 degrees

        print(fail_info)
        return (
            planning_actions,
            planning_success,
            failure_reason,
        )

    def _prepare_neuralmp_observation(self, obstacle_pcd, goal_robot_pcd, current_robot_pcd=None): 
        # (num_envs, num_points, 4)
        nn_pcd_obs = torch.cat((
                torch.zeros(self.num_robot_points, 4), # mask robot pcd with 0
                torch.ones(self.num_obstacle_points, 4), # mask obstacle pcd with 1
                2 * torch.ones(self.num_robot_points, 4), # mask goal obstacle pcd with 2
        ), dim=0).unsqueeze(0).repeat(self.num_envs, 1, 1).cuda()


        # add robot points
        if current_robot_pcd is not None:
            nn_pcd_obs[:, 0:self.num_robot_points, 0:3] = current_robot_pcd
        
        # add obstacle points
        nn_pcd_obs[:, self.num_robot_points:self.num_robot_points+self.num_obstacle_points, 0:3] = obstacle_pcd

        # add goal robot points
        nn_pcd_obs[:, self.num_robot_points+self.num_obstacle_points:, 0:3] = goal_robot_pcd
        return nn_pcd_obs


    def _update_neuralmp_robot_pcd_observation(self, nn_pcd_obs, current_robot_pcd):
        nn_pcd_obs[:, 0:self.num_robot_points, 0:3] = current_robot_pcd
        return nn_pcd_obs

    def reset(self):
        pass



