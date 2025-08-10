"""
Franka + LEAP Hand Pick Env
"""

import time

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.utils.pcd_utils import *
from isaacgymenvs.utils.rotation_conversions import *
from isaacgymenvs.tasks import FrankaLEAP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from tqdm import tqdm



class FrankaLEAPPickTop(FrankaLEAP):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.scene_box_cfg = cfg["env"]["scene"]["safety_box"]
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

        if self.eef_init["enable"]:
            dis_open_range = self.eef_init["dis_open_range"]
            dis_open = torch.rand(self.num_envs, device=self.device) * (dis_open_range[1] - dis_open_range[0]) + dis_open_range[0]
            dis_side_range = self.eef_init["dis_side_range"]
            dis_side_x = torch.rand(self.num_envs, device=self.device) * (self.box_dims[:, 0] + 2*dis_side_range) - (self.box_dims[:, 0]/2 + dis_side_range)
            dis_side_y = torch.rand(self.num_envs, device=self.device) * (self.box_dims[:, 1] + 2*dis_side_range) - (self.box_dims[:, 1]/2 + dis_side_range)

            eef_init_pos = self.box_pos.clone()
            eef_init_pos[:, 0] += dis_side_x
            eef_init_pos[:, 1] += dis_side_y
            eef_init_pos[:, 2] += dis_open + self.box_dims[:, 2]

            eef_init_quat = A2B_quaternion(eef_init_pos, self.box_pos, max_angle_deg=20)
            flip_idx = eef_init_pos[:, 0] < self.box_pos[:, 0]
            rot_local_z_180 = torch.tensor([[0.0, 0.0, 1.0, 0.0]]*sum(flip_idx), device=self.device)  # 180 degrees around local z-axis
            eef_init_quat[flip_idx] = quat_mul(eef_init_quat[flip_idx], rot_local_z_180)  # rotate by 180 degrees around local z-axis

            eef_init_pos7 = torch.cat((eef_init_pos, eef_init_quat), dim=-1)  # (num_envs, 7)

            # TODO: resampling mechanism here when IK failed
            self.canonical_joint_config[:, :7] = self.get_joint_from_ee(eef_init_pos7)

    def _create_envs(self, spacing, num_per_row):
        """
        loading Franka + LEAP + a table in the environment, this is for debugging purposes only
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # setup params
        self.box_dims = []
        self.box_pos = []
        self.box_quats = []

        self.cuboid_dims = []  # xyz
        self.cuboid_pos = []
        self.cuboid_quats = []

        self.mesh_aabb_extents = None  # xyz, axis-aligned bounding box full extents
        self.table_surface_height = torch.zeros((self.num_envs,), device=self.device)
        self.obj_pos_range = torch.zeros((self.num_envs, 4), device=self.device) # x-min, x-max, y-min, y-max
        self.obj_pos_target = torch.zeros((self.num_envs, 3), device=self.device) # x, y, z

        # setup robot (franka + leap)
        robot_dof_props = self._create_franka_leap()
        robot_asset = self.robot_asset
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # make sure robot spawns at the origin, this matches the IK setting with cuRobo
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        max_agg_bodies = num_robot_bodies + 5 + 1  # 5 for box, 1 for object
        max_agg_shapes = num_robot_shapes + 5 + 1  # 5 for box, 1 for object

        self.robots = []
        self.objects = []
        self.env_ptrs = []
        self._object_center_init_state = torch.zeros((self.num_envs, 3), device=self.device)

        # load all meshes first
        all_meshes_list = self.create_all_meshes()

        # Create environments
        for i in tqdm(range(self.num_envs), desc="Creating Envs"):
            # grasp object
            object_asset, object_start_pose, object_scale, object_id, mesh_id = all_meshes_list[i % len(all_meshes_list)]

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create robot (franka + leap)
            robot_actor = self.gym.create_actor(
                env_ptr, robot_asset, robot_start_pose, "franka", i, 0, 0
            )
            self.gym.set_actor_dof_properties(env_ptr, robot_actor, robot_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create box workspace
            self._box_wall_ids = []
            box_info = self._create_box()
            for wall_idx, (wall_asset, wall_start_pose) in enumerate(box_info):
                wall_actor = self.gym.create_actor(
                    env_ptr, wall_asset, wall_start_pose, f"box_wall{wall_idx}", i, 1, 0
                )
                self._box_wall_ids.append(wall_actor)

            self.gym.set_rigid_body_color(
                env_ptr,
                self._box_wall_ids[0],
                0,  # body index; 0 for single-body assets like cubes
                gymapi.MESH_VISUAL_AND_COLLISION,
                gymapi.Vec3(1.0, 0.0, 0.0)  # RGB in [0, 1]
            )

            # Create object
            self._object_id = self.gym.create_actor(
                env_ptr, object_asset, object_start_pose, "object", i, 2, 0
            )
            self._object_center_init_state[i, :3] = torch.tensor([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z], device=self.device)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.env_ptrs.append(env_ptr)
            self.robots.append(robot_actor)
            self.objects.append(self._object_id)

            # object pcd
            object_pcd_i = torch.from_numpy(compute_scene_oracle_pcd(
                num_obstacle_points=self.pcd_spec_dict["num_object_points"],
                mesh_position=np.array([[0.0, 0.0, 0.0]]),
                mesh_scale=np.array([object_scale]),
                mesh_quaternion=np.array([[0.0, 0.0, 0.0, 1.0]]),
                obj_id=np.array([object_id]),
                mesh_id=np.array([mesh_id]),
                meshes_dir=self.mesh_args["mesh_dir"],
            )).to(self.device)
            self.object_pcds.append(object_pcd_i)

        self.cuboid_dims = np.array(self.cuboid_dims).reshape(self.num_envs, -1, 3)
        self.cuboid_pos = np.array(self.cuboid_pos).reshape(self.num_envs, -1, 3)
        self.cuboid_quats = np.array(self.cuboid_quats).reshape(self.num_envs, -1, 4)

        for i in range(self.num_envs):
            self.table_surface_height[i] = self.cuboid_pos[i, 0, 2] + self.cuboid_dims[i, 0, 2] / 2
            self.obj_pos_range[i, 0] = self.cuboid_pos[i, 0, 0] - self.cuboid_dims[i, 0, 0] / 2 # x-min
            self.obj_pos_range[i, 1] = self.cuboid_pos[i, 0, 0] + self.cuboid_dims[i, 0, 0] / 2 # x-max
            self.obj_pos_range[i, 2] = self.cuboid_pos[i, 0, 1] - self.cuboid_dims[i, 0, 1] / 2 # y-min
            self.obj_pos_range[i, 3] = self.cuboid_pos[i, 0, 1] + self.cuboid_dims[i, 0, 1] / 2 # y-max

            # static pcd
            static_pcd_i = torch.from_numpy(compute_scene_oracle_pcd(
                num_obstacle_points=self.pcd_spec_dict["num_static_points"],
                cuboid_dims=np.array(self.cuboid_dims[i]),
                cuboid_centers=np.array(self.cuboid_pos[i]),
                cuboid_quats=np.array(self.cuboid_quats[i]),
            )).to(self.device)
            self.static_pcds.append(static_pcd_i)

        self.box_dims = torch.tensor(self.box_dims, device=self.device) # (num_envs, 3)
        self.box_pos = torch.tensor(self.box_pos, device=self.device) # (num_envs, 3)
        self.box_quats = torch.tensor(self.box_quats, device=self.device)

        self.obj_pos_target[:, :2] = self.box_pos[:, :2]
        self.obj_pos_target[:, 2] = self.box_pos[:, 2] + self.box_dims[:, 2]

        self.cuboid_dims = torch.from_numpy(self.cuboid_dims).to(self.device)
        self.cuboid_pos = torch.from_numpy(self.cuboid_pos).to(self.device)
        self.cuboid_quats = torch.from_numpy(self.cuboid_quats).to(self.device)

        self.static_pcds = torch.stack(self.static_pcds, dim=0).to(self.device).to(torch.float32) # (num_envs, num_points, 3)
        self.object_pcds = torch.stack(self.object_pcds, dim=0).to(self.device).to(torch.float32)
        self.combined_pcds = torch.cat([self.static_pcds, self.object_pcds], dim=1).to(self.device) # (num_envs, num_static_points + num_object_points, 3)

        # get mesh AABB (axis-aligned bounding box) extents
        min_xyz = self.object_pcds.min(axis=1).values
        max_xyz = self.object_pcds.max(axis=1).values
        self.mesh_aabb_extents = max_xyz - min_xyz
        self._object_center_init_state[:, 2] += self.mesh_aabb_extents[:, 2] / 2

        # refine obj_rand_pos_range based on mesh AABB
        self.obj_pos_range[:, 0] += (self.mesh_aabb_extents[:, 0] / 2 + self.scene_box_cfg["obj_wall_tol"])
        self.obj_pos_range[:, 1] -= (self.mesh_aabb_extents[:, 0] / 2 + self.scene_box_cfg["obj_wall_tol"])
        self.obj_pos_range[:, 2] += (self.mesh_aabb_extents[:, 1] / 2 + self.scene_box_cfg["obj_wall_tol"])
        self.obj_pos_range[:, 3] -= (self.mesh_aabb_extents[:, 1] / 2 + self.scene_box_cfg["obj_wall_tol"])

        # Setup data
        actor_num = 1 + 5 + 1  # robot, box, object
        self.init_data(actor_num=actor_num)

    def init_data(self, actor_num):
        super().init_data(actor_num)
        self.reward_settings["target_pos"] = self.obj_pos_target

    def _create_box(self):
        wall_thickness = self.scene_box_cfg["wall_thickness"]
        size_range = self.scene_box_cfg["size"]
        size = np.random.uniform(size_range[0], size_range[1]) # inner size of the box

        # for simple debugging scenario training
        x_shift = self.scene_box_cfg["x_shift"]

        self.box_dims.append(size.tolist())
        self.box_pos.append([x_shift, 0.0, 0.0])
        self.box_quats.append([0.0, 0.0, 0.0, 1.0])

        walls = [
            # bottom wall
            self._create_cube(
                pos=[x_shift+0.0, 0.0, -wall_thickness/2],
                size=[size[0]+wall_thickness*2, size[1]+wall_thickness*2, wall_thickness],
            ),
            # left wall
            self._create_cube(
                pos=[x_shift+0.0, size[1]/2 + wall_thickness/2, size[2]/2],
                size=[size[0]+wall_thickness*2, wall_thickness, size[2]],
            ),
            # right wall
            self._create_cube(
                pos=[x_shift+0.0, -size[1]/2 - wall_thickness/2, size[2]/2],
                size=[size[0]+wall_thickness*2, wall_thickness, size[2]],
            ),
            # front wall
            self._create_cube(
                pos=[x_shift+-size[0]/2 - wall_thickness/2, 0.0, size[2]/2],
                size=[wall_thickness, size[1], size[2]],
            ),
            # back wall
            self._create_cube(
                pos=[x_shift+size[0]/2 + wall_thickness/2, 0.0, size[2]/2],
                size=[wall_thickness, size[1], size[2]],
            ),
        ]

        return walls

    def _reset_box_state(self):
        z_shift = self.scene_box_cfg["z"]
        r_rot_ex = self.scene_box_cfg["r"]
        rot_in = self.scene_box_cfg["rot_in"]
        rot_ex = self.scene_box_cfg["rot_ex"]

    def check_robot_collision(self):
        super().check_robot_collision()
        self.box_collision = torch.any(self.contact_forces[:, 31:35].view(self.num_envs, -1) != 0, dim=1)

    def compute_observations(self):
        self._refresh()

        obs_components = ["q_hand",
                          "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                          "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                          "box_to_eef_pos", "box_dims", "box_to_eef_rot_6d",
                          "hand_to_object", "object_to_eef_rot_6d", "object_to_target"]#, "object_target_6d_diff"]

        states_components = ["q", "qd",
                             "eef_pos", "eef_rot_6d", "eef_vel",
                             "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                             "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                             "box_to_eef_pos", "box_dims", "box_to_eef_rot_6d",
                             "object_center_pos", "object_rot_6d",
                             "hand_to_object", "object_to_eef_rot_6d", "object_to_target"]#, "object_target_6d_diff"]

        obs_buf = torch.cat([self.states[ob] for ob in obs_components], dim=-1)
        states_buf = torch.cat([self.states[st] for st in states_components], dim=-1)

        # TODO： convert box to a local region
        obs_buf = torch.cat([obs_buf, self.mesh_aabb_extents], dim=-1)
        states_buf = torch.cat([states_buf, self.mesh_aabb_extents], dim=-1)

        self.obs_buf = obs_buf
        self.states_buf = states_buf

        return self.obs_buf

    def compute_reward(self):
        self.reset_buf[:] = torch.where((self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf[self.states['object_center_pos'][:, 2] < self.table_surface_height-0.1] = 1

        if self.scene_box_cfg["colli_reset"]:
            self.reset_buf[self.box_collision] = 1

        reward_dict = compute_franka_leap_reward(self.states, self.reward_settings)

        self.rew_buf[:] = reward_dict["r_total"]
        self.extras["sep_reward/r_hand_obj"] = torch.mean(reward_dict["r_hand_obj"]).item()
        self.extras["sep_reward/r_obj_goal"] = torch.mean(reward_dict["r_obj_goal"]).item()
        self.extras["sep_reward/r_lift"] = torch.mean(reward_dict["r_lift"]).item()
        self.extras["sep_reward/r_curl"] = torch.mean(reward_dict["r_curl"]).item()
        self.extras["dis/d_hand_obj"] = torch.mean(reward_dict["d_hand_obj"]).item()
        self.extras["dis/d_lift"] = torch.mean(reward_dict["d_lift"]).item()

        # log metrics
        success_5cm_per_step = (reward_dict["d_obj_goal"] < 0.05)
        self.success_flags[success_5cm_per_step] = 1
        lifting_5cm_per_step = (reward_dict["d_lift"] > 0.05)
        self.lifting_flags[lifting_5cm_per_step] = 1
        self.extras["metrics/success_rate_5cm_per_step"] = torch.mean(success_5cm_per_step.float()).item()
        self.extras["metrics/lifting_rate_5cm_per_step"] = torch.mean(lifting_5cm_per_step.float()).item()
        self.extras["metrics/success_rate_5cm_per_ep"] = torch.mean(self.success_flags).item()
        self.extras["metrics/lifting_rate_5cm_per_ep"] = torch.mean(self.lifting_flags).item()

    def set_viewer(self):
        super().set_viewer(
            pos=[1.0, 0.0, 1.0],
            target=[0.5, 0.0, 0.1],
        )


@torch.jit.script
def compute_franka_leap_reward(states, reward_settings):
    # type: (Dict[str, Tensor], Dict[str, Tensor]) -> Dict[str, Tensor]

    # R1: Hand (palm, fingers) to object distance
    d_palm = torch.norm(states["object_center_pos"] - states["eef_pos"], dim=-1)
    d_finger1 = torch.norm(states["object_center_pos"] - states["eef_finger1_pos"], dim=-1)
    d_finger2 = torch.norm(states["object_center_pos"] - states["eef_finger2_pos"], dim=-1)
    d_finger3 = torch.norm(states["object_center_pos"] - states["eef_finger3_pos"], dim=-1)
    d_finger4 = torch.norm(states["object_center_pos"] - states["eef_finger4_pos"], dim=-1)

    # R1: Max dist component to object: max_i∈{palm_pos,fingertips} ||x^i - x^obj||
    d_hand_obj = torch.stack([d_palm, d_finger1, d_finger2, d_finger3, d_finger4], dim=1)
    d_hand_obj = torch.max(d_hand_obj, dim=1)[0]

    # R1: Hand object distance reward
    beta_hand_object = reward_settings["beta_hand_object"]
    r_hand_obj = torch.exp(-beta_hand_object * d_hand_obj)

    # R2: Lifting bonus: r_lift = 1.0 if object is lifted
    target_pos = reward_settings["target_pos"].squeeze(-1)
    object_height = states["object_center_pos"][:, 2] - reward_settings["object_init_height"].squeeze(-1)
    beta_lift = reward_settings["beta_lift"]
    if beta_lift > 0:
        object_vertical_err = torch.abs(states["object_center_pos"][:, 2] - target_pos[2])
        r_lift = torch.exp(-beta_lift * object_vertical_err)
        r_lift = torch.where(object_height > reward_settings["lift_threshold"], r_lift, 0.0)
    else:
        r_lift = torch.where(object_height > reward_settings["lift_threshold"], 1.0, torch.zeros_like(object_height))

    # R3: Object goal distance reward
    d_obj_goal = torch.norm(states["object_center_pos"] - target_pos, dim=-1)
    beta_object_goal = reward_settings["beta_object_goal"]
    r_obj_goal = torch.exp(-beta_object_goal * d_obj_goal)
    if reward_settings["lift_thres_for_obj_goal"]:
        r_obj_goal = torch.where(object_height > reward_settings["lift_threshold"], r_obj_goal, 0.0)

    # R4: Finger curl
    hand_dof_pos = states["q"][:, 7:] # hand joint angles
    near_object = (d_hand_obj <= reward_settings["curl_reaching_threshold"])
    finger_pos_diff = torch.sum((hand_dof_pos - reward_settings["grasp_finger_dof_pos"]) ** 2, dim=1)

    beta_curl = reward_settings["beta_curl"]
    r_curl= torch.exp(-beta_curl * finger_pos_diff)
    r_curl = torch.where(near_object, r_curl, 0.0)


    w_hand_obj = reward_settings["w_hand_obj"]
    w_obj_goal = reward_settings["w_obj_goal"]
    w_lift = reward_settings["w_lift"]
    w_curl = reward_settings["w_curl"]

    r_total = w_hand_obj*r_hand_obj + w_obj_goal*r_obj_goal + w_lift*r_lift + w_curl*r_curl

    rewards = {
        "r_hand_obj": w_hand_obj*r_hand_obj,
        "r_lift": w_lift*r_lift,
        "r_obj_goal": w_obj_goal*r_obj_goal,
        "r_curl": w_curl*r_curl,
        "r_total": r_total,
        "d_hand_obj": d_hand_obj,
        "d_lift": object_height,
        "d_obj_goal": d_obj_goal,
    }

    return rewards


@hydra.main(config_name="config", config_path="../cfg/")
def launch_test(cfg: DictConfig):
    np.random.seed(0)
    torch.manual_seed(0)
    cfg_dict = omegaconf_to_dict(cfg)
    cfg_task = cfg_dict["task"]
    rl_device = cfg_dict["rl_device"]
    sim_device = cfg_dict["sim_device"]
    headless = cfg_dict["headless"]
    graphics_device_id = 0
    virtual_screen_capture = False
    force_render = False
    env = FrankaLEAPPickTop(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    for i in tqdm(range(1000)):
        t1 = time.time()
        env.reset_idx()
        # env.set_robot_joint_state(env.canonical_grasp_config)
        env.step_sim_multi(1, False)
        env.compute_observations()

        # test fk, ik # need to set eef to panda_link7, otherwise will have offset
        ee_pos = env.get_ee_from_joint(env.states['q'][:, :7])
        fk_pos_err = torch.any((ee_pos[:, :3] - env.states['eef_pos']) > 1e-4)
        fk_ori_err1 = (ee_pos[:, 3:] - env.states['eef_quat']) > 1e-4
        fk_ori_err2 = (ee_pos[:, 3:] + env.states['eef_quat']) > 1e-4
        fk_ori_err = torch.any(fk_ori_err1 & fk_ori_err2)
        print(f"FK pos error: {fk_pos_err}, FK ori error: {fk_ori_err}")

        q_config = env.get_joint_from_ee(ee_pos)
        ee_pos_resolve = env.get_ee_from_joint(q_config)
        ik_pos_err = torch.any((ee_pos_resolve[:, :3] - env.states['eef_pos']) > 1e-4)
        ik_quat_err1 = (ee_pos_resolve[:, 3:] - env.states['eef_quat']) > 1e-4
        ik_quat_err2 = (ee_pos_resolve[:, 3:] + env.states['eef_quat']) > 1e-4
        ik_quat_err = torch.any(ik_quat_err1 & ik_quat_err2)
        print(f"IK pos error: {ik_pos_err}, IK ori error: {ik_quat_err}")

        import ipdb ; ipdb.set_trace()
        t2 = time.time()
        print(f"Reset time: {t2 - t1}")
        env.render()


if __name__ == "__main__":
    launch_test()