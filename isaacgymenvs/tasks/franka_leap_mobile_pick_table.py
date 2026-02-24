"""
Franka + LEAP Hand Pick Env
TODO:1. clean up
"""

import time

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.utils.pcd_utils import *
from isaacgymenvs.utils.rotation_conversions import *
from isaacgymenvs.tasks import FrankaLEAPMobile
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from tqdm import tqdm



class FrankaLEAPMobilePickTable(FrankaLEAPMobile):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

    def _init_params(self):
        z_shift_range = self.cfg["env"]["scene"]["z_shift_range"] # this shifts the table height, not just the safety box
        self.z_shift = torch.rand(self.num_envs, device=self.device) * (z_shift_range[1] - z_shift_range[0]) + z_shift_range[0]

        table_size_range = torch.tensor(self.cfg["env"]["scene"]["table_size_range"], device=self.device)
        self.table_size = torch.rand(self.num_envs, 3, device=self.device) * (table_size_range[1] - table_size_range[0]) + table_size_range[0]

        self.max_objects_per_env = 1

    def _setup_fabric_switching_target(self):
        self.switching_target_pos = self._object_state[:, :3].clone()
        self.switching_target_pos += self.switch_pos_offset
        self.switching_target_pos[:, 2] += self.mesh_aabb_extents[:, 2] / 2
        rot_local_x_180 = torch.tensor([[1.0, 0.0, 0.0, 0.0]]*self.num_envs, device=self.device)  # 180 degrees around local x-axis
        self.switching_target_quat = rot_local_x_180 # default hand orientation is facing up, so need to rotate 180

    def _create_envs(self, spacing, num_per_row):
        """
        loading Franka + LEAP + a table in the environment, this is for debugging purposes only
        """
        self._init_params()

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # setup params
        self.table_pos = []

        self.cuboid_dims = []  # xyz
        self.cuboid_pos = []
        self.cuboid_quats = [] # xyzw

        self.mesh_aabb_extents = None  # xyz, axis-aligned bounding box full extents
        self.table_surface_height = torch.zeros((self.num_envs,), device=self.device)

        self.obj_pos_target = torch.zeros((self.num_envs, 3), device=self.device) # x, y, z

        obj_reset_pos_range = self.cfg["env"]["object_settings"]["obj_reset_pos_range"]
        self.obj_reset_center_xy = torch.tensor(self.cfg["env"]["object_settings"]["obj_reset_center_xy"], device=self.device).repeat(self.num_envs, 1)
        self.obj_reset_center_quat = torch.tensor(self.cfg["env"]["object_settings"]["obj_reset_center_quat"], device=self.device).repeat(self.num_envs, 1)
        self.obj_reset_pos_range = torch.zeros((self.num_envs, 4), device=self.device) # x-min, x-max, y-min, y-max
        # Convert configured global XY range to range relative to box center.
        self.obj_reset_pos_range[:, 0] = obj_reset_pos_range[0][0] # x-min
        self.obj_reset_pos_range[:, 1] = obj_reset_pos_range[1][0] # x-max
        self.obj_reset_pos_range[:, 2] = obj_reset_pos_range[0][1] # y-min
        self.obj_reset_pos_range[:, 3] = obj_reset_pos_range[1][1] # y-max

        # setup robot (franka + leap)
        robot_dof_props = self._create_franka_leap()
        robot_asset = self.robot_asset
        robot_start_pose = gymapi.Transform()
        robot_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # make sure robot spawns at the origin, this matches the IK setting with cuRobo
        robot_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # compute aggregate size
        num_robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        num_robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        max_agg_bodies = num_robot_bodies + 1 + 1 # 1 for object, 1 for table
        max_agg_shapes = num_robot_shapes + 1 + 1 # 1 for object, 1 for table

        self.robots = []
        self.objects = []
        self.add_on_obstacles = []
        self.envs = []
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


            self.table_surface_height[i] = self.z_shift[i].item()

            # Create table
            # setup table
            table_pos = [0.5, 0.0, -self.table_size[i, 2].item()/2+self.z_shift[i].item()]
            table_size = self.table_size[i].cpu().numpy().tolist()
            self.table_pos.append(table_pos)

            table_asset, table_start_pose = self._create_cube(
                pos=table_pos,
                size=table_size,
            )

            if self.enable_fabric:
                self._create_fabric_cube(
                    pos=table_pos,
                    size=table_size,
                    quat=[0, 0, 0, 1],
                    env_id=i,
                )

            self.gym.create_actor(
                env_ptr, table_asset, table_start_pose, "table", i, 1, 0
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
            self.envs.append(env_ptr)
            self.robots.append(robot_actor)
            self.objects.append(self._object_id)
            self.add_on_obstacles.append(self._add_on_obstacle_ids)

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

        self.table_pos = torch.tensor(self.table_pos, device=self.device)

        for i in range(self.num_envs):
            # static pcd
            static_pcd_i = torch.from_numpy(compute_scene_oracle_pcd(
                num_obstacle_points=self.pcd_spec_dict["num_static_points"],
                cuboid_dims=np.array(self.cuboid_dims[i]),
                cuboid_centers=np.array(self.cuboid_pos[i]),
                cuboid_quats=np.array(self.cuboid_quats[i]),
            )).to(self.device)
            self.static_pcds.append(static_pcd_i)

        self.cuboid_dims = torch.from_numpy(self.cuboid_dims).to(self.device)
        self.cuboid_pos = torch.from_numpy(self.cuboid_pos).to(self.device)
        self.cuboid_quats = torch.from_numpy(self.cuboid_quats).to(self.device)

        self.static_pcds = torch.stack(self.static_pcds, dim=0).to(self.device).to(torch.float32) # (num_envs, num_points, 3)
        self.object_pcds = torch.stack(self.object_pcds, dim=0).to(self.device).to(torch.float32)
        self.combined_pcds = torch.cat([self.static_pcds, self.object_pcds], dim=1).to(self.device) # (num_envs, num_static_points + num_object_points, 3)
        if self.distractor_settings["enable"]:
            self._create_distractor_pcd()
            self.combined_pcds = torch.cat([self.combined_pcds, self.distractor_pcds], dim=1).to(self.device) # (num_envs, num_static_points + num_object_points + num_distractor_points, 3)

        # get mesh AABB (axis-aligned bounding box) extents
        min_xyz = self.object_pcds.min(axis=1).values
        max_xyz = self.object_pcds.max(axis=1).values
        self.mesh_aabb_extents = max_xyz - min_xyz
        self._object_center_init_state[:, 2] += self.mesh_aabb_extents[:, 2] / 2

        # Setup data
        actor_num = 1 + 1 + 1 # robot, table, object
        self.init_data(actor_num=actor_num)

        if self.enable_fabric:
            self._init_fabric()

    def init_data(self, actor_num):
        super().init_data(actor_num=actor_num)
        self.reward_settings["target_pos"] = self.obj_pos_target
        self.reward_settings["w_colli"] = to_torch(self.cfg["reward"]["weights"]["w_colli"], device=self.device)

    def _update_states(self):
        super()._update_states()
        lift_5cm = self.states["object_center_pos"][:, 2] - self._object_center_init_state[:, 2] > 0.05

        self.obj_pos_target[~lift_5cm, :2] = self.states["object_center_pos"][~lift_5cm, :2]  # x, y
        self.obj_pos_target[:, 2] = self.table_surface_height + self.reward_settings['target_lift_dis']

        self.switching_target_pos = self.states['object_center_pos'].clone()
        self.switching_target_pos += self.switch_pos_offset

        self.states.update({
            # check whether the object is lifted based on bottom board force contact info
            "lift": lift_5cm,
            "collision": self.scene_collision,
        })

    def check_robot_collision(self):
        super().check_robot_collision()
        self.scene_collision = torch.any(self.contact_forces[:, 59:-1].view(self.num_envs, -1) != 0, dim=1)
        self.table_collision = torch.any(self.contact_forces[:, 58].view(self.num_envs, -1) != 0, dim=1)

    def compute_observations(self):
        self._refresh()

        obs_components = ["q_hand",
                          "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                          "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                          "object_to_eef", "object_to_eef_rot_6d",
                          "target_to_eef", "target_to_eef_rot_6d"]

        states_components = ["q", "qd",
                             "eef_pos", "eef_rot_6d", "eef_vel",
                             "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                             "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                             "object_to_eef", "object_to_eef_rot_6d",
                             "target_to_eef", "target_to_eef_rot_6d"]

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

        reward_dict = compute_franka_leap_reward(self.states, self.reward_settings)

        self.rew_buf[:] = reward_dict["r_total"]
        self.extras["sep_reward/r_hand_obj"] = torch.mean(reward_dict["r_hand_obj"]).item()
        self.extras["sep_reward/r_obj_goal"] = torch.mean(reward_dict["r_obj_goal"]).item()
        self.extras["sep_reward/r_lift"] = torch.mean(reward_dict["r_lift"]).item()
        self.extras["sep_reward/r_curl"] = torch.mean(reward_dict["r_curl"]).item()
        self.extras["sep_reward/r_colli"] = torch.mean(reward_dict["r_colli"]).item()
        self.extras["sep_reward/r_actionreg"] = torch.mean(reward_dict["r_actionreg"]).item()
        self.extras["dis/d_hand_obj"] = torch.mean(reward_dict["d_hand_obj"]).item()
        self.extras["dis/d_lift"] = torch.mean(reward_dict["d_lift"]).item()
        self.extras["dis/d_eef_point_goal"] = torch.mean(reward_dict["d_eef_point_goal"]).item()

        # log metrics
        self.lifting_5cm_per_step = self.states["lift"]
        self.lifting_flags[self.lifting_5cm_per_step] = 1
        self.success_5cm_per_step = (reward_dict["d_eef_point_goal"] < 0.05) & self.lifting_5cm_per_step
        self.success_flags[self.success_5cm_per_step] = 1

        self.extras["metrics/success_rate_5cm_per_step"] = torch.mean(self.success_5cm_per_step.float()).item()
        self.extras["metrics/lifting_rate_5cm_per_step"] = torch.mean(self.lifting_5cm_per_step.float()).item()
        self.extras["metrics/success_rate_5cm_per_ep"] = torch.mean(self.success_flags).item()
        self.extras["metrics/lifting_rate_5cm_per_ep"] = torch.mean(self.lifting_flags).item()
        self.extras["metrics/collision_rate_per_step"] = torch.mean(self.states["collision"].float()).item()

    def set_viewer(self):
        super().set_viewer(
            pos=[2.0, 0.0, 1.2],
            target=[0.3, 0.0, 0.7],
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
        r_lift = torch.where(states["lift"], r_lift, 0.0)
    else:
        r_lift = torch.where(states["lift"], 1.0, torch.zeros_like(object_height))

    # R3: Object goal distance reward (based on average point matching distance)
    d_eef_point_goal = states["point_matching_err"]
    beta_object_goal = reward_settings["beta_object_goal"]
    r_obj_goal = torch.exp(-beta_object_goal * d_eef_point_goal)
    r_obj_goal = torch.where(states["lift"], r_obj_goal, 0.0)

    # R4: Finger curl
    hand_dof_pos = states["q"][:, 10:26] # hand joint angles
    near_object = (d_hand_obj <= reward_settings["curl_reaching_threshold"])
    finger_pos_diff = torch.sum((hand_dof_pos - reward_settings["grasp_finger_dof_pos"]) ** 2, dim=1)

    beta_curl = reward_settings["beta_curl"]
    r_curl= torch.exp(-beta_curl * finger_pos_diff)
    r_curl = torch.where(near_object, r_curl, 0.0)

    # R6: Colli Penalty
    r_colli = torch.where(states["collision"], 1.0, 0.0)

    # R7: Velocity Regularization/Penalty
    actionreg = states["actionreg"]
    r_actionreg = torch.sum(actionreg**2, dim=-1)

    w_hand_obj = reward_settings["w_hand_obj"]
    w_obj_goal = reward_settings["w_obj_goal"]
    w_lift = reward_settings["w_lift"]
    w_curl = reward_settings["w_curl"]
    w_colli = reward_settings["w_colli"]
    w_actionreg = reward_settings["w_actionreg"]

    r_total = w_hand_obj*r_hand_obj + w_obj_goal*r_obj_goal + \
              w_lift*r_lift + w_curl*r_curl + \
              w_colli*r_colli + w_actionreg*r_actionreg

    rewards = {
        "r_hand_obj": w_hand_obj*r_hand_obj,
        "r_lift": w_lift*r_lift,
        "r_obj_goal": w_obj_goal*r_obj_goal,
        "r_curl": w_curl*r_curl,
        "r_colli": w_colli*r_colli,
        "r_actionreg": w_actionreg*r_actionreg,
        "r_total": r_total,
        "d_hand_obj": d_hand_obj,
        "d_lift": object_height,
        "d_eef_point_goal": d_eef_point_goal,
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
    env = FrankaLEAPMobilePickTable(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    for i in tqdm(range(1000)):
        t1 = time.time()
        env.reset_idx()
        # env.set_robot_joint_state(env.canonical_joint_config)
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
