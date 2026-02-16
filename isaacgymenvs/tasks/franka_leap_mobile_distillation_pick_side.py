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
from isaacgymenvs.tasks import FrankaLEAPMobileDistillation
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from tqdm import tqdm



class FrankaLEAPMobileDistillationPickTop(FrankaLEAPMobileDistillation):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # @ray we actually manually design the hand reset position range so avoid ik solver failures, so we need a fixed object spawn position
        # also, since we use eef position control, the policy is agnostic to object position and franka priorioception in the world frame
        # so fixed position won't affect learning
        xyz_range = cfg["env"]["object_settings"]["xyz_range"]
        avg_xyz = [
            0.5 * (xyz_range[0][0] + xyz_range[1][0]),
            0.5 * (xyz_range[0][1] + xyz_range[1][1]),
            0.5 * (xyz_range[0][2] + xyz_range[1][2]),
        ]
        cfg["env"]["object_settings"]["xyz_range"] = [avg_xyz, avg_xyz]
        self.object_grasp_target_z_scale = float(cfg["env"]["object_settings"]["object_grasp_target_z_scale"])
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

    def init_data(self, actor_num):
        super().init_data(actor_num)
        # @ray we use target_quat for reward computation and construct it based on the mode "left/right/both"
        self.target_quat_right = to_torch(self.cfg["reward"]["params"]["target_quat_right"], device=self.device).unsqueeze(0)
        self.target_quat_right = self.target_quat_right / (
            torch.norm(self.target_quat_right, dim=1, keepdim=True) + 1e-10
        )
        self.target_quat_left = to_torch(self.cfg["reward"]["params"]["target_quat_left"], device=self.device).unsqueeze(0)
        self.target_quat_left = self.target_quat_left / (
            torch.norm(self.target_quat_left, dim=1, keepdim=True) + 1e-10
        )
        
        self.reward_settings["beta_hand_orientation"] = to_torch(self.cfg["reward"]["exp"]["beta_hand_orientation"], device=self.device)
        self.reward_settings["w_hand_orientation"] = to_torch(self.cfg["reward"]["weights"]["w_hand_orientation"], device=self.device)

        # @ray to force the hand to grasp on the correct region on the object (in terms of z-axis), we gate the lift and to-goal rewards with a sigmoid based on z-difference 
        # between the grasp_target's z height and the averaged finger height on obejct. 
        self.reward_settings["grasp_on_object_z_height_tolerance"] = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_tolerance"], device=self.device)
        self.reward_settings["grasp_on_object_z_height_slope"] = to_torch(self.cfg["reward"]["params"]["grasp_on_object_z_height_scale"], device=self.device)
 
    def _setup_fabric_switching_target(self):
        self.switching_target_pos = self._object_state[:, :3].clone()
        self.switching_target_pos += self.switch_pos_offset
        self.switching_target_pos[:, 2] += self.mesh_aabb_extents[:, 2] / 2
        rot_local_x_180 = torch.tensor([[1.0, 0.0, 0.0, 0.0]]*self.num_envs, device=self.device)  # 180 degrees around local x-axis
        self.switching_target_quat = rot_local_x_180 # default hand orientation is facing up, so need to rotate 180

    def _update_states(self):
        super()._update_states()
        object_grasp_target_pos = self._object_state[:, :3].clone()
        local_offset = torch.zeros([self.num_envs, 3], dtype=torch.float, device=self.device)
        local_offset[:, 2] = self.mesh_aabb_extents[:, 2] * self.object_grasp_target_z_scale
        object_rot_mat = quaternion_to_matrix_ig(self._object_state[:, 3:7])
        rotated_offset = torch.matmul(object_rot_mat, local_offset.unsqueeze(-1)).squeeze(-1)
        object_grasp_target_pos += rotated_offset

        # @ray not just update but also create new keys here
        self.states.update({
            # Table Contact Status, check whether the object is lifted
            "lift": ~self.table_collision,
            "object_grasp_target_pos": object_grasp_target_pos, # @ray reward-only grasp target
            "object_grasp_target_to_eef": object_grasp_target_pos - self._eef_state[:, :3], # @ray for policy observation
        })
    
    def compute_observations(self):
        self._refresh() # @ray checks table collision and updates states

        obs_components = ["q_hand",
                          "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                          "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                          "object_to_eef", "object_to_eef_rot_6d",
                          "object_grasp_target_to_eef",
                          "target_to_eef", "target_to_eef_rot_6d"]

        states_components = ["q", "qd",
                             "eef_pos", "eef_rot_6d", "eef_vel",
                             "eef_finger1_pos_relative", "eef_finger2_pos_relative",
                             "eef_finger3_pos_relative", "eef_finger4_pos_relative",
                             "object_to_eef", "object_to_eef_rot_6d",
                             "object_grasp_target_to_eef",
                             "target_to_eef", "target_to_eef_rot_6d"]

        obs_buf = torch.cat([self.states[ob] for ob in obs_components], dim=-1)
        states_buf = torch.cat([self.states[st] for st in states_components], dim=-1)

        # @ray optionally append object bbox info
        if self.cfg["observation"]["usage"]["use_xy"]:
            obj_xy_bbox = self.mesh_aabb_extents[:, :2]
            obs_buf = torch.cat([obs_buf, obj_xy_bbox], dim=-1)
            states_buf = torch.cat([states_buf, obj_xy_bbox], dim=-1)
        if self.cfg["observation"]["usage"]["use_z"]:
            obj_height = self.mesh_aabb_extents[:, 2:3]
            obs_buf = torch.cat([obs_buf, obj_height], dim=-1)
            states_buf = torch.cat([states_buf, obj_height], dim=-1)

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
        self.extras["dis/d_eef_point_goal_rot"] = torch.mean(reward_dict["d_eef_point_goal_rot"]).item()

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