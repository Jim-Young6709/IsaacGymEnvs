"""
Franka + LEAP Hand Pick Env
"""

import time

import hydra
import isaacgym
import numpy as np
import torch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks import FrankaLEAP
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from omegaconf import DictConfig
from tqdm import tqdm
from collections import OrderedDict
from neural_mp.real_utils.model import NeuralMPModel



class FrankaLEAPPick(FrankaLEAP):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        # load pretrained encoder TODO: note this is the drp_neural_mp encoder, should use IMPACT one
        self.base_model = NeuralMPModel.from_pretrained("jimyoung6709/DRP_Dagger")
        self.pcd_encoder = self.base_model.policy.nets['policy'].model.nets['encoder'].nets['obs']
        self.pcd_encoder.eval()

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render
        )

        # TODO: add full env loading here

    def _create_envs(self, spacing, num_per_row):
        """
        loading Franka + LEAP + a table in the environment, this is for debugging purposes only
        """
        super()._create_envs(spacing, num_per_row)

    def compute_observations(self):
        self._refresh()

        obs_base = OrderedDict()
        dummy_config = torch.ones((self.num_envs, 7), device=self.device, dtype=torch.float32)
        zero_padding = torch.zeros(self.num_envs, self.combined_pcds.shape[1], 1, device=self.device, dtype=torch.float32)
        # TODO: use different mask for object and obstacles
        input_pcd = torch.cat([self.combined_pcds, zero_padding], dim=-1).to(torch.float32)
        obs_base["current_angles"] = dummy_config.clone()
        obs_base["goal_angles"] = dummy_config.clone()
        obs_base["compute_pcd_params"] = input_pcd

        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.float16):
                pcd_latent = self.pcd_encoder(obs_base) # 1038 (1024 + 7 + 7)

        pcd_latent = pcd_latent[:, :1024]

        obs_components = ["q", "eef_pos", "eef_rot_6d",
                          "eef_finger1_pos", "eef_finger2_pos", "eef_finger3_pos", "eef_finger4_pos",
                          "object_center_pos", "object_rot_6d", "hand_to_object",
                          "object_to_target", "object_target_6d_diff"]

        states_components = ["q", "eef_pos", "eef_rot_6d",
                          "eef_finger1_pos", "eef_finger2_pos", "eef_finger3_pos", "eef_finger4_pos",
                          "object_center_pos", "object_rot_6d", "hand_to_object",
                          "object_to_target", "object_target_6d_diff"]

        obs_buf = torch.cat([self.states[ob] for ob in obs_components] + [pcd_latent], dim=-1)
        states_buf = torch.cat([self.states[st] for st in states_components] + [pcd_latent], dim=-1)

        self.obs_buf = obs_buf
        self.states_buf = states_buf

        return self.obs_buf

    def compute_reward(self, actions):
        self.reset_buf[:] = torch.where((self.progress_buf >= self.max_episode_length - 1), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf[self.states['object_center_pos'][:, 2] < -0.1] = 1
        reward_dict = compute_franka_leap_reward(self.states, self.reward_settings)

        self.rew_buf[:] = reward_dict["r_total"]
        self.extras["sep_reward/r_hand_obj"] = torch.mean(reward_dict["r_hand_obj"]).item()
        self.extras["sep_reward/r_obj_goal"] = torch.mean(reward_dict["r_obj_goal"]).item()
        self.extras["sep_reward/r_lift"] = torch.mean(reward_dict["r_lift"]).item()
        self.extras["sep_reward/r_curl"] = torch.mean(reward_dict["r_curl"]).item()

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
    object_height = states["object_center_pos"][:, 2] - reward_settings["object_init_height"].squeeze(-1)
    r_lift = torch.where(object_height > reward_settings["lift_threshold"], 1.0, torch.zeros_like(object_height))

    # R3: Object goal distance reward
    target_pos = reward_settings["target_pos"].squeeze(-1)
    d_obj_goal = torch.norm(states["object_center_pos"] - target_pos, dim=-1)
    beta_object_goal = reward_settings["beta_object_goal"]
    r_obj_goal = torch.exp(-beta_object_goal * d_obj_goal)
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
    env = FrankaLEAPPick(cfg_task, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    env.reset()

    for i in tqdm(range(1000)):
        t1 = time.time()
        env.reset_idx()
        env.set_robot_joint_state(env.canonical_grasp_config)
        import ipdb ; ipdb.set_trace()
        t2 = time.time()
        print(f"Reset time: {t2 - t1}")
        env.render()


if __name__ == "__main__":
    launch_test()
