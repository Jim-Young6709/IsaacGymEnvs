import torch
import time
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
from collections import OrderedDict

from isaacgymenvs.utils.training_utils import *
from isaacgymenvs.inference.inference_utils import *


TRANSFORMER_CONFIGS = {
    "device": "cuda:0",
    "clip_actions": 1.0,
    "local_pcd_range": 0.5,
    "num_local_points": 512,
    "seed": 42,
    "action_scale": {
        "eef_pos": 0.01,
        "eef_rot": 0.01,
        "hand": 0.05,
    },
    "model_config_path": "isaacgymenvs/cfg/model/deploy_transformer_local_t_ctrldelta.yaml",
    "ckpt_path": "dagger_ckpts/grogu_ckpts/Oct16.pt",
}


class LocalPolicyTransformer:
    def __init__(self, configs):
        self.device = configs["device"]
        set_seed_and_precision(configs["seed"])
        self.local_pcd_range = configs["local_pcd_range"]
        self.num_local_points = configs["num_local_points"]
        self.clip_actions = configs["clip_actions"]
        self.action_scale = configs["action_scale"]

        # load model from config
        model_config_file = Path(configs["model_config_path"])
        assert model_config_file.exists(), f"Model config file {model_config_file} does not exist"
        with open(model_config_file, "r") as f:
            model_config = OmegaConf.load(f)
        self.model = instantiate(model_config).to(self.device)
        self.model = self.model.to(self.device)

        # load ckpt weight
        load_checkpoint_path = configs["ckpt_path"]
        if load_checkpoint_path is not None:
            success_rate_ep = self.load_checkpoint(load_checkpoint_path)
            colorprint(f"Loading ckpt from {load_checkpoint_path}: success_rate_ep={success_rate_ep}", color="magenta")

        self.robot_dof_lower_limits = torch.tensor([
            -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
            -0.3140, -1.0470, -0.5060, -0.3660,
            -0.3490, -0.4700, -1.2000, -1.3400,
            -0.3140, -1.0470, -0.5060, -0.3660,
            -0.3140, -1.0470, -0.5060, -0.3660
        ], device=self.device)
        self.robot_dof_upper_limits = torch.tensor([
            2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,
            2.2300, 1.0470, 1.8850, 2.0420,
            2.0940, 2.4430, 1.9000, 1.8800,
            2.2300, 1.0470, 1.8850, 2.0420,
            2.2300, 1.0470, 1.8850, 2.0420
        ], device=self.device)
        self.hand_dof_lower_limits = self.robot_dof_lower_limits[7:]
        self.hand_dof_upper_limits = self.robot_dof_upper_limits[7:]
        self.abs_hand_actions = torch.zeros(1, 16, device=self.device)
        self.steps = 0


    def generate_random_inputs(self):
        """
        Generate random inputs for the get_action function using torch.rand
        """      
        # Generate random point cloud in EEF frame (N, 3)
        N = 1500  
        # Random point cloud in [0, 1)
        full_pcd_eef_frame_t = torch.rand(N, 3, device=self.device) 
        # Generate random hand configuration (16,)
        q_hand = torch.rand(16, device=self.device)  # In [0, 1)
        # Generate random EEF absolute pose (7,) - xyz + xyzw quaternion
        eef_pos = torch.rand(3, device=self.device)  # xyz position in [0, 1)
        eef_quat = torch.rand(4, device=self.device)  # xyzw quaternion in [0, 1)
        eef_quat = eef_quat / torch.norm(eef_quat)  # Normalize to get valid quaternion
        eef_abs_pose = torch.cat([eef_pos, eef_quat])
        return full_pcd_eef_frame_t, q_hand, eef_abs_pose

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        # remove the DDP ckpt prefix if there are any
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix (7 characters)
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        return checkpoint["success_rate_ep"]

    def unnormalize_leap_hand_joints(self, joint_angles: torch.Tensor, delta: bool = False) -> torch.Tensor:
        """
        Unnormalize joint angles.
        Args:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        Returns:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        """
        assert joint_angles.shape[-1] == 16
        lower_limits = self.hand_dof_lower_limits
        upper_limits = self.hand_dof_upper_limits
        limit_range = upper_limits - lower_limits
        if delta:
            unnormalized = joint_angles * limit_range
        else:
            desired_lower_limits = -1 * torch.ones_like(joint_angles)
            desired_upper_limits = 1 * torch.ones_like(joint_angles)
            unnormalized = (joint_angles - desired_lower_limits) * limit_range / (
                desired_upper_limits - desired_lower_limits
            ) + lower_limits
        return unnormalized

    def inference_policy(self, obs_dict):
        """
        get action from the network, this is the action that goes into IsaacGym's step function, not the final action that's been executed

        Args:
            obs_dict (OrderedDict):
                full_pcd_eef_frame_t (torch.Tensor): (1, N, 3), N should be larger than num_local_points
                q_hand (torch.Tensor): (1, 16)
        Returns:
            step_actions (torch.Tensor): (action_dim)
        """
        obs_dict["local_pcd_t"], _ = crop_local_pcd(
            obs_dict['full_pcd_eef_frame_t'], self.local_pcd_range, self.num_local_points
        ) 

        with torch.no_grad():
            self.model.eval()
            student_actions_chunk = self.model(obs_dict)
        # (Batch, Chunk, Action_dim)
        student_actions = student_actions_chunk[0, 0, :] 
        # get the step action that goes into env.step() in sim
        step_actions = torch.clamp(student_actions, -self.clip_actions, self.clip_actions)
        return step_actions, obs_dict
    

    def get_action(self, full_pcd_eef_frame_t, q_hand, eef_abs_pose):
        """
        get the final action for execution

        Args:
            full_pcd_eef_frame_t (torch.Tensor): (N, 3), N should be larger than num_local_points
            q_hand (torch.Tensor): (16,)
            eef_abs_pose (torch.Tensor): (7,) xyz + xyzw
        Returns:
            step_actions (torch.Tensor): (7+16,)
        """

        # reformatting inputs
        assert full_pcd_eef_frame_t.dim() == 2 \
            and full_pcd_eef_frame_t.size(0) >= self.num_local_points \
            and full_pcd_eef_frame_t.size(1) == 3
        # (1, N, 3)
        full_pcd_eef_frame_t_b = full_pcd_eef_frame_t.unsqueeze(0).to(self.device) 
        q_hand_b = q_hand.unsqueeze(0).to(self.device) # (1, 16)
        obs_dict = OrderedDict([
            ("full_pcd_eef_frame_t", full_pcd_eef_frame_t_b),
            ("q_hand", q_hand_b),
            ("q_hand_ctrl_delta", (q_hand_b - self.abs_hand_actions)*2) # *2 helps with sim-to-real
        ])
        # inference policy
        step_action, obs_dict = self.inference_policy(obs_dict)

        # apply policy action
        pos_actions = step_action[:3] * self.action_scale["eef_pos"]
        ctrl_target_eef_pos = eef_abs_pose[:3] + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = step_action[3:6] * self.action_scale["eef_rot"]
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = quat_from_angle_axis(angle, axis)

        # clamp tiny rotations to avoid numerical issues
        if angle < 1.0e-6:
            rot_actions_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        ctrl_target_eef_quat = quat_mul(
            rot_actions_quat, eef_abs_pose[3:] # xyzw format
        )

        hand_actions = step_action[6:] * self.action_scale["hand"]
        delta_hand_joint_actions_unnormalized = self.unnormalize_leap_hand_joints(hand_actions, delta=True)

        abs_hand_actions = q_hand + delta_hand_joint_actions_unnormalized
        abs_hand_actions = tensor_clamp(
            abs_hand_actions, self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )

        self.abs_hand_actions[0, :] = abs_hand_actions
        self.steps += 1

        return ctrl_target_eef_pos.cpu().numpy(), ctrl_target_eef_quat.cpu().numpy(), abs_hand_actions.cpu().numpy(), obs_dict
    


if __name__ == "__main__":
    device = "cuda:0"
    model = LocalPolicyTransformer(device, TRANSFORMER_CONFIGS)

    print("warm up")
    t_1 = time.time()
    for i in range(3):
        test_input = model.generate_random_inputs()
        action = model.get_action(*test_input)
    t_2 = time.time()
    print(f"warm up time: {t_2 - t_1}")

    test_num = 100.
    print(f"profiling with {test_num} inferences")
    t_3 = time.time()
    for i in range(int(test_num)):
        test_input = generate_random_inputs()
        action = model.get_action(*test_input)
    t_4 = time.time()
    t_test = t_4 - t_3
    print(f"time for {test_num} inference: {t_test} ; HZ: {test_num/t_test}")
