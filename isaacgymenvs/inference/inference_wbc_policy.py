import torch
import time
from hydra.utils import instantiate
from collections import OrderedDict

from omegaconf import OmegaConf

from isaacgymenvs.utils.training_utils import *
from isaacgymenvs.inference.inference_utils import *


TRANSFORMER_CONFIGS = {
    "seed": 42,
    "ckpt_path": "dagger_ckpts/grogu_ckpts/Feb9_wbc_tablemulti_aux_1024_0_1024_expJan22.pt",
}


class WBCPolicyTransformer:
    def __init__(self, configs):
        self.device = "cuda:0"
        set_seed_and_precision(configs["seed"])

        # load ckpt
        load_checkpoint_path = configs["ckpt_path"]
        if load_checkpoint_path is not None:
            success_rate_ep = self.load_checkpoint(load_checkpoint_path)
            colorprint(f"Loading ckpt from {load_checkpoint_path}: success_rate_ep={success_rate_ep}", color="magenta")

        self.local_pcd_range = self.ckpt_cfg["dagger"]["local_pcd_range"]
        self.num_local_points = 1024
        self.clip_actions = self.ckpt_cfg["task"]["env"]["clipActions"]
        self.action_scale = self.ckpt_cfg["task"]["env"]["actionScale"]
        # convert to python dict
        self.action_scale = OmegaConf.to_container(self.action_scale, resolve=True)

        self.dt = self.ckpt_cfg["task"]["sim"]["dt"]

        self.delta_franka_action = self.ckpt_cfg["action_space"]["delta_franka_action"]
        self.delta_leap_action = self.ckpt_cfg["action_space"]["delta_leap_action"]
        self.delta_arx_action = self.ckpt_cfg["action_space"]["delta_arx_action"]

        self.robot_dof_lower_limits = torch.tensor([
           -5.0000e+00, -5.0000e+00, -1.0000e+05,
           -2.9671e+00, -1.8326e+00, -2.9671e+00, -3.1416e+00, -2.9671e+00, -8.7300e-02, -2.9671e+00,
           -3.1400e-01, -1.0470e+00, -5.0600e-01, -3.6600e-01,
           -3.4900e-01, -4.7000e-01, -1.2000e+00, -1.3400e+00,
           -3.1400e-01, -1.0470e+00, -5.0600e-01, -3.6600e-01,
           -3.1400e-01, -1.0470e+00, -5.0600e-01, -3.6600e-01,
           -3.1400e+00,  0.0000e+00,  0.0000e+00, -1.5708e+00, -1.6700e+00, -1.5700e+00,
        ], device=self.device)
        self.robot_dof_upper_limits = torch.tensor([
            5.0000e+00, 5.0000e+00, 1.0000e+05,
            2.9671e+00, 1.8326e+00, 2.9671e+00, 0.0000e+00, 2.9671e+00, 3.8223e+00, 2.9671e+00,
            2.2300e+00, 1.0470e+00, 1.8850e+00, 2.0420e+00,
            2.0940e+00, 2.4430e+00, 1.9000e+00, 1.8800e+00,
            2.2300e+00, 1.0470e+00, 1.8850e+00, 2.0420e+00,
            2.2300e+00, 1.0470e+00, 1.8850e+00, 2.0420e+00,
            2.6180e+00, 3.1400e+00, 3.1400e+00, 1.5708e+00, 1.6700e+00, 1.5700e+00,
        ], device=self.device)

        self.franka_dof_lower_limits = self.robot_dof_lower_limits[3:10]
        self.franka_dof_upper_limits = self.robot_dof_upper_limits[3:10]
        self.leap_dof_lower_limits = self.robot_dof_lower_limits[10:26]
        self.leap_dof_upper_limits = self.robot_dof_upper_limits[10:26]
        self.arx_dof_lower_limits = self.robot_dof_lower_limits[26:32]
        self.arx_dof_upper_limits = self.robot_dof_upper_limits[26:32]
        self.abs_hand_actions = torch.zeros(16, device=self.device)
        self.steps = 0

    def generate_random_inputs(self):
        """
        Generate random inputs for the get_action function using torch.rand
        """      
        # Generate random point cloud in EEF frame (N, 3)
        N = self.num_local_points*2  
        # Random point cloud in [0, 1)
        full_pcd_eef_frame_t = torch.rand(N, 3, device=self.device) 
        # Generate random hand configuration (16,)
        q_hand = torch.rand(16, device=self.device)  # In [0, 1)
        q_arm_manip = torch.rand(7, device=self.device)  # In [0, 1)
        q_arm_vision = torch.rand(6, device=self.device)  # In [0, 1)

        if self.has_aux_input == True:
            aux_inputs = torch.rand(3, device=self.device)
        else:
            aux_inputs = None

        return full_pcd_eef_frame_t, torch.zeros(3, device=self.device), q_hand, q_arm_manip, q_arm_vision, aux_inputs

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # load model from config
        self.ckpt_cfg = checkpoint["cfg"]
        self.model = instantiate(self.ckpt_cfg["model"]).to(self.device)
        self.model = self.model.to(self.device)
        self.has_aux_input = "aux_object_state" in self.ckpt_cfg.model.state_encoders_cfg
        self.aux_prediction_mode = str(self.ckpt_cfg.model.get("aux_prediction_mode", "absolute")).lower()
        self.aux_delta_scale = float(self.ckpt_cfg.model.get("aux_delta_scale", 0.01))
        if self.aux_prediction_mode not in ["absolute", "delta"]:
            raise ValueError(f"aux_prediction_mode must be 'absolute' or 'delta', got {self.aux_prediction_mode}")

        # remove the DDP ckpt prefix if there are any
        state_dict = checkpoint["model_state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix (7 characters)
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        return checkpoint["train_success_rate_ep"]

    def _aux_to_2d(self, aux_tensor):
        if aux_tensor.ndim == 3:
            return aux_tensor[:, 0, :]
        return aux_tensor

    def _decode_aux_prediction(self, aux_pred, prev_abs_aux):
        prev_abs_aux_2d = self._aux_to_2d(prev_abs_aux)
        if self.aux_prediction_mode == "delta":
            aux_delta = torch.clamp(aux_pred, -1.0, 1.0)
            return prev_abs_aux_2d.unsqueeze(1) + self.aux_delta_scale * aux_delta
        return self._aux_to_2d(aux_pred).unsqueeze(1)

    def normalize_robot_joints(self, joint_angles: torch.Tensor, robot: bool, delta: bool = False) -> torch.Tensor:
        """
        Normalize joint angles to be within [-1, 1].
        Args:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        Returns:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        """
        if robot=="franka":
            assert joint_angles.shape[-1] == 7
            lower_limits = self.franka_dof_lower_limits
            upper_limits = self.franka_dof_upper_limits
        elif robot=="leap":
            assert joint_angles.shape[-1] == 16
            lower_limits = self.leap_dof_lower_limits
            upper_limits = self.leap_dof_upper_limits
        elif robot=="arx":
            assert joint_angles.shape[-1] == 6
            lower_limits = self.arx_dof_lower_limits
            upper_limits = self.arx_dof_upper_limits
        else:
            raise ValueError("robot must be in ['franka', 'leap', 'arx']")

        franka_limit_range = upper_limits - lower_limits

        if delta:
            normalized = joint_angles / franka_limit_range
        else:
            desired_lower_limits = -1 * torch.ones_like(joint_angles)
            desired_upper_limits = 1 * torch.ones_like(joint_angles)
            normalized = (joint_angles - lower_limits) / franka_limit_range * (
                desired_upper_limits - desired_lower_limits
            ) + desired_lower_limits
        return normalized

    def unnormalize_robot_joints(self, joint_angles: torch.Tensor, robot: bool, delta: bool = False) -> torch.Tensor:
        """
        Unnormalize joint angles.
        Args:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        Returns:
            joint_angles (torch.Tensor): (num_envs, num_robot_dofs)
        """
        if robot=="franka":
            assert joint_angles.shape[-1] == 7
            lower_limits = self.franka_dof_lower_limits
            upper_limits = self.franka_dof_upper_limits
        elif robot=="leap":
            assert joint_angles.shape[-1] == 16
            lower_limits = self.leap_dof_lower_limits
            upper_limits = self.leap_dof_upper_limits
        elif robot=="arx":
            assert joint_angles.shape[-1] == 6
            lower_limits = self.arx_dof_lower_limits
            upper_limits = self.arx_dof_upper_limits
        else:
            raise ValueError("robot must be in ['franka', 'leap', 'arx']")

        franka_limit_range = upper_limits - lower_limits

        if delta:
            unnormalized = joint_angles * franka_limit_range
        else:
            desired_lower_limits = -1 * torch.ones_like(joint_angles)
            desired_upper_limits = 1 * torch.ones_like(joint_angles)
            unnormalized = (joint_angles - desired_lower_limits) * franka_limit_range / (
                desired_upper_limits - desired_lower_limits
            ) + lower_limits
        return unnormalized

    def inference_policy(self, obs_dict):
        """
        get action from the network, this is the action that goes into IsaacGym's step function, not the final action that's been executed

        Args:
            obs_dict (OrderedDict):
                full_pcd_frankabase_frame_t (torch.Tensor): (1, N, 3), N should be larger than num_local_points
                q_hand (torch.Tensor): (1, 16)
        Returns:
            step_actions (torch.Tensor): (action_dim)
        """

        if "local_pcd_t" in self.ckpt_cfg["model"]["pcd_encoders_cfg"]:
            # Codex
            num_points = self.ckpt_cfg["model"]["pcd_encoders_cfg"]["local_pcd_t"]["num_points"] # [num cylindrical points, num spherical eef points, num spherical aux points]
            num_cyl = int(num_points[0])
            num_eef = int(num_points[1]) if len(num_points) > 1 else 0
            num_aux = int(num_points[2]) if len(num_points) > 2 else 0
            local_ranges = self.local_pcd_range
            local_cyl_range = float(local_ranges[0])
            local_eef_range = float(local_ranges[1]) if len(local_ranges) > 1 else float(local_ranges[0])
            local_aux_range = float(local_ranges[2]) if len(local_ranges) > 2 else local_eef_range

            cylindrical_local_pcd_t, _ = crop_local_pcd(
                obs_dict['full_pcd_frankabase_frame_t'],
                local_cyl_range,
                num_cyl,
                is_cylindrical=True,
            ) # (num_envs, num_local_points, 3)
            spherical_local_pcd_t, _ = crop_local_pcd(
                obs_dict['full_pcd_frankabase_frame_t'] - obs_dict['eef_xyz_frankabase_frame_t'],
                local_eef_range,
                num_eef,
                is_cylindrical=False,
            ) # (num_envs, num_local_points, 3)
            spherical_local_pcd_t = spherical_local_pcd_t + obs_dict['eef_xyz_frankabase_frame_t']

            local_pcd_parts = [cylindrical_local_pcd_t, spherical_local_pcd_t]

            # Codex
            if num_aux > 0 and self.has_aux_input:
                aux_origin = obs_dict["aux_object_state"]
                aux_spherical_local_pcd_t, _ = crop_local_pcd(
                    obs_dict['full_pcd_frankabase_frame_t'] - aux_origin,
                    local_aux_range,
                    num_aux,
                    is_cylindrical=False,
                ) # (num_envs, num_local_points, 3)
                aux_spherical_local_pcd_t = aux_spherical_local_pcd_t + aux_origin
                local_pcd_parts.append(aux_spherical_local_pcd_t)

            obs_dict["local_pcd_t"] = torch.cat(local_pcd_parts, dim=1)

        aux_pred = None
        with torch.no_grad():
            self.model.eval()
            output = self.model(obs_dict)
            student_actions_chunk = output["action"]
            if self.model.aux_prediction:
                aux_pred = self._decode_aux_prediction(output["aux"], obs_dict["aux_object_state"])
                aux_pred = aux_pred[0, 0, :]

        student_actions = student_actions_chunk[0, 0, :32] # TODO: this assumes chunk size is 1

        # get the step action that goes into env.step() in sim
        step_actions = torch.clamp(student_actions, -self.clip_actions, self.clip_actions)
        return step_actions, aux_pred, obs_dict

    def get_action(self, full_pcd_frankabase_frame_t, eef_xyz_frankabase_frame_t, q_hand, q_arm_manip, q_arm_vision, aux_inputs=None):
        """
        get the final action for execution

        Args:
            full_pcd_frankabase_frame_t (torch.Tensor): (N, 3), N should be larger than num_local_points
            q_hand (torch.Tensor): (16,)
            eef_abs_pose (torch.Tensor): (7,) xyz + xyzw
            aux_inputs (torch.Tensor): (3,) optional auxiliary inputs, currently set to object center xyz position
        Returns:
            step_actions (torch.Tensor): (7+16,)
        """

        # reformatting inputs
        assert full_pcd_frankabase_frame_t.dim() == 2 \
            and full_pcd_frankabase_frame_t.size(0) >= self.num_local_points \
            and full_pcd_frankabase_frame_t.size(1) == 3
        # (1, N, 3)
        full_pcd_frankabase_frame_t_b = full_pcd_frankabase_frame_t.unsqueeze(0).to(self.device)
        eef_xyz_frankabase_frame_t_b = eef_xyz_frankabase_frame_t.unsqueeze(0).to(self.device)  # (1, 3)
        if self.has_aux_input:
            if aux_inputs is None:
                aux_inputs_b = torch.rand((1, 3), device=self.device)
            else:
                aux_inputs_b = aux_inputs.unsqueeze(0).to(self.device)
        else:
            aux_inputs_b = None
    
        q_hand_b = self.normalize_robot_joints(q_hand.unsqueeze(0).to(self.device), robot="leap", delta=False) # (1, 16)
        q_hand_ctrl_delta_b = self.normalize_robot_joints(
            (q_hand.to(self.device) - self.abs_hand_actions), robot="leap", delta=True
        ).unsqueeze(0).to(self.device)
        q_arm_manip_b = self.normalize_robot_joints(q_arm_manip.unsqueeze(0).to(self.device), robot="franka", delta=False) # (1, 7)
        q_arm_vision_b = self.normalize_robot_joints(q_arm_vision.unsqueeze(0).to(self.device), robot="arx", delta=False) # (1, 7)

        obs_dict = OrderedDict([
            ("full_pcd_frankabase_frame_t", full_pcd_frankabase_frame_t_b),
            ("eef_xyz_frankabase_frame_t", eef_xyz_frankabase_frame_t_b),
            ("aux_object_state", aux_inputs_b),
            ("q_arm_manip", q_arm_manip_b),
            ("q_arm_vision", q_arm_vision_b),
            ("q_hand", q_hand_b),
            ("q_hand_ctrl_delta", q_hand_ctrl_delta_b*2) # *2 helps with sim-to-real
        ])

        # inference policy
        step_action, aux_pred, obs_dict = self.inference_policy(obs_dict)

        # get unnormalized absolute actions for execution
        actions_abs = step_action.clone() # (32,)

        actions_abs[:3] = step_action[:3] # base vel TODO: we should keep here as velocity right?

        if self.delta_franka_action:
            actions_abs[3:10] = self.unnormalize_robot_joints(actions_abs[3:10], robot="franka", delta=True) * self.action_scale["franka"] * self.dt + q_arm_manip
        else:
            actions_abs[3:10] = self.unnormalize_robot_joints(actions_abs[3:10], robot="franka", delta=False)

        if self.delta_leap_action:
            actions_abs[10:26] = self.unnormalize_robot_joints(actions_abs[10:26], robot="leap", delta=True) * self.action_scale["leap"] * self.dt + q_hand
        else:
            actions_abs[10:26] = self.unnormalize_robot_joints(actions_abs[10:26], robot="leap", delta=False)

        if self.delta_arx_action:
            actions_abs[26:] = self.unnormalize_robot_joints(actions_abs[26:], robot="arx", delta=True) * self.action_scale["arx"] * self.dt + q_arm_vision
        else:
            actions_abs[26:] = self.unnormalize_robot_joints(actions_abs[26:], robot="arx", delta=False)

        actions_abs = tensor_clamp(
            actions_abs, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )
        self.abs_hand_actions[:] = actions_abs[10:26]

        actions_abs_cpu = actions_abs.cpu().numpy()

        base_vel_robot = actions_abs_cpu[0:3]
        franka_joint_pos = actions_abs_cpu[3:10]
        leap_joint_pos = actions_abs_cpu[10:26]
        arx_joint_pos = actions_abs_cpu[26:32]
        aux_pred_cpu = aux_pred.cpu().numpy() if aux_pred is not None else None
        return base_vel_robot, franka_joint_pos, leap_joint_pos, arx_joint_pos, aux_pred_cpu, obs_dict



if __name__ == "__main__":
    model = WBCPolicyTransformer(TRANSFORMER_CONFIGS)

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
        test_input = model.generate_random_inputs()
        action = model.get_action(*test_input)
    t_4 = time.time()
    t_test = t_4 - t_3
    print(f"time for {test_num} inference: {t_test} ; HZ: {test_num/t_test}")
