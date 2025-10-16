import torch
import time
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pathlib import Path
from collections import OrderedDict
from isaacgymenvs.utils.training_utils import *


def shuffle_pcd(pcd: torch.Tensor) -> torch.Tensor:
    """
    Randomize ordering of points in a point cloud.
    
    Args:
        pcd: (B, N, 3) tensor
    Returns:
        shuffled: (B, N, 3) tensor with points randomly permuted
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # Random permutations (different per batch)
    idx = torch.argsort(torch.rand(B, N, device=device), dim=-1)  # (B, N)

    # Build batch index for gather
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)

    # Gather shuffled points
    return pcd[batch_idx, idx]  # (B, N, 3)


def crop_local_pcd(pcd: torch.Tensor, local_range: torch.float, num_local_points: torch.int):
    """
    Crop the point cloud to a local region around the origin with 0 padding.
    Args:
        pcd: (B, N, 3) tensor
        range: float, the radius of the local region
    """
    B, N, _ = pcd.shape
    device = pcd.device

    # get local pcd
    masked_pcds = shuffle_pcd(pcd)
    dist = torch.norm(masked_pcds, dim=-1)
    mask = dist < local_range # nan < X always returns false, so if there are nan values in pcd input, it get automatically filtered out
    masked_pcds[~mask] = float("nan")

    # sort to get all the valid points
    is_valid = mask.int()
    sort_idx = torch.argsort(is_valid, dim=-1, descending=True)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    sorted_pcds = masked_pcds[batch_idx, sort_idx]  # (B, N, 3)
    pcd_local_nan_padding = sorted_pcds[:, :num_local_points]

    avg_num_valid_points = is_valid.sum() / B
    min_num_valid_points = is_valid.sum(dim=-1).min()
    logs = {
        "local_crop/avg_num_valid_points": avg_num_valid_points.item(),
        "local_crop/min_num_valid_points": min_num_valid_points.item(),
    }

    # replace nan values as 0s
    local_pcd_zero_padding = torch.nan_to_num(pcd_local_nan_padding, nan=0.0)

    return local_pcd_zero_padding, logs


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

@torch.jit.script
def quat_unit(a):
    return normalize(a)

@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)



# TODO: hardcode for now for simplicity, but ideally this should be loaded from a config file
TRANSFORMER_CONFIGS = {
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


class Transformer_FrankaLEAP:
    def __init__(self, device, configs):
        self.device = device
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

        self.robot_dof_lower_limits = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
                -0.3140, -1.0470, -0.5060, -0.3660,
                -0.3490, -0.4700, -1.2000, -1.3400,
                -0.3140, -1.0470, -0.5060, -0.3660,
                -0.3140, -1.0470, -0.5060, -0.3660],
            device=self.device)
        self.robot_dof_upper_limits = torch.tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,
                2.2300, 1.0470, 1.8850, 2.0420,
                2.0940, 2.4430, 1.9000, 1.8800,
                2.2300, 1.0470, 1.8850, 2.0420,
                2.2300, 1.0470, 1.8850, 2.0420],
            device=self.device)
        self.hand_dof_lower_limits = self.robot_dof_lower_limits[7:]
        self.hand_dof_upper_limits = self.robot_dof_upper_limits[7:]
        self.abs_hand_actions = torch.zeros(1, 16, device=self.device)
        self.steps = 0

    def preprocess_inputs(self, obs):
        obs_input = OrderedDict()
        obs_input["q_hand"] = obs["q_hand"]
        obs_input["q_hand_ctrl_delta"] = obs["q_hand_ctrl_delta"]
        obs_input["local_pcd_t"], crop_logs = crop_local_pcd(
            obs['full_pcd_eef_frame_t'], self.local_pcd_range, self.num_local_points
        ) # (1, num_local_points, 3)
        return obs_input

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = checkpoint['model_state_dict']  # or whatever key your model state is stored under

        # remove the DDP ckpt prefix if there are any
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # remove 'module.' prefix (7 characters)
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        return checkpoint["success_rate_ep"]

    def unnormalize_leap_hand_joints(self, joint_angles: torch.Tensor, robot: bool, delta: bool = False) -> torch.Tensor:
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

    def get_step_action(self, obs_dict):
        """
        get action from the network, this is the action that goes into IsaacGym's step function, not the final action that's been executed

        Args:
            obs_dict (OrderedDict):
                full_pcd_eef_frame_t (torch.Tensor): (1, N, 3), N should be larger than num_local_points
                q_hand (torch.Tensor): (1, 16)
        Returns:
            step_actions (torch.Tensor): (action_dim)
        """
        obs_input = self.preprocess_inputs(obs_dict)

        with torch.no_grad():
            self.model.eval()
            student_actions_chunk = self.model(obs_input)

        student_actions = student_actions_chunk[0, 0, :] # (Batch, Chunk, Action_dim)

        # get the step action that goes into env.step() in sim
        step_actions = torch.clamp(student_actions, -self.clip_actions, self.clip_actions)

        return step_actions, obs_input

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

        full_pcd_eef_frame_t_b = full_pcd_eef_frame_t.unsqueeze(0).to(self.device) # (1, N, 3)
        q_hand_b = q_hand.unsqueeze(0).to(self.device) # (1, 16)

        obs_dict = OrderedDict([
            ("full_pcd_eef_frame_t", full_pcd_eef_frame_t_b),
            ("q_hand", q_hand_b),
            ("q_hand_ctrl_delta", q_hand_b - self.abs_hand_actions)
        ])

        step_action, obs_input = self.get_step_action(obs_dict)

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
        delta_hand_joint_actions_unnormalized = self.unnormalize_leap_hand_joints(hand_actions, robot="hand", delta=True)

        abs_hand_actions = q_hand + delta_hand_joint_actions_unnormalized
        abs_hand_actions = tensor_clamp(
            abs_hand_actions, self.hand_dof_lower_limits, self.hand_dof_upper_limits
        )

        self.abs_hand_actions[0, :] = abs_hand_actions
        self.steps += 1

        return ctrl_target_eef_pos.cpu().numpy(), ctrl_target_eef_quat.cpu().numpy(), abs_hand_actions.cpu().numpy(), obs_input["local_pcd_t"][0]
    

    def generate_random_inputs(self):
        """
        Generate random inputs for the get_action function using torch.rand
        """      
        # Generate random point cloud in EEF frame (N, 3)
        N = 1500  # Larger than typical num_local_points
        full_pcd_eef_frame_t = torch.rand(N, 3, device=self.device)  # Random point cloud in [0, 1)

        # Generate random hand configuration (16,)
        q_hand = torch.rand(16, device=self.device)  # In [0, 1)

        # Generate random EEF absolute pose (7,) - xyz + xyzw quaternion
        eef_pos = torch.rand(3, device=self.device)  # xyz position in [0, 1)
        eef_quat = torch.rand(4, device=self.device)  # xyzw quaternion in [0, 1)
        eef_quat = eef_quat / torch.norm(eef_quat)  # Normalize to get valid quaternion
        eef_abs_pose = torch.cat([eef_pos, eef_quat])
        return full_pcd_eef_frame_t, q_hand, eef_abs_pose



if __name__ == "__main__":
    device = "cuda:0"

    def generate_random_inputs():
        """
        Generate random inputs for the get_action function using torch.rand
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Generate random point cloud in EEF frame (N, 3)
        N = 1500  # Larger than typical num_local_points
        full_pcd_eef_frame_t = torch.rand(N, 3, device=device)  # Random point cloud in [0, 1)

        # Generate random hand configuration (16,)
        q_hand = torch.rand(16, device=device)  # In [0, 1)

        # Generate random EEF absolute pose (7,) - xyz + xyzw quaternion
        eef_pos = torch.rand(3, device=device)  # xyz position in [0, 1)
        eef_quat = torch.rand(4, device=device)  # xyzw quaternion in [0, 1)
        eef_quat = eef_quat / torch.norm(eef_quat)  # Normalize to get valid quaternion
        eef_abs_pose = torch.cat([eef_pos, eef_quat])

        return full_pcd_eef_frame_t, q_hand, eef_abs_pose

    model = Transformer_FrankaLEAP(device, TRANSFORMER_CONFIGS)

    print("warm up")
    t_1 = time.time()
    for i in range(3):
        test_input = generate_random_inputs()
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
