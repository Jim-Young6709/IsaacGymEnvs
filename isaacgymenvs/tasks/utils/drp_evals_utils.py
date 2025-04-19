
import torch
import numpy as np

from isaacgym.torch_utils import *



def orientation_error(q1, q2):
    """
    batched orientation error computation
    input shape [B, 4(xyzw)], input ordering doesn't matter
    return absolute difference in degrees
    """
    assert q1.shape == q2.shape, "Desired and current orientations must have the same shape"

    cc = quat_conjugate(q2)
    q_r = quat_mul(q1, cc)

    # Compute the angle difference using the scalar part (w) of q_r
    w = torch.abs(q_r[:, 3])
    err = 2 * torch.acos(torch.clamp(w, -1.0, 1.0)) / torch.pi * 180  # Clamp for numerical stability, return in degrees
    return err

def random_quaternion_xyzw():
    u1, u2, u3 = np.random.uniform(0, 1, 3)
    qx = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    qy = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    return np.array([qx, qy, qz, qw])


def quaternion_to_rotation_matrix(q):
    # q: (..., 4) -> (..., 3, 3)
    # (qx, qy, qz, qw)
    x, y, z, w = q.unbind(-1)

    B = q.shape[:-1]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = torch.stack([
        ww + xx - yy - zz, 2 * (xy - wz),       2 * (xz + wy),
        2 * (xy + wz),     ww - xx + yy - zz,   2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),       ww - xx - yy + zz
    ], dim=-1).reshape(*B, 3, 3)
    return rot


def transform_pcds_to_world(pcds_local, poses):
    # pcds_local: (N, D, P, 3)
    # poses: (N, D, 7) -> (x, y, z, qx, qy, qz, qw)
    trans = poses[:, :, :3]  # (N, D, 3)
    quat = poses[:, :, 3:]   # (N, D, 4)

    rot = quaternion_to_rotation_matrix(quat).to(pcds_local.dtype)  # (N, D, 3, 3)
    
    # Transform pointclouds
    pcds_local = pcds_local.unsqueeze(-1)  # (N, D, P, 3, 1)
    pcds_rotated = torch.matmul(rot.unsqueeze(2), pcds_local).squeeze(-1)  # (N, D, P, 3)
    pcds_world = pcds_rotated + trans.unsqueeze(2)  # (N, D, P, 3)
    return pcds_world