# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from isaacgym.torch_utils import quat_from_angle_axis, quat_mul


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_matrix_ig(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions (xyzw) to rotation matrices.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def sample_random_quaternion_torch(batch_size=1):
    u = torch.rand(batch_size, 3).cuda()

    qx = torch.sqrt(1 - u[:, 0]) * torch.sin(2 * torch.pi * u[:, 1])
    qy = torch.sqrt(1 - u[:, 0]) * torch.cos(2 * torch.pi * u[:, 1])
    qz = torch.sqrt(u[:, 0]) * torch.sin(2 * torch.pi * u[:, 2])
    qw = torch.sqrt(u[:, 0]) * torch.cos(2 * torch.pi * u[:, 2])

    return torch.stack([qx, qy, qz, qw], dim=-1)


def sample_spherical_shell(r_range: list, n_samples: int = 1, device='cpu'):
    # Step 1: Random direction using Gaussian normalization
    vec = torch.randn(n_samples, 3, device=device)         # random 3D vector
    vec = vec / vec.norm(dim=1, keepdim=True)              # normalize to unit length

    # Step 2: Random radius in [r_min, r_max]
    radius = torch.empty(n_samples, 1, device=device).uniform_(r_range[0], r_range[1])

    vec[:, 2] = torch.abs(vec[:, 2])  # ensure z-component is non-negative

    return radius * vec  # shape: (n_samples, 3)


def matrix_to_quaternion_ig(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions (xyzw) with real part last.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last (x, y, z, w), as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ))

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    quat_real_first = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
    ].reshape(batch_dim + (4,))

    # Reorder to real-last: (x, y, z, w)
    return torch.roll(quat_real_first, shifts=-1, dims=-1)


def A2B_quaternion(posA: torch.Tensor, posB: torch.Tensor, max_angle_deg=5.0, right_axis="x"):
    """
    right_axis: "x" or "y", which axis is perpendicular to the 'up' direction
    """
    B = posA.shape[0]
    device = posA.device
    up=torch.tensor([[0.0, 0.0, 1.0]]*B, device=device)

    # Forward direction
    forward = posB - posA
    forward = forward / torch.norm(forward, dim=-1, keepdim=True)

    # Right vector
    right = torch.cross(up, forward)
    right = right / torch.norm(right, dim=-1, keepdim=True)

    if right_axis == "x":
        # Corrected up vector
        up_corrected = torch.cross(forward, right)

        # Rotation matrix and quaternion
        rot_mat = torch.stack((right, up_corrected, forward), dim=-1)
    elif right_axis == "y":
        x_axis = torch.cross(right, forward)
        rot_mat = torch.stack((x_axis, right, forward), dim=-1)

    quat = matrix_to_quaternion_ig(rot_mat)

    # Add small random rotation
    if max_angle_deg > 0:
        max_angle_rad = max_angle_deg * torch.pi / 180.0
        rand_axis = torch.randn_like(posA, device=device)
        rand_axis = rand_axis / torch.norm(rand_axis, dim=-1, keepdim=True)
        rand_angle = (torch.rand(posA.shape[0], device=device) - 0.5) * 2 * max_angle_rad
        rand_quat = quat_from_angle_axis(rand_angle, rand_axis)

        quat = quat_mul(rand_quat, quat)

    return quat

def se2_transform(delta_action, theta_B2A):
    """
    se2 transform

    Args:
        delta_action: (B, 3) dx, dy, dtheta; delta action in frame B
        theta: (B,) current yaw angle of frame B in frame A
    """
    dx = delta_action[..., 0]
    dy = delta_action[..., 1]
    dtheta = delta_action[..., 2]

    c = torch.cos(theta_B2A)
    s = torch.sin(theta_B2A)

    dx_A = c * dx - s * dy
    dy_A = s * dx + c * dy

    return torch.stack((dx_A, dy_A, dtheta), dim=-1)
