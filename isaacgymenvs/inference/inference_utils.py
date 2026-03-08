import torch


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


def crop_local_pcd(
    pcd: torch.Tensor,
    local_range: torch.float,
    num_local_points: torch.int,
    is_cylindrical: bool = False,
    crop_center: torch.Tensor = None,
    x_direction_cutoff: torch.float = -0.5,
    log_name: str = "",
):
    """
    Crop the point cloud to a local region around the origin with 0 padding.
    Args:
        pcd: (B, N, 3) tensor
        local_range: float, the radius of the local region
        crop_center: (B, 3)
    """
    B, N, _ = pcd.shape
    device = pcd.device

    if crop_center is None:
        crop_center = torch.zeros((B, 3), device=pcd.device, dtype=pcd.dtype)
    crop_center = crop_center.unsqueeze(1)
    pcd_centered = pcd - crop_center

    # get local pcd
    masked_pcds = shuffle_pcd(pcd_centered)
    if is_cylindrical:
        dist = torch.norm(masked_pcds[..., :2], dim=-1)
    else:
        dist = torch.norm(masked_pcds, dim=-1)
    mask = dist < local_range # nan < X always returns false, so if there are nan values in pcd input, it get automatically filtered out
    masked_pcds[~mask] = float("nan")

    if x_direction_cutoff is not None:
        x_dir_mask = masked_pcds[:, :, 0] > x_direction_cutoff
        masked_pcds[~x_dir_mask] = float("nan")

    # sort to get all the valid points
    is_valid = mask.int()
    sort_idx = torch.argsort(is_valid, dim=-1, descending=True)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    sorted_pcds = masked_pcds[batch_idx, sort_idx]  # (B, N, 3)
    pcd_local_nan_padding = sorted_pcds[:, :num_local_points]

    avg_num_valid_points = is_valid.sum() / B
    min_num_valid_points = is_valid.sum(dim=-1).min()

    crop_type = "cylindrical" if is_cylindrical else "spherical"
    logs = {
        f"{log_name}_local_{crop_type}_crop/avg_num_valid_points": avg_num_valid_points.item(),
        f"{log_name}_local_{crop_type}_crop/min_num_valid_points": min_num_valid_points.item(),
    }

    # replace nan values as 0s
    local_pcd_zero_padding = torch.nan_to_num(pcd_local_nan_padding, nan=0.0)

    # shift the pcd back, TODO: note now the padded zeros will get shifted to the crop center, not sure if this is a good idea
    cropped_pcd = local_pcd_zero_padding + crop_center

    return cropped_pcd, logs
