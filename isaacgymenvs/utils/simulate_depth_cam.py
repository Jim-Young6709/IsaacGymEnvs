import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from isaacgymenvs.utils.pcd_utils import shuffle_pcd


INTEL_455 = {
    'H': 848//4,
    'W': 480//4,
    'fov_x_deg': 87.0,
    'fov_y_deg': 58.0,
    'near_m': 0.4,
    'far_m': 6.0,
}

INTEL_435 = {
    'H': 848//4,
    'W': 480//4,
    'fov_x_deg': 85.2,
    'fov_y_deg': 58.0,
    'near_m': 0.3,
    'far_m': 3.0,
}

# ---------- camera position & view direction sampling ----------
def sample_cameras(
    B: int,
    gaze_target_pos: torch.Tensor,
    gaze_target_xyz_rand: float,
    cam_pos_rand: List[List[float]],
    device=None,
    dtype=None
):
    cam_pos_range = torch.tensor(cam_pos_rand, device=device, dtype=dtype)
    cam_pos = torch.rand((B, 3), device=device) * (cam_pos_range[1] - cam_pos_range[0]) + cam_pos_range[0]
    rand1 = (torch.rand((B, 3), device=device, dtype=dtype) - 0.5) * 2.0 # uniform rand [-1, 1)
    cam_target_pos_rand = gaze_target_pos + rand1 * gaze_target_xyz_rand
    dirs = cam_target_pos_rand - cam_pos
    dirs /= torch.norm(dirs, dim=-1, keepdim=True).clamp_min(1e-12)

    return cam_pos, dirs


# ---------- camera basis (shared) ----------
def _camera_basis_from_view_dirs(view_dirs: torch.Tensor):
    """
    Build per-batch orthonormal camera basis [u | w | v] with v = forward (+Z).
    view_dirs: (B,3)
    Returns u_hat, w_hat, v_hat each (B,3)
    """
    assert view_dirs.ndim == 2 and view_dirs.shape[-1] == 3
    device, dtype = view_dirs.device, view_dirs.dtype

    v = view_dirs / (view_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    ez = torch.tensor([0., 0., 1.], device=device, dtype=dtype).expand_as(v)
    ey = torch.tensor([0., 1., 0.], device=device, dtype=dtype).expand_as(v)
    ref = torch.where((v.abs().mul(ez).sum(-1) > 0.9).unsqueeze(-1), ey, ez)
    u = torch.cross(ref, v, dim=-1)
    u = u / (u.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    w = torch.cross(v, u, dim=-1)
    return u, w, v


# ---------- z-buffer rasterization ----------
@torch.no_grad()
def rasterize_depth_zbuffer(
    pcd: torch.Tensor,           # (B,N,3)
    view_dirs: torch.Tensor,     # (B,3)
    cam_pos: torch.Tensor,       # (B,3)
    H: int = 480, W: int = 640,
    fov_y_deg: float = 60.0,     # vertical FOV
    fov_x_deg: float = 60.0,     # horizontal FOV
    inflate_px: int = 0,         # min-pool radius in pixels (0 = off)
    clip_mode: str = "post",     # "pre" or "post" near/far clipping
    near_m: float = 0.0,
    far_m: float = None,
):
    """
    Returns:
      depth:  (B,H,W) float, +inf where no hit (after optional post-clip/inflate)
      inside: (B,N) bool, points that project in-frame & z>0 (& pass pre-clip if used)
      iu,iv:  (B,N) long pixel indices (floor)
      z:      (B,N) depths in camera space (for debugging/analysis)
      basis:  tuple of (u_hat, w_hat, v_hat), each (B,3)
    """
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
    B, N, _ = pcd.shape
    device, dtype = pcd.device, pcd.dtype

    # Camera basis
    u_hat, w_hat, v_hat = _camera_basis_from_view_dirs(view_dirs)

    # Camera coords
    rel = pcd - cam_pos[:, None, :]              # (B,N,3)
    x = (rel * u_hat[:, None, :]).sum(-1)            # (B,N)
    y = (rel * w_hat[:, None, :]).sum(-1)
    z = (rel * v_hat[:, None, :]).sum(-1)

    # Intrinsics from FOVs
    fx = 0.5 * W / math.tan(0.5 * math.radians(fov_x_deg))
    fy = 0.5 * H / math.tan(0.5 * math.radians(fov_y_deg))
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    # Project
    invz = 1.0 / z.clamp_min(1e-12)
    u_pix = fx * (x * invz) + cx
    v_pix = fy * (y * invz) + cy

    in_front = z > 0
    inside = (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H) & in_front

    # Optional "pre" clipping (ignore points outside range when building z-buffer)
    if clip_mode == "pre":
        rng = z >= near_m
        if far_m is not None:
            rng = rng & (z <= far_m)
        inside = inside & rng

    iu = u_pix.floor().clamp(0, W-1).long()
    iv = v_pix.floor().clamp(0, H-1).long()
    pix = iv * W + iu                                  # (B,N) in [0, W*H)

    # Per-pixel min depth via sort-trick (PyTorch 1.x friendly)
    K = W * H
    lin_idx = pix + (torch.arange(B, device=device)[:, None] * K)    # (B,N)
    lin_idx_flat = lin_idx.reshape(-1)                               # (B*N,)
    z_flat       = z.reshape(-1)
    inside_flat  = inside.reshape(-1)

    inf = torch.tensor(float('inf'), device=device, dtype=dtype)
    z_masked = torch.where(inside_flat, z_flat, inf)                 # outside -> +inf

    M = lin_idx_flat.numel()
    order = torch.argsort(z_masked)                                  # (BN,)
    rank  = torch.empty_like(order); rank[order] = torch.arange(M, device=device, dtype=order.dtype)
    key   = lin_idx_flat * (M + 1) + rank.long()                     # lexicographic (pixel, depth)
    perm  = torch.argsort(key)

    bins_sorted = lin_idx_flat[perm]
    z_sorted    = z_masked[perm]
    is_first = torch.ones_like(bins_sorted, dtype=torch.bool)
    is_first[1:] = (bins_sorted[1:] != bins_sorted[:-1])

    uniq_bins = bins_sorted[is_first]                                # (~U,)
    zmin_vals = z_sorted[is_first]                                   # (~U,)

    min_depth = torch.full((B*K,), float('inf'), device=device, dtype=dtype)
    min_depth[uniq_bins] = zmin_vals.to(dtype)                       # (B*K,)
    depth = min_depth.view(B, H, W)                                  # (B,H,W)

    # Inflate occluders by min-pooling (optional)
    if inflate_px > 0:
        # min-pool by max-pooling on the negative; pad with -inf so borders are handled
        neg = -depth.unsqueeze(1)                                    # (B,1,H,W)
        neg = F.pad(neg, (inflate_px, inflate_px, inflate_px, inflate_px),
                    mode='constant', value=float('-inf'))
        pooled_neg = F.max_pool2d(neg, kernel_size=2*inflate_px+1, stride=1)  # (B,1,H,W)
        depth = (-pooled_neg).squeeze(1)

    # Optional "post" clipping on the completed depth image
    if clip_mode == "post":
        if near_m > 0.0:
            depth = torch.where(depth >= near_m, depth, inf)
        if far_m is not None:
            depth = torch.where(depth <= far_m, depth, inf)

    return depth, inside, iu, iv, z, (u_hat, w_hat, v_hat), (fx, fy, cx, cy)


# ---------- back-projection (dense, one point per valid pixel) ----------
@torch.no_grad()
def backproject_depth_to_world(
    depth: torch.Tensor,         # (B,H,W), +inf where invalid
    view_dirs: torch.Tensor,     # (B,3)
    cam_pos: torch.Tensor,   # (B,3)
    intrinsics: tuple,           # (fx, fy, cx, cy)
):
    """
    Returns:
      pcd_world: (B,H,W,3) world coordinates; NaN where depth is invalid
      valid:     (B,H,W) bool mask where depth is finite
    """
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    fx, fy, cx, cy = intrinsics

    u_hat, w_hat, v_hat = _camera_basis_from_view_dirs(view_dirs)

    # pixel grid (broadcasted)
    u = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
    v = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)

    valid = torch.isfinite(depth)
    z = depth

    # camera-frame coords
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    # turn invalid to NaN for convenience
    nan = torch.tensor(float('nan'), device=device, dtype=dtype)
    x = torch.where(valid, x, nan)
    y = torch.where(valid, y, nan)
    z = torch.where(valid, z, nan)

    # world = o + x*û + y*ŵ + z*v̂
    # expand bases for broadcast: (B,1,1,3)
    uB = u_hat.view(B, 1, 1, 3)
    wB = w_hat.view(B, 1, 1, 3)
    vB = v_hat.view(B, 1, 1, 3)
    oB = cam_pos.view(B, 1, 1, 3)

    pcd_world = oB + x[..., None] * uB + y[..., None] * wB + z[..., None] * vB  # (B,H,W,3)
    return pcd_world, valid


# ---------- convenience: end-to-end helper ----------
@torch.no_grad()
def render_points_to_world_grid(
    pcd: torch.Tensor, view_dirs: torch.Tensor, cam_pos: torch.Tensor,
    cam_spec_dict: dict = INTEL_455,
    inflate_px: int = 0, clip_mode: str = "post",
    jitter_std_m: float = 0.0,         # e.g., 0.002 for ~2 mm
    jitter_mode: str = "xyz",      # "tangent" or "xyz"
):
    """
    High-level: rasterize z-buffer (with optional inflation & clipping),
    then back-project one 3D point per valid pixel into world space.
    Optionally add small metric jitter to reduce grid-like appearance.

    Returns:
      depth:      (B,H,W) float (+inf invalid)
      pcd_world:  (B,H,W,3) float (NaN invalid)
      valid:      (B,H,W) bool
    """
    H = cam_spec_dict['H']
    W = cam_spec_dict['W']
    fov_x_deg = cam_spec_dict['fov_x_deg']
    fov_y_deg = cam_spec_dict['fov_y_deg']
    near_m = cam_spec_dict['near_m']
    far_m = cam_spec_dict['far_m']

    depth, _, _, _, _, _, intr = rasterize_depth_zbuffer(
        pcd, view_dirs, cam_pos, H, W, fov_y_deg, fov_x_deg,
        inflate_px=inflate_px, clip_mode=clip_mode, near_m=near_m, far_m=far_m
    )

    pcd_world, valid = backproject_depth_to_world(depth, view_dirs, cam_pos, intr)

    # ---- Metric jitter (optional) ----
    if jitter_std_m > 0.0:
        B, H_, W_ = depth.shape
        device, dtype = depth.device, depth.dtype

        if jitter_mode.lower() == "xyz":
            # Isotropic 3D jitter in meters
            noise = torch.randn_like(pcd_world) * jitter_std_m
            pcd_world = torch.where(valid[..., None], pcd_world + noise, pcd_world)

        elif jitter_mode.lower() == "tangent":
            # Jitter in the camera's local tangent plane (û,ŵ), roughly preserves depth.
            from math import isfinite  # just to avoid unused import warnings
            u_hat, w_hat, _ = _camera_basis_from_view_dirs(view_dirs)  # (B,3) each
            uB = u_hat.view(B, 1, 1, 3)  # (B,1,1,3)
            wB = w_hat.view(B, 1, 1, 3)

            eps_u = torch.randn(B, H_, W_, 1, device=device, dtype=dtype) * jitter_std_m
            eps_w = torch.randn(B, H_, W_, 1, device=device, dtype=dtype) * jitter_std_m
            jitter = eps_u * uB + eps_w * wB  # (B,H,W,3)

            pcd_world = torch.where(valid[..., None], pcd_world + jitter, pcd_world)

        else:
            raise ValueError(f"Unknown jitter_mode '{jitter_mode}'. Use 'tangent' or 'xyz'.")

    return depth, pcd_world, valid


# ---------- subsample to fixed number of points ----------
def subsample_to_M_rowloop(
    pcs: List[torch.Tensor],   # list of (Xi, 3)
    M: int,
    generator: Optional[torch.Generator] = None,
    fill_when_empty: float = 0.0,
    target_device: Optional[torch.device] = None,
    target_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
    pcd: (B, M, 3)
    idx: (B, M) int64 indices into each original cloud; -1 where Xi == 0.

    Behavior:
    - Xi >= M: sample M without replacement
    - Xi <  M: take all Xi without replacement, then fill the tail with replacement
    """
    assert len(pcs) > 0, "pcs must be non-empty"
    B = len(pcs)

    # Normalize output device/dtype
    base = pcs[0]
    dev  = target_device if target_device is not None else base.device
    dtype = target_dtype if target_dtype is not None else base.dtype

    pcd = torch.empty((B, M, 3), device=dev, dtype=dtype)
    idx = torch.full((B, M), -1, device=dev, dtype=torch.long)

    for b, P in enumerate(pcs):
        # Handle empty
        if P.numel() == 0 or P.shape[0] == 0:
            pcd[b].fill_(fill_when_empty)
            continue

        # Move to common device/dtype only if needed
        if P.device != dev or P.dtype != dtype:
            Q = P.to(device=dev, dtype=dtype)
        else:
            Q = P

        N = Q.shape[0]

        if N >= M:
            choice = torch.randperm(N, device=dev, generator=generator)[:M]
        else:
            # head: all N without replacement; tail: (M-N) with replacement
            head = torch.randperm(N, device=dev, generator=generator)
            tail = torch.randint(N, (M - N,), device=dev, generator=generator)
            choice = torch.cat([head, tail], dim=0)

        pcd[b] = Q[choice]
        idx[b] = choice

    return pcd, idx


# fully integrated single function wrapper
def simulate_depth_cam_render(
    pcd: torch.Tensor, gaze_target_pos: torch.Tensor, num_points: int,
    inflate_px: int = 2, jitter_std_m: float = 0.004,
    gaze_target_xyz_rand: float = 0.1,
    cam_pos_rand=[[-0.3, -0.2, 0.2], [0.3, -0.6, 0.9]],
):
    batch_size = pcd.shape[0]
    device = pcd.device

    cam_pos, view_dirs = sample_cameras(
        B=batch_size,
        gaze_target_pos=gaze_target_pos,
        gaze_target_xyz_rand=gaze_target_xyz_rand,
        cam_pos_rand=cam_pos_rand,
        device=device,
        dtype=pcd.dtype
    )

    depth, pcd_world, valid = render_points_to_world_grid(
        pcd, # (B, N, 3)
        view_dirs, # (B, 3), normalized view direction vector
        cam_pos, # (B, 3), xyz cam pos
        inflate_px=inflate_px, # 'pooling' kernal size, 2 is 5x5
        jitter_std_m=jitter_std_m, # noise
    )

    rendered_pcd = pcd_world.view(batch_size, -1, 3)
    num_total_points = rendered_pcd.shape[1]
    rendered_pcd = shuffle_pcd(rendered_pcd)
    nan_mask = torch.isnan(rendered_pcd).any(dim=-1)
    sort_key = nan_mask.int() # 0: valid ; 1: invalid
    sort_idx = torch.argsort(sort_key, dim=-1)
    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, num_total_points)
    sorted_pcds = rendered_pcd[batch_idx, sort_idx]  # (B, N, 3)

    avg_num_valid_points = num_total_points - nan_mask.sum() / batch_size
    min_num_valid_points = (num_total_points - nan_mask.sum(dim=-1)).min()
    logs = {
        "sim_depth_cam_render/avg_num_valid_points": avg_num_valid_points.item(),
        "sim_depth_cam_render/min_num_valid_points": min_num_valid_points.item(),
    }

    pcd_nan_padding = sorted_pcds[:, :num_points]

    return pcd_nan_padding, logs



# ------------------------------ from pose utils ------------------------------
def _camera_basis_from_pose_x_forward(
    camera_pose: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct camera basis [u | w | v] directly from camera_pose.

    camera_pose: (B, 7) = [x, y, z, qx, qy, qz, qw], quaternion in (xyzw),
                 mapping camera frame -> world frame.

    We assume the *actual* camera forward axis is +X in the camera frame.
    For the renderer's internal coordinates (x_r, y_r, z_r), we use:

        z_r = X_cam   (depth / forward)
        x_r = Y_cam   (horizontal)
        y_r = Z_cam   (vertical)

    So in world coordinates:

        v_hat = R * e_x  (depth axis)
        u_hat = R * e_y  (horizontal axis)
        w_hat = R * e_z  (vertical axis)
    """
    assert camera_pose.shape[-1] == 7, "camera_pose must have last dim = 7 (pos + quat_xyzw)"

    q = camera_pose[..., 3:7]  # (B,4) (qx, qy, qz, qw)
    # Normalize quaternion for safety
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    qx, qy, qz, qw = q.unbind(-1)

    # Precompute products
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    # Rotation matrix R (camera -> world) for quat (x, y, z, w)
    # R = [[1-2(y²+z²), 2(xy-wz),   2(xz+wy)],
    #      [2(xy+wz),   1-2(x²+z²), 2(yz-wx)],
    #      [2(xz-wy),   2(yz+wx),   1-2(x²+y²)]]
    R00 = 1.0 - 2.0 * (yy + zz)
    R01 = 2.0 * (xy - wz)
    R02 = 2.0 * (xz + wy)

    R10 = 2.0 * (xy + wz)
    R11 = 1.0 - 2.0 * (xx + zz)
    R12 = 2.0 * (yz - wx)

    R20 = 2.0 * (xz - wy)
    R21 = 2.0 * (yz + wx)
    R22 = 1.0 - 2.0 * (xx + yy)

    # Columns of R are world-space camera axes:
    # col0 = R * e_x = [R00, R10, R20]ᵀ = X_cam axis (forward)
    # col1 = R * e_y = [R01, R11, R21]ᵀ = Y_cam axis
    # col2 = R * e_z = [R02, R12, R22]ᵀ = Z_cam axis

    # Map camera axes -> renderer basis as described above
    v_hat = torch.stack([R00, R10, R20], dim=-1)  # depth / forward (+X_cam)
    u_hat = torch.stack([R01, R11, R21], dim=-1)  # horizontal (Y_cam)
    w_hat = torch.stack([R02, R12, R22], dim=-1)  # vertical (Z_cam)

    # Re-normalize to be extra safe
    v_hat = v_hat / v_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u_hat = u_hat / u_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w_hat = w_hat / w_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return u_hat, w_hat, v_hat


@torch.no_grad()
def rasterize_depth_zbuffer_from_pose(
    pcd: torch.Tensor,           # (B,N,3)
    camera_pose: torch.Tensor,   # (B,7) [pos, quat_xyzw], forward = +X_cam
    H: int = 480, W: int = 640,
    fov_y_deg: float = 60.0,
    fov_x_deg: float = 60.0,
    inflate_px: int = 0,
    clip_mode: str = "post",
    near_m: float = 0.0,
    far_m: float = None,
):
    """
    Pose-based variant of rasterize_depth_zbuffer.

    Returns:
      depth:  (B,H,W) float, +inf where no hit
      inside: (B,N) bool, points that project in-frame & z>0 (& pre-clip if used)
      iu,iv:  (B,N) long pixel indices
      z:      (B,N) depths in camera space (along +X_cam, mapped to v_hat)
      basis:  (u_hat, w_hat, v_hat), each (B,3)
      intr:   (fx, fy, cx, cy)
    """
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
    B, N, _ = pcd.shape
    device, dtype = pcd.device, pcd.dtype

    cam_pos = camera_pose[:, 0:3]  # (B,3)
    u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)

    # Camera coords in our [u_hat, w_hat, v_hat] basis
    rel = pcd - cam_pos[:, None, :]       # (B,N,3)
    x = (rel * u_hat[:, None, :]).sum(-1) # (B,N)
    y = (rel * w_hat[:, None, :]).sum(-1) # (B,N)
    z = (rel * v_hat[:, None, :]).sum(-1) # (B,N) depth along +X_cam

    # Intrinsics from FOVs
    fx = 0.5 * W / math.tan(0.5 * math.radians(fov_x_deg))
    fy = 0.5 * H / math.tan(0.5 * math.radians(fov_y_deg))
    cx = (W - 1) * 0.5
    cy = (H - 1) * 0.5

    # Project
    invz = 1.0 / z.clamp_min(1e-12)
    u_pix = fx * (x * invz) + cx
    v_pix = fy * (y * invz) + cy

    in_front = z > 0
    inside = (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H) & in_front

    if clip_mode == "pre":
        rng = z >= near_m
        if far_m is not None:
            rng = rng & (z <= far_m)
        inside = inside & rng

    iu = u_pix.floor().clamp(0, W - 1).long()
    iv = v_pix.floor().clamp(0, H - 1).long()
    pix = iv * W + iu  # (B,N) in [0, W*H)

    # ---- same z-buffer sort trick as original ----
    K = W * H
    lin_idx = pix + (torch.arange(B, device=device)[:, None] * K)  # (B,N)
    lin_idx_flat = lin_idx.reshape(-1)                             # (B*N,)
    z_flat = z.reshape(-1)
    inside_flat = inside.reshape(-1)

    inf = torch.tensor(float('inf'), device=device, dtype=dtype)
    z_masked = torch.where(inside_flat, z_flat, inf)

    M = lin_idx_flat.numel()
    order = torch.argsort(z_masked)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(M, device=device, dtype=order.dtype)
    key = lin_idx_flat * (M + 1) + rank.long()
    perm = torch.argsort(key)

    bins_sorted = lin_idx_flat[perm]
    z_sorted = z_masked[perm]
    is_first = torch.ones_like(bins_sorted, dtype=torch.bool)
    is_first[1:] = (bins_sorted[1:] != bins_sorted[:-1])

    uniq_bins = bins_sorted[is_first]
    zmin_vals = z_sorted[is_first]

    min_depth = torch.full((B * K,), float('inf'), device=device, dtype=dtype)
    min_depth[uniq_bins] = zmin_vals.to(dtype)
    depth = min_depth.view(B, H, W)

    # Inflate occluders (optional)
    if inflate_px > 0:
        neg = -depth.unsqueeze(1)  # (B,1,H,W)
        neg = F.pad(
            neg,
            (inflate_px, inflate_px, inflate_px, inflate_px),
            mode="constant",
            value=float("-inf"),
        )
        pooled_neg = F.max_pool2d(neg, kernel_size=2 * inflate_px + 1, stride=1)
        depth = (-pooled_neg).squeeze(1)

    # Post clipping
    if clip_mode == "post":
        if near_m > 0.0:
            depth = torch.where(depth >= near_m, depth, inf)
        if far_m is not None:
            depth = torch.where(depth <= far_m, depth, inf)

    return depth, inside, iu, iv, z, (u_hat, w_hat, v_hat), (fx, fy, cx, cy)


@torch.no_grad()
def backproject_depth_to_world_from_pose(
    depth: torch.Tensor,        # (B,H,W), +inf where invalid
    camera_pose: torch.Tensor,  # (B,7)
    intrinsics: tuple,          # (fx, fy, cx, cy)
):
    """
    Back-project using camera_pose (full orientation, x-forward).
    Returns:
      pcd_world: (B,H,W,3) world coordinates; NaN where depth is invalid
      valid:     (B,H,W) bool mask where depth is finite
    """
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    fx, fy, cx, cy = intrinsics

    cam_pos = camera_pose[:, 0:3]
    u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)

    # pixel grid
    u = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
    v = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)

    valid = torch.isfinite(depth)
    z = depth

    x = (u - cx) / fx * z  # corresponds to Y_cam
    y = (v - cy) / fy * z  # corresponds to Z_cam

    nan = torch.tensor(float("nan"), device=device, dtype=dtype)
    x = torch.where(valid, x, nan)
    y = torch.where(valid, y, nan)
    z = torch.where(valid, z, nan)

    # world = o + x*û + y*ŵ + z*v̂
    uB = u_hat.view(B, 1, 1, 3)
    wB = w_hat.view(B, 1, 1, 3)
    vB = v_hat.view(B, 1, 1, 3)
    oB = cam_pos.view(B, 1, 1, 3)

    pcd_world = oB + x[..., None] * uB + y[..., None] * wB + z[..., None] * vB
    return pcd_world, valid



@torch.no_grad()
def render_points_to_world_grid_from_pose(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,      # (B,7)
    cam_spec_dict: dict = INTEL_455,
    inflate_px: int = 0,
    clip_mode: str = "post",
    jitter_std_m: float = 0.0,
    jitter_mode: str = "xyz",
):
    """
    Pose-based high-level helper:
    - rasterize depth using camera_pose (full orientation, +X forward)
    - back-project one point per valid pixel into world
    - optional metric jitter
    """
    H = cam_spec_dict["H"]
    W = cam_spec_dict["W"]
    fov_x_deg = cam_spec_dict["fov_x_deg"]
    fov_y_deg = cam_spec_dict["fov_y_deg"]
    near_m = cam_spec_dict["near_m"]
    far_m = cam_spec_dict["far_m"]

    depth, _, _, _, _, _, intr = rasterize_depth_zbuffer_from_pose(
        pcd,
        camera_pose,
        H,
        W,
        fov_y_deg,
        fov_x_deg,
        inflate_px=inflate_px,
        clip_mode=clip_mode,
        near_m=near_m,
        far_m=far_m,
    )



    pcd_world, valid = backproject_depth_to_world_from_pose(depth, camera_pose, intr)

    # ---- Metric jitter (optional) ----
    if jitter_std_m > 0.0:
        B, H_, W_ = depth.shape
        device, dtype = depth.device, depth.dtype

        if jitter_mode.lower() == "xyz":
            noise = torch.randn_like(pcd_world) * jitter_std_m
            pcd_world = torch.where(valid[..., None], pcd_world + noise, pcd_world)

        elif jitter_mode.lower() == "tangent":
            u_hat, w_hat, _ = _camera_basis_from_pose_x_forward(camera_pose)
            uB = u_hat.view(B, 1, 1, 3)
            wB = w_hat.view(B, 1, 1, 3)

            eps_u = torch.randn(B, H_, W_, 1, device=device, dtype=dtype) * jitter_std_m
            eps_w = torch.randn(B, H_, W_, 1, device=device, dtype=dtype) * jitter_std_m
            jitter = eps_u * uB + eps_w * wB

            pcd_world = torch.where(valid[..., None], pcd_world + jitter, pcd_world)
        else:
            raise ValueError(f"Unknown jitter_mode '{jitter_mode}'. Use 'tangent' or 'xyz'.")

    return depth, pcd_world, valid



def simulate_depth_cam_render_from_pose(
    pcd: torch.Tensor, 
    camera_pose: torch.Tensor,    # (B,7): [x,y,z,qx,qy,qz,qw], camera->world, +X forward
    num_points: int,
    inflate_px: int = 2,
    jitter_std_m: float = 0.004,
    cam_spec_dict: dict = INTEL_435,
):
    """
    Expects full scene point cloud and camera pose. Renders depth, back-projects to world grid, adds jitter, then subsamples to fixed num_points.
    Args:
        pcd (torch.Tensor): _description_
        camera_pose (torch.Tensor): (B,7): [x,y,z,qx,qy,qz,qw], camera->world, +X forward
        num_points (int): _description_
        inflate_px (int, optional): _description_. Defaults to 2.
        jitter_std_m (float, optional): _description_. Defaults to 0.004.
        cam_spec_dict (dict, optional): Current options are INTEL_435 and INTEL_455.

    Returns:
        _type_: _description_
    """
    batch_size = pcd.shape[0]
    device = pcd.device

    depth, pcd_world, valid = render_points_to_world_grid_from_pose(
        pcd,
        camera_pose,
        cam_spec_dict=cam_spec_dict,
        inflate_px=inflate_px,
        jitter_std_m=jitter_std_m,
    )

    rendered_pcd = pcd_world.view(batch_size, -1, 3)
    num_total_points = rendered_pcd.shape[1]
    rendered_pcd = shuffle_pcd(rendered_pcd)

    nan_mask = torch.isnan(rendered_pcd).any(dim=-1)
    sort_key = nan_mask.int()  # 0: valid ; 1: invalid
    sort_idx = torch.argsort(sort_key, dim=-1)
    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, num_total_points)
    sorted_pcds = rendered_pcd[batch_idx, sort_idx]  # (B, N, 3)

    avg_num_valid_points = num_total_points - nan_mask.sum() / batch_size
    min_num_valid_points = (num_total_points - nan_mask.sum(dim=-1)).min()
    logs = {
        "sim_depth_cam_render/avg_num_valid_points": avg_num_valid_points.item(),
        "sim_depth_cam_render/min_num_valid_points": min_num_valid_points.item(),
    }

    pcd_nan_padding = sorted_pcds[:, :num_points]

    return pcd_nan_padding, logs