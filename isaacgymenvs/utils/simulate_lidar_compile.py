#!/usr/bin/env python3
"""
simulate_lidar.py (FAST + torch.compile for fixed shapes)

CODEX LIDAR MERGE: copied in from old_urdf for the lidar/depth DAgger merge.

Speedups:
1) First-return per angular bin: scatter_reduce_(amin)  (O(N) vs O(N log N) argsort trick)
2) Cache bin-center ray directions (avoid trig / grid construction every call)
3) torch.compile path tuned for FIXED B and FIXED N (dynamic=False)
   - Suppression neighborhood uses constant (python-int) kernel_size
   - Avoids F.pad(mode="circular") by doing wrap via torch.cat (compile-friendly)

Assumptions for best performance:
- Batch size B does not change
- Number of points N does not change
- num_polar / num_azimuth, suppress_bins, near/far, eps params do not change

Coordinate conventions:
- lidar_pose: (B,7) = [x,y,z,qx,qy,qz,qw], quat is xyzw, mapping LiDAR->world.
- LiDAR scans hemisphere where +Z_lidar points outward (we keep points with z_lidar > 0).
- +X_lidar defines azimuth zero reference (doesn’t matter much).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ------------------------------ caches ------------------------------
_dir_cache: dict = {}        # (Hp,Wp,device,dtype) -> dir_flat (K,3) in LiDAR frame
_compiled_cache: dict = {}   # (params+device+dtype) -> compiled callable


# ------------------------------ utils ------------------------------
def shuffle_pcd(pcd: torch.Tensor) -> torch.Tensor:
    """pcd: (B, N, 3) -> shuffle per-batch."""
    B, N, _ = pcd.shape
    device = pcd.device
    idx = torch.argsort(torch.rand(B, N, device=device), dim=-1)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    return pcd[batch_idx, idx]


def rotmat_from_quat_xyzw(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    q: (...,4) in xyzw
    Returns: (...,3,3) rotation matrix mapping local(frame) -> world.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    qx, qy, qz, qw = q.unbind(-1)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R00 = 1.0 - 2.0 * (yy + zz)
    R01 = 2.0 * (xy - wz)
    R02 = 2.0 * (xz + wy)

    R10 = 2.0 * (xy + wz)
    R11 = 1.0 - 2.0 * (xx + zz)
    R12 = 2.0 * (yz - wx)

    R20 = 2.0 * (xz - wy)
    R21 = 2.0 * (yz + wx)
    R22 = 1.0 - 2.0 * (xx + yy)

    return torch.stack(
        [
            torch.stack([R00, R01, R02], dim=-1),
            torch.stack([R10, R11, R12], dim=-1),
            torch.stack([R20, R21, R22], dim=-1),
        ],
        dim=-2,
    )  # (...,3,3)


def _get_dir_flat(Hp: int, Wp: int, device, dtype) -> torch.Tensor:
    """
    Returns dir_flat: (K,3) LiDAR-frame unit directions for bin centers.
    Hemisphere:
      theta in [0, pi/2] from +Z (0 = +Z, pi/2 = XY plane)
      phi   in [-pi, pi) around +Z
    """
    key = (int(Hp), int(Wp), device, dtype)
    if key in _dir_cache:
        return _dir_cache[key]

    # bin centers in normalized coords
    iu_c = (torch.arange(Wp, device=device, dtype=dtype) + 0.5) / float(Wp)  # (Wp,)
    iv_c = (torch.arange(Hp, device=device, dtype=dtype) + 0.5) / float(Hp)  # (Hp,)

    phi = iu_c * (2.0 * math.pi) - math.pi                     # (Wp,)
    theta = iv_c * (0.5 * math.pi)                              # (Hp,)

    # broadcast to (Hp,Wp)
    theta2d = theta[:, None]                                    # (Hp,1)
    phi2d = phi[None, :]                                        # (1,Wp)

    sin_t = torch.sin(theta2d)                                  # (Hp,1)
    cos_t = torch.cos(theta2d)                                  # (Hp,1)
    cos_p = torch.cos(phi2d)                                    # (1,Wp)
    sin_p = torch.sin(phi2d)                                    # (1,Wp)

    x = sin_t * cos_p                                           # (Hp,Wp)
    y = sin_t * sin_p                                           # (Hp,Wp)
    z = cos_t.expand_as(x)                                      # (Hp,Wp)

    dir_hw3 = torch.stack([x, y, z], dim=-1)                    # (Hp,Wp,3)
    dir_flat = dir_hw3.reshape(Hp * Wp, 3).contiguous()         # (K,3)

    _dir_cache[key] = dir_flat
    return dir_flat


# ------------------------------ eager fast core ------------------------------
@torch.no_grad()
def render_lidar_bins_to_world_from_pose_fast(
    pcd: torch.Tensor,          # (B,N,3) world
    lidar_pose: torch.Tensor,   # (B,7) world_T_lidar (xyzw)
    num_azimuth: int,
    num_polar: int,
    near_m: float,
    far_m: Optional[float],
    suppress_bins: int,
    occlusion_eps_m: float,
    occlusion_eps_rel: float,
    jitter_std_m: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      pts_w:      (B,K,3) world-frame points with NaNs for empty bins
      valid_flat: (B,K) bool
    """
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
    assert lidar_pose.ndim == 2 and lidar_pose.shape[-1] == 7

    B, N, _ = pcd.shape
    device, dtype = pcd.device, pcd.dtype

    Hp = int(num_polar)
    Wp = int(num_azimuth)
    K = Hp * Wp

    # pose
    o = lidar_pose[:, 0:3]                     # (B,3)
    q = lidar_pose[:, 3:7]                     # (B,4) xyzw
    R = rotmat_from_quat_xyzw(q).to(dtype=dtype)  # (B,3,3) lidar->world
    Rt = R.transpose(-1, -2)                   # (B,3,3) world->lidar

    # world -> lidar
    rel_w = pcd - o[:, None, :]                                        # (B,N,3)
    rel_l = torch.matmul(Rt[:, None, :, :], rel_w[..., None]).squeeze(-1)  # (B,N,3)
    x, y, z = rel_l[..., 0], rel_l[..., 1], rel_l[..., 2]

    r = torch.sqrt((x * x + y * y + z * z).clamp_min(1e-24))            # (B,N)

    inside = z > 0.0
    if near_m is not None and near_m > 0.0:
        inside = inside & (r >= float(near_m))
    far_val = float("inf") if far_m is None else float(far_m)
    inside = inside & (r <= far_val)

    # angles -> bins
    phi = torch.atan2(y, x)                                             # [-pi, pi]
    u = (phi + math.pi) / (2.0 * math.pi)
    u = u - torch.floor(u)                                              # [0,1)

    zr = (z / r.clamp_min(1e-12)).clamp(0.0, 1.0)                       # [0,1]
    theta = torch.acos(zr)                                              # [0, pi/2]
    v = theta / (0.5 * math.pi)                                         # [0,1]

    iu = torch.floor(u * float(Wp)).clamp(0, Wp - 1).long()             # (B,N)
    iv = torch.floor(v * float(Hp)).clamp(0, Hp - 1).long()             # (B,N)
    pix = iv * Wp + iu                                                  # (B,N) in [0,K)

    # first return per bin via scatter_reduce_(amin)
    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError("Tensor.scatter_reduce_ not found. Need PyTorch >= 1.12 (PyTorch 2.x is fine).")

    inf = torch.full((), float("inf"), device=device, dtype=dtype)
    r_masked = torch.where(inside, r, inf)                              # (B,N)

    range_flat = torch.full((B, K), float("inf"), device=device, dtype=dtype)
    range_flat.scatter_reduce_(dim=1, index=pix, src=r_masked, reduce="amin", include_self=True)
    range_img = range_flat.view(B, Hp, Wp)                              # (B,Hp,Wp)

    # leak-through suppression (no hole filling)
    if suppress_bins > 0:
        s = int(suppress_bins)
        k = int(2 * s + 1)

        neg = -range_img.unsqueeze(1)                                   # (B,1,Hp,Wp)

        # width wrap (azimuth): cat last s + mid + first s
        neg = torch.cat([neg[..., -s:], neg, neg[..., :s]], dim=-1)     # (B,1,Hp,Wp+2s)

        # height replicate: cat repeated first/last rows
        top = neg[:, :, 0:1, :].expand(-1, -1, s, -1)
        bot = neg[:, :, -1:, :].expand(-1, -1, s, -1)
        neg = torch.cat([top, neg, bot], dim=-2)                        # (B,1,Hp+2s,Wp+2s)

        pooled = F.max_pool2d(neg, kernel_size=(k, k), stride=1)        # (B,1,Hp,Wp)
        neighbor_min = (-pooled).squeeze(1)                             # (B,Hp,Wp)

        eps = float(occlusion_eps_m) + float(occlusion_eps_rel) * neighbor_min.clamp_min(0.0)
        suppress = torch.isfinite(range_img) & torch.isfinite(neighbor_min) & (range_img > neighbor_min + eps)

        range_img = torch.where(suppress, inf, range_img)
        range_flat = range_img.view(B, K)

    valid_flat = torch.isfinite(range_flat)                             # (B,K)

    # backproject using cached direction grid (K,3)
    dir_flat = _get_dir_flat(Hp, Wp, device, dtype)                     # (K,3)
    pts_l = dir_flat.unsqueeze(0) * range_flat.unsqueeze(-1)            # (B,K,3)

    # lidar -> world: o + R @ pts_l
    pts_w = torch.matmul(R, pts_l.transpose(1, 2)).transpose(1, 2) + o[:, None, :]  # (B,K,3)

    nan = torch.full((), float("nan"), device=device, dtype=dtype)
    pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w, nan)

    if jitter_std_m > 0.0:
        noise = torch.randn_like(pts_w) * float(jitter_std_m)
        pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w + noise, pts_w)

    return pts_w, valid_flat


# ------------------------------ torch.compile path (FIXED SHAPES) ------------------------------
def get_compiled_lidar_renderer_fixed_shapes(
    num_azimuth: int,
    num_polar: int,
    near_m: float,
    far_m: Optional[float],
    suppress_bins: int,
    occlusion_eps_m: float,
    occlusion_eps_rel: float,
    compile_mode: str = "max-autotune",
):
    """
    Returns compiled callable:
        fn(pcd: (B,N,3), lidar_pose: (B,7), jitter_std: scalar tensor) -> (pts_w: (B,K,3), valid_flat: (B,K))

    IMPORTANT: FIXED shapes (B and N do not change), and all params passed here remain constant.
    """
    Hp = int(num_polar)
    Wp = int(num_azimuth)
    K = Hp * Wp

    far_val = float("inf") if far_m is None else float(far_m)
    DO_SUPPRESS = (int(suppress_bins) > 0)
    s = int(suppress_bins)
    k = int(2 * s + 1)

    def _get_or_build(device, dtype):
        key = (Hp, Wp, float(near_m), float(far_val), int(suppress_bins), float(occlusion_eps_m), float(occlusion_eps_rel),
               compile_mode, device, dtype)
        if key in _compiled_cache:
            return _compiled_cache[key]

        # cached bin-center directions as a constant tensor in the closure
        dir_flat = _get_dir_flat(Hp, Wp, device, dtype)  # (K,3)

        @torch.no_grad()
        def _compiled_fn(pcd: torch.Tensor, lidar_pose: torch.Tensor, jitter_std: torch.Tensor):
            B = pcd.shape[0]

            o = lidar_pose[:, 0:3]
            q = lidar_pose[:, 3:7]
            R = rotmat_from_quat_xyzw(q).to(dtype=pcd.dtype)
            Rt = R.transpose(-1, -2)

            rel_w = pcd - o[:, None, :]
            rel_l = torch.matmul(Rt[:, None, :, :], rel_w[..., None]).squeeze(-1)
            x = rel_l[..., 0]
            y = rel_l[..., 1]
            z = rel_l[..., 2]

            r = torch.sqrt((x * x + y * y + z * z).clamp_min(1e-24))

            inside = z > 0.0
            if near_m is not None and near_m > 0.0:
                inside = inside & (r >= float(near_m))
            inside = inside & (r <= float(far_val))

            phi = torch.atan2(y, x)
            u = (phi + math.pi) / (2.0 * math.pi)
            u = u - torch.floor(u)

            zr = (z / r.clamp_min(1e-12)).clamp(0.0, 1.0)
            theta = torch.acos(zr)
            v = theta / (0.5 * math.pi)

            iu = torch.floor(u * float(Wp)).clamp(0, Wp - 1).long()
            iv = torch.floor(v * float(Hp)).clamp(0, Hp - 1).long()
            pix = iv * Wp + iu  # (B,N)

            inf = torch.full((), float("inf"), device=pcd.device, dtype=pcd.dtype)
            r_masked = torch.where(inside, r, inf)

            range_flat = torch.full((B, K), float("inf"), device=pcd.device, dtype=pcd.dtype)
            range_flat.scatter_reduce_(dim=1, index=pix, src=r_masked, reduce="amin", include_self=True)
            range_img = range_flat.view(B, Hp, Wp)

            if DO_SUPPRESS:
                neg = -range_img.unsqueeze(1)  # (B,1,Hp,Wp)

                # width wrap via cat
                neg = torch.cat([neg[..., -s:], neg, neg[..., :s]], dim=-1)  # (B,1,Hp,Wp+2s)

                # height replicate via cat
                top = neg[:, :, 0:1, :].expand(-1, -1, s, -1)
                bot = neg[:, :, -1:, :].expand(-1, -1, s, -1)
                neg = torch.cat([top, neg, bot], dim=-2)  # (B,1,Hp+2s,Wp+2s)

                pooled = F.max_pool2d(neg, kernel_size=(k, k), stride=1)  # (B,1,Hp,Wp)
                neighbor_min = (-pooled).squeeze(1)

                eps = float(occlusion_eps_m) + float(occlusion_eps_rel) * neighbor_min.clamp_min(0.0)
                suppress = torch.isfinite(range_img) & torch.isfinite(neighbor_min) & (range_img > neighbor_min + eps)
                range_img = torch.where(suppress, inf, range_img)
                range_flat = range_img.view(B, K)

            valid_flat = torch.isfinite(range_flat)

            # backproject with cached directions
            pts_l = dir_flat.unsqueeze(0) * range_flat.unsqueeze(-1)  # (B,K,3)
            pts_w = torch.matmul(R, pts_l.transpose(1, 2)).transpose(1, 2) + o[:, None, :]

            nan = torch.full((), float("nan"), device=pcd.device, dtype=pcd.dtype)
            pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w, nan)

            # jitter inside compiled graph
            noise = torch.randn_like(pts_w) * jitter_std
            pts_w = torch.where(valid_flat.unsqueeze(-1), pts_w + noise, pts_w)

            return pts_w, valid_flat

        compiled = torch.compile(_compiled_fn, mode=compile_mode, dynamic=False)
        _compiled_cache[key] = compiled
        return compiled

    def wrapper(pcd: torch.Tensor, lidar_pose: torch.Tensor, jitter_std_m: float):
        fn = _get_or_build(pcd.device, pcd.dtype)
        jitter_std = torch.tensor(float(jitter_std_m), device=pcd.device, dtype=pcd.dtype)
        return fn(pcd, lidar_pose, jitter_std)

    return wrapper


# ------------------------------ user-facing simulate ------------------------------
@torch.no_grad()
def simulate_lidar_render_from_pose(
    pcd: torch.Tensor,
    lidar_pose: torch.Tensor,  # (B,7): [x,y,z,qx,qy,qz,qw], world_T_lidar, +Z hemisphere
    num_points: int = 10000,
    num_azimuth: int = 512,
    num_polar: int = 512,
    near_m: float = 0.1,
    far_m: Optional[float] = 30.0,
    suppress_bins: int = 2,
    occlusion_eps_m: float = 0.02,
    occlusion_eps_rel: float = 0.01,
    jitter_std_m: float = 0.0,
    shuffle: bool = True,
    use_compile: bool = True,
    compile_mode: str = "max-autotune",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
      lidar_pcd: (B, num_points, 3) valid-first, NaN padded
      logs: dict

    NOTE: Best with fixed shapes B and N, and fixed params above when use_compile=True.
    """
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
    assert lidar_pose.ndim == 2 and lidar_pose.shape[-1] == 7

    B = pcd.shape[0]
    device = pcd.device

    Hp = int(num_polar)
    Wp = int(num_azimuth)
    K = Hp * Wp

    if use_compile:
        renderer = get_compiled_lidar_renderer_fixed_shapes(
            num_azimuth=Wp,
            num_polar=Hp,
            near_m=float(near_m),
            far_m=None if far_m is None else float(far_m),
            suppress_bins=int(suppress_bins),
            occlusion_eps_m=float(occlusion_eps_m),
            occlusion_eps_rel=float(occlusion_eps_rel),
            compile_mode=compile_mode,
        )
        pts_w_flat, valid_flat = renderer(pcd, lidar_pose, jitter_std_m)  # (B,K,3), (B,K)
    else:
        pts_w_flat, valid_flat = render_lidar_bins_to_world_from_pose_fast(
            pcd=pcd,
            lidar_pose=lidar_pose,
            num_azimuth=Wp,
            num_polar=Hp,
            near_m=float(near_m),
            far_m=None if far_m is None else float(far_m),
            suppress_bins=int(suppress_bins),
            occlusion_eps_m=float(occlusion_eps_m),
            occlusion_eps_rel=float(occlusion_eps_rel),
            jitter_std_m=float(jitter_std_m),
        )

    # Shuffle bins (so truncation gives random subset)
    if shuffle:
        pts_w_flat = shuffle_pcd(pts_w_flat)

    # Pack to fixed num_points: move NaNs to end then take prefix
    nan_mask = torch.isnan(pts_w_flat).any(dim=-1)  # (B,K)
    sort_idx = torch.argsort(nan_mask.int(), dim=-1)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, K)
    sorted_out = pts_w_flat[batch_idx, sort_idx]    # (B,K,3)
    lidar_pcd = sorted_out[:, :num_points]          # (B,num_points,3)

    # Logs
    # valid_flat corresponds to bins with finite range after suppression
    num_valid_per_batch = valid_flat.sum(dim=-1)  # (B,)
    logs: Dict[str, float] = {
        "sim_lidar_render/avg_num_valid_points": float(num_valid_per_batch.float().mean().item()),
        "sim_lidar_render/min_num_valid_points": float(num_valid_per_batch.min().item()),
        "sim_lidar_render/num_rays": float(K),
        "sim_lidar_render/suppress_bins": float(suppress_bins),
    }

    return lidar_pcd, logs
