"""
simulate_depth_cam.py (FAST + torch.compile for fixed shapes)

CODEX LIDAR MERGE: copied in from old_urdf for the lidar/depth DAgger merge.

Includes speedups:
1) Z-buffer: argsort trick -> scatter_reduce_(amin)  (O(N) vs O(N log N))
2) Cache intrinsics + pixel grids (avoid re-allocations & trig every call)
3) torch.compile path tuned for FIXED B and FIXED N (dynamic=False)
   - Fixes your SymInt max_pool2d kernel_size crash by making kernel_size + pad
     Python-int tuples captured in the closure.

Assumptions for best performance:
- Batch size B does not change
- Number of points N does not change
- cam_spec_dict (H/W/FOVs), inflate_px, clip_mode, jitter_mode do not change
"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


INTEL_455 = {
    "H": 848 // 4,
    "W": 480 // 4,
    "fov_x_deg": 87.0,
    "fov_y_deg": 58.0,
    "near_m": 0.4,
    "far_m": 6.0,
}

INTEL_435 = {
    "H": 848 // 4,
    "W": 480 // 4,
    "fov_x_deg": 85.2,
    "fov_y_deg": 58.0,
    "near_m": 0.3,
    "far_m": 3.0,
}

# ------------------------------ caches ------------------------------
_uv_cache: dict = {}        # (H,W,device,dtype) -> (u_base,v_base) (1,H,W)
_intr_cache: dict = {}      # (H,W,fovx,fovy,device,dtype) -> (fx,fy,cx,cy) scalar tensors
_compiled_cache: dict = {}  # (spec+params+device+dtype) -> compiled callable


def _get_uv_base(H: int, W: int, device, dtype):
    key = (H, W, device, dtype)
    if key not in _uv_cache:
        u = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(1, H, W)
        v = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(1, H, W)
        _uv_cache[key] = (u, v)
    return _uv_cache[key]


def _get_intrinsics(H: int, W: int, fov_x_deg: float, fov_y_deg: float, device, dtype):
    key = (H, W, float(fov_x_deg), float(fov_y_deg), device, dtype)
    if key not in _intr_cache:
        fx = torch.tensor(0.5 * W / math.tan(0.5 * math.radians(fov_x_deg)), device=device, dtype=dtype)
        fy = torch.tensor(0.5 * H / math.tan(0.5 * math.radians(fov_y_deg)), device=device, dtype=dtype)
        cx = torch.tensor((W - 1) * 0.5, device=device, dtype=dtype)
        cy = torch.tensor((H - 1) * 0.5, device=device, dtype=dtype)
        _intr_cache[key] = (fx, fy, cx, cy)
    return _intr_cache[key]


# ------------------------------ PCD utils ------------------------------
def shuffle_pcd(pcd: torch.Tensor) -> torch.Tensor:
    """
    Randomize ordering of points in a point cloud.
    Args:
        pcd: (B, N, 3) tensor
    Returns:
        shuffled: (B, N, 3)
    """
    B, N, _ = pcd.shape
    device = pcd.device
    idx = torch.argsort(torch.rand(B, N, device=device), dim=-1)  # (B,N)
    batch_idx = torch.arange(B, device=device)[:, None].expand(B, N)
    return pcd[batch_idx, idx]


# ------------------------------ from pose utils ------------------------------
def _camera_basis_from_pose_x_forward(
    camera_pose: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    camera_pose: (B,7) = [x,y,z,qx,qy,qz,qw], quat (xyzw), mapping camera->world.
    Assumes camera forward axis is +X in camera frame.

    Renderer basis:
        v_hat = R * e_x  (depth axis)
        u_hat = R * e_y  (horizontal axis)
        w_hat = R * e_z  (vertical axis)
    """
    assert camera_pose.shape[-1] == 7, "camera_pose must have last dim = 7 (pos + quat_xyzw)"

    q = camera_pose[..., 3:7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
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

    v_hat = torch.stack([R00, R10, R20], dim=-1)  # +X_cam
    u_hat = torch.stack([R01, R11, R21], dim=-1)  # +Y_cam
    w_hat = torch.stack([R02, R12, R22], dim=-1)  # +Z_cam

    v_hat = v_hat / v_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    u_hat = u_hat / u_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w_hat = w_hat / w_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return u_hat, w_hat, v_hat


# ------------------------------ FAST rasterize (scatter-reduce z-buffer) ------------------------------
@torch.no_grad()
def rasterize_depth_zbuffer_from_pose(
    pcd: torch.Tensor,           # (B,N,3)
    camera_pose: torch.Tensor,   # (B,7)
    H: int,
    W: int,
    fov_y_deg: float,
    fov_x_deg: float,
    inflate_px: int = 0,
    clip_mode: str = "post",     # "pre" or "post"
    near_m: float = 0.0,
    far_m: Optional[float] = None,
):
    """
    Returns:
      depth:  (B,H,W) float, +inf where no hit / clipped
      inside: (B,N) bool
      iu,iv:  (B,N) long pixel indices
      z:      (B,N) depths along +X_cam
      basis:  (u_hat,w_hat,v_hat) each (B,3)
      intr:   (fx,fy,cx,cy) scalar tensors
    """
    assert pcd.ndim == 3 and pcd.shape[-1] == 3
    B, N, _ = pcd.shape
    device, dtype = pcd.device, pcd.dtype

    fx, fy, cx, cy = _get_intrinsics(H, W, fov_x_deg, fov_y_deg, device, dtype)
    intr = (fx, fy, cx, cy)

    cam_pos = camera_pose[:, 0:3]
    u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)

    rel = pcd - cam_pos[:, None, :]            # (B,N,3)
    x = (rel * u_hat[:, None, :]).sum(-1)      # (B,N)
    y = (rel * w_hat[:, None, :]).sum(-1)      # (B,N)
    z = (rel * v_hat[:, None, :]).sum(-1)      # (B,N)

    invz = 1.0 / z.clamp_min(1e-12)
    u_pix = fx * (x * invz) + cx
    v_pix = fy * (y * invz) + cy

    in_front = z > 0
    inside = (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H) & in_front

    far_val = float("inf") if far_m is None else float(far_m)

    if clip_mode == "pre":
        inside = inside & (z >= near_m) & (z <= far_val)

    iu = u_pix.floor().clamp(0, W - 1).long()
    iv = v_pix.floor().clamp(0, H - 1).long()
    pix = iv * W + iu  # (B,N)

    if not hasattr(torch.Tensor, "scatter_reduce_"):
        raise RuntimeError("Tensor.scatter_reduce_ not found. Need PyTorch >= 1.12 (PyTorch 2.1 is fine).")

    inf = torch.full((), float("inf"), device=device, dtype=dtype)
    z_masked = torch.where(inside, z, inf)  # (B,N)

    K = H * W
    min_depth = torch.full((B, K), float("inf"), device=device, dtype=dtype)
    min_depth.scatter_reduce_(dim=1, index=pix, src=z_masked, reduce="amin", include_self=True)
    depth = min_depth.view(B, H, W)

    # inflate occluders (optional; eager version)
    if inflate_px > 0:
        neg = -depth.unsqueeze(1)  # (B,1,H,W)
        neg = F.pad(
            neg,
            (inflate_px, inflate_px, inflate_px, inflate_px),
            mode="constant",
            value=float("-inf"),
        )
        k = 2 * inflate_px + 1
        pooled_neg = F.max_pool2d(neg, kernel_size=(k, k), stride=1)
        depth = (-pooled_neg).squeeze(1)

    # post clip
    if near_m > 0.0:
        depth = torch.where(depth >= near_m, depth, inf)
    depth = torch.where(depth <= far_val, depth, inf)

    return depth, inside, iu, iv, z, (u_hat, w_hat, v_hat), intr


# ------------------------------ FAST backproject (cached grids) ------------------------------
@torch.no_grad()
def backproject_depth_to_world_from_pose(
    depth: torch.Tensor,        # (B,H,W), +inf where invalid
    camera_pose: torch.Tensor,  # (B,7)
    intrinsics: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],  # fx,fy,cx,cy scalar tensors
):
    """
    Returns:
      pcd_world: (B,H,W,3) world coords; NaN where invalid
      valid:     (B,H,W) bool
    """
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype
    fx, fy, cx, cy = intrinsics

    cam_pos = camera_pose[:, 0:3]
    u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)

    u_base, v_base = _get_uv_base(H, W, device, dtype)
    u = u_base.expand(B, H, W)
    v = v_base.expand(B, H, W)

    valid = torch.isfinite(depth)
    z = depth

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    nan = torch.full((), float("nan"), device=device, dtype=dtype)
    x = torch.where(valid, x, nan)
    y = torch.where(valid, y, nan)
    z = torch.where(valid, z, nan)

    uB = u_hat.view(B, 1, 1, 3)
    wB = w_hat.view(B, 1, 1, 3)
    vB = v_hat.view(B, 1, 1, 3)
    oB = cam_pos.view(B, 1, 1, 3)

    pcd_world = oB + x[..., None] * uB + y[..., None] * wB + z[..., None] * vB
    return pcd_world, valid


# ------------------------------ High-level helper (eager fast) ------------------------------
@torch.no_grad()
def render_points_to_world_grid_from_pose(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,      # (B,7)
    cam_spec_dict: Dict = INTEL_455,
    inflate_px: int = 0,
    clip_mode: str = "post",
    jitter_std_m: float = 0.0,
    jitter_mode: str = "xyz",
):
    H = int(cam_spec_dict["H"])
    W = int(cam_spec_dict["W"])
    fov_x_deg = float(cam_spec_dict["fov_x_deg"])
    fov_y_deg = float(cam_spec_dict["fov_y_deg"])
    near_m = float(cam_spec_dict["near_m"])
    far_m = cam_spec_dict["far_m"]
    far_m = None if far_m is None else float(far_m)

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

    if jitter_std_m > 0.0:
        if jitter_mode.lower() == "xyz":
            noise = torch.randn_like(pcd_world) * float(jitter_std_m)
            pcd_world = torch.where(valid[..., None], pcd_world + noise, pcd_world)
        elif jitter_mode.lower() == "tangent":
            u_hat, w_hat, _ = _camera_basis_from_pose_x_forward(camera_pose)
            B, H_, W_ = depth.shape
            uB = u_hat.view(B, 1, 1, 3)
            wB = w_hat.view(B, 1, 1, 3)
            eps_u = torch.randn(B, H_, W_, 1, device=pcd.device, dtype=pcd.dtype) * float(jitter_std_m)
            eps_w = torch.randn(B, H_, W_, 1, device=pcd.device, dtype=pcd.dtype) * float(jitter_std_m)
            jitter = eps_u * uB + eps_w * wB
            pcd_world = torch.where(valid[..., None], pcd_world + jitter, pcd_world)
        else:
            raise ValueError(f"Unknown jitter_mode '{jitter_mode}'. Use 'tangent' or 'xyz'.")

    return depth, pcd_world, valid


# ------------------------------ torch.compile path (FIXED SHAPES) ------------------------------
def get_compiled_renderer_fixed_shapes(
    cam_spec_dict: Dict,
    inflate_px: int = 0,
    clip_mode: str = "post",
    jitter_mode: str = "xyz",
    compile_mode: str = "max-autotune",
):
    """
    Returns compiled callable:
        fn(pcd: (B,N,3), camera_pose: (B,7), jitter_std: scalar tensor) -> (depth, pcd_world, valid)

    IMPORTANT: This assumes FIXED shapes (B and N do not change).
    Uses torch.compile(..., dynamic=False) and constant kernel_size tuples to avoid SymInt kernel_size crash.
    """
    if jitter_mode.lower() != "xyz":
        raise ValueError("Compiled renderer supports jitter_mode='xyz' only (compile-friendly).")

    H = int(cam_spec_dict["H"])
    W = int(cam_spec_dict["W"])
    fov_x_deg = float(cam_spec_dict["fov_x_deg"])
    fov_y_deg = float(cam_spec_dict["fov_y_deg"])
    near_m = float(cam_spec_dict["near_m"])
    far_m = cam_spec_dict["far_m"]
    far_val = float("inf") if far_m is None else float(far_m)
    clip_pre = (clip_mode == "pre")

    # Compile is device/dtype-specific; cache by params+device+dtype
    def _get_or_build(device, dtype):
        key = (H, W, fov_x_deg, fov_y_deg, near_m, far_val, int(inflate_px), clip_mode, jitter_mode, compile_mode, device, dtype)
        if key in _compiled_cache:
            return _compiled_cache[key]

        fx, fy, cx, cy = _get_intrinsics(H, W, fov_x_deg, fov_y_deg, device, dtype)
        u_base, v_base = _get_uv_base(H, W, device, dtype)

        # --- IMPORTANT: make PAD/KERNEL python-int tuples in closure (not SymInt) ---
        PAD = (int(inflate_px), int(inflate_px), int(inflate_px), int(inflate_px))
        k = int(2 * inflate_px + 1)
        KERNEL = (k, k)

        @torch.no_grad()
        def _compiled_fn(pcd: torch.Tensor, camera_pose: torch.Tensor, jitter_std: torch.Tensor):
            cam_pos = camera_pose[:, 0:3]
            u_hat, w_hat, v_hat = _camera_basis_from_pose_x_forward(camera_pose)

            rel = pcd - cam_pos[:, None, :]
            x = (rel * u_hat[:, None, :]).sum(-1)
            y = (rel * w_hat[:, None, :]).sum(-1)
            z = (rel * v_hat[:, None, :]).sum(-1)

            invz = 1.0 / z.clamp_min(1e-12)
            u_pix = fx * (x * invz) + cx
            v_pix = fy * (y * invz) + cy

            in_front = z > 0
            inside = (u_pix >= 0) & (u_pix < W) & (v_pix >= 0) & (v_pix < H) & in_front
            if clip_pre:
                inside = inside & (z >= near_m) & (z <= far_val)

            iu = u_pix.floor().clamp(0, W - 1).long()
            iv = v_pix.floor().clamp(0, H - 1).long()
            pix = iv * W + iu  # (B,N)

            inf = torch.full((), float("inf"), device=pcd.device, dtype=pcd.dtype)
            z_masked = torch.where(inside, z, inf)

            K = H * W
            B = pcd.shape[0]
            min_depth = torch.full((B, K), float("inf"), device=pcd.device, dtype=pcd.dtype)
            min_depth.scatter_reduce_(dim=1, index=pix, src=z_masked, reduce="amin", include_self=True)
            depth = min_depth.view(B, H, W)

            # inflate inside compiled graph (kernel_size tuple is python-int, safe)
            if inflate_px > 0:
                neg = -depth.unsqueeze(1)
                neg = F.pad(neg, PAD, mode="constant", value=float("-inf"))
                pooled_neg = F.max_pool2d(neg, kernel_size=KERNEL, stride=1)
                depth = (-pooled_neg).squeeze(1)

            if near_m > 0.0:
                depth = torch.where(depth >= near_m, depth, inf)
            depth = torch.where(depth <= far_val, depth, inf)

            # backproject
            u = u_base.expand(B, H, W)
            v = v_base.expand(B, H, W)

            valid = torch.isfinite(depth)
            zz = depth
            xx = (u - cx) / fx * zz
            yy = (v - cy) / fy * zz

            nan = torch.full((), float("nan"), device=pcd.device, dtype=pcd.dtype)
            xx = torch.where(valid, xx, nan)
            yy = torch.where(valid, yy, nan)
            zz = torch.where(valid, zz, nan)

            uB = u_hat.view(B, 1, 1, 3)
            wB = w_hat.view(B, 1, 1, 3)
            vB = v_hat.view(B, 1, 1, 3)
            oB = cam_pos.view(B, 1, 1, 3)

            pcd_world = oB + xx[..., None] * uB + yy[..., None] * wB + zz[..., None] * vB

            # jitter (xyz): keep inside graph; jitter_std can be 0
            noise = torch.randn_like(pcd_world) * jitter_std
            pcd_world = torch.where(valid[..., None], pcd_world + noise, pcd_world)

            return depth, pcd_world, valid

        compiled = torch.compile(_compiled_fn, mode=compile_mode, dynamic=False)
        _compiled_cache[key] = compiled
        return compiled

    def wrapper(pcd: torch.Tensor, camera_pose: torch.Tensor, jitter_std_m: float):
        compiled = _get_or_build(pcd.device, pcd.dtype)
        jitter_std = torch.tensor(float(jitter_std_m), device=pcd.device, dtype=pcd.dtype)
        return compiled(pcd, camera_pose, jitter_std)

    return wrapper


# ------------------------------ User-facing simulate function ------------------------------
@torch.no_grad()
def simulate_depth_cam_render_from_pose(
    pcd: torch.Tensor,
    camera_pose: torch.Tensor,    # (B,7): [x,y,z,qx,qy,qz,qw]
    num_points: int,
    inflate_px: int = 2,
    jitter_std_m: float = 0.004,
    cam_spec_dict: Dict = INTEL_435,
    clip_mode: str = "post",
    jitter_mode: str = "xyz",
    use_compile: bool = True,
    compile_mode: str = "max-autotune",
):
    """
    Returns:
      pcd_nan_padding: (B, num_points, 3) points (valid first, NaN-padded)
      logs: dict
    """
    batch_size = pcd.shape[0]
    device = pcd.device

    if use_compile:
        renderer = get_compiled_renderer_fixed_shapes(
            cam_spec_dict=cam_spec_dict,
            inflate_px=inflate_px,
            clip_mode=clip_mode,
            jitter_mode=jitter_mode,
            compile_mode=compile_mode,
        )
        depth, pcd_world, valid = renderer(pcd, camera_pose, jitter_std_m)
    else:
        depth, pcd_world, valid = render_points_to_world_grid_from_pose(
            pcd,
            camera_pose,
            cam_spec_dict=cam_spec_dict,
            inflate_px=inflate_px,
            clip_mode=clip_mode,
            jitter_std_m=jitter_std_m,
            jitter_mode=jitter_mode,
        )

    rendered_pcd = pcd_world.view(batch_size, -1, 3)  # (B, H*W, 3)
    num_total_points = rendered_pcd.shape[1]

    rendered_pcd = shuffle_pcd(rendered_pcd)

    nan_mask = torch.isnan(rendered_pcd).any(dim=-1)  # (B, H*W)
    sort_key = nan_mask.int()                         # 0 valid, 1 invalid
    sort_idx = torch.argsort(sort_key, dim=-1)
    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(batch_size, num_total_points)
    sorted_pcds = rendered_pcd[batch_idx, sort_idx]  # (B, H*W, 3)

    avg_num_valid_points = num_total_points - nan_mask.sum().float() / float(batch_size)
    min_num_valid_points = (num_total_points - nan_mask.sum(dim=-1)).min().float()

    logs = {
        "sim_depth_cam_render/avg_num_valid_points": float(avg_num_valid_points.item()),
        "sim_depth_cam_render/min_num_valid_points": float(min_num_valid_points.item()),
    }

    pcd_nan_padding = sorted_pcds[:, :num_points]
    return pcd_nan_padding, logs
