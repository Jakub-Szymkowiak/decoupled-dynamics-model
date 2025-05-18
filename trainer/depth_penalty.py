import torch


def dynamic_depth_alignment_penalty(model, deltas, viewpoint):
    if deltas is None or viewpoint.uid != 1:
        return 0.0

    xyz = model.dynamic.get_xyz + deltas.d_xyz
    P = torch.tensor(viewpoint.pose.inverse.homogeneous, device=xyz.device, dtype=xyz.dtype)
    K = torch.tensor(viewpoint.intrinsics.K, device=xyz.device, dtype=xyz.dtype)

    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    xyz_cam = (P @ xyz_h.T).T[:, :3]
    depth = xyz_cam[:, 2]

    uv_h = (K @ xyz_cam.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    u_px, v_px = uv[:, 0].round().long(), uv[:, 1].round().long()

    H, W = viewpoint.dynamic_depth.shape
    in_bounds = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)

    if not in_bounds.any():
        return torch.tensor(0.0, device=xyz.device)

    u_px, v_px, depth = u_px[in_bounds], v_px[in_bounds], depth[in_bounds]
    idx_flat = v_px * W + u_px

    dmask = viewpoint.dmask.view(-1).to(xyz.device)
    gt_depth = viewpoint.dynamic_depth.view(-1).to(xyz.device, dtype=xyz.dtype)

    valid = dmask[idx_flat] > 0
    if not valid.any():
        return torch.tensor(0.0, device=xyz.device)

    return torch.abs(gt_depth[idx_flat[valid]] - depth[valid]).mean()


def static_depth_alignment_penalty(model, viewpoint):
    xyz = model.static.get_xyz
    P = torch.tensor(viewpoint.pose.inverse.homogeneous, device=xyz.device, dtype=xyz.dtype)
    K = torch.tensor(viewpoint.intrinsics.K, device=xyz.device, dtype=xyz.dtype)

    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    xyz_cam = (P @ xyz_h.T).T[:, :3]
    depth = xyz_cam[:, 2]

    uv_h = (K @ xyz_cam.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    u_px, v_px = uv[:, 0].round().long(), uv[:, 1].round().long()

    H, W = viewpoint.dynamic_depth.shape
    in_bounds = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)

    if not in_bounds.any():
        return torch.tensor(0.0, device=xyz.device)

    u_px, v_px, depth = u_px[in_bounds], v_px[in_bounds], depth[in_bounds]
    idx_flat = v_px * W + u_px

    gt_depth = viewpoint.static_depth.view(-1).to(xyz.device, dtype=xyz.dtype)

    return torch.abs(gt_depth[idx_flat] - depth).mean()