import torch.nn.functional as F

from utils.loss_utils import l1_loss


def sample_flow(flow_map, coords):
    N = coords.shape[0]
    H, W = flow_map.shape[2:]

    coords_norm = coords.clone()
    coords_norm[:, 0] = (coords[:, 0] / (W - 1)) * 2 - 1
    coords_norm[:, 1] = (coords[:, 1] / (H - 1)) * 2 - 1

    grid = coords_norm.view(1, -1, 1, 2)

    sampled = F.grid_sample(flow_map, grid, mode="bilinear", align_corners=True)
    flow = sampled.squeeze(3).squeeze(0).T

    return flow


def optical_flow_consistency_loss(model, render, viewpoint, next_viewpoint, state, directive, utils):
    if next_viewpoint is None or viewpoint.flow is None:
        return 0.0

    if not directive.train_dynamic or not directive.train_deform:
        return 0.0

    screenspace_points = render["dynamic"]["viewspace_points"][:, :2].detach()
    
    next_deltas, _ = model.infer_deltas(fid=next_viewpoint.fid,
                                        iteration=state.iteration,
                                        time_interval=state.time_interval,
                                        smooth_term=utils.smooth_term)

    next_render = utils.run_renderer("dynamic", next_deltas, next_viewpoint)
    next_screenspace_points = next_render["viewspace_points"][:, :2].detach()

    flow_pred = next_screenspace_points - screenspace_points

    flow_map = viewpoint.flow.permute(2, 0, 1).unsqueeze(0).float()
    flow_sampled = sample_flow(flow_map, screenspace_points)

    return l1_loss(flow_pred, flow_sampled)
