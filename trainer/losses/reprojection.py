import torch

from utils.delta_utils import DeformDeltas
from utils.loss_utils import l1_loss


def reprojection_loss(model, deltas, next_viewpoint, fid_input, state, utils):
    if next_viewpoint is None or fid_input is None:
        return 0.0
    
    d_xyz = deltas["dynamic"].d_xyz

    grad_dxyz = torch.autograd.grad(
        outputs=d_xyz,
        inputs=fid_input,
        grad_outputs=torch.ones_like(d_xyz),
        retain_graph=True,
        create_graph=True
    )[0]

    next_deltas, _ = model.infer_deltas(fid=next_viewpoint.fid,
                                        iteration=state.iteration,
                                        time_interval=state.time_interval,
                                        smooth_term=utils.smooth_term)

    d_xyz_pred = deltas["dynamic"].d_xyz + state.time_interval * grad_dxyz

    next_deltas["dynamic"].d_xyz = d_xyz_pred

    rendering_result = utils.run_renderer("dynamic", next_deltas, next_viewpoint)
    rendered_image = torch.clamp(rendering_result["render"], 0.0, 1.0)

    gt_image = torch.clamp(next_viewpoint.dynamic_image, 0.0, 1.0)

    return l1_loss(rendered_image, gt_image)

