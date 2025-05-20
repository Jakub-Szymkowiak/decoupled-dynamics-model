import torch

from gaussian_renderer import render_decoupled


def get_rendering_func(model, pipe, bg_color, is_6dof=False):
    def run_renderer(mode, deltas, viewpoint, override_xyz=None):
        return render_decoupled(model=model.get_models()[mode], 
                                viewpoint=viewpoint, 
                                pipe=pipe, 
                                bg_color=bg_color, 
                                deltas=deltas[mode], 
                                override_xyz=override_xyz)

    return run_renderer

def mask_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert image.shape[0] == 3 and image.ndim == 3
    assert mask.shape[0] == 1 and mask.shape[1:] == image.shape[1:]
    return image * (mask > 0)


