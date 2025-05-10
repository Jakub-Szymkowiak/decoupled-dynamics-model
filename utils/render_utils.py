import torch

from gaussian_renderer import render


def get_rendering_func(model, pipe, background, is_6dof):
    def run_renderer(mode, deltas, viewpoint):
        return render(viewpoint, model.get_models()[mode], pipe, background, 
                      deltas[mode].d_xyz, deltas[mode].d_rotation, deltas[mode].d_scaling, 
                      is_6dof)

    return run_renderer

def mask_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert image.shape[0] == 3 and image.ndim == 3
    assert mask.shape[0] == 1 and mask.shape[1:] == image.shape[1:]
    return image * (mask > 0)
