import torch
from trainer.regularizers import Regularizer


def scaling_anisotropy_penalty(model):
    return torch.std(model.get_scaling, dim=1).mean()

def d_scaling_anisotropy_penalty(model, deltas):
    return torch.std(deltas.d_scaling, dim=1).mean()


DEFAULT_REGULARIZERS = [
    #Regularizer(anisotropic_scale_penalty, weight=5.0, target="gaussians")
    Regularizer(d_scaling_anisotropy_penalty, weight=5.0, target="deform")
]