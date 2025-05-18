import torch

from trainer.depth_penalty import dynamic_depth_alignment_penalty, static_depth_alignment_penalty
from trainer.regularizers import Regularizer



def scaling_anisotropy_penalty(model):
    return torch.std(model.get_scaling, dim=1).mean()

def d_scaling_anisotropy_penalty(deltas):
    if deltas is not None:
        return torch.std(deltas.d_scaling, dim=1).mean()
    return 0.0

# TODO - rotation and d_rotation penalty?

DEFAULT_REGULARIZERS = [
    Regularizer(scaling_anisotropy_penalty, weight=0.1),
    Regularizer(d_scaling_anisotropy_penalty, weight=0.1),
    #Regularizer(dynamic_depth_alignment_penalty, weight=1.0),
    Regularizer(static_depth_alignment_penalty, weight=1.0)
]