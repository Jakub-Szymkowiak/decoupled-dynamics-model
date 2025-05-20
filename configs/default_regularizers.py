import torch

from trainer.losses.depth_alignment import dynamic_depth_alignment_penalty, static_depth_alignment_penalty
from trainer.losses.reprojection import reprojection_loss
from trainer.regularizers import Regularizer



def scaling_anisotropy_penalty(model):
    return torch.std(model.get_scaling, dim=1).mean()

def d_scaling_anisotropy_penalty(deltas, directive):
    if directive.train_deform:
        return torch.std(deltas["dynamic"].d_scaling, dim=1).mean()
    return 0.0

# TODO - rotation and d_rotation penalty?

DEFAULT_REGULARIZERS = [
    Regularizer(scaling_anisotropy_penalty, weight=0.25),
    Regularizer(d_scaling_anisotropy_penalty, weight=0.25),
    #Regularizer(dynamic_depth_alignment_penalty, weight=0.001),
    #Regularizer(static_depth_alignment_penalty, weight=0.001)
    Regularizer(reprojection_loss, weight=0.01)
]