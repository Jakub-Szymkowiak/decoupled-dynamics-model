import torch
from trainer.regularizers import Regularizer


def anisotropic_scale_penalty(model):
    return torch.std(model.get_scaling, dim=1).mean()


DEFAULT_REGULARIZERS = [
    #Regularizer(anisotropic_scale_penalty, weight=5.0)
]