from inspect import signature
from typing import Callable, Dict, List

import torch

from utils.delta_utils import DeformDeltas


class Regularizer:
    def __init__(self, fn: Callable, weight: float, name: str = None):
        self.fn = fn
        self.weight = weight
        self.name = name or fn.__name__
        
        self.sig = signature(fn)

    def __call__(self, **kwargs):
        accepted_args = {k: v for k, v in kwargs.items() if k in self.sig.parameters}
        return self.fn(**accepted_args) * self.weight




class RegularizerRegistry:
    def __init__(self, regularizers: List[Regularizer], device="cuda"):
        self._regs = regularizers
        self._device = device

    def compute(self, **context):
        losses = {}
        for reg in self._regs:
            losses[reg.name] = reg(**context)

        total = sum(losses.values()) if losses else torch.tensor(0.0, device=self._device)
        return total, losses