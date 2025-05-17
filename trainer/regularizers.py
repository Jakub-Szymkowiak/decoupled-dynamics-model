from typing import Callable, Dict, List

import torch

from utils.delta_utils import DeformDeltas


class Regularizer:
    def __init__(self, fn: Callable, weight: float, target: str, name: str=None):
        self.fn = fn
        self.weight = weight
        self.target = target
        self.name = name or fn.__name__

    def __call__(self, model, deltas=None):
        return self.fn(model, deltas) * self.weight

class RegularizerRegistry:
    def __init__(self, regularizers: List[Regularizer], device="cuda"):
        self._regs = regularizers
        self._device = device

    def compute(self, model, deltas: DeformDeltas, target: str):
        loss_terms = {}
        for reg in self._regs:
            if reg.target == target:
                value = reg(model, deltas)
                loss_terms[reg.name] = value
        total = sum(loss_terms.values()) if loss_terms else torch.tensor(0.0, device=self._device)
        return total, loss_terms
