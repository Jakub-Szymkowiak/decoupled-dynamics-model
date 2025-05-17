import torch


class DeformDeltas:
    def __init__(self, d_xyz=None, d_rotation=None, d_scaling=None):
        _default = lambda d: d if d is not None else 0.0

        self.d_xyz = _default(d_xyz)
        self.d_rotation = _default(d_rotation)
        self.d_scaling = _default(d_scaling)

    @staticmethod
    def prepare_composed(Ns: int, dynamic: "DeformDeltas") -> "DeformDeltas":
        assert isinstance(dynamic.d_xyz, torch.Tensor), "Expected dynamic deltas to be tensors."

        def pad_with_static_zeros(d: torch.Tensor):
            shape = (Ns, d.shape[-1])
            static_zeros = torch.zeros(shape, device=d.device)
            return torch.cat([static_zeros, d], dim=0)

        return DeformDeltas(d_xyz=pad_with_static_zeros(dynamic.d_xyz),
                            d_rotation=pad_with_static_zeros(dynamic.d_rotation),
                            d_scaling=pad_with_static_zeros(dynamic.d_scaling))

    @classmethod
    def get_zero_deform(cls):
        return cls(*(0.0, 0.0, 0.0))

    def as_tuple(self):
        return (self.d_xyz, self.d_rotation, self.d_scaling)

    def __repr__(self):
        return f"DeformDeltas(x={type(self.d_xyz)}, rot={type(self.d_rotation)}, scale={type(self.d_scaling)})"
