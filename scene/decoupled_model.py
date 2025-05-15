import json

from dataclasses import asdict, dataclass
from typing import Dict, Literal, Optional, Union

from pathlib import Path

import torch

from scene.deform_model import DeformModel
from scene.gaussian_model import BasicPointCloud, GaussianModel


tensor_or_float = Union[torch.Tensor, float]

@dataclass
class Deltas:
    # deltas can be either tensors
    # or floats (static_deltas are 0.0 floats)
    d_xyz: tensor_or_float
    d_rotation: tensor_or_float
    d_scaling: tensor_or_float

    @classmethod
    def from_tuple(cls, deltas_tuple: tuple):
        assert len(deltas_tuple) == 3, "Expected a 3-tuple"
        return cls(*deltas_tuple)

    @classmethod
    def zeros(cls, Ns: int, device="cuda"):
        return cls(
            d_xyz=torch.zeros(Ns, 3, device=device),
            d_rotation=torch.zeros(Ns, 4, device=device),
            d_scaling=torch.zeros(Ns, 3, device=device)
        )

    @classmethod
    def cat(cls, a: "Deltas", b: "Deltas") -> "Deltas":
        return cls(
            d_xyz=torch.cat([a.d_xyz, b.d_xyz], dim=0),
            d_rotation=torch.cat([a.d_rotation, b.d_rotation], dim=0),
            d_scaling=torch.cat([a.d_scaling, b.d_scaling], dim=0),
        )

    def to_tuple(self):
        return (self.d_xyz, self.d_rotation, self.d_scaling)

    def __repr__(self):
        return f"Deltas(x={self.d_xyz.shape}, rot={self.d_rotation.shape}, scale={self.d_scaling.shape})"

    

class DecoupledModel:
    _static_deltas = Deltas.from_tuple((0.0, 0.0, 0.0))

    def __init__(
            self, 
            sh_degree: int,
            is_blender: bool=False,
            is_6dof: bool=False
        ):

        self.static = GaussianModel(sh_degree)
        self.dynamic = GaussianModel(sh_degree)
        self.deform = DeformModel(is_blender, is_6dof)

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        
        # Set in self.create_from_pcd()
        self._Ns = None
        self._Nd = None

        self._ast_noise = None
        

    def _concat_attr(self, attr_name: str) -> torch.Tensor:
        static_attr = getattr(self.static, attr_name)
        dynamic_attr = getattr(self.dynamic, attr_name)
        return torch.cat([static_attr, dynamic_attr], dim=0)

    def get_models(self):
        return {"static": self.static, "dynamic": self.dynamic, "composed": self}

    def infer_deltas(
            self, 
            fid: torch.Tensor, 
            iteration: Optional[int] = None, 
            time_interval: Optional[float] = None, 
            smooth_term: Optional[torch.tensor] = None,
            noise: bool=True
        ) -> Dict[Literal["static", "dynamic", "composed"], Deltas]:

        # TODO - decide whether to use buffering

        # Noise computation
        if noise:
            assert_msg = "Must provide parameters for noise = True; or use noise = False"
            assert iteration is not None and time_interval is not None and smooth_term is not None, assert_msg
            self._ast_noise.normal_()
            self._ast_noise *= time_interval * smooth_term(iteration)
        else:
            self._ast_noise.zero_()

        fid_input = fid.view(1, 1).expand(self._Nd, 1)
        time_input = fid_input + self._ast_noise

        xyz = self.dynamic.get_xyz.detach().to(torch.float32)
        
        raw_deltas = self.deform.step(xyz, time_input)

        dynamic_deltas = Deltas.from_tuple(raw_deltas)
        static_zeros = Deltas.zeros(self._Ns, device=dynamic_deltas.d_xyz.device)
        composed = Deltas.cat(static_zeros, dynamic_deltas)

        return {"static": self._static_deltas, "dynamic": dynamic_deltas, "composed": composed}

    def get_zero_deltas(self) -> Deltas:
        return {
            "static": Deltas.from_tuple((0.0, 0.0, 0.0)),
            "dynamic": Deltas.from_tuple((0.0, 0.0, 0.0)),
            "composed": Deltas.from_tuple((0.0, 0.0, 0.0))
        }

    @property
    def get_xyz(self):
        return self._concat_attr("get_xyz")

    @property
    def get_scaling(self):
        return self._concat_attr("get_scaling")

    @property
    def get_rotation(self):
        return self._concat_attr("get_rotation")

    @property
    def get_opacity(self):
        return self._concat_attr("get_opacity")

    @property
    def get_features(self):
        return self._concat_attr("get_features")

    def create_from_pcd(self, static_ptc: BasicPointCloud, dynamic_ptc: BasicPointCloud, cameras_extent: float):

        self.static.create_from_pcd(static_ptc, cameras_extent)
        self.dynamic.create_from_pcd(dynamic_ptc, cameras_extent)
        
        self._Ns, self._Nd = self.static.get_xyz.size(0), self.dynamic.get_xyz.size(0)

        print("Number of static background Gaussians at init: ", self._Ns)
        print("Number of dynamic foreground Gaussians at init: ", self._Nd)

        self._ast_noise = torch.zeros(self._Nd, 1, device="cuda")

    def training_setup(self, training_args):
        self.deform.train_setting(training_args)

        self.static.training_setup(training_args)
        self.dynamic.training_setup(training_args)

    def oneupSHdegree(self):
        self.static.oneupSHdegree()
        self.dynamic.oneupSHdegree()

        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def load_plys(self, directory: Path):
        self.static.load_ply(directory / "static.ply")
        self.dynamic.load_ply(directory / "dynamic.ply")

        self._Ns, self._Nd = self.static.get_xyz.size(0), self.dynamic.get_xyz.size(0)
        self._ast_noise = torch.zeros(self._Nd, 1, device="cuda")

    def save(self, iteration, model_path: Path):
        directory = Path(model_path) / "point_cloud" / f"_{iteration}" 

        self.deform.save_weights(model_path, iteration)
        self.static.save_ply(directory / "static.ply")
        self.dynamic.save_ply(directory / "dynamic.ply")
        