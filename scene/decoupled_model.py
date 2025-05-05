from typing import Optional

import torch

from scene.deform_model import DeformModel
from scene.gaussian_model import BasicPointCloud, GaussianModel


class DecoupledModel:
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

    def _concat_attr(self, attr_name: str):
        static_attr = getattr(self.static, attr_name)
        dynamic_attr = getattr(self.dynamic, attr_name)
        return torch.cat([static_attr, dynamic_attr], dim=0)

    def infer_deltas(
            self, 
            fid: torch.Tensor, 
            iteration: Optional[int], 
            time_interval: Optional[float], 
            smooth_term: Optional[torch.tensor],
            noise: bool=True
        ):

        _get_N = lambda g: g.get_xyz.shape[0]
        Ns = _get_N(self.static)
        Nd = _get_N(self.dynamic)
        

        if noise == True:
            assert iteration, "Pass iteration"
            assert time_interval, "Pass time_interval"
            assert smooth_term, "Pass smooth_term"

            ast_noise = torch.randn(Nd, 1, device="cuda")
            ast_noise *= time_interval * smooth_term(iteration)
        else:
            ast_noise = 0.0

        static_deltas = (0.0, 0.0, 0.0)

        time_input = fid.unsqueeze(0).repeat(Nd, 1)

        xyz = self.dynamic.get_xyz.detach()
        d_xyz, d_scaling, d_rotation = self.deform.step(xyz, time_input + ast_noise)

        static_zeros = lambda t: torch.zeros(Ns, t.shape[1], device=t.device, dtype=t.dtype)

        composed_deltas = (
            torch.cat([static_zeros(d_xyz), d_xyz], dim=0),
            torch.cat([static_zeros(d_scaling), d_scaling], dim=0),
            torch.cat([static_zeros(d_rotation), d_rotation], dim=0),
        )

        return static_deltas, (d_xyz, d_scaling, d_rotation), composed_deltas



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

    def create_from_pcd(
            self, 
            static_ptc: BasicPointCloud,
            dynamic_ptc: BasicPointCloud,
            cameras_extent: float
        ):

        self.static.create_from_pcd(static_ptc, cameras_extent)
        print("Number of static background Gaussians at init: ", self.static.get_xyz.shape[0])
        self.dynamic.create_from_pcd(dynamic_ptc, cameras_extent)
        print("Number of dynamic foreground Gaussians at init: ", self.dynamic.get_xyz.shape[0])

    def training_setup(self, training_args):
        self.deform.train_setting(training_args)

        self.static.training_setup(training_args)
        self.dynamic.training_setup(training_args)

    def oneupSHdegree(self):
        self.static.oneupSHdegree()
        self.dynamic.oneupSHdegree()

        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
