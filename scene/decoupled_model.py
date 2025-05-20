from pathlib import Path
from typing import Optional

import torch

from scene.deform_model import DeformModel
from scene.gaussian_model import BasicPointCloud, GaussianModel
from utils.delta_utils import DeformDeltas


class DecoupledModel:
    def __init__(self, sh_degree: int, is_blender: bool=False, is_6dof: bool=False):
        self.static = GaussianModel(sh_degree)
        self.dynamic = GaussianModel(sh_degree)
        self.deform = DeformModel(is_blender, is_6dof)

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        
        self._Ns = None # number of static Gaussians
        self._Nd = None # number of dynamic Gaussians

        self._ast_noise = None

    def get_models(self):
        return {"static": self.static, "dynamic": self.dynamic, "composed": self}

    def infer_deltas(
            self,
            fid: torch.Tensor,
            iteration: Optional[int] = None,
            time_interval: Optional[float] = None,
            smooth_term: Optional[torch.Tensor] = None,
            noise: bool=True,
            with_grad: bool=False
        ):

        if noise:
            assert_msg = "Must provide parameters for noise = True; or use noise = False"
            assert iteration is not None and time_interval is not None and smooth_term is not None, assert_msg
            self._ast_noise.normal_()
            self._ast_noise *= time_interval * smooth_term(iteration)
        else:
            self._ast_noise.zero_()

        xyz = self.dynamic.get_xyz.detach().to(torch.float32)

        fid_input = fid.view(1, 1).expand(self._Nd, 1)
        if with_grad:
            fid_input.requires_grad_(True)

        time_input = fid_input + self._ast_noise

        raw_deltas = self.deform.step(xyz, time_input)

        static_deltas = DeformDeltas.get_zero_deform()
        dynamic_deltas = DeformDeltas(*raw_deltas)
        composed_deltas = DeformDeltas.prepare_composed(self._Ns, dynamic_deltas)

        return {"static": static_deltas, "dynamic": dynamic_deltas, "composed": composed_deltas}, fid_input

    def get_zero_deltas(self) -> DeformDeltas:
        return {mode: DeformDeltas.get_zero_deform() for mode in ("static", "dynamic", "composed")}

    def _concat_attr(self, attr_name: str) -> torch.Tensor:
        static_attr = getattr(self.static, attr_name)
        dynamic_attr = getattr(self.dynamic, attr_name)
        return torch.cat([static_attr, dynamic_attr], dim=0)

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