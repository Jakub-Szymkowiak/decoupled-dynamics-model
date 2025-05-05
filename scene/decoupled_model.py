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

        self._delta_buffers = None
        self._time_input = None
        self._ast_noise = None

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

        # TODO - decide whether to use buffering

        Ns = self.static.get_xyz.shape[0]
        Nd = self.dynamic.get_xyz.shape[0]

        if noise:
            assert iteration is not None and time_interval is not None and smooth_term is not None
            self._ast_noise.normal_()
            self._ast_noise *= time_interval * smooth_term(iteration)
        else:
            self._ast_noise.zero_()

        self._time_input.copy_(fid.view(1, 1).expand(Nd, 1))
        self._time_input += self._ast_noise

        xyz = self.dynamic.get_xyz.detach()
        d_xyz, d_rotation, d_scaling  = self.deform.step(xyz, self._time_input)

        buffers = {
            "d_xyz": d_xyz,
            "d_scaling": d_scaling,
            "d_rotation": d_rotation
        }

        for name in ["d_xyz", "d_scaling", "d_rotation"]:
            buf = self._delta_buffers[name]
            data = buffers[name]
            assert buf[Ns:].shape == data.shape, f"{name} shape mismatch: buf={buf[Ns:].shape}, data={data.shape}"

            buf[:Ns].zero_()
            buf[Ns:] = data.detach()

        static_deltas = (0.0, 0.0, 0.0)
        dynamic_deltas = (d_xyz, d_rotation, d_scaling)
        composed_deltas = (
            self._delta_buffers["d_xyz"].clone(),
            self._delta_buffers["d_rotation"].clone(),
            self._delta_buffers["d_scaling"].clone()
        )

        return static_deltas, dynamic_deltas, composed_deltas


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

        Ns, Nd = self.static.get_xyz.size(0), self.dynamic.get_xyz.size(0)

        print("Number of static background Gaussians at init: ", Ns)
        print("Number of dynamic foreground Gaussians at init: ", Nd)

        self._delta_buffers = {
            "d_xyz": torch.zeros(Ns + Nd, 3, device="cuda"),
            "d_scaling": torch.zeros(Ns + Nd, 3, device="cuda"),
            "d_rotation": torch.zeros(Ns + Nd, 4, device="cuda")
        }

        self._time_input = torch.zeros(Nd, 1, device="cuda")
        self._ast_noise = torch.zeros(Nd, 1, device="cuda")

    def training_setup(self, training_args):
        self.deform.train_setting(training_args)

        self.static.training_setup(training_args)
        self.dynamic.training_setup(training_args)

    def oneupSHdegree(self):
        self.static.oneupSHdegree()
        self.dynamic.oneupSHdegree()

        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
