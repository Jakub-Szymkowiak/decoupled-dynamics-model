#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, fid
                 R, T, FoVx, FoVy, 
                 static_image, static_depth,
                 dynamic_image, dynamic_depth, dmask,  
                 trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, data_device="cuda"):

        super(Camera, self).__init__()

        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.static_image = static_image.clamp(0.0, 1.0).to(self.data_device)
        self.dynamic_image = dynamic_image.clamp(0.0, 1.0).to(self.data_device)
        
        self.fid = torch.tensor(fid).to(self.data_device)

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.static_depth = torch.tensor(static_depth).to(self.data_device)
        self.dynamic_depth = torch.tensor(dynamic_depth).to(self.data_device)

        self.dmask = torch.tensor(dmask).unsqueeze(0).to(device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale))
        self.world_view_transform = self.world_view_transform.transpose(0, 1).to(self.data_device)

        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, 
                                                fovX=self.FoVx, fovY=self.FoVy)                                          
        self.projection_matrix = self.projection_matrix.transpose(0,1).to(self.data_device)

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, data_device='cuda'):
        self.original_image = self.original_image.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
