import os
import random
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos

import torch

from pathlib import Path
from typing import List, Optional

from scene.decoupled_model import DecoupledModel
from scene.gaussian_model import GaussianModel


class Scene:
    def __init__(
            self, 
            args: ModelParams, 
            model: DecoupledModel, 
            resolution_scales: List[float] = [1.0],
            load_iteration: Optional[int] = None, 
            shuffle: bool=False
        ):

        self.model_path = args.model_path
        self.loaded_iter = None
        self.model = model

        scene_info = sceneLoadTypeCallbacks["ours"](args.source_path)

        self.num_frames = len(scene_info. cam_infos)

        # TODO
        self.cameras_extent = 0.05 

        # TODO - implement loading logic for rendering
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

        if self.loaded_iter:
            print(f"Loading at iteration {self.loaded_iter}")
            directory = Path(self.model_path) / "point_cloud" / f"_{self.loaded_iter}"
            model.load_plys(directory)
        else:
            self.model.create_from_pcd(scene_info.pointclouds["static"], 
                                      scene_info.pointclouds["dynamic"], 
                                      self.cameras_extent)

        self.cameras = {}
        for resolution_scale in resolution_scales:
            self.cameras[resolution_scale] = cameraList_from_camInfos(scene_info.cam_infos,
                                                                      resolution_scale, 
                                                                      args)

    def save(self, iteration):
        self.model.save(iteration, self.model_path)

    def getCameras(self, scale=1.0):
        return self.cameras[scale]
