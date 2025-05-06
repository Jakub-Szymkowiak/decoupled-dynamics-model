import os
import random
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos

from typing import List, Optional

from pathlib import Path

from scene.decoupled_model import DecoupledModel
from scene.gaussian_model import GaussianModel


class Scene:
    def __init__(
            self, 
            args: ModelParams, 
            model: DecoupledModel, 
            resolution_scales: List[float]=[1.0],
            load_iteration: Optional[int]=None, 
            shuffle: bool=False):

        self.model_path = args.model_path
        self.loaded_iter = None
        self.model = model

        scene_info = sceneLoadTypeCallbacks["ours"](args.source_path)

        # TODO
        self.cameras_extent = 5.0 

        # TODO - implement loading logic for rendering
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

        if self.loaded_iter:
            directory = Path(self.model_path) / "point_cloud" / f"_{self.loaded_iter}"
            model.load_plys(directory)
        else:
            self.model.create_from_pcd(
                scene_info.static_ptc, 
                scene_info.dynamic_ptc, 
                self.cameras_extent)

        scene_info = sceneLoadTypeCallbacks["ours"](args.source_path)

        # TODO - implement shuffling logic
        if shuffle:
            pass


        self.static_cams, self.dynamic_cams = {}, {}
        for resolution_scale in resolution_scales:
            _get_cam_list = lambda cams: cameraList_from_camInfos(cams, resolution_scale, args)
            self.static_cams[resolution_scale] = _get_cam_list(scene_info.static_cameras)
            self.dynamic_cams[resolution_scale] = _get_cam_list(scene_info.dynamic_cameras)

    def save(self, iteration):
        saving_dir = os.path.join(self.model_path, f"point_cloud/_{iteration}")

        static_ply_path = os.path.join(saving_dir, "static.ply")
        dynamic_ply_path = os.path.join(saving_dir, "dynamic.ply")

        self.model.static.save_ply(static_ply_path)
        self.model.dynamic.save_ply(dynamic_ply_path)

    def getStaticCameras(self, scale=1.0):
        return self.static_cams[scale]

    def getDynamicCameras(self, scale=1.0):
        return self.dynamic_cams[scale]
