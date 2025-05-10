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

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
from glob import glob
import cv2 as cv
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON

from typing import Optional

from pathlib import Path

from dataloader.data_structures import PreviewConfig, SceneSpec
from dataloader.io_utils import load_scene_data
from dataloader.scene import Scene
from dataloader.preview import Preview


class CameraInfo(NamedTuple):
    uid: int

    R: np.array
    T: np.array

    FovY: np.array
    FovX: np.array

    image_name: str

    image: np.array
    image_path: str

    depth: np.array
    depth_path: str

    dmask: np.array
    dmask_path: str

    width: int
    height: int

    fid: Optional[float]=None
    dynamic: bool=False


class SceneInfo(NamedTuple):
    static_ptc: BasicPointCloud
    dynamic_ptc: BasicPointCloud
    static_cameras: list
    dynamic_cameras: list


def read_scene(root: Path, subdir: str, is_dynamic: bool, preview_path: Optional[Path]=None):
    subdir = Path(subdir)
    spec = SceneSpec(root=root, subdir=subdir, dynamic=is_dynamic)
    frames = load_scene_data(spec)
    scene = Scene(frames, dynamic=is_dynamic)

    scene.align_poses()
    
    downsample   = 1   if is_dynamic else 2
    conf_thrs    = 0.6 if is_dynamic else 0.0
    half_window  = None #   if is_dynamic else None
    ref_frame_id = 0   if is_dynamic else None

    scene.create_pointcloud(
        downsample=downsample,
        conf_thrs=conf_thrs,
        half_window=half_window,
        ref_frame_id=ref_frame_id
    )

    scene.normalize()

    if preview_path is not None:
        preview = Preview(scene)

        config = PreviewConfig(pointcloud=True,
                               frame_poses=True,
                               camera_trace=True,
                               bounding_box=True,
                               world_axes=False,
                               save_json=False,
                               frames_downsample=1,
                               azim=180,
                               elev=90,
                               dist=10)

        preview.render(output_path=preview_path, config=config)

    
        
    return scene


def read_cam_info_from_scene(scene: Scene, idx: int, is_dynamic: bool=False):
    frame = scene.get_frame(idx)
    num_frames = scene.num_frames

    uid = idx + 1

    # intrinsics
    intrinsics = frame.intrinsics

    cx, cy = intrinsics.cx, intrinsics.cy
    fx, fy = intrinsics.fx, intrinsics.fy

    _round_int = lambda x: int(round(x))
    width, height = _round_int(cx * 2), _round_int(cy * 2)

    FovX, FovY = focal2fov(fx, width), focal2fov(fy, height)

    # extrinsics
    pose = frame.pose
    view = pose.inverse

    R, T = view.R.T, view.T

    image = frame.image
    depth = frame.depth

    image_name = frame.paths["image"].stem

    image_path = str(frame.paths["image"])
    depth_path = str(frame.paths["depth"])

    if is_dynamic:
        dmask = frame.dmask
        dmask_path = str(frame.paths["dmask"])
        fid = float(idx / (num_frames - 1))
    else:
        dmask = None # no mask for static background frames
        dmask_path = None
        fid = None # no fid for static background frames

    cam_info = CameraInfo(uid=uid, fid=fid, 
                          R=R, T=T, FovX=FovX, FovY=FovY,
                          image_name=image_name,
                          image=image, image_path=image_path,
                          depth=depth, depth_path=depth_path,
                          dmask=dmask, dmask_path=dmask_path,
                          width=width, height=height,
                          dynamic=is_dynamic)

    return cam_info


def readMonST3RSceneInfo(path):
    root = Path(path)

    # Preview
    # preview_path_static = Path("./static_preview.png").resolve()
    # preview_path_dynamic = Path("./dynamic_preview.png").resolve()
    preview_path_static, preview_path_dynamic = None, None

    static_scene = read_scene(root, "static", is_dynamic=False, preview_path=preview_path_static)
    dynamic_scene = read_scene(root, "dynamic", is_dynamic=True, preview_path=preview_path_dynamic)
    
    mismatch_msg = "Static and dynamic sequence length mismatch"
    assert static_scene.num_frames == dynamic_scene.num_frames, mismatch_msg
    num_frames = static_scene.num_frames

    _get_cam_infos = lambda scene, dynamic: [read_cam_info_from_scene(scene, idx, dynamic) for idx in range(num_frames)]
    static_cam_infos = _get_cam_infos(static_scene, False)
    dynamic_cam_infos = _get_cam_infos(dynamic_scene, True)

    _ptc_to_basic_ptc = lambda ptc: BasicPointCloud(points=ptc.xyz, colors=ptc.rgb, normals=ptc.normals)
    static_ptc = _ptc_to_basic_ptc(static_scene.pointcloud)
    dynamic_ptc = _ptc_to_basic_ptc(dynamic_scene.pointcloud)
    
    scene_info = SceneInfo(static_ptc=static_ptc, 
                           dynamic_ptc=dynamic_ptc,
                           static_cameras=static_cam_infos,
                           dynamic_cameras=dynamic_cam_infos)

    return scene_info


sceneLoadTypeCallbacks = {
    "ours": readMonST3RSceneInfo
}
