import numpy as np

from pathlib import Path

from dataloader.loader import load_scene_data
from dataloader.scene import SceneProcessor

from utils.graphics_utils import focal2fov, fov2focal


class CameraInfo(NamedTuple):
    uid: int
    fid: float

    R: np.array
    T: np.array

    FovY: np.array
    FovX: np.array

    static_image: np.array
    static_depth: np.array

    dynamic_image: np.array
    dynamic_depth: np.array
    
    dmask: np.array

    width: int
    height: int



class SceneInfo(NamedTuple):
    cam_infos: list
    pointclouds: dict
    centroids: list


def read_and_process(root: str):
    frames, paths = load_scene_data(root)
    processor = SceneProcessor(frames)
    
    processor.align_poses()
    processor.create_pointclouds()
    processor.trim_distant_static()
    processor.normalize()    
        
    return processor, paths


def read_cam_info_from_scene(processor: SceneProcessor, frame_id: int):
    frame = processor.get_frame(frame_id)
    num_frames = processor.num_frames

    uid = frame_id + 1
    fid = float(idx / (num_frames - 1))

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
    
    cam_info = CameraInfo(uid=uid, fid=fid, 
                          R=R, T=T, FovX=FovX, FovY=FovY,
                          static_image=frame.static.image
                          static_depth=frame.static.depth
                          dynamic_image=frame.dynamic.image
                          dynamic_depth=frame.dynamic.depth
                          dmask=frame.dynamic.dmask
                          width=width, height=height)

    return cam_info


def readSceneInfo(path):
    processor, paths = read_and_process(path)

    cam_infos = [read_cam_info_from_scene(processor, idx) for idx range(processor.num_frames)]

    pointclouds = {
        "static": processor.get_static_pointcloud().to_BasicPointCloud(), 
        "dynamic": processor.get_dynamic_pointcloud().to_BasicPointCloud()
    }

    centroids = processor.get_dynamic_centroids()

    scene_info = SceneInfo(cam_infos=cam_infos,
                           pointclouds=pointclouds,
                           centroids=centroids)

    return scene_info


sceneLoadTypeCallbacks = {
    "ours": readSceneInfo
}
