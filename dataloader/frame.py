import numpy as np

from typing import NamedTuple
from cams import Intrinsics, Pose


class StaticFrameData(NamedTuple):
    image: np.ndarray
    depth: np.ndarray

class DynamicFrameData(NamedTuple):
    image: np.ndarray
    depth: np.ndarray
    confs: np.ndarray
    dmask: np.ndarray

class Frame:
    def __init__(
            self,
            frame_id: int,
            static_data: StaticFrameData,
            dynamic_data: DynamicFrameData,
            pose: Pose,
            intrinsics: Intrinsics
        ):

        self.frame_id = frame_id
        self.static = static_data
        self.dynamic = dynamic_data
        self.pose = pose
        self.intrinsics = intrinsics
        
        self.centroid = None

        self.H, self.W = self.static.image.shape[:2]

        assert self.static.image.shape[:2] == (self.H, self.W), "Static image shape mismatch"
        assert self.static.depth.shape[:2] == (self.H, self.W), "Static depth shape mismatch"
        
        assert self.dynamic.image.shape[:2] == (self.H, self.W), "Dynamic image shape mismatch"
        assert self.dynamic.depth.shape[:2] == (self.H, self.W), "Dynamic depth shape mismatch"
        assert self.dynamic.confs.shape[:2] == (self.H, self.W), "Dynamic confs shape mismatch"
        assert self.dynamic.dmask.shape[:2] == (self.H, self.W), "Dynamic mask shape mismatch"

    def get_static_points(self, stride: int=1):
        return self.to_points(image=self.static.image,
                              depth=self.static.depth,
                              stride=stride)

    def get_dynamic_points(self, conf_thrs: float=0.6, stride: int=1):
        return self.to_points(image=self.dynamic.image,
                              depth=self.dynamic.depth,
                              confs=self.dynamic.confs,
                              dmask=self.dynamic.dmask,
                              stride=stride,
                              conf_thrs=conf_thrs)

    def _to_points(self, 
                   image: np.ndarray,
                   depth: np.ndarray,
                   confs: Optional[np.ndarray] = None,
                   dmask: Optional[np.ndarray] = None,
                   conf_thrs: Optional[float] = 0.6,
                   stride: int=1):
        
        _downsample = lambda arr: arr[::stride, ::stride]

        image, depth = _downsample(image), _downsample(depth)
        H, W = image.shape[:2]

        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        us, vs = us + 0.5, vs + 0.5
        homogeneous = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)

        local_dirs = self.intrinsics.backproject(homogeneous)
        points_cam = local_dirs * depth.reshape(-1, 1)
        points_world = self.pose.transform_points(points_cam)

        image_flat = image.reshape(-1, 3) # / 255.0

        if confs is not None:
            confs = _downsample(confs)
            conf_flat = confs.flatten()
            mask = conf_flat > conf_thrs
        else: 
            mask = np.ones(H * W, dtype=bool)
        
        if dmask is not None:
            dmask = _downsample(dmask)
            mask &= dmask.flatten().astype(bool)

        return points_world[mask], image_flat[mask]
        


    