from typing import List, Optional

import numpy as np

from .data_structures import Intrinsics, PointCloud, Pose


class Frame:
    def __init__(
            self,
            frame_id: int,
            image: np.ndarray,
            depth: np.ndarray,
            confs: np.ndarray,
            paths: dict,
            intrinsics: Intrinsics,
            pose: Pose,
            dmask: Optional[np.ndarray] = None,
        ):

        self.frame_id = frame_id
        self.image = image
        self.depth = depth
        self.confs = confs
        self.dmask = dmask
        self.paths = paths
        self.intrinsics = intrinsics
        self.pose = pose

        self.H, self.W = self.image.shape[:2]

        assert self.image.shape[:2] == (self.H, self.W), "Image shape mismatch"
        assert self.depth.shape[:2] == (self.H, self.W), "Depth shape mismatch"
        assert self.confs.shape[:2] == (self.H, self.W), "Confs shape mismatch"
        
        if self.dmask is not None:
            assert self.dmask.shape[:2] == (self.H, self.W), "Dynamic motion mask shape mismatch"

    def to_points(self, stride: int=1, conf_thrs: float=0.0, use_masks: bool=False):
        _downsample = lambda arr: arr[::stride, ::stride]
        image, depth, confs = _downsample(self.image), _downsample(self.depth), _downsample(self.confs)

        if use_masks:
            assert self.dmask is not None, "Cannot use dynamic motion masks; masks haven't been loaded"
            dmask = _downsample(self.dmask)

        H, W = image.shape[:2]

        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        us, vs = us + 0.5, vs + 0.5

        homogeneous = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)
        local_dirs = self.intrinsics.backproject(homogeneous)

        points_cam = local_dirs * depth.flatten()[:, None]
        
        points_world = self.pose.transform_points(points_cam)
        
        conf_flat = confs.flatten()
        mask = (conf_flat > conf_thrs) 

        if use_masks:
            inv_dmask_flat = dmask.flatten().astype(bool)
            mask = mask & inv_dmask_flat

        points = points_world[mask]
        colors = image.reshape(-1, 3)[mask] / 255.0
        confidences = conf_flat[mask]

        return points, colors, confidences


class Scene:
    def __init__(self, frames: List[Frame], dynamic: bool=False, conf_thrs: float=0.6):
        assert all(isinstance(f, Frame) for f in frames), "All elements must be Frame instances"
        self._frames = frames
        self.dynamic = dynamic
        self.conf_thrs = conf_thrs
        self._pointcloud = None

    @property
    def frames(self) -> List[Frame]:
        return self._frames

    @property
    def pointcloud(self) -> PointCloud:
        if self._pointcloud is None:
            raise RuntimeError("Pointcloud not created yet. Call create_pointcloud() first.")
        return self._pointcloud

    @property
    def num_frames(self) -> int:
        return len(self._frames)

    def get_frame(self, frame_id: int) -> Frame:
        return self._frames[frame_id]

    def create_pointcloud(self, downsample: int=1):
        pts, colors = [], []

        if self.dynamic:
            ref_frame = len(self.frames) // 2
            n_frames = 2
            frames = self._frames[ref_frame-n_frames:ref_frame+n_frames+1]
        else:
            frames = self.frames

        for frame in frames:
            p, c, _ = frame.to_points(stride=downsample, conf_thrs=self.conf_thrs, use_masks=self.dynamic)
            pts.append(p)
            colors.append(c)

        xyz = np.concatenate(pts, axis=0).astype(np.float32)
        rgb = np.concatenate(colors, axis=0).astype(np.float32)

        self._pointcloud = PointCloud(xyz=xyz, rgb=rgb)
        self._pointcloud.estimate_normals()

    def normalize(self, radius: float = 0.01) -> None:
        assert self._pointcloud is not None, "Pointcloud must be created before normalization"

        base = np.array([f.pose.T for f in self._frames])
        center = base.mean(axis=0)
        
        scale = radius / np.max(np.linalg.norm(base - center, axis=1))

        self._pointcloud.rescale(scale=scale, translation=center)
        self._pointcloud.normalize_normals()

        for frame in self._frames:
            frame.pose = frame.pose.rescaled(scale=scale, translation=center)
            frame.intrinsics = frame.intrinsics.rescaled(scale=scale)

        self._switch_opengl()

    def align_poses(self, ref_frame_id: Optional[int]=None):
        if ref_frame_id is None:
            ref_frame_id = len(self._frames) // 2

        ref_pose_inv = np.linalg.inv(self._frames[ref_frame_id].pose.homogeneous)

        for frame in self._frames:
            aligned = ref_pose_inv @ frame.pose.homogeneous
            frame.pose = Pose.from_homogeneous(aligned)

    def trim_distant(self, percent: float=10.0):
        assert self._pointcloud is not None, "Pointcloud must be created first"

        xyz = self._pointcloud.xyz
        distances = np.linalg.norm(xyz, axis=1)
        threshold = np.percentile(distances, 100 - percent)
        mask = distances <= threshold

        self._pointcloud.xyz = xyz[mask]
        self._pointcloud.rgb = self._pointcloud.rgb[mask]
        self._pointcloud.normals = self._pointcloud.normals[mask]

    def _switch_opengl(self):
        F = np.diag([1, -1, -1])
        for frame in self._frames:
            R = F @ frame.pose.R @ F
            T = F @ frame.pose.T
            mat = np.eye(4)
            mat[:3, :3] = R
            mat[:3, 3] = T
            frame.pose = Pose.from_homogeneous(mat)
