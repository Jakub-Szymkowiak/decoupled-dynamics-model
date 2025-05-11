import warnings

import numpy as np
import open3d as o3d

from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.graphics_utils import BasicPointCloud


@dataclass
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray   
    normals: np.ndarray=None

    def rescale(self, scale: float=1.0, translation: np.ndarray=np.zeros(3)):
        self.xyz = (self.xyz - translation) * scale
        if self.normals is not None:
            self.normalize_normals()

    def apply_mask(self, mask):
        self.xyz = self.xyz[mask]
        self.rgb = self.rgb[mask]
        if self.normals is not None:
            self.normals = self.normals[mask]

    def estimate_normals(self, knn=30):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn ))
        self.normals = np.asarray(pcd.normals)

    def normalize_normals(self):
        normals = self.normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = normals / np.where(norms == 0, 1e-8, norms)

    def to_BasicPointCloud(self):
        if self.normals is None:
            self.estimate_normals()
            self.normalize_normals()

        return BasicPointCloud(points=self.xyz, colors=self.rgb, normals=self.normals)


class SceneProcessor:
    def __init__(self, frames: List[Frame]):
        assert all(isinstance(f, Frame) for f in frames), "All elements must be Frame instances"
        self._frames = frames

        self._static_pointcloud = None
        self._dynamic_pointcloud = None
        self._dynamic_centroids = None

    @property
    def num_frames(self):
        return len(self._frames)

    @property
    def frames(self):
        return self._frames

    def get_static_pointcloud(self):
        return self._static_pointcloud
 
    def get_dynamic_pointcloud(self):
        return self._dynamic_pointcloud

    def get_dynamic_centroids(self):
        return self._dynamic_centroids

    def get_frame(self, frame_id):
        return self._frames[frame_id]

    def align_poses(self, ref_frame_id: Optional[int]=None):
        if ref_frame_id is None:
            ref_frame_id = self.num_frames // 2

        ref_pose_inv = self.get_frame(ref_frame_id).inverse.homogeneous
        
        for frame in self._frames:
            aligned = ref_pose_inv @ frame.pose.homogeneous
            frame.pose = Pose.from_homogeneous(aligned)

    def create_pointclouds(self, downsample: int=1, conf_thrs: float=0.6):

        # rescale intrinsics
        for frame in self._frames:
            frame.intrinsics = frame.intrinsics.rescaled(1.0 / downsample)

        # static pointcloud
        pts, colors = [], []
        for frame in self._frames:
            p, c = frame.get_static_points(stride=downsample)
            pts.append(p)
            colors.append(c)
        
        xyz = np.concatenate(pts, axis=0).astype(np.float32)
        rgb = np.concatenate(colors, axis=0).astype(np.float32)

        self._static_pointcloud = PointCloud(xyz=xyz, rgb=rgb)
        self._static_pointcloud.estimate_normals()

        # dynamic pointcloud
        pts, colors, centroids = [], [], []
        for frame in self._frames:
            p, c = frame.get_dynamic_points(stride=downsample)
            pts.append(p)
            colors.append(c)
            
            if p.size == 0:
                if np.all(frame.dynamic.dmask == 0):
                    warnings.warn(f"Empty dynamic motion mask for frame {frame.frame_id}")
            warnings.warn(f"No points created for dynamic frame {frame.frame_id}. Will result in wrong centroid computation.")

            centroids.append(p.mean())
        
        self._dynamic_centroids = np.asarray(centroids)

        xyz = np.concatenate(pts, axis=0).astype(np.float32)
        rgb = np.concatenate(colors, axis=0).astype(np.float32)

        self._dynamic_pointcloud = PointCloud(xyz=xyz, rgb=rgb)
        self._dynamic_pointcloud.estimate_normals()

    def normalize(self, radius: float=0.05):
        base = np.array([f.pose.T for f in self._frames])
        center = base.mean(axis=0)

        scale = radius / np.max(np.linalg.norm(base - center, axis=1))

        self._dynamic_centroids = (self._dynamic_centroids - center) * scale
        self._static_pointcloud.rescale(scale=scale, translation=center)
        self._dynamic_pointcloud.rescale(scale=scale, translation=center)

        for frame in self._frames:
            frame.pose = frame.pose.rescaled(scale=scale, translation=center)

    def trim_distant_static(self, percent: float=10.0):
        xyz = self._static_pointcloud.xyz
        distances = np.linalg.norm(xyz, axis=1)
        thrs = np.percentile(distances, 100 - percent)
        mask = distances <= thrs
        
        self._static_pointcloud.apply_mask(mask)

    

    