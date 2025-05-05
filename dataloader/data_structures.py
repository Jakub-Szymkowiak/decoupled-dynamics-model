from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from pathlib import Path

from scipy.spatial.transform import Rotation


@dataclass
class PointCloud:
    xyz: np.ndarray
    rgb: np.ndarray   
    normals: np.ndarray=None

    def rescale(self, scale: float=1.0, translation: np.ndarray=np.zeros(3)):
        self.xyz = (self.xyz - translation) * scale

    def estimate_normals(self, knn=30):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.xyz)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn ))
        self.normals = np.asarray(pcd.normals)

    def normalize_normals(self):
        normals = self.normals
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = normals / np.where(norms == 0, 1e-8, norms)


@dataclass
class Intrinsics:
    K: np.ndarray
    
    @property
    def fx(self) -> float:
        return self.K[0, 0]

    @property
    def fy(self) -> float:
        return self.K[1, 1]

    @property
    def cx(self) -> float:
        return self.K[0, 2]

    @property
    def cy(self) -> float:
        return self.K[1, 2]

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    def backproject(self, homogeneous: np.ndarray) -> np.ndarray:
        return (self.K_inv @ homogeneous.T).T

    def rescaled(self, scale: float=1.0):
        K = self.K.copy()
        K[:2, :] *= scale
        return Intrinsics(K)


@dataclass 
class Pose:
    pose_vec: np.ndarray # [tx, ty, tz, | qw, qx, qy, qz]

    @property
    def q(self) -> np.ndarray:
        return self.pose_vec[3:]

    @property
    def q_xyzw(self) -> np.ndarray:
        q = self.q 
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
        return q_xyzw

    @property 
    def R(self) -> np.ndarray:
        return Rotation.from_quat(self.q_xyzw).as_matrix()

    @property
    def T(self) -> np.ndarray:
        return self.pose_vec[:3]

    @property
    def homogeneous(self) -> np.ndarray:
        mat = np.eye(4)
        mat[:3, :3] = self.R
        mat[:3, 3] = self.T
        return mat

    @property
    def inverse(self) -> "Pose":
        R = self.R
        T = self.T
        R_inv = R.T
        T_inv = -R_inv @ T

        mat_inv = np.eye(4)
        mat_inv[:3, :3] = R_inv
        mat_inv[:3, 3] = T_inv

        return Pose.from_homogeneous(mat_inv)

    @property 
    def forward_direction(self) -> np.ndarray:
        return -self.R[:, 2] # OpenGL 

    def rescaled(self, scale: float=1.0, translation: np.ndarray=np.zeros(3)):
        H = self.homogeneous
        H[:3, 3] -= translation
        H[:3, 3] *= scale
        return Pose.from_homogeneous(H)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return points @ self.R.T + self.T

    @classmethod
    def from_homogeneous(cls, H: np.ndarray) -> "Pose":
        if H.shape != (4, 4):
            raise ValueError("Homogeneous matrix must be 4x4.")

        R = H[:3, :3]
        t = H[:3, 3]

        q = Rotation.from_matrix(R).as_quat()
        q_wxyz = np.array([q[3], q[0], q[1], q[2]])

        pose_vec = np.concatenate([t, q_wxyz])

        return cls(pose_vec)
        

@dataclass
class SceneSpec:
    root: Path
    subdir: Path = Path(".")
    
    image_dir: Path = Path("images")
    depth_dir: Path = Path("depths")
    conf_dir: Path = Path("confs")
    
    intrinsics_file: Path = Path("pred_intrinsics.txt")
    trajectory_file: Path = Path("pred_traj.txt")

    dynamic: Optional[bool]=False
    dmask_dir: Path = Path("masks")

    def image_path(self, idx: int) -> Path:
        return self.root / self.subdir / self.image_dir / f"frame_{idx:04d}.png"

    def depth_path(self, idx: int) -> Path:
        return self.root / self.subdir / self.depth_dir / f"frame_{idx:04d}.npy"

    def conf_path(self, idx: int) -> Path:
        return self.root / self.subdir / self.conf_dir / f"conf_{idx}.npy"

    def dmask_path(self, idx: int) -> Path:
        assert self.dynamic, "Set dynamic = True to load dynamic motion masks"
        return self.root / self.subdir / self.dmask_dir / f"dynamic_mask_{idx}.png"

    @property
    def intrinsics_path(self) -> Path:
        return self.root / self.intrinsics_file

    @property
    def trajectory_path(self) -> Path:
        return self.root / self.trajectory_file


@dataclass
class PreviewConfig:
    pointcloud: bool=True
    frame_poses: bool=True
    camera_trace: bool=False
    bounding_box: bool=True
    world_axes: bool=True
    frames_downsample: int=1
    figsize: tuple=(10, 10)
    elev: int=20
    azim: int=60
    dist: float=10.0
    axes_range: tuple=(-3, 3)
    save_json: bool=False