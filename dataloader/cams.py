from dataclasses import dataclass
from typing import Optional

from scipy.spatial.transform import Rotation


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
        return -self.R[:, 2] 

    def rescaled(self, scale: float=1.0, translation: np.ndarray=np.zeros(3)):
        H = self.homogeneous.copy()
        H[:3, 3] -= translation
        H[:3, 3] *= scale
        return Pose.from_homogeneous(H)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return points @ self.R.T + self.T

    @classmethod
    def from_homogeneous(cls, H: np.ndarray) -> "Pose":
        assert H.shape == (4, 4), "Homogeneous matrix must be 4x4."

        R = H[:3, :3]
        T = H[:3, 3]

        q = Rotation.from_matrix(R).as_quat()
        q_wxyz = np.array([q[3], q[0], q[1], q[2]])

        pose_vec = np.concatenate([T, q_wxyz])
        return cls(pose_vec)