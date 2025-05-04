from typing import List

import numpy as np
import imageio.v3 as iio

from pathlib import Path

from .data_structures import Intrinsics, Pose, SceneSpec
from .scene import Frame


def load_scene_data(spec: SceneSpec) -> List[Frame]:
    intrinsics = np.loadtxt(spec.intrinsics_path, dtype=np.float32).reshape(-1, 3, 3)
    trajectory = np.loadtxt(spec.trajectory_path, dtype=np.float32)[:, 1:].reshape(-1, 7)

    frames = []
    for idx, (K, pose_vec) in enumerate(zip(intrinsics, trajectory)):
        image_path = spec.image_path(idx)
        depth_path = spec.depth_path(idx)
        conf_path = spec.conf_path(idx)

        image = np.asarray(iio.imread(image_path))
        depth = np.load(depth_path)
        confs = np.load(conf_path)


        paths = {
            "image": image_path,
            "depth": depth_path,
            "confs": conf_path
        }

        frame = Frame(frame_id=idx,
                     image=image,
                     depth=depth,
                     confs=confs,
                     paths=paths,
                     intrinsics=Intrinsics(K),
                     pose=Pose(pose_vec))
                     
        frames.append(frame)

    return frames