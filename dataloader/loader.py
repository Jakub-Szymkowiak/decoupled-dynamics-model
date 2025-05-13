import imageio.v3 as iio
import numpy as np

from pathlib import Path
from tqdm import tqdm

from configs.default_schema import DEFAULT_SCHEMA

from dataloader.cams import Intrinsics, Pose
from dataloader.schema import parse_schema
from dataloader.frame import StaticFrameData, DynamicFrameData, Frame


def load_scene_data(root, schema=DEFAULT_SCHEMA):
    root = Path(root).resolve() if not isinstance(root, Path) else root
    paths = parse_schema(root, schema)

    intrinsics = np.loadtxt(paths["cameras"]["intrinsics"], dtype=np.float32).reshape(-1, 3, 3)
    trajectory = np.loadtxt(paths["cameras"]["trajectory"], dtype=np.float32)[:, 1:].reshape(-1, 7)
    
    frames = []
    for idx, (K_vec, pose_vec) in enumerate(zip(intrinsics, trajectory)):
        static_image_path = paths["static"]["images"] / f"frame_{idx:04d}.png"
        static_depth_path = paths["static"]["depths"] / f"frame_{idx:04d}.npy"

        static_data = StaticFrameData(image=np.asarray(iio.imread(static_image_path)),
                                      depth=np.load(static_depth_path))
        
        dynamic_image_path = paths["dynamic"]["images"] / f"frame_{idx:04d}.png"
        dynamic_depth_path = paths["dynamic"]["depths"] / f"frame_{idx:04d}.npy"
        dynamic_dmask_path = paths["dynamic"]["masks"]  / f"dynamic_mask_{idx}.png" 
        dynamic_confs_path = paths["dynamic"]["confs"]  / f"conf_{idx}.npy"

        dynamic_data = DynamicFrameData(image=np.asarray(iio.imread(dynamic_image_path)),
                                        depth=np.load(dynamic_depth_path),
                                        dmask=np.asarray(iio.imread(dynamic_dmask_path)),
                                        confs=np.load(dynamic_confs_path))

        intrinsics = Intrinsics(K_vec)
        pose = Pose(pose_vec)

        frame = Frame(frame_id=idx, 
                      static_data=static_data, 
                      dynamic_data=dynamic_data,
                      pose=pose,
                      intrinsics=intrinsics)

        frames.append(frame)

    return frames, paths