








        

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