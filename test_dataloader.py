from pathlib import Path

from dataloader.data_structures import SceneSpec, PreviewConfig
from dataloader.io_utils import load_scene_data
from dataloader.scene import Scene
from dataloader.preview import Preview


def main():
    root = Path("./data/bear")
    spec = SceneSpec(root=root)
    frames = load_scene_data(spec)
    scene = Scene(frames)

    scene.align_poses()
    scene.create_pointcloud()
    scene.normalize()

    preview = Preview(scene)
    preview.render(
        output_path=Path("test_scene_preview.jpg"),
        config=PreviewConfig(
            pointcloud=True,
            frame_poses=False,
            camera_trace=True,
            bounding_box=True,
            world_axes=False,
            save_json=True,
            frames_downsample=6
        )
    )

if __name__ == "__main__":
    main()

