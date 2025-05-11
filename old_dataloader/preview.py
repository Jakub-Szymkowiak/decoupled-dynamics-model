import numpy as np
import matplotlib.pyplot as plt
import json

from itertools import product

from pathlib import Path
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from .data_structures import PointCloud, Pose, PreviewConfig
from .scene import Scene 



class Preview:
    def __init__(self, scene: Scene):
        self.scene = scene

    def render(self, output_path: Path, config: PreviewConfig):
        composer = GeometryComposer(self.scene, config)
        renderer = SceneRenderer(output_path, config)

        renderer.draw(composer)
        renderer.save_json(composer.json_data)
        renderer.save_figure()

class GeometryComposer:
    def __init__(self, scene: Scene, config: PreviewConfig):
        self.scene = scene
        self.config = config
        self.json_data = {}

    def add_geometry(self, ax):
        if self.config.world_axes:
            self._add_world_axes(ax)

        if self.config.pointcloud:
            self._add_pointcloud(ax, self.scene.pointcloud)

        if self.config.camera_trace:
            self._add_camera_trace(ax)

        if self.config.frame_poses:
            for frame in self.scene.frames[::self.config.frames_downsample]:
                self._add_pose(ax, frame.pose)

        if self.config.bounding_box:
            self._add_bbox(ax, self.scene.pointcloud)

    def _add_pointcloud(self, ax, ptc: PointCloud):
        ax.scatter(ptc.xyz[:, 0], ptc.xyz[:, 1], ptc.xyz[:, 2], c=ptc.rgb, s=0.2, alpha=0.01)

    def _add_pose(self, ax, pose: Pose):
        _draw_axes(ax, pose.T, pose.R, length=1.5, alpha=1.0)
        self.json_data.setdefault("poses", []).append({
            "position": pose.T.tolist(),
            "rotation": pose.R.tolist()
        })

    def _add_camera_trace(self, ax):
        centers = np.array([f.pose.T for f in self.scene.frames])
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   c="red", s=10, marker="o", alpha=0.8)

    def _add_bbox(self, ax, ptc: PointCloud):
        xyz = ptc.xyz
        min_pt, max_pt = np.min(xyz, axis=0), np.max(xyz, axis=0)
        corners = np.array(list(product(*zip(min_pt, max_pt))))
        edges = [(i, j) for i in range(8) for j in range(i+1, 8) if bin(i ^ j).count("1") == 1]

        for i, j in edges:
            ax.plot(*zip(corners[i], corners[j]), color="k", lw=0.7, alpha=0.5)

        self.json_data["bounding_box"] = {
            "min": min_pt.tolist()
        }

    def _add_world_axes(self, ax):
        origin = np.zeros(3)
        _draw_axes(ax, origin, np.eye(3), length=1.5, alpha=1.0)


class SceneRenderer:
    def __init__(self, output_path: Path, config: PreviewConfig):
        self.output_path = output_path
        self.config = config
        self.fig = plt.figure(figsize=self.config.figsize)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("white")

    def draw(self, composer: GeometryComposer):
        composer.add_geometry(self.ax)

        elev = self.config.elev
        azim = self.config.azim
        dist = self.config.dist
        ax_range = self.config.axes_range

        self.ax.set_xlim(ax_range)
        self.ax.set_ylim(ax_range)
        self.ax.set_zlim(ax_range)

        self.ax.set_xlabel("X", fontsize=8)
        self.ax.set_ylabel("Y", fontsize=8)
        self.ax.set_zlabel("Z", fontsize=8)

        self.ax.view_init(elev=elev, azim=azim)
        self.ax.dist = dist
        self.ax.tick_params(axis="both", which="major", labelsize=6)

    def save_json(self, json_data):
        if self.config.save_json:
            json_path = self.output_path.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=4)

    def save_figure(self):
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=300)
        plt.close()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs, ys, zs = self._verts3d
        xs2d, ys2d, zs2d = proj3d.proj_transform(xs, ys, zs, self.axes.M)
        self.set_positions((xs2d[0], ys2d[0]), (xs2d[1], ys2d[1]))
        return np.min(zs2d)

    def draw(self, renderer):
        super().draw(renderer)


def _draw_axes(ax, origin: np.ndarray, directions: np.ndarray, length: float, alpha: float):
    for i, color in enumerate(["r", "g", "b"]):
        axis = directions[:, i] * length
        arrow = Arrow3D([origin[0], origin[0] + axis[0]],
                        [origin[1], origin[1] + axis[1]],
                        [origin[2], origin[2] + axis[2]],
                        mutation_scale=15, lw=2, arrowstyle="-|>", color=color, alpha=alpha)
        ax.add_artist(arrow)
