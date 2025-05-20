import time

from argparse import ArgumentParser
from pathlib import Path


import torch
import torchvision
import numpy as np

from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, DeformModel
from scene.decoupled_model import DecoupledModel
from utils.general_utils import safe_state
from utils.render_utils import get_rendering_func, mask_image


def render_scene(model, views, iteration, model_cfg, pipe_cfg, background, mode):
    root = Path(model_cfg.model_path).resolve()

    render_dir = mode if mode != "composed" else "images"

    render_path = root / "renders" / f"_{iteration}" / render_dir
    depth_path = root / "renders" / f"_{iteration}" / "depths"

    render_path.mkdir(parents=True, exist_ok=True)
    depth_path.mkdir(parents=True, exist_ok=True)

    run_renderer = get_rendering_func(model, pipe_cfg, background, model_cfg.is_6dof)

    # Saving loop
    for idx, view in enumerate(tqdm(views, desc="Rendering process:")):
        if model_cfg.load2gpu_on_the_fly:
            view.load2device()

        fid = torch.as_tensor(view.fid, device="cuda")
        deltas, _ = model.infer_deltas(fid, noise=False)

        results = run_renderer(mode, deltas, view)
        rendering = results["render"]

        if mode == "dynamic":
            rendering = mask_image(rendering, view.dmask)

        rendering_file = render_path / f"{idx:05d}.png"
        torchvision.utils.save_image(rendering, rendering_file)

        if mode == "composed":
            depth = results["depth"]
            depth = depth / (depth.max() + 1e-5)
            depth_file = depth_path / f"{idx:05d}.png"
            torchvision.utils.save_image(depth, depth_file)

    # Rendering FPS benchmark loop
    t_list = [] # store render times

    for idx, view in enumerate(views):
        fid = torch.as_tensor(view.fid, device="cuda")
        
        torch.cuda.synchronize()
        t_start = time.time()
        
        deltas, _ = model.infer_deltas(fid, noise=False)
        
        results = run_renderer("composed", deltas, view)

        torch.cuda.synchronize()

        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f"Test FPS for mode = {mode}: \033[1;35m{fps:.5f}\033[0m")

def render_sets(model_cfg: ModelParams, pipe_cfg: PipelineParams, args):
    with torch.no_grad():
        model = DecoupledModel(model_cfg.sh_degree, model_cfg.is_blender, model_cfg.is_6dof)
        model.deform.load_weights(model_cfg.model_path)
        
        scene = Scene(model_cfg, model, load_iteration=args.iteration)
        views = scene.getCameras()

        bg_color = [1, 1, 1] if model_cfg.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for mode in args.render_modes:
            render_scene(model, views, scene.loaded_iter, model_cfg, pipe_cfg, background, mode)


if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering parameters")

    model_cfg = ModelParams(parser, sentinel=True)
    pipe_cfg = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--render_modes", nargs="+", choices=["composed", "static", "dynamic"], default=["composed"])

    args = get_combined_args(parser)
    args.render_modes = list(set(args.render_modes + ["composed"]))

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model_cfg.extract(args), pipe_cfg.extract(args), args)
