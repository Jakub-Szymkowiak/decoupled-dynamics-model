#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical, render_wander_path
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
import time


from pathlib import Path

from scene.decoupled_model import DecoupledModel


def render_set(
        model_path, 
        load2gpu_on_the_fly, 
        is_6dof, 
        views,
        iteration, 
        model, 
        pipeline,
        background
    ):

    root = Path(model_path).resolve()
    render_path = root / "renders" / f"_{iteration}"
    depth_path = root / "renders" / f"_{iteration}"

    render_path.mkdir(parents=True, exist_ok=True)
    depth_path.mkdir(parents=True, exist_ok=True)

    t_list = [] # store render times

    # Saving loop
    for idx, view in enumerate(tqdm(views, desc="Rendering process:")):
        if load2gpu_on_the_fly:
            view.load2device()

        fid = torch.as_tensor(view.fid, device="cuda")
        deltas = model.infer_deltas(fid, noise=False)

        def _run_renderer(mode: str, viewpoint):
            return render(viewpoint, 
                          model.get_models()[mode], 
                          pipeline, 
                          background, 
                          deltas[mode].d_xyz, 
                          deltas[mode].d_rotation, 
                          deltas[mode].d_scaling, 
                          is_6dof)

        results = _run_renderer("composed", view)

        rendering = results["render"]
        depth = results["depth"]

        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

    # Rendering FPS benchmark loop
    for idx, view in enumerate(views):
        fid = torch.as_tensor(view.fid, device="cuda")
        
        torch.cuda.synchronize()
        t_start = time.time()
        
        deltas = model.infer_deltas(fid, noise=False)

        def _run_renderer(mode: str, viewpoint):
            return render(viewpoint, 
                          model.get_models()[mode], 
                          pipeline, 
                          background, 
                          deltas[mode].d_xyz, 
                          deltas[mode].d_rotation, 
                          deltas[mode].d_scaling, 
                          is_6dof)

        results = _run_renderer("composed", view)

        torch.cuda.synchronize()

        t_end = time.time()
        t_list.append(t_end - t_start)

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m, Num. of GS: {model.get_xyz.shape[0]}')


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    with torch.no_grad():
        model = DecoupledModel(dataset.sh_degree, dataset.is_blender, dataset.is_6dof)
        model.deform.load_weights(dataset.model_path)
        
        scene = Scene(dataset, model, load_iteration=iteration)
        views = scene.getDynamicCameras()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_func = render_set

        render_func(dataset.model_path, dataset.load2gpu_on_the_fly, dataset.is_6dof, 
                    views, scene.loaded_iter, model, pipeline, background)



if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering parameters")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))
