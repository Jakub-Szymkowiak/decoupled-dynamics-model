import sys

from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import DecoupledModel, Scene
from trainer import SceneOptimizer, OptimizationSpec
from utils.general_utils import safe_state


def run_training(model_cfg, opt_cfg, pipe_cfg, args):
    model = DecoupledModel(model_cfg.sh_degree)
    scene = Scene(model_cfg, model)
    model.training_setup(opt_cfg)

    bg_color = [1, 1, 1] if model_cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    run_renderer = get_rendering_func(model, pipe, background, model_cfg.is_6dof) 
    
    print("Starting optimization.")
    spec = OptimizationSpec.from_params(opt_cfg)
    scene_optimizer = SceneOptimizer(spec, scene, model, run_renderer)
    scene_optimizer.start_training()

    print("Finished optimization.")





def parse_args():
    parser = ArgumentParser(description="Training script parameters")

    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")

    test_interval = 2500
    max_iters = 40_000
    default_test_iterations = list(range(test_interval, max_iters, test_interval))
    default_save_iterations = [1, 5000, 10000, 15000, 30000]

    parser.add_argument("--test_iterations", nargs="+", type=int, default=default_test_iterations)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=default_save_iterations)



    args = parser.parse_args(sys.argv[1:])

    model_cfg = model_params.extract(args)
    opt_cfg = optimization_params.extract(args)
    pipe_cfg = pipeline_params.extract(args)

    args.save_iterations.append(opt_cfg.iterations)

    return model_cfg, opt_cfg, pipe_cfg, args


if __name__ == "__main__":
    model_cfg, opt_cfg, pipe_cfg, args = parse_args()

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    

