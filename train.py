import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from scene import DecoupledModel, Scene
from trainer import SceneOptimizer, OptimizationSpec
from utils.general_utils import safe_state
from utils.render_utils import get_rendering_func


def run_training(model_cfg, opt_cfg, pipe_cfg, args):
    model = DecoupledModel(model_cfg.sh_degree)
    scene = Scene(model_cfg, model)
    model.training_setup(opt_cfg)

    bg_color = [1, 1, 1] if model_cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    run_renderer = get_rendering_func(model, pipe_cfg, background, model_cfg.is_6dof) 
    
    print("Starting optimization.")
    
    spec = OptimizationSpec.from_params(model_cfg, opt_cfg, pipe_cfg, args)
    scene_optimizer = SceneOptimizer(spec, scene, model, run_renderer)
    scene_optimizer.start_training()

    print("Finished optimization.")

def prepare_output(args):
    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)

    cfg_log_path = model_path / "cfg_args"
    cfg_log_path.write_text(str(Namespace(**vars(args))))

def parse_args():
    parser = ArgumentParser(description="Training script parameters")

    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)

    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")

    test_interval = 200
    max_iters = 100_000
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

    prepare_output(args)
    run_training(model_cfg, opt_cfg, pipe_cfg, args)

    

