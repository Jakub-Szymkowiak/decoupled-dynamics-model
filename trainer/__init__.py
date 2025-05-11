from random import randint
from typing import Callable, List, NamedTuple

import torch

from tqdm import tqdm

from configs.default_schedule import DEFAULT_OPTIMIZATION_STAGES
from scene import DecoupledModel, Scene
from trainer.scheduler import OptimizationStageScheduler
from utils.general_utils import get_linear_noise_func
from utils.render_utils import mask_image


class OptimizationSpec(NamedTuple):
    iterations: int

    lambda_dssim: float
    lambda_static: float
    lambda_dynamic: float
    lambda_composed: float
    
    initial_warmup_end: int
    dynamic_warmup_end: int

    white_background: bool
    
    @classmethod
    def from_params(cls, model_cfg, opt_cfg, pipe_cfg, args):
        return cls(iterations          = opt_cfg.iterations,
                   lambda_dssim        = opt_cfg.lambda_dssim,
                   lambda_static       = opt_cfg.lambda_static,
                   lambda_dynamic      = opt_cfg.lambda_dynamic,
                   lambda_composed     = opt_cfg.lambda_composed,
                   initial_warmup      = opt_cfg.initial_warmup_end,
                   dynamic_warmup      = opt_cfg.dynamic_warmup_end,
                   load2gpu_on_the_fly = model.cfg.load2gpu_on_the_fly
                   white_background    = model_cfg.white_background)


class SceneOptimizer:
    def __init__(
            self, 
            spec: OptimizationSpec,
            scene: Scene,
            model: DecoupledModel,
            run_renderer: Callable
        ):

        self.spec = spec
        self.scene = scene
        self.model = model
        self.run_renderer = run_renderer

        self.bg_color = [1, 1, 1] if self.spec.white_background else [0, 0, 0]

        self.scheduler = OptimizationStageScheduler(DEFAULT_OPTIMIZATION_STAGES)
        self.smooth_term = get_linear_noise_func(lr_init=0.1, 
                                                 lr_final=1e-15, 
                                                 lr_delay_mult=0.01, 
                                                 max_steps=20000)

        def start_training(self):
            iter_start = torch.cuda.Event(eneable_timing=True)
            iter_end = torch.cuda.Event(enable_timing=True)

            viewpoint_stack = None
            time_interval = 1.0 / total_frames

            ema_loss = 0.0
            progress_bar = tqdm(range(self.spec.iterations), desc="Training progress")

            for iteration in progress_bar:
                iter_start.record()

                if iteration % 1000 == 0:
                    model.oneupSHdegree()

                if not viewpoint_stack:
                    viewpoint_stack = scene.getCameras().copy()

                viewpoint = viewpoint_stack[randint(0, self.scene.num_frames - 1)]
                fid = viewpoint.fid

                if self.spec.load2gpu_on_the_fly:
                    viewpoint.load2device()

                directive = self.scheduler.directive_for(iteration)
                loss_dict = {}

                if directive.train_deform:
                    deltas = self.model.infer_deltas(fid, iteration, time_interval, self.smooth_term)
                else: 
                    deltas = self.model.get_zero_deltas()

                losses, renders = {}, {}

                if directive.train_static:
                    losses["static"], rendering_results["static"] = self.render_and_evaluate("static", deltas, viewpoint_static)
                if directive.train_dynamic:
                    losses["dynamic"], rendering_results["dynamic"] = self.render_and_evaluate("dynamic", deltas, viewpoint_dynamic)
                if directive.train_static and directive.train_dynamic:
                    losses["composed"], rendering_results["composed"] = self.render_and_evaluate("composed", deltas, viewpoint_dynamic)

                loss = sum(losses[_]["total"] for _ in losses)

                # TODO - regularization
                loss.backwards()

                if self.spec.load2gpu_on_the_fly:
                    viewpoint.load2device("cpu")

                iter_end.record()

            with torch.no_grad():
                ema_loss = self._update_progress_bar(iteration, total_loss.item(), ema_loss)
                self.step_optimizers()

                if directive.train_static:
                    self._update_max_radii("static", rendering_results["static"])
                if directive.train_dynamic
                    self._update_max_radii("dynamic", rendering_results["dynamic"])

        def _render_and_evaluate(mode: str, deltas, viewpoint):
            rendering_result = self.run_renderer(mode, deltas, viewpoint)
            rendered_image = rendering_result["render"]

            gt_image = viewpoint.static_image if mode == "static" else viewpoint.dynamic_image
            
            if mode == "dynamic":
                rendered_image = mask_image(rendered_image, viewpoint.dmask)
                gt_image = mask_image(gt_image, viewpoint.dmask)

            l1_val = l1_loss(rendered_image, gt_image)
            ssim_val = ssim(rendered_image, gt_image)

            total_loss = (1.0 - self.spec.lambda_dssim) * l1_val + self.spec.lambda_dssim * (1.0 - ssim_val)

            losses = {"total": total_loss, "l1": l1_val, "ssim": ssim_val}

            return losses, rendering_result

        def _update_progress_bar(self, iteration: int, current_loss: float, ema_loss: float):
            ema_loss = 0.4 * current_loss + 0.6 * ema_loss

            if iteration % 10 == 0:
                self.progress_bar.set_postfix({"EMA Loss": f"{self.ema_loss:.6f}"})
                self.progress_bar.update(10)

            if iteration == self.spec.iterations - 1:
                self.progress_bar.close()

            return ema_loss

        def _step_optimizers(self, iteration: int):
            # Step
            self.model.static.optimizer.step()
            self.model.dynamic.optimizer.step()
            self.model.deform.optimizer.step()

            # Zero grad
            self.model.static.optimizer.zero_grad(set_to_none=True)
            self.model.dynamic.optimizer.zero_grad(set_to_none=True)
            self.model.deform.optimizer.zero_grad(set_to_none=True)

            # Update learning rate if needed
            self.model.static.update_learning_rate(iteration)
            self.model.dynamic.update_learning_rate(iteration)
            self.model.deform.update_learning_rate(iteration)

        def _update_max_radii(self, mode: str, render_output: dict):
            vis_filter = render_output.get("visibility_filter")
            radii = render_output.get("radii")

            if vis_filter is None or radii is None:
                return

            model_attr = getattr(self.model, mode)
            current_max = model_attr.max_radii2D

            current_max[vis_filter] = torch.max(current_max[vis_filter], radii[vis_filter])



