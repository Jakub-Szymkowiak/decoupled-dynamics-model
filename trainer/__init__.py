import sys

from random import randint
from types import SimpleNamespace as State
from typing import Callable, Dict, List, NamedTuple

import torch

from tqdm import tqdm

from configs.default_schedule import DEFAULT_STAGES
from configs.default_regularizers import DEFAULT_REGULARIZERS

from scene import DecoupledModel, Scene

from trainer.scheduler import OptimizationStageScheduler
from trainer.regularizers import RegularizerRegistry

from utils.general_utils import get_linear_noise_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.render_utils import mask_image


def get_empty_tensor_dict(device="cuda"):
    return {mode: torch.tensor([], device=device) for mode in ["static", "dynamic", "composed"]}

class OptimizationSpec(NamedTuple):
    iterations: int

    lambda_dssim: float
    lambdas_decoupled: Dict[str, float]
    
    load2gpu_on_the_fly: bool

    white_background: bool

    test_iterations: List[int]
    save_iterations: List[int]
    
    @classmethod
    def from_params(cls, model_cfg, opt_cfg, pipe_cfg, args):
        return cls(iterations          = opt_cfg.iterations,
                   lambda_dssim        = opt_cfg.lambda_dssim,
                   load2gpu_on_the_fly = model_cfg.load2gpu_on_the_fly,
                   white_background    = model_cfg.white_background,
                   test_iterations     = args.test_iterations,
                   save_iterations     = args.save_iterations,
                   lambdas_decoupled = {"static": opt_cfg.lambda_static,
                                        "dynamic": opt_cfg.lambda_dynamic,
                                        "composed": opt_cfg.lambda_composed})


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

        self.reg_registry = RegularizerRegistry(DEFAULT_REGULARIZERS)
        self.scheduler = OptimizationStageScheduler(DEFAULT_STAGES)
        self.smooth_term = get_linear_noise_func(lr_init=0.1, 
                                                 lr_final=1e-15, 
                                                 lr_delay_mult=0.01, 
                                                 max_steps=20000)            

    def _init_training_state(self):
        state = State()
        state.iter_start = torch.cuda.Event(enable_timing=True)
        state.iter_end = torch.cuda.Event(enable_timing=True)
        state.viewpoint_stack = None
        state.ema_loss = 0.0
        state.best_psnr = {mode: 0.0 for mode in ["static", "dynamic", "composed"]}
        state.best_iter = {mode: None for mode in ["static", "dynamic", "composed"]}
        state.time_interval = 1.0 / self.scene.num_frames
        state.progress_bar = tqdm(range(self.spec.iterations), desc="Training progress")

        return state

    def start_training(self):
        state = self._init_training_state()

        for iteration in range(1, self.spec.iterations + 1):
            state.iter_start.record()

            # Every 1k iterations increase SH degree until max
            if iteration % 1000 == 0:
                self.model.oneupSHdegree()

            if not state.viewpoint_stack:
                state.viewpoint_stack = self.scene.getCameras().copy()

            # Draw random camera
            random_index = randint(0, self.scene.num_frames - 1)
            viewpoint = state.viewpoint_stack[random_index]
            fid = viewpoint.fid

            if self.spec.load2gpu_on_the_fly:
                viewpoint.load2device()

            # Get training directive from scheduler
            directive = self.scheduler.directive_for(iteration)
            loss_dict = {}

            # Handle deform network
            if directive.train_deform:
                deltas = self.model.infer_deltas(fid, iteration, state.time_interval, self.smooth_term)
            else: 
                deltas = self.model.get_zero_deltas()

            # Render images and compute photometric losses
            losses, renders = {}, {}

            if directive.train_static:
                losses["static"], renders["static"], _ = self._render_and_compute_loss("static", deltas, viewpoint)
            if directive.train_dynamic:
                losses["dynamic"], renders["dynamic"], _ = self._render_and_compute_loss("dynamic", deltas, viewpoint)
            if directive.train_static and directive.train_dynamic:
                losses["composed"], renders["composed"], _ = self._render_and_compute_loss("composed", deltas, viewpoint)

            loss = sum(losses[_]["total"] for _ in losses)

            # Regularize 
            reg_loss, _ = self.reg_registry.compute(
                model=self.model,
                deltas=deltas["dynamic"],
                render=renders,
                viewpoint=viewpoint,
                directive=directive,
                cokolwiek=2
            )

            # DEBUG
            if False:
                if iteration % 100 == 0:
                    print(f"[DEBUG] Iter {iteration:05d} | Viewpoint FID {fid} | Losses:")
                    for k, val in losses.items():
                        print(f"  - {k:<8}: total={val['total']:.4f}, l1={val['l1']:.4f}, ssim={val['ssim']:.4f}")
                    print(f"  -> Reg Loss: {reg_loss.item():.4f} | Total Loss: {loss.item():.4f}")


            loss += reg_loss

            loss.backward()

            if self.spec.load2gpu_on_the_fly:
                viewpoint.load2device("cpu")

            state.iter_end.record()


            ### DEBUG - save gts and renders

            if False:
                with torch.no_grad():
                    from torchvision.utils import save_image
                    save_image(viewpoint.static_image, "gt_static.png")
                    save_image(viewpoint.dynamic_image, "gt_dynamic.png")

                    if directive.train_static:
                        save_image(renders["static"]["render"], "static.png")
                    if directive.train_dynamic:
                        save_image(renders["dynamic"]["render"], "dynamic.png")
                    if directive.train_static and directive.train_dynamic:
                        save_image(renders["composed"]["render"], "compsoed.png")

            ### END DEBUG

            with torch.no_grad():
                self._update_progress_bar(iteration, loss.item(), state)
                self._step_optimizers(iteration)

                # Max radii update
                if directive.train_static:
                    self._update_max_radii("static", renders["static"])
                if directive.train_dynamic:
                    self._update_max_radii("dynamic", renders["dynamic"])

                # Evaluation
                if iteration in self.spec.test_iterations:
                    torch.cuda.empty_cache()
                    current_psnrs = self._evaluate_psnr(iteration)
                    for mode in state.best_psnr:
                        psnr_score = current_psnrs[mode]
                        if psnr_score is not None and psnr_score.item() > state.best_psnr[mode]:
                            state.best_psnr[mode] = psnr_score.item()
                            state.best_iter[mode] = iteration
                    torch.cuda.empty_cache()

                # Saving
                if iteration in self.spec.save_iterations:
                    print(f"\n[ITER {iteration}] Saving Gaussians")
                    self.scene.save(iteration)

            

    def _render_and_compute_loss(self, mode: str, deltas, viewpoint):
        rendering_result = self.run_renderer(mode, deltas, viewpoint)
        rendered_image = rendering_result["render"]

        gt_image = viewpoint.static_image if mode == "static" else viewpoint.dynamic_image
        
        if mode == "dynamic":
            #rendered_image = mask_image(rendered_image, viewpoint.dmask)
            gt_image = mask_image(gt_image, viewpoint.dmask)

        l1_val = l1_loss(rendered_image, gt_image)
        ssim_val = ssim(rendered_image, gt_image)

        total_loss = (1.0 - self.spec.lambda_dssim) * l1_val + self.spec.lambda_dssim * (1.0 - ssim_val)
        total_loss *= self.spec.lambdas_decoupled[mode]

        losses = {"total": total_loss, "l1": l1_val, "ssim": ssim_val}

        return losses, rendering_result, gt_image

    def _update_progress_bar(self, iteration: int, current_loss: float, state: State):
        state.ema_loss = 0.4 * current_loss + 0.6 * state.ema_loss

        if iteration % 10 == 0:
            state.progress_bar.set_postfix({"Loss": f"{state.ema_loss:.6f}"})
            state.progress_bar.update(10)

        if iteration == self.spec.iterations - 1:
            state.progress_bar.close()

    def _step_optimizers(self, iteration: int):
        self.model.static.optimizer.step()
        self.model.dynamic.optimizer.step()
        self.model.deform.optimizer.step()

        self.model.static.optimizer.zero_grad(set_to_none=True)
        self.model.dynamic.optimizer.zero_grad(set_to_none=True)
        self.model.deform.optimizer.zero_grad(set_to_none=True)

        self.model.static.update_learning_rate(iteration)
        self.model.dynamic.update_learning_rate(iteration)
        self.model.deform.update_learning_rate(iteration)

    def _update_max_radii(self, mode: str, render_output: dict):
        vis_filter = render_output["visibility_filter"]
        radii = render_output["radii"]

        if vis_filter is None or radii is None:
            return

        model_attr = getattr(self.model, mode)
        current_max = model_attr.max_radii2D

        current_max[vis_filter] = torch.max(current_max[vis_filter], radii[vis_filter])

    def _evaluate_psnr(self, iteration: int): 
        torch.cuda.empty_cache()
        viewpoint_stack = self.scene.getCameras()
        time_interval = 1.0 / self.scene.num_frames
        
        gts, renders = get_empty_tensor_dict(), get_empty_tensor_dict()

        for viewpoint in viewpoint_stack:
            if self.spec.load2gpu_on_the_fly:
                viewpoint.load2device()

            fid = viewpoint.fid
            deltas = self.model.infer_deltas(fid, iteration, time_interval, noise=False)

            for mode in gts.keys():
                _, rendering_result, gt_image = self._render_and_compute_loss(mode, deltas, viewpoint)
                
                gt_image = torch.clamp(gt_image, 0.0, 1.0)
                rendered_image = torch.clamp(rendering_result["render"], 0.0, 1.0)

                gts[mode] = torch.cat((gts[mode], gt_image.unsqueeze(0)), dim=0)
                renders[mode] = torch.cat((renders[mode], rendered_image.unsqueeze(0)), dim=0)

            if self.spec.load2gpu_on_the_fly:
                viewpoint.load2device("cpu")

            torch.cuda.empty_cache()

        psnrs = {mode: psnr(renders[mode], gts[mode]).mean() for mode in gts.keys()}

        print(f"\n[ITER {iteration}] Evaluation (PSNR)")
        for mode in ["static", "dynamic", "composed"]:
            print(f" — {mode.capitalize():<8}: {psnrs[mode]:.2f} dB")
        sys.stdout.flush()

        return psnrs