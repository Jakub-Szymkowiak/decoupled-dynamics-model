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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


from scene.decoupled_model import DecoupledModel

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    #tb_writer = prepare_output_and_logger(dataset)

    model = DecoupledModel(dataset.sh_degree, dataset.is_blender, dataset.is_6dof)
    model.training_setup(opt)

    scene = Scene(dataset, model)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack_static = None
    viewpoint_stack_dynamic = None


    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            model.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack_static:
            viewpoint_stack_static = scene.getStaticCameras().copy()

        if not viewpoint_stack_dynamic:
            viewpoint_stack_dynamic = scene.getDynamicCameras().copy()

        total_frame = len(viewpoint_stack_static) # == len(viewpoint_stack_dynamic)
        time_interval = 1 / total_frame

        random_idx = randint(0, total_frame - 1)
        viewpoint_cam_static = viewpoint_stack_static.pop(random_idx)
        viewpoint_cam_dynamic = viewpoint_stack_dynamic.pop(random_idx)

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam_static.load2device()
            viewpoint_cam_dynamic.load2device()

        fid = torch.tensor(viewpoint_cam_dynamic.fid, device="cuda")

        # Deform model inference
        if iteration < 0: # < opt.warm_up:
            static_deltas = (0.0, 0.0, 0.0)
            dynamic_deltas = (0.0, 0.0, 0.0)
            composed_deltas = (0.0, 0.0, 0.0)
        else:
            deltas = model.infer_deltas(fid, iteration, time_interval, smooth_term)
            static_deltas, dynamic_deltas, composed_deltas = deltas

        # Loss computation
        loss = torch.tensor(0.0, device="cuda")

        gt_image_static = viewpoint_cam_static.original_image.cuda()
        gt_image = viewpoint_cam_dynamic.original_image.cuda()

        # 1. Render static background
        render_pkg_re_static = render(viewpoint_cam_static, model.static, pipe, background, 
                                      static_deltas[0], static_deltas[1], static_deltas[2], 
                                      dataset.is_6dof)

        static_render = render_pkg_re_static["render"]
        Ll1_static = l1_loss(static_render, gt_image_static)

        ssim_static = ssim(static_render, gt_image)

        loss += (1.0 - opt.lambda_dssim) * Ll1_static + opt.lambda_dssim * (1.0 - ssim_static)

        # 2. Render dynamic object
        render_pkg_re_dynamic = render(viewpoint_cam_dynamic, model.dynamic, pipe, background, 
                                       dynamic_deltas[0], dynamic_deltas[1], dynamic_deltas[2], 
                                       dataset.is_6dof)

        dynamic_render = render_pkg_re_dynamic["render"]
        Ll1_dynamic = l1_loss(dynamic_render, gt_image)
        ssim_dynamic = ssim(dynamic_render, gt_image)

        loss += (1.0 - opt.lambda_dssim) * Ll1_dynamic + opt.lambda_dssim * (1.0 - ssim_dynamic)

        # 3. Render composed 
        render_pkg_re_composed = render(viewpoint_cam_dynamic, model, pipe, background,
                                        composed_deltas[0], composed_deltas[1], composed_deltas[2],
                                        dataset.is_6dof)


        composed_render = render_pkg_re_composed["render"]
        Ll1_composed = l1_loss(composed_render, gt_image)
        ssim_composed = ssim(composed_render, gt_image)

        loss += (1.0 - opt.lambda_dssim) * Ll1_composed + opt.lambda_dssim * (1.0 - ssim_composed)

        loss.backward()

        # DEBUG

        from torchvision.utils import save_image
        save_image(composed_render, "comp.png")
        save_image(static_render, "stat.png")
        save_image(dynamic_render, "dyna.png")
        save_image(gt_image, "gtimage.png")


        if dataset.load2gpu_on_the_fly:
            viewpoint_cam_static.load2device("cpu")
            viewpoint_cam_dynamic.load2device("cpu")

        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            vis_filter_static = render_pkg_re_static["visibility_filter"]
            radii_static = render_pkg_re_static["radii"]
            model.static.max_radii2D[vis_filter_static] = torch.max(model.static.max_radii2D[vis_filter_static], radii_static[vis_filter_static])

            vis_filter_dynamic = render_pkg_re_dynamic["visibility_filter"]
            radii_dynamic = render_pkg_re_dynamic["radii"]
            model.dynamic.max_radii2D[vis_filter_dynamic] = torch.max(model.dynamic.max_radii2D[vis_filter_dynamic], radii_dynamic[vis_filter_dynamic])

            # TODO - implement saving logic
            
            # Log and save
            # cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), deform, dataset.load2gpu_on_the_fly, dataset.is_6dof)
            # if iteration in testing_iterations:
            #     if cur_psnr.item() > best_psnr:
            #         best_psnr = cur_psnr.item()
            #         best_iteration = iteration

            # if iteration in saving_iterations:
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)
            #     deform.save_weights(args.model_path, iteration)
            
            # TODO - implement densification logic

            # Densification
            # if iteration < opt.densify_until_iter:
            #     viewspace_point_tensor_densify = render_pkg_re["viewspace_points_densify"]
            #     gaussians.add_densification_stats(viewspace_point_tensor_densify, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

            #     if iteration % opt.opacity_reset_interval == 0 or (
            #             dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                model.static.optimizer.step()
                model.dynamic.optimizer.step()
                model.deform.optimizer.step()

                model.static.optimizer.zero_grad(set_to_none=True)
                model.dynamic.optimizer.zero_grad(set_to_none=True)
                model.deform.optimizer.zero_grad()

                model.static.update_learning_rate(iteration)
                model.dynamic.update_learning_rate(iteration)
                model.deform.update_learning_rate(iteration)

                

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    test_every = 2500
    test_it_def = list(range(test_every, 30_000, test_every))
    parser.add_argument("--test_iterations", nargs="+", type=int, default=test_it_def)

    save_it_def = [1, 1_000, 7_000, 15_000, 30_000]
    parser.add_argument("--save_iterations", nargs="+", type=int, default=save_it_def)

    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)

    # All done
    print("\nTraining complete.")