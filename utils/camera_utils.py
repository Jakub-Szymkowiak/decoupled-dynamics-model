import numpy as np
import torch

from scene.cameras import Camera


WARNED = False


def loadCam(args, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.static_image.shape[:2]

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    static_image = torch.from_numpy(cam_info.static_image.copy()).permute(2, 0, 1) / 255.0
    static_depth = torch.from_numpy(cam_info.static_depth.copy())
    
    dynamic_image = torch.from_numpy(cam_info.dynamic_image.copy()).permute(2, 0, 1) / 255.0
    dynamic_depth = torch.from_numpy(cam_info.dynamic_depth.copy())
    
    dmask = torch.from_numpy(cam_info.dmask.copy())
    flow = torch.from_numpy(cam_info.flow.copy()) if cam_info.flow is not None else None

    return Camera(uid=cam_info.uid, fid=cam_info.fid,
                  R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  static_image=static_image, static_depth=static_depth,
                  dynamic_image=dynamic_image, dynamic_depth=dynamic_depth,
                  dmask=dmask, flow=flow,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else "cpu",
                  pose=cam_info.pose, intrinsics=cam_info.intrinsics)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    return [loadCam(args, c, resolution_scale) for c in cam_infos]