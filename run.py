#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch

torch.backends.cuda.preferred_linalg_library()

import sys
import termios
import tty
from argparse import ArgumentParser

import rerun as rr  # pip install rerun-sdk
from pytorch3d.renderer import PerspectiveCameras

from metric_depth import img_to_pts_3d_da
from metric_dust import img_to_pts_3d_dust
from misc.imutils import fill
from misc.inpaint import run_inpaint
from misc.merge import merge_and_filter
from misc.prune import density_pruning_py3d
from misc.renderer import pts_3d_to_img_py3d, pts_3d_to_img_raster
from misc.scale import project_and_scale_points, pts_cam_to_pts_world
from misc.supersample import run_supersample


def get_keypress():
    fd = sys.stdin.fileno()
    original_attributes = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)  # Read a single character
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_attributes)
    return key


def move_camera(extrinsics, direction, amount):
    """
    Move the camera considering its orientation or rotate left/right.
    
    :param extrinsics: The current extrinsics matrix.
    :param direction: 'w' (forward), 's' (backward), 'a' (left rotate), 'd' (right rotate).
    :param amount: The amount to move or rotate.
    """
    # Extract rotation matrix R and translation vector t from the extrinsics matrix
    R = extrinsics[:3, :3]
    
    if direction in ['w', 's']:
        # Direction vector for forward/backward 
        amount = 75.0 * (-amount if direction == 'w' else amount)
        extrinsics[:3, 3] += torch.tensor([0, 0, amount], device=extrinsics.device).float()
    elif direction in ['q', 'e']:
        # Direction vector for up/down 
        amount = 75.0 * (-amount if direction == 'q' else amount)
        extrinsics[:3, 3] += torch.tensor([0, amount, 0], device=extrinsics.device).float()
    elif direction in ['a', 'd']:
        # Rotation angle (in radians). Positive for 'd' (right), negative for 'a' (left)
        angle = torch.tensor(-amount if direction == 'd' else amount)
        # Rotation matrix around the Y-axis (assuming Y is up)
        rotation_matrix = torch.tensor([
            [torch.cos(angle), 0, torch.sin(angle), 0],
            [0, 1, 0, 0],
            [-torch.sin(angle), 0, torch.cos(angle), 0],
            [0, 0, 0, 1]
        ], device=extrinsics.device)
        # Apply rotation to the extrinsics matrix
        extrinsics = torch.matmul(rotation_matrix, extrinsics)
    return extrinsics


def main():
    # --- setup rerun args ---
    parser = ArgumentParser(description="Build your own adventure.")
    rr.script_add_args(parser)
    parser.add_argument('--depth', type=str, default='dust', help='da / dust')
    parser.add_argument('--renderer', type=str, default='py3d', help='raster / py3d')
    args = parser.parse_args()
    rr.script_setup(args, "15myst")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    img_to_pts_3d = img_to_pts_3d_da if args.depth == 'da' else img_to_pts_3d_dust
    pts_3d_to_img = pts_3d_to_img_raster if args.renderer == 'raster' else pts_3d_to_img_py3d 

    pts_3d = None
    image = None
    extrinsics = None
    all_images = []
    intrinsics = torch.tensor([[256.0*1.0, 0.0000, 256.0000],
                               [0.0000, 256.0*1.0, 256.0000],
                               [0.0000, 0.000, 1.0000]]).cuda()
    imsize = 512.
    idx = 0
    while True:
        idx += 1
        # --- setup initial scene ---
        if image is None: 
            #prompt = input(f"enter stable diffusion initial scene: ")
            prompt = 'a high-resolution photo of a large kitchen.'
            image = run_inpaint(torch.zeros(512, 512, 3), torch.ones(512, 512), prompt=prompt)
            mask = mask_3d = torch.ones(512, 512)

            all_images.insert(0, image)
        else:
            image = wombo_img.to(torch.uint8)
            mask = image.sum(dim=2) < 10
            image[mask] = 0.0

        # --- establish orientation ---
        if extrinsics == None:
            extrinsics = torch.tensor([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]).float().cuda()

        # --- estimate depth ---
        if pts_3d is None: 
            pts_3d, rgb_3d, focals = img_to_pts_3d_dust(all_images)
            pts_3d = pts_cam_to_pts_world(pts_3d, extrinsics)
            pts_3d, rgb_3d = density_pruning_py3d(pts_3d, rgb_3d)

            if focals is not None:
                intrinsics[0, 0] = focals
                intrinsics[1, 1] = focals
                print(intrinsics)

                if args.renderer != 'raster':
                    cameras = PerspectiveCameras(
                        device='cuda',
                        R=torch.eye(3).unsqueeze(0),
                        in_ndc=False,
                        T=torch.zeros(1, 3),
                        focal_length=-intrinsics[0,0].unsqueeze(0),
                        principal_point=intrinsics[:2,2].unsqueeze(0),
                        image_size=torch.ones(1, 2) * imsize,
                    )

        # --- rerun logging --- 
        rr.set_time_sequence("frame", idx+1)
        rr.log(f"world/points", rr.Points3D(pts_3d.cpu().numpy(), colors=rgb_3d.cpu().numpy()), timeless=True)
        rr.log("world/camera", 
            rr.Transform3D(translation=extrinsics[:3, 3].cpu().numpy(),
                           mat3x3=extrinsics[:3, :3].cpu().numpy(), from_parent=True))
        inpy = intrinsics.cpu().numpy()
        rr.log("world/camera/image", rr.Pinhole(resolution=[imsize, imsize], focal_length=[inpy[0,0], inpy[1,1]], principal_point=[inpy[0,-1], inpy[1,-1]]))
        rr.log("world/camera/image", rr.Image(image.cpu().numpy()).compress(jpeg_quality=75))
        rr.log("world/camera/mask", rr.Pinhole(resolution=[imsize, imsize], focal_length=[inpy[0,0], inpy[1,1]], principal_point=[inpy[0,-1], inpy[1,-1]]))
        rr.log("world/camera/mask", rr.Image((torch.stack([mask_3d, mask_3d, mask_3d], dim=2).float() * 255.0).to(torch.uint8).cpu().numpy()).compress(jpeg_quality=100))

        # --- get user input ---
        infill = False
        inpaint = False
        print("press (w, a, s, d, q, e) move, (f)ill, (u)psample, (k)ill, (b)reakpoint, or (t)ext for stable diffusion...")
        user_input = get_keypress()
        if user_input.lower() in ['w', 'a', 's', 'd', 'q', 'e']:
            extrinsics = move_camera(extrinsics, user_input.lower(), 0.1)  # Assuming an amount of 0.1 for movement/rotation
            print(f"{user_input} --> camera moved/rotated, extrinsics:\n", extrinsics)
        elif user_input.lower() == 'f':
            print(f"{user_input} --> fill...")
            #wombo_img = fill(image_t)      # blur points to make a smooth image
            #infill = True
            inpaint = True
        elif user_input.lower() == 'u':
            print(f"{user_input} --> upsample...")
            breakpoint()
            pts_3d = supersample_point_cloud(pts_3d.to(device))
            image = run_supersample(image, mask, prompt)
            rgb_3d = torch.tensor(np.asarray(color_image[0])).float().to(device)
            rgb_3d = supersample_point_cloud(rgb_3d).to(torch.uint8)
        elif user_input.lower() == 'k':
            print(f"{user_input} --> kill...")
            break
        elif user_input.lower() == 'b':
            print(f"{user_input} --> breakpoint...")
            breakpoint()
        else:
            prompt = input(f"{user_input} --> enter stable diffusion prompt: ")
            inpaint = True

        # --- turn 3d points to image ---
        wombo_img = pts_3d_to_img(pts_3d, rgb_3d, intrinsics, extrinsics, (imsize, imsize), cameras)

        # --- sideways pipeline ---
        if inpaint: 
            wombo_img = fill(wombo_img)      # blur points to make a smooth image
            mask = wombo_img.sum(dim=2) < 10
            wombo_img[mask] = -1.0
            sq_init = run_inpaint(wombo_img, mask.float(), prompt=prompt)
            wombo_img = wombo_img.to(torch.uint8)
            wombo_img[mask] = sq_init[mask]
            all_images.insert(0, sq_init)

        if inpaint or infill:
            # --- lift img to 3d ---
            new_pts_3d, new_rgb_3d, _ = img_to_pts_3d_dust(all_images)
            new_pts_3d, new_rgb_3d = density_pruning_py3d(new_pts_3d, new_rgb_3d)
            new_pts_3d = pts_cam_to_pts_world(new_pts_3d, extrinsics)

            # --- re-aligns two point clouds with partial overlap ---
            _, new_pts_3d, new_rgb_3d, mask_3d = project_and_scale_points(pts_3d, new_pts_3d, rgb_3d, new_rgb_3d, intrinsics, extrinsics, 
                                                                          image_shape=(imsize, imsize), 
                                                                          color_threshold=30, 
                                                                          align_mode='o3d')

            # --- merge and filtering new point cloud ---
            pts_3d, rgb_3d = merge_and_filter(pts_3d, new_pts_3d, rgb_3d, new_rgb_3d)
    rr.script_teardown(args)

if __name__ == "__main__":
    # --- procedure
    # 1. load an input image
    # 2. estimate depth
    # 3. use dust3r to lift to 3D point cloud 
    # 4. move around and apply delta extrinsics
    # 5. re-render new image with mask
    # 6. in paint black with diffusion

    import cProfile
    import io
    import pstats
    with torch.no_grad():
        #with torch.autocast(device_type="cuda"):
        profiler = cProfile.Profile()
        profiler.enable()
        main()
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumtime')

        # Now we filter and print entries with a cumulative time greater than 0.5 seconds
        for func in ps.fcn_list:
            if ps.stats[func][3] > 0.5:  # index 3 is where 'cumulative time' is stored
                print(f'{func[2]} took {ps.stats[func][3]:.2f} seconds')
        #breakpoint()
