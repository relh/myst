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
import torch
from PIL import Image

from metric_depth import img_to_pts_3d_da, pts_cam_to_pts_world
from metric_dust import img_to_pts_3d_dust
from misc.inpaint import run_inpaint
from misc.merge import *
from misc.prune import *
from misc.renderer import *
from misc.scale import *
from misc.utils import *


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
    mask = None
    extrinsics = None
    all_images = []
    intrinsics = torch.tensor([[256.0*1.0, 0.0000, 256.0000],
                               [0.0000, 256.0*1.0, 256.0000],
                               [0.0000, 0.000, 1.0000]]).cuda()
    idx = 0
    while True:
        idx += 1

        # --- setup initial scene ---
        if image is None: 
            #prompt = input(f"enter stable diffusion initial scene: ")
            prompt = 'a high-resolution photo of a large kitchen.'
            image = run_inpaint(torch.zeros(512, 512, 3), torch.ones(512, 512), prompt=prompt)
            mask = torch.ones(512, 512)

            all_images.insert(0, image)
        else:
            image = torch.tensor(wombo_img).to(torch.uint8)
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
            #pts_3d, mask_3d = realign_depth_edges(pts_3d, rgb_3d)
            mask_3d = mask
            pts_3d = pts_cam_to_pts_world(pts_3d, extrinsics)
            pts_3d, rgb_3d = density_pruning(pts_3d, rgb_3d)

            #da_3d, da_colors, _ = img_to_pts_3d_da(pil_img)
            #da_3d = pts_cam_to_pts_world(da_3d, extrinsics)
            #_, pts_3d = project_and_scale_points_with_color(da_3d, pts_3d, da_colors, rgb_3d, intrinsics, extrinsics, image_shape=(512, 512))

            if focals is not None:
                intrinsics[0, 0] = focals
                intrinsics[1, 1] = focals
                print(intrinsics)

        # --- rerun logging --- 
        rr.set_time_sequence("frame", idx+1)
        rr.log(f"world/points", rr.Points3D(pts_3d.cpu().numpy(), colors=rgb_3d.cpu().numpy()), timeless=True)
        rr.log("world/camera", 
            rr.Transform3D(translation=extrinsics[:3, 3].cpu().numpy(),
                           mat3x3=extrinsics[:3, :3].cpu().numpy(), from_parent=True))
        inpy = intrinsics.cpu().numpy()
        rr.log("world/camera/image", rr.Pinhole(resolution=[512., 512.], focal_length=[inpy[0,0], inpy[1,1]], principal_point=[inpy[0,-1], inpy[1,-1]]))
        rr.log("world/camera/image", rr.Image(image.cpu().numpy()).compress(jpeg_quality=75))
        rr.log("world/camera/mask", rr.Pinhole(resolution=[512., 512.], focal_length=[inpy[0,0], inpy[1,1]], principal_point=[inpy[0,-1], inpy[1,-1]]))
        rr.log("world/camera/mask", rr.Image((torch.stack([mask_3d, mask_3d, mask_3d], dim=2).float() * 255.0).to(torch.uint8).cpu().numpy()).compress(jpeg_quality=100))

        # --- get user input ---
        infill = False
        inpaint = False
        print("press (w, a, s, d, q, e) move, (f)ill, (k)ill, (b)reakpoint, or (t)ext for stable diffusion...")
        user_input = get_keypress()
        if user_input.lower() in ['w', 'a', 's', 'd', 'q', 'e']:
            extrinsics = move_camera(extrinsics, user_input.lower(), 0.1)  # Assuming an amount of 0.1 for movement/rotation
            print(f"{user_input} --> camera moved/rotated, extrinsics:\n", extrinsics)
        elif user_input.lower() == 'f':
            print(f"{user_input} --> fill...")
            #wombo_img = fill(image_t)      # blur points to make a smooth image
            #infill = True
            inpaint = True
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
        wombo_img = pts_3d_to_img(pts_3d, rgb_3d, intrinsics, extrinsics, (512, 512))

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
            #pil_img = Image.fromarray(wombo_img.to(torch.uint8).cpu().numpy())

            # --- lift img to 3d ---
            new_pts_3d, new_rgb_3d, _ = img_to_pts_3d_dust(all_images)
            #new_pts_3d, new_rgb_3d = density_pruning(new_pts_3d, new_rgb_3d)
            #new_pts_3d, mask_3d = realign_depth_edges(new_pts_3d, new_rgb_3d)
            #extrinsics_inv = torch.linalg.pinv(extrinsics)
            new_pts_3d = pts_cam_to_pts_world(new_pts_3d, extrinsics)
            #new_da_3d, new_da_colors, _ = img_to_pts_3d_da(pil_img)
            #new_da_3d = pts_cam_to_pts_world(new_da_3d, extrinsics)

            # --- re-aligns two point clouds with partial overlap ---
            #_, new_pts_3d = project_and_scale_points_with_color(new_da_3d, new_pts_3d, new_da_colors, new_rgb_3d, intrinsics, extrinsics, image_shape=(512, 512))
            _, new_pts_3d, new_rgb_3d, mask_3d = project_and_scale_points_with_color(pts_3d, new_pts_3d, rgb_3d, new_rgb_3d, intrinsics, extrinsics, image_shape=(512, 512))

            # --- merge and filtering new point cloud ---
            #new_pts_3d, new_rgb_3d = trim_points(new_pts_3d, new_rgb_3d, border=32)
            #new_pts_3d, new_rgb_3d = prune_based_on_viewpoint(new_pts_3d, new_rgb_3d, intrinsics, extrinsics, image_shape=(512, 512), k=16, density_threshold=0.5)
            # --- setting epsilon explicitly to avoid square image dependency
            pts_3d, rgb_3d = merge_and_filter(pts_3d, new_pts_3d, rgb_3d, new_rgb_3d)
            #pts_3d = torch.cat((pts_3d, new_pts_3d), dim=0)
            #rgb_3d = torch.cat((rgb_3d, new_rgb_3d), dim=0)

            # --- density pruning ---
            pts_3d, rgb_3d = density_pruning(pts_3d, rgb_3d)
    rr.script_teardown(args)

if __name__ == "__main__":
    # load an input image
    # estimate depth
    # use COLMAP to DA scale factor
    # lift to 3D point cloud
    # apply delta extrinsics from EF using nearest neighbors
    # re-render new image with mask
    # in paint black with diffusion
    #with torch.no_grad():
    #with torch.autocast(device_type="cuda"):
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    breakpoint()
