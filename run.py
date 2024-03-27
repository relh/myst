#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch

torch.backends.cuda.preferred_linalg_library()

import os
import re
import sys
import termios
import tty
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import replicate
import rerun as rr  # pip install rerun-sdk
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline
from PIL import Image
from pytorch3d.transforms import matrix_to_quaternion
from torchvision.transforms import ToPILImage, ToTensor

from ek_fields_utils.colmap_rw_utils import read_model
from merge import *
from metric_depth import img_to_pts_3d_da, pts_3d_to_img, pts_cam_to_pts_world
from metric_dust import img_to_pts_3d_dust
from misc.colab import run_inpaint
from misc.control import generate_outpainted_image
from misc.outpainting import run
from misc.replicate_me import run_replicate_with_pil
from utils import *


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
        # Direction vector for forward/backward (assuming the camera looks towards the positive Z in its local space)
        amount = 40.0 * (-amount if direction == 'w' else amount)
        extrinsics[:3, 3] += torch.tensor([0, 0, amount], device=extrinsics.device).float()
    elif direction in ['q', 'e']:
        # Direction vector for forward/backward (assuming the camera looks towards the positive Z in its local space)
        amount = 40.0 * (-amount if direction == 'q' else amount)
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
    args = parser.parse_args()
    rr.script_setup(args, "13myst")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    img_to_pts_3d = img_to_pts_3d_da if args.depth == 'da' else img_to_pts_3d_dust
    depth_3d = None
    image = None
    mask = None
    extrinsics = None
    visualization = False
    intrinsics = torch.tensor([[256.0*1.0, 0.0000, 256.0000],
                               [0.0000, 256.0*1.0, 256.0000],
                               [0.0000, 0.000, 1.0000]]).cuda()
    idx = 0
    while True:
        idx += 1

        # --- setup initial scene ---
        if image is None: 
            #prompt = input(f"enter stable diffusion initial scene: ")
            prompt = 'a high-resolution photo of a large kitchen'
            image = run_inpaint(torch.zeros(512, 512, 3), torch.ones(512, 512), prompt=prompt)
            mask = torch.ones(512, 512)
        else:
            image = wombo_img.to(torch.uint8)
            mask = image.sum(dim=2) < 10

        # --- establish orientation ---
        if extrinsics == None:
            extrinsics = torch.tensor([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]).float().cuda()

        # --- estimate depth ---
        if depth_3d is None: 
            pil_img = Image.fromarray(image.cpu().numpy())
            depth_3d, depth_colors, focals = img_to_pts_3d(pil_img)
            depth_3d = pts_cam_to_pts_world(depth_3d, extrinsics)

            if focals is not None:
                intrinsics[0, 0] = focals
                intrinsics[1, 1] = focals

        # --- rerun logging --- 
        rr.set_time_sequence("frame", idx+1)
        rr.log(f"world/points", rr.Points3D(depth_3d.cpu().numpy(), colors=depth_colors.cpu().numpy()), timeless=True)
        rr.log("world/camera", 
            rr.Transform3D(translation=extrinsics[:3, 3].cpu().numpy(),
                           mat3x3=extrinsics[:3, :3].cpu().numpy(), from_parent=True))
        rr.log("world/camera/image",rr.Pinhole(resolution=[512., 512.], focal_length=[256., 256.], principal_point=[256., 256.]))
        rr.log("world/camera/image", rr.Image(image.cpu().numpy()).compress(jpeg_quality=75))
        rr.log("world/camera/mask", rr.Pinhole(resolution=[512., 512.], focal_length=[256., 256.], principal_point=[256., 256.]))
        rr.log("world/camera/mask", rr.Image((torch.stack([mask, mask, mask], dim=2).float() * 255.0).to(torch.uint8).cpu().numpy()).compress(jpeg_quality=100))

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
        proj_da, vis_depth_3d, vis_depth_colors = pts_3d_to_img(depth_3d, depth_colors, intrinsics, extrinsics, (512, 512))
        image_t = torch.zeros((512, 512, 3), dtype=torch.uint8).cuda()
        proj_da = proj_da.long()
        proj_da[:, 0] = proj_da[:, 0].clamp(0, 512 - 1)
        proj_da[:, 1] = proj_da[:, 1].clamp(0, 512 - 1)
        #image_t[proj_da[:, 1], proj_da[:, 0]] = vis_depth_colors
        image_t[proj_da[:, 1], proj_da[:, 0]] = (vis_depth_colors * 1.0).to(torch.uint8)
        wombo_img = image_t.clone().float() # only use existing points

        # --- sideways pipeline ---
        if inpaint: 
            wombo_img = fill(image_t)      # blur points to make a smooth image
            mask = wombo_img.sum(dim=2) < 10
            wombo_img[mask] = -1.0
            sq_init = run_inpaint(wombo_img, mask.float(), prompt=prompt)
            wombo_img = wombo_img.to(torch.uint8)
            wombo_img[mask] = sq_init[mask]

        if inpaint or infill:
            pil_img = Image.fromarray(wombo_img.to(torch.uint8).cpu().numpy())
            new_depth_3d, new_depth_colors, _ = img_to_pts_3d(pil_img)
            new_depth_3d = pts_cam_to_pts_world(new_depth_3d, extrinsics)
            new_depth_3d, new_depth_colors = trim_points(new_depth_3d, new_depth_colors, border=32)
            depth_3d, depth_colors = merge_and_filter(depth_3d, new_depth_3d, depth_colors, new_depth_colors)

    rr.script_teardown(args)

if __name__ == "__main__":
    # load an input image
    # estimate depth
    # use COLMAP to DA scale factor
    # lift to 3D point cloud
    # apply delta extrinsics from EF using nearest neighbors
    # re-render new image with mask
    # in paint black with diffusion
    with torch.no_grad():
        #with torch.autocast(device_type="cuda"):
        main()
