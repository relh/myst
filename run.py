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
from metric_depth import img_to_pts_3d, pts_3d_to_img
from misc.colab import run_inpainting_pipeline
from misc.control import generate_outpainted_image
from misc.outpainting import run
from misc.replicate_me import run_replicate_with_pil
from utils import *

#from transformers import pipeline


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
    t = extrinsics[:3, 3]
    
    if direction in ['w', 's']:
        # Direction vector for forward/backward (assuming the camera looks towards the positive Z in its local space)
        # Negative for 's' (backward)
        forward_vec = torch.tensor([0, 0, -amount if direction == 'w' else amount], device=extrinsics.device).float()
        
        # Transform the forward vector by the camera's rotation to get the world space direction
        # Note: No need to change the direction based on 'w' or 's' outside of forward_vec calculation
        world_direction = torch.matmul(R, forward_vec.unsqueeze(-1)).squeeze()

        # Update the translation component of the extrinsics matrix by adding the world space direction
        extrinsics[:3, 3] += world_direction
    elif direction in ['a', 'd']:
        # Rotation angle (in radians). Positive for 'd' (right), negative for 'a' (left)
        angle = torch.tensor(-amount if direction == 'd' else amount)
        # Rotation matrix around the Y-axis (assuming Y is up)
        rotation_matrix = torch.tensor([
            [torch.cos(angle), 0, torch.sin(angle), 0],
            [0, 1, 0, 0],
            [-torch.sin(angle), 0, torch.cos(angle), 0],
            [0, 0, 0, 1]
        ])
        # Apply rotation to the extrinsics matrix
        extrinsics = torch.matmul(rotation_matrix, extrinsics)
    
    return extrinsics


def main():
    # --- setup rerun args ---
    parser = ArgumentParser(description="Build your own adventure.")
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "7myst")

    # --- initial logging ---
    rr.log("description", rr.TextDocument('tada',  media_type=rr.MediaType.MARKDOWN), timeless=True)
    rr.log("world", rr.ViewCoordinates.LEFT_HAND_Y_DOWN, timeless=True)

    # Iterate through images (video frames) logging data related to each frame.
    image = None
    extrinsics = None
    visualization = False
    intrinsics = torch.tensor([[256.0*1.0, 0.0000, 256.0000],
                               [0.0000, 256.0*1.0, 256.0000],
                               [0.0000, 0.000, 1.0000]]).cuda()
    idx = 0
    while True:
        idx += 1
        if idx % 15 == 0: print(f'{idx}')

        # --- setup initial scene ---
        if image is None: 
            #user_input = input("Describe initial scene: ")
            user_input = "A photo of a kitchen"
            image = run_inpainting_pipeline(torch.zeros(512, 512, 3), torch.ones(512, 512), strength=0.89, prompt=user_input)
        else:
            image = wombo_img.to(torch.uint8)

        # --- establish orientation ---
        if extrinsics == None:
            extrinsics = torch.tensor([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]).float().cuda()

        # --- estimate depth ---
        pil_img = Image.fromarray(image.cpu().numpy())
        da_3d, da_colors = img_to_pts_3d(pil_img, extrinsics)

        # --- rerun logging --- 
        rr.set_time_sequence("frame", idx+1)
        #rr.log("camera", rr.ViewCoordinates.LUF, timeless=True)  # X=Right, Y=Down, Z=Forward
        rr.log("world/camera", 
            rr.Transform3D(translation=extrinsics[:3, 3].cpu().numpy(),
                           mat3x3=torch.linalg.inv(extrinsics)[:3, :3].cpu().numpy()))
        rr.log("world/camera/image",
            rr.Pinhole(
                resolution=[512., 512.],
                focal_length=[256., 256.],
                principal_point=[256., 256.],
            ),
        )
        rr.log("world/camera/image", rr.Image(image.cpu().numpy()).compress(jpeg_quality=75))
        rr.log("world/points", rr.Points3D(da_3d.cpu().numpy(), colors=da_colors.cpu().numpy()), timeless=True)

        inpaint = False
        print("Press any key...")
        user_input = get_keypress()
        print(f"You pressed: {user_input}\n")

        if user_input.lower() in ['w', 'a', 's', 'd']:
            extrinsics = move_camera(extrinsics, user_input.lower(), 0.1)  # Assuming an amount of 0.1 for movement/rotation
            print("Camera moved/rotated. New extrinsics matrix:\n", extrinsics)
        elif user_input.lower() == 'q':
            print("Exiting...")
            break
        elif user_input.lower() == 'b':
            print("Breakpoint...")
            breakpoint()
        else:
            # Treat the input as a text prompt for Stable Diffusion
            user_input = input("Enter scene description for stable diffusion: ")
            inpaint = True

        # --- turn 3d points to image ---
        proj_da, _, _, vis_da_3d, _, vis_da_colors, _ = pts_3d_to_img(da_3d, da_colors, intrinsics, extrinsics, (512, 512))
        image_t = torch.zeros((512, 512, 3), dtype=torch.uint8).cuda()
        proj_da = proj_da.long()
        proj_da[:, 0] = proj_da[:, 0].clamp(0, 512 - 1)
        proj_da[:, 1] = proj_da[:, 1].clamp(0, 512 - 1)
        image_t[proj_da[:, 1], proj_da[:, 0]] = vis_da_colors

        # --- despeckle pipeline ---
        wombo_img = mod_fill(image_t)

        # --- sideways pipeline ---
        if inpaint: 
            wombo_mask = wombo_img.sum(dim=2) < 10
            wombo_img[wombo_mask] = -1.0
            sq_init = run_inpainting_pipeline(wombo_img, wombo_mask.float(), strength=0.89, prompt=user_input)
            wombo_img = wombo_img.to(torch.uint8)
            wombo_img[wombo_mask] = sq_init[wombo_mask]

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
