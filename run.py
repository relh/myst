#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch

torch.backends.cuda.preferred_linalg_library()

import os
import re
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
from transformers import pipeline

from ek_fields_utils.colmap_rw_utils import read_model
from misc.colab import run_inpainting_pipeline
from misc.control import generate_outpainted_image
from misc.outpainting import run
from misc.replicate_me import run_replicate_with_pil
from utils import *


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
        # Direction vector for forward (assuming the camera looks towards the positive Z in its local space)
        forward_vec = torch.tensor([0, 0, 1 if direction == 'w' else -1]).float()
        # Transform the forward vector by the camera's rotation to get the world space direction
        world_direction = torch.matmul(R, forward_vec)
        # Update the translation component of the extrinsics matrix by the world space direction scaled by the amount
        t += world_direction * amount
    elif direction in ['a', 'd']:
        # Rotation angle (in radians). Positive for 'd' (right), negative for 'a' (left)
        angle = torch.tensor(amount if direction == 'd' else -amount)
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
    rr.script_setup(args, "4myst")

    # --- initial logging ---
    rr.log("description", rr.TextDocument('tada',  media_type=rr.MediaType.MARKDOWN), timeless=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    # --- pipeline setup ---
    depth_pipe = pipeline(task="depth-estimation", \
                    torch_dtype=torch.float32, \
                    model="LiheYoung/depth-anything-large-hf", 
                    device=torch.device("cuda"))

    # Iterate through images (video frames) logging data related to each frame.
    image = None
    extrinsics = None
    visualization = False
    intrinsics = torch.tensor([[256.0, 0.0000, 256.0000],
                               [0.0000, 256.0, 256.0000],
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
            breakpoint()

        # --- establish orientation ---
        if extrinsics == None:
            extrinsics = torch.eye(4).cuda()
        quat_xyzw = matrix_to_quaternion(extrinsics[:3, :3].unsqueeze(0))
        print(quat_xyzw.shape)

        # --- estimate depth ---
        pil_img = Image.fromarray(image.cpu().numpy())
        da_depth = F.interpolate(depth_pipe(pil_img)["predicted_depth"][None].cuda(), (512, 512), mode="bilinear", align_corners=False)[0, 0]
        da_3d, da_colors = depth_to_points_3d(da_depth, intrinsics, extrinsics, image)

        #user_input = input("Enter command (WASD), text for prompt, or 'quit' to exit: ")
        user_input = 'd'
        if user_input.lower() in ['w', 'a', 's', 'd']:
            # Move the camera based on the input
            #extrinsics = move_camera(extrinsics, user_input.lower(), 0.1)  # Assuming an amount of 0.1 for movement/rotation
            print("Camera moved/rotated. New extrinsics matrix:\n", extrinsics)
        elif user_input.lower() == 'quit':
            print("Exiting...")
            break
        elif user_input.lower() == 'break':
            print("Breakpoint...")
            breakpoint()
        else:
            # Treat the input as a text prompt for Stable Diffusion
            stable_diffusion_prompt = user_input
            print(f"Stored prompt for Stable Diffusion: '{stable_diffusion_prompt}'")

        proj_da, _, _, vis_da_3d, _, vis_da_colors, _ = points_3d_to_image(da_3d, da_colors, intrinsics, extrinsics, (512, 512))

        # Convert proj_da to integer and clamp to image size
        image_t = torch.zeros((512, 512, 3), dtype=torch.uint8).cuda()
        proj_da = proj_da.long()
        proj_da[:, 0] = proj_da[:, 0].clamp(0, 512 - 1)
        proj_da[:, 1] = proj_da[:, 1].clamp(0, 512 - 1)
        image_t[proj_da[:, 1], proj_da[:, 0]] = vis_da_colors

        # --- despeckle pipeline ---
        wombo_img = mod_fill(image_t)

        # --- sideways pipeline ---
        if idx % 15 == 0:
            wombo_mask = wombo_img.sum(dim=2) < 10
            wombo_img[wombo_mask] = -1.0
            sq_init = run_inpainting_pipeline(wombo_img, wombo_mask.float(), strength=0.89, prompt="a partial kitchen view")
            wombo_img = wombo_img.to(torch.uint8)
            wombo_img[wombo_mask] = sq_init[wombo_mask]

        # --- visualization ---
        if visualization:
            fig, ax = plt.subplots(2, 3, figsize=(10,7))
            ax[0, 0].imshow(pil_img)
            ax[1, 0].imshow(valid_mask_img.cpu().numpy())
            ax[0, 1].imshow(sq_img)
            ax[1, 1].imshow(sq_mask)
            ax[0, 2].imshow(sq_init.cpu().numpy())
            ax[1, 2].imshow(diffused_img.cpu().numpy())
            plt.tight_layout()
            plt.show()

        # --- rerun logging --- 
        rr.set_time_sequence("frame", idx+1)
        rr.log("da_3d", rr.Points3D(da_3d.cpu().numpy(), colors=[99, 99, 99]))
        rr.log("camera", rr.Transform3D(\
                            translation=extrinsics[:3, 3].cpu().numpy(),\
                            rotation=rr.Quaternion(xyzw=quat_xyzw.squeeze().cpu().numpy()), from_parent=True))
        rr.log("camera", rr.ViewCoordinates.RDF, timeless=True)  # X=Right, Y=Down, Z=Forward
        rr.log(
            "camera/image",
            rr.Pinhole(
                resolution=[512, 512],
                focal_length=[256, 256],
                principal_point=[256, 256],
            ),
        )
        rr.log("camera/image", rr.Image(np.array(pil_img)).compress(jpeg_quality=75))

    rr.script_teardown(args)
    breakpoint()

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
