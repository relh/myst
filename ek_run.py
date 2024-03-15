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
from torchvision.transforms import ToPILImage, ToTensor
from transformers import pipeline

from ek_fields_utils.colmap_rw_utils import read_model
from misc.colab import run_inpainting_pipeline
from misc.control import generate_outpainted_image
from misc.outpainting import run
from misc.replicate_me import run_replicate_with_pil
from utils import *


def read_and_log_sparse_reconstruction(dataset_path: Path, filter_output: bool, resize: tuple[int, int] | None) -> None:
    print("Reading sparse COLMAP reconstruction")
    cameras, images, sparse_points3d = read_model(dataset_path / "old_dense", ext=".bin")
    dense_points3d, colors = load_dense_point_cloud(str(dataset_path) + "/dense/fused.ply")

    if filter_output:
        # Filter out noisy points
        sparse_points3d = {id: point for id, point in sparse_points3d.items() if point.rgb.any() and len(point.image_ids) > 4}

    rr.log("description", rr.TextDocument('tada',  media_type=rr.MediaType.MARKDOWN), timeless=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    # --- pipeline setup ---
    depth_pipe = pipeline(task="depth-estimation", \
                    torch_dtype=torch.float32, \
                    model="LiheYoung/depth-anything-large-hf", 
                    device=torch.device("cuda"))
    #depth_pipe = StableDiffusionInpaintPipeline.from_pretrained( \
    #    "runwayml/stable-diffusion-inpainting", \
    #    torch_dtype=torch.float16)

    #depth_pipe = AutoPipelineForInpainting.from_pretrained(
    #    "kandinsky-community/kandinsky-2-2-decoder-inpaint", 
    #    torch_dtype=torch.float16)
    #depth_pipe = depth_pipe.to("cuda:0")

    # Iterate through images (video frames) logging data related to each frame.
    image_file = None
    visualization = False
    num_images = len(images.values())
    all_images = list(sorted(images.values(), key=lambda im: im.name))
    for idx, image in enumerate(all_images):
        if idx % 15 == 0:
            print(f'{idx} / {num_images}')

        # only get one image, so only do one COLMAP calibration
        if image_file is None: 
            image_file = dataset_path / "images" / image.name
            pil_img = Image.open(str(image_file))
        else:
            pil_img = Image.fromarray(wombo_img.cpu().numpy().astype('uint8'))

        # --- setup camera ---
        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().cuda()
        extrinsics = get_camera_extrinsic_matrix(image).cuda()

        # --- project 3D points ---
        dense_points3d = torch.tensor(dense_points3d).float().cuda()
        proj_colmap, proj_dyn, colmap_depth, vis_colmap_3d, vis_dyn_3d, vis_colmap_colors, vis_dyn_colors = points_3d_to_image(dense_points3d, None, intrinsics, extrinsics, (camera.height, camera.width), this_mask=None)
        da_depth = F.interpolate(depth_pipe(pil_img)["predicted_depth"][None].cuda(), (camera.height, camera.width), mode="bilinear", align_corners=False)[0, 0]
        breakpoint()

        ransac_da_depth, _ = ransac_alignment(da_depth.unsqueeze(0), colmap_depth.unsqueeze(0))
        local_da_depth = adjust_disparity(da_depth.squeeze(), colmap_depth.squeeze())
        wombo_depth = (ransac_da_depth + local_da_depth) / 2.0

        img_tensor = torch.tensor(np.array(pil_img), device=wombo_depth.device)
        da_3d, da_colors = depth_to_points_3d(wombo_depth.squeeze(), intrinsics, extrinsics, img_tensor)

        next_extrinsics = get_camera_extrinsic_matrix(all_images[idx+1]).cuda()
        proj_da, _, _, vis_da_3d, _, vis_da_colors, _ = points_3d_to_image(da_3d, da_colors, intrinsics, next_extrinsics, (camera.height, camera.width))

        # Convert proj_da to integer and clamp to image size
        image_t = torch.zeros((256, 456, 3), dtype=torch.uint8).cuda()
        proj_da = proj_da.long()
        proj_da[:, 0] = proj_da[:, 0].clamp(0, 456 - 1)
        proj_da[:, 1] = proj_da[:, 1].clamp(0, 256 - 1)
        image_t[proj_da[:, 1], proj_da[:, 0]] = vis_da_colors

        # --- despeckle pipeline ---
        #mask_img = Image.fromarray((torch.where((image_t.sum(dim=2) == 0), 1, 0).float() * 1.0).cpu().numpy()).convert('L')
        #valid_mask_img = torch.where((image_t.sum(dim=2) <= 10), 0, 1).float().cuda()
        wombo_img = mod_fill(image_t)

        # --- pipeline outline ---
        # after estimating depth, voxel of radius r set in 3D 
        # when re-estimating for new frame, do lookup, then change
        # --> in-fill? 
        # ---> 

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
        rr.log("dense_points", rr.Points3D(dense_points3d.cpu().numpy(), colors=colors))
        rr.log("da_3d", rr.Points3D(da_3d.cpu().numpy(), colors=[99, 99, 99]))
        #rr.log("camera/image/dense_keypoints", rr.Points2D(proj_da.cpu().numpy(), colors=[167, 138, 34]))

        rr.log("camera", rr.Transform3D(translation=image.tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), from_parent=True))
        rr.log("camera", rr.ViewCoordinates.RDF, timeless=True)  # X=Right, Y=Down, Z=Forward
        rr.log(
            "camera/image",
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=camera.params[:2],
                principal_point=camera.params[2:],
            ),
        )
        rr.log("camera/image", rr.Image(np.array(pil_img)).compress(jpeg_quality=75))
        #rr.log("camera/image", rr.Image(np.array(sq_mask)[100:-100]).compress(jpeg_quality=75))
    breakpoint()

def main() -> None:
    parser = ArgumentParser(description="Visualize the output of COLMAP's sparse reconstruction on a video.")
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "4myst")
    dataset_path = Path('/mnt/sda/epic-fields/p01_04/')
    read_and_log_sparse_reconstruction(dataset_path, filter_output=True, resize=False)
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
        with torch.autocast(device_type="cuda"):
            main()
