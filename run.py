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
from misc.colab import run_inpainting_pipeline, create_outpainting_image_and_mask
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

    #rr.log("description", rr.TextDocument('tada',  media_type=rr.MediaType.MARKDOWN), timeless=True)
    #rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)

    # --- pipeline setup ---
    pipe = pipeline(task="depth-estimation", \
                    torch_dtype=torch.float16, \
                    model="LiheYoung/depth-anything-large-hf", 
                    device="cuda:0")

    #inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained( \
    #    "runwayml/stable-diffusion-inpainting", \
    #    torch_dtype=torch.float16)

    #inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
    #    "kandinsky-community/kandinsky-2-2-decoder-inpaint", 
    #    torch_dtype=torch.float16)
    #inpaint_pipe = inpaint_pipe.to("cuda:0")

    # Iterate through images (video frames) logging data related to each frame.
    image_file = None
    num_images = len(images.values())
    all_images = list(sorted(images.values(), key=lambda im: im.name))
    for idx, image in enumerate(all_images):
        if idx % 10 == 0:
            print(f'{idx} / {num_images}')

        # only get one image, so only do one COLMAP calibration
        if image_file is None: 
            image_file = dataset_path / "images" / image.name
            pil_img = Image.open(str(image_file))

        # --- setup camera ---
        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().cuda()
        extrinsics = get_camera_extrinsic_matrix(image).cuda()

        # --- project 3D points ---
        dense_points3d = torch.tensor(dense_points3d).float().cuda()
        proj_colmap, proj_dyn, colmap_depth, vis_colmap_3d, vis_dyn_3d, vis_colmap_colors, vis_dyn_colors = points_3d_to_image(dense_points3d, None, intrinsics, extrinsics, (camera.height, camera.width), this_mask=None)
        da_depth = F.interpolate(pipe(pil_img)["predicted_depth"][None].cuda(), (camera.height, camera.width), mode="bilinear", align_corners=False)[0, 0]
        aligned_da_depth, _ = ransac_alignment(da_depth.unsqueeze(0), colmap_depth.unsqueeze(0))

        img_tensor = torch.tensor(np.array(pil_img), device=aligned_da_depth.device)
        da_3d, da_colors = depth_to_points_3d(aligned_da_depth.squeeze(), intrinsics, extrinsics, img_tensor)

        next_extrinsics = get_camera_extrinsic_matrix(all_images[idx+1]).cuda()
        proj_da, _, _, vis_da_3d, _, vis_da_colors, _ = points_3d_to_image(da_3d, da_colors, intrinsics, next_extrinsics, (camera.height, camera.width))

        # Initialize a black image of size (256, 456, 3)
        # Convert proj_da to integer and clamp to image size
        image_size = (256, 456, 3)
        image_t = torch.zeros(image_size, dtype=torch.uint8).cuda()
        proj_da = proj_da.long()
        proj_da[:, 0] = proj_da[:, 0].clamp(0, image_size[1] - 1)
        proj_da[:, 1] = proj_da[:, 1].clamp(0, image_size[0] - 1)
        image_t[proj_da[:, 1], proj_da[:, 0]] = vis_da_colors

        # --- despeckle pipeline ---
        mask_img = torch.where((image_t.sum(dim=2) == 0), 1, 0).float()
        mask_img = Image.fromarray((torch.where((image_t.sum(dim=2) == 0), 1, 0).float() * 255.0).cpu().numpy()).convert('L')
        despeckled_img = fill_missing_values_batched(pil_img, mask_img)

        breakpoint()
        # --- apply delta extrinsics ---
        #quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        #next_extrinsics = get_camera_extrinsic_matrix(all_images[idx+1]).cuda()
        # TODO get real

        '''
        # --- sideways pipeline ---
        # split up image into two halves and in-paint
        breakpoint()
        big_img, big_mask = create_outpainting_image_and_mask(despeckled_img, zoom=0.95)
        big_init = run_inpainting_pipeline(big_img, big_mask, strength=1.00, prompt="A photo of a kitchen.")

        fig, ax = plt.subplots(2, 3, figsize=(10,7))
        ax[0, 0].imshow(pil_img)
        ax[1, 0].imshow(mask_img)
        ax[0, 1].imshow(big_img)
        ax[1, 1].imshow(big_mask)
        ax[0, 2].imshow(big_init)
        plt.tight_layout()
        plt.show()
        import sys
        sys.exit()
        breakpoint()
        '''

        # --- rerun logging --- 
        '''
        rr.set_time_sequence("frame", idx+1)
        #rr.log("dense_points", rr.Points3D(dense_points3d, colors=colors))

        rr.log("da_3d", rr.Points3D(da_3d.cpu().numpy(), colors=[99, 99, 99]))
        rr.log("camera/image/dense_keypoints", rr.Points2D(proj_da.cpu().numpy(), colors=[167, 138, 34]))

        # COLMAP's camera transform is "camera from world"
        rr.log("camera", rr.Transform3D(translation=image.tvec, rotation=rr.Quaternion(xyzw=quat_xyzw), from_parent=True))
        rr.log("camera", rr.ViewCoordinates.RDF, timeless=True)  # X=Right, Y=Down, Z=Forward

        # Log camera intrinsics
        rr.log(
            "camera/image",
            rr.Pinhole(
                resolution=[camera.width, camera.height],
                focal_length=camera.params[:2],
                principal_point=camera.params[2:],
            ),
        )

        if True:
            #rgb = None
            bgr = cv2.imread(str(image_file))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rr.log("camera/image", rr.Image(rgb).compress(jpeg_quality=75))
        elif resize:
            bgr = cv2.imread(str(image_file))
            bgr = cv2.resize(bgr, resize)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rr.log("camera/image", rr.Image(rgb).compress(jpeg_quality=75))
        else:
            rr.log("camera/image", rr.ImageEncoded(path=dataset_path / "images" / image.name))
        '''

        if idx > 30: breakpoint()


def main() -> None:
    parser = ArgumentParser(description="Visualize the output of COLMAP's sparse reconstruction on a video.")
    #rr.script_add_args(parser)
    args = parser.parse_args()

    #rr.script_setup(args, "2myst")
    dataset_path = Path('/mnt/sda/epic-fields/p01_04/')
    read_and_log_sparse_reconstruction(dataset_path, filter_output=True, resize=False)
    #rr.script_teardown(args)


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
