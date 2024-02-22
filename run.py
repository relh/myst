#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import rerun as rr  # pip install rerun-sdk
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from transformers import pipeline

from ek_fields_utils.colmap_rw_utils import read_model
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
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device="cuda:0")
    image_file = None

    #inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
    #inpaint_pipe = inpaint_pipe.to("cpu")
    #prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
    #image and mask_image should be PIL images.
    #The mask structure is white for inpainting and black for keeping as is
    #image = inpaint_pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
    #image.save("./yellow_cat_on_park_bench.png")

    # Iterate through images (video frames) logging data related to each frame.
    num_images = len(images.values())
    for idx, image in enumerate(sorted(images.values(), key=lambda im: im.name)):  # type: ignore[no-any-return]
        if idx % 10 == 0:
            print(f'{idx} / {num_images}')

        # only get one image, so only do one COLMAP calibration
        if image_file is None: 
            image_file = dataset_path / "images" / image.name
            img = Image.open(str(image_file))

        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
        extrinsics = get_camera_extrinsic_matrix(image)
        proj_colmap, colmap_depth, vis_colmap_3d, vis_colmap_colors = points_3d_to_image(dense_points3d, None, intrinsics, extrinsics, (camera.height, camera.width), this_mask=None)
        da_depth = F.interpolate(pipe(img)["predicted_depth"][None].cuda(), (camera.height, camera.width), mode="bilinear", align_corners=False)[0, 0]
        aligned_da_depth, _ = ransac_alignment(da_depth.unsqueeze(0), colmap_depth.unsqueeze(0))

        da_3d, da_colors = depth_to_points_3d(aligned_da_depth.squeeze(), intrinsics, extrinsics, torch.tensor(np.array(img)))
        proj_da, _, vis_da_3d, vis_da_colors = points_3d_to_image(da_3d, da_colors, intrinsics, extrinsics, (camera.height, camera.width))

        # Optimizing the image creation process using smart indexing in PyTorch

        # Initialize a black image of size (256, 456, 3)
        image_size = (256, 456, 3)
        image = torch.zeros(image_size, dtype=torch.uint8).cuda()

        # Convert proj_da to integer and clamp to image size
        proj_da = proj_da.long()
        proj_da[:, 0] = proj_da[:, 0].clamp(0, image_size[1] - 1)
        proj_da[:, 1] = proj_da[:, 1].clamp(0, image_size[0] - 1)

        # Using advanced indexing to directly place colors at the projected points
        image[proj_da[:, 1], proj_da[:, 0]] = vis_da_colors

        #mask_image = Image.fromarray((torch.where((image.sum(dim=2) == 0), 1, 0).float() * 255.0).cpu().numpy()).convert('L')
        #image = Image.fromarray(image.cpu().numpy())
        #mask_image.save('./mask_image.png', format='PNG')
        #image.save('./image.png', format='PNG')

        mask_image = torch.where((image.sum(dim=2) == 0), 1, 0).float()
        new_image = fill_missing_values_batched(image, mask_image)

        #image = inpaint_pipe(prompt='a scene from epic kitchens', image=image, mask_image=mask_image).images[0]
        #del inpaint_pipe
        #breakpoint()

        # Visualizing the optimized generated image
        #new_image = new_image / 255.0
        #plt.imshow(new_image.cpu().numpy())
        #plt.axis('off')
        #plt.show()

        # --- rerun logging --- 
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

        if idx > 30: breakpoint()


def main() -> None:
    parser = ArgumentParser(description="Visualize the output of COLMAP's sparse reconstruction on a video.")
    rr.script_add_args(parser)
    args = parser.parse_args()

    rr.script_setup(args, "2myst")
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
    main()
