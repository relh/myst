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
from ek_fields_utils.colmap_rw_utils import read_model
from PIL import Image
from transformers import pipeline

from utils import *


def read_and_log_sparse_reconstruction(dataset_path: Path, filter_output: bool, resize: tuple[int, int] | None) -> None:
    print("Reading sparse COLMAP reconstruction")
    cameras, images, sparse_points3D = read_model(dataset_path / "old_dense", ext=".bin")
    points3D, colors = load_dense_point_cloud(str(dataset_path) + "/dense/fused.ply")

    if filter_output:
        # Filter out noisy points
        sparse_points3D = {id: point for id, point in sparse_points3D.items() if point.rgb.any() and len(point.image_ids) > 4}

    rr.log("description", rr.TextDocument('tada',  media_type=rr.MediaType.MARKDOWN), timeless=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device="cuda:0")
    image_file = None

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
        projected_points, sparse_depth = points_3d_to_image(points3D, intrinsics, extrinsics, (camera.height, camera.width))
        est_depth = F.interpolate(pipe(img)["predicted_depth"][None].cuda(), (camera.height, camera.width), mode="bilinear", align_corners=False)[0, 0]
        aligned_depth, _ = ransac_alignment(est_depth.unsqueeze(0), sparse_depth.unsqueeze(0))
        point_cloud, pc_colors = depth_to_points_3d(aligned_depth.squeeze(), intrinsics, extrinsics, torch.tensor(np.array(img)))

        # --- rerun logging --- 
        rr.set_time_sequence("frame", idx+1)
        #rr.log("dense_points", rr.Points3D(points3D, colors=colors))

        rr.log("now_points", rr.Points3D(point_cloud.cpu().numpy(), colors=[99, 99, 99]))
        rr.log("camera/image/dense_keypoints", rr.Points2D(projected_points.cpu().numpy(), colors=[167, 138, 34]))

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
