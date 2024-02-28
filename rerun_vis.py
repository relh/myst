#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch

torch.backends.cuda.preferred_linalg_library()

import os
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt
import rerun as rr  # pip install rerun-sdk
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline

from ek_fields_utils.colmap_rw_utils import Camera, read_model
from utils import *


def scale_camera(camera: Camera, resize: tuple[int, int]) -> tuple[Camera, npt.NDArray[np.float_]]:
    """Scale the camera intrinsics to match the resized image."""
    assert camera.model == "PINHOLE"
    new_width = resize[0]
    new_height = resize[1]
    scale_factor = np.array([new_width / camera.width, new_height / camera.height])

    # For PINHOLE camera model, params are: [focal_length_x, focal_length_y, principal_point_x, principal_point_y]
    new_params = np.append(camera.params[:2] * scale_factor, camera.params[2:] * scale_factor)

    return (Camera(camera.id, camera.model, new_width, new_height, new_params), scale_factor)

def read_and_log_sparse_reconstruction(dataset_path: Path, filter_output: bool, resize: tuple[int, int] | None) -> None:
    print("Reading sparse COLMAP reconstruction")
    cameras, images, sparse_points3d = read_model(dataset_path / "old_dense", ext=".bin")
    dense_points3d, colors = load_dense_point_cloud(str(dataset_path) + "/dense/fused.ply")

    hand_obj_mask_path = './assets/P01_04/hands23/masks/'
    hand_masks = [hand_obj_mask_path + x for x in os.listdir(hand_obj_mask_path) if x.startswith('2_')]

    if filter_output:
        # Filter out noisy points
        sparse_points3d = {id: point for id, point in sparse_points3d.items() if point.rgb.any() and len(point.image_ids) > 4}

    rr.log("description", rr.TextDocument('tada',  media_type=rr.MediaType.MARKDOWN), timeless=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    rr.log("plot/avg_reproj_err", rr.SeriesLine(color=[240, 45, 58]), timeless=True)

    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device="cuda:0")

    # Iterate through images (video frames) logging data related to each frame.
    num_images = len(images.values())
    for idx, image in enumerate(sorted(images.values(), key=lambda im: im.name)):  # type: ignore[no-any-return]
        print(f'{idx} / {num_images}')
        image_file = dataset_path / "images" / image.name

        if not os.path.exists(image_file):
            breakpoint()
            continue

        if idx > 300: breakpoint()

        # COLMAP sets image ids that don't match the original video frame
        idx_match = re.search(r"\d+", image.name)
        assert idx_match is not None
        frame_idx = int(idx_match.group(0))

        quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
        camera = cameras[image.camera_id]
        scale_factor = np.array([1.0, 1.0])
        if resize:
            camera, scale_factor = scale_camera(camera, resize)

        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
        intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
        extrinsics = get_camera_extrinsic_matrix(image) 

        this_mask_files = [x for x in hand_masks if '_'+str(idx+1)+'.png' in x]
        if len(this_mask_files) > 0:
            this_mask = torch.stack([torch.tensor(np.asarray(Image.open(x).convert('1'))) for x in this_mask_files]).sum(dim=0).cuda()
        else:
            this_mask = torch.zeros((256, 456)).int().cuda()

        proj_colmap, proj_dyn, colmap_depth, vis_colmap_3d, vis_dyn_3d, vis_colmap_colors, vis_dyn_colors = points_3d_to_image(dense_points3d, None, intrinsics, extrinsics, (camera.height, camera.width), this_mask)
        
        img = Image.open(str(image_file))
        da_depth = F.interpolate(pipe(img)["predicted_depth"][None].cuda(), (camera.height, camera.width), mode="bilinear", align_corners=False)[0, 0]

        aligned_da_depth, _ = ransac_alignment(da_depth.unsqueeze(0), colmap_depth.unsqueeze(0))
        #aligned_da_depth = adjust_disparity(da_depth.squeeze(), colmap_depth.squeeze())

        img = torch.tensor(np.array(img))
        img[this_mask, :] = 255
        da_3d, da_colors = depth_to_points_3d(aligned_da_depth.squeeze(), intrinsics, extrinsics, img, mask=this_mask)

        visible = [id != -1 and sparse_points3d.get(id) is not None for id in image.point3D_ids]
        visible_ids = image.point3D_ids[visible]

        visible_xyzs = [sparse_points3d[id] for id in visible_ids]
        visible_xys = image.xys[visible]
        if resize:
            visible_xys *= scale_factor

        points = [point.xyz for point in visible_xyzs]
        point_colors = [point.rgb for point in visible_xyzs]
        point_errors = [point.error for point in visible_xyzs]

        # --- rerun logging --- 
        rr.set_time_sequence("frame", frame_idx)
        rr.log("dense_points", rr.Points3D(dense_points3d, colors=colors))
        rr.log("dynamic_points", rr.Points3D(dense_points3d, colors=colors))

        rr.log("plot/avg_reproj_err", rr.Scalar(np.mean(point_errors)))

        rr.log("points", rr.Points3D(points, colors=point_colors), rr.AnyValues(error=point_errors))
        rr.log("da_3d", rr.Points3D(da_3d.cpu().numpy(), colors=da_colors.cpu().numpy())) #[99, 99, 99]))

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

        if resize:
            bgr = cv2.imread(str(image_file))
            bgr = cv2.resize(bgr, resize)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rr.log("camera/image", rr.Image(rgb).compress(jpeg_quality=75))
        else:
            rr.log("camera/image", rr.ImageEncoded(path=dataset_path / "images" / image.name))

        rr.log("camera/image/keypoints", rr.Points2D(visible_xys, colors=[34, 138, 167]))
        rr.log("camera/image/dense_keypoints", rr.Points2D(proj_colmap.cpu().numpy(), colors=[167, 138, 34]))
        rr.log("camera/image/dynamic_keypoints", rr.Points2D(proj_dyn.cpu().numpy(), colors=[34, 138, 167]))


def main() -> None:
    parser = ArgumentParser(description="Visualize the output of COLMAP's sparse reconstruction on a video.")
    parser.add_argument("--unfiltered", action="store_true", help="If set, we don't filter away any noisy data.")
    parser.add_argument(
        "--dataset",
        action="store",
        # default="colmap_fiat",
        default="pinhole_workspace",
        choices=["colmap_rusty_car", "colmap_fiat", "colmap_espresso_shot", "pinhole_workspace"],
        help="Which dataset to download",
    )
    parser.add_argument("--resize", action="store", help="Target resolution to resize images")

    rr.script_add_args(parser)
    args = parser.parse_args()
    #args.addr = "0.0.0.0:9876"

    if args.resize:
        args.resize = tuple(int(x) for x in args.resize.split("x"))

    rr.script_setup(args, "csfm")
    dataset_path = Path('/mnt/sda/epic-fields/p01_04/')
    read_and_log_sparse_reconstruction(dataset_path, filter_output=not args.unfiltered, resize=args.resize)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
