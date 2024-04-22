#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys

import numpy as np
import open3d as o3d
import rerun as rr  # pip install rerun-sdk
import torch
from pytorch3d.renderer import (NormWeightedCompositor,
                                PointsRasterizationSettings, PointsRasterizer,
                                PointsRenderer)
from pytorch3d.structures import Pointclouds

def pts_world_to_cam(pts_3d, extrinsics):
    pts_homo = torch.cat((pts_3d, torch.ones(pts_3d.shape[0], 1, device=pts_3d.device)), dim=1).T
    pts_cam = extrinsics @ pts_homo 
    return pts_cam

def pts_cam_to_world(pts_cam, extrinsics):
    extrinsics_inv = torch.linalg.inv(extrinsics)
    points_homogeneous = torch.cat((pts_cam, torch.ones(pts_cam.shape[0], 1, device=pts_cam.device)), dim=1)
    pts_3d = torch.mm(extrinsics_inv, points_homogeneous.t()).t()[:, :3]  # Apply extrinsics
    return pts_3d 

def pts_cam_to_proj(pts_cam, intrinsics):
    proj_pts = intrinsics @ pts_cam[:3, :]
    return (proj_pts[:2] / proj_pts[2])

def project_to_image(camera_coords, intrinsics, image_shape):
    # Projects 3D points onto a 2D image plane using the camera's extrinsic and intrinsic matrices.
    im_proj_pts = pts_cam_to_proj(camera_coords, intrinsics)
    y_pixels, x_pixels = torch.round(im_proj_pts).long()
    visible = (y_pixels >= 0) & (y_pixels < image_shape[1]) &\
              (x_pixels >= 0) & (x_pixels < image_shape[0]) &\
              (camera_coords[2, :] > 0.0)
    return torch.stack([x_pixels[visible], y_pixels[visible]], dim=1), visible

def pts_world_to_unique(pts_3d, colors, intrinsics, extrinsics, image_shape):
    camera_coords = pts_world_to_cam(pts_3d, extrinsics)
    proj, visible = project_to_image(camera_coords, intrinsics, image_shape)
    #breakpoint()
    proj, unique_indices = torch.unique(proj, dim=0, sorted=False, return_inverse=True)
    colors = colors[visible][unique_indices].float()
    pts_cam = camera_coords.T[visible][unique_indices][:, :3]
    proj = proj[unique_indices]
    return proj, colors, pts_3d, pts_cam, camera_coords.T

def pts_3d_to_img_raster(points_3d, colors, intrinsics, extrinsics, image_shape):
    image_shape = (int(image_shape[0]), int(image_shape[1]))

    camera_coords = pts_world_to_cam(points_3d, extrinsics)
    im_proj_pts = pts_cam_to_proj(camera_coords, intrinsics)

    # --- TODO replace
    # build depth map
    x_pixels = torch.round(im_proj_pts[0, :]).long()
    y_pixels = torch.round(im_proj_pts[1, :]).long()
    visible = (camera_coords[2] > 0) & (im_proj_pts[0] >= 0) & (im_proj_pts[1] >= 0) & \
          (x_pixels < image_shape[1]) & (y_pixels < image_shape[0]) 

    # screen out invisible and index for hand-obj occlusion
    x_pixels = x_pixels[visible]
    y_pixels = y_pixels[visible]
    proj_pts = proj_pts[:, visible]
    im_proj_pts = im_proj_pts[:, visible]

    depth_map = torch.full(image_shape, float('inf'), device=points_3d.device)
    depth_map[y_pixels, x_pixels] = torch.min(depth_map[y_pixels, x_pixels], proj_pts[2])
    depth_map[depth_map == float('inf')] = 0

    proj_da, vis_depth_3d, vis_depth_colors = im_proj_pts.T[:, :2], \
           points_3d[visible], \
           (None if colors is None else colors[visible])
    # --- TODO replace

    image_t = torch.zeros((512, 512, 3), dtype=torch.uint8).cuda()
    proj_da = proj_da.long()
    proj_da[:, 0] = proj_da[:, 0].clamp(0, 512 - 1)
    proj_da[:, 1] = proj_da[:, 1].clamp(0, 512 - 1)
    image_t[proj_da[:, 1], proj_da[:, 0]] = (vis_depth_colors * 1.0).to(torch.uint8)

    return image_t.clone().float()
