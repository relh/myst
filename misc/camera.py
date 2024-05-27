#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import open3d as o3d
import rerun as rr  # pip install rerun-sdk
import torch
from pytorch3d.renderer import (NormWeightedCompositor,
                                PointsRasterizationSettings, PointsRasterizer,
                                PointsRenderer, PulsarPointsRenderer)
from pytorch3d.structures import Pointclouds


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
        amount = (-amount if direction == 'w' else amount)
        extrinsics[:3, 3] += torch.tensor([0, 0, amount], device=extrinsics.device).float()
    elif direction in ['q', 'e']:
        # Rotation angle (in radians). Positive for 'e' (down), negative for 'q' (up)
        angle = torch.tensor(-amount if direction == 'e' else amount)
        # Rotation matrix around the X-axis (assuming X is forward/backward)
        rotation_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle), 0],
            [0, torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 0, 1]
        ], device=extrinsics.device)
        # Apply rotation to the extrinsics matrix
        extrinsics = torch.matmul(rotation_matrix, extrinsics)
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

def pts_world_to_cam(pts_3d, extrinsics):
    pts_homo = torch.cat((pts_3d, torch.ones(pts_3d.shape[0], 1, device=pts_3d.device)), dim=1).T
    pts_cam = extrinsics @ pts_homo 
    pts_cam = pts_cam[:3] / pts_cam[-1]
    return pts_cam.T

def pts_cam_to_world(pts_cam, extrinsics):
    extrinsics_inv = torch.linalg.pinv(extrinsics)
    pts_homo = torch.cat((pts_cam, torch.ones(pts_cam.shape[0], 1, device=pts_cam.device)), dim=1).T
    pts_3d = extrinsics_inv @ pts_homo 
    pts_3d = pts_3d[:3] / pts_3d[-1]
    return pts_3d.T 

def pts_cam_to_proj(pts_cam, intrinsics):
    proj_pts = (intrinsics @ pts_cam.T)
    return (proj_pts[:2] / proj_pts[2]).T

def project_to_image(camera_coords, intrinsics, image_shape, bbox=None):
    # Projects 3D points onto a 2D image plane using the camera's extrinsic and intrinsic matrices.
    im_proj_pts = pts_cam_to_proj(camera_coords, intrinsics)
    im_proj_pts = torch.round(im_proj_pts).long()
    x_pixels, y_pixels = im_proj_pts[:, 0], im_proj_pts[:, 1]

    visible = (x_pixels >= 0) & (x_pixels < image_shape[1]) &\
              (y_pixels >= 0) & (y_pixels < image_shape[0]) &\
              (camera_coords[:, 2] > 0.0)

    if bbox[0] is not None:
        tl, br = bbox
        not_box = ~((x_pixels >= tl[0]) & (x_pixels < br[0]) &\
                    (y_pixels >= tl[1]) & (y_pixels < br[1]))
        visible = visible & not_box

    return torch.stack([x_pixels[visible], y_pixels[visible]], dim=1), visible

def pts_world_to_visible(pts_3d, intrinsics, extrinsics, image_shape, bbox=None):
    camera_coords = pts_world_to_cam(pts_3d, extrinsics)
    proj, visible = project_to_image(camera_coords, intrinsics, image_shape, bbox=bbox)
    return visible

def pts_world_to_unique(pts_3d, colors, intrinsics, extrinsics, image_shape):
    camera_coords = pts_world_to_cam(pts_3d, extrinsics)
    proj, visible = project_to_image(camera_coords, intrinsics, image_shape)
    #proj, inverse_indices = torch.unique(proj, dim=0, sorted=False, return_inverse=True)
    colors = colors[visible].float() #[inverse_indices].float()
    pts_cam = camera_coords[visible][:, :3].float() #[inverse_indices][:, :3]
    proj = proj#[inverse_indices]
    return proj, colors, pts_3d, pts_cam, camera_coords

def pts_3d_to_img_raster(points_3d, colors, intrinsics, extrinsics, image_shape, cameras=None):
    image_shape = (int(image_shape[0]), int(image_shape[1]))

    camera_coords = pts_world_to_cam(points_3d, extrinsics)
    im_proj_pts = pts_cam_to_proj(camera_coords, intrinsics)

    # build depth map
    x_pixels = torch.round(im_proj_pts[:, 0]).long()
    y_pixels = torch.round(im_proj_pts[:, 1]).long()
    visible = (camera_coords[:, 2] > 0) & \
        (im_proj_pts[:, 0] >= 0) & (im_proj_pts[:, 1] >= 0) & \
        (x_pixels < image_shape[1]) & (y_pixels < image_shape[0]) 

    # screen out invisible and index for hand-obj occlusion
    x_pixels = x_pixels[visible]
    y_pixels = y_pixels[visible]
    proj_pts = camera_coords[visible]
    im_proj_pts = im_proj_pts[visible]

    depth_map = torch.full(image_shape, float('inf'), device=points_3d.device)
    depth_map[y_pixels, x_pixels] = torch.min(depth_map[y_pixels, x_pixels], proj_pts[:, 2])
    depth_map[depth_map == float('inf')] = 0

    proj_da, vis_depth_3d, vis_depth_colors = im_proj_pts[:, :2], \
        points_3d[visible], \
        (None if colors is None else colors[visible])

    image_t = torch.zeros((512, 512, 3), dtype=torch.uint8).cuda()
    proj_da = proj_da.long()
    proj_da[:, 0] = proj_da[:, 0].clamp(0, 512 - 1)
    proj_da[:, 1] = proj_da[:, 1].clamp(0, 512 - 1)
    image_t[proj_da[:, 1], proj_da[:, 0]] = (vis_depth_colors * 1.0).to(torch.uint8)

    return image_t.clone().float()

def pts_cam_to_pytorch3d(points_3d):
    points_3d[:, :2] = points_3d[:, :2] * -1
    return points_3d

def pts_3d_to_img_py3d(points_3d, colors, intrinsics, extrinsics, image_shape, cameras, scale, bbox=None):
    from misc.scale import median_scene_distance
    image_shape = (int(image_shape[0]), int(image_shape[1]))

    # find visible to compute appropriate point radius assuming dense 
    vis_mask = pts_world_to_visible(points_3d, intrinsics, extrinsics, image_shape, bbox)
    points_3d = points_3d[vis_mask]
    colors = colors[vis_mask]

    this_scale = median_scene_distance(points_3d, extrinsics) / 10.0
    radius = 1 / ((image_shape[0] * 0.5) * ((this_scale / scale) ** 2.0) + 1e-5)
    radius = max(radius, 1 / (image_shape[0] * 0.5))
    #print(f'radius: {radius}')
    #print(f'scale: {scale}')
    #print(f'current scale: {this_scale}')

    points_3d = pts_world_to_cam(points_3d, extrinsics)
    points_3d = pts_cam_to_pytorch3d(points_3d)
    point_cloud = Pointclouds(points=[points_3d], features=[colors.float() / 255.0])
    
    raster_settings = PointsRasterizationSettings(
        image_size=image_shape[:2], 
        radius=radius,
        points_per_pixel=3,
    )
    
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=NormWeightedCompositor(background_color=[-1.0, -1.0, -1.0])
    )
    
    image = renderer(point_cloud)[0, ..., :3] * 255.0
    return image
