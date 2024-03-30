#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import rerun as rr  # pip install rerun-sdk
import torch
from pytorch3d.renderer import (AlphaCompositor, PointsRasterizationSettings,
                                PointsRasterizer, PointsRenderer)
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.structures import Pointclouds

sys.path.append('depth_anything/metric_depth/')

# Global settings
FL = 715.0873
FY = 256 * 1.0
FX = 256 * 1.0
NYU_DATA = False
FINAL_HEIGHT = 512 # 256
FINAL_WIDTH = 512 # 256
DATASET = 'nyu' # Lets not pick a fight with the model's dataloader

model = None


def pts_3d_to_img_pulsar(points_3d, colors, intrinsics, extrinsics, image_shape):
    device = points_3d.device
    
    # Convert extrinsics to camera position in world coordinates
    R = extrinsics[:3, :3].T  # Transpose to invert rotation
    T = -R @ extrinsics[:3, 3]
    
    # Since focal_length is not directly accepted, we might need to compute FOV
    # If your application requires exact intrinsics handling, this would need adjusting
    # For simplicity, let's use default FoV and assume principal point is at the center
    cameras = FoVPerspectiveCameras(device=device, R=[R], T=[T.unsqueeze(0)])
    
    # Create a point cloud object
    point_cloud = Pointclouds(points=[points_3d], features=[colors.to(torch.float32) / 255.0])  # Normalize colors
    
    # Define rasterization settings
    raster_settings = PointsRasterizationSettings(
        image_size=image_shape[:2], 
        radius=0.01,
        points_per_pixel=10
    )
    
    # Define the renderer
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=[0.0, 0.0, 0.0])
    )
    
    # Render the image
    images = renderer(point_cloud)
    
    # Convert the rendered image to the expected format
    image_rgba = images[0, ..., :3]  # Extract RGB values
    image_rgb = image_rgba.cpu().numpy()  # Assuming numpy array is needed

    return image_rgb


def pts_3d_to_img(points_3d, colors, intrinsics, extrinsics, image_shape):
    points_homogeneous = torch.cat((points_3d, torch.ones(points_3d.shape[0], 1, device=points_3d.device)), dim=1).T
    camera_coords = extrinsics @ points_homogeneous
    proj_pts = intrinsics @ camera_coords[:3, :]
    im_proj_pts = proj_pts[:2] / proj_pts[2]

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

    image_t = torch.zeros((512, 512, 3), dtype=torch.uint8).cuda()
    proj_da = proj_da.long()
    proj_da[:, 0] = proj_da[:, 0].clamp(0, 512 - 1)
    proj_da[:, 1] = proj_da[:, 1].clamp(0, 512 - 1)
    image_t[proj_da[:, 1], proj_da[:, 0]] = (vis_depth_colors * 1.0).to(torch.uint8)

    return image_t.clone().float()
