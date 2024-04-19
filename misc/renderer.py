#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import pytorch3d.ops
import rerun as rr  # pip install rerun-sdk
import torch
import torch.nn as nn
from pytorch3d.renderer import (AlphaCompositor, MeshRasterizer,
                                NormWeightedCompositor, PerspectiveCameras,
                                PointsRasterizationSettings, PointsRasterizer,
                                PointsRenderer, RasterizationSettings)
from pytorch3d.renderer.blending import BlendParams, hard_rgb_blend
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds

sys.path.append('depth_anything/metric_depth/')


def pts_3d_to_img_py3d(points_3d, colors, intrinsics, extrinsics, image_shape):
    points_3d = torch.cat((points_3d.clone(), torch.ones(points_3d.shape[0], 1, device=points_3d.device)), dim=1).T
    points_3d_trans = extrinsics @ points_3d
    
    cameras = PerspectiveCameras(
        device=points_3d.device,
        R=torch.eye(3).unsqueeze(0),
        in_ndc=False,
        T=torch.zeros(1, 3),
        focal_length=-intrinsics[0,0].unsqueeze(0),
        principal_point=intrinsics[:2,2].unsqueeze(0),
        image_size=torch.ones(1, 2) * 512,
    )
    
    point_cloud = Pointclouds(points=[(points_3d_trans[:3] / points_3d_trans[3]).T], 
                              features=[colors.clone().float() / 255.0])
    
    raster_settings = PointsRasterizationSettings(
        image_size=image_shape[:2], 
        radius=0.003,
        points_per_pixel=3
    )
    
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=NormWeightedCompositor(background_color=[0.0, 0.0, 0.0])
    )
    
    images = renderer(point_cloud)[0, ..., :3] * 255.0
    return images


def pts_3d_to_img_raster(points_3d, colors, intrinsics, extrinsics, image_shape):
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
