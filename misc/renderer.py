#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import rerun as rr  # pip install rerun-sdk
import torch
from pytorch3d.renderer import (NormWeightedCompositor,
                                PointsRasterizationSettings, PointsRasterizer,
                                PointsRenderer)
from pytorch3d.structures import Pointclouds

def pts_3d_to_img_py3d(points_3d, colors, intrinsics, extrinsics, image_shape, cameras):
    image_shape = (int(image_shape[0]), int(image_shape[1]))

    points_3d = torch.cat((points_3d.clone(), torch.ones(points_3d.shape[0], 1, device=points_3d.device)), dim=1).T
    points_3d_trans = extrinsics @ points_3d
    
    point_cloud = Pointclouds(points=[(points_3d_trans[:3] / points_3d_trans[3]).T], 
                              features=[colors.float() / 255.0])
    
    raster_settings = PointsRasterizationSettings(
        image_size=image_shape[:2], 
        radius=0.003,
        points_per_pixel=3
    )
    
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=NormWeightedCompositor(background_color=[0.0, 0.0, 0.0])
    )
    
    return renderer(point_cloud)[0, ..., :3] * 255.0
