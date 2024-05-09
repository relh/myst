#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import rerun as rr  # pip install rerun-sdk
import torch
from pytorch3d.renderer import (NormWeightedCompositor,
                                PointsRasterizationSettings, PointsRasterizer,
                                PointsRenderer, PulsarPointsRenderer)
from pytorch3d.structures import Pointclouds

from misc.camera import pts_world_to_cam

def pts_cam_to_pytorch3d(points_3d):
    points_3d[:, :2] = points_3d[:, :2] * -1
    return points_3d

def pts_3d_to_img_py3d(points_3d, colors, intrinsics, extrinsics, image_shape, cameras):
    image_shape = (int(image_shape[0]), int(image_shape[1]))

    points_3d = pts_world_to_cam(points_3d, extrinsics)
    points_3d = pts_cam_to_pytorch3d(points_3d)
    point_cloud = Pointclouds(points=[points_3d], features=[colors.float() / 255.0])
    
    raster_settings = PointsRasterizationSettings(
        image_size=image_shape[:2], 
        radius=1.0 / (image_shape[0] * 0.5),
    )
    
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=NormWeightedCompositor(background_color=[-1.0, -1.0, -1.0])
    )
    
    image = renderer(point_cloud)[0, ..., :3] * 255.0
    return image
