#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import glob
import os
import struct
import sys
from pathlib import Path
from typing import Mapping

import kornia
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rerun as rr  # pip install rerun-sdk
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange, repeat
from imageio import get_writer
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import ToTensor
from tqdm import tqdm
from transformers import pipeline

sys.path.append('depth_anything/metric_depth/')
from depth_anything.metric_depth.zoedepth.models.builder import build_model
from depth_anything.metric_depth.zoedepth.utils.config import get_config

# Global settings
FL = 715.0873
FY = 256 * 1.0
FX = 256 * 1.0
NYU_DATA = False
FINAL_HEIGHT = 512 # 256
FINAL_WIDTH = 512 # 256
DATASET = 'nyu' # Lets not pick a fight with the model's dataloader

model = None

def img_to_pts_3d(color_image, extrinsics):
    global model
    if model is None:
        config = get_config('zoedepth', "eval", DATASET)
        config.pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
        model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
    #color_image = Image.open(image_path).convert('RGB')
    original_width, original_height = color_image.size
    image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    pred = model(image_tensor, dataset=DATASET)
    if isinstance(pred, dict):
        pred = pred.get('metric_depth', pred.get('out'))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    pred = pred.squeeze().detach().cpu().numpy()

    # Resize color image and depth to final size
    resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
    resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

    focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
    x = (x - FINAL_WIDTH / 2.0) / focal_length_x
    y = (y - FINAL_HEIGHT / 2.0) / focal_length_y
    z = np.array(resized_pred)

    # Compute 3D points in camera coordinates
    points_camera_coord = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3) * 50.0
    extrinsics_inv = torch.linalg.inv(extrinsics)
    
    # Convert to torch tensor and apply extrinsics to get points in world coordinates
    points_camera_coord_tensor = torch.tensor(points_camera_coord, dtype=torch.float32, device='cuda')
    points_homogeneous = torch.cat((points_camera_coord_tensor, torch.ones(points_camera_coord_tensor.shape[0], 1, device=points_camera_coord_tensor.device)), dim=1)
    points_world_coord = torch.mm(extrinsics_inv, points_homogeneous.t()).t()[:, :3]  # Apply extrinsics

    colors = np.array(resized_color_image).reshape(-1, 3) / 255.0
    da_colors = (torch.tensor(colors) * 255.0).float().to('cuda').to(torch.uint8)
    return points_world_coord, da_colors

def pts_3d_to_img(points_3d, colors, intrinsics, extrinsics, image_shape, this_mask=None):
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

    if this_mask == None: this_mask = torch.zeros((512, 512), device=points_3d.device)
    unocc = this_mask[y_pixels, x_pixels] == 0
    occ = this_mask[y_pixels, x_pixels] == 1

    depth_map = torch.full(image_shape, float('inf'), device=points_3d.device)
    depth_map[y_pixels[unocc], x_pixels[unocc]] = torch.min(depth_map[y_pixels[unocc], x_pixels[unocc]], proj_pts[2][unocc])
    depth_map[depth_map == float('inf')] = 0

    return im_proj_pts.T[unocc][:, :2], \
           im_proj_pts.T[occ][:, :2], \
           depth_map, \
           points_3d[visible][unocc], \
           points_3d[visible][occ], \
           (None if colors is None else colors[visible][unocc]), \
           (None if colors is None else colors[visible][occ])


if __name__ == '__main__':
    image_path = "./depth_anything/metric_depth/my_test/input/demo11.png"
    color_image = Image.open(image_path).convert('RGB')
    pcd = image_to_3d(color_image)
    breakpoint()
