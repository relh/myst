#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import collections
import os
import struct
from pathlib import Path
from typing import Mapping

import kornia
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rerun as rr  # pip install rerun-sdk
import torch
import torch.nn.functional as F
from decord import VideoReader
from einops import rearrange, repeat
from imageio import get_writer
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
from torchvision.transforms import ToTensor
from transformers import pipeline

from ek_fields_utils.colmap_rw_utils import read_model, sort_images

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')
torch.backends.cuda.preferred_linalg_library()

def mod_fill(tensor):
    tensor = tensor.to(torch.float32)

    # Mask of zeros in all channels
    zero_mask = (tensor == 0).all(dim=-1)

    # Preparing the tensor for convolution
    tensor_expanded = tensor.permute(2, 0, 1).unsqueeze(0)  # Shape [1, 3, 256, 456]

    # Define the convolution kernel
    kernel = torch.ones((3, 3), dtype=torch.float32).to(tensor.device)
    kernel[1, 1] = 0  # Excluding the center pixel from averaging
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, 3, 3]
    kernel = kernel.repeat(3, 1, 1, 1)  # Adjust kernel for 3 channels, shape [3, 1, 3, 3]

    # Applying padding
    padded_tensor = F.pad(tensor_expanded, (1, 1, 1, 1), mode='reflect')

    # Convolution operations to sum and count neighbors
    sum_neighbors = F.conv2d(padded_tensor, kernel, padding=0, groups=3)
    non_zero_mask = (padded_tensor != 0).float()
    count_nonzero_neighbors = F.conv2d(non_zero_mask, kernel, padding=0, groups=3)

    # Calculate the averages
    average_values = sum_neighbors / count_nonzero_neighbors.clamp(min=1)
    average_values = average_values.squeeze(0).permute(1, 2, 0)  # Back to original shape [256, 456, 3]

    # Apply the averages to zero values
    tensor[zero_mask] = average_values.to(torch.float32)[zero_mask]
    return tensor#.to(torch.uint8)


def make_square_mask(side='left'):
    new_mask = torch.zeros(256, 256)
    if side == 'left':
        new_mask[:, :int(56*0.5)] = 255.0
    else:
        new_mask[:, int(-56*0.5):] = 255.0
    new_mask = Image.fromarray(new_mask.cpu().numpy()).convert("L")
    return new_mask

def prep_pil(pil_img):
    """
    Turns a 456x256 image into two square images padded with noise to make them 256x256.
    """
    # turns ek 456x256 img into two square 256ers 
    #if mask:
    #    image = pil_img
    #else:
    #    b, g, r = pil_img.split()
    # 
    #     # Reassemble the image with bands in the correct order
    #     image = Image.merge("RGB", (r, g, b))
    image = pil_img
    width, height = image.size
    split_point = width // 2
    left_image = image.crop((0, 0, split_point, height))
    right_image = image.crop((split_point, 0, width, height))

    # Pad the images
    left_padded = ImageOps.expand(left_image, (256-split_point, 0, 0, 0), fill='black')
    right_padded = ImageOps.expand(right_image, (0, 0, 256-split_point, 0), fill='black')

    # Save or display the images
    return left_padded, right_padded

def save_rgba_image(rgb_image, mask, file_path):
    # Convert the PyTorch tensor to a numpy array and adjust dimensions
    rgb_array = rgb_image.numpy().astype(np.uint8)

    # Create a PIL image from the numpy array
    rgb_pil = Image.fromarray(rgb_array)

    # Invert the mask: 0 for transparent (mask=1) and 255 for opaque (mask=0)
    alpha = np.where(mask.numpy() == 1, 0, 255).astype(np.uint8)

    # Create an alpha channel image from the mask
    alpha_pil = Image.fromarray(alpha)

    # Combine the RGB image with the alpha channel to get an RGBA image
    rgba_pil = Image.merge("RGBA", [rgb_pil.split()[0], rgb_pil.split()[1], rgb_pil.split()[2], alpha_pil])

    # Save the image
    rgba_pil.save(file_path, 'PNG')

def fill_missing_values_batched(image, mask):
    # Ensure image is in float and mask is repeated for each channel
    image_batch = rearrange(torch.tensor(np.array(image)), 'h w c -> 1 c h w').float().cuda()
    valid_mask_batch = repeat(mask, 'h w -> 1 c h w', c=3).float().cuda().half()

    # Define a 3x3 kernel for convolution
    kernel = torch.ones((3, 1, 3, 3), dtype=torch.float32).cuda()

    # Convolve image and valid_mask with the kernel
    sum_neighbors = F.conv2d(image_batch * valid_mask_batch, kernel, padding=1, groups=3)
    count_neighbors = F.conv2d(valid_mask_batch, kernel, padding=1, groups=3)

    # Normalize sum by the count of valid neighbors
    average_values = sum_neighbors / count_neighbors.clamp(min=1)

    # Replace missing pixels
    image_filled = torch.where(mask == 1, average_values, image_batch)

    # Clamp values to the valid range (assuming 8-bit image)
    image_filled = image_filled.clamp(0, 255)

    return rearrange(image_filled, '1 c h w -> h w c')

def compute_local_scaling_factors(estimated_disparity, colmap_depth, window_size=5):
    """
    Compute local scaling factors using convolution for efficient batch processing.
    """
    # Create a mask of non-zero values in colmap_depth
    mask = colmap_depth != 0
    masked_colmap = colmap_depth * mask.float()
    
    # Avoid division by zero in estimated_disparity
    estimated_disparity[estimated_disparity == 0] = 1e-6

    # Calculate scaling factors where colmap_depth is non-zero
    scaling_factors = torch.zeros_like(estimated_disparity)
    scaling_factors[mask] = masked_colmap[mask] / estimated_disparity[mask]

    # Define a kernel for summing scaling factors and counts within local windows
    kernel = torch.ones((1, 1, window_size, window_size), device=colmap_depth.device)

    # Compute the sum of scaling factors and the count of non-zero values in each window
    local_sum = F.conv2d(scaling_factors.unsqueeze(0).unsqueeze(0), kernel, padding=window_size//2)
    local_count = F.conv2d(mask.float().unsqueeze(0).unsqueeze(0), kernel, padding=window_size//2)

    # Calculate the average scaling factor for each window
    local_avg_scaling = local_sum / local_count
    local_avg_scaling[local_count == 0] = 0  # Avoid division by zero

    return local_avg_scaling.squeeze()#, local_count.squeeze()

def adjust_disparity(estimated_disparity, colmap_depth, window_size=3, iterations=11):
    """
    Iteratively adjust estimated_disparity based on local scaling factors.
    """
    height, width = colmap_depth.shape
    mask = colmap_depth != 0

    # Initial adjustment
    local_scaling = compute_local_scaling_factors(estimated_disparity, colmap_depth, window_size)
    adjusted_disparity = torch.where(mask, colmap_depth, estimated_disparity * local_scaling)
    mask = adjusted_disparity != 0

    for _ in range(iterations - 1):
        # Compute scaling factors for the current adjusted_disparity
        local_scaling = compute_local_scaling_factors(estimated_disparity, adjusted_disparity, window_size)
        adjusted_disparity = torch.where(mask, adjusted_disparity, estimated_disparity * local_scaling)
        mask = adjusted_disparity != 0
        if mask.sum() == 0: break

    return adjusted_disparity

def ransac_alignment(estimated_dense_disparity, colmap_depth, ransac_iters=2096):
    disparity_max = 10000
    disparity_min = 0.0001
    mask = colmap_depth != 0
    target_disparity = torch.zeros_like(colmap_depth)
    target_disparity[mask] = 1.0 / torch.clip(colmap_depth[mask], 1 / disparity_max, 1 / disparity_min)
    prediction1 = estimated_dense_disparity.repeat(ransac_iters, 1, 1)
    target_disparity1 = target_disparity.repeat(ransac_iters, 1, 1)
    mask1 = mask.repeat(ransac_iters, 1, 1)
    scale, shift = compute_scale_and_shift(prediction1, target_disparity1, mask1)
    prediction_aligned = (scale.view(-1, 1, 1) * prediction1 + shift.view(-1, 1, 1)).clamp(disparity_min, disparity_max)
    prediction_depth = 1.0 / prediction_aligned
    err = torch.abs(prediction_depth - colmap_depth.repeat(ransac_iters, 1, 1))
    min_value, min_index = torch.min(torch.mean(err, dim=[1, 2]), dim=0)
    return prediction_depth[min_index].unsqueeze(0), min_value.item()

def compute_scale_and_shift(prediction, target, mask):
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))
    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1

def get_camera_extrinsic_matrix(image):
    quat_xyzw = image.qvec[[1, 2, 3, 0]]  # COLMAP uses wxyz quaternions
    r = R.from_quat(quat_xyzw).as_matrix()
    T = torch.eye(4)
    T[:3, :3] = torch.from_numpy(r)
    T[:3, 3] = torch.tensor(image.tvec)
    return T

def points_3d_to_image(points_3d, colors, intrinsics, extrinsics, image_shape, this_mask=None):
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

def load_dense_point_cloud(ply_file_path: str):
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    return np.asarray(point_cloud.points), np.asarray(point_cloud.colors)

def depth_to_points_3d(depth_map, K, E, image=None, mask=None):
    if mask is None:
        mask = depth_map > 0.0
    mask = mask.bool()
    cam_coords = kornia.geometry.depth_to_3d_v2(depth_map, K)#, normalize_points=True)
    cam_coords_flat = rearrange(cam_coords, 'h w xyz -> xyz (h w)')
    ones = torch.ones(1, cam_coords_flat.shape[1], device=cam_coords.device)
    cam_coords_homogeneous = torch.cat([cam_coords_flat, ones], dim=0).to(torch.float32)
    E_inv = torch.linalg.inv(E.to(torch.float32))
    print(E_inv)
    world_coords_homogeneous = (E @ cam_coords_homogeneous).T
    world_coords = (world_coords_homogeneous[:, :3] / world_coords_homogeneous[:, 3].unsqueeze(1)).view(cam_coords.shape)
    #world_coords[:, :, 1] *= -1.0
    #world_coords[:, :, 2] *= -1.0
    return world_coords[mask], image[mask]

if __name__ == "__main__":
    pass
