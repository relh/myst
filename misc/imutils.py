#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import kornia
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rerun as rr  # pip install rerun-sdk
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R

#torch.set_default_dtype(torch.float32)
#torch.set_default_device('cuda')
torch.backends.cuda.preferred_linalg_library()

def fill(tensor):
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

def select_bounding_box(image: Image.Image):
    # Convert PIL image to numpy array
    image_array = np.array(image)

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(image_array)

    # List to store the coordinates
    coords = []

    def onclick(event):
        # Store the coordinates
        ix, iy = event.xdata, event.ydata
        coords.append((ix, iy))

        # Draw a point on the image
        ax.plot(ix, iy, 'ro')
        fig.canvas.draw()

        # Stop after two points are selected
        if len(coords) == 2:
            plt.close(fig)

    # Connect the click event to the function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Display the image and wait for clicks
    plt.show()

    # Return the coordinates if two points were selected
    if len(coords) == 2:
        return [int(x) for x in coords[0]], [int(x) for x in coords[1]]
    else:
        return None, None

if __name__ == "__main__":
    pass
