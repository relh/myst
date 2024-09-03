#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from pytorch3d.transforms import Transform3d

from misc.camera import project_to_image, pts_cam_to_world, pts_world_to_unique


def median_scene_distance(point_cloud, camera_extrinsics):
    """
    Calculates the median distance from the camera to the points in the point cloud using world coordinates.

    Args:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) containing the coordinates of the points in world coordinates.
    - camera_extrinsics (torch.Tensor): A 4x4 tensor representing the camera extrinsic matrix that transforms points from world coordinates to camera coordinates.

    Returns:
    - float: The median distance of the points from the camera.
    """
    # Extract camera position from the extrinsics matrix
    # The camera position is the negative of the translation components of the matrix, transformed by the rotation part.
    rotation_matrix = camera_extrinsics[:3, :3]
    translation_vector = camera_extrinsics[:3, 3]
    camera_position = -torch.matmul(rotation_matrix.transpose(0, 1), translation_vector)

    distances = torch.norm(point_cloud - camera_position, dim=1)
    return distances.median().item() # Convert to Python float for general use 

def fit_least_squares_shift_scale_with_mask(pc1, pc2, mask1, mask2):
    """
    Fit a least squares shift and scale transformation from pc2 to pc1 using provided masks.
    pc1, pc2 are (N, 3) tensors representing the point clouds.
    mask1, mask2 are boolean tensors indicating corresponding points in pc1 and pc2.
    """
    # Apply masks to select corresponding points
    sub_pc1 = pc1[mask1]
    sub_pc2 = pc2[mask2]

    # Compute centroids of the selected subsets
    centroid1 = torch.mean(sub_pc1, dim=0)
    centroid2 = torch.mean(sub_pc2, dim=0)

    # Compute scale factors along each dimension
    std1 = torch.std(sub_pc1, dim=0, unbiased=False)
    std2 = torch.std(sub_pc2, dim=0, unbiased=False)
    scale_factors = std1 / std2

    # Scale the corresponding part of the second point cloud
    scaled_pc2 = (sub_pc2 - centroid2) * scale_factors + centroid2

    # Compute the translation needed after scaling
    translation = centroid1 - torch.mean(scaled_pc2, dim=0)

    # Apply the transformation to the entire second point cloud
    transformed_pc2 = (pc2 - centroid2) * scale_factors + centroid2 + translation

    rmse = torch.sqrt(torch.mean((transformed_pc2[mask2] - pc1[mask1])**2))
    print(f"Alignment RMSE: {rmse.item()}")
    rmse = torch.sqrt(torch.mean((pc2[mask2] - pc1[mask1])**2))
    print(f"Original RMSE: {rmse.item()}")

    return transformed_pc2, scale_factors, translation

def align_partial_point_clouds(source, target, source_mask, target_mask, threshold=1.0, trans_init=None):
    """
    Aligns parts of two point clouds using the ICP algorithm and applies the transformation to the whole point cloud.

    Parameters:
    - source: The entire source point cloud (type: o3d.geometry.PointCloud).
    - target: The entire target point cloud (type: o3d.geometry.PointCloud).
    - source_mask: Indices of the source point cloud to use for ICP (type: np.ndarray).
    - target_mask: Indices of the target point cloud to use for ICP (type: np.ndarray).
    - threshold: The maximum distance threshold between corresponding points (type: float).
    - trans_init: Initial transformation guess (type: numpy.ndarray).

    Returns:
    - transformed_source: The entire source point cloud transformed (type: o3d.geometry.PointCloud).
    - icp_result: The result of the ICP registration (type: registration.RegistrationResult).
    """
    if trans_init is None:
        trans_init = np.eye(4)  # Default to the identity matrix if no initial guess

    # Extract the corresponding parts using the provided masks
    source_part = source.select_by_index(source_mask)
    target_part = target.select_by_index(target_mask)

    # Perform ICP on the corresponding parts
    icp_result = o3d.pipelines.registration.registration_icp(
        source_part, target_part, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return icp_result

def vis_images(img1, img2):
    img1 = img1.cpu().numpy() + 1 
    img2 = img2.cpu().numpy() + 1
    img1 /= 255.0
    img2 /= 255.0

    # Compute the absolute difference image
    difference = np.abs(img1 - img2)
    difference[img1 == 0] = 0.0
    difference[img2 == 0] = 0.0

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plotting the first image
    axes[0].imshow(img1)
    axes[0].axis('off')  # Turn off axis numbers and ticks
    axes[0].set_title('Image 1')

    # Plotting the second image
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Image 2')

    # Plotting the difference image
    axes[2].imshow(difference)
    axes[2].axis('off')
    axes[2].set_title('Difference')

    # Display the plots
    plt.tight_layout()
    plt.show()

def project_and_scale_points(gt_points_3d, new_points_3d, gt_colors, new_colors, intrinsics, extrinsics, image_shape, color_threshold=30, align_mode='median'):
    gt_proj, mod_gt_colors, gt_3d, select_gt_3d, gt_camera = pts_world_to_unique(gt_points_3d, gt_colors, intrinsics, extrinsics, image_shape)
    new_proj, mod_new_colors, new_3d, select_new_3d, new_camera = pts_world_to_unique(new_points_3d, new_colors, intrinsics, extrinsics, image_shape)

    # Initialize two tensors filled with -1 (indicating empty/invalid)
    gt_c_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')
    new_c_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')

    # Fill the tensors with the RGB colors at the specified coordinates
    gt_c_image[gt_proj[:, 0], gt_proj[:, 1]] = mod_gt_colors
    new_c_image[new_proj[:, 0], new_proj[:, 1]] = mod_new_colors
    gt_c_image = gt_c_image.permute(1, 0, 2)
    new_c_image = new_c_image.permute(1, 0, 2)

    # Find indices where both tensors have valid (non-empty) colors
    valid_indices = ((gt_c_image != -1) & (new_c_image != -1)).all(dim=2)  # Both have valid RGB colors

    # Calculate the difference between colors at valid indices
    color_difference = torch.abs(gt_c_image - new_c_image).sum(dim=2)
    # Check if the difference is within the threshold (60) for all RGB channels
    within_threshold = (color_difference <= color_threshold) & valid_indices
    matches_count = within_threshold.sum()
    print(f'Number of matching colors within threshold: {matches_count}')

    if matches_count == 0: 
        breakpoint()
        return 1.0, new_points_3d, within_threshold

    gt_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')
    new_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')

    gt_image[gt_proj[:, 0], gt_proj[:, 1]] = select_gt_3d
    new_image[new_proj[:, 0], new_proj[:, 1]] = select_new_3d

    vis_images(gt_c_image, new_c_image)
    breakpoint()

    # TODO this is broken because the intrinsics are wrong so the "corresponding" points 
    # are no longer "corresponding"

    scale = None
    shift = None
    if align_mode == 'median':
        diff_3d = new_image[:, :, :] / gt_image[:, :, :]
        scale = diff_3d[within_threshold].median()
        new_3d_colors = new_colors
        new_3d_scaled = new_camera
        new_3d_scaled[:, :3] *= scale

    elif align_mode == 'o3d':
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(gt_image.view(-1, 3).cpu().numpy())
        source.colors = o3d.utility.Vector3dVector(gt_c_image.reshape(-1, 3).cpu().numpy())

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(new_image.view(-1, 3).cpu().numpy())
        target.colors = o3d.utility.Vector3dVector(new_c_image.reshape(-1, 3).cpu().numpy())

        o3d_indices = torch.where(within_threshold.view(-1))[0].tolist()

        # Align the point clouds based on the corresponding parts
        icp_result = align_partial_point_clouds(
            source, target, o3d_indices, o3d_indices, threshold=5.)
        scale = shift = icp_result.transformation

        # Apply the computed transformation to the entire source point cloud
        new_3d_scaled = o3d.geometry.PointCloud()
        new_3d_scaled.points = o3d.utility.Vector3dVector(new_camera[:, :3].cpu().numpy())
        new_3d_scaled.colors = o3d.utility.Vector3dVector(new_colors.cpu().numpy())
        new_3d_scaled = new_3d_scaled.transform(icp_result.transformation)

        new_3d_colors = torch.tensor(np.asarray(new_3d_scaled.colors)).to(torch.uint8).to('cuda')#[within_threshold.view(-1)]
        new_3d_scaled = torch.tensor(np.asarray(new_3d_scaled.points)).float().to('cuda')#[within_threshold.view(-1)]
        new_3d_scaled = torch.cat((new_3d_scaled, new_camera[:, -1].unsqueeze(1)), dim=1)

    elif align_mode == 'lstsq':
        # TODO use original selected points here
        _, scale, shift = fit_least_squares_shift_scale_with_mask(gt_image, new_image, within_threshold, within_threshold)
        new_3d_scaled = new_camera
        new_3d_scaled[:, :3] = new_3d[:, :3] * scale + shift
        new_3d_colors = new_colors
    else:
        new_3d_scaled = new_camera
        new_3d_colors = new_colors

    # --- from cam -> world ---
    new_3d_scaled = pts_cam_to_world(new_3d_scaled, extrinsics)

    print(f'Scale: {scale}, Shift: {shift}')
    return new_3d_scaled, new_3d_colors, within_threshold 
