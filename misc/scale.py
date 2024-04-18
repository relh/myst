#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import torch


def project_to_image(camera_coords, intrinsics, image_shape):
    # Projects 3D points onto a 2D image plane using the camera's extrinsic and intrinsic matrices.
    proj_pts = intrinsics @ camera_coords[:3, :]
    im_proj_pts = proj_pts[:2] / proj_pts[2]
    x_pixels, y_pixels = torch.round(im_proj_pts).long()
    valid_mask = (x_pixels >= 0) & (x_pixels < image_shape[1]) & (y_pixels >= 0) & (y_pixels < image_shape[0])
    return torch.stack([x_pixels[valid_mask], y_pixels[valid_mask]], dim=1), torch.arange(camera_coords.shape[-1], device=camera_coords.device)[valid_mask]

def world_to_filtered(gt_points_3d, gt_colors, intrinsics, extrinsics, image_shape):
    #gt_points_3d = gt_points_3d.clone().floor()
    gt_camera_coords = extrinsics @ torch.cat((gt_points_3d, torch.ones((gt_points_3d.shape[0], 1), device=gt_points_3d.device)), dim=1).T
    gt_proj, gt_indices = project_to_image(gt_camera_coords, intrinsics, image_shape)
    gt_proj, gt_unique_indices = torch.unique(gt_proj, dim=0, return_inverse=True)
    gt_colors = gt_colors[gt_indices][gt_unique_indices].float()
    gt_3d = gt_points_3d[gt_indices][gt_unique_indices]
    gt_proj = gt_proj[gt_unique_indices]
    return gt_proj, gt_colors, gt_camera_coords.T, gt_3d

def estimate_scale_and_shift(gt_depths, new_depths):
    A = torch.vstack((new_depths, torch.ones_like(new_depths))).T  # Transpose to get the correct shape
    
    # Use torch.linalg.lstsq for the least squares solution
    # Note the reversed order of A and gt_depths compared to the deprecated torch.lstsq
    result = torch.linalg.lstsq(A, gt_depths.unsqueeze(1))

    solution = result.solution
    scale, shift = solution.squeeze()  # Extract scale (s) and shift (b), and remove extra dimensions
    return scale.item(), shift.item()

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


def fit_least_squares_shift_scale_with_mask(pc1, pc2, mask1, mask2):
    """
    Fit a least squares shift and scale transformation from pc2 to pc1 using provided masks.
    pc1, pc2 are (N, 3) tensors representing the point clouds.
    mask1, mask2 are boolean tensors indicating corresponding points in pc1 and pc2.
    """
    # Apply masks to select corresponding points
    sub_pc1 = pc1[mask1]
    sub_pc2 = pc2[mask2]

    # Compute the initial RMSE before transformation
    initial_rmse = torch.sqrt(torch.mean((sub_pc1 - sub_pc2)**2))
    print(f"Initial RMSE: {initial_rmse.item()}")

    # Compute centroids of the selected subsets
    #centroid1 = torch.mean(sub_pc1, dim=0)
    #centroid2 = torch.mean(sub_pc2, dim=0)

    # Compute the scale factor as the ratio of norms of point cloud subsets
    norm_pc1 = torch.norm(sub_pc1)# - centroid1)
    norm_pc2 = torch.norm(sub_pc2)# - centroid2)
    scale_factor = norm_pc1 / norm_pc2

    # Scale the corresponding part of the second point cloud
    scaled_pc2 = (sub_pc2) * scale_factor 

    # Compute the translation needed after scaling
    translation = -torch.mean(scaled_pc2, dim=0)

    # Apply the transformation to the entire second point cloud
    transformed_pc2 = (pc2) * scale_factor + translation

    # Compute RMSE after the transformation
    transformed_rmse = torch.sqrt(torch.mean((transformed_pc2[mask2] - pc1[mask1])**2))
    print(f"Transformed RMSE: {transformed_rmse.item()}")

    return transformed_pc2, scale_factor, translation


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

    # Apply the computed transformation to the entire source point cloud
    transformed_source = source.transform(icp_result.transformation)
    
    return transformed_source, icp_result

def project_and_scale_points_with_color(gt_points_3d, new_points_3d, gt_colors, new_colors, intrinsics, extrinsics, image_shape, color_threshold=30):
    gt_proj, gt_colors, gt_3d, select_gt_3d = world_to_filtered(gt_points_3d, gt_colors, intrinsics, extrinsics, image_shape)
    new_proj, new_colors, new_3d, select_new_3d = world_to_filtered(new_points_3d, new_colors, intrinsics, extrinsics, image_shape)

    # Initialize two tensors filled with -1 (indicating empty/invalid)
    gt_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')
    new_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')

    # Fill the tensors with the RGB colors at the specified coordinates
    gt_image[gt_proj[:, 1], gt_proj[:, 0]] = gt_colors
    new_image[new_proj[:, 1], new_proj[:, 0]] = new_colors

    # Find indices where both tensors have valid (non-empty) colors
    valid_indices = ((gt_image != -1) & (new_image != -1)).all(dim=2)  # Both have valid RGB colors

    # Calculate the difference between colors at valid indices
    color_difference = torch.abs(gt_image - new_image).sum(dim=2)

    # Check if the difference is within the threshold (60) for all RGB channels
    within_threshold = (color_difference <= 60) & valid_indices
    matches_count = within_threshold.sum()
    print(f'Number of matching colors within threshold: {matches_count}')

    if matches_count == 0: 
        return 1.0, new_points_3d, within_threshold

    gt_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')
    new_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')

    gt_image[gt_proj[:, 1], gt_proj[:, 0]] = select_gt_3d
    new_image[new_proj[:, 1], new_proj[:, 0]] = select_new_3d

    extrinsics_inv = torch.linalg.pinv(extrinsics)
    scale = None
    shift = None
    align_mode = 'median'
    if align_mode == 'median':
        diff_3d = gt_image[:, :, :] / new_image[:, :, :]
        scale = diff_3d[within_threshold].median()
        new_3d_scaled = new_points_3d * scale 
    elif align_mode == 'use_o3d':
        # Load the point clouds
        #source = o3d.io.read_point_cloud("path_to_source_point_cloud.ply")
        #target = o3d.io.read_point_cloud("path_to_target_point_cloud.ply")

        # Define the masks for the corresponding parts
        #source_mask = [0, 2, 3, 5]  # example indices of corresponding points
        #target_mask = [1, 3, 4, 6]  # example indices of corresponding points

        # Align the point clouds based on the corresponding parts
        transformed_source, icp_result = align_partial_point_clouds(
            source, target, source_mask, target_mask, threshold=0.5)
        print("Transformation is:")
        print(icp_result.transformation)

        # Optionally, visualize the result
        o3d.visualization.draw_geometries([transformed_source, target])
    elif align_mode == 'lstsq':
        # TODO use original selected points here
        _, scale, shift = fit_least_squares_shift_scale_with_mask(gt_image, new_image, within_threshold, within_threshold)
        new_3d_scaled = new_points_3d 
        new_3d_scaled[:, :3] = new_3d[:, :3] * scale + shift
    print(f'Scale: {scale}, Shift: {shift}')

    #breakpoint()
    #points_world_homogeneous = torch.matmul(new_3d_scaled, extrinsics_inv.T)
    #points_world = points_world_homogeneous[:, :3] / points_world_homogeneous[:, 3:]
    points_world = new_3d_scaled

    # TODO FIX within_threshold
    return scale, points_world[:, :3], within_threshold 
