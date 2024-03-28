#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def project_to_image(points_3d, intrinsics, image_shape):
    """
    Projects 3D points onto a 2D image plane using the camera's intrinsic matrix.
    Returns the projected 2D points and a mask indicating if points are within the image bounds.
    """
    # Homogeneous coordinates transformation
    ones = torch.ones((points_3d.shape[0], 1), device=points_3d.device)
    points_homogeneous = torch.cat((points_3d, ones), dim=1).T
    #projected_points = intrinsics @ points_homogeneous
    projected_points = intrinsics @ points_homogeneous[:3, :]
    
    # Normalize by depth and convert to pixel coordinates
    projected_points = projected_points[:2, :] / projected_points[2, :]
    x_pixels = torch.round(projected_points[0]).long()
    y_pixels = torch.round(projected_points[1]).long()

    # Create mask for points within image bounds
    valid_mask = (x_pixels >= 0) & (x_pixels < image_shape[1]) & \
                 (y_pixels >= 0) & (y_pixels < image_shape[0])
    
    return x_pixels[valid_mask], y_pixels[valid_mask], valid_mask

def project_and_scale_points_with_color(gt_points_3d, new_points_3d, gt_colors, new_colors, intrinsics, extrinsics, image_shape, color_threshold=30):
    # Project both point clouds to the image plane
    gt_x, gt_y, gt_mask = project_to_image(gt_points_3d, intrinsics, image_shape)
    new_x, new_y, new_mask = project_to_image(new_points_3d, intrinsics, image_shape)

    # Create a mask of valid points for both point clouds
    valid_mask = torch.zeros(image_shape, dtype=torch.bool)
    valid_mask[gt_y[gt_mask], gt_x[gt_mask]] = True
    valid_mask[new_y[new_mask], new_x[new_mask]] = True

    # Find the indices of the valid points in the original point clouds
    gt_indices = torch.arange(gt_points_3d.shape[0])[gt_mask]
    new_indices = torch.arange(new_points_3d.shape[0])[new_mask]

    # Create a tensor to store the projected points and their corresponding indices
    gt_projected = torch.stack((gt_x[gt_mask], gt_y[gt_mask], gt_indices), dim=1)
    new_projected = torch.stack((new_x[new_mask], new_y[new_mask], new_indices), dim=1)

    # Combine the projected points and find unique collisions
    combined_projected = torch.cat((gt_projected, new_projected), dim=0)
    unique_projected, counts = torch.unique(combined_projected[:, :2], dim=0, return_counts=True)

    # Filter the collisions based on count and create a mask
    collision_mask = counts > 1
    collision_points = unique_projected[collision_mask]

    # Extract the indices of the colliding points in the original point clouds
    gt_collision_indices = collision_points[:, 2].long()
    new_collision_indices = collision_points[:, 2].long() - gt_projected.shape[0]

    # Extract the colors of the colliding points
    gt_collision_colors = gt_colors[gt_collision_indices]
    new_collision_colors = new_colors[new_collision_indices]

    # Compute the color differences and create a mask for valid correspondences
    color_diffs = torch.norm(gt_collision_colors - new_collision_colors, dim=1)
    valid_correspondences_mask = color_diffs < color_threshold

    # Extract the valid corresponding 3D points
    gt_valid_points = gt_points_3d[gt_collision_indices[valid_correspondences_mask]]
    new_valid_points = new_points_3d[new_collision_indices[valid_correspondences_mask]]

    # Compute the median scaling factor
    gt_depths = gt_valid_points[:, 2]
    new_depths = new_valid_points[:, 2]

    # Calculate scale factors for each valid correspondence based on depth ratios
    scale_factors = gt_depths / (new_depths + 1e-6)

    # Compute the median of these scale factors
    median_scale = torch.median(scale_factors)

    # Apply the median scale to adjust the new point cloud
    new_points_3d_scaled = new_points_3d * median_scale

    return median_scale, new_points_3d_scaled
