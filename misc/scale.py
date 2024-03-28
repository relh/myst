#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def project_to_image(points_3d, intrinsics, extrinsics, image_shape):
    """
    Projects 3D points onto a 2D image plane using the camera's extrinsic and intrinsic matrices.
    Returns the projected 2D points, their original indices, and a mask indicating if points are within the image bounds.
    """
    # Transform points to camera coordinates
    points_homogeneous = torch.cat((points_3d, torch.ones((points_3d.shape[0], 1), device=points_3d.device)), dim=1).T
    camera_coords = extrinsics @ points_homogeneous

    # Project points using the intrinsic matrix
    proj_pts = intrinsics @ camera_coords[:3, :]
    im_proj_pts = proj_pts[:2] / proj_pts[2]

    # Convert to pixel coordinates and filter based on image bounds
    x_pixels = torch.round(im_proj_pts[0]).long()
    y_pixels = torch.round(im_proj_pts[1]).long()
    valid_mask = (x_pixels >= 0) & (x_pixels < image_shape[1]) & (y_pixels >= 0) & (y_pixels < image_shape[0])

    # Filter out invalid points
    valid_x = x_pixels[valid_mask]
    valid_y = y_pixels[valid_mask]
    valid_indices = torch.arange(points_3d.shape[0], device=points_3d.device)[valid_mask]

    return torch.stack([valid_x, valid_y], dim=1), valid_indices

def composite_key_for_projections(projections, image_shape):
    """
    Generate a composite key for each projection to facilitate finding unique projections and collisions.
    This assumes projections are already filtered to be within image bounds.
    """
    return projections[:, 0] + projections[:, 1] * image_shape[1]

def find_shared_projections_and_compare_colors(gt_proj, new_proj, gt_indices, new_indices, gt_colors, new_colors, color_threshold):
    """
    Identifies shared projections between two point clouds and compares colors of corresponding points.
    Uses efficient operations for GPU execution.
    """
    # Combine projections and find unique projections
    combined_proj = torch.cat([gt_proj, new_proj], dim=0)
    unique_proj, inverse_indices, counts = torch.unique(combined_proj, dim=0, return_inverse=True, return_counts=True)

    # Identify projections that appear in both point clouds (shared projections)
    shared_mask = counts > 1
    shared_proj = unique_proj[shared_mask]

    # Find indices of the first occurrence of shared projections in each point cloud
    gt_shared_indices = torch.where(shared_mask[:gt_proj.shape[0]])[0]
    new_shared_indices = torch.where(shared_mask[gt_proj.shape[0]:])[0]

    # Get colors of shared projections
    gt_shared_colors = gt_colors[gt_indices[gt_shared_indices]]
    new_shared_colors = new_colors[new_indices[new_shared_indices]]

    # Ensure gt_shared_colors and new_shared_colors have the same length
    min_length = min(gt_shared_colors.shape[0], new_shared_colors.shape[0])
    gt_shared_colors = gt_shared_colors[:min_length]
    new_shared_colors = new_shared_colors[:min_length]

    # Compare colors and find valid correspondences
    color_diffs = torch.norm(gt_shared_colors - new_shared_colors, dim=1)
    valid_correspondences_mask = color_diffs < color_threshold

    # Filter indices based on valid correspondences
    valid_gt_indices = gt_indices[gt_shared_indices[valid_correspondences_mask]]
    valid_new_indices = new_indices[new_shared_indices[valid_correspondences_mask]]

    return valid_gt_indices, valid_new_indices

def project_and_scale_points_with_color(gt_points_3d, new_points_3d, gt_colors, new_colors, intrinsics, extrinsics, image_shape, color_threshold=30):
    """
    Projects 3D points to 2D, identifies shared projections, and compares colors to find valid matches efficiently.
    """
    # Project points to 2D space
    gt_proj, gt_indices = project_to_image(gt_points_3d, intrinsics, extrinsics, image_shape)
    new_proj, new_indices = project_to_image(new_points_3d, intrinsics, extrinsics, image_shape)

    # Find shared projections and compare colors
    valid_gt_indices, valid_new_indices = find_shared_projections_and_compare_colors(
        gt_proj, new_proj, gt_indices, new_indices, gt_colors, new_colors, color_threshold
    )

    if len(valid_gt_indices) == 0 or len(valid_new_indices) == 0:
        return torch.tensor(1.0), new_points_3d  # No scaling if no matches

    # Compute scaling factor and apply to new point cloud
    scale_factors = gt_points_3d[valid_gt_indices][:, 2] / (new_points_3d[valid_new_indices][:, 2] + 1e-6)
    median_scale = torch.median(scale_factors)
    new_points_3d_scaled = new_points_3d * median_scale

    return median_scale, new_points_3d_scaled
