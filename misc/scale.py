#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def project_to_image(camera_coords, intrinsics, image_shape):
    # Projects 3D points onto a 2D image plane using the camera's extrinsic and intrinsic matrices.
    proj_pts = intrinsics @ camera_coords[:3, :]
    im_proj_pts = proj_pts[:2] / proj_pts[2]
    x_pixels, y_pixels = torch.round(im_proj_pts).long()
    valid_mask = (x_pixels >= 0) & (x_pixels < image_shape[1]) & (y_pixels >= 0) & (y_pixels < image_shape[0])
    return torch.stack([x_pixels[valid_mask], y_pixels[valid_mask]], dim=1), torch.arange(camera_coords.shape[-1], device=camera_coords.device)[valid_mask]

def world_to_filtered(gt_points_3d, gt_colors, intrinsics, extrinsics, image_shape):
    gt_camera_coords = extrinsics @ torch.cat((gt_points_3d, torch.ones((gt_points_3d.shape[0], 1), device=gt_points_3d.device)), dim=1).T
    gt_proj, gt_indices = project_to_image(gt_camera_coords, intrinsics, image_shape)
    gt_proj, gt_unique_indices = torch.unique(gt_proj, dim=0, return_inverse=True)
    gt_colors = gt_colors[gt_indices][gt_unique_indices].float()
    gt_3d = gt_camera_coords[:3].T[gt_indices][gt_unique_indices]
    gt_proj = gt_proj[gt_unique_indices]
    return gt_proj, gt_colors, gt_camera_coords.T, gt_3d

def estimate_scale_and_shift(gt_depths, new_depths):
    # Ensure inputs are 1D tensors and of the same length
    assert gt_depths.ndim == 1 and new_depths.ndim == 1
    assert gt_depths.size(0) == new_depths.size(0)
    
    # Construct the design matrix for the linear model Z = s*Z + b
    A = torch.stack([new_depths, torch.ones_like(new_depths)], dim=1)
    
    # Use torch.linalg.lstsq for the least squares solution
    # Note the reversed order of A and gt_depths compared to the deprecated torch.lstsq
    result = torch.linalg.lstsq(A, gt_depths.unsqueeze(1))
    solution = result.solution
    
    scale, shift = solution.squeeze()  # Extract scale (s) and shift (b), and remove extra dimensions
    return scale.item(), shift.item()

def project_and_scale_points_with_color(gt_points_3d, new_points_3d, gt_colors, new_colors, intrinsics, extrinsics, image_shape, color_threshold=30):
    gt_proj, gt_colors, gt_3d, select_gt_3d = world_to_filtered(gt_points_3d, gt_colors, intrinsics, extrinsics, image_shape)
    new_proj, new_colors, new_3d, select_new_3d = world_to_filtered(new_points_3d, new_colors, intrinsics, extrinsics, image_shape)

    # Initialize two tensors filled with -1 (indicating empty/invalid)
    gt_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')
    new_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')

    # Fill the tensors with the RGB colors at the specified coordinates
    gt_image[gt_proj[:, 0], gt_proj[:, 1]] = gt_colors
    new_image[new_proj[:, 0], new_proj[:, 1]] = new_colors

    # Find indices where both tensors have valid (non-empty) colors
    valid_indices = ((gt_image != -1) & (new_image != -1)).all(dim=2)  # Both have valid RGB colors

    # Calculate the difference between colors at valid indices
    color_difference = torch.abs(gt_image - new_image).sum(dim=2)

    # Check if the difference is within the threshold (30) for all RGB channels
    within_threshold = (color_difference <= 30) & valid_indices
    matches_count = within_threshold.sum()
    print(f'Number of matching colors within threshold: {matches_count}')

    if matches_count == 0: 
        return 1.0, new_points_3d

    gt_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')
    new_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')

    gt_image[gt_proj[:, 0], gt_proj[:, 1]] = select_gt_3d
    new_image[new_proj[:, 0], new_proj[:, 1]] = select_new_3d

    median = True
    if median:
        diff_3d = gt_image[:, :, :] / new_image[:, :, :]
        median = diff_3d[within_threshold].median()
        print(f'Median: {median}')
        new_3d_scaled = new_3d * median 
    else:
        # Get depth values of matched points
        gt_depths = gt_image[within_threshold][2]
        new_depths = new_image[within_threshold][2]
        
        # Estimate scale and shift
        scale, shift = estimate_scale_and_shift(gt_depths, new_depths)
        print(f'Scale: {scale}, Shift: {shift}')
        
        # Apply scale and shift to the new_points' depth
        new_3d_scaled = new_3d.clone()  # Clone to avoid modifying the original
        new_3d_scaled[:, 2] = new_3d[:, 2] * scale + shift

    extrinsics_inv = torch.linalg.pinv(extrinsics)
    points_world_homogeneous = torch.matmul(new_3d_scaled, extrinsics_inv.T)
    points_world = points_world_homogeneous[:, :3] / points_world_homogeneous[:, 3:]
    return median, points_world[:, :3] 
