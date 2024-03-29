#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def project_to_image(camera_coords, intrinsics, image_shape):
    # Projects 3D points onto a 2D image plane using the camera's extrinsic and intrinsic matrices.
    proj_pts = intrinsics @ camera_coords[:3, :]
    im_proj_pts = proj_pts[:2] / proj_pts[2]
    x_pixels = torch.round(im_proj_pts[0]).long()
    y_pixels = torch.round(im_proj_pts[1]).long()
    valid_mask = (x_pixels >= 0) & (x_pixels < image_shape[1]) & (y_pixels >= 0) & (y_pixels < image_shape[0])
    valid_x = x_pixels[valid_mask]
    valid_y = y_pixels[valid_mask]
    return torch.stack([valid_x, valid_y], dim=1), torch.arange(camera_coords.shape[-1], device=camera_coords.device)[valid_mask]

def project_and_scale_points_with_color(gt_points_3d, new_points_3d, gt_colors, new_colors, intrinsics, extrinsics, image_shape, color_threshold=30):
    gt_points_hg = torch.cat((gt_points_3d, torch.ones((gt_points_3d.shape[0], 1), device=gt_points_3d.device)), dim=1).T
    new_points_hg = torch.cat((new_points_3d, torch.ones((new_points_3d.shape[0], 1), device=new_points_3d.device)), dim=1).T

    gt_camera_coords = extrinsics @ gt_points_hg
    new_camera_coords = extrinsics @ new_points_hg

    gt_proj, gt_indices = project_to_image(gt_camera_coords, intrinsics, image_shape)
    new_proj, new_indices = project_to_image(new_camera_coords, intrinsics, image_shape)

    # Ensure unique projections within each point cloud
    # This also reorders the points and colors according to the unique operation
    gt_proj, gt_unique_indices = torch.unique(gt_proj, dim=0, return_inverse=True)
    new_proj, new_unique_indices = torch.unique(new_proj, dim=0, return_inverse=True)

    gt_colors = gt_colors[gt_indices][gt_unique_indices].float()
    new_colors = new_colors[new_indices][new_unique_indices].float()
    
    gt_3d = gt_camera_coords[:3].T[gt_indices][gt_unique_indices]
    new_3d = new_camera_coords[:3].T[new_indices][new_unique_indices]

    gt_proj = gt_proj[gt_unique_indices]
    new_proj = new_proj[new_unique_indices]

    # Initialize two tensors filled with -1 (indicating empty/invalid)
    gt_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')
    new_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')

    # Fill the tensors with the RGB colors at the specified coordinates
    gt_image[gt_proj[:, 0], gt_proj[:, 1]] = gt_colors
    new_image[new_proj[:, 0], new_proj[:, 1]] = new_colors

    # Find indices where both tensors have valid (non-empty) colors
    valid_gt = gt_image != -1
    valid_new = new_image != -1
    valid_indices = (valid_gt & valid_new).all(dim=2)  # Both have valid RGB colors

    # Calculate the difference between colors at valid indices
    color_difference = torch.abs(gt_image - new_image)
    color_difference = color_difference.sum(dim=2)
    # Check if the difference is within the threshold (30) for all RGB channels
    within_threshold = (color_difference <= 30) & valid_indices
    matches_count = within_threshold.sum()
    print(f'Number of matching colors within threshold: {matches_count}')

    if matches_count == 0: 
        return 1.0, new_points_3d

    gt_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')
    new_image = torch.full((512, 512, 3), -1, dtype=torch.float32, device='cuda:0')

    gt_image[gt_proj[:, 0], gt_proj[:, 1]] = gt_3d
    new_image[new_proj[:, 0], new_proj[:, 1]] = new_3d

    # Find indices where both tensors have valid (non-empty) colors
    valid_gt = gt_image != -1
    valid_new = new_image != -1

    diff_3d = gt_image[:, :, -1] / new_image[:, :, -1]
    median = diff_3d[within_threshold].median()
    new_points_3d_scaled = new_points_3d * median 
    print(f'Median: {median}')
    return median, new_points_3d_scaled
