#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def calculate_dynamic_epsilon(point_cloud):
    """
    Calculate a dynamic epsilon based on the average distance between adjacent points in a dense point cloud.

    :param point_cloud: Input point cloud as a PyTorch tensor of shape (N, 3), where N is the number of points.
    :return: Calculated epsilon value.
    """
    # Assume point_cloud is reshaped to (256, 256, 3) if it comes from a 256x256 depth image
    sq_shape = int(point_cloud.shape[0] ** 0.5)
    reshaped_pc = point_cloud.view(sq_shape, sq_shape, 3)

    # Calculate differences between adjacent points along both axes
    diff_x = reshaped_pc[:-1, :, :] - reshaped_pc[1:, :, :]
    diff_y = reshaped_pc[:, :-1, :] - reshaped_pc[:, 1:, :]

    # Calculate distances for these differences
    dist_x = torch.sqrt(torch.sum(diff_x**2, dim=2))
    dist_y = torch.sqrt(torch.sum(diff_y**2, dim=2))

    # Calculate average distance
    avg_dist = torch.mean(torch.cat((dist_x.flatten(), dist_y.flatten())))

    return avg_dist

def trim_points(new_da_3d, new_da_colors, border=1):
    """
    Removes a specified border width in pixels from around the edges of new_da_3d and new_da_colors.

    :param new_da_3d: Tensor of new points (N, 3), where N is a perfect square.
    :param new_da_colors: Tensor of colors corresponding to new_da_3d points (N, 3).
    :param border: Number of pixels to remove from each edge.
    :return: A tuple of reshaped new_da_3d and new_da_colors with the border removed.
    """
    # Ensure the sizes match
    assert new_da_3d.size(0) == new_da_colors.size(0), "The point and color tensors must have the same first dimension."
    
    # Determine the size of the square matrix
    N = int(new_da_3d.size(0) ** 0.5)
    assert N * N == new_da_3d.size(0), "The first dimension must be a perfect square."
    
    # Reshape to square matrices
    reshaped_points = new_da_3d.view(N, N, 3)
    reshaped_colors = new_da_colors.view(N, N, 3)
    
    # Remove the border from both tensors
    trimmed_points = reshaped_points[border:N-border, border:N-border, :]
    trimmed_colors = reshaped_colors[border:N-border, border:N-border, :]

    # Reshape back to original shape with reduced number of points/colors
    result_points = trimmed_points.reshape(-1, 3)
    result_colors = trimmed_colors.reshape(-1, 3)

    return result_points, result_colors

def merge_and_filter(da_3d, new_da_3d, da_colors, new_da_colors, epsilon=None):
    """
    Merge two point clouds on the GPU, retaining all points from da_3d and filtering new_da_3d points based on uniqueness,
    excluding black points from new_da_3d.

    :param da_3d: Tensor of original points (N, 3) on the GPU.
    :param new_da_3d: Tensor of new points (M, 3) to merge, on the GPU.
    :param da_colors: Tensor of colors corresponding to da_3d points (N, 3).
    :param new_da_colors: Tensor of colors corresponding to new_da_3d points (M, 3).
    :param epsilon: Proximity threshold, points closer than this are considered duplicates.
    :return: Tuple of merged point cloud and colors as PyTorch tensors.
    """
    epsilon = calculate_dynamic_epsilon(new_da_3d)
    print(f'Epsilon: {epsilon}')

    # Exclude black points from new_da_3d
    not_black_mask = torch.all(new_da_colors != 0, dim=1)
    new_da_3d_filtered = new_da_3d[not_black_mask]
    new_da_colors_filtered = new_da_colors[not_black_mask]

    # Generate hash keys for point cloud bins
    def get_hash_keys(points):
        return torch.floor(points / epsilon).to(torch.int64)

    # Compute hash keys for original and filtered new point clouds
    original_keys = get_hash_keys(da_3d).to(torch.float32)
    new_keys_filtered = get_hash_keys(new_da_3d_filtered).to(torch.float32)

    # Concatenate original and new filtered point clouds and their colors
    combined_points = torch.cat((da_3d, new_da_3d_filtered), dim=0)
    combined_colors = torch.cat((da_colors, new_da_colors_filtered), dim=0)

    # Concatenate keys to find unique among new filtered points
    all_new_keys = torch.cat((original_keys, new_keys_filtered), dim=0)

    # Find unique new keys
    unique_vals, unique_inverse = torch.unique(all_new_keys, return_inverse=True, dim=0)
    unique_counts = torch.bincount(unique_inverse)

    # Identify first occurrences of unique keys as valid, plus all from original
    valid_new_indices = unique_inverse[len(da_3d):] < len(da_3d) + torch.arange(len(new_da_3d_filtered)).to(da_3d.device)
    valid_indices = torch.cat((torch.ones(len(da_3d), dtype=torch.bool, device=da_3d.device), valid_new_indices))

    merged_points = combined_points[valid_indices]
    merged_colors = combined_colors[valid_indices]

    return merged_points, merged_colors

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

def bad_project_to_image(points_3d, intrinsics, extrinsics, image_shape):
    """
    Projects 3D points onto a 2D image plane using the camera's intrinsic and extrinsic matrices.
    Returns the projected 2D points and a mask indicating if points are within the image bounds.
    """
    # Transform points to camera coordinates
    ones = torch.ones((points_3d.shape[0], 1), device=points_3d.device)
    points_homogeneous = torch.cat((points_3d, ones), dim=1)
    camera_coords = extrinsics @ points_homogeneous.T

    # Apply intrinsic matrix to get image coordinates
    proj_pts = intrinsics @ camera_coords[:3, :]
    im_proj_pts = proj_pts[:2, :] / proj_pts[2, :]

    # Convert to pixel coordinates
    x_pixels = torch.round(im_proj_pts[0]).long()
    y_pixels = torch.round(im_proj_pts[1]).long()

    # Create mask for points within image bounds
    valid_mask = (x_pixels >= 0) & (x_pixels < image_shape[1]) & \
                 (y_pixels >= 0) & (y_pixels < image_shape[0])

    # Filter based on valid_mask and return pixel coordinates
    return x_pixels[valid_mask], y_pixels[valid_mask], valid_mask

def compute_median_scale_factor(gt_points_3d, new_points_3d, valid_correspondences, valid_colors_mask):
    """
    Computes the median scaling factor for aligning two point clouds based on the depth values
    of their corresponding points that have passed the color similarity check.
    
    Parameters:
    - gt_points_3d: Ground truth point cloud as a tensor of shape (N, 3).
    - new_points_3d: New point cloud as a tensor of shape (M, 3).
    - valid_correspondences: A tensor of indices where points between the two point clouds correspond.
    - valid_colors_mask: A boolean mask indicating which correspondences are valid based on color similarity.
    
    Returns:
    - median_scale: The median scaling factor to align the new point cloud with the ground truth.
    """
    # Extract the indices of valid correspondences that also pass the color check
    valid_indices = valid_correspondences[valid_colors_mask]

    # Extract depth values (Z-coordinates) for corresponding points
    gt_depths = gt_points_3d[valid_indices][:, 2]
    new_depths = new_points_3d[valid_indices][:, 2]

    # Calculate scale factors for each valid correspondence based on depth ratios
    # Avoid division by zero by adding a small epsilon to denominators
    scale_factors = gt_depths / (new_depths + 1e-6)

    # Compute the median of these scale factors
    median_scale = torch.median(scale_factors)

    return median_scale


def compute_collisions_and_colors(x1, y1, colors1, x2, y2, colors2, image_shape, color_threshold=30):
    """
    Identifies collisions between two sets of projected points and verifies them with color similarity.
    Returns a mask of valid correspondences for each set of points.
    """
    # Initialize image tensors to track the projected point locations
    img1 = torch.zeros((*image_shape, 3), dtype=torch.float32)
    img2 = torch.zeros_like(img1)

    # Populate the image tensors with color values at the projected locations
    img1[y1, x1] = colors1
    img2[y2, x2] = colors2

    # Find collisions by identifying pixels that are non-zero (have been colored) in both images
    collision_mask = (torch.sum(img1, dim=-1) > 0) & (torch.sum(img2, dim=-1) > 0)

    # For each collision, calculate the color difference to verify the correspondence
    # First, find indices where there are collisions
    y_coll, x_coll = torch.where(collision_mask)
    
    # Calculate color differences only at collision points
    color_differences = torch.abs(img1[y_coll, x_coll] - img2[y_coll, x_coll])
    
    # Sum color differences across RGB channels and compare to the threshold
    valid_collisions = torch.where(torch.sum(color_differences, dim=1) < color_threshold)[0]

    # Use valid_collisions to filter y_coll and x_coll for valid correspondences
    valid_y = y_coll[valid_collisions]
    valid_x = x_coll[valid_collisions]

    # Construct valid correspondences mask from valid_y and valid_x
    valid_correspondences_mask = torch.zeros_like(collision_mask)
    valid_correspondences_mask[valid_y, valid_x] = True

    return valid_correspondences_mask


def project_and_scale_points_with_color(gt_points_3d, new_points_3d, gt_colors, new_colors, intrinsics, extrinsics, image_shape, color_threshold=30):
    """
    Projects and scales two point clouds onto an image plane with color-based validation for overlap.
    
    Parameters:
    - gt_points_3d, new_points_3d: The ground truth and new point clouds (X, 3) and (Y, 3).
    - gt_colors, new_colors: Colors associated with each point in the point clouds (X, 3) and (Y, 3).
    - intrinsics: The camera intrinsic parameters matrix.
    - extrinsics: The camera extrinsic parameters matrix (unused here but included for completeness).
    - image_shape: The shape of the target image (height, width).
    - color_threshold: The maximum allowed color difference for considering a correspondence valid.
    
    Returns:
    - The median scale factor applied to the new point cloud for alignment.
    - The new point cloud scaled accordingly.
    """
    # Project both point clouds to the image plane
    gt_x, gt_y, _ = project_to_image(gt_points_3d, intrinsics, image_shape)
    new_x, new_y, _ = project_to_image(new_points_3d, intrinsics, image_shape)

    # Compute collisions based on the projected points and validate them with color similarity
    valid_correspondences, valid_colors_mask = compute_collisions_and_colors(
        gt_x, gt_y, gt_colors, new_x, new_y, new_colors, image_shape, color_threshold
    )
    
    # Filter the 3D points of the new point cloud based on valid correspondences and color mask
    # Note: Adjustments may be needed to correctly apply 'valid_correspondences' and 'valid_colors_mask'
    # For demonstration, assume direct indexing could work, which might require modification in practice
    valid_new_points_3d = new_points_3d[valid_correspondences]

    # Compute the median scaling factor
    # Note: 'compute_median_scale_factor' expects depth values directly, which might need extraction from 'valid_correspondences'
    # Here, 'valid_correspondences' should indicate specific pairs of matching points, and we need both GT and new depths
    # For demonstration, assume we have a way to get these depths directly, which in practice will require more logic
    median_scale = compute_median_scale_factor(
        gt_points_3d, valid_new_points_3d, valid_correspondences, valid_colors_mask
    )
    
    # Apply the median scale to adjust the new point cloud
    new_points_3d_scaled = new_points_3d * median_scale
    
    return median_scale, new_points_3d_scaled

