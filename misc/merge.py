#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from pytorch3d.ops import knn_points


def calculate_dynamic_epsilon(point_cloud):
    """
    Calculate a dynamic epsilon based on the average distance to the four nearest neighbors in a point cloud.
    This version uses PyTorch3D for GPU acceleration.
    
    :param point_cloud: Input point cloud as a PyTorch tensor of shape (N, 3), where N is the number of points.
    :return: Calculated epsilon value.
    """
    # Ensure the point cloud is on the appropriate device (e.g., GPU)
    point_cloud = point_cloud.to(device='cuda')  # Move data to GPU if available

    # Calculate the 5 nearest neighbors using PyTorch3D (including itself)
    # We use k=5 to get four neighbors plus the point itself
    neighbors = knn_points(point_cloud[None, :, :], point_cloud[None, :, :], K=5, return_sorted=False)
    
    # Compute distances to the nearest neighbors, shape is (1, N, 5)
    distances = neighbors.dists.squeeze(0)  # Remove batch dimension
    
    # Remove the distance to itself (which is the smallest, and hence the first)
    valid_distances = distances[:, 1:]  # Skip the first column, which is distance to itself
    
    # Compute the average distance to the four nearest neighbors
    avg_distance = torch.mean(valid_distances)
    
    return avg_distance.item()


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

def merge_and_filter(da_3d, new_da_3d, da_colors, new_da_colors, epsilon=5.0):
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
    if epsilon is None: 
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
    torch.bincount(unique_inverse)

    # Identify first occurrences of unique keys as valid, plus all from original
    valid_new_indices = unique_inverse[len(da_3d):] < len(da_3d) + torch.arange(len(new_da_3d_filtered)).to(da_3d.device)
    valid_indices = torch.cat((torch.ones(len(da_3d), dtype=torch.bool, device=da_3d.device), valid_new_indices))

    merged_points = combined_points[valid_indices]
    merged_colors = combined_colors[valid_indices]

    return merged_points, merged_colors
