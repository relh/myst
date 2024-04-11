import torch
import kornia as kn

def detect_edges_and_color_directionally(depth_map, low_threshold=0.3, high_threshold=0.3, num_pixels=4):
    """
    Detects edges using Canny edge detector, calculates gradient directions, 
    and colors additional pixels in the direction of the gradient on a depth map.
    
    Args:
    - depth_map (Tensor): Depth image tensor of shape ((H*W), 3).
    - low_threshold (float): Lower threshold for hysteresis in Canny edge detection.
    - high_threshold (float): Upper threshold for hysteresis in Canny edge detection.
    - num_pixels (int): Number of pixels to color in the direction of the edge.

    Returns:
    - Tensor: Binary mask indicating edges and colored pixels along the edges.
    """

    # Assuming depth_map needs reshaping and normalization
    depth_map = depth_map.reshape((512, 512, 3))[:, :, -1].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    depth_map = depth_map.float() / 255.0  # Normalize assuming the depth values are in [0, 255]

    # Apply Canny edge detector
    _, edges = kn.filters.canny(depth_map, low_threshold, high_threshold)

    # Calculate spatial gradients to get directions
    gradients = kn.filters.spatial_gradient(depth_map, mode='sobel', normalized=True).squeeze()
    grad_x, grad_y = gradients

    # Calculate gradient direction for each edge pixel
    grad_direction = torch.atan2(grad_y, grad_x).squeeze()  # Shape: (H, W)

    # Get coordinates of edge pixels
    edge_coords = torch.where(edges.squeeze() > 0)  # Removing unnecessary dimensions

    # Calculate offsets based on gradient direction
    dx = torch.round(torch.cos(grad_direction[edge_coords])).to(torch.long)
    dy = torch.round(torch.sin(grad_direction[edge_coords])).to(torch.long)

    # Prepare the output mask
    output_mask = edges.squeeze().clone()  # Shape: (H, W)

    # Color pixels in the direction
    H, W = output_mask.shape
    for i in range(1, num_pixels + 1):
        new_x = torch.clamp(edge_coords[1] + i * dx, 0, W - 1)
        new_y = torch.clamp(edge_coords[0] + i * dy, 0, H - 1)

        # Ensure we index valid positions only
        valid_indices = (new_x >= 0) & (new_x < W) & (new_y >= 0) & (new_y < H)
        output_mask[new_y[valid_indices], new_x[valid_indices]] = 1

    return output_mask
