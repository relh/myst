import kornia as kn
import open3d as o3d
import numpy as np
import torch

def density_pruning(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    #pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())

    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return points[ind], colors[ind]

def realign_depth_edges(pts_3d, rgb_3d, low_threshold=0.3, high_threshold=0.3, num_pixels=10):
    H, W = 512, 512

    pts_3d = pts_3d.reshape((H, W, 3))
    rgb_image = rgb_3d.reshape((H, W, 3)).unsqueeze(0) # Shape: (1, 3, H, W)

    depth_map = pts_3d[:, :, -1].clone().float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    og_depth_map = depth_map.clone()
    depth_map = depth_map / 255.0  # Normalize assuming the depth values are in [0, 255]

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
    og_x = torch.clamp(edge_coords[1], 0, W - 1)
    og_y = torch.clamp(edge_coords[0], 0, H - 1)
    end_x = torch.clamp(edge_coords[1] + num_pixels * dx, 0, W - 1)
    end_y = torch.clamp(edge_coords[0] + num_pixels * dy, 0, H - 1)
    og_color = rgb_image.squeeze()[og_y, og_x].float()
    og_depth = og_depth_map.squeeze()[og_y, og_x].float()
    end_color = rgb_image.squeeze()[end_y, end_x].float()
    end_depth = og_depth_map.squeeze()[end_y, end_x].float()

    # Color pixels in the direction
    for i in range(1, num_pixels + 1):
        new_x = torch.clamp(edge_coords[1] + i * dx, 0, W - 1)
        new_y = torch.clamp(edge_coords[0] + i * dy, 0, H - 1)

        #new_depth = og_depth_map.squeeze()[new_y, new_x].float()
        new_color = rgb_image.squeeze()[new_y, new_x].float()
        new_to_og = abs(new_color - og_color).sum(dim=1)
        new_to_end = abs(new_color - end_color).sum(dim=1)
        depth = torch.where(new_to_og > new_to_end, end_depth, og_depth)
        ratio_depth = depth / pts_3d[new_y, new_x, -1]
        pts_3d[new_y, new_x] = pts_3d[new_y, new_x] * ratio_depth.unsqueeze(1)

    return pts_3d.reshape(-1, 3), edges.squeeze()
    #return ~(output_mask.bool().view(-1))
