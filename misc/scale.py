import torch

def project_to_image(points_3d, intrinsics, extrinsics, image_shape):
    """
    Projects 3D points onto a 2D image plane using the camera's extrinsic and intrinsic matrices.
    """
    points_homogeneous = torch.cat((points_3d, torch.ones((points_3d.shape[0], 1), device=points_3d.device)), dim=1).T
    camera_coords = extrinsics @ points_homogeneous
    proj_pts = intrinsics @ camera_coords[:3, :]
    im_proj_pts = proj_pts[:2] / proj_pts[2]

    x_pixels = torch.round(im_proj_pts[0]).long()
    y_pixels = torch.round(im_proj_pts[1]).long()
    valid_mask = (x_pixels >= 0) & (x_pixels < image_shape[1]) & (y_pixels >= 0) & (y_pixels < image_shape[0])

    valid_x = x_pixels[valid_mask]
    valid_y = y_pixels[valid_mask]

    return torch.stack([valid_x, valid_y], dim=1), torch.arange(points_3d.shape[0], device=points_3d.device)[valid_mask]

def composite_key_for_projections(projections, image_shape):
    """
    Generate a composite key for each projection.
    """
    return projections[:, 0] + projections[:, 1] * image_shape[1]

def project_and_scale_points_with_color(gt_points_3d, new_points_3d, gt_colors, new_colors, intrinsics, extrinsics, image_shape, color_threshold=30):
    gt_proj, gt_indices = project_to_image(gt_points_3d, intrinsics, extrinsics, image_shape)
    new_proj, new_indices = project_to_image(new_points_3d, intrinsics, extrinsics, image_shape)

    gt_keys = composite_key_for_projections(gt_proj, image_shape)
    new_keys = composite_key_for_projections(new_proj, image_shape)

    all_keys = torch.cat([gt_keys, new_keys])
    sorted_keys, sorted_indices = torch.sort(all_keys)

    duplicates = torch.roll(sorted_keys, shifts=-1) == sorted_keys
    duplicates[-1] = False  # Exclude the last comparison which is invalid

    shared_indices = sorted_indices[duplicates]

    # Split back into original sets
    gt_shared = shared_indices[shared_indices < len(gt_keys)]
    new_shared = shared_indices[shared_indices >= len(gt_keys)] - len(gt_keys)

    if len(gt_shared) == 0 or len(new_shared) == 0:
        print('broke!')
        return torch.tensor(1.0, device=gt_points_3d.device), new_points_3d

    # Find the common shared indices
    common_shared_indices = torch.tensor([i for i in gt_shared if i in new_shared], device=gt_points_3d.device)

    if len(common_shared_indices) == 0:
        print('broke!')
        return torch.tensor(1.0, device=gt_points_3d.device), new_points_3d

    # Ensure color tensors are float for norm calculation
    gt_colors_float = gt_colors[gt_indices].float()
    new_colors_float = new_colors[new_indices].float()

    # Compute color differences for common shared projections and filter based on threshold
    color_diffs = torch.norm(gt_colors_float[common_shared_indices] - new_colors_float[common_shared_indices], dim=1)
    valid_mask = color_diffs < color_threshold

    valid_gt_indices = gt_indices[common_shared_indices][valid_mask]
    valid_new_indices = new_indices[common_shared_indices][valid_mask]

    # Compute scaling factor based on depth (z-coordinate) of valid matching points
    if len(valid_gt_indices) > 0:
        scale_factors = gt_points_3d[valid_gt_indices, 2] / (new_points_3d[valid_new_indices, 2] + 1e-6)
        median_scale = torch.median(scale_factors)
        new_points_3d_scaled = new_points_3d * median_scale
    else:
        median_scale = torch.tensor(1.0, device=gt_points_3d.device)
        new_points_3d_scaled = new_points_3d

    return median_scale, new_points_3d_scaled
