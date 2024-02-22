"""
Depth alignment scripts between COLMAP and a monocular depth estimation pipeline
This allows us to get depth estimations with respect to a world coordinate system
Credit to Gene Chou, who wrote the original code.
"""

import numpy as np
import torch
import torch.nn.functional as F


def get_dense_depthmaps(
    pilimg, extrinsics, intrinsics, points3D, depth_pipe, device="cuda"
):
    """
    Gets an aligned dense depthmap from an image, camera information from COLMAP,
    and a depth prediction pipeline

    Args:
    pilimg: PIL image
    extrinsics: extrinsics of camera (i.e. rotation and translation)
    intrinsics: intrinsics of camera (i.e. focal length, principal point, etc.)
    points3D: 3D points from COLMAP
    depth_pipe: depth prediction pipeline
    return_est: whether to return the estimated depthmap
    device: cuda or cpu
    """
    # create sparse depthmap by projecting 3d points to 2d image
    # load xyz
    points3D_xyz = np.stack([point.xyz for point in points3D.values()])

    # convert to depth (scalar value for each pixel)
    img = np.array(pilimg) / 255.0
    sparsedepth = create_sparse_depth_map(
        points3D_xyz, intrinsics, extrinsics, img.shape[0:2]
    )

    # print("Sparse depth: ", sparsedepth.shape, sparsedepth.max(), sparsedepth.min())

    h, w = img.shape[:2]
    # Transform back to PIL
    est_depth = depth_pipe(pilimg)["predicted_depth"]
    est_depth = F.interpolate(
        est_depth[None], (h, w), mode="bilinear", align_corners=False
    )[0, 0]

    ##

    colmap_depth = torch.tensor(sparsedepth).unsqueeze(0)  # 1 x H x W
    estimated_dense_disparity = est_depth.unsqueeze(0)  # 1 x H x W

    """
    print("Colmap depth: ", colmap_depth)
    print("Colmap max and min depth: ", colmap_depth.max(), colmap_depth.min())
    print(
        "Estimated max and min disparity: ",
        estimated_dense_disparity.max(),
        estimated_dense_disparity.min(),
    )
    """

    aligned_depth, _ = ransac_pc_depth_alignment(
        estimated_dense_disparity, colmap_depth, ransac_iters=100, device=device
    )  # 10% points used for least squares each iteration

    return aligned_depth, estimated_dense_disparity, colmap_depth


def create_sparse_depth_map(points_3d, intrinsic_matrix, extrinsic_matrix, image_shape):
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    # Apply the extrinsic transformation (world to camera coordinates)
    camera_coords = extrinsic_matrix @ points_homogeneous

    # Convert to non-homogeneous camera coordinates
    camera_coords_non_homogeneous = camera_coords[:3, :]

    # Project the points onto the 2D image plane (camera to image coordinates)
    projected_points = intrinsic_matrix @ camera_coords_non_homogeneous

    # Normalize the coordinates and round them to get pixel indices
    x_pixels = np.round(projected_points[0, :] / projected_points[2, :]).astype(int)
    y_pixels = np.round(projected_points[1, :] / projected_points[2, :]).astype(int)
    depths = projected_points[2, :]

    # Print out min and max depths
    # print("Min and max sparse depths: ", depths.min(), depths.max())
    # print("Min and max x pixels: ", x_pixels.min(), x_pixels.max())
    # print("Min and max y pixels: ", y_pixels.min(), y_pixels.max())
    # print("Initial image shape: ", image_shape)

    # Initialize the depth map with infinity (or a very large number)
    depth_map = np.full(image_shape, np.inf)

    # Update the depth map with the nearest depth at each pixel location if that depth is positive
    # This is because negative depth indicates points that are actually behind the camera
    # which can mess up fitting since those points aren't visible in the image
    for x, y, depth in zip(x_pixels, y_pixels, depths):
        if depth > 0 and 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            depth_map[y, x] = min(depth_map[y, x], depth)

    # Replace infinities with zeros or an appropriate background value
    # later zeros are replaced by mask
    depth_map[depth_map == np.inf] = 0
    return depth_map


def ransac_pc_depth_alignment(
    estimated_dense_disparity, colmap_depth, ransac_iters=100, device="cuda"
):
    if device == "cpu":
        return cpu_ransac_alignment(
            estimated_dense_disparity, colmap_depth, ransac_iters=ransac_iters
        )
    else:
        return gpu_ransac_alignment(
            estimated_dense_disparity,
            colmap_depth,
            ransac_iters=ransac_iters,
            device="cuda",
        )


def cpu_ransac_alignment(estimated_dense_disparity, colmap_depth, ransac_iters=100):
    disparity_max = 10000
    disparity_min = 0.0001
    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    # print(colmap_depth.shape, estimated_dense_disparity.shape)
    assert colmap_depth.shape == estimated_dense_disparity.shape, (
        colmap_depth.shape,
        estimated_dense_disparity.shape,
    )

    colmap_depth = colmap_depth.float()
    estimated_dense_disparity = estimated_dense_disparity.float()

    mask = colmap_depth != 0

    target = colmap_depth
    target = torch.clip(target, depth_min, depth_max)

    prediction = estimated_dense_disparity

    # zeros where colmap_depth is zero, 1/depth for disparity where colmap_depth is not zero
    # target[mask == 1] equal to colmap_depth[mask == 1]
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]  # convert to inverse depth

    nonzero_indices = torch.nonzero(mask[0], as_tuple=False)
    num_to_zero_out = len(nonzero_indices) - 2  # int(percent * len(nonzero_indices))

    min_error = 1.0
    best_aligned = 0.0

    for _ in range(ransac_iters):
        mask1 = mask.clone()
        target_disparity1 = target_disparity.clone()
        rand_indices = torch.randperm(len(nonzero_indices))[:num_to_zero_out]
        coords = nonzero_indices[rand_indices].long()
        rows, cols = coords[:, 0], coords[:, 1]
        mask1[0, rows, cols] = 0
        target_disparity1[0, rows, cols] = 0.0

        scale, shift = compute_scale_and_shift(prediction, target_disparity1, mask1)

        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        prediction_aligned[prediction_aligned > disparity_max] = disparity_max
        prediction_aligned[prediction_aligned < disparity_min] = disparity_min
        prediction_depth = (1.0 / prediction_aligned).float()

        # calculate errors
        threshold = 1.05
        # bad pixel
        err = torch.zeros_like(prediction_depth, dtype=torch.float)

        # target is ground truth sparse depth map, 0 where no points and 1 where there is a 3d reconstruction point
        # prediction_depth / target or target / prediction_depth is the error...should ideally be 1 (i.e. prediction after alignment == ground truth)
        err[mask == 1] = torch.max(
            prediction_depth[mask == 1] / target[mask == 1],
            target[mask == 1] / prediction_depth[mask == 1],
        )
        err[mask == 1] = (err[mask == 1] > threshold).float()

        # err is 0 where no points or prediction == ground truth
        # err is 1 where prediction != ground truth
        # mask is 1 where there is a point
        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        if p.squeeze().item() < min_error:
            min_error = p.squeeze().item()
            best_aligned = prediction_depth

    return best_aligned, min_error


def gpu_ransac_alignment(
    estimated_dense_disparity,
    colmap_depth,
    ransac_iters=100,
    mask_percent=0.9,
    device="cuda",
):
    disparity_max = 10000
    disparity_min = 0.0001

    depth_max = 1 / disparity_min
    depth_min = 1 / disparity_max

    # print(colmap_depth.shape, estimated_dense_disparity.shape)
    assert colmap_depth.shape == estimated_dense_disparity.shape, (
        colmap_depth.shape,
        estimated_dense_disparity.shape,
    )

    mask = colmap_depth != 0

    target = colmap_depth
    target = torch.clip(target, depth_min, depth_max)

    prediction = estimated_dense_disparity

    # zeros where colmap_depth is zero, 1/depth for disparity where colmap_depth is not zero
    # target[mask == 1] equal to colmap_depth[mask == 1]
    target_disparity = torch.zeros_like(target)
    target_disparity[mask == 1] = 1.0 / target[mask == 1]  # convert to inverse depth

    # ransac
    # ipdb.set_trace()
    prediction1 = prediction.repeat(ransac_iters, 1, 1).to(device)
    target_disparity1 = target_disparity.repeat(ransac_iters, 1, 1).to(device)
    mask1 = mask.repeat(ransac_iters, 1, 1).to(device)
    target_depth = target.repeat(ransac_iters, 1, 1).to(device)
    original_mask = mask.repeat(ransac_iters, 1, 1).to(device)

    percent = mask_percent  # how much to mask out --> changing to only using 2 points
    nonzero_indices = torch.nonzero(mask1[0], as_tuple=False)
    num_to_zero_out = len(nonzero_indices) - 2  # int(percent * len(nonzero_indices))

    all_indices = torch.stack(
        [torch.randperm(len(nonzero_indices)) for _ in range(ransac_iters)]
    )
    rand_indices = all_indices[:, :num_to_zero_out]

    range_tensor = torch.arange(ransac_iters).view(-1, 1)
    nonzeros = nonzero_indices.unsqueeze(0).repeat(ransac_iters, 1, 1)
    masked_indices = nonzeros[range_tensor, rand_indices].long()

    # Extract batch indices (which is just a range from 0 to B-1)
    batch_indices = torch.arange(mask1.shape[0])[:, None]

    # Use advanced indexing to set the specified pixels to 0
    mask1[batch_indices, masked_indices[..., 0], masked_indices[..., 1]] = 0
    target_disparity1[batch_indices, masked_indices[..., 0], masked_indices[..., 1]] = (
        0.0
    )

    scale, shift = compute_scale_and_shift(prediction1, target_disparity1, mask1)

    prediction_aligned = scale.view(-1, 1, 1) * prediction1 + shift.view(-1, 1, 1)

    prediction_aligned[prediction_aligned > disparity_max] = disparity_max
    prediction_aligned[prediction_aligned < disparity_min] = disparity_min
    prediction_depth = 1.0 / prediction_aligned

    # calculate errors
    threshold = 1.05
    prediction_depth = prediction_depth.float()
    original_mask = original_mask.float()
    target_depth = target_depth.float()
    # bad pixel
    err = torch.zeros_like(prediction_depth, dtype=torch.float)

    # target is ground truth sparse depth map, 0 where no points and 1 where there is a 3d reconstruction point
    # prediction_depth / target or target / prediction_depth is the error...should ideally be 1 (i.e. prediction after alignment == ground truth)
    err[original_mask == 1] = torch.max(
        prediction_depth[original_mask == 1] / target_depth[original_mask == 1],
        target_depth[original_mask == 1] / prediction_depth[original_mask == 1],
    )
    err[original_mask == 1] = (err[original_mask == 1] > threshold).float()

    # err is 0 where no points or prediction == ground truth
    # err is 1 where prediction != ground truth
    # mask is 1 where there is a point
    p = torch.sum(err, (1, 2)) / torch.sum(original_mask, (1, 2))
    min_value, min_index = torch.min(p, 0)

    return (
        prediction_depth[min_index].unsqueeze(0).cpu(),
        min_value.cpu().item(),
    )  # aligned depth, error


def compute_scale_and_shift(prediction, target, mask):
    """
    https://gist.github.com/ranftlr/45f4c7ddeb1bbb88d606bc600cab6c8d
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(
        mask, (1, 2)
    )  # shape of B, where each b is the sum of every element at index b

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1
