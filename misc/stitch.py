#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json

import cv2
import kornia
import numpy as np
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import torch
from matplotlib import pyplot as plt
from PIL import Image
from pytorch3d.ops import iterative_closest_point
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import Transform3d, quaternion_to_matrix
from torchvision import transforms


def extract_rotation_translation(extrinsics):
    """
    Extract rotation matrix and translation vector from an extrinsic matrix.
    """
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    return rotation, translation

def compute_relative_pose(extrinsics1, extrinsics2):
    """
    Compute the relative pose between two camera positions.
    """
    rotation1, translation1 = extract_rotation_translation(extrinsics1)
    rotation2, translation2 = extract_rotation_translation(extrinsics2)

    # Compute relative rotation and translation
    relative_rotation = torch.matmul(rotation2, rotation1.transpose(0, 1))
    relative_translation = translation2 - torch.matmul(relative_rotation, translation1)

    return relative_rotation, relative_translation

def apply_relative_pose(point_cloud, relative_rotation, relative_translation):
    """
    Apply the relative pose to align the point cloud.
    """
    transform = Transform3d().rotate(relative_rotation).translate(relative_translation)
    aligned_point_cloud = transform.transform_points(point_cloud)
    return aligned_point_cloud


def process_depth(depth_map, scale_factor=(33.0 / 65536), shift=0.9, max_depth=33.0):
    """
    Convert and scale depth map values.
    
    :param depth_map: Input depth map with integer values.
    :param scale_factor: Factor to scale depth values (default assumes mm to m).
    :param max_depth: Maximum depth value to consider (in scaled units).
    :return: Scaled and processed depth map.
    """
    # Convert to a floating-point representation and scale
    depth_map = depth_map.float() * scale_factor + shift
    #depth_map *= 0.005

    return depth_map


def create_point_cloud(rgb_image, depth_map, intrinsics):
    """
    Create a point cloud from an RGB image, depth map, and camera intrinsics.
    """
    # Reshape depth map to include a batch and channel dimension
    depth_map = depth_map.unsqueeze(0)  # Shape: (1, 1, H, W)

    # Unproject to 3D space using the new depth_to_3d_v2
    xyz = kornia.geometry.depth.depth_to_3d_v2(depth_map, intrinsics, normalize_points=True)

    # Squeeze the batch dimension and reshape
    xyz = xyz.squeeze(0).reshape(-1, 3)

    # Reshape and add colors
    rgb = rgb_image.permute(1, 2, 0).reshape(-1, 3)
    point_cloud = torch.cat([xyz, rgb], dim=1)

    return point_cloud


def apply_extrinsics(point_cloud, extrinsics):
    """
    Apply extrinsic transformations to the point cloud.
    This function now handles and preserves the RGB color data.
    """
    # Separate XYZ coordinates and RGB colors
    xyz = point_cloud[:, :3]
    rgb = point_cloud[:, 3:]

    # Apply transformations only to XYZ coordinates
    transform = Transform3d(matrix=extrinsics)
    transformed_xyz = transform.transform_points(xyz)

    # Reattach the RGB data to the transformed XYZ coordinates
    transformed_point_cloud = torch.cat([transformed_xyz, rgb], dim=1)

    return transformed_point_cloud


def align_point_clouds(source_xyz, target_xyz, source_rgb):
    """
    Align two point clouds (source and target) using Iterative Closest Point (ICP).
    Returns the aligned source point cloud with its original colors.
    """
    source_pc = Pointclouds(points=[source_xyz])
    target_pc = Pointclouds(points=[target_xyz])

    # Run ICP
    icp_solution = iterative_closest_point(source_pc, target_pc)

    if icp_solution.converged:
        # Apply the estimated transformation to the source point cloud
        transformed_source_xyz = icp_solution.Xt.points_packed()
        # Recombine with original color data
        transformed_source = torch.cat([transformed_source_xyz, source_rgb], dim=1)
        return transformed_source
    else:
        print("ICP did not converge.")
        # Return the original source with colors if ICP fails
        return torch.cat([source_xyz, source_rgb], dim=1)


def compute_relative_pose(extrinsics1, extrinsics2):
    # Assuming extrinsics are 4x4 matrices with rotation and translation
    R1, t1 = extrinsics1[:3, :3], extrinsics1[:3, 3]
    R2, t2 = extrinsics2[:3, :3], extrinsics2[:3, 3]

    # Compute relative rotation and translation (from frame 1 to frame 2)
    relative_rotation = R2 @ R1.T
    relative_translation = t2 - (relative_rotation @ t1)

    return relative_rotation, relative_translation

def warp_image(image, intrinsics, relative_rotation, relative_translation):
    # Warps image from the perspective of the first camera to that of the second
    image = image.permute(2,0,1)
    height, width = image.shape[1:]
    image = image.unsqueeze(0)
    transform_matrix = torch.cat([relative_rotation, relative_translation.unsqueeze(1)], dim=1)
    warping_matrix = intrinsics @ transform_matrix
    warping_matrix = warping_matrix[:, :3]
    warping_matrix = warping_matrix.unsqueeze(0)
    image = image.float()
    warped_image = kornia.geometry.transform.warp_perspective(image, warping_matrix, dsize=(height, width))
    return warped_image.squeeze()


def calculate_offset(extrinsics1, extrinsics2, intrinsics):
    # Calculate the relative transformation from camera 1 to camera 2
    relative_transform = torch.linalg.inv(extrinsics2) @ extrinsics1
    relative_transform = relative_transform.to(intrinsics.device)

    # Project the center of the first image to the second image's frame
    # Assuming the first image's center is (0, 0, 0) in its own frame
    center_image1 = torch.tensor([0., 0., 0., 1.]).to(intrinsics.device)
    center_image1_in_image2 = relative_transform @ center_image1
    breakpoint()

    # Convert this point to pixel coordinates using the intrinsics
    center_image1_in_image2_pixel = intrinsics @ center_image1_in_image2[:3]
    center_image1_in_image2_pixel = center_image1_in_image2_pixel.clone()
    breakpoint()
    center_image1_in_image2_pixel = center_image1_in_image2_pixel / center_image1_in_image2_pixel[2]  # Normalize

    # Calculate the offset in pixels (rounding to the nearest pixel)
    offset_x = round(center_image1_in_image2_pixel[0].item())
    offset_y = round(center_image1_in_image2_pixel[1].item())

    return (offset_x, offset_y)


def stitch_images(image1, image2, offset2):
    # Calculate the dimensions of the canvas
    canvas_height = max(image1.shape[1], image2.shape[1] + abs(offset2[1]))
    canvas_width = max(image1.shape[2], image2.shape[2] + abs(offset2[0]))

    # Create the canvas
    canvas = torch.zeros(3, canvas_height, canvas_width)

    # Calculate where to place image1 on the canvas
    y_start1 = max(0, -offset2[1])
    y_end1 = y_start1 + image1.shape[1]
    x_start1 = max(0, -offset2[0])
    x_end1 = x_start1 + image1.shape[2]

    # Place image1 on the canvas
    canvas[:, y_start1:y_end1, x_start1:x_end1] = image1

    # Calculate where to place image2 on the canvas
    y_start2 = max(0, offset2[1])
    y_end2 = y_start2 + image2.shape[1]
    x_start2 = max(0, offset2[0])
    x_end2 = x_start2 + image2.shape[2]

    # Place image2 on the canvas
    canvas[:, y_start2:y_end2, x_start2:x_end2] = image2

    return canvas


if __name__ == "__main__":
    json_file_path = '/mnt/sda/epic-fields/json/P01_04.json'
    image_dir = "/mnt/sda/epic-kitchens/og/frames_rgb_flow/rgb/train/P01/P01_04"

    with open(json_file_path) as f:
        a_cam = json.load(f)

    width, height = a_cam['camera']['width'], a_cam['camera']['height']
    fx, fy, cx, cy, *_ = a_cam['camera']['params']
    intrinsics_matrix = torch.tensor([[fx, 0, cx],
                                      [0, fy, cy],
                                      [0, 0, 1]])
    extrinsics = []
    #images = []
    image_transform = transforms.ToTensor()
    indices = [0, 1]
    for frame_number in indices:
        frame_name = f"frame_{(frame_number+1)*5:010}.jpg"
        params = a_cam['images'][frame_name]

        # Processing extrinsics
        qw, qx, qy, qz, tx, ty, tz = params
        quaternion = torch.tensor([qw, qx, qy, qz])
        rotation_matrix = quaternion_to_matrix(quaternion)
        extrinsic_matrix = torch.eye(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = torch.tensor([tx, ty, tz])
        extrinsics.append(torch.linalg.inv(extrinsic_matrix) * torch.tensor([1, -1, -1, 1]).reshape(4, 1))

        # Loading images
        #image_path = f"{image_dir}/{frame_name}"
        #image = Image.open(image_path)
        #image_tensor = image_transform(image).unsqueeze(0).unsqueeze(0)
        #images.append(image_tensor)

    # Stack images along view dimension
    #padding = int((width - height) / 2)
    #image_tensor = torch.cat(images, dim=1)#[..., padding:-padding] # TODO check if need modify intrinsics
    intrinsics = intrinsics_matrix.cuda()
    extrinsics_tensor = torch.stack(extrinsics).cuda()

    image1 = torch.tensor(np.asarray(Image.open('/home/relh/Downloads/marigold example/frame_0000000005.jpg'))).cuda()
    image2 = torch.tensor(np.asarray(Image.open('/home/relh/Downloads/marigold example/frame_0000000010.jpg'))).cuda()
    depth1 = torch.tensor(np.asarray(Image.open('/home/relh/Downloads/marigold example/frame_0000000005_depth_16bit.png'))).cuda()
    depth2 = torch.tensor(np.asarray(Image.open('/home/relh/Downloads/marigold example/frame_0000000010_depth_16bit.png'))).cuda()

    image1 = image1.permute(2, 0, 1)
    image2 = image2.permute(2, 0, 1)

    extrinsics1 = extrinsics_tensor[0]
    extrinsics2 = extrinsics_tensor[1]

    #stitcher = Stitcher(try_use_gpu=True)
    #image1 = '/home/relh/Downloads/marigold example/frame_0000000005.jpg'
    #image2 = '/home/relh/Downloads/marigold example/frame_0000000010.jpg'
    #panorama = stitcher.stitch([cv2.imread(image1), cv2.imread(image2)])
    #panorama = stitcher.stitch([cv2.imread("img1.jpg"), cv.imread("img2.jpg")])
    #breakpoint()

    # Load images
    #image1 = load_image('path_to_your_first_image.jpg')
    #image2 = load_image('path_to_your_second_image.jpg')

    # Compute the relative pose
    relative_rotation, relative_translation = compute_relative_pose(extrinsics1, extrinsics2)

    # Warp the second image to the perspective of the first
    warped_image2 = warp_image(image2, intrinsics, relative_rotation, relative_translation)

    # Stitch the images together
    # Assuming intrinsics, extrinsics1, and extrinsics2 are your camera parameters
    offset2 = calculate_offset(extrinsics1, extrinsics2, intrinsics)

    # Assuming image1 and image2 are your loaded images, in (C, H, W) format
    stitched_image = stitch_images(image1, image2, offset2)

    # Convert back to numpy array and display/save
    stitched_image = stitched_image.permute(1, 2, 0).numpy()
    stitched_image = (stitched_image * 1.0).astype(np.uint8)
    stitched_image = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR)
    plt.imshow(stitched_image)
    plt.show()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    #depth1 = process_depth(depth1)
    #depth2 = process_depth(depth2)

    # Convert images and depth maps to point clouds
    point_cloud1 = create_point_cloud(image1, depth1, intrinsics)
    point_cloud2 = create_point_cloud(image2, depth2, intrinsics)

    '''
    # Apply extrinsic transformations to one of the point clouds
    transformed_point_cloud1 = apply_extrinsics(point_cloud1, extrinsics_tensor[1])

    # Usage
    # Split the point clouds into XYZ coordinates and RGB colors
    source_xyz = transformed_point_cloud1[:, :3]
    source_rgb = transformed_point_cloud1[:, 3:]
    target_xyz = point_cloud2[:, :3]

    # Align the point clouds
    aligned_point_cloud1 = align_point_clouds(source_xyz, target_xyz, source_rgb)
    aligned_point_cloud1 = aligned_point_cloud1.cpu().numpy()
    '''

    # Example usage
    # Assuming extrinsics1 and extrinsics2 are the extrinsic matrices for the two camera positions
    relative_rotation, relative_translation = compute_relative_pose(extrinsics1, extrinsics2)

    # Assuming point_cloud2 is the point cloud from the second camera position
    aligned_point_cloud2 = apply_relative_pose(point_cloud2, relative_rotation, relative_translation)

    point_cloud1 = point_cloud1.cpu().numpy()
    rr.init("rerun_lala", spawn=True)
    rr.log("1", rr.Points3D(point_cloud1[:, :3], colors=point_cloud1[:, 3:], radii=1.0))
    rr.log("2", rr.Points3D(aligned_point_cloud2[:, :3], colors=aligned_point_cloud2[:, 3:], radii=1.0))
