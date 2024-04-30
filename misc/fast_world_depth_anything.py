import os

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr  # pip install rerun-sdk
import torch
import torch.nn.functional as F
from decord import VideoReader
from imageio import get_writer
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor
from transformers import pipeline

torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

from misc.colmap_rw_utils import read_model, sort_images
from misc.utils import *


def get_3D_coordinates(depth, extrinsics):
    H, W = depth.shape[-2:]
    i_coords, j_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords_3D = torch.stack((j_coords, i_coords, depth.squeeze()), dim=-1)
    coords_3D_homogeneous = torch.cat((coords_3D, torch.ones(H, W, 1)), dim=-1)
    return torch.matmul(coords_3D_homogeneous, extrinsics.T)[..., :3]

def get_aligned_depth(image, img, intrinsics, points3D_xyz, pipe, frame_idx=0):
    extrinsics = get_camera_extrinsic_matrix(image)
    if extrinsics is None:
        return None
    img_tensor = ToTensor()(img)
    h, w = img_tensor.shape[1:3]
    _, sparse_depth = points_3d_to_image(points3D_xyz, intrinsics, extrinsics, (h, w))
    est_depth = F.interpolate(pipe(img)["predicted_depth"][None].cuda(), (h, w), mode="bilinear", align_corners=False)[0, 0]
    aligned_depth, _ = ransac_alignment(est_depth.unsqueeze(0), sparse_depth.unsqueeze(0))
    return aligned_depth.squeeze(), est_depth, get_3D_coordinates(aligned_depth, extrinsics)

def save_frames(video_reader, output_frame_dir, max_frames=None):
    for i, frame in enumerate(video_reader):
        if max_frames and i >= max_frames:
            break
        Image.fromarray(frame.asnumpy()).save(os.path.join(output_frame_dir, f'frame_{i}.jpg'))

def save_depth_frame(depth_tensor, file_path):
    depth_image = depth_tensor.squeeze().cpu().clamp(0, 255).numpy()
    depth_image_colored = plt.get_cmap('magma')(depth_image / 255)  # Normalized for coloring
    depth_image_colored_uint8 = (depth_image_colored * 255).astype(np.uint8)
    Image.fromarray(depth_image_colored_uint8).save(file_path)

def make_depth_video(depth_dir):
    frame_files = sorted(os.listdir(depth_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    with get_writer(depth_dir[:-1] + '.mp4', fps=30) as writer:
        for frame_file in frame_files:
            frame_path = os.path.join(depth_dir, frame_file)
            frame = Image.open(frame_path)
            writer.append_data(np.array(frame)[:, :, :3])

def process_video(video_path, output_frame_dir, output_depth_video, output_disp_video, \
                  cameras, images, points3D, dense_points3D, intrinsics, pipe):
    os.makedirs(output_frame_dir, exist_ok=True)
    os.makedirs(output_depth_video, exist_ok=True)
    os.makedirs(output_disp_video, exist_ok=True)

    if not os.listdir(output_frame_dir):
        video_reader = VideoReader(video_path)
        save_frames(video_reader, output_frame_dir)


    if not os.listdir(output_depth_video):
        frame_files = sorted(os.listdir(output_frame_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for frame_file in frame_files:
            frame_path = os.path.join(output_frame_dir, frame_file)
            frame = Image.open(frame_path)
            frame_idx = int(frame_file.split('_')[-1].split('.')[0]) + 1
            image = images[frame_idx]
            result = get_aligned_depth(image, frame, intrinsics, dense_points3D, pipe, frame_idx)
            
            if result is not None:
                pred_depth, est_depth, world_3D_coords = result
                print(pred_depth)
                print("processed frame {}".format(frame_idx), end="\r")
                save_depth_frame(pred_depth, os.path.join(output_depth_video, f'depth_{frame_idx}.png'))
                save_depth_frame(est_depth, os.path.join(output_disp_video, f'disparity_{frame_idx}.png'))

    make_depth_video(output_depth_video)
    make_depth_video(output_disp_video)

if __name__ == "__main__":
    video_name = "P01_04"
    io_args = {
        "input_video": f"./assets/{video_name}/{video_name}.MP4",
        "input_field_dir": f"./assets/{video_name}/field/",
        "input_dense_field_dir": f"./assets/{video_name}/dense/",
        "output_frame_dir": f"./assets/{video_name}/frames/",
        "output_depth_dir": f"./assets/{video_name}/depth/",
        "output_disp_dir": f"./assets/{video_name}/disparity/",
    }

    pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf", device="cuda:0")
    cameras, images, points3D = read_model(io_args["input_field_dir"], "")
    images = sort_images(images)
    fx, fy, cx, cy, k1, k2, p1, p2 = cameras.get(1).params
    intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float()
    points3D_xyz = torch.stack([torch.tensor(point.xyz).float() for point in points3D.values()])
    dense_points3D, colors = load_dense_point_cloud(io_args['input_dense_field_dir'] + "fused.ply")
    dense_points3D = torch.tensor(dense_points3D)

    process_video(
        io_args["input_video"], io_args["output_frame_dir"], io_args["output_depth_dir"],
        io_args["output_disp_dir"],
        cameras, images, points3D_xyz, dense_points3D, intrinsics, pipe
    )
