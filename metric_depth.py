#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np
import rerun as rr  # pip install rerun-sdk
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.append('depth_anything/metric_depth/')
from depth_anything.metric_depth.zoedepth.models.builder import build_model
from depth_anything.metric_depth.zoedepth.utils.config import get_config

# Global settings
FL = 715.0873
FY = 256 * 1.0
FX = 256 * 1.0
NYU_DATA = False
FINAL_HEIGHT = 512 # 256
FINAL_WIDTH = 512 # 256
DATASET = 'nyu' # Lets not pick a fight with the model's dataloader

model = None

def img_to_pts_3d_da(color_image):
    global model
    if model is None:
        config = get_config('zoedepth', "eval", DATASET)
        config.pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
        model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
    #color_image = Image.open(image_path).convert('RGB')
    original_width, original_height = color_image.size
    image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    pred = model(image_tensor, dataset=DATASET)
    if isinstance(pred, dict):
        pred = pred.get('metric_depth', pred.get('out'))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    pred = pred.squeeze().detach().cpu().numpy()

    # Resize color image and depth to final size
    resized_color_image = color_image.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)
    resized_pred = Image.fromarray(pred).resize((FINAL_WIDTH, FINAL_HEIGHT), Image.NEAREST)

    focal_length_x, focal_length_y = (FX, FY) if not NYU_DATA else (FL, FL)
    x, y = np.meshgrid(np.arange(FINAL_WIDTH), np.arange(FINAL_HEIGHT))
    x = (x - FINAL_WIDTH / 2.0) / focal_length_x
    y = (y - FINAL_HEIGHT / 2.0) / focal_length_y
    z = np.array(resized_pred)

    # Compute 3D points in camera coordinates
    points_camera_coord = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3) * 50.0
    points_camera_coord_tensor = torch.tensor(points_camera_coord, dtype=torch.float32, device='cuda')

    colors = np.array(resized_color_image).reshape(-1, 3) / 255.0
    colors = (torch.tensor(colors) * 255.0).float().to('cuda').to(torch.uint8)
    return points_camera_coord_tensor, colors, None


if __name__ == '__main__':
    image_path = "./depth_anything/metric_depth/my_test/input/demo11.png"
    color_image = Image.open(image_path).convert('RGB')
    pcd = image_to_3d(color_image)
    breakpoint()
