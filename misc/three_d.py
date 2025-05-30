#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('depth_anything/metric_depth/')
sys.path.append('mast3r/dust3r/')
sys.path.append('mast3r/')

import argparse
import copy
import functools
import math
import os
import tempfile

import cv2
import einops
import gradio
import matplotlib.pyplot as pl
import numpy as np
import PIL.Image
import rerun as rr  # pip install rerun-sdk
import torch
import torchvision.transforms as transforms
import torchvision.transforms as tvf
import trimesh
from PIL import Image
from PIL.ImageOps import exif_transpose
from scipy.spatial.transform import Rotation

from depth_anything.metric_depth.zoedepth.models.builder import build_model
from depth_anything.metric_depth.zoedepth.utils.config import get_config
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.image import rgb
from dust3r.viz import (CAM_COLORS, OPENGL, add_scene_cam, cat_meshes,
                        pts3d_to_trimesh)
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from misc.camera import pts_cam_to_world
from misc.supersample import supersample_point_cloud

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
metric_model = None
da_model = None
dust_model = None
intr_model = None

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(images, size, square_ok=True):
    imgs = []
    filelist = []
    for i, image in enumerate(images):
        filelist.append(str(i))
        img = exif_transpose(image)
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))
    print(f' (Found {len(imgs)} images)')
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    return imgs, filelist

def img_to_pts_3d_dust(images, world2cam=None, intrinsics=None, dm=None, conf=None, tmp_dir=None):
    global dust_model
    device = 'cuda'
    batch_size = 1
    if dust_model is None:
        #weights_path = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        weights_path = 'mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
        dust_model = AsymmetricMASt3R.from_pretrained(weights_path).to('cuda')

    # --- whether to standalone index 0 image or not ---
    images = [Image.fromarray(image.cpu().numpy()) for image in images]
    num_images = len(images)
    images, filelist = load_images(images, size=512)

    # --- run dust3r ---
    pairs = make_pairs(images, scene_graph='complete', prefilter=None,\
                       symmetrize=True)# if num_images > 2 else True)
    #output = inference(pairs, dust_model, device, batch_size=batch_size)

    scene = sparse_global_alignment(filelist, pairs, tmp_dir,
                                dust_model, lr1=0.07, niter1=100, lr2=0.014, niter2=100, device=device,
                                opt_depth='depth' in 'refine', shared_intrinsics=False,
                                matching_conf_thr=5.)#, **kw)

    # --- post processing ---
    use = lambda x: x.float().cuda().detach()
    all_cam2world = [use(x) for x in scene.get_im_poses()]
    world2cam = torch.linalg.inv(all_cam2world[-1])
    intrinsics = use(scene.intrinsics[-1])
    pts3d, depth_maps, confs = scene.get_dense_pts3d()
    pts_3d = use(torch.stack(pts3d))
    rgb_3d = use(torch.stack([torch.tensor(x) for x in scene.imgs])) * 255.0
    rgb_3d = einops.rearrange(rgb_3d, 'b h w c -> b (h w) c')
    depth_maps = use(torch.stack(depth_maps))
    conf = use(torch.stack(confs))
    conf = conf.reshape(conf.shape[0], -1)

    pts_3d = pts_3d#[conf > 0.5]
    rgb_3d = rgb_3d#[conf > 0.5]

    return pts_3d.reshape(-1, 3),\
           rgb_3d.reshape(-1, 3)[:, :3].to(torch.uint8),\
           world2cam,\
           intrinsics,\
           depth_maps,\
           conf.reshape(-1, 1)

def img_to_pts_3d_da(color_image, world2cam=None, intrinsics=None, tmp_dir=None):
    global da_model, intr_model
    if da_model is None:
        config = get_config('zoedepth', "eval", 'nyu')
        config.pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
        da_model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
        da_model.eval()
    if intr_model is None:
        intr_model = torch.hub.load('ShngJZ/WildCamera', "WildCamera", pretrained=True).cuda()
    original_width, original_height = 512, 512
    #color_image = Image.open(image_path).convert('RGB')
    color_image = color_image[-1]
    #original_width, original_height = color_image.size()
    color_image = Image.fromarray(color_image.cpu().numpy())
    image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

    pred = da_model(image_tensor, dataset='nyu')
    if isinstance(pred, dict):
        pred = pred.get('metric_depth', pred.get('out'))
    elif isinstance(pred, (list, tuple)):
        pred = pred[-1]
    pred = pred.squeeze().detach().cpu().numpy()

    # Resize color image and depth to final size
    resized_color_image = color_image.resize((original_width, original_height), Image.LANCZOS)
    resized_pred = Image.fromarray(pred).resize((original_width, original_height), Image.NEAREST)

    focal_length_x, focal_length_y = (256.0, 256.0)
    x, y = np.meshgrid(np.arange(original_width), np.arange(original_height))
    x = (x - original_width / 2.0) / focal_length_x
    y = (y - original_height / 2.0) / focal_length_y
    z = np.array(resized_pred)

    # Compute 3D points in camera coordinates
    points_camera_coord = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3) * 50.0
    points_camera_coord_tensor = torch.tensor(points_camera_coord, dtype=torch.float32, device='cuda')

    colors = np.array(resized_color_image).reshape(-1, 3) / 255.0
    colors = (torch.tensor(colors) * 255.0).float().to('cuda').to(torch.uint8)

    if world2cam == None:
        world2cam = torch.tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).float().cuda()
    if intrinsics == None:
        intrinsics = torch.tensor([[256.0*1.0, 0.0000, 256.0000],
                               [0.0000, 256.0*1.0, 256.0000],
                               [0.0000, 0.000, 1.0000]]).cuda()

    depth_3d = pts_cam_to_world(points_camera_coord_tensor, world2cam)
    return depth_3d, colors, world2cam, intrinsics, None, None

def calculate_intrinsic_matrix(preds, image_width, image_height):
    vfov_rad = preds['pred_vfov'].item() * (math.pi / 180)  # Convert degrees to radians
    
    # Compute focal length from vFOV
    focal_length_vfov = image_height / (2 * torch.tan(torch.tensor(vfov_rad) / 2))

    # Use pred_rel_focal to compute focal length
    focal_length_rel = preds['pred_rel_focal'].item() * image_height

    # Check for consistency between vFOV-derived and pred_rel_focal-derived focal lengths
    #if not torch.isclose(focal_length_vfov, torch.tensor(focal_length_rel), atol=1e-3):
    #    print(f"Warning: Focal lengths differ. vFOV-derived: {focal_length_vfov}, rel_focal-derived: {focal_length_rel}")

    # Using the vFOV-derived focal length for intrinsic matrix to ensure consistency with previous calculations
    # or switch to `focal_length_rel` if it proves to be more accurate in your application context.
    focal_length = focal_length_vfov  # or focal_length_rel

    # Principal point assumed at the center
    cx = image_width / 2
    cy = image_height / 2
    
    # Create the intrinsic matrix
    K = torch.tensor([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=preds['pred_vfov'].device)  # Ensure the tensor is on the same device as the input

    return K

def img_to_pts_3d_metric(color_image, world2cam=None, intrinsics=None, tmp_dir=None):
    global metric_model, intr_model
    if metric_model is None:
        metric_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_giant2', pretrain=True).cuda()
        #metric_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True).cuda()
    if intr_model is None and intrinsics is None:
        from perspective2d import PerspectiveFields
        version = 'Paramnet-360Cities-edina-centered'
        intr_model = PerspectiveFields(version).eval().cuda()
    original_width, original_height = 512, 512
    color_image = color_image[-1].cpu().numpy()
    pf_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    color_image = Image.fromarray(color_image)

    if intrinsics is None:
        preds = intr_model.inference(img_bgr=pf_image)
        intrinsics = calculate_intrinsic_matrix(preds, 512, 512)
    metric_intrinsics = [intrinsics[0,0], intrinsics[1,1], intrinsics[0, -1], intrinsics[1, -1]]

    image_tensor = transforms.ToTensor()(color_image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    #color_image = einops.rearrange(color_image, 'h w c -> 1 c h w').float().cuda()
    pred_depth, confidence, output_dict = metric_model.inference({'input': image_tensor, 'intrinsics': metric_intrinsics})

    # TODO use these
    pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
    normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details
    pred = pred_depth.squeeze().detach().cpu().numpy()

    # Resize color image and depth to final size
    resized_color_image = color_image.resize((original_width, original_height), Image.LANCZOS)
    resized_pred = Image.fromarray(pred).resize((original_width, original_height), Image.NEAREST)

    focal_length_x, focal_length_y = (256.0, 256.0)
    x, y = np.meshgrid(np.arange(original_width), np.arange(original_height))
    x = (x - original_width / 2.0) / focal_length_x
    y = (y - original_height / 2.0) / focal_length_y
    z = np.array(resized_pred)

    # Compute 3D points in camera coordinates
    points_camera_coord = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3) * 50.0
    points_camera_coord_tensor = torch.tensor(points_camera_coord, dtype=torch.float32, device='cuda')

    colors = np.array(resized_color_image).reshape(-1, 3) / 255.0
    colors = (torch.tensor(colors) * 255.0).float().to('cuda').to(torch.uint8)

    if world2cam == None:
        world2cam = torch.tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).float().cuda()
    if intrinsics == None:
        intrinsics = torch.tensor([[256.0*1.0, 0.0000, 256.0000],
                               [0.0000, 256.0*1.0, 256.0000],
                               [0.0000, 0.000, 1.0000]]).cuda()

    depth_3d = pts_cam_to_world(points_camera_coord_tensor, world2cam)
    return depth_3d, colors, world2cam, intrinsics, None, None

if __name__ == '__main__':
    image_path = "./depth_anything/metric_depth/my_test/input/demo11.png"
    color_image = Image.open(image_path).convert('RGB')
    pcd = image_to_3d(color_image)
    breakpoint()
