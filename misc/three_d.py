#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('depth_anything/metric_depth/')
sys.path.append('mast3r/dust3r/')
sys.path.append('mast3r/')

import einops
import argparse
import copy
import functools
import math
import os
import tempfile

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
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.utils.misc import hash_md5
#from dust3r.model import AsymmetricCroCo3DStereo
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.utils.image import rgb
from dust3r.viz import (CAM_COLORS, OPENGL, add_scene_cam, cat_meshes,
                        pts3d_to_trimesh)
from misc.supersample import supersample_point_cloud

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
da_model = None
dust_model = None

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

def img_to_pts_3d_dust(images, all_cam2world=None, intrinsics=None, dm=None, conf=None, tmp_dir=None):
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
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=False if num_images > 2 else True)
    #output = inference(pairs, dust_model, device, batch_size=batch_size)

    scene = sparse_global_alignment(filelist, pairs, tmp_dir,
                                dust_model, lr1=0.07, niter1=100, lr2=0.014, niter2=100, device=device,
                                opt_depth='depth' in 'refine', shared_intrinsics=False,
                                matching_conf_thr=5.)#, **kw)

    #mode = GlobalAlignerMode.ModularPointCloudOptimizer if num_images > 2 else GlobalAlignerMode.PairViewer
    #scene = global_aligner(output, device=device, mode=mode)

    # --- either get pts or run global optimization ---
    '''
    #mode = GlobalAlignerMode.PointCloudOptimizer if num_images > 2 else GlobalAlignerMode.PairViewer

    if mode is GlobalAlignerMode.ModularPointCloudOptimizer and all_cam2world is not None:
    #if mode is GlobalAlignerMode.PointCloudOptimizer and all_cam2world is not None:

        # --- format pose and intrinsics ---
        all_cam2world = [x for x in all_cam2world]
        known_poses = [False if x is None else True for x in all_cam2world]

        #repl_intrinsics = [intrinsics.cpu().numpy() for x in all_cam2world]
        #known_intrinsics = [True for x in repl_intrinsics]

        # --- set pose and focal ---
        scene.preset_pose(all_cam2world, known_poses)
        #scene.preset_intrinsics(repl_intrinsics, known_intrinsics)

        # --- set depth maps ---
        #for i, this_dm in enumerate(dm):
        #    scene._set_depthmap(i, this_dm, force=True)

        # --- set focal and pp ---
        #scene.preset_focal([x[0, 0] for x in repl_intrinsics], known_intrinsics)

        #pp = [np.array((x[0, -1], x[1, -1])) for x in repl_intrinsics]
        #for i in range(len(pp)):
        #    pp[i].requires_grad = True
        #scene.preset_principal_point(pp, known_intrinsics)
    '''

    # --- either get pts or run global optimization ---
    #loss = scene.compute_global_alignment(init='mst', niter=200, schedule='cosine', lr=0.01) # 60/s
    #loss = scene.compute_global_alignment(init='msp', niter=200, schedule='cosine', lr=0.01) # 50/s
    #loss = scene.compute_global_alignment(init='known_poses', niter=200, schedule='cosine', lr=0.01)
    #scene = scene.clean_pointcloud()
    #scene = scene.mask_sky()

    # --- post processing ---
    use = lambda x: x.float().cuda().detach()
    all_cam2world = [use(x) for x in scene.get_im_poses()]
    world2cam = torch.linalg.inv(all_cam2world[-1])
    intrinsics = use(scene.intrinsics[-1])
    pts3d, depth_maps, confs = scene.get_dense_pts3d()
    pts_3d = use(torch.stack(pts3d))[-1]
    rgb_3d = use(torch.stack([torch.tensor(x) for x in scene.imgs])) * 255.0
    rgb_3d = einops.rearrange(rgb_3d, 'b h w c -> b c (h w)')[-1]
    depth_maps = use(torch.stack(depth_maps))
    conf = use(torch.stack(confs))
    conf = conf.reshape(conf.shape[0], -1)[-1]

    pts_3d = pts_3d#[conf > 0.5]
    rgb_3d = einops.rearrange(rgb_3d, 'c x -> x c')#[conf > 0.5]

    return pts_3d.reshape(-1, 3),\
           rgb_3d.reshape(-1, 3)[:, :3].to(torch.uint8),\
           world2cam,\
           all_cam2world,\
           intrinsics,\
           depth_maps,\
           conf

def img_to_pts_3d_da(color_image, views=None, tmp_dir=None):
    global da_model
    if da_model is None:
        config = get_config('zoedepth', "eval", 'nyu')
        config.pretrained_resource = 'local::./checkpoints/depth_anything_metric_depth_indoor.pt'
        da_model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
        da_model.eval()
    original_width, original_height = 512, 512
    #color_image = Image.open(image_path).convert('RGB')
    color_image = color_image[0]
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
    return points_camera_coord_tensor, colors, None

if __name__ == '__main__':
    image_path = "./depth_anything/metric_depth/my_test/input/demo11.png"
    color_image = Image.open(image_path).convert('RGB')
    pcd = image_to_3d(color_image)
    breakpoint()
