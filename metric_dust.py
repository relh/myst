#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import functools
import math
import os
import sys
import tempfile

import gradio
import matplotlib.pyplot as pl
import numpy as np
import PIL.Image
import torch
import torchvision.transforms as tvf
import trimesh
from PIL import Image
from PIL.ImageOps import exif_transpose
from scipy.spatial.transform import Rotation

sys.path.append('dust3r/')

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import rgb
from dust3r.viz import (CAM_COLORS, OPENGL, add_scene_cam, cat_meshes,
                        pts3d_to_trimesh)
from misc.supersample import supersample_point_cloud

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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
    for image in images:
        img = exif_transpose(image)
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        print(f' - adding image with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    print(f' (Found {len(imgs)} images)')
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    return imgs

def img_to_pts_3d_dust(images, poses=None, intrinsics=None):
    global dust_model
    device = 'cuda'
    batch_size = 1
    if dust_model is None:
        weights_path = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        dust_model = AsymmetricCroCo3DStereo.from_pretrained(weights_path).to('cuda')

    # --- whether to standalone index 0 image or not ---
    images = [Image.fromarray(image.cpu().numpy()) for image in images]
    num_images = len(images)
    images = load_images(images, size=512)

    # --- run dust3r ---
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, dust_model, device, batch_size=batch_size)
    mode = GlobalAlignerMode.ModularPointCloudOptimizer if num_images > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)

    # --- either get pts or run global optimization ---
    if mode is GlobalAlignerMode.ModularPointCloudOptimizer and poses is not None:
        # pretend we don't know final pose to appease dust3r
        poses = [x for x in poses] + [None]
        known_poses = [True if x != None else False for x in poses]
        known_intrinsics = [intrinsics for x in poses]

        scene.preset_pose([x.cpu().numpy() for x in poses], known_poses)
        scene.preset_intrinsics([x.cpu().numpy() for x in known_intrinsics])

    # --- either get pts or run global optimization ---
    loss = scene.compute_global_alignment(init='mst', niter=50, schedule='cosine', lr=0.01)
    #scene = scene.clean_pointcloud()
    #scene = scene.mask_sky()

    # --- post processing ---
    clean = lambda x: x.float().cuda()#.detach()
    all_cam2world = [clean(x) for x in scene.get_im_poses()]
    world2cam = torch.linalg.inv(all_cam2world[-1])
    intrinsics = clean(scene.get_intrinsics()[-1])
    pts_3d = clean(torch.stack(scene.get_pts3d()))
    rgb_3d = clean(torch.stack([torch.tensor(x) for x in scene.imgs])) * 255.0

    return pts_3d.reshape(-1, 3),\
           rgb_3d.reshape(-1, 3)[:, :3].to(torch.uint8),\
           world2cam,\
           all_cam2world,\
           intrinsics
            





