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
from dust3r.inference import inference, load_model
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
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

def img_to_pts_3d_dust(images, views='single'):
    global dust_model
    device = 'cuda'
    batch_size = 1
    if dust_model is None:
        model_path = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        dust_model = load_model(model_path, device)

    # --- whether to standalone index 0 image or not ---
    images = [Image.fromarray(image.cpu().numpy()) for image in images]
    if views == 'single':
        images = [images[0]]
    images = load_images(images, size=512)

    # --- run dust3r ---
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, dust_model, device, batch_size=batch_size)
    mode = GlobalAlignerMode.PointCloudOptimizer if len(images) > 1 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)

    # --- either get pts or run global optimization ---
    if len(images) > 1:
        loss = scene.compute_global_alignment(init='mst', niter=100, schedule='cosine', lr=0.01)
    else:
        scene = scene.get_pts3d()[0]

    #scene = scene.clean_pointcloud()
    #scene = scene.mask_sky()

    # --- post processing ---
    imgs = to_numpy(scene.imgs)
    focals = to_numpy(scene.get_focals().cpu())
    cams2world = to_numpy(scene.get_im_poses().cpu())
    pts3d = to_numpy(scene.get_pts3d())
    mask = to_numpy(scene.get_masks())
    intrinsics = scene.get_intrinsics()[0].float().cuda()

    # full pointcloud
    pts = np.concatenate([p for p, m in zip(pts3d, mask)])
    col = np.concatenate([p for p, m in zip(imgs, mask)])
    scene = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))

    pts_3d = torch.tensor(scene.vertices, device='cuda', dtype=torch.float32)
    rgb_3d = torch.tensor(scene.colors, device='cuda', dtype=torch.uint8)
    return pts_3d * 1000.0, rgb_3d[:, :3], intrinsics.detach()
