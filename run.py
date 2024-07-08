#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

torch.backends.cuda.preferred_linalg_library()

import os
import pickle
import sys
import termios
import tty
from argparse import ArgumentParser

import lpips
import rerun as rr  # pip install rerun-sdk
from matplotlib import pyplot as plt
from PIL import Image
from pytorch3d.renderer import OrthographicCameras, PerspectiveCameras

from misc.camera import (move_camera, pts_3d_to_img_py3d, pts_3d_to_img_raster,
                         pts_cam_to_world)
from misc.control import generate_control
from misc.imutils import fill, resize_and_pad, select_bounding_box
from misc.inpaint import run_inpaint
from misc.perceptual import *
from misc.prune import density_pruning_py3d
from misc.scale import median_scene_distance
from misc.supersample import run_supersample, supersample_point_cloud
from misc.text import generate_prompt
from misc.three_d import img_to_pts_3d_da, img_to_pts_3d_dust


def main(args, meta_idx):
    # --- setup rerun args ---
    rr.script_setup(args, f"{meta_idx}myst")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    img_to_pts_3d = img_to_pts_3d_da if args.depth == 'da' else img_to_pts_3d_dust
    pts_3d_to_img = pts_3d_to_img_raster if args.renderer == 'raster' else pts_3d_to_img_py3d 

    cameras = None
    pts_3d = None
    image = None
    size = 512
    idx = 0
    while True:
        # --- setup initial scene ---
        if image is None: 
            mask = torch.ones(size, size)
            if args.image != 'gen':
                orig_prompt = prompt = ''
                image = torch.tensor(np.array(resize_and_pad(Image.open(args.image))))
            else:
                orig_prompt = prompt = generate_prompt(args.prompt)
                print(prompt)
                with torch.no_grad():
                    image = run_inpaint(torch.zeros(size, size, 3), mask,\
                                        prompt=prompt, guidance_scale=7.0, model=args.model).to(torch.uint8)
                    #image = run_supersample(image, mask, prompt)
            all_images = [image.detach()]
        else:
            image = gen_image.to(torch.uint8)

        # --- estimate depth ---
        if pts_3d is None: 
            pts_3d, rgb_3d, world2cam, all_cam2world, intrinsics, dm, conf = img_to_pts_3d(all_images)
            scale = median_scene_distance(pts_3d, world2cam) / 10.0
            pts_3d, rgb_3d = density_pruning_py3d(pts_3d, rgb_3d)

        # --- establish camera parameters ---
        if args.renderer == 'py3d':
            cameras = PerspectiveCameras(
                in_ndc=False,
                focal_length=((intrinsics[0,0], intrinsics[1,1]),),
                principal_point=((intrinsics[0,2], intrinsics[1,2]),),
                image_size=torch.ones(1, 2) * size,
                device='cuda',
            )

        # --- rerun logging --- 
        see = lambda x: x.detach().cpu().numpy()
        inpy = see(intrinsics)
        rr.set_time_sequence("frame", idx+1)
        rr.log("world/points", rr.Points3D(see(pts_3d), colors=see(rgb_3d)))
        rr.log("world/camera", rr.Transform3D(translation=see(world2cam[:3, 3]),
                                              mat3x3=see(world2cam[:3, :3]), from_parent=True))
        rr.log("world/camera/image", rr.Pinhole(resolution=[size, size], focal_length=[inpy[0,0], inpy[1,1]], principal_point=[inpy[0,-1], inpy[1,-1]]))
        rr.log("world/camera/image", rr.Image(see(image)).compress(jpeg_quality=75))
        rr.log("world/camera/mask", rr.Pinhole(resolution=[size, size], focal_length=[inpy[0,0], inpy[1,1]], principal_point=[inpy[0,-1], inpy[1,-1]]))
        rr.log("world/camera/mask", rr.Image((mask.float() * 255.0).to(torch.uint8).cpu().numpy()).compress(jpeg_quality=100))

        # --- get user input ---
        inpaint = False
        tl, br = None, None
        print("press (w, a, s, d, q, e) move, (f)ill, (u)psample, (k)ill, (b)reakpoint, (i)npaint region, or (t)ext for stable diffusion...")
        user_input, scale = generate_control(args.control, scale, idx)
        if user_input.lower() in ['w', 'a', 's', 'd', 'q', 'e']:
            world2cam = move_camera(world2cam, user_input.lower(), scale)  # Assuming an amount of 0.1 for movement/rotation
            print(f"{user_input} --> camera moved/rotated, extrinsics.") #, world2cam)
        elif user_input.lower() == 'f':
            print(f"{user_input} --> fill...")
            inpaint = True
            #prompt = ''
        elif user_input.lower() == 'u':
            print(f"{user_input} --> upsample...")
            breakpoint()
        elif user_input.lower() == 'k':
            print(f"{user_input} --> kill...")
            break
        elif user_input.lower() == 'b':
            print(f"{user_input} --> breakpoint...")
            breakpoint()
        elif user_input.lower() == 'i':
            print(f"{user_input} --> inpaint region...")
            tl, br = select_bounding_box(see(image))
            prompt = input(f"{user_input} --> enter stable diffusion prompt: ")
            inpaint = True
        else:
            prompt = input(f"{user_input} --> enter stable diffusion prompt: ")
            inpaint = True

        # --- turn 3d points to image ---
        gen_image = pts_3d_to_img(pts_3d, rgb_3d, intrinsics, world2cam, (size, size), cameras, scale, bbox=(tl, br))
        mask = ((gen_image == -255).sum(dim=2) == 3)# | (conf[-1] < 1.0)
        #gen_image = fill(gen_image) # blur points to make a smooth image

        if inpaint: 
            # --- inpaint pipeline ---
            gen_image[mask] = -1.0
            with torch.no_grad():
                gen_image[mask] = run_inpaint(gen_image, mask.float(), prompt=prompt, model=args.model)[mask]
            gen_image = gen_image.to(torch.uint8)
            #gen_image = run_supersample(gen_image, mask, prompt)

            '''
            # use LPIPS to ditch ugly
            score_map, unrealistic_mask = evaluate_image_patches(all_images[0], gen_image)
            urm = unrealistic_mask.reshape(8, 8).T
            urm = einops.repeat(urm, 'h w -> 1 3 h w')
            urm = F.interpolate(urm.float(), (512, 512))
            urm = einops.rearrange(urm.bool(), 'b c h w -> h w (b c)')
            gen_image[urm] = 0
            '''

            # --- add to duster list ---
            all_images.append(gen_image.detach())
            all_cam2world.append(torch.linalg.inv(world2cam))

            # --- lift img to 3d ---
            pts_3d, rgb_3d, world2cam, all_cam2world, intrinsics, dm, conf = img_to_pts_3d(all_images, all_cam2world, intrinsics, dm, conf)
            scale = median_scene_distance(pts_3d, world2cam) / 10.0
            #pts_3d, rgb_3d = density_pruning_py3d(pts_3d, rgb_3d)
        idx += 1
    rr.script_teardown(args)

    if args.control != 'me':
        data = {'meta_idx': meta_idx,\
                'prompt': orig_prompt,\
                'sequence': sequence}#, 'images': all_images, 'cam2world': all_cam2world, 'intrinsics': intrinsics}

        start = Image.fromarray(all_images[0].cpu().numpy())
        end = Image.fromarray(all_images[-1].cpu().numpy())
        start.save(f'./outputs/imgs/{meta_idx}_start.png')
        end.save(f'./outputs/imgs/{meta_idx}_end.png')
        pickle.dump(data, open(f'./outputs/pickles/{meta_idx}.pkl', 'wb'))

if __name__ == "__main__":
    # --- procedure ---
    # 1. load an input image
    # 2. estimate depth
    # 3. use dust3r to lift to 3D point cloud 
    # 4. move around and apply delta extrinsics
    # 5. re-render new image with mask
    # 6. in paint black with diffusion
    parser = ArgumentParser(description="Build your own adventure.")
    rr.script_add_args(parser)
    parser.add_argument('--depth', type=str, default='dust', help='da / dust')
    parser.add_argument('--renderer', type=str, default='py3d', help='raster / py3d')
    parser.add_argument('--prompt', type=str, default='combo', help='me / doors / auto / combo / default')
    parser.add_argument('--control', type=str, default='auto', help='me / doors / auto')
    parser.add_argument('--image', type=str, default='gen', help='gen / path')
    parser.add_argument('--model', type=str, default='sd2', help='sd2 / if')
    args = parser.parse_args()

    #with torch.autocast(device_type="cuda"):
    #with torch.no_grad():
    pickles = sorted([x for x in os.listdir('./outputs/pickles/') if 'pkl' in x], key=lambda x: int(x.split('.')[0]))
    how_far = int(pickles[-1].split('.')[0]) if len(pickles) > 0 else 0

    # OOM after 130 or so
    for meta_idx in range(100):
        main(args, meta_idx+how_far)  
