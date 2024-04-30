#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch

torch.backends.cuda.preferred_linalg_library()

import os
import pickle
import sys
import termios
import tty
from argparse import ArgumentParser

import rerun as rr  # pip install rerun-sdk
from PIL import Image
from pytorch3d.renderer import OrthographicCameras, PerspectiveCameras

from misc.camera import pts_3d_to_img_raster, pts_cam_to_world
from misc.controller import generate_control
from misc.da_3d import img_to_pts_3d_da
from misc.dust_3d import img_to_pts_3d_dust
from misc.imutils import fill
from misc.inpaint import run_inpaint
from misc.merge import merge_and_filter
from misc.prompt import generate_prompt
from misc.prune import density_pruning_py3d
from misc.renderer import pts_3d_to_img_py3d
from misc.scale import project_and_scale_points
from misc.supersample import run_supersample
from misc.write import write_inference_html


def get_keypress():
    fd = sys.stdin.fileno()
    original_attributes = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)  # Read a single character
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original_attributes)
    return key


def move_camera(extrinsics, direction, amount):
    """
    Move the camera considering its orientation or rotate left/right.
    
    :param extrinsics: The current extrinsics matrix.
    :param direction: 'w' (forward), 's' (backward), 'a' (left rotate), 'd' (right rotate).
    :param amount: The amount to move or rotate.
    """
    # Extract rotation matrix R and translation vector t from the extrinsics matrix
    R = extrinsics[:3, :3]
    
    if direction in ['w', 's']:
        # Direction vector for forward/backward 
        amount = 0.5 * (-amount if direction == 'w' else amount)
        extrinsics[:3, 3] += torch.tensor([0, 0, amount], device=extrinsics.device).float()
    elif direction in ['q', 'e']:
        # Rotation angle (in radians). Positive for 'e' (down), negative for 'q' (up)
        angle = torch.tensor(-amount if direction == 'e' else amount)
        # Rotation matrix around the X-axis (assuming X is forward/backward)
        rotation_matrix = torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(angle), -torch.sin(angle), 0],
            [0, torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 0, 1]
        ], device=extrinsics.device)
        # Apply rotation to the extrinsics matrix
        extrinsics = torch.matmul(rotation_matrix, extrinsics)
    elif direction in ['a', 'd']:
        # Rotation angle (in radians). Positive for 'd' (right), negative for 'a' (left)
        angle = torch.tensor(-amount if direction == 'd' else amount)
        # Rotation matrix around the Y-axis (assuming Y is up)
        rotation_matrix = torch.tensor([
            [torch.cos(angle), 0, torch.sin(angle), 0],
            [0, 1, 0, 0],
            [-torch.sin(angle), 0, torch.cos(angle), 0],
            [0, 0, 0, 1]
        ], device=extrinsics.device)
        # Apply rotation to the extrinsics matrix
        extrinsics = torch.matmul(rotation_matrix, extrinsics)
    return extrinsics


def main(args, meta_idx):
    # --- setup rerun args ---
    rr.script_setup(args, f"{meta_idx}myst")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    img_to_pts_3d = img_to_pts_3d_da if args.depth == 'da' else img_to_pts_3d_dust
    pts_3d_to_img = pts_3d_to_img_raster if args.renderer == 'raster' else pts_3d_to_img_py3d 

    if args.controller == 'ai':
        sequence = generate_control()
        print(f'ai sequence is... {sequence}')

    pts_3d = None
    image = None
    imsize = 512
    idx = 0
    while True:
        idx += 1

        # --- setup initial scene ---
        if image is None: 
            #prompt = input(f"enter stable diffusion initial scene: ")
            #prompt = 'a high-resolution photo of a large kitchen.'
            prompt = generate_prompt()
            print(prompt)

            image = run_inpaint(torch.zeros(imsize, imsize, 3), torch.ones(imsize, imsize), prompt=prompt)
            mask_3d = torch.ones(imsize, imsize)
            all_images = [image]
        else:
            image = gen_image.to(torch.uint8)
            image[image.sum(dim=2) < 10] = 0.0

        # --- estimate depth ---
        if pts_3d is None: 
            pts_3d, rgb_3d, world2cam, all_cam2world, intrinsics, dm = img_to_pts_3d(all_images, None, None)
            pts_3d, rgb_3d = density_pruning_py3d(pts_3d, rgb_3d)

            # --- establish camera parameters ---
            if args.renderer == 'py3d':
                cameras = PerspectiveCameras(
                    device='cuda',
                    R=torch.eye(3).unsqueeze(0),
                    in_ndc=False,
                    T=torch.zeros(1, 3),
                    focal_length=-intrinsics[0,0].unsqueeze(0),
                    principal_point=intrinsics[:2,2].unsqueeze(0),
                    image_size=torch.ones(1, 2) * imsize,
                )

        # --- rerun logging --- 
        see = lambda x: x.detach().cpu().numpy()
        rr.set_time_sequence("frame", idx+1)
        rr.log(f"world/points", rr.Points3D(see(pts_3d), colors=see(rgb_3d)))
        rr.log("world/camera", 
            rr.Transform3D(translation=see(world2cam[:3, 3]),
                           mat3x3=see(world2cam[:3, :3]), from_parent=True))
        inpy = see(intrinsics)
        rr.log("world/camera/image", rr.Pinhole(resolution=[imsize, imsize], focal_length=[inpy[0,0], inpy[1,1]], principal_point=[inpy[0,-1], inpy[1,-1]]))
        rr.log("world/camera/image", rr.Image(see(image)).compress(jpeg_quality=75))
        rr.log("world/camera/mask", rr.Pinhole(resolution=[imsize, imsize], focal_length=[inpy[0,0], inpy[1,1]], principal_point=[inpy[0,-1], inpy[1,-1]]))
        rr.log("world/camera/mask", rr.Image((torch.stack([mask_3d, mask_3d, mask_3d], dim=2).float() * 255.0).to(torch.uint8).cpu().numpy()).compress(jpeg_quality=100))

        # --- get user input ---
        inpaint = False
        print("press (w, a, s, d, q, e) move, (f)ill, (u)psample, (k)ill, (b)reakpoint, or (t)ext for stable diffusion...")
        if args.controller == 'me':
            user_input = get_keypress()
        else:
            if idx >= len(sequence): break
            user_input = sequence[idx]
        if user_input.lower() in ['w', 'a', 's', 'd', 'q', 'e']:
            world2cam = move_camera(world2cam, user_input.lower(), 0.1)  # Assuming an amount of 0.1 for movement/rotation
            print(f"{user_input} --> camera moved/rotated, extrinsics:\n", world2cam)
        elif user_input.lower() == 'f':
            print(f"{user_input} --> fill...")
            inpaint = True
        elif user_input.lower() == 'u':
            print(f"{user_input} --> upsample...")
            breakpoint()
            pts_3d = supersample_point_cloud(pts_3d.to(device))
            image = run_supersample(image, mask, prompt)
            rgb_3d = torch.tensor(np.asarray(color_image[0])).float().to(device)
            rgb_3d = supersample_point_cloud(rgb_3d).to(torch.uint8)
        elif user_input.lower() == 'k':
            print(f"{user_input} --> kill...")
            break
        elif user_input.lower() == 'b':
            print(f"{user_input} --> breakpoint...")
            breakpoint()
        else:
            prompt = input(f"{user_input} --> enter stable diffusion prompt: ")
            inpaint = True

        # --- turn 3d points to image ---
        gen_image = pts_3d_to_img(pts_3d, rgb_3d, intrinsics, world2cam, (imsize, imsize), cameras)

        if inpaint: 
            # --- inpaint pipeline ---
            #gen_image = fill(gen_image)      # blur points to make a smooth image
            mask = gen_image.sum(dim=2) < 10
            gen_image[mask] = -1.0
            sq_init = run_inpaint(gen_image, mask.float(), prompt=prompt)
            gen_image = gen_image.to(torch.uint8)
            gen_image[mask] = sq_init[mask]

            # --- add to duster list ---
            all_images.append(gen_image)
            all_cam2world.append(torch.linalg.inv(world2cam))

            # --- lift img to 3d ---
            pts_3d, rgb_3d, world2cam, all_cam2world, _, dm = img_to_pts_3d(all_images, all_cam2world, intrinsics, dm=dm)
            pts_3d, rgb_3d = density_pruning_py3d(pts_3d, rgb_3d)
    rr.script_teardown(args)

    if args.controller == 'ai':
        data = {'meta_idx': meta_idx,\
                'prompt': prompt,\
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
    parser.add_argument('--views', type=str, default='multi', help='multi / single')
    parser.add_argument('--controller', type=str, default='ai', help='me / ai')
    args = parser.parse_args()

    #with torch.no_grad():
    #with torch.autocast(device_type="cuda"):
    pickles = sorted([x for x in os.listdir('./outputs/pickles/') if 'pkl' in x], key=lambda x: int(x.split('.')[0]))
    how_far = 0
    if len(pickles) > 0: 
        how_far = int(pickles[-1].split('.')[0])

    # OOM after 130 or so
    for meta_idx in range(100):
        main(args, meta_idx+how_far)  
