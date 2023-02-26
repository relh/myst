#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

os.environ['DECORD_EOF_RETRY_MAX'] = str(20480)

import gc
import getopt
import glob
import json
import math
import os
import pdb
import pickle
import random
import shutil
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict

import albumentations as albu
import cv2
import decord as de
import einops
import gflags
import numpy
import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from cuml.cluster import HDBSCAN
from cupyx.profiler import benchmark
from decord import VideoLoader, VideoReader, cpu, gpu
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad
from kornia.geometry.ransac import RANSAC
from PIL import Image
from torch import distributed as dist
from torch import nn as nn
from torch.cuda.amp import autocast
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

import seeem
from dataset import Coherence_Dataset, MiniDataset
from functions import *

random.seed(0)
inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

'''
This standalone file generates all the labels needed for the present iteration
'''

torch.set_grad_enabled(False)
dataset = 'ego4d'

if dataset == 'epickitchens':
    data_path = '/x/relh/epickitchens/'
    video_path = '/y/relh/epickitchens/videos/'
    depth_path = '/y/relh/epickitchens/depth/'
    dummy_frame_path = '/y/relh/epickitchens/new_frames/'
    frame_path = '/x/relh/epickitchens/frames/'
    cache_path = '/x/relh/epickitchens/cache/'
    people_path = '/x/relh/epickitchens/people/'
    root_flow_path = '/x/relh/epickitchens/flow/'
elif dataset == 'ego4d':
    data_path = '/z/relh/ego4d/'
    video_path = '/Pool2/users/relh/ego4d_data/v1/full_scale/'
    depth_path = '/z/relh/ego4d/depth/'
    dummy_frame_path = '/z/relh/ego4d/new_frames/'
    frame_path = '/z/relh/ego4d/frames/'
    cache_path = '/z/relh/ego4d/cache/'
    people_path = '/z/relh/ego4d/people/'
    root_flow_path = '/z/relh/ego4d/flow/'

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

if __name__ == "__main__":
    # for all datasets
    # find places where flow bridges
    # calculate epipole
    # then load contact points
    # from contact points make trajectories

    img_size = (576, 1024)
    embed_size = (144, 256)
    anno_size = (1080, 1920)
    fps_offset_mult = 1

    if 'ego4d' in dataset:
        img_size = (648, 864)
        embed_size = (162, 216)
        anno_size = (1440, 1920)
        fps_offset_mult = 2
        object_threshold = 0.88

    ddp = False
    batch_size = 20
    args = {'fps_offset_mult': fps_offset_mult, 'offset': 10}
    args = Struct(**args)
    args.ransac_threshold = 3.3e-7 * args.offset * args.fps_offset_mult

    #train_frames = pickle.load(open(f'./util/10_{dataset}_train_frames.pkl', 'rb'))
    valid_frames = pickle.load(open(f'./util/10_{dataset}_valid_frames.pkl', 'rb'))

    #random.shuffle(train_frames)
    random.shuffle(valid_frames)

    #train_data = Coherence_Dataset(train_frames, dataset, data_path, img_size, train=True)
    valid_data = Coherence_Dataset(valid_frames, dataset, data_path, img_size)

    #train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0 if ddp else 2)#, sampler=train_sampler)#, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=0 if ddp else 0)#, sampler=valid_sampler)#, pin_memory=True)

    mesh_grids = torch.stack(list(torch.meshgrid(torch.linspace(-1, 1, steps=img_size[1]), torch.linspace(-1, 1, img_size[0]), indexing='xy')))
    mesh_grids = einops.repeat(mesh_grids, 'c h w -> b h w c', b=batch_size).cuda(non_blocking=True)

    ransac = RANSAC(model_type='fundamental', inl_th=args.ransac_threshold, batch_size=4096, max_iter=5, confidence=0.99999, max_lo_iters=5)
    my_seeem = seeem.WebPage('/home/relh/public_html/experiments/test_new_epipoles_v8/index.html')

    for i, batch in enumerate(valid_loader):
        now_future_flow = batch['now']['flow_n_f'].cuda(non_blocking=True).float()
        future_now_flow = batch['future']['flow_f_n'].cuda(non_blocking=True).float()

        now_rgb = batch['now']['rgb'].float().cuda(non_blocking=True)
        future_rgb = batch['future']['rgb'].float().cuda(non_blocking=True)

        now_people = batch['now']['people'].float().bool().cuda(non_blocking=True)
        future_people = batch['future']['people'].float().bool().cuda(non_blocking=True)

        now_future_corr_grid = build_corr_grid(now_future_flow, mesh_grids, args)
        future_now_corr_grid = build_corr_grid(future_now_flow, mesh_grids, args)

        now_future_rewarp_corr_grid = build_rewarp_grid(mesh_grids, future_now_corr_grid, now_future_corr_grid)
        future_now_rewarp_corr_grid = build_rewarp_grid(mesh_grids, now_future_corr_grid, future_now_corr_grid)

        now_future_cycle_inconsistent = cycle_inconsistent(now_future_flow, now_future_rewarp_corr_grid, now_future_corr_grid, args)
        future_now_cycle_inconsistent = cycle_inconsistent(future_now_flow, future_now_rewarp_corr_grid, future_now_corr_grid, args)

        now_future_F_mat, _ = cv2_fit_motion_model(now_rgb, future_rgb, ~now_people, now_future_cycle_inconsistent, now_future_corr_grid, ransac, 1.0, mesh_grids, args)
        future_now_F_mat, _ = cv2_fit_motion_model(future_rgb, now_rgb, ~future_people, future_now_cycle_inconsistent, future_now_corr_grid, ransac, 1.0, mesh_grids, args)
        #now_future_F_mat, _ = fit_motion_model(~now_people, now_future_cycle_inconsistent, now_future_corr_grid, ransac, 1.0, mesh_grids, args)
        #future_now_F_mat, _ = fit_motion_model(~future_people, future_now_cycle_inconsistent, future_now_corr_grid, ransac, 1.0, mesh_grids, args)

        for b in range(now_rgb.shape[0]):
            this_F = now_future_F_mat[b].detach().cpu().numpy()
            that_F = future_now_F_mat[b].detach().cpu().numpy()

            #this_epipole = np.matmul(this_F, now_future_epipoles[b].cpu().numpy())
            #that_epipole = np.matmul(that_F, future_now_epipoles[b].cpu().numpy())

            # Select a point in the first image to project to the second image
            #pt1 = np.array([100, 200], dtype=np.float32).reshape(1, 1, -1)

            # Project the point to the second image using the fundamental matrix
            #pt2, _ = cv2.correctMatches(F, pts1.reshape(1, -1, 2), pts2.reshape(1, -1, 2), pt1)

            # Draw the correspondences and projected point on the images
            #img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)
            print(this_F)
            F = torch.tensor(this_F).float()
            F = np.array(F)

            now_future_epipoles = scipy.linalg.null_space(F) / scipy.linalg.null_space(F)[-1]
            future_now_epipoles = scipy.linalg.null_space(F.T) / scipy.linalg.null_space(F.T)[-1]

            #unnorm_now_epipoles = now_future_epipoles.squeeze() / now_future_epipoles[:, -1]
            #unnorm_future_epipoles = future_now_epipoles.squeeze() / future_now_epipoles[:, -1]
            #now_future_epipoles = torch.tensor(now_future_epipoles).clamp(0, 648)
            #future_now_epipoles = torch.tensor(future_now_epipoles).clamp(0, 648)

            now_epipoles = torch.zeros((now_rgb.shape[0], 1, now_rgb.shape[2], now_rgb.shape[3]))
            future_epipoles = torch.zeros((now_rgb.shape[0], 1, now_rgb.shape[2], now_rgb.shape[3]))

            #now_future_epipoles[0] = ((now_future_epipoles[0] + 1.0) * now_rgb.shape[3] / 2.0)#.clamp(0, now_rgb.shape[2]-1)
            #now_future_epipoles[1] = ((now_future_epipoles[1] + 1.0) * now_rgb.shape[2] / 2.0)#.clamp(0, now_rgb.shape[3]-1)
            #future_now_epipoles[0] = ((future_now_epipoles[0] + 1.0) * now_rgb.shape[3] / 2.0)#.clamp(0, now_rgb.shape[2]-1)
            #future_now_epipoles[1] = ((future_now_epipoles[1] + 1.0) * now_rgb.shape[2] / 2.0)#.clamp(0, now_rgb.shape[3]-1)

            #now_future_epipoles = now_future_epipoles#.int()
            #future_now_epipoles = future_now_epipoles#.int()

            #now_epipoles = F.interpolate(now_epipoles, size=(now_rgb.shape[2] // 5, now_rgb.shape[3] // 5))
            #(now_future_epipoles[:, 0]-15).clamp(0, now_rgb.shape[2]).int():(now_future_epipoles[:, 0]+15).clamp(0, now_rgb.shape[2]).int(),\
            #(now_future_epipoles[:, 1]-15).clamp(0, now_rgb.shape[3]).int():(now_future_epipoles[:, 1]+15).clamp(0, now_rgb.shape[3]).int()] = 1
            #now_future_epipoles = #abs(now_future_epipoles)
            #future_now_epipoles = #abs(future_now_epipoles)

            img1 = inv_normalize(now_rgb[b].clone()).permute(1, 2, 0).detach().cpu().numpy()  # convert PyTorch tensor to numpy array
            img2 = inv_normalize(future_rgb[b].clone()).permute(1, 2, 0).detach().cpu().numpy()

            img_left = cv2.cvtColor(np.uint8(img1*255), cv2.COLOR_RGB2BGR)  # convert numpy array to OpenCV image
            img_right = cv2.cvtColor(np.uint8(img2*255), cv2.COLOR_RGB2BGR)

            #if b == 1:
            #    pdb.set_trace()
            points_left = np.array([[50, 50], [500, 50], [50, 500], [500, 500], [250, 250], [800, 50], [800, 600]])
            points_right = np.array([[50, 50], [500, 50], [50, 500], [500, 500], [250, 250], [800, 50], [800, 600]])

            # Compute the epipolar lines on the right image
            lines_right = cv2.computeCorrespondEpilines(points_left.reshape(-1, 1, 2), 1, F)
            lines_right = lines_right.reshape(-1, 3)

            # Draw the epipolar lines on the right image
            #for line in lines_right:
            #    x0, y0 = map(int, [0, -line[2] / line[1]])
            #    x1, y1 = map(int, [img_right.shape[1], -(line[2] + line[0] * img_right.shape[1]) / line[1]])
            #    cv2.line(img_right, (x0, y0), (x1, y1), (0, 255, 0), 1)

                # Draw the epipolar lines and line segments on the right image
            for point_left, line_right, point_right in zip(points_left, lines_right, points_right):
                color = tuple(np.random.randint(0, 255, size=3).tolist())  # Random color for each line
                x0, y0 = map(int, [0, -line_right[2] / line_right[1]])
                x1, y1 = map(int, [img_right.shape[1], -(line_right[2] + line_right[0] * img_right.shape[1]) / line_right[1]])
                cv2.line(img_right, (x0, y0), (x1, y1), color, 3)
                #cv2.line(img_right, tuple(point_right), (x0, y0), color, 1)
                cv2.circle(img_left, tuple(point_left), 5, color, 3)

            b = int(b)
            for x in range(-30,30):
                for y in range(-30,30):
                    now_epipoles[b, 0, \
                                 int(min(max(now_future_epipoles[1][0]+x, 0), now_epipoles.shape[2]-1)),\
                                 int(min(max(now_future_epipoles[0][0]+y, 0), now_epipoles.shape[3]-1))] = 0.1 * (31 - abs(x)) * (31 - abs(y))
                    future_epipoles[b, 0, \
                                 int(min(max(future_now_epipoles[1][0]+x, 0), now_epipoles.shape[2]-1)),\
                                 int(min(max(future_now_epipoles[0][0]+y, 0), now_epipoles.shape[3]-1))] = 0.1 * (31 - abs(x)) * (31 - abs(y))

            # Save the image with the epipolar lines drawn
            my_seeem.store_image(img_left, 'epipolar', i*batch_size+b, option='cv2')
            my_seeem.store_image(now_rgb[b], 'rgb', i*batch_size+b, option='rgb')
            my_seeem.store_image(now_future_flow[b], 'flow', i*batch_size+b, option='flow')
            my_seeem.store_image(now_epipoles[b], 'epipole', i*batch_size+b, option='save')
            my_seeem.store_image(now_future_cycle_inconsistent[b], 'cycle', i*batch_size+b, option='save')

            my_seeem.store_image(img_right, 'epipolar', i*batch_size+b+0.5, option='cv2')
            my_seeem.store_image(future_rgb[b], f'rgb', i*batch_size+b+0.5, option='rgb')
            my_seeem.store_image(future_now_flow[b], 'flow', i*batch_size+b+0.5, option='flow')
            my_seeem.store_image(future_epipoles[b], 'epipole', i*batch_size+b+0.5, option='save')
            my_seeem.store_image(future_now_cycle_inconsistent[b], 'cycle', i*batch_size+b+0.5, option='save')

        #if i > 3:
        break

    my_seeem.write()
