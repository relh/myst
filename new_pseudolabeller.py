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

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

if __name__ == "__main__":
    # for all datasets
    # find places where flow bridges
    # calculate epipole
    # then load contact points
    # from contact points make trajectories
    ddp = False
    batch_size = 10
    args = {'fps_offset_mult': fps_offset_mult, 'offset': 10}
    args = Struct(**args)

    train_frames = pickle.load(open(f'./util/10_{dataset}_train_frames.pkl', 'rb'))
    valid_frames = pickle.load(open(f'./util/10_{dataset}_valid_frames.pkl', 'rb'))

    train_data = Coherence_Dataset(train_frames, dataset, data_path, img_size, train=True)
    valid_data = Coherence_Dataset(valid_frames, dataset, data_path, img_size)

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=0 if ddp else 2)#, sampler=train_sampler)#, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=0 if ddp else 2)#, sampler=valid_sampler)#, pin_memory=True)

    mesh_grids = torch.stack(list(torch.meshgrid(torch.linspace(-1, 1, steps=img_size[1]), torch.linspace(-1, 1, img_size[0]), indexing='xy')))
    mesh_grids = einops.repeat(mesh_grids, 'c h w -> b h w c', b=batch_size).cuda(non_blocking=True)

    ransac = RANSAC(model_type='fundamental', inl_th=3.3e-7, batch_size=4096, max_iter=5, confidence=0.99999, max_lo_iters=5)
    my_seeem = seeem.WebPage('/home/relh/public_html/experiments/test_epipoles/index.html')

    for i, batch in enumerate(train_loader):
        print(i)
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

        now_future_F_mat, _ = fit_motion_model(~now_people, now_future_cycle_inconsistent, now_future_corr_grid, ransac, 1.0, mesh_grids, args)
        future_now_F_mat, _ = fit_motion_model(~future_people, future_now_cycle_inconsistent, future_now_corr_grid, ransac, 1.0, mesh_grids, args)

        now_future_epipoles = calculate_epipoles(now_future_F_mat)
        future_now_epipoles = calculate_epipoles(future_now_F_mat)

        for b in range(now_rgb.shape[0]):
            print(b)
            my_seeem.store_image(now_rgb[b], 'rgb_0', i*batch_size+b, option='rgb')
            my_seeem.store_image(future_rgb[b], f'rgb_{0+args.offset}', i*batch_size+b, option='rgb')

            my_seeem.store_image(now_future_flow[b], 'flow_pos', i*batch_size+b, option='flow')
            my_seeem.store_image(future_now_flow[b], 'flow_neg', i*batch_size+b, option='flow')

            #my_seeem.store_image(dummy_image, 'save', i*batch_size+b, option='save')
            #my_seeem.store_image(dummy_image, 'pca', i*batch_size+b, option='pca')
            #my_seeem.store_image(dummy_image, 'rgb', i*batch_size+b, option='rgb')

        my_seeem.write()
        break
