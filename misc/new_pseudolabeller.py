#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

os.environ['DECORD_EOF_RETRY_MAX'] = str(20480)

import os
import pickle
import random
import sys

import cv2
import einops
import numpy
import numpy as np
import scipy
import seeem
import torch
import torch.utils.data
from functions import *
from kornia.geometry.ransac import RANSAC
from torch.utils.data import DataLoader

from dataset import Coherence_Dataset

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
    batch_size = 50
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

    #ransac = RANSAC(model_type='fundamental', inl_th=0.010, batch_size=4096, max_iter=25, confidence=0.999999, max_lo_iters=25)
    my_seeem = seeem.WebPage('/home/relh/public_html/experiments/test_new_epipoles_v12/index.html')

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

        #alt_now_future_F_mat, _ = cv2_fit_motion_model(now_rgb, future_rgb, ~now_people, now_future_cycle_inconsistent, now_future_corr_grid, ransac, 1.0, mesh_grids, args)
        #alt_future_now_F_mat, _ = cv2_fit_motion_model(future_rgb, now_rgb, ~future_people, future_now_cycle_inconsistent, future_now_corr_grid, ransac, 1.0, mesh_grids, args)
        confidences = [0.999, 0.9999, 0.99999, 0.999999, 0.9999999]
        max_lo_iters = [1, 5, 10, 15, 20, 25, 50]
        max_iters = [1, 5, 10, 15, 20, 25, 50]
        batch_sizes = [512, 1024, 2096, 4096, 8100]
        ransac_thresholds = [3.3e-7, 1e-7, 1e-6, 1e-8]
        ransac_thresholds = [x * args.offset * args.fps_offset_mult for x in ransac_thresholds]

        def ransac_tune(bs: int, mli: int, mi: int, c: float, rt: float) -> float:
            # optimal for learning_rate=0.2, batch_size=4, architecture="conv"
            ransac = RANSAC(model_type='fundamental', inl_th=rt, batch_size=bs, max_iter=mi, confidence=c, max_lo_iters=mli)
            now_future_F_mat, _ = fit_motion_model(~now_people, now_future_cycle_inconsistent, now_future_corr_grid, ransac, 1.0, mesh_grids, args)
            future_now_F_mat, _ = fit_motion_model(~future_people, future_now_cycle_inconsistent, future_now_corr_grid, ransac, 1.0, mesh_grids, args)

            diff = []
            for b in range(now_rgb.shape[0]):
                that_F = future_now_F_mat[b].detach().cpu().numpy()
                this_F = now_future_F_mat[b].detach().cpu().numpy()

                diff.append((abs(this_F.T - that_F)).sum())
            return diff

        # Instrumentation class is used for functions with multiple inputs
        # (positional and/or keywords)
        '''
        parametrization = ng.p.Instrumentation(
            bs=ng.p.Scalar(lower=512, upper=8000).set_integer_casting(),
            mli=ng.p.Scalar(lower=1, upper=50).set_integer_casting(),
            mi=ng.p.Scalar(lower=1, upper=50).set_integer_casting(),
            c=ng.p.Scalar(lower=0.999, upper=0.99999999),
            rt=ng.p.Scalar(lower=1e-7, upper=1e-3),
        )
        '''
        #bs = 8400
        #mli = 97
        #mi = 64
        #c = 0.9997846007267676 
        #rt = 2.5163064521501138e-05

        #bs = 1412
        #mli = 18
        #mi = 64
        #c = 0.9992188484015111
        #rt = 1.2368851783195201e-07

        #bs = 4096 
        #mi = 25 
        #mli = 25 
        #c = 0.999999
        #rt = 3.3e-7 * args.offset * args.fps_offset_mult

        bs = 8096
        mi = 1000
        mli = 100
        c = 0.9999999
        rt = 0.5
        #3.3e-8 * args.offset * args.fps_offset_mult 

        #optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=100)
        #recommendation = optimizer.minimize(ransac_tune)

        #print(recommendation.kwargs)  # shows the recommended keyword arguments of the function
        # >>> {'learning_rate': 0.1998, 'batch_size': 4, 'architecture': 'conv'}

        #pdb.set_trace()
        #Instrumentation(Tuple(),Dict(bs=Scalar{Cl(512,10000,b),Int}[sigma=Scalar{exp=2.03}],c=Scalar{Cl(0.999,0.999999999,b)}[sigma=Scalar{exp=2.03}],mi=Scalar{Cl(1,100,b),Int}[sigma=Scalar{exp=2.03}],mli=Scalar{Cl(1,100,b),Int}[sigma=Scalar{exp=2.03}],rt=Log{Cl(-12.000000000000002,-6.000000000000002,b),exp=4.64})):((), 
        #{'bs': 1412, 'mli': 18, 'mi': 64, 'c': 0.9992188484015111,
        #'rt': 1.2368851783195201e-07}) with losses 371854.70301332476

        # least impact to most 
        smallest_diff = sys.maxsize
        '''
        for bs in batch_sizes:
            for mli in max_lo_iters:
                for mi in max_iters:
                    for c in confidences:
                        for rt in ransac_thresholds:
        '''
        ransac = RANSAC(model_type='fundamental', inl_th=rt, batch_size=bs, max_iter=mi, confidence=c, max_lo_iters=mli)
        now_future_F_mat, _ = fit_motion_model(~now_people, now_future_cycle_inconsistent, now_future_corr_grid, ransac, 1.0, mesh_grids, args)
        future_now_F_mat, _ = fit_motion_model(~future_people, future_now_cycle_inconsistent, future_now_corr_grid, ransac, 1.0, mesh_grids, args)

        diff = []
        for b in range(now_rgb.shape[0]):
            that_F = future_now_F_mat[b].detach().cpu().numpy()
            this_F = now_future_F_mat[b].detach().cpu().numpy()
            diff.append((abs(this_F.T - that_F)).sum())

        sum_diff = sum(diff)
        if sum_diff < smallest_diff:
            smallest_diff = sum_diff
            print(f'rt: {rt}, c: {c}, mli: {mli}, mi: {mi}, bs: {bs}, sd: {sum_diff}')

        for b in range(now_rgb.shape[0]):
            that_F = future_now_F_mat[b].detach().cpu().numpy()
            this_F = now_future_F_mat[b].detach().cpu().numpy()

            '''
            h, w = 648.0, 864.0

            sx, sy = w/2.0, h/2.0
            tx, ty = w/2.0, h/2.0

            H = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]])
            H_inv = np.linalg.inv(H)
            H_inv_transpose = H_inv.T
            '''

            pts_nonnorm = np.array([[50, 50], [500, 50], [50, 500], [500, 500], [250, 250], [800, 50], [800, 600]])

            # Normalized pixel coordinates of correspondences
            #pts_norm = (H_inv @ np.hstack((pts_nonnorm, np.ones((pts_nonnorm.shape[0], 1)))).T).T[:,:2]

            # Compute the epipolar lines on the right image
            #epilines_norm = F @ np.hstack((points_left, np.ones((points_left.shape[0], 1)))).T

            this_F = torch.tensor(this_F).float()
            this_F = np.array(this_F)
            this_lines_norm = cv2.computeCorrespondEpilines(pts_nonnorm.reshape(-1, 1, 2), 2, that_F)
            this_lines_norm = this_lines_norm.reshape(-1, 3)
            this_lines_nonnorm = (this_lines_norm.T).T
            #this_lines_nonnorm = (H_inv_transpose @ this_lines_norm.T).T

            that_F = torch.tensor(that_F).float()
            that_F = np.array(that_F)
            that_lines_norm = cv2.computeCorrespondEpilines(pts_nonnorm.reshape(-1, 1, 2), 2, this_F)
            that_lines_norm = that_lines_norm.reshape(-1, 3)
            that_lines_nonnorm = (that_lines_norm.T).T
            #that_lines_nonnorm = (H_inv_transpose @ that_lines_norm.T).T

            img1 = inv_normalize(now_rgb[b].clone()).permute(1, 2, 0).detach().cpu().numpy()  # convert PyTorch tensor to numpy array
            img2 = inv_normalize(future_rgb[b].clone()).permute(1, 2, 0).detach().cpu().numpy()

            img_left = cv2.cvtColor(np.uint8(img1*255), cv2.COLOR_RGB2BGR)  # convert numpy array to OpenCV image
            img_right = cv2.cvtColor(np.uint8(img2*255), cv2.COLOR_RGB2BGR)

            # Draw the epipolar lines and line segments on the right image
            for point, this_line, that_line in zip(pts_nonnorm, this_lines_nonnorm, that_lines_nonnorm):
                color = tuple(np.random.randint(0, 255, size=3).tolist())  # Random color for each line
                x0, y0 = map(int, [0, -this_line[2]/this_line[1]])
                x1, y1 = map(int, [img_right.shape[1], -(this_line[2]+this_line[0]*img_right.shape[1])/this_line[1]])
                cv2.line(img_right, (x0, y0), (x1, y1), color, 3)
                cv2.circle(img_right, tuple(point), 5, color, -3)

                x0, y0 = map(int, [0, -that_line[2]/that_line[1]])
                x1, y1 = map(int, [img_right.shape[1], -(that_line[2]+that_line[0]*img_right.shape[1])/that_line[1]])
                cv2.line(img_left, (x0, y0), (x1, y1), color, 3)
                cv2.circle(img_left, tuple(point), 5, color, -3)

            now_future_epipoles = scipy.linalg.null_space(this_F)# / scipy.linalg.null_space(F)[-1]
            future_now_epipoles = scipy.linalg.null_space(that_F)# / scipy.linalg.null_space(F.T)[-1]

            #now_future_epipoles = np.dot(H, norm_now_future_epipoles)
            now_future_epipoles /= now_future_epipoles[2]

            #future_now_epipoles = np.dot(H, norm_future_now_epipoles)
            future_now_epipoles /= future_now_epipoles[2]

            now_future_epipoles = tuple(now_future_epipoles.T[0][:2].astype(int))
            future_now_epipoles = tuple(future_now_epipoles.T[0][:2].astype(int))

            now_epipoles = torch.zeros((now_rgb.shape[0], 1, now_rgb.shape[2], now_rgb.shape[3]))
            future_epipoles = torch.zeros((now_rgb.shape[0], 1, now_rgb.shape[2], now_rgb.shape[3]))

            color = tuple(np.random.randint(0, 255, size=3).tolist())  # Random color for each line
            cv2.circle(img_right, future_now_epipoles, 5, color, -3)
            cv2.circle(img_left, now_future_epipoles, 5, color, -3)

            b = int(b)
            for x in range(-30,30):
                for y in range(-30,30):
                    now_epipoles[b, 0, \
                                 int(min(max(now_future_epipoles[1]+x, 0), now_epipoles.shape[2]-1)),\
                                 int(min(max(now_future_epipoles[0]+y, 0), now_epipoles.shape[3]-1))] = 0.1 * (31 - abs(x)) * (31 - abs(y))
                    future_epipoles[b, 0, \
                                 int(min(max(future_now_epipoles[1]+x, 0), now_epipoles.shape[2]-1)),\
                                 int(min(max(future_now_epipoles[0]+y, 0), now_epipoles.shape[3]-1))] = 0.1 * (31 - abs(x)) * (31 - abs(y))

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

        #if i > 1:
        break

    my_seeem.write()
