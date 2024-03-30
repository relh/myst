#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from cuml.cluster import HDBSCAN
from functions import (build_assoc_inputs, build_corr_grid, build_group_inputs,
                       build_rewarp_grid, calculate_average_embedding,
                       cleanse_component, connected_components,
                       cycle_inconsistent, d_flow, epipolar_distance,
                       fit_motion_model, fit_motion_models, flow_above_mean,
                       get_pixel_groups, merge_component, rebase_components,
                       segment_embeddings, store_image)
from inference import make_association_portable
from kornia.geometry.ransac import RANSAC
from models.segmentation_pytorch.configs.segformer_config import config as cfg
from torch import distributed as dist
from torch import nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm
from write import write_index_html


def run_epoch(loader, net, scaler, optimizer, scheduler, epoch, args, is_train=True, visualize=False):
    # --- synchronize workers ---
    if args.ddp:
        dist.barrier()
        if is_train: loader.sampler.set_epoch(epoch)

    # --- initialize variables ---
    pbar = tqdm(loader, ncols=200)
    epoch_len = (len(pbar) * loader.batch_size * args.num_gpus) if args.epoch_len == -1 else args.epoch_len
    torch.set_grad_enabled(is_train)
    torch.backends.cudnn.benchmark = True
    net.train() if is_train else net.eval()
    if not is_train: epoch_len = args.queue_len

    losses, loss, step = 0, 0, 0
    start = time.time()

    dd_ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    aa_ce_loss = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([0.1, 2.0]).cuda())
    ransac = RANSAC(model_type='fundamental', inl_th=(args.ransac_threshold), batch_size=4096, max_iter=5, confidence=0.99999, max_lo_iters=5)
    if not is_train and visualize:
        clust = HDBSCAN(min_samples=args.cluster_min_samples, min_cluster_size=args.cluster_min_size)#, cluster_selection_method='leaf')

    inconsistent = cycle_inconsistent if 'aa' in args.target else cycle_inconsistent # boundary_inconsistent

    with autocast(enabled=True):
        with torch.set_grad_enabled(False):
            # x,y grid for optical flow
            mesh_grids = torch.stack(list(torch.meshgrid(torch.linspace(-1, 1, steps=args.img_size[1]), torch.linspace(-1, 1, args.img_size[0]), indexing='xy')))
            mesh_grids = einops.repeat(mesh_grids, 'c h w -> b h w c', b=args.batch_size // args.num_gpus).cuda(non_blocking=True)

    # --- training loop ---
    for i, batch in enumerate(pbar):
        with autocast(enabled=True):
            if is_train: optimizer.zero_grad(set_to_none=True)
            ddl, aal, ccl, ssl, dfl, pfl = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            with torch.set_grad_enabled(False):
                #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #    with record_function("load_data"):
                # ============= load input/output ===============
                now_future_flow = batch['now']['flow_n_f'].cuda(non_blocking=True)
                future_now_flow = batch['future']['flow_f_n'].cuda(non_blocking=True)

                now_rgb = batch['now']['rgb'].float().cuda(non_blocking=True)
                future_rgb = batch['future']['rgb'].float().cuda(non_blocking=True)

                now_people = batch['now']['people'].float().bool().cuda(non_blocking=True)
                future_people = batch['future']['people'].float().bool().cuda(non_blocking=True)

                if args.double:
                    now_double_flow = batch['double']['flow_n_d'].cuda(non_blocking=True)
                    double_now_flow = batch['double']['flow_d_n'].cuda(non_blocking=True)

                if args.triple:
                    now_triple_flow = batch['triple']['flow_n_t'].cuda(non_blocking=True)
                    triple_now_flow = batch['triple']['flow_t_n'].cuda(non_blocking=True)
                #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                # =============== labels from flow derivative ====================
                if 'df' in args.target or 'pf' in args.target:
                    now_d_flow, future_d_flow = d_flow(now_future_flow), d_flow(future_now_flow)

                    now_future_flow_norm = (torch.linalg.vector_norm(now_future_flow, dim=-1).float() + 1e-5)
                    future_now_flow_norm = (torch.linalg.vector_norm(future_now_flow, dim=-1).float() + 1e-5)

                    now_d_flow_norm = (torch.linalg.vector_norm(now_d_flow, dim=1).float() + 1e-5)
                    future_d_flow_norm = (torch.linalg.vector_norm(future_d_flow, dim=1).float() + 1e-5)
                elif args.dataset == 'playroom' or args.pseudolabels == 'flow':
                    now_sed = (torch.linalg.vector_norm(now_future_flow, dim=-1).float() + 1e-5).detach()
                    future_sed = (torch.linalg.vector_norm(future_now_flow, dim=-1).float() + 1e-5).detach()

                # =============== correlation grids from flow ====================
                #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #    with record_function("corr_grids"):
                        # add norm flow to -1 to 1 for warping
                now_future_corr_grid = build_corr_grid(now_future_flow, mesh_grids, args)
                future_now_corr_grid = build_corr_grid(future_now_flow, mesh_grids, args)
                if args.double: 
                    now_double_corr_grid = build_corr_grid(now_double_flow, mesh_grids, args)
                    double_now_corr_grid = build_corr_grid(double_now_flow, mesh_grids, args)
                if args.triple: 
                    now_triple_corr_grid = build_corr_grid(now_triple_flow, mesh_grids, args)
                    triple_now_corr_grid = build_corr_grid(triple_now_flow, mesh_grids, args)

                now_future_rewarp_corr_grid = build_rewarp_grid(mesh_grids, future_now_corr_grid, now_future_corr_grid)
                future_now_rewarp_corr_grid = build_rewarp_grid(mesh_grids, now_future_corr_grid, future_now_corr_grid)
                if args.double: now_double_rewarp_corr_grid = build_rewarp_grid(mesh_grids, double_now_corr_grid, now_double_corr_grid)
                if args.triple: now_triple_rewarp_corr_grid = build_rewarp_grid(mesh_grids, triple_now_corr_grid, now_triple_corr_grid)
                #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                # =============== mask of cycle consistent warps ====================
                #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #    with record_function("inconsistent"):
                now_future_cycle_inconsistent = inconsistent(now_future_flow, now_future_rewarp_corr_grid, now_future_corr_grid, args)
                future_now_cycle_inconsistent = inconsistent(future_now_flow, future_now_rewarp_corr_grid, future_now_corr_grid, args)
                if args.double: now_double_cycle_inconsistent = inconsistent(now_double_flow, now_double_rewarp_corr_grid, now_double_corr_grid, args)
                if args.triple: now_triple_cycle_inconsistent = inconsistent(now_triple_flow, now_triple_rewarp_corr_grid, now_triple_corr_grid, args)
                #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # =============== background and hand motion models ====================
        with torch.set_grad_enabled(False):
            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #    with record_function("motion_models"):
            if args.dataset == 'playroom':
                now_future_F_mat = torch.zeros((now_people.shape[0], 3, 3)).float().cuda()
                future_now_F_mat = torch.zeros((now_people.shape[0], 3, 3)).float().cuda()
            elif args.motion_model == 'cv2':
                now_future_F_mat, _ = cv2_fit_motion_model(now_rgb, future_rgb)
                future_now_F_mat, _ = cv2_fit_motion_model(future_rgb, now_rgb)
            else:
                now_future_F_mat, _ = fit_motion_model(~now_people, now_future_cycle_inconsistent, now_future_corr_grid, ransac, 1.0, mesh_grids, args)
                future_now_F_mat, _ = fit_motion_model(~future_people, future_now_cycle_inconsistent, future_now_corr_grid, ransac, 1.0, mesh_grids, args)
            if args.double: now_double_F_mat, _ = fit_motion_model(~now_people, now_double_cycle_inconsistent, now_double_corr_grid, ransac, 0.66, mesh_grids, args)
            if args.triple: now_triple_F_mat, _ = fit_motion_model(~now_people, now_triple_cycle_inconsistent, now_triple_corr_grid, ransac, 0.33, mesh_grids, args)
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

            with autocast(enabled=True):
                #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #    with record_function("epipolar"):
                # =============== mask of background and hand motion models ====================
                # if this happens, static camera. Want all flow about mean
                if args.dataset == 'playroom' or args.pseudolabels == 'flow':
                    now_sed = flow_above_mean(now_sed)
                    future_sed = flow_above_mean(future_sed)
                else:
                    now_sed = epipolar_distance(now_future_corr_grid, now_future_F_mat, mesh_grids, args)
                    future_sed = epipolar_distance(future_now_corr_grid, future_now_F_mat, mesh_grids, args)
                    if args.double: now_sed += epipolar_distance(now_double_corr_grid, now_double_F_mat, mesh_grids, args)
                    if args.triple: now_sed += epipolar_distance(now_triple_corr_grid, now_triple_F_mat, mesh_grids, args)
                #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                # =============== grouping based on epipolar distance ====================
                #args.epipolar_threshold = 0.5
                now_thresh = 0.0 #math.log(args.epipolar_threshold) if args.epipolar_log else args.epipolar_threshold
                if args.double: now_thresh += 0.0 #args.epipolar_threshold 
                if args.triple: now_thresh += 0.0 #args.epipolar_threshold 

                now_outl = (now_sed > now_thresh).float()
                future_outl = (future_sed > 0.0).float()

                # =============== people involvement begins ====================
                # threshold epipolar distance and then group connected components
                #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #    with record_function("connected_components"):
                #print(benchmark(connected_components, (now_outl, args), n_repeat=20))
                now_labels, future_labels = connected_components(now_outl, args), connected_components(future_outl, args)
                now_cc_people, future_cc_people = connected_components(now_people.float(), args), connected_components(future_people.float(), args)
                #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                # merge connected components suavely
                future_labels[future_labels != 0.0] += (now_labels.max() + 1)
                future_cc_people[future_cc_people != 0.0] += (now_cc_people.max() + 1)

                if args.merge:
                    now_labels = merge_component(future_labels, now_labels, now_future_corr_grid, future_now_cycle_inconsistent, now_future_cycle_inconsistent)
                    future_labels = merge_component(now_labels, future_labels, future_now_corr_grid, now_future_cycle_inconsistent, future_now_cycle_inconsistent)

                    now_cc_people = merge_component(future_cc_people, now_cc_people, now_future_corr_grid, future_now_cycle_inconsistent, now_future_cycle_inconsistent)
                    future_cc_people = merge_component(now_cc_people, future_cc_people, future_now_corr_grid, now_future_cycle_inconsistent, future_now_cycle_inconsistent)

                now_labels, future_labels = cleanse_component(now_labels), cleanse_component(future_labels)
                future_cc_people, now_cc_people = cleanse_component(future_cc_people), cleanse_component(now_cc_people)

                # split the connected components into people + objects
                now_labels, future_labels = rebase_components(now_labels, future_labels)
                max_label = max((now_labels.max(), future_labels.max()))
                now_split_labels, future_split_labels = now_labels.clone(), future_labels.clone()

                now_split_labels[now_labels != 0 & now_people] = (max_label + now_cc_people[now_labels != 0 & now_people])
                future_split_labels[future_labels != 0 & future_people] = (max_label + future_cc_people[future_labels != 0 & future_people])

                if 'mm' in args.target:
                    # split into cc, then repeat 
                    # threshold epipolar distance and then group connected components
                    # for each cc fit new model

                    now_future_cc_F_mat, now_future_cc_outl = fit_motion_models(now_labels, now_future_corr_grid, ransac, 1, mesh_grids, args)
                    future_now_cc_F_mat, future_now_cc_outl = fit_motion_models(future_labels, future_now_corr_grid, ransac, 1, mesh_grids, args)

                    for b in range(len(now_future_cc_outl)):
                        for k,_ in now_future_cc_outl[b].items():
                            now_future_cc_outl[b][k] = F.interpolate(einops.repeat(now_future_cc_outl[b][k], 'h w -> b c h w', b=1, c=1), size=args.embed_size).squeeze().bool()

                        for k,_ in future_now_cc_outl[b].items():
                            future_now_cc_outl[b][k] = F.interpolate(einops.repeat(future_now_cc_outl[b][k], 'h w -> b c h w', b=1, c=1), size=args.embed_size).squeeze().bool()
                    #now_cc_sed = epipolar_distance(now_future_corr_grid, now_future_cc_F_mat, mesh_grids)
                    #future_cc_sed = epipolar_distance(future_now_corr_grid, future_now_cc_F_mat, mesh_grids)

                    # just need inliers for this
                    #now_inl = (~now_outl.bool()).float() # + ~now_double_outl.bool() + ~now_triple_outl.bool()).float()
                    #future_inl = (~future_outl.bool()).float()

                    # build triplets
                    # cc_outl are negatives, cc_inl are positives
                    # pass cc_outl mask to build_mlp

                # =============== downsample everything to embedding size ====================
                if 'df' in args.target or 'pf' in args.target:
                    now_d_flow = F.interpolate(now_d_flow, size=args.embed_size)
                    future_d_flow = F.interpolate(future_d_flow, size=args.embed_size)

                now_future_cycle_inconsistent = F.interpolate(einops.repeat(now_future_cycle_inconsistent.float(), 'b h w -> b c h w', c=1), size=args.embed_size).squeeze()
                future_now_cycle_inconsistent = F.interpolate(einops.repeat(future_now_cycle_inconsistent.float(), 'b h w -> b c h w', c=1), size=args.embed_size).squeeze()

                now_people = F.interpolate(einops.repeat(now_people.float(), 'b h w -> b c h w', c=1), size=args.embed_size).squeeze()
                future_people = F.interpolate(einops.repeat(future_people.float(), 'b h w -> b c h w', c=1), size=args.embed_size).squeeze()

                now_future_corr_grid = einops.rearrange(F.interpolate(einops.rearrange(now_future_corr_grid, 'b h w c -> b c h w'), size=args.embed_size), 'b c h w -> b h w c')
                future_now_corr_grid = einops.rearrange(F.interpolate(einops.rearrange(future_now_corr_grid, 'b h w c -> b c h w'), size=args.embed_size), 'b c h w -> b h w c')

                now_labels = F.interpolate(einops.repeat(now_labels.float(), 'b h w -> b c h w', c=1), size=args.embed_size)
                future_labels = F.interpolate(einops.repeat(future_labels.float(), 'b h w -> b c h w', c=1), size=args.embed_size)

                if 'dd' in args.target or 'aa' in args.target:
                    now_pixels = get_pixel_groups(now_labels)
                    future_pixels = get_pixel_groups(future_labels)

                now_weight = (1 - now_future_cycle_inconsistent.sum() / (2 * args.embed_size[0] * args.embed_size[1])) 
                future_weight = (1 - future_now_cycle_inconsistent.sum() / (2 * args.embed_size[0] * args.embed_size[1])) 

        # =============== forward ====================
        with autocast(enabled=True):
            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #    with record_function("network"):
            n_o, f_o = net(now_rgb), net(future_rgb)
            #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            #if 'resnet' in args.model:
            #    n_o = {'e': n_o[0][1]}
            #    f_o = {'e': f_o[0][1]}

            if args.rgb:
                now_cat_rgb = F.interpolate(now_rgb.clone(), size=args.embed_size)#, mode='bilinear')
                future_cat_rgb = F.interpolate(future_rgb.clone(), size=args.embed_size)#, mode='bilinear')

                n_o['e'] = torch.cat((n_o['e'], now_cat_rgb), dim=1)
                f_o['e'] = torch.cat((f_o['e'], future_cat_rgb), dim=1)

            if args.xy:
                coord_conv = F.interpolate(einops.rearrange(mesh_grids.clone(), 'b h w c -> b c h w'), size=args.embed_size)

                n_o['e'] = torch.cat((n_o['e'], coord_conv), dim=1)
                f_o['e'] = torch.cat((f_o['e'], coord_conv), dim=1)

            if args.norm:
                n_o['e'], f_o['e'] = F.normalize(n_o['e'], dim=1), F.normalize(f_o['e'], dim=1)
            
            if 'dd' in args.target:
                simseg = net.module if args.ddp else net

                now_group_batch, now_group_labels = build_group_inputs(now_pixels, future_pixels, n_o['e'], f_o['e'], args)
                now_group_out = simseg.compare_nn(now_group_batch)

                future_group_batch, future_group_labels = build_group_inputs(future_pixels, now_pixels, f_o['e'], n_o['e'], args)
                future_group_out = simseg.compare_nn(future_group_batch)

            if 'aa' in args.target:
                simseg = net.module if args.ddp else net

                now_assoc_batch, now_assoc_labels = build_assoc_inputs(now_people, now_pixels, n_o['e']) 
                now_assoc_out = simseg.compare_assoc(now_assoc_batch)

                future_assoc_batch, future_assoc_labels = build_assoc_inputs(future_people, future_pixels, f_o['e'])
                future_assoc_out = simseg.compare_assoc(future_assoc_batch)

            if 'df' in args.target or 'pf' in args.target:
                # =============== make d_flow loss ====================
                # offset by +1, -1 both x and y
                m_now_x, p_now_x = torch.roll(n_o['e'], -1, dims=3), torch.roll(n_o['e'], 1, dims=3)
                m_now_y, p_now_y = torch.roll(n_o['e'], -1, dims=2), torch.roll(n_o['e'], 1, dims=2)
                m_future_x, p_future_x = torch.roll(f_o['e'], -1, dims=3), torch.roll(f_o['e'], 1, dims=3)
                m_future_y, p_future_y = torch.roll(f_o['e'], -1, dims=2), torch.roll(f_o['e'], 1, dims=2)

                # calculate x and y similarities
                sim_mp_nx = torch.nan_to_num(F.cosine_similarity(m_now_x, p_now_x, dim=1, eps=1e-3), nan=sys.maxsize)
                sim_mp_ny = torch.nan_to_num(F.cosine_similarity(m_now_y, p_now_y, dim=1, eps=1e-3), nan=sys.maxsize)
                sim_mp_fx = torch.nan_to_num(F.cosine_similarity(m_future_x, p_future_x, dim=1, eps=1e-3), nan=sys.maxsize)
                sim_mp_fy = torch.nan_to_num(F.cosine_similarity(m_future_y, p_future_y, dim=1, eps=1e-3), nan=sys.maxsize)

                sim_mp_nx[now_future_cycle_inconsistent != 0.0] = 1.0
                sim_mp_ny[now_future_cycle_inconsistent != 0.0] = 1.0
                sim_mp_fx[future_now_cycle_inconsistent != 0.0] = 1.0
                sim_mp_fy[future_now_cycle_inconsistent != 0.0] = 1.0

            if 'df' in args.target:
                dfl += (abs(now_d_flow[:, 0]) * sim_mp_nx).clip(min=0.0)
                dfl += (abs(now_d_flow[:, 1]) * sim_mp_ny).clip(min=0.0)
                dfl += (abs(future_d_flow[:, 0]) * sim_mp_fx).clip(min=0.0)
                dfl += (abs(future_d_flow[:, 1]) * sim_mp_fy).clip(min=0.0)
                dfl = dfl.mean()

            if 'pf' in args.target:
                pfl += ((1.0 / (abs(now_d_flow[:, 0]) + 1e-3)) * (1 - sim_mp_nx).clip(min=0.0))
                pfl += ((1.0 / (abs(now_d_flow[:, 1]) + 1e-3)) * (1 - sim_mp_ny).clip(min=0.0))
                pfl += ((1.0 / (abs(future_d_flow[:, 0]) + 1e-3)) * (1 - sim_mp_fx).clip(min=0.0))
                pfl += ((1.0 / (abs(future_d_flow[:, 1]) + 1e-3)) * (1 - sim_mp_fy).clip(min=0.0))
                pfl = pfl.mean()

            if 'dd' in args.target:
                ddl += now_weight * torch.nan_to_num(dd_ce_loss(now_group_out, now_group_labels), nan=0.99)
                ddl += future_weight * torch.nan_to_num(dd_ce_loss(future_group_out, future_group_labels), nan=0.99)
                ddl = 10.0 * ddl

            if 'aa' in args.target:
                aal += now_weight * torch.nan_to_num(aa_ce_loss(now_assoc_out, now_assoc_labels), nan=0.99)
                aal += future_weight * torch.nan_to_num(aa_ce_loss(future_assoc_out, future_assoc_labels), nan=0.99)
                aal = 10.0 * aal

            if 'cc' in args.target:
                # connected component embeddings
                for b in range(now_labels.shape[0]):
                    with torch.set_grad_enabled(False):
                        # =============== calculate average mask embeddings ====================
                        now_ccs = [calculate_average_embedding(n_o['e'][b], now_labels[b], cc) for cc in now_labels[b].unique() if cc != 0]
                        future_ccs = [calculate_average_embedding(f_o['e'][b], future_labels[b], cc) for cc in future_labels[b].unique() if cc != 0]

                    # =============== find similarities within masks ====================
                    now_sim_ccs = [torch.nan_to_num(F.cosine_similarity(n_o['e'][b], exp_avg_embed, dim=0, eps=1e-3), nan=sys.maxsize) for (avg_embed, exp_avg_embed, mask) in now_ccs]
                    future_sim_ccs = [torch.nan_to_num(F.cosine_similarity(f_o['e'][b], exp_avg_embed, dim=0, eps=1e-3), nan=sys.maxsize) for (avg_embed, exp_avg_embed, mask) in future_ccs]

                    for (avg, expavg, mask), sim in zip(now_ccs, now_sim_ccs):
                        ccl += now_weight * sum(
                          [x.mean() for x in [torch.masked_select(-torch.log((torch.exp(sim) + 1e-3) / (sum([torch.exp(sim_o) for (avg_o, exp_o, mask_o), sim_o in zip(now_ccs, now_sim_ccs)]) + 1e-3)), mask.squeeze())]])
                    for (avg, expavg, mask), sim in zip(future_ccs, future_sim_ccs):
                        ccl += future_weight * sum(
                          [x.mean() for x in [torch.masked_select(-torch.log((torch.exp(sim) + 1e-3) / (sum([torch.exp(sim_o) for (avg_o, exp_o, mask_o), sim_o in zip(now_ccs, now_sim_ccs)]) + 1e-3)), mask.squeeze())]])

            if 'ss' in args.target:
                # warp the embeddings
                n_w_e = F.grid_sample(n_o['e'], future_now_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)
                f_w_e = F.grid_sample(f_o['e'], now_future_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)

                # =============== make simsiam loss from warped embeddings ====================
                now_to_future_simsiam = torch.nan_to_num(F.cosine_similarity(n_w_e, f_o['pred'].clone(), dim=1, eps=1e-3), nan=1.0)
                future_to_now_simsiam = torch.nan_to_num(F.cosine_similarity(f_w_e, n_o['pred'].clone(), dim=1, eps=1e-3), nan=1.0)

                now_to_future_simsiam[future_now_cycle_inconsistent != 0.0] = 1.0
                future_to_now_simsiam[now_future_cycle_inconsistent != 0.0] = 1.0

                ssl += 10.0 * (now_weight * (1.0 - now_to_future_simsiam).mean() + future_weight * (1.0 - future_to_now_simsiam).mean())

            if not is_train and visualize:
                now_segments = segment_embeddings(n_o['e'][:, :min(args.embed_dim, n_o['e'].shape[1])], clust).float()
                future_segments = segment_embeddings(f_o['e'][:, :min(args.embed_dim, f_o['e'].shape[1])], clust).float()

                # record which images were used to annotate them later
                #with open(f'annotation/{args.dataset}_to_annotate_{args.global_rank}.txt', 'a') as f:
                #    for frame in batch['now']['frame']:
                #        f.write(frame + '\n')
                #    for frame in batch['future']['frame']:
                #        f.write(frame + '\n')

                now_association, now_cluster_assoc, now_obj_pred, now_box2seg = make_association_portable(batch['now']['frame'], n_o['e'], now_segments, net, args)
                future_association, future_cluster_assoc, future_obj_pred, future_box2seg = make_association_portable(batch['future']['frame'], f_o['e'], future_segments, net, args)

                now_association_1, now_cluster_assoc_1, now_obj_pred_1, now_box2seg_1 = make_association_portable(batch['now']['frame'], n_o['e'], now_segments, net, args, reverse=True)
                future_association_1, future_cluster_assoc_1, future_obj_pred_1, future_box2seg_1 = make_association_portable(batch['future']['frame'], f_o['e'], future_segments, net, args, reverse=True)

            loss = torch.tensor(0.0).cuda()
            if 'dd' in args.target: loss += ddl
            if 'aa' in args.target: loss += aal
            if 'cc' in args.target: loss += ccl
            if 'ss' in args.target: loss += ssl
            if 'df' in args.target: loss += dfl
            if 'pf' in args.target: loss += pfl

            if loss != loss:
                print('NaNed!')
                sys.exit()
                #pdb.set_trace()

            local_step = ((i + 1) * args.batch_size)

        if is_train:
            scaler.scale(loss).backward()

            if False and 'smoothformer' in args.model:
                # Weight update
                step = i
                if ((step + 1) % cfg.TRAIN.ACCUM_STEPS == 0) or (step + 1 == len(loader)):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # update average loss
                    losses += float(loss.detach())

                    # update learning schedule
                    scheduler.before_train_iter()
                    scheduler.get_lr(int(epoch_len+step), cfg.TRAIN.BASE_LR)
            else:
                scaler.step(optimizer)
                scaler.update()
            #scheduler.step()
            #loss.backward()

        # ================ logging ===================
        #if True or 'smoothformer' not in args.model or not is_train:
        losses += float(loss.detach())

        status = '{} epoch {}: itr {:<6}/ {}- {}- '.format('TRAIN' if is_train else 'VAL  ', epoch, local_step, epoch_len, args.name)
        if 'aa' in args.target: status += f'aal {aal:.4f}- '
        if 'dd' in args.target: status += f'ddl {ddl:.4f}- '
        if 'cc' in args.target: status += f'ccl {ccl:.4f}- '
        if 'ss' in args.target: status += f'ssl {ssl:.4f}- '
        if 'df' in args.target: status += f'dfl {dfl:.4f}- '
        if 'pf' in args.target: status += f'pfl {pfl:.4f}- '
        status += 'avg l {:.4f}- lr {}- dt {:.4f}'.format(
          losses / (i + 1), # print batch loss and avg loss
          str(optimizer.param_groups[0]['lr'])[:7] if optimizer is not None else args.lr,
          time.time() - start) # batch time
        pbar.set_description(status)

        if not is_train and i < args.queue_len: 
            iii = i * args.num_gpus + args.global_rank

            # final dim is spatial, so embed_dim (144x256) 
            b = now_rgb.shape[0]

            try:
                store_image([n_o['e'], f_o['e']], ['now_xy-rgb-with-feat', 'future_xy-rgb-with-feat'], 'pca', iii, b, args)
                store_image([n_o['e'][:, :args.embed_dim], f_o['e'][:, :args.embed_dim]], ['now_feat', 'future_feat'], 'pca', iii, b, args)
                if 'ss' in args.target:
                    store_image([n_w_e, f_w_e, n_o['pred'], f_o['pred']], ['now_warp-proj', 'future_warp-proj', 'now_pred', 'future_pred'], 'pca', iii, b, args)
            except Exception as e:
                print(e)

            if visualize:
                store_image(now_segments, 'now_clusters', 'nipy_spectral', iii, b, args)
                store_image(future_segments, 'future_clusters', 'nipy_spectral', iii, b, args)

                store_image(now_association, 'now_association-0', 'save', iii, b, args)
                store_image(now_association_1, 'now_association-1', 'save', iii, b, args)
                store_image(future_association, 'future_association-0', 'save', iii, b, args)
                store_image(future_association_1, 'future_association-1', 'save', iii, b, args)

                store_image(now_cluster_assoc, 'now_association-0-by-cluster', 'save', iii, b, args)
                store_image(now_cluster_assoc_1, 'now_association-1-by-cluster', 'save', iii, b, args)
                store_image(future_cluster_assoc, 'future_association-0-by-cluster', 'save', iii, b, args)
                store_image(future_cluster_assoc_1, 'future_association-1-by-cluster', 'save', iii, b, args)

                store_image(now_box2seg, 'now_fig-box2seg-0-mask', 'save', iii, b, args)
                store_image(future_box2seg, 'future_fig-box2seg-0-mask', 'save', iii, b, args)

                store_image([now_box2seg, now_rgb], 'now_fig-box2seg-0', 'overlay', iii, b, args)
                store_image([future_box2seg, future_rgb], 'future_fig-box2seg-0', 'overlay', iii, b, args)
                #store_image([now_box2seg_1, now_rgb], 'now_fig-box2seg-1', 'overlay', iii, b, args)
                #store_image([future_box2seg_1, future_rgb], 'future_fig-box2seg-1', 'overlay', iii, b, args)

                store_image(now_obj_pred, 'now_fig-objpred-0-mask', 'save', iii, b, args)
                store_image(now_obj_pred_1, 'now_fig-objpred-1-mask', 'save', iii, b, args)
                store_image(future_obj_pred, 'future_fig-objpred-0-mask', 'save', iii, b, args)
                store_image(future_obj_pred_1, 'future_fig-objpred-1-mask', 'save', iii, b, args)

                store_image([now_obj_pred, now_rgb], 'now_fig-objpred-0', 'overlay', iii, b, args)
                store_image([now_obj_pred_1, now_rgb], 'now_fig-objpred-1', 'overlay', iii, b, args)
                store_image([future_obj_pred, future_rgb], 'future_fig-objpred-0', 'overlay', iii, b, args)
                store_image([future_obj_pred_1, future_rgb], 'future_fig-objpred-1', 'overlay', iii, b, args)

            store_image(now_rgb, 'now_frame', 'rgb', iii, b, args)
            store_image(future_rgb, 'future_frame', 'rgb', iii, b, args)

            store_image(now_labels.squeeze(), 'now_pseudolabels', 'nipy_spectral', iii, b, args)
            store_image(future_labels.squeeze(), 'future_pseudolabels', 'nipy_spectral', iii, b, args)

            store_image(now_split_labels.squeeze(), 'now_pseudolabels-assoc', 'nipy_spectral', iii, b, args)
            store_image(future_split_labels.squeeze(), 'future_pseudolabels-assoc', 'nipy_spectral', iii, b, args)

            store_image(now_sed, 'now_sampson-error', 'save', iii, b, args)
            store_image(future_sed, 'future_sampson-error', 'save', iii, b, args)

            store_image(now_future_cycle_inconsistent.unsqueeze(1), 'now_pixels-inconsistent', 'save', iii, b, args)
            store_image(future_now_cycle_inconsistent.unsqueeze(1), 'future_pixels-inconsistent', 'save', iii, b, args)

            store_image([F.interpolate(now_people.unsqueeze(1), size=args.img_size).squeeze(), now_rgb], 'now_people', 'overlay', iii, b, args)
            store_image([F.interpolate(future_people.unsqueeze(1), size=args.img_size).squeeze(), future_rgb], 'future_people', 'overlay', iii, b, args)

            store_image(now_future_flow, 'now_xy-flow-all', 'flow', iii, b, args)
            store_image(future_now_flow, 'future_xy-flow-all', 'flow', iii, b, args)

            if 'mm' in args.target:
                store_image(now_outl, 'now_outliers', 'save', iii, b, args)
                store_image(future_outl, 'future_outliers', 'save', iii, b, args)

            if 'df' in args.target or 'pf' in args.target:
                store_image(now_d_flow_norm, 'now_flow-d', 'save', iii, b, args)
                store_image(future_d_flow_norm, 'future_flow-d', 'save', iii, b, args)

                store_image(now_future_flow_norm, 'now_mag-flow', 'save', iii, b, args)
                store_image(future_now_flow_norm, 'future_mag-flow', 'save', iii, b, args)

            if 'df' in args.target:
                store_image((abs(now_d_flow[:, 0]) * sim_mp_nx).clip(min=0.0), 'now_flow-dfl-x', 'save', iii, b, args)
                store_image((abs(now_d_flow[:, 1]) * sim_mp_ny).clip(min=0.0), 'now_flow-dfl-y', 'save', iii, b, args)
                store_image((abs(future_d_flow[:, 0]) * sim_mp_fx).clip(min=0.0), 'future_flow-dfl-x', 'save', iii, b, args)
                store_image((abs(future_d_flow[:, 1]) * sim_mp_fy).clip(min=0.0), 'future_flow-dfl-y', 'save', iii, b, args)

            if 'pf' in args.target:
                store_image((1.0 / (abs(now_d_flow[:, 0]) + 1e-3)) * (1 - sim_mp_nx).clip(min=0.0), 'now_flow-pfl-x', 'save', iii, b, args)
                store_image((1.0 / (abs(now_d_flow[:, 1]) + 1e-3)) * (1 - sim_mp_ny).clip(min=0.0), 'now_flow-pfl-y', 'save', iii, b, args)
                store_image((1.0 / (abs(future_d_flow[:, 0]) + 1e-3)) * (1 - sim_mp_fx).clip(min=0.0), 'future_flow-pfl-x', 'save', iii, b, args)
                store_image((1.0 / (abs(future_d_flow[:, 1]) + 1e-3)) * (1 - sim_mp_fy).clip(min=0.0), 'future_flow-pfl-y', 'save', iii, b, args)

            if 'ss' in args.target:
                store_image(1.0 - now_to_future_simsiam, 'now_simsiam', 'save', iii, b, args)
                store_image(1.0 - future_to_now_simsiam, 'future_simsiam', 'save', iii, b, args)

            if 'df' in args.target or 'pf' in args.target:
                store_image(now_future_flow[:, :, :, 0], 'now_x-flow', 'bwr', iii, b, args)
                store_image(now_future_flow[:, :, :, 1], 'now_y-flow', 'bwr', iii, b, args)

                store_image(future_now_flow[:, :, :, 0], 'future_x-flow', 'bwr', iii, b, args)
                store_image(future_now_flow[:, :, :, 1], 'future_y-flow', 'bwr', iii, b, args)

                store_image(abs(now_d_flow[:, 0]).float(), 'now_x-dflow', 'save', iii, b, args)
                store_image(abs(now_d_flow[:, 1]).float(), 'now_y-dflow', 'save', iii, b, args)

                store_image(abs(future_d_flow[:, 0]).float(), 'future_x-dflow', 'save', iii, b, args)
                store_image(abs(future_d_flow[:, 1]).float(), 'future_y-dflow', 'save', iii, b, args)

        # =============== termination ================
        if local_step > epoch_len and epoch_len > 0: break
    avg_loss = losses / (i + 1e-3) # (i * loader.batch_size)

    if not is_train and args.global_rank == 0:
        write_index_html(args)

    return avg_loss


if __name__ == "__main__":
    pass
