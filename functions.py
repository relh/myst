#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import pdb
import random
import sys

import cupy as cp
import cv2
import einops
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy.ndimage
import torch
import torch.nn.functional as F
import torch.utils.data
from cupyx.profiler import benchmark
from cupyx.scipy.ndimage import label
from kornia.filters import spatial_gradient
from kornia.geometry.epipolar import sampson_epipolar_distance
from PIL import Image, ImageEnhance
from shapely.geometry import Polygon
from skimage.color import *  # lab2rgb
from torchvision import transforms
from torchvision.ops import masks_to_boxes
from torchvision.utils import save_image

inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
normalize = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
  ),
])

values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
)

def make_association_automatic(handlocs, output, segments, net, args, reverse=False):
    if len(handlocs) > 0 and reverse == False:
        handloc, hand_bbox = handlocs[0]
    elif len(handlocs) > 1 and reverse == True:
        handloc, hand_bbox = handlocs[-1] 
    else:
        probits = torch.zeros(args.img_size)
        cluster_probits = torch.zeros(args.img_size)
        obj_preds = torch.zeros(args.img_size)
        return torch.stack([probits]).cuda(), torch.stack([cluster_probits]).cuda(), torch.stack([obj_preds]).cuda()

    #hand_pred
    handloc = [int(x * args.embed_size[0] / (args.anno_size[0] * 1.0)) for x in handloc]
    handloc[0] = min(handloc[0], args.embed_size[1]-1)
    handloc[1] = min(handloc[0], args.embed_size[0]-1)
    hand_bbox = [int(x * args.embed_size[0] / (args.anno_size[0] * 1.0)) for x in hand_bbox]
    hand_bbox[0] = min(hand_bbox[0], args.embed_size[1]-1)
    hand_bbox[1] = min(hand_bbox[1], args.embed_size[0]-1)
    hand_bbox[2] = min(hand_bbox[2], args.embed_size[1]-1)
    hand_bbox[3] = min(hand_bbox[3], args.embed_size[0]-1)

    obj_pred, cluster_probits, probits = make_obj_pred(handloc, output, segments, net, args, obj_bbox=None)
    obj_pred[handloc[1], handloc[0]] = 2.0

    hand_pred = make_hand_pred(handloc, output, segments, args, hand_bbox, size=None)
    pair_pred = obj_pred + hand_pred

    probits = F.interpolate(einops.repeat(probits, 'h w -> b c h w', b=1, c=1), size=args.img_size).squeeze()
    cluter_probits = F.interpolate(einops.repeat(cluster_probits, 'h w -> b c h w', b=1, c=1), size=args.img_size).squeeze()
    pair_preds = F.interpolate(einops.repeat(pair_pred.float(), 'h w -> b c h w', b=1, c=1), size=args.img_size).squeeze()
    return torch.stack([probits]), torch.stack([cluster_probits]), torch.stack([pair_preds])


def make_obj_pred(handloc, output, segments, net, args, obj_bbox=None, size=None):
    if 'grabcut' in args.model:
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        # apply GrabCut using the the bounding box segmentation method
        inp_image = (inv_normalize(output[0]).flip(dims=(0,)).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
        mask = np.zeros(output[0,1].shape, np.uint8)

        #rect = [obj_bbox[1], obj_bbox[0], obj_bbox[3], obj_bbox[2]]
        rect = obj_bbox
        rect[2] = rect[2] - rect[0]
        rect[3] = rect[3] - rect[1]
        rect = np.array([int(x * (args.img_size[0] / size[0])) for x in rect])
        (obj_pred, bgModel, fgModel) = cv2.grabCut(inp_image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
        obj_pred = np.where((obj_pred == cv2.GC_BGD) | (obj_pred == cv2.GC_PR_BGD), 0, 1)
        obj_pred = F.interpolate(einops.repeat(torch.tensor(obj_pred).float(), 'h w -> b c h w', b=1, c=1), size=size).squeeze().bool().cuda()
        cluster_probits = torch.zeros(segments.shape).float().cuda()
        reshaped_probits = torch.zeros(segments.shape).float().cuda()
    elif handloc is None:
        try:
            obj_pred = box_2_seg(output, segments, obj_bbox, args)
        except:
            obj_pred = torch.zeros(segments.shape).bool().cuda()
        cluster_probits = torch.zeros(segments.shape).float().cuda()
        reshaped_probits = torch.zeros(segments.shape).float().cuda()
    else:
        handloc = [int(x * args.embed_size[0] / (size[0] * 1.0)) if size is not None else int(x) for x in handloc]
        handloc[0] = min(handloc[0], args.embed_size[1]-1)
        handloc[1] = min(handloc[0], args.embed_size[0]-1)
        # compare handlocation to all other embeddings
        flat_segments = einops.repeat(segments, 'h w -> (h w)')
        flat_all_embeds = einops.repeat(output, 'b c h w -> (h w) (c b)')
        tile_point = einops.repeat(output[:, :, handloc[1], handloc[0]], 'b c -> z (b c)', z=flat_all_embeds.shape[0])
        nm = net.module if args.ddp else net
        #logits = nm.compare_assoc(torch.cat((tile_point, flat_all_embeds), dim=1)) # whether query point is in position 1 or position 2
        logits = nm.compare_assoc(torch.cat((flat_all_embeds, tile_point), dim=1))

        # normalize logits and reshape them
        probits = F.softmax(logits)[:, 1]
        reshaped_probits = probits.reshape(output.shape[2:])

        # mean affinities across clusters
        cluster_probits = torch.zeros(segments.shape).cuda()
        for cc in segments.unique():
            if cc == -1: continue
            cc_mask = torch.where((segments == cc), torch.ones(segments.shape).cuda(), torch.zeros(segments.shape).cuda()).bool()
            cluster_probits[segments == cc] = reshaped_probits[cc_mask].mean()

        # object is probits above threshold
        obj_pred = cluster_probits > args.object_threshold
    return obj_pred, cluster_probits**2.0, reshaped_probits**2.0


def make_hand_pred(handloc, output, segments, args, hand_bbox, size=None):
    handloc = [int(x * args.embed_size[0] / (size[0] * 1.0)) if size is not None else int(x) for x in handloc]
    hand_cluster = segments[handloc[1], handloc[0]]
    if 'grabcut' in args.model:
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")
        # apply GrabCut using the the bounding box segmentation method
        inp_image = (inv_normalize(output[0]).flip(dims=(0,)).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
        mask = np.zeros(output[0,1].shape, np.uint8)

        rect = hand_bbox
        rect[2] = rect[2] - rect[0]
        rect[3] = rect[3] - rect[1]
        rect = np.array([int(x * (args.img_size[0] / size[0])) for x in rect])
        (hand_pred, bgModel, fgModel) = cv2.grabCut(inp_image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
        hand_pred = np.where((hand_pred == cv2.GC_BGD) | (hand_pred == cv2.GC_PR_BGD), 0, 1)
        hand_pred = F.interpolate(einops.repeat(torch.tensor(hand_pred).float(), 'h w -> b c h w', b=1, c=1), size=size).squeeze().bool().cuda()
    elif hand_cluster == -1:
        #good_clusters = box_2_good_clusters(segments, hand_bbox)
        hand_pred = box_2_seg(output, segments, hand_bbox, args)
    else:
        hand_pred = torch.where((segments == hand_cluster), torch.ones(segments.shape).cuda(), torch.zeros(segments.shape).cuda())

    # prune hand pred
    #hand_pred[:, :hand_bbox[0]] = False
    #hand_pred[:hand_bbox[1], :] = False
    #hand_pred[:, hand_bbox[0]+hand_bbox[2]:] = False
    #hand_pred[hand_bbox[1]+hand_bbox[3]:, :] = False

    #hand_pred[:, hand_bbox[2]:] = False
    #hand_pred[hand_bbox[3]:, :] = False
    return hand_pred


def load_test_annotations(person, video, now_frame, annotations, coco_annotations, hos_annotations, args):
    hos_details = None
    handlocs = []
    entities = {}
    if args.dataset == 'epickitchens':
        try:
            entities = annotations[now_frame]
            hos_key = person + '_' + video + '/' + now_frame
            hos_details = hos_annotations[hos_key] if hos_key in hos_annotations else None
        except Exception as e:
            print(str(e))

    this_frame = [x for x in coco_annotations['images'] if x['file_name'] == now_frame]
    if len(this_frame) > 0:
        these_anno = [x for x in coco_annotations['annotations'] if x['image_id'] == this_frame[0]['id']]

        for this_anno in these_anno:
            # only modify boxes for hands
            if args.dataset == 'epickitchens': 
                this_offset = this_anno['offset']
                quarter_bbox = [x for x in this_anno['bbox']]
                if this_anno['isincontact'] == 1:
                    quarter_bbox[2] /= 2
                    quarter_bbox[3] /= 2
                    if this_offset[0] > 0: quarter_bbox[0] = quarter_bbox[0] + quarter_bbox[2]
                    if this_offset[1] > 0: quarter_bbox[1] = quarter_bbox[1] + quarter_bbox[3]
                quarter_bbox[2] += quarter_bbox[0]
                quarter_bbox[3] += quarter_bbox[1]

            if args.dataset == 'ego4d':
                quarter_bbox = [int(x * args.anno_size[0] / args.img_size[0]) for x in this_anno['bbox']]
                tabb = [int(x * args.anno_size[0] / args.img_size[0]) for x in this_anno['bbox']]
            else:
                tabb = this_anno['bbox']
                tabb[2] += tabb[0]
                tabb[3] += tabb[1]

            # get segmentation within quarter box
            for quarter_segmentation in this_anno['segmentation']:
                # only want to make incontact handlocs
                if args.dataset == 'epickitchens' and this_anno['isincontact'] != 1: continue
                if args.dataset == 'ego4d' and this_anno['category_id'] != 1: continue

                if args.dataset == 'ego4d':
                    quarter_pts = [(min(int(x * args.anno_size[0] / args.img_size[0]), args.anno_size[0]-1), min(int(y * args.anno_size[0] / args.img_size[0]), args.anno_size[1]-1)) for (x, y) in zip(quarter_segmentation[0::2], quarter_segmentation[1::2]) if x > quarter_bbox[0] and y > quarter_bbox[1] and x < (quarter_bbox[0]+quarter_bbox[2]) and y < (quarter_bbox[1]+quarter_bbox[3])]
                else:
                    quarter_pts = [(x, y) for (x, y) in zip(quarter_segmentation[0::2], quarter_segmentation[1::2]) if x > quarter_bbox[0] and y > quarter_bbox[1] and x < (quarter_bbox[0]+quarter_bbox[2]) and y < (quarter_bbox[1]+quarter_bbox[3])]

                try:
                    handlocs.append((get_representative_point(np.array(quarter_pts)), tabb))
                except:
                    if args.dataset == 'ego4d':
                        all_pts = [(min(int(x * args.anno_size[0] / args.img_size[0]), args.anno_size[0]-1), min(int(y * args.anno_size[0] / args.img_size[0]), args.anno_size[1]-1)) for (x, y) in zip(quarter_segmentation[0::2], quarter_segmentation[1::2])]
                    else:
                        all_pts = [(x, y) for (x, y) in zip(quarter_segmentation[0::2], quarter_segmentation[1::2])]
                    handlocs.append((get_representative_point(np.array(all_pts)), tabb))

            if args.dataset == 'ego4d':
                polygon_list = []
                for poly in this_anno['segmentation']:
                    points = [(int(x*args.anno_size[0]/args.img_size[0]),int(y*args.anno_size[0]/args.img_size[0])) for (x,y) in zip(poly[0::2], poly[1::2])]
                    polygon_list.append(np.array(points).astype(np.int32))

                if this_anno['category_id'] == 1:
                    entities[('right_hand' if 'left_hand' in entities else 'left_hand')] = polygon_list 
                else:
                    entities[('right_object' if 'left_object' in entities else 'left_object')] = polygon_list 

    return entities, hos_details, handlocs


def _object_id_hash(objects, val=256, dtype=torch.long):
    C = objects.shape[0]
    objects = objects.to(dtype)
    out = torch.zeros_like(objects[0:1, ...])
    for c in range(C):
        scale = val ** (C - 1 - c)
        out += scale * objects[c:c + 1, ...]
    return out


def process_segmentation_color(meta, seg_color, meta_key):
    # convert segmentation color to integer segment id
    raw_segment_map = _object_id_hash(seg_color, val=256, dtype=torch.long)
    raw_segment_map = raw_segment_map.squeeze(0)

    # remove zone id from the raw_segment_map
    #meta_key = 'playroom_large_v3_images/' + file_name.split('/images/')[-1] + '.hdf5'
    zone_id = int(meta[meta_key]['zone'])
    raw_segment_map[raw_segment_map == zone_id] = 0

    # convert raw segment ids to a range in [0, n]
    _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
    segment_map -= segment_map.min()

    return segment_map


def input_multiplexer(image_path, args):
    with Image.open(image_path) as keyframe_im:
        if args.sharpness != 1.0:
            inp_image_enh = keyframe_im.resize(args.img_size[::-1], resample=PIL.Image.BILINEAR)
            enhancer = ImageEnhance.Sharpness(inp_image_enh)
            inp_image_enh = enhancer.enhance(args.sharpness)
            keyframe = normalize(np.array(inp_image_enh)).unsqueeze(0).cuda()
            keyframe = F.interpolate(keyframe, size=args.img_size, mode='bilinear')
        else:
            keyframe = normalize(np.array(keyframe_im)).unsqueeze(0).cuda()
    return keyframe


def output_multiplexer(keyframe, net, mesh_grids, args):
    if args.model == 'convnext' or \
       args.model == 'unet' or \
       'former' in args.model or \
            'hrnet' in args.model:
        o = net(keyframe)

        if args.rgb:
            cat_rgb = F.interpolate(keyframe.clone(), size=args.embed_size)
            o['e'] = torch.cat((o['e'], cat_rgb), dim=1)

        if args.xy:
            coord_conv = F.interpolate(einops.rearrange(mesh_grids.clone(), 'b h w c -> b c h w'), size=args.embed_size)
            o['e'] = torch.cat((o['e'], coord_conv), dim=1)

        output = o['e']
    elif args.model == 'rgb':
        output = F.interpolate(keyframe, size=args.embed_size)
    elif args.model == 'coco':
        output = net(keyframe)
        output = F.interpolate(output['out'], size=args.embed_size)
    elif args.model == 'resnet':
        feat_pyramid, gfeat = net(keyframe)
        output0 = F.interpolate(feat_pyramid[0], size=args.embed_size)
        output1 = F.interpolate(feat_pyramid[1], size=args.embed_size)
        output = torch.cat([output0, output1], dim=1)
    elif args.model == 'cohesiv':
        output = net.run_image(inv_normalize(keyframe))
        output = F.interpolate(output['e'], size=args.embed_size)
    elif args.model == 'dino':
        output = net.extract_features(keyframe.cpu(), transform=False, upsample=False)
        #output = F.interpolate(output[:, :args.embed_dim], size=args.embed_size)
        output = F.interpolate(output, size=args.embed_size)
    elif args.model == 'spatialgrid':
        output = einops.rearrange(mesh_grids, 'b h w c -> b c h w')
        output = F.interpolate(torch.cat([keyframe, output], dim=1), size=args.embed_size)
    elif net is None:
        output = F.interpolate(keyframe, size=args.embed_size)
    else:
        output = net(keyframe)
        output = F.interpolate(output, size=args.embed_size)

    if args.norm: output = F.normalize(output, dim=1)
    output = torch.nan_to_num(output)
    return output


def get_representative_point(cnt):
    poly = Polygon(cnt.squeeze())
    cx = poly.representative_point().x
    cy = poly.representative_point().y
    return cx, cy


def load_bbox(segments, this_frame, args, bbox_type='hands'):
    bbox_path = None
    if args.bbox:
        if args.slurm_partition is not None:
            if args.dataset == 'epickitchens':  bbox_path = f'/home/relh/100DOH/experiments/bbox/epick/'#hands/'
            elif args.dataset == 'doh':  bbox_path = f'/home/relh/100DOH/experiments/bbox/doh/'#hands/'
            elif args.dataset == 'ho3d': bbox_path = f'/home/relh/100DOH/experiments/bbox/ho3d/'#hands/'
        else:
            if args.dataset == 'epickitchens':  bbox_path = f'/w/relh/bbox/epick/'#hands/'
            elif args.dataset == 'doh':  bbox_path = f'/w/relh/bbox/doh/'#hands/'
            elif args.dataset == 'ho3d': bbox_path = f'/w/relh/bbox/ho3d/'#hands/'

    if bbox_path is not None:
        with Image.open(bbox_path + f'{bbox_type}/' + this_frame[:-4] + '.png') as im_bbox_pred:
            bbox_pred = (torch.tensor(np.array(im_bbox_pred)).int() / 255).bool().cuda()
    else:
        bbox_pred = torch.ones(segments.shape)

    try:
        box = masks_to_boxes(einops.repeat(bbox_pred, 'h w -> b h w', b=1)).int()[0]
    except:
        box = [0, 0, bbox_pred.shape[0]-1, bbox_pred.shape[1]-1]

    return box#, bbox_path


def box_2_seg(output, segments, box, args):
    # box always pair of coordinates
    if box is not None and len(box) != 4:
        try:
            box = masks_to_boxes(einops.repeat(box, 'h w -> b h w', b=1)).int()[0]
        except Exception as e:
            print(f'badbox -- {box}')
            box = [0, 0, segments.shape[0], segments.shape[1]]
    else:
        box = [0, 0, segments.shape[0], segments.shape[1]]

    good_clusters = box_2_good_clusters(segments, box)
    if len(good_clusters) == 1 and good_clusters[0] == -1:
        box_segments = segments[box[1]:box[3], box[0]:box[2]]
        good_clusters = [int(x) for x in box_segments.unique() if x != -1]
        if len(good_clusters) == 0: good_clusters = [-1]

    if args.congealing_method == 'feature':
        top3 = calculate_top3(output, segments)
        binary_cc = feature_congeal_outliers(segments, box, top3)
        this_mask = sum(segments==i for i in good_clusters).bool()
        this_mask += binary_cc
    else:
        try:
            segments = spatial_congeal_outliers(segments)
        except Exception as e:
            print('no outliers to congeal!')
            print(str(e))
        if 'grabcut' in args.model:
            try:
                fgModel = np.zeros((1, 65), dtype="float")
                bgModel = np.zeros((1, 65), dtype="float")
                # apply GrabCut using the the bounding box segmentation method
                output = F.interpolate(output, size=segments.shape)
                inp_image = (inv_normalize(output[0]).flip(dims=(0,)).permute(1,2,0).cpu().numpy() * 255.0).astype(np.uint8)
                mask = np.zeros(output[0,1].shape, np.uint8)

                rect = [box[0], box[1], box[2] - box[0], box[3] - box[1]] 
                rect = np.array([int(x) for x in rect])
                #rect = np.array([int(x * (args.img_size[0] / segments.shape[0])) for x in rect])
                (obj_pred, bgModel, fgModel) = cv2.grabCut(inp_image, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
                obj_pred = np.where((obj_pred == cv2.GC_BGD) | (obj_pred == cv2.GC_PR_BGD), 0, 1)
                this_mask = F.interpolate(einops.repeat(torch.tensor(obj_pred).float(), 'h w -> b c h w', b=1, c=1), size=segments.shape).squeeze().bool().cuda()
            except:
                print('broken grabcut!')
                this_mask = torch.zeros(segments.shape).squeeze().bool().cuda()
        else:
            this_mask = sum(segments==i for i in good_clusters).bool()

    this_mask[:, :box[0]] = False
    this_mask[:box[1], :] = False 
    this_mask[:, box[2]:] = False 
    this_mask[box[3]:, :] = False

    if this_mask.sum() == 0: # or this_mask.sum == (args.embed_size[0] * args.embed_size[1]):
        this_mask[box[1]:box[3], box[0]:box[2]] = True

    #best_box_segments, best_box_cluster, boxed_mask = clusters_2_seg(segments, box, good_clusters)
    return this_mask.squeeze() #best_box_segments.squeeze()


def box_2_good_clusters(segments, box):
    # find clusters in box
    box_segments = segments[box[1]:box[3], box[0]:box[2]]
    box_seg_unique = [int(x) for x in box_segments.unique()]

    # see if any clusters cross border
    not_box_segments = segments.clone()
    not_box_segments[box[1]:box[3], box[0]:box[2]] = -2

    good_clusters = []
    max_cluster, max_sum = -1, 0.0

    for cc in box_seg_unique:
        #if bsu == -1 or bsu == 0: continue
        if cc == -1: continue
        cc_sum = (box_segments == cc).sum()
        n_cc_sum = (not_box_segments == cc).sum()

        # if percentage of cluster outside of box < 10%, add to good_clusters
        if n_cc_sum < (cc_sum * 0.10): # 10
            good_clusters.append(int(cc))
        if cc_sum > max_sum:
            max_sum = cc_sum 
            max_cluster = int(cc)
    if len(good_clusters) == 0: 
        good_clusters = [max_cluster]
    #if len(good_clusters) == 0: 
    #    good_clusters = [int(x) for x in box_segments.unique() if x != -1]
    return good_clusters


def draw_box_on_mask(this_mask, box):
    # box always pair of coordinates
    if len(box) != 4:
        try:
            box = masks_to_boxes(einops.repeat(box, 'h w -> b h w', b=1)).int()[0]
        except Exception as e:
            box = [0, 0, this_mask.shape[1]-1, this_mask.shape[0]-1]

    boxed_mask = this_mask.clone()
    boxed_mask[box[1]:box[3], box[0]] = True
    boxed_mask[box[1], box[0]:box[2]] = True
    boxed_mask[box[1]:box[3], box[2]] = True
    boxed_mask[box[3], box[0]:box[2]] = True

    return boxed_mask


def box_iou(box, this_box):
    x_left = max(box[0], this_box[0])
    y_top = max(box[1], this_box[1])
    x_right = min(box[2], this_box[2])
    y_bottom = min(box[3], this_box[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (box[2] - box[0]) * (box[3] - box[1])
    bb2_area = (this_box[2] - this_box[0]) * (this_box[3] - this_box[3])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def spatial_congeal_outliers(segments):
    if len(segments.shape) == 2:
        segments = segments.unsqueeze(0)
    outliers = (segments == -1).permute(1,2,0).cpu()
    distanceIndMulti = scipy.ndimage.distance_transform_edt(outliers, return_distances=False, return_indices=True)
    distanceInd = np.ravel_multi_index(distanceIndMulti, outliers.shape)
    distanceInd = distanceInd.transpose(2,0,1)
    return segments.ravel()[distanceInd].squeeze().cuda()


def feature_congeal_outliers(segments, box, top3_cc):
    # check clust = -1
    outliers = (segments == -1)

    # if top 2 are in good_clusters, add to top 
    top3_cc[:, outliers == False] = -2
    top3_cc[:, :, :box[0]] = -2
    top3_cc[:, :box[1], :] = -2
    top3_cc[:, :, box[2]:] = -2
    top3_cc[:, box[3]:, :] = -2

    sum_cc = sum(top3_cc==i for i in good_clusters).sum(dim=0)
    binary_cc = sum_cc >= min(2, len(good_clusters))
    return binary_cc



def get_pixel_groups(this_labels, batch_size=4096):
    all_pixels = []
    for b in range(this_labels.shape[0]):
        label_unique = this_labels[b].unique()
        this_max = batch_size // len(label_unique)
        spatial_label_unique = dict([(int(cc), (this_labels[b, 0] == cc).nonzero()) for cc in label_unique])

        # need to sample batch_size from these, sort so can always get enough from last
        spatial_label_unique = sorted(spatial_label_unique.items(), key=lambda i: i[1].shape[0])
        pixels = {}
        pixels_remaining = batch_size
        for zzz, (cc, embed_group) in enumerate(spatial_label_unique):
            # how many points to choose from this group
            indices_to_sample = min(pixels_remaining, this_max, embed_group.shape[0])
            pixels_remaining -= indices_to_sample

            # points to choose from this group
            indices = random.sample(range(0, embed_group.shape[0]), indices_to_sample)
            pixels[cc] = embed_group[indices, :]

        if pixels_remaining > 0:
            indices = random.choices(range(0, embed_group.shape[0]), k=pixels_remaining)
            pixels[cc] = torch.cat((pixels[cc], embed_group[indices, :]), dim=0)
        all_pixels.append(pixels)
    return all_pixels


def build_group_inputs(this_pixels, that_pixels, this_embedding, that_embedding, args):
    # David's pairwise loss
    mlp_inputs = []
    mlp_labels = []

    for b in range(len(this_pixels)):
        # build batch
        mlp_input = []
        mlp_label = []

        for chosen_cc, chosen_pixels in this_pixels[b].items():
            # negative points from background pixels
            pixels, labels = [], []

            # find matching connected component to understand how many positive we can take (max 25%)
            that_amount = 0
            if chosen_cc in that_pixels[b].keys(): 
                pos_that_pixels = that_pixels[b][chosen_cc]
                that_amount = min(chosen_pixels.shape[0] // 4, pos_that_pixels.shape[0])
                indices = random.sample(range(0, pos_that_pixels.shape[0]), that_amount)
                pixels.append(that_embedding[b, :, pos_that_pixels[indices, 0], pos_that_pixels[indices, 1]])
                labels.append((torch.zeros(that_amount) if args.negbg else -torch.ones(that_amount)) if chosen_cc == 0 else torch.ones(that_amount))

            # how many total negative pixels do we need to supplement this
            neg_that_amount, neg_this_amount = 0, 0
            if 0 in that_pixels[b].keys():
                neg_that_pixels = that_pixels[b][0]
                neg_that_amount = min(chosen_pixels.shape[0] // 4, neg_that_pixels.shape[0])
                indices = random.sample(range(0, neg_that_pixels.shape[0]), neg_that_amount)
                pixels.append(that_embedding[b, :, neg_that_pixels[indices, 0], neg_that_pixels[indices, 1]])
                labels.append((torch.zeros(neg_that_amount) if args.negbg else -torch.ones(neg_that_amount)) if chosen_cc == 0 else torch.zeros(neg_that_amount))

            if 0 in this_pixels[b].keys(): 
                neg_this_pixels = this_pixels[b][0]
                neg_this_amount = min(chosen_pixels.shape[0] // 4, neg_this_pixels.shape[0])
                indices = random.sample(range(0, neg_this_pixels.shape[0]), neg_this_amount)
                pixels.append(this_embedding[b, :, neg_this_pixels[indices, 0], neg_this_pixels[indices, 1]])
                labels.append((torch.zeros(neg_this_amount) if args.negbg else -torch.ones(neg_this_amount)) if chosen_cc == 0 else torch.zeros(neg_this_amount))

            this_amount = chosen_pixels.shape[0] - that_amount - neg_that_amount - neg_this_amount
            indices = random.sample(range(0, chosen_pixels.shape[0]), this_amount)
            pixels.append(this_embedding[b, :, chosen_pixels[indices, 0], chosen_pixels[indices, 1]])
            labels.append((torch.zeros(this_amount) if args.negbg else -torch.ones(this_amount)) if chosen_cc == 0 else torch.ones(this_amount))

            mlp_input.append(torch.cat((this_embedding[b, :, chosen_pixels[:, 0], chosen_pixels[:, 1]], torch.cat(pixels, dim=1)), dim=0))
            mlp_label.append(torch.cat(labels))
        mlp_input = torch.cat(mlp_input, dim=1)
        mlp_label = torch.cat(mlp_label, dim=0)

        mlp_inputs.append(mlp_input)
        mlp_labels.append(mlp_label)

    # run through an MLP with inputs being two embeddings
    return torch.cat(mlp_inputs, dim=1).permute(1,0), torch.cat(mlp_labels, dim=0).long().cuda()


def build_assoc_inputs(this_people, this_pixels, this_embedding):
    mlp_inputs = []
    mlp_labels = []

    # before split ccs, check if any single cc has people components and non
    # if so, make positive batches
    # if people

    for b in range(len(this_pixels)):
        # build batch
        mlp_input = []
        mlp_label = []
        how_many = len(this_pixels[b].items())

        for chosen_cc, chosen_pixels in this_pixels[b].items():
            # negative points from background pixels
            # split chosen_cc into people and non people
            # batches are positive between people and non people
            # negatives come from background
            
            # need all pixels with hands to pair with non hands
            # need all pixels with nonhands to pair with hands
            chosen_people = (this_people[b, chosen_pixels[:,0], chosen_pixels[:,1]]).bool()
            people_choices = [int(x) for x in list(torch.where(chosen_people == True)[0])]
            nonpeople_choices = [int(x) for x in list(torch.where(chosen_people == False)[0])]
            pos_this_amount = min(len(people_choices), len(nonpeople_choices))
            neg_needed = chosen_pixels.shape[0] - (pos_this_amount*2)
            random.shuffle(people_choices)
            random.shuffle(nonpeople_choices)

            #if len(people_choices) > len(nonpeople_choices):
            #    ordered_choices = people_choices + nonpeople_choices
            #    paired_choices = nonpeople_choices[:pos_this_amount]
            #else:
            #    ordered_choices = nonpeople_choices + people_choices
            #    paired_choices = people_choices[:pos_this_amount]
            if pos_this_amount > 0:
                ordered_choices = people_choices[:pos_this_amount] + nonpeople_choices[-pos_this_amount:] + ((people_choices + nonpeople_choices)[:neg_needed])
                paired_choices = nonpeople_choices[:pos_this_amount] + people_choices[-pos_this_amount:]
            else:
                ordered_choices = ((people_choices + nonpeople_choices)[:neg_needed])
                paired_choices = []

            # how many total negative pixels do we need to supplement this
            other_neg_embeds = []
            for i, (other_cc, other_pixels) in enumerate(this_pixels[b].items()):
                if chosen_cc == other_cc: 
                    if len(people_choices) > 0:
                        indices = random.sample(people_choices, k=min(len(people_choices), neg_needed // how_many))
                    else:                   
                        indices = random.sample(nonpeople_choices, k=min(len(nonpeople_choices), neg_needed // how_many))
                else:
                    indices = random.sample(range(0, other_pixels.shape[0]), k=min(other_pixels.shape[0], neg_needed // how_many))
                neg_needed -= len(indices)
                other_neg_embeds.append(this_embedding[b, :, other_pixels[indices, 0], other_pixels[indices, 1]])

            if neg_needed > 0:
                #for i, (other_cc, other_pixels) in enumerate(sorted(this_pixels[b].items(), key=lambda x: x[0])):
                for i, (other_cc, other_pixels) in enumerate(this_pixels[b].items()):
                    if chosen_cc == other_cc:
                        if len(people_choices) > 0: 
                            indices = random.choices(people_choices, k=neg_needed)
                        else: 
                            indices = random.choices(nonpeople_choices, k=neg_needed)
                    else:
                        indices = random.choices(range(0, other_pixels.shape[0]), k=neg_needed)
                    indices = random.choices(range(0, other_pixels.shape[0]), k=neg_needed)
                    neg_needed -= len(indices)
                    other_neg_embeds.append(this_embedding[b, :, other_pixels[indices, 0], other_pixels[indices, 1]])
                    if neg_needed <= 0: break
            other_neg_embeds = torch.cat(other_neg_embeds, dim=1)

            ordered_pixels = chosen_pixels[ordered_choices]
            ordered_embed = this_embedding[b, :, ordered_pixels[:, 0], ordered_pixels[:, 1]]

            pos_pixels = chosen_pixels[paired_choices] 
            pos_embed = this_embedding[b, :, pos_pixels[:, 0], pos_pixels[:, 1]]

            paired_embed = torch.cat((pos_embed, other_neg_embeds), dim=1)

            mlp_input.append(torch.cat((ordered_embed, paired_embed), dim=0))
            mlp_label.append(torch.cat((torch.ones(pos_this_amount * 2), torch.zeros(chosen_pixels.shape[0] - (pos_this_amount * 2)))))

        mlp_input = torch.cat(mlp_input, dim=1)
        mlp_label = torch.cat(mlp_label, dim=0)

        mlp_inputs.append(mlp_input)
        mlp_labels.append(mlp_label)

    # run through an MLP with inputs being two embeddings
    return torch.cat(mlp_inputs, dim=1).permute(1,0), torch.cat(mlp_labels, dim=0).long().cuda()


def build_worse_assoc_inputs(this_people, that_people, this_pixels, that_pixels, this_labels, that_labels, this_embedding, that_embedding, merge=False):
    mlp_inputs = []
    mlp_labels = []

    # best way to sample
    # find min of the 4
    # sample that many
    # find remaining
    # sample from each valid remaining pair
    # before split ccs, check if any single cc has people components and non
    # if so, make positive batches
    # if people
    # split chosen_cc into people and non people
    # batches are positive between people and non people
    # negatives come from background and other half
    
    # need all pixels with hands to pair with non hands
    # need all pixels with nonhands to pair with hands
    # negative points from background pixels
    for b in range(len(this_pixels)):
        # build batch
        mlp_input = []
        mlp_label = []

        for chosen_cc, chosen_this_pixels in this_pixels[b].items():
            chosen_this_people = (this_people[b, chosen_this_pixels[:,0], chosen_this_pixels[:,1]]).bool()
            people_this_choices = [int(x) for x in list(torch.where(chosen_this_people == True)[0])]
            nonpeople_this_choices = [int(x) for x in list(torch.where(chosen_this_people == False)[0])]

            peoples = [(people_this_choices, 'this')]
            nonpeoples = [(nonpeople_this_choices, 'this')]

            chosen_that_pixels = None
            if merge and chosen_cc in that_pixels[b].keys(): 
                chosen_that_pixels = that_pixels[b][chosen_cc]
                chosen_that_people = (that_people[b, chosen_that_pixels[:,0], chosen_that_pixels[:,1]]).bool()
                people_that_choices = [int(x) for x in list(torch.where(chosen_that_people == True)[0])]
                nonpeople_that_choices = [int(x) for x in list(torch.where(chosen_that_people == False)[0])]

                peoples.append((people_that_choices, 'that'))
                nonpeoples.append((nonpeople_that_choices, 'that'))

            # from all valid hand+held and held+hand pairs, find out how much we can sample
            p_e_s, pos_embeds = 0, [] 
            for p_i, p_where in peoples:
                for np_i, np_where in nonpeoples:
                    min_ij = min(len(p_i), len(np_i))
                    if min_ij <= 0: continue
                    p_e_s += (min_ij * 1.0) # 2.0
                    pos_embeds.append(((random.sample(p_i, min_ij), p_where), (random.sample(np_i, min_ij), np_where)))
                    #pos_embeds.append(((random.sample(np_i, min_ij), np_where), (random.sample(p_i, min_ij), p_where)))

            # if positive association pairs exist, winnow them down the be at most half of the sub-batch
            if len(pos_embeds) > 0:
                if p_e_s > chosen_this_pixels.shape[0] // 2: #subsample
                    ratio = (chosen_this_pixels.shape[0] / 2) / p_e_s
                    mod_pos_embeds = []
                    for g1, g2 in pos_embeds:
                        new_indices = max(int(len(g1[0]) * ratio), 1)
                        mod_pos_embeds.append(((g1[0][:new_indices], g1[1]), (g2[0][:new_indices], g2[1])))
                else:
                    mod_pos_embeds = pos_embeds

                pos_embeds = []
                for (p_i, p_where), (np_i, np_where) in mod_pos_embeds:
                    p_ee = this_embedding if p_where == 'this' else that_embedding
                    np_ee = this_embedding if np_where == 'this' else that_embedding

                    p_p = chosen_this_pixels if p_where == 'this' else chosen_that_pixels 
                    np_p = chosen_this_pixels if np_where == 'this' else chosen_that_pixels 

                    pos_embeds.append(torch.cat((p_ee[b, :, p_p[p_i, 0], p_p[p_i, 1]],
                                                 np_ee[b, :, np_p[np_i, 0], np_p[np_i, 1]]), dim=0))
                pos_embeds = torch.cat(pos_embeds, dim=1)
                p_e_s = pos_embeds.shape[1]

            # from all valid hand+hand and held+held, find out how many negatives we can sample 
            n_e_s, neg_embeds = 0, []
            to_neg = chosen_this_pixels.shape[0] - p_e_s
            for x_i, x_where in peoples + nonpeoples:
                min_ij = min(len(x_i), to_neg)
                if min_ij <= 0: continue
                n_e_s += min_ij
            ratio = to_neg / n_e_s

            # sample negatives from hand+other 
            for p_i, p_where in peoples:
                min_ij = min(len(p_i), to_neg)
                if min_ij <= 0: continue
                min_ij = max(math.ceil(min_ij * ratio), 1)

                p_l = this_labels[b] if p_where == 'this' else that_labels[b]
                p_pe = this_people[b] if p_where == 'this' else that_people[b]
                p_pe = p_pe.clone().bool()

                not_me_labels = (p_l != chosen_cc) 
                not_me_or_people = (not_me_labels | p_pe)
                nmorp = not_me_or_people[0].nonzero()

                p_neg_1 = random.sample(p_i, min_ij)
                p_neg_2 = random.sample(range(0, nmorp.shape[0]), min_ij)

                p_ee = this_embedding if p_where == 'this' else that_embedding
                p_p = chosen_this_pixels if p_where == 'this' else chosen_that_pixels 

                neg_embeds.append(torch.cat((p_ee[b, :, p_p[p_neg_1, 0], p_p[p_neg_1, 1]],
                                             p_ee[b, :, nmorp[p_neg_2, 0], nmorp[p_neg_2, 1]]), dim=0))

            # sample negatives from held+other 
            for np_i, np_where in nonpeoples:
                min_ij = min(len(np_i), to_neg)
                if min_ij <= 0: continue
                min_ij = max(math.ceil(min_ij * ratio), 1)

                np_l = this_labels[b] if np_where == 'this' else that_labels[b]
                np_pe = this_people[b] if np_where == 'this' else that_people[b]
                np_pe = np_pe.clone().bool()

                not_me_labels = (np_l != chosen_cc) 
                not_me_or_not_people = (not_me_labels | ~np_pe)
                nmornp = not_me_or_not_people[0].nonzero()

                np_neg_1 = random.sample(np_i, min_ij)
                np_neg_2 = random.sample(range(0, nmornp.shape[0]), min_ij)

                np_ee = this_embedding if np_where == 'this' else that_embedding
                np_p = chosen_this_pixels if np_where == 'this' else chosen_that_pixels 

                neg_embeds.append(torch.cat((np_ee[b, :, np_p[np_neg_1, 0], np_p[np_neg_1, 1]],
                                             np_ee[b, :, nmornp[np_neg_2, 0], nmornp[np_neg_2, 1]]), dim=0))

            # accumulate negatives and ditch if overflow
            if len(neg_embeds) > 0:
                neg_embeds = torch.cat(neg_embeds, dim=1)

                if neg_embeds.shape[1] > to_neg:
                    max_neg = random.sample(range(0, neg_embeds.shape[1]), chosen_this_pixels.shape[0] - p_e_s)
                    neg_embeds = neg_embeds[:, max_neg]

            mlp_input.append((torch.cat((pos_embeds, neg_embeds), dim=1) if p_e_s > 0 else neg_embeds))
            mlp_label.append(torch.cat((torch.ones(p_e_s), torch.zeros(chosen_this_pixels.shape[0] - p_e_s))))

        mlp_input = torch.cat(mlp_input, dim=1)
        mlp_label = torch.cat(mlp_label, dim=0)

        mlp_inputs.append(mlp_input)
        mlp_labels.append(mlp_label)

    # run through an MLP with inputs being two embeddings
    return torch.cat(mlp_inputs, dim=1).permute(1,0), torch.cat(mlp_labels, dim=0).long().cuda()


def d_flow(flow):
    # this method calculates the derivative of the flow
    _flow = einops.rearrange(flow, 'b h w c -> b c h w')
    _sg_flow = spatial_gradient(_flow, normalized=False)
    _d_flow = einops.rearrange([_sg_flow[:, 0, 0], _sg_flow[:, 1, 1]], 's b h w -> b s h w')
    return _d_flow

def d_embedding(flow):
    # this method calculates the derivative of the embedding 
    _flow = einops.repeat(flow.clone(), 'c h w -> b c h w', b=1)
    _sg_flow = spatial_gradient(_flow, normalized=False)
    _d_flow = einops.rearrange([_sg_flow[:, :, 0], _sg_flow[:, :, 1]], 's b c h w -> b (s c) h w')
    _dd_flow = fft_integrate(_d_flow.float()).squeeze()
    return einops.rearrange(pca_image(fft_integrate(_d_flow.float()).squeeze()).float(), 'h w c -> c h w')


def build_corr_grid(flow, mesh_grids, args):
    # this grid will describe how this flow field warps to another
    #corr_grid = einops.repeat(mesh_grid, 'h w c -> b h w c', b=flow.shape[0])

    #newX, newY = X+dx, Y+dy
    #reverseDx, reverseDy = ...
    corr_grid = mesh_grids.clone()
    corr_grid[:, :, :, 0] += (flow[:, :, :, 0] / 512)
    corr_grid[:, :, :, 1] += (flow[:, :, :, 1] / 288)
    return corr_grid


def build_rewarp_grid(mesh_grids, that_corr_grid, this_corr_grid):
    #einops.repeat(mesh_grid.clone(), 'h w c -> b c h w', b=that_corr_grid.shape[0])
    mesh_grid_s = einops.rearrange(mesh_grids.clone(), 'b h w c -> b c h w')

    #interp(reverseDx AT newX, newY)
    #| X - interp(reversed, X+dx, Y+dy) |
    re_warped_corr_grid = F.grid_sample(mesh_grid_s, that_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)
    re_warped_corr_grid = F.grid_sample(re_warped_corr_grid, this_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)
    re_warped_corr_grid = einops.rearrange(re_warped_corr_grid, 'b c h w -> b h w c')
    return re_warped_corr_grid


def boundary_inconsistent(flow, re_warped_corr_grid, that_corr_grid, args):
    # optical flow for visualizing -> (X,Y) EVERYWHERE
    vis_flow = einops.rearrange(flow.clone(), 'b h w c -> b c h w')
    warped_flow = F.grid_sample(vis_flow, that_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)
    return (warped_flow.sum(dim=1) == 0.0)


def cycle_inconsistent(flow, re_warped_corr_grid, that_corr_grid, args):
    # optical flow for visualizing -> (X,Y) EVERYWHERE
    vis_flow = einops.rearrange(flow.clone(), 'b h w c -> b c h w')

    # check sum diff and skip bad matches with weighting -- previously 10
    re_warped_flow = F.grid_sample(vis_flow.clone(), re_warped_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)
    cycle_inconsistent = (abs(vis_flow - re_warped_flow).sum(dim=1) > (args.fps_offset_mult * args.offset * 0.33))
    return cycle_inconsistent


def connected_components(mask, args):
    # gets connected components for all masks above the pre-filtered threshold
    all_labels = []
    for b in range(mask.shape[0]):
        this_labels = [torch.zeros(mask[b].shape)]
        this_max = 0
        for v in mask[b].unique().tolist():
            if v == 0: continue

            # get unique mask pieces
            this_label = (torch.as_tensor(label(cp.asarray((mask[b] == v).int()))[0]))

            # limit to 250 pixels+
            this_label = cleanse_component(this_label)

            # limit to 10 ccs
            #this_label[this_label > 20] = 0

            this_label[this_label != 0] += this_max
            this_labels.append(this_label)
            this_max = this_label.max()

        all_labels.append(torch.stack(this_labels).max(dim=0).values)
    return torch.stack(all_labels).cuda()


def merge_component(this_labels, that_labels, that_corr_grid, this_cycle_inconsistent, that_cycle_inconsistent):
    # gets connected components for all masks above the pre-filtered threshold
    clean_this_labels = this_labels.clone().float()
    clean_this_labels[this_cycle_inconsistent != 0.0] = 0.0
    this_warped_labels = F.grid_sample(clean_this_labels.unsqueeze(1), that_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True).int().squeeze()
    this_warped_labels[that_cycle_inconsistent != 0.0] = 0.0

    # fill that labels
    for v in sorted(this_labels.unique().tolist(), reverse=True): 
        if v == 0: continue

        that_values = that_labels[(this_warped_labels == v) & (that_labels != 0)].unique().tolist()
        max_val = max([v] + that_values)

        this_labels[this_labels == v] = max_val
        this_warped_labels[this_warped_labels == v] = max_val
        for vv in that_values: that_labels[that_labels == vv] = max_val

    #that_labels = torch.stack((this_warped_labels, that_labels)).max(dim=0).values
    return that_labels


def cleanse_component(this_label, min_size=250):
    if len(this_label.shape) < 3:
        this_label = this_label.unsqueeze(0)

    for b in range(this_label.shape[0]):
        for v in this_label[b].unique().tolist():
            if (this_label[b] == v).sum() < min_size:
                this_label[b][this_label[b] == v] = 0.0
    return this_label.squeeze()


def rebase_components(this_label, that_label):
    real_this_label, real_that_label = this_label.clone(), that_label.clone()
    for i, val in enumerate(sorted(list(set(this_label.unique().tolist() + that_label.unique().tolist())))):
        real_this_label[this_label == val] = i
        real_that_label[that_label == val] = i
    return real_this_label, real_that_label


def fit_motion_model(mask, cycle_inconsistent, that_corr_grid, ransac, acceptable, mesh_grids, args):
    # fits a motion model between two frames, within mask pixels
    all_F_mats, all_inl = [], []
    for b in range(that_corr_grid.shape[0]):
        this_F_mats, this_inl = [], []
        pts_mask = (~cycle_inconsistent[b] & mask[b])

        if pts_mask.sum() <= 8: pts_mask = mask[b]
        if pts_mask.sum() <= 8: pts_mask = torch.ones(mask[b].shape).bool()
        if pts_mask.sum() <= 8:
            this_F_mats.append(torch.zeros((F_mat.shape)).cuda())
            this_inl.append(torch.zeros((inl.shape)).cuda())

        ptsA = (mesh_grids[b][pts_mask]).float()
        ptsB = (that_corr_grid[b][pts_mask]).float()
        pts = random.sample(range(0, ptsA.shape[0]), min(2500, ptsA.shape[0]))

        F_mat, inlier_mask = ransac.forward(ptsA[pts], ptsB[pts])

        # ensure good fit
        errors = ransac.error_fn(mesh_grids[b], that_corr_grid[b], einops.repeat(F_mat, 'h w -> c h w', c=1))
        inl = (errors <= (ransac.inl_th * acceptable)).cuda()

        all_F_mats.append(F_mat)
        all_inl.append((inl & pts_mask).float().max(dim=0).values)

        pts_mask = ((~inl & mask[b]) & ~cycle_inconsistent[b]).cuda()
    return torch.stack(all_F_mats), torch.stack(all_inl).bool()


def fit_motion_models(this_labels, that_corr_grid, ransac, acceptable, mesh_grid, args):
    # fits a motion model between two frames, within mask pixels
    all_F_mats, all_outl = [], []
    for b in range(that_corr_grid.shape[0]):
        this_F_mats, this_outl = {}, {} 
        for i, cc in enumerate(this_labels[b].unique()):
            pts_mask = (this_labels[b] == cc)
            if pts_mask.sum() <= 8: pts_mask = torch.ones(this_labels[b].shape).bool()

            ptsA = (mesh_grid[pts_mask]).double()
            ptsB = (that_corr_grid[b][pts_mask]).double()
            pts = random.sample(range(0, ptsA.shape[0]), min(5000, ptsA.shape[0]))

            F_mat, inlier_mask = ransac.forward(ptsA[pts], ptsB[pts])
            F_mat = F_mat.float()

            # ensure good fit
            errors = ransac.error_fn(mesh_grid, that_corr_grid[b], einops.repeat(F_mat, 'h w -> c h w', c=1))
            inl = (errors <= (ransac.inl_th * acceptable))

            # check how many inl are in my cc
            # check how many are in other cc
            this_F_mats[int(cc)] = F_mat
            this_outl[int(cc)] = (~inl).float()

        all_F_mats.append(this_F_mats)
        all_outl.append(this_outl)
    return all_F_mats, all_outl


def epipolar_distance(that_corr_grid, this_F_mats, mesh_grids, args):
    # produces a h x w map of epipolar error for the given motion model
    sed = sampson_epipolar_distance(mesh_grids.flatten(1, 2), that_corr_grid.flatten(1, 2), this_F_mats).reshape((mesh_grids.shape[0], mesh_grids.shape[1], mesh_grids.shape[2]))

    if args.epipolar_log:
        sed = torch.log(sed + 1e-7)
    for b in range(that_corr_grid.shape[0]):
        sed[b] -= sed[b].min()
        sed[b] /= sed[b].max()
        sed[b] -= sed[b].mean()
    return sed #einops.rearrange(torch.stack(seds), 'i b h w -> b i h w')


def calculate_epipoles(this_F_mats):
    epipoles = []
    for b in range(this_F_mats.shape[0]):
        epipoles.append(torch.tensor(scipy.linalg.null_space(this_F_mats[b].T.cpu().numpy())))
    unnorm_epipoles = torch.stack(epipoles)
    return unnorm_epipoles.squeeze() / unnorm_epipoles[:, -1]

def flow_above_mean(sed):
    #sed = (torch.linalg.vector_norm(this_flow, dim=-1).float() + 1e-5)

    for b in range(sed.shape[0]):
        sed[b] -= sed[b].min()
        sed[b] /= sed[b].max()
        sed[b] -= sed[b].mean()

    return sed


def cyclize(now_measure, future_measure, now_corr_grid, future_corr_grid, now_cycle_inconsistent, future_cycle_inconsistent):
    # warps a grid forwards and backwards to see where the correlation grids are consistent
    if len(now_measure.shape) == 3:
        now_measure = einops.repeat(now_measure, 'b h w -> b c h w', c=1)
        future_measure = einops.repeat(future_measure, 'b h w -> b c h w', c=1)

    warped_now_measure = now_measure.clone()
    warped_future_measure = future_measure.clone()

    warped_now_measure = F.grid_sample(warped_now_measure, future_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)
    warped_future_measure = F.grid_sample(warped_future_measure, now_corr_grid, mode='nearest', padding_mode='zeros', align_corners=True)

    double_now_measure = now_measure + warped_future_measure
    double_future_measure = future_measure + warped_now_measure

    return double_now_measure, double_future_measure


def semantics(sim_ccs):
    # check output against embedding
    # cosine similarity each embedding to cls_avg_embed
    # choose top-k

    # or each cluster chooses top-1
    # find best match from entity 
    cls_cluster_map = {}
    for cc, embeds in sim_ccs.items():
        avg_embed = embeds[0]
        my_best_sim = -1
        my_cls = None
        for cls, cls_embed in cls_avg_embed.items():
            this_sim = F.cosine_similarity(avg_embed, cls_embed, dim=0)
            if this_sim > my_best_sim:
                my_best_sim = this_sim
                my_cls = cls
        cls_cluster_map[int(cc)] = my_cls
    return cls_cluster_map


def calculate_top3(output, segments):
    # check closest clusters cosine sim 
    #outliers = (segments[0] == -1)

    unique_ccs = [int(x) for x in segments.unique()]
    sim_ccs = {}
    for cc in unique_ccs:
        sim_ccs[cc] = calculate_average_embedding(output[0], segments, cc)

    sim_ccs_items = list(sim_ccs.items())
    cc_similarity = torch.stack([v[1] for (k, v) in sim_ccs_items])
    this_k = min(3, len(unique_ccs))
    closest_cc_idx = cc_similarity.topk(k=this_k, dim=0).indices
    top3 = torch.zeros((3, segments.shape[0], segments.shape[1])).cuda()
    top3[:this_k] = (torch.stack([torch.tensor(k) for (k, v) in sim_ccs_items]).cuda())[closest_cc_idx]
    return top3


def calculate_average_embedding(this_e, this_labels, cc):
    # calculates average embeddings wherever a label equals cc
    this_cc = torch.einsum('chw,ihw->chw', torch.nan_to_num(this_e), (this_labels == cc).squeeze().unsqueeze(0))
    this_cc[this_cc == 0.0] = float('nan')
    avg_embed = torch.nan_to_num((F.normalize(this_cc.nanmean(dim=(1, 2)), dim=-1)))
    exp_avg_embed = avg_embed.unsqueeze(-1).unsqueeze(-1).expand_as(this_e)
    sim_avg_embed = torch.nan_to_num(F.cosine_similarity(this_e, exp_avg_embed, dim=0, eps=1e-3))
    return avg_embed, sim_avg_embed, this_labels == cc


def segment_embeddings(embed, clust):
    real_segments = []
    shape = (embed.shape[2], embed.shape[3])
    embed = einops.rearrange(embed, 'b c h w -> b (h w) c').cuda()

    for b in range(embed.shape[0]):
        try:
            real_segments.append(torch.as_tensor(clust.fit_predict(embed[b]).reshape(shape)))
        except Exception as e:
            print('clustering failed!')
            print(str(e))
            new_segment = torch.zeros(shape)
            new_segment[shape[0]-1:shape[0]+1, shape[1]-1:shape[1]+1] = 1
            real_segments.append(new_segment)

    return torch.stack(real_segments).cuda()


def store_affinities(output, i, args):
    # grid sample 100 points
    chosen_100 = output[0][:, 3::6, 3::6]
    expand_me = lambda x: x.unsqueeze(-1).unsqueeze(-1).expand_as(output[0])
    expand_100 = lambda x: x.unsqueeze(-1).unsqueeze(-1).expand_as(chosen_100)

    # calculate affinities
    im = Image.fromarray(np.uint8((inv_normalize(now_rgb[0]) * 255.0).permute(1, 2, 0).cpu().numpy()))
    im.save(f'/home/relh/public_html/affinities/images/{i}_image.png', format='PNG')

    print(f'image {i}..')
    affinities, chosen = [], []
    for x in range(chosen_100.shape[-2]):
        for y in range(chosen_100.shape[-1]):
            #print(f'{x}, {y}')
            if 'mlp' in args.affinity:
                flat_embed = einops.repeat(output[0], 'c h w -> (h w) c')
                tile_point = einops.repeat(chosen_100[:, x, y], 'c -> z c', z=flat_embed.shape[0])
                if args.ddp:
                    logits = net.module.compare_nn(torch.cat((flat_embed, tile_point), dim=1))
                else:
                    logits = net.compare_nn(torch.cat((flat_embed, tile_point), dim=1))
                point_sim = F.softmax(logits, dim=1)[:, 1].reshape(output[0].shape[-2], output[0].shape[-1])
                plt.imsave(f'/home/relh/public_html/affinities/{args.affinity}/{i}_{x}_{y}_{args.name}.png', (point_sim).float().cpu(), cmap='magma')
            elif 'cosine' in args.affinity:
                # save as 0->1 images indexed by coordinate
                point_sim = (F.cosine_similarity(output[0], expand_me(chosen_100[:, x, y]), dim=0))
                plt.imsave(f'/home/relh/public_html/affinities/{args.affinity}/{i}_{x}_{y}_{args.name}.png', (point_sim).float().cpu(), cmap='magma')
            elif 'stacked' in args.affinity:
                # save as 0->1 images indexed by coordinate
                point_sim = (F.cosine_similarity(output[0], expand_me(chosen_100[:, x, y]), dim=0)).cuda()
                chosen_sim = (F.cosine_similarity(chosen_100, expand_100(chosen_100[:, x, y]), dim=0)).cuda()

                # flatten below mean
                point_sim[point_sim < point_sim.mean()] = point_sim.mean()
                chosen_sim[chosen_sim < chosen_sim.mean()] = chosen_sim.mean()

                # stack the affinities
                affinities.append(((point_sim - point_sim.min()) / (point_sim - point_sim.min()).max()))
                chosen.append(((chosen_sim - chosen_sim.min()) / (chosen_sim - chosen_sim.min()).max()))
            elif 'median' in args.affinity:
                point_sim = (F.cosine_similarity(output[0], expand_me(chosen_100[:, x, y]), dim=0))
                point_sim[point_sim < point_sim.median()] = point_sim.median()
                plt.imsave(f'/home/relh/public_html/affinities/{args.affinity}/{i}_{x}_{y}_{args.name}.png', (point_sim / point_sim.max()).float().cpu(), cmap='magma', format='png')

    # stack the affinities weighted by the relevance
    if 'stacked' in args.affinity:
        affinities = torch.stack(affinities).cuda()
        chosen = torch.stack(chosen).cuda()
        counter = 0
        for x in range(chosen_100.shape[-2]):
            print(x)
            for y in range(chosen_100.shape[-1]):
                this_sim = (chosen[counter].flatten().unsqueeze(-1).unsqueeze(-1) * affinities).sum(dim=0)
                plt.imsave(f'/home/relh/public_html/affinities/{args.affinity}/{i}_{x}_{y}_{args.name}.png', (this_sim / this_sim.max()).float().cpu(), cmap='magma', format='png')
                counter += 1

    if i >= 10:
        sys.exit()

    # visualize and listen for mouseover

def change_brightness(img):
    '''
    input: BGR from cv2
    output: BGR
    '''
    img = Image.fromarray(np.uint8((inv_normalize(img) * 255.0).permute(1, 2, 0).cpu().numpy()))#[:, : , ::-1])
    #image brightness enhancer
    enhancer = ImageEnhance.Brightness(img)
    factor = 0.4 #darkens the image
    im_output = enhancer.enhance(factor)
    im_output = np.asarray(im_output)[:, :, ::-1]
    return im_output


def store_image(inp=None, label='features', option='save', iii=0, bb=0, args=None):
    # swiss army knife method for saving outputs condition on what type of output it is
    for b in range(bb):
        if option == 'save':
            if 'score' in label:
                save_image(inp[b].float().cpu(), f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', format='png')
            elif 'association' in label:
                if inp[b].sum() != 0:
                    save_image((inp[b] / inp[b].max()).cpu(), f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', format='png')
            else:
                save_image((inp[b] / inp[b].max()).cpu(), f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', format='png')
        elif option == 'overlay':
            mask, rgb = inp
            rgb_dimmed = change_brightness(rgb[b])
            this_mask = (cv2.cvtColor(np.float32(mask[b].cpu().numpy()), cv2.COLOR_GRAY2RGB) * (255, 0, 0)).astype(np.uint8)
            final_image = cv2.addWeighted(rgb_dimmed, 1.0, this_mask, 0.7, 0)
            cv2.imwrite(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', final_image)
        elif option == 'pca':
            # combo pca
            combo_pca = einops.rearrange(pca_image(torch.nan_to_num(torch.cat([x[b] for x in inp], dim=-1))), 'h w c -> c h w').cpu()
            for i, l in zip(range(len(inp)), label):
                save_image(combo_pca[:, :, i * args.embed_size[1]:(i + 1) * args.embed_size[1]], f'{args.experiment_path}/outputs/{iii}_{b}_{l}.png', format='png')
        elif option == 'flow':
            import flow_vis
            flow_vis_out = flow_vis.flow_to_color(inp[b].cpu().numpy(), convert_to_bgr=False)
            flow_vis_out = Image.fromarray(np.uint8(flow_vis_out))
            flow_vis_out.save(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png')
        elif option == 'rgb':
            im = Image.fromarray(np.uint8((inv_normalize(inp[b]) * 255.0).permute(1, 2, 0).cpu().numpy()))
            im.save(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', format='png')
        else:
            if 'bwr' in option:
                plt.imsave(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', (inp[b]).float().cpu(), cmap=option, vmin=-100, vmax=100)
            else:
                plt.imsave(f'{args.experiment_path}/outputs/{iii}_{b}_{label}.png', (inp[b] / inp[b].max()).float().cpu(), cmap=option)


def skip_other_bad_indices(i):
    if i == 16:
        print('skipping!')
        return True
    elif i == 36:
        print('skipping!')
        return True
    elif i == 44:
        print('skipping!')
        return True
    return False

def skip_bad_indices(i):
    if i > 1613 and i < 1616:
        print('skipping!')
        return True
    if i > 2367 and i < 2370:
        print('skipping!')
        return True
    if i > 3220 and i < 3223:
        print('skipping!')
        return True
    if i > 3804 and i < 3807:
        print('skipping!')
        return True
    if i > 5609 and i < 5612:
        print('skipping!')
        return True
    if i > 6147 and i < 6150:
        print('skipping!')
        return True
    if i > 6813 and i < 6816:
        print('skipping!')
        return True
    return False


'''
def draw_contour_on_mask(size, cnt, color:int = 255):
    mask = np.zeros(size, dtype='uint8')
    mask = cv2.drawContours(mask, [cnt], -1, color, -1)
    return mask


def get_furthest_point_from_edge_cv2(cnt):
    mask = draw_contour_on_mask((H,W), cnt)
    dist_img = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    cy, cx = np.where(dist_img==dist_img.max())
    cx, cy = cx.mean(), cy.mean()  # there are sometimes cases where there are multiple values returned for the visual center
    return cx, cy


def get_center_of_mass(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx, cy
'''
