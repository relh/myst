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
import scipy.io as sio
#import tensorflow as tf
import torch
import torch.nn.functional as F
from decord import VideoLoader, VideoReader, cpu, gpu
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import pad
#from keras.utils.generic_utils import Progbar
from mmflow.apis import inference_model, init_model
from people_segmentation.pre_trained_models import create_model
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from dataset import MiniDataset

#from models.adversarial_learner import AdversarialLearner
#from models.pytorch_pwc.run import estimate
#from models.utils.general_utils import (compute_boundary_score,
#                                        postprocess_image, postprocess_mask)

#try:
#    import imageio
#except ImportError:
#    imageio = None
'''
This standalone file generates all the labels needed for the present iteration
'''
dataset = 'ego4d'

if dataset == 'epickitchens':
    video_path = '/y/relh/epickitchens/videos/'
    depth_path = '/y/relh/epickitchens/depth/'
    dummy_frame_path = '/y/relh/epickitchens/new_frames/'
    frame_path = '/x/relh/epickitchens/frames/'
    cache_path = '/x/relh/epickitchens/cache/'
    people_path = '/x/relh/epickitchens/people/'
    root_flow_path = '/x/relh/epickitchens/flow/'
elif dataset == 'ego4d':
    video_path = '/Pool2/users/relh/ego4d_data/v1/full_scale/'
    #depth_path = '/y/relh/epickitchens/depth/'
    dummy_frame_path = '/z/relh/ego4d/new_frames/'
    frame_path = '/z/relh/ego4d/frames/'
    cache_path = '/z/relh/ego4d/cache/'
    people_path = '/z/relh/ego4d/people/'
    root_flow_path = '/z/relh/ego4d/flow/'
elif dataset == 'playroom':
    frame_path = '/home/relh/eisen/datasets/Playroom/images/'
    cache_path = '/z/relh/playroom/cache/'
    people_path = '/z/relh/playroom/people/'
    root_flow_path = '/z/relh/playroom/flow/'

#os.makedirs(dummy_frame_path, exist_ok=True)
#os.makedirs(frame_path, exist_ok=True)
#os.makedirs(cache_path, exist_ok=True)
#os.makedirs(people_path, exist_ok=True)

#de.bridge.set_bridge('torch')

def frames_and_people(videos, split='val'):
    my_gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])

    model = create_model("Unet_2020-07-20").to(0)
    model.eval()
    size = (576, 1024) if dataset == 'epickitchens' else (648, 864) 

    metadata = json.load(open(f'/home/relh/annotated_ego4d/ego4d.json'))

    fho_hands = json.load(open(f'/home/relh/annotated_ego4d/fho_hands_{split}.json'))
    fho_scod = json.load(open(f'/home/relh/annotated_ego4d/fho_scod_{split}.json'))
    fho_sta = json.load(open(f'/home/relh/annotated_ego4d/fho_sta_{split}.json'))

    my_videos = list(videos)[my_gpu::7]
    my_videos = my_videos[::-1]

    for vvv in tqdm(my_videos):
        # want to find all the frames and their offset pairs we may care about
        my_video = vvv['video_uid']

        # my hand clips frames
        my_hands_clips = [x for x in fho_hands['clips'] if x['video_uid'] == my_video]
        to_process = []
        for mhc in my_hands_clips:
            for frame in mhc['frames']:
                to_process.append(frame['action_start_frame'])
                to_process.append(frame['action_end_frame'])
                # everything below here also boxes into right and left hand
                if 'pre_45' in frame:
                    to_process.append(frame['pre_45']['frame'])
                if 'pre_30' in frame:
                    to_process.append(frame['pre_30']['frame'])
                if 'pre_15' in frame:
                    to_process.append(frame['pre_15']['frame'])
                if 'pre_frame' in frame:
                    to_process.append(frame['pre_frame']['frame'])
                if 'post_frame' in frame:
                    to_process.append(frame['post_frame']['frame'])
                if 'pnr_frame' in frame:
                    to_process.append(frame['pnr_frame']['frame'])
                if 'contact_frame' in frame:
                    to_process.append(frame['contact_frame']['frame'])

        # my scod clips frames
        my_scod_clips = [x for x in fho_scod['clips'] if x['video_uid'] == my_video]
        for msc in my_scod_clips:
            if 'post_frame' in msc:
                to_process.append(msc['post_frame']['frame_number'])
            if 'pnr_frame' in msc:
                to_process.append(msc['pnr_frame']['frame_number'])
            if 'contact_frame' in msc:
                to_process.append(msc['contact_frame']['frame_number'])

        # my sta clips frames
        my_sta_clips = [x for x in fho_sta['annotations'] if x['video_id'] == my_video]
        for mstac in my_sta_clips:
            to_process.append(mstac['frame'])

        # now have list of frames from video we want
        to_process = list(set(to_process))
        print(f'to_process: {len(to_process)}')

        #if os.path.exists(frame_path + v) and len(os.listdir(frame_path + v)) > 60: 
        #    print('skipped!')
        #    continue

        #print(v)
        v = vvv['video_uid']
        this_video_path = vvv['video_uid'] + '.mp4' 
        try:
            vr = VideoReader(video_path + this_video_path, ctx=gpu())
        except Exception as e:
            print('--- LOAD SKIP ---')
            print(str(e))
            continue

        os.makedirs(frame_path + f'{v}/', exist_ok=True)
        os.makedirs(people_path + f'{v}/', exist_ok=True)

        if dataset == 'epickitchens':
            group_indices = list(range(len(vr)))
        else:
            #if len(vr) < 100: continue
            indices = to_process
            indices += [x+10 for x in indices]
            indices = [x for x in indices if x < len(vr)]
            indices = sorted(indices)
            group_indices = [indices[i:i+50] for i in range(0, len(indices), 50)]
        try:
            with torch.no_grad():
                for xyz, i_group in enumerate(group_indices):
                    print(f'{my_gpu}: {xyz} / {len(group_indices)}.. {(xyz / len(group_indices)) * 100.0}.. {v}')
                    # 0. generate frame
                    if dataset == 'epickitchens':
                        og_frames = [vr[i]]
                    else:
                        og_frames = vr.get_batch(i_group) #[i, i+10, i+20, i+30])#, i+60, i+120, i+300])

                    for iii, og in zip(i_group, og_frames): 
                        #if os.path.exists(people_path + f'{v}/{v}_{iii}.png'):
                        #    data = Image.open(people_path + f'{v}/{v}_{iii}.png')
                        #    print(data.size)
                        #    if data.size == tuple(list(size)[::-1]):
                        #        print('skipped!')
                        #        continue
                        frame = og.cpu().numpy()
                        image = np.uint8(frame)

                        #if not os.path.exists(frame_path + v + f'/{v}_{iii}.png'):
                        original_frame = F.interpolate(einops.repeat(og, 'h w c -> b c h w', b=1), size=size) #(648, 864))
                        original_frame = np.uint8(einops.rearrange(original_frame.squeeze(), 'c h w -> h w c').cpu().numpy())
                        original_frame = Image.fromarray(original_frame)
                        original_frame.save(frame_path + v + f'/{v}_{iii}.png', format='PNG')

                        # 1. generate people
                        transform = albu.Compose([albu.Normalize(p=1)], p=1)
                        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
                        x = transform(image=padded_image)["image"]
                        x = torch.unsqueeze(tensor_from_rgb_image(x), 0).to(0)
                        prediction = model(x)[0][0]
                        mask = (prediction > 0).bool()

                        x_min_pad, y_min_pad, x_max_pad, y_max_pad = pads
                        height, width = mask.shape[:2]
                        mask = mask[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]
                        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=size).squeeze() #(548, 864)
                        mask = mask.bool().cpu().numpy()
                        mask = Image.fromarray(np.asarray(mask))
                        mask.save(people_path + f'{v}/{v}_{iii}.png', format='PNG')
        except Exception as e:
            print('--- BATCH SKIP ---')
            print(str(e))
            continue
        gc.collect()
        torch.cuda.empty_cache()

def cache(videos, offset=10):
    # 2. save cache
    os.makedirs(cache_path, exist_ok=True)
    flow_path = root_flow_path + f'{offset}/'
    negflow_path = root_flow_path + f'{-offset}/'
    os.makedirs(flow_path, exist_ok=True)
    os.makedirs(negflow_path, exist_ok=True)

    paired_frames = []
    for v in videos:
        v = v.split('.')[0]
        print(v)
        this_flow_path = flow_path + v + '/'
        os.makedirs(this_flow_path, exist_ok=True)
        this_negflow_path = negflow_path + v + '/'
        os.makedirs(this_negflow_path, exist_ok=True)

        this_frame_path = frame_path + v + '/'
        if os.path.exists(this_frame_path):
            frames = os.listdir(this_frame_path)
            print(f'{this_frame_path}.. {len(frames)}')

            for frame in frames:
                frame_num = int(frame.split('_')[-1].split('.png')[0])
                future_frame = frame.replace(str(frame_num) + '.png', str(frame_num + offset) + '.png')
                if os.path.exists(this_frame_path + future_frame):
                    paired_frames.append(frame.split('.')[0]) # sans extension

    os.makedirs(cache_path, exist_ok=True)
    with open(cache_path + f'{offset}_{dataset}_frame_cache.pkl', 'wb') as handle:
        pickle.dump(paired_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)


def filter_cache(offset=10):
    with open(cache_path + f'{offset}_{dataset}_frame_cache.pkl', 'rb') as handle:
        paired_frames = pickle.load(handle)

    filtered_paired_frames = sorted(paired_frames, key=lambda x: int(x.split('_')[2]))
    filtered_paired_frames = sorted(filtered_paired_frames, key=lambda x: int(x.split('_')[1]))
    filtered_paired_frames = sorted(filtered_paired_frames, key=lambda x: x.split('_')[0])
    old_person, old_video, old_frame_num = None, None, None

    new_filtered_paired_frames = []
    for i, frame in enumerate(filtered_paired_frames):
        print(f'{i} / {len(filtered_paired_frames)}.. {(i / len(filtered_paired_frames)) * 100.0}')
        person, video, now_frame_num = frame.split('_')
        name = '_'.join([person, video])
        future_frame_num = str(int(now_frame_num) + offset)
        future_frame = '_'.join([name, str(future_frame_num)])

        if old_frame_num is not None:
            if person == old_person and video == old_video:
                if dataset == 'epickitchens':
                    if (int(now_frame_num) - int(old_frame_num)) > 10:
                        new_filtered_paired_frames.append(frame)
                    else:
                        continue
                else:
                    new_filtered_paired_frames.append(frame)
            else:
                new_filtered_paired_frames.append(frame)
        else:
            new_filtered_paired_frames.append(frame)

        old_person, old_video, old_frame_num = person, video, now_frame_num

    with open(cache_path + f'{offset}_{dataset}_filtered_frame_cache.pkl', 'wb') as handle:
        pickle.dump(new_filtered_paired_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)


def refine_cache(offset=10):
    with open(cache_path + f'{offset}_filtered_frame_cache.pkl', 'rb') as handle:
        filtered_paired_frames = pickle.load(handle)

    old_person, old_video, old_frame_num = None, None, None

    new_filtered_paired_frames = []
    for i, frame in enumerate(filtered_paired_frames):
        print(f'{i} / {len(filtered_paired_frames)}.. {(i / len(filtered_paired_frames)) * 100.0}')
        person, video, now_frame_num = frame.split('_')
        name = '_'.join([person, video])
        future_frame_num = str(int(now_frame_num) + offset)
        future_frame = '_'.join([name, str(future_frame_num)])

        now_frame_path = (frame_path + name + '/' + frame + '.png')
        future_frame_path = (frame_path + name + '/' + future_frame + '.png')
        # TODO check for cycle inconsistency

        new_filtered_paired_frames.append(frame)
        old_person, old_video, old_frame_num = person, video, now_frame_num

    with open(cache_path + f'{offset}_refined_frame_cache.pkl', 'wb') as handle:
        pickle.dump(new_filtered_paired_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)

def merge_cache(offset=10):
    with open(cache_path + f'{offset}_single_frame_cache.pkl', 'rb') as handle:
        one_filtered_paired_frames = pickle.load(handle)
    with open(cache_path + f'{offset*3}_filtered_frame_cache.pkl', 'rb') as handle:
        three_filtered_paired_frames = pickle.load(handle)

    old_person, old_video, old_frame_num = None, None, None

    new_filtered_paired_frames = []
    special_counter = 0
    for i, frame in enumerate(three_filtered_paired_frames):
        print(f'{i} / {len(three_filtered_paired_frames)}.. {(i / len(three_filtered_paired_frames)) * 100.0}')
        person, video, now_frame_num = frame.split('_')
        name = '_'.join([person, video])
        future_frame_num = str(int(now_frame_num) + 3*offset)
        future_frame = '_'.join([name, str(future_frame_num)])

        one_frame = one_filtered_paired_frames[i + special_counter]
        one_person, one_video, one_frame_num = one_frame.split('_')

        #now_frame_path = (frame_path + name + '/' + frame + '.png')
        #future_frame_path = (frame_path + name + '/' + future_frame + '.png')

        if old_frame_num is not None:
            if person == old_person and video == old_video:
                if one_filtered_paired_frames[i + special_counter] == frame:
                    new_filtered_paired_frames.append(frame)
                else:
                    special_counter += 1
                    print(f'inc special counter! {special_counter}')
                    if one_filtered_paired_frames[i + special_counter] == frame:
                        new_filtered_paired_frames.append(frame)
            else:
                if one_person == person and one_video == video:
                    new_filtered_paired_frames.append(frame)
                else:
                    print(f'inc special counter! {special_counter}')
                    special_counter += 1
        else:
            new_filtered_paired_frames.append(frame)

        old_person, old_video, old_frame_num = person, video, now_frame_num

    with open(cache_path + f'{offset}_new_frame_cache.pkl', 'wb') as handle:
        pickle.dump(new_filtered_paired_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pseudolabel(paired_frames, offset=30, neg=False, pwc=False):
    sys.path.append("./labelling/unsupervised_detection/")
    from common_flags import FLAGS
    from my_models.adversarial_learner import AdversarialLearner
    from my_models.utils.general_utils import (compute_boundary_score,
                                               postprocess_image,
                                               postprocess_mask)
    argv = sys.argv
    argv = FLAGS(argv)  # parse flags
    FLAGS.img_width = '1024'
    FLAGS.img_height = '576'
    FLAGS.batch_size = '2'
    FLAGS.root_dir = '2'
    FLAGS.ckpt_file = '/home/relh/inferring_actions/labelling/unsupervised_detection_models/davis_best_model/model.best'
    learner = AdversarialLearner()
    #learner.setup_inference(FLAGS, aug_test=False)
    saver = tf.compat.v1.train.Saver([var for var in tf.compat.v1.trainable_variables()])
    #sv = tf.compat.v1.train.Supervisor(logdir=FLAGS.test_save_dir,
    #                         save_summaries_secs=0,
    #                         saver=None)
    #with sv.managed_session() as sess:
    #with tf.Session() as sess:
    if True:
        checkpoint = FLAGS.ckpt_file
        if checkpoint:
            saver.restore(learner, checkpoint)
            print("Resume model from checkpoint {}".format(checkpoint))
        else:
            raise IOError("Checkpoint file not found")

        #sess.run(learner.test_iterator.initializer)

        n_steps = int(np.ceil(learner.test_samples / float(FLAGS.batch_size)))

        progbar = Progbar(target=n_steps)

        i = 0

        for step in range(n_steps):
            #if sv.should_stop():
            #    break
            try:
                inference = learner.inference(sess)
            except tf.errors.OutOfRangeError:
                  print("End of testing dataset")  # ==> "End of dataset"
                  break
            # Now write images in the test folder
            for batch_num in range(inference['input_image'].shape[0]):

                # select mask
                generated_mask = inference['gen_masks'][batch_num]
                gt_mask = inference['gt_masks'][batch_num]
                category = inference['img_fname'][batch_num].decode("utf-8").split('/')[-2]

                iou, out_mask = compute_IoU(gt_mask=gt_mask, pred_mask_f=generated_mask)
                pdb.set_trace()


def flow(paired_frames, offset=30, neg=False, pwc=False):
    # 3. read offset
    if dataset == 'epickitchens':
        root_flow_path = '/x/relh/epickitchens/flow/'
    elif dataset == 'ego4d':
        root_flow_path = '/z/relh/ego4d/flow/'
    elif dataset == 'playroom':
        root_flow_path = '/z/relh/playroom/flow/'

    if not pwc:
        def parse_args():
            parser = ArgumentParser()
            parser.add_argument('--video', help='video file')
            parser.add_argument('--config', help='Config file', default='labelling/mmflow/configs/gma/gma_8x2_120k_flyingthings3d_sintel_368x768.py')
            parser.add_argument('--checkpoint', help='Checkpoint file', default='labelling/mmflow/gma_8x2_120k_flyingthings3d_sintel_368x768.pth')
            parser.add_argument('--out', help='File to save visualized flow map', default='out_demo.mp4')
            parser.add_argument('--gt', default=None, help='video file of ground truth for input video')
            parser.add_argument('--device', default='cuda:0', help='Device used for inference')
            args = parser.parse_args()
            return args

        args = parse_args()
        # build the model from a config file and a checkpoint file
        model = init_model(args.config, args.checkpoint, device=args.device)
    else:
        root_flow_path = root_flow_path.replace('flow', 'pwc-flow')
        root_flow_path = root_flow_path.replace('/x/', '/y/')

    with torch.no_grad():
        flow_path = root_flow_path + f'{-offset if neg else offset}/'
        print(flow_path)
        os.makedirs(flow_path, exist_ok=True)
        #with open(cache_path + f'{offset}_{dataset}_frame_cache.pkl', 'rb') as handle:
        #    paired_frames = pickle.load(handle)
        #paired_frames = paired_frames[::-1]

        for i, now_frame in enumerate(paired_frames):
            if dataset == 'epickitchens':
                person, video, now_frame_num = now_frame.split('_')
                name = '_'.join([person, video])
            elif dataset == 'ego4d':
                name, now_frame_num = now_frame.split('_')
            elif dataset == 'playroom':
                name = '/'.join(now_frame.split('/')[-3:-1])
                now_frame_num = 0
                now_frame = 'frame0'

            future_frame_num = str(int(now_frame_num) + offset)

            if dataset == 'playroom':
                future_frame = 'frame1'
            else:
                future_frame = '_'.join([name, str(future_frame_num)])
            os.makedirs(flow_path + name + '/', exist_ok=True)

            if neg:
                rgb_future_flow_file = flow_path + name + '/' + future_frame + '.npz'
                print(f'{i} / {len(paired_frames)}.. {(i / len(paired_frames)) * 100.0} {rgb_future_flow_file}')
                if os.path.exists(rgb_future_flow_file): 
                    try:
                        abc = np.load(rgb_future_flow_file)['arr_0']
                    except:
                        continue # and 'P01_102' not in rgb_future_flow_file: continue
                        pass

                if not pwc:
                    rgb_now_frame = cv2.imread(frame_path + name + '/' + now_frame + '.png')

                    if not os.path.exists(frame_path + name + '/' + future_frame + '.png'):
                        continue

                    rgb_future_frame = cv2.imread(frame_path + name + '/' + future_frame + '.png')


                    if rgb_now_frame is None or rgb_future_frame is None: continue

                    flow = inference_model(model, rgb_future_frame, rgb_now_frame)
                    flow = flow.astype(np.float16)
                    np.savez_compressed(rgb_future_flow_file, flow)
                else:
                    tenTwo = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(frame_path+name+'/'+now_frame+'.png'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
                    tenOne = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(frame_path+name+'/'+future_frame+'.png'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

                    tenOutput = estimate(tenOne, tenTwo)

                    flow = tenOutput.cpu().numpy().astype(np.float16)
                    np.savez_compressed(rgb_future_flow_file, flow)
            else:
                rgb_now_flow_file = flow_path + name + '/' + now_frame + '.npz'
                print(f'{i} / {len(paired_frames)}.. {(i / len(paired_frames)) * 100.0} {rgb_now_flow_file}')
                if os.path.exists(rgb_now_flow_file):
                    try:
                        abc = np.load(rgb_now_flow_file)['arr_0']
                    except:
                        continue # and 'P01_102' not in rgb_future_flow_file: continue
                        pass

                if not pwc:
                    rgb_now_frame = cv2.imread(frame_path + name + '/' + now_frame + '.png')

                    if not os.path.exists(frame_path + name + '/' + future_frame + '.png'):
                        continue

                    rgb_future_frame = cv2.imread(frame_path + name + '/' + future_frame + '.png')

                    if rgb_now_frame is None or rgb_future_frame is None: continue

                    flow = inference_model(model, rgb_now_frame, rgb_future_frame)
                    flow = flow.astype(np.float16)
                    np.savez_compressed(rgb_now_flow_file, flow)
                else:
                    tenOne = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(frame_path+name+'/'+now_frame+'.png'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
                    tenTwo = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(frame_path+name+'/'+future_frame+'.png'))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

                    tenOutput = estimate(tenOne, tenTwo)

                    flow = tenOutput.cpu().numpy().astype(np.float16)
                    np.savez_compressed(rgb_now_flow_file, flow)


def depth(offset=10):
    pass
    # 5. generate depth now+future

def cleanup_frame_cache(offset):
    with open(cache_path + f'{offset}_{dataset}_modified_frame_cache.pkl', 'rb') as handle:
        paired_frames = pickle.load(handle)
    #paired_frames = paired_frames[159000:]

    size = (576, 1024) if dataset == 'epickitchens' else (648, 864) 
    clean_frame_cache = []
    start = time.time()

    data = MiniDataset(paired_frames, offset)
    bs = 500
    loader = DataLoader(data, batch_size=bs, num_workers=10)

    for i, batch in enumerate(loader):
        if i % 10 == 0: 
            print(f'{i*bs} / {(i*bs)-len(clean_frame_cache)} / {len(paired_frames)} / {time.time() - start}')

        for ii in range(batch['now'].shape[0]):
            now_data = batch['now'][ii]
            if now_data is None or now_data.max() == 0: continue

            future_data = batch['future'][ii]
            if future_data is None or future_data.max() == 0: continue

            clean_frame_cache.append(batch['frame'][ii])

        #if i > 100: 
        #    break

    '''
    for i, now_frame in enumerate(paired_frames):
        if i % 100 == 0: print(f'{i} / {i-len(clean_frame_cache)} / {len(paired_frames)} / {time.time() - start}')

        if dataset == 'epickitchens': 
            person, video, now_frame_num = now_frame.split('_')
            name = '_'.join([person, video])
        elif dataset == 'ego4d': 
            name, now_frame_num = now_frame.split('_')

        future_frame_num = str(int(now_frame_num) + offset)
        future_frame = '_'.join([name, str(future_frame_num)])

        now_frame_path = (frame_path + name + '/' + now_frame + '.png')
        future_frame_path = (frame_path + name + '/' + future_frame + '.png')

        if os.path.exists(now_frame_path) and os.path.exists(future_frame_path):
            now_data = Image.open(now_frame_path)
            if now_data.size != tuple(list(size)[::-1]):
                print('skipped!')
                continue

            #if np.array(now_data).max() == 0: continue
            if now_data.getbbox() is None: continue

            future_data = Image.open(future_frame_path)
            if future_data.size != tuple(list(size)[::-1]):
                print('skipped!')
                continue

            #if np.array(future_data).max() == 0: continue
            if future_data.getbbox() is None: continue

            now_extrema = now_data.convert("L").getextrema()
            future_extrema = future_data.convert("L").getextrema()
            if now_extrema == (0, 0) or future_extrema == (0, 0):
                # all black
                continue

            now_image = cv2.imread(now_frame_path, 0)
            future_image = cv2.imread(future_frame_path, 0)
            now_gray_version = cv2.cvtColor(now_image, cv2.COLOR_BGR2GRAY)
            future_gray_version = cv2.cvtColor(future_image, cv2.COLOR_BGR2GRAY)
            if cv2.countNonZero(now_gray_version) == 0 or cv2.countNonzero(future_gray_version) == 0:
                #print("Error")
                continue
            else:
                pass
                #print("Image is fine")

            clean_frame_cache.append(now_frame)
        else:
            continue
    '''

    with open(cache_path + f'{offset}_{dataset}_clean_frame_cache.pkl', 'wb') as handle:
        pickle.dump(clean_frame_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

def remove_broken_frames(dataset):
    with open(f'/w/relh/{dataset}/cache/{10}_{dataset}_frame_cache.pkl', 'rb') as handle:
        pfc = pickle.load(handle)
    with open(f'/w/relh/{dataset}/cache/{10}_{dataset}_clean_frame_cache.pkl', 'rb') as handle:
        cpfc = pickle.load(handle)
    xpfc = list(set(pfc) - set(cpfc))
    print(len(pfc))
    print(len(cpfc))
    print(len(xpfc))
    main_path = '/z/relh/ego4d/'
    for i, x in enumerate(xpfc):
        if i % 100 == 0: print(f'{i} / {len(xpfc)}')

        name, num = x.split('_')
        frame = os.path.join(main_path, 'frames', name, x + '.png') 
        people = os.path.join(main_path, 'people', name, x + '.png') 
        flow = os.path.join(main_path, 'flow', str(offset), name, x + '.npz') 
        negflow = os.path.join(main_path, 'flow', str(-offset), name, x + '.npz') 
        if os.path.exists(frame):
            os.remove(frame)
        if os.path.exists(people):
            os.remove(people)
        if os.path.exists(flow):
            os.remove(flow)
        if os.path.exists(negflow):
            os.remove(negflow)

def find_mini_paired(videos, offset=10, split='val'):
    my_gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])

    metadata = json.load(open(f'/home/relh/annotated_ego4d/ego4d.json'))

    fho_hands = json.load(open(f'/home/relh/annotated_ego4d/fho_hands_{split}.json'))
    fho_scod = json.load(open(f'/home/relh/annotated_ego4d/fho_scod_{split}.json'))
    fho_sta = json.load(open(f'/home/relh/annotated_ego4d/fho_sta_{split}.json'))

    my_videos = list(videos)[my_gpu::7]
    my_videos = my_videos[::-1]

    mini_paired_frames = []
    for vvv in tqdm(my_videos):
        # want to find all the frames and their offset pairs we may care about
        my_video = vvv['video_uid']

        # my hand clips frames
        my_hands_clips = [x for x in fho_hands['clips'] if x['video_uid'] == my_video]
        to_process = []
        for mhc in my_hands_clips:
            for frame in mhc['frames']:
                to_process.append(frame['action_start_frame'])
                to_process.append(frame['action_end_frame'])
                # everything below here also boxes into right and left hand
                if 'pre_45' in frame:
                    to_process.append(frame['pre_45']['frame'])
                if 'pre_30' in frame:
                    to_process.append(frame['pre_30']['frame'])
                if 'pre_15' in frame:
                    to_process.append(frame['pre_15']['frame'])
                if 'pre_frame' in frame:
                    to_process.append(frame['pre_frame']['frame'])
                if 'post_frame' in frame:
                    to_process.append(frame['post_frame']['frame'])
                if 'pnr_frame' in frame:
                    to_process.append(frame['pnr_frame']['frame'])
                if 'contact_frame' in frame:
                    to_process.append(frame['contact_frame']['frame'])

        # my scod clips frames
        my_scod_clips = [x for x in fho_scod['clips'] if x['video_uid'] == my_video]
        for msc in my_scod_clips:
            if 'post_frame' in msc:
                to_process.append(msc['post_frame']['frame_number'])
            if 'pnr_frame' in msc:
                to_process.append(msc['pnr_frame']['frame_number'])
            if 'contact_frame' in msc:
                to_process.append(msc['contact_frame']['frame_number'])

        # my sta clips frames
        my_sta_clips = [x for x in fho_sta['annotations'] if x['video_id'] == my_video]
        for mstac in my_sta_clips:
            to_process.append(mstac['frame'])

        # now have list of frames from video we want
        to_process = list(set(to_process))
        print(f'to_process: {len(to_process)}')

        v = vvv['video_uid']
        this_video_path = vvv['video_uid'] + '.mp4' 

        for index in to_process:
            mini_paired_frames.append(v + '_' + str(index))

    return mini_paired_frames


def build_paired_frame_cache(offset=10):
    paired_frames = []

    videos = os.listdir(frame_path)
    for v in videos:
        this_frame_path = frame_path + v + '/'
        frames = os.listdir(this_frame_path)
        print(f'{this_frame_path}.. {len(frames)}')

        for frame in frames:
            frame_num = int(frame.split('_')[-1].split('.png')[0])
            future_frame = frame.replace(str(frame_num) + '.png', str(frame_num + offset) + '.png')
            if os.path.exists(this_frame_path + future_frame):
                paired_frames.append(frame.split('.')[0]) # sans extension

    with open(cache_path + f'{offset}_{dataset}_modified_frame_cache.pkl', 'wb') as handle:
        pickle.dump(paired_frames, handle, protocol=pickle.HIGHEST_PROTOCOL)

def build_ego4d_images(split='train'):
    metadata = json.load(open(f'/home/relh/annotated_ego4d/ego4d.json'))
    videos = list([x for x in metadata['videos']])# if x['split_fho'] == 'val'])# or x['split_fho'] == 'multi'])

    fho_hands = json.load(open(f'/home/relh/annotated_ego4d/fho_hands_{split}.json'))
    fho_scod = json.load(open(f'/home/relh/annotated_ego4d/fho_scod_{split}.json'))
    fho_sta = json.load(open(f'/home/relh/annotated_ego4d/fho_sta_{split}.json'))

    my_videos = list(videos)
    new_frame_path = f'/home/relh/VISOR-HOS/datasets/ego4d_box2seg/{split}/'

    full_paired_frames = []
    new_paths = []
    for vvv in tqdm(my_videos):
        # want to find all the frames and their offset pairs we may care about
        my_video = vvv['video_uid']

        # my hand clips frames
        my_hands_clips = [x for x in fho_hands['clips'] if x['video_uid'] == my_video]
        to_process = []
        for mhc in my_hands_clips:
            for frame in mhc['frames']:
                to_process.append(frame['action_start_frame'])
                to_process.append(frame['action_end_frame'])
                # everything below here also boxes into right and left hand
                if 'pre_45' in frame:
                    to_process.append(frame['pre_45']['frame'])
                if 'pre_30' in frame:
                    to_process.append(frame['pre_30']['frame'])
                if 'pre_15' in frame:
                    to_process.append(frame['pre_15']['frame'])
                if 'pre_frame' in frame:
                    to_process.append(frame['pre_frame']['frame'])
                if 'post_frame' in frame:
                    to_process.append(frame['post_frame']['frame'])
                if 'pnr_frame' in frame:
                    to_process.append(frame['pnr_frame']['frame'])
                if 'contact_frame' in frame:
                    to_process.append(frame['contact_frame']['frame'])

        # my scod clips frames
        my_scod_clips = [x for x in fho_scod['clips'] if x['video_uid'] == my_video]
        for msc in my_scod_clips:
            if 'post_frame' in msc:
                to_process.append(msc['post_frame']['frame_number'])
            if 'pnr_frame' in msc:
                to_process.append(msc['pnr_frame']['frame_number'])
            if 'contact_frame' in msc:
                to_process.append(msc['contact_frame']['frame_number'])

        # my sta clips frames
        my_sta_clips = [x for x in fho_sta['annotations'] if x['video_id'] == my_video]
        for mstac in my_sta_clips:
            to_process.append(mstac['frame'])

        # now have list of frames from video we want
        to_process = list(set(to_process))
        print(f'to_process: {len(to_process)}')

        v = vvv['video_uid']
        this_video_path = vvv['video_uid'] + '.mp4' 

        mini_paired_frames = []
        for index in to_process:
            mini_paired_frames.append(v + '_' + str(index) + '.png')

        this_frame_path = frame_path + v + '/'
        for mpf in mini_paired_frames:
            if os.path.exists(this_frame_path + mpf):
                try:
                    this_image = np.array(Image.open(this_frame_path + mpf))
                except:
                    continue
                if this_image is None or this_image.max() == 0: continue

                full_paired_frames.append(this_frame_path + mpf)
                new_paths.append(new_frame_path + mpf)

        print(len(new_paths))
        #if len(full_paired_frames) > 0:
        #    pdb.set_trace()

        # TODO
        # get annotations from this then use

    for source, dest in zip(full_paired_frames, new_paths):
        shutil.copyfile(source, dest)

def process_fho_seg(frame):
    to_process = frame['frame']
    boxes = frame['boxes']
    return to_process

def build_annotated_frame_cache(split='train'):
    metadata = json.load(open(f'/home/relh/annotated_ego4d/ego4d.json'))
    videos = list([x for x in metadata['videos']])# if x['split_fho'] == 'val'])# or x['split_fho'] == 'multi'])

    fho_hands = json.load(open(f'/home/relh/annotated_ego4d/fho_hands_{split}.json'))
    fho_scod = json.load(open(f'/home/relh/annotated_ego4d/fho_scod_{split}.json'))
    fho_sta = json.load(open(f'/home/relh/annotated_ego4d/fho_sta_{split}.json'))

    my_videos = list(videos)
    new_frame_path = f'/home/relh/VISOR-HOS/datasets/ego4d_box2seg/{split}/'

    full_paired_frames = []
    annotations = defaultdict(list)
    for zzz, vvv in enumerate(tqdm(my_videos)):
        # want to find all the frames and their offset pairs we may care about
        my_video = vvv['video_uid']

        # my hand clips frames
        '''
        my_hands_clips = [x for x in fho_hands['clips'] if x['video_uid'] == my_video]
        for mhc in my_hands_clips:
            for frame in mhc['frames']:
                #to_process.append(frame['action_start_frame'])
                #to_process.append(frame['action_end_frame'])
                # everything below here also boxes into right and left hand
                for key in ['pre_45', 'pre_30', 'pre_15', 'pre_frame', 'post_frame', 'pnr_frame', 'contact_frame']:
                    if key in frame:
                        frame_id = my_video + '_' + str(frame[key]['frame'])
                        if 'boxes' in frame[key]:
                            nono_annotations[frame_id].append(frame[key]['boxes'])
        '''

        # my scod clips frames
        my_scod_clips = [x for x in fho_scod['clips'] if x['video_uid'] == my_video]
        for msc in my_scod_clips:
            for key in ['pre_frame', 'pnr_frame', 'post_frame']:
                if key in msc:
                    frame_id = my_video + '_' + str(msc[key]['frame_number'])
                    if 'bbox' in msc[key]:
                        for bbox in msc[key]['bbox']:
                            this_box = [bbox['bbox']['x'], bbox['bbox']['y'], bbox['bbox']['width'], bbox['bbox']['height']]
                            this_box = [int(x * (648 / 1080.0)) for x in this_box]
                            bbox['bbox'] = this_box
                            annotations[frame_id].append(bbox)

        # my sta clips frames
        my_sta_clips = [x for x in fho_sta['annotations'] if x['video_id'] == my_video]
        for mstac in my_sta_clips:
            frame_id = my_video + '_' + str(mstac['frame'])
            if 'objects' in mstac:
                for obj in mstac['objects']:
                    # 1080, 1440
                    this_box = obj['box']
                    this_box = [x * (648 / 1080.0) for x in this_box]
                    if this_box[0] > this_box[2]:
                        pdb.set_trace()
                    if this_box[1] > this_box[3]:
                        pdb.set_trace()
                    obj['bbox'] = [this_box[0], this_box[1], this_box[2]-this_box[0], this_box[3]-this_box[1]]
                    annotations[frame_id].append(obj)

    my_json = {'images': [], 'annotations': []}

    for image, all_anno in tqdm(annotations.items()):
        real_path = frame_path + image.split('_')[0] + '/' + image + '.png'
        if not os.path.exists(real_path):
            continue
        try:
            this_image = np.array(Image.open(real_path))
        except:
            continue
        if this_image is None or this_image.max() == 0: continue

        my_json['images'].append({'id': image, 'height': 648, 'file_name': image + '.png'})

        for anno in all_anno:
            anno['image_id'] =  image
            this_box = anno['bbox']
            this_box = [int(x) for x in this_box]
            this_box[0] = min(max(this_box[0], 0), 864)
            this_box[1] = min(max(this_box[1], 0), 648)
            this_box[2] = min(max(this_box[2], 2), 864-this_box[0])
            this_box[3] = min(max(this_box[3], 2), 648-this_box[1])
            anno['bbox'] = this_box
            my_json['annotations'].append(anno)

    with open(f'/home/relh/VISOR-HOS/datasets/ego4d_box2seg/annotations/{split}_boxes_only_filtered_v3.json', 'w') as j: json.dump(my_json, j)


def vis_annotated_frame_cache(split='train'):
    to_vis_json = json.load(open(f'/home/relh/VISOR-HOS/datasets/ego4d_box2seg/annotations/{split}_congeal_max_box2seg_READY_v6.json'))

    # 648, 864
    with open('/home/relh/public_html/VISOR-HOS/my_vis_product/index.html', 'w') as f:
        f.write('<html><head></head><body>')
        all_images = to_vis_json['images'][:5000]
        random.shuffle(all_images)
        for fff, image in enumerate(tqdm(all_images)):
            #f.write(f'<img style="width: 10uw;" src="ego4d_vis/{split}/{image["file_name"]}">')
            img = cv2.imread(f'/home/relh/public_html/VISOR-HOS/my_vis_product/ego4d_vis/{split}/{image["file_name"]}')
            this_annotations = [(jjj, x) for (jjj, x) in enumerate(to_vis_json['annotations']) if x['image_id'] == image['id']]

            bbox_img = img
            img_segs = img
            for image_id, anno in this_annotations:
                my_box = anno['bbox']
                my_segments = anno['segmentation']
                if 'object_type' in anno: continue

                my_box = [x for x in my_box]
                #mask = torch.zeros((648, 864)).bool()
                #mask[my_box[1]:my_box[1]+my_box[3], my_box[0]:my_box[0]+my_box[2]] = True
                #contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                bbox_img = cv2.rectangle(bbox_img, (my_box[0], my_box[1]), (my_box[0] + my_box[2], my_box[1] + my_box[3]), (255,0,0), 2)
                #img_bbox = cv2.polylines(img, [my_box], True, (255,120,255), 3)
                #bbox_write_path = f'/home/relh/public_html/VISOR-HOS/my_vis_product/modified/{split}/bbox_{image["file_name"]}'
                #cv2.imwrite(bbox_write_path, bbox_img)

                my_new_segments = []
                for poly in my_segments:
                    my_new_segments.append([x for x in poly])

                img_segs = img
                i = 0
                for poly in my_new_segments:
                    i += 1
                    points = []
                    for x,y in zip(poly[::2], poly[1::2]):
                        points.append((x,y))
                    #points.append(points[0])
                    points = np.array(points).astype(np.int32)
                    points = points.reshape((-1, 1, 2))
                    img_segs = cv2.polylines(img_segs, [points], True, (0,0,255), 3)

                #img_segs = cv2.polylines(img, my_new_segments, True, (255,120,255), 3)
                cv2.imwrite(f'/home/relh/public_html/VISOR-HOS/my_vis_product/modified/{split}/segs_{image["file_name"]}', img_segs)

            #f.write(f'<img src="modified/{split}/bbox_{image["file_name"]}">')
            f.write(f'<img src="modified/{split}/segs_{image["file_name"]}">')
            #f.write('<br>')
            if fff > 500: break
        f.write('</body></html>')


def fix_annotated_frame_cache(split='train'):
    to_fix_json = json.load(open(f'/home/relh/VISOR-HOS/datasets/ego4d_box2seg/annotations/{split}_congeal_max_box2seg_READY_v5.json'))
    #to_fix_json = json.load(open(f'/home/relh/VISOR-HOS/datasets/ego4d_box2seg/annotations/{split}_congeal_max_box2seg_UPDATED.json'))
    #example_json = json.load(open(f'/home/relh/VISOR-HOS/datasets/epick_visor_coco_hos/annotations/{split}.json'))

    #fho_hands = json.load(open(f'/home/relh/annotated_ego4d/fho_hands_{split}.json'))
    #fho_scod = json.load(open(f'/home/relh/annotated_ego4d/fho_scod_{split}.json'))
    '''
    fho_sta = json.load(open(f'/home/relh/annotated_ego4d/fho_sta_{split}.json'))

    to_fix_json['info'] = {
            "year": 2022,
            "version": 1.0,
            "description": 'ego4d coco', 
            "contributor": 'relh' ,
            "url": 'none' ,
            "date_created": '11/10' 
    }

    to_fix_json['categories'] = [
        {
            "id": 1,
            "name": 'hand',
        },
        {
            "id": 2,
            "name": 'object',
        },
    ]


    #to_fix_json['categories'] = {}
    all_categories = []
    for jjj, anno in enumerate(tqdm(to_fix_json['annotations'])):
        to_fix_json['annotations'][jjj]['id'] = jjj+1
        if 'object_type' in anno:
            all_categories.append(anno['object_type'])
            if anno['object_type'] == 'left_hand' or anno['object_type'] == 'right_hand':
                to_fix_json['annotations'][jjj]['category_id'] = 1
            else:
                to_fix_json['annotations'][jjj]['category_id'] = 2
        if 'noun_category_id' in anno:
            my_type = anno['noun_category_id']
            my_name = [x['name'] for x in fho_sta['noun_categories'] if x['id'] == my_type]
            if len(my_name) > 0:
                my_name = my_name[0]
                all_categories.append(my_name)
            if my_name == 'left_hand' or my_name == 'right_hand':
                to_fix_json['annotations'][jjj]['category_id'] = 1
            else:
                to_fix_json['annotations'][jjj]['category_id'] = 2
        to_fix_json['annotations'][jjj]['area'] = 9.0
        to_fix_json['annotations'][jjj]['iscrowd'] = 0
        to_fix_json['annotations'][jjj]['attributes'] = {"occluded": 0}
    all_categories = list(set(all_categories))
    '''

    all_images = to_fix_json['images']
    for jjj, image in enumerate(all_images):
        wanted_path = '/home/relh/VISOR-HOS/datasets/ego4d_box2seg/train/' + image['file_name']
        original_path = frame_path + image['id'].split('_')[0] + '/' + image['file_name']
        if not os.path.exists(wanted_path):
            if os.path.exists(original_path):
                print(f'copied! {original_path} to {wanted_path}')
                shutil.copyfile(original_path, wanted_path)
            else:
                pdb.set_trace()
        #image['width'] = 864
        #to_fix_json['images'][jjj] = image

    '''
    broken = 0
    for jjj, anno in enumerate(tqdm(to_fix_json['annotations'])):
        if 'segmentation' not in anno or len(anno['segmentation']) == 0:
            broken += 1
            print(jjj)
            box = anno['bbox']
            if box[2] <= 1:
                box[0] = max(box[0] - 2, 0)
                box[2] = 2
            if box[3] <= 1:
                box[1] = max(box[1] - 2, 0)
                box[3] = 2
            best_box_segments = torch.zeros((1, 648, 864)).bool().cuda()
            best_box_segments[:, box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = True 
            bbs = best_box_segments.permute(1,2,0).cpu().numpy()
            contours, hierarchy = cv2.findContours(bbs.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            segmentation = []
            for contour in contours:
                # Valid polygons have >= 6 coordinates (3 points)
                if contour.size >= 6:
                    segmentation.append(contour.flatten().tolist())

            resized_segmentation = []
            for seg in segmentation:
                resized_segmentation.append([x for x in seg])

            if len(resized_segmentation) == 0:
                pdb.set_trace()

            to_fix_json['annotations'][jjj]['segmentation'] = resized_segmentation

    print(broken / len(to_fix_json['annotations']))
    '''

    #random.shuffle(all_images)
    # 648, 864
    #with open(f'/home/relh/VISOR-HOS/datasets/ego4d_box2seg/annotations/{split}_congeal_max_box2seg_READY_v5.json', 'w') as j: json.dump(to_fix_json, j)

    ego_4d_train = json.load(open(f'/home/relh/VISOR-HOS/datasets/ego4d_box2seg/annotations/train_congeal_max_box2seg_READY_v5.json'))

    '''
    for iii, image in enumerate(ego_4d_train['images']):
        print(f'{iii} / {len(ego_4d_train["images"])}')
        real_id = image['id'] 

        this_annotations = [(jjj, x) for (jjj, x) in enumerate(ego_4d_train['annotations']) if x['image_id'] == real_id]

        for jjj, anno in this_annotations:
            anno['image_id'] = iii+1
            ego_4d_train['annotations'][jjj] = anno

        image['id'] = iii+1
        ego_4d_train['images'][iii] = image
    '''

    with open(f'/home/relh/VISOR-HOS/datasets/ego4d_box2seg/annotations/train_congeal_max_box2seg_READY_v6.json', 'w') as j: json.dump(ego_4d_train, j)

def build_ego4d_val_json(root_path, split='val'):
    to_fix_json = {}
    to_fix_json['info'] = {
            "year": 2022,
            "version": 1.0,
            "description": 'ego4d coco val', 
            "contributor": 'relh' ,
            "url": 'none' ,
            "date_created": '11/11' 
    }
    to_fix_json['categories'] = [
        {
            "id": 1,
            "name": 'hand',
        },
        {
            "id": 2,
            "name": 'object',
        },
    ]
    to_fix_json['images'] = []
    to_fix_json['annotations'] = []

    paths = [root_path + x for x in ['test_indomain/image/', 'test_outdomain/image/', 'val/image/']] 
    # 648, 864
    ego4d_images = []
    ego4d_labels = []
    new_paths = []
    for path in paths:
        image_names = os.listdir(path)
        ego4d_images += [path + x for x in image_names if 'ego4d' in x]
        my_path = path.replace('image', 'label')
        ego4d_labels += [my_path + x.replace('.jpg', '.png') for x in image_names if 'ego4d' in x]
        new_paths += ['/home/relh/VISOR-HOS/datasets/ego4d_eval/val/' + x for x in image_names if 'ego4d' in x]

    jjj = 0 
    for iii, (source_image, target_image, annotation) in enumerate(zip(ego4d_images, new_paths, ego4d_labels)):
        print(f'{iii} / {len(ego4d_labels)}')
        image_name = source_image.split('/')[-1].split('.')[0]
        to_fix_json['images'].append({'id': iii+1, 'height': 648, 'width': 864, 'file_name': image_name + '.jpg'})

        # resize image here
        this_image = Image.open(source_image)
        new_image = this_image.resize((864,648), Image.ANTIALIAS)
        #this_image.thumbnail((864, 648))
        new_image.save(target_image)

        '''
        this_label = Image.open(annotation)
        this_label = np.array(this_label)
        for index in [1,2,3,4,5]:
            this_mask = np.array(this_label == index)
            try:
                rect = masks_to_boxes(einops.repeat(torch.tensor(this_mask), 'h w -> b h w', b=1)).int()[0]
            except:
                rect = [1, 1, 1919, 1439]
            rect[2] = rect[2] - rect[0]
            rect[3] = rect[3] - rect[1]
            rect = [int(x * (648 / this_label.shape[0])) for x in rect]

            contours, hierarchy = cv2.findContours(this_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            segmentation = []
            for contour in contours:
                # Valid polygons have >= 6 coordinates (3 points)
                if contour.size >= 6:
                    segmentation.append(contour.flatten().tolist())

            resized_segmentation = []
            for seg in segmentation:
                resized_segmentation.append([x * (648 / this_label.shape[0]) for x in seg])


            if len(resized_segmentation) == 0: continue

            jjj += 1
            to_fix_json['annotations'].append({'id': jjj, 'image_id': iii+1, 'area': 9.0, 'iscrowd': 0, 'attributes': {'occluded': 0}})
            #to_fix_json['annotations'][jjj]['id'] = jjj+1
            if index == 1 or index == 2:
                to_fix_json['annotations'][-1]['category_id'] = 1
            else:
                to_fix_json['annotations'][-1]['category_id'] = 2
            to_fix_json['annotations'][-1]['segmentation'] = resized_segmentation
            to_fix_json['annotations'][-1]['bbox'] = rect 
        
    with open(f'/home/relh/VISOR-HOS/datasets/ego4d_eval/annotations/val_v2.json', 'w') as j: json.dump(to_fix_json, j)
    '''

if __name__ == "__main__":
    #pass
    #build_annotated_frame_cache(split='train')
    #vis_annotated_frame_cache(split='val')
    #fix_annotated_frame_cache(split='train')

    #build_annotated_frame_cache(split='train')
    #annotations = json.load(open(f'/home/relh/annotated_ego4d/ego4d.json'))
    #all_videos = list([x for x in annotations['videos']])
    #test_videos = list([x for x in annotations['videos'] if x['split_fho'] == 'test' or x['split_fho'] == 'multi'])
    #zzz = [x['video_uid'] for x in all_videos]
    #pdb.set_trace()
    #videos = [x for i, x in enumerate(os.listdir(video_path))]
    #random.shuffle(videos)
    #frames_and_people(all_videos)
    #valid_videos = list([x for x in annotations['videos'] if x['split_fho'] == 'val'])# or x['split_fho'] == 'multi'])
    #frames_and_people(valid_videos, split='val')
    #train_videos = list([x for x in annotations['videos'] if x['split_fho'] == 'train'])
    #frames_and_people(train_videos, split='train')
    my_gpu = int(os.environ["CUDA_VISIBLE_DEVICES"])

    #root_path = '/home/relh/EgoHOS/data/'
    #build_ego4d_val_json(root_path)
    #vis_annotated_frame_cache()
    #ego_4d_val = json.load(open(f'/home/relh/VISOR-HOS/datasets/ego4d_eval/annotations/val_v1.json'))
    '''
    file_list = glob.glob(os.path.join(frame_path, 'model_split_*', '*')) # glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-9]*', '*[0-8]')) #+ \
    file_list = [x + '/frame0.png' for x in file_list]

    with open(cache_path + f'{1}_{dataset}_frame_cache.pkl', 'wb') as handle:
        pickle.dump(file_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    '''

    offsets = [20, 30]#, 60, 120, 300]
    for split in ['valid', 'train']:
        for offset in offsets:
            #with open(cache_path + f'{offset}_{dataset}_frame_cache.pkl', 'rb') as handle:
            #    paired_frames = pickle.load(handle)
            with open(f'./util/10_{dataset}_{split}_frames.pkl', 'rb') as handle:
                paired_frames = pickle.load(handle)
            my_paired_frames = paired_frames[my_gpu::7]
            my_paired_frames = my_paired_frames[::-1]
            #my_paired_frames = my_paired_frames[:len(my_paired_frames) // 2]
            random.shuffle(my_paired_frames)
            print('starting flow..')
            flow(my_paired_frames, offset, neg=False)

    for split in ['valid', 'train']:
        for offset in offsets:
            #with open(cache_path + f'{offset}_{dataset}_frame_cache.pkl', 'rb') as handle:
            #    paired_frames = pickle.load(handle)
            with open(f'./util/10_{dataset}_{split}_frames.pkl', 'rb') as handle:
                paired_frames = pickle.load(handle)
            my_paired_frames = paired_frames[my_gpu::7]
            my_paired_frames = my_paired_frames[::-1]
            #my_paired_frames = my_paired_frames[:len(my_paired_frames) // 2]
            random.shuffle(my_paired_frames)
            print('starting flow..')
            flow(my_paired_frames, offset, neg=True)

    #my_paired_frames = paired_frames[4-2::6]
    #flow(my_paired_frames, offset, neg=True)
    #build_paired_frame_cache(offset)
    #cache(videos, offset)
    #pdb.set_trace()
    #filter_cache(offset)
    #mpf = find_mini_paired(valid_videos) 
    #flow(mpf, offset, neg=True)
    #mpf = find_mini_paired(train_videos) 
    #flow(mpf, offset, neg=False)
    #mpf = find_mini_paired(train_videos) 
    #depth(offset)
    #refine_cache(offset)
    #merge_cache(offset)
    #cleanup_frame_cache(offset)
