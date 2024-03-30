#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import json
import logging
import os
import pdb
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image


class Coherence_Dataset(Dataset):
    def __init__(self, pfc, dataset, data_path, img_size, train=False):
        self.pfc = pfc
        self.frame_path = data_path + 'frames/'
        self.flow_path = data_path + 'flow/'
        self.people_path = data_path + 'people/'

        # data transform
        self.transform = []
        self.transform = transforms.Compose(self.transform + [
          transforms.ToTensor(),
          transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
          ),
        ])

        self.dataset = dataset
        self.img_size = img_size
        self.pfc = [x for (i, x) in enumerate(pfc)] # if i % 8 == b][::-1]
        self.offset = 10 
        self.people = False 

    def __len__(self):
        return len(self.pfc)

    def __getitem__(self, idx):
        now_frame = self.pfc[idx]
        if self.dataset == 'epickitchens': 
            person, video, now_frame_num = now_frame.split('_')
            name = '_'.join([person, video])
        elif self.dataset == 'ego4d': 
            name, now_frame_num = now_frame.split('_')
        elif self.dataset == 'playroom':
            name = '/'.join(now_frame.split('/')[-2:])
            now_frame_num = 0
            now_frame = 'frame0'

        future_frame_num = str(int(now_frame_num) + self.offset)

        if self.dataset == 'playroom':
            future_frame = 'frame1'
        else:
            future_frame = '_'.join([name, str(future_frame_num)])

        try:
            returner = {
              #'name': name,
              #'idx': idx,
              'now': {
                'frame': (name + '/frame0.png') if self.dataset == 'playroom' else now_frame,
                'rgb': self.transform(Image.open(self.frame_path + name + '/' + now_frame + '.png')),
                'flow_n_f': np.load(os.path.join(self.flow_path, str(self.offset), name, now_frame + '.npz'), allow_pickle=True, mmap_mode='r')['arr_0'],
                'people': (np.array(Image.open(self.people_path + name + '/' + now_frame + '.png')) if self.people else np.zeros(self.img_size)),
              },
              'future': {
                'frame': (name + '/frame1.png') if self.dataset == 'playroom' else future_frame,
                'rgb': self.transform(Image.open(self.frame_path + name + '/' + future_frame + '.png')),
                'flow_f_n': np.load(os.path.join(self.flow_path, str(-self.offset), name, future_frame + '.npz'), allow_pickle=True, mmap_mode='r')['arr_0'],
                'people': (np.array(Image.open(self.people_path + name + '/' + future_frame + '.png')) if self.people else np.zeros(self.img_size)),
              },
            }
            return returner
        except Exception as e:
            print(e)
            print(f'bad index! {idx} {self.pfc[idx]}')
            return self[random.randint(0, len(self))]


class EgoHOSDataset(Dataset):
    def __init__(self, pfc, offset):
        self.pfc = pfc
        self.offset = offset
        self.frame_path = '/z/relh/ego4d/frames/'
        self.dataset = 'ego4d'

    def __len__(self):
        return len(self.pfc)

    def __getitem__(self, idx):
        now_frame = self.pfc[idx]
        name, now_frame_num = now_frame.split('_')

        future_frame_num = str(int(now_frame_num) + self.offset)
        future_frame = '_'.join([name, str(future_frame_num)])

        try:
            now_data = np.array(Image.open(self.frame_path + name + '/' + now_frame + '.png'))
            future_data = np.array(Image.open(self.frame_path + name + '/' + future_frame + '.png'))
            return {'frame': now_frame, 'now': now_data, 'future': future_data}
        except:
            return {'frame': str(0), 'now': np.zeros((648, 864, 3), dtype=np.byte), 'future': np.zeros((648, 864, 3), dtype=np.byte)}


class MiniDataset(Dataset):
    def __init__(self, pfc, offset):
        self.pfc = pfc
        self.offset = offset
        self.frame_path = '/z/relh/ego4d/frames/'
        self.dataset = 'ego4d'

    def __len__(self):
        return len(self.pfc)

    def __getitem__(self, idx):
        now_frame = self.pfc[idx]
        name, now_frame_num = now_frame.split('_')

        future_frame_num = str(int(now_frame_num) + self.offset)
        future_frame = '_'.join([name, str(future_frame_num)])

        try:
            now_data = np.array(Image.open(self.frame_path + name + '/' + now_frame + '.png'))
            future_data = np.array(Image.open(self.frame_path + name + '/' + future_frame + '.png'))
            return {'frame': now_frame, 'now': now_data, 'future': future_data}
        except:
            return {'frame': str(0), 'now': np.zeros((648, 864, 3), dtype=np.byte), 'future': np.zeros((648, 864, 3), dtype=np.byte)}


class PlayroomDataset(Dataset):
    def __init__(self, training, args, frame_idx=0, dataset_dir = '/z/relh/playroom'):

        self.training = training
        self.frame_idx = frame_idx
        #self.args = args

        # meta.json is only required for TDW datasets
        meta_path = os.path.join(dataset_dir, 'meta.json')
        self.meta = json.loads(Path(meta_path).open().read())

        if self.training:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'frames', 'model_split_*', '*[0-8]'))
        else:
            self.file_list = sorted(glob.glob(os.path.join(dataset_dir, 'frames', 'model_split_[0-3]', '*9')))

        #if args.precompute_flow: # precompute flows for training and validation dataset
        #    self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_*', '*')) # glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-9]*', '*[0-8]')) #+ \


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        frame_idx = self.frame_idx if os.path.exists(self.get_image_path(file_name, self.frame_idx)) else 0
        img1 = read_image(self.get_image_path(file_name, frame_idx))


        flag = os.path.exists(self.get_image_path(file_name, self.frame_idx+1))
        img2 = read_image(self.get_image_path(file_name, frame_idx+1)) if flag else img1
        segment_colors = read_image(self.get_image_path(file_name.replace('/frames/', '/segments/'), frame_idx))
        gt_segment = self.process_segmentation_color(segment_colors, file_name)

        ret = {'img1': img1, 'img2': img2, 'gt_segment': gt_segment}

        #if not self.args.compute_flow and not self.args.precompute_flow:
        #    flow_path = os.path.join(file_name.replace('/images/', '/flows/'), f'frame{frame_idx}.npy')
        #    flow = np.load(flow_path)
        #    magnitude = torch.tensor((flow ** 2).sum(0, keepdims=True) ** 0.5)
        #    segment_target = (magnitude > self.args.flow_threshold)
        #    ret['segment_target'] = segment_target
        #elif self.args.precompute_flow:
        ret['file_name'] = self.get_image_path(file_name, frame_idx)

        return ret

    @staticmethod
    def get_image_path(file_name, frame_idx):
        return os.path.join(file_name, f'frame{frame_idx}' + '.png')

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]
        return out

    def process_segmentation_color(self, seg_color, file_name):
        # convert segmentation color to integer segment id
        raw_segment_map = self._object_id_hash(seg_color, val=256, dtype=torch.long)
        raw_segment_map = raw_segment_map.squeeze(0)

        # remove zone id from the raw_segment_map
        meta_key = 'playroom_large_v3_images/' + file_name.split('/images/')[-1] + '.hdf5'
        zone_id = int(self.meta[meta_key.replace('/frames/', '/images/')]['zone'])
        raw_segment_map[raw_segment_map == zone_id] = 0

        # convert raw segment ids to a range in [0, n]
        _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
        segment_map -= segment_map.min()

        return segment_map


def fetch_dataloader(args, training=True, drop_last=True):
    """ Create the data loader for the corresponding trainign set """
    if args.dataset == 'playroom':
        dataset = PlayroomDataset(training=training, args=args)
    else:
        raise ValueError(f'Expect dataset in [playroom], but got {args.dataset} instead')

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            pin_memory=False,
                            shuffle=training,
                            num_workers=args.num_workers,
                            drop_last=drop_last)

    logging.info(f"Load dataset [{args.dataset}-{'train' if training else 'val'}] with {len(dataset)} image pairs")
    return dataloader

if __name__ == "__main__":
    #ds = PlayroomDataset(training=False, args=None)
    ds = Coherence_Dataset(training=False, args=None)

    pdb.set_trace()
