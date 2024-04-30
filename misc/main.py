#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

import argparse
import datetime
import getpass
import glob
import json
import os
import pickle
import random
import socket
import sys
import traceback
import warnings
from pathlib import Path

import submitit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from cohesiv import COHESIV
from dataset import Coherence_Dataset
from mmcv.runner import build_optimizer
from models import resnet
from models.N3F.feature_extractor.lib.baselines import get_model
from models.segmentation_pytorch.configs.segformer_config import config as cfg
from models.segmentation_pytorch.utils.lr_schedule import PolyLrUpdater
from simseg import SimSeg
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

warnings.filterwarnings("ignore")
random.seed(30)

# pdb post mortem always drops in debugger on crash
#pdb.post_mortem(tb) # more "modern"

def setup(rank, args):
    args.global_rank = rank

    #if args.debug:
    #    print("LETS LOG THIS")
    #    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    #    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    # --- initialize ddp ---
    # process_group needs to know real rank
    # mp only sees 1 ranks, needs local rank of 0
    # slurm sees all ranks, needs local rank = real rank

    if args.slurm_partition is not None:
        dist_env = submitit.helpers.TorchDistributedEnvironment().export()
        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(f"rank: {dist_env.rank}")
        print(f"world size: {dist_env.world_size}")
        print(f"local rank: {dist_env.local_rank}")
        print(f"local world size: {dist_env.local_world_size}")
        os.environ['MASTER_ADDR'] = str(dist_env.master_addr)
        os.environ['MASTER_PORT'] = str(dist_env.master_port)
        #os.environ['NCCL_P2P_DISABLE'] = str(1)
        args.global_rank = dist_env.rank 
        dist.init_process_group(backend='nccl', rank=args.global_rank, world_size=args.num_gpus, init_method='env://')
    elif args.ddp:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = args.port
        os.environ['NCCL_P2P_DISABLE'] = str(1)
        start_rank = int(os.environ["CUDA_VISIBLE_DEVICES"].split(',')[0]) # to avoid ruoyuw
        rank += start_rank
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
        dist.init_process_group(backend='nccl', rank=args.global_rank, world_size=args.num_gpus, init_method='env://')
        #rank = 0 
    else:
        args.num_gpus = 1
        args.batch_size = 4

    # --- initialize network ---
    torch.cuda.set_device(rank)
    net_mtime = datetime.datetime.fromtimestamp(60*60*24.0)
    if 'cohesiv' in args.model:
        # woohoo made cohesiv-demo already
        net = COHESIV().cuda()

        if args.ddp:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = DistributedDataParallel(net, device_ids=[rank], output_device=rank) #, broadcast_buffers=False) #, find_unused_parameters=True)

        net = net.eval()
    elif 'resnet' in args.model:
        net = resnet.ResNet().cuda()
        net = net.eval()
    elif 'dino' in args.model:
        net = get_model('dino', './models/N3F/feature_extractor/ckpts/dino_vitbase8_pretrain.pth', f"cuda:0")
        #net = get_model('dino', './models/N3F/feature_extractor/ckpts/checkpoint.pth', f"cuda:0")
    elif 'rgb' in args.model:
        net = None
    elif 'grabcut' in args.model:
        net = None
    elif 'spatialgrid' in args.model:
        net = None
    else:
        try:
            net = SimSeg(args).cuda()
        except Exception as e:
            print(str(e))
            return 0.0 

        if args.ddp:
            net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = DistributedDataParallel(net, device_ids=[rank], output_device=rank) #, broadcast_buffers=False) #, find_unused_parameters=True)

    # --- load filtered paired frames ---
    if 'playroom' not in args.dataset:
        with open(f'{args.data_path}cache/{args.offset}_{args.dataset}_frame_cache.pkl', 'rb') as handle:
            pfc = pickle.load(handle)

    # --- split into train/val ---
    if not os.path.exists(f'./util/{args.offset}_{args.dataset}_train_frames.pkl'):
        if 'epickitchens' in args.dataset:

            valid_annotations = json.load(open(f'{args.anno_path}/Annotations/val_annotations.json'))
            valid_videos = sorted(list(set(['_'.join(x.split('_')[:2]) for x in valid_annotations.keys()])))

            train_frames = [x for x in pfc if '_'.join(x.split('_')[:2]) not in valid_videos]
            valid_frames = [x for x in pfc if '_'.join(x.split('_')[:2]) in valid_videos]
        elif 'ego4d' in args.dataset:
            annotations = json.load(open(f'{args.anno_path}/ego4d.json'))
            train_videos = list([x['video_uid'] for x in annotations['videos'] if x['split_fho'] == 'train'])
            valid_videos = list([x['video_uid'] for x in annotations['videos'] if x['split_fho'] == 'val'])# or x['split_fho'] == 'multi'])
            test_videos = list([x['video_uid'] for x in annotations['videos'] if x['split_fho'] == 'test' or x['split_fho'] == 'multi'])

            train_frames = [x for x in pfc if x.split('_')[0] not in valid_videos and x not in test_videos]
            valid_frames = [x for x in pfc if x.split('_')[0] in valid_videos] 
        elif 'playroom' in args.dataset:
            train_frames = glob.glob(os.path.join(args.data_path, 'frames', 'model_split_*', '*[0-8]'))
            valid_frames = sorted(glob.glob(os.path.join(args.data_path, 'frames', 'model_split_[0-3]', '*9')))

        pickle.dump(train_frames, open(f'./util/{args.offset}_{args.dataset}_train_frames.pkl', 'wb'))
        pickle.dump(valid_frames, open(f'./util/{args.offset}_{args.dataset}_valid_frames.pkl', 'wb'))
    else:
        train_frames = pickle.load(open(f'./util/{args.offset}_{args.dataset}_train_frames.pkl', 'rb'))
        valid_frames = pickle.load(open(f'./util/{args.offset}_{args.dataset}_valid_frames.pkl', 'rb'))

        if False and args.visualize:
            from inference import hand_map
            valid_frames = [x for x in valid_frames if x in hand_map]

    # --- shuffle frames for visualization diversity ---
    random.Random(args.global_rank).shuffle(valid_frames)

    # --- make datasets, samplers depend on DDP ---
    sampler = lambda x, y: (DistributedSampler(x, shuffle=y) if args.ddp else SequentialSampler(x))
    train_data, valid_data = Coherence_Dataset(train_frames, args, train=True), Coherence_Dataset(valid_frames, args)
    train_sampler, valid_sampler = sampler(train_data, True), sampler(valid_data, False)

    # --- make dataloaders, workers depend on DDP ---
    train_loader = DataLoader(train_data, batch_size=args.batch_size // args.num_gpus, num_workers=0 if args.ddp else 2, sampler=train_sampler)#, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size // args.num_gpus, num_workers=0 if args.ddp else 2, sampler=valid_sampler)#, pin_memory=True)

    # --- load checkpoint ---
    def load(rank, net, compile=True):
        net_mtime = datetime.datetime.fromtimestamp(60*60*24.0)
        if net == None: return None, net_mtime
        checkpoints = list(Path(args.experiment_path + 'models/').rglob("*.ckpt"))
        #checkpoints = [x for x in checkpoints if time.time() - os.path.getmtime(x) > (24*11*60*60)] # get only checkpoints from last 48 hours
        #if len(checkpoints) == 0:
        #    checkpoints = list(Path(args.experiment_path + 'models/').rglob("*.ckpt"))
        if len(checkpoints) > 0:
            if rank == 0: print(f'checkpoints! {checkpoints}')
            checkpoints.sort(key=lambda x: float(str(x).split('/')[-1].split('_')[0]))
            #checkpoints.sort(key=lambda x: -float(str(os.path.getmtime(x)))) 
            print(os.path.getmtime(checkpoints[0]))
            model_path = os.path.join(args.experiment_path + 'models/', checkpoints[0])
            net_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(model_path))
            load_dict = torch.load(model_path, map_location={'cuda:%d' % 0: 'cuda:%d' % rank})

            try:
                if 'cohesiv' in args.model:
                    net.load_state_dict(load_dict['model_state_dict'])
                else:
                    net.load_state_dict(load_dict['model'])
            except Exception as e:
                print(f'failed to load model0! {str(checkpoints[0])}')
                print(str(e))
                load_dict_model = load_dict['model']
                for key in list(load_dict_model.keys()):
                    if 'simsiam' not in key:
                        load_dict_model[key.replace('module.', 'module.backbone.')] = load_dict_model.pop(key)
                try:
                    net.load_state_dict(load_dict_model)
                except Exception as e:
                    print(f'failed to load model1! {str(checkpoints[0])}')
                    print(str(e))
                    for key in list(load_dict_model.keys()):
                        if ('1.weight' in key or '4.weight' in key or '7.weight' in key) and 'simsiam' in key:
                            new_key = key.replace('weight', 'g')
                            value = load_dict_model.pop(key)
                            load_dict_model[new_key] = value.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                        if ('1.bias' in key or '4.bias' in key or '7.bias' in key) and 'simsiam' in key:
                            new_key = key.replace('bias', 'b')
                            value = load_dict_model.pop(key)
                            load_dict_model[new_key] = value.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                    try:
                        net.load_state_dict(load_dict_model)
                    except Exception as e:
                        print(f'failed to load model2! {str(checkpoints[0])}')
                        print(str(e))
                        return net
            if rank == 0: print(f'loaded model! {str(checkpoints[0])}')
        #if compile:
        #    net = torch.compile(net)
        return net, net_mtime

    # --- train model ---
    if args.train:
        from train import run_epoch
        print('training..')
        if args.load: net, net_mtime = load(rank, net, compile=False)

        # --- setup optimizer ---
        #if args.finetune:
        # only finetune association head
        #if 'aa' in args.target:   optimizer = torch.optim.AdamW(net.module.compare_assoc.parameters(), betas=(0.9, 0.95), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
        #elif 'dd' in args.target: optimizer = torch.optim.AdamW(net.module.compare_nn.parameters(), betas=(0.9, 0.95), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
        #else:                     optimizer = torch.optim.AdamW(net.compare_nn.parameters(), betas=(0.9, 0.95), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
        #else:
        if False and 'smoothformer' in args.model:
            optimizer_cfg = dict(
                type=cfg.TRAIN.OPTIMIZER, 
                lr=cfg.TRAIN.BASE_LR,  
                weight_decay=cfg.TRAIN.WD,
                paramwise_cfg=dict(
                    custom_keys={
                        'pos_block': dict(decay_mult=0.), 
                        'norm': dict(decay_mult=0.),
                        'head': dict(lr_mult=10.)
                    }
            ))

            optimizer = build_optimizer(net, optimizer_cfg)

            scheduler = PolyLrUpdater(
                optimizer = optimizer,
                power = cfg.TRAIN.POWER,
                min_lr = cfg.TRAIN.MIN_LR,
                max_iters = cfg.TRAIN.DECAY_STEPS,
                epoch_len = len(train_loader),
                warmup = cfg.TRAIN.WARMUP,
                warmup_iters = cfg.TRAIN.WARMUP_ITERS,
                warmup_ratio = cfg.TRAIN.WARMUP_RATIO,
                by_epoch = cfg.TRAIN.BY_EPOCH,
            )
        else:
            scheduler = None
            optimizer = torch.optim.AdamW(net.parameters(), betas=(0.9, 0.95), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
        rlrop = ReduceLROnPlateau(optimizer, 'min', patience=int(args.early_stopping / 2), factor=0.5)
        #if not args.debug: net = torch.compile(net)
        scaler = GradScaler()
        train_losses, valid_losses = [], []
        min_loss = sys.maxsize
        failed_epochs = 0

        try:
            for epoch in range(args.num_epoch):
                if args.ddp:
                    train_sampler.set_epoch(epoch)
                    valid_sampler.set_epoch(epoch)

                train_loss = run_epoch(train_loader, net, scaler, optimizer, scheduler, epoch, args)
                valid_loss = run_epoch(valid_loader, net, None, None, None, epoch, args, is_train=False)

                if args.ddp:
                    train_loss = torch.as_tensor(train_loss).cuda()
                    valid_loss = torch.as_tensor(valid_loss).cuda()

                    dist.all_reduce(train_loss)
                    dist.all_reduce(valid_loss)

                    train_loss = float(train_loss) / args.num_gpus
                    valid_loss = float(valid_loss) / args.num_gpus

                rlrop.step(valid_loss)
                print(valid_loss)

                if valid_loss == 0.0: break

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                if valid_losses[-1] < min_loss:
                    min_loss = valid_losses[-1]
                    failed_epochs = 0
                    model_out_path = args.experiment_path + 'models/' + '_'.join([str(float(min_loss)), str(epoch), 'model']) + '.ckpt'
                    if args.global_rank == 0 and not args.debug:
                        torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict()}, model_out_path)
                        print('saved model..\t\t\t {}'.format(model_out_path))
                else:
                    failed_epochs += 1
                    print('--> loss failed to decrease {} epochs..\t\t\tthreshold is {}, {} all..{}'.format(failed_epochs, args.early_stopping, valid_losses, min_loss))

                # --- load best model every epoch ---
                #if args.load: net, net_mtime = load(rank, net)
                if failed_epochs > args.early_stopping: break
        except KeyboardInterrupt:
            pass

    # --- gather here for calibration / testing / evaluation / visualization ---
    try:
        if args.load: net, net_mtime = load(rank, net)
    except Exception as e:
        print(str(e))
        return 0.0

    torch.set_grad_enabled(False)

    if args.dataset == 'epickitchens': 
        train_annotations = json.load(open(f'{args.anno_path}/Annotations/train_annotations.json'))
        valid_annotations = json.load(open(f'{args.anno_path}/Annotations/val_annotations.json'))
    elif args.dataset == 'ego4d': 
        # todo
        train_annotations = train_frames 
        valid_annotations = [x for x in os.listdir(f'{args.anno_path}/VISOR-HOS/datasets/ego4d_eval/val')]
    elif args.dataset == 'playroom': 
        train_annotations = train_frames 
        valid_annotations = valid_frames #[x for x in os.listdir('/z/relh/playroom/segments/')]

    try:
        # --- calibrate ---
        cls_avg_embed, association_threshold = None, 0.5
        if args.calibrate: 
            if args.cohesiv:
                from calibrate import cohesiv_calibrate 
                print('calibration cohesiv..')
                cohesiv_calibrate(train_annotations, net, args, args.experiment_path + f'evaluation/{args.dataset}/')
            if args.visor:
                from calibrate import visor_calibrate 
                print('calibration visor..')
                cls_avg_embed = visor_calibrate(train_annotations, net, args, args.experiment_path + f'evaluation/{args.dataset}/')
                print(cls_avg_embed.keys())

        # --- testing ---
        visor_scores, visor_entities = {'all': 0.0, 'thresh': 0.0, 'argmax': 0.0, 'hands': 0.0, 'held': 0.0, 'active': 0.0}, {'auroc': {}}
        if args.test: 
            if args.cohesiv:
                from test import cohesiv_test
                print('testing cohesiv..')
                cohesiv_test(net, args, args.experiment_path + f'evaluation/{args.dataset}/cohesiv/')

            if args.visor:
                from test import visor_test 
                print('testing visor..')
                visor_scores, visor_entities = visor_test(valid_annotations, net, cls_avg_embed, args, args.experiment_path + f'evaluation/{args.dataset}/visor/')

        # --- genBox2Seg ---
        if args.genBox2Seg and args.global_rank == 0: 
            from coco_from_box2seg import genBox2Seg 
            print('generating box2Seg annotations..')
            genBox2Seg(net, args, args.experiment_path + f'evaluation/{args.dataset}/box2seg/')

        # --- evaluate ---
        if args.evaluate: 
            cohesiv_scores = None
            if args.cohesiv and args.global_rank == 0:
                from evaluate import cohesiv_eval
                print('evaluating cohesiv..')
                cohesiv_scores = cohesiv_eval(args, net_mtime, args.experiment_path + f'evaluation/{args.dataset}/cohesiv/')

            if args.visor:
                from evaluate import visor_eval 
                print('evaluating visor..')
                visor_scores = visor_eval(valid_annotations, cls_avg_embed, visor_scores, visor_entities, args, args.experiment_path + f'evaluation/{args.dataset}/visor/')

            print(cohesiv_scores)
            print(visor_scores)

        # --- visualize ---
        if args.visualize: 
            from train import run_epoch
            print('visualizing..')
            valid_loss = run_epoch(valid_loader, net, None, None, None, 999, args, is_train=False, visualize=True)

        # --- make figures ---
        if args.inference: 
            print('inference..')
            from cuml.cluster import HDBSCAN
            clust = HDBSCAN(min_samples=args.cluster_min_samples, min_cluster_size=args.cluster_min_size)#, cluster_selection_method='leaf')
            inference(net, clust, args)

    except KeyboardInterrupt:
        sys.exit()

    if args.ddp:
        dist.barrier()
        dist.destroy_process_group()
    print('ending')
    return 0.0

if __name__ == "__main__":
    os.nice(20)
    #os.nice(0)
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', dest='train', action='store_true', help='whether to train')
    parser.add_argument('--test', dest='test', action='store_true', help='whether to test')
    parser.add_argument('--calibrate', dest='calibrate', action='store_true', help='whether to calibrate')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='whether to evaluate')
    parser.add_argument('--visualize', dest='visualize', action='store_true', help='whether to visualize')
    parser.add_argument('--inference', dest='inference', action='store_true', help='whether to run inference')
    parser.add_argument('--write', dest='write', action='store_true', help='just write outputs')
    parser.add_argument('--cohesiv', dest='cohesiv', action='store_true', help='whether to perform cohesiv test/eval')
    parser.add_argument('--visor', dest='visor', action='store_true', help='whether to perform visor test/eval')
    parser.add_argument('--auroc', dest='auroc', action='store_true', help='whether to perform auroc testeval')
    parser.add_argument('--genBox2Seg', dest='genBox2Seg', action='store_true', help='whether to generate Box2Seg')

    # experiment parameters
    parser.add_argument('--name', type=str, default='demo', help='exp name')
    parser.add_argument('--target', type=str, default='cc', help='people / objects / background / attention')
    parser.add_argument('--decode', type=str, default='cluster', help='how to decode segments')
    parser.add_argument('--affinity', type=str, default=None, help='what kind of affinity')
    parser.add_argument('--dataset', type=str, default='epickitchens', help='doh / epickitchens / ho3d / ego4d / egohos / playroom')
    parser.add_argument('--model', type=str, default='hrnet', help='simseg / convnext / hrnet')
    parser.add_argument('--embed_dim', type=int, default=128, help='what size embedding')
    parser.add_argument('--offset', type=int, default=30, help='number of frames between samples')

    parser.add_argument('--people', dest='people', action='store_true', help='using people or not')
    parser.add_argument('--double', dest='double', action='store_true', help='using double hop')
    parser.add_argument('--triple', dest='triple', action='store_true', help='using triple hop')
    parser.add_argument('--merge', dest='merge', action='store_true', help='whether merge across time')
    parser.add_argument('--aug', dest='aug', action='store_true', help='using aug')
    parser.add_argument('--negbg', dest='negbg', action='store_true', help='using negatives in background')
    parser.add_argument('--rgb', dest='rgb', action='store_true', help='concat rgb to embedding')
    parser.add_argument('--xy', dest='xy', action='store_true', help='concat xy to embedding')
    parser.add_argument('--first_xy', dest='first_xy', action='store_true', help='whether to first add CoordConv then run')
    parser.add_argument('--norm', dest='norm', action='store_true', help='norm embeddings')

    # decoding parameters
    # 2, 100, 1.25, 5.0, 7.89 checkpoint
    # 2, 100, 1.15, 5.0, 7.89 checkpoint
    # 2, 90, 1.10, 4.0
    parser.add_argument('--cluster_min_samples', type=int, default=1, help='min core samples in a cluster')
    parser.add_argument('--cluster_min_size', type=int, default=90, help='min size of a cluster')
    parser.add_argument('--object_threshold', type=float, default=0.945, help='dynamic threshold to filter objects at')
    parser.add_argument('--first_interpolate', dest='first_interpolate', action='store_true', help='whether to first interpolate then cluster')
    parser.add_argument('--sharpness', type=float, default=4.0, help='cohesiv sharpness enhancement')

    # pseudolabel parameters
    parser.add_argument('--epipolar_log', dest='epipolar_log', action='store_true', help='epi log')
    parser.add_argument('--epipolar_threshold', type=float, default=3.3e-7, help='epi thresh')
    parser.add_argument('--ransac_threshold', type=float, default=3.3e-7, help='ransac thresh')
    parser.add_argument('--congealing_method', type=str, default='spatial', help='whether spatial or feature outlier congealing')
    parser.add_argument('--pseudolabels', type=str, default='MOVES', help='whether to use MOVES or CIS')
    parser.add_argument('--flow', type=str, default='gma', help='whether to use GMA or PWC')
    parser.add_argument('--motion_model', type=str, default='kornia', help='whether to use kornia or cv2')

    # optimization parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='what lr')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='what decay')
    parser.add_argument('--early_stopping', type=int, default=5, help='number of epochs before early stopping')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='whether to finetune')

    # machine parameters
    parser.add_argument('--load', dest='load', action='store_true', help='whether to load')
    parser.add_argument('--ddp', dest='ddp', action='store_true', help='whether to run ddp')
    parser.add_argument('--device', type=str, default='cuda:0', help='what GPU')
    parser.add_argument('--port', type=str, default='12489', help='what ddp port')
    parser.add_argument('--debug', dest='debug', action='store_true', help='whether debug mode')
    parser.add_argument('--k40', dest='k40', action='store_true', help='whether to use k40s')
    parser.add_argument('--bbox', dest='bbox', action='store_true', help='whether to use bbox to help cohesiv eval')
    parser.add_argument('--predPK', dest='predPK', action='store_true', help='whether to use predPK')

    # slurm parameters
    parser.add_argument('--num_gpus', type=int, default=None, help='num_gpus')

    # training parameters
    parser.add_argument('--num_epoch', type=int, default=1000, help='num_epoch')
    parser.add_argument('--epoch_len', type=int, default=5000, help='number of samples per epoch')
    parser.add_argument('--queue_len', type=int, default=500, help='number of outputs to write')

    parser.set_defaults(train=False)
    parser.set_defaults(test=False)
    parser.set_defaults(calibrate=False)
    parser.set_defaults(evaluate=False)
    parser.set_defaults(visualize=False)
    parser.set_defaults(inference=False)
    parser.set_defaults(cohesiv=False)
    parser.set_defaults(visor=False)
    parser.set_defaults(auroc=False)
    parser.set_defaults(genBox2Seg=False)
    #parser.set_defaults(write=True)

    parser.set_defaults(load=True)
    parser.set_defaults(ddp=False)
    parser.set_defaults(debug=False)
    parser.set_defaults(k40=False)
    parser.set_defaults(bbox=True)
    parser.set_defaults(predPK=False)

    parser.set_defaults(people=False)
    parser.set_defaults(double=False)
    parser.set_defaults(triple=False)
    parser.set_defaults(merge=True)
    parser.set_defaults(aug=True)
    parser.set_defaults(negbg=False)
    parser.set_defaults(rgb=False)
    parser.set_defaults(xy=False)
    parser.set_defaults(norm=True)

    parser.set_defaults(epipolar_log=False)
    parser.set_defaults(first_interpolate=True)
    parser.set_defaults(first_xy=False)
    args = parser.parse_args()

    if args.debug:
        args.num_epoch = 1
        args.epoch_len = 10
        args.queue_len = 10
        print(args)

    args.people = True if 'aa' in args.target else False

    args.name = '-'.join([args.name, args.model, str(args.target), str(args.offset), str(args.lr), 'aug' if args.aug else 'noaug'])
    args.name += '-rgb' if args.rgb else ''
    args.name += '-xy' if args.xy else ''

    if args.finetune:
        args.lr = args.lr * 0.01

    args.img_size = (576, 1024)
    args.embed_size = (144, 256)
    args.anno_size = (1080, 1920)
    args.fps_offset_mult = 1
    if 'ego4d' in args.dataset:
        args.img_size = (648, 864)
        args.embed_size = (162, 216)
        args.anno_size = (1440, 1920)
        args.fps_offset_mult = 2
        args.object_threshold = 0.88
    elif 'playroom' in args.dataset:
        #pdb.set_trace()
        args.img_size = (512, 512)
        args.embed_size = (128, 128)
        args.anno_size = (512, 512)
        args.fps_offset_mult = 1
        args.object_threshold = 0.88

    #args.epipolar_threshold = args.epipolar_threshold * args.offset * args.fps_offset_mult
    args.ransac_threshold = args.ransac_threshold * args.offset * args.fps_offset_mult

    # --- machine setup: directories, one output path for everything ---
    uniq = getpass.getuser()
    hostname = socket.gethostname()
    print(hostname)

    cpus_per_node = 40
    mem_per_node = 250
    if 'lh' in hostname and 'arc-ts.umich.edu' in hostname:
        args.experiment_path = f'/scratch/jiadeng_root/jiadeng/{uniq}/experiments/{args.name}/'
        args.data_path = '' # TODO
        args.slurm_partition = '' # TODO
        gpus_per_node = 8
        if args.num_gpus is None:
            args.num_gpus = 8
        args.batch_size = args.num_gpus * 2
    elif 'gl' in hostname and 'arc-ts.umich.edu' in hostname:
        args.experiment_path = f'/nfs/turbo/fouheyTemp/{uniq}/experiments/{args.name}/'
        args.data_path = f'/nfs/turbo/fouheyTemp/{uniq}/{args.dataset}/'
        args.anno_path = f'/nfs/turbo/fouheyTemp/{uniq}/annotated_{args.dataset}/'
        args.coco_path = f'/nfs/turbo/fouheyTemp/{uniq}/VISOR-HOS'
        args.slurm_partition = None #'spgpu'
        gpus_per_node = 8
        cpus_per_node = 32
        mem_per_node = 350
        if args.num_gpus is None:
            args.num_gpus = 8
        args.batch_size = args.num_gpus * 8
    elif 'vl-fb' in hostname or 'compute' in hostname:
        args.experiment_path = f'/home/{uniq}/experiments/{args.name}/'
        args.data_path = f'/home/{uniq}/{args.dataset}/'
        args.anno_path = f'/home/{uniq}/annotated_{args.dataset}/'
        args.coco_path = f'/home/{uniq}/VISOR-HOS'
        args.slurm_partition = 'vl-fb' if args.k40 else 'vl-fb-gtx1080'
        gpus_per_node = 4 if args.k40 else 10
        if args.num_gpus is None:
            args.num_gpus = 3 if args.k40 else 10
        args.batch_size = args.num_gpus * 2
    else:
        args.experiment_path = f'/y/{uniq}/experiments/{args.name}/'
        if 'epickitchens' in args.dataset: args.data_path = f'/x/{uniq}/{args.dataset}/'
        elif 'ego4d' in args.dataset:      args.data_path = f'/z/{uniq}/{args.dataset}/'
        elif 'playroom' in args.dataset:   args.data_path = f'/z/{uniq}/{args.dataset}/'
        args.anno_path = f'/home/{uniq}/annotated_{args.dataset}/'
        args.coco_path = f'/home/{uniq}/VISOR-HOS'
        args.slurm_partition = None
        gpus_per_node = 8 
        if args.num_gpus is None:
            args.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        args.batch_size = args.num_gpus * 2
    print(f'args.num_gpus: {args.num_gpus}')

    if args.debug:
        args.slurm_partition = None

    if args.dataset == 'playroom':
        args.batch_size *= 2

    if 'dino' in args.model: args.batch_size /= 2

    cpus_per_gpu = int(cpus_per_node * (args.num_gpus / gpus_per_node) / args.num_gpus)
    mem_gb = int(mem_per_node * (args.num_gpus / gpus_per_node))

    print(args.experiment_path)
    #args.dist_url = "file://" + str(args.experiment_path) + f"{uuid.uuid4().hex}_init"
    #print(args.dist_url)
    os.makedirs(args.experiment_path, exist_ok=True)
    os.makedirs(args.experiment_path + 'models/', exist_ok=True)
    os.makedirs(args.experiment_path + 'outputs/', exist_ok=True)
    os.makedirs(args.experiment_path + 'test_outputs/', exist_ok=True)
    os.makedirs(args.experiment_path + 'coco/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/', exist_ok=True)
    if args.affinity is not None and not os.path.exists(f'/home/{uniq}/public_html/affinities/{args.affinity}/'):
        os.makedirs(f'/home/{uniq}/public_html/affinities/{args.affinity}/')

    # visor folders
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/visor/left-hands/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/visor/right-hands/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/visor/pair-objects/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/visor/active-objects/', exist_ok=True)

    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/visor/rgb/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/visor/outs/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/visor/pca/', exist_ok=True)

    # cohesiv folders
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/cohesiv/hands/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/cohesiv/objects/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/cohesiv/pairs/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/cohesiv/all/', exist_ok=True)

    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/cohesiv/rgb/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/cohesiv/outs/', exist_ok=True)
    os.makedirs(args.experiment_path + f'evaluation/{args.dataset}/cohesiv/pca/', exist_ok=True)

    # box2seg folders

    if args.write:
        from write import write_index_html
        write_index_html(args)
        sys.exit()

    try:
        if args.slurm_partition is not None:
            executor = submitit.AutoExecutor(folder=f"logs/{args.name}")
            executor.update_parameters(
              nodes=1,
              tasks_per_node=args.num_gpus,
              cpus_per_task=cpus_per_gpu,
              mem_gb=mem_gb,
              timeout_min=(60*24*7) if args.train else (5*1*1),
              slurm_job_name=args.name,
              slurm_gres=f"gpu:{args.num_gpus}",
              slurm_gpus_per_task=1 if args.k40 or 'gl' in hostname else args.num_gpus,
              slurm_partition=args.slurm_partition)
            job = executor.submit(setup, 0, args)
            print(job.job_id) # ID of your job
            #job.results()
        elif args.ddp: 
            mp.spawn(setup, args=(args,), nprocs=args.num_gpus, daemon=True)
        else: 
            args.global_rank = 0 
            setup(0, args)
    except Exception as e:
        print(str(e))
        traceback.print_exception(*sys.exc_info())
        #mp.spawn(setup, args=(args,), nprocs=args.num_gpus)
