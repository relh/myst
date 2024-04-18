#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import sys

import numpy as np
import PIL.Image
from PIL import Image
import torch
import torchvision.transforms as tvf
from PIL.ImageOps import exif_transpose

sys.path.append('dust3r/')

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference, load_model

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dust_model = None

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)


def load_images(color_image, size, square_ok=True):
    imgs = []
    for image in color_image:
        img = exif_transpose(image)
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W//2, H//2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

        W2, H2 = img.size
        print(f' - adding image with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    print(f' (Found {len(imgs)} images)')
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    return imgs


def img_to_pts_3d_dust(color_image):
    global dust_model
    device = 'cuda'
    batch_size = 1
    get_focals = False
    if dust_model is None:
        model_path = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        dust_model = load_model(model_path, device)
        get_focals = True

    color_image = [Image.fromarray(image.cpu().numpy()) for image in color_image]
    images = load_images(color_image, size=512)

    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, dust_model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    #view1, pred1 = output['view1'], output['pred1']
    #view2, pred2 = output['view2'], output['pred2']

    # retrieve useful values from scene:
    mode = GlobalAlignerMode.PointCloudOptimizer if len(color_image) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)
    loss = scene.compute_global_alignment(init='mst', niter=100, schedule='cosine', lr=0.01)

    print(f'loss: {loss}')
    if loss != loss: 
        pass
        breakpoint()
    else:
        pts3d = scene.get_pts3d()[0]
    #pts3d = output[f'pred1']['pts3d'][0].to(device)
    #imgs = scene.imgs
    #confidence_masks = scene.get_masks()

    focals = None
    if get_focals:
        focals = scene.get_focals()[0]
        #scene.get_im_poses()[0]

    if len(color_image) >= 2:
        # TODO stick to paired image setup
        #breakpoint()
        pass

    return torch.tensor(pts3d.reshape(-1, 3) * 1000.0), \
           torch.tensor(np.asarray(color_image[0]).reshape(-1, 3)).to('cuda'), focals


if __name__ == '__main__':
    model_path = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model = load_model(model_path, device)
    # load_images can take a list of images or a directory
    images = load_images(['dust3r/croco/assets/Chateau1.png'], size=512)
    imgs = [imgs[0], copy.deepcopy(imgs[0])]
    imgs[1]['idx'] = 1
    breakpoint()
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    #view1, pred1 = output['view1'], output['pred1']
    #view2, pred2 = output['view2'], output['pred2']

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # visualize reconstruction
    #scene.show()
    breakpoint()
