#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision.transforms import ToTensor, ToPILImage
from kornia.geometry.transform import get_affine_matrix2d, warp_affine


# Zooms out of a given image, and creates an outpainting mask for the external area.
def create_outpainting_image_and_mask(image, zoom):
    image_tensor = ToTensor()(image).unsqueeze(0)
    _, c, h, w = image_tensor.shape

    center = torch.tensor((h / 2, w / 2)).unsqueeze(0)

    zoom = torch.tensor([zoom, zoom]).unsqueeze(0)
    translate = torch.tensor((0.0, 0.0)).unsqueeze(0)
    angle = torch.tensor([0.0])

    M = get_affine_matrix2d(
        center=center, translations=translate, angle=angle, scale=zoom
    )

    mask_image_tensor = warp_affine(
        image_tensor,
        M=M[:, :2],
        dsize=image_tensor.shape[2:],
        padding_mode="fill",
        fill_value=-1*torch.ones(3),
    )
    mask = torch.where(mask_image_tensor < 0, 1.0, 0.0)

    transformed_image_tensor = warp_affine(
        image_tensor,
        M=M[:, :2],
        dsize=image_tensor.shape[2:],
        padding_mode="border"
    )

    output_mask = ToPILImage()(mask[0])
    output_image = ToPILImage()(transformed_image_tensor[0])

    return output_image, output_mask

from diffusers import StableDiffusionXLInpaintPipeline

MODEL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, variant='fp16')
pipe.enable_model_cpu_offload()

import PIL
from diffusers.utils import load_image

init_image = load_image(
    "https://www.mauritshuis.nl/media/wlola5to/0670_repro_2.jpg?center=0.42060129980199951,0.47243107769423559&mode=crop&width=480&rnd=133443375703200000&quality=70"
)
init_image = init_image.resize((512, 512))
conditioning_image, outpaint_mask = create_outpainting_image_and_mask(init_image, 0.5)

prompt = "A girl sitting in a library full of books"
generator = torch.Generator()
seed = 12345

output = pipe(
    prompt,
    image=conditioning_image,
    mask_image=outpaint_mask,
    height=1024,
    width=1024,
    generator=generator.manual_seed(seed),
)
output.images[0]
