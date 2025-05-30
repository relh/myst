#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
from io import BytesIO

import numpy as np
import PIL
import requests
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForInpainting, StableDiffusionUpscalePipeline
from einops import rearrange, repeat
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from PIL import Image
from torchvision.transforms.functional import pad

#torch.set_default_dtype(torch.float32)
#torch.set_default_device('cuda')

# Global variable for the pipeline
pipeline = None

def supersample_point_cloud(point_cloud):
    """
    Supersample a point cloud by interpolating points between adjacent points in both x and y directions.

    :param point_cloud: Input point cloud as a PyTorch tensor of shape (height, width, 3).
    :return: Supersampled point cloud.
    """
    height, width, _ = point_cloud.shape
    
    # Interpolate along x-axis
    interpolated_x = (point_cloud[:, :-1, :] + point_cloud[:, 1:, :]) / 2
    
    # Concatenate the original and the interpolated points along x-axis
    supersampled_x = torch.empty(height, 2 * width - 1, 3)
    supersampled_x[:, 0::2, :] = point_cloud
    supersampled_x[:, 1::2, :] = interpolated_x

    # Interpolate along y-axis
    interpolated_y = (supersampled_x[:-1, :, :] + supersampled_x[1:, :, :]) / 2
    
    # Concatenate the original and the interpolated points along y-axis
    supersampled_xy = torch.empty(2 * height - 1, 2 * width - 1, 3)
    supersampled_xy[0::2, :, :] = supersampled_x
    supersampled_xy[1::2, :, :] = interpolated_y

    return supersampled_xy.to('cuda')


def tensor_to_square_pil(image, mask, zoom=(456.0 / 512.0)):
    image_tensor = rearrange(image / 255.0, 'h w c -> 1 c h w').to(torch.float32)
    mask_tensor = rearrange(mask, 'h w -> 1 1 h w').to(torch.float32)

    # Calculate zoom and get affine transformation
    _, c, h, w = image_tensor.shape
    center = torch.tensor([w / 2, h / 2]).unsqueeze(0)
    zoom_tensor = torch.tensor([zoom, zoom]).unsqueeze(0)
    translate = torch.tensor([0.0, 0.0]).unsqueeze(0)
    angle = torch.tensor([0.0])

    M = get_affine_matrix2d(center=center, angle=angle, scale=zoom_tensor, translations=translate).to(image_tensor.device)

    # Apply affine transformation to zoom out
    image_tensor = warp_affine(image_tensor, M[:, :2], dsize=(math.ceil(h / zoom), math.ceil(w / zoom)), padding_mode="border")
    mask_tensor = F.interpolate(mask_tensor, size=(math.ceil(h / zoom), math.ceil(w / zoom)), mode='nearest')

    # Determine new size to make the image square and calculate padding
    new_h, new_w = image_tensor.shape[2:]
    pad_h = (max(new_h, new_w) - new_h) // 2
    pad_w = (max(new_h, new_w) - new_w) // 2

    # Apply padding to maintain border effect
    padded_image = pad(image_tensor, [pad_w, pad_h, pad_w, pad_h], padding_mode="edge")
    padded_mask = pad(mask_tensor, [pad_w, pad_h, pad_w, pad_h], padding_mode="constant", fill=1)

    padded_image = rearrange(padded_image, '1 c h w -> h w c').cpu().numpy()
    padded_mask = rearrange(padded_mask, '1 1 h w -> h w').cpu().numpy()

    padded_image = (padded_image * 255).astype(np.uint8)
    padded_mask = (padded_mask * 255).astype(np.uint8)

    output_image = Image.fromarray(padded_image)
    output_mask = Image.fromarray(padded_mask, 'L')  # 'L' mode for grayscale mask

    return output_image, output_mask, pad_h, pad_w


def initialize_pipeline():
    global pipeline

    # load model and scheduler
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16,
    )
    pipeline.vae.scaling_factor=0.08333

    #pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline = pipeline.to("cuda")


# Function to run inpainting pipeline
def run_supersample(image: Image, mask_image: Image, prompt: str):
    global pipeline
    if pipeline is None:
        initialize_pipeline()
    seed = random.randint(0, 99999)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    image, mask_image, pad_h, pad_w = tensor_to_square_pil(image, mask_image, zoom=1.0)
    output = pipeline(
        prompt=prompt, 
        image=image,
        num_inference_steps=10,
    )

    output = repeat(torch.tensor(np.array(output.images[0])).float().to('cuda'), 'h w c -> 1 c h w')
    output = F.interpolate(output, size=(512, 512), mode='nearest')
    output = output[:, :, pad_h:(None if pad_h == 0 else -pad_h), pad_w:(None if pad_w == 0 else -pad_w)]
    return rearrange(output, '1 c h w -> h w c').to(torch.uint8)


# You might want to initialize the pipeline when the script is imported
# But it can also be lazily initialized on the first call to run_inpainting_pipeline
# initialize_pipeline()  # Uncomment this if you want to initialize on import

# Example usage
if __name__ == "__main__":
    prompt = "A girl sitting in a library full of books"
    input_url = "https://www.mauritshuis.nl/media/wlola5to/0670_repro_2.jpg?center=0.42060129980199951,0.47243107769423559&mode=crop&width=480&rnd=133443375703200000&quality=70"
    init_image = Image.open(PIL.Requests.get(input_url, stream=True).raw).resize((512, 512))
    conditioning_image, outpaint_mask = create_outpainting_image_and_mask(init_image, 0.5)

    output_image = run_inpainting_pipeline(conditioning_image, outpaint_mask, prompt)
    output_image.show()  # Or save it with output_image.save('output_path.jpg')
