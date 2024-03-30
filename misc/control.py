#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from diffusers import (ControlNetModel, DPMSolverMultistepScheduler,
                       StableDiffusionControlNetInpaintPipeline)

# Assuming these imports are done and necessary packages are installed

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:2] == image_mask.shape[0:2], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

# Initialize models and pipeline once
controlnet_inpaint = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", controlnet=controlnet_inpaint, torch_dtype=torch.float16
)

pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.enable_model_cpu_offload()

def generate_outpainted_image(init_image, mask_image, prompt="indoor kitchen scene"):
    inpaint_condition_image = make_inpaint_condition(init_image, mask_image)
    
    # Generate image
    generated_img = pipe(
        prompt=prompt,
        negative_prompt=None,
        num_inference_steps=20,
        guidance_scale=6.0,
        eta=1.0,
        image=init_image,
        mask_image=mask_image,
        control_image=inpaint_condition_image,
    ).images[0]

    return generated_img
