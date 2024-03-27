# inpainting_pipeline.py
import math
import random

import einops
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers import (AutoPipelineForInpainting, DPMSolverMultistepScheduler,
                       StableDiffusionPipeline,
                       StableDiffusionXLInpaintPipeline)
from einops import rearrange, repeat
from kornia.geometry.transform import (get_affine_matrix2d,
                                       get_rotation_matrix2d, warp_affine)
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms.functional import pad, to_pil_image

#torch.set_default_dtype(torch.float32)
#torch.set_default_device('cuda')

# Global variable for the pipeline
pipeline = None

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

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipeline.initial_strength = 1.0
    pipeline.next_strength = 1.0

    #pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    #pipeline.enable_xformers_memory_efficient_attention()
    pipeline = pipeline.to("cuda")


# Function to run inpainting pipeline
def run_inpaint(image: Image, mask_image: Image, prompt: str):
    global pipeline
    if pipeline is None:
        initialize_pipeline()
        strength = pipeline.initial_strength
        #seed = 50618
        seed = random.randint(0, 99999)
    else:
        strength = pipeline.next_strength 
        seed = random.randint(0, 99999)
    print(f'seed is.. {seed}')
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image, mask_image, pad_h, pad_w = tensor_to_square_pil(image, mask_image, zoom=1.0)
    #mask_image = pipeline.mask_processor.blur(mask_image, blur_factor=33)

    #'''
    output = pipeline(
      prompt=prompt,
      image=image,
      mask_image=mask_image,
      guidance_scale=8.0,
      num_inference_steps=25,  # steps between 15 and 30 work well for us
      strength=strength,  # make sure to use `strength` below 1.0
      generator=generator,
    )
    #'''
    '''
    output = pipeline(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        height=512,
        width=512,
        generator=generator,
        strength=strength
    )
    '''

    output_image = repeat(torch.tensor(np.array(output.images[0])).float().to('cuda'), 'h w c -> 1 c h w')
    output_image = F.interpolate(output_image, size=(512, 512), mode='nearest')
    output_image = output_image[:, :, pad_h:(None if pad_h == 0 else -pad_h), pad_w:(None if pad_w == 0 else -pad_w)]

    if pad_h > 0 and pad_w > 0:
        to_unpad_h = int((512 - 456) / 2)
        to_unpad_w = int((288 - 256) / 2)
        output_image = output_image[:, :, to_unpad_w:-to_unpad_w, to_unpad_h:-to_unpad_h]
    return rearrange(output_image, '1 c h w -> h w c').to(torch.uint8)


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
