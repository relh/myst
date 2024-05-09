# inpainting_pipeline.py
import math
import random

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForInpainting, StableDiffusionInpaintPipeline
from einops import rearrange, repeat
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from PIL import Image
from torchvision.transforms.functional import pad

#torch.set_default_dtype(torch.float32)
#torch.set_default_device('cuda')

# Global variable for the pipeline
pipeline = None

def initialize_pipeline():
    global pipeline

    #pipeline = AutoPipelineForInpainting.from_pretrained(
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        #"runwayml/stable-diffusion-inpainting", 
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16#, variant="fp16"
    ).to("cuda")
    pipeline.initial_strength = 1.0
    pipeline.next_strength = 1.0

    #pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline = pipeline.to("cuda")


# Function to run inpainting pipeline
def run_inpaint(image: Image, mask_image: Image, prompt: str):
    global pipeline
    if pipeline is None:
        initialize_pipeline()
        strength = pipeline.initial_strength
        #seed = 78631 
    else:
        strength = pipeline.next_strength 
    seed = random.randint(0, 99999)
    print(f'seed is.. {seed}')
    generator = torch.Generator(device="cuda").manual_seed(seed)

    pipe_image = Image.fromarray((image.cpu().numpy()).astype(np.uint8))
    pipe_mask = Image.fromarray(((mask_image.sum(dim=2) / 3.0 * 255.0).cpu().numpy()).astype(np.uint8), 'L')

    output = pipeline(
      prompt=prompt,
      image=pipe_image,
      mask_image=pipe_mask,
      strength=strength,  
      generator=generator,
      guidance_scale=7.0,
      num_inference_steps=50,  # steps between 15 and 30 work well for us
    )

    return torch.tensor(np.array(output.images[0])).float().to('cuda')

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
