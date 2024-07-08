# inpainting_pipeline.py
import math
import random

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers import (AutoPipelineForInpainting, DiffusionPipeline,
                       IFInpaintingPipeline,
                       IFInpaintingSuperResolutionPipeline,
                       StableDiffusionInpaintPipeline)
from diffusers.utils import pt_to_pil
from einops import rearrange, repeat
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from PIL import Image
from torchvision.transforms.functional import pad
from transformers import T5EncoderModel

#torch.set_default_dtype(torch.float32)
#torch.set_default_device('cuda')

# Global variable for the pipeline
pipeline = None

def initialize_pipeline(model):
    global pipeline

    if model == 'sd2': 
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

    elif model == 'if':
        text_encoder = T5EncoderModel.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0",
            subfolder="text_encoder", 
            load_in_8bit=True, 
            variant="8bit"
        )

        pipeline = IFInpaintingPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0", 
            text_encoder=text_encoder, 
            unet=None, 
        )

        sr_pipeline = IFInpaintingSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0", 
            text_encoder=None, 
            variant="fp16", 
            torch_dtype=torch.float16, 
        )

# Function to run inpainting pipeline
def run_inpaint(image, mask_image, prompt, model, guidance_scale=25.0):
    global pipeline
    if pipeline is None:
        initialize_pipeline(model)
        #seed = 12887 #78631 
        seed = 78631 
    else:
        seed = random.randint(0, 99999)
    strength = 1.0
    print(f'seed is.. {seed}')

    generator = torch.Generator(device="cuda").manual_seed(seed)
    pipe_image = Image.fromarray((image.cpu().numpy()).astype(np.uint8))
    pipe_mask = Image.fromarray(((mask_image * 255.0).cpu().numpy()).astype(np.uint8), 'L')

    if model == 'sd2':
        print(guidance_scale)
        output = pipeline(
          prompt=prompt,
          image=pipe_image,
          mask_image=pipe_mask,
          strength=strength,  
          generator=generator,
          guidance_scale=guidance_scale,
          num_inference_steps=30,  # steps between 15 and 30 work well for us
        )
    elif model == 'if':
        prompt_embeds, negative_embeds = pipeline.encode_prompt(prompt)

        output = pipeline(
            image=pipe_image,
            mask_image=pipe_mask,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds, 
            #output_type="pt",
            generator=generator,
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
