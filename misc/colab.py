# inpainting_pipeline.py
import numpy as np
import PIL
import einops
import torch
import torch.nn.functional as F
from diffusers import (AutoPipelineForInpainting,
                       StableDiffusionXLInpaintPipeline)
from einops import rearrange, repeat
from kornia.geometry.transform import (get_affine_matrix2d,
                                       get_rotation_matrix2d, warp_affine)
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms.functional import pad, to_pil_image

# Global variable for the pipeline
pipeline = None

def create_outpainting_image_and_mask(image, mask, zoom):
    # Convert the PIL image to a PyTorch tensor
    image_tensor = torch.tensor(rearrange(image / 255.0, 'h w c -> 1 c h w'), dtype=torch.float32)
    '''
    _, c, h, w = image_tensor.shape
    # Calculate zoom and get affine transformation
    center = torch.tensor([w / 2, h / 2]).unsqueeze(0)
    zoom_tensor = torch.tensor([zoom, zoom]).unsqueeze(0)
    translate = torch.tensor([0.0, 0.0]).unsqueeze(0)
    angle = torch.tensor([0.0])

    M = get_affine_matrix2d(center=center, angle=angle, scale=zoom_tensor, translations=translate)

    # Apply affine transformation to zoom out
    transformed_image_tensor = warp_affine(image_tensor, M[:, :2], dsize=(h, w), padding_mode="border")

    #sq_img, sq_mask, pad_h, pad_w = create_outpainting_image_and_mask(wombo_img, wombo_mask, zoom=1.00)
    #diffused_img = sq_init[pad_h:(None if pad_h == 0 else -pad_h),\
    #                       pad_w:(None if pad_w == 0 else -pad_w)]
    '''

    # Determine new size to make the image square and calculate padding
    new_h, new_w = image_tensor.shape[2:]
    pad_h = (max(new_h, new_w) - new_h) // 2
    pad_w = (max(new_h, new_w) - new_w) // 2

    # Apply padding to maintain border effect
    padded_image_tensor = pad(image_tensor, [pad_w, pad_h, pad_w, pad_h], padding_mode="edge")
    padded_mask = pad(mask, [pad_w, pad_h, pad_w, pad_h], mode="constant", value=1)

    padded_image_array = rearrange(padded_image_tensor, '1 c h w -> h w c').cpu().numpy()
    padded_image_array = (padded_image_array * 255).astype(np.uint8)
    mask_array = (padded_mask.cpu().numpy() * 255).astype(np.uint8)

    output_image = Image.fromarray(padded_image_array)
    output_mask = Image.fromarray(mask_array, 'L')  # 'L' mode for grayscale mask

    return output_image, output_mask, pad_h, pad_w

def initialize_pipeline():
    global pipeline
    '''
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    torch_dtype=torch.float16, variant='fp16')
    '''

    pipeline = AutoPipelineForInpainting.from_pretrained(\
                 "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",\
                 torch_dtype=torch.float16, variant="fp16").to("cuda")

    pipeline.enable_model_cpu_offload()

# Function to run inpainting pipeline
def run_inpainting_pipeline(image: Image, mask_image: Image, prompt: str, seed: int = 12345, strength: float = 1.0):
    global pipeline
    if pipeline is None:
        initialize_pipeline()

    #generator = torch.Generator().manual_seed(seed)
    generator = torch.Generator(device="cuda").manual_seed(0)

    output = pipeline(
      prompt=prompt,
      image=Image.fromarray(image.cpu().numpy().astype(np.uint8)), #, 'h w c -> c h w'),
      mask_image=Image.fromarray((mask_image * 255.0).cpu().numpy().astype(np.uint8), 'L'),
      guidance_scale=8.0,
      num_inference_steps=15,  # steps between 15 and 30 work well for us
      strength=strength,  # make sure to use `strength` below 1.0
      generator=generator,
    )

    '''
    output = pipeline(
        prompt,
        image=image,
        mask_image=mask_image,
        height=456,
        width=456,
        generator=generator,
        strength=strength
    )
    '''
    img = repeat(torch.tensor(np.array(output.images[0])).float().to('cuda'), 'h w c -> 1 c h w')
    return rearrange(F.interpolate(img, size=(256, 456), mode='bilinear', align_corners=False), '1 c h w -> h w c').to(torch.uint8)


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
