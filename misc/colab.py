# inpainting_pipeline.py
import numpy as np
import PIL
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from einops import rearrange, repeat
from kornia.geometry.transform import (get_affine_matrix2d,
                                       get_rotation_matrix2d, warp_affine)
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

# Global variable for the pipeline
pipeline = None

def create_outpainting_image_and_mask(image: Image, zoom: float):
    # Convert PIL Image to tensor without normalization, in floating point
    image_np = np.array(image).astype(np.float32)
    image_tensor = rearrange(image_np, 'h w c -> 1 c h w')
    image_tensor = torch.from_numpy(image_tensor)

    _, c, h, w = image_tensor.shape
    new_dim = max(h, w)
    square_size = int(new_dim / zoom)  # Square canvas size after zoom

    # Center the image on the new canvas by calculating the translation
    translate_x = (square_size - w * zoom) / 2.0
    translate_y = (square_size - h * zoom) / 2.0

    # Prepare for affine transformation
    center = torch.tensor([w / 2, h / 2]).unsqueeze(0)
    scale = torch.tensor([zoom, zoom]).unsqueeze(0)
    translate = torch.tensor([translate_x, translate_y]).unsqueeze(0)
    angle = torch.tensor([0.0])

    # Calculate the affine transformation matrix
    M = get_affine_matrix2d(center=center, angle=angle, scale=scale, translations=translate)

    # Apply affine transformation with specific fill_value for inpainting areas
    transformed_image_tensor = warp_affine(image_tensor, M=M[:, :2], dsize=(square_size, square_size), 
                                           padding_mode="fill", fill_value=-1.0*torch.ones(3))

    # Creating mask: Identify areas with fill_value as needing inpainting
    mask = torch.where(transformed_image_tensor == -1.0, torch.ones_like(transformed_image_tensor), torch.zeros_like(transformed_image_tensor))

    # Convert the transformed image tensor back to a PIL Image, handling negative values appropriately
    transformed_image_np = rearrange(transformed_image_tensor, '1 c h w -> h w c').numpy()
    transformed_image_np = np.clip(transformed_image_np, 0, 255)  # Ensure values are within byte range
    transformed_image = Image.fromarray(transformed_image_np.astype(np.uint8))

    # Convert the mask tensor to a PIL Image, considering only one channel since all will be the same
    mask_np = rearrange(mask, '1 c h w -> h w c').squeeze().numpy()
    mask_image = Image.fromarray((mask_np * 255).astype(np.uint8))

    return transformed_image, mask_image

def initialize_pipeline():
    global pipeline
    MODEL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, variant='fp16')
    pipeline.enable_model_cpu_offload()

# Function to run inpainting pipeline
def run_inpainting_pipeline(input_image: Image, mask_image: Image, prompt: str, seed: int = 12345, strength: float = 1.0):
    global pipeline
    if pipeline is None:
        initialize_pipeline()

    generator = torch.Generator().manual_seed(seed)

    output = pipeline(
        prompt,
        image=input_image,
        mask_image=mask_image,
        height=512,
        width=512,
        generator=generator,
        strength=strength
    )
    return output.images[0]

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
