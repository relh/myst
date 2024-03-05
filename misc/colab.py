# inpainting_pipeline.py
import PIL
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

# Global variable for the pipeline
pipeline = None

# Zooms out of a given image, and creates an outpainting mask for the external area.
def create_outpainting_image_and_mask(image, zoom):
    image_tensor = ToTensor()(image).unsqueeze(0)  # Convert PIL Image to tensor
    _, c, h, w = image_tensor.shape

    center = torch.tensor([h / 2, w / 2]).unsqueeze(0)
    zoom = torch.tensor([zoom, zoom]).unsqueeze(0)
    translate = torch.tensor([0.0, 0.0]).unsqueeze(0)
    angle = torch.tensor([0.0])

    M = get_affine_matrix2d(center=center, translations=translate, angle=angle, scale=zoom)

    mask_image_tensor = warp_affine(image_tensor, M=M[:, :2], dsize=(h, w), padding_mode="fill", fill_value=-1*torch.ones(3))
    mask = torch.where(mask_image_tensor < 0, 1.0, 0.0)

    transformed_image_tensor = warp_affine(image_tensor, M=M[:, :2], dsize=(h, w), padding_mode="border")

    output_mask = ToPILImage()(mask[0])
    output_image = ToPILImage()(transformed_image_tensor[0])

    return output_image, output_mask

def initialize_pipeline():
    global pipeline
    MODEL_NAME = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, variant='fp16')
    pipeline.enable_model_cpu_offload()

# Function to run inpainting pipeline
def run_inpainting_pipeline(input_image: Image, mask_image: Image, prompt: str, seed: int = 12345):
    global pipeline
    if pipeline is None:
        initialize_pipeline()

    generator = torch.Generator().manual_seed(seed)

    output = pipeline(
        prompt,
        image=input_image,
        mask_image=mask_image,
        height=1024,
        width=1024,
        generator=generator,
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
