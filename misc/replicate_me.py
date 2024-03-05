#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BytesIO

import replicate
from PIL import Image


def run_replicate_with_pil(image: Image, mask: Image, prompt: str):
    # Convert PIL Images to bytes
    image_bytes = BytesIO()
    mask_bytes = BytesIO()
    image.save(image_bytes, format='JPEG') # Adjust format as necessary
    mask.save(mask_bytes, format='JPEG')   # Adjust format as necessary
    image_bytes.seek(0)
    mask_bytes.seek(0)

    # Run the Replicate model with the bytes
    output = replicate.run(
        "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        input={
            "prompt": prompt,
            "image": image_bytes,
            "mask": mask_bytes
        }
    )
    return output

# Example usage:
# image = Image.open("path/to/your_image.jpg")
# mask = Image.open("path/to/your_mask.jpg")
# prompt = "an armchair in a room full of plants"
# output = run_replicate_with_pil(image, mask, prompt)
# print(output)

