#!/usr/bin/env python
# -*- coding: utf-8 -*-

import einops
import torch
import torch.nn.functional as F
from torchvision import transforms
import lpips
import numpy as np

# Lazy load LPIPS model
lpips_fn = None

def get_lpips_model():
    global lpips_fn
    if lpips_fn is None:
        lpips_fn = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            lpips_fn = lpips_fn.cuda()
    return lpips_fn

# Function to extract non-overlapping patches
def extract_non_overlapping_patches(image, patch_size=64):
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(-1, 3, patch_size, patch_size)
    return patches

# Function to preprocess images (adapted for PyTorch tensors)
def preprocess_image(image_tensor):
    # Convert from uint8 [0, 255] to float [0, 1]
    image_tensor = image_tensor.float() / 255.0
    
    # Normalize to [-1, 1]
    image_tensor = image_tensor * 2 - 1
    return image_tensor.unsqueeze(0)

# Function to compute LPIPS distance
def compute_lpips_distance(patch, real_patches, lpips_fn):
    distances = []
    for real_patch in real_patches:
        distance = lpips_fn(patch.unsqueeze(0), real_patch.unsqueeze(0))
        distances.append(distance.item())
    return min(distances)

# Main function to be called with images
def evaluate_image_patches(real_image_tensor, generated_image_tensor, patch_size=64, threshold=0.4):
    lpips_fn = get_lpips_model()

    real_image_tensor = einops.rearrange(real_image_tensor, 'h w c -> c h w')
    generated_image_tensor = einops.rearrange(generated_image_tensor, 'h w c -> c h w')
    
    # Preprocess images (adapted for PyTorch tensors)
    real_image_tensor = preprocess_image(real_image_tensor)
    generated_image_tensor = preprocess_image(generated_image_tensor)

    # Extract non-overlapping patches
    real_patches = extract_non_overlapping_patches(real_image_tensor, patch_size)
    generated_patches = extract_non_overlapping_patches(generated_image_tensor, patch_size)

    # Compute LPIPS distance for each generated patch
    lpips_distances = [compute_lpips_distance(patch, real_patches, lpips_fn) for patch in generated_patches]

    # Convert distances to tensor
    lpips_distances = torch.tensor(lpips_distances)

    # Determine which patches are unrealistic
    unrealistic_mask = lpips_distances > threshold

    return lpips_distances, unrealistic_mask

