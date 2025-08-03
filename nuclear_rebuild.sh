#!/bin/bash

echo "=== NUCLEAR REBUILD: Complete PyTorch ecosystem rebuild for RTX 5080 ==="

# Set the correct PYTHONPATH
export PYTHONPATH="/home/relh/.venv/lib/python3.12/site-packages:$PYTHONPATH"

echo "Step 1: Removing ALL PyTorch-related packages..."
uv pip uninstall -y torch torchvision torchaudio
uv pip uninstall -y xformers diffusers kornia flash-attn
uv pip uninstall -y pytorch3d trimesh open3d
uv pip uninstall -y transformers accelerate
uv pip uninstall -y timm mmcv mmengine
uv pip uninstall -y lpips

echo "Step 2: Cleaning up compiled binaries..."
rm -rf /home/relh/.venv/lib/python3.12/site-packages/torch*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/xformers*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/flash_attn*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/kornia*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/pytorch3d*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/trimesh*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/open3d*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/transformers*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/accelerate*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/timm*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/mmcv*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/mmengine*
rm -rf /home/relh/.venv/lib/python3.12/site-packages/lpips*

echo "Step 3: Installing PyTorch 2.7.1 with CUDA 12.4 support..."
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

echo "Step 4: Installing core dependencies..."
uv pip install transformers accelerate

echo "Step 5: Installing diffusers..."
uv pip install diffusers

echo "Step 6: Installing kornia (without flash attention)..."
FLASH_ATTN_SKIP_CUDA_BUILD=1 uv pip install kornia

echo "Step 7: Installing other required packages..."
uv pip install pytorch3d trimesh open3d timm mmcv mmengine lpips

echo "Step 8: Installing xformers (without flash attention)..."
XFORMERS_DISABLE_FLASH_ATTN=1 uv pip install xformers --index-url https://download.pytorch.org/whl/cu124

echo "Step 9: Testing all imports..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

import diffusers
print(f'Diffusers: {diffusers.__version__}')

import kornia
print('Kornia loaded successfully')

import xformers
print('Xformers loaded successfully')

import pytorch3d
print('PyTorch3D loaded successfully')

print('=== ALL PACKAGES LOADED SUCCESSFULLY! ===')
"

echo "=== REBUILD COMPLETE ==="
echo "You can now run: python run.py" 