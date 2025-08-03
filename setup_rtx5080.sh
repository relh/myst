#!/bin/bash

echo "=== MYST Setup for RTX 5080 ==="
echo "This script sets up the environment for RTX 5080 (sm_120 architecture)"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Not in a virtual environment. Creating one..."
    uv venv
    source .venv/bin/activate
fi

echo "Step 1: Installing PyTorch with CUDA 12.4 support for RTX 5080..."
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124

echo "Step 2: Installing core dependencies..."
uv pip install -r requirements.txt

echo "Step 3: Installing xformers (without flash attention for compatibility)..."
XFORMERS_DISABLE_FLASH_ATTN=1 uv pip install xformers --index-url https://download.pytorch.org/whl/cu124

echo "Step 4: Installing kornia (without flash attention)..."
FLASH_ATTN_SKIP_CUDA_BUILD=1 uv pip install kornia

echo "Step 5: Testing installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')

import diffusers
print(f'Diffusers: {diffusers.__version__}')

import kornia
print('Kornia loaded successfully')

import xformers
print('Xformers loaded successfully')

print('=== Installation successful! ===')
"

echo "Step 6: Creating output directories..."
mkdir -p outputs/imgs outputs/pickles

echo "=== Setup complete! ==="
echo "You can now run: python run.py" 