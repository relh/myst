#!/bin/bash

# VGGT Setup Script
# This script installs VGGT (Visual Geometry Grounded Transformer) as a replacement for Dust3r/Mast3r

echo "Setting up VGGT (Visual Geometry Grounded Transformer)..."

# Check if we're in a conda/mamba environment
if [ -z "$CONDA_PREFIX" ]; then
    echo "Warning: No conda environment detected. It's recommended to use a conda environment."
    echo "Create one with: mamba create -n myst python=3.10"
    echo "Then activate it: mamba activate myst"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install PyTorch with CUDA support (adjust CUDA version as needed)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn (required for VGGT)
echo "Installing Flash Attention..."
pip install flash-attn --no-build-isolation

# Install other dependencies
echo "Installing other dependencies..."
pip install einops huggingface-hub transformers>=4.30.0

# Clone and install VGGT
echo "Installing VGGT from GitHub..."
if [ -d "vggt" ]; then
    echo "VGGT directory already exists. Removing old version..."
    rm -rf vggt
fi

git clone https://github.com/facebookresearch/vggt.git
cd vggt
pip install -e .
cd ..

# Download VGGT model weights
echo "Downloading VGGT model weights..."
python -c "
import torch
print('Downloading VGGT model...')
try:
    # This will download the model weights
    model = torch.hub.load('facebookresearch/vggt', 'vggt', trust_repo=True)
    print('VGGT model downloaded successfully!')
except Exception as e:
    print(f'Error downloading model: {e}')
    print('You may need to download manually on first run.')
"

echo "VGGT setup complete!"
echo ""
echo "To use VGGT in myst, run:"
echo "  python run.py --depth vggt"
echo ""
echo "Note: The first run may take time to download model weights (~2GB)." 