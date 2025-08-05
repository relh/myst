#!/bin/bash

# CroCo Setup Script for uv venv environment
# CroCo (Cross-view Completion) is a foundation model used by Dust3r/Mast3r

echo "=== Setting up CroCo dependencies ==="

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No virtual environment detected. Please activate your venv first."
    echo "Run: source .venv/bin/activate"
    exit 1
fi

# Install PyTorch if not already installed
echo "Installing PyTorch and core dependencies..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies for CroCo
echo "Installing CroCo dependencies..."
uv pip install scikit-learn tqdm opencv-python matplotlib ipywidgets
uv pip install quaternion-numpy  # quaternion package for numpy

# Optional: Install habitat-sim for pretraining (skip if not needed)
# Note: habitat-sim can be tricky to install, so we make it optional
echo "Note: habitat-sim installation is optional and can be skipped"
echo "If you need it for pretraining, run:"
echo "  uv pip install habitat-sim --index-url https://aihabitat.org/packages/habitat-sim"

echo "CroCo dependencies installed successfully!"