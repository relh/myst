#!/bin/bash

# Dust3r Setup Script
# Sets up DUSt3R (Dense Unconstrained Stereo 3D Reconstruction)

echo "=== Setting up DUSt3R ==="

# Check if we're in the myst directory
if [ ! -f "run.py" ]; then
    echo "Error: This script must be run from the myst directory"
    exit 1
fi

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: No virtual environment detected. Please activate your venv first."
    echo "Run: source .venv/bin/activate"
    exit 1
fi

# Get parent directory
PARENT_DIR="$(dirname "$(pwd)")"
DUST3R_DIR="$PARENT_DIR/dust3r"

echo "Step 1: Installing CroCo dependencies..."
bash scripts/croco_setup.sh

echo "Step 2: Cloning DUSt3R repository..."
if [ ! -d "$DUST3R_DIR" ]; then
    cd "$PARENT_DIR"
    git clone --recursive https://github.com/naver/dust3r.git
    cd dust3r
    # Checkout stable commit (optional, remove if you want latest)
    # git checkout 4545e79
else
    echo "DUSt3R already exists at $DUST3R_DIR"
fi

echo "Step 3: Installing DUSt3R requirements..."
cd "$DUST3R_DIR"
uv pip install -r requirements.txt
uv pip install -r requirements_optional.txt || echo "Some optional requirements failed (this is ok)"

echo "Step 4: Downloading DUSt3R model checkpoint..."
mkdir -p "$DUST3R_DIR/checkpoints"
DUST3R_CHECKPOINT="$DUST3R_DIR/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "$DUST3R_CHECKPOINT" ]; then
    echo "Downloading DUSt3R checkpoint (700MB)..."
    cd "$DUST3R_DIR/checkpoints"
    wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
else
    echo "DUSt3R checkpoint already exists"
fi

echo ""
echo "=== DUSt3R Setup Complete ==="
echo ""
echo "To use DUSt3R in myst, run:"
echo "  ./run_myst.sh --depth dust"
echo ""
echo "Note: DUSt3R is automatically imported in misc/three_d.py"