#!/bin/bash

# MASt3R Setup Script
# Sets up MASt3R (Matching And Stereo 3D Reconstruction)

echo "=== Setting up MASt3R ==="

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
MAST3R_DIR="$PARENT_DIR/mast3r"

echo "Step 1: Installing CroCo dependencies..."
bash scripts/croco_setup.sh

echo "Step 2: Cloning MASt3R repository..."
if [ ! -d "$MAST3R_DIR" ]; then
    cd "$PARENT_DIR"
    git clone --recursive https://github.com/naver/mast3r.git
else
    echo "MASt3R already exists at $MAST3R_DIR"
fi

echo "Step 3: Installing MASt3R requirements..."
cd "$MAST3R_DIR"
uv pip install -r requirements.txt

# MASt3R has dust3r as a submodule
echo "Step 4: Setting up dust3r submodule..."
if [ ! -d "$MAST3R_DIR/dust3r" ]; then
    git submodule update --init --recursive
fi

echo "Step 5: Downloading MASt3R model checkpoint..."
mkdir -p "$MAST3R_DIR/checkpoints"
MAST3R_CHECKPOINT="$MAST3R_DIR/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
if [ ! -f "$MAST3R_CHECKPOINT" ]; then
    echo "Downloading MASt3R checkpoint (700MB)..."
    cd "$MAST3R_DIR/checkpoints"
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
else
    echo "MASt3R checkpoint already exists"
fi

echo ""
echo "=== MASt3R Setup Complete ==="
echo ""
echo "To use MASt3R in myst, run:"
echo "  ./run_myst.sh --depth mast3r"
echo ""
echo "Note: MASt3R provides metric depth estimation (better for 3D reconstruction)"
echo "      DUSt3R provides relative depth (use --depth dust)"