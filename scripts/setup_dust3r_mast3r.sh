#!/bin/bash

# Dust3r/Mast3r Setup Script
# This script sets up Dust3r and Mast3r for use with Myst

echo "=== Setting up Dust3r/Mast3r for Myst ==="
echo "Note: VGGT is now the recommended depth estimator, but this script is provided for compatibility."

# Check if we're in the myst directory
if [ ! -f "run.py" ]; then
    echo "Error: This script must be run from the myst directory"
    exit 1
fi

# Create directories for dust3r and mast3r as siblings to myst
PARENT_DIR="$(dirname "$(pwd)")"
DUST3R_DIR="$PARENT_DIR/dust3r"
MAST3R_DIR="$PARENT_DIR/mast3r"

echo "Step 1: Cloning repositories..."

# Clone dust3r
if [ ! -d "$DUST3R_DIR" ]; then
    echo "Cloning dust3r..."
    cd "$PARENT_DIR"
    git clone --recursive https://github.com/naver/dust3r.git
else
    echo "dust3r already exists at $DUST3R_DIR"
fi

# Clone mast3r
if [ ! -d "$MAST3R_DIR" ]; then
    echo "Cloning mast3r..."
    cd "$PARENT_DIR"
    git clone --recursive https://github.com/naver/mast3r.git
else
    echo "mast3r already exists at $MAST3R_DIR"
fi

echo "Step 2: Installing dependencies..."

# Install dust3r requirements
cd "$DUST3R_DIR"
pip install -r requirements.txt
pip install -r requirements_optional.txt || echo "Some optional requirements failed"

# Install mast3r requirements
cd "$MAST3R_DIR"
pip install -r requirements.txt

echo "Step 3: Downloading model checkpoints..."

# Create checkpoint directories
mkdir -p "$DUST3R_DIR/checkpoints"
mkdir -p "$MAST3R_DIR/checkpoints"

# Download dust3r checkpoint
DUST3R_CHECKPOINT="$DUST3R_DIR/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "$DUST3R_CHECKPOINT" ]; then
    echo "Downloading DUSt3R checkpoint..."
    cd "$DUST3R_DIR/checkpoints"
    wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
fi

# Download mast3r checkpoint
MAST3R_CHECKPOINT="$MAST3R_DIR/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
if [ ! -f "$MAST3R_CHECKPOINT" ]; then
    echo "Downloading MASt3R checkpoint..."
    cd "$MAST3R_DIR/checkpoints"
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
fi

echo "Step 4: Setting up Python paths..."

cd "$(dirname "$0")/.."  # Back to myst directory

# Update three_d.py to use dust3r/mast3r
echo ""
echo "=== IMPORTANT: Manual steps required ==="
echo ""
echo "1. Add these paths to the beginning of misc/three_d.py:"
echo "   sys.path.append('$PARENT_DIR/dust3r/')"
echo "   sys.path.append('$PARENT_DIR/mast3r/')"
echo ""
echo "2. Uncomment the dust3r/mast3r imports in misc/three_d.py"
echo ""
echo "3. Remove or comment out the NotImplementedError in img_to_pts_3d_dust()"
echo ""
echo "4. Run with: python run.py --depth dust"
echo ""
echo "Note: Dust3r/Mast3r are slower than VGGT. Consider using --depth vggt for better performance."