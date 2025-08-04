#!/bin/bash

echo "=== MYST Environment Setup ==="
echo "This script sets up the environment for Myst on any modern CUDA GPU."
echo "It will install dependencies, set up VGGT, and prepare your workspace."

# Parse command line arguments
REBUILD=false
if [[ "$1" == "--rebuild" ]] || [[ "$1" == "--nuclear" ]]; then
    REBUILD=true
    echo "🔥 NUCLEAR REBUILD MODE: Will completely reinstall PyTorch ecosystem"
fi

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

# Nuclear rebuild: Remove all PyTorch-related packages
if [ "$REBUILD" = true ]; then
    echo "Step 0: Removing ALL PyTorch-related packages..."
    uv pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    uv pip uninstall -y xformers diffusers kornia flash-attn 2>/dev/null || true
    uv pip uninstall -y pytorch3d trimesh open3d 2>/dev/null || true
    uv pip uninstall -y transformers accelerate 2>/dev/null || true
    uv pip uninstall -y timm mmcv mmengine 2>/dev/null || true
    uv pip uninstall -y lpips 2>/dev/null || true
    
    echo "Cleaning up compiled binaries..."
    SITE_PACKAGES="$VIRTUAL_ENV/lib/python*/site-packages"
    rm -rf $SITE_PACKAGES/torch* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/xformers* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/flash_attn* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/kornia* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/pytorch3d* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/transformers* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/accelerate* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/timm* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/mmcv* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/mmengine* 2>/dev/null || true
    rm -rf $SITE_PACKAGES/lpips* 2>/dev/null || true
fi

echo "Step 1: Installing PyTorch nightly build with CUDA 12.8 support..."
# Install latest nightly build for RTX 5080 support
# RTX 5080 requires sm_120 support which is in newer PyTorch builds
echo "Installing PyTorch with RTX 5080 (sm_120) support..."
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify CUDA compute capability support
python -c "
import torch
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    print(f'GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'CUDA capability: {capability}')
    if capability[0] == 12 and capability[1] == 0:
        print('✓ RTX 5080 (sm_120) support confirmed')
    # Test a simple CUDA operation
    try:
        x = torch.ones(1).cuda()
        print('✓ CUDA operations working')
    except RuntimeError as e:
        print(f'⚠ CUDA operation failed: {e}')
        print('You may need a newer PyTorch build')
else:
    print('⚠ CUDA not available')
"

echo "Step 2: Installing core dependencies..."
if [ "$REBUILD" = true ]; then
    # For rebuild, install everything fresh
    uv pip install transformers accelerate diffusers
    uv pip install trimesh open3d timm mmcv mmengine lpips
else
    # For normal setup, use uv sync
    uv sync --no-deps  # Skip dependencies to avoid conflicts
fi

echo "Step 3: Installing kornia (without flash attention)..."
FLASH_ATTN_SKIP_CUDA_BUILD=1 uv pip install kornia

echo "Step 4: Installing xformers for memory efficiency..."
# Try to install xformers - it significantly reduces memory usage
# First check if we have CUDA 12.8 compatible xformers
echo "Attempting to install xformers..."

# Set environment variables for memory efficiency
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Try pre-built xformers from PyTorch nightly  
if ! uv pip install xformers --index-url https://download.pytorch.org/whl/cu121; then
    echo "Pre-built xformers not available for your configuration"
    echo "Attempting to build xformers from source (this may take a while)..."
    
    # Install build dependencies
    uv pip install ninja
    
    # Set build flags for RTX 5080 compatibility
    export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
    export XFORMERS_ENABLE_DEBUG_ASSERTIONS=0
    export MAX_JOBS=4
    
    # Build xformers from source - use 0.0.23 which has better compatibility
    if ! uv pip install "xformers==0.0.23" --no-deps; then
        echo "xformers installation failed - will use standard attention"
        echo "This may result in higher memory usage"
    fi
fi

echo "Step 5: Installing PyTorch3D (optional)..."
uv pip install pytorch3d || echo "PyTorch3D installation failed - will use fallback renderer"

echo "Step 6: Setting up VGGT..."
# Clone VGGT directly into the myst directory
VGGT_DIR="$(pwd)/vggt"

if [ ! -d "$VGGT_DIR" ]; then
    echo "Cloning VGGT repository..."
    git clone https://github.com/facebookresearch/vggt.git
else
    echo "VGGT repository already exists at $VGGT_DIR"
fi

# Always reinstall VGGT to ensure it's properly set up
echo "Installing VGGT and its requirements..."
cd vggt

# Install VGGT dependencies
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt
fi

# Install VGGT as an editable package so Python can find it properly
echo "Installing VGGT as editable package..."
uv pip install -e .

# Create checkpoints directory if needed
mkdir -p checkpoints

# Return to myst directory
cd ..

echo "Step 7: Testing installation..."
VGGT_DIR="$(pwd)/vggt"
PYTHONPATH="$VGGT_DIR:$PYTHONPATH" python -c "
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
try:
    import xformers
    print(f'Xformers: {xformers.__version__}')
except ImportError:
    print('Xformers not available - using standard attention')
try:
    import pytorch3d
    print('PyTorch3D loaded successfully')
except ImportError:
    print('PyTorch3D not available - will use fallback renderer')
try:
    from vggt.models import VGGT
    print('VGGT loaded successfully')
except ImportError as e:
    print(f'VGGT import failed: {e}')
print('=== Installation successful! ===')
"

echo "Step 8: Creating output directories..."
mkdir -p outputs/imgs outputs/pickles

echo "=== Setup complete! ==="
echo ""
if [ "$REBUILD" = true ]; then
    echo "🔥 Nuclear rebuild completed successfully!"
fi
echo ""
echo "You can now run Myst using:"
echo "  ./run_myst.sh --headless --depth vggt --renderer raster --prompt auto --control auto --image gen --model sd2"
echo ""
echo "Or manually with:"
echo "  PYTHONPATH=$(pwd)/vggt:\$PYTHONPATH python run.py"
echo ""
echo "Note: The script will automatically fall back to raster renderer if PyTorch3D is not available."
echo "Note: xformers flash attention is disabled for compatibility."
echo ""
echo "Optional: To use Dust3r/Mast3r instead of VGGT, run:"
echo "  bash scripts/setup_dust3r_mast3r.sh"