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
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

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

echo "Step 4: Installing xformers..."
# Check if xformers needs to be built from source for sm_120 (RTX 5080)
BUILD_FROM_SOURCE=false
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')" 2>/dev/null | grep -q "(12, 0)" && BUILD_FROM_SOURCE=true
fi

if [ "$BUILD_FROM_SOURCE" = true ]; then
    echo "Detected sm_120 GPU (RTX 5080), building xformers from source..."
    # Set environment variables for building with sm_120 support
    export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;8.9;9.0;12.0"
    export XFORMERS_ENABLE_DEBUG_ASSERTIONS=0
    export XFORMERS_DISABLE_FLASH_ATTN=1
    export MAX_JOBS=4
    
    # Install build dependencies
    uv pip install ninja packaging
    
    # Build and install xformers from source
    uv pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
else
    echo "Installing pre-built xformers nightly..."
    # Try nightly build first for better compatibility
    XFORMERS_DISABLE_FLASH_ATTN=1 uv pip install --pre xformers --index-url https://download.pytorch.org/whl/nightly/cu128 || {
        echo "Nightly xformers failed, trying stable..."
        XFORMERS_DISABLE_FLASH_ATTN=1 uv pip install xformers --index-url https://download.pytorch.org/whl/cu128
    }
fi

echo "Step 5: Installing PyTorch3D (optional)..."
uv pip install pytorch3d || echo "PyTorch3D installation failed - will use fallback renderer"

echo "Step 6: Setting up VGGT..."
# Get the parent directory of the current myst directory
PARENT_DIR="$(dirname "$(pwd)")"
VGGT_DIR="$PARENT_DIR/vggt"

if [ ! -d "$VGGT_DIR" ]; then
    echo "Cloning VGGT repository to $VGGT_DIR..."
    cd "$PARENT_DIR"
    git clone https://github.com/facebookresearch/vggt.git
    cd vggt
    # Create missing __init__.py files
    echo "Creating missing __init__.py files..."
    echo "# VGGT package initialization" > vggt/__init__.py
    echo "# VGGT models package initialization" > vggt/models/__init__.py
    echo "from .vggt import VGGT" >> vggt/models/__init__.py
    echo "__all__ = ['VGGT']" >> vggt/models/__init__.py
    cd "$(pwd)/myst"
else
    echo "VGGT repository already exists at $VGGT_DIR"
fi

echo "Step 7: Testing installation..."
VGGT_DIR="$PARENT_DIR/vggt"
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
import xformers
print(f'Xformers: {xformers.__version__}')
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
echo "  PYTHONPATH=$VGGT_DIR:\$PYTHONPATH python run.py"
echo ""
echo "Note: The script will automatically fall back to raster renderer if PyTorch3D is not available."
echo "Note: xformers flash attention is disabled for compatibility."