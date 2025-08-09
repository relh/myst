#!/bin/bash

echo "=== MYST Environment Setup ==="
echo "This script sets up the environment for Myst on any modern CUDA GPU."
echo "It will install dependencies, set up VGGT, and prepare your workspace."

# Parse command line arguments
REBUILD=false
WITH_DUST3R=false
for arg in "$@"; do
    case $arg in
        --rebuild|--nuclear)
            REBUILD=true
            echo "🔥 NUCLEAR REBUILD MODE: Will completely reinstall PyTorch ecosystem"
            ;;
        --with-dust3r)
            WITH_DUST3R=true
            echo "📦 Will also install Dust3r/Mast3r models"
            ;;
    esac
done

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
    uv pip uninstall -y xformers diffusers kornia flash-attn flash-attn-2 2>/dev/null || true
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

echo "Step 1: Installing PyTorch (stable, pinned) with CUDA 12.8 support..."
# Pin explicit versions to ensure API compatibility with xformers and sm_120
echo "Installing torch==2.7.1, torchvision==0.22.1, torchaudio==2.7.1 (cu128)..."
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 --upgrade

# Verify CUDA compute capability support
python - <<'PY'
import torch
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    print(f'GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'CUDA capability: {capability}')
    if capability[0] == 12 and capability[1] == 0:
        print('✓ RTX 5080 (sm_120) support confirmed')
    # Check flash attention attribute availability (required by recent xformers)
    try:
        print('flash attention attribute present:', hasattr(torch.backends.cuda, 'is_flash_attention_available'))
    except Exception as e:
        print(f'flash attention attribute check failed: {e}')
    # Test a simple CUDA operation
    try:
        x = torch.ones(1).cuda()
        print('✓ CUDA operations working')
    except RuntimeError as e:
        print(f'⚠ CUDA operation failed: {e}')
        print('You may need a newer PyTorch build')
else:
    print('⚠ CUDA not available')
PY

echo "Step 2: Installing core dependencies..."
if [ "$REBUILD" = true ]; then
    # For rebuild, install everything fresh
    uv pip install transformers accelerate diffusers
    uv pip install trimesh open3d timm mmcv mmengine lpips
else
    # For normal setup, use uv sync
    uv sync --no-deps  # Skip dependencies to avoid conflicts
fi

echo "Step 3: Installing kornia (pinned to <0.8 to avoid flash attention path)..."
uv pip install "kornia>=0.7.2,<0.8.0"

echo "Step 4: Installing xformers with pre-built stable wheels (conditional)..."
# xformers is useful but only if PyTorch exposes flash attention capability API expected by recent xformers

# Set environment variables for memory efficiency
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check torch flash attention attribute availability
if python - <<'PY'
import sys, torch
sys.exit(0 if hasattr(torch.backends.cuda, 'is_flash_attention_available') else 1)
PY
then
    echo "PyTorch reports flash attention attribute is available; proceeding to install xformers."
    # Check if xformers is already installed and working
    if python -c "import xformers; import xformers.ops; xformers.ops.memory_efficient_attention(torch.randn(1,8,128,64).cuda().half(), torch.randn(1,8,128,64).cuda().half(), torch.randn(1,8,128,64).cuda().half())" 2>/dev/null; then
        echo "✓ xformers is already installed and working"
        python -c "import xformers; print(f'  Version: {xformers.__version__}')"
    else
        echo "Installing xformers stable matching PyTorch stable (CUDA 12.8)..."
        uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu128
        if python -c "import xformers; import xformers.ops; xformers.ops.memory_efficient_attention(torch.randn(1,8,128,64).cuda().half(), torch.randn(1,8,128,64).cuda().half(), torch.randn(1,8,128,64).cuda().half())" 2>/dev/null; then
            echo "✓ xformers installed successfully"
            python -c "import xformers; print(f'  Version: {xformers.__version__}')"
        else
            echo "WARNING: xformers installation failed; continuing without xformers"
        fi
    fi
else
    echo "PyTorch does not expose torch.backends.cuda.is_flash_attention_available; removing xformers to avoid import errors."
    echo "Diffusers will fall back to PyTorch SDPA."
    # Force uninstall xformers to prevent import errors
    echo "Uninstalling xformers..."
    uv pip uninstall -y xformers 2>/dev/null || true
    # Double check it's gone
    if python -c "import xformers" 2>/dev/null; then
        echo "WARNING: xformers still present, forcing removal..."
        SITE_PACKAGES="$VIRTUAL_ENV/lib/python*/site-packages"
        rm -rf $SITE_PACKAGES/xformers* 2>/dev/null || true
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

# Install VGGT dependencies WITHOUT overriding PyTorch
if [ -f "requirements.txt" ]; then
    # Install requirements but skip torch/torchvision to keep our CUDA 12.8 versions
    uv pip install -r requirements.txt --no-deps
    # Install only the non-torch dependencies
    uv pip install einops opencv-python matplotlib
fi

# Install VGGT as an editable package so Python can find it properly
echo "Installing VGGT as editable package..."
uv pip install -e . --no-deps

# Create checkpoints directory if needed
mkdir -p checkpoints

# Return to myst directory
cd ..

echo "Step 7: Ensuring correct PyTorch version and removing xformers..."
# Re-install PyTorch to ensure it wasn't downgraded by VGGT
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 --upgrade

# Final xformers check and removal
if python - <<'PY'
import sys, torch
sys.exit(0 if hasattr(torch.backends.cuda, 'is_flash_attention_available') else 1)
PY
then
    echo "PyTorch flash attention API available, xformers can be used if needed"
else
    echo "Removing xformers to prevent import errors..."
    uv pip uninstall -y xformers 2>/dev/null || true
fi

echo "Step 8: Testing installation..."
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

echo "Step 9: Creating output directories..."
mkdir -p outputs/imgs outputs/pickles

# Optional: Install Dust3r/Mast3r
if [ "$WITH_DUST3R" = true ]; then
    echo ""
    echo "Step 10: Installing Dust3r/Mast3r models..."
    
    # Get parent directory
    PARENT_DIR="$(dirname "$(pwd)")"
    
    # Setup CroCo (required by both dust3r and mast3r)
    echo "Installing CroCo dependencies..."
    CROCO_DIR="$PARENT_DIR/croco"
    if [ ! -d "$CROCO_DIR" ]; then
        cd "$PARENT_DIR"
        git clone https://github.com/naver/croco.git
        cd croco
        # Download CroCo checkpoint
        mkdir -p checkpoints/
        if [ ! -f "checkpoints/CroCo_V2_ViTLarge_BaseDecoder.pth" ]; then
            cd checkpoints/
            wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth
            cd ..
        fi
        cd ..
    fi
    cd "$PARENT_DIR/myst"
    
    # Setup Dust3r
    echo "Installing Dust3r..."
    DUST3R_DIR="$PARENT_DIR/dust3r"
    if [ ! -d "$DUST3R_DIR" ]; then
        cd "$PARENT_DIR"
        git clone --recursive https://github.com/naver/dust3r.git
        cd dust3r
        uv pip install -r requirements.txt
        uv pip install -r requirements_optional.txt || echo "Some optional requirements failed (this is ok)"
        
        # Download checkpoint
        mkdir -p checkpoints
        if [ ! -f "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth" ]; then
            cd checkpoints
            wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
            cd ..
        fi
        cd ..
    fi
    cd "$PARENT_DIR/myst"
    
    # Setup Mast3r
    echo "Installing Mast3r..."
    MAST3R_DIR="$PARENT_DIR/mast3r"
    if [ ! -d "$MAST3R_DIR" ]; then
        cd "$PARENT_DIR"
        git clone --recursive https://github.com/naver/mast3r.git
        cd mast3r
        uv pip install -r requirements.txt
        
        # Download checkpoint
        mkdir -p checkpoints
        if [ ! -f "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth" ]; then
            cd checkpoints
            wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
            cd ..
        fi
        cd ..
    fi
    cd "$PARENT_DIR/myst"
    
    echo "✓ Dust3r/Mast3r models installed"
fi

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
if [ "$WITH_DUST3R" = true ]; then
    echo "Available depth models:"
    echo "  --depth vggt    : VGGT (default, fastest)"
    echo "  --depth dust    : Dust3r (relative depth)"
    echo "  --depth mast3r  : Mast3r (metric depth)"
    echo "  --depth da      : Depth Anything"
    echo "  --depth metric  : Metric3D"
else
    echo "Available depth models:"
    echo "  --depth vggt    : VGGT (default, fastest)"
    echo "  --depth da      : Depth Anything"
    echo "  --depth metric  : Metric3D"
    echo ""
    echo "Optional: To also install Dust3r/Mast3r models, run:"
    echo "  ./setup_env.sh --with-dust3r"
fi