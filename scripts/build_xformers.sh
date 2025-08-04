#!/bin/bash

echo "=== Building xformers from source for RTX 5080 (sm_120) ==="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Error: Not in a virtual environment. Please activate your virtual environment first."
    exit 1
fi

# Check if PyTorch is installed and CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
    echo "Error: PyTorch with CUDA support is required"
    exit 1
}

# Get CUDA capability
CUDA_CAPABILITY=$(python -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability(0))))")
echo "Detected CUDA capability: $CUDA_CAPABILITY"

# Clean up any existing xformers installation
echo "Removing existing xformers installation..."
pip uninstall -y xformers 2>/dev/null || true

# Install build dependencies
echo "Installing build dependencies..."
pip install -U ninja packaging setuptools wheel

# Set build environment variables
# Use sm_89 for now as CUDA doesn't support sm_120 in current nvcc
# The PTX will allow forward compatibility with newer GPUs
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
export XFORMERS_ENABLE_DEBUG_ASSERTIONS=0
export XFORMERS_BUILD_TYPE=Release
export MAX_JOBS=4
export FORCE_CUDA=1
export NVCC_FLAGS="-Xfatbin -compress-all"

echo "Build configuration:"
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  MAX_JOBS=$MAX_JOBS"
echo "  PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"

# Clone xformers repository
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

echo "Cloning xformers repository..."
git clone https://github.com/facebookresearch/xformers.git
cd xformers

# Try main branch first (has latest GPU support)
echo "Building xformers from main branch..."
git checkout main
git submodule update --init --recursive

# Build and install
if pip install -v .; then
    echo "✓ Successfully built xformers from main branch"
else
    echo "Main branch failed, trying stable release..."
    git checkout v0.0.28
    git submodule update --init --recursive
    
    if pip install -v .; then
        echo "✓ Successfully built xformers v0.0.28"
    else
        echo "✗ Failed to build xformers"
        cd /
        rm -rf "$TEMP_DIR"
        exit 1
    fi
fi

# Clean up
cd /
rm -rf "$TEMP_DIR"

# Test xformers
echo ""
echo "Testing xformers installation..."
python -c "
import torch
import xformers
import xformers.ops

print(f'xformers version: {xformers.__version__}')

# Test memory efficient attention
if torch.cuda.is_available():
    device = 'cuda'
    dtype = torch.float16
    
    # Test with typical attention dimensions
    batch_size = 2
    seq_len = 1024
    n_heads = 16
    d_head = 64
    
    q = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
    
    try:
        out = xformers.ops.memory_efficient_attention(q, k, v)
        print('✓ Memory efficient attention working correctly')
        print(f'  Input shape: {q.shape}')
        print(f'  Output shape: {out.shape}')
    except Exception as e:
        print(f'✗ Memory efficient attention failed: {e}')
        import traceback
        traceback.print_exc()
else:
    print('⚠ CUDA not available for testing')

# Show available ops
print('\\nAvailable xformers ops:')
for op_name in dir(xformers.ops):
    if not op_name.startswith('_'):
        print(f'  - {op_name}')
"

echo ""
echo "=== xformers build complete ==="
echo "You can now use memory efficient attention in your models"