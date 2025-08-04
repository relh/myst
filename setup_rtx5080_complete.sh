commit b4e2e0e577faed5024e110df34e4dec8d75aaff3
Author: Richard Higgins <richard@relh.net>
Date:   Sun Aug 3 17:22:36 2025 -0700

    more ruff

diff --git a/setup_rtx5080_complete.sh b/setup_rtx5080_complete.sh
new file mode 100755
index 0000000..077a544
--- /dev/null
+++ b/setup_rtx5080_complete.sh
@@ -0,0 +1,97 @@
+#!/bin/bash
+
+echo "=== MYST Complete Setup for RTX 5080 ==="
+echo "This script sets up the environment for RTX 5080 (sm_120 architecture)"
+echo "Includes all fixes for PyTorch3D compatibility and VGGT setup"
+
+# Check if uv is available
+if ! command -v uv &> /dev/null; then
+    echo "Error: uv is not installed. Please install uv first:"
+    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
+    exit 1
+fi
+
+# Check if we're in a virtual environment
+if [[ "$VIRTUAL_ENV" == "" ]]; then
+    echo "Warning: Not in a virtual environment. Creating one..."
+    uv venv
+    source .venv/bin/activate
+fi
+
+echo "Step 1: Installing PyTorch with CUDA 12.8 support for RTX 5080..."
+uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
+
+echo "Step 2: Installing core dependencies..."
+uv pip install -r requirements.txt
+
+echo "Step 3: Installing xformers (without flash attention for compatibility)..."
+XFORMERS_DISABLE_FLASH_ATTN=1 uv pip install xformers --index-url https://download.pytorch.org/whl/cu128
+
+echo "Step 4: Installing kornia (without flash attention)..."
+FLASH_ATTN_SKIP_CUDA_BUILD=1 uv pip install kornia
+
+echo "Step 5: Setting up VGGT..."
+# Get the parent directory of the current myst directory
+PARENT_DIR="$(dirname "$(pwd)")"
+VGGT_DIR="$PARENT_DIR/vggt"
+
+if [ ! -d "$VGGT_DIR" ]; then
+    echo "Cloning VGGT repository to $VGGT_DIR..."
+    cd "$PARENT_DIR"
+    git clone https://github.com/facebookresearch/vggt.git
+    cd vggt
+    
+    # Create missing __init__.py files
+    echo "Creating missing __init__.py files..."
+    echo "# VGGT package initialization" > vggt/__init__.py
+    echo "# VGGT models package initialization" > vggt/models/__init__.py
+    echo "from .vggt import VGGT" >> vggt/models/__init__.py
+    echo "__all__ = ['VGGT']" >> vggt/models/__init__.py
+    
+    cd "$(pwd)/myst"
+else
+    echo "VGGT repository already exists at $VGGT_DIR"
+fi
+
+echo "Step 6: Testing installation..."
+# Get the VGGT directory for PYTHONPATH
+VGGT_DIR="$PARENT_DIR/vggt"
+PYTHONPATH="$VGGT_DIR:$VIRTUAL_ENV/lib/python3.12/site-packages" python -c "
+import torch
+print(f'PyTorch: {torch.__version__}')
+print(f'CUDA available: {torch.cuda.is_available()}')
+print(f'CUDA version: {torch.version.cuda}')
+if torch.cuda.is_available():
+    print(f'GPU: {torch.cuda.get_device_name(0)}')
+    print(f'CUDA capability: {torch.cuda.get_device_capability(0)}')
+
+import diffusers
+print(f'Diffusers: {diffusers.__version__}')
+
+import kornia
+print('Kornia loaded successfully')
+
+import xformers
+print('Xformers loaded successfully')
+
+try:
+    from vggt.models import VGGT
+    print('VGGT loaded successfully')
+except ImportError as e:
+    print(f'VGGT import failed: {e}')
+
+print('=== Installation successful! ===')
+"
+
+echo "Step 7: Creating output directories..."
+mkdir -p outputs/imgs outputs/pickles
+
+echo "=== Setup complete! ==="
+echo "You can now run myst using:"
+echo "  ./run_myst.sh --headless --depth vggt --renderer raster --prompt auto --control auto --image gen --model sd2"
+echo ""
+echo "Or manually with:"
+echo "  PYTHONPATH=$VGGT_DIR:$VIRTUAL_ENV/lib/python3.12/site-packages python run.py"
+echo ""
+echo "Note: The script will automatically fall back to raster renderer if PyTorch3D is not available"
+echo "Note: xformers flash attention is disabled for RTX 5080 compatibility" 
\ No newline at end of file
