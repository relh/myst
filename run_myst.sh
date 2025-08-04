#!/bin/bash

# Myst runner script - automatically sets up environment and bypasses Cursor AppImage issues

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Use the correct Python from the virtual environment
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

# Set up PYTHONPATH to include VGGT
export PYTHONPATH="$SCRIPT_DIR/../vggt:$PYTHONPATH"

# Check if VGGT directory exists
if [ ! -d "$SCRIPT_DIR/../vggt" ]; then
    echo "Warning: VGGT directory not found at $SCRIPT_DIR/../vggt"
    echo "Please run ./setup_env.sh first"
    exit 1
fi

# Run the myst script with all arguments passed through
echo "Using Python: $PYTHON_BIN"
echo "PYTHONPATH: $PYTHONPATH"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run.py" "$@"