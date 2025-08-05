#!/bin/bash

# Combined Dust3r/Mast3r Setup Script
# This script sets up both Dust3r and Mast3r

echo "=== Setting up both DUSt3R and MASt3R ==="
echo ""

# Run individual setup scripts
echo "Installing DUSt3R..."
bash scripts/dust3r_setup.sh

echo ""
echo "Installing MASt3R..."
bash scripts/mast3r_setup.sh

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Available depth estimation methods:"
echo "  --depth dust    : Use DUSt3R (relative depth)"
echo "  --depth mast3r  : Use MASt3R (metric depth, recommended)"
echo "  --depth vggt    : Use VGGT (fastest, default)"
echo ""