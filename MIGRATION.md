# Migration Guide: VGGT vs Dust3r/Mast3r

This guide explains how to switch between VGGT (default) and the original Dust3r/Mast3r implementation.

## Using VGGT (Recommended - Default)

VGGT is now the default 3D reconstruction method. To use it:

```bash
python run.py --depth vggt
```

### Benefits:
- Much faster reconstruction (<1 second vs tens of seconds)
- Better memory efficiency
- State-of-the-art accuracy
- Excellent single-view reconstruction

## Reverting to Dust3r/Mast3r

If you need to use the original Dust3r/Mast3r implementation:

### 1. Re-enable Dust3r imports

Edit `misc/three_d.py` and uncomment the following lines:

```python
# Around line 7-9, uncomment:
sys.path.append('mast3r/dust3r/')
sys.path.append('mast3r/')

# Around line 33-44, uncomment:
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.image import rgb
from dust3r.viz import (CAM_COLORS, OPENGL, add_scene_cam, cat_meshes,
                        pts3d_to_trimesh)
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
```

### 2. Remove the NotImplementedError

In the `img_to_pts_3d_dust` function, remove or comment out:
```python
raise NotImplementedError("Dust3r/Mast3r support has been replaced by VGGT...")
```

### 3. Install Dust3r/Mast3r

Follow the original setup instructions for Dust3r/Mast3r:

```bash
# Clone mast3r repository
git clone --recursive https://github.com/naver/mast3r
cd mast3r
# Install dependencies
pip install -r requirements.txt
pip install -r dust3r/requirements.txt
# Download model weights
mkdir -p checkpoints
# Download MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth to checkpoints/
cd ..
```

### 4. Run with Dust3r

```bash
python run.py --depth dust
```

## Performance Comparison

| Method | Speed | Memory Usage | Accuracy | Single-view |
|--------|-------|--------------|----------|-------------|
| VGGT | <1s | Low | Excellent | Excellent |
| Dust3r/Mast3r | 10-30s | High | Good | Limited |
| Metric3D | 2-5s | Medium | Good | Good |
| Depth Anything | 1-2s | Low | Fair | Good |

## Troubleshooting

### VGGT Issues
- If VGGT fails to load, run: `bash scripts/vggt_setup.sh`
- For CUDA errors, ensure PyTorch is installed with the correct CUDA version
- Model download issues: The first run downloads ~2GB of weights

### Dust3r/Mast3r Issues
- Import errors: Ensure all imports are uncommented
- Model not found: Download the checkpoint file manually
- OOM errors: Dust3r uses more memory, reduce batch size or image resolution

## Recommendations

We strongly recommend using VGGT for:
- Faster iteration cycles
- Better memory efficiency
- State-of-the-art results
- Easier setup and maintenance

Only use Dust3r/Mast3r if you need:
- Specific compatibility with existing Dust3r workflows
- To reproduce previous results
- To compare methodologies 