# myst

Myst is a combination of Stable Diffusion and VGGT (Visual Geometry Grounded Transformer)/DepthAnything/Metric3D to create worlds that are 3D aware and go beyond outpainting.

Simply, Myst lets you start with a text prompt, "an image of a kitchen", generates a 2D image from it using diffusion, lifts the 2D image to 3D, and then lets you navigate around the scene. As you rotate/move to reveal unexplored parts of the scene, you can run additional diffusion steps and new images are generated and then combined into a *single, coherent, 3D scene*. 

**Update**: We've replaced Dust3r/Mast3r with [VGGT](https://github.com/facebookresearch/vggt), the CVPR 2025 Best Paper Award winner, for faster and more accurate 3D reconstruction. 

We can create infinite 3D scenes, for use as a potential dataset. We can manually create these worlds, or do it automatically.

**Architecture:** *Showing how we created these worlds.*

![Architecture](./img/architecture.png)

**Real-time Generation:** *A real-time recording of me creating a kitchen scene using Stable Diffusion and Dust3r.*

<p align="center">
  <img src="./img/demo.gif" alt="Real-time generation">
</p>

**Automatic Dataset:** *Showing a few automatic datasets.*

![Automatic Dataset](./img/automatic_dataset.png)

---

## A Few Scenes 
<table>
  <tr>
    <td align="center">
      <strong>LOTS of bay windows..</strong><br>
      <img src="./img/screencast10.gif" alt="Screencast 10">
    </td>
    <td align="center">
      <strong>What happens in a long hallway?</strong><br>
      <img src="./img/screencast11.gif" alt="Screencast 11">
    </td>
    <td align="center">
      <strong>Monastery tunnels</strong><br>
      <img src="./img/screencast12.gif" alt="Screencast 12">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Full 360 beach views</strong><br>
      <img src="./img/screencast01.gif" alt="Screencast 01">
    </td>
    <td align="center">
      <strong>Kitchen meets a fireplace</strong><br>
      <img src="./img/screencast02.gif" alt="Screencast 02">
    </td>
    <td align="center">
      <strong>Nice wood oak paneling</strong><br>
      <img src="./img/screencast03.gif" alt="Screencast 03">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Super mario kitchen-land</strong><br>
      <img src="./img/screencast04.gif" alt="Screencast 04">
    </td>
    <td align="center">
      <strong>Severance hallway?</strong><br>
      <img src="./img/screencast05.gif" alt="Screencast 05">
    </td>
    <td align="center">
      <strong>More beach and ocean views</strong><br>
      <img src="./img/screencast06.gif" alt="Screencast 06">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Spacious bedroom kitchens</strong><br>
      <img src="./img/screencast07.gif" alt="Screencast 07">
    </td>
    <td align="center">
      <strong>Fireplace bedroom kitchens</strong><br>
      <img src="./img/screencast08.gif" alt="Screencast 08">
    </td>
    <td align="center">
      <strong>Interesting ceilings</strong><br>
      <img src="./img/screencast09.gif" alt="Screencast 09">
    </td>
  </tr>
</table>

## Synthetic Dataset 

<table>
  <tr>
    <td align="center">
      <strong>Urban spook</strong><br>
      <img src="./img/auto1.png" alt="Screencast 10">
    </td>
    <td align="center">
      <strong>Mountains and ducks</strong><br>
      <img src="./img/auto2.png" alt="Screencast 11">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Marble, books, plants</strong><br>
      <img src="./img/auto3.png" alt="Screencast 01">
    </td>
    <td align="center">
      <strong>Buddha, cape town, aerial</strong><br>
      <img src="./img/auto4.png" alt="Screencast 02">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>Venice and ruins</strong><br>
      <img src="./img/auto5.png" alt="Screencast 03">
    </td>
    <td align="center">
      <strong>More kitchens</strong><br>
      <img src="./img/auto6.png" alt="Screencast 03">
    </td>
  </tr>
</table>

## Installation

### Prerequisites

- **Python 3.12+** (recommended)
- **CUDA 12.8+** for GPU acceleration (RTX 5080 compatibility)
- **NVIDIA GPU** with at least 8GB VRAM (16GB+ recommended)
- **uv** package manager (recommended) or **pip**

### Quick Setup

#### Option 1: Automated Setup (Recommended)

For **RTX 5080** and other newer GPUs (sm_120 architecture):

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/your-repo/myst.git
cd myst

# Run the complete automated setup script
chmod +x setup_rtx5080_complete.sh
./setup_rtx5080_complete.sh
```

For **older GPUs** (RTX 30/40 series, sm_80/sm_89):

```bash
# Use the nuclear rebuild script for compatibility
chmod +x nuclear_rebuild.sh
./nuclear_rebuild.sh
```

#### Option 2: Manual Setup

1. **Install uv** (recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Create virtual environment**:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install PyTorch with correct CUDA version**:

For **RTX 5080** (sm_120):
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For **RTX 30/40 series** (sm_80/sm_89):
```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Install dependencies**:
```bash
uv pip install -r requirements.txt
```

5. **Install xformers** (optional, for memory optimization):
```bash
XFORMERS_DISABLE_FLASH_ATTN=1 uv pip install xformers --index-url https://download.pytorch.org/whl/cu128
```

6. **Install VGGT** (for 3D reconstruction):
```bash
uv pip install "git+https://github.com/facebookresearch/vggt.git"
```

#### Option 3: Conda Setup (Legacy)

```bash
# Create conda environment
mamba create -n myst python=3.10
mamba activate myst

# Install PyTorch
mamba install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
mamba install -y diffusers xformers pytorch3d -c pytorch -c nvidia -c pytorch3d -c conda-forge
pip install -r requirements.txt
```

### Troubleshooting

#### RTX 5080 Compatibility Issues

If you encounter CUDA architecture errors with RTX 5080:

1. **Check your CUDA version**:
```bash
nvidia-smi
```

2. **Verify PyTorch installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

3. **Reinstall with correct CUDA version**:
```bash
# Remove existing PyTorch
uv pip uninstall torch torchvision torchaudio -y

# Install with CUDA 12.4
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124
```

#### Common Issues

- **"No module named pip"**: You're using `uv` - use `uv pip` instead of `python -m pip`
- **CUDA architecture mismatch**: Use the correct CUDA version for your GPU
- **xformers compilation errors**: Use `XFORMERS_DISABLE_FLASH_ATTN=1` flag
- **Memory issues**: Reduce batch size or use CPU mode for testing

## Run

Default mode now uses VGGT for 3D reconstruction:
```bash
python run.py --depth vggt
```

You can also use other depth estimation methods:
- `--depth vggt` (default): VGGT - fastest and most accurate
- `--depth metric`: Metric3D
- `--depth da`: Depth Anything
- `--depth dust`: Dust3r/Mast3r (requires uncommenting imports in misc/three_d.py)

### Command Line Options

```bash
python run.py [OPTIONS]

Options:
  --headless              Don't show GUI
  --depth DEPTH           vggt / metric / da / dust
  --renderer RENDERER     raster / py3d
  --prompt PROMPT         me / doors / auto / combo / default
  --control CONTROL       me / doors / auto
  --intrinsics INTRINSICS dummy / pf
  --image IMAGE           gen / path
  --model MODEL           sd2 / if
```

### Examples

```bash
# Interactive mode with VGGT
python run.py --depth vggt --prompt auto --control auto

# Headless mode for dataset generation
python run.py --headless --depth vggt --prompt auto --control auto

# Use specific image as starting point
python run.py --image path/to/your/image.jpg --depth vggt
```

