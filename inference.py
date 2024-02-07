#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import svg
import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Float, install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from PIL import Image
from pytorch3d.transforms import Rotate, Translate, quaternion_to_matrix
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import Dataset, default_collate
from torchvision import transforms

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.dataset_re10k import DatasetRE10k, DatasetRE10kCfg
    from src.dataset.types import Stage
    from src.dataset.view_sampler import get_view_sampler
    from src.dataset.view_sampler.view_sampler_arbitrary import \
        ViewSamplerArbitraryCfg
    from src.geometry.projection import homogenize_points, project
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.decoder.cuda_splatting import render_cuda_orthographic
    from src.model.encoder import (Encoder, EncoderEpipolar,
                                   EncoderEpipolarCfg, get_encoder)
    from src.model.encoder.visualization.encoder_visualizer import \
        EncoderVisualizer
    from src.model.encoder.visualization.encoder_visualizer_epipolar import \
        EncoderVisualizerEpipolar
    from src.model.model_wrapper import ModelWrapper
    from src.model.ply_export import export_ply
    from src.paper.common import encode_image, save_svg
    from src.visualization.color_map import apply_color_map_to_image
    from src.visualization.colors import get_distinct_color
    from src.visualization.drawing.cameras import unproject_frustum_corners
    from src.visualization.drawing.lines import draw_lines
    from src.visualization.drawing.points import draw_points


SCENE = "2177ca3a775a9ee9"
CONTEXT_INDICES = (135, 195)
QUERIES = (
    # x, y
    (238, 168),  # sofa pillow corner
    (238, 80),  # painting corner
    (159, 195),  # plant leaves
    (227, 277),  # carpet corner
    (300, 80),  # random spot on wall
)
QUERIES = tuple((x / 400, y / 400) for x, y in QUERIES)
LAYER = 1
HEAD = 2
IMAGE_SHAPE = (256, 256)
FIGURE_WIDTH = 240
MARGIN = 4
LINE_WIDTH = 4
RAY_RADIUS = 2
RAY_BACKER_RADIUS = 2.5

SCENES = (
    # scene, context 1, context 2, far plane
    # ("fc60dbb610046c56", 28, 115, 10.0),d7c9abc0b221c799
    # ("1eca36ec55b88fe4", 0, 120, 3.0, [110]), # teaser fig.
    ("1", 0, 0, 20.0, [60]),
    ("2", 0, 0, 20.0, [90]),
    ("3", 0, 0, 20.0, [120]),
    ("4", 0, 0, 20.0, [150]),
    ("5", 0, 0, 20.0, [180]),
)
SCENES = (
    # scene, context 1, context 2, far plane
    # ("fc60dbb610046c56", 28, 115, 10.0),d7c9abc0b221c799
    # ("1eca36ec55b88fe4", 0, 120, 3.0, [110]), # teaser fig.
    ("2c52d9d606a3ece2", 87, 112, 35.0, [105]),
    ("71a1121f817eb913", 139, 164, 10.0, [65]),
    ("d70fc3bef87bffc1", 67, 92, 10.0, [60]),
    ("f0feab036acd7195", 44, 69, 25.0, [125]),
    ("a93d9d0fd69071aa", 57, 82, 15.0, [60]),
)
FIGURE_WIDTH = 500
MARGIN = 4
GAUSSIAN_TRIM = 8
LINE_WIDTH = 2
LINE_COLOR = [0, 0, 0]
POINT_DENSITY = 0.5

ENCODERS = {"epipolar": (EncoderEpipolar, EncoderVisualizerEpipolar)}
EncoderCfg = EncoderEpipolarCfg

DATASETS: dict[str, Dataset] = {"re10k": DatasetRE10k,}
DatasetCfg = DatasetRE10kCfg

def to_hex(color: Float[Tensor, "3"]) -> str:
    r, g, b = color.tolist()
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def generate_attention_figure(cfg_dict):
    # File path
    json_file_path = '/mnt/sda/epic-fields/P01_04.json'
    image_dir = "/mnt/sda/epic-kitchens/og/frames_rgb_flow/rgb/train/P01/P01_04"

    with open(json_file_path) as f:
        a_cam = json.load(f)

    width, height = a_cam['camera']['width'], a_cam['camera']['height']
    fx, fy, cx, cy, *_ = a_cam['camera']['params']
    intrinsics_matrix = torch.tensor([[fx / height, 0, cx / width], # TODO broken intrinsics
                                                [0, fy / height, cy / height],
                                                [0, 0, 1]])
    extrinsics = []
    images = []
    image_transform = transforms.ToTensor()
    indices = [0, 1, 2, 3, 4]  # Modify as needed
    for frame_number in indices:
        frame_name = f"frame_{(frame_number+1)*15:010}.jpg"
        params = a_cam['images'][frame_name]

        # Processing extrinsics
        qw, qx, qy, qz, tx, ty, tz = params
        quaternion = torch.tensor([qw, qx, qy, qz])
        rotation_matrix = quaternion_to_matrix(quaternion)
        extrinsic_matrix = torch.eye(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = torch.tensor([tx, ty, tz])
        extrinsics.append(torch.linalg.inv(extrinsic_matrix) * torch.tensor([1, -1, -1, 1]).reshape(4, 1))

        # Loading images
        image_path = f"{image_dir}/{frame_name}"
        image = Image.open(image_path)
        image_tensor = image_transform(image).unsqueeze(0).unsqueeze(0)
        images.append(image_tensor)

    # Stack images along view dimension
    padding = int((width - height) / 2)
    image_tensor = torch.cat(images, dim=1)[..., padding:-padding] # TODO check if need modify intrinsics
    extrinsics_tensor = torch.stack(extrinsics)

    # Compute near and far bounds
    #points = a_cam['points']
    #distances = [np.linalg.norm(np.array(point[:3])) for point in points]
    near = 0.9 #min(distances)
    far = 33.5 #max(distances)

    # Building the final dictionary structure
    example = {
        "context": {
            "image": image_tensor[:, [0, -1], :, :, :],
            "intrinsics": intrinsics_matrix.unsqueeze(0).repeat(1, 2, 1, 1),
            "extrinsics": extrinsics_tensor.unsqueeze(0)[:, [0, -1], :, :],
            "near": torch.tensor([[near, near]]),
            "far": torch.tensor([[far, far]])
        },
        "target": {
            "image": image_tensor[:, [1,2,3], :, :, :],
            "intrinsics": intrinsics_matrix.unsqueeze(0).repeat(1, 3, 1, 1),
            "extrinsics": extrinsics_tensor.unsqueeze(0)[:, [1,2,3], :, :],
            "near": torch.tensor([[near, near, near]]),
            "far": torch.tensor([[far, far, far]])
        }
    }

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    #global cfg
    #cfg = new_cfg
    torch.manual_seed(cfg_dict.seed)
    device = torch.device('cuda:0') # cpu

    # Prepare the checkpoint for loading.
    #checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    checkpoint_path = './checkpoints/re10k.ckpt'

    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    encoder, encoder_visualizer = ENCODERS[cfg.model.encoder.name]
    encoder = encoder(cfg.model.encoder)
    if encoder_visualizer is not None:
        encoder_visualizer = encoder_visualizer(cfg.model.encoder.visualizer, encoder)
    #encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_wrapper = ModelWrapper.load_from_checkpoint(
        checkpoint_path,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder, #get_decoder(cfg.model.decoder, cfg.dataset),
        losses=[],
        step_tracker=None,
    ).to(device)
    model_wrapper.eval()
    #dataset = iter(get_dataset(cfg.dataset, "test", None))

    # Create a dataset that always returns the desired scene.
    #view_sampler_cfg = ViewSamplerArbitraryCfg(
    #    "arbitrary",
    #    2,
    #    2,
    #    context_views=list(CONTEXT_INDICES),
    #    target_views=list(CONTEXT_INDICES),
    #)
    #cfg.dataset.view_sampler = view_sampler_cfg
    #cfg.dataset.overfit_to_scene = SCENE

    # Get the scene.
    view_sampler = get_view_sampler(
        cfg.dataset.view_sampler,
        "test",
        cfg.dataset.overfit_to_scene is not None,
        cfg.dataset.cameras_are_circular,
        None,
    )
    dataset = DATASETS[cfg.dataset.name](cfg.dataset, "test", view_sampler)
    #dataset = iter(dataset)
    #dataset = get_dataset(cfg.dataset, "test", None)

    old_example = default_collate([next(iter(dataset))])
    old_example = apply_to_collection(old_example, Tensor, lambda x: x.to(device))
    example = apply_to_collection(example, Tensor, lambda x: x.to(device))
    example = old_example
    #breakpoint()

    # GENERATE POINT CLOUD FIG
    #'''
    for idx, (scene, *context_indices, far, angles) in enumerate(SCENES):
        #far = 33
        # Create a dataset that always returns the desired scene.
        #view_sampler_cfg = ViewSamplerArbitraryCfg(
        #    "arbitrary",
        #    2,
        #    2,
        #    context_views=list(context_indices),
        #    target_views=[0, 0],  # use [40, 80] for teaser
        #)
        #cfg.dataset.view_sampler = view_sampler_cfg
        #cfg.dataset.overfit_to_scene = scene

        # Get the scene.
        #dataset = get_dataset(cfg.dataset, "test", None)
        #example = default_collate([next(iter(dataset))])
        #example = apply_to_collection(example, Tensor, lambda x: x.to(device))

        # Generate the Gaussians.
        visualization_dump = {}
        gaussians = encoder.forward(
            example["context"], False, visualization_dump=visualization_dump
        )

        # Figure out which Gaussians to mask off/throw away.
        _, _, _, h, w = example["context"]["image"].shape

        # Transform means into camera space.
        means = rearrange(
            gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=2, h=h, w=w
        )
        means = homogenize_points(means)
        w2c = example["context"]["extrinsics"].inverse()[0]
        means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3]

        # Create a mask to filter the Gaussians. First, throw away Gaussians at the
        # borders, since they're generally of lower quality.
        mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
        mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        # Then, drop Gaussians that are really far away.
        mask = mask & (means[..., 2] < far)

        def trim(element):
            element = rearrange(
                element, "() (v h w spp) ... -> h w spp v ...", v=2, h=h, w=w
            )
            return element[mask][None]

        for angle in angles:
            # Define the pose we render from.
            pose = torch.eye(4, dtype=torch.float32, device=device)
            rotation = R.from_euler("xyz", [-15, angle - 90, 0], True).as_matrix()
            pose[:3, :3] = torch.tensor(rotation, dtype=torch.float32, device=device)
            translation = torch.eye(4, dtype=torch.float32, device=device)
            # visual balance, 0.5x pyramid/frustum volume
            translation[2, 3] = far * (0.5 ** (1 / 3))
            pose = translation @ pose

            ones = torch.ones((1,), dtype=torch.float32, device=device)
            render_args = {
                "extrinsics": example["context"]["extrinsics"][0, :1] @ pose,
                "width": ones * far * 2,
                "height": ones * far * 2,
                "near": ones * 0,
                "far": ones * far,
                "image_shape": (256, 256),
                "background_color": torch.zeros(
                    (1, 3), dtype=torch.float32, device=device
                ),
                "gaussian_means": trim(gaussians.means),
                "gaussian_covariances": trim(gaussians.covariances),
                "gaussian_sh_coefficients": trim(gaussians.harmonics),
                "gaussian_opacities": trim(gaussians.opacities),
            }

            # Render alpha (opacity).
            dump = {}
            alpha_args = {
                **render_args,
                "gaussian_sh_coefficients": torch.ones_like(
                    render_args["gaussian_sh_coefficients"][..., :1]
                ),
                "use_sh": False,
            }
            alpha = render_cuda_orthographic(**alpha_args, dump=dump)[0]

            # Render (premultiplied) color.
            color = render_cuda_orthographic(**render_args)[0]

            # Render depths. Without modifying the renderer, we can only render
            # premultiplied depth, then hackily transform it into straight alpha depth,
            # which is needed for sorting.
            depth = render_args["gaussian_means"] - dump["extrinsics"][0, :3, 3]
            depth = depth.norm(dim=-1)
            depth_args = {
                **render_args,
                "gaussian_sh_coefficients": repeat(depth, "() g -> () g c ()", c=3),
                "use_sh": False,
            }
            depth_premultiplied = render_cuda_orthographic(**depth_args)
            depth = (depth_premultiplied / alpha).nan_to_num(posinf=1e10, nan=1e10)[0]

            # Save the rendering for later depth-based alpha compositing.
            layers = [(color, alpha, depth)]

            # Figure out the intrinsics from the FOV.
            fx = 0.5 / (0.5 * dump["fov_x"]).tan()
            fy = 0.5 / (0.5 * dump["fov_y"]).tan()
            dump_intrinsics = torch.eye(3, dtype=torch.float32, device=device)
            dump_intrinsics[0, 0] = fx
            dump_intrinsics[1, 1] = fy
            dump_intrinsics[:2, 2] = 0.5

            # Compute frustum corners for the context views.
            frustum_corners = unproject_frustum_corners(
                example["context"]["extrinsics"][0],
                example["context"]["intrinsics"][0],
                torch.ones((2,), dtype=torch.float32, device=device) * far / 8,
            )
            camera_origins = example["context"]["extrinsics"][0, :, :3, 3]

            # Generate the 3D lines that have to be computed.
            lines = []
            for corners, origin in zip(frustum_corners, camera_origins):
                for i in range(4):
                    lines.append((corners[i], corners[i - 1]))
                    lines.append((corners[i], origin))

            # Generate an alpha compositing layer for each line.
            for a, b in lines:
                # Start with the point whose depth is further from the camera.
                a_depth = (dump["extrinsics"].inverse() @ homogenize_points(a))[..., 2]
                b_depth = (dump["extrinsics"].inverse() @ homogenize_points(b))[..., 2]
                start = a if (a_depth > b_depth).all() else b
                end = b if (a_depth > b_depth).all() else a

                # Create the alpha mask (this one is clean).
                start_2d = project(start, dump["extrinsics"], dump_intrinsics)[0][0]
                end_2d = project(end, dump["extrinsics"], dump_intrinsics)[0][0]
                alpha = draw_lines(
                    torch.zeros_like(color),
                    start_2d[None],
                    end_2d[None],
                    (1, 1, 1),
                    LINE_WIDTH,
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                # Create the color.
                lc = torch.tensor(LINE_COLOR, dtype=torch.float32, device=device)
                color = draw_lines(
                    torch.zeros_like(color),
                    start_2d[None],
                    end_2d[None],
                    lc,
                    LINE_WIDTH,
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                # Create the depth. We just individually render points.
                wh = torch.tensor((w, h), dtype=torch.float32, device=device)
                delta = (wh * (start_2d - end_2d)).norm()
                num_points = delta / POINT_DENSITY
                t = torch.linspace(0, 1, int(num_points) + 1, device=device)
                xyz = start[None] * t[:, None] + end[None] * (1 - t)[:, None]
                depth = (xyz - dump["extrinsics"][0, :3, 3]).norm(dim=-1)
                depth = repeat(depth, "p -> p c", c=3)
                xy = project(xyz, dump["extrinsics"], dump_intrinsics)[0]
                depth = draw_points(
                    torch.ones_like(color) * 1e10,
                    xy,
                    depth,
                    LINE_WIDTH,  # makes it 2x as wide as line
                    x_range=(0, 1),
                    y_range=(0, 1),
                )

                layers.append((color, alpha, depth))

            # Do the alpha compositing.
            canvas = torch.ones_like(color)
            colors = torch.stack([x for x, _, _ in layers])
            alphas = torch.stack([x for _, x, _ in layers])
            depths = torch.stack([x for _, _, x in layers])
            index = depths.argsort(dim=0)
            colors = colors.gather(index=index, dim=0)
            alphas = alphas.gather(index=index, dim=0)
            t = (1 - alphas).cumprod(dim=0)
            t = torch.cat([torch.ones_like(t[:1]), t[:-1]], dim=0)
            image = (t * colors).sum(dim=0)
            total_alpha = (t * alphas).sum(dim=0)
            image = total_alpha * image + (1 - total_alpha) * canvas

            base = Path(f"point_clouds/{idx:0>6}_{scene}")
            save_image(image, f"{base}_angle_{angle:0>3}.png")

            # Render depth.
            *_, h, w = example["context"]["image"].shape
            rendered = decoder.forward(
                gaussians,
                example["context"]["extrinsics"],
                example["context"]["intrinsics"],
                example["context"]["near"],
                example["context"]["far"],
                (h, w),
                "depth",
            )

            export_ply(
                example["context"]["extrinsics"][0, 0],
                trim(gaussians.means)[0],
                trim(visualization_dump["scales"])[0],
                trim(visualization_dump["rotations"])[0],
                trim(gaussians.harmonics)[0],
                trim(gaussians.opacities)[0],
                base / "gaussians.ply",
            )

            result = rendered.depth
            depth_near = result[result > 0].quantile(0.01).log()
            depth_far = result.quantile(0.99).log()
            result = result.log()
            result = 1 - (result - depth_near) / (depth_far - depth_near)
            result = apply_color_map_to_image(result, "turbo")
            save_image(result[0, 0], f"{base}_depth_0.png")
            save_image(result[0, 1], f"{base}_depth_1.png")
            a = 1
        a = 1
    a = 1
    #'''

    # GENERATE EPIPOLAR LINES FIGURE
    '''
    # Run the encoder with hooks to capture the attention output.
    softmax_weights = []

    def hook(module, input, output):
        softmax_weights.append(output)

    handles = [
        layer[0].fn.attend.register_forward_hook(hook)
        for layer in encoder.epipolar_transformer.transformer.layers
    ]
    visualization_dump = {}
    encoder.forward(example["context"], False, visualization_dump=visualization_dump)
    for handle in handles:
        handle.remove()

    attention = torch.stack(softmax_weights)
    sampling = visualization_dump["sampling"]
    context_images = example["context"]["image"]

    # Pick a random batch element, view, and other view.
    _, _, _, h, w = context_images.shape
    ds = cfg.model.encoder.epipolar_transformer.downscale
    wh = torch.tensor((w // ds, h // ds), dtype=torch.float32, device=device)
    queries = torch.tensor(QUERIES, dtype=torch.float32, device=device) * wh
    col, row = queries.type(torch.int64).unbind(dim=-1)
    rr = row * (w // ds) + col

    b, v, _, r, s, _ = sampling.xy_sample.shape
    rb = 0
    rv = 0
    rov = 0

    # Visualize attention in the sample view.
    attention = rearrange(attention, "l (b v r) hd () s -> l b v r hd s", b=b, v=v, r=r)
    attention = attention[:, rb, rv, rr, :, :]

    # Create colors according to attention.
    color = [get_distinct_color(i) for i, _ in enumerate(rr)]
    color = torch.tensor(color, device=attention.device)
    color = rearrange(color, "r c -> r () c")
    attn = rearrange(attention[LAYER, :, HEAD], "r s -> r s ()")
    attn = attn / reduce(attn, "r s () -> r () ()", "max")

    left_image = context_images[rb, rv]
    right_image = context_images[rb, encoder.sampler.index_v[rv, rov]]

    # Generate the SVG.
    # Create an SVG canvas.
    image_width = (FIGURE_WIDTH - MARGIN) / 2
    image_height = image_width * IMAGE_SHAPE[0] / IMAGE_SHAPE[1]
    fig = svg.SVG(
        width=FIGURE_WIDTH,
        height=image_height,
        elements=[],
        viewBox=svg.ViewBoxSpec(0, 0, FIGURE_WIDTH, image_height),
    )

    # Draw the left image.
    left_image = svg.Image(
        width=image_width,
        height=image_height,
        href=encode_image(left_image, "jpeg"),
    )
    fig.elements.append(left_image)

    # Draw the right image.
    right_image = svg.Image(
        width=image_width,
        height=image_height,
        href=encode_image(right_image, "jpeg"),
        x=image_width + MARGIN,
    )
    fig.elements.append(right_image)

    # Create a mask for the epipolar line.
    mask = svg.Mask(
        elements=[svg.Rect(width=FIGURE_WIDTH, height=image_height, fill="white")],
        id="mask",
        maskUnits="userSpaceOnUse",
    )
    fig.elements.append(mask)

    scale = torch.tensor(
        (image_width, image_height), dtype=torch.float32, device=device
    )
    for rrv_idx, rrv in enumerate(rr):
        # Draw the sample.
        ray_xy = (sampling.xy_ray[rb, rv, rrv] * scale).tolist()
        ray = svg.Circle(
            cx=ray_xy[0],
            cy=ray_xy[1],
            r=RAY_BACKER_RADIUS,
            fill="#000000",
        )
        fig.elements.append(ray)
        ray = svg.Circle(
            cx=ray_xy[0],
            cy=ray_xy[1],
            r=RAY_RADIUS,
            fill=to_hex(color[rrv_idx, 0]),
        )
        fig.elements.append(ray)

        # Draw the epipolar line.
        start = (sampling.xy_sample_near[rb, rv, rov, rrv, 0] * scale).tolist()
        start[0] += image_width + MARGIN
        end = (sampling.xy_sample_far[rb, rv, rov, rrv, -1] * scale).tolist()
        end[0] += image_width + MARGIN
        epipolar_line = svg.Line(
            x1=2 * start[0] - end[0],  # extra length that gets clipped
            y1=2 * start[1] - end[1],  # extra length that gets clipped
            x2=end[0],
            y2=end[1],
            stroke="#000000",
            stroke_width=LINE_WIDTH,
            mask="url(#mask)",
        )
        fig.elements.append(epipolar_line)

        # Draw lines for attention.
        for sv in range(s):
            start = (sampling.xy_sample_near[rb, rv, rov, rrv, sv] * scale).tolist()
            start[0] += image_width + MARGIN
            end = (sampling.xy_sample_far[rb, rv, rov, rrv, sv] * scale).tolist()
            end[0] += image_width + MARGIN
            epipolar_line = svg.Line(
                x1=start[0],
                y1=start[1],
                x2=end[0],
                y2=end[1],
                stroke=to_hex((color * attn)[rrv_idx, sv]),
                stroke_width=LINE_WIDTH,
                mask="url(#mask)",
            )
            fig.elements.append(epipolar_line)

    save_svg(fig, Path("attention.svg"))
    '''

    # INPUT MAPPING
    '''
    (Pdb) example.keys()
    dict_keys(['context', 'target', 'scene'])
    (Pdb) example['context'].keys()
    dict_keys(['extrinsics', 'intrinsics', 'image', 'near', 'far', 'index'])
    (Pdb) example['target'].keys()
    dict_keys(['extrinsics', 'intrinsics', 'image', 'near', 'far', 'index'])
    (Pdb) example['scene']
    ['5aca87f95a9412c6']

    (Pdb) example['target']['near']
    tensor([[0.1193, 0.1193, 0.1193]], device='cuda:0')
    (Pdb) example['target']['far']
    tensor([[1193.0017, 1193.0017, 1193.0017]], device='cuda:0')
    (Pdb) example['target']['image'].shape
    torch.Size([1, 3, 3, 256, 256])
    (Pdb) example['context']['image'].shape
    torch.Size([1, 2, 3, 256, 256])

    (Pdb) example['context']['intrinsics'].shape
    torch.Size([1, 2, 3, 3])
    (Pdb) example['context']['extrinsics'].shape
    torch.Size([1, 2, 4, 4])

    (Pdb) example['target']['intrinsics'].shape
    torch.Size([1, 3, 3, 3])
    (Pdb) example['target']['extrinsics'].shape
    torch.Size([1, 3, 4, 4])
    '''


if __name__ == "__main__":
    with torch.no_grad():
        #generate_point_cloud_figure()
        generate_attention_figure()
