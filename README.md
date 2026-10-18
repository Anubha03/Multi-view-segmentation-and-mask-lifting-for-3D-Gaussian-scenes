# Multi-view segmentation and mask lifting for 3D Gaussian scenes

**Team:** Anubha Vyasamudri (av3329) · Sanjita Chandan Ballapur (sb5216)
**Course:** COMS 4732W — Computer Vision, Columbia University, Spring 2025

---

## Overview

This codebase implements a training free pipeline for semantically segmenting
objects in pretrained 3D Gaussian Splatting (3DGS) scenes using improved
multiview mask lifting.

Given a pretrained scene and a text prompt (e.g. `"potted plant"`), the pipeline:

1. Renders the scene from N=18 orbit cameras (3 elevation rings: 25°, 5°, -10°)
2. Runs GroundingDINO and SAM2 on each render to get 2D masks
3. Erodes masks by 4px to remove boundary ambiguity
4. Lifts masks to 3D using **depth-aware, radius-aware** voting
5. Thresholds with Otsu's method plus spatial and color consistency filters
6. Applies a downstream operation: **recolor**, **remove**, or **NL instruction edit**

### Key algorithmic improvements over naive lifting

| Improvement | Problem it solves |
|---|---|
| Multiview cameras (3 elevation rings) | Single ring misses top and base surfaces |
| Depth-aware lifting | Occluded Gaussians accumulate false votes |
| Radius-aware voting | Edge Gaussians miss the mask by 1px |
| Visibility normalisation | Centrally-placed Gaussians are over-counted |
| Mask erosion (r=4px) | Boundary pixels are ambiguous |
| Otsu threshold | Fixed percentile is arbitrary |
| Color consistency filter | Background sneaks through spatial filter |

## Prerequisites

- macOS with Apple Silicon (M1 / M2 / M3), macOS 12.3+
- Python 3.10 or later
- ~8 GB free disk space
- Xcode command-line tools: `xcode-select --install`

---

## Quick Start

### 1. One-shot setup

```bash
bash setup.sh
```

This creates a virtual environment, installs all dependencies, downloads
SAM2 and GroundingDINO checkpoints, and downloads the bonsai and bicycle
scenes from the Mip-NeRF 360 dataset.

### 2. Activate environment

```bash
source .venv/bin/activate
```

### 3. Run the pipeline

```bash
# Recolor the bonsai tree red
python scripts/run_pipeline.py \
    --scene bonsai \
    --prompt "potted plant" \
    --edit recolor \
    --recolor_rgb 1.0 0.1 0.05

# Recolor the bicycle blue
python scripts/run_pipeline.py \
    --scene bicycle \
    --prompt "bicycle" \
    --edit recolor \
    --recolor_rgb 0.05 0.55 1.0

# Remove the bicycle from the scene
python scripts/run_pipeline.py \
    --scene bicycle \
    --prompt "bicycle" \
    --edit remove

# Natural language instruction edit
python scripts/run_pipeline.py \
    --scene bonsai \
    --instruction "change the flowers to green"
```

---

## Project Structure

```
cv2_extra_credit/
├── setup.sh                    one-shot environment and data setup
├── requirements.txt            Python dependencies
├── configs/
│   └── default.yaml            all pipeline hyperparameters
└── scripts/
    ├── run_pipeline.py         main entry point
    ├── gaussian_scene.py       PLY loader and PyTorch rasteriser (MPS)
    ├── camera_utils.py         camera loading from cameras.json / COLMAP
    ├── segmentation.py         GroundingDINO and SAM2 wrappers
    ├── mask_lifting.py         improved multiview lifting (all 7 improvements)
    ├── evaluate.py             CLIP similarity and LPIPS metrics
    ├── visualize.py            orbit GIFs and comparison images
    └── fetch_cameras.py        downloads COLMAP cameras for Mip-NeRF 360
```

---

## Configuration

All hyperparameters are in `configs/default.yaml`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `render.n_views_per_elevation` | 6 | Cameras per elevation ring (18 total) |
| `render.elevation_angles` | [25, 5, -10] | Three elevation rings in degrees |
| `lifting.depth_rel_tol` | 0.22 | Depth filter tolerance (22% of surface depth) |
| `lifting.mask_erode_radius` | 4 | Mask erosion in pixels before lifting |
| `lifting.radius_weight` | true | Enable radius-aware voting |
| `lifting.normalise` | true | Enable visibility normalisation |
| `segmentation.grounding_dino.box_threshold` | 0.28 | Detection confidence |

---

## Output Structure

```
outputs/{scene}/{prompt}/
├── comparison.png          original / mask overlay / edited strip
├── {scene}_before.gif      orbit animation before edit
├── {scene}_{edit}.gif      orbit animation after edit
├── score_distribution.png  lifting score histogram
└── renders/                per-view renders and 2D masks
```

---

## Troubleshooting

**`MPS not available`**: Requires macOS 12.3+ on an M-series chip.

**`No .ply found`**: Run `bash scripts/get_ply.sh <scene>` to download the scene,
or set `scenes_dir` in `configs/default.yaml` to your scenes folder.

**Out of memory**: Reduce `render.n_views_per_elevation` from 6 to 4 in config.

**SAM2 import error**: Run `pip install -e .` inside `third_party/sam2/`.
