"""
visualize.py
────────────
Visualisation utilities:
  - Orbit GIF  (before and after editing)
  - Mask overlay on rendered images  (segmented Gaussians highlighted)
  - Side-by-side comparison strips
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import torch
from rich.console import Console

from gaussian_scene import GaussianScene, render_view
from camera_utils import make_orbit_cameras, Camera

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
#  Orbit GIF
# ──────────────────────────────────────────────────────────────────────────────

def render_orbit_gif(
    scene:          GaussianScene,
    cfg:            dict,
    out_path:       Path,
    label:          str = "",
    background:     Optional[torch.Tensor] = None,
    reference_cams: Optional[List[Camera]] = None,
) -> None:
    """Render an orbit around the scene centre and save as a GIF.

    Args:
        scene:          Scene to render.
        cfg:            Pipeline config dict.
        out_path:       Output .gif path.
        label:          Text label drawn on each frame (e.g. "before" / "after").
        background:     (3,) float tensor for background colour.
        reference_cams: If provided, use their average position to determine
                        orbit radius and centre; otherwise fall back to scene bounds.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    orbit_cfg = cfg["orbit"]
    render_cfg = cfg["render"]
    device = scene.device

    # Determine scene centre and orbit radius
    if reference_cams:
        cam_positions = torch.stack([c.world_center for c in reference_cams], dim=0)
        center = cam_positions.mean(dim=0)
        dists  = (cam_positions - center).norm(dim=-1)
        radius = float(dists.mean()) * orbit_cfg["radius_scale"]
    else:
        center = scene.xyz.mean(dim=0)
        dists  = (scene.xyz - center).norm(dim=-1)
        radius = float(dists.quantile(0.85)) * orbit_cfg["radius_scale"]

    # Approximate focal lengths from config
    W = render_cfg["image_width"]
    H = render_cfg["image_height"]
    fov_deg = 60.0
    fx = fy = 0.5 * W / math.tan(math.radians(fov_deg / 2))

    # Override focal if we have reference cameras
    if reference_cams:
        fx = reference_cams[0].fx
        fy = reference_cams[0].fy
        W  = reference_cams[0].width
        H  = reference_cams[0].height

    cameras = make_orbit_cameras(
        center       = center,
        radius       = radius,
        num_frames   = orbit_cfg["num_frames"],
        elevation_deg= orbit_cfg["elevation_deg"],
        fx=fx, fy=fy,
        width=W, height=H,
        device=device,
    )

    if background is None:
        background = torch.tensor(cfg["render"]["background"], device=device)

    frames = []
    console.log(f"Rendering orbit GIF ({len(cameras)} frames) → {out_path.name}")
    for i, cam in enumerate(cameras):
        result = render_view(scene, cam, background=background)
        frame_f = result["rgb"].cpu().numpy()           # (H, W, 3) float
        frame   = (frame_f * 255).astype(np.uint8)      # (H, W, 3) uint8

        # Burn in label
        if label:
            frame = _burn_label(frame, f"{label}  [frame {i+1}/{len(cameras)}]")

        frames.append(frame)

    imageio.mimwrite(
        str(out_path),
        frames,
        fps=orbit_cfg["fps"],
        loop=0,
        format="GIF",
    )
    console.log(f"  Saved [cyan]{out_path}[/cyan]")


# ──────────────────────────────────────────────────────────────────────────────
#  Mask overlay
# ──────────────────────────────────────────────────────────────────────────────

def overlay_mask(
    render_rgb: np.ndarray,   # (H, W, 3) uint8
    mask_2d:    np.ndarray,   # (H, W) bool
    color:      tuple = (255, 80, 80),
    alpha:      float = 0.45,
) -> np.ndarray:
    """Blend a coloured mask overlay onto a render."""
    out = render_rgb.astype(np.float32).copy()
    overlay = np.zeros_like(out)
    overlay[mask_2d] = color
    out = out * (1 - alpha) + overlay * alpha
    return out.clip(0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
#  Comparison strip
# ──────────────────────────────────────────────────────────────────────────────

def make_comparison_strip(
    renders: List[np.ndarray],   # list of (H, W, 3) uint8
    labels:  List[str],
    out_path: Path,
) -> None:
    """Save a horizontal strip of renders with text labels."""
    assert len(renders) == len(labels), "Need one label per render"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labelled = [_burn_label(r, l) for r, l in zip(renders, labels)]
    strip = np.concatenate(labelled, axis=1)
    imageio.imwrite(str(out_path), strip)
    console.log(f"  Saved comparison strip → [cyan]{out_path}[/cyan]")


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _burn_label(img: np.ndarray, text: str, margin: int = 8) -> np.ndarray:
    """Draw a white text label in the top-left corner using PIL (no cv2 dep)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # Try to load a basic font; fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()
        # Shadow for readability
        draw.text((margin + 1, margin + 1), text, fill=(0, 0, 0), font=font)
        draw.text((margin, margin), text, fill=(255, 255, 255), font=font)
        return np.array(pil)
    except Exception:
        # If PIL draw fails, return unchanged
        return img
