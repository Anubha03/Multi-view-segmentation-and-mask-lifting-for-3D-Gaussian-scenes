"""
run_pipeline.py
───────────────
End-to-end orchestration:

  1. Load pre-trained 3DGS scene (.ply)
  2. Load cameras (cameras.json / COLMAP)
  3. Run multi-view Grounded-SAM2 (or SAM3) lifting
  4. Apply downstream operations (recolor / remove)
  5. Evaluate (CLIP, LPIPS, optional mIoU)
  6. Generate orbit GIFs (before / after)

Usage:
    python scripts/run_pipeline.py \
        --scene garden \
        --prompt "the plant" \
        --edit recolor \
        --recolor_rgb 0.2 0.8 0.2

    python scripts/run_pipeline.py \
        --scene kitchen \
        --prompt "the kettle" \
        --edit remove \
        --backbone sam3
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from rich.console import Console

console = Console()

# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_device(cfg: dict) -> torch.device:
    d = cfg.get("device", "cpu")
    if d == "mps":
        if not torch.backends.mps.is_available():
            console.log("[yellow]MPS not available – using CPU[/yellow]")
            d = "cpu"
        else:
            console.log("[green]Using MPS (Apple Silicon)[/green]")
    return torch.device(d)


def _find_ply(scene_dir: Path) -> Path:
    """Search for the .ply file in the standard 3DGS directory layout."""
    candidates = [
        scene_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply",
        scene_dir / "point_cloud" / "iteration_7000"  / "point_cloud.ply",
        *sorted(scene_dir.rglob("*.ply")),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No .ply file found in {scene_dir}.\n"
        "Make sure the scene was downloaded by setup.sh."
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    scene_name:  str,
    prompt:      str,
    cfg:         dict,
    device:      torch.device,
    edit:        str = "recolor",      # "recolor" | "remove" | "none"
    recolor_rgb: tuple = None,
    gt_mask_path: Path = None,
    out_root:    Path = Path("outputs"),
) -> dict:
    """
    Full pipeline. Returns a metrics dict.
    """
    # ── Import all modules (deferred to keep startup fast) ────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from gaussian_scene import GaussianScene, render_view
    from camera_utils import load_cameras
    from segmentation import build_segmenter
    from mask_lifting import run_lifting_pipeline
    from evaluate import (
        eval_clip_similarity, eval_lpips_preservation,
        eval_miou, project_seg_mask_to_2d, run_evaluation,
    )
    from visualize import render_orbit_gif, overlay_mask, make_comparison_strip

    t_total = time.time()
    out_dir = out_root / scene_name / prompt.replace(" ", "_")[:40]
    out_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(exist_ok=True)

    # ── 1. Load scene ─────────────────────────────────────────────────────────
    console.rule("[bold]Stage 1 – Load Scene[/bold]")
    scene_dir = Path(cfg["scenes_dir"]) / scene_name
    ply_path  = _find_ply(scene_dir)
    scene     = GaussianScene(ply_path, device)

    # ── 2. Load cameras ───────────────────────────────────────────────────────
    console.rule("[bold]Stage 2 – Load Cameras[/bold]")
    cameras = load_cameras(
        scene_dir,
        device = device,
        max_cameras = max(cfg["render"].get("num_views", 24),
                         cfg["evaluation"]["num_eval_views"] + 5),
    )

    if not cameras:
        console.log("  [yellow]No camera files found – generating orbit cameras from scene bounds[/yellow]")
        import math
        from camera_utils import make_orbit_cameras
        W = cfg["render"]["image_width"]
        H = cfg["render"]["image_height"]
        fov_deg = 60.0
        fx = fy = 0.5 * W / math.tan(math.radians(fov_deg / 2))
        center = scene.xyz.mean(dim=0)
        dists  = (scene.xyz - center).norm(dim=-1)
        radius = float(dists.quantile(0.85)) * cfg["orbit"]["radius_scale"]
        n_cams = max(cfg["render"].get("num_views", 24),
                     cfg["evaluation"]["num_eval_views"] + 5)
        cameras = make_orbit_cameras(
            center=center, radius=radius,
            num_frames=n_cams, elevation_deg=cfg["orbit"]["elevation_deg"],
            fx=fx, fy=fy, width=W, height=H, device=device,
        )
        console.log(f"  Generated [bold]{len(cameras)}[/bold] orbit cameras (radius={radius:.2f})")
    else:
        console.log(f"  Loaded [bold]{len(cameras)}[/bold] cameras")

    lift_cameras  = cameras[:cfg["render"].get("num_views", 24)]
    eval_cameras  = cameras[:cfg["evaluation"]["num_eval_views"]]

    # ── 3. Build segmenter ────────────────────────────────────────────────────
    console.rule("[bold]Stage 3 – Build Segmenter[/bold]")
    segmenter = build_segmenter(cfg, device)

    # ── 4. Render before-edit views (for LPIPS and GIF) ──────────────────────
    console.rule("[bold]Stage 4 – Pre-edit Renders[/bold]")
    background = torch.tensor(cfg["render"]["background"], device=device)
    renders_before = []
    for cam in eval_cameras:
        r = render_view(scene, cam, background=background)
        renders_before.append((r["rgb"].cpu().numpy() * 255).astype(np.uint8))
    console.log(f"  Rendered {len(renders_before)} pre-edit views")

    # ── 5. Multi-view mask lifting ────────────────────────────────────────────
    console.rule("[bold]Stage 5 – Multi-view Mask Lifting[/bold]")
    t_lift = time.time()
    run_lifting_pipeline(
        scene, lift_cameras, segmenter, prompt, cfg,
        render_dir=renders_dir,
    )
    t_lift = time.time() - t_lift
    console.log(f"  Lifting done in {t_lift:.1f}s")
    n_seg = int(scene.seg_mask.sum().item())

    # ── 6. Downstream edit ───────────────────────────────────────────────────
    console.rule(f"[bold]Stage 6 – Edit: {edit}[/bold]")
    # Save a copy of the scene for "after" renders
    from copy import deepcopy

    scene_edited = _copy_scene_minimal(scene)

    if edit == "recolor":
        rgb_target = recolor_rgb or tuple(cfg["editing"]["recolor_rgb"])
        scene_edited.recolor(rgb_target)
    elif edit == "remove":
        scene_edited.remove_segmented()
    else:
        console.log("  No edit applied (none).")

    # Save edited .ply
    edited_ply = out_dir / f"{scene_name}_{edit}.ply"
    scene_edited.save_ply(edited_ply)

    # ── 7. Post-edit renders ──────────────────────────────────────────────────
    console.rule("[bold]Stage 7 – Post-edit Renders[/bold]")
    renders_after = []
    seg_masks_2d  = []
    for cam in eval_cameras:
        r = render_view(scene_edited, cam, background=background)
        renders_after.append((r["rgb"].cpu().numpy() * 255).astype(np.uint8))
        seg_masks_2d.append(project_seg_mask_to_2d(scene_edited, cam))
    console.log(f"  Rendered {len(renders_after)} post-edit views")

    # Save comparison strip for first eval view
    if renders_before and renders_after:
        before_overlay = overlay_mask(renders_before[0], seg_masks_2d[0] if seg_masks_2d else np.zeros_like(renders_before[0][:,:,0], dtype=bool))
        make_comparison_strip(
            [renders_before[0], before_overlay, renders_after[0]],
            ["Before", "Segmentation", f"After ({edit})"],
            out_dir / "comparison.png",
        )

    # ── 8. Evaluation ────────────────────────────────────────────────────────
    console.rule("[bold]Stage 8 – Evaluation[/bold]")
    metrics = run_evaluation(
        scene_dir    = scene_dir,
        prompt       = prompt,
        cfg          = cfg,
        device       = device,
        renders_before = renders_before,
        renders_after  = renders_after,
        seg_masks_2d   = seg_masks_2d,
        gt_mask_path   = gt_mask_path,
    )
    metrics["n_gaussians_segmented"] = n_seg
    metrics["fraction_segmented"]    = n_seg / scene.num_gaussians
    metrics["lifting_runtime_s"]     = t_lift
    metrics["total_runtime_s"]       = time.time() - t_total

    # Save metrics JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # ── 9. Orbit GIFs ────────────────────────────────────────────────────────
    console.rule("[bold]Stage 9 – Orbit GIFs[/bold]")
    render_orbit_gif(
        scene, cfg,
        out_path       = out_dir / f"{scene_name}_before.gif",
        label          = f"{scene_name} – original",
        background     = background,
        reference_cams = cameras,
    )
    render_orbit_gif(
        scene_edited, cfg,
        out_path       = out_dir / f"{scene_name}_{edit}.gif",
        label          = f"{scene_name} – {edit}: {prompt}",
        background     = background,
        reference_cams = cameras,
    )

    console.rule("[bold green]Pipeline complete[/bold green]")
    console.log(f"  Total time: {metrics['total_runtime_s']:.1f}s")
    console.log(f"  Output dir: [cyan]{out_dir}[/cyan]")

    return metrics


def _copy_scene_minimal(scene):
    """Lightweight scene copy preserving the segmentation mask."""
    from gaussian_scene import GaussianScene
    new = object.__new__(GaussianScene)
    new.device   = scene.device
    new.ply_path = scene.ply_path
    new.xyz      = scene.xyz.clone()
    new.opacity  = scene.opacity.clone()
    new.scales   = scene.scales.clone()
    new.quats    = scene.quats.clone()
    new.sh_dc    = scene.sh_dc.clone()
    new.sh_rest  = scene.sh_rest.clone() if scene.sh_rest is not None else None
    new.seg_mask = scene.seg_mask.clone()
    return new


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="3DGS Semantic Segmentation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scene",       required=True,
                        help="Scene name (must be downloaded by setup.sh), e.g. garden")
    parser.add_argument("--prompt",      required=True,
                        help="Text prompt for the object, e.g. 'the plant'")
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--backbone",    choices=["grounded_sam2", "sam3"],
                        default=None, help="Override backbone from config")
    parser.add_argument("--edit",        choices=["recolor", "remove", "none"],
                        default="recolor")
    parser.add_argument("--recolor_rgb", nargs=3, type=float, metavar=("R", "G", "B"),
                        default=None, help="RGB target for recolor (0-1 range)")
    parser.add_argument("--gt_mask",     default=None,
                        help="Path to ground-truth mask image for mIoU")
    parser.add_argument("--out_dir",     default="outputs")
    parser.add_argument("--num_views",   type=int, default=None,
                        help="Override number of views for lifting")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.backbone:
        cfg["segmentation"]["backbone"] = args.backbone
    if args.num_views:
        cfg["render"]["num_views"] = args.num_views

    device = _get_device(cfg)

    console.print(f"\n[bold]3DGS Semantic Segmentation Pipeline[/bold]")
    console.print(f"  Scene:    [cyan]{args.scene}[/cyan]")
    console.print(f"  Prompt:   [green]{args.prompt}[/green]")
    console.print(f"  Backbone: [yellow]{cfg['segmentation']['backbone']}[/yellow]")
    console.print(f"  Edit:     [magenta]{args.edit}[/magenta]")
    console.print(f"  Device:   [white]{device}[/white]\n")

    recolor_rgb = tuple(args.recolor_rgb) if args.recolor_rgb else None
    gt_mask_path = Path(args.gt_mask) if args.gt_mask else None

    metrics = run_pipeline(
        scene_name   = args.scene,
        prompt       = args.prompt,
        cfg          = cfg,
        device       = device,
        edit         = args.edit,
        recolor_rgb  = recolor_rgb,
        gt_mask_path = gt_mask_path,
        out_root     = Path(args.out_dir),
    )

    console.print("\n[bold green]Done![/bold green]")
    for k, v in metrics.items():
        if isinstance(v, float):
            console.print(f"  {k}: {v:.4f}")
        else:
            console.print(f"  {k}: {v}")


if __name__ == "__main__":
    main()