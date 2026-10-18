"""
evaluate.py
───────────
Evaluation metrics for the segmentation pipeline:

  1. CLIP Similarity  – cosine similarity between text prompt embedding
                        and CLIP image embedding of novel-view renders
                        of the segmented region.

  2. LPIPS            – perceptual distance between original and edited renders
                        in NON-segmented regions (scene preservation).

  3. mIoU             – (optional) intersection-over-union against a ground-truth
                        mask file, e.g., from the LERF figurines dataset.

Usage from CLI:
    python evaluate.py --scene garden --prompt "the plant" \
                       [--gt_mask path/to/gt.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from rich.console import Console
from rich.table import Table

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
#  CLIP similarity
# ──────────────────────────────────────────────────────────────────────────────

class CLIPEvaluator:
    def __init__(self, model_name: str = "ViT-B/32", device: torch.device = torch.device("cpu")):
        import clip
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=str(device))
        self.model.eval()

    @torch.no_grad()
    def image_features(self, img_rgb: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 → normalised CLIP image feature (1, D)."""
        pil = Image.fromarray(img_rgb)
        x   = self.preprocess(pil).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x).float()
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def text_features(self, text: str) -> torch.Tensor:
        """Text string → normalised CLIP text feature (1, D)."""
        import clip
        tokens = clip.tokenize([text]).to(self.device)
        feat   = self.model.encode_text(tokens).float()
        return F.normalize(feat, dim=-1)

    def similarity(self, img_rgb: np.ndarray, text: str) -> float:
        img_f  = self.image_features(img_rgb)
        text_f = self.text_features(text)
        return (img_f * text_f).sum().item()


def eval_clip_similarity(
    renders: List[np.ndarray],   # list of (H, W, 3) uint8 rendered images
    masks:   List[np.ndarray],   # list of (H, W) bool masks (segmented region)
    prompt:  str,
    model_name: str = "ViT-B/32",
    device:  torch.device = torch.device("cpu"),
) -> float:
    """
    Compute mean CLIP similarity between `prompt` and novel-view renders
    of the *segmented region only* (background zeroed).
    """
    evaluator = CLIPEvaluator(model_name, device)
    sims = []
    for img, mask in zip(renders, masks):
        # Black out non-segmented region
        img_seg = img.copy()
        img_seg[~mask] = 0
        sim = evaluator.similarity(img_seg, prompt)
        sims.append(sim)
    mean_sim = float(np.mean(sims)) if sims else 0.0
    console.log(f"  CLIP similarity (segmented region): [bold]{mean_sim:.4f}[/bold]")
    return mean_sim


# ──────────────────────────────────────────────────────────────────────────────
#  LPIPS scene preservation
# ──────────────────────────────────────────────────────────────────────────────

class LPIPSEvaluator:
    def __init__(self, device: torch.device = torch.device("cpu")):
        import lpips
        # Use CPU for LPIPS regardless of device (MPS can have issues with VGG)
        self.device = torch.device("cpu")
        self.loss_fn = lpips.LPIPS(net="vgg").to(self.device)
        self.loss_fn.eval()

    def to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """(H, W, 3) uint8 → (1, 3, H, W) float32 in [-1, 1]."""
        x = torch.from_numpy(img).float() / 127.5 - 1.0
        return x.permute(2, 0, 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def distance(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        non_seg_mask: Optional[np.ndarray] = None,
    ) -> float:
        """LPIPS perceptual distance between two renders.

        If non_seg_mask provided (H, W bool), only consider those pixels.
        """
        if non_seg_mask is not None:
            img_a = img_a.copy(); img_b = img_b.copy()
            # Zero out segmented region before computing
            img_a[~non_seg_mask] = 128
            img_b[~non_seg_mask] = 128

        ta = self.to_tensor(img_a)
        tb = self.to_tensor(img_b)
        return self.loss_fn(ta, tb).item()


def eval_lpips_preservation(
    renders_before: List[np.ndarray],
    renders_after:  List[np.ndarray],
    seg_masks:      List[np.ndarray],
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Mean LPIPS between before/after edits in the *non-segmented* region.
    Lower = better scene preservation.
    """
    evaluator = LPIPSEvaluator(device)
    dists = []
    for before, after, mask in zip(renders_before, renders_after, seg_masks):
        non_seg = ~mask
        d = evaluator.distance(before, after, non_seg_mask=non_seg)
        dists.append(d)
    mean_lpips = float(np.mean(dists)) if dists else 0.0
    console.log(f"  LPIPS (non-segmented region): [bold]{mean_lpips:.4f}[/bold]  (lower = better)")
    return mean_lpips


# ──────────────────────────────────────────────────────────────────────────────
#  mIoU (vs ground-truth mask)
# ──────────────────────────────────────────────────────────────────────────────

def eval_miou(
    pred_mask:  np.ndarray,    # (H, W) bool – lifted 3D mask projected to 2D
    gt_mask:    np.ndarray,    # (H, W) bool – ground-truth
) -> float:
    """Binary intersection-over-union."""
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    intersection = (pred & gt).sum()
    union        = (pred | gt).sum()
    iou = float(intersection) / float(union + 1e-8)
    console.log(f"  mIoU (vs ground truth): [bold]{iou:.4f}[/bold]")
    return iou


def project_seg_mask_to_2d(
    scene,          # GaussianScene
    cam,            # Camera
) -> np.ndarray:
    """
    Project the 3D segmentation mask to a 2D image for mIoU evaluation.
    Each pixel is True if any segmented Gaussian alpha-composites to it.
    """
    from gaussian_scene import render_view, GaussianScene, Camera
    import torch

    device = scene.device
    H, W   = cam.height, cam.width

    uv, z = cam.project(scene.xyz)
    px = uv[:, 0].round().long()
    py = uv[:, 1].round().long()
    valid = (z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)

    mask_2d = torch.zeros(H, W, dtype=torch.bool, device=device)
    seg_valid = scene.seg_mask & valid
    py_s = py[seg_valid].clamp(0, H - 1)
    px_s = px[seg_valid].clamp(0, W - 1)
    mask_2d[py_s, px_s] = True

    return mask_2d.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(
    scene_dir:       Path,
    prompt:          str,
    cfg:             dict,
    device:          torch.device,
    renders_before:  Optional[List[np.ndarray]] = None,
    renders_after:   Optional[List[np.ndarray]] = None,
    seg_masks_2d:    Optional[List[np.ndarray]] = None,
    gt_mask_path:    Optional[Path] = None,
) -> dict:
    """Run all available evaluations and return a metrics dict."""
    metrics = {}

    # CLIP
    if renders_after and seg_masks_2d:
        clip_model = cfg["evaluation"]["clip_model"]
        metrics["clip_similarity"] = eval_clip_similarity(
            renders_after, seg_masks_2d, prompt, clip_model, device
        )

    # LPIPS
    if renders_before and renders_after and seg_masks_2d:
        metrics["lpips_preservation"] = eval_lpips_preservation(
            renders_before, renders_after, seg_masks_2d, device
        )

    # mIoU
    if gt_mask_path and gt_mask_path.exists() and seg_masks_2d:
        gt_mask = np.array(Image.open(str(gt_mask_path)).convert("L")) > 127
        # Use the first eval view projection
        metrics["miou"] = eval_miou(seg_masks_2d[0], gt_mask)

    # Pretty-print
    table = Table(title="Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold green")
    for k, v in metrics.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3DGS segmentation")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--gt_mask", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cpu"))
    console.print(f"Running evaluation for scene=[cyan]{args.scene}[/cyan], prompt=[green]{args.prompt}[/green]")
    # Full evaluation is orchestrated by run_pipeline.py; this script is for standalone use.
    console.print("[yellow]Run `python scripts/run_pipeline.py` for full evaluation.[/yellow]")
