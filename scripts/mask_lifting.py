"""
mask_lifting.py
---------------
Improved multi-view mask lifting for 3D Gaussian Splatting segmentation.

Implements seven improvements over naive alpha-weighted projection:

  [1] Multi-elevation cameras     Complete top/side/base coverage
  [2] Depth-aware lifting         Filters occluded Gaussians via Z-buffer
  [3] Radius-aware voting         Partial credit for edge Gaussians
  [4] Visibility normalisation    Corrects for view-count bias
  [5] Mask erosion                Removes boundary-ambiguous pixels
  [6] Otsu threshold              Finds natural foreground/background split
  [7] Color consistency filter    Final post-filter removes background noise

Reference: Farquhar et al. style depth-aware lifting adapted for 3DGS.
See report Section 3.2 for full algorithmic description.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from rich.console import Console
from rich.progress import track

console = Console()


# =============================================================================
# Mask preprocessing
# =============================================================================

def erode_masks(masks: np.ndarray, radius: int = 4) -> np.ndarray:
    """
    Erode 2D binary masks to remove boundary-ambiguous pixels [Improvement 5].

    SAM2 mask edges are noisy. Pixels right at the detected object boundary
    straddle the foreground/background boundary; lifting them causes "halo"
    Gaussians to be selected. Erosion ensures only clearly-interior pixels vote.

    Args:
        masks:  (V, H, W) bool array of per-view masks.
        radius: Erosion radius in pixels. Default 4px.

    Returns:
        Eroded masks, same shape.
    """
    struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    eroded = np.stack([
        binary_erosion(m, structure=struct) if m.any() else m
        for m in masks
    ])
    n_before = masks.any(axis=(1, 2)).sum()
    n_after  = eroded.any(axis=(1, 2)).sum()
    console.log(f"  Mask erosion r={radius}: {n_before} → {n_after} views with content")
    return eroded


# =============================================================================
# Otsu threshold
# =============================================================================

def otsu_threshold(scores: np.ndarray) -> float:
    """
    Otsu's method on non-zero lifting scores [Improvement 6].

    Finds the threshold that maximally separates foreground (object) and
    background Gaussian score populations by minimising within-class variance.
    More principled than a fixed percentile because it adapts to the actual
    score distribution rather than assuming a fixed fraction of Gaussians
    belong to the object.

    Args:
        scores: Per-Gaussian lifting scores. Only non-zero values are used.

    Returns:
        Optimal threshold float.
    """
    nz = scores[scores > 0]
    if len(nz) < 2:
        return float("inf")

    hist, edges = np.histogram(nz, bins=256)
    total = hist.sum()
    cum_w = np.cumsum(hist)
    cum_m = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2)
    gm    = cum_m[-1]

    best_var, best_t = -1.0, edges[0]
    for i in range(1, len(hist)):
        w0 = cum_w[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        m0  = cum_m[i] / w0
        m1  = (gm * total - cum_m[i]) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var = var
            best_t   = edges[i]

    return float(best_t)


# =============================================================================
# Color consistency filter
# =============================================================================

def color_consistency_filter(
    seg: np.ndarray,
    colors: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Remove Gaussians whose color is far from the segment's median [Improvement 7].

    After lifting, background Gaussians at the right depth (e.g. cloth behind
    the bonsai, bench behind the bicycle) can still receive enough votes to
    pass the score threshold. This filter exploits the fact that object and
    background colors are usually distinct.

    Args:
        seg:    (N,) bool — current segment mask.
        colors: (N, 3) float32 — per-Gaussian RGB in [0, 1].
        sigma:  Maximum L2 color distance from the segment's median centroid.
                Tight (0.22) for bonsai; loose (0.32) for multicolor bicycle.

    Returns:
        Refined (N,) bool mask.
    """
    if seg.sum() == 0:
        return seg

    centroid = np.median(colors[seg], axis=0)          # (3,)
    dist     = np.linalg.norm(colors - centroid, axis=1)  # (N,)
    refined  = seg & (dist < sigma)

    # Safety: if we deleted more than 85% of the segment, skip
    if refined.sum() < 0.15 * seg.sum():
        console.log("  [yellow]Color filter too aggressive — skipping[/yellow]")
        return seg

    n_removed = int(seg.sum() - refined.sum())
    console.log(
        f"  Color filter σ<{sigma:.2f}: removed {n_removed:,} off-color Gaussians "
        f"(centroid ≈ {centroid.round(2)})"
    )
    return refined


# =============================================================================
# Depth-aware, radius-aware multi-view lifting
# =============================================================================

class ImprovedLifter:
    """
    Depth-aware, radius-aware, visibility-normalised multi-view lifting.

    Improvements over naive lifting:
      [2] Depth filter: only vote for Gaussians on the visible surface.
      [3] Radius voting: partial credit if the projected Gaussian ellipse
          overlaps the mask, even when its centre is just outside.
      [4] Visibility normalisation: score divided by view-count per Gaussian.

    Usage:
        lifter = ImprovedLifter(positions, scales, opacities, cfg)
        for cam, mask, depth_map in zip(cameras, masks, depth_maps):
            lifter.accumulate(cam, mask, depth_map)
        scores = lifter.finalise()
    """

    def __init__(
        self,
        positions: np.ndarray,   # (N, 3)
        scales:    np.ndarray,   # (N, 3)
        opacities: np.ndarray,   # (N,)
        cfg:       dict,
    ):
        self.positions = positions
        self.scales    = scales
        self.opacities = opacities
        self.cfg       = cfg

        N = len(positions)
        self.scores  = np.zeros(N, dtype=np.float32)
        self.vis_cnt = np.zeros(N, dtype=np.float32)  # in-frustum count

        # Pre-compute max scale per Gaussian (for projected radius)
        self.max_scale = scales.max(axis=1)            # (N,)

        self._ph = np.concatenate(                     # homogeneous positions
            [positions, np.ones((N, 1), dtype=np.float32)], axis=1
        )

    def accumulate(
        self,
        viewmat: np.ndarray,     # (4, 4) world-to-camera
        K:       np.ndarray,     # (3, 3) pinhole intrinsics
        mask:    np.ndarray,     # (H, W) bool  (already eroded)
        depth_map: np.ndarray,   # (H, W) float  Z-buffer surface depth
        H: int,
        W: int,
    ) -> None:
        """Accumulate one view's evidence."""
        cfg_lift  = self.cfg["lifting"]
        depth_tol = cfg_lift["depth_rel_tol"]
        do_radius = cfg_lift.get("radius_weight", True)

        R, t = viewmat[:3, :3], viewmat[:3, 3]

        # Project Gaussian centres to camera space
        cam   = self._ph[:, :3] @ R.T + t             # (N, 3)
        cam_z = cam[:, 2]
        valid = cam_z > 0.05                           # in front of camera

        ds = np.where(valid, cam_z, 1.0)
        px = (K[0, 0] * cam[:, 0] / ds + K[0, 2]).astype(np.int32)
        py = (K[1, 1] * cam[:, 1] / ds + K[1, 2]).astype(np.int32)

        in_bounds = valid & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        idx = np.where(in_bounds)[0]
        if len(idx) == 0:
            return

        self.vis_cnt[idx] += 1  # [4] visibility count

        # [2] Depth filter — skip Gaussians behind the visible surface
        surf_z   = depth_map[py[idx], px[idx]]
        z_ratio  = np.abs(cam_z[idx] - surf_z) / np.maximum(surf_z, 0.01)
        on_surf  = (surf_z > 0.01) & (z_ratio < depth_tol)
        sidx     = idx[on_surf]
        if len(sidx) == 0:
            return

        # [3] Radius-aware voting via distance transform
        mask_f = mask.astype(np.float32)
        if do_radius:
            dt  = distance_transform_edt(1.0 - mask_f)       # dist from mask inside
            r   = (K[0, 0] * self.max_scale[sidx] / cam_z[sidx]).clip(1, 30)
            cv  = mask_f[py[sidx], px[sidx]]                  # centre vote
            de  = dt[py[sidx], px[sidx]]
            rb  = np.clip(1.0 - de / r, 0, 1)                # radius bonus
            vote = np.maximum(cv, rb * 0.5)
        else:
            vote = mask_f[py[sidx], px[sidx]]

        self.scores[sidx] += self.opacities[sidx] * vote.astype(np.float32)

    def finalise(self) -> np.ndarray:
        """Return visibility-normalised per-Gaussian scores (N,)."""
        # [4] Normalise by visibility count
        norm   = np.maximum(self.vis_cnt, 1.0)
        scores = self.scores / norm
        nz     = (scores > 0).sum()
        console.log(f"  Gaussians with score > 0: {nz:,}  ({100*nz/len(scores):.1f}%)")
        return scores


# =============================================================================
# Full lifting + threshold pipeline
# =============================================================================

def lift_and_segment(
    positions:  np.ndarray,           # (N, 3)
    scales:     np.ndarray,           # (N, 3)
    opacities:  np.ndarray,           # (N,)
    colors:     np.ndarray,           # (N, 3)
    viewmats:   np.ndarray,           # (V, 4, 4)
    Ks:         np.ndarray,           # (V, 3, 3)
    masks_raw:  np.ndarray,           # (V, H, W) bool  (pre-erosion)
    depth_maps: np.ndarray,           # (V, H, W) float Z-buffer
    center:     np.ndarray,           # (3,) scene centre
    cfg:        dict,
    scene_key:  str = "default",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete improved lifting pipeline.

    Steps:
      1. Erode masks                   [Improvement 5]
      2. Accumulate depth-aware votes  [Improvements 2, 3, 4]
      3. Otsu threshold                [Improvement 6]
      4. Spatial filter (Y, XZ)
      5. Color consistency filter      [Improvement 7]

    Returns:
        seg    (N,) bool — final segment mask
        scores (N,) float — raw normalised scores (for diagnostics)
    """
    H, W = masks_raw.shape[1], masks_raw.shape[2]
    N    = len(positions)

    # [5] Erode masks
    erode_r = cfg["lifting"].get("mask_erode_radius", 4)
    masks   = erode_masks(masks_raw, radius=erode_r)

    # Accumulate votes
    lifter = ImprovedLifter(positions, scales, opacities, cfg)
    for v in track(range(len(viewmats)), description="Lifting views"):
        lifter.accumulate(viewmats[v], Ks[v], masks[v], depth_maps[v], H, W)
    scores = lifter.finalise()

    nz = scores[scores > 0]
    if len(nz) == 0:
        console.log("[red]  No non-zero scores — check segmentation output[/red]")
        return np.zeros(N, dtype=bool), scores

    # [6] Otsu threshold with fallback guards
    thresh    = otsu_threshold(scores)
    seg_score = scores >= thresh
    console.log(f"  Otsu threshold = {thresh:.5f} → {seg_score.sum():,} Gaussians")

    cfg_lift = cfg["lifting"]
    if seg_score.mean() > 0.30:
        p = cfg_lift.get("otsu_fallback_high_pctile", 75)
        thresh    = float(np.percentile(nz, p))
        seg_score = scores >= thresh
        console.log(f"  Too loose — fallback p{p} = {thresh:.5f} → {seg_score.sum():,}")

    if seg_score.sum() < 300:
        p = cfg_lift.get("otsu_fallback_low_pctile", 50)
        thresh    = float(np.percentile(nz, p))
        seg_score = scores >= thresh
        console.log(f"  Too tight — fallback p{p} = {thresh:.5f} → {seg_score.sum():,}")

    # Spatial filter (Y-down convention: smaller Y = physically higher)
    spatial_cfg = cfg["lifting"].get("spatial", {})
    scene_spatial = spatial_cfg.get(scene_key, spatial_cfg.get("default", {}))

    if scene_spatial:
        y_pct = scene_spatial.get("y_pctile", 55)
        y_max = float(np.percentile(positions[seg_score, 1], y_pct))
        y_max = np.clip(y_max,
                        scene_spatial.get("y_min_clamp", -0.5),
                        scene_spatial.get("y_max_clamp",  1.0))
        seg   = seg_score & (positions[:, 1] < y_max)
        console.log(f"  Y filter (Y < {y_max:.3f}): {seg.sum():,}")

        if "xz_radius" in scene_spatial:
            xz   = np.sqrt((positions[:, 0] - center[0]) ** 2 +
                            (positions[:, 2] - center[2]) ** 2)
            seg  = seg & (xz < scene_spatial["xz_radius"])
            console.log(f"  XZ filter (< {scene_spatial['xz_radius']}): {seg.sum():,}")

        if "brightness_min" in scene_spatial:
            bright = colors.mean(axis=1)
            seg    = seg & (bright > scene_spatial["brightness_min"])
            console.log(
                f"  Brightness filter (> {scene_spatial['brightness_min']}): {seg.sum():,}"
            )
    else:
        seg = seg_score

    # [7] Color consistency filter
    color_sigma_cfg = cfg["lifting"].get("color_sigma", {})
    sigma = color_sigma_cfg.get(scene_key, color_sigma_cfg.get("default", 0.28))
    seg   = color_consistency_filter(seg, colors, sigma)

    # Auto-relax if over-filtered
    if seg.sum() < 300:
        console.log("  [yellow]Too few after filters — using score-only segment[/yellow]")
        seg = seg_score

    frac = 100 * seg.mean()
    console.log(f"  [green]Final: {seg.sum():,} / {N:,} Gaussians  ({frac:.2f}%)[/green]")
    return seg, scores
