# # """
# # gaussian_scene.py
# # ─────────────────
# # Loads a pre-trained 3D Gaussian Splatting scene from a .ply file and provides:
# #   - Differentiable rasterisation via a pure-PyTorch (MPS-friendly) renderer
# #   - Alpha-map extraction per view (needed for multi-view mask lifting)
# #   - Per-Gaussian attribute access (position, colour, opacity, scale, rotation)
# #   - Downstream edit operations: recolor, remove

# # The renderer is a simplified tile-based alpha-compositing rasteriser that runs
# # entirely in PyTorch, so it works on MPS without any CUDA extensions.
# # """

# # from __future__ import annotations

# # import math
# # from pathlib import Path
# # from typing import Optional

# # import numpy as np
# # import torch
# # import torch.nn.functional as F
# # from plyfile import PlyData
# # from rich.console import Console

# # console = Console()

# # # ──────────────────────────────────────────────────────────────────────────────
# # #  Spherical Harmonics helpers (degree-0 DC only for colour)
# # # ──────────────────────────────────────────────────────────────────────────────
# # SH_C0 = 0.28209479177387814   # 1 / (2 * sqrt(pi))


# # def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
# #     """Convert SH degree-0 (DC) coefficients → linear RGB in [0, 1]."""
# #     return (sh_dc * SH_C0 + 0.5).clamp(0.0, 1.0)


# # def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
# #     """Convert linear RGB → SH degree-0 DC coefficients."""
# #     return (rgb - 0.5) / SH_C0


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  Camera
# # # ──────────────────────────────────────────────────────────────────────────────
# # class Camera:
# #     """Simple pinhole camera with world→camera extrinsics."""

# #     def __init__(
# #         self,
# #         R: torch.Tensor,          # (3, 3)  world→cam rotation
# #         t: torch.Tensor,          # (3,)    world→cam translation
# #         fx: float, fy: float,
# #         cx: float, cy: float,
# #         width: int, height: int,
# #         device: torch.device,
# #     ):
# #         self.R = R.to(device)
# #         self.t = t.to(device)
# #         self.fx, self.fy = fx, fy
# #         self.cx, self.cy = cx, cy
# #         self.width = width
# #         self.height = height
# #         self.device = device

# #     @property
# #     def world_center(self) -> torch.Tensor:
# #         """Camera centre in world coordinates:  -R^T @ t"""
# #         return -(self.R.T @ self.t)

# #     def project(self, xyz_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
# #         """Project world-space points → pixel coords and per-point depths.

# #         Args:
# #             xyz_world: (N, 3) world-space positions.

# #         Returns:
# #             uv: (N, 2) pixel coordinates (float).
# #             z:  (N,)   depth (camera-space z).
# #         """
# #         # World → Camera
# #         xyz_cam = (xyz_world @ self.R.T) + self.t  # (N, 3)
# #         z = xyz_cam[:, 2].clamp(min=1e-6)          # depth

# #         u = self.fx * xyz_cam[:, 0] / z + self.cx
# #         v = self.fy * xyz_cam[:, 1] / z + self.cy
# #         uv = torch.stack([u, v], dim=-1)            # (N, 2)
# #         return uv, z


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  GaussianScene
# # # ──────────────────────────────────────────────────────────────────────────────
# # class GaussianScene:
# #     """Holds all per-Gaussian attributes loaded from a .ply file."""

# #     def __init__(self, ply_path: Path, device: torch.device):
# #         self.device = device
# #         self.ply_path = Path(ply_path)
# #         self._load_ply()

# #     # ── Loading ───────────────────────────────────────────────────────────────

# #     def _load_ply(self):
# #         console.log(f"Loading scene: [cyan]{self.ply_path}[/cyan]")
# #         ply = PlyData.read(str(self.ply_path))
# #         vertex = ply["vertex"]

# #         def col(name: str) -> torch.Tensor:
# #             return torch.tensor(np.array(vertex[name]), dtype=torch.float32)

# #         # Positions  (N, 3)
# #         self.xyz = torch.stack([col("x"), col("y"), col("z")], dim=-1).to(self.device)
# #         N = self.xyz.shape[0]

# #         # Opacity logit → sigmoid  (N,)
# #         self.opacity = torch.sigmoid(col("opacity")).to(self.device)

# #         # Scale (log-space in file) → exp  (N, 3)
# #         scales = torch.stack([col(f"scale_{i}") for i in range(3)], dim=-1)
# #         self.scales = torch.exp(scales).to(self.device)

# #         # Rotation quaternion (N, 4)  wxyz convention
# #         self.quats = F.normalize(
# #             torch.stack([col(f"rot_{i}") for i in range(4)], dim=-1),
# #             dim=-1
# #         ).to(self.device)

# #         # SH features: at minimum DC (degree 0) per channel
# #         sh_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_dc")])
# #         if sh_names:
# #             sh_dc = torch.stack([col(n) for n in sh_names], dim=-1)  # (N, 3)
# #         else:
# #             # Fallback: try "red","green","blue" attributes (some exporters)
# #             try:
# #                 sh_dc = torch.stack([col("red"), col("green"), col("blue")], dim=-1).float() / 255.0
# #                 sh_dc = rgb_to_sh(sh_dc)
# #             except Exception:
# #                 sh_dc = torch.zeros(N, 3)
# #         self.sh_dc = sh_dc.to(self.device)  # (N, 3)

# #         # Higher-order SH (stored but not used in rendering here for speed)
# #         sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest")])
# #         if sh_rest_names:
# #             self.sh_rest = torch.stack([col(n) for n in sh_rest_names], dim=-1).to(self.device)
# #         else:
# #             self.sh_rest = None

# #         console.log(f"  Loaded [bold]{N:,}[/bold] Gaussians")

# #         # Segmentation mask (all False initially)
# #         self.seg_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

# #     # ── Properties ────────────────────────────────────────────────────────────

# #     @property
# #     def num_gaussians(self) -> int:
# #         return self.xyz.shape[0]

# #     @property
# #     def rgb(self) -> torch.Tensor:
# #         """Per-Gaussian colour from DC SH coefficients. (N, 3)"""
# #         return sh_to_rgb(self.sh_dc)

# #     # ── Edit operations ───────────────────────────────────────────────────────

# #     def recolor(self, rgb: tuple[float, float, float]):
# #         """Set all segmented Gaussians to a fixed RGB colour (in-place)."""
# #         if not self.seg_mask.any():
# #             console.log("[yellow]No Gaussians segmented – recolor skipped[/yellow]")
# #             return
# #         target = torch.tensor(rgb, dtype=torch.float32, device=self.device)
# #         self.sh_dc[self.seg_mask] = rgb_to_sh(target).unsqueeze(0)
# #         # Zero out higher-order SH for recolored Gaussians to avoid colour bleeding
# #         if self.sh_rest is not None:
# #             self.sh_rest[self.seg_mask] = 0.0
# #         console.log(f"  Recolored [bold]{self.seg_mask.sum().item():,}[/bold] Gaussians → RGB{rgb}")

# #     def remove_segmented(self):
# #         """Remove all segmented Gaussians from the scene (in-place)."""
# #         keep = ~self.seg_mask
# #         n_removed = self.seg_mask.sum().item()
# #         self.xyz = self.xyz[keep]
# #         self.opacity = self.opacity[keep]
# #         self.scales = self.scales[keep]
# #         self.quats = self.quats[keep]
# #         self.sh_dc = self.sh_dc[keep]
# #         if self.sh_rest is not None:
# #             self.sh_rest = self.sh_rest[keep]
# #         self.seg_mask = self.seg_mask[keep]
# #         console.log(f"  Removed [bold]{n_removed:,}[/bold] Gaussians – "
# #                     f"[bold]{self.num_gaussians:,}[/bold] remaining")

# #     def save_ply(self, out_path: Path):
# #         """Write the current scene back to a .ply file."""
# #         from plyfile import PlyElement, PlyData
# #         out_path = Path(out_path)
# #         out_path.parent.mkdir(parents=True, exist_ok=True)

# #         xyz = self.xyz.cpu().numpy()
# #         opacity_logit = torch.logit(self.opacity.clamp(1e-6, 1 - 1e-6)).cpu().numpy()
# #         scales_log = torch.log(self.scales.clamp(min=1e-8)).cpu().numpy()
# #         quats = self.quats.cpu().numpy()
# #         sh_dc = self.sh_dc.cpu().numpy()

# #         dtype_list = [
# #             ("x", "f4"), ("y", "f4"), ("z", "f4"),
# #             ("opacity", "f4"),
# #             ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
# #             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
# #             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
# #         ]
# #         N = xyz.shape[0]
# #         arr = np.empty(N, dtype=dtype_list)
# #         arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
# #         arr["opacity"] = opacity_logit
# #         for i in range(3):
# #             arr[f"scale_{i}"] = scales_log[:, i]
# #         for i in range(4):
# #             arr[f"rot_{i}"] = quats[:, i]
# #         for i in range(3):
# #             arr[f"f_dc_{i}"] = sh_dc[:, i]

# #         el = PlyElement.describe(arr, "vertex")
# #         PlyData([el]).write(str(out_path))
# #         console.log(f"  Saved scene → [cyan]{out_path}[/cyan]")


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  Pure-PyTorch Gaussian Rasteriser  (MPS-compatible)
# # # ──────────────────────────────────────────────────────────────────────────────

# # def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
# #     """(N, 4) wxyz quaternion → (N, 3, 3) rotation matrices."""
# #     w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
# #     R = torch.stack([
# #         1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
# #         2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x),
# #         2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y),
# #     ], dim=-1).reshape(-1, 3, 3)
# #     return R


# # def _compute_2d_covariance(
# #     scales: torch.Tensor,    # (N, 3)
# #     quats: torch.Tensor,     # (N, 4)
# #     cam: Camera,
# # ) -> torch.Tensor:
# #     """3D world-space covariance:  Sigma_world = R_g S S^T R_g^T

# #     Returns (N, 3, 3). Caller is responsible for rotating into camera space
# #     and applying the perspective Jacobian.
# #     """
# #     R_g  = _quat_to_rotmat(quats)          # (N, 3, 3)  Gaussian rotation
# #     S    = torch.diag_embed(scales)         # (N, 3, 3)  scale diagonal
# #     RS   = R_g @ S                          # (N, 3, 3)
# #     return RS @ RS.transpose(-1, -2)        # (N, 3, 3)  world-space covariance


# # # def render_view(
# # #     scene: GaussianScene,
# # #     cam: Camera,
# # #     background: Optional[torch.Tensor] = None,
# # #     return_alpha_map: bool = False,
# # #     max_gaussians: int = 50_000,
# # # ) -> dict:
# # #     """Render a single view using a vectorised alpha-compositing rasteriser.

# # #     This is *not* as fast as the CUDA rasteriser but is fully MPS/CPU compatible.

# # #     Returns dict with keys:
# # #         "rgb"        (H, W, 3) float32 rendered image
# # #         "alpha_map"  (N, H, W) float32  per-Gaussian alpha contribution  [optional]
# # #         "depth"      (H, W)    float32
# # #     """
# # #     device = scene.device
# # #     H, W = cam.height, cam.width

# # #     if background is None:
# # #         background = torch.ones(3, device=device)

# # #     # ── 1. Project all Gaussians to 2D ────────────────────────────────────────
# # #     uv, z = cam.project(scene.xyz)                    # (N,2), (N,)

# # #     # Cull Gaussians behind camera or outside a generous image margin
# # #     margin = max(H, W) * 0.5
# # #     valid = (z > 0.1) & \
# # #             (uv[:, 0] > -margin) & (uv[:, 0] < W + margin) & \
# # #             (uv[:, 1] > -margin) & (uv[:, 1] < H + margin)

# # #     # Limit for memory (take highest-opacity Gaussians first)
# # #     if valid.sum() > max_gaussians:
# # #         opac_valid = scene.opacity.clone()
# # #         opac_valid[~valid] = -1.0
# # #         _, top_idx = opac_valid.topk(max_gaussians)
# # #         mask2 = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=device)
# # #         mask2[top_idx] = True
# # #         valid = valid & mask2

# # #     idx = torch.where(valid)[0]                       # indices of visible Gaussians
# # #     uv_v = uv[idx]                                    # (M, 2)
# # #     z_v = z[idx]                                      # (M,)
# # #     opac_v = scene.opacity[idx]                       # (M,)
# # #     rgb_v = scene.rgb[idx]                            # (M, 3)
# # #     scales_v = scene.scales[idx]                      # (M, 3)
# # #     quats_v = scene.quats[idx]                        # (M, 4)

# # #     # ── 2. Compute 2D Gaussian radius (screen-space extent) ───────────────────
# # #     # Approximate: project the largest scale axis onto screen via focal lengths
# # #     Sigma3D = _compute_2d_covariance(scales_v, quats_v, cam)  # (M,3,3)

# # #     # Jacobian (approximate at optical axis)
# # #     fx, fy = cam.fx, cam.fy
# # #     z_safe = z_v.clamp(min=0.1).unsqueeze(-1).unsqueeze(-1)  # (M,1,1)

# # #     J = torch.zeros(len(idx), 2, 3, device=device)
# # #     J[:, 0, 0] = fx / z_v
# # #     J[:, 0, 2] = -fx * uv_v[:, 0] / (z_v * cam.fx + 1e-8) / (z_v + 1e-8)   # simplified
# # #     J[:, 1, 1] = fy / z_v
# # #     J[:, 1, 2] = -fy * uv_v[:, 1] / (z_v * cam.fy + 1e-8) / (z_v + 1e-8)

# # #     # 2D covariance:  Sigma2D = J Sigma3D J^T  +  small regulariser
# # #     Sigma2D = J @ Sigma3D @ J.transpose(-1, -2)      # (M, 2, 2)
# # #     Sigma2D[:, 0, 0] += 0.3
# # #     Sigma2D[:, 1, 1] += 0.3

# # #     # Extract cov components
# # #     cov_xx = Sigma2D[:, 0, 0]  # (M,)
# # #     cov_xy = Sigma2D[:, 0, 1]
# # #     cov_yy = Sigma2D[:, 1, 1]

# # #     # Pixel radius for bounding box  (3 sigma)
# # #     det = (cov_xx * cov_yy - cov_xy ** 2).clamp(min=1e-6)
# # #     trace = cov_xx + cov_yy
# # #     lambda1 = trace / 2 + torch.sqrt((trace / 2) ** 2 - det).clamp(min=0)
# # #     radius = (3.0 * lambda1.sqrt()).ceil().int().clamp(1, max(H, W) // 2)

# # #     # ── 3. Sort by depth (back-to-front) ──────────────────────────────────────
# # #     order = torch.argsort(z_v, descending=True)
# # #     uv_v   = uv_v[order];  opac_v = opac_v[order];  rgb_v = rgb_v[order]
# # #     cov_xx = cov_xx[order]; cov_xy = cov_xy[order]; cov_yy = cov_yy[order]
# # #     radius = radius[order]; det    = det[order]
# # #     orig_idx = idx[order]   # original Gaussian indices (for alpha map)

# # #     # ── 4. Rasterise ──────────────────────────────────────────────────────────
# # #     # canvas_rgb   = background.clone().unsqueeze(0).unsqueeze(0).expand(H, W, 3).clone()
# # #     # transmittance = torch.ones(H, W, device=device)
# # #     canvas_rgb = torch.zeros(H, W, 3, device=device)
# # #     transmittance = torch.ones(H, W, device=device)

# # #     # We'll accumulate alpha contributions per-Gaussian for the lifting step.
# # #     # To avoid an (N, H, W) tensor (too large), we use a sparse accumulation:
# # #     # For each Gaussian we compute its contribution to its bounding-box tiles,
# # #     # then scatter-add a per-Gaussian scalar (max alpha over all pixels).
# # #     gauss_alpha_score = torch.zeros(scene.num_gaussians, device=device)

# # #     M = len(order)
# # #     ux = uv_v[:, 0]; uy = uv_v[:, 1]

# # #     # Inverse covariance  (2x2 matrix components)
# # #     inv_det = 1.0 / det.clamp(min=1e-8)
# # #     inv_xx = cov_yy * inv_det
# # #     inv_xy = -cov_xy * inv_det
# # #     inv_yy = cov_xx * inv_det

# # #     # Process Gaussians one by one (vectorised over pixels in their bbox)
# # #     for i in range(M):
# # #         r  = radius[i].item()
# # #         cx = int(round(ux[i].item()))
# # #         cy = int(round(uy[i].item()))

# # #         x0 = max(0, cx - r);  x1 = min(W, cx + r + 1)
# # #         y0 = max(0, cy - r);  y1 = min(H, cy + r + 1)
# # #         if x0 >= x1 or y0 >= y1:
# # #             continue

# # #         # Pixel grid in bounding box
# # #         px = torch.arange(x0, x1, device=device, dtype=torch.float32)  # (bw,)
# # #         py = torch.arange(y0, y1, device=device, dtype=torch.float32)  # (bh,)
# # #         grid_x, grid_y = torch.meshgrid(px, py, indexing="xy")          # (bh, bw)

# # #         dx = grid_x - ux[i]   # (bh, bw)
# # #         dy = grid_y - uy[i]

# # #         # Mahalanobis distance squared
# # #         maha = (inv_xx[i] * dx * dx
# # #                 + 2 * inv_xy[i] * dx * dy
# # #                 + inv_yy[i] * dy * dy)                  # (bh, bw)

# # #         gauss_alpha = opac_v[i] * torch.exp(-0.5 * maha) * (maha < 8.0).float()

# # #         # transmittance slice
# # #         T = transmittance[y0:y1, x0:x1]                 # (bh, bw)
# # #         contrib = gauss_alpha * T                        # weighted contribution

# # #         # Update canvas
# # #         canvas_rgb[y0:y1, x0:x1] += contrib.unsqueeze(-1) * rgb_v[i]
# # #         transmittance[y0:y1, x0:x1] *= (1.0 - gauss_alpha)

# # #         # Record max pixel alpha-contribution for this Gaussian
# # #         gauss_alpha_score[orig_idx[i]] = contrib.max().item()

# # #     # Final composite: add background where transmittance remains
# # #     final_rgb = canvas_rgb + transmittance.unsqueeze(-1) * background


# # #     result = {
# # #         "rgb": final_rgb.clamp(0.0, 1.0),       # (H, W, 3)
# # #         "depth": 1.0 - transmittance,            # (H, W)  accumulated opacity
# # #         "alpha_score": gauss_alpha_score,        # (N,)
# # #     }
# # #     return result

# # def render_view(
# #     scene: GaussianScene,
# #     cam: Camera,
# #     background: Optional[torch.Tensor] = None,
# #     return_alpha_map: bool = False,
# #     max_gaussians: int = 100_000,
# # ) -> dict:
# #     """Fully-vectorised rasteriser — zero Python loops over Gaussians.

# #     Evaluates all Gaussians over fixed-size P×P patches, builds a sparse
# #     (touched_pixels × M) alpha matrix, then computes alpha-compositing
# #     transmittance with a single cumprod. No Python iteration over Gaussians,
# #     tiles, or pixels. Runs entirely in PyTorch tensor ops (MPS/CPU compatible).
# #     """
# #     device = scene.device
# #     H, W = cam.height, cam.width

# #     if background is None:
# #         background = torch.ones(3, device=device)

# #     # ── 1. Project all Gaussians ──────────────────────────────────────────────
# #     uv, z = cam.project(scene.xyz)          # (N, 2), (N,)

# #     margin = max(H, W) * 0.5
# #     valid = (
# #         (z > 0.1)
# #         & (uv[:, 0] > -margin) & (uv[:, 0] < W + margin)
# #         & (uv[:, 1] > -margin) & (uv[:, 1] < H + margin)
# #     )

# #     # Cap total Gaussians for memory safety (keep highest-opacity ones)
# #     if valid.sum() > max_gaussians:
# #         opac_tmp = scene.opacity.clone()
# #         opac_tmp[~valid] = -1.0
# #         _, top_idx = opac_tmp.topk(max_gaussians)
# #         mask2 = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=device)
# #         mask2[top_idx] = True
# #         valid = valid & mask2

# #     idx      = torch.where(valid)[0]          # (M,)
# #     uv_v     = uv[idx]                        # (M, 2)
# #     z_v      = z[idx]                         # (M,)
# #     opac_v   = scene.opacity[idx]             # (M,)
# #     rgb_v    = scene.rgb[idx]                 # (M, 3)
# #     scales_v = scene.scales[idx]
# #     quats_v  = scene.quats[idx]
# #     M        = idx.shape[0]

# #     if M == 0:
# #         final_rgb = background.unsqueeze(0).unsqueeze(0).expand(H, W, 3).clone()
# #         return {
# #             "rgb":         final_rgb.clamp(0.0, 1.0),
# #             "depth":       torch.zeros(H, W, device=device),
# #             "alpha_score": torch.zeros(scene.num_gaussians, device=device),
# #         }

# #     # ── 2. 2D covariance for all visible Gaussians (vectorised) ───────────────
# #     # Project to camera space to get x_cam, y_cam for the Jacobian.
# #     # cam.project() only returns pixel uv, so we recompute xyz_cam here.
# #     xyz_cam_v = (scene.xyz[idx] @ cam.R.T) + cam.t   # (M, 3)  camera-space coords
# #     x_cam_v   = xyz_cam_v[:, 0]                        # (M,)
# #     y_cam_v   = xyz_cam_v[:, 1]
# #     # z_v is already xyz_cam[:,2] from cam.project() — consistent.

# #     Sigma3D = _compute_2d_covariance(scales_v, quats_v, cam)   # (M, 3, 3)

# #     fx, fy = cam.fx, cam.fy
# #     # EWA splatting Jacobian (Zwicker et al. 2002, eq. 29):
# #     #   J = [[fx/z,  0,     -fx*x/z^2],
# #     #        [0,     fy/z,  -fy*y/z^2]]
# #     # where x, y, z are CAMERA-SPACE coordinates (not pixel coords).
# #     z2 = z_v * z_v
# #     J = torch.zeros(M, 2, 3, device=device)
# #     J[:, 0, 0] =  fx / z_v
# #     J[:, 0, 2] = -fx * x_cam_v / z2
# #     J[:, 1, 1] =  fy / z_v
# #     J[:, 1, 2] = -fy * y_cam_v / z2

# #     # Rotate Sigma3D into camera space first: Sigma_cam = R Sigma3D R^T
# #     R = cam.R.unsqueeze(0)                              # (1, 3, 3)
# #     Sigma_cam = R @ Sigma3D @ R.transpose(-1, -2)       # (M, 3, 3)

# #     Sigma2D = J @ Sigma_cam @ J.transpose(-1, -2)       # (M, 2, 2)
# #     Sigma2D[:, 0, 0] += 0.3
# #     Sigma2D[:, 1, 1] += 0.3

# #     cov_xx = Sigma2D[:, 0, 0]    # (M,)
# #     cov_xy = Sigma2D[:, 0, 1]
# #     cov_yy = Sigma2D[:, 1, 1]

# #     det     = (cov_xx * cov_yy - cov_xy ** 2).clamp(min=1e-6)
# #     inv_det = 1.0 / det
# #     inv_xx  = cov_yy * inv_det   # (M,)
# #     inv_xy  = -cov_xy * inv_det
# #     inv_yy  = cov_xx * inv_det

# #     # Per-Gaussian screen-space radius (3-sigma bounding circle)
# #     trace   = cov_xx + cov_yy
# #     lambda1 = trace / 2 + torch.sqrt(((trace / 2) ** 2 - det).clamp(min=0))
# #     radii   = (3.0 * lambda1.sqrt()).ceil().int().clamp(1, max(H, W) // 2)  # (M,)

# #     # ── 3. Sort all Gaussians back-to-front (depth order) ────────────────────
# #     order    = torch.argsort(z_v, descending=True)
# #     uv_v     = uv_v[order]
# #     opac_v   = opac_v[order]
# #     rgb_v    = rgb_v[order]
# #     inv_xx   = inv_xx[order]
# #     inv_xy   = inv_xy[order]
# #     inv_yy   = inv_yy[order]
# #     radii    = radii[order]
# #     orig_idx = idx[order]          # original scene Gaussian indices

# #     ux = uv_v[:, 0]   # (M,)
# #     uy = uv_v[:, 1]

# #     # ── 4. Chunked bbox rasterisation ────────────────────────────────────────────
# #     #
# #     # Process Gaussians in small depth-sorted chunks. For each chunk we:
# #     #   a) compute each Gaussian's pixel bbox (vectorised over the chunk)
# #     #   b) evaluate the Gaussian kernel only inside that bbox
# #     #   c) scatter-add colour and transmittance updates onto the canvas
# #     #
# #     # The key difference from the old per-Gaussian loop: we batch CHUNK Gaussians
# #     # together and do all their bbox evaluations as tensor ops, then write results
# #     # with a single index_put_ per chunk. No .item() calls inside the hot path.
# #     #
# #     # Memory per chunk: CHUNK × (2*MAX_BBOX+1)^2 floats.
# #     # With CHUNK=64, MAX_BBOX=24 → 64×49×49×4 ≈ 0.6 MB — negligible.

# #     CHUNK   = 64          # Gaussians processed per iteration
# #     MAX_BBOX = 24         # hard cap on per-Gaussian bbox half-size (pixels)

# #     canvas_rgb        = torch.zeros(H * W, 3, device=device)
# #     transmittance     = torch.ones(H * W,    device=device)
# #     gauss_alpha_score = torch.zeros(scene.num_gaussians, device=device)

# #     # Clamp radii to MAX_BBOX so patch tensors stay small
# #     radii_capped = radii.clamp(max=MAX_BBOX)   # (M,)

# #     for start in range(0, M, CHUNK):
# #         end   = min(start + CHUNK, M)
# #         C     = end - start                    # actual chunk size

# #         # Per-Gaussian bbox half-sizes for this chunk
# #         r_c   = radii_capped[start:end]        # (C,)
# #         ux_c  = ux[start:end]                  # (C,)
# #         uy_c  = uy[start:end]
# #         cx_c  = ux_c.round().long()
# #         cy_c  = uy_c.round().long()

# #         # Use the maximum radius in this chunk as the common patch size.
# #         # Gaussians with smaller radii simply get near-zero alpha outside their
# #         # true extent — the maha < 8 gate ensures they don't contribute.
# #         R = int(r_c.max().item())
# #         if R < 1:
# #             R = 1
# #         P  = 2 * R + 1
# #         PP = P * P

# #         # Pixel offsets grid: (1, P) and (P, 1)
# #         off = torch.arange(-R, R + 1, device=device)   # (P,)

# #         # Absolute pixel coords for every (Gaussian, patch-row, patch-col)
# #         # row_abs: (C, P, 1) + (1, P, 1) broadcast isn't needed — build directly
# #         row_abs = (cy_c.view(C, 1, 1) + off.view(1, P, 1)).expand(C, P, P).contiguous()  # (C,P,P)
# #         col_abs = (cx_c.view(C, 1, 1) + off.view(1, 1, P)).expand(C, P, P).contiguous()  # (C,P,P)

# #         in_bounds = (
# #             (row_abs >= 0) & (row_abs < H) &
# #             (col_abs >= 0) & (col_abs < W)
# #         )  # (C, P, P)

# #         # Sub-pixel dx/dy
# #         dx = (col_abs.float() - ux_c.view(C, 1, 1))   # (C, P, P)
# #         dy = (row_abs.float() - uy_c.view(C, 1, 1))

# #         ixx = inv_xx[start:end].view(C, 1, 1)
# #         ixy = inv_xy[start:end].view(C, 1, 1)
# #         iyy = inv_yy[start:end].view(C, 1, 1)

# #         maha = ixx * dx * dx + 2.0 * ixy * dx * dy + iyy * dy * dy  # (C,P,P)

# #         alpha = (
# #             opac_v[start:end].view(C, 1, 1)
# #             * torch.exp(-0.5 * maha)
# #             * (maha < 8.0)
# #             * in_bounds
# #         ).contiguous()   # (C, P, P)

# #         # Flatten to (C, PP) and compute linear pixel indices
# #         alpha_flat = alpha.view(C, PP)                          # (C, PP)
# #         pix_idx    = (row_abs.view(C, PP).clamp(0, H-1) * W
# #                       + col_abs.view(C, PP).clamp(0, W-1))     # (C, PP)

# #         # For each Gaussian in the chunk, scatter its contribution onto the canvas.
# #         # We iterate over C (≤64) Gaussians — cheap Python loop.
# #         for k in range(C):
# #             pix_k   = pix_idx[k]          # (PP,) pixel indices
# #             alpha_k = alpha_flat[k]       # (PP,) alpha values

# #             T_k     = transmittance[pix_k]                     # (PP,) current transmittance
# #             contrib = alpha_k * T_k                            # (PP,)

# #             # Accumulate colour
# #             canvas_rgb.index_add_(0, pix_k,
# #                                   contrib.unsqueeze(1) * rgb_v[start + k])

# #             # Update transmittance (multiply by 1 - alpha)
# #             transmittance.index_put_((pix_k,),
# #                                      T_k * (1.0 - alpha_k))

# #             # Record peak contribution for lifting
# #             pk = contrib.max()
# #             g_orig = orig_idx[start + k]
# #             if pk > gauss_alpha_score[g_orig]:
# #                 gauss_alpha_score[g_orig] = pk

# #     final_rgb     = (canvas_rgb + transmittance.unsqueeze(1) * background).view(H, W, 3)
# #     transmittance = transmittance.view(H, W)

# #     return {
# #         "rgb":         final_rgb.clamp(0.0, 1.0),
# #         "depth":       1.0 - transmittance,
# #         "alpha_score": gauss_alpha_score,
# #     }

# """
# gaussian_scene.py  (fixed)
# ──────────────────────────
# Loads a pre-trained 3DGS scene from a .ply file and provides a correct
# pure-PyTorch rasteriser (MPS / CPU compatible).

# Key fixes vs original:
#   1. _quat_to_rotmat: row-major reshape was transposing every matrix → fixed
#   2. EWA Jacobian: was using pixel coords (u,v) instead of camera-space (x,y,z)
#   3. Sigma3D was not rotated into camera space before applying J
#   4. dx/dy broadcast shapes were wrong at tile edges (bh ≠ bw)
#   5. Tile loop inner `for k` still hit .item() per Gaussian → replaced with
#      vectorised cumulative-product transmittance within each tile
# """

# from __future__ import annotations

# import math
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import torch
# import torch.nn.functional as F
# from plyfile import PlyData
# from rich.console import Console

# console = Console()

# # ─────────────────────────────────────────────────────────────────────────────
# #  Spherical Harmonics helpers  (degree-0 DC only)
# # ─────────────────────────────────────────────────────────────────────────────
# SH_C0 = 0.28209479177387814   # 1 / (2√π)

# def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
#     return (sh_dc * SH_C0 + 0.5).clamp(0.0, 1.0)

# def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
#     return (rgb - 0.5) / SH_C0


# # ─────────────────────────────────────────────────────────────────────────────
# #  Camera
# # ─────────────────────────────────────────────────────────────────────────────
# class Camera:
#     """Pinhole camera.  R, t are world→camera (OpenCV convention)."""

#     def __init__(self, R, t, fx, fy, cx, cy, width, height, device):
#         self.R = R.to(device)
#         self.t = t.to(device)
#         self.fx, self.fy = fx, fy
#         self.cx, self.cy = cx, cy
#         self.width  = width
#         self.height = height
#         self.device = device

#     @property
#     def world_center(self):
#         return -(self.R.T @ self.t)

#     def project(self, xyz_world: torch.Tensor):
#         """(N,3) world → (N,2) pixel uv,  (N,) depth z."""
#         xyz_cam = (xyz_world @ self.R.T) + self.t   # (N,3)
#         z  = xyz_cam[:, 2].clamp(min=1e-6)
#         u  = self.fx * xyz_cam[:, 0] / z + self.cx
#         v  = self.fy * xyz_cam[:, 1] / z + self.cy
#         return torch.stack([u, v], dim=-1), z


# # ─────────────────────────────────────────────────────────────────────────────
# #  GaussianScene
# # ─────────────────────────────────────────────────────────────────────────────
# class GaussianScene:

#     def __init__(self, ply_path: Path, device: torch.device):
#         self.device   = device
#         self.ply_path = Path(ply_path)
#         self._load_ply()

#     def _load_ply(self):
#         console.log(f"Loading scene: [cyan]{self.ply_path}[/cyan]")
#         ply    = PlyData.read(str(self.ply_path))
#         vertex = ply["vertex"]

#         def col(name):
#             return torch.tensor(np.array(vertex[name]), dtype=torch.float32)

#         self.xyz     = torch.stack([col("x"), col("y"), col("z")], dim=-1).to(self.device)
#         N            = self.xyz.shape[0]
#         self.opacity = torch.sigmoid(col("opacity")).to(self.device)

#         scales      = torch.stack([col(f"scale_{i}") for i in range(3)], dim=-1)
#         self.scales = torch.exp(scales).to(self.device)
#         # No scale clamping. Mip-NeRF 360 scenes (bicycle, bonsai, etc.) contain
#         # large-scale Gaussians for branches, spokes, and background planes that are
#         # valid geometry. Clamping them collapses their alpha contribution and breaks
#         # the multi-view lifting step (Gaussians never accumulate enough score to be
#         # segmented). Log the scale range for diagnostics only.
#         scale_max_val = self.scales.max().item()
#         console.log(f"  Scale range: max={scale_max_val:.3f}")

#         # FIX: quaternion convention — official 3DGS PLY stores (w,x,y,z)
#         # as rot_0..rot_3, which matches our _quat_to_rotmat expectation.
#         self.quats = F.normalize(
#             torch.stack([col(f"rot_{i}") for i in range(4)], dim=-1), dim=-1
#         ).to(self.device)

#         sh_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_dc")])
#         if sh_names:
#             sh_dc = torch.stack([col(n) for n in sh_names], dim=-1)
#         else:
#             try:
#                 sh_dc = torch.stack([col("red"), col("green"), col("blue")], dim=-1).float() / 255.0
#                 sh_dc = rgb_to_sh(sh_dc)
#             except Exception:
#                 sh_dc = torch.zeros(N, 3)
#         self.sh_dc = sh_dc.to(self.device)

#         sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest")])
#         self.sh_rest  = (
#             torch.stack([col(n) for n in sh_rest_names], dim=-1).to(self.device)
#             if sh_rest_names else None
#         )

#         console.log(f"  Loaded [bold]{N:,}[/bold] Gaussians")
#         self.seg_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

#     @property
#     def num_gaussians(self): return self.xyz.shape[0]

#     @property
#     def rgb(self): return sh_to_rgb(self.sh_dc)

#     def recolor(self, rgb: tuple):
#         if not self.seg_mask.any():
#             console.log("[yellow]No Gaussians segmented – recolor skipped[/yellow]")
#             return
#         target = torch.tensor(rgb, dtype=torch.float32, device=self.device)
#         self.sh_dc[self.seg_mask] = rgb_to_sh(target).unsqueeze(0)
#         if self.sh_rest is not None:
#             self.sh_rest[self.seg_mask] = 0.0
#         console.log(f"  Recolored {self.seg_mask.sum().item():,} Gaussians → RGB{rgb}")

#     def remove_segmented(self):
#         keep      = ~self.seg_mask
#         n_removed = self.seg_mask.sum().item()
#         self.xyz     = self.xyz[keep]
#         self.opacity = self.opacity[keep]
#         self.scales  = self.scales[keep]
#         self.quats   = self.quats[keep]
#         self.sh_dc   = self.sh_dc[keep]
#         if self.sh_rest is not None:
#             self.sh_rest = self.sh_rest[keep]
#         self.seg_mask = self.seg_mask[keep]
#         console.log(f"  Removed {n_removed:,} Gaussians – {self.num_gaussians:,} remaining")

#     def save_ply(self, out_path: Path):
#         from plyfile import PlyElement, PlyData as PD
#         out_path = Path(out_path)
#         out_path.parent.mkdir(parents=True, exist_ok=True)
#         xyz           = self.xyz.cpu().numpy()
#         opacity_logit = torch.logit(self.opacity.clamp(1e-6, 1-1e-6)).cpu().numpy()
#         scales_log    = torch.log(self.scales.clamp(min=1e-8)).cpu().numpy()
#         quats         = self.quats.cpu().numpy()
#         sh_dc         = self.sh_dc.cpu().numpy()
#         dtype_list = [
#             ("x","f4"),("y","f4"),("z","f4"),("opacity","f4"),
#             ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
#             ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
#             ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
#         ]
#         N   = xyz.shape[0]
#         arr = np.empty(N, dtype=dtype_list)
#         arr["x"], arr["y"], arr["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
#         arr["opacity"] = opacity_logit
#         for i in range(3): arr[f"scale_{i}"] = scales_log[:,i]
#         for i in range(4): arr[f"rot_{i}"]   = quats[:,i]
#         for i in range(3): arr[f"f_dc_{i}"]  = sh_dc[:,i]
#         PD([PlyElement.describe(arr, "vertex")]).write(str(out_path))
#         console.log(f"  Saved → [cyan]{out_path}[/cyan]")


# # ─────────────────────────────────────────────────────────────────────────────
# #  Rasteriser helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
#     """(N,4) wxyz quaternion → (N,3,3) rotation matrices.

#     FIX: original code called .reshape(-1,3,3) on a flat stack of 9 scalars
#     per Gaussian which transposes the matrix (row-major ≠ column-major).
#     We build each row explicitly to be unambiguous.
#     """
#     w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
#     # Row 0
#     r00 = 1 - 2*(y*y + z*z);  r01 = 2*(x*y - w*z);  r02 = 2*(x*z + w*y)
#     # Row 1
#     r10 = 2*(x*y + w*z);       r11 = 1 - 2*(x*x + z*z); r12 = 2*(y*z - w*x)
#     # Row 2
#     r20 = 2*(x*z - w*y);       r21 = 2*(y*z + w*x);  r22 = 1 - 2*(x*x + y*y)
#     # Stack rows first, then reshape → correct (N,3,3) layout
#     row0 = torch.stack([r00, r01, r02], dim=-1)   # (N,3)
#     row1 = torch.stack([r10, r11, r12], dim=-1)
#     row2 = torch.stack([r20, r21, r22], dim=-1)
#     return torch.stack([row0, row1, row2], dim=-2)  # (N,3,3)


# def _world_covariance(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
#     """3D world-space covariance:  Σ_world = R_g S Sᵀ R_gᵀ   shape (N,3,3)."""
#     R_g = _quat_to_rotmat(quats)       # (N,3,3)
#     S   = torch.diag_embed(scales)     # (N,3,3)
#     RS  = R_g @ S                      # (N,3,3)
#     return RS @ RS.transpose(-1, -2)   # (N,3,3)


# # ─────────────────────────────────────────────────────────────────────────────
# #  render_view
# # ─────────────────────────────────────────────────────────────────────────────

# def render_view(
#     scene: GaussianScene,
#     cam: Camera,
#     background: Optional[torch.Tensor] = None,
#     max_gaussians: int = 100_000,
#     tile_size: int = 16,
#     **kwargs,
# ) -> dict:
#     """Tile-based alpha-compositing rasteriser.  MPS / CPU compatible.

#     Returns dict with keys "rgb" (H,W,3), "depth" (H,W), "alpha_score" (N,).
#     """
#     device = scene.device
#     H, W   = cam.height, cam.width

#     if background is None:
#         background = torch.ones(3, device=device)

#     # ── 1. Project & cull ────────────────────────────────────────────────────
#     uv, z = cam.project(scene.xyz)          # (N,2), (N,)

#     margin = max(H, W) * 0.5
#     valid  = (
#         (z > 0.1)
#         & (uv[:,0] > -margin) & (uv[:,0] < W + margin)
#         & (uv[:,1] > -margin) & (uv[:,1] < H + margin)
#     )

#     if valid.sum() > max_gaussians:
#         opac_tmp = scene.opacity.clone()
#         opac_tmp[~valid] = -1.0
#         _, top_idx = opac_tmp.topk(max_gaussians)
#         mask2 = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=device)
#         mask2[top_idx] = True
#         valid = valid & mask2

#     idx      = torch.where(valid)[0]
#     uv_v     = uv[idx]
#     z_v      = z[idx]
#     opac_v   = scene.opacity[idx]
#     rgb_v    = scene.rgb[idx]
#     scales_v = scene.scales[idx]
#     quats_v  = scene.quats[idx]
#     M        = idx.shape[0]

#     if M == 0:
#         return {
#             "rgb":         background.view(1,1,3).expand(H,W,3).clone(),
#             "depth":       torch.zeros(H, W, device=device),
#             "alpha_score": torch.zeros(scene.num_gaussians, device=device),
#         }

#     # ── 2. 2D covariance (EWA splatting) ─────────────────────────────────────
#     # Camera-space positions (needed for correct Jacobian)
#     xyz_cam_v = (scene.xyz[idx] @ cam.R.T) + cam.t   # (M,3)
#     xc = xyz_cam_v[:,0]   # (M,)
#     yc = xyz_cam_v[:,1]
#     # z_v == xyz_cam_v[:,2]  (already computed above, same values)

#     # World-space 3D covariance
#     Sigma_world = _world_covariance(scales_v, quats_v)   # (M,3,3)

#     # Rotate into camera space:  Σ_cam = R_cam Σ_world R_camᵀ
#     R  = cam.R.unsqueeze(0)                              # (1,3,3)
#     Sigma_cam = R @ Sigma_world @ R.transpose(-1,-2)     # (M,3,3)

#     # EWA perspective Jacobian (Zwicker et al. 2002, eq. 29):
#     #   J = [[fx/z,  0,      -fx·x/z²],
#     #        [0,     fy/z,   -fy·y/z²]]
#     fx, fy = cam.fx, cam.fy
#     z2 = z_v * z_v
#     J  = torch.zeros(M, 2, 3, device=device)
#     J[:,0,0] =  fx / z_v
#     J[:,0,2] = -fx * xc / z2
#     J[:,1,1] =  fy / z_v
#     J[:,1,2] = -fy * yc / z2

#     Sigma2D = J @ Sigma_cam @ J.transpose(-1,-2)     # (M,2,2)
#     Sigma2D[:,0,0] += 0.3                             # low-pass filter (prevent aliasing)
#     Sigma2D[:,1,1] += 0.3

#     cov_xx = Sigma2D[:,0,0]
#     cov_xy = Sigma2D[:,0,1]
#     cov_yy = Sigma2D[:,1,1]

#     det     = (cov_xx * cov_yy - cov_xy**2).clamp(min=1e-6)
#     inv_det = 1.0 / det
#     inv_xx  =  cov_yy * inv_det
#     inv_xy  = -cov_xy * inv_det
#     inv_yy  =  cov_xx * inv_det

#     # Per-Gaussian screen-space bounding radius (3σ)
#     trace   = cov_xx + cov_yy
#     lambda1 = trace/2 + torch.sqrt(((trace/2)**2 - det).clamp(min=0))
#     radii   = (3.0 * lambda1.sqrt()).ceil().int().clamp(1, max(H,W)//2)

#     # ── 3. Depth sort (back-to-front) ────────────────────────────────────────
#     order    = torch.argsort(z_v, descending=True)
#     uv_v     = uv_v[order]
#     opac_v   = opac_v[order]
#     rgb_v    = rgb_v[order]
#     inv_xx   = inv_xx[order]
#     inv_xy   = inv_xy[order]
#     inv_yy   = inv_yy[order]
#     radii    = radii[order]
#     orig_idx = idx[order]

#     ux = uv_v[:,0]
#     uy = uv_v[:,1]

#     # ── 4. Tile rasterisation ─────────────────────────────────────────────────
#     T     = tile_size
#     n_ty  = math.ceil(H / T)
#     n_tx  = math.ceil(W / T)

#     r_np  = radii.cpu().numpy()
#     ux_np = ux.cpu().numpy()
#     uy_np = uy.cpu().numpy()

#     canvas_rgb        = torch.zeros(H, W, 3, device=device)
#     transmittance     = torch.ones(H, W,    device=device)
#     gauss_alpha_score = torch.zeros(scene.num_gaussians, device=device)

#     for ty in range(n_ty):
#         y0 = ty * T;  y1 = min(y0 + T, H)
#         for tx in range(n_tx):
#             x0 = tx * T;  x1 = min(x0 + T, W)

#             # Gaussians whose bbox overlaps this tile
#             overlap = (
#                 (ux_np + r_np >= x0) & (ux_np - r_np < x1)
#                 & (uy_np + r_np >= y0) & (uy_np - r_np < y1)
#             )
#             if not overlap.any():
#                 continue

#             t_idx  = torch.from_numpy(overlap.nonzero()[0]).to(device)
#             K      = t_idx.shape[0]
#             t_ux   = ux[t_idx]
#             t_uy   = uy[t_idx]
#             t_opac = opac_v[t_idx]
#             t_rgb  = rgb_v[t_idx]
#             t_ixx  = inv_xx[t_idx]
#             t_ixy  = inv_xy[t_idx]
#             t_iyy  = inv_yy[t_idx]

#             bh = y1 - y0;  bw = x1 - x0

#             # FIX: use .view() with explicit shapes to avoid non-contiguous broadcast
#             px = torch.arange(x0, x1, device=device, dtype=torch.float32)  # (bw,)
#             py = torch.arange(y0, y1, device=device, dtype=torch.float32)  # (bh,)

#             dx = px.view(1, 1, bw) - t_ux.view(K, 1, 1)   # (K, 1, bw)
#             dy = py.view(1, bh, 1) - t_uy.view(K, 1, 1)   # (K, bh, 1)
#             # Both broadcast correctly to (K, bh, bw)

#             maha = (
#                 t_ixx.view(K,1,1) * dx * dx
#                 + 2.0 * t_ixy.view(K,1,1) * dx * dy
#                 + t_iyy.view(K,1,1) * dy * dy
#             )   # (K, bh, bw)

#             g_alpha = t_opac.view(K,1,1) * torch.exp(-0.5*maha) * (maha < 8.0)
#             # (K, bh, bw)

#             # Vectorised alpha-compositing within tile:
#             # T_before[k] = Π_{j<k}(1 - α_j)  for each pixel
#             # Use cumprod then shift right by 1.
#             one_minus = (1.0 - g_alpha)                      # (K, bh, bw)
#             cumprod   = torch.cumprod(one_minus, dim=0)       # (K, bh, bw)
#             T_before  = torch.cat([
#                 torch.ones(1, bh, bw, device=device),
#                 cumprod[:-1]
#             ], dim=0)                                          # (K, bh, bw)

#             weights = g_alpha * T_before                       # (K, bh, bw)

#             # Colour accumulation: Σ_k w_k · rgb_k
#             # rgb: (K,3) → (K,1,1,3); weights: (K,bh,bw,1)
#             colour_contrib = (weights.unsqueeze(-1) * t_rgb.view(K,1,1,3)).sum(dim=0)
#             canvas_rgb[y0:y1, x0:x1] += colour_contrib        # (bh,bw,3)

#             # Residual transmittance for this tile
#             T_final = cumprod[-1]                              # (bh, bw)
#             transmittance[y0:y1, x0:x1] *= T_final

#             # Alpha score for mask lifting: max weight per Gaussian
#             peak_per_gauss = weights.view(K, -1).max(dim=1).values  # (K,)
#             gauss_alpha_score.scatter_reduce_(
#                 0, orig_idx[t_idx], peak_per_gauss, reduce="amax", include_self=True
#             )

#     final_rgb = canvas_rgb + transmittance.unsqueeze(-1) * background

#     return {
#         "rgb":         final_rgb.clamp(0.0, 1.0),
#         "depth":       1.0 - transmittance,
#         "alpha_score": gauss_alpha_score,
#     }

# """
# gaussian_scene.py
# ─────────────────
# Loads a pre-trained 3D Gaussian Splatting scene from a .ply file and provides:
#   - Differentiable rasterisation via a pure-PyTorch (MPS-friendly) renderer
#   - Alpha-map extraction per view (needed for multi-view mask lifting)
#   - Per-Gaussian attribute access (position, colour, opacity, scale, rotation)
#   - Downstream edit operations: recolor, remove

# The renderer is a simplified tile-based alpha-compositing rasteriser that runs
# entirely in PyTorch, so it works on MPS without any CUDA extensions.
# """

# from __future__ import annotations

# import math
# from pathlib import Path
# from typing import Optional

# import numpy as np
# import torch
# import torch.nn.functional as F
# from plyfile import PlyData
# from rich.console import Console

# console = Console()

# # ──────────────────────────────────────────────────────────────────────────────
# #  Spherical Harmonics helpers (degree-0 DC only for colour)
# # ──────────────────────────────────────────────────────────────────────────────
# SH_C0 = 0.28209479177387814   # 1 / (2 * sqrt(pi))


# def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
#     """Convert SH degree-0 (DC) coefficients → linear RGB in [0, 1]."""
#     return (sh_dc * SH_C0 + 0.5).clamp(0.0, 1.0)


# def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
#     """Convert linear RGB → SH degree-0 DC coefficients."""
#     return (rgb - 0.5) / SH_C0


# # ──────────────────────────────────────────────────────────────────────────────
# #  Camera
# # ──────────────────────────────────────────────────────────────────────────────
# class Camera:
#     """Simple pinhole camera with world→camera extrinsics."""

#     def __init__(
#         self,
#         R: torch.Tensor,          # (3, 3)  world→cam rotation
#         t: torch.Tensor,          # (3,)    world→cam translation
#         fx: float, fy: float,
#         cx: float, cy: float,
#         width: int, height: int,
#         device: torch.device,
#     ):
#         self.R = R.to(device)
#         self.t = t.to(device)
#         self.fx, self.fy = fx, fy
#         self.cx, self.cy = cx, cy
#         self.width = width
#         self.height = height
#         self.device = device

#     @property
#     def world_center(self) -> torch.Tensor:
#         """Camera centre in world coordinates:  -R^T @ t"""
#         return -(self.R.T @ self.t)

#     def project(self, xyz_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         """Project world-space points → pixel coords and per-point depths.

#         Args:
#             xyz_world: (N, 3) world-space positions.

#         Returns:
#             uv: (N, 2) pixel coordinates (float).
#             z:  (N,)   depth (camera-space z).
#         """
#         # World → Camera
#         xyz_cam = (xyz_world @ self.R.T) + self.t  # (N, 3)
#         z = xyz_cam[:, 2].clamp(min=1e-6)          # depth

#         u = self.fx * xyz_cam[:, 0] / z + self.cx
#         v = self.fy * xyz_cam[:, 1] / z + self.cy
#         uv = torch.stack([u, v], dim=-1)            # (N, 2)
#         return uv, z


# # ──────────────────────────────────────────────────────────────────────────────
# #  GaussianScene
# # ──────────────────────────────────────────────────────────────────────────────
# class GaussianScene:
#     """Holds all per-Gaussian attributes loaded from a .ply file."""

#     def __init__(self, ply_path: Path, device: torch.device):
#         self.device = device
#         self.ply_path = Path(ply_path)
#         self._load_ply()

#     # ── Loading ───────────────────────────────────────────────────────────────

#     def _load_ply(self):
#         console.log(f"Loading scene: [cyan]{self.ply_path}[/cyan]")
#         ply = PlyData.read(str(self.ply_path))
#         vertex = ply["vertex"]

#         def col(name: str) -> torch.Tensor:
#             return torch.tensor(np.array(vertex[name]), dtype=torch.float32)

#         # Positions  (N, 3)
#         self.xyz = torch.stack([col("x"), col("y"), col("z")], dim=-1).to(self.device)
#         N = self.xyz.shape[0]

#         # Opacity logit → sigmoid  (N,)
#         self.opacity = torch.sigmoid(col("opacity")).to(self.device)

#         # Scale (log-space in file) → exp  (N, 3)
#         scales = torch.stack([col(f"scale_{i}") for i in range(3)], dim=-1)
#         self.scales = torch.exp(scales).to(self.device)

#         # Rotation quaternion (N, 4)  wxyz convention
#         self.quats = F.normalize(
#             torch.stack([col(f"rot_{i}") for i in range(4)], dim=-1),
#             dim=-1
#         ).to(self.device)

#         # SH features: at minimum DC (degree 0) per channel
#         sh_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_dc")])
#         if sh_names:
#             sh_dc = torch.stack([col(n) for n in sh_names], dim=-1)  # (N, 3)
#         else:
#             # Fallback: try "red","green","blue" attributes (some exporters)
#             try:
#                 sh_dc = torch.stack([col("red"), col("green"), col("blue")], dim=-1).float() / 255.0
#                 sh_dc = rgb_to_sh(sh_dc)
#             except Exception:
#                 sh_dc = torch.zeros(N, 3)
#         self.sh_dc = sh_dc.to(self.device)  # (N, 3)

#         # Higher-order SH (stored but not used in rendering here for speed)
#         sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest")])
#         if sh_rest_names:
#             self.sh_rest = torch.stack([col(n) for n in sh_rest_names], dim=-1).to(self.device)
#         else:
#             self.sh_rest = None

#         console.log(f"  Loaded [bold]{N:,}[/bold] Gaussians")

#         # Segmentation mask (all False initially)
#         self.seg_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

#     # ── Properties ────────────────────────────────────────────────────────────

#     @property
#     def num_gaussians(self) -> int:
#         return self.xyz.shape[0]

#     @property
#     def rgb(self) -> torch.Tensor:
#         """Per-Gaussian colour from DC SH coefficients. (N, 3)"""
#         return sh_to_rgb(self.sh_dc)

#     # ── Edit operations ───────────────────────────────────────────────────────

#     def recolor(self, rgb: tuple[float, float, float]):
#         """Set all segmented Gaussians to a fixed RGB colour (in-place)."""
#         if not self.seg_mask.any():
#             console.log("[yellow]No Gaussians segmented – recolor skipped[/yellow]")
#             return
#         target = torch.tensor(rgb, dtype=torch.float32, device=self.device)
#         self.sh_dc[self.seg_mask] = rgb_to_sh(target).unsqueeze(0)
#         # Zero out higher-order SH for recolored Gaussians to avoid colour bleeding
#         if self.sh_rest is not None:
#             self.sh_rest[self.seg_mask] = 0.0
#         console.log(f"  Recolored [bold]{self.seg_mask.sum().item():,}[/bold] Gaussians → RGB{rgb}")

#     def remove_segmented(self):
#         """Remove all segmented Gaussians from the scene (in-place)."""
#         keep = ~self.seg_mask
#         n_removed = self.seg_mask.sum().item()
#         self.xyz = self.xyz[keep]
#         self.opacity = self.opacity[keep]
#         self.scales = self.scales[keep]
#         self.quats = self.quats[keep]
#         self.sh_dc = self.sh_dc[keep]
#         if self.sh_rest is not None:
#             self.sh_rest = self.sh_rest[keep]
#         self.seg_mask = self.seg_mask[keep]
#         console.log(f"  Removed [bold]{n_removed:,}[/bold] Gaussians – "
#                     f"[bold]{self.num_gaussians:,}[/bold] remaining")

#     def save_ply(self, out_path: Path):
#         """Write the current scene back to a .ply file."""
#         from plyfile import PlyElement, PlyData
#         out_path = Path(out_path)
#         out_path.parent.mkdir(parents=True, exist_ok=True)

#         xyz = self.xyz.cpu().numpy()
#         opacity_logit = torch.logit(self.opacity.clamp(1e-6, 1 - 1e-6)).cpu().numpy()
#         scales_log = torch.log(self.scales.clamp(min=1e-8)).cpu().numpy()
#         quats = self.quats.cpu().numpy()
#         sh_dc = self.sh_dc.cpu().numpy()

#         dtype_list = [
#             ("x", "f4"), ("y", "f4"), ("z", "f4"),
#             ("opacity", "f4"),
#             ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
#             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
#             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
#         ]
#         N = xyz.shape[0]
#         arr = np.empty(N, dtype=dtype_list)
#         arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
#         arr["opacity"] = opacity_logit
#         for i in range(3):
#             arr[f"scale_{i}"] = scales_log[:, i]
#         for i in range(4):
#             arr[f"rot_{i}"] = quats[:, i]
#         for i in range(3):
#             arr[f"f_dc_{i}"] = sh_dc[:, i]

#         el = PlyElement.describe(arr, "vertex")
#         PlyData([el]).write(str(out_path))
#         console.log(f"  Saved scene → [cyan]{out_path}[/cyan]")


# # ──────────────────────────────────────────────────────────────────────────────
# #  Pure-PyTorch Gaussian Rasteriser  (MPS-compatible)
# # ──────────────────────────────────────────────────────────────────────────────

# def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
#     """(N, 4) wxyz quaternion → (N, 3, 3) rotation matrices."""
#     w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
#     R = torch.stack([
#         1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
#         2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x),
#         2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y),
#     ], dim=-1).reshape(-1, 3, 3)
#     return R


# def _compute_2d_covariance(
#     scales: torch.Tensor,    # (N, 3)
#     quats: torch.Tensor,     # (N, 4)
#     cam: Camera,
# ) -> torch.Tensor:
#     """3D world-space covariance:  Sigma_world = R_g S S^T R_g^T

#     Returns (N, 3, 3). Caller is responsible for rotating into camera space
#     and applying the perspective Jacobian.
#     """
#     R_g  = _quat_to_rotmat(quats)          # (N, 3, 3)  Gaussian rotation
#     S    = torch.diag_embed(scales)         # (N, 3, 3)  scale diagonal
#     RS   = R_g @ S                          # (N, 3, 3)
#     return RS @ RS.transpose(-1, -2)        # (N, 3, 3)  world-space covariance


# # def render_view(
# #     scene: GaussianScene,
# #     cam: Camera,
# #     background: Optional[torch.Tensor] = None,
# #     return_alpha_map: bool = False,
# #     max_gaussians: int = 50_000,
# # ) -> dict:
# #     """Render a single view using a vectorised alpha-compositing rasteriser.

# #     This is *not* as fast as the CUDA rasteriser but is fully MPS/CPU compatible.

# #     Returns dict with keys:
# #         "rgb"        (H, W, 3) float32 rendered image
# #         "alpha_map"  (N, H, W) float32  per-Gaussian alpha contribution  [optional]
# #         "depth"      (H, W)    float32
# #     """
# #     device = scene.device
# #     H, W = cam.height, cam.width

# #     if background is None:
# #         background = torch.ones(3, device=device)

# #     # ── 1. Project all Gaussians to 2D ────────────────────────────────────────
# #     uv, z = cam.project(scene.xyz)                    # (N,2), (N,)

# #     # Cull Gaussians behind camera or outside a generous image margin
# #     margin = max(H, W) * 0.5
# #     valid = (z > 0.1) & \
# #             (uv[:, 0] > -margin) & (uv[:, 0] < W + margin) & \
# #             (uv[:, 1] > -margin) & (uv[:, 1] < H + margin)

# #     # Limit for memory (take highest-opacity Gaussians first)
# #     if valid.sum() > max_gaussians:
# #         opac_valid = scene.opacity.clone()
# #         opac_valid[~valid] = -1.0
# #         _, top_idx = opac_valid.topk(max_gaussians)
# #         mask2 = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=device)
# #         mask2[top_idx] = True
# #         valid = valid & mask2

# #     idx = torch.where(valid)[0]                       # indices of visible Gaussians
# #     uv_v = uv[idx]                                    # (M, 2)
# #     z_v = z[idx]                                      # (M,)
# #     opac_v = scene.opacity[idx]                       # (M,)
# #     rgb_v = scene.rgb[idx]                            # (M, 3)
# #     scales_v = scene.scales[idx]                      # (M, 3)
# #     quats_v = scene.quats[idx]                        # (M, 4)

# #     # ── 2. Compute 2D Gaussian radius (screen-space extent) ───────────────────
# #     # Approximate: project the largest scale axis onto screen via focal lengths
# #     Sigma3D = _compute_2d_covariance(scales_v, quats_v, cam)  # (M,3,3)

# #     # Jacobian (approximate at optical axis)
# #     fx, fy = cam.fx, cam.fy
# #     z_safe = z_v.clamp(min=0.1).unsqueeze(-1).unsqueeze(-1)  # (M,1,1)

# #     J = torch.zeros(len(idx), 2, 3, device=device)
# #     J[:, 0, 0] = fx / z_v
# #     J[:, 0, 2] = -fx * uv_v[:, 0] / (z_v * cam.fx + 1e-8) / (z_v + 1e-8)   # simplified
# #     J[:, 1, 1] = fy / z_v
# #     J[:, 1, 2] = -fy * uv_v[:, 1] / (z_v * cam.fy + 1e-8) / (z_v + 1e-8)

# #     # 2D covariance:  Sigma2D = J Sigma3D J^T  +  small regulariser
# #     Sigma2D = J @ Sigma3D @ J.transpose(-1, -2)      # (M, 2, 2)
# #     Sigma2D[:, 0, 0] += 0.3
# #     Sigma2D[:, 1, 1] += 0.3

# #     # Extract cov components
# #     cov_xx = Sigma2D[:, 0, 0]  # (M,)
# #     cov_xy = Sigma2D[:, 0, 1]
# #     cov_yy = Sigma2D[:, 1, 1]

# #     # Pixel radius for bounding box  (3 sigma)
# #     det = (cov_xx * cov_yy - cov_xy ** 2).clamp(min=1e-6)
# #     trace = cov_xx + cov_yy
# #     lambda1 = trace / 2 + torch.sqrt((trace / 2) ** 2 - det).clamp(min=0)
# #     radius = (3.0 * lambda1.sqrt()).ceil().int().clamp(1, max(H, W) // 2)

# #     # ── 3. Sort by depth (back-to-front) ──────────────────────────────────────
# #     order = torch.argsort(z_v, descending=True)
# #     uv_v   = uv_v[order];  opac_v = opac_v[order];  rgb_v = rgb_v[order]
# #     cov_xx = cov_xx[order]; cov_xy = cov_xy[order]; cov_yy = cov_yy[order]
# #     radius = radius[order]; det    = det[order]
# #     orig_idx = idx[order]   # original Gaussian indices (for alpha map)

# #     # ── 4. Rasterise ──────────────────────────────────────────────────────────
# #     # canvas_rgb   = background.clone().unsqueeze(0).unsqueeze(0).expand(H, W, 3).clone()
# #     # transmittance = torch.ones(H, W, device=device)
# #     canvas_rgb = torch.zeros(H, W, 3, device=device)
# #     transmittance = torch.ones(H, W, device=device)

# #     # We'll accumulate alpha contributions per-Gaussian for the lifting step.
# #     # To avoid an (N, H, W) tensor (too large), we use a sparse accumulation:
# #     # For each Gaussian we compute its contribution to its bounding-box tiles,
# #     # then scatter-add a per-Gaussian scalar (max alpha over all pixels).
# #     gauss_alpha_score = torch.zeros(scene.num_gaussians, device=device)

# #     M = len(order)
# #     ux = uv_v[:, 0]; uy = uv_v[:, 1]

# #     # Inverse covariance  (2x2 matrix components)
# #     inv_det = 1.0 / det.clamp(min=1e-8)
# #     inv_xx = cov_yy * inv_det
# #     inv_xy = -cov_xy * inv_det
# #     inv_yy = cov_xx * inv_det

# #     # Process Gaussians one by one (vectorised over pixels in their bbox)
# #     for i in range(M):
# #         r  = radius[i].item()
# #         cx = int(round(ux[i].item()))
# #         cy = int(round(uy[i].item()))

# #         x0 = max(0, cx - r);  x1 = min(W, cx + r + 1)
# #         y0 = max(0, cy - r);  y1 = min(H, cy + r + 1)
# #         if x0 >= x1 or y0 >= y1:
# #             continue

# #         # Pixel grid in bounding box
# #         px = torch.arange(x0, x1, device=device, dtype=torch.float32)  # (bw,)
# #         py = torch.arange(y0, y1, device=device, dtype=torch.float32)  # (bh,)
# #         grid_x, grid_y = torch.meshgrid(px, py, indexing="xy")          # (bh, bw)

# #         dx = grid_x - ux[i]   # (bh, bw)
# #         dy = grid_y - uy[i]

# #         # Mahalanobis distance squared
# #         maha = (inv_xx[i] * dx * dx
# #                 + 2 * inv_xy[i] * dx * dy
# #                 + inv_yy[i] * dy * dy)                  # (bh, bw)

# #         gauss_alpha = opac_v[i] * torch.exp(-0.5 * maha) * (maha < 8.0).float()

# #         # transmittance slice
# #         T = transmittance[y0:y1, x0:x1]                 # (bh, bw)
# #         contrib = gauss_alpha * T                        # weighted contribution

# #         # Update canvas
# #         canvas_rgb[y0:y1, x0:x1] += contrib.unsqueeze(-1) * rgb_v[i]
# #         transmittance[y0:y1, x0:x1] *= (1.0 - gauss_alpha)

# #         # Record max pixel alpha-contribution for this Gaussian
# #         gauss_alpha_score[orig_idx[i]] = contrib.max().item()

# #     # Final composite: add background where transmittance remains
# #     final_rgb = canvas_rgb + transmittance.unsqueeze(-1) * background


# #     result = {
# #         "rgb": final_rgb.clamp(0.0, 1.0),       # (H, W, 3)
# #         "depth": 1.0 - transmittance,            # (H, W)  accumulated opacity
# #         "alpha_score": gauss_alpha_score,        # (N,)
# #     }
# #     return result

# def render_view(
#     scene: GaussianScene,
#     cam: Camera,
#     background: Optional[torch.Tensor] = None,
#     return_alpha_map: bool = False,
#     max_gaussians: int = 100_000,
# ) -> dict:
#     """Fully-vectorised rasteriser — zero Python loops over Gaussians.

#     Evaluates all Gaussians over fixed-size P×P patches, builds a sparse
#     (touched_pixels × M) alpha matrix, then computes alpha-compositing
#     transmittance with a single cumprod. No Python iteration over Gaussians,
#     tiles, or pixels. Runs entirely in PyTorch tensor ops (MPS/CPU compatible).
#     """
#     device = scene.device
#     H, W = cam.height, cam.width

#     if background is None:
#         background = torch.ones(3, device=device)

#     # ── 1. Project all Gaussians ──────────────────────────────────────────────
#     uv, z = cam.project(scene.xyz)          # (N, 2), (N,)

#     margin = max(H, W) * 0.5
#     valid = (
#         (z > 0.1)
#         & (uv[:, 0] > -margin) & (uv[:, 0] < W + margin)
#         & (uv[:, 1] > -margin) & (uv[:, 1] < H + margin)
#     )

#     # Cap total Gaussians for memory safety (keep highest-opacity ones)
#     if valid.sum() > max_gaussians:
#         opac_tmp = scene.opacity.clone()
#         opac_tmp[~valid] = -1.0
#         _, top_idx = opac_tmp.topk(max_gaussians)
#         mask2 = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=device)
#         mask2[top_idx] = True
#         valid = valid & mask2

#     idx      = torch.where(valid)[0]          # (M,)
#     uv_v     = uv[idx]                        # (M, 2)
#     z_v      = z[idx]                         # (M,)
#     opac_v   = scene.opacity[idx]             # (M,)
#     rgb_v    = scene.rgb[idx]                 # (M, 3)
#     scales_v = scene.scales[idx]
#     quats_v  = scene.quats[idx]
#     M        = idx.shape[0]

#     if M == 0:
#         final_rgb = background.unsqueeze(0).unsqueeze(0).expand(H, W, 3).clone()
#         return {
#             "rgb":         final_rgb.clamp(0.0, 1.0),
#             "depth":       torch.zeros(H, W, device=device),
#             "alpha_score": torch.zeros(scene.num_gaussians, device=device),
#         }

#     # ── 2. 2D covariance for all visible Gaussians (vectorised) ───────────────
#     # Project to camera space to get x_cam, y_cam for the Jacobian.
#     # cam.project() only returns pixel uv, so we recompute xyz_cam here.
#     xyz_cam_v = (scene.xyz[idx] @ cam.R.T) + cam.t   # (M, 3)  camera-space coords
#     x_cam_v   = xyz_cam_v[:, 0]                        # (M,)
#     y_cam_v   = xyz_cam_v[:, 1]
#     # z_v is already xyz_cam[:,2] from cam.project() — consistent.

#     Sigma3D = _compute_2d_covariance(scales_v, quats_v, cam)   # (M, 3, 3)

#     fx, fy = cam.fx, cam.fy
#     # EWA splatting Jacobian (Zwicker et al. 2002, eq. 29):
#     #   J = [[fx/z,  0,     -fx*x/z^2],
#     #        [0,     fy/z,  -fy*y/z^2]]
#     # where x, y, z are CAMERA-SPACE coordinates (not pixel coords).
#     z2 = z_v * z_v
#     J = torch.zeros(M, 2, 3, device=device)
#     J[:, 0, 0] =  fx / z_v
#     J[:, 0, 2] = -fx * x_cam_v / z2
#     J[:, 1, 1] =  fy / z_v
#     J[:, 1, 2] = -fy * y_cam_v / z2

#     # Rotate Sigma3D into camera space first: Sigma_cam = R Sigma3D R^T
#     R = cam.R.unsqueeze(0)                              # (1, 3, 3)
#     Sigma_cam = R @ Sigma3D @ R.transpose(-1, -2)       # (M, 3, 3)

#     Sigma2D = J @ Sigma_cam @ J.transpose(-1, -2)       # (M, 2, 2)
#     Sigma2D[:, 0, 0] += 0.3
#     Sigma2D[:, 1, 1] += 0.3

#     cov_xx = Sigma2D[:, 0, 0]    # (M,)
#     cov_xy = Sigma2D[:, 0, 1]
#     cov_yy = Sigma2D[:, 1, 1]

#     det     = (cov_xx * cov_yy - cov_xy ** 2).clamp(min=1e-6)
#     inv_det = 1.0 / det
#     inv_xx  = cov_yy * inv_det   # (M,)
#     inv_xy  = -cov_xy * inv_det
#     inv_yy  = cov_xx * inv_det

#     # Per-Gaussian screen-space radius (3-sigma bounding circle)
#     trace   = cov_xx + cov_yy
#     lambda1 = trace / 2 + torch.sqrt(((trace / 2) ** 2 - det).clamp(min=0))
#     radii   = (3.0 * lambda1.sqrt()).ceil().int().clamp(1, max(H, W) // 2)  # (M,)

#     # ── 3. Sort all Gaussians back-to-front (depth order) ────────────────────
#     order    = torch.argsort(z_v, descending=True)
#     uv_v     = uv_v[order]
#     opac_v   = opac_v[order]
#     rgb_v    = rgb_v[order]
#     inv_xx   = inv_xx[order]
#     inv_xy   = inv_xy[order]
#     inv_yy   = inv_yy[order]
#     radii    = radii[order]
#     orig_idx = idx[order]          # original scene Gaussian indices

#     ux = uv_v[:, 0]   # (M,)
#     uy = uv_v[:, 1]

#     # ── 4. Chunked bbox rasterisation ────────────────────────────────────────────
#     #
#     # Process Gaussians in small depth-sorted chunks. For each chunk we:
#     #   a) compute each Gaussian's pixel bbox (vectorised over the chunk)
#     #   b) evaluate the Gaussian kernel only inside that bbox
#     #   c) scatter-add colour and transmittance updates onto the canvas
#     #
#     # The key difference from the old per-Gaussian loop: we batch CHUNK Gaussians
#     # together and do all their bbox evaluations as tensor ops, then write results
#     # with a single index_put_ per chunk. No .item() calls inside the hot path.
#     #
#     # Memory per chunk: CHUNK × (2*MAX_BBOX+1)^2 floats.
#     # With CHUNK=64, MAX_BBOX=24 → 64×49×49×4 ≈ 0.6 MB — negligible.

#     CHUNK   = 64          # Gaussians processed per iteration
#     MAX_BBOX = 24         # hard cap on per-Gaussian bbox half-size (pixels)

#     canvas_rgb        = torch.zeros(H * W, 3, device=device)
#     transmittance     = torch.ones(H * W,    device=device)
#     gauss_alpha_score = torch.zeros(scene.num_gaussians, device=device)

#     # Clamp radii to MAX_BBOX so patch tensors stay small
#     radii_capped = radii.clamp(max=MAX_BBOX)   # (M,)

#     for start in range(0, M, CHUNK):
#         end   = min(start + CHUNK, M)
#         C     = end - start                    # actual chunk size

#         # Per-Gaussian bbox half-sizes for this chunk
#         r_c   = radii_capped[start:end]        # (C,)
#         ux_c  = ux[start:end]                  # (C,)
#         uy_c  = uy[start:end]
#         cx_c  = ux_c.round().long()
#         cy_c  = uy_c.round().long()

#         # Use the maximum radius in this chunk as the common patch size.
#         # Gaussians with smaller radii simply get near-zero alpha outside their
#         # true extent — the maha < 8 gate ensures they don't contribute.
#         R = int(r_c.max().item())
#         if R < 1:
#             R = 1
#         P  = 2 * R + 1
#         PP = P * P

#         # Pixel offsets grid: (1, P) and (P, 1)
#         off = torch.arange(-R, R + 1, device=device)   # (P,)

#         # Absolute pixel coords for every (Gaussian, patch-row, patch-col)
#         # row_abs: (C, P, 1) + (1, P, 1) broadcast isn't needed — build directly
#         row_abs = (cy_c.view(C, 1, 1) + off.view(1, P, 1)).expand(C, P, P).contiguous()  # (C,P,P)
#         col_abs = (cx_c.view(C, 1, 1) + off.view(1, 1, P)).expand(C, P, P).contiguous()  # (C,P,P)

#         in_bounds = (
#             (row_abs >= 0) & (row_abs < H) &
#             (col_abs >= 0) & (col_abs < W)
#         )  # (C, P, P)

#         # Sub-pixel dx/dy
#         dx = (col_abs.float() - ux_c.view(C, 1, 1))   # (C, P, P)
#         dy = (row_abs.float() - uy_c.view(C, 1, 1))

#         ixx = inv_xx[start:end].view(C, 1, 1)
#         ixy = inv_xy[start:end].view(C, 1, 1)
#         iyy = inv_yy[start:end].view(C, 1, 1)

#         maha = ixx * dx * dx + 2.0 * ixy * dx * dy + iyy * dy * dy  # (C,P,P)

#         alpha = (
#             opac_v[start:end].view(C, 1, 1)
#             * torch.exp(-0.5 * maha)
#             * (maha < 8.0)
#             * in_bounds
#         ).contiguous()   # (C, P, P)

#         # Flatten to (C, PP) and compute linear pixel indices
#         alpha_flat = alpha.view(C, PP)                          # (C, PP)
#         pix_idx    = (row_abs.view(C, PP).clamp(0, H-1) * W
#                       + col_abs.view(C, PP).clamp(0, W-1))     # (C, PP)

#         # For each Gaussian in the chunk, scatter its contribution onto the canvas.
#         # We iterate over C (≤64) Gaussians — cheap Python loop.
#         for k in range(C):
#             pix_k   = pix_idx[k]          # (PP,) pixel indices
#             alpha_k = alpha_flat[k]       # (PP,) alpha values

#             T_k     = transmittance[pix_k]                     # (PP,) current transmittance
#             contrib = alpha_k * T_k                            # (PP,)

#             # Accumulate colour
#             canvas_rgb.index_add_(0, pix_k,
#                                   contrib.unsqueeze(1) * rgb_v[start + k])

#             # Update transmittance (multiply by 1 - alpha)
#             transmittance.index_put_((pix_k,),
#                                      T_k * (1.0 - alpha_k))

#             # Record peak contribution for lifting
#             pk = contrib.max()
#             g_orig = orig_idx[start + k]
#             if pk > gauss_alpha_score[g_orig]:
#                 gauss_alpha_score[g_orig] = pk

#     final_rgb     = (canvas_rgb + transmittance.unsqueeze(1) * background).view(H, W, 3)
#     transmittance = transmittance.view(H, W)

#     return {
#         "rgb":         final_rgb.clamp(0.0, 1.0),
#         "depth":       1.0 - transmittance,
#         "alpha_score": gauss_alpha_score,
#     }

"""
gaussian_scene.py  (fixed)
──────────────────────────
Loads a pre-trained 3DGS scene from a .ply file and provides a correct
pure-PyTorch rasteriser (MPS / CPU compatible).

Key fixes vs original:
  1. _quat_to_rotmat: row-major reshape was transposing every matrix → fixed
  2. EWA Jacobian: was using pixel coords (u,v) instead of camera-space (x,y,z)
  3. Sigma3D was not rotated into camera space before applying J
  4. dx/dy broadcast shapes were wrong at tile edges (bh ≠ bw)
  5. Tile loop inner `for k` still hit .item() per Gaussian → replaced with
     vectorised cumulative-product transmittance within each tile
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from rich.console import Console

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
#  Spherical Harmonics helpers  (degree-0 DC only)
# ─────────────────────────────────────────────────────────────────────────────
SH_C0 = 0.28209479177387814   # 1 / (2√π)

def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    return (sh_dc * SH_C0 + 0.5).clamp(0.0, 1.0)

def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


# ─────────────────────────────────────────────────────────────────────────────
#  Camera
# ─────────────────────────────────────────────────────────────────────────────
class Camera:
    """Pinhole camera.  R, t are world→camera (OpenCV convention)."""

    def __init__(self, R, t, fx, fy, cx, cy, width, height, device, image_path=None):
        self.R = R.to(device)
        self.t = t.to(device)
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.width  = width
        self.height = height
        self.device = device
        # Path to the original training image for this camera (used by the
        # segmenter instead of the 3DGS render — real photos give much better
        # GroundingDINO + SAM2 results than washed-out splat renders).
        self.image_path = image_path  # Path | None

    @property
    def world_center(self):
        return -(self.R.T @ self.t)

    def project(self, xyz_world: torch.Tensor):
        """(N,3) world → (N,2) pixel uv,  (N,) depth z."""
        xyz_cam = (xyz_world @ self.R.T) + self.t   # (N,3)
        z  = xyz_cam[:, 2].clamp(min=1e-6)
        u  = self.fx * xyz_cam[:, 0] / z + self.cx
        v  = self.fy * xyz_cam[:, 1] / z + self.cy
        return torch.stack([u, v], dim=-1), z


# ─────────────────────────────────────────────────────────────────────────────
#  GaussianScene
# ─────────────────────────────────────────────────────────────────────────────
class GaussianScene:

    def __init__(self, ply_path: Path, device: torch.device):
        self.device   = device
        self.ply_path = Path(ply_path)
        self._load_ply()

    def _load_ply(self):
        console.log(f"Loading scene: [cyan]{self.ply_path}[/cyan]")
        ply    = PlyData.read(str(self.ply_path))
        vertex = ply["vertex"]

        def col(name):
            return torch.tensor(np.array(vertex[name]), dtype=torch.float32)

        self.xyz     = torch.stack([col("x"), col("y"), col("z")], dim=-1).to(self.device)
        N            = self.xyz.shape[0]
        self.opacity = torch.sigmoid(col("opacity")).to(self.device)

        scales      = torch.stack([col(f"scale_{i}") for i in range(3)], dim=-1)
        self.scales = torch.exp(scales).to(self.device)
        # No scale clamping. Mip-NeRF 360 scenes (bicycle, bonsai, etc.) contain
        # large-scale Gaussians for branches, spokes, and background planes that are
        # valid geometry. Clamping them collapses their alpha contribution and breaks
        # the multi-view lifting step (Gaussians never accumulate enough score to be
        # segmented). Log the scale range for diagnostics only.
        scale_max_val = self.scales.max().item()
        console.log(f"  Scale range: max={scale_max_val:.3f}")

        # FIX: quaternion convention — official 3DGS PLY stores (w,x,y,z)
        # as rot_0..rot_3, which matches our _quat_to_rotmat expectation.
        self.quats = F.normalize(
            torch.stack([col(f"rot_{i}") for i in range(4)], dim=-1), dim=-1
        ).to(self.device)

        sh_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_dc")])
        if sh_names:
            sh_dc = torch.stack([col(n) for n in sh_names], dim=-1)
        else:
            try:
                sh_dc = torch.stack([col("red"), col("green"), col("blue")], dim=-1).float() / 255.0
                sh_dc = rgb_to_sh(sh_dc)
            except Exception:
                sh_dc = torch.zeros(N, 3)
        self.sh_dc = sh_dc.to(self.device)

        sh_rest_names = sorted([p.name for p in vertex.properties if p.name.startswith("f_rest")])
        self.sh_rest  = (
            torch.stack([col(n) for n in sh_rest_names], dim=-1).to(self.device)
            if sh_rest_names else None
        )

        console.log(f"  Loaded [bold]{N:,}[/bold] Gaussians")
        self.seg_mask = torch.zeros(N, dtype=torch.bool, device=self.device)

    @property
    def num_gaussians(self): return self.xyz.shape[0]

    @property
    def rgb(self): return sh_to_rgb(self.sh_dc)

    def recolor(self, rgb: tuple):
        if not self.seg_mask.any():
            console.log("[yellow]No Gaussians segmented – recolor skipped[/yellow]")
            return
        target = torch.tensor(rgb, dtype=torch.float32, device=self.device)
        self.sh_dc[self.seg_mask] = rgb_to_sh(target).unsqueeze(0)
        if self.sh_rest is not None:
            self.sh_rest[self.seg_mask] = 0.0
        console.log(f"  Recolored {self.seg_mask.sum().item():,} Gaussians → RGB{rgb}")

    def remove_segmented(self):
        keep      = ~self.seg_mask
        n_removed = self.seg_mask.sum().item()
        self.xyz     = self.xyz[keep]
        self.opacity = self.opacity[keep]
        self.scales  = self.scales[keep]
        self.quats   = self.quats[keep]
        self.sh_dc   = self.sh_dc[keep]
        if self.sh_rest is not None:
            self.sh_rest = self.sh_rest[keep]
        self.seg_mask = self.seg_mask[keep]
        console.log(f"  Removed {n_removed:,} Gaussians – {self.num_gaussians:,} remaining")

    def save_ply(self, out_path: Path):
        from plyfile import PlyElement, PlyData as PD
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        xyz           = self.xyz.cpu().numpy()
        opacity_logit = torch.logit(self.opacity.clamp(1e-6, 1-1e-6)).cpu().numpy()
        scales_log    = torch.log(self.scales.clamp(min=1e-8)).cpu().numpy()
        quats         = self.quats.cpu().numpy()
        sh_dc         = self.sh_dc.cpu().numpy()
        dtype_list = [
            ("x","f4"),("y","f4"),("z","f4"),("opacity","f4"),
            ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
            ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4"),
            ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
        ]
        N   = xyz.shape[0]
        arr = np.empty(N, dtype=dtype_list)
        arr["x"], arr["y"], arr["z"] = xyz[:,0], xyz[:,1], xyz[:,2]
        arr["opacity"] = opacity_logit
        for i in range(3): arr[f"scale_{i}"] = scales_log[:,i]
        for i in range(4): arr[f"rot_{i}"]   = quats[:,i]
        for i in range(3): arr[f"f_dc_{i}"]  = sh_dc[:,i]
        PD([PlyElement.describe(arr, "vertex")]).write(str(out_path))
        console.log(f"  Saved → [cyan]{out_path}[/cyan]")


# ─────────────────────────────────────────────────────────────────────────────
#  Rasteriser helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz quaternion → (N,3,3) rotation matrices.

    FIX: original code called .reshape(-1,3,3) on a flat stack of 9 scalars
    per Gaussian which transposes the matrix (row-major ≠ column-major).
    We build each row explicitly to be unambiguous.
    """
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    # Row 0
    r00 = 1 - 2*(y*y + z*z);  r01 = 2*(x*y - w*z);  r02 = 2*(x*z + w*y)
    # Row 1
    r10 = 2*(x*y + w*z);       r11 = 1 - 2*(x*x + z*z); r12 = 2*(y*z - w*x)
    # Row 2
    r20 = 2*(x*z - w*y);       r21 = 2*(y*z + w*x);  r22 = 1 - 2*(x*x + y*y)
    # Stack rows first, then reshape → correct (N,3,3) layout
    row0 = torch.stack([r00, r01, r02], dim=-1)   # (N,3)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)  # (N,3,3)


def _world_covariance(scales: torch.Tensor, quats: torch.Tensor) -> torch.Tensor:
    """3D world-space covariance:  Σ_world = R_g S Sᵀ R_gᵀ   shape (N,3,3)."""
    R_g = _quat_to_rotmat(quats)       # (N,3,3)
    S   = torch.diag_embed(scales)     # (N,3,3)
    RS  = R_g @ S                      # (N,3,3)
    return RS @ RS.transpose(-1, -2)   # (N,3,3)


# ─────────────────────────────────────────────────────────────────────────────
#  render_view
# ─────────────────────────────────────────────────────────────────────────────

def render_view(
    scene: GaussianScene,
    cam: Camera,
    background: Optional[torch.Tensor] = None,
    max_gaussians: int = 100_000,
    tile_size: int = 16,
    **kwargs,
) -> dict:
    """Tile-based alpha-compositing rasteriser.  MPS / CPU compatible.

    Returns dict with keys "rgb" (H,W,3), "depth" (H,W), "alpha_score" (N,).
    """
    device = scene.device
    H, W   = cam.height, cam.width

    if background is None:
        background = torch.ones(3, device=device)

    # ── 1. Project & cull ────────────────────────────────────────────────────
    uv, z = cam.project(scene.xyz)          # (N,2), (N,)

    margin = max(H, W) * 0.5
    valid  = (
        (z > 0.1)
        & (uv[:,0] > -margin) & (uv[:,0] < W + margin)
        & (uv[:,1] > -margin) & (uv[:,1] < H + margin)
    )

    if valid.sum() > max_gaussians:
        opac_tmp = scene.opacity.clone()
        opac_tmp[~valid] = -1.0
        _, top_idx = opac_tmp.topk(max_gaussians)
        mask2 = torch.zeros(scene.num_gaussians, dtype=torch.bool, device=device)
        mask2[top_idx] = True
        valid = valid & mask2

    idx      = torch.where(valid)[0]
    uv_v     = uv[idx]
    z_v      = z[idx]
    opac_v   = scene.opacity[idx]
    rgb_v    = scene.rgb[idx]
    scales_v = scene.scales[idx]
    quats_v  = scene.quats[idx]
    M        = idx.shape[0]

    if M == 0:
        return {
            "rgb":         background.view(1,1,3).expand(H,W,3).clone(),
            "depth":       torch.zeros(H, W, device=device),
            "alpha_score": torch.zeros(scene.num_gaussians, device=device),
        }

    # ── 2. 2D covariance (EWA splatting) ─────────────────────────────────────
    # Camera-space positions (needed for correct Jacobian)
    xyz_cam_v = (scene.xyz[idx] @ cam.R.T) + cam.t   # (M,3)
    xc = xyz_cam_v[:,0]   # (M,)
    yc = xyz_cam_v[:,1]
    # z_v == xyz_cam_v[:,2]  (already computed above, same values)

    # World-space 3D covariance
    Sigma_world = _world_covariance(scales_v, quats_v)   # (M,3,3)

    # Rotate into camera space:  Σ_cam = R_cam Σ_world R_camᵀ
    R  = cam.R.unsqueeze(0)                              # (1,3,3)
    Sigma_cam = R @ Sigma_world @ R.transpose(-1,-2)     # (M,3,3)

    # EWA perspective Jacobian (Zwicker et al. 2002, eq. 29):
    #   J = [[fx/z,  0,      -fx·x/z²],
    #        [0,     fy/z,   -fy·y/z²]]
    fx, fy = cam.fx, cam.fy
    z2 = z_v * z_v
    J  = torch.zeros(M, 2, 3, device=device)
    J[:,0,0] =  fx / z_v
    J[:,0,2] = -fx * xc / z2
    J[:,1,1] =  fy / z_v
    J[:,1,2] = -fy * yc / z2

    Sigma2D = J @ Sigma_cam @ J.transpose(-1,-2)     # (M,2,2)
    Sigma2D[:,0,0] += 0.3                             # low-pass filter (prevent aliasing)
    Sigma2D[:,1,1] += 0.3

    cov_xx = Sigma2D[:,0,0]
    cov_xy = Sigma2D[:,0,1]
    cov_yy = Sigma2D[:,1,1]

    det     = (cov_xx * cov_yy - cov_xy**2).clamp(min=1e-6)
    inv_det = 1.0 / det
    inv_xx  =  cov_yy * inv_det
    inv_xy  = -cov_xy * inv_det
    inv_yy  =  cov_xx * inv_det

    # Per-Gaussian screen-space bounding radius (3σ)
    trace   = cov_xx + cov_yy
    lambda1 = trace/2 + torch.sqrt(((trace/2)**2 - det).clamp(min=0))
    radii   = (3.0 * lambda1.sqrt()).ceil().int().clamp(1, max(H,W)//2)

    # ── 3. Depth sort (back-to-front) ────────────────────────────────────────
    order    = torch.argsort(z_v, descending=True)
    uv_v     = uv_v[order]
    opac_v   = opac_v[order]
    rgb_v    = rgb_v[order]
    inv_xx   = inv_xx[order]
    inv_xy   = inv_xy[order]
    inv_yy   = inv_yy[order]
    radii    = radii[order]
    orig_idx = idx[order]

    ux = uv_v[:,0]
    uy = uv_v[:,1]

    # ── 4. Tile rasterisation ─────────────────────────────────────────────────
    T     = tile_size
    n_ty  = math.ceil(H / T)
    n_tx  = math.ceil(W / T)

    r_np  = radii.cpu().numpy()
    ux_np = ux.cpu().numpy()
    uy_np = uy.cpu().numpy()

    canvas_rgb        = torch.zeros(H, W, 3, device=device)
    transmittance     = torch.ones(H, W,    device=device)
    gauss_alpha_score = torch.zeros(scene.num_gaussians, device=device)

    for ty in range(n_ty):
        y0 = ty * T;  y1 = min(y0 + T, H)
        for tx in range(n_tx):
            x0 = tx * T;  x1 = min(x0 + T, W)

            # Gaussians whose bbox overlaps this tile
            overlap = (
                (ux_np + r_np >= x0) & (ux_np - r_np < x1)
                & (uy_np + r_np >= y0) & (uy_np - r_np < y1)
            )
            if not overlap.any():
                continue

            t_idx  = torch.from_numpy(overlap.nonzero()[0]).to(device)
            K      = t_idx.shape[0]
            t_ux   = ux[t_idx]
            t_uy   = uy[t_idx]
            t_opac = opac_v[t_idx]
            t_rgb  = rgb_v[t_idx]
            t_ixx  = inv_xx[t_idx]
            t_ixy  = inv_xy[t_idx]
            t_iyy  = inv_yy[t_idx]

            bh = y1 - y0;  bw = x1 - x0

            # FIX: use .view() with explicit shapes to avoid non-contiguous broadcast
            px = torch.arange(x0, x1, device=device, dtype=torch.float32)  # (bw,)
            py = torch.arange(y0, y1, device=device, dtype=torch.float32)  # (bh,)

            dx = px.view(1, 1, bw) - t_ux.view(K, 1, 1)   # (K, 1, bw)
            dy = py.view(1, bh, 1) - t_uy.view(K, 1, 1)   # (K, bh, 1)
            # Both broadcast correctly to (K, bh, bw)

            maha = (
                t_ixx.view(K,1,1) * dx * dx
                + 2.0 * t_ixy.view(K,1,1) * dx * dy
                + t_iyy.view(K,1,1) * dy * dy
            )   # (K, bh, bw)

            g_alpha = t_opac.view(K,1,1) * torch.exp(-0.5*maha) * (maha < 8.0)
            # (K, bh, bw)

            # Vectorised alpha-compositing within tile:
            # T_before[k] = Π_{j<k}(1 - α_j)  for each pixel
            # Use cumprod then shift right by 1.
            one_minus = (1.0 - g_alpha)                      # (K, bh, bw)
            cumprod   = torch.cumprod(one_minus, dim=0)       # (K, bh, bw)
            T_before  = torch.cat([
                torch.ones(1, bh, bw, device=device),
                cumprod[:-1]
            ], dim=0)                                          # (K, bh, bw)

            weights = g_alpha * T_before                       # (K, bh, bw)

            # Colour accumulation: Σ_k w_k · rgb_k
            # rgb: (K,3) → (K,1,1,3); weights: (K,bh,bw,1)
            colour_contrib = (weights.unsqueeze(-1) * t_rgb.view(K,1,1,3)).sum(dim=0)
            canvas_rgb[y0:y1, x0:x1] += colour_contrib        # (bh,bw,3)

            # Residual transmittance for this tile
            T_final = cumprod[-1]                              # (bh, bw)
            transmittance[y0:y1, x0:x1] *= T_final

            # Alpha score for mask lifting: max weight per Gaussian
            peak_per_gauss = weights.view(K, -1).max(dim=1).values  # (K,)
            gauss_alpha_score.scatter_reduce_(
                0, orig_idx[t_idx], peak_per_gauss, reduce="amax", include_self=True
            )

    final_rgb = canvas_rgb + transmittance.unsqueeze(-1) * background

    return {
        "rgb":         final_rgb.clamp(0.0, 1.0),
        "depth":       1.0 - transmittance,
        "alpha_score": gauss_alpha_score,
    }