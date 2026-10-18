# # # """
# # # camera_utils.py
# # # ───────────────
# # # Load cameras from a pre-trained 3DGS scene directory.

# # # 3DGS scenes (Kerbl et al.) store camera info in:
# # #   - cameras.json  (written by the official viewer / training code)
# # #   - OR sparse/0/cameras.bin + images.bin  (raw COLMAP)

# # # We support both formats and return a list of Camera objects.
# # # """

# # # from __future__ import annotations

# # # import json
# # # import math
# # # import struct
# # # from pathlib import Path
# # # from typing import List, Optional

# # # import numpy as np
# # # import torch

# # # from gaussian_scene import Camera


# # # # ──────────────────────────────────────────────────────────────────────────────
# # # #  cameras.json loader  (preferred – output by the 3DGS training code)
# # # # ──────────────────────────────────────────────────────────────────────────────

# # # def _load_cameras_json(
# # #     json_path: Path,
# # #     width: int,
# # #     height: int,
# # #     device: torch.device,
# # #     max_cameras: int = 200,
# # # ) -> List[Camera]:
# # #     """Load cameras from the cameras.json written by the 3DGS trainer."""
# # #     with open(json_path) as f:
# # #         data = json.load(f)

# # #     cameras = []
# # #     for entry in data[:max_cameras]:
# # #         # The JSON stores the c2w (camera-to-world) 4x4 matrix
# # #         # We need w2c (world-to-camera)
# # #         img_w = entry.get("width", width)
# # #         img_h = entry.get("height", height)
# # #         fx = entry.get("fx", entry.get("fl_x", 800.0))
# # #         fy = entry.get("fy", entry.get("fl_y", 800.0))
# # #         cx = entry.get("cx", img_w / 2.0)
# # #         cy = entry.get("cy", img_h / 2.0)

# # #         # c2w matrix
# # #         c2w = np.array(entry["transform_matrix"], dtype=np.float32)
# # #         if c2w.shape == (4, 4):
# # #             R_c2w = c2w[:3, :3]
# # #             t_c2w = c2w[:3, 3]
# # #         else:
# # #             raise ValueError(f"Unexpected transform_matrix shape: {c2w.shape}")

# # #         # w2c = inverse of c2w
# # #         R_w2c = R_c2w.T
# # #         t_w2c = -R_c2w.T @ t_c2w

# # #         cam = Camera(
# # #             R=torch.from_numpy(R_w2c),
# # #             t=torch.from_numpy(t_w2c),
# # #             fx=float(fx), fy=float(fy),
# # #             cx=float(cx), cy=float(cy),
# # #             width=int(img_w), height=int(img_h),
# # #             device=device,
# # #         )
# # #         cameras.append(cam)

# # #     return cameras


# # # # ──────────────────────────────────────────────────────────────────────────────
# # # #  COLMAP binary loader  (fallback)
# # # # ──────────────────────────────────────────────────────────────────────────────

# # # def _read_colmap_cameras_bin(cameras_bin: Path) -> dict:
# # #     cameras = {}
# # #     with open(cameras_bin, "rb") as f:
# # #         num_cameras = struct.unpack("<Q", f.read(8))[0]
# # #         for _ in range(num_cameras):
# # #             cam_id   = struct.unpack("<I", f.read(4))[0]
# # #             model_id = struct.unpack("<I", f.read(4))[0]
# # #             w        = struct.unpack("<Q", f.read(8))[0]
# # #             h        = struct.unpack("<Q", f.read(8))[0]
# # #             num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 5, 5: 8, 6: 8, 7: 12}.get(model_id, 4)
# # #             params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
# # #             cameras[cam_id] = {"w": w, "h": h, "model_id": model_id, "params": params}
# # #     return cameras


# # # def _read_colmap_images_bin(images_bin: Path) -> list:
# # #     images = []
# # #     with open(images_bin, "rb") as f:
# # #         num_images = struct.unpack("<Q", f.read(8))[0]
# # #         for _ in range(num_images):
# # #             img_id    = struct.unpack("<I", f.read(4))[0]
# # #             qvec      = struct.unpack("<4d", f.read(32))   # qw, qx, qy, qz
# # #             tvec      = struct.unpack("<3d", f.read(24))
# # #             cam_id    = struct.unpack("<I", f.read(4))[0]
# # #             # read name (null-terminated)
# # #             name = b""
# # #             while True:
# # #                 c = f.read(1)
# # #                 if c == b"\x00":
# # #                     break
# # #                 name += c
# # #             # skip 2D points
# # #             num_pts = struct.unpack("<Q", f.read(8))[0]
# # #             f.read(num_pts * 24)
# # #             images.append({"id": img_id, "qvec": qvec, "tvec": tvec, "cam_id": cam_id, "name": name.decode()})
# # #     return images


# # # def _qvec_to_rotmat(qvec) -> np.ndarray:
# # #     """COLMAP quaternion (qw, qx, qy, qz) → 3×3 rotation matrix."""
# # #     qw, qx, qy, qz = qvec
# # #     R = np.array([
# # #         [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
# # #         [2*(qx*qy + qw*qz),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
# # #         [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),   1 - 2*(qx**2 + qy**2)],
# # #     ], dtype=np.float32)
# # #     return R


# # # def _load_colmap_cameras(
# # #     scene_dir: Path,
# # #     width: int,
# # #     height: int,
# # #     device: torch.device,
# # #     max_cameras: int = 200,
# # # ) -> List[Camera]:
# # #     cameras_bin = scene_dir / "sparse" / "0" / "cameras.bin"
# # #     images_bin  = scene_dir / "sparse" / "0" / "images.bin"

# # #     if not cameras_bin.exists() or not images_bin.exists():
# # #         return []

# # #     colmap_cams   = _read_colmap_cameras_bin(cameras_bin)
# # #     colmap_images = _read_colmap_images_bin(images_bin)

# # #     cameras = []
# # #     for img_entry in colmap_images[:max_cameras]:
# # #         cc = colmap_cams[img_entry["cam_id"]]
# # #         params = cc["params"]

# # #         # SIMPLE_PINHOLE: f, cx, cy
# # #         # PINHOLE: fx, fy, cx, cy
# # #         if cc["model_id"] == 0:
# # #             fx = fy = params[0]
# # #             cx, cy = params[1], params[2]
# # #         else:
# # #             fx, fy = params[0], params[1]
# # #             cx, cy = params[2], params[3]

# # #         R = _qvec_to_rotmat(img_entry["qvec"])
# # #         t = np.array(img_entry["tvec"], dtype=np.float32)

# # #         cam = Camera(
# # #             R=torch.from_numpy(R),
# # #             t=torch.from_numpy(t),
# # #             fx=float(fx), fy=float(fy),
# # #             cx=float(cx), cy=float(cy),
# # #             width=int(cc["w"]), height=int(cc["h"]),
# # #             device=device,
# # #         )
# # #         cameras.append(cam)

# # #     return cameras


# # # # ──────────────────────────────────────────────────────────────────────────────
# # # #  Public API
# # # # ──────────────────────────────────────────────────────────────────────────────

# # # def load_cameras(
# # #     scene_dir: Path,
# # #     width: int = 800,
# # #     height: int = 600,
# # #     device: torch.device = torch.device("cpu"),
# # #     max_cameras: int = 200,
# # # ) -> List[Camera]:
# # #     """Auto-detect camera format and load cameras from a 3DGS scene directory.

# # #     Search order:
# # #     1. cameras.json (written by 3DGS viewer)
# # #     2. transforms_train.json (NeRF-style, also supported by 3DGS)
# # #     3. sparse/0/cameras.bin + images.bin (raw COLMAP)

# # #     Args:
# # #         scene_dir: Root of the 3DGS scene (contains the .ply + camera files).
# # #         width, height: Override image resolution (used when not stored in file).
# # #         device: PyTorch device.
# # #         max_cameras: Cap on cameras to load (to limit memory usage).

# # #     Returns:
# # #         List of Camera objects.
# # #     """
# # #     scene_dir = Path(scene_dir)

# # #     # 1. cameras.json
# # #     cameras_json = scene_dir / "cameras.json"
# # #     if cameras_json.exists():
# # #         cams = _load_cameras_json(cameras_json, width, height, device, max_cameras)
# # #         if cams:
# # #             return cams

# # #     # 2. transforms_train.json  (NeRF-Synthetic / instant-ngp format)
# # #     for tf_name in ["transforms_train.json", "transforms.json"]:
# # #         tf_path = scene_dir / tf_name
# # #         if tf_path.exists():
# # #             # NeRF format uses fov → derive fx/fy
# # #             with open(tf_path) as f:
# # #                 tf = json.load(f)
# # #             fov_x = tf.get("camera_angle_x", math.pi / 3)
# # #             fx = 0.5 * width / math.tan(0.5 * fov_x)
# # #             fy = fx
# # #             cx, cy = width / 2.0, height / 2.0
# # #             cams = []
# # #             for frame in tf.get("frames", [])[:max_cameras]:
# # #                 c2w = np.array(frame["transform_matrix"], dtype=np.float32)
# # #                 R_c2w = c2w[:3, :3]
# # #                 t_c2w = c2w[:3, 3]
# # #                 R_w2c = R_c2w.T
# # #                 t_w2c = -R_c2w.T @ t_c2w
# # #                 cams.append(Camera(
# # #                     R=torch.from_numpy(R_w2c),
# # #                     t=torch.from_numpy(t_w2c),
# # #                     fx=fx, fy=fy, cx=cx, cy=cy,
# # #                     width=width, height=height, device=device,
# # #                 ))
# # #             if cams:
# # #                 return cams

# # #     # 3. COLMAP binary
# # #     cams = _load_colmap_cameras(scene_dir, width, height, device, max_cameras)
# # #     if cams:
# # #         return cams

# # #     # No camera files found – return empty list; caller should fall back to orbit cameras
# # #     return []


# # # def make_orbit_cameras(
# # #     center: torch.Tensor,       # (3,) scene centre
# # #     radius: float,
# # #     num_frames: int,
# # #     elevation_deg: float,
# # #     fx: float, fy: float,
# # #     width: int, height: int,
# # #     device: torch.device,
# # # ) -> List[Camera]:
# # #     """Generate an orbit path around `center` at a fixed elevation."""
# # #     cameras = []
# # #     elevation = math.radians(elevation_deg)
# # #     for i in range(num_frames):
# # #         azimuth = 2 * math.pi * i / num_frames
# # #         # Camera position in world coords
# # #         cam_pos = torch.tensor([
# # #             radius * math.cos(elevation) * math.cos(azimuth),
# # #             radius * math.sin(elevation),
# # #             radius * math.cos(elevation) * math.sin(azimuth),
# # #         ], device=device, dtype=torch.float32) + center

# # #         # Look-at → R, t
# # #         forward = F.normalize(center.float() - cam_pos, dim=0)
# # #         # world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
# # #         world_up = torch.tensor([0.0, -1.0, 0.0], device=device)

# # #         right    = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
# # #         up       = torch.cross(right, forward, dim=0)

# # #         # R_w2c rows = right, up, -forward  (OpenCV convention)
# # #         # R_w2c = torch.stack([right, up, -forward], dim=0)  # (3, 3)
# # #         R_w2c = torch.stack([right, -up, forward], dim=0)

# # #         t_w2c = R_w2c @ (-cam_pos)

# # #         cameras.append(Camera(
# # #             R=R_w2c, t=t_w2c,
# # #             fx=fx, fy=fy,
# # #             cx=width / 2.0, cy=height / 2.0,
# # #             width=width, height=height, device=device,
# # #         ))
# # #     return cameras


# # # # Avoid circular import
# # # import torch.nn.functional as F

# # """
# # camera_utils.py  (fixed)
# # ────────────────────────
# # Fixes vs original:
# #   1. Searches for COLMAP data in ALL standard Mip-NeRF 360 / 3DGS locations,
# #      including sparse/0/, sparse/, and the 3DGS-written cameras.json.
# #   2. make_orbit_cameras: fixed broken up-vector (was double-negating).
# #   3. Added diagnostic print when no cameras found so failure is obvious.
# # """

# # from __future__ import annotations

# # import json
# # import math
# # import struct
# # from pathlib import Path
# # from typing import List

# # import numpy as np
# # import torch
# # import torch.nn.functional as F

# # from gaussian_scene import Camera


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  cameras.json  (written by 3DGS trainer / viewer)
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _load_cameras_json(json_path, device, max_cameras=200):
# #     with open(json_path) as f:
# #         data = json.load(f)
# #     cameras = []
# #     for entry in data[:max_cameras]:
# #         img_w = int(entry.get("width",  entry.get("w", 800)))
# #         img_h = int(entry.get("height", entry.get("h", 600)))
# #         fx = float(entry.get("fx", entry.get("fl_x", img_w * 0.8)))
# #         fy = float(entry.get("fy", entry.get("fl_y", img_h * 0.8)))
# #         cx = float(entry.get("cx", img_w / 2.0))
# #         cy = float(entry.get("cy", img_h / 2.0))

# #         c2w = np.array(entry["transform_matrix"], dtype=np.float32)
# #         R_c2w = c2w[:3, :3]
# #         t_c2w = c2w[:3, 3]
# #         R_w2c = R_c2w.T
# #         t_w2c = (-R_c2w.T @ t_c2w).astype(np.float32)

# #         cameras.append(Camera(
# #             R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
# #             fx=fx, fy=fy, cx=cx, cy=cy,
# #             width=img_w, height=img_h, device=device,
# #         ))
# #     return cameras


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  COLMAP binary
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _read_colmap_cameras_bin(path):
# #     cams = {}
# #     with open(path, "rb") as f:
# #         n = struct.unpack("<Q", f.read(8))[0]
# #         for _ in range(n):
# #             cam_id   = struct.unpack("<I", f.read(4))[0]
# #             model_id = struct.unpack("<I", f.read(4))[0]
# #             w = struct.unpack("<Q", f.read(8))[0]
# #             h = struct.unpack("<Q", f.read(8))[0]
# #             np_ = {0:3, 1:4, 2:4, 3:5, 4:5, 5:8, 6:8, 7:12}.get(model_id, 4)
# #             params = struct.unpack(f"<{np_}d", f.read(8 * np_))
# #             cams[cam_id] = {"w":w, "h":h, "model_id":model_id, "params":params}
# #     return cams

# # def _read_colmap_images_bin(path):
# #     images = []
# #     with open(path, "rb") as f:
# #         n = struct.unpack("<Q", f.read(8))[0]
# #         for _ in range(n):
# #             img_id = struct.unpack("<I", f.read(4))[0]
# #             qvec   = struct.unpack("<4d", f.read(32))
# #             tvec   = struct.unpack("<3d", f.read(24))
# #             cam_id = struct.unpack("<I", f.read(4))[0]
# #             name = b""
# #             while True:
# #                 c = f.read(1)
# #                 if c == b"\x00": break
# #                 name += c
# #             num_pts = struct.unpack("<Q", f.read(8))[0]
# #             f.read(num_pts * 24)
# #             images.append({"qvec":qvec, "tvec":tvec, "cam_id":cam_id})
# #     return images

# # def _qvec_to_rotmat(qvec):
# #     qw, qx, qy, qz = qvec
# #     return np.array([
# #         [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
# #         [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
# #         [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)],
# #     ], dtype=np.float32)

# # def _load_colmap_cameras(sparse_dir, device, max_cameras=200,
# #                          render_width=None, render_height=None):
# #     cb = sparse_dir / "cameras.bin"
# #     ib = sparse_dir / "images.bin"
# #     if not cb.exists() or not ib.exists():
# #         return []
# #     colmap_cams   = _read_colmap_cameras_bin(cb)
# #     colmap_images = _read_colmap_images_bin(ib)
# #     cameras = []
# #     for img in colmap_images[:max_cameras]:
# #         cc = colmap_cams[img["cam_id"]]
# #         p  = cc["params"]
# #         if cc["model_id"] == 0:   # SIMPLE_PINHOLE
# #             fx = fy = p[0]; cx, cy = p[1], p[2]
# #         elif cc["model_id"] == 2: # SIMPLE_RADIAL
# #             fx = fy = p[0]; cx, cy = p[1], p[2]
# #         else:                     # PINHOLE and others
# #             fx, fy = p[0], p[1]; cx, cy = p[2], p[3]

# #         orig_w, orig_h = int(cc["w"]), int(cc["h"])

# #         # FIX: Rescale intrinsics to the render resolution so cam.project()
# #         # returns pixel coords in [0, render_width) x [0, render_height).
# #         # Without this, projected coords are in original camera space
# #         # (e.g. 1546x1032 for bonsai) but the rendered mask is 400x300 —
# #         # all mask lookups in _alpha_weighted_mask_score are clamped garbage.
# #         if render_width is not None and render_height is not None:
# #             sx = render_width  / orig_w
# #             sy = render_height / orig_h
# #             fx = fx * sx;  fy = fy * sy
# #             cx = cx * sx;  cy = cy * sy
# #             out_w, out_h = render_width, render_height
# #         else:
# #             out_w, out_h = orig_w, orig_h

# #         R = _qvec_to_rotmat(img["qvec"])
# #         t = np.array(img["tvec"], dtype=np.float32)
# #         cameras.append(Camera(
# #             R=torch.from_numpy(R), t=torch.from_numpy(t),
# #             fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
# #             width=out_w, height=out_h, device=device,
# #         ))
# #     return cameras


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  NeRF-style transforms.json
# # # ─────────────────────────────────────────────────────────────────────────────

# # def _load_transforms_json(tf_path, device, max_cameras=200):
# #     with open(tf_path) as f:
# #         tf = json.load(f)
# #     # Prefer per-frame intrinsics; fall back to top-level fov
# #     frames = tf.get("frames", [])[:max_cameras]
# #     cameras = []
# #     for frame in frames:
# #         w   = int(frame.get("w", tf.get("w", 800)))
# #         h   = int(frame.get("h", tf.get("h", 600)))
# #         if "fl_x" in frame or "fl_x" in tf:
# #             fx = float(frame.get("fl_x", tf.get("fl_x")))
# #             fy = float(frame.get("fl_y", tf.get("fl_y", fx)))
# #         else:
# #             fov_x = float(tf.get("camera_angle_x", math.pi/3))
# #             fx = fy = 0.5 * w / math.tan(0.5 * fov_x)
# #         cx = float(frame.get("cx", tf.get("cx", w/2.0)))
# #         cy = float(frame.get("cy", tf.get("cy", h/2.0)))
# #         c2w    = np.array(frame["transform_matrix"], dtype=np.float32)
# #         R_c2w  = c2w[:3,:3]; t_c2w = c2w[:3,3]
# #         R_w2c  = R_c2w.T
# #         t_w2c  = (-R_c2w.T @ t_c2w).astype(np.float32)
# #         cameras.append(Camera(
# #             R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
# #             fx=fx, fy=fy, cx=cx, cy=cy,
# #             width=w, height=h, device=device,
# #         ))
# #     return cameras


# # # ─────────────────────────────────────────────────────────────────────────────
# # #  Public API
# # # ─────────────────────────────────────────────────────────────────────────────


# # def load_cameras(
# #     scene_dir,
# #     width: int = 800,
# #     height: int = 600,
# #     device=torch.device("cpu"),
# #     max_cameras: int = 200,
# # ) -> List[Camera]:
# #     """Auto-detect and load cameras from a 3DGS / Mip-NeRF 360 scene directory.

# #     The `width` and `height` arguments are the RENDER resolution (from config).
# #     COLMAP intrinsics are rescaled from original image resolution to render
# #     resolution so that cam.project() returns pixel coords consistent with the
# #     rendered images that the segmenter and mask-lifting code will operate on.
# #     """
# #     scene_dir = Path(scene_dir)

# #     # 1. cameras.json (3DGS trainer output) — check multiple locations.
# #     # NOTE: if cameras.json exists alongside COLMAP sparse/ from a previous
# #     # viewer session, this would be picked first. We skip it when the user
# #     # explicitly wants COLMAP cameras (sparse/0/ present).
# #     colmap_present = any(
# #         (scene_dir / sd / "cameras.bin").exists()
# #         for sd in ["sparse/0", "sparse", "colmap/sparse/0"]
# #     )
# #     if not colmap_present:
# #         for p in [scene_dir / "cameras.json",
# #                   scene_dir / "camera_calibration" / "cameras.json"]:
# #             if p.exists():
# #                 cams = _load_cameras_json(p, device, max_cameras)
# #                 if cams:
# #                     print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
# #                     return cams

# #     # 2. COLMAP binary — always preferred when sparse/0/ is present.
# #     for sd_rel in ["sparse/0", "sparse", "colmap/sparse/0"]:
# #         sparse_dir_path = scene_dir / sd_rel
# #         cams = _load_colmap_cameras(
# #             sparse_dir_path, device, max_cameras,
# #             render_width=width, render_height=height,
# #         )
# #         if cams:
# #             print(f"[camera_utils] Loaded {len(cams)} cameras from {sparse_dir_path} "
# #                   f"(rescaled intrinsics to {width}x{height})")
# #             return cams

# #     # 3. NeRF-style transforms JSON
# #     for name in ["transforms_train.json", "transforms.json", "transforms_test.json"]:
# #         p = scene_dir / name
# #         if p.exists():
# #             cams = _load_transforms_json(p, device, max_cameras)
# #             if cams:
# #                 print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
# #                 return cams

# #     print(f"[camera_utils] WARNING: No camera files found in {scene_dir}")
# #     print(f"  Searched: cameras.json, sparse/0/cameras.bin, transforms_train.json")
# #     print(f"  Contents of scene_dir: {list(scene_dir.iterdir()) if scene_dir.exists() else 'DIR NOT FOUND'}")
# #     return []

# # def make_orbit_cameras(
# #     center: torch.Tensor,
# #     radius: float,
# #     num_frames: int,
# #     elevation_deg: float,
# #     fx: float, fy: float,
# #     width: int, height: int,
# #     device: torch.device,
# # ) -> List[Camera]:
# #     """Orbit cameras around `center`. Uses standard OpenCV convention."""
# #     cameras = []
# #     elevation = math.radians(elevation_deg)
# #     # World up = -Y in OpenCV/3DGS convention (Y points down in camera space)
# #     world_up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)

# #     for i in range(num_frames):
# #         azimuth = 2 * math.pi * i / num_frames
# #         cam_pos = center.float() + torch.tensor([
# #             radius * math.cos(elevation) * math.cos(azimuth),
# #             -radius * math.sin(elevation),   # negative: up in world = -Y
# #             radius * math.cos(elevation) * math.sin(azimuth),
# #         ], device=device, dtype=torch.float32)

# #         forward = F.normalize(center.float() - cam_pos, dim=0)  # points toward scene
# #         right   = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
# #         up      = torch.cross(right, forward, dim=0)  # recompute for orthogonality

# #         # OpenCV R_w2c: rows are right, down, forward
# #         # "down" in camera space = -up in world space
# #         R_w2c = torch.stack([right, -up, forward], dim=0)  # (3,3)
# #         t_w2c = R_w2c @ (-cam_pos)

# #         cameras.append(Camera(
# #             R=R_w2c, t=t_w2c,
# #             fx=fx, fy=fy,
# #             cx=width/2.0, cy=height/2.0,
# #             width=width, height=height, device=device,
# #         ))
# #     return cameras

# # """
# # camera_utils.py
# # ───────────────
# # Load cameras from a pre-trained 3DGS scene directory.

# # 3DGS scenes (Kerbl et al.) store camera info in:
# #   - cameras.json  (written by the official viewer / training code)
# #   - OR sparse/0/cameras.bin + images.bin  (raw COLMAP)

# # We support both formats and return a list of Camera objects.
# # """

# # from __future__ import annotations

# # import json
# # import math
# # import struct
# # from pathlib import Path
# # from typing import List, Optional

# # import numpy as np
# # import torch

# # from gaussian_scene import Camera


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  cameras.json loader  (preferred – output by the 3DGS training code)
# # # ──────────────────────────────────────────────────────────────────────────────

# # def _load_cameras_json(
# #     json_path: Path,
# #     width: int,
# #     height: int,
# #     device: torch.device,
# #     max_cameras: int = 200,
# # ) -> List[Camera]:
# #     """Load cameras from the cameras.json written by the 3DGS trainer."""
# #     with open(json_path) as f:
# #         data = json.load(f)

# #     cameras = []
# #     for entry in data[:max_cameras]:
# #         # The JSON stores the c2w (camera-to-world) 4x4 matrix
# #         # We need w2c (world-to-camera)
# #         img_w = entry.get("width", width)
# #         img_h = entry.get("height", height)
# #         fx = entry.get("fx", entry.get("fl_x", 800.0))
# #         fy = entry.get("fy", entry.get("fl_y", 800.0))
# #         cx = entry.get("cx", img_w / 2.0)
# #         cy = entry.get("cy", img_h / 2.0)

# #         # c2w matrix
# #         c2w = np.array(entry["transform_matrix"], dtype=np.float32)
# #         if c2w.shape == (4, 4):
# #             R_c2w = c2w[:3, :3]
# #             t_c2w = c2w[:3, 3]
# #         else:
# #             raise ValueError(f"Unexpected transform_matrix shape: {c2w.shape}")

# #         # w2c = inverse of c2w
# #         R_w2c = R_c2w.T
# #         t_w2c = -R_c2w.T @ t_c2w

# #         cam = Camera(
# #             R=torch.from_numpy(R_w2c),
# #             t=torch.from_numpy(t_w2c),
# #             fx=float(fx), fy=float(fy),
# #             cx=float(cx), cy=float(cy),
# #             width=int(img_w), height=int(img_h),
# #             device=device,
# #         )
# #         cameras.append(cam)

# #     return cameras


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  COLMAP binary loader  (fallback)
# # # ──────────────────────────────────────────────────────────────────────────────

# # def _read_colmap_cameras_bin(cameras_bin: Path) -> dict:
# #     cameras = {}
# #     with open(cameras_bin, "rb") as f:
# #         num_cameras = struct.unpack("<Q", f.read(8))[0]
# #         for _ in range(num_cameras):
# #             cam_id   = struct.unpack("<I", f.read(4))[0]
# #             model_id = struct.unpack("<I", f.read(4))[0]
# #             w        = struct.unpack("<Q", f.read(8))[0]
# #             h        = struct.unpack("<Q", f.read(8))[0]
# #             num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 5, 5: 8, 6: 8, 7: 12}.get(model_id, 4)
# #             params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
# #             cameras[cam_id] = {"w": w, "h": h, "model_id": model_id, "params": params}
# #     return cameras


# # def _read_colmap_images_bin(images_bin: Path) -> list:
# #     images = []
# #     with open(images_bin, "rb") as f:
# #         num_images = struct.unpack("<Q", f.read(8))[0]
# #         for _ in range(num_images):
# #             img_id    = struct.unpack("<I", f.read(4))[0]
# #             qvec      = struct.unpack("<4d", f.read(32))   # qw, qx, qy, qz
# #             tvec      = struct.unpack("<3d", f.read(24))
# #             cam_id    = struct.unpack("<I", f.read(4))[0]
# #             # read name (null-terminated)
# #             name = b""
# #             while True:
# #                 c = f.read(1)
# #                 if c == b"\x00":
# #                     break
# #                 name += c
# #             # skip 2D points
# #             num_pts = struct.unpack("<Q", f.read(8))[0]
# #             f.read(num_pts * 24)
# #             images.append({"id": img_id, "qvec": qvec, "tvec": tvec, "cam_id": cam_id, "name": name.decode()})
# #     return images


# # def _qvec_to_rotmat(qvec) -> np.ndarray:
# #     """COLMAP quaternion (qw, qx, qy, qz) → 3×3 rotation matrix."""
# #     qw, qx, qy, qz = qvec
# #     R = np.array([
# #         [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
# #         [2*(qx*qy + qw*qz),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
# #         [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),   1 - 2*(qx**2 + qy**2)],
# #     ], dtype=np.float32)
# #     return R


# # def _load_colmap_cameras(
# #     scene_dir: Path,
# #     width: int,
# #     height: int,
# #     device: torch.device,
# #     max_cameras: int = 200,
# # ) -> List[Camera]:
# #     cameras_bin = scene_dir / "sparse" / "0" / "cameras.bin"
# #     images_bin  = scene_dir / "sparse" / "0" / "images.bin"

# #     if not cameras_bin.exists() or not images_bin.exists():
# #         return []

# #     colmap_cams   = _read_colmap_cameras_bin(cameras_bin)
# #     colmap_images = _read_colmap_images_bin(images_bin)

# #     cameras = []
# #     for img_entry in colmap_images[:max_cameras]:
# #         cc = colmap_cams[img_entry["cam_id"]]
# #         params = cc["params"]

# #         # SIMPLE_PINHOLE: f, cx, cy
# #         # PINHOLE: fx, fy, cx, cy
# #         if cc["model_id"] == 0:
# #             fx = fy = params[0]
# #             cx, cy = params[1], params[2]
# #         else:
# #             fx, fy = params[0], params[1]
# #             cx, cy = params[2], params[3]

# #         R = _qvec_to_rotmat(img_entry["qvec"])
# #         t = np.array(img_entry["tvec"], dtype=np.float32)

# #         cam = Camera(
# #             R=torch.from_numpy(R),
# #             t=torch.from_numpy(t),
# #             fx=float(fx), fy=float(fy),
# #             cx=float(cx), cy=float(cy),
# #             width=int(cc["w"]), height=int(cc["h"]),
# #             device=device,
# #         )
# #         cameras.append(cam)

# #     return cameras


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  Public API
# # # ──────────────────────────────────────────────────────────────────────────────

# # def load_cameras(
# #     scene_dir: Path,
# #     width: int = 800,
# #     height: int = 600,
# #     device: torch.device = torch.device("cpu"),
# #     max_cameras: int = 200,
# # ) -> List[Camera]:
# #     """Auto-detect camera format and load cameras from a 3DGS scene directory.

# #     Search order:
# #     1. cameras.json (written by 3DGS viewer)
# #     2. transforms_train.json (NeRF-style, also supported by 3DGS)
# #     3. sparse/0/cameras.bin + images.bin (raw COLMAP)

# #     Args:
# #         scene_dir: Root of the 3DGS scene (contains the .ply + camera files).
# #         width, height: Override image resolution (used when not stored in file).
# #         device: PyTorch device.
# #         max_cameras: Cap on cameras to load (to limit memory usage).

# #     Returns:
# #         List of Camera objects.
# #     """
# #     scene_dir = Path(scene_dir)

# #     # 1. cameras.json
# #     cameras_json = scene_dir / "cameras.json"
# #     if cameras_json.exists():
# #         cams = _load_cameras_json(cameras_json, width, height, device, max_cameras)
# #         if cams:
# #             return cams

# #     # 2. transforms_train.json  (NeRF-Synthetic / instant-ngp format)
# #     for tf_name in ["transforms_train.json", "transforms.json"]:
# #         tf_path = scene_dir / tf_name
# #         if tf_path.exists():
# #             # NeRF format uses fov → derive fx/fy
# #             with open(tf_path) as f:
# #                 tf = json.load(f)
# #             fov_x = tf.get("camera_angle_x", math.pi / 3)
# #             fx = 0.5 * width / math.tan(0.5 * fov_x)
# #             fy = fx
# #             cx, cy = width / 2.0, height / 2.0
# #             cams = []
# #             for frame in tf.get("frames", [])[:max_cameras]:
# #                 c2w = np.array(frame["transform_matrix"], dtype=np.float32)
# #                 R_c2w = c2w[:3, :3]
# #                 t_c2w = c2w[:3, 3]
# #                 R_w2c = R_c2w.T
# #                 t_w2c = -R_c2w.T @ t_c2w
# #                 cams.append(Camera(
# #                     R=torch.from_numpy(R_w2c),
# #                     t=torch.from_numpy(t_w2c),
# #                     fx=fx, fy=fy, cx=cx, cy=cy,
# #                     width=width, height=height, device=device,
# #                 ))
# #             if cams:
# #                 return cams

# #     # 3. COLMAP binary
# #     cams = _load_colmap_cameras(scene_dir, width, height, device, max_cameras)
# #     if cams:
# #         return cams

# #     # No camera files found – return empty list; caller should fall back to orbit cameras
# #     return []


# # def make_orbit_cameras(
# #     center: torch.Tensor,       # (3,) scene centre
# #     radius: float,
# #     num_frames: int,
# #     elevation_deg: float,
# #     fx: float, fy: float,
# #     width: int, height: int,
# #     device: torch.device,
# # ) -> List[Camera]:
# #     """Generate an orbit path around `center` at a fixed elevation."""
# #     cameras = []
# #     elevation = math.radians(elevation_deg)
# #     for i in range(num_frames):
# #         azimuth = 2 * math.pi * i / num_frames
# #         # Camera position in world coords
# #         cam_pos = torch.tensor([
# #             radius * math.cos(elevation) * math.cos(azimuth),
# #             radius * math.sin(elevation),
# #             radius * math.cos(elevation) * math.sin(azimuth),
# #         ], device=device, dtype=torch.float32) + center

# #         # Look-at → R, t
# #         forward = F.normalize(center.float() - cam_pos, dim=0)
# #         # world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
# #         world_up = torch.tensor([0.0, -1.0, 0.0], device=device)

# #         right    = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
# #         up       = torch.cross(right, forward, dim=0)

# #         # R_w2c rows = right, up, -forward  (OpenCV convention)
# #         # R_w2c = torch.stack([right, up, -forward], dim=0)  # (3, 3)
# #         R_w2c = torch.stack([right, -up, forward], dim=0)

# #         t_w2c = R_w2c @ (-cam_pos)

# #         cameras.append(Camera(
# #             R=R_w2c, t=t_w2c,
# #             fx=fx, fy=fy,
# #             cx=width / 2.0, cy=height / 2.0,
# #             width=width, height=height, device=device,
# #         ))
# #     return cameras


# # # Avoid circular import
# # import torch.nn.functional as F

# """
# camera_utils.py  (fixed)
# ────────────────────────
# Fixes vs original:
#   1. Searches for COLMAP data in ALL standard Mip-NeRF 360 / 3DGS locations,
#      including sparse/0/, sparse/, and the 3DGS-written cameras.json.
#   2. make_orbit_cameras: fixed broken up-vector (was double-negating).
#   3. Added diagnostic print when no cameras found so failure is obvious.
# """

# from __future__ import annotations

# import json
# import math
# import struct
# from pathlib import Path
# from typing import List

# import numpy as np
# import torch
# import torch.nn.functional as F

# from gaussian_scene import Camera


# # ─────────────────────────────────────────────────────────────────────────────
# #  cameras.json  (written by 3DGS trainer / viewer)
# # ─────────────────────────────────────────────────────────────────────────────

# def _load_cameras_json(json_path, device, max_cameras=200):
#     with open(json_path) as f:
#         data = json.load(f)
#     cameras = []
#     for entry in data[:max_cameras]:
#         img_w = int(entry.get("width",  entry.get("w", 800)))
#         img_h = int(entry.get("height", entry.get("h", 600)))
#         fx = float(entry.get("fx", entry.get("fl_x", img_w * 0.8)))
#         fy = float(entry.get("fy", entry.get("fl_y", img_h * 0.8)))
#         cx = float(entry.get("cx", img_w / 2.0))
#         cy = float(entry.get("cy", img_h / 2.0))

#         c2w = np.array(entry["transform_matrix"], dtype=np.float32)
#         R_c2w = c2w[:3, :3]
#         t_c2w = c2w[:3, 3]
#         R_w2c = R_c2w.T
#         t_w2c = (-R_c2w.T @ t_c2w).astype(np.float32)

#         cameras.append(Camera(
#             R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
#             fx=fx, fy=fy, cx=cx, cy=cy,
#             width=img_w, height=img_h, device=device,
#         ))
#     return cameras


# # ─────────────────────────────────────────────────────────────────────────────
# #  COLMAP binary
# # ─────────────────────────────────────────────────────────────────────────────

# def _read_colmap_cameras_bin(path):
#     cams = {}
#     with open(path, "rb") as f:
#         n = struct.unpack("<Q", f.read(8))[0]
#         for _ in range(n):
#             cam_id   = struct.unpack("<I", f.read(4))[0]
#             model_id = struct.unpack("<I", f.read(4))[0]
#             w = struct.unpack("<Q", f.read(8))[0]
#             h = struct.unpack("<Q", f.read(8))[0]
#             np_ = {0:3, 1:4, 2:4, 3:5, 4:5, 5:8, 6:8, 7:12}.get(model_id, 4)
#             params = struct.unpack(f"<{np_}d", f.read(8 * np_))
#             cams[cam_id] = {"w":w, "h":h, "model_id":model_id, "params":params}
#     return cams

# def _read_colmap_images_bin(path):
#     images = []
#     with open(path, "rb") as f:
#         n = struct.unpack("<Q", f.read(8))[0]
#         for _ in range(n):
#             img_id = struct.unpack("<I", f.read(4))[0]
#             qvec   = struct.unpack("<4d", f.read(32))
#             tvec   = struct.unpack("<3d", f.read(24))
#             cam_id = struct.unpack("<I", f.read(4))[0]
#             name = b""
#             while True:
#                 c = f.read(1)
#                 if c == b"\x00": break
#                 name += c
#             num_pts = struct.unpack("<Q", f.read(8))[0]
#             f.read(num_pts * 24)
#             images.append({"qvec":qvec, "tvec":tvec, "cam_id":cam_id, "name":name.decode()})
#     return images

# def _qvec_to_rotmat(qvec):
#     qw, qx, qy, qz = qvec
#     return np.array([
#         [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
#         [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
#         [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)],
#     ], dtype=np.float32)

# def _load_colmap_cameras(sparse_dir, device, max_cameras=200,
#                          render_width=None, render_height=None):
#     cb = sparse_dir / "cameras.bin"
#     ib = sparse_dir / "images.bin"
#     if not cb.exists() or not ib.exists():
#         return []
#     colmap_cams   = _read_colmap_cameras_bin(cb)
#     colmap_images = _read_colmap_images_bin(ib)
#     cameras = []
#     for img in colmap_images[:max_cameras]:
#         cc = colmap_cams[img["cam_id"]]
#         p  = cc["params"]
#         if cc["model_id"] == 0:   # SIMPLE_PINHOLE
#             fx = fy = p[0]; cx, cy = p[1], p[2]
#         elif cc["model_id"] == 2: # SIMPLE_RADIAL
#             fx = fy = p[0]; cx, cy = p[1], p[2]
#         else:                     # PINHOLE and others
#             fx, fy = p[0], p[1]; cx, cy = p[2], p[3]

#         orig_w, orig_h = int(cc["w"]), int(cc["h"])

#         # FIX: Rescale intrinsics to the render resolution so cam.project()
#         # returns pixel coords in [0, render_width) x [0, render_height).
#         # Without this, projected coords are in original camera space
#         # (e.g. 1546x1032 for bonsai) but the rendered mask is 400x300 —
#         # all mask lookups in _alpha_weighted_mask_score are clamped garbage.
#         if render_width is not None and render_height is not None:
#             sx = render_width  / orig_w
#             sy = render_height / orig_h
#             fx = fx * sx;  fy = fy * sy
#             cx = cx * sx;  cy = cy * sy
#             out_w, out_h = render_width, render_height
#         else:
#             out_w, out_h = orig_w, orig_h

#         R = _qvec_to_rotmat(img["qvec"])
#         t = np.array(img["tvec"], dtype=np.float32)

#         # Resolve original training image path.
#         # Mip-NeRF 360 stores images under images_4/ (quarter-res) or images_2/.
#         # We prefer the smallest available resolution that is still larger than
#         # the render resolution, so GroundingDINO gets a real photo without
#         # loading unnecessarily large files.
#         img_name = img.get("name", "")
#         scene_dir = sparse_dir.parent.parent if sparse_dir.name == "0" else sparse_dir.parent
#         image_path = None
#         for img_subdir in ["images_4", "images_2", "images"]:
#             candidate = scene_dir / img_subdir / img_name
#             if candidate.exists():
#                 image_path = candidate
#                 break

#         cameras.append(Camera(
#             R=torch.from_numpy(R), t=torch.from_numpy(t),
#             fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
#             width=out_w, height=out_h, device=device,
#             image_path=image_path,
#         ))
#     return cameras


# # ─────────────────────────────────────────────────────────────────────────────
# #  NeRF-style transforms.json
# # ─────────────────────────────────────────────────────────────────────────────

# def _load_transforms_json(tf_path, device, max_cameras=200):
#     with open(tf_path) as f:
#         tf = json.load(f)
#     # Prefer per-frame intrinsics; fall back to top-level fov
#     frames = tf.get("frames", [])[:max_cameras]
#     cameras = []
#     for frame in frames:
#         w   = int(frame.get("w", tf.get("w", 800)))
#         h   = int(frame.get("h", tf.get("h", 600)))
#         if "fl_x" in frame or "fl_x" in tf:
#             fx = float(frame.get("fl_x", tf.get("fl_x")))
#             fy = float(frame.get("fl_y", tf.get("fl_y", fx)))
#         else:
#             fov_x = float(tf.get("camera_angle_x", math.pi/3))
#             fx = fy = 0.5 * w / math.tan(0.5 * fov_x)
#         cx = float(frame.get("cx", tf.get("cx", w/2.0)))
#         cy = float(frame.get("cy", tf.get("cy", h/2.0)))
#         c2w    = np.array(frame["transform_matrix"], dtype=np.float32)
#         R_c2w  = c2w[:3,:3]; t_c2w = c2w[:3,3]
#         R_w2c  = R_c2w.T
#         t_w2c  = (-R_c2w.T @ t_c2w).astype(np.float32)
#         cameras.append(Camera(
#             R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
#             fx=fx, fy=fy, cx=cx, cy=cy,
#             width=w, height=h, device=device,
#         ))
#     return cameras


# # ─────────────────────────────────────────────────────────────────────────────
# #  Public API
# # ─────────────────────────────────────────────────────────────────────────────


# def load_cameras(
#     scene_dir,
#     width: int = 800,
#     height: int = 600,
#     device=torch.device("cpu"),
#     max_cameras: int = 200,
# ) -> List[Camera]:
#     """Auto-detect and load cameras from a 3DGS / Mip-NeRF 360 scene directory.

#     The `width` and `height` arguments are the RENDER resolution (from config).
#     COLMAP intrinsics are rescaled from original image resolution to render
#     resolution so that cam.project() returns pixel coords consistent with the
#     rendered images that the segmenter and mask-lifting code will operate on.
#     """
#     scene_dir = Path(scene_dir)

#     # 1. cameras.json (3DGS trainer output) — check multiple locations.
#     # NOTE: if cameras.json exists alongside COLMAP sparse/ from a previous
#     # viewer session, this would be picked first. We skip it when the user
#     # explicitly wants COLMAP cameras (sparse/0/ present).
#     colmap_present = any(
#         (scene_dir / sd / "cameras.bin").exists()
#         for sd in ["sparse/0", "sparse", "colmap/sparse/0"]
#     )
#     if not colmap_present:
#         for p in [scene_dir / "cameras.json",
#                   scene_dir / "camera_calibration" / "cameras.json"]:
#             if p.exists():
#                 cams = _load_cameras_json(p, device, max_cameras)
#                 if cams:
#                     print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
#                     return cams

#     # 2. COLMAP binary — always preferred when sparse/0/ is present.
#     for sd_rel in ["sparse/0", "sparse", "colmap/sparse/0"]:
#         sparse_dir_path = scene_dir / sd_rel
#         cams = _load_colmap_cameras(
#             sparse_dir_path, device, max_cameras,
#             render_width=width, render_height=height,
#         )
#         if cams:
#             n_with_img = sum(1 for c in cams if c.image_path is not None)
#         print(f"[camera_utils] Loaded {len(cams)} cameras from {sparse_dir_path} "
#                   f"(rescaled intrinsics to {width}x{height}, "
#                   f"{n_with_img}/{len(cams)} have original image paths)")
#         return cams

#     # 3. NeRF-style transforms JSON
#     for name in ["transforms_train.json", "transforms.json", "transforms_test.json"]:
#         p = scene_dir / name
#         if p.exists():
#             cams = _load_transforms_json(p, device, max_cameras)
#             if cams:
#                 print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
#                 return cams

#     print(f"[camera_utils] WARNING: No camera files found in {scene_dir}")
#     print(f"  Searched: cameras.json, sparse/0/cameras.bin, transforms_train.json")
#     print(f"  Contents of scene_dir: {list(scene_dir.iterdir()) if scene_dir.exists() else 'DIR NOT FOUND'}")
#     return []

# def make_orbit_cameras(
#     center: torch.Tensor,
#     radius: float,
#     num_frames: int,
#     elevation_deg: float,
#     fx: float, fy: float,
#     width: int, height: int,
#     device: torch.device,
# ) -> List[Camera]:
#     """Orbit cameras around `center`. Uses standard OpenCV convention."""
#     cameras = []
#     elevation = math.radians(elevation_deg)
#     # World up = -Y in OpenCV/3DGS convention (Y points down in camera space)
#     world_up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)

#     for i in range(num_frames):
#         azimuth = 2 * math.pi * i / num_frames
#         cam_pos = center.float() + torch.tensor([
#             radius * math.cos(elevation) * math.cos(azimuth),
#             -radius * math.sin(elevation),   # negative: up in world = -Y
#             radius * math.cos(elevation) * math.sin(azimuth),
#         ], device=device, dtype=torch.float32)

#         forward = F.normalize(center.float() - cam_pos, dim=0)  # points toward scene
#         right   = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
#         up      = torch.cross(right, forward, dim=0)  # recompute for orthogonality

#         # OpenCV R_w2c: rows are right, down, forward
#         # "down" in camera space = -up in world space
#         R_w2c = torch.stack([right, -up, forward], dim=0)  # (3,3)
#         t_w2c = R_w2c @ (-cam_pos)

#         cameras.append(Camera(
#             R=R_w2c, t=t_w2c,
#             fx=fx, fy=fy,
#             cx=width/2.0, cy=height/2.0,
#             width=width, height=height, device=device,
#         ))
#     return cameras
# # """
# # camera_utils.py
# # ───────────────
# # Load cameras from a pre-trained 3DGS scene directory.

# # 3DGS scenes (Kerbl et al.) store camera info in:
# #   - cameras.json  (written by the official viewer / training code)
# #   - OR sparse/0/cameras.bin + images.bin  (raw COLMAP)

# # We support both formats and return a list of Camera objects.
# # """

# # from __future__ import annotations

# # import json
# # import math
# # import struct
# # from pathlib import Path
# # from typing import List, Optional

# # import numpy as np
# # import torch

# # from gaussian_scene import Camera


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  cameras.json loader  (preferred – output by the 3DGS training code)
# # # ──────────────────────────────────────────────────────────────────────────────

# # def _load_cameras_json(
# #     json_path: Path,
# #     width: int,
# #     height: int,
# #     device: torch.device,
# #     max_cameras: int = 200,
# # ) -> List[Camera]:
# #     """Load cameras from the cameras.json written by the 3DGS trainer."""
# #     with open(json_path) as f:
# #         data = json.load(f)

# #     cameras = []
# #     for entry in data[:max_cameras]:
# #         # The JSON stores the c2w (camera-to-world) 4x4 matrix
# #         # We need w2c (world-to-camera)
# #         img_w = entry.get("width", width)
# #         img_h = entry.get("height", height)
# #         fx = entry.get("fx", entry.get("fl_x", 800.0))
# #         fy = entry.get("fy", entry.get("fl_y", 800.0))
# #         cx = entry.get("cx", img_w / 2.0)
# #         cy = entry.get("cy", img_h / 2.0)

# #         # c2w matrix
# #         c2w = np.array(entry["transform_matrix"], dtype=np.float32)
# #         if c2w.shape == (4, 4):
# #             R_c2w = c2w[:3, :3]
# #             t_c2w = c2w[:3, 3]
# #         else:
# #             raise ValueError(f"Unexpected transform_matrix shape: {c2w.shape}")

# #         # w2c = inverse of c2w
# #         R_w2c = R_c2w.T
# #         t_w2c = -R_c2w.T @ t_c2w

# #         cam = Camera(
# #             R=torch.from_numpy(R_w2c),
# #             t=torch.from_numpy(t_w2c),
# #             fx=float(fx), fy=float(fy),
# #             cx=float(cx), cy=float(cy),
# #             width=int(img_w), height=int(img_h),
# #             device=device,
# #         )
# #         cameras.append(cam)

# #     return cameras


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  COLMAP binary loader  (fallback)
# # # ──────────────────────────────────────────────────────────────────────────────

# # def _read_colmap_cameras_bin(cameras_bin: Path) -> dict:
# #     cameras = {}
# #     with open(cameras_bin, "rb") as f:
# #         num_cameras = struct.unpack("<Q", f.read(8))[0]
# #         for _ in range(num_cameras):
# #             cam_id   = struct.unpack("<I", f.read(4))[0]
# #             model_id = struct.unpack("<I", f.read(4))[0]
# #             w        = struct.unpack("<Q", f.read(8))[0]
# #             h        = struct.unpack("<Q", f.read(8))[0]
# #             num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 5, 5: 8, 6: 8, 7: 12}.get(model_id, 4)
# #             params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
# #             cameras[cam_id] = {"w": w, "h": h, "model_id": model_id, "params": params}
# #     return cameras


# # def _read_colmap_images_bin(images_bin: Path) -> list:
# #     images = []
# #     with open(images_bin, "rb") as f:
# #         num_images = struct.unpack("<Q", f.read(8))[0]
# #         for _ in range(num_images):
# #             img_id    = struct.unpack("<I", f.read(4))[0]
# #             qvec      = struct.unpack("<4d", f.read(32))   # qw, qx, qy, qz
# #             tvec      = struct.unpack("<3d", f.read(24))
# #             cam_id    = struct.unpack("<I", f.read(4))[0]
# #             # read name (null-terminated)
# #             name = b""
# #             while True:
# #                 c = f.read(1)
# #                 if c == b"\x00":
# #                     break
# #                 name += c
# #             # skip 2D points
# #             num_pts = struct.unpack("<Q", f.read(8))[0]
# #             f.read(num_pts * 24)
# #             images.append({"id": img_id, "qvec": qvec, "tvec": tvec, "cam_id": cam_id, "name": name.decode()})
# #     return images


# # def _qvec_to_rotmat(qvec) -> np.ndarray:
# #     """COLMAP quaternion (qw, qx, qy, qz) → 3×3 rotation matrix."""
# #     qw, qx, qy, qz = qvec
# #     R = np.array([
# #         [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
# #         [2*(qx*qy + qw*qz),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
# #         [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),   1 - 2*(qx**2 + qy**2)],
# #     ], dtype=np.float32)
# #     return R


# # def _load_colmap_cameras(
# #     scene_dir: Path,
# #     width: int,
# #     height: int,
# #     device: torch.device,
# #     max_cameras: int = 200,
# # ) -> List[Camera]:
# #     cameras_bin = scene_dir / "sparse" / "0" / "cameras.bin"
# #     images_bin  = scene_dir / "sparse" / "0" / "images.bin"

# #     if not cameras_bin.exists() or not images_bin.exists():
# #         return []

# #     colmap_cams   = _read_colmap_cameras_bin(cameras_bin)
# #     colmap_images = _read_colmap_images_bin(images_bin)

# #     cameras = []
# #     for img_entry in colmap_images[:max_cameras]:
# #         cc = colmap_cams[img_entry["cam_id"]]
# #         params = cc["params"]

# #         # SIMPLE_PINHOLE: f, cx, cy
# #         # PINHOLE: fx, fy, cx, cy
# #         if cc["model_id"] == 0:
# #             fx = fy = params[0]
# #             cx, cy = params[1], params[2]
# #         else:
# #             fx, fy = params[0], params[1]
# #             cx, cy = params[2], params[3]

# #         R = _qvec_to_rotmat(img_entry["qvec"])
# #         t = np.array(img_entry["tvec"], dtype=np.float32)

# #         cam = Camera(
# #             R=torch.from_numpy(R),
# #             t=torch.from_numpy(t),
# #             fx=float(fx), fy=float(fy),
# #             cx=float(cx), cy=float(cy),
# #             width=int(cc["w"]), height=int(cc["h"]),
# #             device=device,
# #         )
# #         cameras.append(cam)

# #     return cameras


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  Public API
# # # ──────────────────────────────────────────────────────────────────────────────

# # def load_cameras(
# #     scene_dir: Path,
# #     width: int = 800,
# #     height: int = 600,
# #     device: torch.device = torch.device("cpu"),
# #     max_cameras: int = 200,
# # ) -> List[Camera]:
# #     """Auto-detect camera format and load cameras from a 3DGS scene directory.

# #     Search order:
# #     1. cameras.json (written by 3DGS viewer)
# #     2. transforms_train.json (NeRF-style, also supported by 3DGS)
# #     3. sparse/0/cameras.bin + images.bin (raw COLMAP)

# #     Args:
# #         scene_dir: Root of the 3DGS scene (contains the .ply + camera files).
# #         width, height: Override image resolution (used when not stored in file).
# #         device: PyTorch device.
# #         max_cameras: Cap on cameras to load (to limit memory usage).

# #     Returns:
# #         List of Camera objects.
# #     """
# #     scene_dir = Path(scene_dir)

# #     # 1. cameras.json
# #     cameras_json = scene_dir / "cameras.json"
# #     if cameras_json.exists():
# #         cams = _load_cameras_json(cameras_json, width, height, device, max_cameras)
# #         if cams:
# #             return cams

# #     # 2. transforms_train.json  (NeRF-Synthetic / instant-ngp format)
# #     for tf_name in ["transforms_train.json", "transforms.json"]:
# #         tf_path = scene_dir / tf_name
# #         if tf_path.exists():
# #             # NeRF format uses fov → derive fx/fy
# #             with open(tf_path) as f:
# #                 tf = json.load(f)
# #             fov_x = tf.get("camera_angle_x", math.pi / 3)
# #             fx = 0.5 * width / math.tan(0.5 * fov_x)
# #             fy = fx
# #             cx, cy = width / 2.0, height / 2.0
# #             cams = []
# #             for frame in tf.get("frames", [])[:max_cameras]:
# #                 c2w = np.array(frame["transform_matrix"], dtype=np.float32)
# #                 R_c2w = c2w[:3, :3]
# #                 t_c2w = c2w[:3, 3]
# #                 R_w2c = R_c2w.T
# #                 t_w2c = -R_c2w.T @ t_c2w
# #                 cams.append(Camera(
# #                     R=torch.from_numpy(R_w2c),
# #                     t=torch.from_numpy(t_w2c),
# #                     fx=fx, fy=fy, cx=cx, cy=cy,
# #                     width=width, height=height, device=device,
# #                 ))
# #             if cams:
# #                 return cams

# #     # 3. COLMAP binary
# #     cams = _load_colmap_cameras(scene_dir, width, height, device, max_cameras)
# #     if cams:
# #         return cams

# #     # No camera files found – return empty list; caller should fall back to orbit cameras
# #     return []


# # def make_orbit_cameras(
# #     center: torch.Tensor,       # (3,) scene centre
# #     radius: float,
# #     num_frames: int,
# #     elevation_deg: float,
# #     fx: float, fy: float,
# #     width: int, height: int,
# #     device: torch.device,
# # ) -> List[Camera]:
# #     """Generate an orbit path around `center` at a fixed elevation."""
# #     cameras = []
# #     elevation = math.radians(elevation_deg)
# #     for i in range(num_frames):
# #         azimuth = 2 * math.pi * i / num_frames
# #         # Camera position in world coords
# #         cam_pos = torch.tensor([
# #             radius * math.cos(elevation) * math.cos(azimuth),
# #             radius * math.sin(elevation),
# #             radius * math.cos(elevation) * math.sin(azimuth),
# #         ], device=device, dtype=torch.float32) + center

# #         # Look-at → R, t
# #         forward = F.normalize(center.float() - cam_pos, dim=0)
# #         # world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
# #         world_up = torch.tensor([0.0, -1.0, 0.0], device=device)

# #         right    = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
# #         up       = torch.cross(right, forward, dim=0)

# #         # R_w2c rows = right, up, -forward  (OpenCV convention)
# #         # R_w2c = torch.stack([right, up, -forward], dim=0)  # (3, 3)
# #         R_w2c = torch.stack([right, -up, forward], dim=0)

# #         t_w2c = R_w2c @ (-cam_pos)

# #         cameras.append(Camera(
# #             R=R_w2c, t=t_w2c,
# #             fx=fx, fy=fy,
# #             cx=width / 2.0, cy=height / 2.0,
# #             width=width, height=height, device=device,
# #         ))
# #     return cameras


# # # Avoid circular import
# # import torch.nn.functional as F

# """
# camera_utils.py  (fixed)
# ────────────────────────
# Fixes vs original:
#   1. Searches for COLMAP data in ALL standard Mip-NeRF 360 / 3DGS locations,
#      including sparse/0/, sparse/, and the 3DGS-written cameras.json.
#   2. make_orbit_cameras: fixed broken up-vector (was double-negating).
#   3. Added diagnostic print when no cameras found so failure is obvious.
# """

# from __future__ import annotations

# import json
# import math
# import struct
# from pathlib import Path
# from typing import List

# import numpy as np
# import torch
# import torch.nn.functional as F

# from gaussian_scene import Camera


# # ─────────────────────────────────────────────────────────────────────────────
# #  cameras.json  (written by 3DGS trainer / viewer)
# # ─────────────────────────────────────────────────────────────────────────────

# def _load_cameras_json(json_path, device, max_cameras=200):
#     with open(json_path) as f:
#         data = json.load(f)
#     cameras = []
#     for entry in data[:max_cameras]:
#         img_w = int(entry.get("width",  entry.get("w", 800)))
#         img_h = int(entry.get("height", entry.get("h", 600)))
#         fx = float(entry.get("fx", entry.get("fl_x", img_w * 0.8)))
#         fy = float(entry.get("fy", entry.get("fl_y", img_h * 0.8)))
#         cx = float(entry.get("cx", img_w / 2.0))
#         cy = float(entry.get("cy", img_h / 2.0))

#         c2w = np.array(entry["transform_matrix"], dtype=np.float32)
#         R_c2w = c2w[:3, :3]
#         t_c2w = c2w[:3, 3]
#         R_w2c = R_c2w.T
#         t_w2c = (-R_c2w.T @ t_c2w).astype(np.float32)

#         cameras.append(Camera(
#             R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
#             fx=fx, fy=fy, cx=cx, cy=cy,
#             width=img_w, height=img_h, device=device,
#         ))
#     return cameras


# # ─────────────────────────────────────────────────────────────────────────────
# #  COLMAP binary
# # ─────────────────────────────────────────────────────────────────────────────

# def _read_colmap_cameras_bin(path):
#     cams = {}
#     with open(path, "rb") as f:
#         n = struct.unpack("<Q", f.read(8))[0]
#         for _ in range(n):
#             cam_id   = struct.unpack("<I", f.read(4))[0]
#             model_id = struct.unpack("<I", f.read(4))[0]
#             w = struct.unpack("<Q", f.read(8))[0]
#             h = struct.unpack("<Q", f.read(8))[0]
#             np_ = {0:3, 1:4, 2:4, 3:5, 4:5, 5:8, 6:8, 7:12}.get(model_id, 4)
#             params = struct.unpack(f"<{np_}d", f.read(8 * np_))
#             cams[cam_id] = {"w":w, "h":h, "model_id":model_id, "params":params}
#     return cams

# def _read_colmap_images_bin(path):
#     images = []
#     with open(path, "rb") as f:
#         n = struct.unpack("<Q", f.read(8))[0]
#         for _ in range(n):
#             img_id = struct.unpack("<I", f.read(4))[0]
#             qvec   = struct.unpack("<4d", f.read(32))
#             tvec   = struct.unpack("<3d", f.read(24))
#             cam_id = struct.unpack("<I", f.read(4))[0]
#             name = b""
#             while True:
#                 c = f.read(1)
#                 if c == b"\x00": break
#                 name += c
#             num_pts = struct.unpack("<Q", f.read(8))[0]
#             f.read(num_pts * 24)
#             images.append({"qvec":qvec, "tvec":tvec, "cam_id":cam_id, "name":name.decode()})
#     return images

# def _qvec_to_rotmat(qvec):
#     qw, qx, qy, qz = qvec
#     return np.array([
#         [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
#         [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
#         [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)],
#     ], dtype=np.float32)

# def _load_colmap_cameras(sparse_dir, device, max_cameras=200,
#                          target_long_edge=800):
#     """Load COLMAP cameras with UNIFORM rescaling to preserve aspect ratio.

#     CRITICAL: render_width/render_height are NOT accepted as separate params.
#     Instead we use target_long_edge and derive render_w, render_h per-camera
#     by uniformly scaling so the longest edge = target_long_edge.

#     Why uniform scaling matters
#     ---------------------------
#     The previous code accepted render_width=800, render_height=600 and
#     applied NON-UNIFORM scaling:
#         sx = render_width  / orig_w
#         sy = render_height / orig_h
#     For a PORTRAIT camera (e.g. bonsai: 1520x2028):
#         sx = 800/1520 = 0.526
#         sy = 600/2028 = 0.296   ← 2x different!
#     This stretches the camera frustum, making cam.project() return pixel
#     coordinates that don't align with what the renderer produces.
#     The segmentation mask and the Gaussian projections are in different
#     coordinate spaces → lifting scores are computed on wrong pixels.

#     With uniform scaling (sx == sy):
#         scale = 800 / max(1520, 2028) = 800/2028 = 0.395
#         render_w = round_even(1520 * 0.395) = 598
#         render_h = round_even(2028 * 0.395) = 800
#         sx = sy = 0.395 → fx, fy both scaled correctly
#     """
#     cb = sparse_dir / "cameras.bin"
#     ib = sparse_dir / "images.bin"
#     if not cb.exists() or not ib.exists():
#         return []
#     colmap_cams   = _read_colmap_cameras_bin(cb)
#     colmap_images = _read_colmap_images_bin(ib)
#     cameras = []
#     for img in colmap_images[:max_cameras]:
#         cc = colmap_cams[img["cam_id"]]
#         p  = cc["params"]
#         if cc["model_id"] == 0:   # SIMPLE_PINHOLE
#             fx = fy = p[0]; cx, cy = p[1], p[2]
#         elif cc["model_id"] == 2: # SIMPLE_RADIAL
#             fx = fy = p[0]; cx, cy = p[1], p[2]
#         else:                     # PINHOLE, OPENCV, FULL_OPENCV
#             fx, fy = p[0], p[1]; cx, cy = p[2], p[3]

#         orig_w, orig_h = int(cc["w"]), int(cc["h"])

#         # Uniform scale: longest edge → target_long_edge
#         scale  = target_long_edge / max(orig_w, orig_h)
#         out_w  = int(orig_w * scale)
#         out_h  = int(orig_h * scale)
#         # Round to nearest even (required by some video/image encoders)
#         out_w  = max(2, (out_w  // 2) * 2)
#         out_h  = max(2, (out_h  // 2) * 2)
#         # Recompute exact scale from rounded output dims
#         sx     = out_w / orig_w
#         sy     = out_h / orig_h

#         fx = fx * sx;  fy = fy * sy
#         cx = cx * sx;  cy = cy * sy

#         R = _qvec_to_rotmat(img["qvec"])
#         t = np.array(img["tvec"], dtype=np.float32)

#         # Resolve original training image path.
#         img_name  = img.get("name", "")
#         scene_dir = sparse_dir.parent.parent if sparse_dir.name == "0" else sparse_dir.parent
#         image_path = None
#         for img_subdir in ["images_4", "images_2", "images"]:
#             candidate = scene_dir / img_subdir / img_name
#             if candidate.exists():
#                 image_path = candidate
#                 break

#         cameras.append(Camera(
#             R=torch.from_numpy(R), t=torch.from_numpy(t),
#             fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
#             width=out_w, height=out_h, device=device,
#             image_path=image_path,
#         ))
#     return cameras


# # ─────────────────────────────────────────────────────────────────────────────
# #  NeRF-style transforms.json
# # ─────────────────────────────────────────────────────────────────────────────

# def _load_transforms_json(tf_path, device, max_cameras=200):
#     with open(tf_path) as f:
#         tf = json.load(f)
#     # Prefer per-frame intrinsics; fall back to top-level fov
#     frames = tf.get("frames", [])[:max_cameras]
#     cameras = []
#     for frame in frames:
#         w   = int(frame.get("w", tf.get("w", 800)))
#         h   = int(frame.get("h", tf.get("h", 600)))
#         if "fl_x" in frame or "fl_x" in tf:
#             fx = float(frame.get("fl_x", tf.get("fl_x")))
#             fy = float(frame.get("fl_y", tf.get("fl_y", fx)))
#         else:
#             fov_x = float(tf.get("camera_angle_x", math.pi/3))
#             fx = fy = 0.5 * w / math.tan(0.5 * fov_x)
#         cx = float(frame.get("cx", tf.get("cx", w/2.0)))
#         cy = float(frame.get("cy", tf.get("cy", h/2.0)))
#         c2w    = np.array(frame["transform_matrix"], dtype=np.float32)
#         R_c2w  = c2w[:3,:3]; t_c2w = c2w[:3,3]
#         R_w2c  = R_c2w.T
#         t_w2c  = (-R_c2w.T @ t_c2w).astype(np.float32)
#         cameras.append(Camera(
#             R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
#             fx=fx, fy=fy, cx=cx, cy=cy,
#             width=w, height=h, device=device,
#         ))
#     return cameras


# # ─────────────────────────────────────────────────────────────────────────────
# #  Public API
# # ─────────────────────────────────────────────────────────────────────────────


# def load_cameras(
#     scene_dir,
#     width: int = 800,
#     height: int = 600,
#     device=torch.device("cpu"),
#     max_cameras: int = 200,
#     target_long_edge: int = 800,
# ) -> List[Camera]:
#     """Load cameras with UNIFORM aspect-ratio-preserving scaling.

#     width/height kept for API compat but IGNORED for COLMAP paths.
#     Render resolution is derived per-camera from original dimensions
#     using uniform scaling so sx == sy and intrinsics stay valid.
#     """
#     scene_dir = Path(scene_dir)

#     colmap_present = any(
#         (scene_dir / sd / "cameras.bin").exists()
#         for sd in ["sparse/0", "sparse", "colmap/sparse/0"]
#     )
#     if not colmap_present:
#         for p in [scene_dir / "cameras.json",
#                   scene_dir / "camera_calibration" / "cameras.json"]:
#             if p.exists():
#                 cams = _load_cameras_json(p, device, max_cameras)
#                 if cams:
#                     print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
#                     return cams

#     for sd_rel in ["sparse/0", "sparse", "colmap/sparse/0"]:
#         sparse_dir_path = scene_dir / sd_rel
#         cams = _load_colmap_cameras(
#             sparse_dir_path, device, max_cameras,
#             target_long_edge=target_long_edge,
#         )
#         if cams:
#             n_with_img = sum(1 for c in cams if c.image_path is not None)
#             s = cams[0]
#             print(f"[camera_utils] Loaded {len(cams)} cameras from {sparse_dir_path} "
#                   f"(render={s.width}x{s.height}, uniform scale, "
#                   f"{n_with_img}/{len(cams)} have original image paths)")
#             return cams

#     for name in ["transforms_train.json", "transforms.json", "transforms_test.json"]:
#         p = scene_dir / name
#         if p.exists():
#             cams = _load_transforms_json(p, device, max_cameras)
#             if cams:
#                 print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
#                 return cams

#     print(f"[camera_utils] WARNING: No camera files found in {scene_dir}")
#     print(f"  Contents: {list(scene_dir.iterdir()) if scene_dir.exists() else 'DIR NOT FOUND'}")
#     return []

# def make_orbit_cameras(
#     center: torch.Tensor,
#     radius: float,
#     num_frames: int,
#     elevation_deg: float,
#     fx: float, fy: float,
#     width: int, height: int,
#     device: torch.device,
# ) -> List[Camera]:
#     """Orbit cameras around `center`. Uses standard OpenCV convention."""
#     cameras = []
#     elevation = math.radians(elevation_deg)
#     # World up = -Y in OpenCV/3DGS convention (Y points down in camera space)
#     world_up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)

#     for i in range(num_frames):
#         azimuth = 2 * math.pi * i / num_frames
#         cam_pos = center.float() + torch.tensor([
#             radius * math.cos(elevation) * math.cos(azimuth),
#             -radius * math.sin(elevation),   # negative: up in world = -Y
#             radius * math.cos(elevation) * math.sin(azimuth),
#         ], device=device, dtype=torch.float32)

#         forward = F.normalize(center.float() - cam_pos, dim=0)  # points toward scene
#         right   = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
#         up      = torch.cross(right, forward, dim=0)  # recompute for orthogonality

#         # OpenCV R_w2c: rows are right, down, forward
#         # "down" in camera space = -up in world space
#         R_w2c = torch.stack([right, -up, forward], dim=0)  # (3,3)
#         t_w2c = R_w2c @ (-cam_pos)

#         cameras.append(Camera(
#             R=R_w2c, t=t_w2c,
#             fx=fx, fy=fy,
#             cx=width/2.0, cy=height/2.0,
#             width=width, height=height, device=device,
#         ))
#     return cameras

# """
# camera_utils.py
# ───────────────
# Load cameras from a pre-trained 3DGS scene directory.

# 3DGS scenes (Kerbl et al.) store camera info in:
#   - cameras.json  (written by the official viewer / training code)
#   - OR sparse/0/cameras.bin + images.bin  (raw COLMAP)

# We support both formats and return a list of Camera objects.
# """

# from __future__ import annotations

# import json
# import math
# import struct
# from pathlib import Path
# from typing import List, Optional

# import numpy as np
# import torch

# from gaussian_scene import Camera


# # ──────────────────────────────────────────────────────────────────────────────
# #  cameras.json loader  (preferred – output by the 3DGS training code)
# # ──────────────────────────────────────────────────────────────────────────────

# def _load_cameras_json(
#     json_path: Path,
#     width: int,
#     height: int,
#     device: torch.device,
#     max_cameras: int = 200,
# ) -> List[Camera]:
#     """Load cameras from the cameras.json written by the 3DGS trainer."""
#     with open(json_path) as f:
#         data = json.load(f)

#     cameras = []
#     for entry in data[:max_cameras]:
#         # The JSON stores the c2w (camera-to-world) 4x4 matrix
#         # We need w2c (world-to-camera)
#         img_w = entry.get("width", width)
#         img_h = entry.get("height", height)
#         fx = entry.get("fx", entry.get("fl_x", 800.0))
#         fy = entry.get("fy", entry.get("fl_y", 800.0))
#         cx = entry.get("cx", img_w / 2.0)
#         cy = entry.get("cy", img_h / 2.0)

#         # c2w matrix
#         c2w = np.array(entry["transform_matrix"], dtype=np.float32)
#         if c2w.shape == (4, 4):
#             R_c2w = c2w[:3, :3]
#             t_c2w = c2w[:3, 3]
#         else:
#             raise ValueError(f"Unexpected transform_matrix shape: {c2w.shape}")

#         # w2c = inverse of c2w
#         R_w2c = R_c2w.T
#         t_w2c = -R_c2w.T @ t_c2w

#         cam = Camera(
#             R=torch.from_numpy(R_w2c),
#             t=torch.from_numpy(t_w2c),
#             fx=float(fx), fy=float(fy),
#             cx=float(cx), cy=float(cy),
#             width=int(img_w), height=int(img_h),
#             device=device,
#         )
#         cameras.append(cam)

#     return cameras


# # ──────────────────────────────────────────────────────────────────────────────
# #  COLMAP binary loader  (fallback)
# # ──────────────────────────────────────────────────────────────────────────────

# def _read_colmap_cameras_bin(cameras_bin: Path) -> dict:
#     cameras = {}
#     with open(cameras_bin, "rb") as f:
#         num_cameras = struct.unpack("<Q", f.read(8))[0]
#         for _ in range(num_cameras):
#             cam_id   = struct.unpack("<I", f.read(4))[0]
#             model_id = struct.unpack("<I", f.read(4))[0]
#             w        = struct.unpack("<Q", f.read(8))[0]
#             h        = struct.unpack("<Q", f.read(8))[0]
#             num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 5, 5: 8, 6: 8, 7: 12}.get(model_id, 4)
#             params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
#             cameras[cam_id] = {"w": w, "h": h, "model_id": model_id, "params": params}
#     return cameras


# def _read_colmap_images_bin(images_bin: Path) -> list:
#     images = []
#     with open(images_bin, "rb") as f:
#         num_images = struct.unpack("<Q", f.read(8))[0]
#         for _ in range(num_images):
#             img_id    = struct.unpack("<I", f.read(4))[0]
#             qvec      = struct.unpack("<4d", f.read(32))   # qw, qx, qy, qz
#             tvec      = struct.unpack("<3d", f.read(24))
#             cam_id    = struct.unpack("<I", f.read(4))[0]
#             # read name (null-terminated)
#             name = b""
#             while True:
#                 c = f.read(1)
#                 if c == b"\x00":
#                     break
#                 name += c
#             # skip 2D points
#             num_pts = struct.unpack("<Q", f.read(8))[0]
#             f.read(num_pts * 24)
#             images.append({"id": img_id, "qvec": qvec, "tvec": tvec, "cam_id": cam_id, "name": name.decode()})
#     return images


# def _qvec_to_rotmat(qvec) -> np.ndarray:
#     """COLMAP quaternion (qw, qx, qy, qz) → 3×3 rotation matrix."""
#     qw, qx, qy, qz = qvec
#     R = np.array([
#         [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qw*qz),   2*(qx*qz + qw*qy)],
#         [2*(qx*qy + qw*qz),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
#         [2*(qx*qz - qw*qy),       2*(qy*qz + qw*qx),   1 - 2*(qx**2 + qy**2)],
#     ], dtype=np.float32)
#     return R


# def _load_colmap_cameras(
#     scene_dir: Path,
#     width: int,
#     height: int,
#     device: torch.device,
#     max_cameras: int = 200,
# ) -> List[Camera]:
#     cameras_bin = scene_dir / "sparse" / "0" / "cameras.bin"
#     images_bin  = scene_dir / "sparse" / "0" / "images.bin"

#     if not cameras_bin.exists() or not images_bin.exists():
#         return []

#     colmap_cams   = _read_colmap_cameras_bin(cameras_bin)
#     colmap_images = _read_colmap_images_bin(images_bin)

#     cameras = []
#     for img_entry in colmap_images[:max_cameras]:
#         cc = colmap_cams[img_entry["cam_id"]]
#         params = cc["params"]

#         # SIMPLE_PINHOLE: f, cx, cy
#         # PINHOLE: fx, fy, cx, cy
#         if cc["model_id"] == 0:
#             fx = fy = params[0]
#             cx, cy = params[1], params[2]
#         else:
#             fx, fy = params[0], params[1]
#             cx, cy = params[2], params[3]

#         R = _qvec_to_rotmat(img_entry["qvec"])
#         t = np.array(img_entry["tvec"], dtype=np.float32)

#         cam = Camera(
#             R=torch.from_numpy(R),
#             t=torch.from_numpy(t),
#             fx=float(fx), fy=float(fy),
#             cx=float(cx), cy=float(cy),
#             width=int(cc["w"]), height=int(cc["h"]),
#             device=device,
#         )
#         cameras.append(cam)

#     return cameras


# # ──────────────────────────────────────────────────────────────────────────────
# #  Public API
# # ──────────────────────────────────────────────────────────────────────────────

# def load_cameras(
#     scene_dir: Path,
#     width: int = 800,
#     height: int = 600,
#     device: torch.device = torch.device("cpu"),
#     max_cameras: int = 200,
# ) -> List[Camera]:
#     """Auto-detect camera format and load cameras from a 3DGS scene directory.

#     Search order:
#     1. cameras.json (written by 3DGS viewer)
#     2. transforms_train.json (NeRF-style, also supported by 3DGS)
#     3. sparse/0/cameras.bin + images.bin (raw COLMAP)

#     Args:
#         scene_dir: Root of the 3DGS scene (contains the .ply + camera files).
#         width, height: Override image resolution (used when not stored in file).
#         device: PyTorch device.
#         max_cameras: Cap on cameras to load (to limit memory usage).

#     Returns:
#         List of Camera objects.
#     """
#     scene_dir = Path(scene_dir)

#     # 1. cameras.json
#     cameras_json = scene_dir / "cameras.json"
#     if cameras_json.exists():
#         cams = _load_cameras_json(cameras_json, width, height, device, max_cameras)
#         if cams:
#             return cams

#     # 2. transforms_train.json  (NeRF-Synthetic / instant-ngp format)
#     for tf_name in ["transforms_train.json", "transforms.json"]:
#         tf_path = scene_dir / tf_name
#         if tf_path.exists():
#             # NeRF format uses fov → derive fx/fy
#             with open(tf_path) as f:
#                 tf = json.load(f)
#             fov_x = tf.get("camera_angle_x", math.pi / 3)
#             fx = 0.5 * width / math.tan(0.5 * fov_x)
#             fy = fx
#             cx, cy = width / 2.0, height / 2.0
#             cams = []
#             for frame in tf.get("frames", [])[:max_cameras]:
#                 c2w = np.array(frame["transform_matrix"], dtype=np.float32)
#                 R_c2w = c2w[:3, :3]
#                 t_c2w = c2w[:3, 3]
#                 R_w2c = R_c2w.T
#                 t_w2c = -R_c2w.T @ t_c2w
#                 cams.append(Camera(
#                     R=torch.from_numpy(R_w2c),
#                     t=torch.from_numpy(t_w2c),
#                     fx=fx, fy=fy, cx=cx, cy=cy,
#                     width=width, height=height, device=device,
#                 ))
#             if cams:
#                 return cams

#     # 3. COLMAP binary
#     cams = _load_colmap_cameras(scene_dir, width, height, device, max_cameras)
#     if cams:
#         return cams

#     # No camera files found – return empty list; caller should fall back to orbit cameras
#     return []


# def make_orbit_cameras(
#     center: torch.Tensor,       # (3,) scene centre
#     radius: float,
#     num_frames: int,
#     elevation_deg: float,
#     fx: float, fy: float,
#     width: int, height: int,
#     device: torch.device,
# ) -> List[Camera]:
#     """Generate an orbit path around `center` at a fixed elevation."""
#     cameras = []
#     elevation = math.radians(elevation_deg)
#     for i in range(num_frames):
#         azimuth = 2 * math.pi * i / num_frames
#         # Camera position in world coords
#         cam_pos = torch.tensor([
#             radius * math.cos(elevation) * math.cos(azimuth),
#             radius * math.sin(elevation),
#             radius * math.cos(elevation) * math.sin(azimuth),
#         ], device=device, dtype=torch.float32) + center

#         # Look-at → R, t
#         forward = F.normalize(center.float() - cam_pos, dim=0)
#         # world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
#         world_up = torch.tensor([0.0, -1.0, 0.0], device=device)

#         right    = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
#         up       = torch.cross(right, forward, dim=0)

#         # R_w2c rows = right, up, -forward  (OpenCV convention)
#         # R_w2c = torch.stack([right, up, -forward], dim=0)  # (3, 3)
#         R_w2c = torch.stack([right, -up, forward], dim=0)

#         t_w2c = R_w2c @ (-cam_pos)

#         cameras.append(Camera(
#             R=R_w2c, t=t_w2c,
#             fx=fx, fy=fy,
#             cx=width / 2.0, cy=height / 2.0,
#             width=width, height=height, device=device,
#         ))
#     return cameras


# # Avoid circular import
# import torch.nn.functional as F

"""
camera_utils.py  (fixed)
────────────────────────
Fixes vs original:
  1. Searches for COLMAP data in ALL standard Mip-NeRF 360 / 3DGS locations,
     including sparse/0/, sparse/, and the 3DGS-written cameras.json.
  2. make_orbit_cameras: fixed broken up-vector (was double-negating).
  3. Added diagnostic print when no cameras found so failure is obvious.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from gaussian_scene import Camera


# ─────────────────────────────────────────────────────────────────────────────
#  cameras.json  (written by 3DGS trainer / viewer)
# ─────────────────────────────────────────────────────────────────────────────

def _load_cameras_json(json_path, device, max_cameras=200):
    with open(json_path) as f:
        data = json.load(f)
    cameras = []
    for entry in data[:max_cameras]:
        img_w = int(entry.get("width",  entry.get("w", 800)))
        img_h = int(entry.get("height", entry.get("h", 600)))
        fx = float(entry.get("fx", entry.get("fl_x", img_w * 0.8)))
        fy = float(entry.get("fy", entry.get("fl_y", img_h * 0.8)))
        cx = float(entry.get("cx", img_w / 2.0))
        cy = float(entry.get("cy", img_h / 2.0))

        c2w = np.array(entry["transform_matrix"], dtype=np.float32)
        R_c2w = c2w[:3, :3]
        t_c2w = c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = (-R_c2w.T @ t_c2w).astype(np.float32)

        cameras.append(Camera(
            R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=img_w, height=img_h, device=device,
        ))
    return cameras


# ─────────────────────────────────────────────────────────────────────────────
#  COLMAP binary
# ─────────────────────────────────────────────────────────────────────────────

def _read_colmap_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id   = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            w = struct.unpack("<Q", f.read(8))[0]
            h = struct.unpack("<Q", f.read(8))[0]
            np_ = {0:3, 1:4, 2:4, 3:5, 4:5, 5:8, 6:8, 7:12}.get(model_id, 4)
            params = struct.unpack(f"<{np_}d", f.read(8 * np_))
            cams[cam_id] = {"w":w, "h":h, "model_id":model_id, "params":params}
    return cams

def _read_colmap_images_bin(path):
    images = []
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec   = struct.unpack("<4d", f.read(32))
            tvec   = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            num_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts * 24)
            images.append({"qvec":qvec, "tvec":tvec, "cam_id":cam_id, "name":name.decode()})
    return images

def _qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz),   1-2*(qx**2+qz**2),   2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx),   1-2*(qx**2+qy**2)],
    ], dtype=np.float32)

def _load_colmap_cameras(sparse_dir, device, max_cameras=200,
                         target_long_edge=800):
    """Load COLMAP cameras with UNIFORM rescaling to preserve aspect ratio.

    CRITICAL: render_width/render_height are NOT accepted as separate params.
    Instead we use target_long_edge and derive render_w, render_h per-camera
    by uniformly scaling so the longest edge = target_long_edge.

    Why uniform scaling matters
    ---------------------------
    The previous code accepted render_width=800, render_height=600 and
    applied NON-UNIFORM scaling:
        sx = render_width  / orig_w
        sy = render_height / orig_h
    For a PORTRAIT camera (e.g. bonsai: 1520x2028):
        sx = 800/1520 = 0.526
        sy = 600/2028 = 0.296   ← 2x different!
    This stretches the camera frustum, making cam.project() return pixel
    coordinates that don't align with what the renderer produces.
    The segmentation mask and the Gaussian projections are in different
    coordinate spaces → lifting scores are computed on wrong pixels.

    With uniform scaling (sx == sy):
        scale = 800 / max(1520, 2028) = 800/2028 = 0.395
        render_w = round_even(1520 * 0.395) = 598
        render_h = round_even(2028 * 0.395) = 800
        sx = sy = 0.395 → fx, fy both scaled correctly
    """
    cb = sparse_dir / "cameras.bin"
    ib = sparse_dir / "images.bin"
    if not cb.exists() or not ib.exists():
        return []
    colmap_cams   = _read_colmap_cameras_bin(cb)
    colmap_images = _read_colmap_images_bin(ib)
    cameras = []
    for img in colmap_images[:max_cameras]:
        cc = colmap_cams[img["cam_id"]]
        p  = cc["params"]
        if cc["model_id"] == 0:   # SIMPLE_PINHOLE
            fx = fy = p[0]; cx, cy = p[1], p[2]
        elif cc["model_id"] == 2: # SIMPLE_RADIAL
            fx = fy = p[0]; cx, cy = p[1], p[2]
        else:                     # PINHOLE, OPENCV, FULL_OPENCV
            fx, fy = p[0], p[1]; cx, cy = p[2], p[3]

        orig_w, orig_h = int(cc["w"]), int(cc["h"])

        # Uniform scale: longest edge → target_long_edge
        scale  = target_long_edge / max(orig_w, orig_h)
        out_w  = int(orig_w * scale)
        out_h  = int(orig_h * scale)
        # Round to nearest even (required by some video/image encoders)
        out_w  = max(2, (out_w  // 2) * 2)
        out_h  = max(2, (out_h  // 2) * 2)
        # Recompute exact scale from rounded output dims
        sx     = out_w / orig_w
        sy     = out_h / orig_h

        fx = fx * sx;  fy = fy * sy
        cx = cx * sx;  cy = cy * sy

        R = _qvec_to_rotmat(img["qvec"])
        t = np.array(img["tvec"], dtype=np.float32)

        # Resolve original training image path.
        img_name  = img.get("name", "")
        scene_dir = sparse_dir.parent.parent if sparse_dir.name == "0" else sparse_dir.parent
        image_path = None
        for img_subdir in ["images_4", "images_2", "images"]:
            candidate = scene_dir / img_subdir / img_name
            if candidate.exists():
                image_path = candidate
                break

        cameras.append(Camera(
            R=torch.from_numpy(R), t=torch.from_numpy(t),
            fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
            width=out_w, height=out_h, device=device,
            image_path=image_path,
        ))
    return cameras


# ─────────────────────────────────────────────────────────────────────────────
#  NeRF-style transforms.json
# ─────────────────────────────────────────────────────────────────────────────

def _load_transforms_json(tf_path, device, max_cameras=200):
    with open(tf_path) as f:
        tf = json.load(f)
    # Prefer per-frame intrinsics; fall back to top-level fov
    frames = tf.get("frames", [])[:max_cameras]
    cameras = []
    for frame in frames:
        w   = int(frame.get("w", tf.get("w", 800)))
        h   = int(frame.get("h", tf.get("h", 600)))
        if "fl_x" in frame or "fl_x" in tf:
            fx = float(frame.get("fl_x", tf.get("fl_x")))
            fy = float(frame.get("fl_y", tf.get("fl_y", fx)))
        else:
            fov_x = float(tf.get("camera_angle_x", math.pi/3))
            fx = fy = 0.5 * w / math.tan(0.5 * fov_x)
        cx = float(frame.get("cx", tf.get("cx", w/2.0)))
        cy = float(frame.get("cy", tf.get("cy", h/2.0)))
        c2w    = np.array(frame["transform_matrix"], dtype=np.float32)
        R_c2w  = c2w[:3,:3]; t_c2w = c2w[:3,3]
        R_w2c  = R_c2w.T
        t_w2c  = (-R_c2w.T @ t_c2w).astype(np.float32)
        cameras.append(Camera(
            R=torch.from_numpy(R_w2c), t=torch.from_numpy(t_w2c),
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=w, height=h, device=device,
        ))
    return cameras


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_cameras(
    scene_dir,
    width: int = 800,
    height: int = 600,
    device=torch.device("cpu"),
    max_cameras: int = 200,
    target_long_edge: int = 800,
) -> List[Camera]:
    """Load cameras with UNIFORM aspect-ratio-preserving scaling.

    width/height kept for API compat but IGNORED for COLMAP paths.
    Render resolution is derived per-camera from original dimensions
    using uniform scaling so sx == sy and intrinsics stay valid.
    """
    scene_dir = Path(scene_dir)

    colmap_present = any(
        (scene_dir / sd / "cameras.bin").exists()
        for sd in ["sparse/0", "sparse", "colmap/sparse/0"]
    )
    if not colmap_present:
        for p in [scene_dir / "cameras.json",
                  scene_dir / "camera_calibration" / "cameras.json"]:
            if p.exists():
                cams = _load_cameras_json(p, device, max_cameras)
                if cams:
                    print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
                    return cams

    for sd_rel in ["sparse/0", "sparse", "colmap/sparse/0"]:
        sparse_dir_path = scene_dir / sd_rel
        cams = _load_colmap_cameras(
            sparse_dir_path, device, max_cameras,
            target_long_edge=target_long_edge,
        )
        if cams:
            n_with_img = sum(1 for c in cams if c.image_path is not None)
            s = cams[0]
            print(f"[camera_utils] Loaded {len(cams)} cameras from {sparse_dir_path} "
                  f"(render={s.width}x{s.height}, uniform scale, "
                  f"{n_with_img}/{len(cams)} have original image paths)")
            return cams

    for name in ["transforms_train.json", "transforms.json", "transforms_test.json"]:
        p = scene_dir / name
        if p.exists():
            cams = _load_transforms_json(p, device, max_cameras)
            if cams:
                print(f"[camera_utils] Loaded {len(cams)} cameras from {p}")
                return cams

    print(f"[camera_utils] WARNING: No camera files found in {scene_dir}")
    print(f"  Contents: {list(scene_dir.iterdir()) if scene_dir.exists() else 'DIR NOT FOUND'}")
    return []

def make_orbit_cameras(
    center: torch.Tensor,
    radius: float,
    num_frames: int,
    elevation_deg: float,
    fx: float, fy: float,
    width: int, height: int,
    device: torch.device,
) -> List[Camera]:
    """Orbit cameras around `center`. Uses standard OpenCV convention."""
    cameras = []
    elevation = math.radians(elevation_deg)
    # World up = -Y in OpenCV/3DGS convention (Y points down in camera space)
    world_up = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float32)

    for i in range(num_frames):
        azimuth = 2 * math.pi * i / num_frames
        cam_pos = center.float() + torch.tensor([
            radius * math.cos(elevation) * math.cos(azimuth),
            -radius * math.sin(elevation),   # negative: up in world = -Y
            radius * math.cos(elevation) * math.sin(azimuth),
        ], device=device, dtype=torch.float32)

        forward = F.normalize(center.float() - cam_pos, dim=0)  # points toward scene
        right   = F.normalize(torch.cross(forward, world_up, dim=0), dim=0)
        up      = torch.cross(right, forward, dim=0)  # recompute for orthogonality

        # OpenCV R_w2c: rows are right, down, forward
        # "down" in camera space = -up in world space
        R_w2c = torch.stack([right, -up, forward], dim=0)  # (3,3)
        t_w2c = R_w2c @ (-cam_pos)

        cameras.append(Camera(
            R=R_w2c, t=t_w2c,
            fx=fx, fy=fy,
            cx=width/2.0, cy=height/2.0,
            width=width, height=height, device=device,
        ))
    return cameras