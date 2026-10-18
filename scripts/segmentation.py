# # """
# # segmentation.py
# # ───────────────
# # 2D segmentation backends:
# #   - GroundedSAM2:  GroundingDINO (bounding box) → SAM2 (mask)
# #   - SAM3Segmenter: same API, uses the SAM3 checkpoint

# # Both expose a common interface:
# #     segmenter.segment(image_rgb_np, text_prompt) → (mask_bool_HW, score_float)
# # """

# # from __future__ import annotations

# # import os
# # import sys
# # from pathlib import Path
# # from typing import Optional, Tuple

# # import numpy as np
# # import torch
# # from PIL import Image
# # from rich.console import Console

# # console = Console()


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  Shared utilities
# # # ──────────────────────────────────────────────────────────────────────────────

# # def _tensor_to_uint8(img_rgb: np.ndarray) -> np.ndarray:
# #     """Convert float [0,1] or uint8 [0,255] RGB ndarray to uint8."""
# #     if img_rgb.dtype != np.uint8:
# #         img_rgb = (img_rgb.clip(0, 1) * 255).astype(np.uint8)
# #     return img_rgb


# # def _load_image_for_gdino(img_rgb: np.ndarray, device: torch.device):
# #     """Prepare image for GroundingDINO inference."""
# #     from torchvision import transforms
# #     transform = transforms.Compose([
# #         transforms.ToTensor(),
# #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# #     ])
# #     img_pil = Image.fromarray(_tensor_to_uint8(img_rgb))
# #     return transform(img_pil).unsqueeze(0).to(device)


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  GroundingDINO loader (handles both official and pip-only installs)
# # # ──────────────────────────────────────────────────────────────────────────────

# # def _load_grounding_dino(config_path: str, checkpoint_path: str, device: torch.device):
# #     """Load GroundingDINO model, trying multiple import paths."""
# #     # Try official source install first
# #     try:
# #         from groundingdino.util.inference import load_model
# #         # model = load_model(config_path, checkpoint_path, device=str(device))
# #         model = load_model(config_path, checkpoint_path, device="cpu")
# #         return model, "official"
# #     except ImportError:
# #         pass

# #     # Try pip-installed groundingdino-py
# #     try:
# #         import groundingdino
# #         from groundingdino.util.inference import load_model
# #         # model = load_model(config_path, checkpoint_path, device=str(device))
# #         model = load_model(config_path, checkpoint_path, device="cpu")
# #         return model, "pip"
# #     except Exception as e:
# #         raise ImportError(
# #             f"GroundingDINO not available: {e}\n"
# #             "Run setup.sh to install it, or check the third_party/ directory."
# #         )
    
# # # def _run_grounding_dino(model, image_tensor, caption, box_threshold, text_threshold, device):
# # #     try:
# # #         from groundingdino.util.inference import predict
# # #         # GroundingDINO must run on CPU — MPS is unsupported
# # #         image_cpu = image_tensor.squeeze(0).cpu()
# # #         boxes, scores, labels = predict(
# # #             model=model,
# # #             image=image_cpu,
# # #             caption=caption,
# # #             box_threshold=box_threshold,
# # #             text_threshold=text_threshold,
# # #             device="cpu",   # <-- force CPU
# # #         )
# # #         return boxes.to(device), scores.to(device)
# # #     except Exception as e:
# # #         console.log(f"[red]GroundingDINO predict failed: {e}[/red]")
# # #         return torch.zeros(0, 4), torch.zeros(0)


# # def _run_grounding_dino(
# #     model,
# #     image_tensor,
# #     caption: str,
# #     box_threshold: float,
# #     text_threshold: float,
# #     device: torch.device,
# # ):
# #     """Run GroundingDINO and return boxes (N,4) and scores (N,)."""
# #     try:
# #         from groundingdino.util.inference import predict
# #         # predict works with cpu/mps
# #         boxes, scores, labels = predict(
# #             model=model,
# #             image=image_tensor.squeeze(0),
# #             caption=caption,
# #             box_threshold=box_threshold,
# #             text_threshold=text_threshold,
# #             device=str(device),
# #         )
# #         return boxes, scores
# #     except Exception as e:
# #         console.log(f"[yellow]GroundingDINO predict failed: {e} – returning empty[/yellow]")
# #         return torch.zeros(0, 4), torch.zeros(0)


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  SAM2 mask predictor wrapper
# # # ──────────────────────────────────────────────────────────────────────────────

# # def _load_sam2(checkpoint: str, model_cfg: str, device: torch.device):
# #     """Load SAM2 (or SAM3, same API) predictor."""
# #     try:
# #         from sam2.build_sam import build_sam2
# #         from sam2.sam2_image_predictor import SAM2ImagePredictor
# #         sam2_model = build_sam2(model_cfg, checkpoint, device=str(device))
# #         predictor  = SAM2ImagePredictor(sam2_model)
# #         return predictor
# #     except Exception as e:
# #         raise ImportError(
# #             f"SAM2 not found: {e}\n"
# #             "Make sure you ran `pip install -e .` inside third_party/sam2/"
# #         )


# # def _boxes_xyxy_from_cxcywh(boxes_cxcywh: torch.Tensor, H: int, W: int) -> np.ndarray:
# #     """Convert normalised cx,cy,w,h → absolute x1,y1,x2,y2."""
# #     if len(boxes_cxcywh) == 0:
# #         return np.zeros((0, 4), dtype=np.float32)
# #     cx, cy, w, h = boxes_cxcywh.unbind(-1)
# #     x1 = (cx - w / 2) * W
# #     y1 = (cy - h / 2) * H
# #     x2 = (cx + w / 2) * W
# #     y2 = (cy + h / 2) * H
# #     return torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()


# # def _run_sam2(predictor, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> Tuple[np.ndarray, float]:
# #     """Run SAM2 with box prompts and return the best merged mask + score."""
# #     predictor.set_image(image_rgb)

# #     if len(boxes_xyxy) == 0:
# #         H, W = image_rgb.shape[:2]
# #         return np.zeros((H, W), dtype=bool), 0.0

# #     # Use all detected boxes as prompts; take the union of all masks
# #     masks, iou_preds, _ = predictor.predict(
# #         box=boxes_xyxy,
# #         multimask_output=False,
# #     )
# #     # masks: (N_boxes, 1, H, W) or (N_boxes, H, W)
# #     if masks.ndim == 4:
# #         masks = masks[:, 0]     # (N_boxes, H, W)
# #     merged_mask = masks.any(axis=0)      # (H, W) union
# #     score = float(iou_preds.mean())
# #     return merged_mask.astype(bool), score


# # # ──────────────────────────────────────────────────────────────────────────────
# # #  Public segmenter classes
# # # ──────────────────────────────────────────────────────────────────────────────

# # class GroundedSAM2:
# #     """
# #     Two-stage zero-shot segmenter:
# #       1. GroundingDINO → bounding boxes from text prompt
# #       2. SAM2 → masks from boxes
# #     """

# #     def __init__(
# #         self,
# #         gdino_config:     str,
# #         gdino_checkpoint: str,
# #         sam2_checkpoint:  str,
# #         sam2_model_cfg:   str,
# #         box_threshold:    float = 0.30,
# #         text_threshold:   float = 0.25,
# #         device:           torch.device = torch.device("cpu"),
# #     ):
# #         self.box_threshold  = box_threshold
# #         self.text_threshold = text_threshold
# #         self.device         = device

# #         console.log("Loading GroundingDINO …")
# #         self.gdino, self._gdino_variant = _load_grounding_dino(
# #             gdino_config, gdino_checkpoint, device
# #         )
# #         console.log(f"  GroundingDINO loaded ({self._gdino_variant})")

# #         console.log("Loading SAM2 …")
# #         self.sam2 = _load_sam2(sam2_checkpoint, sam2_model_cfg, device)
# #         console.log("  SAM2 loaded")

# #     def segment(
# #         self,
# #         image_rgb: np.ndarray,   # (H, W, 3) uint8 or float32
# #         text_prompt: str,
# #     ) -> Tuple[np.ndarray, float]:
# #         """
# #         Returns:
# #             mask  (H, W) bool   – True = object pixels
# #             score float         – mean IoU prediction from SAM2
# #         """
# #         image_rgb = _tensor_to_uint8(image_rgb)
# #         H, W = image_rgb.shape[:2]

# #         # Stage 1: GroundingDINO
# #         img_tensor = _load_image_for_gdino(image_rgb, self.device)
# #         boxes, scores = _run_grounding_dino(
# #             self.gdino, img_tensor, text_prompt,
# #             self.box_threshold, self.text_threshold, self.device,
# #         )

# #         if len(boxes) == 0:
# #             return np.zeros((H, W), dtype=bool), 0.0

# #         # Convert normalised cx,cy,w,h → absolute xyxy
# #         boxes_xyxy = _boxes_xyxy_from_cxcywh(boxes, H, W)

# #         # Stage 2: SAM2
# #         mask, score = _run_sam2(self.sam2, image_rgb, boxes_xyxy)
# #         return mask, score


# # class SAM3Segmenter:
# #     """
# #     SAM3 segmenter – uses the SAM3 checkpoint with the same SAM2 API.
# #     SAM3 integrates semantic text-prompted segmentation into training,
# #     so we pass the text caption directly to the predictor when available,
# #     and fall back to a full-image bounding box otherwise.
# #     """

# #     def __init__(
# #         self,
# #         checkpoint:  str,
# #         model_cfg:   str,
# #         device:      torch.device = torch.device("cpu"),
# #     ):
# #         self.device = device
# #         console.log("Loading SAM3 …")
# #         self.predictor = _load_sam2(checkpoint, model_cfg, device)
# #         console.log("  SAM3 loaded")

# #     def segment(
# #         self,
# #         image_rgb: np.ndarray,
# #         text_prompt: str,
# #     ) -> Tuple[np.ndarray, float]:
# #         """
# #         SAM3 can accept text prompts natively; the current release mirrors
# #         the SAM2 image predictor API. We pass a full-image box as prompt and
# #         use the automatic mask generation path which ranks masks by semantic
# #         similarity when a text token is available.
# #         """
# #         image_rgb = _tensor_to_uint8(image_rgb)
# #         H, W = image_rgb.shape[:2]

# #         # Attempt text-prompted segmentation (SAM3 extended API)
# #         try:
# #             from sam2.sam2_image_predictor import SAM2ImagePredictor
# #             self.predictor.set_image(image_rgb)
# #             # SAM3 exposes predict_with_text in some builds
# #             if hasattr(self.predictor, "predict_with_text"):
# #                 masks, scores, _ = self.predictor.predict_with_text(
# #                     text=text_prompt,
# #                     multimask_output=True,
# #                 )
# #             else:
# #                 # Fallback: use automatic mask generator and pick largest mask
# #                 from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# #                 gen = SAM2AutomaticMaskGenerator(self.predictor.model)
# #                 ann  = gen.generate(image_rgb)
# #                 if not ann:
# #                     return np.zeros((H, W), dtype=bool), 0.0
# #                 # Pick largest mask (heuristic for dominant object)
# #                 ann_sorted = sorted(ann, key=lambda x: x["area"], reverse=True)
# #                 mask = ann_sorted[0]["segmentation"].astype(bool)
# #                 return mask, float(ann_sorted[0].get("predicted_iou", 0.5))

# #             best = np.argmax(scores)
# #             mask = masks[best].astype(bool)
# #             return mask, float(scores[best])

# #         except Exception as e:
# #             console.log(f"[yellow]SAM3 segment failed: {e} – returning empty mask[/yellow]")
# #             return np.zeros((H, W), dtype=bool), 0.0


# # def build_segmenter(cfg: dict, device: torch.device):
# #     """Factory: build the configured segmenter from the YAML config dict."""
# #     backbone = cfg["segmentation"]["backbone"]

# #     if backbone == "grounded_sam2":
# #         gdino_cfg  = cfg["segmentation"]["grounding_dino"]
# #         sam2_cfg   = cfg["segmentation"]["sam2"]
# #         return GroundedSAM2(
# #             gdino_config     = gdino_cfg["config_path"],
# #             gdino_checkpoint = gdino_cfg["checkpoint_path"],
# #             sam2_checkpoint  = sam2_cfg["checkpoint"],
# #             sam2_model_cfg   = sam2_cfg["model_cfg"],
# #             box_threshold    = gdino_cfg["box_threshold"],
# #             text_threshold   = gdino_cfg["text_threshold"],
# #             device           = device,
# #         )

# #     elif backbone == "sam3":
# #         sam3_cfg = cfg["segmentation"]["sam3"]
# #         return SAM3Segmenter(
# #             checkpoint = sam3_cfg["checkpoint"],
# #             model_cfg  = sam3_cfg["model_cfg"],
# #             device     = device,
# #         )

# #     else:
# #         raise ValueError(f"Unknown segmentation backbone: {backbone!r}")
# """
# segmentation.py
# ───────────────
# 2D segmentation backends:
#   - GroundedSAM2:  GroundingDINO (CPU) → SAM2 (MPS/CPU)
#   - SAM3Segmenter: SAM3 checkpoint (same API style)

# Optimized for Apple Silicon:
#   • GroundingDINO forced to CPU (avoids MPS OOM)
#   • SAM2 can stay on MPS
#   • Optional image downscaling before GroundingDINO
#   • Automatic cache cleanup
# """

# from __future__ import annotations

# import gc
# from pathlib import Path
# from typing import Tuple

# import numpy as np
# import torch
# from PIL import Image
# from rich.console import Console

# console = Console()


# # ──────────────────────────────────────────────────────────────────────────────
# # Utilities
# # ──────────────────────────────────────────────────────────────────────────────

# def _cleanup_memory():
#     gc.collect()
#     if torch.backends.mps.is_available():
#         try:
#             torch.mps.empty_cache()
#         except Exception:
#             pass


# def _tensor_to_uint8(img_rgb: np.ndarray) -> np.ndarray:
#     """Convert float [0,1] or uint8 [0,255] RGB ndarray to uint8."""
#     if img_rgb.dtype != np.uint8:
#         img_rgb = (img_rgb.clip(0, 1) * 255).astype(np.uint8)
#     return img_rgb


# def _resize_keep_aspect(img: np.ndarray, max_side: int = 640) -> tuple[np.ndarray, float]:
#     """
#     Resize image so longest side=max_side.
#     Returns resized_img, scale_factor
#     scale_factor = resized / original
#     """
#     h, w = img.shape[:2]
#     longest = max(h, w)

#     if longest <= max_side:
#         return img, 1.0

#     scale = max_side / longest
#     new_w = int(w * scale)
#     new_h = int(h * scale)

#     pil = Image.fromarray(img)
#     pil = pil.resize((new_w, new_h), Image.BILINEAR)
#     return np.array(pil), scale


# def _load_image_for_gdino(img_rgb: np.ndarray):
#     """Prepare image for GroundingDINO."""
#     from torchvision import transforms

#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.485, 0.456, 0.406],
#             [0.229, 0.224, 0.225]
#         ),
#     ])

#     img_pil = Image.fromarray(_tensor_to_uint8(img_rgb))
#     return transform(img_pil).unsqueeze(0)   # CPU tensor


# # ──────────────────────────────────────────────────────────────────────────────
# # GroundingDINO
# # ──────────────────────────────────────────────────────────────────────────────

# def _load_grounding_dino(config_path: str, checkpoint_path: str):
#     """Always load GroundingDINO on CPU."""
#     try:
#         from groundingdino.util.inference import load_model
#         model = load_model(config_path, checkpoint_path, device="cpu")
#         return model, "official"
#     except Exception:
#         pass

#     try:
#         import groundingdino
#         from groundingdino.util.inference import load_model
#         model = load_model(config_path, checkpoint_path, device="cpu")
#         return model, "pip"
#     except Exception as e:
#         raise ImportError(
#             f"GroundingDINO not available: {e}\n"
#             "Install it correctly first."
#         )


# def _run_grounding_dino(
#     model,
#     image_tensor,
#     caption: str,
#     box_threshold: float,
#     text_threshold: float,
# ):
#     """Run GroundingDINO on CPU only."""
#     try:
#         from groundingdino.util.inference import predict

#         boxes, scores, labels = predict(
#             model=model,
#             image=image_tensor.squeeze(0).cpu(),
#             caption=caption,
#             box_threshold=box_threshold,
#             text_threshold=text_threshold,
#             device="cpu",
#         )

#         return boxes, scores

#     except Exception as e:
#         console.log(f"[yellow]GroundingDINO failed: {e}[/yellow]")
#         return torch.zeros(0, 4), torch.zeros(0)


# # ──────────────────────────────────────────────────────────────────────────────
# # SAM2 / SAM3
# # ──────────────────────────────────────────────────────────────────────────────

# def _load_sam2(checkpoint: str, model_cfg: str, device: torch.device):
#     try:
#         from sam2.build_sam import build_sam2
#         from sam2.sam2_image_predictor import SAM2ImagePredictor

#         model = build_sam2(model_cfg, checkpoint, device=str(device))
#         predictor = SAM2ImagePredictor(model)
#         return predictor

#     except Exception as e:
#         raise ImportError(f"SAM2 load failed: {e}")


# def _boxes_xyxy_from_cxcywh(boxes_cxcywh: torch.Tensor, H: int, W: int):
#     if len(boxes_cxcywh) == 0:
#         return np.zeros((0, 4), dtype=np.float32)

#     cx, cy, bw, bh = boxes_cxcywh.unbind(-1)

#     x1 = (cx - bw / 2) * W
#     y1 = (cy - bh / 2) * H
#     x2 = (cx + bw / 2) * W
#     y2 = (cy + bh / 2) * H

#     return torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()


# def _run_sam2(predictor, image_rgb: np.ndarray, boxes_xyxy: np.ndarray):
#     predictor.set_image(image_rgb)

#     if len(boxes_xyxy) == 0:
#         H, W = image_rgb.shape[:2]
#         return np.zeros((H, W), dtype=bool), 0.0

#     masks, iou_preds, _ = predictor.predict(
#         box=boxes_xyxy,
#         multimask_output=False,
#     )

#     if masks.ndim == 4:
#         masks = masks[:, 0]

#     merged = masks.any(axis=0)
#     score = float(iou_preds.mean())

#     return merged.astype(bool), score


# # ──────────────────────────────────────────────────────────────────────────────
# # Main Classes
# # ──────────────────────────────────────────────────────────────────────────────

# class GroundedSAM2:
#     def __init__(
#         self,
#         gdino_config: str,
#         gdino_checkpoint: str,
#         sam2_checkpoint: str,
#         sam2_model_cfg: str,
#         box_threshold: float = 0.30,
#         text_threshold: float = 0.25,
#         device: torch.device = torch.device("cpu"),
#     ):
#         self.box_threshold = box_threshold
#         self.text_threshold = text_threshold
#         self.device = device

#         console.log("Loading GroundingDINO (CPU)…")
#         self.gdino, self.variant = _load_grounding_dino(
#             gdino_config,
#             gdino_checkpoint
#         )
#         console.log("GroundingDINO loaded")

#         console.log(f"Loading SAM2 ({device})…")
#         self.sam2 = _load_sam2(
#             sam2_checkpoint,
#             sam2_model_cfg,
#             device
#         )
#         console.log("SAM2 loaded")

#     def segment(self, image_rgb: np.ndarray, text_prompt: str):
#         image_rgb = _tensor_to_uint8(image_rgb)
#         H, W = image_rgb.shape[:2]

#         _cleanup_memory()

#         # Resize for GroundingDINO only
#         small_img, scale = _resize_keep_aspect(image_rgb, 640)

#         img_tensor = _load_image_for_gdino(small_img)

#         boxes, scores = _run_grounding_dino(
#             self.gdino,
#             img_tensor,
#             text_prompt,
#             self.box_threshold,
#             self.text_threshold,
#         )

#         if len(boxes) == 0:
#             return np.zeros((H, W), dtype=bool), 0.0

#         # boxes are on resized image -> convert using resized dims first
#         h2, w2 = small_img.shape[:2]
#         boxes_xyxy = _boxes_xyxy_from_cxcywh(boxes, h2, w2)

#         # Scale back to original image size
#         boxes_xyxy[:, [0, 2]] /= scale
#         boxes_xyxy[:, [1, 3]] /= scale

#         mask, score = _run_sam2(self.sam2, image_rgb, boxes_xyxy)

#         _cleanup_memory()

#         return mask, score


# class SAM3Segmenter:
#     def __init__(
#         self,
#         checkpoint: str,
#         model_cfg: str,
#         device: torch.device = torch.device("cpu"),
#     ):
#         self.device = device

#         console.log("Loading SAM3…")
#         self.predictor = _load_sam2(checkpoint, model_cfg, device)
#         console.log("SAM3 loaded")

#     def segment(self, image_rgb: np.ndarray, text_prompt: str):
#         H, W = image_rgb.shape[:2]

#         try:
#             self.predictor.set_image(image_rgb)

#             if hasattr(self.predictor, "predict_with_text"):
#                 masks, scores, _ = self.predictor.predict_with_text(
#                     text=text_prompt,
#                     multimask_output=True,
#                 )
#                 best = np.argmax(scores)
#                 return masks[best].astype(bool), float(scores[best])

#             full_box = np.array([[0, 0, W, H]], dtype=np.float32)
#             return _run_sam2(self.predictor, image_rgb, full_box)

#         except Exception as e:
#             console.log(f"[yellow]SAM3 failed: {e}[/yellow]")
#             return np.zeros((H, W), dtype=bool), 0.0


# # ──────────────────────────────────────────────────────────────────────────────
# # Factory
# # ──────────────────────────────────────────────────────────────────────────────

# def build_segmenter(cfg: dict, device: torch.device):
#     backbone = cfg["segmentation"]["backbone"]

#     if backbone == "grounded_sam2":
#         gd = cfg["segmentation"]["grounding_dino"]
#         sm = cfg["segmentation"]["sam2"]

#         return GroundedSAM2(
#             gdino_config=gd["config_path"],
#             gdino_checkpoint=gd["checkpoint_path"],
#             sam2_checkpoint=sm["checkpoint"],
#             sam2_model_cfg=sm["model_cfg"],
#             box_threshold=gd["box_threshold"],
#             text_threshold=gd["text_threshold"],
#             device=device,
#         )

#     elif backbone == "sam3":
#         sm = cfg["segmentation"]["sam3"]

#         return SAM3Segmenter(
#             checkpoint=sm["checkpoint"],
#             model_cfg=sm["model_cfg"],
#             device=device,
#         )

#     else:
#         raise ValueError(f"Unknown backbone: {backbone}")

# """
# segmentation.py
# ───────────────
# 2D segmentation backends:
#   - GroundedSAM2:  GroundingDINO (bounding box) → SAM2 (mask)
#   - SAM3Segmenter: same API, uses the SAM3 checkpoint

# Both expose a common interface:
#     segmenter.segment(image_rgb_np, text_prompt) → (mask_bool_HW, score_float)
# """

# from __future__ import annotations

# import os
# import sys
# from pathlib import Path
# from typing import Optional, Tuple

# import numpy as np
# import torch
# from PIL import Image
# from rich.console import Console

# console = Console()


# # ──────────────────────────────────────────────────────────────────────────────
# #  Shared utilities
# # ──────────────────────────────────────────────────────────────────────────────

# def _tensor_to_uint8(img_rgb: np.ndarray) -> np.ndarray:
#     """Convert float [0,1] or uint8 [0,255] RGB ndarray to uint8."""
#     if img_rgb.dtype != np.uint8:
#         img_rgb = (img_rgb.clip(0, 1) * 255).astype(np.uint8)
#     return img_rgb


# def _load_image_for_gdino(img_rgb: np.ndarray, device: torch.device):
#     """Prepare image for GroundingDINO inference."""
#     from torchvision import transforms
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ])
#     img_pil = Image.fromarray(_tensor_to_uint8(img_rgb))
#     return transform(img_pil).unsqueeze(0).to(device)


# # ──────────────────────────────────────────────────────────────────────────────
# #  GroundingDINO loader (handles both official and pip-only installs)
# # ──────────────────────────────────────────────────────────────────────────────

# def _load_grounding_dino(config_path: str, checkpoint_path: str, device: torch.device):
#     """Load GroundingDINO model, trying multiple import paths."""
#     # Try official source install first
#     try:
#         from groundingdino.util.inference import load_model
#         # model = load_model(config_path, checkpoint_path, device=str(device))
#         model = load_model(config_path, checkpoint_path, device="cpu")
#         return model, "official"
#     except ImportError:
#         pass

#     # Try pip-installed groundingdino-py
#     try:
#         import groundingdino
#         from groundingdino.util.inference import load_model
#         # model = load_model(config_path, checkpoint_path, device=str(device))
#         model = load_model(config_path, checkpoint_path, device="cpu")
#         return model, "pip"
#     except Exception as e:
#         raise ImportError(
#             f"GroundingDINO not available: {e}\n"
#             "Run setup.sh to install it, or check the third_party/ directory."
#         )
    
# # def _run_grounding_dino(model, image_tensor, caption, box_threshold, text_threshold, device):
# #     try:
# #         from groundingdino.util.inference import predict
# #         # GroundingDINO must run on CPU — MPS is unsupported
# #         image_cpu = image_tensor.squeeze(0).cpu()
# #         boxes, scores, labels = predict(
# #             model=model,
# #             image=image_cpu,
# #             caption=caption,
# #             box_threshold=box_threshold,
# #             text_threshold=text_threshold,
# #             device="cpu",   # <-- force CPU
# #         )
# #         return boxes.to(device), scores.to(device)
# #     except Exception as e:
# #         console.log(f"[red]GroundingDINO predict failed: {e}[/red]")
# #         return torch.zeros(0, 4), torch.zeros(0)


# def _run_grounding_dino(
#     model,
#     image_tensor,
#     caption: str,
#     box_threshold: float,
#     text_threshold: float,
#     device: torch.device,
# ):
#     """Run GroundingDINO and return boxes (N,4) and scores (N,)."""
#     try:
#         from groundingdino.util.inference import predict
#         # predict works with cpu/mps
#         boxes, scores, labels = predict(
#             model=model,
#             image=image_tensor.squeeze(0),
#             caption=caption,
#             box_threshold=box_threshold,
#             text_threshold=text_threshold,
#             device=str(device),
#         )
#         return boxes, scores
#     except Exception as e:
#         console.log(f"[yellow]GroundingDINO predict failed: {e} – returning empty[/yellow]")
#         return torch.zeros(0, 4), torch.zeros(0)


# # ──────────────────────────────────────────────────────────────────────────────
# #  SAM2 mask predictor wrapper
# # ──────────────────────────────────────────────────────────────────────────────

# def _load_sam2(checkpoint: str, model_cfg: str, device: torch.device):
#     """Load SAM2 (or SAM3, same API) predictor."""
#     try:
#         from sam2.build_sam import build_sam2
#         from sam2.sam2_image_predictor import SAM2ImagePredictor
#         sam2_model = build_sam2(model_cfg, checkpoint, device=str(device))
#         predictor  = SAM2ImagePredictor(sam2_model)
#         return predictor
#     except Exception as e:
#         raise ImportError(
#             f"SAM2 not found: {e}\n"
#             "Make sure you ran `pip install -e .` inside third_party/sam2/"
#         )


# def _boxes_xyxy_from_cxcywh(boxes_cxcywh: torch.Tensor, H: int, W: int) -> np.ndarray:
#     """Convert normalised cx,cy,w,h → absolute x1,y1,x2,y2."""
#     if len(boxes_cxcywh) == 0:
#         return np.zeros((0, 4), dtype=np.float32)
#     cx, cy, w, h = boxes_cxcywh.unbind(-1)
#     x1 = (cx - w / 2) * W
#     y1 = (cy - h / 2) * H
#     x2 = (cx + w / 2) * W
#     y2 = (cy + h / 2) * H
#     return torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()


# def _run_sam2(predictor, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> Tuple[np.ndarray, float]:
#     """Run SAM2 with box prompts and return the best merged mask + score."""
#     predictor.set_image(image_rgb)

#     if len(boxes_xyxy) == 0:
#         H, W = image_rgb.shape[:2]
#         return np.zeros((H, W), dtype=bool), 0.0

#     # Use all detected boxes as prompts; take the union of all masks
#     masks, iou_preds, _ = predictor.predict(
#         box=boxes_xyxy,
#         multimask_output=False,
#     )
#     # masks: (N_boxes, 1, H, W) or (N_boxes, H, W)
#     if masks.ndim == 4:
#         masks = masks[:, 0]     # (N_boxes, H, W)
#     merged_mask = masks.any(axis=0)      # (H, W) union
#     score = float(iou_preds.mean())
#     return merged_mask.astype(bool), score


# # ──────────────────────────────────────────────────────────────────────────────
# #  Public segmenter classes
# # ──────────────────────────────────────────────────────────────────────────────

# class GroundedSAM2:
#     """
#     Two-stage zero-shot segmenter:
#       1. GroundingDINO → bounding boxes from text prompt
#       2. SAM2 → masks from boxes
#     """

#     def __init__(
#         self,
#         gdino_config:     str,
#         gdino_checkpoint: str,
#         sam2_checkpoint:  str,
#         sam2_model_cfg:   str,
#         box_threshold:    float = 0.30,
#         text_threshold:   float = 0.25,
#         device:           torch.device = torch.device("cpu"),
#     ):
#         self.box_threshold  = box_threshold
#         self.text_threshold = text_threshold
#         self.device         = device

#         console.log("Loading GroundingDINO …")
#         self.gdino, self._gdino_variant = _load_grounding_dino(
#             gdino_config, gdino_checkpoint, device
#         )
#         console.log(f"  GroundingDINO loaded ({self._gdino_variant})")

#         console.log("Loading SAM2 …")
#         self.sam2 = _load_sam2(sam2_checkpoint, sam2_model_cfg, device)
#         console.log("  SAM2 loaded")

#     def segment(
#         self,
#         image_rgb: np.ndarray,   # (H, W, 3) uint8 or float32
#         text_prompt: str,
#     ) -> Tuple[np.ndarray, float]:
#         """
#         Returns:
#             mask  (H, W) bool   – True = object pixels
#             score float         – mean IoU prediction from SAM2
#         """
#         image_rgb = _tensor_to_uint8(image_rgb)
#         H, W = image_rgb.shape[:2]

#         # Stage 1: GroundingDINO
#         img_tensor = _load_image_for_gdino(image_rgb, self.device)
#         boxes, scores = _run_grounding_dino(
#             self.gdino, img_tensor, text_prompt,
#             self.box_threshold, self.text_threshold, self.device,
#         )

#         if len(boxes) == 0:
#             return np.zeros((H, W), dtype=bool), 0.0

#         # Convert normalised cx,cy,w,h → absolute xyxy
#         boxes_xyxy = _boxes_xyxy_from_cxcywh(boxes, H, W)

#         # Stage 2: SAM2
#         mask, score = _run_sam2(self.sam2, image_rgb, boxes_xyxy)
#         return mask, score


# class SAM3Segmenter:
#     """
#     SAM3 segmenter – uses the SAM3 checkpoint with the same SAM2 API.
#     SAM3 integrates semantic text-prompted segmentation into training,
#     so we pass the text caption directly to the predictor when available,
#     and fall back to a full-image bounding box otherwise.
#     """

#     def __init__(
#         self,
#         checkpoint:  str,
#         model_cfg:   str,
#         device:      torch.device = torch.device("cpu"),
#     ):
#         self.device = device
#         console.log("Loading SAM3 …")
#         self.predictor = _load_sam2(checkpoint, model_cfg, device)
#         console.log("  SAM3 loaded")

#     def segment(
#         self,
#         image_rgb: np.ndarray,
#         text_prompt: str,
#     ) -> Tuple[np.ndarray, float]:
#         """
#         SAM3 can accept text prompts natively; the current release mirrors
#         the SAM2 image predictor API. We pass a full-image box as prompt and
#         use the automatic mask generation path which ranks masks by semantic
#         similarity when a text token is available.
#         """
#         image_rgb = _tensor_to_uint8(image_rgb)
#         H, W = image_rgb.shape[:2]

#         # Attempt text-prompted segmentation (SAM3 extended API)
#         try:
#             from sam2.sam2_image_predictor import SAM2ImagePredictor
#             self.predictor.set_image(image_rgb)
#             # SAM3 exposes predict_with_text in some builds
#             if hasattr(self.predictor, "predict_with_text"):
#                 masks, scores, _ = self.predictor.predict_with_text(
#                     text=text_prompt,
#                     multimask_output=True,
#                 )
#             else:
#                 # Fallback: use automatic mask generator and pick largest mask
#                 from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
#                 gen = SAM2AutomaticMaskGenerator(self.predictor.model)
#                 ann  = gen.generate(image_rgb)
#                 if not ann:
#                     return np.zeros((H, W), dtype=bool), 0.0
#                 # Pick largest mask (heuristic for dominant object)
#                 ann_sorted = sorted(ann, key=lambda x: x["area"], reverse=True)
#                 mask = ann_sorted[0]["segmentation"].astype(bool)
#                 return mask, float(ann_sorted[0].get("predicted_iou", 0.5))

#             best = np.argmax(scores)
#             mask = masks[best].astype(bool)
#             return mask, float(scores[best])

#         except Exception as e:
#             console.log(f"[yellow]SAM3 segment failed: {e} – returning empty mask[/yellow]")
#             return np.zeros((H, W), dtype=bool), 0.0


# def build_segmenter(cfg: dict, device: torch.device):
#     """Factory: build the configured segmenter from the YAML config dict."""
#     backbone = cfg["segmentation"]["backbone"]

#     if backbone == "grounded_sam2":
#         gdino_cfg  = cfg["segmentation"]["grounding_dino"]
#         sam2_cfg   = cfg["segmentation"]["sam2"]
#         return GroundedSAM2(
#             gdino_config     = gdino_cfg["config_path"],
#             gdino_checkpoint = gdino_cfg["checkpoint_path"],
#             sam2_checkpoint  = sam2_cfg["checkpoint"],
#             sam2_model_cfg   = sam2_cfg["model_cfg"],
#             box_threshold    = gdino_cfg["box_threshold"],
#             text_threshold   = gdino_cfg["text_threshold"],
#             device           = device,
#         )

#     elif backbone == "sam3":
#         sam3_cfg = cfg["segmentation"]["sam3"]
#         return SAM3Segmenter(
#             checkpoint = sam3_cfg["checkpoint"],
#             model_cfg  = sam3_cfg["model_cfg"],
#             device     = device,
#         )

#     else:
#         raise ValueError(f"Unknown segmentation backbone: {backbone!r}")
"""
segmentation.py
───────────────
2D segmentation backends:
  - GroundedSAM2:  GroundingDINO (CPU) → SAM2 (MPS/CPU)
  - SAM3Segmenter: SAM3 checkpoint (same API style)

Optimized for Apple Silicon:
  • GroundingDINO forced to CPU (avoids MPS OOM)
  • SAM2 can stay on MPS
  • Optional image downscaling before GroundingDINO
  • Automatic cache cleanup
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from rich.console import Console

console = Console()


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _cleanup_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _tensor_to_uint8(img_rgb: np.ndarray) -> np.ndarray:
    """Convert float [0,1] or uint8 [0,255] RGB ndarray to uint8."""
    if img_rgb.dtype != np.uint8:
        img_rgb = (img_rgb.clip(0, 1) * 255).astype(np.uint8)
    return img_rgb


def _resize_keep_aspect(img: np.ndarray, max_side: int = 640) -> tuple[np.ndarray, float]:
    """
    Resize image so longest side=max_side.
    Returns resized_img, scale_factor
    scale_factor = resized / original
    """
    h, w = img.shape[:2]
    longest = max(h, w)

    if longest <= max_side:
        return img, 1.0

    scale = max_side / longest
    new_w = int(w * scale)
    new_h = int(h * scale)

    pil = Image.fromarray(img)
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    return np.array(pil), scale


def _load_image_for_gdino(img_rgb: np.ndarray):
    """Prepare image for GroundingDINO."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    img_pil = Image.fromarray(_tensor_to_uint8(img_rgb))
    return transform(img_pil).unsqueeze(0)   # CPU tensor


# ──────────────────────────────────────────────────────────────────────────────
# GroundingDINO
# ──────────────────────────────────────────────────────────────────────────────

def _load_grounding_dino(config_path: str, checkpoint_path: str):
    """Always load GroundingDINO on CPU."""
    try:
        from groundingdino.util.inference import load_model
        model = load_model(config_path, checkpoint_path, device="cpu")
        return model, "official"
    except Exception:
        pass

    try:
        import groundingdino
        from groundingdino.util.inference import load_model
        model = load_model(config_path, checkpoint_path, device="cpu")
        return model, "pip"
    except Exception as e:
        raise ImportError(
            f"GroundingDINO not available: {e}\n"
            "Install it correctly first."
        )


def _run_grounding_dino(
    model,
    image_tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
):
    """Run GroundingDINO on CPU only."""
    try:
        from groundingdino.util.inference import predict

        boxes, scores, labels = predict(
            model=model,
            image=image_tensor.squeeze(0).cpu(),
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device="cpu",
        )

        return boxes, scores

    except Exception as e:
        console.log(f"[yellow]GroundingDINO failed: {e}[/yellow]")
        return torch.zeros(0, 4), torch.zeros(0)


# ──────────────────────────────────────────────────────────────────────────────
# SAM2 / SAM3
# ──────────────────────────────────────────────────────────────────────────────

def _load_sam2(checkpoint: str, model_cfg: str, device: torch.device):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(model_cfg, checkpoint, device=str(device))
        predictor = SAM2ImagePredictor(model)
        return predictor

    except Exception as e:
        raise ImportError(f"SAM2 load failed: {e}")


def _boxes_xyxy_from_cxcywh(boxes_cxcywh: torch.Tensor, H: int, W: int):
    if len(boxes_cxcywh) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    cx, cy, bw, bh = boxes_cxcywh.unbind(-1)

    x1 = (cx - bw / 2) * W
    y1 = (cy - bh / 2) * H
    x2 = (cx + bw / 2) * W
    y2 = (cy + bh / 2) * H

    return torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()


def _run_sam2(predictor, image_rgb: np.ndarray, boxes_xyxy: np.ndarray):
    predictor.set_image(image_rgb)

    if len(boxes_xyxy) == 0:
        H, W = image_rgb.shape[:2]
        return np.zeros((H, W), dtype=bool), 0.0

    masks, iou_preds, _ = predictor.predict(
        box=boxes_xyxy,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks[:, 0]

    merged = masks.any(axis=0)
    score = float(iou_preds.mean())

    return merged.astype(bool), score


# ──────────────────────────────────────────────────────────────────────────────
# Main Classes
# ──────────────────────────────────────────────────────────────────────────────

class GroundedSAM2:
    def __init__(
        self,
        gdino_config: str,
        gdino_checkpoint: str,
        sam2_checkpoint: str,
        sam2_model_cfg: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        device: torch.device = torch.device("cpu"),
        max_coverage: float = 0.25,
    ):
        self.box_threshold  = box_threshold
        self.text_threshold = text_threshold
        self.device         = device
        # Masks covering more than this fraction of the image are rejected
        # as background false-positives. 25% is right for most foreground objects.
        self.max_coverage   = max_coverage


        console.log("Loading GroundingDINO (CPU)…")
        self.gdino, self.variant = _load_grounding_dino(
            gdino_config,
            gdino_checkpoint
        )
        console.log("GroundingDINO loaded")

        console.log(f"Loading SAM2 ({device})…")
        self.sam2 = _load_sam2(
            sam2_checkpoint,
            sam2_model_cfg,
            device
        )
        console.log("SAM2 loaded")

    def segment(self, image_rgb: np.ndarray, text_prompt: str):
        image_rgb = _tensor_to_uint8(image_rgb)
        H, W = image_rgb.shape[:2]

        _cleanup_memory()

        # Resize for GroundingDINO only
        small_img, scale = _resize_keep_aspect(image_rgb, 640)
        img_tensor = _load_image_for_gdino(small_img)

        boxes, scores = _run_grounding_dino(
            self.gdino,
            img_tensor,
            text_prompt,
            self.box_threshold,
            self.text_threshold,
        )

        # FIX: log detection results so failures are visible in the terminal.
        # If you see "GDino: 0 boxes" every view, the prompt doesn't match the scene.
        # If you see large boxes (area > 0.5), it's likely a background false-positive.
        if len(boxes) == 0:
            console.log(f"    [yellow]GDino: 0 boxes for '{text_prompt}' – returning empty mask[/yellow]")
            return np.zeros((H, W), dtype=bool), 0.0

        # Log box areas (in normalised cxcywh, area = w*h)
        box_areas = (boxes[:, 2] * boxes[:, 3]).cpu().numpy()
        console.log(
            f"    GDino: {len(boxes)} box(es), scores={scores.cpu().numpy().round(3).tolist()}, "
            f"norm_areas={box_areas.round(3).tolist()}"
        )

        # boxes are on resized image -> convert using resized dims first
        h2, w2 = small_img.shape[:2]
        boxes_xyxy = _boxes_xyxy_from_cxcywh(boxes, h2, w2)

        # Scale back to original image size
        boxes_xyxy[:, [0, 2]] /= scale
        boxes_xyxy[:, [1, 3]] /= scale

        mask, score = _run_sam2(self.sam2, image_rgb, boxes_xyxy)

        # FIX: reject masks that cover more than 60% of the image.
        # When GroundingDINO detects the background instead of the object (common
        # in scenes with bright backgrounds like the bicycle scene), SAM2 will
        # mask the dominant background region. A real object mask rarely covers
        # more than 50% of the image. If coverage > 60%, this is almost certainly
        # a background false-positive — return empty mask rather than poisoning
        # the lifting scores for every background Gaussian.
        coverage = float(mask.sum()) / (H * W)
        max_coverage = getattr(self, "max_coverage", 0.25)
        console.log(f"    SAM2: coverage={coverage:.1%}, iou_score={score:.3f}, "
                    f"max_allowed={max_coverage:.0%}")
        if coverage > max_coverage:
            console.log(
                f"    [yellow]Coverage {coverage:.1%} > {max_coverage:.0%} threshold – "
                f"likely background false-positive. Discarding.[/yellow]"
            )
            _cleanup_memory()
            return np.zeros((H, W), dtype=bool), 0.0

        _cleanup_memory()
        return mask, score


class SAM3Segmenter:
    def __init__(
        self,
        checkpoint: str,
        model_cfg: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device

        console.log("Loading SAM3…")
        self.predictor = _load_sam2(checkpoint, model_cfg, device)
        console.log("SAM3 loaded")

    def segment(self, image_rgb: np.ndarray, text_prompt: str):
        H, W = image_rgb.shape[:2]

        try:
            self.predictor.set_image(image_rgb)

            if hasattr(self.predictor, "predict_with_text"):
                masks, scores, _ = self.predictor.predict_with_text(
                    text=text_prompt,
                    multimask_output=True,
                )
                best = np.argmax(scores)
                return masks[best].astype(bool), float(scores[best])

            full_box = np.array([[0, 0, W, H]], dtype=np.float32)
            return _run_sam2(self.predictor, image_rgb, full_box)

        except Exception as e:
            console.log(f"[yellow]SAM3 failed: {e}[/yellow]")
            return np.zeros((H, W), dtype=bool), 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_segmenter(cfg: dict, device: torch.device):
    backbone = cfg["segmentation"]["backbone"]

    if backbone == "grounded_sam2":
        gd = cfg["segmentation"]["grounding_dino"]
        sm = cfg["segmentation"]["sam2"]

        return GroundedSAM2(
            gdino_config=gd["config_path"],
            gdino_checkpoint=gd["checkpoint_path"],
            sam2_checkpoint=sm["checkpoint"],
            sam2_model_cfg=sm["model_cfg"],
            box_threshold=gd["box_threshold"],
            text_threshold=gd["text_threshold"],
            device=device,
        )

    elif backbone == "sam3":
        sm = cfg["segmentation"]["sam3"]

        return SAM3Segmenter(
            checkpoint=sm["checkpoint"],
            model_cfg=sm["model_cfg"],
            device=device,
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone}")