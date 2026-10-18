#!/usr/bin/env bash
# =============================================================================
#  setup.sh  –  One-shot environment setup for M1 Mac (MPS backend)
#  Usage:  bash setup.sh
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
SAM2_DIR="$PROJECT_DIR/third_party/sam2"
GDINO_DIR="$PROJECT_DIR/third_party/GroundingDINO"

echo "=============================================="
echo "  3DGS Semantic Segmentation — Environment Setup"
echo "  Project: $PROJECT_DIR"
echo "=============================================="

# ── 0. Prerequisites check ────────────────────────────────────────────────────
for cmd in python3 git; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "ERROR: '$cmd' not found. Install it first." && exit 1
  fi
done

PYTHON_VERSION=$(python3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" < "3.10" ]]; then
  echo "ERROR: Python 3.10+ required (found $PYTHON_VERSION)."
  echo "Install via: brew install python@3.11"
  exit 1
fi

ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
  echo "WARNING: This script is optimised for Apple Silicon (arm64). Found: $ARCH"
fi

# ── 1. Create venv ────────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  echo ""
  echo ">>> Creating virtual environment at $VENV_DIR"
  python3.11 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools --quiet

# ── 2. PyTorch (ships MPS on macOS arm64) ────────────────────────────────────
echo ""
echo ">>> Installing PyTorch (latest stable, MPS-enabled macOS wheel)"
# Install latest available torch – PyPI only carries recent builds on arm64.
# torchvision version is inferred automatically by pip to match torch.
pip install torch torchvision --quiet

# Verify MPS
python3.11 - <<'EOF'
import torch
if torch.backends.mps.is_available():
    print("  ✓ MPS backend available")
else:
    print("  ⚠ MPS not available – will fall back to CPU")
EOF

# ── 3. Core requirements ──────────────────────────────────────────────────────
echo ""
echo ">>> Installing Python requirements"
pip install -r "$PROJECT_DIR/requirements.txt" --quiet

# ── 4. SAM2 from source ───────────────────────────────────────────────────────
echo ""
echo ">>> Installing SAM2 from source"
mkdir -p "$PROJECT_DIR/third_party"

if [[ ! -d "$SAM2_DIR" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
fi

cd "$SAM2_DIR"
# Install in editable mode without building CUDA extensions (MPS path uses Python ops)
pip install -e ".[demo]" --no-build-isolation --quiet || pip install -e . --quiet
cd "$PROJECT_DIR"

# Download SAM2 checkpoints
CKPT_DIR="$PROJECT_DIR/checkpoints/sam2"
mkdir -p "$CKPT_DIR"

echo ">>> Downloading SAM2 checkpoints"
SAM2_BASE="https://dl.fbaipublicfiles.com/segment_anything_2/072824"
for CKPT in "sam2_hiera_tiny.pt" "sam2_hiera_small.pt"; do
  if [[ ! -f "$CKPT_DIR/$CKPT" ]]; then
    echo "  Downloading $CKPT …"
    curl -L --progress-bar "$SAM2_BASE/$CKPT" -o "$CKPT_DIR/$CKPT"
  else
    echo "  ✓ $CKPT already present"
  fi
done

# ── 5. GroundingDINO ──────────────────────────────────────────────────────────
echo ""
echo ">>> Installing GroundingDINO"
if [[ ! -d "$GDINO_DIR" ]]; then
  git clone --depth 1 https://github.com/IDEA-Research/GroundingDINO.git "$GDINO_DIR"
fi

cd "$GDINO_DIR"
# Force CPU/MPS build (no CUDA)
TORCH_CUDA_ARCH_LIST="" pip install -e . --quiet 2>/dev/null || {
  echo "  GroundingDINO C++ ext failed – installing Python-only fallback"
  pip install groundingdino-py --quiet
}
cd "$PROJECT_DIR"

# Download GroundingDINO weights
GDINO_CKPT_DIR="$PROJECT_DIR/checkpoints/grounding_dino"
mkdir -p "$GDINO_CKPT_DIR"
GDINO_WEIGHT="groundingdino_swint_ogc.pth"
if [[ ! -f "$GDINO_CKPT_DIR/$GDINO_WEIGHT" ]]; then
  echo "  Downloading GroundingDINO weights …"
  curl -L --progress-bar \
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/$GDINO_WEIGHT" \
    -o "$GDINO_CKPT_DIR/$GDINO_WEIGHT"
fi

GDINO_CFG="GroundingDINO_SwinT_OGC.py"
if [[ ! -f "$GDINO_CKPT_DIR/$GDINO_CFG" ]]; then
  curl -L --progress-bar \
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/$GDINO_CFG" \
    -o "$GDINO_CKPT_DIR/$GDINO_CFG"
fi

# ── 5b. Fix supervision version conflict with groundingdino-py ───────────────
# groundingdino-py pins supervision==0.6.0 but we need a newer version.
# Force-reinstall our version after groundingdino-py is installed.
echo ""
echo ">>> Fixing supervision version conflict"
pip install "supervision>=0.21" --force-reinstall --quiet
# groundingdino-py is incompatible with transformers>=4.36 (BertModel.get_head_mask removed)
pip install "transformers==4.35.2" --quiet

# ── 6. Download pre-trained 3DGS scenes ──────────────────────────────────────
echo ""
echo ">>> Downloading pre-trained 3DGS scenes"
SCENE_DIR="$PROJECT_DIR/scenes"
mkdir -p "$SCENE_DIR"

# Fix: previous setup.sh versions used local_dir=scene_dir/scene_name which caused
# hf_hub_download to save at scene_dir/scene_name/scene_name/point_cloud/...
# Move files to the correct location if this happened.
for SCENE in bonsai bicycle garden room stump; do
  WRONG="$SCENE_DIR/$SCENE/$SCENE/point_cloud/iteration_30000/point_cloud.ply"
  RIGHT="$SCENE_DIR/$SCENE/point_cloud/iteration_30000/point_cloud.ply"
  if [[ -f "$WRONG" && ! -f "$RIGHT" ]]; then
    echo "  Fixing misplaced $SCENE PLY …"
    mkdir -p "$(dirname "$RIGHT")"
    mv "$WRONG" "$RIGHT"
    echo "  ✓ Moved $SCENE/point_cloud.ply to correct location"
  fi
done

python3.11 - "$SCENE_DIR" <<'PYEOF'
import sys, shutil, numpy as np
from pathlib import Path

scene_dir = Path(sys.argv[1])

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("  huggingface_hub not installed – skipping")
    sys.exit(0)

# ── What's actually in dylanebert/3dgs ───────────────────────────────────────
# Scenes WITH point_cloud/iteration_30000/point_cloud.ply:
#   garden, room, bonsai, bicycle, stump, playroom
# Scenes with ONLY .splat files (no point_cloud subfolder):
#   counter → counter/counter-7k.splat
#   kitchen → kitchen/kitchen-7k.splat
#
# Key fix: hf_hub_download with local_dir=scene_dir saves the file at
#   scene_dir / hf_path  (preserves the full repo path including scene prefix)
# So for hf_path="bonsai/point_cloud/iteration_30000/point_cloud.ply"
# with local_dir=scene_dir, the file lands at:
#   scene_dir/bonsai/point_cloud/iteration_30000/point_cloud.ply  ✓

PLY_SCENES = [
    # (scene_name, hf_path)  — all from dylanebert/3dgs dataset
    ("garden",  "garden/point_cloud/iteration_30000/point_cloud.ply"),
    ("room",    "room/point_cloud/iteration_30000/point_cloud.ply"),
    ("bonsai",  "bonsai/point_cloud/iteration_30000/point_cloud.ply"),
    ("bicycle", "bicycle/point_cloud/iteration_30000/point_cloud.ply"),
]

# counter and kitchen only have .splat files → download and convert
SPLAT_SCENES = [
    # (scene_name, hf_path_to_splat)
    ("counter", "counter/counter-7k.splat"),
    ("kitchen", "kitchen/kitchen-7k.splat"),
]

REPO_ID   = "dylanebert/3dgs"
REPO_TYPE = "dataset"


def splat_to_ply(splat_path: Path, ply_path: Path):
    """Convert .splat binary → standard 3DGS .ply.
    .splat layout (32 bytes/Gaussian):
      [0:12]  xyz        3×f32
      [12:24] scales     3×f32  (linear, not log)
      [24:28] rgba       4×u8   (r,g,b=colour 0-255; a=opacity 0-255)
      [28:32] quaternion 4×u8   packed: val = u8/127.5 - 1.0, then normalise
    """
    from plyfile import PlyData, PlyElement
    SH_C0 = 0.28209479177387814
    data = splat_path.read_bytes()
    n = len(data) // 32
    raw = np.frombuffer(data[:n*32], dtype=np.uint8).reshape(n, 32)

    xyz    = raw[:, 0:12].view(np.float32).reshape(n, 3).copy()
    scales = raw[:, 12:24].view(np.float32).reshape(n, 3).copy()
    rgba   = raw[:, 24:28].astype(np.float32)
    rot_u8 = raw[:, 28:32].astype(np.float32)

    sh_dc         = (rgba[:, :3] / 255.0 - 0.5) / SH_C0
    opacity_01    = np.clip(rgba[:, 3] / 255.0, 1e-6, 1-1e-6)
    opacity_logit = np.log(opacity_01 / (1 - opacity_01))
    scales_log    = np.log(np.clip(scales, 1e-8, None))
    quat          = rot_u8 / 127.5 - 1.0
    quat         /= np.linalg.norm(quat, axis=1, keepdims=True).clip(min=1e-8)

    dt = [("x","f4"),("y","f4"),("z","f4"),
          ("scale_0","f4"),("scale_1","f4"),("scale_2","f4"),
          ("f_dc_0","f4"),("f_dc_1","f4"),("f_dc_2","f4"),
          ("opacity","f4"),
          ("rot_0","f4"),("rot_1","f4"),("rot_2","f4"),("rot_3","f4")]
    arr = np.empty(n, dtype=dt)
    for i,k in enumerate(["x","y","z"]):       arr[k] = xyz[:,i]
    for i in range(3): arr[f"scale_{i}"]        = scales_log[:,i]
    for i in range(3): arr[f"f_dc_{i}"]          = sh_dc[:,i]
    arr["opacity"] = opacity_logit
    for i in range(4): arr[f"rot_{i}"]           = quat[:,i]

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")]).write(str(ply_path))
    print(f"    Converted {n:,} Gaussians → {ply_path.name}")


def cleanup(scene_dir_path: Path):
    for d in scene_dir_path.rglob(".cache"):
        shutil.rmtree(d, ignore_errors=True)


# ── Download full .ply scenes ─────────────────────────────────────────────────
for scene_name, hf_path in PLY_SCENES:
    ply_dest = scene_dir / scene_name / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if ply_dest.exists():
        print(f"  ✓ {scene_name} already present  ({ply_dest.stat().st_size/1e6:.0f} MB)")
        continue
    print(f"  Downloading {scene_name} …")
    try:
        # local_dir=scene_dir  →  file saved at  scene_dir/hf_path  ✓
        hf_hub_download(repo_id=REPO_ID, filename=hf_path,
                        repo_type=REPO_TYPE, local_dir=str(scene_dir))
        cleanup(scene_dir / scene_name)
        if ply_dest.exists():
            print(f"  ✓ {scene_name}  ({ply_dest.stat().st_size/1e6:.0f} MB)")
        else:
            print(f"  ⚠ {scene_name}: unexpected path, check {scene_dir/scene_name}")
    except Exception as e:
        print(f"  ⚠ {scene_name} failed: {e}")


# ── Download + convert .splat scenes ─────────────────────────────────────────
for scene_name, hf_path in SPLAT_SCENES:
    ply_dest = scene_dir / scene_name / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if ply_dest.exists():
        print(f"  ✓ {scene_name} already present  ({ply_dest.stat().st_size/1e6:.0f} MB)")
        continue
    print(f"  Downloading {scene_name} (.splat) …")
    try:
        # Download splat to a temp dir, then convert
        tmp_dir = scene_dir / scene_name / "_tmp"
        hf_hub_download(repo_id=REPO_ID, filename=hf_path,
                        repo_type=REPO_TYPE, local_dir=str(tmp_dir))
        splat_file = tmp_dir / hf_path   # e.g. tmp_dir/counter/counter-7k.splat
        if not splat_file.exists():
            # fallback: find any .splat in tmp_dir
            found = list(tmp_dir.rglob("*.splat"))
            if not found:
                raise FileNotFoundError(f"No .splat found in {tmp_dir}")
            splat_file = found[0]
        print(f"    Converting {splat_file.name} → point_cloud.ply …")
        splat_to_ply(splat_file, ply_dest)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  ✓ {scene_name}  ({ply_dest.stat().st_size/1e6:.0f} MB)")
    except Exception as e:
        print(f"  ⚠ {scene_name} failed: {e}")


print("")
print("  Scene download complete.")
print("  NOTE: No camera files – pipeline auto-generates orbit cameras.")
PYEOF

# ── 7. Final check ────────────────────────────────────────────────────────────
echo ""
echo ">>> Verifying installation"
python3.11 - <<'PYEOF'
import importlib, sys
ok = True
for mod in ["torch", "torchvision", "numpy", "PIL", "plyfile", "cv2", "clip", "lpips", "tqdm", "rich"]:
    try:
        importlib.import_module(mod)
        print(f"  ✓ {mod}")
    except ImportError as e:
        print(f"  ✗ {mod}: {e}")
        ok = False
if not ok:
    sys.exit(1)
PYEOF

echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Activate env:  source .venv/bin/activate"
echo "  Run pipeline:  python scripts/run_pipeline.py --scene garden --prompt \"the plant\""
echo "  Run eval only: python scripts/evaluate.py --scene garden"
echo "=============================================="