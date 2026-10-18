#!/usr/bin/env bash
# setup_and_train.sh
# Downloads Mip-NeRF 360 scenes via nerfbaselines and trains 3DGS with nerfstudio.
# Run from inside your project directory (anubha_proj or 3dgs_segmentation).
#
# Usage:
#   bash setup_and_train.sh                  # trains all 6 scenes
#   bash setup_and_train.sh garden bonsai    # trains specific scenes
#
# Prerequisites (already installed in your venv):
#   nerfbaselines, nerfstudio (ns-train), plyfile

set -euo pipefail
SCENES=("${@:-garden bonsai kitchen counter room bicycle}")
if [ "$#" -eq 0 ]; then
  SCENES=(garden bonsai kitchen counter room bicycle)
fi

PROJECT_DIR="$(pwd)"
SCENE_DIR="$PROJECT_DIR/scenes"
mkdir -p "$SCENE_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
section() { echo -e "\n${YELLOW}════════════════════════════════════════${NC}"; \
            echo -e "${YELLOW}  $*${NC}"; \
            echo -e "${YELLOW}════════════════════════════════════════${NC}"; }

# ── Find PLY in a scene dir (searches common nerfstudio output locations) ─────
find_ply() {
  local scene_dir="$1"
  # nerfstudio saves to: outputs/<scene>/splatfacto/<timestamp>/nerfstudio_models/
  # but exports PLY to:  outputs/<scene>/splatfacto/<timestamp>/point_cloud/iteration_30000/
  local ply
  ply=$(find "$scene_dir" "$PROJECT_DIR/outputs" -name "point_cloud.ply" \
        2>/dev/null | head -1)
  echo "$ply"
}

# ── Detect colmap path inside a downloaded scene ──────────────────────────────
detect_colmap_args() {
  local scene_dir="$1"
  local colmap_path="" images_path=""

  # Check nb-info.json first (nerfbaselines writes this)
  local nb_info="$scene_dir/nb-info.json"
  if [ -f "$nb_info" ]; then
    colmap_path=$(python3 -c "
import json, sys
d = json.load(open('$nb_info'))
print(d.get('loader_kwargs',{}).get('colmap_path',''))
" 2>/dev/null || echo "")
    images_path=$(python3 -c "
import json, sys
d = json.load(open('$nb_info'))
print(d.get('loader_kwargs',{}).get('images_path','images'))
" 2>/dev/null || echo "images")
  fi

  # Fallback: search for cameras.bin
  if [ -z "$colmap_path" ]; then
    local found
    found=$(find "$scene_dir" -name "cameras.bin" 2>/dev/null | head -1)
    if [ -n "$found" ]; then
      colmap_path=$(dirname "$found" | sed "s|$scene_dir/||")
    fi
  fi

  # Fallback: search for images dir
  if [ -z "$images_path" ]; then
    for candidate in images_4 images_2 images; do
      if [ -d "$scene_dir/$candidate" ]; then
        images_path="$candidate"
        break
      fi
    done
  fi

  echo "$colmap_path|$images_path"
}

# ── Export PLY from nerfstudio checkpoint ─────────────────────────────────────
export_ply_from_checkpoint() {
  local scene="$1"
  local output_dir="$PROJECT_DIR/outputs/$scene/splatfacto"
  local dest_ply="$SCENE_DIR/$scene/point_cloud/iteration_30000/point_cloud.ply"

  if [ -f "$dest_ply" ]; then
    info "$scene PLY already exists at $dest_ply"
    echo "$dest_ply"; return 0
  fi

  # Find the latest checkpoint dir
  local ckpt_dir
  ckpt_dir=$(ls -td "$output_dir"/*/  2>/dev/null | head -1)
  if [ -z "$ckpt_dir" ]; then
    error "No nerfstudio checkpoint found for $scene in $output_dir"
    return 1
  fi

  info "Exporting PLY from checkpoint: $ckpt_dir"
  mkdir -p "$(dirname "$dest_ply")"

  # nerfstudio export gaussians
  ns-export gaussian-splat \
    --load-config "$ckpt_dir/config.yml" \
    --output-dir "$SCENE_DIR/$scene/point_cloud/iteration_30000/" \
    2>/dev/null || true

  # The export names it differently sometimes — rename if needed
  local exported
  exported=$(find "$SCENE_DIR/$scene/point_cloud/iteration_30000/" \
             -name "*.ply" 2>/dev/null | head -1)
  if [ -n "$exported" ] && [ "$exported" != "$dest_ply" ]; then
    mv "$exported" "$dest_ply"
  fi

  if [ -f "$dest_ply" ]; then
    info "PLY exported → $dest_ply"
    echo "$dest_ply"
  else
    error "PLY export failed for $scene"
    return 1
  fi
}

# ── Process one scene ─────────────────────────────────────────────────────────
process_scene() {
  local scene="$1"
  section "Scene: $scene"

  local scene_dir="$SCENE_DIR/$scene"
  local dest_ply="$scene_dir/point_cloud/iteration_30000/point_cloud.ply"

  # ── Already have PLY? ──────────────────────────────────────────────────────
  if [ -f "$dest_ply" ]; then
    info "✓ $scene PLY already exists — skipping download and training."
    return 0
  fi

  # ── Step 1: Download raw images via nerfbaselines ──────────────────────────
  if [ ! -f "$scene_dir/nb-info.json" ]; then
    info "Downloading $scene via nerfbaselines..."
    mkdir -p "$scene_dir"
    nerfbaselines download-dataset \
      "external://mipnerf360/$scene" \
      -o "$scene_dir" || {
      error "Download failed for $scene"
      return 1
    }
    info "Download complete."
  else
    info "$scene already downloaded (nb-info.json found)."
  fi

  # ── Step 2: Detect colmap/images paths ────────────────────────────────────
  local args
  args=$(detect_colmap_args "$scene_dir")
  local colmap_path images_path
  colmap_path=$(echo "$args" | cut -d'|' -f1)
  images_path=$(echo "$args" | cut -d'|' -f2)

  info "COLMAP path: $colmap_path"
  info "Images path: $images_path"

  if [ -z "$colmap_path" ]; then
    error "Could not detect COLMAP path for $scene. Contents:"
    find "$scene_dir" -maxdepth 3 -type d
    return 1
  fi

  # ── Step 3: Fix val_list.txt ───────────────────────────────────────────────
  if [ -f "$scene_dir/test_list.txt" ] && [ ! -f "$scene_dir/val_list.txt" ]; then
    cp "$scene_dir/test_list.txt" "$scene_dir/val_list.txt"
    info "Created val_list.txt from test_list.txt"
  fi

  # ── Step 4: Train with nerfstudio splatfacto ──────────────────────────────
  info "Training $scene with splatfacto (MPS)..."
  info "This takes 20-40 minutes. Do not close the terminal."

  PYTORCH_ENABLE_MPS_FALLBACK=1 \
  TORCHDYNAMO_DISABLE=1 \
  ns-train splatfacto \
    --data "$scene_dir" \
    --output-dir "$PROJECT_DIR/outputs" \
    --experiment-name "$scene" \
    --machine.device-type mps \
    --max-num-iterations 15000 \
    --steps-per-save 5000 \
    --vis viewer \
    colmap \
    --colmap-path "$colmap_path" \
    --images-path "$images_path" || {
    error "Training failed for $scene"
    return 1
  }

  # ── Step 5: Export PLY ────────────────────────────────────────────────────
  export_ply_from_checkpoint "$scene" || {
    # Fallback: find any ply nerfstudio wrote
    local any_ply
    any_ply=$(find "$PROJECT_DIR/outputs/$scene" -name "*.ply" 2>/dev/null | head -1)
    if [ -n "$any_ply" ]; then
      mkdir -p "$(dirname "$dest_ply")"
      cp "$any_ply" "$dest_ply"
      info "Copied PLY from $any_ply → $dest_ply"
    else
      error "Could not find any PLY for $scene after training."
      return 1
    fi
  }

  info "✓ $scene complete → $dest_ply"
}

# ── Main ──────────────────────────────────────────────────────────────────────
section "3DGS Scene Setup & Training"
echo "Project dir: $PROJECT_DIR"
echo "Scenes:      ${SCENES[*]}"
echo ""

# Check dependencies
for cmd in nerfbaselines ns-train python3; do
  if ! command -v "$cmd" &>/dev/null; then
    error "$cmd not found. Make sure your venv is activated."
    exit 1
  fi
done

FAILED=()
for scene in "${SCENES[@]}"; do
  if process_scene "$scene"; then
    info "✓ $scene done"
  else
    warn "✗ $scene failed — continuing with other scenes"
    FAILED+=("$scene")
  fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
section "Summary"
for scene in "${SCENES[@]}"; do
  ply="$SCENE_DIR/$scene/point_cloud/iteration_30000/point_cloud.ply"
  if [ -f "$ply" ]; then
    size=$(du -sh "$ply" | cut -f1)
    echo -e "  ${GREEN}✓${NC}  $scene  ($size)  →  $ply"
  else
    echo -e "  ${RED}✗${NC}  $scene  FAILED"
  fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  warn "Failed scenes: ${FAILED[*]}"
  echo "  You can retry individually:"
  for s in "${FAILED[@]}"; do
    echo "    bash setup_and_train.sh $s"
  done
fi

echo ""
info "Done! Run the pipeline with e.g.:"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/run_pipeline.py \\"
echo "    --scene garden --prompt 'the garden table' --edit recolor"