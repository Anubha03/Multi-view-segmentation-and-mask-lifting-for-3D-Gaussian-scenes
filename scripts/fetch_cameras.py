"""
fetch_cameras.py
────────────────
Downloads COLMAP camera files for a Mip-NeRF 360 scene and places them
exactly where camera_utils.py expects them:

    scenes/<scene>/sparse/0/cameras.bin
    scenes/<scene>/sparse/0/images.bin
    scenes/<scene>/sparse/0/points3D.bin   (not needed for rendering, included anyway)

Usage:
    python scripts/fetch_cameras.py --scene bicycle
    python scripts/fetch_cameras.py --scene garden
    python scripts/fetch_cameras.py --scene counter
    python scripts/fetch_cameras.py --scene kitchen

Available outdoor scenes : bicycle, garden, stump, treehill, flowers
Available indoor scenes  : counter, kitchen, room, bonsai

The Mip-NeRF 360 dataset is hosted by the original authors at:
    https://storage.googleapis.com/gresearch/refraw360/

Each scene is ~1–3 GB total. This script only downloads the COLMAP sparse
reconstruction (~5 MB) so it's fast.
"""

import argparse
import hashlib
import io
import os
import struct
import sys
import zipfile
from pathlib import Path
import urllib.request
import urllib.error

# ── Dataset manifest ──────────────────────────────────────────────────────────
# The Mip-NeRF 360 paper provides two zip archives:
#   360_v2.zip  – outdoor scenes (bicycle, garden, stump, treehill, flowers)
#   360_indoor.zip – indoor scenes (counter, kitchen, room, bonsai)
# Both contain full images + COLMAP sparse data.
# We stream-extract only the sparse/0/ subfolder to save bandwidth.

OUTDOOR_SCENES = {"bicycle", "garden", "stump", "treehill", "flowers"}
INDOOR_SCENES  = {"counter", "kitchen", "room", "bonsai"}

BASE_URL = "https://storage.googleapis.com/gresearch/refraw360"
OUTDOOR_ZIP = "360_v2.zip"
INDOOR_ZIP  = "360_indoor.zip"      # some mirrors call this 360_v2_extra.zip

# Alternative: the authors also host per-scene tarballs on their project page.
# If the Google Storage URLs are slow, try the academic torrents mirror.
ALT_URLS = {
    # Per-scene COLMAP-only tarballs (much smaller, ~5 MB each).
    # Hosted by the nerfstudio team who re-packaged the dataset.
    "bicycle": "https://huggingface.co/datasets/nerfstudio/nerf-datasets/resolve/main/mipnerf360/bicycle.zip",
    "garden":  "https://huggingface.co/datasets/nerfstudio/nerf-datasets/resolve/main/mipnerf360/garden.zip",
    "counter": "https://huggingface.co/datasets/nerfstudio/nerf-datasets/resolve/main/mipnerf360/counter.zip",
    "kitchen": "https://huggingface.co/datasets/nerfstudio/nerf-datasets/resolve/main/mipnerf360/kitchen.zip",
    "room":    "https://huggingface.co/datasets/nerfstudio/nerf-datasets/resolve/main/mipnerf360/room.zip",
    "bonsai":  "https://huggingface.co/datasets/nerfstudio/nerf-datasets/resolve/main/mipnerf360/bonsai.zip",
    "stump":   "https://huggingface.co/datasets/nerfstudio/nerf-datasets/resolve/main/mipnerf360/stump.zip",
    "treehill":"https://huggingface.co/datasets/nerfstudio/nerf-datasets/resolve/main/mipnerf360/treehill.zip",
    "flowers": "https://huggingface.co/datasets/nerfstudio/nerf-datasets/resolve/main/mipnerf360/flowers.zip",
}

NEEDED_SUFFIXES = (
    "sparse/0/cameras.bin",
    "sparse/0/images.bin",
    "sparse/0/points3D.bin",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def bar(downloaded, total, width=40):
    if total <= 0:
        print(f"\r  {downloaded // 1024 // 1024} MB downloaded...", end="", flush=True)
        return
    frac = downloaded / total
    filled = int(width * frac)
    pct = frac * 100
    mb_done  = downloaded / 1024 / 1024
    mb_total = total / 1024 / 1024
    print(f"\r  [{'█'*filled}{'░'*(width-filled)}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB",
          end="", flush=True)


def download_bytes(url: str) -> bytes:
    """Download url → bytes with a progress bar."""
    print(f"  Downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fetch_cameras/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            buf   = io.BytesIO()
            done  = 0
            chunk = 1 << 17   # 128 KB
            while True:
                data = resp.read(chunk)
                if not data:
                    break
                buf.write(data)
                done += len(data)
                bar(done, total)
            print()   # newline after bar
            return buf.getvalue()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {url}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error: {e.reason}  →  {url}") from e


def extract_sparse(zip_bytes: bytes, scene: str, out_dir: Path):
    """Extract only sparse/0/*.bin from a zip into out_dir/sparse/0/."""
    sparse_out = out_dir / "sparse" / "0"
    sparse_out.mkdir(parents=True, exist_ok=True)

    extracted = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        all_names = zf.namelist()
        # The zip may have a top-level scene folder: bicycle/sparse/0/cameras.bin
        # or just: sparse/0/cameras.bin  — handle both.
        for name in all_names:
            for suffix in NEEDED_SUFFIXES:
                # Match  <anything>/sparse/0/cameras.bin  or  sparse/0/cameras.bin
                if name.endswith(suffix) and (
                    f"/{scene}/" in f"/{name}" or name.startswith(f"{scene}/") or
                    not any(s in name for s in OUTDOOR_SCENES | INDOOR_SCENES)
                ):
                    dest = sparse_out / Path(suffix).name
                    print(f"  Extracting {name} → {dest}")
                    dest.write_bytes(zf.read(name))
                    extracted.append(dest)
                    break

    return extracted


def verify_colmap_bins(sparse_dir: Path) -> bool:
    """Quick sanity check: try reading the first camera from each bin."""
    cb = sparse_dir / "cameras.bin"
    ib = sparse_dir / "images.bin"
    if not cb.exists() or not ib.exists():
        return False
    try:
        with open(cb, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        with open(ib, "rb") as f:
            n2 = struct.unpack("<Q", f.read(8))[0]
        print(f"  ✓ cameras.bin has {n} camera(s), images.bin has {n2} image(s)")
        return n > 0 and n2 > 0
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fetch Mip-NeRF 360 COLMAP cameras for a scene.")
    ap.add_argument("--scene",      default="bicycle",
                    help="Scene name (bicycle, garden, counter, kitchen, …)")
    ap.add_argument("--scenes_dir", default="scenes",
                    help="Root scenes directory (default: scenes/)")
    ap.add_argument("--force",      action="store_true",
                    help="Re-download even if files already exist")
    args = ap.parse_args()

    scene     = args.scene.lower()
    all_known = OUTDOOR_SCENES | INDOOR_SCENES
    if scene not in all_known:
        print(f"Unknown scene '{scene}'. Known: {sorted(all_known)}")
        sys.exit(1)

    out_dir    = Path(args.scenes_dir) / scene
    sparse_dir = out_dir / "sparse" / "0"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Check if already present ───────────────────────────────────────────
    if not args.force and verify_colmap_bins(sparse_dir):
        print(f"\n✓ Camera files already present at {sparse_dir}/")
        print("  Run with --force to re-download.")
        return

    print(f"\nFetching COLMAP cameras for scene: {scene}")
    print(f"Output directory: {sparse_dir}/\n")

    # ── Try Hugging Face per-scene zip first (smallest download) ──────────
    url = ALT_URLS.get(scene)
    zip_bytes = None

    if url:
        try:
            print(f"Trying per-scene zip (HuggingFace mirror)…")
            zip_bytes = download_bytes(url)
        except RuntimeError as e:
            print(f"  HuggingFace mirror failed: {e}")
            zip_bytes = None

    # ── Fall back to Google Storage full-dataset zip ───────────────────────
    if zip_bytes is None:
        zip_name = OUTDOOR_ZIP if scene in OUTDOOR_SCENES else INDOOR_ZIP
        url = f"{BASE_URL}/{zip_name}"
        print(f"\nFalling back to full dataset zip ({zip_name})…")
        print("  Warning: this is a large file (1–3 GB). Only sparse/ will be extracted.")
        try:
            zip_bytes = download_bytes(url)
        except RuntimeError as e:
            print(f"\n✗ Download failed: {e}")
            print("\nManual download instructions:")
            print(f"  1. Download: {url}")
            print(f"  2. Extract: {scene}/sparse/0/cameras.bin")
            print(f"              {scene}/sparse/0/images.bin")
            print(f"              {scene}/sparse/0/points3D.bin")
            print(f"  3. Place in: {sparse_dir}/")
            sys.exit(1)

    # ── Extract ────────────────────────────────────────────────────────────
    print(f"\nExtracting sparse reconstruction…")
    extracted = extract_sparse(zip_bytes, scene, out_dir)

    if not extracted:
        print("\n✗ No sparse/0/*.bin files found in the downloaded zip.")
        print("  The zip structure may have changed. Please extract manually:")
        print(f"  Place cameras.bin, images.bin, points3D.bin in: {sparse_dir}/")
        sys.exit(1)

    # ── Verify ────────────────────────────────────────────────────────────
    print("\nVerifying extracted files…")
    ok = verify_colmap_bins(sparse_dir)

    if ok:
        print(f"\n✓ Done! Camera files ready at: {sparse_dir}/")
        print(f"\nYou can now run:")
        print(f"  python scripts/debug_render.py --scene {scene}")
        print(f"  python scripts/run_pipeline.py --scene {scene} --prompt 'the {scene}' --edit recolor")
    else:
        print(f"\n✗ Verification failed. Check {sparse_dir}/ manually.")
        sys.exit(1)


if __name__ == "__main__":
    main()