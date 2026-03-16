"""
STEP 1c: Traditional Augmentation — Generate 500 images per minority breed
Uses only classical image transforms on existing real images.
No GPU needed. Runs in seconds.

Transforms applied:
  - Horizontal flip
  - Rotation (±10°, ±20°)
  - Colour jitter (brightness, contrast, saturation, hue)
  - Random crop + resize
  - Gaussian blur
  - Combinations of the above

Requirements:
    pip install pillow pandas numpy
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import json

# ── Config ────────────────────────────────────────────────────────────────────
SPLIT_DIR   = Path("data/oxford_pet/splits")
RESULTS_DIR = Path("results")
SYNTH_DIR   = Path("data/synthetic/traditional")
N_GENERATE  = 500
RANDOM_SEED = 42
IMG_SIZE    = 224

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Individual transforms ─────────────────────────────────────────────────────
def apply_hflip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def apply_rotation(img, angle=None):
    if angle is None:
        angle = random.choice([-20, -15, -10, -5, 5, 10, 15, 20])
    return img.rotate(angle, resample=Image.BILINEAR, expand=False)

def apply_color_jitter(img):
    # Randomly adjust brightness, contrast, saturation, hue
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    return img

def apply_crop(img, size=IMG_SIZE):
    w, h = img.size
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    x1 = random.randint(0, margin_x)
    y1 = random.randint(0, margin_y)
    x2 = random.randint(w - margin_x, w)
    y2 = random.randint(h - margin_y, h)
    return img.crop((x1, y1, x2, y2)).resize((size, size), Image.LANCZOS)

def apply_blur(img):
    radius = random.uniform(0.5, 1.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def apply_sharpen(img):
    return img.filter(ImageFilter.SHARPEN)

# ── Transform pipeline: pick a random combination each time ──────────────────
def random_augment(img: Image.Image) -> Image.Image:
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Always apply at least one transform
    transforms_pool = [
        lambda i: apply_hflip(i),
        lambda i: apply_rotation(i),
        lambda i: apply_color_jitter(i),
        lambda i: apply_crop(i),
        lambda i: apply_blur(i),
        lambda i: apply_sharpen(i),
    ]

    # Pick 1–3 transforms randomly
    n = random.randint(1, 3)
    chosen = random.sample(transforms_pool, n)
    for t in chosen:
        try:
            img = t(img)
        except Exception:
            pass  # skip if transform fails on edge case

    # Ensure final size is correct
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return img

# ── Generate images for one breed ─────────────────────────────────────────────
def generate_for_breed(breed: str, real_paths: list, out_dir: Path, n: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    existing = list(out_dir.glob("*.jpg"))
    if len(existing) >= n:
        print(f"  [skip] {breed} already has {len(existing)} images")
        return

    print(f"  Generating {n} images for {breed} from {len(real_paths)} real images...")

    count = 0
    while count < n:
        # Pick a random real image to augment
        src_path = random.choice(real_paths)
        try:
            img = Image.open(src_path).convert("RGB")
            aug = random_augment(img)
            aug.save(out_dir / f"{breed}_trad_{count:04d}.jpg", quality=95)
            count += 1
        except Exception as e:
            print(f"  [warn] Failed on {src_path}: {e}")
            continue

    print(f"  ✓ Saved {count} images to {out_dir}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(RESULTS_DIR / "metadata.json") as f:
        meta = json.load(f)
    minority_breeds = meta["minority_breeds"]
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    for breed in minority_breeds:
        breed_paths = train_df[train_df["breed"] == breed]["path"].tolist()
        out_dir     = SYNTH_DIR / breed.replace(" ", "_")
        print(f"\n{'='*50}")
        print(f"Breed: {breed}  |  Real images: {len(breed_paths)}")
        generate_for_breed(breed, breed_paths, out_dir, N_GENERATE)

    print(f"""
╔══════════════════════════════════════════════════════╗
║  Traditional augmentation complete                  ║
║  500 images generated per minority breed            ║
║  Saved to: data/synthetic/traditional/              ║
║                                                      ║
║  Next: run 02_train_baseline.py (if not done)       ║
║  Then: run 05_train_augmented.py                    ║
╚══════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
