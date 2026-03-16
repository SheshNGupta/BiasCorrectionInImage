"""
STEP 1: Dataset Preparation & Bias Analysis
Oxford-IIIT Pet Dataset — download, split, identify minority breeds.

Requirements:
    pip install torch torchvision matplotlib seaborn pandas requests
"""

import os
import shutil
import tarfile
import requests
import random
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data/oxford_pet")
IMAGES_DIR    = DATA_DIR / "images"
SPLIT_DIR     = DATA_DIR / "splits"
RESULTS_DIR   = Path("results")
RANDOM_SEED   = 42
TRAIN_RATIO   = 0.80
N_MINORITY    = 5          # number of minority breeds to augment

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

IMAGES_URL    = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTS_URL    = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

# ── Download ──────────────────────────────────────────────────────────────────
def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  Downloading {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"  Saved to {dest}")

def extract(archive: Path, dest: Path):
    if dest.exists():
        print(f"  [skip] {dest} already extracted")
        return
    print(f"  Extracting {archive.name} ...")
    with tarfile.open(archive) as t:
        t.extractall(dest)
    print(f"  Extracted to {dest}")

# ── Parse breed labels ────────────────────────────────────────────────────────
def get_breed(filename: str) -> str:
    """
    Oxford-IIIT filenames are like: Abyssinian_100.jpg
    Breed = everything before the last underscore+number.
    """
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    return parts[0]  # e.g. "Abyssinian" or "american_bulldog"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Download
    images_archive = DATA_DIR / "images.tar.gz"
    annots_archive = DATA_DIR / "annotations.tar.gz"
    download_file(IMAGES_URL, images_archive)
    download_file(ANNOTS_URL, annots_archive)
    extract(images_archive, IMAGES_DIR)
    extract(annots_archive, DATA_DIR / "annotations")

    # 2. Collect all images
    all_images = sorted([
        f for f in (IMAGES_DIR / "images").glob("*.jpg")
    ])
    print(f"\nTotal images found: {len(all_images)}")

    # 3. Count per breed
    breed_counts = Counter(get_breed(f.name) for f in all_images)
    df = pd.DataFrame(breed_counts.items(), columns=["breed", "count"])
    df = df.sort_values("count").reset_index(drop=True)
    print(f"\nBreed count summary:\n{df.describe()}")

    # 4. Identify minority and majority breeds
    minority_breeds = df.head(N_MINORITY)["breed"].tolist()
    majority_breeds = df.tail(5)["breed"].tolist()
    print(f"\nMinority breeds (will be augmented): {minority_breeds}")
    print(f"Majority breeds (reference): {majority_breeds}")

    # 5. Visualise distribution
    plt.figure(figsize=(16, 6))
    colors = ["#C00000" if b in minority_breeds else "#2E75B6" for b in df["breed"]]
    plt.bar(df["breed"], df["count"], color=colors)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Image Count")
    plt.title("Oxford-IIIT Pet — Per-Breed Image Count\n(Red = minority breeds selected for augmentation)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "breed_distribution.png", dpi=150)
    plt.close()
    print(f"\nSaved breed distribution chart to results/breed_distribution.png")

    # 6. Build train/test splits (stratified per breed)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_files, test_files = [], []

    for breed in breed_counts:
        breed_imgs = [f for f in all_images if get_breed(f.name) == breed]
        random.shuffle(breed_imgs)
        n_train = max(1, int(len(breed_imgs) * TRAIN_RATIO))
        train_files.extend([(str(f), breed) for f in breed_imgs[:n_train]])
        test_files.extend([(str(f), breed) for f in breed_imgs[n_train:]])

    print(f"\nTrain split: {len(train_files)} images")
    print(f"Test  split: {len(test_files)}  images")

    # Save splits as CSV
    pd.DataFrame(train_files, columns=["path", "breed"]).to_csv(
        SPLIT_DIR / "train.csv", index=False)
    pd.DataFrame(test_files,  columns=["path", "breed"]).to_csv(
        SPLIT_DIR / "test.csv",  index=False)

    # 7. Save metadata
    metadata = {
        "minority_breeds": minority_breeds,
        "majority_breeds": majority_breeds,
        "breed_counts": breed_counts,
        "n_train": len(train_files),
        "n_test":  len(test_files),
        "random_seed": RANDOM_SEED,
    }
    with open(RESULTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n✓ Dataset preparation complete.")
    print(f"  Splits saved to: {SPLIT_DIR}")
    print(f"  Metadata saved to: {RESULTS_DIR / 'metadata.json'}")


if __name__ == "__main__":
    main()
