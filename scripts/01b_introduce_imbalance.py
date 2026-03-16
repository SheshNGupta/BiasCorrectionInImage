"""
STEP 1b: Introduce Artificial Class Imbalance
=============================================
The Oxford-IIIT Pet dataset is naturally well-balanced (152-160 images
per breed, ratio 1.1x). This script creates a realistic imbalanced
training split by subsampling selected minority breeds.

This is standard practice in bias correction research. The test set
is NEVER modified — only the training split is subsampled.

Imbalance structure:
  - Severe minority  : 3 breeds → 20 training images each
  - Moderate minority: 5 breeds → 50 training images each
  - Majority         : 29 breeds → all images kept (~155 each)

Resulting max/min ratio: ~8x  (realistic for real-world datasets)

Requirements:
    pip install pandas matplotlib seaborn
"""

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SPLIT_DIR   = Path("data/oxford_pet/splits")
RESULTS_DIR = Path("results")
RANDOM_SEED = 42

# Imbalance targets
SEVERE_COUNT   = 20    # 3 breeds reduced to this
MODERATE_COUNT = 50    # 5 breeds reduced to this

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load existing balanced train split
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    test_df  = pd.read_csv(SPLIT_DIR / "test.csv")

    # Get all breeds sorted by name for deterministic selection
    all_breeds = sorted(train_df["breed"].unique())
    print(f"Total breeds: {len(all_breeds)}")
    print(f"Current train size: {len(train_df)}")

    # ── Select minority breeds deterministically ──────────────────────────────
    # Pick first 3 alphabetically as severe minority
    # Pick next 5 alphabetically as moderate minority
    # This is deterministic and reproducible
    severe_breeds   = all_breeds[:3]
    moderate_breeds = all_breeds[3:8]
    majority_breeds = all_breeds[8:]

    print(f"\nSevere minority breeds   ({SEVERE_COUNT} images each): {severe_breeds}")
    print(f"Moderate minority breeds ({MODERATE_COUNT} images each): {moderate_breeds}")
    print(f"Majority breeds          (all images kept): {len(majority_breeds)} breeds")

    # ── Build imbalanced training split ───────────────────────────────────────
    imbalanced_rows = []

    for breed in all_breeds:
        breed_rows = train_df[train_df["breed"] == breed].copy()
        breed_rows = breed_rows.sample(frac=1, random_state=RANDOM_SEED)  # shuffle

        if breed in severe_breeds:
            kept = breed_rows.head(SEVERE_COUNT)
        elif breed in moderate_breeds:
            kept = breed_rows.head(MODERATE_COUNT)
        else:
            kept = breed_rows   # keep all majority images

        imbalanced_rows.append(kept)

    imbalanced_df = pd.concat(imbalanced_rows, ignore_index=True)
    imbalanced_df = imbalanced_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\nImbalanced train size: {len(imbalanced_df)}  (was {len(train_df)})")
    print(f"Test set unchanged:    {len(test_df)} images")

    # ── Verify counts ─────────────────────────────────────────────────────────
    counts = imbalanced_df.groupby("breed").size().sort_values()
    print(f"\nPer-breed count summary:")
    print(f"  Min  : {counts.min()} images  ({counts.idxmin()})")
    print(f"  Max  : {counts.max()} images  ({counts.idxmax()})")
    print(f"  Mean : {counts.mean():.1f} images")
    print(f"  Ratio: {counts.max() / counts.min():.1f}x")
    print(f"\nFull breed count list (sorted):")
    print(counts.to_string())

    # ── Save imbalanced split (overwrites original train.csv) ─────────────────
    # Back up original first
    original_backup = SPLIT_DIR / "train_balanced_original.csv"
    if not original_backup.exists():
        train_df.to_csv(original_backup, index=False)
        print(f"\n[backup] Original balanced split saved to {original_backup}")

    imbalanced_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    print(f"[saved] Imbalanced train split saved to {SPLIT_DIR / 'train.csv'}")

    # ── Update metadata.json ──────────────────────────────────────────────────
    minority_breeds = list(severe_breeds) + list(moderate_breeds)

    metadata = {
        "severe_minority_breeds":   list(severe_breeds),
        "moderate_minority_breeds": list(moderate_breeds),
        "minority_breeds":          minority_breeds,
        "majority_breeds":          list(majority_breeds),
        "severe_count":             SEVERE_COUNT,
        "moderate_count":           MODERATE_COUNT,
        "breed_counts":             counts.to_dict(),
        "n_train":                  len(imbalanced_df),
        "n_test":                   len(test_df),
        "imbalance_ratio":          round(counts.max() / counts.min(), 1),
        "random_seed":              RANDOM_SEED,
    }

    with open(RESULTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[saved] metadata.json updated")

    # ── Plot imbalanced distribution ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Before
    original_counts = train_df.groupby("breed").size().sort_values()
    colors_before   = ["#2E75B6"] * len(original_counts)
    axes[0].bar(original_counts.index, original_counts.values, color=colors_before)
    axes[0].set_title("Before: Balanced Split\n(ratio 1.1x)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Training Images")
    axes[0].tick_params(axis='x', rotation=90, labelsize=6)
    axes[0].axhline(original_counts.mean(), color="orange", linestyle="--",
                    label=f"Mean={original_counts.mean():.0f}")
    axes[0].legend()

    # After
    colors_after = []
    for b in counts.index:
        if b in severe_breeds:
            colors_after.append("#C00000")
        elif b in moderate_breeds:
            colors_after.append("#E97132")
        else:
            colors_after.append("#2E75B6")

    axes[1].bar(counts.index, counts.values, color=colors_after)
    axes[1].set_title("After: Artificially Imbalanced Split\n(ratio ~8x)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Training Images")
    axes[1].tick_params(axis='x', rotation=90, labelsize=6)
    axes[1].axhline(counts.mean(), color="orange", linestyle="--",
                    label=f"Mean={counts.mean():.0f}")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C00000", label=f"Severe minority ({SEVERE_COUNT} imgs)"),
        Patch(facecolor="#E97132", label=f"Moderate minority ({MODERATE_COUNT} imgs)"),
        Patch(facecolor="#2E75B6", label="Majority (all imgs)"),
    ]
    axes[1].legend(handles=legend_elements, fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "breed_distribution_imbalanced.png", dpi=150)
    plt.close()
    print(f"[saved] Distribution plot saved to results/breed_distribution_imbalanced.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Imbalance successfully introduced                          ║
║                                                              ║
║  Severe minority  ({SEVERE_COUNT:3d} imgs): {', '.join(severe_breeds[:2])}...
║  Moderate minority ({MODERATE_COUNT:3d} imgs): {', '.join(moderate_breeds[:2])}...
║  Imbalance ratio : {counts.max() / counts.min():.1f}x                              ║
║  Train set size  : {len(imbalanced_df)} (was {len(train_df)})              ║
║  Test set        : {len(test_df)} (UNCHANGED)                    ║
║                                                              ║
║  Next step: run 02_train_baseline.py                        ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
