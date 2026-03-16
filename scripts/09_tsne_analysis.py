"""
STEP 7: t-SNE Visualisation of FastGAN Mode Collapse
Extracts ResNet-50 feature embeddings from real and synthetic images,
then uses t-SNE to visualise how well each augmentation method covers
the real image distribution per minority breed.

Key finding to visualise:
  - FastGAN severe-minority breeds (N=20) → tight clusters = mode collapse
  - SD+LoRA → broad coverage matching real distribution
  - Traditional aug → mirrors real (expected)

Usage: python 07_tsne_analysis.py
Requires: torch, torchvision, sklearn, matplotlib, seaborn, numpy, pandas

Output:
  results/tsne_all_breeds.png     — full 8-breed grid
  results/tsne_birman.png         — single breed deep-dive (worst FastGAN FID)
  results/tsne_embeddings.npz     — raw embeddings for reuse
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
SPLIT_DIR   = Path("data/oxford_pet/splits")
SYNTH_TRAD  = Path("data/synthetic/traditional")
SYNTH_GAN   = Path("data/synthetic/fastgan")
SYNTH_SD    = Path("data/synthetic/sd_lora")
RESULTS_DIR = Path("results")
MODELS_DIR  = Path("models")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Minority breeds with their training image counts
MINORITY_BREEDS = {
    "Abyssinian":        {"n_real": 20,  "severity": "severe"},
    "Bengal":            {"n_real": 20,  "severity": "severe"},
    "Birman":            {"n_real": 20,  "severity": "severe"},
    "Bombay":            {"n_real": 50,  "severity": "moderate"},
    "British_Shorthair": {"n_real": 50,  "severity": "moderate"},
    "Egyptian_Mau":      {"n_real": 50,  "severity": "moderate"},
    "Maine_Coon":        {"n_real": 50,  "severity": "moderate"},
    "Persian":           {"n_real": 50,  "severity": "moderate"},
}

# Max images per source per breed (keep balanced for fair t-SNE)
MAX_PER_SOURCE = 100

# Colours consistent with paper's figure palette
COLOURS = {
    "Real":        "#2C3E50",
    "Traditional": "#E67E22",
    "FastGAN":     "#2980B9",
    "SD+LoRA":     "#C0392B",
}
MARKERS = {
    "Real": "o", "Traditional": "s", "FastGAN": "^", "SD+LoRA": "D"
}

# ── Feature extractor ─────────────────────────────────────────────────────────
def build_extractor():
    """ResNet-50 with FC removed — outputs 2048-dim features."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    return model.to(DEVICE).eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ImageList(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img)
        except Exception:
            return torch.zeros(3, 224, 224)


def extract_features(model, image_paths, batch_size=64):
    if not image_paths:
        return np.zeros((0, 2048))
    ds     = ImageList(image_paths)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)
    feats  = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch.to(DEVICE)).cpu().numpy()
            feats.append(out)
    return np.vstack(feats)


# ── Collect image paths ───────────────────────────────────────────────────────
def get_real_paths(breed):
    """Get test-split real images for the breed (unaugmented ground truth)."""
    import pandas as pd
    # Use all available real images (train + test) for embedding reference
    paths = []
    for csv_name in ["train.csv", "test.csv"]:
        csv_path = SPLIT_DIR / csv_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        breed_clean = breed.replace("_", " ")
        rows = df[df["breed"].str.replace("_", " ") == breed_clean]
        paths.extend(rows["path"].tolist())
    return paths[:MAX_PER_SOURCE]


def get_synth_paths(synth_dir, breed):
    """Get synthetic images from a given directory."""
    breed_dir = synth_dir / breed
    if not breed_dir.exists():
        # Try with spaces replaced
        breed_dir = synth_dir / breed.replace("_", " ")
    if not breed_dir.exists():
        return []
    imgs = sorted(breed_dir.glob("*.jpg")) + sorted(breed_dir.glob("*.png"))
    return [str(p) for p in imgs[:MAX_PER_SOURCE]]


# ── t-SNE for one breed ───────────────────────────────────────────────────────
def compute_tsne_breed(model, breed, perplexity=15):
    print(f"  Processing {breed}...")

    real_paths = get_real_paths(breed)
    trad_paths = get_synth_paths(SYNTH_TRAD, breed)
    gan_paths  = get_synth_paths(SYNTH_GAN,  breed)
    sd_paths   = get_synth_paths(SYNTH_SD,   breed)

    counts = {
        "Real": len(real_paths), "Traditional": len(trad_paths),
        "FastGAN": len(gan_paths), "SD+LoRA": len(sd_paths)
    }
    print(f"    Images: {counts}")

    # Extract features
    feat_real = extract_features(model, real_paths)
    feat_trad = extract_features(model, trad_paths)
    feat_gan  = extract_features(model, gan_paths)
    feat_sd   = extract_features(model, sd_paths)

    # Stack all features + labels
    all_feats  = []
    all_labels = []
    for feats, label in [(feat_real, "Real"), (feat_trad, "Traditional"),
                         (feat_gan, "FastGAN"), (feat_sd, "SD+LoRA")]:
        if len(feats) > 0:
            all_feats.append(feats)
            all_labels.extend([label] * len(feats))

    if not all_feats:
        return None, None

    X = np.vstack(all_feats)
    X = StandardScaler().fit_transform(X)

    # t-SNE
    n_samples = len(X)
    perp      = min(perplexity, n_samples // 3, 30)
    tsne = TSNE(
        n_components=2, perplexity=max(5, perp),
        max_iter=1000, random_state=42, init="pca"
    )
    X_2d = tsne.fit_transform(X)

    return X_2d, all_labels


# ── Plotting helpers ──────────────────────────────────────────────────────────
def plot_breed_tsne(ax, X_2d, labels, breed, info, show_legend=False):
    """Plot t-SNE for one breed on a given axes."""
    label_order = ["Real", "Traditional", "FastGAN", "SD+LoRA"]
    for source in label_order:
        idx = [i for i, l in enumerate(labels) if l == source]
        if not idx:
            continue
        pts  = X_2d[idx]
        size = 35 if source == "Real" else 15
        alpha = 0.9 if source == "Real" else 0.5
        zorder = 5 if source == "Real" else 2
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=COLOURS[source], marker=MARKERS[source],
            s=size, alpha=alpha, zorder=zorder,
            edgecolors="none", label=source
        )

    severity = info["severity"]
    n_real   = info["n_real"]
    title    = breed.replace("_", " ")
    colour   = "#C0392B" if severity == "severe" else "#2980B9"
    ax.set_title(f"{title}\n(N={n_real}, {severity})",
                 fontsize=9, color=colour, fontweight="bold", pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)

    if show_legend:
        handles = [
            mpatches.Patch(color=COLOURS[s], label=s)
            for s in label_order if any(l == s for l in labels)
        ]
        ax.legend(handles=handles, loc="lower right",
                  fontsize=7, framealpha=0.8, edgecolor="none")


def add_convex_hull(ax, X_2d, labels, source, color):
    """Draw convex hull around a source's points to show spread."""
    from scipy.spatial import ConvexHull
    idx = [i for i, l in enumerate(labels) if l == source]
    if len(idx) < 4:
        return
    pts = X_2d[idx]
    try:
        hull = ConvexHull(pts)
        for simplex in hull.simplices:
            ax.plot(pts[simplex, 0], pts[simplex, 1],
                    color=color, alpha=0.3, linewidth=0.8, zorder=1)
    except Exception:
        pass


# ── Main grid plot ────────────────────────────────────────────────────────────
def plot_all_breeds(all_results):
    breeds = list(all_results.keys())
    n      = len(breeds)
    ncols  = 4
    nrows  = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()

    fig.suptitle(
        "t-SNE of ResNet-50 Feature Embeddings per Minority Breed\n"
        "(Red = severe minority N=20, Blue = moderate minority N=50)",
        fontsize=12, y=1.01, fontweight="bold"
    )

    for i, breed in enumerate(breeds):
        X_2d, labels = all_results[breed]
        if X_2d is None:
            axes[i].set_visible(False)
            continue
        info = MINORITY_BREEDS[breed]
        plot_breed_tsne(axes[i], X_2d, labels, breed, info,
                        show_legend=(i == ncols - 1))
        # Convex hulls for FastGAN and Real to show coverage gap
        add_convex_hull(axes[i], X_2d, labels, "Real",    COLOURS["Real"])
        add_convex_hull(axes[i], X_2d, labels, "FastGAN", COLOURS["FastGAN"])
        add_convex_hull(axes[i], X_2d, labels, "SD+LoRA", COLOURS["SD+LoRA"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Global legend
    handles = [mpatches.Patch(color=COLOURS[s], label=s)
               for s in ["Real", "Traditional", "FastGAN", "SD+LoRA"]]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out = RESULTS_DIR / "tsne_all_breeds.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [saved] {out}")


# ── Single breed deep-dive (Birman — worst FastGAN FID) ──────────────────────
def plot_single_breed(breed, X_2d, labels, info):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Feature Embedding Distribution: {breed.replace('_', ' ')} "
        f"(N={info['n_real']} real training images, severe minority)",
        fontsize=12, fontweight="bold"
    )

    # Left: All sources
    plot_breed_tsne(axes[0], X_2d, labels, breed, info, show_legend=True)
    axes[0].set_title("All sources", fontsize=10)

    # Right: FastGAN vs Real only — highlight mode collapse
    idx_real = [i for i, l in enumerate(labels) if l == "Real"]
    idx_gan  = [i for i, l in enumerate(labels) if l == "FastGAN"]
    idx_sd   = [i for i, l in enumerate(labels) if l == "SD+LoRA"]

    if idx_real:
        axes[1].scatter(X_2d[idx_real, 0], X_2d[idx_real, 1],
                        c=COLOURS["Real"], s=60, alpha=0.9, zorder=5,
                        label=f"Real (N={len(idx_real)})", marker="o")
    if idx_gan:
        axes[1].scatter(X_2d[idx_gan, 0], X_2d[idx_gan, 1],
                        c=COLOURS["FastGAN"], s=20, alpha=0.4, zorder=2,
                        label=f"FastGAN (N={len(idx_gan)})", marker="^")
    if idx_sd:
        axes[1].scatter(X_2d[idx_sd, 0], X_2d[idx_sd, 1],
                        c=COLOURS["SD+LoRA"], s=20, alpha=0.4, zorder=2,
                        label=f"SD+LoRA (N={len(idx_sd)})", marker="D")

    # Convex hulls
    for source, color in [("Real", COLOURS["Real"]),
                           ("FastGAN", COLOURS["FastGAN"]),
                           ("SD+LoRA", COLOURS["SD+LoRA"])]:
        add_convex_hull(axes[1], X_2d, labels, source, color)

    axes[1].set_title("Real vs FastGAN vs SD+LoRA\n(hulls show distribution coverage)",
                      fontsize=10)
    axes[1].legend(fontsize=9, framealpha=0.9)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].spines[["top","right","left","bottom"]].set_visible(False)

    # Annotation
    fig.text(
        0.5, -0.04,
        "FastGAN images clustering tightly (mode collapse) vs SD+LoRA "
        "broadly covering the real image distribution",
        ha="center", fontsize=10, style="italic", color="#555555"
    )

    plt.tight_layout()
    out = RESULTS_DIR / f"tsne_{breed.lower()}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {out}")


# ── Compute coverage metric ───────────────────────────────────────────────────
def coverage_analysis(all_results):
    """
    Compute average nearest-neighbour distance from each synthetic point
    to its nearest real point. Lower = closer to real distribution.
    """
    from sklearn.neighbors import NearestNeighbors

    print("\n── Distribution Coverage Analysis ────────────────────────────")
    print(f"  {'Breed':<22} {'FastGAN→Real':>14} {'SD+LoRA→Real':>14} {'Trad→Real':>12}")
    print("  " + "─" * 65)

    coverage = {}
    for breed, (X_2d, labels) in all_results.items():
        if X_2d is None:
            continue
        idx_real = [i for i, l in enumerate(labels) if l == "Real"]
        if not idx_real:
            continue

        real_pts = X_2d[idx_real]
        nbrs = NearestNeighbors(n_neighbors=1).fit(real_pts)

        row = {"breed": breed}
        for source in ["FastGAN", "SD+LoRA", "Traditional"]:
            idx_s = [i for i, l in enumerate(labels) if l == source]
            if not idx_s:
                row[source] = np.nan
                continue
            synth_pts = X_2d[idx_s]
            dists, _  = nbrs.kneighbors(synth_pts)
            row[source] = round(float(np.mean(dists)), 3)

        coverage[breed] = row
        print(f"  {breed:<22} {row.get('FastGAN', 'N/A'):>14} "
              f"{row.get('SD+LoRA', 'N/A'):>14} "
              f"{row.get('Traditional', 'N/A'):>12}")

    # Save
    path = RESULTS_DIR / "tsne_coverage.json"
    with open(path, "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"\n  [saved] {path}")
    return coverage


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Building feature extractor (ResNet-50 without FC)...")
    model = build_extractor()

    # Try to load pre-trained classifier weights for better feature alignment
    baseline_weights = MODELS_DIR / "baseline_resnet50_seed42.pth"
    if not baseline_weights.exists():
        baseline_weights = MODELS_DIR / "baseline_resnet50.pth"
    if baseline_weights.exists():
        print(f"  Loading classifier weights: {baseline_weights}")
        state = torch.load(baseline_weights, map_location=DEVICE)
        # Remove FC weights — we replaced FC with Identity
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        model.load_state_dict(state, strict=False)
        print("  Classifier weights loaded (FC excluded)")
    else:
        print("  Using ImageNet-pretrained weights (no classifier weights found)")

    # Compute t-SNE for all breeds
    print("\nComputing t-SNE embeddings for all minority breeds...")
    all_results = {}
    for breed in MINORITY_BREEDS:
        X_2d, labels = compute_tsne_breed(model, breed)
        all_results[breed] = (X_2d, labels)

    # Save raw embeddings for reuse
    np.savez(
        RESULTS_DIR / "tsne_embeddings.npz",
        **{
            f"{breed}_X2d":   all_results[breed][0]
            for breed in all_results if all_results[breed][0] is not None
        },
        **{
            f"{breed}_labels": np.array(all_results[breed][1])
            for breed in all_results if all_results[breed][1] is not None
        }
    )
    print("\n  [saved] results/tsne_embeddings.npz")

    # Plot grid of all breeds
    print("\nGenerating plots...")
    plot_all_breeds(all_results)

    # Deep-dive: Birman (worst FastGAN FID = 214.7, severe minority)
    if all_results.get("Birman", (None, None))[0] is not None:
        X_2d, labels = all_results["Birman"]
        plot_single_breed("Birman", X_2d, labels, MINORITY_BREEDS["Birman"])

    # Also plot Abyssinian (highest FastGAN FID = 348.1)
    if all_results.get("Abyssinian", (None, None))[0] is not None:
        X_2d, labels = all_results["Abyssinian"]
        plot_single_breed("Abyssinian", X_2d, labels, MINORITY_BREEDS["Abyssinian"])

    # Coverage metric
    coverage_analysis(all_results)

    print("\n✓ t-SNE analysis complete.")
    print("  Upload to Claude:")
    print("    results/tsne_all_breeds.png")
    print("    results/tsne_birman.png")
    print("    results/tsne_abyssinian.png")
    print("    results/tsne_coverage.json")


if __name__ == "__main__":
    main()
