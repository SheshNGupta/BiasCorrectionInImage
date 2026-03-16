"""
STEP 5: Train Augmented Classifiers for All Conditions
Conditions: (1) Traditional aug  (2) FastGAN aug  (3) SD+LoRA aug  (4) Hybrid aug
Re-uses the same ResNet-50 architecture and training config as baseline.

Requirements: same as Step 2
"""

import json
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed (42, 123, or 456 for the three runs)")
args, _ = parser.parse_known_args()

# ── Config (must match Step 2) ────────────────────────────────────────────────
SPLIT_DIR   = Path("data/oxford_pet/splits")
SYNTH_TRAD  = Path("data/synthetic/traditional")
SYNTH_GAN   = Path("data/synthetic/fastgan")
SYNTH_SD    = Path("data/synthetic/sd_lora")
RESULTS_DIR = Path("results")
MODELS_DIR  = Path("models")
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-4
N_SYNTHETIC = 500         # per minority breed per condition
RANDOM_SEED = args.seed
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*60}")
print(f"  RANDOM SEED: {RANDOM_SEED}")
print(f"{'='*60}\n")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────
class PetDataset(Dataset):
    def __init__(self, csv_path, label_map, transform, extra_rows=None):
        self.df = pd.read_csv(csv_path)
        if extra_rows is not None:
            self.df = pd.concat([self.df, extra_rows], ignore_index=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), self.label_map[row["breed"]]

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Build extra rows DataFrame from synthetic images ──────────────────────────
def build_synthetic_rows(synth_dir: Path, minority_breeds: list,
                         n_per_class: int = None) -> pd.DataFrame:
    """Build dataframe of synthetic images.
    If n_per_class is None, uses ALL images found in each breed folder.
    """
    rows = []
    for breed in minority_breeds:
        breed_dir = synth_dir / breed.replace(" ", "_")
        if not breed_dir.exists():
            print(f"  [WARN] Synthetic dir not found: {breed_dir}")
            continue
        imgs = sorted(breed_dir.glob("*.jpg"))
        if n_per_class is not None:
            imgs = imgs[:n_per_class]
        print(f"    {breed}: using {len(imgs)} synthetic images from {synth_dir.name}")
        for p in imgs:
            rows.append({"path": str(p), "breed": breed})
    return pd.DataFrame(rows)

def build_hybrid_rows(minority_breeds: list) -> pd.DataFrame:
    """Split available images evenly between GAN and SD per breed.
    Uses min(available_gan, available_sd) // 2 from each source per breed.
    """
    rows = []
    for breed in minority_breeds:
        safe      = breed.replace(" ", "_")
        gan_imgs  = sorted((SYNTH_GAN  / safe).glob("*.jpg")) if (SYNTH_GAN  / safe).exists() else []
        sd_imgs   = sorted((SYNTH_SD   / safe).glob("*.jpg")) if (SYNTH_SD   / safe).exists() else []
        half      = min(len(gan_imgs), len(sd_imgs)) // 2
        selected  = [(p, breed) for p in gan_imgs[:half]] + [(p, breed) for p in sd_imgs[:half]]
        print(f"    {breed}: hybrid using {half} GAN + {half} SD = {half*2} images")
        for p, b in selected:
            rows.append({"path": str(p), "breed": b})
    return pd.DataFrame(rows)

# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(n_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m.to(DEVICE)

# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, label_map):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(DEVICE)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    idx2breed   = {v: k for k, v in label_map.items()}
    target_names = [idx2breed[i] for i in range(len(label_map))]
    macro_f1    = f1_score(all_labels, all_preds, average="macro")
    report      = classification_report(all_labels, all_preds,
                                        target_names=target_names, output_dict=True)
    return macro_f1, report

def save_metrics(report, macro_f1, condition, minority_breeds, gpu_mins):
    breeds = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    min_accs = [report[b]["recall"] for b in breeds if b     in minority_breeds]
    maj_accs = [report[b]["recall"] for b in breeds if b not in minority_breeds]
    summary  = {
        "condition":             condition,
        "seed":                  RANDOM_SEED,
        "macro_f1":              round(macro_f1, 4),
        "minority_avg_accuracy": round(np.mean(min_accs), 4),
        "majority_avg_accuracy": round(np.mean(maj_accs), 4),
        "bias_gap":              round(np.mean(maj_accs) - np.mean(min_accs), 4),
        "gpu_minutes":           round(gpu_mins, 1),
        "per_class_accuracy":    {b: round(report[b]["recall"], 4) for b in breeds},
    }

    # Per-seed file (safe to overwrite — one file per condition per seed)
    seed_path = RESULTS_DIR / f"metrics_{condition}_seed{RANDOM_SEED}.json"
    with open(seed_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] {seed_path}")

    # Aggregate file: append this run to all_runs_{condition}.json
    agg_path = RESULTS_DIR / f"all_runs_{condition}.json"
    if agg_path.exists():
        with open(agg_path) as f:
            all_runs = json.load(f)
    else:
        all_runs = []
    # Remove any previous entry for this seed so re-runs overwrite cleanly
    all_runs = [r for r in all_runs if r.get("seed") != RANDOM_SEED]
    all_runs.append(summary)
    with open(agg_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"  [updated] {agg_path} ({len(all_runs)} seed(s) so far)")

    return summary

def plot_per_class_accuracy(report, label, minority_breeds, out_path):
    breeds = [k for k in report if k not in ("accuracy", "macro avg", "weighted avg")]
    accs   = [report[b]["recall"] for b in breeds]
    colors = ["#C00000" if b in minority_breeds else "#2E75B6" for b in breeds]
    order  = np.argsort(accs)
    plt.figure(figsize=(16, 6))
    plt.bar([breeds[i] for i in order], [accs[i] for i in order],
            color=[colors[i] for i in order])
    plt.axhline(np.mean(accs), color="orange", linestyle="--",
                label=f"Mean={np.mean(accs):.2f}")
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Per-class Accuracy"); plt.title(f"Per-class Accuracy — {label}")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=150); plt.close()

# ── Run one condition ─────────────────────────────────────────────────────────
def run_condition(condition: str, extra_rows, label_map, minority_breeds):
    n_classes = len(label_map)
    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    if extra_rows is not None:
        print(f"Extra synthetic rows: {len(extra_rows)}")
    print(f"{'='*60}")

    train_ds = PetDataset(SPLIT_DIR / "train.csv", label_map, train_tf, extra_rows)
    test_ds  = PetDataset(SPLIT_DIR / "test.csv",  label_map, val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)  # num_workers=0 is faster on Windows
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)  # num_workers=0 is faster on Windows

    model     = build_model(n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        if epoch % 10 == 0:
            f1, _ = evaluate(model, test_loader, label_map)
            print(f"  Epoch {epoch}/{EPOCHS} | val_macro_f1={f1:.3f}")

    gpu_mins = (time.time() - t0) / 60
    macro_f1, report = evaluate(model, test_loader, label_map)
    print(f"\nFinal {condition} Macro F1: {macro_f1:.4f}")

    torch.save(model.state_dict(), MODELS_DIR / f"resnet50_{condition}_seed{RANDOM_SEED}.pth")
    plot_per_class_accuracy(report, condition, minority_breeds,
                            RESULTS_DIR / f"accuracy_{condition}_seed{RANDOM_SEED}.png")
    return save_metrics(report, macro_f1, condition, minority_breeds, gpu_mins)

# ── One-time migration: seed existing seed-42 results into all_runs files ────
def migrate_seed42_results():
    """
    If all_runs_{condition}.json doesn't exist yet, bootstrap it from the
    original metrics_{condition}.json files (which were produced with seed 42).
    Safe to call every run — skips conditions that are already migrated.
    """
    condition_map = {
        "traditional": "metrics_traditional.json",
        "fastgan":     "metrics_fastgan.json",
        "sd_lora":     "metrics_sd_lora.json",
        "hybrid":      "metrics_hybrid.json",
    }
    for condition, filename in condition_map.items():
        agg_path = RESULTS_DIR / f"all_runs_{condition}.json"
        src_path = RESULTS_DIR / filename

        # Skip if aggregate already has a seed-42 entry
        if agg_path.exists():
            with open(agg_path) as f:
                existing = json.load(f)
            if any(r.get("seed") == 42 for r in existing):
                continue  # already migrated

        if not src_path.exists():
            print(f"  [migrate] {src_path} not found, skipping.")
            continue

        with open(src_path) as f:
            original = json.load(f)

        # Inject seed=42 if not already present
        original["seed"] = original.get("seed", 42)

        all_runs = []
        if agg_path.exists():
            with open(agg_path) as f:
                all_runs = json.load(f)
        all_runs.append(original)

        with open(agg_path, "w") as f:
            json.dump(all_runs, f, indent=2)
        print(f"  [migrate] Seeded {agg_path} with existing seed-42 results from {filename}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Migrate existing seed-42 results into aggregate files (safe no-op if already done)
    migrate_seed42_results()

    with open(RESULTS_DIR / "metadata.json") as f:
        meta = json.load(f)
    minority_breeds = meta["minority_breeds"]

    with open(RESULTS_DIR / "label_map.json") as f:
        label_map = json.load(f)

    results = []

    # Condition 2: Traditional augmentation
    trad_rows = build_synthetic_rows(SYNTH_TRAD, minority_breeds)  # uses all available
    r2 = run_condition("traditional", trad_rows, label_map, minority_breeds)
    results.append(r2)

    # Condition 3: FastGAN augmented
    gan_rows = build_synthetic_rows(SYNTH_GAN, minority_breeds)   # uses all available
    r3 = run_condition("fastgan", gan_rows, label_map, minority_breeds)
    results.append(r3)

    # Condition 4: SD+LoRA augmented
    sd_rows = build_synthetic_rows(SYNTH_SD, minority_breeds)     # uses all available
    r4 = run_condition("sd_lora", sd_rows, label_map, minority_breeds)
    results.append(r4)

    # Condition 5: Hybrid (GAN + SD)
    hybrid_rows = build_hybrid_rows(minority_breeds)              # splits available evenly
    r5 = run_condition("hybrid", hybrid_rows, label_map, minority_breeds)
    results.append(r5)

    # Save per-seed summary
    seed_summary_path = RESULTS_DIR / f"augmented_results_seed{RANDOM_SEED}.json"
    with open(seed_summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {seed_summary_path}")

    # Print mean ± SD across seeds seen so far for each condition
    print("\n── Multi-seed summary (seeds completed so far) ──")
    for cond in ["traditional", "fastgan", "sd_lora", "hybrid"]:
        agg_path = RESULTS_DIR / f"all_runs_{cond}.json"
        if not agg_path.exists():
            continue
        with open(agg_path) as f:
            runs = json.load(f)
        if len(runs) < 2:
            continue
        f1s  = [r["macro_f1"]              for r in runs]
        mins = [r["minority_avg_accuracy"] for r in runs]
        gaps = [r["bias_gap"]              for r in runs]
        print(f"  {cond:12s} | F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f} "
              f"| MinAcc: {np.mean(mins):.4f} ± {np.std(mins):.4f} "
              f"| Gap: {np.mean(gaps):.4f} ± {np.std(gaps):.4f}  "
              f"[n={len(runs)} seeds]")

    print("\n✓ All augmented conditions trained and evaluated.")


if __name__ == "__main__":
    main()
