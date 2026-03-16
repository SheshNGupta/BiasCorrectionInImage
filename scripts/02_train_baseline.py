"""
STEP 2: Train Baseline ResNet-50 Classifier (Real Data Only)
Records per-class accuracy and identifies the bias gap.

Requirements:
    pip install torch torchvision scikit-learn pandas matplotlib seaborn
"""

import os
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed (42, 123, or 456 for the three runs)")
args, _ = parser.parse_known_args()

# ── Config ────────────────────────────────────────────────────────────────────
SPLIT_DIR    = Path("data/oxford_pet/splits")
RESULTS_DIR  = Path("results")
MODELS_DIR   = Path("models")
BATCH_SIZE   = 32
EPOCHS       = 50
LR           = 1e-4
RANDOM_SEED  = args.seed
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'='*60}")
print(f"  RANDOM SEED: {RANDOM_SEED}")
print(f"{'='*60}\n")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# ── Dataset ───────────────────────────────────────────────────────────────────
class PetDataset(Dataset):
    def __init__(self, csv_path, label_map, transform=None, extra_rows=None):
        self.df = pd.read_csv(csv_path)
        if extra_rows is not None:            # for augmented conditions
            self.df = pd.concat([self.df, extra_rows], ignore_index=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label_map[row["breed"]]
        return img, label

# ── Transforms ────────────────────────────────────────────────────────────────
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

# ── Build model ───────────────────────────────────────────────────────────────
def build_model(n_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(DEVICE)

# ── Training loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, label_map):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    idx_to_breed = {v: k for k, v in label_map.items()}
    target_names = [idx_to_breed[i] for i in range(len(label_map))]

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report   = classification_report(all_labels, all_preds,
                                     target_names=target_names, output_dict=True)
    return macro_f1, report, all_labels, all_preds

# ── Visualisations ────────────────────────────────────────────────────────────
def plot_per_class_accuracy(report, label, minority_breeds, out_path):
    breeds = [k for k in report if k not in ("accuracy","macro avg","weighted avg")]
    accs   = [report[b]["recall"] for b in breeds]
    colors = ["#C00000" if b in minority_breeds else "#2E75B6" for b in breeds]

    order  = np.argsort(accs)
    breeds = [breeds[i] for i in order]
    accs   = [accs[i]   for i in order]
    colors = [colors[i] for i in order]

    plt.figure(figsize=(16, 6))
    plt.bar(breeds, accs, color=colors)
    plt.axhline(np.mean(accs), color="orange", linestyle="--", label=f"Mean={np.mean(accs):.2f}")
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("Per-class Accuracy (Recall)")
    plt.title(f"Per-class Accuracy — {label}\n(Red = minority breeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_metrics(report, macro_f1, condition, minority_breeds, gpu_mins, out_dir):
    breeds = [k for k in report if k not in ("accuracy","macro avg","weighted avg")]
    minority_accs = [report[b]["recall"] for b in breeds if b in minority_breeds]
    majority_accs = [report[b]["recall"] for b in breeds if b not in minority_breeds]

    summary = {
        "condition":              condition,
        "seed":                   RANDOM_SEED,
        "macro_f1":               round(macro_f1, 4),
        "minority_avg_accuracy":  round(np.mean(minority_accs), 4),
        "majority_avg_accuracy":  round(np.mean(majority_accs), 4),
        "bias_gap":               round(np.mean(majority_accs) - np.mean(minority_accs), 4),
        "gpu_minutes":            round(gpu_mins, 1),
        "per_class_accuracy":     {b: round(report[b]["recall"], 4) for b in breeds},
    }

    # Per-seed file (safe to overwrite)
    seed_path = out_dir / f"metrics_{condition}_seed{RANDOM_SEED}.json"
    with open(seed_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {seed_path}")

    # Aggregate file: append/update this seed
    agg_path = out_dir / f"all_runs_{condition}.json"
    if agg_path.exists():
        with open(agg_path) as f:
            all_runs = json.load(f)
    else:
        all_runs = []
    all_runs = [r for r in all_runs if r.get("seed") != RANDOM_SEED]
    all_runs.append(summary)
    with open(agg_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"[Updated] {agg_path} ({len(all_runs)} seed(s) so far)")

    return summary

# ── One-time migration: seed existing seed-42 result into all_runs file ─────
def migrate_seed42_results():
    """
    Bootstrap all_runs_baseline.json from the original metrics_baseline.json
    (produced with seed 42). Safe to call every run — skips if already done.
    """
    agg_path = RESULTS_DIR / "all_runs_baseline.json"
    src_path = RESULTS_DIR / "metrics_baseline.json"

    if agg_path.exists():
        with open(agg_path) as f:
            existing = json.load(f)
        if any(r.get("seed") == 42 for r in existing):
            return  # already migrated

    if not src_path.exists():
        print(f"  [migrate] {src_path} not found, skipping.")
        return

    with open(src_path) as f:
        original = json.load(f)

    original["seed"] = original.get("seed", 42)

    all_runs = []
    if agg_path.exists():
        with open(agg_path) as f:
            all_runs = json.load(f)
    all_runs.append(original)

    with open(agg_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"  [migrate] Seeded {agg_path} with existing seed-42 results from metrics_baseline.json")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # Migrate existing seed-42 results (safe no-op if already done)
    migrate_seed42_results()

    # Load metadata
    with open(RESULTS_DIR / "metadata.json") as f:
        meta = json.load(f)
    minority_breeds = meta["minority_breeds"]

    # Build label map
    train_df = pd.read_csv(SPLIT_DIR / "train.csv")
    breeds   = sorted(train_df["breed"].unique())
    label_map = {b: i for i, b in enumerate(breeds)}

    with open(RESULTS_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    n_classes = len(breeds)
    print(f"Number of classes: {n_classes}")
    print(f"Device: {DEVICE}")

    # Datasets & loaders
    train_ds = PetDataset(SPLIT_DIR / "train.csv", label_map, train_tf)
    test_ds  = PetDataset(SPLIT_DIR / "test.csv",  label_map, val_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model, criterion, optimiser
    model     = build_model(n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    t0 = time.time()
    history = []
    print("\nTraining baseline model...")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        if epoch % 10 == 0:
            val_f1, val_report, _, _ = evaluate(model, test_loader, label_map)
            print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={tr_loss:.4f} | train_acc={tr_acc:.3f} | val_macro_f1={val_f1:.3f}")
            history.append({"epoch": epoch, "loss": tr_loss, "train_acc": tr_acc, "val_f1": val_f1})

    gpu_mins = (time.time() - t0) / 60

    # Final evaluation
    macro_f1, report, labels, preds = evaluate(model, test_loader, label_map)
    print(f"\nFinal Baseline Macro F1: {macro_f1:.4f}")

    # Save model
    torch.save(model.state_dict(), MODELS_DIR / f"baseline_resnet50_seed{RANDOM_SEED}.pth")

    # Plots
    plot_per_class_accuracy(
        report, "Baseline (real data only)", minority_breeds,
        RESULTS_DIR / f"accuracy_baseline_seed{RANDOM_SEED}.png"
    )

    # Save metrics
    save_metrics(report, macro_f1, "baseline", minority_breeds, gpu_mins, RESULTS_DIR)

    # Training curve
    if history:
        df_h = pd.DataFrame(history)
        plt.figure()
        plt.plot(df_h["epoch"], df_h["val_f1"], marker="o")
        plt.xlabel("Epoch"); plt.ylabel("Val Macro F1")
        plt.title("Baseline Training Curve")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"training_curve_baseline_seed{RANDOM_SEED}.png", dpi=150)
        plt.close()

    # Print mean ± SD across seeds seen so far
    agg_path = RESULTS_DIR / "all_runs_baseline.json"
    if agg_path.exists():
        with open(agg_path) as f:
            runs = json.load(f)
        if len(runs) >= 2:
            f1s  = [r["macro_f1"]              for r in runs]
            mins = [r["minority_avg_accuracy"] for r in runs]
            gaps = [r["bias_gap"]              for r in runs]
            print(f"\n── Baseline multi-seed summary ({len(runs)} seeds) ──")
            print(f"  F1:     {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"  MinAcc: {np.mean(mins):.4f} ± {np.std(mins):.4f}")
            print(f"  Gap:    {np.mean(gaps):.4f} ± {np.std(gaps):.4f}")

    print("\n✓ Baseline training complete.")


if __name__ == "__main__":
    main()
