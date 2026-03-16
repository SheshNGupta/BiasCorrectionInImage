"""
STEP 7: Final Comparison, Fairness Metrics & Paper-Ready Figures
Collects all results, computes fairness metrics, and produces
publication-quality charts and a summary table.

Requirements:
    pip install pandas matplotlib seaborn fairlearn scikit-learn
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_LABELS = {
    "baseline":    "Baseline\n(Real Only)",
    "traditional": "Traditional\nAugmentation",
    "fastgan":     "FastGAN\nAugmented",
    "sd_lora":     "SD+LoRA\nAugmented",
    "hybrid":      "Hybrid\n(GAN+SD)",
}
COLORS = {
    "baseline":    "#888888",
    "traditional": "#E97132",
    "fastgan":     "#2E75B6",
    "sd_lora":     "#C00000",
    "hybrid":      "#1E6B3C",
}

# ── Load all metrics ──────────────────────────────────────────────────────────
def load_all_metrics(minority_breeds):
    conditions = ["baseline", "traditional", "fastgan", "sd_lora", "hybrid"]
    records = []
    for c in conditions:
        p = RESULTS_DIR / f"metrics_{c}.json"
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        with open(p) as f:
            d = json.load(f)
        records.append(d)
    return records

# ── Summary Table ─────────────────────────────────────────────────────────────
def build_summary_table(records):
    rows = []
    for r in records:
        rows.append({
            "Condition":                 CONDITION_LABELS.get(r["condition"], r["condition"]),
            "Macro F1":                  f"{r['macro_f1']:.3f}",
            "Minority Avg Acc":          f"{r['minority_avg_accuracy']:.3f}",
            "Majority Avg Acc":          f"{r['majority_avg_accuracy']:.3f}",
            "Bias Gap (↓ better)":       f"{r['bias_gap']:.3f}",
            "GPU Mins":                  f"{r['gpu_minutes']}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print("\n── Summary Table ──────────────────────────────────────────")
    print(df.to_string(index=False))
    return df

# ── Figure 1: Macro F1 bar chart ──────────────────────────────────────────────
def plot_macro_f1(records):
    conds  = [r["condition"] for r in records]
    f1s    = [r["macro_f1"]  for r in records]
    colors = [COLORS[c] for c in conds]
    labels = [CONDITION_LABELS[c] for c in conds]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, f1s, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Macro F1 Score by Augmentation Condition", fontsize=13, fontweight="bold")
    ax.axhline(f1s[0], color="grey", linestyle="--", linewidth=0.8, label="Baseline")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_macro_f1.png", dpi=200)
    plt.close()
    print("  Saved fig1_macro_f1.png")

# ── Figure 2: Minority vs Majority accuracy ───────────────────────────────────
def plot_minority_majority(records):
    labels  = [CONDITION_LABELS[r["condition"]] for r in records]
    min_acc = [r["minority_avg_accuracy"] for r in records]
    maj_acc = [r["majority_avg_accuracy"] for r in records]

    x = np.arange(len(labels))
    w = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, min_acc, width=w, label="Minority Breeds", color="#C00000", alpha=0.85)
    b2 = ax.bar(x + w/2, maj_acc, width=w, label="Majority Breeds", color="#2E75B6", alpha=0.85)
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Average Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Minority vs Majority Breed Accuracy by Condition",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_minority_majority.png", dpi=200)
    plt.close()
    print("  Saved fig2_minority_majority.png")

# ── Figure 3: Bias gap reduction ──────────────────────────────────────────────
def plot_bias_gap(records):
    labels = [CONDITION_LABELS[r["condition"]] for r in records]
    gaps   = [r["bias_gap"] for r in records]
    colors = [COLORS[r["condition"]] for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, gaps, color=colors, width=0.5, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylabel("Bias Gap (Majority Acc − Minority Acc)", fontsize=11)
    ax.set_title("Bias Gap by Condition (Lower is Better)",
                 fontsize=13, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_bias_gap.png", dpi=200)
    plt.close()
    print("  Saved fig3_bias_gap.png")

# ── Figure 4: Per-class accuracy heatmap across conditions ────────────────────
def plot_per_class_heatmap(records, minority_breeds):
    conditions = [r["condition"] for r in records]
    # Get all breeds from first record
    breeds = sorted(records[0]["per_class_accuracy"].keys())

    data = []
    for r in records:
        row = [r["per_class_accuracy"].get(b, 0) for b in breeds]
        data.append(row)

    df = pd.DataFrame(data, index=[CONDITION_LABELS[c] for c in conditions], columns=breeds)

    # Highlight minority breed columns
    minority_mask = [b in minority_breeds for b in breeds]

    fig, ax = plt.subplots(figsize=(max(16, len(breeds) * 0.45), 5))
    sns.heatmap(df, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.3, annot=False, cbar_kws={"label": "Accuracy"})

    # Mark minority breeds with asterisk on x-axis
    xlabels = [f"*{b}" if b in minority_breeds else b for b in breeds]
    ax.set_xticklabels(xlabels, rotation=90, fontsize=6)
    ax.set_title("Per-Class Accuracy Heatmap (* = minority breed)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_heatmap.png", dpi=200)
    plt.close()
    print("  Saved fig4_heatmap.png")

# ── Figure 5: FID scores ──────────────────────────────────────────────────────
def plot_fid(minority_breeds):
    fid_path = RESULTS_DIR / "fid_scores.json"
    if not fid_path.exists():
        print("  [skip] FID scores not found")
        return

    with open(fid_path) as f:
        fid_data = json.load(f)

    breeds   = [b for b in minority_breeds if b in fid_data]
    trad_fids = [fid_data[b].get("traditional", 0) for b in breeds]
    gan_fids  = [fid_data[b].get("fastgan", 0) for b in breeds]
    sd_fids   = [fid_data[b].get("sd_lora",  0) for b in breeds]

    x = np.arange(len(breeds))
    w = 0.22

    fig, ax = plt.subplots(figsize=(11, 5))
    b0 = ax.bar(x - w,     trad_fids, width=w, label="Traditional", color="#E97132", alpha=0.85)
    b1 = ax.bar(x,         gan_fids,  width=w, label="FastGAN",     color="#2E75B6", alpha=0.85)
    b2 = ax.bar(x + w,     sd_fids,   width=w, label="SD+LoRA",     color="#C00000", alpha=0.85)
    ax.bar_label(b0, fmt="%.1f", padding=3, fontsize=7)
    ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=7)
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(breeds, fontsize=9)
    ax.set_ylabel("FID Score (↓ = more realistic)", fontsize=11)
    ax.set_title("FID Score per Minority Breed (Lower is Better)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_fid.png", dpi=200)
    plt.close()
    print("  Saved fig5_fid.png")

# ── Bias Reduction Index ───────────────────────────────────────────────────────
def compute_bias_reduction(records):
    baseline = next((r for r in records if r["condition"] == "baseline"), None)
    if not baseline:
        return
    baseline_gap = baseline["bias_gap"]
    print("\n── Bias Reduction Index ─────────────────────────────────")
    for r in records:
        if r["condition"] == "baseline":
            continue
        reduction = (baseline_gap - r["bias_gap"]) / baseline_gap * 100
        print(f"  {r['condition']:12s}: {reduction:+.1f}% bias gap reduction")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(RESULTS_DIR / "metadata.json") as f:
        meta = json.load(f)
    minority_breeds = meta["minority_breeds"]

    records = load_all_metrics(minority_breeds)
    if not records:
        print("No metrics found. Run steps 2–5 first.")
        return

    print(f"\nLoaded results for {len(records)} conditions.")

    # Summary table
    build_summary_table(records)

    # Figures
    print("\nGenerating figures...")
    plot_macro_f1(records)
    plot_minority_majority(records)
    plot_bias_gap(records)
    plot_per_class_heatmap(records, minority_breeds)
    plot_fid(minority_breeds)

    # Bias reduction
    compute_bias_reduction(records)

    # Add FID to summary
    fid_path = RESULTS_DIR / "fid_scores.json"
    if fid_path.exists():
        with open(fid_path) as f:
            fid_data = json.load(f)
        avg = fid_data.get("average", {})
        print(f"\n── Average FID ────────────────────────────────────────")
        print(f"  Traditional avg FID : {avg.get('traditional', 'N/A')}")
        print(f"  FastGAN     avg FID : {avg.get('fastgan',     'N/A')}")
        print(f"  SD+LoRA     avg FID : {avg.get('sd_lora',     'N/A')}")

    print(f"\n✓ All figures saved to {FIGURES_DIR}")
    print("  Ready to copy figures into your manuscript.")


if __name__ == "__main__":
    main()
