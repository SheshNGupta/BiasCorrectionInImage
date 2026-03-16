"""
STEP 6: Statistical Significance Testing
Computes paired t-tests and Wilcoxon signed-rank tests across seeds
for all primary metrics. Produces a publication-ready summary table
and updates results/significance_tests.json.

Usage: python 06_significance_tests.py
Requires: scipy, pandas, numpy
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import combinations

RESULTS_DIR = Path("results")
CONDITIONS   = ["baseline", "traditional", "fastgan", "sd_lora", "hybrid"]
METRICS      = ["macro_f1", "minority_avg_accuracy", "majority_avg_accuracy", "bias_gap"]
METRIC_LABELS = {
    "macro_f1":              "Macro F1",
    "minority_avg_accuracy": "Minority Acc.",
    "majority_avg_accuracy": "Majority Acc.",
    "bias_gap":              "Bias Gap",
}
ALPHA = 0.05


# ── Load all runs ─────────────────────────────────────────────────────────────
def load_all_runs():
    data = {}
    for cond in CONDITIONS:
        path = RESULTS_DIR / f"all_runs_{cond}.json"
        if not path.exists():
            print(f"  [WARN] {path} not found — skipping {cond}")
            continue
        with open(path) as f:
            runs = json.load(f)
        # Sort by seed for consistent pairing
        runs = sorted(runs, key=lambda r: r.get("seed", 0))
        data[cond] = runs
        seeds = [r.get("seed", "?") for r in runs]
        print(f"  {cond:14s}: {len(runs)} runs (seeds {seeds})")
    return data


# ── Per-condition summary ─────────────────────────────────────────────────────
def summarise(data):
    rows = []
    for cond, runs in data.items():
        row = {"condition": cond, "n_seeds": len(runs)}
        for metric in METRICS:
            vals = [r[metric] for r in runs]
            row[f"{metric}_mean"] = np.mean(vals)
            row[f"{metric}_sd"]   = np.std(vals, ddof=1)
            row[f"{metric}_vals"] = vals
        rows.append(row)
    return rows


# ── Pairwise significance tests ───────────────────────────────────────────────
def pairwise_tests(summary):
    """
    For each metric, compare every condition against baseline and
    against SD+LoRA. Uses:
      - Paired t-test (parametric, assumes normality — valid for n≥3 seeds)
      - Wilcoxon signed-rank (non-parametric backup)
    With only 3 seeds, p-values are indicative — report effect size (Cohen's d)
    alongside p-values for honest reporting.
    """
    results = []
    cond_map = {r["condition"]: r for r in summary}

    comparisons = []
    # All conditions vs baseline
    for cond in CONDITIONS:
        if cond != "baseline":
            comparisons.append(("baseline", cond))
    # All conditions vs sd_lora
    for cond in CONDITIONS:
        if cond not in ("sd_lora", "baseline"):
            comparisons.append(("sd_lora", cond))

    for metric in METRICS:
        for ref, comp in comparisons:
            if ref not in cond_map or comp not in cond_map:
                continue

            ref_vals  = cond_map[ref][f"{metric}_vals"]
            comp_vals = cond_map[comp][f"{metric}_vals"]

            # Align by seed (already sorted)
            n = min(len(ref_vals), len(comp_vals))
            ref_vals  = ref_vals[:n]
            comp_vals = comp_vals[:n]

            diffs = [c - r for r, c in zip(ref_vals, comp_vals)]
            mean_diff = np.mean(diffs)

            # Cohen's d (paired)
            sd_diff = np.std(diffs, ddof=1) if n > 1 else 0
            cohens_d = mean_diff / sd_diff if sd_diff > 0 else float("inf")

            # Paired t-test
            if n >= 2:
                t_stat, t_pval = stats.ttest_rel(comp_vals, ref_vals)
            else:
                t_stat, t_pval = np.nan, np.nan

            # Wilcoxon (requires n >= 3 with non-zero diffs for exact test)
            try:
                if n >= 3 and any(d != 0 for d in diffs):
                    w_stat, w_pval = stats.wilcoxon(comp_vals, ref_vals)
                else:
                    w_stat, w_pval = np.nan, np.nan
            except Exception:
                w_stat, w_pval = np.nan, np.nan

            significant = t_pval < ALPHA if not np.isnan(t_pval) else False

            results.append({
                "metric":        metric,
                "reference":     ref,
                "comparison":    comp,
                "mean_diff":     round(mean_diff, 5),
                "cohens_d":      round(cohens_d, 3),
                "t_stat":        round(t_stat, 4) if not np.isnan(t_stat) else None,
                "t_pval":        round(t_pval, 4) if not np.isnan(t_pval) else None,
                "w_stat":        round(w_stat, 4) if not np.isnan(w_stat) else None,
                "w_pval":        round(w_pval, 4) if not np.isnan(w_pval) else None,
                "significant":   bool(significant),
                "direction":     "improvement" if mean_diff > 0 else "degradation",
            })

    return results


# ── Bias gap specific analysis ────────────────────────────────────────────────
def bias_gap_analysis(summary):
    """
    Focused analysis on bias gap — the paper's primary fairness metric.
    Reports Bias Reduction Index (BRI) with confidence intervals.
    """
    cond_map = {r["condition"]: r for r in summary}
    baseline_vals = cond_map["baseline"]["bias_gap_vals"]
    baseline_mean = np.mean(baseline_vals)

    print("\n── Bias Gap Analysis ─────────────────────────────────────────")
    print(f"  Baseline gap: {baseline_mean:.4f} ± {np.std(baseline_vals, ddof=1):.4f}")
    print()

    bri_results = []
    for cond in ["traditional", "fastgan", "sd_lora", "hybrid"]:
        if cond not in cond_map:
            continue
        cond_vals  = cond_map[cond]["bias_gap_vals"]
        cond_mean  = np.mean(cond_vals)
        bri        = (baseline_mean - cond_mean) / baseline_mean * 100

        # Bootstrap 95% CI for BRI
        n_boot = 10000
        boot_bri = []
        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            b_idx  = rng.integers(0, len(baseline_vals), len(baseline_vals))
            c_idx  = rng.integers(0, len(cond_vals),     len(cond_vals))
            b_mean = np.mean([baseline_vals[i] for i in b_idx])
            c_mean = np.mean([cond_vals[i]     for i in c_idx])
            if b_mean > 0:
                boot_bri.append((b_mean - c_mean) / b_mean * 100)

        ci_lo = np.percentile(boot_bri, 2.5)
        ci_hi = np.percentile(boot_bri, 97.5)

        direction = "reduces" if bri > 0 else "INCREASES"
        print(f"  {cond:14s}: BRI = {bri:+.1f}% (95% CI [{ci_lo:+.1f}%, {ci_hi:+.1f}%])  "
              f"→ {direction} bias gap")

        bri_results.append({
            "condition": cond,
            "bri":       round(bri, 2),
            "ci_lo":     round(ci_lo, 2),
            "ci_hi":     round(ci_hi, 2),
        })

    return bri_results


# ── Print summary table ───────────────────────────────────────────────────────
def print_summary_table(summary, sig_results):
    print("\n── Per-condition summary (mean ± SD) ─────────────────────────")
    header = f"{'Condition':<14}  {'Macro F1':>16}  {'Min Acc':>16}  {'Maj Acc':>16}  {'Bias Gap':>16}"
    print(header)
    print("─" * len(header))
    for row in summary:
        cond = row["condition"]
        parts = []
        for m in METRICS:
            mean = row[f"{m}_mean"]
            sd   = row[f"{m}_sd"]
            parts.append(f"{mean:.4f}±{sd:.4f}")
        print(f"  {cond:<12}  {'  '.join(f'{p:>16}' for p in parts)}")

    print("\n── Pairwise significance tests (vs baseline) ─────────────────")
    print(f"  {'Metric':<22}  {'Comparison':<20}  {'Diff':>8}  {'d':>6}  {'t-p':>7}  {'Sig':>5}")
    print("─" * 76)
    vs_baseline = [r for r in sig_results if r["reference"] == "baseline"]
    for r in vs_baseline:
        sig_str = "  ✓" if r["significant"] else "  –"
        pval    = f"{r['t_pval']:.3f}" if r["t_pval"] is not None else "  N/A"
        print(f"  {METRIC_LABELS[r['metric']]:<22}  "
              f"baseline→{r['comparison']:<11}  "
              f"{r['mean_diff']:>+8.4f}  "
              f"{r['cohens_d']:>6.2f}  "
              f"{pval:>7}  {sig_str}")

    print("\n  NOTE: With n=3 seeds, p-values are indicative only.")
    print("  Cohen's d: |d|<0.2 small, 0.2–0.8 medium, >0.8 large.")
    print("  Report effect sizes alongside p-values in the manuscript.")


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(summary, sig_results, bri_results):
    output = {
        "summary": [
            {
                "condition": r["condition"],
                "n_seeds":   r["n_seeds"],
                **{f"{m}_mean": round(r[f"{m}_mean"], 5) for m in METRICS},
                **{f"{m}_sd":   round(r[f"{m}_sd"],   5) for m in METRICS},
            }
            for r in summary
        ],
        "pairwise_tests":    sig_results,
        "bias_gap_analysis": bri_results,
        "note": (
            "With n=3 seeds, p-values are indicative only. "
            "Effect sizes (Cohen's d) and bootstrap CIs are the primary evidence. "
            "For publication, report both p-values and d with this caveat."
        )
    }
    path = RESULTS_DIR / "significance_tests.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  [saved] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading multi-seed results...")
    data    = load_all_runs()
    summary = summarise(data)

    print_summary_table(summary, pairwise_tests(summary))
    sig_results = pairwise_tests(summary)
    bri_results = bias_gap_analysis(summary)
    save_results(summary, sig_results, bri_results)

    print("\n✓ Significance testing complete.")
    print("  Upload results/significance_tests.json to Claude to update the manuscript.")


if __name__ == "__main__":
    main()
