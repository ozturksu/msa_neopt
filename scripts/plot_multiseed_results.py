#!/usr/bin/env python3
"""
plot_multiseed_results.py
generates the two multi-seed comparison figures used in the
report (regret across all SPO+ variations, and secondary metrics) directly
from the saved results/multiseed/*_multiseed_summary.csv files produced by
train_{model}_multiseed.py.

Unlike a one-off script with hardcoded numbers, this reads the real CSVs
and computes mean/std itself, so it stays correct automatically if you add
more seeds later and rerun.

Usage:
    python plot_multiseed_results.py

Requires all 5 of the following to already exist:
    results/multiseed/pto_multiseed_summary.csv
    results/multiseed/sspo_multiseed_summary.csv
    results/multiseed/msa_neopt_multiseed_summary.csv
    results/multiseed/gru_neopt_multiseed_summary.csv
    results/multiseed/neopt_multiseed_summary.csv

Output:
    figures/spo_variations_regret.png
    figures/spo_variations_secondary.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT     = Path(__file__).parent.parent
RES_DIR  = ROOT / "results" / "multiseed"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Display order and labels for the 5 models. Matches the CSV filenames
# produced by each train_{model}_multiseed.py script.
MODEL_SPECS = [
    {"key": "pto",       "label": "PTO\n(MSE)",            "color": "#888888"},
    {"key": "sspo",      "label": "SSPO\n(surrogate)",      "color": "#aaaaaa"},
    {"key": "msa_neopt", "label": "MSA-NeOpt\n(SPO+, CNN)", "color": "#2ca02c"},
    {"key": "gru_neopt", "label": "GRU-NeOpt\n(SPO+, RNN)", "color": "#1f77b4"},
    {"key": "neopt",     "label": "NeOpt\n(SPO+, Attn)",    "color": "#ff7f0e"},
]

# Which models count as "SPO+ variations" for the overlap-band shading in
# the regret figure. Update this list if you add more SPO+-trained models.
SPO_PLUS_KEYS = {"msa_neopt", "gru_neopt", "neopt"}


def load_summary(key: str) -> pd.DataFrame:
    """Load one model's multiseed summary CSV, produced by
    train_{key}_multiseed.py's final pd.merge(...).to_csv(...) call."""
    path = RES_DIR / f"{key}_multiseed_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run train_{key}_multiseed.py to completion "
            f"first (all seeds must finish, not just be in progress)."
        )
    return pd.read_csv(path)


def compute_stats() -> pd.DataFrame:
    """Load all 5 models' CSVs and compute mean/std for each test metric.
    Returns one row per model, in MODEL_SPECS order."""
    rows = []
    for spec in MODEL_SPECS:
        df = load_summary(spec["key"])
        n_seeds = len(df)
        rows.append({
            "key":            spec["key"],
            "label":          spec["label"],
            "color":          spec["color"],
            "n_seeds":        n_seeds,
            "regret_mean":    df["test_rel_regret"].mean(),
            "regret_std":     df["test_rel_regret"].std(),
            "peak_red_mean":  df["test_peak_reduction"].mean(),
            "lp_regret_mean": df["test_lp_regret"].mean(),
            "mse_mean":       df["test_mse"].mean(),
            "mse_std":        df["test_mse"].std(),
            "mae_mean":       df["test_mae"].mean(),
            "mae_std":        df["test_mae"].std(),
        })
    return pd.DataFrame(rows)


def plot_regret_comparison(stats: pd.DataFrame):
    """Figure 1: mean regret with error bars across all 5 models, with the
    SPO+ overlap band shaded to make the RQ2 finding visually obvious."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(stats))

    ax.bar(x, stats["regret_mean"], yerr=stats["regret_std"], capsize=6,
           color=stats["color"], alpha=0.85, edgecolor="black", linewidth=0.7)

    spo_rows = stats[stats["key"].isin(SPO_PLUS_KEYS)]
    spo_lo = (spo_rows["regret_mean"] - spo_rows["regret_std"]).min()
    spo_hi = (spo_rows["regret_mean"] + spo_rows["regret_std"]).max()
    ax.axhspan(max(0, spo_lo), spo_hi, color="green", alpha=0.08, zorder=0)
    ax.text(len(stats) - 0.5, spo_hi + 0.03, "SPO+ overlap band",
            fontsize=9, style="italic", ha="right", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(stats["label"], fontsize=10)
    n_seeds_note = stats["n_seeds"].mode()[0]
    ax.set_ylabel(f"Mean Relative Regret (%)  [{n_seeds_note}-seed mean $\\pm$ std]",
                   fontsize=11)
    ax.set_title("Regret Across All SPO+ Variations vs. Non-SPO+ Baselines\n"
                 "(EirGrid test set)", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for i, row in stats.iterrows():
        ax.text(i, row["regret_mean"] + row["regret_std"] + 0.04,
                f"{row['regret_mean']:.2f}%", ha="center", fontsize=9,
                fontweight="bold")

    plt.tight_layout()
    out = FIG_DIR / "spo_variations_regret.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


def plot_secondary_metrics(stats: pd.DataFrame):
    """Figure 2: peak reduction and LP-oracle regret side by side, for all
    5 models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(stats))

    axes[0].bar(x, stats["peak_red_mean"], color=stats["color"], alpha=0.85,
                edgecolor="black", linewidth=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stats["label"], fontsize=9)
    axes[0].set_ylabel("Mean Peak Reduction (%)", fontsize=10)
    axes[0].set_title("Peak Reduction vs No-Battery Baseline", fontsize=11)
    axes[0].yaxis.grid(True, linestyle="--", alpha=0.4)
    axes[0].set_axisbelow(True)
    for i, v in enumerate(stats["peak_red_mean"]):
        axes[0].text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8,
                     fontweight="bold")

    axes[1].bar(x, stats["lp_regret_mean"], color=stats["color"], alpha=0.85,
                edgecolor="black", linewidth=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stats["label"], fontsize=9)
    axes[1].set_ylabel("LP-Oracle Regret (%)", fontsize=10)
    axes[1].set_title("Regret vs Theoretically Optimal (LP) Dispatch", fontsize=11)
    axes[1].yaxis.grid(True, linestyle="--", alpha=0.4)
    axes[1].set_axisbelow(True)
    for i, v in enumerate(stats["lp_regret_mean"]):
        axes[1].text(i, v + 0.08, f"{v:.2f}%", ha="center", fontsize=8,
                     fontweight="bold")

    n_seeds_note = stats["n_seeds"].mode()[0]
    fig.suptitle(f"Secondary Metrics Across All Model Variations "
                 f"({n_seeds_note}-seed means)", fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG_DIR / "spo_variations_secondary.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    print("Loading multi-seed summary CSVs...")
    stats = compute_stats()

    print("\nComputed statistics (from real saved data, not hardcoded):")
    print(stats[["label", "n_seeds", "regret_mean", "regret_std",
                 "peak_red_mean", "lp_regret_mean"]].to_string(index=False))

    if stats["n_seeds"].nunique() > 1:
        print(f"\n  Warning: models have different seed counts "
              f"({dict(zip(stats['key'], stats['n_seeds']))}). Figure "
              f"labels will use the most common count; consider noting "
              f"this discrepancy in the report if it is not intentional.")

    print()
    plot_regret_comparison(stats)
    plot_secondary_metrics(stats)
    print("\nDone. Copy both PNGs into report's figures/ folder if not")
    print("already saved there directly.")


if __name__ == "__main__":
    main()