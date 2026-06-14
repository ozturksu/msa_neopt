#!/usr/bin/env python3
"""
step06_evaluate.py
MSA-NeOpt — Final evaluation on held-out test set
Produces the main results table and plots for the thesis.

Metrics (Mandi et al. 2024, Equation 50):
  - Relative Regret (%) = (peak_hat - peak_oracle) / peak_oracle × 100
  - Absolute Regret (pu) = peak_hat - peak_oracle
  - Peak Reduction (%) vs no-battery baseline
  - MSE, MAE on demand forecast

Outputs:
  results/test_results.csv   — per-sample results for all 4 models
  results/summary_table.csv  — main results table
  figures/regret_boxplot.png
  figures/learning_curves.png
  figures/peak_reduction_bar.png

Fix applied vs original draft:
  [FIX-1] load_dbb / "DBB Ablation" renamed to load_sspo / "SSPO Ablation"
          to match the checkpoint saved in step04 (sspo_best.pt)
  [FIX-2] dbb_log.csv reference updated to sspo_log.csv in learning curves
  [FIX-3] get_predictions: SSPO uses KimBackbone (dual output) so model_key
          check updated from "dbb" to "sspo"
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from step02_train_pto    import MultiDeT_Adapted
from step03_train_neopt  import KimBackbone, greedy_peak, EirGridDataset, DEVICE
from step05_train_msa_neopt import MSABlock

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
RES_DIR   = ROOT / "results"
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

IN_FEATURES = 19
SEQ_LEN     = 672
PRED_LEN    = 96
BATCH_SIZE  = 64

BAT_CAP   = 0.5
BAT_POWER = 0.25


# ── LP oracle ──────────────────────────────────────────────────────────────────

def lp_peak_shave(load: np.ndarray, bat_cap=BAT_CAP, bat_power=BAT_POWER,
                  eta_c=0.95, eta_d=0.95) -> float:
    """
    LP battery dispatch (Kim et al. 2025, Equations 7-14).
    Solved via cvxpy + GLPK. Fallback: greedy solver.
    """
    try:
        import cvxpy as cp
        T    = len(load)
        u_d  = cp.Variable(T, nonneg=True)
        u_c  = cp.Variable(T, nonneg=True)
        s    = cp.Variable(T + 1)
        peak = cp.Variable(nonneg=True)
        cons = [s[0] == bat_cap / 2]
        for t in range(T):
            cons += [
                load[t] + u_d[t] - u_c[t] <= peak,
                s[t+1] == s[t] - u_d[t] / eta_d + u_c[t] * eta_c,
                s[t+1] >= 0, s[t+1] <= bat_cap,
                u_d[t] <= bat_power, u_c[t] <= bat_power,
            ]
        prob = cp.Problem(cp.Minimize(peak), cons)
        prob.solve(solver=cp.GLPK, verbose=False)
        return float(peak.value) if peak.value is not None else float(load.max())
    except Exception:
        t = torch.tensor(load, dtype=torch.float32).unsqueeze(0)
        return greedy_peak(t).item()


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_pto():
    m = MultiDeT_Adapted().to(DEVICE)
    ckpt = torch.load(MODEL_DIR / "pto_best.pt", map_location=DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

def load_neopt():
    m = KimBackbone().to(DEVICE)
    ckpt = torch.load(MODEL_DIR / "neopt_best.pt", map_location=DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

def load_sspo():
    # [FIX-1] was load_dbb / dbb_best.pt — step04 was renamed to SSPO
    m = KimBackbone().to(DEVICE)
    ckpt = torch.load(MODEL_DIR / "sspo_best.pt", map_location=DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m

def load_msa_neopt():
    m = MSABlock().to(DEVICE)
    ckpt = torch.load(MODEL_DIR / "msa_neopt_best.pt", map_location=DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, model_key: str):
    """Return predictions for all test samples."""
    preds, trues = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        # KimBackbone returns (mean, log_var) — PTO, SSPO, NeOpt use dual head
        # MSABlock returns a single tensor
        if model_key in ("neopt", "sspo"):   # [FIX-3] was "dbb"
            mu, _ = model(X)
        else:
            mu = model(X)
        preds.append(mu.cpu())
        trues.append(y)
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


def compute_metrics(preds: np.ndarray, trues: np.ndarray,
                    use_lp: bool = False, n_lp_samples: int = 200) -> dict:
    """Compute all evaluation metrics for one model."""
    N = len(preds)

    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))

    p_hat    = greedy_peak(torch.tensor(preds.clip(0, 2), dtype=torch.float32)).numpy()
    p_oracle = greedy_peak(torch.tensor(trues,            dtype=torch.float32)).numpy()
    p_nobat  = trues.max(axis=1)

    rel_regret    = np.maximum(0, (p_hat - p_oracle) / np.maximum(p_oracle, 1e-6)) * 100
    abs_regret    = np.maximum(0, p_hat - p_oracle)
    peak_reduction = (p_nobat - p_hat) / np.maximum(p_nobat, 1e-6) * 100

    if use_lp:
        idx = np.random.choice(N, min(n_lp_samples, N), replace=False)
        lp_peaks  = np.array([lp_peak_shave(trues[i]) for i in idx])
        lp_regret = np.maximum(0, (p_hat[idx] - lp_peaks) / np.maximum(lp_peaks, 1e-6)) * 100
    else:
        lp_regret = None

    return {
        "mse":             mse,
        "mae":             mae,
        "rel_regret_mean": float(rel_regret.mean()),
        "rel_regret_std":  float(rel_regret.std()),
        "abs_regret_mean": float(abs_regret.mean()),
        "peak_reduction":  float(peak_reduction.mean()),
        "rel_regret_arr":  rel_regret,
        "abs_regret_arr":  abs_regret,
        "lp_regret_arr":   lp_regret,
    }


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_regret_boxplot(metrics_dict: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(metrics_dict.keys())
    data   = [metrics_dict[m]["rel_regret_arr"] for m in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, c in zip(bp["boxes"], colours):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_ylabel("Relative Regret (%)", fontsize=12)
    ax.set_title("Battery Dispatch Regret by Model — EirGrid Test Set", fontsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regret_boxplot.png", dpi=150)
    plt.close()
    print("  Saved → figures/regret_boxplot.png")


def plot_learning_curves():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    configs = [
        ("pto",       "PTO (MultiDeT)",    "results/pto_log.csv",       "val_mse"),
        ("neopt",     "NeOpt (SPO+)",       "results/neopt_log.csv",     "val_regret"),
        ("sspo",      "SSPO Ablation",      "results/sspo_log.csv",      "val_regret"),  # [FIX-2]
        ("msa_neopt", "MSA-NeOpt (Ours)",  "results/msa_neopt_log.csv", "val_regret"),
    ]
    for ax, (key, title, log_path, metric) in zip(axes.flat, configs):
        try:
            log = pd.read_csv(ROOT / log_path)
            ax.plot(log["epoch"], log[metric], lw=1.5, label=metric)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.axvline(x=15, color="grey", linestyle=":", alpha=0.6, label="DFL starts")
            ax.legend(fontsize=8)
        except FileNotFoundError:
            ax.text(0.5, 0.5, f"{log_path}\nnot found", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)

    fig.suptitle("Training Curves — MSA-NeOpt Experiments", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "learning_curves.png", dpi=150)
    plt.close()
    print("  Saved → figures/learning_curves.png")


def plot_peak_reduction_bar(metrics_dict: dict):
    labels = list(metrics_dict.keys())
    values = [metrics_dict[m]["peak_reduction"] for m in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(labels, values, color=colours, alpha=0.8, edgecolor="white", linewidth=1.2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean Peak Reduction vs No-Battery (%)", fontsize=11)
    ax.set_title("Peak Shaving Performance — EirGrid Test Set", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "peak_reduction_bar.png", dpi=150)
    plt.close()
    print("  Saved → figures/peak_reduction_bar.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MSA-NeOpt — Step 06: Final Evaluation (Test Set)")
    print("=" * 60)

    test_ds = EirGridDataset("test")
    test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                         num_workers=2, pin_memory=False)
    print(f"  Test samples: {len(test_ds)}")

    # Model registry — [FIX-1] DBB → SSPO
    models = {
        "PTO (MultiDeT)":   (load_pto,       "pto"),
        "NeOpt (SPO+)":     (load_neopt,     "neopt"),
        "SSPO Ablation":    (load_sspo,      "sspo"),
        "MSA-NeOpt (Ours)": (load_msa_neopt, "msa_neopt"),
    }

    all_metrics = {}
    all_rows    = []

    for display_name, (load_fn, model_key) in models.items():
        ckpt_name = f"{model_key}_best.pt"
        if not (MODEL_DIR / ckpt_name).exists():
            print(f"  Skipping {display_name} — {ckpt_name} not found")
            continue

        print(f"\n  Evaluating {display_name} ...")
        model   = load_fn()
        preds, trues = get_predictions(model, test_dl, model_key)
        metrics = compute_metrics(preds, trues, use_lp=True, n_lp_samples=200)

        all_metrics[display_name] = metrics
        all_rows.append({
            "Model":              display_name,
            "MSE":                f"{metrics['mse']:.6f}",
            "MAE":                f"{metrics['mae']:.6f}",
            "Rel. Regret (%)":    f"{metrics['rel_regret_mean']:.4f} ± {metrics['rel_regret_std']:.4f}",
            "Abs. Regret (pu)":   f"{metrics['abs_regret_mean']:.4f}",
            "Peak Reduction (%)": f"{metrics['peak_reduction']:.4f}",
        })

        print(f"    MSE:              {metrics['mse']:.6f}")
        print(f"    MAE:              {metrics['mae']:.6f}")
        print(f"    Rel. Regret (%):  {metrics['rel_regret_mean']:.4f} ± {metrics['rel_regret_std']:.4f}")
        print(f"    Abs. Regret (pu): {metrics['abs_regret_mean']:.4f}")
        print(f"    Peak Reduction:   {metrics['peak_reduction']:.4f}%")
        if metrics["lp_regret_arr"] is not None:
            print(f"    LP Regret (%):    {metrics['lp_regret_arr'].mean():.4f} (n=200 subsample)")

    # Save summary
    summary = pd.DataFrame(all_rows)
    summary.to_csv(RES_DIR / "summary_table.csv", index=False)
    print(f"\n  Summary → results/summary_table.csv")
    print("\n" + summary.to_string(index=False))

    # Figures
    if all_metrics:
        print("\n  Generating figures...")
        plot_regret_boxplot(all_metrics)
        plot_learning_curves()
        plot_peak_reduction_bar(all_metrics)

    print("\n  Done. All results in results/ and figures/")


if __name__ == "__main__":
    main()
