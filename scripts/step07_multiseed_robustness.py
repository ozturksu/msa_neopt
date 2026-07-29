#!/usr/bin/env python3
"""
step07_multiseed_robustness.py
MSA-NeOpt — Multi-seed robustness test for the three SPO+-trained models.

Motivation
----------
Every result reported so far (NeOpt, GRU-NeOpt, MSA-NeOpt) comes from a
SINGLE training run per model. Neural network training is stochastic —
weight initialisation and batch shuffling order both vary with the random
seed — so a single run cannot distinguish "this architecture is reliably
better" from "this run got a favourable draw". This is flagged as the
top-priority limitation of the study (Section 6.8 / Future Work).

This script retrains NeOpt, GRU-NeOpt, and MSA-NeOpt across N_SEEDS
different random seeds each, evaluates every resulting checkpoint on the
held-out test set, and reports mean +/- std of relative regret, peak
reduction, and LP-oracle regret per model. If GRU-NeOpt's advantage over
NeOpt survives this — i.e. the two do not overlap once seed variance is
accounted for — the finding is robust. If they overlap, the original
single-seed result should be reported as inconclusive.

Design
------
- Reuses EirGridDataset, greedy_peak, spo_plus_loss, and all constants
  from step03_train_neopt.py — imported, not reimplemented, so the ONLY
  thing that changes across runs is the random seed.
- Reuses KimBackbone + training functions from step03, GRUBackbone +
  training functions from step03b, MSABlock + training functions from
  step05 — same reuse pattern already used throughout this project.
- Runs entirely in one process (no subprocess calls) to avoid repeatedly
  reloading the ~1GB training dataset from disk.
- RESUMABLE: before training any (model, seed) pair, checks whether its
  checkpoint already exists and skips it if so. Safe to stop and restart.

Expected runtime
-----------------
~45-70 minutes per training run on Apple Silicon CPU. With the default
N_SEEDS = 5 and 3 models, that is 15 runs, roughly 10-12 hours total.
Reduce N_SEEDS below (e.g. to [0, 1, 2] for 3 seeds, ~6-7 hours) if that
is too long — 3 seeds is still a defensible minimum for reporting
mean +/- std, though 5 is preferable.

Output
------
results/multiseed_summary.csv       — per (model, seed) test metrics
results/multiseed_aggregate.csv     — mean +/- std per model across seeds
figures/multiseed_regret_errorbar.png
models/multiseed/{model}_seed{seed}_best.pt
"""

import time
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

from step03_train_neopt import (
    EirGridDataset, greedy_peak, spo_plus_loss, KimBackbone,
    train_warmup_epoch, train_dfl_epoch, evaluate as evaluate_kim,
    IN_FEATURES, SEQ_LEN, PRED_LEN, BATCH_SIZE,
    LR_WARMUP, LR_DFL, WARMUP_EPOCHS, FINETUNE_EPOCHS, PATIENCE, DEVICE,
)
from step03b_train_gru_neopt import (
    GRUBackbone, train_gru_warmup_epoch, train_gru_dfl_epoch,
    evaluate_gru,
)
from step05_train_msa_neopt import (
    MSABlock, train_msa_warmup_epoch, train_msa_dfl_epoch, evaluate_msa,
)

ROOT        = Path(__file__).parent.parent
MODEL_DIR   = ROOT / "models" / "multiseed"
RES_DIR     = ROOT / "results"
FIG_DIR     = ROOT / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
N_SEEDS = [0, 1, 2, 3, 4]     # reduce to [0, 1, 2] for a faster 3-seed test
N_LP_SAMPLES = 100            # smaller than step06's 200 to keep total
                               # multi-seed evaluation time reasonable

MODEL_SPECS = {
    "neopt": {
        "display_name":  "NeOpt (PatchTST)",
        "build_model":   lambda: KimBackbone().to(DEVICE),
        "train_warmup":  train_warmup_epoch,
        "train_dfl":     train_dfl_epoch,
        "evaluate":      evaluate_kim,
        "dual_output":   True,   # returns (mu, logvar) — needs unpacking
    },
    "gru_neopt": {
        "display_name":  "GRU-NeOpt",
        "build_model":   lambda: GRUBackbone().to(DEVICE),
        "train_warmup":  train_gru_warmup_epoch,
        "train_dfl":     train_gru_dfl_epoch,
        "evaluate":      evaluate_gru,
        "dual_output":   False,
    },
    "msa_neopt": {
        "display_name":  "MSA-NeOpt",
        "build_model":   lambda: MSABlock().to(DEVICE),
        "train_warmup":  train_msa_warmup_epoch,
        "train_dfl":     train_msa_dfl_epoch,
        "evaluate":      evaluate_msa,
        "dual_output":   False,
    },
}


# ── Single (model, seed) training run ───────────────────────────────────────

def train_one_seed(model_key: str, seed: int, train_dl, val_dl) -> dict:
    """Train one model with one seed using its own existing training
    functions (imported above), following the identical two-phase
    schedule used everywhere else in this project. Returns a dict with
    the best validation regret and the checkpoint path."""

    spec = MODEL_SPECS[model_key]
    ckpt_path = MODEL_DIR / f"{model_key}_seed{seed}_best.pt"

    if ckpt_path.exists():
        print(f"  [skip] {model_key} seed={seed} — checkpoint already exists")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        return {"model": model_key, "seed": seed,
                "best_val_regret": ckpt["val_regret"],
                "best_epoch": ckpt["epoch"], "ckpt_path": str(ckpt_path)}

    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = spec["build_model"]()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR_WARMUP)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    print(f"\n  {'='*50}")
    print(f"  {spec['display_name']}  —  seed={seed}")
    print(f"  {'='*50}")

    t0 = time.time()

    # Phase 1: warm-up
    for epoch in range(1, WARMUP_EPOCHS + 1):
        spec["train_warmup"](model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = spec["evaluate"](model, val_dl, DEVICE)
        scheduler.step(val_mse)
        if epoch % 5 == 0 or epoch == 1:
            print(f"    warmup epoch {epoch:>3}  val_regret={val_regret:.6f}")

    # Phase 2: DFL fine-tuning
    for pg in optimiser.param_groups:
        pg["lr"] = LR_DFL
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    best_val_regret, best_epoch, patience_counter = float("inf"), 0, 0

    for epoch_ft in range(1, FINETUNE_EPOCHS + 1):
        epoch = WARMUP_EPOCHS + epoch_ft
        spec["train_dfl"](model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = spec["evaluate"](model, val_dl, DEVICE)
        scheduler.step(val_regret)

        if epoch_ft % 5 == 0 or epoch_ft == 1:
            print(f"    dfl epoch {epoch:>3}  val_regret={val_regret:.6f}")

        if val_regret < best_val_regret:
            best_val_regret = val_regret
            best_epoch      = epoch
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_regret": val_regret, "val_mse": val_mse, "seed": seed,
            }, ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    early stopping at epoch {epoch} (best {best_epoch})")
                break

    elapsed_min = (time.time() - t0) / 60
    print(f"  Done in {elapsed_min:.1f} min. Best val_regret={best_val_regret:.6f} "
          f"(epoch {best_epoch})")

    return {"model": model_key, "seed": seed, "best_val_regret": best_val_regret,
            "best_epoch": best_epoch, "ckpt_path": str(ckpt_path),
            "train_minutes": elapsed_min}


# ── Test-set evaluation for one checkpoint ──────────────────────────────────

def lp_peak_shave(load, bat_cap=0.5, bat_power=0.25, eta_c=0.95, eta_d=0.95):
    try:
        import cvxpy as cp
        T = len(load)
        u_d, u_c = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)
        s, peak  = cp.Variable(T + 1), cp.Variable(nonneg=True)
        cons = [s[0] == bat_cap / 2]
        for t in range(T):
            cons += [
                load[t] + u_d[t] - u_c[t] <= peak,
                s[t+1] == s[t] - u_d[t]/eta_d + u_c[t]*eta_c,
                s[t+1] >= 0, s[t+1] <= bat_cap,
                u_d[t] <= bat_power, u_c[t] <= bat_power,
            ]
        cp.Problem(cp.Minimize(peak), cons).solve(solver=cp.GLPK, verbose=False)
        return float(peak.value) if peak.value is not None else float(load.max())
    except Exception:
        t = torch.tensor(load, dtype=torch.float32).unsqueeze(0)
        return greedy_peak(t).item()


@torch.no_grad()
def evaluate_checkpoint_on_test(model_key: str, ckpt_path: Path,
                                 test_dl) -> dict:
    spec  = MODEL_SPECS[model_key]
    model = spec["build_model"]()
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds, trues = [], []
    for X, y in test_dl:
        X = X.to(DEVICE)
        out = model(X)
        mu = out[0] if spec["dual_output"] else out
        preds.append(mu.cpu())
        trues.append(y)
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    mse = float(np.mean((preds - trues) ** 2))
    mae = float(np.mean(np.abs(preds - trues)))
    p_hat    = greedy_peak(torch.tensor(preds.clip(0, 2), dtype=torch.float32)).numpy()
    p_oracle = greedy_peak(torch.tensor(trues, dtype=torch.float32)).numpy()
    p_nobat  = trues.max(axis=1)

    rel_regret = np.maximum(0, (p_hat - p_oracle) / np.maximum(p_oracle, 1e-6)) * 100
    peak_reduction = (p_nobat - p_hat) / np.maximum(p_nobat, 1e-6) * 100

    N = len(preds)
    idx = np.random.RandomState(0).choice(N, min(N_LP_SAMPLES, N), replace=False)
    lp_peaks  = np.array([lp_peak_shave(trues[i]) for i in idx])
    lp_regret = np.maximum(0, (p_hat[idx] - lp_peaks) / np.maximum(lp_peaks, 1e-6)) * 100

    return {
        "mse": mse, "mae": mae,
        "rel_regret_mean": float(rel_regret.mean()),
        "rel_regret_std":  float(rel_regret.std()),
        "peak_reduction":  float(peak_reduction.mean()),
        "lp_regret_mean":  float(lp_regret.mean()),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MSA-NeOpt — Step 07: Multi-Seed Robustness Test")
    print(f"  Models: {list(MODEL_SPECS.keys())}")
    print(f"  Seeds:  {N_SEEDS}")
    print(f"  Total runs: {len(MODEL_SPECS) * len(N_SEEDS)}")
    print("=" * 60)

    train_ds = EirGridDataset("train")
    val_ds   = EirGridDataset("val")
    test_ds  = EirGridDataset("test")
    # [FIX] num_workers=0, not 2. EirGridDataset.__getitem__ does nothing
    # but slice an already in-memory tensor -- there is no real per-sample
    # work for background workers to parallelise. On macOS, PyTorch's
    # 'spawn' multiprocessing start method relaunches a fresh Python
    # process per worker, which re-executes every top-level import in
    # this script -- including step03/step03b/step05's own module-level
    # code. Because this script imports from all three model files at
    # once, that respawn overhead compounds badly and was observed to
    # cause a ~100x slowdown (76 min/epoch instead of the expected
    # ~45 sec/epoch). num_workers=0 runs data loading in the main
    # process and avoids this entirely.
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # ── Phase A: train every (model, seed) pair ────────────────────────────
    training_log = []
    overall_start = time.time()
    total_runs = len(MODEL_SPECS) * len(N_SEEDS)
    run_i = 0

    for model_key in MODEL_SPECS:
        for seed in N_SEEDS:
            run_i += 1
            print(f"\n[{run_i}/{total_runs}] elapsed so far: "
                  f"{(time.time()-overall_start)/60:.1f} min")
            result = train_one_seed(model_key, seed, train_dl, val_dl)
            training_log.append(result)

    pd.DataFrame(training_log).to_csv(RES_DIR / "multiseed_training_log.csv", index=False)

    # ── Phase B: evaluate every checkpoint on the test set ─────────────────
    print("\n" + "=" * 60)
    print("Evaluating all checkpoints on held-out test set")
    print("=" * 60)

    eval_rows = []
    for model_key in MODEL_SPECS:
        for seed in N_SEEDS:
            ckpt_path = MODEL_DIR / f"{model_key}_seed{seed}_best.pt"
            if not ckpt_path.exists():
                print(f"  Skipping {model_key} seed={seed} — no checkpoint")
                continue
            metrics = evaluate_checkpoint_on_test(model_key, ckpt_path, test_dl)
            metrics["model"] = model_key
            metrics["seed"]  = seed
            eval_rows.append(metrics)
            print(f"  {MODEL_SPECS[model_key]['display_name']:20s} seed={seed}  "
                  f"regret={metrics['rel_regret_mean']:.4f}%  "
                  f"peak_red={metrics['peak_reduction']:.2f}%  "
                  f"lp_regret={metrics['lp_regret_mean']:.4f}%")

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(RES_DIR / "multiseed_summary.csv", index=False)

    # ── Aggregate: mean +/- std per model across seeds ─────────────────────
    agg = eval_df.groupby("model").agg(
        n_seeds           = ("seed", "count"),
        regret_mean       = ("rel_regret_mean", "mean"),
        regret_std        = ("rel_regret_mean", "std"),
        peak_red_mean     = ("peak_reduction", "mean"),
        peak_red_std      = ("peak_reduction", "std"),
        lp_regret_mean    = ("lp_regret_mean", "mean"),
        lp_regret_std     = ("lp_regret_mean", "std"),
    ).reset_index()
    agg["display_name"] = agg["model"].map(
        lambda k: MODEL_SPECS[k]["display_name"])
    agg.to_csv(RES_DIR / "multiseed_aggregate.csv", index=False)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (mean +/- std across seeds)")
    print("=" * 60)
    print(agg[["display_name", "n_seeds", "regret_mean", "regret_std",
               "peak_red_mean", "peak_red_std"]].to_string(index=False))

    # ── Robustness check: does GRU-NeOpt's advantage survive seed variance? ─
    if "neopt" in agg["model"].values and "gru_neopt" in agg["model"].values:
        neopt_row = agg[agg["model"] == "neopt"].iloc[0]
        gru_row   = agg[agg["model"] == "gru_neopt"].iloc[0]
        neopt_lo  = neopt_row["regret_mean"] - neopt_row["regret_std"]
        gru_hi    = gru_row["regret_mean"] + gru_row["regret_std"]
        overlap   = gru_hi >= neopt_lo
        print("\n" + "-" * 60)
        print("ROBUSTNESS CHECK: NeOpt vs GRU-NeOpt regret ranges")
        print(f"  NeOpt:     {neopt_row['regret_mean']:.4f} +/- {neopt_row['regret_std']:.4f}")
        print(f"  GRU-NeOpt: {gru_row['regret_mean']:.4f} +/- {gru_row['regret_std']:.4f}")
        if overlap:
            print("  -> Ranges OVERLAP. GRU-NeOpt's single-seed advantage over")
            print("     NeOpt is NOT clearly distinguishable from seed noise.")
        else:
            print("  -> Ranges DO NOT overlap. GRU-NeOpt's advantage appears")
            print("     robust across seeds, not a single lucky run.")
        print("-" * 60)

    # ── Figure: error-bar comparison ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agg))
    ax.bar(x, agg["regret_mean"], yerr=agg["regret_std"], capsize=6,
           color=["#DD8452", "#8172B2", "#C44E52"][:len(agg)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["display_name"])
    ax.set_ylabel("Relative Regret (%) — mean \u00b1 std across seeds")
    ax.set_title(f"Multi-Seed Robustness ({len(N_SEEDS)} seeds per model)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "multiseed_regret_errorbar.png", dpi=150)
    plt.close()

    total_hours = (time.time() - overall_start) / 3600
    print(f"\nTotal wall-clock time: {total_hours:.2f} hours")
    print(f"Saved: results/multiseed_summary.csv")
    print(f"Saved: results/multiseed_aggregate.csv")
    print(f"Saved: figures/multiseed_regret_errorbar.png")


if __name__ == "__main__":
    main()