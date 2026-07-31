#!/usr/bin/env python3
"""
train_pto_multiseed.py
MSA-NeOpt — PTO (MultiDeT) across 3 random seeds, run as its OWN process.

Per supervisor feedback: run each model separately (not looped inside
one script alongside other architectures, this was the likely cause
of the ~8x slowdown observed when step07 imported from three model
files simultaneously). This file imports ONLY from step02_train_pto.py.

Also per supervisor feedback:
  - Every epoch's timing and metrics logged, plus total training time,
    for direct use in report tables
  - Resumable: skips any seed whose checkpoint already exists

PTO uses MSE loss only, no DFL phase, so there is only one training
loop here (not the two-phase warmup+DFL schedule used by the other
four models).

Output:
  models/multiseed/pto_seed{N}_best.pt
  results/multiseed/pto_seed{N}_epochlog.csv
  results/multiseed/pto_multiseed_summary.csv
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from step02_train_pto import (
    EirGridDataset, MultiDeT_Adapted,
    IN_FEATURES, SEQ_LEN, PRED_LEN, BATCH_SIZE, LR, DEVICE,
)

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "multiseed"
RES_DIR   = ROOT / "results" / "multiseed"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS       = [0, 1, 2]   # reduced from 5 due to time constraint
PATIENCE      = 10          # [UNIFORM] matches NeOpt, GRU-NeOpt, SSPO, and
                            
TOTAL_EPOCHS  = 75
N_LP_SAMPLES  = 100
BAT_CAP, BAT_POWER = 0.5, 0.25


def greedy_peak(y: torch.Tensor) -> torch.Tensor:
    B, H = y.shape
    sols = []
    for i in range(B):
        load = y[i].detach().cpu().numpy()
        soc  = 0.0
        disp = np.zeros(H)
        thr  = np.percentile(load, 85)
        for t, p in enumerate(load):
            if p > thr and soc > 0:
                d = min(BAT_POWER, soc, p - thr)
                disp[t] = -d; soc -= d
            elif p < thr * 0.6 and soc < BAT_CAP:
                c = min(BAT_POWER, BAT_CAP - soc)
                disp[t] = c; soc += c
        sols.append((load + disp).max())
    return torch.tensor(sols, dtype=torch.float32, device=y.device)


def lp_peak_shave(load, bat_cap=BAT_CAP, bat_power=BAT_POWER, eta_c=0.95, eta_d=0.95):
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


def train_one_seed(seed: int, train_dl, val_dl) -> dict:
    ckpt_path = MODEL_DIR / f"pto_seed{seed}_best.pt"
    if ckpt_path.exists():
        print(f"  [skip] pto seed={seed} — checkpoint already exists")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        return {"seed": seed, "best_val_mse": ckpt["val_mse"],
                "best_epoch": ckpt["epoch"], "train_minutes": None}

    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = MultiDeT_Adapted().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    print(f"\n  {'='*50}\n  PTO — seed={seed}\n  {'='*50}")
    t0 = time.time()
    epoch_log = []
    best_val_mse, best_epoch, patience_counter = float("inf"), 0, 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        e0 = time.time()
        model.train()
        total = 0.0
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimiser.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item() * len(X)
        train_mse = total / len(train_dl.dataset)

        model.eval()
        val_total, preds, trues = 0.0, [], []
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                val_total += criterion(pred, y).item() * len(X)
                preds.append(pred.cpu()); trues.append(y.cpu())
        val_mse = val_total / len(val_dl.dataset)
        val_mae = (torch.cat(preds) - torch.cat(trues)).abs().mean().item()
        scheduler.step(val_mse)

        e_time = time.time() - e0
        epoch_log.append({"epoch": epoch, "train_mse": train_mse,
                          "val_mse": val_mse, "val_mae": val_mae,
                          "epoch_seconds": e_time})
        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:>3}  val_mse={val_mse:.6f}  ({e_time:.1f}s)")

        if val_mse < best_val_mse:
            best_val_mse, best_epoch, patience_counter = val_mse, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                       "val_mse": val_mse, "seed": seed}, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch} (best {best_epoch})")
                break

    train_minutes = (time.time() - t0) / 60
    pd.DataFrame(epoch_log).to_csv(RES_DIR / f"pto_seed{seed}_epochlog.csv", index=False)
    print(f"  Done in {train_minutes:.1f} min. Best val_mse={best_val_mse:.6f} (epoch {best_epoch})")

    return {"seed": seed, "best_val_mse": best_val_mse,
            "best_epoch": best_epoch, "train_minutes": train_minutes}


@torch.no_grad()
def evaluate_on_test(seed: int, test_dl) -> dict:
    ckpt_path = MODEL_DIR / f"pto_seed{seed}_best.pt"
    model = MultiDeT_Adapted().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds, trues = [], []
    for X, y in test_dl:
        X = X.to(DEVICE)
        pred = model(X)
        preds.append(pred.cpu()); trues.append(y)
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
    lp_peaks = np.array([lp_peak_shave(trues[i]) for i in idx])
    lp_regret = np.maximum(0, (p_hat[idx] - lp_peaks) / np.maximum(lp_peaks, 1e-6)) * 100

    return {"seed": seed, "test_mse": mse, "test_mae": mae,
            "test_rel_regret": float(rel_regret.mean()),
            "test_peak_reduction": float(peak_reduction.mean()),
            "test_lp_regret": float(lp_regret.mean())}


def main():
    print("=" * 60)
    print("MSA-NeOpt — PTO Multi-Seed Training")
    print(f"  Seeds: {N_SEEDS}")
    print("=" * 60)

    train_ds = EirGridDataset("train")
    val_ds   = EirGridDataset("val")
    test_ds  = EirGridDataset("test")
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, num_workers=0)

    train_results = [train_one_seed(s, train_dl, val_dl) for s in N_SEEDS]

    print("\nEvaluating all seeds on test set...")
    eval_results = [evaluate_on_test(s, test_dl) for s in N_SEEDS]

    summary = pd.merge(pd.DataFrame(train_results), pd.DataFrame(eval_results), on="seed")
    summary.to_csv(RES_DIR / "pto_multiseed_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("PTO — mean +/- std across seeds")
    print("=" * 60)
    print(f"  Test regret:        {summary['test_rel_regret'].mean():.4f} +/- {summary['test_rel_regret'].std():.4f}")
    print(f"  Peak reduction:     {summary['test_peak_reduction'].mean():.4f} +/- {summary['test_peak_reduction'].std():.4f}")
    print(f"  LP regret:          {summary['test_lp_regret'].mean():.4f} +/- {summary['test_lp_regret'].std():.4f}")
    print(f"  Train time (min):   {summary['train_minutes'].mean():.1f} +/- {summary['train_minutes'].std():.1f}")
    print(f"\nSaved: results/multiseed/pto_multiseed_summary.csv")


if __name__ == "__main__":
    main()
