#!/usr/bin/env python3
"""
train_msa_neopt_multiseed.py
MSA-NeOpt (MSABlock + SPO+) across 3 random seeds, own process.

Per supervisor feedback:
  - Full per-epoch timing + metrics logged
  - Resumable: skips any seed whose checkpoint already exists

Output:
  models/multiseed/msa_neopt_seed{N}_best.pt
  results/multiseed/msa_neopt_seed{N}_epochlog.csv
  results/multiseed/msa_neopt_multiseed_summary.csv
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
from step03_train_neopt import (
    EirGridDataset, greedy_peak, spo_plus_loss,
    IN_FEATURES, SEQ_LEN, PRED_LEN, BATCH_SIZE,
    LR_WARMUP, LR_DFL, WARMUP_EPOCHS, FINETUNE_EPOCHS, DEVICE,
)
from step05_train_msa_neopt import MSABlock

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models" / "multiseed"
RES_DIR   = ROOT / "results" / "multiseed"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS      = [0, 1, 2]   # reduced from 5, due to time constraint
PATIENCE     = 10          
                            
N_LP_SAMPLES = 100
BAT_CAP, BAT_POWER = 0.5, 0.25


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
    ckpt_path = MODEL_DIR / f"msa_neopt_seed{seed}_best.pt"
    if ckpt_path.exists():
        print(f"  [skip] msa_neopt seed={seed} — checkpoint already exists")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        return {"seed": seed, "best_val_regret": ckpt["val_regret"],
                "best_epoch": ckpt["epoch"], "train_minutes": None}

    torch.manual_seed(seed)
    np.random.seed(seed)

    model     = MSABlock().to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR_WARMUP)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    print(f"\n  {'='*50}\n  MSA-NeOpt — seed={seed}\n  {'='*50}")
    t0 = time.time()
    epoch_log = []

    criterion = nn.MSELoss()
    for epoch in range(1, WARMUP_EPOCHS + 1):
        e0 = time.time()
        model.train()
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimiser.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        val_mse_total, val_mae_total, val_regret_total, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                val_mse_total += criterion(pred, y).item() * len(X)
                val_mae_total += (pred - y).abs().mean().item() * len(X)
                pred_c = pred.clamp(0.0, 2.0)
                peak_hat = greedy_peak(pred_c)
                peak_oracle = greedy_peak(y)
                regret = ((peak_hat - peak_oracle) / peak_oracle.clamp(min=1e-6)).clamp(min=0)
                val_regret_total += regret.mean().item() * len(X)
                n += len(X)
        val_mse, val_mae, val_regret = val_mse_total/n, val_mae_total/n, val_regret_total/n
        scheduler.step(val_mse)

        e_time = time.time() - e0
        epoch_log.append({"phase": "warmup", "epoch": epoch, "val_mse": val_mse,
                          "val_mae": val_mae, "val_regret": val_regret,
                          "epoch_seconds": e_time})
        if epoch % 5 == 0 or epoch == 1:
            print(f"    [warmup] epoch {epoch:>3}  val_regret={val_regret:.6f}  ({e_time:.1f}s)")

    for pg in optimiser.param_groups:
        pg["lr"] = LR_DFL
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    best_val_regret, best_epoch, patience_counter = float("inf"), 0, 0

    for epoch_ft in range(1, FINETUNE_EPOCHS + 1):
        epoch = WARMUP_EPOCHS + epoch_ft
        e0 = time.time()
        model.train()
        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimiser.zero_grad()
            pred = model(X)
            pred_c = pred.clamp(0.0, 2.0)
            sol_true = greedy_peak(y)
            loss = spo_plus_loss(pred_c, y, sol_true)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        val_mse_total, val_mae_total, val_regret_total, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                pred = model(X)
                val_mse_total += nn.functional.mse_loss(pred, y).item() * len(X)
                val_mae_total += (pred - y).abs().mean().item() * len(X)
                pred_c = pred.clamp(0.0, 2.0)
                peak_hat = greedy_peak(pred_c)
                peak_oracle = greedy_peak(y)
                regret = ((peak_hat - peak_oracle) / peak_oracle.clamp(min=1e-6)).clamp(min=0)
                val_regret_total += regret.mean().item() * len(X)
                n += len(X)
        val_mse, val_mae, val_regret = val_mse_total/n, val_mae_total/n, val_regret_total/n
        scheduler.step(val_regret)

        e_time = time.time() - e0
        epoch_log.append({"phase": "dfl", "epoch": epoch, "val_mse": val_mse,
                          "val_mae": val_mae, "val_regret": val_regret,
                          "epoch_seconds": e_time})
        if epoch_ft % 5 == 0 or epoch_ft == 1:
            print(f"    [dfl]    epoch {epoch:>3}  val_regret={val_regret:.6f}  ({e_time:.1f}s)")

        if val_regret < best_val_regret:
            best_val_regret, best_epoch, patience_counter = val_regret, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                       "val_regret": val_regret, "val_mse": val_mse, "seed": seed}, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch} (best {best_epoch})")
                break

    train_minutes = (time.time() - t0) / 60
    pd.DataFrame(epoch_log).to_csv(RES_DIR / f"msa_neopt_seed{seed}_epochlog.csv", index=False)
    print(f"  Done in {train_minutes:.1f} min. Best val_regret={best_val_regret:.6f} (epoch {best_epoch})")

    return {"seed": seed, "best_val_regret": best_val_regret,
            "best_epoch": best_epoch, "train_minutes": train_minutes}


@torch.no_grad()
def evaluate_on_test(seed: int, test_dl) -> dict:
    ckpt_path = MODEL_DIR / f"msa_neopt_seed{seed}_best.pt"
    model = MSABlock().to(DEVICE)
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
    print("MSA-NeOpt — MSA-NeOpt Multi-Seed Training")
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
    summary.to_csv(RES_DIR / "msa_neopt_multiseed_summary.csv", index=False)

    print("\n" + "=" * 60)
    print("MSA-NeOpt — mean +/- std across seeds")
    print("=" * 60)
    print(f"  Test regret:        {summary['test_rel_regret'].mean():.4f} +/- {summary['test_rel_regret'].std():.4f}")
    print(f"  Peak reduction:     {summary['test_peak_reduction'].mean():.4f} +/- {summary['test_peak_reduction'].std():.4f}")
    print(f"  LP regret:          {summary['test_lp_regret'].mean():.4f} +/- {summary['test_lp_regret'].std():.4f}")
    print(f"  Train time (min):   {summary['train_minutes'].mean():.1f} +/- {summary['train_minutes'].std():.1f}")
    print(f"\nSaved: results/multiseed/msa_neopt_multiseed_summary.csv")


if __name__ == "__main__":
    main()
