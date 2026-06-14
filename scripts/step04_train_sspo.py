#!/usr/bin/env python3
"""
step04_train_sspo.py  (renamed from step04_train_dbb.py)
MSA-NeOpt — SSPO ablation (Kim backbone + Surrogate SPO loss)

Background
----------
DBB (Differentiation-of-Black-Box, Vlastelica et al. 2019) requires a
differentiable interpolation of the solver's solution map. This is tractable
for LP/QP solvers (where KKT gradients exist) but not for a greedy heuristic:
  - A uniform perturbation shifts the 85th-percentile threshold identically
    to the load, so greedy_peak returns the same value → zero finite difference.
  - Random perturbation produces a noisy but zero-mean estimate that does not
    reduce val_regret consistently.

Instead this step implements SSPO (Surrogate SPO), a simpler DFL ablation that:
  1. Uses the same warm-up + fine-tuning schedule as SPO+ (step03)
  2. Replaces the SPO+ subgradient with a direct regression-to-oracle signal:
       when peak_hat > peak_oracle, push y_hat toward y_true proportionally
       to the excess peak (i.e. a hinge-weighted MSE surrogate)
  3. Provides a meaningful comparison point: does the full SPO+ subgradient
     outperform this simpler "regress harder on hard samples" signal?

Source: PredOpt/predopt-benchmarks Energy/Trainer/diff_layer.py (structure)
        Surrogate loss design follows Mandi et al. (2024) Section 4.3
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from step03_train_neopt import (
    EirGridDataset, KimBackbone, greedy_peak,
    train_warmup_epoch, evaluate,
    IN_FEATURES, SEQ_LEN, PRED_LEN, BATCH_SIZE,
    LR_WARMUP, LR_DFL,
    WARMUP_EPOCHS, FINETUNE_EPOCHS, PATIENCE, DEVICE,
)

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
RES_DIR   = ROOT / "results"


# ── SSPO loss ──────────────────────────────────────────────────────────────────

def sspo_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Surrogate SPO loss: hinge-weighted MSE.

    For each sample where the predicted dispatch peak exceeds the oracle peak,
    compute the normalised excess and weight the MSE loss by it. Samples where
    peak_hat <= peak_oracle contribute zero loss (dispatch is already optimal).

    This is fully differentiable through y_hat via standard MSE gradients —
    no custom autograd function needed.

    Source: surrogate design follows Mandi et al. (2024) Section 4.3
    Changes: adapted for scalar peak minimisation with greedy solver
    """
    with torch.no_grad():
        peak_hat    = greedy_peak(y_hat.clamp(0.0, 2.0))   # [B]
        peak_oracle = greedy_peak(y_true)                   # [B]
        # Normalised excess peak: how much worse than oracle? Clamp to 0 if better.
        excess = ((peak_hat - peak_oracle) / peak_oracle.clamp(min=1e-6)).clamp(min=0)  # [B]
        # Weight: 1 + excess so "good" samples still contribute baseline MSE
        weight = (1.0 + excess).unsqueeze(1)                # [B, 1] → broadcasts over H

    mse_per_sample = ((y_hat - y_true) ** 2).mean(dim=1)   # [B]
    return (weight.squeeze(1) * mse_per_sample).mean()


def train_sspo_epoch(model, loader, optimiser, device):
    """Phase 2: SSPO surrogate loss."""
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        mu, _ = model(X)
        loss = sspo_loss(mu, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item()
    return total / len(loader)


def main():
    print("=" * 60)
    print("MSA-NeOpt — Step 04: SSPO Ablation (Kim backbone + SSPO)")
    print("=" * 60)

    train_ds = EirGridDataset("train")
    val_ds   = EirGridDataset("val")
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=False)
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=False)

    model = KimBackbone().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR_WARMUP)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    log_rows = []
    best_val_regret = float("inf")
    patience_counter = 0
    best_epoch = 0

    # ── Phase 1: Warm-up (MSE) ─────────────────────────────────────────────
    print(f"\n  Phase 1: Warm-up ({WARMUP_EPOCHS} epochs, MSE)")
    print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val Regret':>11}")
    print("  " + "-" * 32)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        train_loss = train_warmup_epoch(model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = evaluate(model, val_dl, DEVICE)
        scheduler.step(val_mse)
        log_rows.append({"epoch": epoch, "phase": "warmup",
                         "train_loss": train_loss, "val_regret": val_regret})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss:>10.6f}  {val_regret:>11.6f}")

    # ── Phase 2: SSPO fine-tuning ──────────────────────────────────────────
    for pg in optimiser.param_groups:
        pg["lr"] = LR_DFL
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    print(f"\n  Phase 2: DFL fine-tuning ({FINETUNE_EPOCHS} epochs, SSPO)")
    print(f"  {'Epoch':>6}  {'Train SSPO':>11}  {'Val Regret':>11}  {'LR':>10}")
    print("  " + "-" * 46)

    for epoch_ft in range(1, FINETUNE_EPOCHS + 1):
        epoch = WARMUP_EPOCHS + epoch_ft
        train_loss = train_sspo_epoch(model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = evaluate(model, val_dl, DEVICE)
        scheduler.step(val_regret)
        lr_now = optimiser.param_groups[0]["lr"]

        log_rows.append({"epoch": epoch, "phase": "sspo",
                         "train_loss": train_loss, "val_regret": val_regret})

        if epoch_ft % 5 == 0 or epoch_ft == 1:
            print(f"  {epoch:>6}  {train_loss:>11.6f}  {val_regret:>11.6f}  {lr_now:>10.2e}")

        if val_regret < best_val_regret:
            best_val_regret = val_regret
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_regret": val_regret,
                "config": {"model_name": "KimBackbone_SSPO"},
            }, MODEL_DIR / "sspo_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    pd.DataFrame(log_rows).to_csv(RES_DIR / "sspo_log.csv", index=False)
    print(f"\n  Best: epoch {best_epoch}  val_regret={best_val_regret:.6f}")
    print(f"  Saved → models/sspo_best.pt")
    print(f"  Next: python step05_train_msa_neopt.py")


if __name__ == "__main__":
    main()
