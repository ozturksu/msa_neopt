#!/usr/bin/env python3
"""
step03b_train_gru_neopt.py
MSA-NeOpt — GRU-NeOpt (GRU backbone + SPO+ DFL)

Architecture:
  - 2-layer GRU, hidden_size=256, batch_first=True
  - Final hidden state -> Linear(256, 96)
  - Single output head (no uncertainty)

Training schedule:
  Phase 1: 15 warm-up epochs — MSE
  Phase 2: 60 fine-tuning epochs — SPO+

Purpose:
  Tests whether the Transformer attention mechanism is specifically
  necessary for DFL to work, or whether a simpler recurrent model
  trained with SPO+ achieves comparable dispatch quality.

Comparison:
  Step 02   — MultiDeT Transformer + MSE      (no DFL)
  Step 03   — KimBackbone Transformer + SPO+  (DFL, attention-based)
  Step 03b  — GRU + SPO+                      (DFL, recurrent)
  Step 05   — MSABlock CNN + SPO+             (DFL, multi-scale)
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
    EirGridDataset, greedy_peak, spo_plus_loss,
    IN_FEATURES, SEQ_LEN, PRED_LEN, BATCH_SIZE,
    LR_WARMUP, LR_DFL,
    WARMUP_EPOCHS, FINETUNE_EPOCHS, PATIENCE, DEVICE,
)

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
RES_DIR   = ROOT / "results"

# GRU hyperparameters
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
DROPOUT_GRU = 0.1


# ── GRUBackbone ───────────────────────────────────────────────────────────────

class GRUBackbone(nn.Module):
    """
    GRU backbone for GRU-NeOpt.

    A 2-layer GRU that processes the 7-day demand sequence and maps
    the final hidden state to a 96-slot demand forecast.

    Input:  [B, 672, 19]
    Output: [B, 96]

    The GRU processes the sequence step by step, maintaining a hidden
    state that summarises everything seen so far. After 672 steps the
    final hidden state captures the full 7-day history. This is then
    mapped to a 96-slot forecast by a linear head.

    This is fundamentally different from KimBackbone (Transformer) which
    processes all 672 steps in parallel through self-attention, and from
    MSABlock which uses parallel CNN branches. The GRU is sequential and
    recurrent,the simplest possible sequence model that can be trained
    with SPO+.
    """
    def __init__(
        self,
        in_features: int   = IN_FEATURES,
        hidden_size: int   = HIDDEN_SIZE,
        num_layers:  int   = NUM_LAYERS,
        pred_len:    int   = PRED_LEN,
        dropout:     float = DROPOUT_GRU,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 672, 19]
        _, h_n = self.gru(x)
        # h_n: [num_layers, B, hidden_size]
        # take the final layer hidden state only
        last_hidden = h_n[-1]           # [B, hidden_size]
        return self.head(last_hidden)   # [B, pred_len]


# ── Training loops ─────────────────────────────────────────────────────────────

def train_gru_warmup_epoch(model, loader, optimiser, device):
    """Phase 1: MSE warm-up for GRUBackbone."""
    model.train()
    total     = 0.0
    criterion = nn.MSELoss()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item()
    return total / len(loader)


def train_gru_dfl_epoch(model, loader, optimiser, device):
    """Phase 2: SPO+ DFL fine-tuning for GRUBackbone."""
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        pred         = model(X)
        pred_clamped = pred.clamp(0.0, 2.0)
        sol_true     = greedy_peak(y)
        loss         = spo_plus_loss(pred_clamped, y, sol_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate_gru(model, loader, device):
    """Evaluation for GRUBackbone."""
    model.eval()
    mse_fn = nn.MSELoss()
    total_mse, total_mae, total_regret, n = 0.0, 0.0, 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred         = model(X)
        total_mse   += mse_fn(pred, y).item() * len(X)
        total_mae   += (pred - y).abs().mean().item() * len(X)
        peak_hat     = greedy_peak(pred.clamp(0.0, 2.0))
        peak_oracle  = greedy_peak(y)
        regret       = ((peak_hat - peak_oracle)
                        / peak_oracle.clamp(min=1e-6)).clamp(min=0)
        total_regret += regret.mean().item() * len(X)
        n            += len(X)
    return total_mse / n, total_mae / n, total_regret / n


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MSA-NeOpt — Step 03b: GRU-NeOpt (GRU + SPO+)")
    print(f"  hidden_size={HIDDEN_SIZE}  num_layers={NUM_LAYERS}")
    print("=" * 60)

    train_dl = DataLoader(EirGridDataset("train"), BATCH_SIZE,
                          shuffle=True,  num_workers=2, pin_memory=False)
    val_dl   = DataLoader(EirGridDataset("val"),   BATCH_SIZE,
                          shuffle=False, num_workers=2, pin_memory=False)

    model    = GRUBackbone().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Sanity check
    dummy = torch.rand(4, SEQ_LEN, IN_FEATURES).to(DEVICE)
    out   = model(dummy)
    assert out.shape == (4, PRED_LEN), f"Shape mismatch: {out.shape}"
    print(f"  Forward pass OK: {list(dummy.shape)} -> {list(out.shape)}")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR_WARMUP)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    log_rows         = []
    best_val_regret  = float("inf")
    patience_counter = 0
    best_epoch       = 0

    # ── Phase 1: Warm-up (MSE) ─────────────────────────────────────────────
    print(f"\n  Phase 1: MSE warm-up ({WARMUP_EPOCHS} epochs)")
    print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val Regret':>11}  {'Val MAE':>9}")
    print("  " + "-" * 50)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        train_loss                   = train_gru_warmup_epoch(model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = evaluate_gru(model, val_dl, DEVICE)
        scheduler.step(val_mse)
        log_rows.append({"epoch": epoch, "phase": "warmup",
                         "train_loss": train_loss, "val_mse": val_mse,
                         "val_mae": val_mae, "val_regret": val_regret})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss:>10.6f}  "
                  f"{val_regret:>11.6f}  {val_mae:>9.6f}")

    # ── Phase 2: SPO+ DFL fine-tuning ─────────────────────────────────────
    for pg in optimiser.param_groups:
        pg["lr"] = LR_DFL
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    print(f"\n  Phase 2: SPO+ DFL fine-tuning ({FINETUNE_EPOCHS} epochs)")
    print(f"  {'Epoch':>6}  {'SPO+ Loss':>10}  {'Val Regret':>11}"
          f"  {'Val MAE':>9}  {'LR':>10}")
    print("  " + "-" * 57)

    for epoch_ft in range(1, FINETUNE_EPOCHS + 1):
        epoch      = WARMUP_EPOCHS + epoch_ft
        train_loss = train_gru_dfl_epoch(model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = evaluate_gru(model, val_dl, DEVICE)
        scheduler.step(val_regret)
        lr_now = optimiser.param_groups[0]["lr"]

        log_rows.append({"epoch": epoch, "phase": "dfl",
                         "train_loss": train_loss, "val_mse": val_mse,
                         "val_mae": val_mae, "val_regret": val_regret})

        if epoch_ft % 5 == 0 or epoch_ft == 1:
            print(f"  {epoch:>6}  {train_loss:>10.6f}  {val_regret:>11.6f}"
                  f"  {val_mae:>9.6f}  {lr_now:>10.2e}")

        if val_regret < best_val_regret:
            best_val_regret  = val_regret
            best_epoch       = epoch
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_regret":       val_regret,
                "val_mse":          val_mse,
                "config": {
                    "model_name":  "GRUBackbone_SPOPlus",
                    "in_features": IN_FEATURES,
                    "seq_len":     SEQ_LEN,
                    "pred_len":    PRED_LEN,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers":  NUM_LAYERS,
                }
            }, MODEL_DIR / "gru_neopt_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best epoch {best_epoch})")
                break

    pd.DataFrame(log_rows).to_csv(RES_DIR / "gru_neopt_log.csv", index=False)
    print(f"\n  Best: epoch {best_epoch}  val_regret={best_val_regret:.6f}")
    print(f"  Saved -> models/gru_neopt_best.pt")
    print(f"  Next:  python scripts/step06_evaluate.py")


if __name__ == "__main__":
    main()

# ── Quick architecture test ───────────────────────────────────────────────────
# Run: python step03b_train_gru_neopt.py --test
# to verify shapes before running the full training

if __name__ != "__main__":
    pass  # imported as module