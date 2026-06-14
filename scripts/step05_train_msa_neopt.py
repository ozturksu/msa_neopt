#!/usr/bin/env python3
"""
step05_train_msa_neopt.py
MSA-NeOpt — Proposed model (MSABlock + SPO+)
This is the main contribution of the thesis.

MSABlock architecture (Debnath et al. 2026, Figure 1 + Keras notebook):
  - 4 parallel nn.Conv1d branches: kernels {4, 16, 96, 192}
    = 1-hour, 4-hour, 24-hour, 48-hour windows at 15-min resolution
  - AdaptiveAvgPool1d → fixed 96-slot output per branch
  - Cross-scale nn.MultiheadAttention fuses branch outputs
  - Single linear head → 96-slot demand forecast

Changes from Debnath source:
  (1) LSTM layers removed
  (2) 4 parallel Conv1d branches replace sequential CNN
  (3) PyTorch port (original is Keras .h5 + deployment app)
  (4) Wrapped in SPO+ DFL training loop

Training schedule:
  Phase 1: 15 warm-up epochs (MSE)
  Phase 2: 60 DFL fine-tuning epochs (SPO+)

Source: Debnath et al. (2026) IEEE Access 14:13423-13444
        DOI: 10.1109/ACCESS.2026.3656545

Fixes applied vs. original draft:
  [FIX-1] Import LR_WARMUP / LR_DFL instead of LR (LR no longer exists in step03)
  [FIX-2] Logging: per-batch average (total / n_batches) in both training loops
  [FIX-3] Predictions clamped to [0, 2] before greedy_peak in DFL loop
  [FIX-4] LR reset to LR_DFL before Phase 2, fresh scheduler created
  [FIX-5] pin_memory=False (no MPS support on Apple Silicon)
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
    EirGridDataset, greedy_peak, spo_plus_loss, evaluate,
    IN_FEATURES, SEQ_LEN, PRED_LEN, BATCH_SIZE,
    LR_WARMUP, LR_DFL,                          # [FIX-1]
    WARMUP_EPOCHS, FINETUNE_EPOCHS, PATIENCE, DEVICE,
)

ROOT      = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
RES_DIR   = ROOT / "results"

# MSABlock hyperparameters
D_BRANCH     = 64
KERNELS      = [4, 16, 96, 192]   # 1h, 4h, 24h, 48h at 15-min resolution
N_HEADS_ATTN = 4


# ── MSABlock ───────────────────────────────────────────────────────────────────

class MSABlock(nn.Module):
    """
    Multi-Scale Attention CNN backbone for MSA-NeOpt.

    Debnath et al. (2026) Figure 1 — PyTorch port with 3 changes:
      (1) LSTM layers removed
      (2) 4 parallel Conv1d branches: kernels k ∈ {4, 16, 96, 192}
      (3) Cross-scale MultiheadAttention fuses branch outputs

    Input:  [B, 672, 19]
    Output: [B, 96]
    """
    def __init__(
        self,
        in_features: int  = IN_FEATURES,
        d_branch:    int  = D_BRANCH,
        kernels:     list = None,
        pred_len:    int  = PRED_LEN,
        n_heads:     int  = N_HEADS_ATTN,
    ):
        super().__init__()
        if kernels is None:
            kernels = KERNELS

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_features, d_branch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(d_branch),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(pred_len),
            )
            for k in kernels
        ])

        fused_dim = d_branch * len(kernels)   # 64 * 4 = 256

        self.attn = nn.MultiheadAttention(fused_dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(fused_dim)
        self.head = nn.Linear(fused_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.permute(0, 2, 1)                        # [B, in_features, seq_len]
        branch_outs = [b(x_t) for b in self.branches]   # each [B, d_branch, pred_len]
        z = torch.cat(branch_outs, dim=1)                # [B, fused_dim, pred_len]
        z = z.permute(0, 2, 1)                           # [B, pred_len, fused_dim]
        a, _ = self.attn(z, z, z)
        z = self.norm(z + a)
        return self.head(z).squeeze(-1)                  # [B, pred_len]


# ── Training loops ─────────────────────────────────────────────────────────────

def train_msa_warmup_epoch(model, loader, optimiser, device):
    """Phase 1: MSE warm-up for MSABlock (single output head)."""
    model.train()
    total = 0.0
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
    return total / len(loader)               # [FIX-2] per-batch average


def train_msa_dfl_epoch(model, loader, optimiser, device):
    """Phase 2: SPO+ DFL fine-tuning for MSABlock."""
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        pred = model(X)
        pred_clamped = pred.clamp(0.0, 2.0)  # [FIX-3] keep solver stable
        sol_true = greedy_peak(y)
        loss = spo_plus_loss(pred_clamped, y, sol_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item()
    return total / len(loader)               # [FIX-2] per-batch average


@torch.no_grad()
def evaluate_msa(model, loader, device):
    """Evaluation for MSABlock (single output head)."""
    model.eval()
    mse_fn = nn.MSELoss()
    total_mse, total_mae, total_regret, n = 0.0, 0.0, 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_mse += mse_fn(pred, y).item() * len(X)
        total_mae += (pred - y).abs().mean().item() * len(X)
        peak_hat    = greedy_peak(pred.clamp(0.0, 2.0))  # [FIX-3] consistent
        peak_oracle = greedy_peak(y)
        regret = ((peak_hat - peak_oracle) / peak_oracle.clamp(min=1e-6)).clamp(min=0)
        total_regret += regret.mean().item() * len(X)
        n += len(X)
    return total_mse / n, total_mae / n, total_regret / n


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MSA-NeOpt — Step 05: MSA-NeOpt (Proposed Model)")
    print(f"  MSABlock kernels: {KERNELS}")
    print(f"  Branch channels:  {D_BRANCH}  →  fused: {D_BRANCH * len(KERNELS)}")
    print("=" * 60)

    train_ds = EirGridDataset("train")
    val_ds   = EirGridDataset("val")
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=False)   # [FIX-5]
    val_dl   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=False)

    model = MSABlock().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    dummy = torch.rand(4, SEQ_LEN, IN_FEATURES).to(DEVICE)
    out = model(dummy)
    assert out.shape == (4, PRED_LEN), f"Shape mismatch: {out.shape}"
    print(f"  Forward pass OK: {list(dummy.shape)} → {list(out.shape)}")
    print(f"  Kernels {KERNELS} = [1h, 4h, 24h, 48h] at 15-min resolution")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR_WARMUP)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    log_rows = []
    best_val_regret = float("inf")
    patience_counter = 0
    best_epoch = 0

    # ── Phase 1: Warm-up (MSE) ─────────────────────────────────────────────
    print(f"\n  Phase 1: MSE warm-up ({WARMUP_EPOCHS} epochs)")
    print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val Regret':>11}  {'Val MAE':>9}")
    print("  " + "-" * 50)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        train_loss = train_msa_warmup_epoch(model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = evaluate_msa(model, val_dl, DEVICE)
        scheduler.step(val_mse)
        log_rows.append({"epoch": epoch, "phase": "warmup",
                         "train_loss": train_loss, "val_mse": val_mse,
                         "val_mae": val_mae, "val_regret": val_regret})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss:>10.6f}  {val_regret:>11.6f}  {val_mae:>9.6f}")

    # ── Phase 2: SPO+ DFL fine-tuning ──────────────────────────────────────
    # [FIX-4] Reset LR for DFL phase
    for pg in optimiser.param_groups:
        pg["lr"] = LR_DFL
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    print(f"\n  Phase 2: SPO+ DFL fine-tuning ({FINETUNE_EPOCHS} epochs)")
    print(f"  {'Epoch':>6}  {'SPO+ Loss':>10}  {'Val Regret':>11}  {'Val MAE':>9}  {'LR':>10}")
    print("  " + "-" * 57)

    for epoch_ft in range(1, FINETUNE_EPOCHS + 1):
        epoch = WARMUP_EPOCHS + epoch_ft
        train_loss = train_msa_dfl_epoch(model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = evaluate_msa(model, val_dl, DEVICE)
        scheduler.step(val_regret)
        lr_now = optimiser.param_groups[0]["lr"]

        log_rows.append({"epoch": epoch, "phase": "dfl",
                         "train_loss": train_loss, "val_mse": val_mse,
                         "val_mae": val_mae, "val_regret": val_regret})

        if epoch_ft % 5 == 0 or epoch_ft == 1:
            print(f"  {epoch:>6}  {train_loss:>10.6f}  {val_regret:>11.6f}"
                  f"  {val_mae:>9.6f}  {lr_now:>10.2e}")

        if val_regret < best_val_regret:
            best_val_regret = val_regret
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_regret": val_regret,
                "val_mse": val_mse,
                "config": {
                    "model_name": "MSABlock_SPOPlus",
                    "in_features": IN_FEATURES,
                    "seq_len": SEQ_LEN,
                    "pred_len": PRED_LEN,
                    "kernels": KERNELS,
                    "d_branch": D_BRANCH,
                    "n_heads_attn": N_HEADS_ATTN,
                }
            }, MODEL_DIR / "msa_neopt_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    pd.DataFrame(log_rows).to_csv(RES_DIR / "msa_neopt_log.csv", index=False)
    print(f"\n  Best: epoch {best_epoch}  val_regret={best_val_regret:.6f}")
    print(f"  Saved → models/msa_neopt_best.pt")
    print(f"  Next: python step06_evaluate.py")


if __name__ == "__main__":
    main()
