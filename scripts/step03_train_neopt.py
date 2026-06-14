#!/usr/bin/env python3
"""
step03_train_neopt.py
MSA-NeOpt — NeOpt DFL baseline (Kim et al. 2025, Section III-A)
Built from paper specification (no public repo exists).

Architecture:
  - PatchTST-style patch embedding: patch_len=16, stride=8
  - 3-layer, 8-head TransformerEncoder, d_model=128
  - Mean + log-variance output heads (aleatoric uncertainty)
  - Trained with SPO+ gradient loss (Decision-Focused Learning)

Training schedule:
  Phase 1: 15 warm-up epochs — MSE on mean head
  Phase 2: 60 fine-tuning epochs — SPO+ loss (DFL)

Source: Kim et al. (2025) IEEE Trans. Industrial Informatics 21:9286-9296
        DOI: 10.1109/TII.2025.3597951

Changes from source (PredOpt/predopt-benchmarks diff_layer.py):
  [FIX-1] SPO+ forward: normalised by sol_true → loss stays 0-1 scale
  [FIX-2] SPO+ backward: gradient sign NEGATED for peak MINIMISATION
          Original SPO+ subgradient x*(c) - x*(2c_hat-c) is for cost
          maximisation. For minimisation the sign must flip so that
          over-forecasting (sol_spo > sol_true) produces a positive
          gradient that pushes y_hat downward toward y_true.
  [FIX-3] DFL phase LR reset to 1e-4 before fine-tuning begins
  [FIX-4] Predictions clamped to [0, 2] before greedy_peak to prevent
          out-of-distribution values destabilising the solver
  [FIX-5] Logging: per-batch average (total / n_batches)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
RES_DIR   = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

IN_FEATURES  = 19
SEQ_LEN      = 672
PRED_LEN     = 96
PATCH_LEN    = 16
STRIDE       = 8
D_MODEL      = 128
N_HEADS      = 8
N_LAYERS     = 3
DROPOUT      = 0.1
BATCH_SIZE   = 32
LR_WARMUP    = 1e-3   # Phase 1
LR_DFL       = 1e-4   # Phase 2 — lower for SPO+ stability [FIX-3]
WARMUP_EPOCHS   = 15
FINETUNE_EPOCHS = 60
PATIENCE     = 10
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Battery parameters for greedy peak-shave solver
BAT_CAP   = 0.5    # pu-h
BAT_POWER = 0.25   # pu

print(f"Device: {DEVICE}")


# ── Dataset ────────────────────────────────────────────────────────────────────

class EirGridDataset(Dataset):
    def __init__(self, split: str):
        self.X = torch.from_numpy(np.load(DATA_DIR / f"X_{split}.npy")).float()
        self.y = torch.from_numpy(np.load(DATA_DIR / f"y_{split}.npy")).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Greedy peak-shave solver ───────────────────────────────────────────────────

def greedy_peak(y: torch.Tensor) -> torch.Tensor:
    """
    Greedy battery dispatch: charge below 60% of 85th-percentile,
    discharge above 85th-percentile.
    Returns peak demand after dispatch. Shape: [B].
    Source: adapted from PredOpt/predopt-benchmarks Energy/Trainer/diff_layer.py
    """
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
                disp[t] = -d
                soc     -= d
            elif p < thr * 0.6 and soc < BAT_CAP:
                c = min(BAT_POWER, BAT_CAP - soc)
                disp[t] = c
                soc     += c
        sols.append((load + disp).max())
    return torch.tensor(sols, dtype=torch.float32, device=y.device)


# ── SPO+ layer ─────────────────────────────────────────────────────────────────

class SPOPlusFunction(torch.autograd.Function):
    """
    SPO+ decision-focused loss for scalar peak MINIMISATION.
    Source: PredOpt/predopt-benchmarks Energy/Trainer/diff_layer.py

    Changes from source:
      (1) solver argument removed — greedy_peak used directly
      (2) forward normalised by sol_true [FIX-1]
      (3) backward gradient sign negated for minimisation [FIX-2]
      (4) gradient spread over horizon: [B] → [B, H]
    """
    @staticmethod
    def forward(ctx, y_hat, y_true, sol_true):
        sol_hat = greedy_peak(y_hat)
        ctx.save_for_backward(y_hat, y_true, sol_true)
        ref = sol_true.clamp(min=1e-6)
        return ((sol_hat - sol_true).clamp(min=0) / ref).mean()

    @staticmethod
    def backward(ctx, grad_output):
        y_hat, y_true, sol_true = ctx.saved_tensors
        sol_spo = greedy_peak(2 * y_hat - y_true)
        # [FIX-2] Negated vs. original: (sol_spo - sol_true) is POSITIVE when
        # the model over-forecasts, giving a gradient that pushes y_hat DOWN.
        grad = (sol_spo - sol_true).unsqueeze(1).expand(-1, y_hat.shape[1]) / y_hat.shape[1]
        return grad * grad_output, None, None


def spo_plus_loss(y_hat, y_true, sol_true):
    return SPOPlusFunction.apply(y_hat, y_true, sol_true)


# ── Kim et al. backbone ────────────────────────────────────────────────────────

class KimBackbone(nn.Module):
    """
    Kim et al. (2025) Sec III-A, Equations (3)-(5).
    PatchTST-style: overlapping patches → TransformerEncoder → mean + log-var heads.
    Input:  [B, 672, 19]
    Output: (mean [B, 96], log_var [B, 96])
    """
    def __init__(
        self,
        in_features: int = IN_FEATURES,
        seq_len:     int = SEQ_LEN,
        patch_len:   int = PATCH_LEN,
        stride:      int = STRIDE,
        d_model:     int = D_MODEL,
        n_heads:     int = N_HEADS,
        n_layers:    int = N_LAYERS,
        pred_len:    int = PRED_LEN,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride    = stride
        n_patches = (seq_len - patch_len) // stride + 1

        self.proj   = nn.Linear(patch_len * in_features, d_model)
        self.pos    = nn.Parameter(torch.zeros(1, n_patches, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True
        )
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        flat_dim = n_patches * d_model
        self.h_mean = nn.Linear(flat_dim, pred_len)
        self.h_lvar = nn.Linear(flat_dim, pred_len)

    def forward(self, x: torch.Tensor):
        B, L, F = x.shape
        p = x.unfold(1, self.patch_len, self.stride)
        p = p.contiguous().view(B, p.size(1), self.patch_len * F)
        z = self.tf(self.proj(p) + self.pos)
        z = z.reshape(B, -1)
        return self.h_mean(z), self.h_lvar(z)


# ── Training loops ─────────────────────────────────────────────────────────────

def train_warmup_epoch(model, loader, optimiser, device):
    """Phase 1: MSE on mean head only."""
    model.train()
    total = 0.0
    criterion = nn.MSELoss()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        mu, _ = model(X)
        loss = criterion(mu, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item()
    return total / len(loader)   # per-batch average [FIX-5]


def train_dfl_epoch(model, loader, optimiser, device):
    """Phase 2: SPO+ decision-focused loss."""
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        mu, _ = model(X)
        # [FIX-4] Clamp to valid demand range before solver
        mu_clamped = mu.clamp(0.0, 2.0)
        sol_true = greedy_peak(y)
        loss = spo_plus_loss(mu_clamped, y, sol_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item()
    return total / len(loader)   # per-batch average [FIX-5]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse_fn = nn.MSELoss()
    total_mse, total_mae, total_regret = 0.0, 0.0, 0.0
    n = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        mu, _ = model(X)
        total_mse += mse_fn(mu, y).item() * len(X)
        total_mae += (mu - y).abs().mean().item() * len(X)

        # Normalised regret: (peak_hat - peak_oracle) / peak_oracle
        mu_clamped  = mu.clamp(0.0, 2.0)   # [FIX-4] consistent with train
        peak_hat    = greedy_peak(mu_clamped)
        peak_oracle = greedy_peak(y)
        regret = ((peak_hat - peak_oracle) / peak_oracle.clamp(min=1e-6)).clamp(min=0)
        total_regret += regret.mean().item() * len(X)
        n += len(X)

    return total_mse / n, total_mae / n, total_regret / n


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MSA-NeOpt — Step 03: NeOpt (Kim et al. 2025, SPO+)")
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

    dummy = torch.rand(4, SEQ_LEN, IN_FEATURES).to(DEVICE)
    mu, lv = model(dummy)
    assert mu.shape == (4, PRED_LEN)
    print(f"  Forward pass OK: {list(dummy.shape)} → mean {list(mu.shape)}")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR_WARMUP)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    log_rows = []
    best_val_regret = float("inf")
    patience_counter = 0
    best_epoch = 0

    # ── Phase 1: Warm-up (MSE) ─────────────────────────────────────────────
    print(f"\n  Phase 1: Warm-up ({WARMUP_EPOCHS} epochs, MSE)")
    print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}"
          f"  {'Val MAE':>9}  {'Val Regret':>11}")
    print("  " + "-" * 58)

    for epoch in range(1, WARMUP_EPOCHS + 1):
        train_loss = train_warmup_epoch(model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = evaluate(model, val_dl, DEVICE)
        scheduler.step(val_mse)
        log_rows.append({"epoch": epoch, "phase": "warmup",
                         "train_loss": train_loss, "val_mse": val_mse,
                         "val_mae": val_mae, "val_regret": val_regret})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss:>10.6f}  {val_mse:>10.6f}"
                  f"  {val_mae:>9.6f}  {val_regret:>11.6f}")

    # ── Phase 2: DFL fine-tuning (SPO+) ────────────────────────────────────
    # [FIX-3] Reset LR — SPO+ gradients need a smaller step than MSE
    for pg in optimiser.param_groups:
        pg["lr"] = LR_DFL
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, "min", factor=0.5, patience=5)

    print(f"\n  Phase 2: DFL fine-tuning ({FINETUNE_EPOCHS} epochs, SPO+)")
    print(f"  {'Epoch':>6}  {'Train SPO+':>11}  {'Val Regret':>11}"
          f"  {'Val MAE':>9}  {'LR':>10}")
    print("  " + "-" * 58)

    for epoch_ft in range(1, FINETUNE_EPOCHS + 1):
        epoch = WARMUP_EPOCHS + epoch_ft
        train_loss = train_dfl_epoch(model, train_dl, optimiser, DEVICE)
        val_mse, val_mae, val_regret = evaluate(model, val_dl, DEVICE)
        scheduler.step(val_regret)
        lr_now = optimiser.param_groups[0]["lr"]

        log_rows.append({"epoch": epoch, "phase": "dfl",
                         "train_loss": train_loss, "val_mse": val_mse,
                         "val_mae": val_mae, "val_regret": val_regret})

        if epoch_ft % 5 == 0 or epoch_ft == 1:
            print(f"  {epoch:>6}  {train_loss:>11.6f}  {val_regret:>11.6f}"
                  f"  {val_mae:>9.6f}  {lr_now:>10.2e}")

        if val_regret < best_val_regret:
            best_val_regret = val_regret
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_regret": val_regret,
                "val_mse":    val_mse,
                "config": {
                    "model_name": "KimBackbone_SPOPlus",
                    "in_features": IN_FEATURES, "seq_len": SEQ_LEN,
                    "pred_len": PRED_LEN, "patch_len": PATCH_LEN,
                    "stride": STRIDE, "d_model": D_MODEL,
                    "n_heads": N_HEADS, "n_layers": N_LAYERS,
                }
            }, MODEL_DIR / "neopt_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    pd.DataFrame(log_rows).to_csv(RES_DIR / "neopt_log.csv", index=False)
    print(f"\n  Best: epoch {best_epoch}  val_regret={best_val_regret:.6f}")
    print(f"  Saved → models/neopt_best.pt")
    print(f"  Next: python step04_train_dbb.py")


if __name__ == "__main__":
    main()
