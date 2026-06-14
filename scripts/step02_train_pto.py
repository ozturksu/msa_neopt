#!/usr/bin/env python3
"""
step02_train_pto.py
MSA-NeOpt — PTO baseline (MultiDeT, Wang et al. 2022)
Source:  https://github.com/wangc1073/multidet
Changes: (1) input adapted from 13->19 features, seq_len 72->672 slots
         (2) single 96-slot output head replaces three-decoder structure
         (3) Transformer encoder layers unchanged

Trains with MSE loss (predict-then-optimise paradigm).
Saves best checkpoint to models/pto_best.pt
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
RES_DIR   = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

IN_FEATURES  = 19
SEQ_LEN      = 672
PRED_LEN     = 96
D_MODEL      = 64
N_HEADS      = 8
N_LAYERS     = 2
DROPOUT      = 0.1
BATCH_SIZE   = 32
LR           = 1e-3
WARMUP_EPOCHS   = 15
FINETUNE_EPOCHS = 60
TOTAL_EPOCHS    = WARMUP_EPOCHS + FINETUNE_EPOCHS
PATIENCE     = 10
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


class EirGridDataset(Dataset):
    def __init__(self, split: str):
        self.X = torch.from_numpy(np.load(DATA_DIR / f"X_{split}.npy")).float()
        self.y = torch.from_numpy(np.load(DATA_DIR / f"y_{split}.npy")).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class _Encoder(nn.Module):
    """Unchanged TransformerEncoder from Wang et al. 2022."""
    def __init__(self, num_layers, d_model, n_heads, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.enc  = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x): return self.norm(self.enc(x))


class MultiDeT_Adapted(nn.Module):
    """
    Wang et al. (2022) MultiDeT adapted for EirGrid / MSA-NeOpt baseline.
    Input:  [B, 672, 19]
    Output: [B, 96]
    """
    def __init__(self, in_features=IN_FEATURES, seq_len=SEQ_LEN,
                 pred_len=PRED_LEN, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.enc  = _Encoder(n_layers, d_model, n_heads, dropout)
        self.head = nn.Linear(seq_len * d_model, pred_len)
    def forward(self, x):
        z = self.enc(self.proj(x))
        z = z.reshape(z.size(0), -1)
        return self.head(z)


def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimiser.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, preds, trues = 0.0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total += criterion(pred, y).item() * len(X)
        preds.append(pred.cpu()); trues.append(y.cpu())
    preds = torch.cat(preds); trues = torch.cat(trues)
    return total / len(loader.dataset), (preds - trues).abs().mean().item()


def main():
    print("=" * 60)
    print("MSA-NeOpt — Step 02: PTO Baseline (MultiDeT, Wang et al. 2022)")
    print("=" * 60)

    train_dl = DataLoader(EirGridDataset("train"), BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=False)
    val_dl   = DataLoader(EirGridDataset("val"),   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

    model = MultiDeT_Adapted().to(DEVICE)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    dummy = torch.rand(4, SEQ_LEN, IN_FEATURES).to(DEVICE)
    out   = model(dummy)
    assert out.shape == (4, PRED_LEN)
    print(f"  Forward pass OK: {list(dummy.shape)} -> {list(out.shape)}")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, "min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    log_rows = []
    best_val, patience_counter, best_epoch = float("inf"), 0, 0

    print(f"\n  Training {TOTAL_EPOCHS} epochs (MSE throughout — PTO baseline)")
    print(f"  {'Epoch':>6}  {'Train MSE':>10}  {'Val MSE':>10}  {'Val MAE':>9}")
    print("  " + "-" * 45)

    for epoch in range(1, TOTAL_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_dl, optimiser, criterion, DEVICE)
        val_loss, val_mae = evaluate(model, val_dl, criterion, DEVICE)
        scheduler.step(val_loss)
        log_rows.append({"epoch": epoch, "train_mse": train_loss,
                         "val_mse": val_loss, "val_mae": val_mae})
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>6}  {train_loss:>10.6f}  {val_loss:>10.6f}  {val_mae:>9.6f}")
        if val_loss < best_val:
            best_val = val_loss; best_epoch = epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_mse": val_loss,
                        "config": {"in_features": IN_FEATURES, "seq_len": SEQ_LEN,
                                   "pred_len": PRED_LEN, "d_model": D_MODEL,
                                   "n_heads": N_HEADS, "n_layers": N_LAYERS,
                                   "model_name": "MultiDeT_Adapted"}},
                       MODEL_DIR / "pto_best.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best was {best_epoch})")
                break

    pd.DataFrame(log_rows).to_csv(RES_DIR / "pto_log.csv", index=False)
    print(f"\n  Best: epoch {best_epoch}  val_mse={best_val:.6f}")
    print(f"  Saved -> models/pto_best.pt")
    print(f"  Next:  python scripts/step03_train_neopt.py")


if __name__ == "__main__":
    main()
