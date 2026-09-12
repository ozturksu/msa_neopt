"""
generate_prediction_plots.py

Produces real predicted-vs-actual demand plots and error plots for all five
models, using the checkpoints and test arrays already saved from training.
Nothing here is simulated, every number comes from an actual forward pass
of actual trained models on actual test set.

Run from the project root:
    python scripts/generate_prediction_plots.py

Outputs (in figures/):
    prediction_vs_actual_<model>.png   -- one representative test week
    error_analysis_<model>.png         -- residual histogram + error-over-time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")


# 1. Load test data (already saved by step01_prepare_data.py)

X_test  = np.load(DATA_DIR / "X_test.npy")   # [4288, 672, 19]
y_test  = np.load(DATA_DIR / "y_test.npy")   # [4288, 96]
ts_test = np.load(DATA_DIR / "ts_test.npy", allow_pickle=True)  # [4288] timestamps

# ---------------------------------------------------------------------------
# 2. Denormalise back to real MW using the saved min/max stats
#    (X_test / y_test are min-max normalised 0-1; a plot in raw 0-1 values
#    is not interpretable to a reader, so we invert it here)
# ---------------------------------------------------------------------------
stats = pd.read_csv(DATA_DIR / "normalisation_stats.csv", index_col=0)
demand_min, demand_max = stats.loc["demand_mw", ["min", "max"]]

def denorm_demand(arr_normalised):
    return arr_normalised * (demand_max - demand_min) + demand_min

# ---------------------------------------------------------------------------
# 3. Pick one representative test week: 7 consecutive, NON-overlapping
#    daily samples. Consecutive INDEX positions in X_test/y_test are only
#    one 15-min slot apart (not one day apart), so we must step by 96 to
#    get 7 genuinely different, back-to-back days rather than 7 heavily
#    overlapping windows.
# ---------------------------------------------------------------------------
START_IDX = 0  # change this to pick a different week, must satisfy START_IDX + 6*96 < len(X_test)
week_indices = [START_IDX + i * 96 for i in range(7)]

def get_week(arr_2d, indices):
    """Stitch 7 daily 96-step arrays into one 672-step week."""
    return np.concatenate([arr_2d[i] for i in indices])

y_true_week = denorm_demand(get_week(y_test, week_indices))
week_start_ts = ts_test[week_indices[0]]


# 4. MODEL IMPORTS 
sys.path.insert(0, "scripts")

MODELS = {}

# PTO (MultiDeT) returns a single tensor [B, 96]
from step02_train_pto import MultiDeT_Adapted
MODELS["PTO"] = (MultiDeT_Adapted(), MODELS_DIR / "pto_best.pt", "single")

# NeOpt (Kim backbone) confirmed: returns (mean, log_var), both [B, 96]
from step03_train_neopt import KimBackbone
MODELS["NeOpt"] = (KimBackbone(), MODELS_DIR / "neopt_best.pt", "mean_logvar")

# GRU-NeOpt returns a single tensor [B, 96]
from step03b_train_gru_neopt import GRUBackbone
MODELS["GRU-NeOpt"] = (GRUBackbone(), MODELS_DIR / "gru_neopt_best.pt", "single")

# SSPO confirmed: step04 defines no new class, reuses KimBackbone,
#     so it also returns (mean, log_var)
MODELS["SSPO"] = (KimBackbone(), MODELS_DIR / "sspo_best.pt", "mean_logvar")

# MSA-NeOpt returns a single tensor [B, 96]
from step05_train_msa_neopt import MSABlock
MODELS["MSA-NeOpt"] = (MSABlock(), MODELS_DIR / "msa_neopt_best.pt", "single")


# 5. Run each model on the same test week, plot predicted vs actual + errors

X_week_tensor = torch.from_numpy(
    np.stack([X_test[i] for i in week_indices])
).float().to(DEVICE)  # [7, 672, 19]

x_axis = np.arange(672) * 15 / 60  # hours into the week, for the plot x-axis

for name, (model, ckpt_path, output_type) in MODELS.items():
    if not ckpt_path.exists():
        print(f"  Skipping {name}: checkpoint not found: {ckpt_path}")
        continue

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  {name}: loaded checkpoint from epoch {checkpoint.get(chr(39)+chr(101)+chr(112)+chr(111)+chr(99)+chr(104)+chr(39), chr(63))}")

    with torch.no_grad():
        output = model(X_week_tensor)
        if output_type == "mean_logvar":
            y_pred_norm, log_var = output
            y_pred_norm = y_pred_norm.cpu().numpy()
        else:
            y_pred_norm = output.cpu().numpy()

    y_pred_week = denorm_demand(np.concatenate(y_pred_norm))
    errors = y_pred_week - y_true_week

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x_axis, y_true_week, label="Actual demand", color="#1B2A4A", linewidth=1.5)
    ax.plot(x_axis, y_pred_week, label="Predicted demand", color="#E8A33D", linewidth=1.3, linestyle="--")
    ax.set_xlabel("Hours into test week")
    ax.set_ylabel("Demand (MW)")
    ax.set_title(f"{name}: Predicted vs Actual Demand, One Test Week (starting {week_start_ts})")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"prediction_vs_actual_{name.replace(' ', '_')}.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(errors, bins=40, color="#2C7DA0", edgecolor="white")
    axes[0].set_xlabel("Prediction error (MW)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Error distribution")
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].plot(x_axis, errors, color="#B02A2A", linewidth=1)
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_xlabel("Hours into test week")
    axes[1].set_ylabel("Prediction error (MW)")
    axes[1].set_title("Error over time")
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle(f"{name}: Error Analysis")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"error_analysis_{name.replace(' ', '_')}.png", dpi=200)
    plt.close()

    print(f"  {name}: saved both plots. Mean abs error = {np.abs(errors).mean():.2f} MW")

print("\nDone. Check figures/ for prediction_vs_actual_*.png and error_analysis_*.png")
