"""
score_sarima_baseline.py

Stage 2 of the SARIMA baseline: takes Stage 1's forecasts
(results/sarima_forecasts.csv, raw MW) and scores them using the EXACT
same compute_metrics()/lp_peak_shave() functions used for the other five
models, imported directly from step06_evaluate.py rather than
reimplemented.

Critical step this script handles: compute_metrics operates on
NORMALISED data (BAT_CAP=0.5, BAT_POWER=0.25, clip bounds [0,2] are all
normalised-scale constants). sarima_export.csv has raw MW values. Feeding
raw MW into the battery model directly would silently produce meaningless
numbers, a 0.5-capacity battery against a ~3,000+ MW load is negligible,
not an error, just wrong. This script normalises using the SAME
normalisation_stats.csv min/max used everywhere else in this project
before scoring.

Run from the project root (same conda env, predopt310):
    python scripts/score_sarima_baseline.py

Requires results/sarima_forecasts.csv to already exist (run
build_sarima_baseline.py first).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from step06_evaluate import compute_metrics, lp_peak_shave  

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
RES_DIR    = ROOT / "results"
FORECASTS  = RES_DIR / "sarima_forecasts.csv"
SUMMARY    = RES_DIR / "summary_table.csv"

HORIZON = 96


def normalise(arr, lo, hi):
    return (arr - lo) / (hi - lo)


def main():
    if not FORECASTS.exists():
        raise FileNotFoundError(
            f"{FORECASTS} not found, run build_sarima_baseline.py first."
        )

    long_df = pd.read_csv(FORECASTS, parse_dates=["timestamp"])
    n_days = long_df["date"].nunique()
    print(f"Loaded {len(long_df):,} forecast rows across {n_days} test day(s).")

    if n_days < 5:
        print(f"NOTE: only {n_days} day(s) of forecasts present, this looks "
              f"like output from a --test-days quick run, not the full test "
              f"period. Regret/peak-reduction numbers from this few days are "
              f"not yet comparable to Table 4's full-test-set means. Re-run "
              f"build_sarima_baseline.py with the full day count first if "
              f"this is meant to be a final result, not a timing check.")

    # Reshape long format (one row per day+step) into [n_days, 96] arrays,
    # matching the shape compute_metrics expects (same as y_test.npy).
    true_wide = long_df.pivot(index="date", columns="step", values="true_demand").sort_index()
    pred_wide = long_df.pivot(index="date", columns="step", values="sarima_forecast").sort_index()
    assert true_wide.shape[1] == HORIZON and pred_wide.shape[1] == HORIZON, \
        f"Expected {HORIZON} steps per day, got {true_wide.shape[1]}/{pred_wide.shape[1]} -- check Stage 1's output."

    trues_mw = true_wide.to_numpy()
    preds_mw = pred_wide.to_numpy()

    # Normalise using the SAME stats used for every other model in this project.
    stats = pd.read_csv(DATA_DIR / "normalisation_stats.csv", index_col=0)
    d_min, d_max = stats.loc["demand_mw", ["min", "max"]]
    print(f"Normalising with demand_mw min={d_min}, max={d_max} (from normalisation_stats.csv)")

    trues_norm = normalise(trues_mw, d_min, d_max)
    preds_norm = normalise(preds_mw, d_min, d_max)

    print(f"\nScoring {trues_norm.shape[0]} days using the real compute_metrics()...")
    metrics = compute_metrics(preds_norm, trues_norm, use_lp=True, n_lp_samples=min(200, trues_norm.shape[0]))

    print(f"\n{'='*50}")
    print(f"SARIMA baseline results")
    print(f"{'='*50}")
    print(f"  MSE:              {metrics['mse']:.6f}")
    print(f"  MAE:              {metrics['mae']:.6f}")
    print(f"  Rel. Regret (%):  {metrics['rel_regret_mean']:.4f} \u00b1 {metrics['rel_regret_std']:.4f}")
    print(f"  Abs. Regret (pu): {metrics['abs_regret_mean']:.4f}")
    print(f"  Peak Reduction:   {metrics['peak_reduction']:.4f}%")
    if metrics["lp_regret_arr"] is not None:
        print(f"  LP Regret (%):    {metrics['lp_regret_arr'].mean():.4f} "
              f"(n={len(metrics['lp_regret_arr'])} subsample)")

    # Append to a copy of summary_table.csv, if present, for direct comparison
    row = {
        "Model":              "SARIMA (baseline)",
        "MSE":                f"{metrics['mse']:.6f}",
        "MAE":                f"{metrics['mae']:.6f}",
        "Rel. Regret (%)":    f"{metrics['rel_regret_mean']:.4f} \u00b1 {metrics['rel_regret_std']:.4f}",
        "Abs. Regret (pu)":   f"{metrics['abs_regret_mean']:.4f}",
        "Peak Reduction (%)": f"{metrics['peak_reduction']:.4f}",
    }
    out_path = RES_DIR / "summary_table_with_sarima.csv"
    if SUMMARY.exists():
        existing = pd.read_csv(SUMMARY)
        combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        combined = pd.DataFrame([row])
        print(f"NOTE: {SUMMARY} not found, saving SARIMA's row alone, "
              f"not merged with the other four models.")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined table -> {out_path}")


if __name__ == "__main__":
    main()