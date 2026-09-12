"""
build_sarima_baseline.py  (redesigned)

Fits SARIMA ONCE on the training period, then for each test day builds a
small, FIXED-SIZE window of real recent history (default: 90 days
immediately preceding that day) and applies the already-estimated
parameters via .smooth(). no re-optimization, no .append(), no
ever-growing model. Memory cost per day is bounded and identical whether
it's test day 2 or test day 180, which is what the previous .append() based
design could not guarantee.

This also means SARIMA's parameters are fixed after the initial fit and
never updated during testing, the same way the other five models in
this project use fixed weights throughout evaluation, not something
unique to SARIMA.

Requires: pip install statsmodels

Run:
    python build_sarima_baseline.py --test-days 5      # quick timing test
    python build_sarima_baseline.py --test-days 185    # full test period
"""

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

HORIZON = 96              # one day, 15-min slots
SEASONAL_PERIOD = 96      # daily seasonality only; weekly (672) not attempted
CONTEXT_DAYS = 90         # fixed-size window of real history before each test day

# Anchored to this script's own location, not the current working directory --
# matches score_sarima_baseline.py's approach, so both scripts always agree on
# where results/ actually is, regardless of which folder you run them from.
ROOT = Path(__file__).parent.parent
EXPORT_PATH = ROOT / "results" / "sarima_export.csv"
OUTPUT_PATH = ROOT / "results" / "sarima_forecasts.csv"
CACHE_PATH  = ROOT / "results" / "sarima_init_params.pkl"


def load_full_series():
    """One continuous real-valued series spanning train+val+test, so a
    context window for any test day can pull whatever real prior data
    exists, regardless of which split it originally came from."""
    df = pd.read_csv(EXPORT_PATH, index_col="timestamp", parse_dates=True)
    full = df["demand_mw"].sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full, df


import pickle


def main(n_test_days: int, train_days_limit: int, refit: bool):
    full_series, df = load_full_series()
    train = df[df["split"] == "train"]["demand_mw"].asfreq("15min")
    if train_days_limit:
        cutoff = train.index.max() - pd.Timedelta(days=train_days_limit)
        train = train[train.index >= cutoff]
        print(f"Initial fit uses the most recent {train_days_limit} training days "
              f"({len(train):,} points).")

    test_df = df[df["split"] == "test"]
    test_days = sorted(set(test_df.index.date))[:n_test_days]
    print(f"Forecasting {len(test_days)} test day(s), starting {test_days[0]}")
    print(f"(Full test period has {len(set(test_df.index.date))} days total.)")

    init_model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(0, 1, 0, SEASONAL_PERIOD),
        enforce_stationarity=False, enforce_invertibility=False,
    )

    if CACHE_PATH.exists() and not refit:
        print(f"\n[checkpoint] loading cached fitted parameters from {CACHE_PATH} "
              f"(pass --refit to force re-estimation)...")
        with open(CACHE_PATH, "rb") as f:
            params = pickle.load(f)
    else:
        print("\n[checkpoint] fitting SARIMAX ONCE on the training window to estimate "
              "parameters (order=(1,1,1), seasonal_order=(0,1,0,96))...")
        t0 = time.time()
        init_fit = init_model.fit(disp=False, low_memory=True)
        params = init_fit.params
        print(f"[checkpoint] initial fit done, {time.time()-t0:.1f}s.")
        CACHE_PATH.parent.mkdir(exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(params, f)
        print(f"[checkpoint] cached fitted parameters to {CACHE_PATH}, future runs "
              f"will reuse these instantly instead of re-fitting, unless refit is passed.")

    print("These parameters are now FIXED for every test day below  "
          "no re-optimization happens in the loop.")

    results = []
    for i, day in enumerate(test_days):
        day_t0 = time.time()
        day_start = pd.Timestamp(day)
        day_index = pd.date_range(day_start, periods=HORIZON, freq="15min")

        true_values = test_df["demand_mw"].reindex(day_index)
        if true_values.isna().any():
            n_present = true_values.notna().sum()
            print(f"  Day {i+1}/{len(test_days)} ({day}): partial ({n_present}/{HORIZON} "
                  f"slots) -- skipped, not enough slots for a fair comparison")
            continue

        # Fixed-size context window: the CONTEXT_DAYS immediately before this day.
        # Real gaps in the underlying data are expected here (this project has
        # documented real missing periods), statsmodels' state-space models
        # handle NaN in endog natively as genuinely missing observations, so we
        # pass gaps through rather than skip the day, and only bail out if the
        # window is mostly missing (not usably informative at all).
        context_end   = day_start - pd.Timedelta(minutes=15)
        context_start = context_end - pd.Timedelta(days=CONTEXT_DAYS)
        context = full_series.loc[context_start:context_end].asfreq("15min")

        expected_len = CONTEXT_DAYS * HORIZON
        pct_missing = context.isna().mean() if len(context) > 0 else 1.0
        if len(context) < expected_len * 0.5 or pct_missing > 0.5:
            print(f"  Day {i+1}/{len(test_days)} ({day}): context window is more than "
                  f"half missing ({context.isna().sum()}/{len(context)}) -- skipped, "
                  f"not enough real data to be informative")
            continue
        elif context.isna().any():
            print(f"    (context has {context.isna().sum()}/{len(context)} missing slots "
                  f"passing through to SARIMAX's native missing-data handling, not skipping)")

        day_model = SARIMAX(
            context, order=(1, 1, 1), seasonal_order=(0, 1, 0, SEASONAL_PERIOD),
            enforce_stationarity=False, enforce_invertibility=False,
        )
        day_fit = day_model.smooth(params)
        forecast = day_fit.get_forecast(steps=HORIZON).predicted_mean

        for step, (ts, pred, true) in enumerate(zip(day_index, forecast, true_values)):
            results.append({"date": day, "step": step, "timestamp": ts,
                             "true_demand": true, "sarima_forecast": pred})

        print(f"  Day {i+1}/{len(test_days)} ({day}): forecast + scored, "
              f"{time.time()-day_t0:.1f}s (context: {len(context)} points)")

    out = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out):,} forecast rows -> {OUTPUT_PATH}")
    print("This is forecasts only, see score_sarima_baseline.py for scoring "
          "against the real dispatch/regret code.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-days", type=int, default=5,
                         help="Number of test days to forecast (default 5, a quick timing test)")
    parser.add_argument("--train-days-limit", type=int, default=730,
                         help="Only use the most recent N days of training data for the "
                              "initial parameter fit (default 730, ~2 years). Pass 0 for "
                              "the full training set.")
    parser.add_argument("--refit", action="store_true",
                         help="Force re-estimating parameters even if a cached fit exists.")
    args = parser.parse_args()
    main(args.test_days, args.train_days_limit, args.refit)