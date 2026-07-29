#!/usr/bin/env python3
"""
step01b_prepare_data_powertransform.py
MSA-NeOpt — EirGrid data pipeline, POWER TRANSFORM variant

Added following supervisor feedback to test whether reshaping skewed
input feature distributions (via a power transform) improves input
processing quality, tested independently of the five existing models.

This script is a variant of step01_prepare_data.py. It does NOT modify
step01 or overwrite its output. It reads the same raw EirGrid CSVs and
performs identical feature engineering, but replaces the normalisation
step with:

    Yeo-Johnson power transform  ->  min-max normalisation

instead of step01's:

    min-max normalisation only

Why Yeo-Johnson and not Box-Cox:
  Box-Cox requires strictly positive input values. Several engineered
  features can be zero or negative (demand_diff, generation_ratio when
  generation < demand, snsp before 2021 which is zero-filled). Yeo-Johnson
  handles zero and negative values and reduces to a Box-Cox-equivalent
  transform on strictly positive data, so it is the safer general choice.

Which features are transformed:
  Only the 11 continuous, potentially skewed signals:
    demand_mw, wind_mw, generation_mw, co2_intensity, snsp,
    wind_penetration, generation_ratio,
    demand_lag_1d, demand_lag_2d, demand_lag_1w, demand_diff

  The remaining 8 features are left untouched because a power transform
  is not meaningful for them:
    sin_slot, cos_slot, sin_dow, cos_dow, sin_woy, cos_woy   (already
      bounded in [-1, 1] with circular meaning that a power transform
      would destroy)
    is_weekend, is_holiday                                   (binary
      flags, not continuous distributions)

Output:
  data_pt/  (separate from data/ — step01's output is never touched)
    X_train.npy, X_val.npy, X_test.npy
    y_train.npy, y_val.npy, y_test.npy
    ts_train.npy, ts_val.npy, ts_test.npy
    normalisation_stats.csv       (min/max, post-power-transform, train only)
    power_transformer.pkl         (fitted sklearn PowerTransformer, train only)
    skew_comparison.csv           (before/after skewness per transformed feature)

Both data/ and data_pt/ can be pointed at by any training script via the
DATA_DIR constant, so any of the five existing model scripts can be run
unmodified against this data by copying them and changing one line.
"""

import warnings
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"       # step01's original output — READ ONLY, never written
OUT_DIR  = ROOT / "data_pt"    # this script's output — completely separate

# ── UPDATE THIS PATH to your EirGrid CSV directory (same as step01)
RAW_DIR  = Path("/Users/suley/Desktop/NCI/thesis/eir_grid_data/Downloaded_Data/ROI")


OUT_DIR.mkdir(exist_ok=True)

START_YEAR = 2014
END_YEAR   = 2024

CATEGORIES = {
    "demandactual":     "demand_mw",
    "generationactual": "generation_mw",
    "windactual":       "wind_mw",
    "co2intensity":     "co2_intensity",
    "SnspALL":          "snsp",
}

TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
LOOKBACK_SLOTS = 672
HORIZON_SLOTS  = 96

# Features that receive the power transform.
# Order matches engineer_features() output columns 1-5 (raw), 6-7 (derived),
# and 16-19 (lags/diff). Cyclical (8-13) and calendar (14-15) are excluded.
POWER_TRANSFORM_COLS = [
    "demand_mw", "wind_mw", "generation_mw", "co2_intensity", "snsp",
    "wind_penetration", "generation_ratio",
    "demand_lag_1d", "demand_lag_2d", "demand_lag_1w", "demand_diff",
]


# ── Load CSVs (identical to step01) ─────────────────────────────────────────

def load_category(category: str, col_name: str) -> pd.Series:
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        yy   = str(year)[2:]
        path = RAW_DIR / f"ROI_{category}_{yy}_Eirgrid.csv"
        if not path.exists():
            continue
        if path.stat().st_size < 10:
            continue
        try:
            raw = pd.read_csv(path, header=None)
            if raw.empty or raw.shape[1] == 0:
                continue
            if raw.shape[1] >= 4:
                df = raw[[0, 3]].copy()
            else:
                df = raw[[0, 1]].copy()
            df.columns = ["ts", "value"]
            df["ts"]    = pd.to_datetime(df["ts"], dayfirst=True, errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["ts", "value"])
            if df.empty:
                continue
            s = df.set_index("ts")["value"]
            frames.append(s)
        except Exception as e:
            print(f"    Warning: {path.name}: {e}")
    if not frames:
        return pd.Series(dtype=float, name=col_name)
    s = pd.concat(frames).sort_index()
    s.name = col_name
    s = s[~s.index.duplicated(keep="first")]
    print(f"    {col_name:20s}  {len(s):>8,} rows  "
          f"{s.index[0].date()} to {s.index[-1].date()}")
    return s


def load_all() -> pd.DataFrame:
    # Reuses step01's cache if present — same raw signals, only the
    # normalisation step differs, so re-parsing all CSVs is unnecessary.
    cache = DATA_DIR / "eirgrid_raw.parquet"
    if cache.exists():
        print(f"  [cache, shared with step01] {cache}")
        return pd.read_parquet(cache)

    print(f"  Reading CSVs from {RAW_DIR} ...")
    idx    = pd.date_range(f"{START_YEAR}-01-01",
                           f"{END_YEAR}-12-31 23:45", freq="15min")
    merged = pd.DataFrame(index=idx)
    for cat, col in CATEGORIES.items():
        s = load_category(cat, col)
        if not s.empty:
            merged = merged.join(s.reindex(idx), how="left")
        else:
            merged[col] = 0.0
            print(f"    {col:22s}  zero-filled (no data found)")
    if "demand_mw" not in merged.columns or merged["demand_mw"].isna().all():
        raise RuntimeError(f"No demand data loaded from {RAW_DIR}")
    return merged


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    print("  Preprocessing...")
    df = raw.copy()
    if "snsp" in df.columns:
        df["snsp"] = df["snsp"].fillna(0.0)
    df = df.ffill(limit=16)
    before = len(df)
    df     = df[df["demand_mw"].notna()]
    lost   = before - len(df)
    if lost:
        print(f"    Dropped {lost:,} rows with unrecoverable missing demand")
    print(f"    Clean shape: {df.shape}")
    return df


def _irish_holidays(idx: pd.DatetimeIndex) -> np.ndarray:
    holidays = set()
    for year in range(START_YEAR, END_YEAR + 2):
        for m, d in [(1,1),(3,17),(5,5),(6,2),(8,4),(10,27),(12,25),(12,26)]:
            try:
                holidays.add(datetime(year, m, d).date())
            except ValueError:
                pass
    return np.array([1.0 if d.date() in holidays else 0.0 for d in idx])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Engineering 19 features...")
    fe = df.copy()
    if "snsp" not in fe.columns:
        fe["snsp"] = 0.0
    fe["wind_penetration"] = (fe["wind_mw"] / fe["generation_mw"].replace(0, np.nan))
    fe["generation_ratio"] = (fe["generation_mw"] / fe["demand_mw"].replace(0, np.nan))
    slot = fe.index.hour * 4 + fe.index.minute // 15
    fe["sin_slot"] = np.sin(2 * np.pi * slot / 96)
    fe["cos_slot"] = np.cos(2 * np.pi * slot / 96)
    dow = fe.index.dayofweek
    fe["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    fe["cos_dow"] = np.cos(2 * np.pi * dow / 7)
    woy = fe.index.isocalendar().week.values.astype(float)
    fe["sin_woy"] = np.sin(2 * np.pi * woy / 52)
    fe["cos_woy"] = np.cos(2 * np.pi * woy / 52)
    fe["is_weekend"] = (dow >= 5).astype(float)
    fe["is_holiday"] = _irish_holidays(fe.index)
    fe["demand_lag_1d"] = fe["demand_mw"].shift(96)
    fe["demand_lag_2d"] = fe["demand_mw"].shift(192)
    fe["demand_lag_1w"] = fe["demand_mw"].shift(672)
    fe["demand_diff"]   = fe["demand_mw"].diff()
    fe = fe.dropna()
    cols = [
        "demand_mw", "wind_mw", "generation_mw", "co2_intensity", "snsp",
        "wind_penetration", "generation_ratio",
        "sin_slot", "cos_slot", "sin_dow", "cos_dow", "sin_woy", "cos_woy",
        "is_weekend", "is_holiday",
        "demand_lag_1d", "demand_lag_2d", "demand_lag_1w", "demand_diff",
    ]
    for c in cols:
        if c not in fe.columns:
            fe[c] = 0.0
    fe = fe[cols]
    assert fe.shape[1] == 19, f"Expected 19 features, got {fe.shape[1]}"
    print(f"    Feature matrix: {fe.shape}")
    return fe


def split(fe: pd.DataFrame):
    n  = len(fe)
    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    tr, va, te = fe.iloc[:i1], fe.iloc[i1:i2], fe.iloc[i2:]
    print(f"    Train  {tr.index[0].date()} to {tr.index[-1].date()}  ({len(tr):,} slots)")
    print(f"    Val    {va.index[0].date()} to {va.index[-1].date()}  ({len(va):,} slots)")
    print(f"    Test   {te.index[0].date()} to {te.index[-1].date()}  ({len(te):,} slots)")
    return tr, va, te


# ── Power transform + normalise (THE PART THAT DIFFERS FROM STEP01) ────────

def power_transform_and_normalise(tr_fe, va_fe, te_fe):
    """
    Two-stage transform, fit on TRAINING DATA ONLY, applied to all splits:
      1. Yeo-Johnson power transform on the 11 skewed continuous features
      2. Min-max normalisation on all 19 features (same final step as step01)

    Returns transformed (train, val, test) DataFrames, the fitted
    PowerTransformer, the min-max stats dict, and a before/after skewness
    comparison DataFrame for the transformed columns.
    """
    print("  Fitting Yeo-Johnson power transform on training data...")

    skew_before = {c: float(skew(tr_fe[c].values)) for c in POWER_TRANSFORM_COLS}

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(tr_fe[POWER_TRANSFORM_COLS].values)

    def apply_pt(fe: pd.DataFrame) -> pd.DataFrame:
        out = fe.copy()
        out[POWER_TRANSFORM_COLS] = pt.transform(fe[POWER_TRANSFORM_COLS].values)
        return out

    tr_pt = apply_pt(tr_fe)
    va_pt = apply_pt(va_fe)
    te_pt = apply_pt(te_fe)

    skew_after = {c: float(skew(tr_pt[c].values)) for c in POWER_TRANSFORM_COLS}

    skew_df = pd.DataFrame({
        "feature":         POWER_TRANSFORM_COLS,
        "skew_before_pt":  [skew_before[c] for c in POWER_TRANSFORM_COLS],
        "skew_after_pt":   [skew_after[c]  for c in POWER_TRANSFORM_COLS],
    })
    skew_df["abs_skew_reduction"] = (
        skew_df["skew_before_pt"].abs() - skew_df["skew_after_pt"].abs()
    )

    print("\n  Skewness before -> after power transform (0 = perfectly symmetric):")
    for _, row in skew_df.iterrows():
        print(f"    {row['feature']:20s}  {row['skew_before_pt']:>7.3f} -> "
              f"{row['skew_after_pt']:>7.3f}")

    # Stage 2: standard min-max normalisation, same as step01, fit on
    # the POWER-TRANSFORMED training data only.
    print("\n  Applying min-max normalisation (post-power-transform)...")
    stats = {c: (tr_pt[c].min(), tr_pt[c].max()) for c in tr_pt.columns}

    def minmax(fe: pd.DataFrame) -> pd.DataFrame:
        out = fe.copy()
        for c in fe.columns:
            lo, hi = stats[c]
            out[c] = (fe[c] - lo) / (hi - lo) if hi > lo else 0.0
        return out

    tr_n = minmax(tr_pt)
    va_n = minmax(va_pt)
    te_n = minmax(te_pt)

    return tr_n, va_n, te_n, pt, stats, skew_df


def make_windows(fe: pd.DataFrame):
    arr = fe.values.astype(np.float32)
    X, y, ts = [], [], []
    for i in range(LOOKBACK_SLOTS, len(arr) - HORIZON_SLOTS + 1):
        X.append(arr[i - LOOKBACK_SLOTS : i])
        y.append(arr[i : i + HORIZON_SLOTS, 0])
        ts.append(fe.index[i])
    return np.stack(X), np.stack(y), np.array(ts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MSA-NeOpt — Step 01b: Data Pipeline (Power Transform variant)")
    print(f"  Output: {OUT_DIR}  (step01's data/ directory is untouched)")
    print("=" * 60)

    raw   = load_all()
    clean = preprocess(raw)
    fe    = engineer_features(clean)

    tr_fe, va_fe, te_fe = split(fe)

    tr_n, va_n, te_n, pt, stats, skew_df = power_transform_and_normalise(
        tr_fe, va_fe, te_fe)

    # Save fitted transformer and stats
    with open(OUT_DIR / "power_transformer.pkl", "wb") as f:
        pickle.dump(pt, f)
    pd.DataFrame(stats, index=["min", "max"]).T.to_csv(
        OUT_DIR / "normalisation_stats.csv")
    skew_df.to_csv(OUT_DIR / "skew_comparison.csv", index=False)

    # Save sliding-window arrays — same shapes/dtypes as step01, so any
    # existing model script works unmodified if DATA_DIR is repointed here.
    for name, normed in [("train", tr_n), ("val", va_n), ("test", te_n)]:
        X, y, ts = make_windows(normed)
        np.save(OUT_DIR / f"X_{name}.npy",  X)
        np.save(OUT_DIR / f"y_{name}.npy",  y)
        np.save(OUT_DIR / f"ts_{name}.npy", ts)
        print(f"  {name:5s}  X={X.shape}  y={y.shape}")

    print(f"\n  Done — {OUT_DIR} ready.")
    print(f"  Original data/ (step01 output) was not modified.")
    print(f"  Saved: power_transformer.pkl, normalisation_stats.csv, skew_comparison.csv")
    print(f"\n  Mean |skew| before: {skew_df['skew_before_pt'].abs().mean():.3f}")
    print(f"  Mean |skew| after:  {skew_df['skew_after_pt'].abs().mean():.3f}")


if __name__ == "__main__":
    main()