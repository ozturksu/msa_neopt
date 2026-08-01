#!/usr/bin/env python3
"""
step01_data_pipeline.py
MSA-NeOpt — EirGrid data pipeline (reads pre-downloaded CSVs)

CSV format (4 columns, no header):
  col0=EffectiveTime, col1=category, col2=region, col3=Value

Outputs → MSA_NEOPT/data/
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Paths
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR  = Path("/Users/suley/Desktop/NCI/thesis/eir_grid_data/Downloaded_Data/ROI") # change this to the path of the raw data
DATA_DIR.mkdir(exist_ok=True)

# Config 
START_YEAR = 2014
END_YEAR   = 2025

CATEGORIES = {
    "demandactual":     "demand_mw",
    "generationactual": "generation_mw",
    "windactual":       "wind_mw",
    "co2intensity":     "co2_intensity",
    "SnspALL":          "snsp",
}

TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
LOOKBACK_SLOTS = 672   # 7 days × 96 slots
HORIZON_SLOTS  = 96    # 1 day


# Load CSVs

def load_category(category: str, col_name: str) -> pd.Series:
    """
    Read all yearly CSVs for one category.
    Handles two layouts automatically:
      4-col:  timestamp, label, region, value   (demand/gen/wind/co2)
      2-col:  timestamp, value                  (snsp, if present)
    """
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        yy   = str(year)[2:]
        path = RAW_DIR / f"ROI_{category}_{yy}_Eirgrid.csv"
        if not path.exists():
            continue
        # Skip empty files
        if path.stat().st_size < 10:
            continue
        try:
            raw = pd.read_csv(path, header=None)
            if raw.empty or raw.shape[1] == 0:
                continue

            # Auto-detect layout
            if raw.shape[1] >= 4:
                # 4-col layout: col0=ts, col3=value
                df = raw[[0, 3]].copy()
            else:
                # 2-col layout: col0=ts, col1=value
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
          f"{s.index[0].date()} → {s.index[-1].date()}")
    return s


def load_all() -> pd.DataFrame:
    cache = DATA_DIR / "eirgrid_raw.parquet"
    if cache.exists():
        print(f"  [cache] {cache}")
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
            # Zero-fill missing signals (e.g. SNSP before 2021)
            merged[col] = 0.0
            print(f"    {'(' + col + ')':22s}  zero-filled (no data found)")

    if "demand_mw" not in merged.columns or merged["demand_mw"].isna().all():
        raise RuntimeError(
            f"No demand data loaded from {RAW_DIR}\n"
            f"Expected files like: ROI_demandactual_14_Eirgrid.csv"
        )

    merged.to_parquet(cache)
    print(f"  Saved → {cache}  shape={merged.shape}")
    return merged


# Preprocessing

def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    print("  Preprocessing...")
    df = raw.copy()

    # SNSP: zero-fill NaNs (pre-2021 or missing)
    if "snsp" in df.columns:
        df["snsp"] = df["snsp"].fillna(0.0)

    # Forward-fill short gaps ≤ 4 hours (16 slots)
    df = df.ffill(limit=16)

    # Drop rows where demand is still missing after fill
    before = len(df)
    df     = df[df["demand_mw"].notna()]
    lost   = before - len(df)
    if lost:
        print(f"    Dropped {lost:,} rows with unrecoverable missing demand")

    # Per-year missingness report
    for year in range(START_YEAR, END_YEAR + 1):
        yr = df[df.index.year == year]
        if len(yr) == 0:
            continue
        miss = yr["demand_mw"].isna().mean() * 100
        if miss > 2:
            print(f"    Warning: {year} demand missingness = {miss:.1f}%")

    print(f"    Clean shape: {df.shape}")
    return df


# Feature engineering 

def _irish_holidays(idx: pd.DatetimeIndex) -> np.ndarray:
    holidays = set()
    for year in range(START_YEAR, END_YEAR + 2):
        for m, d in [(1,1),(3,17),(5,5),(6,2),(8,4),(10,27),(12,25),(12,26)]:
            try:
                holidays.add(datetime(year, m, d).date())
            except ValueError:
                pass
        if year >= 2023:
            feb = datetime(year, 2, 1)
            while feb.weekday() != 0:
                feb += timedelta(days=1)
            holidays.add(feb.date())
    return np.array([1.0 if d.date() in holidays else 0.0 for d in idx])


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("  Engineering 19 features...")
    fe = df.copy()

    # Ensure snsp column exists even if load_all zero-filled it
    if "snsp" not in fe.columns:
        fe["snsp"] = 0.0

    # Derived signals
    fe["wind_penetration"] = (fe["wind_mw"]
                              / fe["generation_mw"].replace(0, np.nan))
    fe["generation_ratio"] = (fe["generation_mw"]
                              / fe["demand_mw"].replace(0, np.nan))

    # Cyclical time encodings
    slot = fe.index.hour * 4 + fe.index.minute // 15
    fe["sin_slot"] = np.sin(2 * np.pi * slot / 96)
    fe["cos_slot"] = np.cos(2 * np.pi * slot / 96)

    dow = fe.index.dayofweek
    fe["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    fe["cos_dow"] = np.cos(2 * np.pi * dow / 7)

    woy = fe.index.isocalendar().week.values.astype(float)
    fe["sin_woy"] = np.sin(2 * np.pi * woy / 52)
    fe["cos_woy"] = np.cos(2 * np.pi * woy / 52)

    # Calendar flags
    fe["is_weekend"] = (dow >= 5).astype(float)
    fe["is_holiday"] = _irish_holidays(fe.index)

    # Demand lags
    fe["demand_lag_1d"] = fe["demand_mw"].shift(96)
    fe["demand_lag_2d"] = fe["demand_mw"].shift(192)
    fe["demand_lag_1w"] = fe["demand_mw"].shift(672)
    fe["demand_diff"]   = fe["demand_mw"].diff()

    # Drop rows without full lag history
    fe = fe.dropna()

    # Final 19-column selection, the order matters for model input
    cols = [
        "demand_mw",          # 1
        "wind_mw",            # 2
        "generation_mw",      # 3
        "co2_intensity",      # 4
        "snsp",               # 5
        "wind_penetration",   # 6
        "generation_ratio",   # 7
        "sin_slot",           # 8
        "cos_slot",           # 9
        "sin_dow",            # 10
        "cos_dow",            # 11
        "sin_woy",            # 12
        "cos_woy",            # 13
        "is_weekend",         # 14
        "is_holiday",         # 15
        "demand_lag_1d",      # 16
        "demand_lag_2d",      # 17
        "demand_lag_1w",      # 18
        "demand_diff",        # 19
    ]

    # Safety: add any missing column as zeros rather than crashing
    for c in cols:
        if c not in fe.columns:
            print(f"    Warning: '{c}' missing — zero-filled")
            fe[c] = 0.0

    fe = fe[cols]
    assert fe.shape[1] == 19, f"Expected 19 features, got {fe.shape[1]}"
    print(f"    Feature matrix: {fe.shape}")
    return fe


#Normalise

def normalise(fe: pd.DataFrame, stats: dict = None):
    if stats is None:
        stats = {c: (fe[c].min(), fe[c].max()) for c in fe.columns}
    normed = fe.copy()
    for c in fe.columns:
        lo, hi = stats[c]
        normed[c] = (fe[c] - lo) / (hi - lo) if hi > lo else 0.0
    return normed, stats


# Sliding-window dataset

def make_windows(fe: pd.DataFrame):
    arr = fe.values.astype(np.float32)
    X, y, ts = [], [], []
    for i in range(LOOKBACK_SLOTS, len(arr) - HORIZON_SLOTS + 1):
        X.append(arr[i - LOOKBACK_SLOTS : i])       # [672, 19]
        y.append(arr[i : i + HORIZON_SLOTS, 0])     # [96]  demand only
        ts.append(fe.index[i])
    return np.stack(X), np.stack(y), np.array(ts)


# Train / val / test split

def split(fe: pd.DataFrame):
    n  = len(fe)
    i1 = int(n * TRAIN_RATIO)
    i2 = int(n * (TRAIN_RATIO + VAL_RATIO))
    tr, va, te = fe.iloc[:i1], fe.iloc[i1:i2], fe.iloc[i2:]
    print(f"    Train  {tr.index[0].date()} → {tr.index[-1].date()}  ({len(tr):,} slots)")
    print(f"    Val    {va.index[0].date()} → {va.index[-1].date()}  ({len(va):,} slots)")
    print(f"    Test   {te.index[0].date()} → {te.index[-1].date()}  ({len(te):,} slots)")
    return tr, va, te


# Main

def main():
    print("=" * 60)
    print("MSA-NeOpt — Step 01: Data Pipeline")
    print("=" * 60)

    raw   = load_all()
    clean = preprocess(raw)
    fe    = engineer_features(clean)

    tr_fe, va_fe, te_fe = split(fe)

    tr_n, stats = normalise(tr_fe)
    va_n, _     = normalise(va_fe, stats)
    te_n, _     = normalise(te_fe, stats)

    # Save normalisation stats for inference
    pd.DataFrame(stats, index=["min", "max"]).T.to_csv(
        DATA_DIR / "normalisation_stats.csv")

    # Save sliding-window arrays
    for name, normed in [("train", tr_n), ("val", va_n), ("test", te_n)]:
        X, y, ts = make_windows(normed)
        np.save(DATA_DIR / f"X_{name}.npy",  X)
        np.save(DATA_DIR / f"y_{name}.npy",  y)
        np.save(DATA_DIR / f"ts_{name}.npy", ts)
        print(f"  {name:5s}  X={X.shape}  y={y.shape}")

    # Save full normalised feature DataFrame for inspection
    pd.concat([tr_n, va_n, te_n]).to_parquet(
        DATA_DIR / "eirgrid_features_norm.parquet")

    print("\n  Done — data/ ready.")
    print("  Next: python3 step02_train_pto.py")


if __name__ == "__main__":
    main()
