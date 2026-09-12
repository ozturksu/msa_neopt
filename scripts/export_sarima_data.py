#!/usr/bin/env python3
"""
export_sarima_data.py
MSA-NeOpt — Export raw demand data for SARIMA baseline

WHY THIS SCRIPT EXISTS (not just a copy of step01):
  step01's own feature-engineered dataframe (`fe`) drops ~84% of rows
  (207,138 -> 33,697), mostly due to gaps in *other* signals (wind,
  generation), not demand itself. Exporting demand filtered to only
  those surviving rows would hand SARIMA a series full of large,
  irregular calendar gaps, a bad input for a seasonal ARIMA fit.

  Instead, this script:
    1. Runs the full step01 pipeline once (load_all -> preprocess ->
       engineer_features -> split) ONLY to extract the exact train/
       val/test boundary DATES, the same ones behind Table 4.
    2. Re-slices the much more complete `clean` dataframe (output of
       preprocess(), which only drops rows where demand itself is
       unrecoverable) using those same calendar boundaries.
    3. Exports that, a mostly-continuous demand series, split into
       the identical train/val/test calendar windows used everywhere
       else in this project.


Run from the scripts/ folder (same conda env, predopt310):
    python export_sarima_data.py

Output:
    results/sarima_export.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step01_data_pipeline import load_all, preprocess, engineer_features, split, RAW_DIR

ROOT    = Path(__file__).parent.parent
RES_DIR = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("MSA-NeOpt — Export data for SARIMA baseline")
    print("=" * 60)
    #  Step A: run the REAL pipeline once, just to get boundary dates 
    print("\n  [1/2] Running full pipeline once to extract split boundaries...")
    print(f"  (RAW_DIR only matters if data/eirgrid_raw.parquet isn't cached yet)")
    raw   = load_all()
    clean = preprocess(raw)
    fe    = engineer_features(clean)          # the aggressive dropna happens here
    tr_fe, va_fe, te_fe = split(fe)

    train_start, train_end = tr_fe.index[0], tr_fe.index[-1]
    val_start,   val_end   = va_fe.index[0],  va_fe.index[-1]
    test_start,  test_end  = te_fe.index[0],  te_fe.index[-1]

    print(f"\n  Boundary dates (from the real Table-4 split):")
    print(f"    Train  {train_start.date()} -> {train_end.date()}")
    print(f"    Val    {val_start.date()} -> {val_end.date()}")
    print(f"    Test   {test_start.date()} -> {test_end.date()}")

    # Sanity check against the report's stated dates (Section 3.1)
    expected_train_start = pd.Timestamp("2014-02-08")
    expected_test_start  = pd.Timestamp("2024-10-22")
    if abs((train_start - expected_train_start).days) > 2:
        print(f"    NOTE: train start differs from report's stated "
              f"2014-02-08 by more than 2 days, worth double-checking "
              f"nothing has changed upstream.")
    if abs((test_start - expected_test_start).days) > 2:
        print(f"    NOTE: test start differs from report's stated "
              f"2024-10-22 by more than 2 days, worth double-checking.")

    # ── Step B: re-slice the MORE COMPLETE `clean` series by those dates ──
    print(f"\n  [2/2] Re-slicing the more complete pre-feature-engineering "
          f"series by these dates...")
    export = clean.copy()
    export["split"] = "excluded"   # rows outside all three windows (if any)
    export.loc[train_start:train_end, "split"] = "train"
    export.loc[val_start:val_end,     "split"] = "val"
    export.loc[test_start:test_end,   "split"] = "test"

    # Keep only rows that fall inside one of the three windows
    export = export[export["split"] != "excluded"]

    # Raw signal columns only -- SARIMA needs the plain series, not the
    # 14 derived/engineered features (cyclical encodings, lags, etc.)
    raw_cols = [c for c in ["demand_mw", "wind_mw", "generation_mw",
                             "co2_intensity", "snsp"] if c in export.columns]
    export = export[raw_cols + ["split"]]
    export.index.name = "timestamp"

    out_path = RES_DIR / "sarima_export.csv"
    export.to_csv(out_path)

    n_train = (export["split"] == "train").sum()
    n_val   = (export["split"] == "val").sum()
    n_test  = (export["split"] == "test").sum()
    n_gaps_in_demand = export["demand_mw"].isna().sum()

    print(f"\n  Exported {len(export):,} rows -> {out_path}")
    print(f"    train={n_train:,}  val={n_val:,}  test={n_test:,}")
    print(f"    remaining demand_mw gaps (>4hr, unfilled): {n_gaps_in_demand:,}")
    print(f"\n  This series is from the pre-feature-engineering stage, so it's")
    print(f"  far more continuous than the 33,697-row model-training data")
    print(f"  but the train/val/test date boundaries are IDENTICAL to the")
    print(f"  ones behind Table 4, so any regret it computes from a")
    print(f"  SARIMA forecast over the 'test' rows is directly comparable.")
    print(f"\n  Send results/sarima_export.csv.")


if __name__ == "__main__":
    main()