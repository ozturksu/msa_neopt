# Data Setup

## Source

EirGrid publishes historical Irish electricity demand at 15-minute resolution:

**Download portal:** https://www.eirgrid.ie/grid-and-markets/market-data

Free to access for academic use.

## What to download

Download yearly CSV files for these five signals, years 2014–2024:

| Signal | EirGrid category | Our column name |
|--------|-----------------|----------------|
| System demand | `demandactual` | `demand_mw` |
| Wind generation | `windactual` | `wind_mw` |
| Total generation | `generationactual` | `generation_mw` |
| CO2 intensity | `co2intensity` | `co2_intensity` |
| SNSP | `SnspALL` | `snsp` |

## File naming

Files follow the pattern `ROI_{category}_{YY}_Eirgrid.csv`:

```
ROI_demandactual_14_Eirgrid.csv   ← 2014 demand
ROI_demandactual_15_Eirgrid.csv   ← 2015 demand
...
ROI_SnspALL_21_Eirgrid.csv        ← 2021 SNSP (not available before 2021)
```

## Setup

1. Place all downloaded CSV files in one folder, e.g. `/Users/yourname/eirgrid_data/`

2. Update `RAW_DIR` at the top of `scripts/step01_prepare_data.py`:

```python
RAW_DIR = Path("/Users/yourname/eirgrid_data/")
```

3. Run:

```bash
python scripts/step01_prepare_data.py
```

## Output

Step01 produces these files in `data/`:

```
X_train.npy   [22848, 672, 19]   7-day input windows, training set
X_val.npy     [4288,  672, 19]   validation set
X_test.npy    [4288,  672, 19]   test set (2023, held out)
y_train.npy   [22848, 96]        next-day demand targets
y_val.npy     [4288,  96]
y_test.npy    [4288,  96]
normalisation_stats.csv           min/max per feature from training set
eirgrid_raw.parquet               merged raw signals (cache)
```

## Notes

- SNSP data was not published before 2021 — automatically zero-filled for earlier years
- Training: 2014–~2021 | Validation: ~2022 | Test: ~2023
- All normalisation uses training-set statistics only (no data leakage)
