# Data Setup

## Source

EirGrid historical Irish electricity demand data at 15-minute resolution. Two ways to get it: use the provided cached file (recommended), or download fresh.

## Recommended: use the cached file

`eirgrid_raw.parquet` (~3.8 MB) is the exact merged raw data behind the results reported in the paper. Place it at:

```
data/eirgrid_raw.parquet
```

`step01_prepare_data.py` checks for exactly this file before reading anything else. If present, it loads directly and skips CSV parsing entirely — no `RAW_DIR` configuration needed. This is the only way to reproduce the reported splits exactly (see "Why not download fresh?" below).

## Alternative: download fresh

Not recommended for reproducing reported numbers, but documented for completeness.

### Step 1 — Clone the downloader

```bash
git clone https://github.com/Daniel-Parke/EirGrid_Data_Download.git
cd EirGrid_Data_Download
python -m pip install pandas
python -m pip install requests
python -m pip install httpx
python -m pip install "httpx[http2]"
python -m pip install backoff
python -m pip install aiofiles
```

Install each package on its own line, with `httpx[http2]` quoted exactly as above. On zsh (the default shell on current macOS), an unquoted `[http2]` is treated as filename globbing rather than reaching pip, and if combined with other packages on one line, the whole line silently aborts.

### Step 2 — Run the downloader

```bash
python async_eirgrid_downloader.py
```

This downloads the five signal categories below for 2014 through 2025. Expect roughly an hour, not the ~11 minutes suggested by the script's own internal comment.

| Signal | EirGrid category | Column in our pipeline |
|--------|-----------------|----------------------|
| System demand | `demandactual` | `demand_mw` |
| Wind generation | `windactual` | `wind_mw` |
| Total generation | `generationactual` | `generation_mw` |
| CO2 intensity | `co2intensity` | `co2_intensity` |
| SNSP | `SnspALL` | `snsp` |

### Step 3 — Know where the files land

The downloader creates a **nested** folder per region — it does not write into one flat directory:

```
Downloaded_Data/ROI/ROI_demandactual_14_Eirgrid.csv
Downloaded_Data/ROI/ROI_windactual_23_Eirgrid.csv
...
```

### Step 4 — Update the pipeline path

`RAW_DIR` must point at the `ROI` subfolder specifically, not its parent:

```python
RAW_DIR = Path("/absolute/path/to/EirGrid_Data_Download/Downloaded_Data/ROI")
```

### Step 5 — Run the data pipeline

```bash
python scripts/step01_prepare_data.py
```

## Why not download fresh?

Three things go wrong, and none of them raises an error:

- **The EirGrid API returns inconsistent historical coverage between attempts.** Two downloads on the same day, no code changes in between, produced very different row counts and date coverage.
- **SNSP is published island-wide only, not per region.** The `ROI_SnspALL_*.csv` files the downloader produces are genuinely empty (0 bytes) as a result; `step01` zero-fills that column. You can confirm this afterwards — if `data/normalisation_stats.csv` shows `min` equal to `max` for `snsp`, the column was zero-filled.
- **Row counts will not match the reported results.** A fresh download does not reproduce the reported dataset even when it succeeds without error.

## Output

Step01 saves these files to `data/`:

```
X_train.npy   [22820, 672, 19]   7-day input windows — training set
X_val.npy     [4288,  672, 19]   validation set
X_test.npy    [4288,  672, 19]   test set
y_train.npy   [22820, 96]        next-day demand targets
y_val.npy     [4288,  96]
y_test.npy    [4288,  96]
normalisation_stats.csv           min/max per feature from training set only
eirgrid_raw.parquet               cached merged signals (speeds up re-runs)
```

Reference figures, confirmed against actual output: 207,138 rows after cleaning, 33,697 after feature engineering, split into 22,820 training / 4,288 validation / 4,288 test samples. Sample counts are the reliable check. Calendar dates are close but not exact for one boundary: the validation split's start date can be off from the paper by several months even when every count matches exactly, most likely due to a genuine gap in the underlying data rather than a pipeline error.

## Notes

- The split is by ratio (70% / 15% / 15%) over however many rows survive feature engineering, not by fixed calendar years.
- SNSP is published by EirGrid on an all-island basis only, not per region — always zero-filled here regardless of year, for the reason described above, not because of a specific missing date range.
- Normalisation uses training-set statistics only — no data leakage into validation or test.
