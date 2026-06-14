# Data Setup

## Source

EirGrid historical Irish electricity demand data at 15-minute resolution,
downloaded using the **EirGrid Data Downloader** by Daniel Parke:

https://github.com/Daniel-Parke/EirGrid_Data_Download

## How to download the data

### Step 1 — Clone the downloader

```bash
git clone https://github.com/Daniel-Parke/EirGrid_Data_Download.git
cd EirGrid_Data_Download
pip install -r requirements.txt
```

### Step 2 — Run the downloader

Follow the instructions in the EirGrid_Data_Download README to download
the following five signal categories for years 2014 to 2024:

| Signal | EirGrid category | Column in our pipeline |
|--------|-----------------|----------------------|
| System demand | `demandactual` | `demand_mw` |
| Wind generation | `windactual` | `wind_mw` |
| Total generation | `generationactual` | `generation_mw` |
| CO2 intensity | `co2intensity` | `co2_intensity` |
| SNSP | `SnspALL` | `snsp` |

### Step 3 — Check the file naming

The downloader produces files in this format:

```
ROI_demandactual_14_Eirgrid.csv   ← 2014 demand
ROI_windactual_23_Eirgrid.csv     ← 2023 wind
ROI_SnspALL_21_Eirgrid.csv        ← 2021 SNSP
```

Place all downloaded CSV files in one directory, for example:

```
/Users/yourname/eirgrid_data/
```

### Step 4 — Update the pipeline path

Open `scripts/step01_prepare_data.py` and update `RAW_DIR`:

```python
RAW_DIR = Path("/Users/yourname/eirgrid_data/")
```

### Step 5 — Run the data pipeline

```bash
python scripts/step01_prepare_data.py
```

## Output

Step01 saves these files to `data/`:

```
X_train.npy   [22848, 672, 19]   7-day input windows — training set
X_val.npy     [4288,  672, 19]   validation set
X_test.npy    [4288,  672, 19]   test set (2023, held out)
y_train.npy   [22848, 96]        next-day demand targets
y_val.npy     [4288,  96]
y_test.npy    [4288,  96]
normalisation_stats.csv           min/max per feature from training set only
eirgrid_raw.parquet               cached merged signals (speeds up re-runs)
```

## Notes

- SNSP data was not published by EirGrid before 2021 — automatically zero-filled for earlier years
- Split: Train 2014–2021 | Validation 2022 | Test 2023
- Normalisation uses training-set statistics only — no data leakage into validation or test
