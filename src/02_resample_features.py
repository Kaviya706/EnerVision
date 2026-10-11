"""
EnerVision - Step 2: Resample to hourly + feature engineering (nothing dropped)

Input : data/processed/enervision_cleaned.csv   (minute-level, 44 columns)
Output: data/processed/enervision_hourly.csv     (hourly, ALL columns kept)
        data/processed/train.csv
        data/processed/test.csv

What this does:
1. Loads the cleaned file, sets `date` as the index for resampling.
2. Resamples minute-level data to hourly:
     - numeric columns (all *_kW, weather)  -> hourly mean
     - text columns (season, summary, icon) -> most frequent value in
       that hour (season never changes within an hour; summary/icon
       occasionally do)
   NOTHING is dropped - every appliance column and season/summary/icon
   stay in the file.
3. Rebuilds calendar features from the hourly timestamp (averaging
   hour/day/month would be meaningless, so these are recomputed).
4. Adds cyclic encoding for hour (so 23:00 is "close" to 00:00).
5. Adds lag features for House_overall_kW: usage 1 hour ago, 24 hours
   ago, 1 week ago, plus a rolling 24-hour average.
6. Puts `date` back as a normal column and orders columns the same
   way as Step 1: date/time first, appliances next, everything else
   (weather, season/summary/icon, lag/cyclic features) at the end.
7. Splits chronologically into train.csv / test.csv (80/20, never
   shuffled - this is time-series data) and saves both into
   data/processed/, alongside the full hourly file.

NOTE on using this for model training:
  House_overall_kW is (approximately) the sum of every other *_kW
  column. If you train a model to PREDICT House_overall_kW, do not
  also feed it the other *_kW columns as inputs - it will just learn
  to add them up instead of finding real patterns, and will look
  perfect here but fail on new data. This script keeps all columns
  in the file; the EXCLUDE_FROM_MODEL list below documents which
  ones to leave out of X when you get to the training step. Nothing
  is removed from the CSV itself.
"""
import numpy as np
import pandas as pd

IN_FILE = "data/processed/enervision_cleaned.csv"
OUT_HOURLY = "data/processed/enervision_hourly.csv"
OUT_TRAIN = "data/processed/train.csv"
OUT_TEST = "data/processed/test.csv"

TARGET = "House_overall_kW"

# --- 1. load ---
df = pd.read_csv(IN_FILE, parse_dates=["date"]).set_index("date").sort_index()
print("Loaded:", df.shape)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
text_cols = ["season", "summary", "icon"]

# --- 2. resample: numeric -> mean, text -> most frequent ---
hourly_numeric = df[numeric_cols].resample("h").mean()

def most_frequent(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else np.nan

hourly_text = df[text_cols].resample("h").agg(most_frequent)

hourly = hourly_numeric.join(hourly_text)
print("Resampled to hourly:", hourly.shape)

# --- 3. rebuild calendar features from the hourly index ---
hourly["hour"] = hourly.index.hour
hourly["day"] = hourly.index.day
hourly["month"] = hourly.index.month
hourly["day_of_week"] = hourly.index.dayofweek
hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)

# --- 4. cyclic encoding for hour ---
hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24)
hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24)

# --- 5. lag features (based on target only) ---
hourly["lag_1h"] = hourly[TARGET].shift(1)
hourly["lag_24h"] = hourly[TARGET].shift(24)
hourly["lag_168h"] = hourly[TARGET].shift(168)  # 1 week
hourly["roll_24h_mean"] = hourly[TARGET].shift(1).rolling(24).mean()

hourly = hourly.dropna().reset_index()  # date becomes a normal column again
print("Final shape (after dropping NaN rows from lags):", hourly.shape)

# --- 6. column order: date/time -> appliances -> everything else ---
datetime_cols = ["date", "hour", "day", "month", "day_of_week", "is_weekend"]
appliance_priority = [TARGET, "Refrigerator_kW", "Solar_kW"]
other_appliance_cols = sorted(
    c for c in hourly.columns
    if c.endswith("_kW") and c not in appliance_priority
)
appliance_cols = appliance_priority + other_appliance_cols
rest_cols = [c for c in hourly.columns if c not in datetime_cols + appliance_cols]
final_order = datetime_cols + appliance_cols + rest_cols
hourly = hourly[final_order]

# --- 7. chronological train/test split ---
split = int(len(hourly) * 0.8)
train, test = hourly.iloc[:split], hourly.iloc[split:]

hourly.to_csv(OUT_HOURLY, index=False)
train.to_csv(OUT_TRAIN, index=False)
test.to_csv(OUT_TEST, index=False)

print("\nFinal column order:")
for i, c in enumerate(final_order, 1):
    print(f"  {i:2d}. {c}")

print("\nTrain:", train.shape, train["date"].min(), "->", train["date"].max())
print("Test :", test.shape, test["date"].min(), "->", test["date"].max())
print("\nSaved (all columns kept, nothing dropped):")
print(" -", OUT_HOURLY)
print(" -", OUT_TRAIN)
print(" -", OUT_TEST)

# --- reference only: what to exclude from X if predicting House_overall_kW ---
EXCLUDE_FROM_MODEL = [c for c in appliance_cols if c != TARGET] + ["summary", "icon", "season"]
print("\n(Reference) If training a model on", TARGET, "- exclude these from X:")
print(EXCLUDE_FROM_MODEL)