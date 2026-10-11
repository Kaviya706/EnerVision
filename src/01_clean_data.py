"""
EnerVision - Phase 1, Step: Data cleaning & column reordering

What this does:
1. Loads enerVision_combined_master.csv, parses `date`.
2. Drops exact-duplicate columns:
     use_kW    (== House_overall_kW)
     Fridge_kW (== Refrigerator_kW)
     gen_kW    (== Solar_kW)
3. Verifies hour/day/month/day_of_week/is_weekend match values derived
   from `date`; rebuilds them from `date` if any row disagrees.
4. Reorders columns:
     a. date/time:  date, hour, day, month, day_of_week, is_weekend
     b. appliances:  House_overall_kW, Refrigerator_kW, Solar_kW,
                      + every other *_kW submeter column
     c. everything else (weather + summary/icon/season) at the end
5. Saves enervision_cleaned.csv
"""
import pandas as pd

# --- change these two paths to match where the files are on your PC ---
RAW = "data/raw/enerVision_combined_master.csv"
OUT = "data/processed/enervision_cleaned.csv"

df = pd.read_csv(RAW, parse_dates=["date"])
print("Loaded:", df.shape)

# --- 1. drop exact duplicate columns ---
dupe_pairs = [("use_kW", "House_overall_kW"),
              ("Fridge_kW", "Refrigerator_kW"),
              ("gen_kW", "Solar_kW")]
for drop_col, keep_col in dupe_pairs:
    assert (df[drop_col] == df[keep_col]).all(), f"{drop_col} != {keep_col}, check data!"
df = df.drop(columns=[p[0] for p in dupe_pairs])
print("Dropped duplicate columns:", [p[0] for p in dupe_pairs])

# --- 2. verify date-derived features, rebuild if needed ---
checks = {
    "hour": df["date"].dt.hour,
    "day": df["date"].dt.day,
    "month": df["date"].dt.month,
    "day_of_week": df["date"].dt.dayofweek,
}
for col, derived in checks.items():
    match = (df[col] == derived).mean()
    if match < 1.0:
        print(f"  {col}: only {match:.4%} matched date -> rebuilding from date")
        df[col] = derived
    else:
        print(f"  {col}: OK (100% matches date)")

derived_weekend = (df["day_of_week"] >= 5).astype(int)
match = (df["is_weekend"] == derived_weekend).mean()
if match < 1.0:
    print(f"  is_weekend: only {match:.4%} matched -> rebuilding")
    df["is_weekend"] = derived_weekend
else:
    print("  is_weekend: OK (100% matches date)")

# --- 3. reorder columns ---
datetime_cols = ["date", "hour", "day", "month", "day_of_week", "is_weekend"]

appliance_priority = ["House_overall_kW", "Refrigerator_kW", "Solar_kW"]
other_appliance_cols = sorted(
    c for c in df.columns
    if c.endswith("_kW") and c not in appliance_priority
)
appliance_cols = appliance_priority + other_appliance_cols

rest_cols = [c for c in df.columns if c not in datetime_cols + appliance_cols]

final_order = datetime_cols + appliance_cols + rest_cols
assert sorted(final_order) == sorted(df.columns), "column mismatch after reorder!"
df = df[final_order]

# --- 4. save ---
df.to_csv(OUT, index=False)

print("\nFinal column order:")
for i, c in enumerate(final_order, 1):
    print(f"  {i:2d}. {c}")
print("\nShape:", df.shape)
print("Saved to:", OUT)