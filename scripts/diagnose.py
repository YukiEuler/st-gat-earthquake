# diagnostic_max_mw.py
# Script to diagnose why max_mw predictions are flat

import pandas as pd
import numpy as np

print("=" * 70)
print(" DIAGNOSTIC: MAX MAGNITUDE PREDICTION ISSUE")
print("=" * 70)

# Load raw data
df = pd.read_csv('../Amatrice_CAT5.v20210504.csv')

print("\n=== 1. RAW MAGNITUDE DISTRIBUTION ===")
print(df['mw'].describe())

print("\n=== 2. MAGNITUDE PERCENTILES ===")
for p in [50, 75, 90, 95, 99]:
    val = np.percentile(df['mw'], p)
    print(f"  {p}th percentile: {val:.2f}")

print("\n=== 3. MAGNITUDE RANGES ===")
bins = [0, 1, 2, 3, 4, 5, 10]
labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5+']
df['mw_bin'] = pd.cut(df['mw'], bins=bins, labels=labels)
print(df['mw_bin'].value_counts().sort_index())

# Time binning simulation
print("\n=== 4. HOURLY AGGREGATION SPARSITY ===")
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
df['hour_bin'] = df['datetime'].dt.floor('1h')

# Count unique hours
n_hours = df['hour_bin'].nunique()
total_hours = int((df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600)
print(f"  Total hours in dataset: {total_hours:,}")
print(f"  Hours with events: {n_hours:,}")
print(f"  Sparsity (empty hours): {100*(total_hours - n_hours)/total_hours:.1f}%")

# Hourly max magnitude
hourly_max = df.groupby('hour_bin')['mw'].max()
print(f"\n=== 5. HOURLY MAX MAGNITUDE DISTRIBUTION ===")
print(hourly_max.describe())

print("\n=== 6. HOURLY MAX MAGNITUDE RANGES ===")
hourly_bins = pd.cut(hourly_max, bins=bins, labels=labels)
print(hourly_bins.value_counts().sort_index())

# The core problem
print("\n" + "=" * 70)
print(" DIAGNOSIS SUMMARY")
print("=" * 70)

zero_pct = (hourly_max < 1).sum() / len(hourly_max) * 100
low_pct = (hourly_max < 2).sum() / len(hourly_max) * 100
high_pct = (hourly_max >= 4).sum() / len(hourly_max) * 100

print(f"""
CORE ISSUE: Data Imbalance

- {zero_pct:.1f}% of hourly bins have max_mw < 1
- {low_pct:.1f}% of hourly bins have max_mw < 2
- Only {high_pct:.1f}% of hourly bins have max_mw >= 4 (significant)

CONSEQUENCE:
The model learns to predict the MEAN (~{hourly_max.mean():.2f}) because:
1. Most targets are low values (0-2)
2. MSE loss penalizes overprediction heavily
3. Predicting mean minimizes average error

RECOMMENDATIONS:
1. Use class-weighted or asymmetric loss (already added)
2. Oversample significant events (M >= 3)
3. Use classification instead of regression for high magnitude
4. Add more contextual features (precursor events)
""")
