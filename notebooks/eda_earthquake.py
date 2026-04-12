# %% [markdown]
# # Exploratory Data Analysis - Earthquake Prediction
# 
# Notebook ini berisi analisis lengkap dataset gempa bumi Amatrice untuk memahami:
# 1. Distribusi data
# 2. Pola temporal dan spasial
# 3. Karakteristik magnitude
# 4. Implikasi untuk pemodelan

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
pd.set_option('display.max_columns', 50)

print("Libraries loaded successfully!")

# %% [markdown]
# ## 1. Load Data

# %%
# Load raw earthquake data
DATA_PATH = 'Amatrice_CAT5.v20210504.csv'
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
df.head()

# %%
# Basic info
print("Data Types:")
print(df.dtypes)
print(f"\nMissing values:\n{df.isnull().sum()}")

# %% [markdown]
# ## 2. Data Cleaning & Preprocessing

# %%
# Parse datetime from separate columns
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
df['date'] = df['datetime'].dt.date
df['hour_of_day'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month_of_year'] = df['datetime'].dt.month

# Calculate energy (Gutenberg-Richter)
df['log_energy'] = 1.5 * df['mw'] + 4.8

print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
print(f"Total events: {len(df):,}")

# %% [markdown]
# ## 3. Magnitude Analysis

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 Magnitude distribution (histogram)
ax1 = axes[0, 0]
ax1.hist(df['mw'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(df['mw'].mean(), color='red', linestyle='--', label=f'Mean: {df["mw"].mean():.2f}')
ax1.axvline(df['mw'].median(), color='orange', linestyle='--', label=f'Median: {df["mw"].median():.2f}')
ax1.set_xlabel('Magnitude (Mw)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribusi Magnitude')
ax1.legend()

# 3.2 Log-scale histogram
ax2 = axes[0, 1]
ax2.hist(df['mw'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax2.set_yscale('log')
ax2.set_xlabel('Magnitude (Mw)')
ax2.set_ylabel('Frequency (log scale)')
ax2.set_title('Distribusi Magnitude (Log Scale)')
for m in [1, 2, 3, 4, 5]:
    ax2.axvline(m, color='gray', linestyle=':', alpha=0.5)
    count = (df['mw'] >= m).sum()
    ax2.text(m+0.05, ax2.get_ylim()[1]*0.5, f'M≥{m}: {count:,}', rotation=90, fontsize=9)

# 3.3 Boxplot
ax3 = axes[1, 0]
ax3.boxplot(df['mw'], vert=True)
ax3.set_ylabel('Magnitude (Mw)')
ax3.set_title('Boxplot Magnitude')

# 3.4 CDF
ax4 = axes[1, 1]
sorted_mw = np.sort(df['mw'])
cdf = np.arange(1, len(sorted_mw)+1) / len(sorted_mw)
ax4.plot(sorted_mw, cdf, color='steelblue')
ax4.set_xlabel('Magnitude (Mw)')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('CDF Magnitude')
for m in [1, 2, 3]:
    pct = (df['mw'] < m).mean() * 100
    ax4.axvline(m, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(pct/100, color='gray', linestyle=':', alpha=0.5)
    ax4.text(m+0.05, 0.1, f'{pct:.1f}%', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/eda_magnitude_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Magnitude statistics
print("=" * 50)
print("MAGNITUDE STATISTICS")
print("=" * 50)
print(df['mw'].describe())
print(f"\nPercentiles:")
for p in [50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {df['mw'].quantile(p/100):.2f}")

print(f"\nMagnitude Ranges:")
ranges = [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 7)]
for low, high in ranges:
    count = ((df['mw'] >= low) & (df['mw'] < high)).sum()
    pct = count / len(df) * 100
    print(f"  M{low}-{high}: {count:>8,} ({pct:>5.2f}%)")

# %% [markdown]
# ## 4. Temporal Analysis

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 4.1 Events per day
daily_counts = df.groupby('date').size()
ax1 = axes[0, 0]
ax1.plot(daily_counts.index, daily_counts.values, alpha=0.7, linewidth=0.8)
ax1.set_xlabel('Date')
ax1.set_ylabel('Events per Day')
ax1.set_title('Daily Event Count')
ax1.axhline(daily_counts.mean(), color='red', linestyle='--', label=f'Mean: {daily_counts.mean():.1f}')
ax1.legend()

# 4.2 Hourly pattern
ax2 = axes[0, 1]
hourly = df.groupby('hour_of_day').size()
ax2.bar(hourly.index, hourly.values, color='steelblue', edgecolor='black')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Total Events')
ax2.set_title('Events by Hour of Day')

# 4.3 Day of week pattern
ax3 = axes[1, 0]
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow = df.groupby('day_of_week').size()
ax3.bar(dow.index, dow.values, color='steelblue', edgecolor='black')
ax3.set_xticks(range(7))
ax3.set_xticklabels(dow_labels)
ax3.set_xlabel('Day of Week')
ax3.set_ylabel('Total Events')
ax3.set_title('Events by Day of Week')

# 4.4 Max magnitude per day
daily_max = df.groupby('date')['mw'].max()
ax4 = axes[1, 1]
ax4.scatter(daily_max.index, daily_max.values, alpha=0.5, s=10)
ax4.set_xlabel('Date')
ax4.set_ylabel('Max Daily Magnitude')
ax4.set_title('Maximum Daily Magnitude')
ax4.axhline(3, color='red', linestyle='--', label='M=3 threshold')
ax4.legend()

plt.tight_layout()
plt.savefig('outputs/eda_temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Significant events timeline
print("=" * 50)
print("SIGNIFICANT EVENTS (M >= 3)")
print("=" * 50)
sig_events = df[df['mw'] >= 3].sort_values('mw', ascending=False)
print(f"Total significant events: {len(sig_events)}")
print(f"Percentage: {len(sig_events)/len(df)*100:.2f}%")
print(f"\nTop 10 largest events:")
print(sig_events[['datetime', 'mw', 'lat', 'lon', 'dep']].head(10).to_string())

# %% [markdown]
# ## 5. Spatial Analysis

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 5.1 All events
ax1 = axes[0]
scatter = ax1.scatter(df['lon'], df['lat'], c=df['mw'], s=1, alpha=0.3, cmap='YlOrRd')
plt.colorbar(scatter, ax=ax1, label='Magnitude')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('Spatial Distribution - All Events')

# 5.2 Significant events only
ax2 = axes[1]
sig = df[df['mw'] >= 3]
ax2.scatter(df['lon'], df['lat'], c='lightgray', s=1, alpha=0.2, label='M<3')
scatter2 = ax2.scatter(sig['lon'], sig['lat'], c=sig['mw'], s=sig['mw']**2*5, 
                       alpha=0.7, cmap='YlOrRd', edgecolor='black', linewidth=0.5)
plt.colorbar(scatter2, ax=ax2, label='Magnitude')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('Spatial Distribution - Significant Events (M≥3)')

plt.tight_layout()
plt.savefig('outputs/eda_spatial_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Spatial statistics
print("=" * 50)
print("SPATIAL STATISTICS")
print("=" * 50)
print(f"Latitude range: {df['lat'].min():.4f} to {df['lat'].max():.4f}")
print(f"Longitude range: {df['lon'].min():.4f} to {df['lon'].max():.4f}")
print(f"Depth range: {df['dep'].min():.1f} km to {df['dep'].max():.1f} km")
print(f"Mean depth: {df['dep'].mean():.2f} km")

# %% [markdown]
# ## 6. Depth Analysis

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 6.1 Depth distribution
ax1 = axes[0]
ax1.hist(df['dep'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Depth (km)')
ax1.set_ylabel('Frequency')
ax1.set_title('Depth Distribution')

# 6.2 Magnitude vs Depth
ax2 = axes[1]
ax2.scatter(df['dep'], df['mw'], alpha=0.1, s=1)
ax2.set_xlabel('Depth (km)')
ax2.set_ylabel('Magnitude')
ax2.set_title('Magnitude vs Depth')

# 6.3 Depth by magnitude class
ax3 = axes[2]
df['mw_class'] = pd.cut(df['mw'], bins=[-1, 0, 1, 2, 3, 4, 7], 
                        labels=['<0', '0-1', '1-2', '2-3', '3-4', '4+'])
df.boxplot(column='dep', by='mw_class', ax=ax3)
ax3.set_xlabel('Magnitude Class')
ax3.set_ylabel('Depth (km)')
ax3.set_title('Depth by Magnitude Class')
plt.suptitle('')  # Remove automatic title

plt.tight_layout()
plt.savefig('outputs/eda_depth_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 7. Hourly Aggregation Analysis (Model Input)

# %%
# Create hourly bins (as used in the ST-GAT model)
df['hour_bin'] = df['datetime'].dt.floor('H')

hourly_agg = df.groupby('hour_bin').agg({
    'mw': ['count', 'max', 'mean', 'std'],
    'log_energy': 'sum',
    'dep': 'mean'
}).reset_index()
hourly_agg.columns = ['hour_bin', 'count', 'max_mw', 'mean_mw', 'std_mw', 'log_energy', 'avg_depth']

print(f"Total hourly bins: {len(hourly_agg)}")
print(f"Hourly aggregation stats:")
print(hourly_agg.describe())

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1 Events per hour distribution
ax1 = axes[0, 0]
ax1.hist(hourly_agg['count'], bins=50, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Events per Hour')
ax1.set_ylabel('Frequency (hours)')
ax1.set_title('Distribution of Hourly Event Counts')
ax1.axvline(hourly_agg['count'].mean(), color='red', linestyle='--', 
            label=f'Mean: {hourly_agg["count"].mean():.1f}')
ax1.legend()

# 7.2 Max magnitude per hour distribution
ax2 = axes[0, 1]
ax2.hist(hourly_agg['max_mw'], bins=50, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Max Magnitude per Hour')
ax2.set_ylabel('Frequency (hours)')
ax2.set_title('Distribution of Hourly Max Magnitude')

# 7.3 Zero event hours
ax3 = axes[1, 0]
# Create complete hourly range
date_range = pd.date_range(start=df['datetime'].min().floor('H'), 
                           end=df['datetime'].max().ceil('H'), freq='H')
complete_hourly = pd.DataFrame({'hour_bin': date_range})
merged = complete_hourly.merge(hourly_agg, on='hour_bin', how='left').fillna(0)
zero_hours = (merged['count'] == 0).sum()
nonzero_hours = (merged['count'] > 0).sum()

ax3.bar(['No Events', 'Has Events'], [zero_hours, nonzero_hours], 
        color=['lightcoral', 'steelblue'], edgecolor='black')
ax3.set_ylabel('Number of Hours')
ax3.set_title('Hours with vs without Seismic Activity')
for i, v in enumerate([zero_hours, nonzero_hours]):
    ax3.text(i, v + 50, f'{v:,}\n({v/(zero_hours+nonzero_hours)*100:.1f}%)', 
             ha='center', fontsize=10)

# 7.4 Max magnitude classes distribution
ax4 = axes[1, 1]
# Classify hourly max_mw
bins = [0, 1, 2, 3, 7]
labels = ['M<1', 'M1-2', 'M2-3', 'M≥3']
# Only for hours with events
has_events = merged[merged['count'] > 0]
has_events['mw_class'] = pd.cut(has_events['max_mw'], bins=bins, labels=labels, include_lowest=True)
class_counts = has_events['mw_class'].value_counts().sort_index()

ax4.bar(class_counts.index, class_counts.values, color='steelblue', edgecolor='black')
ax4.set_xlabel('Max Magnitude Class')
ax4.set_ylabel('Number of Hours')
ax4.set_title('Distribution of Hourly Max Magnitude Classes\n(Hours with Events Only)')
for i, (label, count) in enumerate(class_counts.items()):
    pct = count / len(has_events) * 100
    ax4.text(i, count + 20, f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/eda_hourly_aggregation.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# KEY INSIGHT: Empty bins
print("=" * 60)
print("KEY INSIGHT: DATA SPARSITY")
print("=" * 60)
print(f"Total hours in period: {len(merged):,}")
print(f"Hours with NO events: {zero_hours:,} ({zero_hours/len(merged)*100:.1f}%)")
print(f"Hours WITH events: {nonzero_hours:,} ({nonzero_hours/len(merged)*100:.1f}%)")
print()
print("When using spatial grid with N nodes:")
print(f"  If 205 nodes × {len(merged)} hours = {205*len(merged):,} total bins")
print(f"  Most bins will be EMPTY (no earthquake activity)")
print()
print("This explains why classification shows mostly M<1:")
print("  → Empty bins have no magnitude → assigned to lowest class")

# %% [markdown]
# ## 8. Correlation Analysis

# %%
# Correlation between features
correlation_cols = ['mw', 'dep', 'lat', 'lon', 'log_energy']
corr_matrix = df[correlation_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - Event Features')
plt.tight_layout()
plt.savefig('outputs/eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 9. Summary & Implications for Modeling

# %%
print("=" * 70)
print("SUMMARY & MODELING IMPLICATIONS")
print("=" * 70)

print("""
📊 DATA CHARACTERISTICS:
─────────────────────────────────────────────────────────────────────
• Total events: {:,}
• Duration: ~1 year
• Magnitude range: {:.2f} to {:.2f} (mean: {:.2f}, median: {:.2f})
• Most events are small: {:.1f}% have M < 2
• Significant events (M≥3): only {:,} ({:.2f}%)

🔍 KEY INSIGHTS:
─────────────────────────────────────────────────────────────────────
1. EXTREME CLASS IMBALANCE (for classification approach):
   • When aggregating hourly per grid cell, most bins are EMPTY
   • This causes model to predict "no significant event" almost always
   • Classification accuracy is misleading (high accuracy ≠ useful predictions)

2. TEMPORAL SPARSITY:
   • Events are clustered (aftershock sequences)
   • Most hours have 0 or very few events per grid cell
   
3. SPATIAL CONCENTRATION:
   • Events concentrated around fault lines
   • Many grid cells have no activity at all

💡 RECOMMENDATIONS FOR MODELING:
─────────────────────────────────────────────────────────────────────
Option A: REGRESSION (Current approach with improvements)
   • Use weighted loss (higher weight for large magnitude events)
   • Evaluate ONLY on time bins with activity
   • Focus on count and energy prediction (more continuous)
   
Option B: EVENT-BASED PREDICTION
   • Instead of predicting all time bins, only predict GIVEN an event occurs
   • What is its likely magnitude?
   • Uses only rows where count > 0
   
Option C: AGGREGATE TO DAILY/WEEKLY
   • Reduces sparsity problem
   • Daily max_mw distribution is more balanced
   
Option D: SEQUENCE-TO-SEQUENCE for Active Periods
   • Identify active periods (sequences with multiple events)
   • Predict evolution of sequences only
""".format(
    len(df),
    df['mw'].min(), df['mw'].max(), df['mw'].mean(), df['mw'].median(),
    (df['mw'] < 2).mean() * 100,
    (df['mw'] >= 3).sum(), (df['mw'] >= 3).mean() * 100
))

# %% [markdown]
# ## 10. Save Processed Data Summary

# %%
# Save summary statistics
summary = {
    'total_events': len(df),
    'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}",
    'duration_days': (df['datetime'].max() - df['datetime'].min()).days,
    'mw_min': df['mw'].min(),
    'mw_max': df['mw'].max(),
    'mw_mean': df['mw'].mean(),
    'mw_median': df['mw'].median(),
    'events_m_ge_3': (df['mw'] >= 3).sum(),
    'pct_m_ge_3': (df['mw'] >= 3).mean() * 100,
    'hourly_bins_total': len(merged),
    'hourly_bins_empty_pct': zero_hours / len(merged) * 100,
}

summary_df = pd.DataFrame([summary]).T
summary_df.columns = ['Value']
print(summary_df)
summary_df.to_csv('outputs/eda_summary.csv')
print("\nSummary saved to outputs/eda_summary.csv")

# %%
print("\n✅ EDA Complete! Check the 'outputs' folder for saved figures.")
