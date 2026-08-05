# ==============================================================================
# GENERATE_DATA_EDA.PY - Generate Data Tables and Visualizations for LaTeX
# ==============================================================================
"""
Script untuk menghasilkan statistik data dan visualisasi untuk laporan LaTeX.
Termasuk visualisasi preprocessing: original data, cropped area, dan graph structure.

Usage:
    python scripts/generate_data_eda.py
    
Outputs:
    - outputs/eda/tables/          LaTeX-ready tables
    - outputs/eda/figures/         Visualization images (including maps)
    - outputs/eda/data_summary.txt Full summary report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx
from pathlib import Path
from scipy.spatial.distance import cdist
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

# Try to import cartopy for basemap
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
    print("Cartopy available - will use real basemaps")
except ImportError:
    HAS_CARTOPY = False
    print("Cartopy not available - using simple plots (install with: pip install cartopy)")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_raw_data():
    """Load raw earthquake catalog."""
    filepath = CONFIG['filename']
    print(f"Loading: {filepath}")
    
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Calculate seismic energy
    df['log_energy'] = 4.8 + 1.5 * df['mw']
    
    # Fill NaN
    df['mw'] = df['mw'].fillna(0)
    df['dep'] = df['dep'].fillna(df['dep'].median())
    
    print(f"Loaded {len(df):,} events")
    return df

# ==============================================================================
# PREPROCESSING VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_preprocessing_stages(df, output_dir):
    """
    Visualize preprocessing stages with detailed crop justification:
    1. Original data distribution
    2. Area after cropping (with bounding box)
    3. Events removed vs retained
    4. Generate crop justification statistics
    """
    print("\n Plotting preprocessing stages...")
    
    # Get bounds from config
    lat_min = CONFIG.get('lat_min')
    lat_max = CONFIG.get('lat_max')
    lon_min = CONFIG.get('lon_min')
    lon_max = CONFIG.get('lon_max')
    
    # Original data extent
    orig_lat_min, orig_lat_max = df['lat'].min(), df['lat'].max()
    orig_lon_min, orig_lon_max = df['lon'].min(), df['lon'].max()
    
    # Filter to bounds
    if lat_min and lat_max and lon_min and lon_max:
        mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & \
               (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        df_cropped = df[mask]
        df_removed = df[~mask]
    else:
        df_cropped = df
        df_removed = pd.DataFrame()
    
    n_total = len(df)
    n_cropped = len(df_cropped)
    n_removed = len(df_removed)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ===== Panel 1: Original Data =====
    ax1 = axes[0]
    scatter1 = ax1.scatter(df['lon'], df['lat'], c=df['mw'], cmap='YlOrRd', 
                           alpha=0.3, s=2, vmin=0, vmax=5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'1. Original Data\n(n = {n_total:,} events)')
    plt.colorbar(scatter1, ax=ax1, label='Mw', shrink=0.8)
    
    # Draw crop box on original
    if lat_min and lon_min:
        rect = mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                   linewidth=2, edgecolor='red', facecolor='none',
                                   linestyle='--', label='Area Crop')
        ax1.add_patch(rect)
        ax1.legend(loc='upper right')
    
    # ===== Panel 2: Cropped vs Removed =====
    ax2 = axes[1]
    if len(df_removed) > 0:
        ax2.scatter(df_removed['lon'], df_removed['lat'], c='gray', alpha=0.2, s=2, label=f'Removed: {n_removed:,}')
    ax2.scatter(df_cropped['lon'], df_cropped['lat'], c='steelblue', alpha=0.5, s=2, label=f'Retained: {n_cropped:,}')
    
    if lat_min and lon_min:
        rect = mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                   linewidth=2, edgecolor='red', facecolor='none')
        ax2.add_patch(rect)
    
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'2. Cropping Process\n({n_removed:,} events removed, {n_cropped/n_total*100:.1f}% retained)')
    ax2.legend(loc='upper right')
    
    # ===== Panel 3: Zoomed Cropped Area =====
    ax3 = axes[2]
    scatter3 = ax3.scatter(df_cropped['lon'], df_cropped['lat'], c=df_cropped['mw'], 
                           cmap='YlOrRd', alpha=0.5, s=5, vmin=0, vmax=5)
    ax3.set_xlim(lon_min - 0.01, lon_max + 0.01)
    ax3.set_ylim(lat_min - 0.01, lat_max + 0.01)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title(f'3. Study Area (Zoom)\n(n = {n_cropped:,} events)')
    plt.colorbar(scatter3, ax=ax3, label='Mw', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'preprocessing_stages.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: preprocessing_stages.png")
    print(f"   Original events: {n_total:,}")
    print(f"   Removed events: {n_removed:,} ({n_removed/n_total*100:.1f}%)")
    print(f"   Retained events: {n_cropped:,} ({n_cropped/n_total*100:.1f}%)")
    
    return df_cropped, {'total': n_total, 'retained': n_cropped, 'removed': n_removed}


def generate_crop_justification(df, output_dir):
    """
    Generate detailed analysis justifying the crop area selection.
    Includes density comparison, event concentration, magnitude distribution.
    """
    print("\n Generating crop justification analysis...")
    
    lat_min = CONFIG.get('lat_min')
    lat_max = CONFIG.get('lat_max')
    lon_min = CONFIG.get('lon_min')
    lon_max = CONFIG.get('lon_max')
    
    # Areas
    orig_lat_range = df['lat'].max() - df['lat'].min()
    orig_lon_range = df['lon'].max() - df['lon'].min()
    orig_area_deg2 = orig_lat_range * orig_lon_range
    
    crop_lat_range = lat_max - lat_min
    crop_lon_range = lon_max - lon_min
    crop_area_deg2 = crop_lat_range * crop_lon_range
    
    # Approximate area in km²
    km_per_deg = 111.0
    orig_area_km2 = orig_area_deg2 * (km_per_deg ** 2)
    crop_area_km2 = crop_area_deg2 * (km_per_deg ** 2)
    
    # Filter data
    mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & \
           (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    df_crop = df[mask]
    df_outside = df[~mask]
    
    n_total = len(df)
    n_crop = len(df_crop)
    n_outside = len(df_outside)
    
    # Density comparison
    density_original = n_total / orig_area_km2
    density_crop = n_crop / crop_area_km2
    density_outside = n_outside / (orig_area_km2 - crop_area_km2) if (orig_area_km2 - crop_area_km2) > 0 else 0
    
    # Magnitude statistics
    mw_crop_mean = df_crop['mw'].mean()
    mw_crop_max = df_crop['mw'].max()
    mw_outside_mean = df_outside['mw'].mean() if len(df_outside) > 0 else 0
    mw_outside_max = df_outside['mw'].max() if len(df_outside) > 0 else 0
    
    # Significant events (Mw >= 4)
    sig_crop = len(df_crop[df_crop['mw'] >= 4])
    sig_outside = len(df_outside[df_outside['mw'] >= 4]) if len(df_outside) > 0 else 0
    sig_total = sig_crop + sig_outside
    
    # Concentration metrics
    area_pct = crop_area_km2 / orig_area_km2 * 100
    event_pct = n_crop / n_total * 100
    concentration_factor = event_pct / area_pct if area_pct > 0 else 0
    
    # Generate summary table
    justification = {
        'Metrik': [
            'Luas area original (km²)',
            'Luas area crop (km²)',
            'Persentase area crop (%)',
            '',
            'Total event original',
            'Event dalam crop area',
            'Event di luar crop area',
            'Persentase event crop (%)',
            '',
            'Density original (event/km²)',
            'Density crop area (event/km²)',
            'Density luar crop (event/km²)',
            'Faktor konsentrasi',
            '',
            'Mw rata-rata (crop)',
            'Mw maksimum (crop)',
            'Mw rata-rata (luar)',
            'Mw maksimum (luar)',
            '',
            'Event signifikan Mw≥4 (crop)',
            'Event signifikan Mw≥4 (luar)',
            '% event signifikan dalam crop',
        ],
        'Nilai': [
            f'{orig_area_km2:,.0f}',
            f'{crop_area_km2:,.0f}',
            f'{area_pct:.1f}%',
            '',
            f'{n_total:,}',
            f'{n_crop:,}',
            f'{n_outside:,}',
            f'{event_pct:.1f}%',
            '',
            f'{density_original:.2f}',
            f'{density_crop:.2f}',
            f'{density_outside:.2f}',
            f'{concentration_factor:.1f}x',
            '',
            f'{mw_crop_mean:.2f}',
            f'{mw_crop_max:.2f}',
            f'{mw_outside_mean:.2f}',
            f'{mw_outside_max:.2f}',
            '',
            f'{sig_crop:,}',
            f'{sig_outside:,}',
            f'{sig_crop/sig_total*100:.1f}%' if sig_total > 0 else 'N/A',
        ]
    }
    
    df_justification = pd.DataFrame(justification)
    df_justification.to_csv(output_dir / 'tables' / 'crop_justification.csv', index=False)
    
    # Generate text summary
    summary_text = f"""
================================================================================
JUSTIFIKASI PEMILIHAN AREA CROP
================================================================================

PERBANDINGAN AREA:
- Area original: {orig_area_km2:,.0f} km² (lat: {df['lat'].min():.4f}° - {df['lat'].max():.4f}°, lon: {df['lon'].min():.4f}° - {df['lon'].max():.4f}°)
- Area crop: {crop_area_km2:,.0f} km² (lat: {lat_min}° - {lat_max}°, lon: {lon_min}° - {lon_max}°)
- Persentase area: {area_pct:.1f}% dari area original

DISTRIBUSI EVENT:
- Total event: {n_total:,}
- Event dalam crop: {n_crop:,} ({event_pct:.1f}%)
- Event di luar crop: {n_outside:,} ({100-event_pct:.1f}%)

ANALISIS DENSITAS:
- Densitas original: {density_original:.2f} event/km²
- Densitas crop area: {density_crop:.2f} event/km²
- Densitas luar crop: {density_outside:.2f} event/km²
- Faktor konsentrasi: {concentration_factor:.1f}x
  (Artinya: {event_pct:.1f}% event terkandung dalam {area_pct:.1f}% area)

ANALISIS MAGNITUDO:
- Crop: mean Mw = {mw_crop_mean:.2f}, max Mw = {mw_crop_max:.2f}
- Luar: mean Mw = {mw_outside_mean:.2f}, max Mw = {mw_outside_max:.2f}

EVENT SIGNIFIKAN (Mw ≥ 4.0):
- Dalam crop: {sig_crop:,} event
- Di luar crop: {sig_outside:,} event
- {sig_crop/sig_total*100:.1f}% event signifikan berada dalam crop area

KESIMPULAN:
Area crop dipilih karena:
1. Konsentrasi tinggi: {concentration_factor:.1f}x lipat lebih padat dari rata-rata
2. {event_pct:.1f}% total event dalam {area_pct:.1f}% area
3. {sig_crop/sig_total*100:.1f}% event signifikan (Mw≥4) berada di area ini
4. Area ini mencakup zona aktif Amatrice sequence 2016-2017
================================================================================
"""
    
    with open(output_dir / 'crop_justification.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    
    print(f"   Saved: crop_justification.csv")
    print(f"   Saved: crop_justification.txt")
    print(f"   Concentration factor: {concentration_factor:.1f}x")
    print(f"   {event_pct:.1f}% events in {area_pct:.1f}% area")
    
    return {
        'area_pct': area_pct,
        'event_pct': event_pct,
        'concentration_factor': concentration_factor,
        'density_crop': density_crop,
        'density_outside': density_outside,
        'sig_pct': sig_crop/sig_total*100 if sig_total > 0 else 0
    }


def generate_crop_sensitivity_analysis(df, output_dir):
    """
    Generate cumulative analysis showing how statistics change 
    when adjusting crop bounds (expanding or shrinking).
    """
    print("\n Generating crop sensitivity analysis...")
    
    # Current crop bounds from config
    lat_min_cfg = CONFIG.get('lat_min')
    lat_max_cfg = CONFIG.get('lat_max')
    lon_min_cfg = CONFIG.get('lon_min')
    lon_max_cfg = CONFIG.get('lon_max')
    
    # Center point of current crop
    lat_center = (lat_min_cfg + lat_max_cfg) / 2
    lon_center = (lon_min_cfg + lon_max_cfg) / 2
    
    # Current half-widths
    lat_half = (lat_max_cfg - lat_min_cfg) / 2
    lon_half = (lon_max_cfg - lon_min_cfg) / 2
    
    # Test different scale factors (0.5x to 2.0x of current bounds)
    scale_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    
    results = []
    n_total = len(df)
    km_per_deg = 111.0
    
    for scale in scale_factors:
        # Calculate new bounds
        lat_min = lat_center - lat_half * scale
        lat_max = lat_center + lat_half * scale
        lon_min = lon_center - lon_half * scale
        lon_max = lon_center + lon_half * scale
        
        # Filter events
        mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & \
               (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        df_crop = df[mask]
        n_crop = len(df_crop)
        
        # Area
        area_km2 = (lat_max - lat_min) * (lon_max - lon_min) * (km_per_deg ** 2)
        
        # Density
        density = n_crop / area_km2 if area_km2 > 0 else 0
        
        # Magnitude stats
        mw_mean = df_crop['mw'].mean() if n_crop > 0 else 0
        mw_max = df_crop['mw'].max() if n_crop > 0 else 0
        
        # Significant events
        n_sig = len(df_crop[df_crop['mw'] >= 4]) if n_crop > 0 else 0
        
        # Event percentage
        event_pct = n_crop / n_total * 100
        
        results.append({
            'Scale': f'{scale:.2f}x',
            'Lat Range': f'{lat_min:.3f} - {lat_max:.3f}',
            'Lon Range': f'{lon_min:.3f} - {lon_max:.3f}',
            'Area (km²)': round(area_km2, 0),
            'Events': n_crop,
            'Event %': round(event_pct, 1),
            'Density': round(density, 2),
            'Mean Mw': round(mw_mean, 2),
            'Max Mw': round(mw_max, 2),
            'Mw≥4': n_sig,
        })
    
    df_sensitivity = pd.DataFrame(results)
    df_sensitivity.to_csv(output_dir / 'tables' / 'crop_sensitivity.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scales = [float(r['Scale'].replace('x', '')) for r in results]
    events = [r['Events'] for r in results]
    densities = [r['Density'] for r in results]
    event_pcts = [r['Event %'] for r in results]
    areas = [r['Area (km²)'] for r in results]
    
    # Panel 1: Events vs Scale
    ax1 = axes[0, 0]
    ax1.plot(scales, events, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax1.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Current Crop')
    ax1.set_xlabel('Scale Factor')
    ax1.set_ylabel('Number of Events')
    ax1.set_title('Events vs Crop Scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Event % vs Scale
    ax2 = axes[0, 1]
    ax2.plot(scales, event_pcts, 'o-', color='green', linewidth=2, markersize=8)
    ax2.axvline(1.0, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(99, color='orange', linestyle=':', alpha=0.7, label='99% threshold')
    ax2.set_xlabel('Scale Factor')
    ax2.set_ylabel('Event Percentage (%)')
    ax2.set_title('Event Coverage vs Crop Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Density vs Scale
    ax3 = axes[1, 0]
    ax3.plot(scales, densities, 'o-', color='purple', linewidth=2, markersize=8)
    ax3.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Current Crop')
    ax3.set_xlabel('Scale Factor')
    ax3.set_ylabel('Density (events/km²)')
    ax3.set_title('Event Density vs Crop Scale')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Area vs Scale (log)
    ax4 = axes[1, 1]
    ax4.plot(scales, areas, 'o-', color='orange', linewidth=2, markersize=8)
    ax4.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Current Crop')
    ax4.set_xlabel('Scale Factor')
    ax4.set_ylabel('Area (km²)')
    ax4.set_title('Crop Area vs Scale Factor')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'crop_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate cumulative table by absolute bounds
    print("\n Generating cumulative bounds analysis...")
    
    # Test expanding each bound individually
    bound_tests = []
    
    for lat_expand in np.arange(-0.2, 0.25, 0.05):
        for lon_expand in np.arange(-0.2, 0.25, 0.05):
            lat_min = lat_min_cfg + lat_expand
            lat_max = lat_max_cfg - lat_expand
            lon_min = lon_min_cfg + lon_expand
            lon_max = lon_max_cfg - lon_expand
            
            if lat_min >= lat_max or lon_min >= lon_max:
                continue
            
            mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & \
                   (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
            n = mask.sum()
            
            area = (lat_max - lat_min) * (lon_max - lon_min) * (km_per_deg ** 2)
            
            bound_tests.append({
                'Lat Shrink': round(lat_expand, 2),
                'Lon Shrink': round(lon_expand, 2),
                'Events': n,
                'Event %': round(n / n_total * 100, 1),
                'Area (km²)': round(area, 0),
                'Density': round(n / area, 2) if area > 0 else 0
            })
    
    df_bounds = pd.DataFrame(bound_tests)
    df_bounds.to_csv(output_dir / 'tables' / 'crop_bounds_analysis.csv', index=False)
    
    print(f"   Saved: crop_sensitivity.csv")
    print(f"   Saved: crop_sensitivity.png")
    print(f"   Saved: crop_bounds_analysis.csv")
    
    return df_sensitivity


def generate_graph_degree_analysis(df, output_dir):
    """
    Generate EDA for graph degree/radius selection.
    Analyzes how different radius_km values affect graph structure.
    """
    print("\n Generating graph degree analysis...")
    
    lat_min = CONFIG.get('lat_min')
    lat_max = CONFIG.get('lat_max')
    lon_min = CONFIG.get('lon_min')
    lon_max = CONFIG.get('lon_max')
    grid_size = CONFIG.get('grid_size', 0.02)
    min_events = CONFIG.get('min_events_per_node', 10)
    
    # Filter to crop area
    mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & \
           (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    df_crop = df[mask].copy()
    
    # Create grid
    n_rows = int(np.ceil((lat_max - lat_min) / grid_size))
    n_cols = int(np.ceil((lon_max - lon_min) / grid_size))
    
    df_crop['row_idx'] = ((df_crop['lat'] - lat_min) / grid_size).astype(int).clip(0, n_rows - 1)
    df_crop['col_idx'] = ((df_crop['lon'] - lon_min) / grid_size).astype(int).clip(0, n_cols - 1)
    df_crop['node_id'] = df_crop['row_idx'] * n_cols + df_crop['col_idx']
    
    # Node stats
    node_stats = df_crop.groupby('node_id').agg({
        'lat': 'mean',
        'lon': 'mean',
        'mw': 'count'
    }).reset_index()
    node_stats.columns = ['node_id', 'lat', 'lon', 'count']
    
    # Qualified nodes
    qualified = node_stats[node_stats['count'] >= min_events]
    n_nodes = len(qualified)
    
    if n_nodes < 2:
        print("   Not enough qualified nodes for graph analysis")
        return None
    
    # Get coordinates in km
    coords = qualified[['lat', 'lon']].values
    coords_km = coords * 111.0
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    distances = cdist(coords_km, coords_km, metric='euclidean')
    
    # Test different radius values
    radius_values = [2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    
    results = []
    for radius in radius_values:
        # Count edges
        edge_mask = (distances > 0) & (distances <= radius)
        n_edges = edge_mask.sum() // 2
        
        # Degree statistics
        degrees = edge_mask.sum(axis=1)
        avg_degree = degrees.mean()
        max_degree = degrees.max()
        min_degree = degrees.min()
        
        # Isolated nodes (degree 0)
        n_isolated = (degrees == 0).sum()
        
        # Graph density
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges * 100 if max_edges > 0 else 0
        
        # Connectivity (% nodes with at least 1 edge)
        connectivity = (degrees > 0).sum() / n_nodes * 100
        
        results.append({
            'Radius (km)': radius,
            'Edges': n_edges,
            'Avg Degree': round(avg_degree, 2),
            'Min Degree': int(min_degree),
            'Max Degree': int(max_degree),
            'Isolated Nodes': n_isolated,
            'Connectivity %': round(connectivity, 1),
            'Density %': round(density, 2)
        })
    
    df_degree = pd.DataFrame(results)
    df_degree.to_csv(output_dir / 'tables' / 'graph_degree_analysis.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    radii = [r['Radius (km)'] for r in results]
    edges = [r['Edges'] for r in results]
    avg_degrees = [r['Avg Degree'] for r in results]
    connectivity = [r['Connectivity %'] for r in results]
    densities = [r['Density %'] for r in results]
    
    current_radius = CONFIG.get('radius_km', 15)
    
    # Panel 1: Edges vs Radius
    ax1 = axes[0, 0]
    ax1.plot(radii, edges, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax1.axvline(current_radius, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_radius} km)')
    ax1.set_xlabel('Radius (km)')
    ax1.set_ylabel('Number of Edges')
    ax1.set_title('Graph Edges vs Radius')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Average Degree vs Radius
    ax2 = axes[0, 1]
    ax2.plot(radii, avg_degrees, 'o-', color='green', linewidth=2, markersize=8)
    ax2.axvline(current_radius, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_radius} km)')
    ax2.axhline(4, color='orange', linestyle=':', alpha=0.7, label='Typical GAT (4)')
    ax2.axhline(8, color='purple', linestyle=':', alpha=0.7, label='Dense (8)')
    ax2.set_xlabel('Radius (km)')
    ax2.set_ylabel('Average Degree')
    ax2.set_title('Average Node Degree vs Radius')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Connectivity vs Radius
    ax3 = axes[1, 0]
    ax3.plot(radii, connectivity, 'o-', color='purple', linewidth=2, markersize=8)
    ax3.axvline(current_radius, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_radius} km)')
    ax3.axhline(100, color='green', linestyle=':', alpha=0.7, label='Full connectivity')
    ax3.set_xlabel('Radius (km)')
    ax3.set_ylabel('Connectivity (%)')
    ax3.set_title('Node Connectivity vs Radius')
    ax3.set_ylim(0, 105)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Density vs Radius
    ax4 = axes[1, 1]
    ax4.plot(radii, densities, 'o-', color='orange', linewidth=2, markersize=8)
    ax4.axvline(current_radius, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_radius} km)')
    ax4.set_xlabel('Radius (km)')
    ax4.set_ylabel('Graph Density (%)')
    ax4.set_title('Graph Density vs Radius')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Graph Structure Analysis (n={n_nodes} nodes, grid={grid_size}°, min_events={min_events})', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'graph_degree_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate degree distribution histogram for current radius
    current_edge_mask = (distances > 0) & (distances <= current_radius)
    current_degrees = current_edge_mask.sum(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(current_degrees, bins=range(0, int(current_degrees.max()) + 2), 
            edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(current_degrees.mean(), color='red', linestyle='--', 
               label=f'Mean: {current_degrees.mean():.1f}')
    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Degree Distribution (radius={current_radius} km)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'degree_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary text
    summary = f"""
================================================================================
GRAPH DEGREE ANALYSIS SUMMARY
================================================================================

KONFIGURASI:
- Grid size: {grid_size}° (~{grid_size*111:.1f} km)
- Min events per node: {min_events}
- Qualified nodes: {n_nodes}
- Current radius: {current_radius} km

STATISTIK PADA RADIUS {current_radius} km:
- Total edges: {edges[radii.index(current_radius)] if current_radius in radii else 'N/A'}
- Average degree: {avg_degrees[radii.index(current_radius)] if current_radius in radii else 'N/A'}
- Connectivity: {connectivity[radii.index(current_radius)] if current_radius in radii else 'N/A'}%
- Density: {densities[radii.index(current_radius)] if current_radius in radii else 'N/A'}%

REKOMENDASI:
- Untuk sparse graph (avg degree 2-4): radius 5-10 km
- Untuk medium density (avg degree 4-8): radius 10-20 km
- Untuk dense graph (avg degree 8+): radius 20+ km

DISTRIBUSI DEGREE (radius={current_radius} km):
- Min degree: {int(current_degrees.min())}
- Max degree: {int(current_degrees.max())}
- Mean degree: {current_degrees.mean():.2f}
- Std degree: {current_degrees.std():.2f}
================================================================================
"""
    
    with open(output_dir / 'graph_degree_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"   Saved: graph_degree_analysis.csv")
    print(f"   Saved: graph_degree_analysis.png")
    print(f"   Saved: degree_distribution.png")
    print(f"   Saved: graph_degree_summary.txt")
    print(f"   Nodes: {n_nodes}, Current radius: {current_radius} km, Avg degree: {current_degrees.mean():.1f}")
    
    return df_degree


def generate_grid_size_analysis(df, output_dir):
    """
    Generate EDA for grid_size selection.
    Analyzes how different grid sizes affect node count, event distribution, and coverage.
    """
    print("\n Generating grid size analysis...")
    
    lat_min = CONFIG.get('lat_min')
    lat_max = CONFIG.get('lat_max')
    lon_min = CONFIG.get('lon_min')
    lon_max = CONFIG.get('lon_max')
    min_events = CONFIG.get('min_events_per_node', 10)
    current_grid = CONFIG.get('grid_size', 0.02)
    
    # Filter to crop area
    mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & \
           (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    df_crop = df[mask].copy()
    n_total = len(df_crop)
    
    # Test different grid sizes
    grid_sizes = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2]
    
    results = []
    for grid_size in grid_sizes:
        # Create grid
        n_rows = int(np.ceil((lat_max - lat_min) / grid_size))
        n_cols = int(np.ceil((lon_max - lon_min) / grid_size))
        total_cells = n_rows * n_cols
        
        # Assign events to grid cells
        df_temp = df_crop.copy()
        df_temp['row_idx'] = ((df_temp['lat'] - lat_min) / grid_size).astype(int).clip(0, n_rows - 1)
        df_temp['col_idx'] = ((df_temp['lon'] - lon_min) / grid_size).astype(int).clip(0, n_cols - 1)
        df_temp['node_id'] = df_temp['row_idx'] * n_cols + df_temp['col_idx']
        
        # Node statistics
        node_counts = df_temp.groupby('node_id').size()
        n_active = len(node_counts)
        
        # Qualified nodes (meeting min_events threshold)
        qualified_mask = node_counts >= min_events
        n_qualified = qualified_mask.sum()
        
        # Events retained
        events_retained = node_counts[qualified_mask].sum()
        retention_pct = events_retained / n_total * 100 if n_total > 0 else 0
        
        # Events lost
        events_lost = n_total - events_retained
        
        # Grid cell size in km
        cell_size_km = grid_size * 111
        
        # Average events per qualified node
        avg_events = node_counts[qualified_mask].mean() if n_qualified > 0 else 0
        std_events = node_counts[qualified_mask].std() if n_qualified > 0 else 0
        
        # Coverage (% of cells with at least 1 event)
        coverage = n_active / total_cells * 100 if total_cells > 0 else 0
        
        results.append({
            'Grid Size (°)': grid_size,
            'Cell Size (km)': round(cell_size_km, 1),
            'Grid Dims': f'{n_rows}x{n_cols}',
            'Total Cells': total_cells,
            'Active Cells': n_active,
            'Qualified Nodes': n_qualified,
            'Events Retained': events_retained,
            'Retention %': round(retention_pct, 1),
            'Events Lost': events_lost,
            'Avg Events/Node': round(avg_events, 1),
            'Std Events/Node': round(std_events, 1),
            'Coverage %': round(coverage, 1)
        })
    
    df_grid = pd.DataFrame(results)
    df_grid.to_csv(output_dir / 'tables' / 'grid_size_analysis.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    grids = [r['Grid Size (°)'] for r in results]
    qualified = [r['Qualified Nodes'] for r in results]
    retention = [r['Retention %'] for r in results]
    avg_events = [r['Avg Events/Node'] for r in results]
    total_cells = [r['Total Cells'] for r in results]
    coverage = [r['Coverage %'] for r in results]
    events_lost = [r['Events Lost'] for r in results]
    
    # Panel 1: Qualified Nodes vs Grid Size
    ax1 = axes[0, 0]
    ax1.plot(grids, qualified, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax1.axvline(current_grid, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_grid}°)')
    ax1.set_xlabel('Grid Size (°)')
    ax1.set_ylabel('Qualified Nodes')
    ax1.set_title('Qualified Nodes vs Grid Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Retention % vs Grid Size
    ax2 = axes[0, 1]
    ax2.plot(grids, retention, 'o-', color='green', linewidth=2, markersize=8)
    ax2.axvline(current_grid, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_grid}°)')
    ax2.axhline(95, color='orange', linestyle=':', alpha=0.7, label='95% threshold')
    ax2.set_xlabel('Grid Size (°)')
    ax2.set_ylabel('Event Retention (%)')
    ax2.set_title('Event Retention vs Grid Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Average Events per Node vs Grid Size
    ax3 = axes[0, 2]
    ax3.plot(grids, avg_events, 'o-', color='purple', linewidth=2, markersize=8)
    ax3.axvline(current_grid, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_grid}°)')
    ax3.set_xlabel('Grid Size (°)')
    ax3.set_ylabel('Avg Events/Node')
    ax3.set_title('Average Events per Node vs Grid Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Total Cells vs Grid Size (log scale)
    ax4 = axes[1, 0]
    ax4.semilogy(grids, total_cells, 'o-', color='orange', linewidth=2, markersize=8)
    ax4.axvline(current_grid, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_grid}°)')
    ax4.set_xlabel('Grid Size (°)')
    ax4.set_ylabel('Total Cells (log)')
    ax4.set_title('Grid Complexity vs Grid Size')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Events Lost vs Grid Size
    ax5 = axes[1, 1]
    ax5.plot(grids, events_lost, 'o-', color='red', linewidth=2, markersize=8)
    ax5.axvline(current_grid, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_grid}°)')
    ax5.set_xlabel('Grid Size (°)')
    ax5.set_ylabel('Events Lost')
    ax5.set_title('Data Loss vs Grid Size')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Trade-off visualization (Nodes vs Retention)
    ax6 = axes[1, 2]
    scatter = ax6.scatter(qualified, retention, c=grids, cmap='viridis', s=100)
    for i, g in enumerate(grids):
        ax6.annotate(f'{g}°', (qualified[i], retention[i]), fontsize=8, ha='left')
    # Highlight current
    current_idx = grids.index(current_grid) if current_grid in grids else None
    if current_idx is not None:
        ax6.scatter([qualified[current_idx]], [retention[current_idx]], 
                   s=200, facecolors='none', edgecolors='red', linewidths=2, label='Current')
    ax6.set_xlabel('Qualified Nodes')
    ax6.set_ylabel('Retention %')
    ax6.set_title('Trade-off: Nodes vs Retention')
    plt.colorbar(scatter, ax=ax6, label='Grid Size (°)')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Grid Size Analysis (min_events={min_events}, total events={n_total:,})', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'grid_size_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary text
    current_result = next((r for r in results if r['Grid Size (°)'] == current_grid), None)
    
    summary = f"""
================================================================================
GRID SIZE ANALYSIS SUMMARY
================================================================================

KONFIGURASI:
- Crop area: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]
- Min events per node: {min_events}
- Total events in crop: {n_total:,}
- Current grid size: {current_grid}° (~{current_grid*111:.1f} km)

STATISTIK PADA GRID SIZE {current_grid}°:
- Grid dimensions: {current_result['Grid Dims'] if current_result else 'N/A'}
- Total cells: {current_result['Total Cells'] if current_result else 'N/A'}
- Qualified nodes: {current_result['Qualified Nodes'] if current_result else 'N/A'}
- Event retention: {current_result['Retention %'] if current_result else 'N/A'}%
- Events lost: {current_result['Events Lost'] if current_result else 'N/A'}
- Avg events/node: {current_result['Avg Events/Node'] if current_result else 'N/A'}

TRADE-OFF:
- Grid lebih kecil → lebih banyak nodes, resolusi tinggi, tapi lebih banyak data loss
- Grid lebih besar → lebih sedikit nodes, resolusi rendah, tapi retention lebih tinggi

REKOMENDASI:
- Fine resolution (high nodes, lower retention): 0.01-0.015°
- Balanced (medium nodes, good retention): 0.02-0.03°
- Coarse (fewer nodes, high retention): 0.05-0.1°

PERBANDINGAN:
"""
    for r in results:
        is_current = ' ← CURRENT' if r['Grid Size (°)'] == current_grid else ''
        summary += f"  {r['Grid Size (°)']}° → {r['Qualified Nodes']} nodes, {r['Retention %']}% retention{is_current}\n"
    
    summary += """
================================================================================
"""
    
    with open(output_dir / 'grid_size_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"   Saved: grid_size_analysis.csv")
    print(f"   Saved: grid_size_analysis.png")
    print(f"   Saved: grid_size_summary.txt")
    if current_result:
        print(f"   Current: {current_grid}° → {current_result['Qualified Nodes']} nodes, {current_result['Retention %']}% retention")
    
    return df_grid


def generate_time_bin_analysis(df, output_dir):
    """
    Generate EDA for time_bin selection.
    Analyzes how different temporal binning affects sequence length, sparsity, and event distribution.
    """
    print("\n Generating time bin analysis...")
    
    lat_min = CONFIG.get('lat_min')
    lat_max = CONFIG.get('lat_max')
    lon_min = CONFIG.get('lon_min')
    lon_max = CONFIG.get('lon_max')
    
    # Parse time_bin from config (e.g., "4h" -> 4)
    time_bin_str = CONFIG.get('time_bin', '4h')
    if isinstance(time_bin_str, str):
        current_bin = int(time_bin_str.replace('h', ''))
    else:
        current_bin = time_bin_str
    
    # Filter to crop area
    mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & \
           (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    df_crop = df[mask].copy()
    
    # Parse datetime
    if 'datetime' not in df_crop.columns:
        df_crop['datetime'] = pd.to_datetime(df_crop[['year', 'month', 'day', 'hour', 'minute', 'second']])
    
    df_crop['timestamp'] = df_crop['datetime'].astype(np.int64) // 10**9
    
    n_events = len(df_crop)
    min_ts = df_crop['timestamp'].min()
    max_ts = df_crop['timestamp'].max()
    total_hours = (max_ts - min_ts) / 3600
    total_days = total_hours / 24
    
    # Test different time bin sizes (in hours)
    time_bins = [1, 2, 4, 6, 8, 12, 24, 48, 72, 168]  # 168h = 1 week
    
    results = []
    for bin_hours in time_bins:
        bin_seconds = bin_hours * 3600
        
        # Number of time steps
        n_steps = int(np.ceil((max_ts - min_ts) / bin_seconds)) + 1
        
        # Assign events to time bins
        df_crop['time_bin'] = ((df_crop['timestamp'] - min_ts) // bin_seconds).astype(int)
        
        # Count events per bin
        events_per_bin = df_crop.groupby('time_bin').size()
        
        # Statistics
        n_active_bins = len(events_per_bin)
        n_empty_bins = n_steps - n_active_bins
        sparsity = n_empty_bins / n_steps * 100 if n_steps > 0 else 0
        
        avg_events = events_per_bin.mean()
        std_events = events_per_bin.std()
        max_events = events_per_bin.max()
        min_events = events_per_bin.min()
        
        # Bins with significant activity (>10 events)
        active_10 = (events_per_bin >= 10).sum()
        
        results.append({
            'Time Bin (h)': bin_hours,
            'Time Bin Label': f'{bin_hours}h' if bin_hours < 24 else f'{bin_hours//24}d',
            'Total Steps': n_steps,
            'Active Bins': n_active_bins,
            'Empty Bins': n_empty_bins,
            'Sparsity %': round(sparsity, 1),
            'Avg Events/Bin': round(avg_events, 1),
            'Std Events/Bin': round(std_events, 1),
            'Max Events/Bin': int(max_events),
            'Min Events/Bin': int(min_events),
            'Bins with ≥10': active_10
        })
    
    df_time = pd.DataFrame(results)
    df_time.to_csv(output_dir / 'tables' / 'time_bin_analysis.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    bins = [r['Time Bin (h)'] for r in results]
    steps = [r['Total Steps'] for r in results]
    sparsity = [r['Sparsity %'] for r in results]
    avg_events = [r['Avg Events/Bin'] for r in results]
    max_events = [r['Max Events/Bin'] for r in results]
    active_bins = [r['Active Bins'] for r in results]
    bins_10 = [r['Bins with ≥10'] for r in results]
    
    # Panel 1: Sequence Length vs Time Bin
    ax1 = axes[0, 0]
    ax1.semilogy(bins, steps, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax1.axvline(current_bin, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_bin}h)')
    ax1.set_xlabel('Time Bin (hours)')
    ax1.set_ylabel('Sequence Length (log)')
    ax1.set_title('Sequence Length vs Time Bin')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Sparsity vs Time Bin
    ax2 = axes[0, 1]
    ax2.plot(bins, sparsity, 'o-', color='orange', linewidth=2, markersize=8)
    ax2.axvline(current_bin, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_bin}h)')
    ax2.axhline(50, color='green', linestyle=':', alpha=0.7, label='50% threshold')
    ax2.set_xlabel('Time Bin (hours)')
    ax2.set_ylabel('Sparsity (%)')
    ax2.set_title('Temporal Sparsity vs Time Bin')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Average Events per Bin vs Time Bin
    ax3 = axes[0, 2]
    ax3.plot(bins, avg_events, 'o-', color='green', linewidth=2, markersize=8)
    ax3.axvline(current_bin, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_bin}h)')
    ax3.set_xlabel('Time Bin (hours)')
    ax3.set_ylabel('Avg Events/Bin')
    ax3.set_title('Average Events per Bin vs Time Bin')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Max Events per Bin vs Time Bin
    ax4 = axes[1, 0]
    ax4.plot(bins, max_events, 'o-', color='purple', linewidth=2, markersize=8)
    ax4.axvline(current_bin, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_bin}h)')
    ax4.set_xlabel('Time Bin (hours)')
    ax4.set_ylabel('Max Events/Bin')
    ax4.set_title('Peak Activity vs Time Bin')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Active Bins vs Time Bin
    ax5 = axes[1, 1]
    ax5.plot(bins, active_bins, 'o-', color='teal', linewidth=2, markersize=8)
    ax5.axvline(current_bin, color='red', linestyle='--', alpha=0.7, label=f'Current ({current_bin}h)')
    ax5.set_xlabel('Time Bin (hours)')
    ax5.set_ylabel('Active Bins (non-empty)')
    ax5.set_title('Active Time Steps vs Time Bin')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Trade-off (Sequence Length vs Sparsity)
    ax6 = axes[1, 2]
    scatter = ax6.scatter(steps, sparsity, c=bins, cmap='viridis', s=100)
    for i, b in enumerate(bins):
        ax6.annotate(f'{b}h', (steps[i], sparsity[i]), fontsize=8, ha='left')
    current_idx = bins.index(current_bin) if current_bin in bins else None
    if current_idx is not None:
        ax6.scatter([steps[current_idx]], [sparsity[current_idx]], 
                   s=200, facecolors='none', edgecolors='red', linewidths=2, label='Current')
    ax6.set_xlabel('Sequence Length')
    ax6.set_ylabel('Sparsity (%)')
    ax6.set_title('Trade-off: Length vs Sparsity')
    plt.colorbar(scatter, ax=ax6, label='Time Bin (h)')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Time Bin Analysis (total events={n_events:,}, duration={total_days:.0f} days)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'time_bin_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate event distribution histogram for current time bin
    bin_seconds = current_bin * 3600
    df_crop['time_bin'] = ((df_crop['timestamp'] - min_ts) // bin_seconds).astype(int)
    events_per_bin = df_crop.groupby('time_bin').size()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(events_per_bin)), events_per_bin.values, width=1.0, alpha=0.7, color='steelblue')
    ax.axhline(events_per_bin.mean(), color='red', linestyle='--', label=f'Mean: {events_per_bin.mean():.1f}')
    ax.set_xlabel(f'Time Step ({current_bin}h bins)')
    ax.set_ylabel('Event Count')
    ax.set_title(f'Temporal Event Distribution (time_bin={current_bin}h)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'temporal_event_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary text
    current_result = next((r for r in results if r['Time Bin (h)'] == current_bin), None)
    
    summary = f"""
================================================================================
TIME BIN ANALYSIS SUMMARY
================================================================================

DATA OVERVIEW:
- Total events: {n_events:,}
- Time range: {df_crop['datetime'].min()} to {df_crop['datetime'].max()}
- Duration: {total_days:.1f} days ({total_hours:.0f} hours)
- Current time bin: {current_bin}h

STATISTIK PADA TIME BIN {current_bin}h:
- Sequence length: {current_result['Total Steps'] if current_result else 'N/A'} steps
- Active bins: {current_result['Active Bins'] if current_result else 'N/A'}
- Empty bins: {current_result['Empty Bins'] if current_result else 'N/A'}
- Sparsity: {current_result['Sparsity %'] if current_result else 'N/A'}%
- Avg events/bin: {current_result['Avg Events/Bin'] if current_result else 'N/A'}
- Max events/bin: {current_result['Max Events/Bin'] if current_result else 'N/A'}

TRADE-OFF:
- Time bin kecil → sequence panjang, sparsity tinggi, granularity tinggi
- Time bin besar → sequence pendek, sparsity rendah, granularity rendah

REKOMENDASI:
- High resolution (fine temporal detail): 1-2h
- Balanced (moderate seq length, acceptable sparsity): 4-6h
- Low resolution (shorter sequence, low sparsity): 12-24h

UNTUK LSTM TRAINING:
- Sequence terlalu panjang → training lambat, memory tinggi
- Sparsity tinggi → banyak zero-padding, less informative
- Target: sparsity < 50%, sequence < 10000 steps

PERBANDINGAN:
"""
    for r in results:
        is_current = ' ← CURRENT' if r['Time Bin (h)'] == current_bin else ''
        summary += f"  {r['Time Bin (h)']}h → {r['Total Steps']} steps, {r['Sparsity %']}% sparse, avg {r['Avg Events/Bin']} events{is_current}\n"
    
    summary += """
================================================================================
"""
    
    with open(output_dir / 'time_bin_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"   Saved: time_bin_analysis.csv")
    print(f"   Saved: time_bin_analysis.png")
    print(f"   Saved: temporal_event_distribution.png")
    print(f"   Saved: time_bin_summary.txt")
    if current_result:
        print(f"   Current: {current_bin}h → {current_result['Total Steps']} steps, {current_result['Sparsity %']}% sparse")
    
    return df_time

def plot_grid_and_graph(df_cropped, output_dir):
    """
    Visualize:
    1. Spatial grid structure
    2. Active nodes (nodes with min_events threshold)
    3. Graph adjacency structure
    """
    print("\n Plotting grid and graph structure...")
    
    grid_size = CONFIG['grid_size']
    lat_min = CONFIG.get('lat_min')
    lat_max = CONFIG.get('lat_max')
    lon_min = CONFIG.get('lon_min')
    lon_max = CONFIG.get('lon_max')
    min_events = CONFIG.get('min_events_per_node', 1)
    radius_km = CONFIG.get('radius_km', 15.0)
    
    # Create grid
    n_rows = int(np.ceil((lat_max - lat_min) / grid_size))
    n_cols = int(np.ceil((lon_max - lon_min) / grid_size))
    total_cells = n_rows * n_cols
    
    # Assign events to grid cells
    df_cropped = df_cropped.copy()
    df_cropped['row_idx'] = ((df_cropped['lat'] - lat_min) / grid_size).astype(int).clip(0, n_rows - 1)
    df_cropped['col_idx'] = ((df_cropped['lon'] - lon_min) / grid_size).astype(int).clip(0, n_cols - 1)
    df_cropped['node_id'] = df_cropped['row_idx'] * n_cols + df_cropped['col_idx']
    
    # Count events per cell
    cell_counts = df_cropped.groupby('node_id').size()
    
    # Active nodes (meeting threshold)
    active_nodes = cell_counts[cell_counts >= min_events].index.values
    n_active = len(active_nodes)
    
    # Calculate node coordinates
    node_coords = {}
    for node_id in range(total_cells):
        row = node_id // n_cols
        col = node_id % n_cols
        lat = lat_min + (row + 0.5) * grid_size
        lon = lon_min + (col + 0.5) * grid_size
        node_coords[node_id] = (lon, lat)
    
    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # ===== Panel 1: Grid Structure =====
    ax1 = axes[0]
    
    # Draw grid lines
    for i in range(n_rows + 1):
        y = lat_min + i * grid_size
        ax1.axhline(y, color='gray', linewidth=0.5, alpha=0.5)
    for j in range(n_cols + 1):
        x = lon_min + j * grid_size
        ax1.axvline(x, color='gray', linewidth=0.5, alpha=0.5)
    
    # Plot events
    ax1.scatter(df_cropped['lon'], df_cropped['lat'], c='steelblue', alpha=0.3, s=1)
    ax1.set_xlim(lon_min - 0.01, lon_max + 0.01)
    ax1.set_ylim(lat_min - 0.01, lat_max + 0.01)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'1. Grid Structure\n({n_rows} × {n_cols} = {total_cells} cells, grid = {grid_size}°)')
    
    # ===== Panel 2: Active Nodes =====
    ax2 = axes[1]
    
    # Create heatmap of event counts
    count_matrix = np.zeros((n_rows, n_cols))
    for node_id, count in cell_counts.items():
        row = node_id // n_cols
        col = node_id % n_cols
        count_matrix[row, col] = count
    
    # Plot heatmap
    im = ax2.imshow(count_matrix, origin='lower', cmap='YlOrRd',
                    extent=[lon_min, lon_max, lat_min, lat_max],
                    aspect='auto')
    plt.colorbar(im, ax=ax2, label='Event Count', shrink=0.8)
    
    # Highlight active nodes
    for node_id in active_nodes:
        row = node_id // n_cols
        col = node_id % n_cols
        lon, lat = node_coords[node_id]
        ax2.plot(lon, lat, 'ko', markersize=5, alpha=0.7)
    
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title(f'2. Active Nodes (≥ {min_events} events)\n({n_active} / {total_cells} active nodes)')
    
    # ===== Panel 3: Graph Structure =====
    ax3 = axes[2]
    
    # Build graph for active nodes only
    G = nx.Graph()
    active_coords = {}
    
    for i, node_id in enumerate(sorted(active_nodes)):
        lon, lat = node_coords[node_id]
        active_coords[i] = (lon, lat)
        G.add_node(i, pos=(lon, lat))
    
    # Add edges based on distance
    coord_array = np.array(list(active_coords.values()))
    
    # Convert lat/lon to approximate km (rough approximation)
    lat_km = 111.0  # km per degree latitude
    lon_km = 111.0 * np.cos(np.radians(np.mean(coord_array[:, 1])))  # km per degree longitude
    
    for i in range(len(coord_array)):
        for j in range(i + 1, len(coord_array)):
            dx = (coord_array[j, 0] - coord_array[i, 0]) * lon_km
            dy = (coord_array[j, 1] - coord_array[i, 1]) * lat_km
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radius_km:
                G.add_edge(i, j, weight=1.0 / (dist + 0.1))
    
    # Draw graph
    pos = {i: active_coords[i] for i in range(len(active_coords))}
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax3, alpha=0.3, edge_color='gray', width=0.5)
    
    # Draw nodes with color based on event count
    node_colors = [cell_counts.get(sorted(active_nodes)[i], 0) for i in range(len(active_nodes))]
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax3, node_size=50, 
                                    node_color=node_colors, cmap='YlOrRd',
                                    alpha=0.8)
    
    ax3.set_xlim(lon_min - 0.01, lon_max + 0.01)
    ax3.set_ylim(lat_min - 0.01, lat_max + 0.01)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_title(f'3. Graph Structure\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges, radius = {radius_km} km)')
    
    # Add colorbar for nodes
    sm = ScalarMappable(cmap='YlOrRd', norm=Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax3, label='Event Count', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'grid_and_graph.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: grid_and_graph.png")
    print(f"   Grid: {n_rows} × {n_cols} = {total_cells} cells")
    print(f"   Active nodes: {n_active}")
    print(f"   Graph edges: {G.number_of_edges()}")
    
    return G, {'total_cells': total_cells, 'active_nodes': n_active, 'edges': G.number_of_edges()}

def plot_spatial_with_basemap(df, output_dir):
    """
    Plot spatial distribution with real Italy basemap.
    Uses cartopy for coastlines, borders, and terrain.
    Falls back to simple plot if cartopy not available.
    """
    print("\n Plotting spatial distribution with real map...")
    
    lat_min = CONFIG.get('lat_min')
    lat_max = CONFIG.get('lat_max')
    lon_min = CONFIG.get('lon_min')
    lon_max = CONFIG.get('lon_max')
    
    if HAS_CARTOPY:
        # ===== CARTOPY VERSION (Real Basemap) =====
        fig = plt.figure(figsize=(18, 7))
        
        # Panel 1: Wide view of Central Italy
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        
        # Set extent for central Italy
        ax1.set_extent([11.5, 14.5, 41.0, 44.0], crs=ccrs.PlateCarree())
        
        # Add map features
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax1.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax1.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
        ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax1.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.5)
        ax1.add_feature(cfeature.RIVERS, edgecolor='blue', alpha=0.3)
        
        # Add gridlines
        gl = ax1.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Plot earthquakes
        scatter1 = ax1.scatter(df['lon'], df['lat'], c=df['mw'], cmap='YlOrRd',
                               alpha=0.5, s=3, vmin=0, vmax=5, transform=ccrs.PlateCarree())
        
        # Draw study area box
        if lat_min and lon_min:
            rect = mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                       linewidth=2, edgecolor='blue', facecolor='none',
                                       linestyle='-', transform=ccrs.PlateCarree())
            ax1.add_patch(rect)
        
        # Add cities
        cities = {
            'Roma': (12.4964, 41.9028),
            'Amatrice': (13.2867, 42.6288),
            'Norcia': (13.0917, 42.7917),
            "L'Aquila": (13.3996, 42.3505),
            'Perugia': (12.3908, 43.1107),
        }
        for city, (lon, lat) in cities.items():
            ax1.plot(lon, lat, 'k^', markersize=6, transform=ccrs.PlateCarree())
            ax1.text(lon + 0.05, lat + 0.05, city, fontsize=8, transform=ccrs.PlateCarree())
        
        ax1.set_title('Geographic Context: Central Italy\n(Amatrice Sequence 2016-2017)')
        plt.colorbar(scatter1, ax=ax1, label='Magnitude (Mw)', shrink=0.7, pad=0.02)
        
        # Panel 2: Zoomed study area
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        
        # Set extent to study area with margin
        margin = 0.05
        ax2.set_extent([lon_min - margin, lon_max + margin, 
                        lat_min - margin, lat_max + margin], crs=ccrs.PlateCarree())
        
        # Add map features
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
        
        # Add gridlines
        gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
        gl2.top_labels = False
        gl2.right_labels = False
        
        # Filter to study area
        mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & \
               (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        df_study = df[mask]
        df_sig = df_study[df_study['mw'] >= 4.0]
        
        # Plot events
        scatter2 = ax2.scatter(df_study['lon'], df_study['lat'], c=df_study['mw'],
                               cmap='YlOrRd', alpha=0.6, s=8, vmin=0, vmax=5,
                               transform=ccrs.PlateCarree())
        
        # Highlight significant events
        ax2.scatter(df_sig['lon'], df_sig['lat'], c='none', edgecolors='black',
                    s=80, linewidths=2, transform=ccrs.PlateCarree(),
                    label=f'Mw ≥ 4.0 (n={len(df_sig)})')
        
        ax2.set_title(f'Study Area (Zoom)\nn = {len(df_study):,} events')
        ax2.legend(loc='upper right')
        plt.colorbar(scatter2, ax=ax2, label='Magnitude (Mw)', shrink=0.7, pad=0.02)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'figures' / 'spatial_real_map.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: spatial_real_map.png (with real basemap)")
        
    else:
        # ===== FALLBACK VERSION (No cartopy) =====
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Panel 1: Wide view
        ax1 = axes[0]
        scatter1 = ax1.scatter(df['lon'], df['lat'], c=df['mw'], cmap='YlOrRd',
                              alpha=0.4, s=3, vmin=0, vmax=5)
        
        if lat_min and lon_min:
            rect = mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                       linewidth=2, edgecolor='blue', facecolor='none',
                                       linestyle='-', label='Study Area')
            ax1.add_patch(rect)
        
        cities = {
            'Roma': (12.4964, 41.9028),
            'Amatrice': (13.2867, 42.6288),
            'Norcia': (13.0917, 42.7917),
            "L'Aquila": (13.3996, 42.3505),
        }
        for city, (lon, lat) in cities.items():
            ax1.plot(lon, lat, 'k^', markersize=8)
            ax1.annotate(city, (lon, lat), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax1.set_xlim(11.5, 14.5)
        ax1.set_ylim(41.0, 44.0)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Geographic Context: Central Italy')
        ax1.legend(loc='upper left')
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1, label='Magnitude (Mw)', shrink=0.7)
        
        # Panel 2: Zoomed area
        ax2 = axes[1]
        mask = (df['lat'] >= lat_min) & (df['lat'] <= lat_max) & \
               (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        df_study = df[mask]
        df_sig = df_study[df_study['mw'] >= 4.0]
        
        scatter2 = ax2.scatter(df_study['lon'], df_study['lat'], c=df_study['mw'],
                              cmap='YlOrRd', alpha=0.5, s=5, vmin=0, vmax=5)
        ax2.scatter(df_sig['lon'], df_sig['lat'], c='none', edgecolors='black',
                   s=50, linewidths=1.5, label=f'Mw ≥ 4.0 (n={len(df_sig)})')
        
        ax2.set_xlim(lon_min - 0.02, lon_max + 0.02)
        ax2.set_ylim(lat_min - 0.02, lat_max + 0.02)
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(f'Study Area: Amatrice Sequence\n(n = {len(df_study):,} events)')
        ax2.legend(loc='upper right')
        ax2.set_aspect('equal')
        plt.colorbar(scatter2, ax=ax2, label='Magnitude (Mw)', shrink=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'figures' / 'spatial_with_context.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   Saved: spatial_with_context.png (simple plot - install cartopy for real map)")

# ==============================================================================
# TABLE GENERATION FUNCTIONS
# ==============================================================================

def generate_overview_table(df, output_dir):
    """Generate dataset overview table."""
    print("\n Generating overview table...")
    
    overview = {
        'Parameter': [
            'Total event',
            'Periode observasi',
            'Durasi (hari)',
            'Rentang latitude',
            'Rentang longitude',
            'Rentang magnitudo (Mw)',
            'Kedalaman min (km)',
            'Kedalaman max (km)',
            'Kedalaman rata-rata (km)',
        ],
        'Nilai': [
            f'{len(df):,}',
            f'{df["datetime"].min().strftime("%Y-%m-%d")} - {df["datetime"].max().strftime("%Y-%m-%d")}',
            f'{(df["datetime"].max() - df["datetime"].min()).days}',
            f'{df["lat"].min():.4f}° - {df["lat"].max():.4f}°',
            f'{df["lon"].min():.4f}° - {df["lon"].max():.4f}°',
            f'{df["mw"].min():.2f} - {df["mw"].max():.2f}',
            f'{df["dep"].min():.1f}',
            f'{df["dep"].max():.1f}',
            f'{df["dep"].mean():.1f}',
        ]
    }
    
    df_overview = pd.DataFrame(overview)
    df_overview.to_csv(output_dir / 'tables' / 'dataset_overview.csv', index=False)
    
    print(f"   Saved: dataset_overview.csv")
    return df_overview

def generate_magnitude_statistics(df, output_dir):
    """Generate magnitude statistics table."""
    print("\n Generating magnitude statistics...")
    
    bins = [-1, 0, 1, 2, 3, 4, 5, 7]
    labels = ['< 0', '0-1', '1-2', '2-3', '3-4', '4-5', '> 5']
    df = df.copy()
    df['mw_bin'] = pd.cut(df['mw'], bins=bins, labels=labels)
    
    mag_stats = df.groupby('mw_bin', observed=True).agg({
        'mw': ['count', 'mean', 'std', 'min', 'max']
    }).round(3)
    mag_stats.columns = ['Jumlah', 'Mean', 'Std', 'Min', 'Max']
    mag_stats['Persentase'] = (mag_stats['Jumlah'] / len(df) * 100).round(2)
    mag_stats = mag_stats.reset_index()
    mag_stats.columns = ['Rentang Mw', 'Jumlah', 'Mean', 'Std', 'Min', 'Max', 'Persentase (%)']
    
    mag_stats.to_csv(output_dir / 'tables' / 'magnitude_distribution.csv', index=False)
    
    print(f"   Saved: magnitude_distribution.csv")
    return mag_stats

def generate_preprocessing_summary(crop_stats, graph_stats, output_dir):
    """Generate preprocessing summary table."""
    print("\n Generating preprocessing summary...")
    
    summary = {
        'Tahap': [
            'Data mentah',
            'Setelah crop area',
            'Event dihapus',
            'Total sel grid',
            'Node aktif',
            'Edge dalam graph',
        ],
        'Nilai': [
            f'{crop_stats["total"]:,}',
            f'{crop_stats["retained"]:,}',
            f'{crop_stats["removed"]:,} ({crop_stats["removed"]/crop_stats["total"]*100:.1f}%)',
            f'{graph_stats["total_cells"]:,}',
            f'{graph_stats["active_nodes"]}',
            f'{graph_stats["edges"]}',
        ]
    }
    
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(output_dir / 'tables' / 'preprocessing_summary.csv', index=False)
    
    print(f"   Saved: preprocessing_summary.csv")
    return df_summary

# ==============================================================================
# ADDITIONAL VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_magnitude_histogram(df, output_dir):
    """Plot magnitude distribution histogram."""
    print("\n Plotting magnitude histogram...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df['mw'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Magnitude (Mw)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Magnitude Distribution (Linear Scale)')
    axes[0].axvline(df['mw'].mean(), color='red', linestyle='--', label=f'Mean: {df["mw"].mean():.2f}')
    axes[0].legend()
    
    axes[1].hist(df['mw'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1].set_xlabel('Magnitude (Mw)')
    axes[1].set_ylabel('Frequency (log scale)')
    axes[1].set_title('Magnitude Distribution (Log Scale)')
    axes[1].set_yscale('log')
    axes[1].axvline(df['mw'].mean(), color='red', linestyle='--', label=f'Mean: {df["mw"].mean():.2f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'magnitude_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: magnitude_histogram.png")

def plot_temporal_distribution(df, output_dir):
    """Plot earthquake count over time."""
    print("\n Plotting temporal distribution...")
    
    df = df.copy()
    df['date'] = df['datetime'].dt.date
    daily = df.groupby('date').agg({
        'mw': ['count', 'max', 'mean']
    }).reset_index()
    daily.columns = ['date', 'count', 'max_mw', 'mean_mw']
    daily['date'] = pd.to_datetime(daily['date'])
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Daily count
    axes[0].fill_between(daily['date'], daily['count'], alpha=0.5, color='steelblue')
    axes[0].plot(daily['date'], daily['count'], color='steelblue', linewidth=0.5)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Event Count')
    axes[0].set_title('Daily Earthquake Event Count')
    
    # Mark significant events
    mainshock_date = pd.Timestamp('2016-08-24')
    if mainshock_date >= daily['date'].min() and mainshock_date <= daily['date'].max():
        axes[0].axvline(mainshock_date, color='red', linestyle='--', alpha=0.7, label='Mainshock 24 Aug 2016')
        axes[0].legend()
    
    # Max magnitude per day
    axes[1].scatter(daily['date'], daily['max_mw'], alpha=0.5, s=15, c='indianred')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Max Magnitude (Mw)')
    axes[1].set_title('Daily Maximum Magnitude')
    axes[1].axhline(y=4.0, color='orange', linestyle='--', alpha=0.7, label='Mw = 4.0')
    axes[1].axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Mw = 5.0')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'temporal_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: temporal_distribution.png")

def plot_gutenberg_richter(df, output_dir):
    """Plot Gutenberg-Richter distribution."""
    print("\n Plotting Gutenberg-Richter relation...")
    
    mags = np.arange(0, 6.5, 0.1)
    counts = [len(df[df['mw'] >= m]) for m in mags]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.semilogy(mags, counts, 'o-', markersize=4, color='steelblue')
    ax.set_xlabel('Magnitude (Mw)')
    ax.set_ylabel('Cumulative Event Count (≥ M)')
    ax.set_title('Gutenberg-Richter Relation')
    ax.grid(True, alpha=0.3)
    
    # Fit for b-value
    mask = (mags >= 1) & (mags <= 4) & (np.array(counts) > 0)
    if mask.sum() > 2:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            mags[mask], np.log10(np.array(counts)[mask])
        )
        b_value = -slope
        
        fit_y = 10**(intercept + slope * mags)
        ax.semilogy(mags, fit_y, '--', color='red', 
                   label=f'b-value = {b_value:.2f} (R² = {r_value**2:.3f})')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'gutenberg_richter.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved: gutenberg_richter.png")

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main function to generate all EDA outputs."""
    print("=" * 70)
    print(" EARTHQUAKE DATA EDA - PREPROCESSING VISUALIZATION")
    print("=" * 70)
    
    # Setup output directories
    output_dir = Path(CONFIG.get('output_dir', 'outputs')) / 'eda'
    (output_dir / 'tables').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_raw_data()
    
    # === PREPROCESSING VISUALIZATIONS ===
    print("\n" + "=" * 50)
    print(" PREPROCESSING VISUALIZATIONS")
    print("=" * 50)
    
    # 1. Preprocessing stages (original -> cropped)
    df_cropped, crop_stats = plot_preprocessing_stages(df, output_dir)
    
    # 2. Grid and graph structure
    G, graph_stats = plot_grid_and_graph(df_cropped, output_dir)
    
    # 3. Spatial with geographic context
    plot_spatial_with_basemap(df, output_dir)
    
    # === TABLES ===
    print("\n" + "=" * 50)
    print(" GENERATING TABLES")
    print("=" * 50)
    
    generate_overview_table(df, output_dir)
    generate_magnitude_statistics(df, output_dir)
    generate_preprocessing_summary(crop_stats, graph_stats, output_dir)
    
    # 4. Crop justification analysis
    generate_crop_justification(df, output_dir)
    
    # 5. Crop sensitivity/cumulative analysis
    generate_crop_sensitivity_analysis(df, output_dir)
    
    # 6. Graph degree/radius analysis
    generate_graph_degree_analysis(df, output_dir)
    
    # 7. Grid size analysis
    generate_grid_size_analysis(df, output_dir)
    
    # 8. Time bin analysis
    generate_time_bin_analysis(df, output_dir)
    
    # === ADDITIONAL VISUALIZATIONS ===
    print("\n" + "=" * 50)
    print(" ADDITIONAL VISUALIZATIONS")
    print("=" * 50)
    
    plot_magnitude_histogram(df, output_dir)
    plot_temporal_distribution(df, output_dir)
    plot_gutenberg_richter(df, output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print(" EDA COMPLETED!")
    print("=" * 70)
    print(f"\n Output directory: {output_dir}")
    print(f" Tables: {len(list((output_dir / 'tables').glob('*')))} files")
    print(f" Figures: {len(list((output_dir / 'figures').glob('*')))} files")
    
    # Generate summary file
    with open(output_dir / 'data_summary.txt', 'w', encoding='utf-8') as f:
        f.write("EARTHQUAKE DATA PREPROCESSING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total events (raw): {crop_stats['total']:,}\n")
        f.write(f"Events after crop: {crop_stats['retained']:,}\n")
        f.write(f"Events removed: {crop_stats['removed']:,} ({crop_stats['removed']/crop_stats['total']*100:.1f}%)\n\n")
        f.write(f"Grid cells: {graph_stats['total_cells']}\n")
        f.write(f"Active nodes: {graph_stats['active_nodes']}\n")
        f.write(f"Graph edges: {graph_stats['edges']}\n")
    
    print(f" Summary: {output_dir / 'data_summary.txt'}")

if __name__ == '__main__':
    main()
