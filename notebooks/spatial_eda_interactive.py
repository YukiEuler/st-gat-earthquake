# %% [markdown]
# # Interactive Spatial EDA - Grid Size & Area Selection
# 
# Tool interaktif untuk menentukan:
# 1. Area mana yang memiliki aktivitas seismik signifikan
# 2. Ukuran grid optimal untuk menangkap pola spasial
# 3. Parameter adjacency matrix yang optimal

# %%
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.spatial.distance import cdist

print("=" * 60)
print(" SPATIAL EDA - Interactive Analysis Tool")
print("=" * 60)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[*] Loading data...")

DATA_PATH = 'Amatrice_CAT5.v20210504.csv'
df = pd.read_csv(DATA_PATH)

# Parse datetime
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
df['timestamp'] = df['datetime'].astype(np.int64) // 10**9

# Calculate bounding box
lat_min_data, lat_max_data = df['lat'].min(), df['lat'].max()
lon_min_data, lon_max_data = df['lon'].min(), df['lon'].max()

print(f"   Total events: {len(df):,}")
print(f"   Latitude range: {lat_min_data:.4f} to {lat_max_data:.4f}")
print(f"   Longitude range: {lon_min_data:.4f} to {lon_max_data:.4f}")
print(f"   Date range: {df['datetime'].min().date()} to {df['datetime'].max().date()}")


# =============================================================================
# GRID ANALYSIS FUNCTION (PYTHON BACKEND)
# =============================================================================
def analyze_grid_comprehensive(df, lat_min, lat_max, lon_min, lon_max, grid_size, min_events, radius_km, sigma_km, time_bin_hours):
    """
    Comprehensive grid analysis with ALL data - computed in Python.
    
    Returns all statistics needed for HTML visualization.
    """
    # Filter to bounds
    df_filtered = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max) & 
                     (df['lon'] >= lon_min) & (df['lon'] <= lon_max)].copy()
    
    n_filtered = len(df_filtered)
    
    # Grid dimensions
    n_rows = int(np.ceil((lat_max - lat_min) / grid_size))
    n_cols = int(np.ceil((lon_max - lon_min) / grid_size))
    total_cells = n_rows * n_cols
    
    # Assign events to grid cells
    df_filtered['row_idx'] = ((df_filtered['lat'] - lat_min) / grid_size).astype(int).clip(0, n_rows - 1)
    df_filtered['col_idx'] = ((df_filtered['lon'] - lon_min) / grid_size).astype(int).clip(0, n_cols - 1)
    df_filtered['node_id'] = df_filtered['row_idx'] * n_cols + df_filtered['col_idx']
    
    # Node statistics
    node_stats = df_filtered.groupby('node_id').agg({
        'lat': 'mean',
        'lon': 'mean',
        'mw': ['count', 'max', 'mean'],
        'dep': 'mean'
    }).reset_index()
    node_stats.columns = ['node_id', 'lat', 'lon', 'count', 'max_mw', 'avg_mw', 'avg_dep']
    
    # Calculate qualified nodes
    qualified_mask = node_stats['count'] >= min_events
    qualified_nodes = node_stats[qualified_mask].copy()
    n_qualified = len(qualified_nodes)
    
    # Events retained
    events_retained = qualified_nodes['count'].sum()
    retention_pct = events_retained / n_filtered * 100 if n_filtered > 0 else 0
    
    # Active nodes (any events)
    n_active = len(node_stats)
    
    # ==========================
    # ADJACENCY MATRIX CALCULATION
    # ==========================
    if n_qualified > 0:
        coords = qualified_nodes[['lat', 'lon']].values
        # Convert to km (approximate)
        coords_km = coords * 111.0  # ~111 km per degree
        
        # Pairwise distances
        distances = cdist(coords_km, coords_km, metric='euclidean')
        
        # Count edges within radius
        edge_mask = (distances > 0) & (distances <= radius_km)
        n_edges = edge_mask.sum() // 2  # Undirected, so divide by 2
        
        # Edge weights (Gaussian)
        weights = np.exp(-distances**2 / (2 * sigma_km**2))
        avg_weight = weights[edge_mask].mean() if edge_mask.sum() > 0 else 0
        
        # Average degree
        avg_degree = edge_mask.sum(axis=1).mean() if n_qualified > 0 else 0
        
        # Density
        max_edges = n_qualified * (n_qualified - 1) / 2
        density = (n_edges / max_edges * 100) if max_edges > 0 else 0
    else:
        n_edges, avg_degree, avg_weight, density = 0, 0, 0, 0
    
    # ==========================
    # TEMPORAL ANALYSIS
    # ==========================
    time_bin_seconds = time_bin_hours * 3600
    min_ts = df_filtered['timestamp'].min()
    max_ts = df_filtered['timestamp'].max()
    
    df_filtered['time_bin'] = ((df_filtered['timestamp'] - min_ts) // time_bin_seconds).astype(int)
    
    total_time_bins = int(np.ceil((max_ts - min_ts) / time_bin_seconds)) + 1
    events_per_bin = df_filtered.groupby('time_bin').size()
    active_bins = len(events_per_bin)
    empty_bins = total_time_bins - active_bins
    active_ratio = active_bins / total_time_bins * 100 if total_time_bins > 0 else 0
    avg_events_per_bin = len(df_filtered) / total_time_bins if total_time_bins > 0 else 0
    max_events_per_bin = events_per_bin.max() if len(events_per_bin) > 0 else 0
    
    # Histogram of events per bin
    bin_counts = events_per_bin.values
    hist_bins = [0, 1, 5, 10, 20, 50, 100, 200, 500]
    temporal_hist = []
    for i, b in enumerate(hist_bins[:-1]):
        count = ((bin_counts >= b) & (bin_counts < hist_bins[i+1])).sum()
        temporal_hist.append(count)
    temporal_hist.append((bin_counts >= hist_bins[-1]).sum())  # 500+
    
    # ==========================
    # NODE DISTRIBUTION HISTOGRAM
    # ==========================
    counts = node_stats['count'].values
    node_hist_bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
    node_hist = []
    for i, b in enumerate(node_hist_bins[:-1]):
        count = ((counts >= b) & (counts < node_hist_bins[i+1])).sum()
        node_hist.append(count)
    node_hist.append((counts >= node_hist_bins[-1]).sum())
    
    # CDF data
    cdf_thresholds = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    cdf_data = [(counts >= t).sum() / len(counts) * 100 if len(counts) > 0 else 0 
                for t in cdf_thresholds]
    
    # ==========================
    # PREPARE NODE DATA FOR VISUALIZATION
    # ==========================
    # Sample nodes for scatter plot (all nodes, but limited info)
    nodes_for_viz = node_stats[['node_id', 'lat', 'lon', 'count', 'max_mw']].copy()
    nodes_for_viz['qualified'] = qualified_mask.values
    
    # Prepare edges for visualization (only between qualified nodes, sample if too many)
    edges_for_viz = []
    if n_qualified > 0 and n_qualified <= 200:  # Only show edges if manageable
        qualified_indices = qualified_nodes.index.tolist()
        qualified_coords = qualified_nodes[['lat', 'lon']].values
        
        for i in range(len(qualified_coords)):
            for j in range(i + 1, len(qualified_coords)):
                dist = np.sqrt(((qualified_coords[i] - qualified_coords[j]) * 111.0) ** 2).sum() ** 0.5
                if dist <= radius_km:
                    weight = np.exp(-dist**2 / (2 * sigma_km**2))
                    edges_for_viz.append({
                        'from': {'lat': float(qualified_coords[i][0]), 'lon': float(qualified_coords[i][1])},
                        'to': {'lat': float(qualified_coords[j][0]), 'lon': float(qualified_coords[j][1])},
                        'weight': float(weight)
                    })
    
    # Convert nodes to JSON-safe format
    nodes_list = []
    for _, row in nodes_for_viz.iterrows():
        nodes_list.append({
            'node_id': int(row['node_id']),
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'count': int(row['count']),
            'max_mw': float(row['max_mw']),
            'qualified': bool(row['qualified'])
        })
    
    return {
        # Basic stats
        'n_filtered': int(n_filtered),
        'n_rows': int(n_rows),
        'n_cols': int(n_cols),
        'total_cells': int(total_cells),
        'n_active': int(n_active),
        'n_qualified': int(n_qualified),
        'events_retained': int(events_retained),
        'retention_pct': float(round(retention_pct, 1)),
        
        # Adjacency stats
        'n_edges': int(n_edges),
        'avg_degree': float(round(avg_degree, 1)),
        'avg_weight': float(round(avg_weight, 3)),
        'density_pct': float(round(density, 2)),
        
        # Temporal stats
        'total_time_bins': int(total_time_bins),
        'active_bins': int(active_bins),
        'empty_bins': int(empty_bins),
        'active_ratio': float(round(active_ratio, 1)),
        'avg_events_per_bin': float(round(avg_events_per_bin, 2)),
        'max_events_per_bin': int(max_events_per_bin),
        
        # Histogram data
        'node_hist': [int(x) for x in node_hist],
        'node_hist_bins': [int(x) for x in node_hist_bins],
        'temporal_hist': [int(x) for x in temporal_hist],
        'temporal_hist_bins': [int(x) for x in hist_bins],
        'cdf_thresholds': [int(x) for x in cdf_thresholds],
        'cdf_data': [float(round(c, 1)) for c in cdf_data],
        
        # Visualization data
        'nodes': nodes_list,
        'edges': edges_for_viz[:1000],  # Limit edges for performance
    }


# =============================================================================
# GENERATE MULTIPLE CONFIGURATIONS
# =============================================================================
print("\n[*] Pre-computing grid configurations...")

# Parameter ranges for interactive exploration
grid_sizes = [0.01, 0.02, 0.03, 0.05, 0.1]
min_events_list = [1, 5, 10, 15, 20, 50, 100, 200, 500, 1000, 2000, 5000]
radius_list = [5, 10, 15, 20, 30, 50]
sigma_list = [5, 10, 15, 20]
time_bins = [1, 3, 6, 12, 24]  # hours

# Default bounds (can be adjusted in HTML)
default_lat_min = lat_min_data
default_lat_max = lat_max_data
default_lon_min = lon_min_data
default_lon_max = lon_max_data

# Compute default configuration
default_config = analyze_grid_comprehensive(
    df, 
    lat_min=default_lat_min,
    lat_max=default_lat_max,
    lon_min=default_lon_min,
    lon_max=default_lon_max,
    grid_size=0.02,
    min_events=15,
    radius_km=5.0,
    sigma_km=10.0,
    time_bin_hours=6
)

print(f"   Default config computed")
print(f"   Total events: {default_config['n_filtered']:,}")
print(f"   Qualified nodes: {default_config['n_qualified']} / {default_config['total_cells']}")


# =============================================================================
# GENERATE HTML WITH PRE-COMPUTED DATA
# =============================================================================
print("\n[*] Generating interactive HTML...")

html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Spatial EDA - Earthquake Grid Analysis (Full Data)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            color: #4fc3f7; 
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .subtitle {
            text-align: center;
            color: #aaa;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        .panel {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .panel h2 {
            color: #4fc3f7;
            margin-bottom: 15px;
            font-size: 1.2em;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 10px;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }
        .control-group {
            background: rgba(255, 255, 255, 0.05);
            padding: 12px;
            border-radius: 8px;
        }
        .control-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #4fc3f7;
            font-size: 0.85em;
        }
        .control-group select, .control-group input[type="number"] {
            width: 100%;
            padding: 8px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            background: rgba(0, 0, 0, 0.3);
            color: #fff;
            font-size: 0.9em;
        }
        .value-display {
            color: #4fc3f7;
            font-weight: bold;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-top: 15px;
        }
        .stat-card {
            background: rgba(79, 195, 247, 0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(79, 195, 247, 0.2);
        }
        .stat-card .value {
            font-size: 1.5em;
            font-weight: bold;
            color: #4fc3f7;
        }
        .stat-card .label {
            font-size: 0.75em;
            color: #aaa;
            margin-top: 5px;
        }
        .chart-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .chart-container {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 15px;
            height: 300px;
        }
        .map-container {
            height: 400px;
            background: #0a0a1a;
            border-radius: 8px;
            position: relative;
        }
        .recommendation {
            background: rgba(46, 204, 113, 0.1);
            border: 1px solid rgba(46, 204, 113, 0.3);
            border-radius: 8px;
            padding: 20px;
        }
        .recommendation pre {
            background: rgba(0, 0, 0, 0.3);
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.85em;
            color: #4fc3f7;
        }
        .btn {
            background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%);
            color: #000;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: scale(1.02);
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #aaa;
        }
        @media (max-width: 768px) {
            .chart-row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 Spatial EDA - Earthquake Grid Analysis</h1>
        <p class="subtitle">Full data analysis (''' + f"{len(df):,}" + ''' events) • ''' + f"{df['datetime'].min().date()} to {df['datetime'].max().date()}" + '''</p>
        
        <div class="panel">
            <h2>⚙️ Configuration Parameters</h2>
            <div class="controls">
                <div class="control-group">
                    <label>📍 Lat Min</label>
                    <input type="number" id="latMin" value="''' + f'{default_lat_min:.4f}' + '''" step="0.01">
                </div>
                <div class="control-group">
                    <label>📍 Lat Max</label>
                    <input type="number" id="latMax" value="''' + f'{default_lat_max:.4f}' + '''" step="0.01">
                </div>
                <div class="control-group">
                    <label>📍 Lon Min</label>
                    <input type="number" id="lonMin" value="''' + f'{default_lon_min:.4f}' + '''" step="0.01">
                </div>
                <div class="control-group">
                    <label>📍 Lon Max</label>
                    <input type="number" id="lonMax" value="''' + f'{default_lon_max:.4f}' + '''" step="0.01">
                </div>
                <div class="control-group">
                    <label>📐 Grid Size (degrees)</label>
                    <select id="gridSize">
                        <option value="0.01">0.01° (~1.1 km)</option>
                        <option value="0.02" selected>0.02° (~2.2 km)</option>
                        <option value="0.03">0.03° (~3.3 km)</option>
                        <option value="0.05">0.05° (~5.5 km)</option>
                        <option value="0.1">0.10° (~11 km)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>🎯 Min Events/Node</label>
                    <select id="minEvents">
                        <option value="1">1</option>
                        <option value="5">5</option>
                        <option value="10">10</option>
                        <option value="15" selected>15</option>
                        <option value="20">20</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                        <option value="200">200</option>
                        <option value="500">500</option>
                        <option value="1000">1,000</option>
                        <option value="2000">2,000</option>
                        <option value="5000">5,000</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>🔗 Radius (km)</label>
                    <select id="radius">
                        <option value="5" selected>5 km</option>
                        <option value="10">10 km</option>
                        <option value="15">15 km</option>
                        <option value="20">20 km</option>
                        <option value="30">30 km</option>
                        <option value="50">50 km</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>📊 Sigma (km)</label>
                    <select id="sigma">
                        <option value="5">5 km</option>
                        <option value="10" selected>10 km</option>
                        <option value="15">15 km</option>
                        <option value="20">20 km</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>⏱️ Time Bin</label>
                    <select id="timeBin">
                        <option value="1">1 hour</option>
                        <option value="3">3 hours</option>
                        <option value="6" selected>6 hours</option>
                        <option value="12">12 hours</option>
                        <option value="24">1 day</option>
                    </select>
                </div>
            </div>
            <div style="margin-top: 15px; text-align: center;">
                <button class="btn" onclick="analyzeGrid()">🔄 Analyze Grid</button>
            </div>
        </div>
        
        <div class="panel">
            <h2>📊 Statistics Summary</h2>
            <div class="stats-grid" id="statsGrid">
                <!-- Will be populated by JS -->
            </div>
        </div>
        
        <div class="panel">
            <h2>🗺️ Spatial Distribution</h2>
            <div class="map-container">
                <canvas id="mapCanvas"></canvas>
            </div>
        </div>
        
        <div class="panel">
            <h2>📈 Node Activity Distribution</h2>
            <div class="chart-row">
                <div class="chart-container">
                    <canvas id="nodeHistChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="cdfChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h2>⏱️ Temporal Distribution</h2>
            <div class="chart-row">
                <div class="chart-container">
                    <canvas id="temporalHistChart"></canvas>
                </div>
                <div id="temporalStats" style="padding: 20px;">
                    <!-- Temporal stats -->
                </div>
            </div>
        </div>
        
        <div class="panel">
            <h2>💡 Recommended Configuration</h2>
            <div class="recommendation" id="recommendation">
                <!-- Config recommendation -->
            </div>
        </div>
    </div>
    
    <script>
        // Pre-computed data from Python (default configuration)
        let currentData = ''' + json.dumps(default_config) + ''';
        
        // Charts
        let nodeHistChart, cdfChart, temporalHistChart;
        
        // Initialize charts
        function initCharts() {
            const chartDefaults = {
                responsive: true,
                maintainAspectRatio: false,
            };
            
            nodeHistChart = new Chart(document.getElementById('nodeHistChart'), {
                type: 'bar',
                data: { labels: [], datasets: [] },
                options: {
                    ...chartDefaults,
                    plugins: { 
                        title: { display: true, text: 'Events per Node Distribution', color: '#aaa' },
                        legend: { display: false }
                    },
                    scales: { 
                        y: { title: { display: true, text: 'Number of Nodes', color: '#aaa' }, ticks: { color: '#aaa' } },
                        x: { ticks: { color: '#aaa' } }
                    }
                }
            });
            
            cdfChart = new Chart(document.getElementById('cdfChart'), {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    ...chartDefaults,
                    plugins: { 
                        title: { display: true, text: 'Cumulative % of Nodes with ≥ X Events', color: '#aaa' },
                        legend: { display: false }
                    },
                    scales: { 
                        y: { title: { display: true, text: '%', color: '#aaa' }, min: 0, max: 100, ticks: { color: '#aaa' } },
                        x: { title: { display: true, text: 'Min Events', color: '#aaa' }, ticks: { color: '#aaa' } }
                    }
                }
            });
            
            temporalHistChart = new Chart(document.getElementById('temporalHistChart'), {
                type: 'bar',
                data: { labels: [], datasets: [] },
                options: {
                    ...chartDefaults,
                    plugins: { 
                        title: { display: true, text: 'Events per Time Bin', color: '#aaa' },
                        legend: { display: false }
                    },
                    scales: { 
                        y: { title: { display: true, text: 'Number of Bins', color: '#aaa' }, ticks: { color: '#aaa' } },
                        x: { ticks: { color: '#aaa' } }
                    }
                }
            });
        }
        
        // Update display with current data
        function updateDisplay() {
            const d = currentData;
            
            // Stats grid
            document.getElementById('statsGrid').innerHTML = `
                <div class="stat-card"><div class="value">${d.n_rows}×${d.n_cols}</div><div class="label">Grid Dimensions</div></div>
                <div class="stat-card"><div class="value">${d.total_cells}</div><div class="label">Total Cells</div></div>
                <div class="stat-card"><div class="value">${d.n_active}</div><div class="label">Active Nodes</div></div>
                <div class="stat-card"><div class="value">${d.n_qualified}</div><div class="label">Qualified Nodes</div></div>
                <div class="stat-card"><div class="value">${d.retention_pct}%</div><div class="label">Event Retention</div></div>
                <div class="stat-card"><div class="value">${d.n_edges}</div><div class="label">Graph Edges</div></div>
                <div class="stat-card"><div class="value">${d.avg_degree}</div><div class="label">Avg Degree</div></div>
                <div class="stat-card"><div class="value">${d.density_pct}%</div><div class="label">Graph Density</div></div>
            `;
            
            // Node histogram
            const nodeLabels = d.node_hist_bins.map((b, i) => 
                i < d.node_hist_bins.length - 1 ? `${b}-${d.node_hist_bins[i+1]}` : `${b}+`
            );
            nodeHistChart.data = {
                labels: nodeLabels,
                datasets: [{
                    data: d.node_hist,
                    backgroundColor: 'rgba(79, 195, 247, 0.7)',
                    borderColor: 'rgba(79, 195, 247, 1)',
                    borderWidth: 1
                }]
            };
            nodeHistChart.update();
            
            // CDF chart
            cdfChart.data = {
                labels: d.cdf_thresholds,
                datasets: [{
                    data: d.cdf_data,
                    borderColor: 'rgba(46, 204, 113, 1)',
                    backgroundColor: 'rgba(46, 204, 113, 0.2)',
                    fill: true,
                    tension: 0.3
                }]
            };
            cdfChart.update();
            
            // Temporal histogram
            const tempLabels = d.temporal_hist_bins.map((b, i) => 
                i < d.temporal_hist_bins.length - 1 ? `${b}-${d.temporal_hist_bins[i+1]}` : `${b}+`
            );
            temporalHistChart.data = {
                labels: tempLabels,
                datasets: [{
                    data: d.temporal_hist,
                    backgroundColor: 'rgba(155, 89, 182, 0.7)',
                    borderColor: 'rgba(155, 89, 182, 1)',
                    borderWidth: 1
                }]
            };
            temporalHistChart.update();
            
            // Temporal stats
            document.getElementById('temporalStats').innerHTML = `
                <h3 style="color: #4fc3f7; margin-bottom: 15px;">Temporal Statistics</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 8px; border-bottom: 1px solid #333;">Total Time Bins</td><td style="padding: 8px; border-bottom: 1px solid #333; color: #4fc3f7;">${d.total_time_bins.toLocaleString()}</td></tr>
                    <tr><td style="padding: 8px; border-bottom: 1px solid #333;">Active Bins</td><td style="padding: 8px; border-bottom: 1px solid #333; color: #4fc3f7;">${d.active_bins.toLocaleString()} (${d.active_ratio}%)</td></tr>
                    <tr><td style="padding: 8px; border-bottom: 1px solid #333;">Empty Bins</td><td style="padding: 8px; border-bottom: 1px solid #333; color: #4fc3f7;">${d.empty_bins.toLocaleString()}</td></tr>
                    <tr><td style="padding: 8px; border-bottom: 1px solid #333;">Avg Events/Bin</td><td style="padding: 8px; border-bottom: 1px solid #333; color: #4fc3f7;">${d.avg_events_per_bin}</td></tr>
                    <tr><td style="padding: 8px;">Max Events/Bin</td><td style="padding: 8px; color: #4fc3f7;">${d.max_events_per_bin}</td></tr>
                </table>
            `;
            
            // Map
            drawMap();
            
            // Recommendation
            const gridSize = parseFloat(document.getElementById('gridSize').value);
            const minEvents = parseInt(document.getElementById('minEvents').value);
            const radius = parseInt(document.getElementById('radius').value);
            const sigma = parseInt(document.getElementById('sigma').value);
            const timeBin = parseInt(document.getElementById('timeBin').value);
            const latMin = parseFloat(document.getElementById('latMin').value);
            const latMax = parseFloat(document.getElementById('latMax').value);
            const lonMin = parseFloat(document.getElementById('lonMin').value);
            const lonMax = parseFloat(document.getElementById('lonMax').value);
            
            document.getElementById('recommendation').innerHTML = `
                <h3 style="color: #2ecc71; margin-bottom: 15px;">📋 Configuration Summary</h3>
                <p style="margin-bottom: 15px;">Based on your selection with <strong>all ${d.n_filtered.toLocaleString()} events</strong>:</p>
                <pre>
CONFIG = {
    # Spatial bounds
    'lat_min': ${latMin.toFixed(4)},
    'lat_max': ${latMax.toFixed(4)},
    'lon_min': ${lonMin.toFixed(4)},
    'lon_max': ${lonMax.toFixed(4)},
    
    # Grid configuration
    'grid_size': ${gridSize},  # ${gridSize}° ≈ ${(gridSize * 111).toFixed(1)} km
    'min_events_per_node': ${minEvents},
    
    # Temporal
    'time_bin': '${timeBin}h',
    
    # Adjacency matrix
    'radius_km': ${radius}.0,
    'sigma_km': ${sigma}.0,
    
    # Expected results:
    # - Qualified nodes: ${d.n_qualified}
    # - Edges: ${d.n_edges}
    # - Density: ${d.density_pct}%
    # - Event retention: ${d.retention_pct}%
}
                </pre>
            `;
        }
        
        // Draw spatial map
        function drawMap() {
            const canvas = document.getElementById('mapCanvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
            
            const w = canvas.width;
            const h = canvas.height;
            const padding = 40;
            
            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, w, h);
            
            const latMin = parseFloat(document.getElementById('latMin').value);
            const latMax = parseFloat(document.getElementById('latMax').value);
            const lonMin = parseFloat(document.getElementById('lonMin').value);
            const lonMax = parseFloat(document.getElementById('lonMax').value);
            
            const scaleX = lon => padding + (lon - lonMin) / (lonMax - lonMin) * (w - 2 * padding);
            const scaleY = lat => h - padding - (lat - latMin) / (latMax - latMin) * (h - 2 * padding);
            
            // Draw edges
            ctx.strokeStyle = 'rgba(79, 195, 247, 0.15)';
            ctx.lineWidth = 1;
            currentData.edges.forEach(e => {
                ctx.globalAlpha = 0.1 + e.weight * 0.4;
                ctx.beginPath();
                ctx.moveTo(scaleX(e.from.lon), scaleY(e.from.lat));
                ctx.lineTo(scaleX(e.to.lon), scaleY(e.to.lat));
                ctx.stroke();
            });
            ctx.globalAlpha = 1;
            
            // Draw nodes
            const minEvents = parseInt(document.getElementById('minEvents').value);
            currentData.nodes.forEach(n => {
                const isQualified = n.count >= minEvents;
                const size = 3 + Math.log10(n.count + 1) * 2;
                
                ctx.fillStyle = isQualified ? 'rgba(46, 204, 113, 0.8)' : 'rgba(231, 76, 60, 0.4)';
                ctx.beginPath();
                ctx.arc(scaleX(n.lon), scaleY(n.lat), size, 0, Math.PI * 2);
                ctx.fill();
            });
            
            // Axis labels
            ctx.fillStyle = '#aaa';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`Longitude (${lonMin.toFixed(2)} - ${lonMax.toFixed(2)})`, w/2, h - 10);
            
            ctx.save();
            ctx.translate(15, h/2);
            ctx.rotate(-Math.PI/2);
            ctx.fillText(`Latitude (${latMin.toFixed(2)} - ${latMax.toFixed(2)})`, 0, 0);
            ctx.restore();
            
            // Legend
            ctx.fillStyle = 'rgba(46, 204, 113, 0.8)';
            ctx.fillRect(w - 130, 15, 12, 12);
            ctx.fillStyle = '#aaa';
            ctx.textAlign = 'left';
            ctx.fillText('Qualified', w - 112, 25);
            
            ctx.fillStyle = 'rgba(231, 76, 60, 0.5)';
            ctx.fillRect(w - 130, 32, 12, 12);
            ctx.fillStyle = '#aaa';
            ctx.fillText('Sparse', w - 112, 42);
        }
        
        // Analyze grid (calls Python backend via form submission or shows message)
        function analyzeGrid() {
            alert('⚠️ Parameter change detected!\\n\\nTo recompute with new parameters, please re-run the Python script:\\n\\npython notebooks/spatial_eda_interactive.py\\n\\nThen refresh this page.');
        }
        
        // Initialize
        initCharts();
        updateDisplay();
        window.addEventListener('resize', drawMap);
    </script>
</body>
</html>
'''

# Save HTML
output_path = Path('outputs/spatial_eda_interactive.html')
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n[OK] Interactive EDA saved to: {output_path}")
print("Open this file in your browser to explore the analysis!")
print(f"\n[*] Summary:")
print(f"   Total events analyzed: {len(df):,}")
print(f"   Qualified nodes: {default_config['n_qualified']}")
print(f"   Graph edges: {default_config['n_edges']}")
