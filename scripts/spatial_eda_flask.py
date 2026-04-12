# ==============================================================================
# SPATIAL_EDA_FLASK.PY - Interactive Spatial EDA with Flask Backend
# ==============================================================================
"""
Flask-based interactive EDA for earthquake data.
Backend processes settings from frontend without reloading.

Usage:
    cd earthquake_prediction
    python scripts/spatial_eda_flask.py
    
Then open: http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from pathlib import Path
import sys

# Add parent for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

app = Flask(__name__)

# ==============================================================================
# LOAD DATA ONCE AT STARTUP
# ==============================================================================
print("[*] Loading earthquake data...")
DATA_PATH = Path(__file__).parent / 'Amatrice_CAT5.v20210504.csv'
df = pd.read_csv(DATA_PATH)

# Parse datetime
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
df['timestamp'] = df['datetime'].astype(np.int64) // 10**9

# Data bounds
DATA_BOUNDS = {
    'lat_min': float(df['lat'].min()),
    'lat_max': float(df['lat'].max()),
    'lon_min': float(df['lon'].min()),
    'lon_max': float(df['lon'].max()),
    'date_min': str(df['datetime'].min().date()),
    'date_max': str(df['datetime'].max().date()),
    'total_events': len(df)
}

print(f"   Loaded {len(df):,} events")
print(f"   Bounds: lat [{DATA_BOUNDS['lat_min']:.4f}, {DATA_BOUNDS['lat_max']:.4f}]")
print(f"   Bounds: lon [{DATA_BOUNDS['lon_min']:.4f}, {DATA_BOUNDS['lon_max']:.4f}]")

# ==============================================================================
# ANALYSIS FUNCTION (Called by API)
# ==============================================================================
def analyze_grid(lat_min, lat_max, lon_min, lon_max, grid_size, min_events, radius_km, sigma_km, time_bin_hours):
    """
    Comprehensive grid analysis - called by Flask API.
    Returns JSON-serializable statistics for frontend.
    """
    # Filter to bounds
    df_filtered = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max) & 
                     (df['lon'] >= lon_min) & (df['lon'] <= lon_max)].copy()
    
    n_filtered = len(df_filtered)
    if n_filtered == 0:
        return {'error': 'No events in selected area'}
    
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
    
    # Qualified nodes
    qualified_mask = node_stats['count'] >= min_events
    qualified_nodes = node_stats[qualified_mask].copy()
    n_qualified = len(qualified_nodes)
    
    # Events retained
    events_retained = qualified_nodes['count'].sum() if n_qualified > 0 else 0
    retention_pct = events_retained / n_filtered * 100 if n_filtered > 0 else 0
    
    n_active = len(node_stats)
    
    # ==========================
    # ADJACENCY MATRIX
    # ==========================
    n_edges, avg_degree, avg_weight, density = 0, 0, 0, 0
    edges_for_viz = []
    
    if n_qualified > 0:
        coords = qualified_nodes[['lat', 'lon']].values
        coords_km = coords * 111.0
        
        distances = cdist(coords_km, coords_km, metric='euclidean')
        
        edge_mask = (distances > 0) & (distances <= radius_km)
        n_edges = int(edge_mask.sum() // 2)
        
        weights = np.exp(-distances**2 / (2 * sigma_km**2))
        avg_weight = float(weights[edge_mask].mean()) if edge_mask.sum() > 0 else 0
        
        avg_degree = float(edge_mask.sum(axis=1).mean())
        
        max_edges = n_qualified * (n_qualified - 1) / 2
        density = (n_edges / max_edges * 100) if max_edges > 0 else 0
        
        # Edges for visualization (limit to 500)
        if n_qualified <= 150:
            qualified_coords = qualified_nodes[['lat', 'lon']].values
            edge_count = 0
            for i in range(len(qualified_coords)):
                if edge_count >= 500:
                    break
                for j in range(i + 1, len(qualified_coords)):
                    if edge_count >= 500:
                        break
                    dist = np.sqrt(((qualified_coords[i] - qualified_coords[j]) * 111.0) ** 2).sum() ** 0.5
                    if dist <= radius_km:
                        weight = np.exp(-dist**2 / (2 * sigma_km**2))
                        edges_for_viz.append({
                            'from': {'lat': float(qualified_coords[i][0]), 'lon': float(qualified_coords[i][1])},
                            'to': {'lat': float(qualified_coords[j][0]), 'lon': float(qualified_coords[j][1])},
                            'weight': float(weight)
                        })
                        edge_count += 1
    
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
    max_events_per_bin = int(events_per_bin.max()) if len(events_per_bin) > 0 else 0
    
    # Histograms
    bin_counts = events_per_bin.values
    hist_bins = [0, 1, 5, 10, 20, 50, 100, 200, 500]
    temporal_hist = []
    for i, b in enumerate(hist_bins[:-1]):
        count = ((bin_counts >= b) & (bin_counts < hist_bins[i+1])).sum()
        temporal_hist.append(int(count))
    temporal_hist.append(int((bin_counts >= hist_bins[-1]).sum()))
    
    counts = node_stats['count'].values
    node_hist_bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
    node_hist = []
    for i, b in enumerate(node_hist_bins[:-1]):
        count = ((counts >= b) & (counts < node_hist_bins[i+1])).sum()
        node_hist.append(int(count))
    node_hist.append(int((counts >= node_hist_bins[-1]).sum()))
    
    # CDF
    cdf_thresholds = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    cdf_data = [float(round((counts >= t).sum() / len(counts) * 100, 1)) if len(counts) > 0 else 0 
                for t in cdf_thresholds]
    
    # ==========================
    # NODES FOR VISUALIZATION
    # ==========================
    nodes_list = []
    for _, row in node_stats.iterrows():
        # Handle NaN values for JSON compatibility
        max_mw = row['max_mw']
        if pd.isna(max_mw):
            max_mw = 0.0
        
        nodes_list.append({
            'node_id': int(row['node_id']),
            'lat': float(row['lat']),
            'lon': float(row['lon']),
            'count': int(row['count']),
            'max_mw': float(max_mw),
            'qualified': bool(row['count'] >= min_events)
        })
    
    return {
        'n_filtered': int(n_filtered),
        'n_rows': int(n_rows),
        'n_cols': int(n_cols),
        'total_cells': int(total_cells),
        'n_active': int(n_active),
        'n_qualified': int(n_qualified),
        'events_retained': int(events_retained),
        'retention_pct': float(round(retention_pct, 1)),
        
        'n_edges': int(n_edges),
        'avg_degree': float(round(avg_degree, 1)),
        'avg_weight': float(round(avg_weight, 3)),
        'density_pct': float(round(density, 2)),
        
        'total_time_bins': int(total_time_bins),
        'active_bins': int(active_bins),
        'empty_bins': int(empty_bins),
        'active_ratio': float(round(active_ratio, 1)),
        'avg_events_per_bin': float(round(avg_events_per_bin, 2)),
        'max_events_per_bin': int(max_events_per_bin),
        
        'node_hist': node_hist,
        'node_hist_bins': [int(x) for x in node_hist_bins],
        'temporal_hist': temporal_hist,
        'temporal_hist_bins': [int(x) for x in hist_bins],
        'cdf_thresholds': [int(x) for x in cdf_thresholds],
        'cdf_data': cdf_data,
        
        'nodes': nodes_list[:1000],  # Limit for performance
        'edges': edges_for_viz,
    }


# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template_string(HTML_TEMPLATE, bounds=DATA_BOUNDS)


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint to analyze grid with given parameters."""
    try:
        data = request.get_json()
        
        result = analyze_grid(
            lat_min=float(data.get('lat_min', DATA_BOUNDS['lat_min'])),
            lat_max=float(data.get('lat_max', DATA_BOUNDS['lat_max'])),
            lon_min=float(data.get('lon_min', DATA_BOUNDS['lon_min'])),
            lon_max=float(data.get('lon_max', DATA_BOUNDS['lon_max'])),
            grid_size=float(data.get('grid_size', 0.02)),
            min_events=int(data.get('min_events', 15)),
            radius_km=float(data.get('radius_km', 15.0)),
            sigma_km=float(data.get('sigma_km', 10.0)),
            time_bin_hours=int(data.get('time_bin_hours', 6))
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bounds')
def api_bounds():
    """Return data bounds."""
    return jsonify(DATA_BOUNDS)


# ==============================================================================
# HTML TEMPLATE
# ==============================================================================
HTML_TEMPLATE = '''<!DOCTYPE html>
<html>
<head>
    <title>Spatial EDA - Grid Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            color: #2c3e50; 
            margin-bottom: 5px;
            font-size: 1.6em;
            font-weight: 600;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
            font-size: 0.85em;
        }
        .panel {
            background: #fff;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .panel h2 {
            color: #2c3e50;
            margin-bottom: 12px;
            font-size: 1em;
            font-weight: 600;
            border-bottom: 1px solid #eee;
            padding-bottom: 8px;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 10px;
        }
        .control-group label {
            display: block;
            margin-bottom: 4px;
            font-weight: 500;
            color: #555;
            font-size: 0.75em;
        }
        .control-group select, .control-group input[type="number"] {
            width: 100%;
            padding: 6px 8px;
            border-radius: 4px;
            border: 1px solid #ddd;
            background: #fff;
            color: #333;
            font-size: 0.85em;
        }
        .control-group select:focus, .control-group input:focus {
            outline: none;
            border-color: #3498db;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 8px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #eee;
        }
        .stat-card .value {
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
        }
        .stat-card .label {
            font-size: 0.65em;
            color: #888;
            margin-top: 2px;
        }
        .chart-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .chart-container {
            background: #fafafa;
            border-radius: 6px;
            padding: 10px;
            height: 220px;
            border: 1px solid #eee;
        }
        .map-container {
            height: 450px;
            background: #fff;
            border-radius: 6px;
            border: 1px solid #ddd;
        }
        .btn {
            background: #3498db;
            color: #fff;
            border: none;
            padding: 8px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.85em;
        }
        .btn:hover { background: #2980b9; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .recommendation pre {
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.75em;
            color: #333;
            border: 1px solid #eee;
            overflow-x: auto;
        }
        .loading {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(255,255,255,0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            font-size: 1.2em;
            color: #333;
        }
        .loading.hidden { display: none; }
        @media (max-width: 768px) {
            .chart-row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="loading hidden" id="loading">⏳ Analyzing...</div>
    
    <div class="container">
        <h1>🌍 Interactive Spatial EDA</h1>
        <p class="subtitle">Flask Backend • {{ bounds.total_events | default(0) }} events • {{ bounds.date_min }} to {{ bounds.date_max }}</p>
        
        <div class="panel">
            <h2>⚙️ Parameters (Real-time Analysis)</h2>
            <div class="controls">
                <div class="control-group">
                    <label>📍 Lat Min</label>
                    <input type="number" id="latMin" value="{{ bounds.lat_min }}" step="0.01">
                </div>
                <div class="control-group">
                    <label>📍 Lat Max</label>
                    <input type="number" id="latMax" value="{{ bounds.lat_max }}" step="0.01">
                </div>
                <div class="control-group">
                    <label>📍 Lon Min</label>
                    <input type="number" id="lonMin" value="{{ bounds.lon_min }}" step="0.01">
                </div>
                <div class="control-group">
                    <label>📍 Lon Max</label>
                    <input type="number" id="lonMax" value="{{ bounds.lon_max }}" step="0.01">
                </div>
                <div class="control-group">
                    <label>📐 Grid Size</label>
                    <select id="gridSize">
                        <option value="0.01">0.01° (~1.1 km)</option>
                        <option value="0.02" selected>0.02° (~2.2 km)</option>
                        <option value="0.03">0.03° (~3.3 km)</option>
                        <option value="0.05">0.05° (~5.5 km)</option>
                        <option value="0.1">0.10° (~11 km)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>🎯 Min Events</label>
                    <select id="minEvents">
                        <option value="1">1</option>
                        <option value="5">5</option>
                        <option value="10">10</option>
                        <option value="15" selected>15</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                        <option value="500">500</option>
                        <option value="1000">1000</option>
                        <option value="1500">1500</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>🔗 Radius (km)</label>
                    <select id="radius">
                        <option value="5">5 km</option>
                        <option value="10">10 km</option>
                        <option value="15" selected>15 km</option>
                        <option value="20">20 km</option>
                        <option value="30">30 km</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>📊 Sigma (km)</label>
                    <select id="sigma">
                        <option value="5">5 km</option>
                        <option value="10" selected>10 km</option>
                        <option value="15">15 km</option>
                        <option value="150">150 km</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>⏱️ Time Bin</label>
                    <select id="timeBin">
                        <option value="1">1 hour</option>
                        <option value="4" selected>4 hours</option>
                        <option value="6">6 hours</option>
                        <option value="12">12 hours</option>
                        <option value="24">1 day</option>
                    </select>
                </div>
            </div>
            <div style="margin-top: 15px; text-align: center;">
                <button class="btn" id="analyzeBtn" onclick="analyzeGrid()">🔄 Analyze Grid</button>
            </div>
        </div>
        
        <div class="panel">
            <h2>📊 Statistics</h2>
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card"><div class="value">-</div><div class="label">Grid Size</div></div>
                <div class="stat-card"><div class="value">-</div><div class="label">Total Cells</div></div>
                <div class="stat-card"><div class="value">-</div><div class="label">Active Nodes</div></div>
                <div class="stat-card"><div class="value">-</div><div class="label">Qualified Nodes</div></div>
                <div class="stat-card"><div class="value">-</div><div class="label">Retention %</div></div>
                <div class="stat-card"><div class="value">-</div><div class="label">Graph Edges</div></div>
                <div class="stat-card"><div class="value">-</div><div class="label">Avg Degree</div></div>
                <div class="stat-card"><div class="value">-</div><div class="label">Density %</div></div>
            </div>
        </div>
        
        <div class="panel">
            <h2>🗺️ Spatial Distribution</h2>
            <div class="map-container">
                <canvas id="mapCanvas"></canvas>
            </div>
        </div>
        
        <div class="panel">
            <h2>📈 Distributions</h2>
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
            <h2>💡 Configuration Output</h2>
            <div class="recommendation" id="recommendation">
                <pre>Click "Analyze Grid" to generate config...</pre>
            </div>
        </div>
    </div>
    
    <script>
        let currentData = null;
        let nodeHistChart, cdfChart;
        
        // Initialize charts
        function initCharts() {
            const defaults = { responsive: true, maintainAspectRatio: false };
            
            nodeHistChart = new Chart(document.getElementById('nodeHistChart'), {
                type: 'bar',
                data: { labels: [], datasets: [] },
                options: {
                    ...defaults,
                    plugins: { 
                        title: { display: true, text: 'Events per Node', color: '#aaa' },
                        legend: { display: false }
                    },
                    scales: { 
                        y: { ticks: { color: '#aaa' } },
                        x: { ticks: { color: '#aaa' } }
                    }
                }
            });
            
            cdfChart = new Chart(document.getElementById('cdfChart'), {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    ...defaults,
                    plugins: { 
                        title: { display: true, text: 'CDF: % Nodes with ≥ X Events', color: '#aaa' },
                        legend: { display: false }
                    },
                    scales: { 
                        y: { min: 0, max: 100, ticks: { color: '#aaa' } },
                        x: { ticks: { color: '#aaa' } }
                    }
                }
            });
        }
        
        // Call Flask API
        async function analyzeGrid() {
            const btn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            
            btn.disabled = true;
            loading.classList.remove('hidden');
            
            const params = {
                lat_min: parseFloat(document.getElementById('latMin').value),
                lat_max: parseFloat(document.getElementById('latMax').value),
                lon_min: parseFloat(document.getElementById('lonMin').value),
                lon_max: parseFloat(document.getElementById('lonMax').value),
                grid_size: parseFloat(document.getElementById('gridSize').value),
                min_events: parseInt(document.getElementById('minEvents').value),
                radius_km: parseFloat(document.getElementById('radius').value),
                sigma_km: parseFloat(document.getElementById('sigma').value),
                time_bin_hours: parseInt(document.getElementById('timeBin').value)
            };
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
                
                currentData = await response.json();
                
                if (currentData.error) {
                    alert('Error: ' + currentData.error);
                } else {
                    updateDisplay(params);
                }
            } catch (e) {
                alert('Request failed: ' + e.message);
            } finally {
                btn.disabled = false;
                loading.classList.add('hidden');
            }
        }
        
        // Update UI with data
        function updateDisplay(params) {
            const d = currentData;
            
            // Stats grid
            document.getElementById('statsGrid').innerHTML = `
                <div class="stat-card"><div class="value">${d.n_rows}×${d.n_cols}</div><div class="label">Grid Size</div></div>
                <div class="stat-card"><div class="value">${d.total_cells.toLocaleString()}</div><div class="label">Total Cells</div></div>
                <div class="stat-card"><div class="value">${d.n_active}</div><div class="label">Active Nodes</div></div>
                <div class="stat-card"><div class="value">${d.n_qualified}</div><div class="label">Qualified Nodes</div></div>
                <div class="stat-card"><div class="value">${d.retention_pct}%</div><div class="label">Retention %</div></div>
                <div class="stat-card"><div class="value">${d.n_edges.toLocaleString()}</div><div class="label">Graph Edges</div></div>
                <div class="stat-card"><div class="value">${d.avg_degree}</div><div class="label">Avg Degree</div></div>
                <div class="stat-card"><div class="value">${d.density_pct}%</div><div class="label">Density %</div></div>
            `;
            
            // Charts
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
            
            // Map
            drawMap();
            
            // Config output
            document.getElementById('recommendation').innerHTML = `
                <pre>CONFIG = {
    # Spatial bounds
    'lat_min': ${params.lat_min.toFixed(4)},
    'lat_max': ${params.lat_max.toFixed(4)},
    'lon_min': ${params.lon_min.toFixed(4)},
    'lon_max': ${params.lon_max.toFixed(4)},
    
    # Grid
    'grid_size': ${params.grid_size},
    'min_events_per_node': ${params.min_events},
    
    # Temporal
    'time_bin': '${params.time_bin_hours}h',
    
    # Adjacency
    'radius_km': ${params.radius_km},
    'sigma_km': ${params.sigma_km},
    
    # Results:
    # - Qualified nodes: ${d.n_qualified}
    # - Edges: ${d.n_edges}
    # - Retention: ${d.retention_pct}%
}</pre>
            `;
        }
        
        // Draw map canvas - fill available space
        function drawMap() {
            const canvas = document.getElementById('mapCanvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
            
            const w = canvas.width, h = canvas.height;
            const padding = 35;
            
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, w, h);
            
            if (!currentData || !currentData.nodes || currentData.nodes.length === 0) return;
            
            // Get actual extent from NODE DATA (not input bounds)
            const lats = currentData.nodes.map(n => n.lat);
            const lons = currentData.nodes.map(n => n.lon);
            const latMin = Math.min(...lats);
            const latMax = Math.max(...lats);
            const lonMin = Math.min(...lons);
            const lonMax = Math.max(...lons);
            const minEvents = parseInt(document.getElementById('minEvents').value);
            
            // Add small margin around data
            const latMargin = (latMax - latMin) * 0.05;
            const lonMargin = (lonMax - lonMin) * 0.05;
            const adjLatMin = latMin - latMargin;
            const adjLatMax = latMax + latMargin;
            const adjLonMin = lonMin - lonMargin;
            const adjLonMax = lonMax + lonMargin;
            
            // Use FULL available canvas space (no aspect ratio constraint)
            const drawWidth = w - 2 * padding;
            const drawHeight = h - 2 * padding;
            
            const scaleX = lon => padding + (lon - adjLonMin) / (adjLonMax - adjLonMin) * drawWidth;
            const scaleY = lat => padding + drawHeight - (lat - adjLatMin) / (adjLatMax - adjLatMin) * drawHeight;
            
            // Draw edges
            ctx.strokeStyle = 'rgba(52, 152, 219, 0.4)';
            ctx.lineWidth = 1;
            currentData.edges.forEach(e => {
                ctx.globalAlpha = 0.3 + e.weight * 0.5;
                ctx.beginPath();
                ctx.moveTo(scaleX(e.from.lon), scaleY(e.from.lat));
                ctx.lineTo(scaleX(e.to.lon), scaleY(e.to.lat));
                ctx.stroke();
            });
            ctx.globalAlpha = 1;
            
            // Draw nodes
            currentData.nodes.forEach(n => {
                const isQualified = n.count >= minEvents;
                const size = 4 + Math.log10(n.count + 1) * 3;
                
                ctx.fillStyle = isQualified ? 'rgba(39, 174, 96, 0.8)' : 'rgba(192, 57, 43, 0.4)';
                ctx.beginPath();
                ctx.arc(scaleX(n.lon), scaleY(n.lat), size, 0, Math.PI * 2);
                ctx.fill();
            });
            
            // Axis labels
            ctx.fillStyle = '#666';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`Longitude: ${lonMin.toFixed(2)}° - ${lonMax.toFixed(2)}°`, w / 2, h - 8);
            
            ctx.save();
            ctx.translate(12, h / 2);
            ctx.rotate(-Math.PI / 2);
            ctx.fillText(`Latitude: ${latMin.toFixed(2)}° - ${latMax.toFixed(2)}°`, 0, 0);
            ctx.restore();
            
            // Legend
            ctx.fillStyle = 'rgba(39, 174, 96, 0.8)';
            ctx.fillRect(w - 110, 12, 10, 10);
            ctx.fillStyle = '#555';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText('Qualified', w - 95, 21);
            
            ctx.fillStyle = 'rgba(192, 57, 43, 0.6)';
            ctx.fillRect(w - 110, 27, 10, 10);
            ctx.fillStyle = '#555';
            ctx.fillText('Not Qualified', w - 95, 36);
        }
        
        // Initialize
        initCharts();
        analyzeGrid();  // Load default on page load
    </script>
</body>
</html>
'''


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" FLASK SPATIAL EDA SERVER")
    print("=" * 60)
    print(" Open in browser: http://localhost:5000")
    print(" Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
