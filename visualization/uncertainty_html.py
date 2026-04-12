# ==============================================================================
# UNCERTAINTY_HTML.PY - Interactive HTML Visualization with Uncertainty
# ==============================================================================

import numpy as np
import json
from pathlib import Path
from datetime import datetime


def generate_uncertainty_html(predictions, uncertainties, targets, 
                               node_info, feature_names, output_path,
                               title="Earthquake Prediction with Uncertainty"):
    """
    Generate interactive HTML visualization with uncertainty bands.
    
    Args:
        predictions: np.array (T, N, F) - Mean predictions from ensemble
        uncertainties: np.array (T, N, F) - Std from ensemble
        targets: np.array (T, N, F) - Ground truth
        node_info: dict with 'lat', 'lon' for each node
        feature_names: list of feature names
        output_path: Path to save HTML file
        title: Title for the visualization
    """
    
    # Get dimensions
    n_timesteps, n_nodes, n_features = predictions.shape
    
    # Find magnitude index
    mw_idx = feature_names.index('max_mw') if 'max_mw' in feature_names else 1
    
    # Prepare data for visualization
    timesteps = list(range(n_timesteps))
    
    # Create traces for each node (top 5 most active)
    node_activity = targets[:, :, mw_idx].sum(axis=0)
    top_nodes = np.argsort(node_activity)[-5:][::-1]
    
    # Build Plotly data
    traces_data = []
    for node_idx in top_nodes:
        pred = predictions[:, node_idx, mw_idx].tolist()
        std = uncertainties[:, node_idx, mw_idx].tolist()
        actual = targets[:, node_idx, mw_idx].tolist()
        
        # Upper and lower bounds
        upper = (predictions[:, node_idx, mw_idx] + 1.96 * uncertainties[:, node_idx, mw_idx]).tolist()
        lower = (predictions[:, node_idx, mw_idx] - 1.96 * uncertainties[:, node_idx, mw_idx]).tolist()
        lower = [max(0, l) for l in lower]  # Cap at 0
        
        node_lat = node_info.get('lat', [0]*n_nodes)[node_idx] if isinstance(node_info.get('lat'), list) else 0
        node_lon = node_info.get('lon', [0]*n_nodes)[node_idx] if isinstance(node_info.get('lon'), list) else 0
        
        traces_data.append({
            'node_id': int(node_idx),
            'lat': float(node_lat),
            'lon': float(node_lon),
            'pred': pred,
            'upper': upper,
            'lower': lower,
            'actual': actual,
            'std': std
        })
    
    # Calculate summary statistics
    mean_uncertainty = float(uncertainties[:, :, mw_idx].mean())
    max_uncertainty = float(uncertainties[:, :, mw_idx].max())
    rmse = float(np.sqrt(((predictions[:, :, mw_idx] - targets[:, :, mw_idx])**2).mean()))
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: #00d4ff;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0, 212, 255, 0.2);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .stat-label {{
            color: #888;
            margin-top: 5px;
        }}
        .chart-container {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .node-selector {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
            justify-content: center;
        }}
        .node-btn {{
            padding: 10px 20px;
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 25px;
            color: #00d4ff;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .node-btn:hover, .node-btn.active {{
            background: rgba(0, 212, 255, 0.3);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }}
        .legend-info {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }}
        .uncertainty-band {{
            width: 20px;
            height: 15px;
            background: rgba(0, 212, 255, 0.2);
            border-radius: 3px;
        }}
        footer {{
            text-align: center;
            margin-top: 40px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌍 {title}</h1>
        <p class="subtitle">Deep Ensemble ST-GAT Predictions with 95% Confidence Intervals</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{n_nodes}</div>
                <div class="stat-label">Spatial Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_timesteps}</div>
                <div class="stat-label">Time Steps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{rmse:.3f}</div>
                <div class="stat-label">RMSE (Mw)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">±{mean_uncertainty:.3f}</div>
                <div class="stat-label">Mean Uncertainty</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3 style="text-align: center; margin-bottom: 15px; color: #aaa;">
                Prediction vs Actual with Uncertainty Bands
            </h3>
            <div class="node-selector" id="nodeSelector"></div>
            <div id="mainChart" style="height: 450px;"></div>
            <div class="legend-info">
                <div class="legend-item">
                    <div class="legend-color" style="background: #00d4ff;"></div>
                    <span>Prediction</span>
                </div>
                <div class="legend-item">
                    <div class="uncertainty-band"></div>
                    <span>95% CI</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ff6b6b;"></div>
                    <span>Actual</span>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h3 style="text-align: center; margin-bottom: 15px; color: #aaa;">
                Uncertainty Distribution Over Time
            </h3>
            <div id="uncertaintyChart" style="height: 350px;"></div>
        </div>
        
        <footer>
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
            ST-GAT Earthquake Prediction System
        </footer>
    </div>
    
    <script>
        const tracesData = {json.dumps(traces_data)};
        const timesteps = {json.dumps(timesteps)};
        let currentNode = 0;
        
        // Create node selector buttons
        const selector = document.getElementById('nodeSelector');
        tracesData.forEach((trace, idx) => {{
            const btn = document.createElement('button');
            btn.className = 'node-btn' + (idx === 0 ? ' active' : '');
            btn.textContent = `Node ${{trace.node_id}}`;
            btn.onclick = () => selectNode(idx);
            selector.appendChild(btn);
        }});
        
        function selectNode(idx) {{
            currentNode = idx;
            document.querySelectorAll('.node-btn').forEach((btn, i) => {{
                btn.classList.toggle('active', i === idx);
            }});
            updateMainChart();
        }}
        
        function updateMainChart() {{
            const trace = tracesData[currentNode];
            
            const data = [
                // Uncertainty band (fill between)
                {{
                    x: timesteps.concat(timesteps.slice().reverse()),
                    y: trace.upper.concat(trace.lower.slice().reverse()),
                    fill: 'toself',
                    fillcolor: 'rgba(0, 212, 255, 0.15)',
                    line: {{ color: 'transparent' }},
                    name: '95% CI',
                    hoverinfo: 'skip'
                }},
                // Prediction line
                {{
                    x: timesteps,
                    y: trace.pred,
                    mode: 'lines',
                    name: 'Prediction',
                    line: {{ color: '#00d4ff', width: 2 }},
                    hovertemplate: 'Pred: %{{y:.3f}}<extra></extra>'
                }},
                // Actual line
                {{
                    x: timesteps,
                    y: trace.actual,
                    mode: 'lines+markers',
                    name: 'Actual',
                    line: {{ color: '#ff6b6b', width: 1.5, dash: 'dot' }},
                    marker: {{ size: 4 }},
                    hovertemplate: 'Actual: %{{y:.3f}}<extra></extra>'
                }}
            ];
            
            const layout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {{ color: '#aaa' }},
                xaxis: {{
                    title: 'Time Step',
                    gridcolor: 'rgba(255,255,255,0.1)',
                    zerolinecolor: 'rgba(255,255,255,0.1)'
                }},
                yaxis: {{
                    title: 'Max Magnitude (Mw)',
                    gridcolor: 'rgba(255,255,255,0.1)',
                    zerolinecolor: 'rgba(255,255,255,0.1)'
                }},
                showlegend: false,
                margin: {{ t: 20, b: 50, l: 60, r: 20 }},
                hovermode: 'x unified'
            }};
            
            Plotly.newPlot('mainChart', data, layout, {{ responsive: true }});
        }}
        
        function createUncertaintyChart() {{
            // Aggregate uncertainty across all nodes
            const meanUncertainty = timesteps.map((t, i) => {{
                let sum = 0;
                tracesData.forEach(trace => {{ sum += trace.std[i]; }});
                return sum / tracesData.length;
            }});
            
            const maxUncertainty = timesteps.map((t, i) => {{
                let max = 0;
                tracesData.forEach(trace => {{ max = Math.max(max, trace.std[i]); }});
                return max;
            }});
            
            const data = [
                {{
                    x: timesteps,
                    y: maxUncertainty,
                    fill: 'tozeroy',
                    fillcolor: 'rgba(255, 107, 107, 0.2)',
                    line: {{ color: '#ff6b6b', width: 1 }},
                    name: 'Max Uncertainty'
                }},
                {{
                    x: timesteps,
                    y: meanUncertainty,
                    fill: 'tozeroy',
                    fillcolor: 'rgba(0, 212, 255, 0.3)',
                    line: {{ color: '#00d4ff', width: 2 }},
                    name: 'Mean Uncertainty'
                }}
            ];
            
            const layout = {{
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: {{ color: '#aaa' }},
                xaxis: {{
                    title: 'Time Step',
                    gridcolor: 'rgba(255,255,255,0.1)'
                }},
                yaxis: {{
                    title: 'Uncertainty (Std)',
                    gridcolor: 'rgba(255,255,255,0.1)'
                }},
                legend: {{ x: 0.02, y: 0.98 }},
                margin: {{ t: 20, b: 50, l: 60, r: 20 }}
            }};
            
            Plotly.newPlot('uncertaintyChart', data, layout, {{ responsive: true }});
        }}
        
        // Initialize charts
        updateMainChart();
        createUncertaintyChart();
    </script>
</body>
</html>
'''
    
    # Save HTML file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')
    
    print(f"   Saved HTML visualization: {output_path}")
    return output_path


def generate_spatial_uncertainty_html(predictions, uncertainties, targets,
                                       node_info, output_path,
                                       title="Spatial Uncertainty Map"):
    """
    Generate spatial map showing uncertainty across nodes.
    
    Args:
        predictions: np.array (T, N, F)
        uncertainties: np.array (T, N, F)
        targets: np.array (T, N, F)
        node_info: dict with 'lat', 'lon', 'center_lat', 'center_lon'
        output_path: Path to save HTML
    """
    
    n_timesteps, n_nodes, n_features = predictions.shape
    mw_idx = 1  # Assume magnitude is at index 1
    
    # Aggregate over time
    mean_pred = predictions[:, :, mw_idx].mean(axis=0)  # (N,)
    mean_uncertainty = uncertainties[:, :, mw_idx].mean(axis=0)  # (N,)
    mean_actual = targets[:, :, mw_idx].mean(axis=0)  # (N,)
    
    # Get node coordinates
    lats = node_info.get('lat', [42.7] * n_nodes)
    lons = node_info.get('lon', [13.2] * n_nodes)
    
    if not isinstance(lats, list):
        lats = list(lats)
        lons = list(lons)
    
    # Build node data
    nodes_data = []
    for i in range(n_nodes):
        nodes_data.append({
            'lat': float(lats[i]) if i < len(lats) else 42.7,
            'lon': float(lons[i]) if i < len(lons) else 13.2,
            'pred': float(mean_pred[i]),
            'uncertainty': float(mean_uncertainty[i]),
            'actual': float(mean_actual[i]),
            'node_id': i
        })
    
    center_lat = node_info.get('center_lat', np.mean(lats))
    center_lon = node_info.get('center_lon', np.mean(lons))
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        #map {{
            height: 100vh;
            width: 100%;
        }}
        .info-box {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 1000;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            min-width: 200px;
        }}
        .info-box h3 {{
            margin: 0 0 10px 0;
            color: #00d4ff;
        }}
        .legend {{
            position: absolute;
            bottom: 20px;
            left: 10px;
            z-index: 1000;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 5px 0;
        }}
        .legend-circle {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid white;
        }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="info-box">
        <h3>🌍 Spatial Uncertainty</h3>
        <div id="nodeInfo">Hover over a node</div>
    </div>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-circle" style="background: rgba(0, 212, 255, 0.7);"></div>
            <span>Low Uncertainty</span>
        </div>
        <div class="legend-item">
            <div class="legend-circle" style="background: rgba(255, 107, 107, 0.7);"></div>
            <span>High Uncertainty</span>
        </div>
    </div>
    
    <script>
        const nodesData = {json.dumps(nodes_data)};
        
        const map = L.map('map').setView([{center_lat}, {center_lon}], 11);
        
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '© OpenStreetMap contributors © CARTO'
        }}).addTo(map);
        
        // Find uncertainty range for color scaling
        const uncertainties = nodesData.map(n => n.uncertainty);
        const minU = Math.min(...uncertainties);
        const maxU = Math.max(...uncertainties);
        
        function getColor(uncertainty) {{
            const ratio = (uncertainty - minU) / (maxU - minU + 0.001);
            const r = Math.round(0 + ratio * 255);
            const g = Math.round(212 - ratio * 105);
            const b = Math.round(255 - ratio * 148);
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}
        
        nodesData.forEach(node => {{
            const radius = 300 + node.uncertainty * 500;  // Size by uncertainty
            const color = getColor(node.uncertainty);
            
            const circle = L.circle([node.lat, node.lon], {{
                radius: radius,
                fillColor: color,
                fillOpacity: 0.6,
                color: 'white',
                weight: 1
            }}).addTo(map);
            
            circle.on('mouseover', function() {{
                document.getElementById('nodeInfo').innerHTML = `
                    <strong>Node ${{node.node_id}}</strong><br>
                    Lat: ${{node.lat.toFixed(4)}}, Lon: ${{node.lon.toFixed(4)}}<br>
                    <hr style="border-color: #333; margin: 8px 0;">
                    Prediction: ${{node.pred.toFixed(3)}}<br>
                    Actual: ${{node.actual.toFixed(3)}}<br>
                    <span style="color: ${{color}}">Uncertainty: ±${{node.uncertainty.toFixed(3)}}</span>
                `;
            }});
        }});
    </script>
</body>
</html>
'''
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')
    
    print(f"   Saved spatial HTML visualization: {output_path}")
    return output_path
