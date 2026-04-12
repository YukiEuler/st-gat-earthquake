# ==============================================================================
# MAIN_HYBRID.PY - Hybrid Classification-Regression Training
# ==============================================================================
"""
Train ST-GAT Hybrid model:
- Classification: max_mw significance (M >= 3 or not)
- Regression: count, log_energy, avg_depth

Usage:
    python main_hybrid.py
"""

import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, DEVICE
from data.preprocessing import DataPreprocessor
from data.adjacency import AdjacencyBuilder
from data.dataset import HybridSeismicDataset
from torch.utils.data import DataLoader

from models import STGATHybrid
from training import HybridLoss

# ==============================================================================
# CONFIGURATION FOR HYBRID MODEL
# ==============================================================================
HYBRID_CONFIG = {
    **CONFIG,
    'n_classes': 4,              # 4-class: M<1, M1-2, M2-3, M>=3
    'cls_weight': 2.0,           # Weight for classification loss
    'reg_weight': 1.0,           # Weight for regression loss
    'class_weights': [10.0, 1.0, 2.0, 15.0],  # Weights for [M<1, M1-2, M2-3, M>=3]
}


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, criterion, optimizer, adj_sparse, device):
    """Train for one epoch."""
    from tqdm import tqdm
    
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc="  Training", leave=False)
    for batch_idx, (data, targets) in enumerate(pbar):
        data = data.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        
        optimizer.zero_grad()
        outputs = model(data, adj_sparse)
        
        loss = criterion(outputs, targets)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{total_loss/max(1,n_batches):.4f}'})
    
    return total_loss / max(1, n_batches)


def evaluate(model, test_loader, criterion, adj_sparse, device):
    """Evaluate on test set with multiclass support."""
    model.eval()
    total_loss = 0.0
    all_cls_preds = []
    all_cls_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            outputs = model(data, adj_sparse)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Collect classification predictions for metrics (multiclass)
            cls_proba = outputs['classification_proba']
            cls_preds = cls_proba.argmax(dim=-1)  # (B, H, N) - class with highest prob
            all_cls_preds.append(cls_preds.cpu().numpy())
            all_cls_targets.append(targets['classification'].cpu().numpy())
    
    # Classification metrics (multiclass)
    all_cls_preds = np.concatenate(all_cls_preds, axis=0).flatten()
    all_cls_targets = np.concatenate(all_cls_targets, axis=0).flatten()
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(all_cls_targets, all_cls_preds)
    # Use weighted average for multiclass
    precision = precision_score(all_cls_targets, all_cls_preds, average='weighted', zero_division=0)
    recall = recall_score(all_cls_targets, all_cls_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_cls_targets, all_cls_preds, average='weighted', zero_division=0)
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    set_seed(HYBRID_CONFIG['seed'])
    
    print("\n" + "=" * 70)
    print(" ST-GAT HYBRID: Classification + Regression")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    print(f"   Classes: {HYBRID_CONFIG['n_classes']} (M<1, M1-2, M2-3, M>=3)")
    print("=" * 70)
    
    # ====================
    # DATA PREPARATION
    # ====================
    print("\n Loading and preprocessing data...")
    
    preprocessor = DataPreprocessor(HYBRID_CONFIG)
    data = preprocessor.process(HYBRID_CONFIG['filename'])
    
    adj_builder = AdjacencyBuilder(HYBRID_CONFIG)
    adj_scipy = adj_builder.build_distance_weighted_adj(
        data['num_nodes'], 
        data['node_info'], 
        data['grid_params'],
        use_distance_weighting=True
    )
    adj_sparse = adj_builder.scipy_to_torch_sparse(adj_scipy, device=DEVICE)
    
    # Create hybrid datasets
    train_dataset = HybridSeismicDataset(
        data['train_data'],
        window_size=HYBRID_CONFIG['window_size'],
        horizon=HYBRID_CONFIG['horizon'],
        magnitude_idx=1,  # max_mw is index 1
        regression_indices=[0, 2, 3],  # count, log_energy, avg_depth
        feature_stats=data['feature_stats']  # For proper threshold calculation
    )
    
    test_dataset = HybridSeismicDataset(
        data['test_data'],
        window_size=HYBRID_CONFIG['window_size'],
        horizon=HYBRID_CONFIG['horizon'],
        magnitude_idx=1,
        regression_indices=[0, 2, 3],
        feature_stats=data['feature_stats']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=HYBRID_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=HYBRID_CONFIG['batch_size'], shuffle=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # ====================
    # MODEL
    # ====================
    print("\n Creating Hybrid Model...")
    
    model = STGATHybrid(
        num_nodes=data['num_nodes'],
        in_features=4,  # count, max_mw, log_energy, avg_depth
        hidden_dim=HYBRID_CONFIG['hidden_dim'],
        n_regression_features=3,  # count, log_energy, avg_depth
        n_classes=HYBRID_CONFIG['n_classes'],  # 4-class classification
        horizon=HYBRID_CONFIG['horizon'],
        num_gat_layers=HYBRID_CONFIG['num_gat_layers'],
        num_heads=HYBRID_CONFIG['num_heads'],
        dropout=HYBRID_CONFIG['dropout'],
    ).to(DEVICE)
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ====================
    # TRAINING
    # ====================
    criterion = HybridLoss(
        cls_weight=HYBRID_CONFIG['cls_weight'],
        reg_weight=HYBRID_CONFIG['reg_weight'],
        class_weights=HYBRID_CONFIG['class_weights'],
        active_weight=HYBRID_CONFIG['active_weight'],
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=HYBRID_CONFIG['learning_rate'],
        weight_decay=HYBRID_CONFIG.get('weight_decay', 1e-5)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    print("\n" + "=" * 70)
    print(" Starting Training")
    print("=" * 70)
    
    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    patience = HYBRID_CONFIG['early_stopping_patience']
    
    for epoch in range(HYBRID_CONFIG['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, adj_sparse, DEVICE)
        test_metrics = evaluate(model, test_loader, criterion, adj_sparse, DEVICE)
        
        scheduler.step(test_metrics['loss'])
        
        # Track best by F1 score
        if test_metrics['f1'] > best_f1:
            best_f1 = test_metrics['f1']
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            status = " Best"
        else:
            patience_counter += 1
            status = f"  ({patience_counter}/{patience})"
        
        print(f"Epoch {epoch+1:3d}/{HYBRID_CONFIG['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_metrics['loss']:.4f} | "
              f"Acc: {test_metrics['accuracy']:.3f} | "
              f"Prec: {test_metrics['precision']:.3f} | "
              f"Rec: {test_metrics['recall']:.3f} | "
              f"F1: {test_metrics['f1']:.3f}{status}")
        
        if patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n Loaded best model (F1: {best_f1:.4f})")
    
    print("=" * 70)
    print(" Training Complete!")
    print("=" * 70)
    
    # Final evaluation
    final_metrics = evaluate(model, test_loader, criterion, adj_sparse, DEVICE)
    print("\n Final Test Metrics:")
    print(f"   Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"   Precision: {final_metrics['precision']:.4f}")
    print(f"   Recall:    {final_metrics['recall']:.4f}")
    print(f"   F1 Score:  {final_metrics['f1']:.4f}")
    
    # Save model
    output_dir = Path(HYBRID_CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': HYBRID_CONFIG,
        'metrics': final_metrics,
    }, output_dir / 'hybrid_model.pt')
    
    print(f"\n Model saved to {output_dir / 'hybrid_model.pt'}")
    
    # ====================
    # SAVE PREDICTIONS TO CSV
    # ====================
    print("\n Generating and saving predictions...")
    
    predictions_df = generate_predictions_csv(
        model, test_loader, adj_sparse, DEVICE,
        data['feature_stats'], HYBRID_CONFIG['n_classes']
    )
    
    csv_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(csv_path, index=False)
    print(f"   Predictions saved to {csv_path}")
    
    # Generate HTML visualization
    html_path = output_dir / 'predictions_viewer.html'
    generate_html_viewer(predictions_df, html_path, HYBRID_CONFIG['n_classes'])
    print(f"   HTML viewer saved to {html_path}")
    
    print("\n Done!")


def generate_predictions_csv(model, test_loader, adj_sparse, device, feature_stats, n_classes):
    """Generate predictions and save to DataFrame with DENORMALIZED values."""
    import pandas as pd
    from tqdm import tqdm
    
    model.eval()
    
    all_predictions = []
    sample_idx = 0
    
    class_names = ['M<1', 'M1-2', 'M2-3', 'M>=3']
    reg_names = ['count', 'log_energy', 'avg_depth']
    
    # Get denormalization stats (indices: 0=count, 1=max_mw, 2=log_energy, 3=avg_depth)
    # Regression uses indices [0, 2, 3] so we map accordingly
    reg_idx_map = [0, 2, 3]  # count, log_energy, avg_depth
    mean = feature_stats['mean']
    std = feature_stats['std']
    
    # DEBUG: Print feature stats
    print("\n   Feature Stats for Denormalization:")
    print(f"   Mean: {mean}")
    print(f"   Std: {std}")
    print(f"   Regression indices: {reg_idx_map}")
    for i, name in enumerate(reg_names):
        orig_idx = reg_idx_map[i]
        print(f"   {name}: mean={mean[orig_idx]:.4f}, std={std[orig_idx]:.4f}")
    
    print("\n   Generating predictions...")
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="   Predicting", leave=False)):
            data = data.to(device)
            outputs = model(data, adj_sparse)
            
            # Get predictions
            cls_proba = outputs['classification_proba'].cpu().numpy()
            cls_pred = cls_proba.argmax(axis=-1)
            reg_pred_norm = outputs['regression'].cpu().numpy()
            
            # Get targets
            cls_target = targets['classification'].numpy()
            reg_target_norm = targets['regression'].numpy()
            
            B, H, N, _ = cls_proba.shape
            
            for b in range(B):
                for h in range(H):
                    for n in range(N):
                        row = {
                            'sample_idx': sample_idx,
                            'horizon': h + 1,
                            'node_id': n,
                            # Classification
                            'cls_target': int(cls_target[b, h, n]),
                            'cls_target_name': class_names[int(cls_target[b, h, n])],
                            'cls_pred': int(cls_pred[b, h, n]),
                            'cls_pred_name': class_names[int(cls_pred[b, h, n])],
                            'cls_correct': int(cls_target[b, h, n] == cls_pred[b, h, n]),
                        }
                        
                        # Class probabilities
                        for c in range(n_classes):
                            row[f'prob_{class_names[c]}'] = float(cls_proba[b, h, n, c])
                        
                        # Regression - DENORMALIZED
                        for i, name in enumerate(reg_names):
                            orig_idx = reg_idx_map[i]
                            # Denormalize: value = normalized * std + mean
                            target_raw = float(reg_target_norm[b, h, n, i] * std[orig_idx] + mean[orig_idx])
                            pred_raw = float(reg_pred_norm[b, h, n, i] * std[orig_idx] + mean[orig_idx])
                            row[f'{name}_target'] = target_raw
                            row[f'{name}_pred'] = pred_raw
                        
                        all_predictions.append(row)
                
                sample_idx += 1
    
    # DEBUG: Print sample of first row
    if all_predictions:
        print("\n   Sample output (first row):")
        first = all_predictions[0]
        for name in reg_names:
            print(f"   {name}: target={first[f'{name}_target']:.4f}, pred={first[f'{name}_pred']:.4f}")
    
    return pd.DataFrame(all_predictions)


def generate_html_viewer(df, output_path, n_classes):
    """Generate simple HTML viewer with node selector and time series."""
    import json
    import numpy as np
    
    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    # Get unique nodes
    nodes = sorted(df['node_id'].unique())
    n_nodes = len(nodes)
    
    # Prepare data per node (aggregate by sample_idx for each node)
    node_data = {}
    for node in nodes:
        node_df = df[df['node_id'] == node].groupby('sample_idx').agg({
            'cls_target': 'first',
            'cls_pred': 'first',
            'cls_correct': 'mean',
            'count_target': 'mean',
            'count_pred': 'mean',
            'log_energy_target': 'mean',
            'log_energy_pred': 'mean',
        }).reset_index()
        # Convert to Python int for JSON serialization
        node_data[int(node)] = node_df.to_dict('records')
    
    data_json = json.dumps(node_data, cls=NumpyEncoder)
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Earthquake Prediction Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .controls {{
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .controls label {{
            margin-right: 10px;
            font-weight: bold;
        }}
        .controls select, .controls input {{
            padding: 8px;
            margin-right: 20px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }}
        .stats {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background: #fff;
            padding: 15px 25px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-box .value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .stat-box .label {{
            color: #666;
            font-size: 12px;
        }}
        .chart-container {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        canvas {{
            max-height: 250px;
        }}
    </style>
</head>
<body>
    <h1>Earthquake Prediction Viewer</h1>
    
    <div class="controls">
        <label>Node:</label>
        <select id="nodeSelect">
            {" ".join([f'<option value="{n}">Node {n}</option>' for n in nodes])}
        </select>
        
        <label>Time Range:</label>
        <input type="range" id="rangeSlider" min="20" max="300" value="100">
        <span id="rangeValue">100</span>
        
        <label>Start:</label>
        <input type="range" id="startSlider" min="0" max="500" value="0">
        <span id="startValue">0</span>
    </div>
    
    <div class="stats" id="statsContainer"></div>
    
    <div class="chart-container">
        <div class="chart-title">Magnitude Class: Actual vs Predicted</div>
        <canvas id="magChart"></canvas>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Count: Actual vs Predicted</div>
        <canvas id="countChart"></canvas>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">Log Energy: Actual vs Predicted</div>
        <canvas id="energyChart"></canvas>
    </div>
    
    <script>
        const allData = {data_json};
        const classNames = ['M<1', 'M1-2', 'M2-3', 'M>=3'];
        
        let magChart, countChart, energyChart;
        
        function initCharts() {{
            magChart = new Chart(document.getElementById('magChart'), {{
                type: 'line',
                data: {{ labels: [], datasets: [] }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{ min: 0, max: 3, ticks: {{ stepSize: 1, callback: v => classNames[v] || v }} }}
                    }}
                }}
            }});
            
            countChart = new Chart(document.getElementById('countChart'), {{
                type: 'line',
                data: {{ labels: [], datasets: [] }},
                options: {{ responsive: true }}
            }});
            
            energyChart = new Chart(document.getElementById('energyChart'), {{
                type: 'line',
                data: {{ labels: [], datasets: [] }},
                options: {{ responsive: true }}
            }});
        }}
        
        function updateCharts() {{
            const node = parseInt(document.getElementById('nodeSelect').value);
            const range = parseInt(document.getElementById('rangeSlider').value);
            const start = parseInt(document.getElementById('startSlider').value);
            
            const nodeData = allData[node] || [];
            const end = Math.min(start + range, nodeData.length);
            const data = nodeData.slice(start, end);
            
            document.getElementById('rangeValue').textContent = range;
            document.getElementById('startValue').textContent = start;
            document.getElementById('startSlider').max = Math.max(0, nodeData.length - range);
            
            const labels = data.map(d => d.sample_idx);
            
            // Magnitude chart
            magChart.data.labels = labels;
            magChart.data.datasets = [
                {{ label: 'Actual', data: data.map(d => d.cls_target), borderColor: '#333', borderWidth: 2, fill: false, pointRadius: 2 }},
                {{ label: 'Predicted', data: data.map(d => d.cls_pred), borderColor: '#e74c3c', borderWidth: 2, fill: false, pointRadius: 2, borderDash: [5, 5] }}
            ];
            magChart.update();
            
            // Count chart
            countChart.data.labels = labels;
            countChart.data.datasets = [
                {{ label: 'Actual', data: data.map(d => d.count_target), borderColor: '#333', borderWidth: 2, fill: false, pointRadius: 1 }},
                {{ label: 'Predicted', data: data.map(d => d.count_pred), borderColor: '#3498db', borderWidth: 2, fill: false, pointRadius: 1, borderDash: [5, 5] }}
            ];
            countChart.update();
            
            // Energy chart
            energyChart.data.labels = labels;
            energyChart.data.datasets = [
                {{ label: 'Actual', data: data.map(d => d.log_energy_target), borderColor: '#333', borderWidth: 2, fill: false, pointRadius: 1 }},
                {{ label: 'Predicted', data: data.map(d => d.log_energy_pred), borderColor: '#27ae60', borderWidth: 2, fill: false, pointRadius: 1, borderDash: [5, 5] }}
            ];
            energyChart.update();
            
            // Stats
            const accuracy = data.length > 0 ? (data.reduce((a, b) => a + b.cls_correct, 0) / data.length * 100).toFixed(1) : 0;
            const class3 = data.filter(d => d.cls_target === 3).length;
            const class3Correct = data.filter(d => d.cls_target === 3 && d.cls_pred === 3).length;
            const class3Recall = class3 > 0 ? (class3Correct / class3 * 100).toFixed(1) : 'N/A';
            
            document.getElementById('statsContainer').innerHTML = `
                <div class="stat-box"><div class="value">${{data.length}}</div><div class="label">Samples</div></div>
                <div class="stat-box"><div class="value">${{accuracy}}%</div><div class="label">Accuracy</div></div>
                <div class="stat-box"><div class="value">${{class3}}</div><div class="label">M>=3 Events</div></div>
                <div class="stat-box"><div class="value">${{class3Recall}}%</div><div class="label">M>=3 Recall</div></div>
            `;
        }}
        
        document.getElementById('nodeSelect').addEventListener('change', updateCharts);
        document.getElementById('rangeSlider').addEventListener('input', updateCharts);
        document.getElementById('startSlider').addEventListener('input', updateCharts);
        
        initCharts();
        updateCharts();
    </script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == '__main__':
    main()


