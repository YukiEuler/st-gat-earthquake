# ==============================================================================
# MAIN.PY - Entry Point for Earthquake Prediction
# ==============================================================================
"""
ST-GAT Earthquake Prediction Pipeline

Usage:
    python main.py --mode train          # Train model
    python main.py --mode eval           # Evaluate saved model
    python main.py --mode ablation       # Run ablation study
    python main.py --mode ensemble       # Train Deep Ensemble
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, DEVICE, print_config

# Set random seeds for full reproducibility
def set_seed(seed=42):
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For CUDA >= 10.2, also set this for full determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # For PyTorch >= 1.8, enable deterministic algorithms globally
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations don't have deterministic implementations
            pass


def main(args):
    set_seed(CONFIG['seed'])
    
    print("\n" + "=" * 70)
    print(" ST-GAT EARTHQUAKE PREDICTION")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    print(f"   Mode: {args.mode}")
    print("=" * 70)
    
    if args.print_config:
        print_config()
    
    # ====================
    # DATA PREPARATION
    # ====================
    print("\n Loading and preprocessing data...")
    
    from data.preprocessing import DataPreprocessor
    from data.adjacency import AdjacencyBuilder
    from data.dataset import SeismicDataset, create_dataloaders
    from torch.utils.data import DataLoader
    
    preprocessor = DataPreprocessor(CONFIG)
    data = preprocessor.process(CONFIG['filename'])
    
    # Build adjacency matrix
    adj_builder = AdjacencyBuilder(CONFIG)
    adj_scipy = adj_builder.build_distance_weighted_adj(
        data['num_nodes'], 
        data['node_info'], 
        data['grid_params'],
        use_distance_weighting=True
    )
    adj_sparse = adj_builder.scipy_to_torch_sparse(adj_scipy, device=DEVICE)
    
    # Get target feature indices
    all_features = CONFIG['features']
    target_features = CONFIG.get('target_features', all_features)
    target_indices = [all_features.index(f) for f in target_features if f in all_features]
    
    # Create dataloaders
    train_dataset = SeismicDataset(
        data['train_data'],
        window_size=CONFIG['window_size'],
        horizon=CONFIG['horizon'],
        target_indices=target_indices
    )
    val_dataset = SeismicDataset(
        data['val_data'],
        window_size=CONFIG['window_size'],
        horizon=CONFIG['horizon'],
        target_indices=target_indices
    )
    test_dataset = SeismicDataset(
        data['test_data'],
        window_size=CONFIG['window_size'],
        horizon=CONFIG['horizon'],
        target_indices=target_indices
    )
    
    # Create DataLoaders with reproducible shuffling
    g = torch.Generator()
    g.manual_seed(CONFIG['seed'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    data_info = {
        'num_nodes': data['num_nodes'],
        'n_input_features': data['train_data'].shape[-1], 
        'n_target_features': len(target_features),
        'target_features': target_features,
        'feature_stats': data['feature_stats'],  # For denormalization
        'target_indices': target_indices,         # For target feature mapping
    }
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # ====================
    # MODE: TRAIN
    # ====================
    if args.mode == 'train':
        from models import STGAT, STTFT, STGATWarmStart, STGATMultiScale
        from training import WeightedMSELoss, AsymmetricMSELoss, QuantileLoss, Trainer
        from evaluation import MetricsCalculator
        from visualization import PredictionVisualizer, AttentionVisualizer, SpatialVisualizer
        from utils import save_all_outputs
        from utils.io import generate_summary_report
        
        # Create model based on temporal_model config
        temporal_model = CONFIG.get('temporal_model', 'lstm')
        
        if temporal_model == 'tft':
            print(f"\n Using ST-TFT (GAT + Temporal Fusion Transformer)")
            model = STTFT(
                num_nodes=data['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=CONFIG['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=CONFIG['horizon'],
                num_gat_layers=CONFIG['num_gat_layers'],
                num_heads=CONFIG['num_heads'],
                tft_layers=CONFIG.get('tft_layers', 2),
                dropout=CONFIG['dropout'],
            ).to(DEVICE)
        elif temporal_model == 'multiscale':
            print(f"\n Using ST-GAT Multi-Scale (GAT + Multi-Resolution LSTM)")
            model = STGATMultiScale(
                num_nodes=data['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=CONFIG['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=CONFIG['horizon'],
                num_gat_layers=CONFIG['num_gat_layers'],
                num_heads=CONFIG['num_heads'],
                dropout=CONFIG['dropout'],
                scales=CONFIG.get('multiscale_scales', [1, 2, 4]),
                fusion_type=CONFIG.get('multiscale_fusion', 'concat'),
                node_embed_dim=16,
                pool_type='avg',
            ).to(DEVICE)
        else:
            print(f"\n Using ST-GAT (GAT + LSTM)")
            model = STGAT(
                        num_nodes=data['num_nodes'],
                        in_features=data_info['n_input_features'],
                        hidden_dim=CONFIG['hidden_dim'],
                        out_features=data_info['n_target_features'],
                        horizon=CONFIG['horizon'],
                        num_gat_layers=CONFIG['num_gat_layers'],
                        num_heads=CONFIG['num_heads'],
                        dropout=CONFIG['dropout'],
                    ).to(DEVICE)
                            
        print(f"\n Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Select loss function based on config (NEW)
        loss_type = CONFIG.get('loss_type', 'weighted_mse')
        print(f"\n Using Loss Function: {loss_type}")
        
        if loss_type == 'asymmetric':
            criterion = AsymmetricMSELoss(
                alpha=CONFIG.get('asymmetric_alpha', 0.8),
                magnitude_idx=CONFIG.get('magnitude_idx', 1),
                active_weight=CONFIG['active_weight'],
                feature_weights=CONFIG.get('feature_weights')
            )
        elif loss_type == 'quantile':
            criterion = QuantileLoss(quantile=0.9)
        elif loss_type == 'active_only':
            from training.losses import ActiveOnlyMSELoss
            criterion = ActiveOnlyMSELoss(
                feature_weights=CONFIG.get('feature_weights')
            )
        elif loss_type == 'sparse_aware':
            from training.losses import SparseAwareLoss
            criterion = SparseAwareLoss(
                active_loss_weight=CONFIG['active_weight'],
                underpredict_penalty=CONFIG.get('underpredict_penalty', 2.0),
                feature_weights=CONFIG.get('feature_weights')
            )
        elif loss_type == 'focal':
            from training.losses import FocalMSELoss
            criterion = FocalMSELoss(
                gamma=CONFIG.get('focal_gamma', 2.0),
                active_weight=CONFIG['active_weight'],
                feature_weights=CONFIG.get('feature_weights')
            )
        elif loss_type == 'multiscale':
            from training.losses import MultiScaleLoss
            criterion = MultiScaleLoss(
                scales=CONFIG.get('multiscale_horizons', [1, 2, 4]),  # 6h, 12h, 24h
                scale_weights=CONFIG.get('multiscale_weights', [1.0, 1.0, 1.0]),
                magnitude_idx=CONFIG.get('magnitude_idx', 1),
                active_weight=CONFIG['active_weight'],
                feature_weights=CONFIG.get('feature_weights')
            )
        else:  # default: weighted_mse
            criterion = WeightedMSELoss(
                active_weight=CONFIG['active_weight'],
                feature_weights=CONFIG.get('feature_weights')
            )
        
        trainer = Trainer(model, criterion, CONFIG, DEVICE)
        train_result = trainer.fit(train_loader, val_loader, adj_sparse)
        
        # Generate predictions
        print("\n Generating predictions...")
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                batch_data = batch_data.to(DEVICE)
                output = model(batch_data, adj_sparse)
                predictions.append(output.cpu().numpy())
                targets.append(batch_target.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Denormalize predictions and targets for proper metrics calculation
        print(" Denormalizing for metrics calculation...")
        feature_stats = data['feature_stats']
        
        # Get target feature stats (slice based on target_features)
        target_mean = feature_stats['mean'][target_indices]
        target_std = feature_stats['std'][target_indices]
        
        print(f"   Target features: {data_info['target_features']}")
        print(f"   Mean: {target_mean}")
        print(f"   Std: {target_std}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Targets shape: {targets.shape}")
        
        # Denormalize: x_denorm = x_norm * std + mean
        # predictions shape: (B, H, N, F) - need to broadcast correctly
        # stats shape: (F,) -> need to reshape for broadcasting
        n_dim = predictions.ndim
        if n_dim == 4:  # (B, H, N, F)
            target_mean = target_mean.reshape(1, 1, 1, -1)
            target_std = target_std.reshape(1, 1, 1, -1)
        elif n_dim == 3:  # (B, N, F)
            target_mean = target_mean.reshape(1, 1, -1)
            target_std = target_std.reshape(1, 1, -1)
        
        predictions_denorm = predictions * target_std + target_mean
        targets_denorm = targets * target_std + target_mean
        
        # Calculate metrics on DENORMALIZED data
        metrics_calc = MetricsCalculator(feature_names=data_info['target_features'])
        metrics = metrics_calc.calculate_all_metrics(targets_denorm, predictions_denorm, magnitude_idx=CONFIG.get('magnitude_idx', 1))
        metrics_calc.print_metrics(metrics)
        
        # Visualizations
        output_dir = Path(CONFIG['output_dir'])
        
        pred_viz = PredictionVisualizer(
            feature_names=data['feature_names'],
            save_dir=output_dir / 'figures'
        )
        pred_viz.plot_training_history(train_result['train_losses'], 
                                       train_result['val_losses'],
                                       train_result['best_loss'])
        pred_viz.plot_scatter_pred_vs_actual(targets_denorm, predictions_denorm)
        pred_viz.plot_timeseries_prediction(targets_denorm, predictions_denorm, 
                                            node_idx=0, horizon_idx=0)
        pred_viz.plot_per_horizon_metrics(metrics['per_horizon'])
        
        # Plot predictions at most active node
        print("\n Visualizing most active node...")
        pred_viz.plot_most_active_node(targets_denorm, predictions_denorm, 
                                       horizon_idx=0,
                                       save_name='most_active_node_prediction')
        
        # Attention visualization
        if CONFIG.get('save_attention', True):
            att_viz = AttentionVisualizer(
                node_coords=adj_builder.coords,
                save_dir=output_dir / 'figures'
            )
            sample_data = data['test_data'][:CONFIG['window_size']]
            attention_weights = att_viz.extract_attention_weights(
                model, sample_data, adj_sparse, DEVICE
            )
            att_viz.plot_attention_distribution(attention_weights, 
                                                save_name='attention_distribution')
            att_viz.plot_top_attention_edges(attention_weights, adj_sparse,
                                            save_name='top_attention_edges')
        
        # Spatial visualization
        spatial_viz = SpatialVisualizer(
            node_coords=adj_builder.coords,
            save_dir=output_dir / 'figures'
        )
        spatial_viz.plot_spatial_heatmap(targets, predictions)
        spatial_viz.plot_node_activity(targets)
        
        # Save outputs
        save_all_outputs(
            output_dir, model, CONFIG, metrics, adj_scipy,
            data['feature_stats'], data['node_info'], train_result
        )
        
        # ====================
        # SAVE PREDICTIONS TO CSV & HTML VIEWER
        # ====================
        print("\n Saving predictions to CSV and generating HTML viewer...")
        
        predictions_df = generate_regression_predictions_csv(
            predictions_denorm, targets_denorm,
            data_info['target_features'], output_dir
        )
        
        html_path = output_dir / 'predictions_viewer.html'
        generate_regression_html_viewer(predictions_df, html_path, data_info['target_features'])
        print(f"   HTML viewer saved to {html_path}")
        generate_summary_report(output_dir, CONFIG, metrics, train_result)
        
    # ====================
    # MODE: ENSEMBLE
    # ====================
    elif args.mode == 'ensemble':
        from models import STGAT
        from training import WeightedMSELoss, DeepEnsemble
        from evaluation import MetricsCalculator
        from visualization import PredictionVisualizer, SpatialVisualizer
        
        print("\n Training Deep Ensemble for Uncertainty Estimation...")
        
        model_kwargs = {
            'num_nodes': data['num_nodes'],
            'in_features': data_info['n_input_features'],
            'hidden_dim': CONFIG['hidden_dim'],
            'out_features': data_info['n_target_features'],
            'horizon': CONFIG['horizon'],
            'num_gat_layers': CONFIG['num_gat_layers'],
            'num_heads': CONFIG['num_heads'],
            'dropout': CONFIG['dropout'],
        }
        
        criterion = WeightedMSELoss(
            active_weight=CONFIG['active_weight'],
            feature_weights=CONFIG.get('feature_weights')
        )
        
        ensemble = DeepEnsemble(
            model_class=STGAT,
            model_kwargs=model_kwargs,
            n_models=CONFIG.get('n_ensemble_models', 5),
            device=DEVICE
        )
        
        output_dir = Path(CONFIG['output_dir'])
        ensemble.fit(train_loader, val_loader, adj_sparse, criterion, CONFIG,
                    save_dir=output_dir / 'models')
        
        # Generate predictions with uncertainty
        results = ensemble.generate_predictions(test_loader, adj_sparse)
        
        # Calculate metrics
        metrics_calc = MetricsCalculator(feature_names=data['feature_names'])
        metrics = metrics_calc.calculate_all_metrics(
            results['targets'], results['mean'], results['std']
        )
        metrics_calc.print_metrics(metrics)
        
        # Visualizations
        pred_viz = PredictionVisualizer(
            feature_names=data['feature_names'],
            save_dir=output_dir / 'figures'
        )
        pred_viz.plot_scatter_pred_vs_actual(results['targets'], results['mean'])
        pred_viz.plot_timeseries_prediction(
            results['targets'], results['mean'],
            node_idx=0, horizon_idx=0, y_pred_std=results['std']
        )
        
        # Spatial uncertainty
        spatial_viz = SpatialVisualizer(
            node_coords=adj_builder.coords,
            save_dir=output_dir / 'figures'
        )
        spatial_viz.plot_uncertainty_spatial(results['mean'], results['std'])
        
        # Save
        metrics_calc.save_metrics(metrics, output_dir / 'metrics' / 'ensemble_metrics.json')
        
    # ====================
    # MODE: ABLATION
    # ====================
    elif args.mode == 'ablation':
        from evaluation import AblationStudy
        from visualization import PredictionVisualizer
        
        print("\n Running Ablation Study...")
        
        ablation = AblationStudy(CONFIG, DEVICE)
        
        # Run all ablation configurations
        # Architecture: full_stgat, single_head, stgat_8heads, st_tft, gcn_lstm, stgat_no_skip, stgat_no_embed
        # Dual-Path: dualpath_concat, dualpath_gate, dualpath_add
        # Learnable Graph: learnable_geo30, learnable_geo50, learnable_geo70
        # Loss: loss_mse, loss_focal, loss_multiscale, loss_asymmetric
        # Regularization: dropout_low, dropout_high
        # Baselines: lstm_only, tft_only, etas, etas_fast, naive, moving_avg
        results = ablation.run_all(
            train_loader, val_loader, test_loader, adj_sparse, data_info,
            configs_to_run=[
                # Architecture
                'full_stgat', 'single_head', 'stgat_8heads', #'st_tft', 
                'gcn_lstm', 'stgat_no_skip', 'stgat_no_embed',
                # Dual-Path
                'dualpath_concat', 'dualpath_gate', 'dualpath_add',
                # Learnable Graph
                'learnable_geo30', 'learnable_geo50', 'learnable_geo70',
                # Warm Start (Curriculum Learning)
                'warmstart_5', 'warmstart_10', 'warmstart_20',
                # Loss functions
                'loss_mse', 'loss_focal', 'loss_multiscale', 'loss_asymmetric',
                # Regularization
                'dropout_low', 'dropout_high',
                # Baselines (Neural)
                'lstm_only', 'tft_only',
                # Baselines (Statistical)
                'etas', 'etas_fast', 'naive', 'moving_avg'
            ]
        )
        
        df = ablation.save_results(Path(CONFIG['output_dir']) / 'metrics')
        print("\n Ablation Results:")
        print(df.to_string())
        
        # Visualization
        pred_viz = PredictionVisualizer(save_dir=Path(CONFIG['output_dir']) / 'figures')
        pred_viz.plot_ablation_comparison(df)
    
    print("\n Done!")


def generate_regression_predictions_csv(predictions, targets, feature_names, output_dir):
    """Generate predictions CSV for regression model."""
    import pandas as pd
    
    all_rows = []
    B, H, N, F = predictions.shape
    
    sample_idx = 0
    for b in range(B):
        for h in range(H):
            for n in range(N):
                row = {
                    'sample_idx': sample_idx,
                    'horizon': h + 1,
                    'node_id': n,
                }
                for f_idx, fname in enumerate(feature_names):
                    row[f'{fname}_target'] = float(targets[b, h, n, f_idx])
                    row[f'{fname}_pred'] = float(predictions[b, h, n, f_idx])
                all_rows.append(row)
        sample_idx += 1
    
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / 'predictions.csv'
    df.to_csv(csv_path, index=False)
    print(f"   Predictions saved to {csv_path}")
    return df


def generate_regression_html_viewer(df, output_path, feature_names):
    """Generate simple HTML viewer with node selector for regression predictions."""
    import json
    import numpy as np
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    nodes = sorted(df['node_id'].unique())
    
    # Prepare data per node
    node_data = {}
    for node in nodes:
        agg_dict = {'sample_idx': 'first'}
        for fname in feature_names:
            agg_dict[f'{fname}_target'] = 'mean'
            agg_dict[f'{fname}_pred'] = 'mean'
        
        node_df = df[df['node_id'] == node].groupby('sample_idx').agg(agg_dict).reset_index(drop=True)
        node_df['sample_idx'] = range(len(node_df))
        node_data[int(node)] = node_df.to_dict('records')
    
    data_json = json.dumps(node_data, cls=NumpyEncoder)
    feature_names_json = json.dumps(feature_names)
    
    # Generate chart sections
    chart_divs = ""
    for fname in feature_names:
        chart_divs += f'''
    <div class="chart-container">
        <div class="chart-title">{fname}: Actual vs Predicted</div>
        <canvas id="chart_{fname}"></canvas>
    </div>'''
    
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
    <h1>Earthquake Prediction Viewer (Regression)</h1>
    
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
    {chart_divs}
    
    <script>
        const allData = {data_json};
        const featureNames = {feature_names_json};
        const charts = {{}};
        
        function initCharts() {{
            featureNames.forEach(fname => {{
                const ctx = document.getElementById('chart_' + fname);
                if (ctx) {{
                    charts[fname] = new Chart(ctx, {{
                        type: 'line',
                        data: {{ labels: [], datasets: [] }},
                        options: {{ responsive: true }}
                    }});
                }}
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
            
            const labels = data.map((d, i) => start + i);
            
            featureNames.forEach(fname => {{
                if (charts[fname]) {{
                    charts[fname].data.labels = labels;
                    charts[fname].data.datasets = [
                        {{ label: 'Actual', data: data.map(d => d[fname + '_target']), borderColor: '#333', borderWidth: 2, fill: false, pointRadius: 1 }},
                        {{ label: 'Predicted', data: data.map(d => d[fname + '_pred']), borderColor: '#3498db', borderWidth: 2, fill: false, pointRadius: 1, borderDash: [5, 5] }}
                    ];
                    charts[fname].update();
                }}
            }});
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
    parser = argparse.ArgumentParser(description='ST-GAT Earthquake Prediction')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'ensemble', 'ablation'],
                       help='Mode: train, eval, ensemble, or ablation')
    parser.add_argument('--print_config', action='store_true',
                       help='Print configuration')
    
    args = parser.parse_args()
    main(args)
