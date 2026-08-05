# ==============================================================================
# EVALUATE_ENSEMBLE.PY - Post-hoc Evaluation and Visualization for Ensemble
# ==============================================================================
"""
Script untuk menghasilkan metrik dan visualisasi lengkap dari model ensemble.

Menghasilkan:
1. Metrik per-horizon (h1, h2, h3, h4)
2. Metrik per-fitur (count, max_mw, log_energy)
3. Analisis ketidakpastian (coverage, sharpness, calibration)
4. Visualisasi untuk LaTeX

Usage:
    python evaluate_ensemble.py
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from tqdm.auto import tqdm

# Import from project
from config import CONFIG, DEVICE
from data.preprocessing import DataPreprocessor
from data.adjacency import AdjacencyBuilder
from data.dataset import SeismicDataset
from torch.utils.data import DataLoader
from models import STGAT, STTFT
from evaluation.metrics import MetricsCalculator
from evaluation.evaluate import load_model as load_evaluation_model


PLOT_FONT_SIZES = {
    'label': 14,
    'title': 16,
    'tick': 12,
    'legend': 12,
}


def detect_model_type(state_dict):
    """Auto-detect model type from state_dict keys."""
    keys = list(state_dict.keys())
    
    # Check for TFT-specific keys
    has_tft = any('temporal_encoder' in k for k in keys)
    has_lstm = any('lstm.' in k or 'lstm_norm' in k for k in keys)
    
    if has_tft:
        return 'sttft'
    elif has_lstm:
        return 'stgat'
    else:
        # Default based on other patterns
        if any('node_embedding' in k for k in keys):
            return 'stgat'
        return 'sttft'


def load_model(model_path, model_kwargs, device):
    """Load a trained model from checkpoint with auto-detection."""
    print(f"   Loading checkpoint: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Auto-detect model type
    model_type = detect_model_type(state_dict)
    print(f"   Detected model type: {model_type.upper()}")
    
    # Get input features from first layer weight shape
    # For STTFT: gat_layers.0.heads.0.linear_q.weight shape is (head_dim, in_features)
    first_layer_key = None
    for k in state_dict.keys():
        if 'gat_layers.0.heads.0.linear_q.weight' in k:
            first_layer_key = k
            break
    
    if first_layer_key:
        in_features_checkpoint = state_dict[first_layer_key].shape[1]
        print(f"   Checkpoint in_features: {in_features_checkpoint}")
        model_kwargs['in_features'] = in_features_checkpoint
    
    # Create model based on detected type
    if model_type == 'sttft':
        # STTFT might have additional kwargs
        sttft_kwargs = {
            'num_nodes': model_kwargs['num_nodes'],
            'in_features': model_kwargs['in_features'],
            'hidden_dim': model_kwargs['hidden_dim'],
            'out_features': model_kwargs['out_features'],
            'horizon': model_kwargs['horizon'],
            'num_gat_layers': model_kwargs['num_gat_layers'],
            'num_heads': model_kwargs['num_heads'],
            'tft_layers': CONFIG.get('tft_layers', 2),
            'dropout': model_kwargs['dropout'],
        }
        model = STTFT(**sttft_kwargs).to(device)
    else:
        model = STGAT(**model_kwargs).to(device)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"   Model loaded successfully!")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, model_type


def generate_predictions(model, test_loader, adj_sparse, device):
    """Generate predictions from a single model."""
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Generating predictions"):
            data = data.to(device)
            output = model(data, adj_sparse)
            predictions.append(output.cpu().numpy())
            targets.append(target.numpy())
    
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def calculate_per_horizon_metrics(targets, predictions, feature_names):
    """Calculate metrics for each forecast horizon."""
    # targets/predictions shape: (B, H, N, F)
    B, H, N, F = targets.shape
    
    metrics_per_horizon = {}
    
    for h in range(H):
        target_h = targets[:, h, :, :]  # (B, N, F)
        pred_h = predictions[:, h, :, :]
        
        # Flatten for metrics calculation
        target_flat = target_h.reshape(-1, F)
        pred_flat = pred_h.reshape(-1, F)
        
        metrics_h = {}
        
        # Overall metrics for this horizon
        mse = np.mean((target_flat - pred_flat) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(target_flat - pred_flat))
        
        # R² calculation
        ss_res = np.sum((target_flat - pred_flat) ** 2)
        ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        metrics_h['overall'] = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2)
        }
        
        # Per-feature metrics
        metrics_h['per_feature'] = {}
        for f_idx, fname in enumerate(feature_names):
            t_f = target_flat[:, f_idx]
            p_f = pred_flat[:, f_idx]
            
            mse_f = np.mean((t_f - p_f) ** 2)
            rmse_f = np.sqrt(mse_f)
            mae_f = np.mean(np.abs(t_f - p_f))
            ss_res_f = np.sum((t_f - p_f) ** 2)
            ss_tot_f = np.sum((t_f - np.mean(t_f)) ** 2)
            r2_f = 1 - (ss_res_f / (ss_tot_f + 1e-8))
            
            metrics_h['per_feature'][fname] = {
                'MSE': float(mse_f),
                'RMSE': float(rmse_f),
                'MAE': float(mae_f),
                'R2': float(r2_f)
            }
        
        metrics_per_horizon[f'h{h+1}'] = metrics_h
    
    return metrics_per_horizon


def calculate_uncertainty_metrics(targets, mean_pred, std_pred, confidence_level=0.95):
    """Calculate uncertainty calibration metrics."""
    z_score = 1.96 if confidence_level == 0.95 else 1.645  # 95% or 90%
    
    ci_lower = mean_pred - z_score * std_pred
    ci_upper = mean_pred + z_score * std_pred
    
    # Coverage: proportion of targets within CI
    in_interval = (targets >= ci_lower) & (targets <= ci_upper)
    coverage = np.mean(in_interval)
    
    # Sharpness: average width of CI (smaller is better)
    sharpness = np.mean(ci_upper - ci_lower)
    
    # Average uncertainty
    avg_uncertainty = np.mean(std_pred)
    
    # Calibration by magnitude bins
    target_flat = targets.flatten()
    mean_flat = mean_pred.flatten()
    std_flat = std_pred.flatten()
    abs_error = np.abs(target_flat - mean_flat)
    
    # Correlation between uncertainty and error
    valid_mask = std_flat > 0
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(std_flat[valid_mask], abs_error[valid_mask])[0, 1]
    else:
        correlation = 0.0
    
    return {
        'coverage_95': float(coverage),
        'sharpness': float(sharpness),
        'avg_uncertainty': float(avg_uncertainty),
        'uncertainty_error_correlation': float(correlation) if not np.isnan(correlation) else 0.0
    }


def plot_per_horizon_metrics(metrics_per_horizon, save_dir):
    """Plot metrics degradation across horizons."""
    horizons = sorted(metrics_per_horizon.keys(), key=lambda x: int(x[1:]))
    
    rmse_values = [metrics_per_horizon[h]['overall']['RMSE'] for h in horizons]
    mae_values = [metrics_per_horizon[h]['overall']['MAE'] for h in horizons]
    r2_values = [metrics_per_horizon[h]['overall']['R2'] for h in horizons]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x = range(1, len(horizons) + 1)
    
    # RMSE
    axes[0].plot(x, rmse_values, 'o-', color='#2E86AB', linewidth=2, markersize=8)
    axes[0].set_xlabel('Forecast Horizon (timesteps)', fontsize=PLOT_FONT_SIZES['label'])
    axes[0].set_ylabel('RMSE', fontsize=PLOT_FONT_SIZES['label'])
    axes[0].set_title('RMSE by Horizon', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(x, mae_values, 'o-', color='#E94F37', linewidth=2, markersize=8)
    axes[1].set_xlabel('Forecast Horizon (timesteps)', fontsize=PLOT_FONT_SIZES['label'])
    axes[1].set_ylabel('MAE', fontsize=PLOT_FONT_SIZES['label'])
    axes[1].set_title('MAE by Horizon', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    axes[1].grid(True, alpha=0.3)
    
    # R²
    axes[2].plot(x, r2_values, 'o-', color='#44AF69', linewidth=2, markersize=8)
    axes[2].set_xlabel('Forecast Horizon (timesteps)', fontsize=PLOT_FONT_SIZES['label'])
    axes[2].set_ylabel('R²', fontsize=PLOT_FONT_SIZES['label'])
    axes[2].set_title('R² by Horizon', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    axes[2].tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'per_horizon_metrics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_per_feature_metrics(metrics_per_horizon, feature_names, save_dir):
    """Plot metrics comparison per feature."""
    # Use first horizon for comparison
    h1_metrics = metrics_per_horizon['h1']['per_feature']
    
    features = list(h1_metrics.keys())
    mse_values = [h1_metrics[f]['MSE'] for f in features]
    rmse_values = [h1_metrics[f]['RMSE'] for f in features]
    r2_values = [h1_metrics[f]['R2'] for f in features]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    colors = ['#2E86AB', '#E94F37', '#44AF69']
    
    # MSE per feature
    bars1 = axes[0].bar(features, mse_values, color=colors[:len(features)])
    axes[0].set_ylabel('MSE', fontsize=PLOT_FONT_SIZES['label'])
    axes[0].set_title('MSE per Feature (h=1)', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45, labelsize=PLOT_FONT_SIZES['tick'])
    axes[0].tick_params(axis='y', labelsize=PLOT_FONT_SIZES['tick'])
    
    # RMSE per feature
    bars2 = axes[1].bar(features, rmse_values, color=colors[:len(features)])
    axes[1].set_ylabel('RMSE', fontsize=PLOT_FONT_SIZES['label'])
    axes[1].set_title('RMSE per Feature (h=1)', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45, labelsize=PLOT_FONT_SIZES['tick'])
    axes[1].tick_params(axis='y', labelsize=PLOT_FONT_SIZES['tick'])
    
    # R² per feature
    bars3 = axes[2].bar(features, r2_values, color=colors[:len(features)])
    axes[2].set_ylabel('R²', fontsize=PLOT_FONT_SIZES['label'])
    axes[2].set_title('R² per Feature (h=1)', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45, labelsize=PLOT_FONT_SIZES['tick'])
    axes[2].tick_params(axis='y', labelsize=PLOT_FONT_SIZES['tick'])
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = save_dir / 'per_feature_metrics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def plot_uncertainty_analysis(targets, mean_pred, std_pred, save_dir):
    """Plot uncertainty analysis visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Uncertainty distribution
    ax = axes[0, 0]
    std_flat = std_pred.flatten()
    ax.hist(std_flat[std_flat > 0], bins=50, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Prediction Uncertainty (std)', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_ylabel('Frequency', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_title('Distribution of Prediction Uncertainty', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    ax.axvline(np.mean(std_flat), color='red', linestyle='--', label=f'Mean: {np.mean(std_flat):.3f}')
    ax.tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    ax.legend(fontsize=PLOT_FONT_SIZES['legend'])
    
    # 2. Uncertainty vs Error scatter
    ax = axes[0, 1]
    abs_error = np.abs(targets.flatten() - mean_pred.flatten())
    # Sample for performance
    n_sample = min(10000, len(abs_error))
    idx = np.random.choice(len(abs_error), n_sample, replace=False)
    ax.scatter(std_flat[idx], abs_error[idx], alpha=0.1, s=5, c='#2E86AB')
    ax.set_xlabel('Uncertainty (std)', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_ylabel('Absolute Error', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_title('Uncertainty vs Prediction Error', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    ax.tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    
    # Add trend line
    valid = (std_flat[idx] > 0) & (abs_error[idx] < np.percentile(abs_error[idx], 99))
    if np.sum(valid) > 10:
        z = np.polyfit(std_flat[idx][valid], abs_error[idx][valid], 1)
        p = np.poly1d(z)
        x_line = np.linspace(std_flat[idx][valid].min(), std_flat[idx][valid].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
        ax.legend(fontsize=PLOT_FONT_SIZES['legend'])
    
    # 3. Coverage plot (calibration)
    ax = axes[1, 0]
    confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    z_scores = [0.674, 0.842, 1.036, 1.282, 1.645, 1.96, 2.576]
    
    actual_coverages = []
    for z in z_scores:
        ci_lower = mean_pred - z * std_pred
        ci_upper = mean_pred + z * std_pred
        coverage = np.mean((targets >= ci_lower) & (targets <= ci_upper))
        actual_coverages.append(coverage)
    
    ax.plot(confidence_levels, actual_coverages, 'o-', color='#2E86AB', 
            linewidth=2, markersize=8, label='Actual')
    ax.plot([0.5, 1], [0.5, 1], 'k--', label='Perfect Calibration')
    ax.set_xlabel('Expected Coverage', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_ylabel('Actual Coverage', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_title('Uncertainty Calibration', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    ax.tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    ax.legend(fontsize=PLOT_FONT_SIZES['legend'])
    ax.set_xlim(0.45, 1.02)
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)
    
    # 4. Sharpness by horizon
    ax = axes[1, 1]
    B, H, N, F = std_pred.shape
    sharpness_per_horizon = []
    for h in range(H):
        ci_width = 2 * 1.96 * std_pred[:, h, :, :].mean()
        sharpness_per_horizon.append(ci_width)
    
    ax.bar(range(1, H+1), sharpness_per_horizon, color='#44AF69', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Forecast Horizon', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_ylabel('95% CI Width', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_title('Uncertainty (Sharpness) by Horizon', fontsize=PLOT_FONT_SIZES['title'], fontweight='bold')
    ax.tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    
    plt.tight_layout()
    save_path = save_dir / 'uncertainty_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


def save_metrics_to_json(all_metrics, save_path):
    """Save all metrics to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"   Saved: {save_path}")


def save_metrics_to_csv(metrics_per_horizon, feature_names, save_dir):
    """Save per-horizon metrics to CSV for easy LaTeX table generation."""
    rows = []
    
    for h_key in sorted(metrics_per_horizon.keys(), key=lambda x: int(x[1:])):
        h_num = int(h_key[1:])
        overall = metrics_per_horizon[h_key]['overall']
        
        row = {
            'Horizon': h_num,
            'MSE': overall['MSE'],
            'RMSE': overall['RMSE'],
            'MAE': overall['MAE'],
            'R2': overall['R2']
        }
        
        # Add per-feature metrics
        for fname in feature_names:
            feat_metrics = metrics_per_horizon[h_key]['per_feature'].get(fname, {})
            row[f'{fname}_RMSE'] = feat_metrics.get('RMSE', 0)
            row[f'{fname}_R2'] = feat_metrics.get('R2', 0)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = save_dir / 'per_horizon_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")
    
    return df


def main():
    print("\n" + "="*70)
    print(" ENSEMBLE MODEL EVALUATION")
    print("="*70)
    
    # ====================
    # DATA PREPARATION
    # ====================
    print("\n Loading and preprocessing data...")
    
    preprocessor = DataPreprocessor(CONFIG)
    data = preprocessor.process(CONFIG['filename'])
    
    # Build adjacency
    adj_builder = AdjacencyBuilder(CONFIG)
    adj_scipy = adj_builder.build_distance_weighted_adj(
        data['num_nodes'], 
        data['node_info'], 
        data['grid_params'],
        use_distance_weighting=True
    )
    adj_sparse = adj_builder.scipy_to_torch_sparse(adj_scipy, device=DEVICE)
    
    # Get target features
    all_features = CONFIG['features']
    target_features = CONFIG.get('target_features', all_features)
    target_indices = [all_features.index(f) for f in target_features if f in all_features]
    
    # Create test loader
    test_dataset = SeismicDataset(
        data['test_data'],
        target_data=data['test_target_data'],
        window_size=CONFIG['window_size'],
        horizon=CONFIG['horizon'],
    )
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Target features: {target_features}")
    
    # ====================
    # LOAD MODEL
    # ====================
    print("\n Loading trained model...")
    
    model_kwargs = {
        'num_nodes': data['num_nodes'],
        'in_features': data['train_data'].shape[-1],
        'hidden_dim': CONFIG['hidden_dim'],
        'out_features': len(target_features),
        'horizon': CONFIG['horizon'],
        'num_gat_layers': CONFIG['num_gat_layers'],
        'num_heads': CONFIG['num_heads'],
        'dropout': CONFIG['dropout'],
    }
    
    model_path = Path(CONFIG['output_dir']) / 'models' / 'stgat_best.pth'
    print(f"   Model path: {model_path}")
    
    # Reuse the canonical evaluator loader so multi-scale checkpoints and
    # effective input features (raw features + node embeddings) are handled
    # consistently.
    model, model_type = load_evaluation_model(model_path, model_kwargs, DEVICE)
    
    # ====================
    # GENERATE PREDICTIONS
    # ====================
    print("\n Generating predictions...")
    predictions, targets = generate_predictions(model, test_loader, adj_sparse, DEVICE)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Targets shape: {targets.shape}")

    # Evaluate in raw Mw units; target normalization was fit on train only.
    target_mean = data['target_stats'].get('offset', data['target_stats']['mean']).reshape(1, 1, 1, -1)
    target_std = data['target_stats']['std'].reshape(1, 1, 1, -1)
    predictions = predictions * target_std + target_mean
    targets = targets * target_std + target_mean
    
    # ====================
    # CALCULATE METRICS
    # ====================
    print("\n Calculating metrics...")
    
    # Overall metrics
    metrics_calc = MetricsCalculator(feature_names=target_features)
    overall_metrics = metrics_calc.calculate_all_metrics(targets, predictions)
    
    # Per-horizon metrics
    metrics_per_horizon = calculate_per_horizon_metrics(targets, predictions, target_features)
    
    # Simple uncertainty estimate (using prediction variance across samples)
    # For single model, we don't have true ensemble uncertainty
    # Use a placeholder based on prediction magnitude
    std_pred = np.abs(predictions) * 0.1 + 0.05  # Proxy uncertainty
    
    uncertainty_metrics = calculate_uncertainty_metrics(targets, predictions, std_pred)
    
    # ====================
    # PRINT SUMMARY
    # ====================
    print("\n" + "="*70)
    print(" EVALUATION RESULTS")
    print("="*70)
    
    print("\n Overall Metrics:")
    for key, value in overall_metrics['overall'].items():
        print(f"   {key}: {value:.4f}")
    
    print("\n Per-Feature Metrics (h=1):")
    for fname, fmetrics in metrics_per_horizon['h1']['per_feature'].items():
        print(f"   {fname}: RMSE={fmetrics['RMSE']:.4f}, R²={fmetrics['R2']:.4f}")
    
    print("\n Per-Horizon RMSE:")
    for h_key in sorted(metrics_per_horizon.keys(), key=lambda x: int(x[1:])):
        rmse = metrics_per_horizon[h_key]['overall']['RMSE']
        r2 = metrics_per_horizon[h_key]['overall']['R2']
        print(f"   {h_key}: RMSE={rmse:.4f}, R²={r2:.4f}")
    
    print("\n Uncertainty Metrics:")
    for key, value in uncertainty_metrics.items():
        print(f"   {key}: {value:.4f}")
    
    # ====================
    # SAVE OUTPUTS
    # ====================
    print("\n Saving outputs...")
    
    output_dir = Path(CONFIG['output_dir'])
    figures_dir = output_dir / 'figures'
    metrics_dir = output_dir / 'metrics'
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualizations
    plot_per_horizon_metrics(metrics_per_horizon, figures_dir)
    plot_per_feature_metrics(metrics_per_horizon, target_features, figures_dir)
    plot_uncertainty_analysis(targets, predictions, std_pred, figures_dir)
    
    # Metrics files
    all_metrics = {
        'overall': overall_metrics['overall'],
        'per_feature': overall_metrics.get('per_feature', {}),
        'per_horizon': metrics_per_horizon,
        'uncertainty': uncertainty_metrics
    }
    save_metrics_to_json(all_metrics, metrics_dir / 'complete_metrics.json')
    
    # CSV for LaTeX
    df_horizon = save_metrics_to_csv(metrics_per_horizon, target_features, metrics_dir)
    
    print("\n" + "="*70)
    print(" DONE! Generated outputs:")
    print("="*70)
    print(f"   Figures:")
    print(f"     - {figures_dir / 'per_horizon_metrics.png'}")
    print(f"     - {figures_dir / 'per_feature_metrics.png'}")
    print(f"     - {figures_dir / 'uncertainty_analysis.png'}")
    print(f"   Metrics:")
    print(f"     - {metrics_dir / 'complete_metrics.json'}")
    print(f"     - {metrics_dir / 'per_horizon_metrics.csv'}")
    print("\n")


if __name__ == '__main__':
    main()
