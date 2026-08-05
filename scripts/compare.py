# ==============================================================================
# COMPARE_HORIZONS_ALIGNED.PY - Aligned Multi-Horizon Visualization
# ==============================================================================
"""
Visualize predictions with ALIGNED timestamps using EXISTING model.
Uses the multi_horizon_model.pth from compare_mr_mh.py.

NO TRAINING NEEDED - just visualization!

Usage:
    python compare_horizons_aligned.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import copy

warnings.filterwarnings('ignore')

PLOT_FONT_SIZES = {
    'label': 14,
    'title': 16,
    'tick': 12,
    'legend': 12,
    'suptitle': 18,
}

from config import CONFIG, DEVICE
from data.preprocessing import DataPreprocessor
from data.adjacency import AdjacencyBuilder
from data.dataset import SeismicDataset
from torch.utils.data import DataLoader
from models.stgat import STGAT

# ==============================================================================
# SEED SETTING
# ==============================================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# ==============================================================================
# LOAD EXISTING MODEL AND DATA
# ==============================================================================
def load_model_and_data():
    """Load existing multi_horizon_model and prepare data."""
    print("\n" + "=" * 70)
    print(" LOADING EXISTING MODEL")
    print("=" * 70)
    
    filepath = Path(CONFIG['filename'])
    model_path = Path('outputs/comparison_mr_mh/models/multi_horizon_model.pth')
    
    # Config for multi-horizon (same as compare_mr_mh.py)
    mh_config = copy.deepcopy(CONFIG)
    mh_config['horizon'] = 6  # Predicts 6 steps: 4h, 8h, 12h, 16h, 20h, 24h
    mh_config['window_size'] = max(1, int(round(
        mh_config.get('history_hours', 96) /
        (pd.Timedelta(mh_config['time_bin']).total_seconds() / 3600.0)
    )))
    
    # Preprocess
    print("   Loading data...")
    preprocessor = DataPreprocessor(mh_config)
    data = preprocessor.process(filepath)
    
    # Build adjacency
    adj_builder = AdjacencyBuilder(mh_config)
    adj_scipy = adj_builder.build_distance_weighted_adj(
        data['num_nodes'], 
        data['node_info'], 
        data['grid_params'],
        use_distance_weighting=True
    )
    adj_sparse = adj_builder.scipy_to_torch_sparse(adj_scipy, device=DEVICE)
    
    # Get target feature indices
    all_features = mh_config['features']
    target_features = mh_config.get('target_features', all_features)
    target_indices = [all_features.index(f) for f in target_features if f in all_features]
    
    # Create test dataset
    test_dataset = SeismicDataset(
        data['test_data'],
        target_data=data['test_target_data'],
        window_size=mh_config['window_size'],
        horizon=mh_config['horizon'],
    )
    test_loader = DataLoader(test_dataset, batch_size=mh_config['batch_size'], shuffle=False)
    
    # Build and load model
    print("   Building model...")
    model = STGAT(
        num_nodes=data['num_nodes'],
        in_features=data['train_data'].shape[-1],
        hidden_dim=mh_config['hidden_dim'],
        out_features=len(target_indices),
        horizon=mh_config['horizon'],
        num_gat_layers=mh_config['num_gat_layers'],
        num_heads=mh_config['num_heads'],
        dropout=mh_config['dropout'],
        use_attention=True,
        use_multihead=True,
        use_skip=True
    ).to(DEVICE)
    
    print(f"   Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Find max_mw index
    max_mw_idx = target_features.index('max_mw') if 'max_mw' in target_features else 0
    
    print(f"   Num nodes: {data['num_nodes']}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Horizon: 6 steps (4h, 8h, 12h, 16h, 20h, 24h)")
    print(f"   Model loaded successfully!")
    
    return {
        'model': model,
        'test_loader': test_loader,
        'adj_sparse': adj_sparse,
        'num_nodes': data['num_nodes'],
        'max_mw_idx': max_mw_idx,
        'config': mh_config,
        'horizon_hours': [4, 8, 12, 16, 20, 24]  # Actual hours for each horizon step
    }


# ==============================================================================
# PREDICTION AND CUMULATIVE MAX
# ==============================================================================
def predict_with_cumulative_max(data_dict):
    """
    Generate predictions and compute cumulative max.
    
    For each sample, the model predicts 6 horizons: +4h, +8h, +12h, +16h, +20h, +24h
    
    Cumulative max:
    - At +4h:  max(+4h)
    - At +8h:  max(+4h, +8h)  
    - At +12h: max(+4h, +8h, +12h)
    - etc.
    
    This ensures monotonic increase: max(0-24h) >= max(0-12h) >= max(0-4h)
    """
    model = data_dict['model']
    test_loader = data_dict['test_loader']
    adj_sparse = data_dict['adj_sparse']
    max_mw_idx = data_dict['max_mw_idx']
    horizon_hours = data_dict['horizon_hours']
    
    model.eval()
    
    # Results for each cumulative horizon
    results = {h: {'pred_cummax': [], 'target_cummax': [], 'pred_instant': [], 'target_instant': []} 
               for h in horizon_hours}
    
    # Store sample data for visualization
    sample_data = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting")):
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Forward pass - output shape: (B, 6, N, F)
            outputs = model(x, adj_sparse)
            
            # Get max_mw: (B, 6, N)
            pred_mw = outputs[:, :, :, max_mw_idx].cpu().numpy()
            target_mw = y[:, :, :, max_mw_idx].cpu().numpy()
            
            batch_size, n_horizons, n_nodes = pred_mw.shape
            
            for b in range(batch_size):
                for n in range(n_nodes):
                    pred_series = pred_mw[b, :, n]  # (6,)
                    target_series = target_mw[b, :, n]  # (6,)
                    
                    # Compute cumulative max at each horizon
                    for h_idx, h in enumerate(horizon_hours):
                        # Cumulative max from horizon 0 to h_idx (inclusive)
                        pred_cummax = np.max(pred_series[:h_idx+1])
                        target_cummax = np.max(target_series[:h_idx+1])
                        
                        results[h]['pred_cummax'].append(pred_cummax)
                        results[h]['target_cummax'].append(target_cummax)
                        
                        # Instant value at this horizon
                        results[h]['pred_instant'].append(pred_series[h_idx])
                        results[h]['target_instant'].append(target_series[h_idx])
                        
                # Store first few samples for visualization
                if len(sample_data) < 100:
                    sample_data.append({
                        'batch_idx': batch_idx,
                        'sample_idx': b,
                        'pred_series': pred_mw[b, :, :].copy(),  # (6, N)
                        'target_series': target_mw[b, :, :].copy()
                    })
                    
    return results, sample_data


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def visualize_aligned(results, sample_data, data_dict, output_dir):
    """Generate visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    horizon_hours = data_dict['horizon_hours']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # =========================================================================
    # 1. Sample Timeline with Cumulative Max
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    np.random.seed(42)
    sample_indices = np.random.choice(len(sample_data), min(4, len(sample_data)), replace=False)
    
    for idx, (ax, s_idx) in enumerate(zip(axes, sample_indices)):
        sample = sample_data[s_idx]
        node_idx = np.random.randint(0, sample['pred_series'].shape[1])
        
        pred_series = sample['pred_series'][:, node_idx]
        target_series = sample['target_series'][:, node_idx]
        
        # Compute cumulative max
        pred_cummax = [np.max(pred_series[:i+1]) for i in range(len(horizon_hours))]
        target_cummax = [np.max(target_series[:i+1]) for i in range(len(horizon_hours))]
        
        # Plot
        ax.bar(np.array(horizon_hours) - 0.8, pred_series, width=1.5, alpha=0.7, 
               label='Predicted (instant)', color='steelblue', edgecolor='navy')
        ax.bar(np.array(horizon_hours) + 0.8, target_series, width=1.5, alpha=0.7, 
               label='Actual (instant)', color='coral', edgecolor='darkred', hatch='//')
        
        # Cumulative max line
        ax.plot(horizon_hours, pred_cummax, 'o-', color='navy', linewidth=2, 
                markersize=8, label='Predicted (cummax)')
        ax.plot(horizon_hours, target_cummax, 's--', color='darkred', linewidth=2, 
                markersize=8, label='Actual (cummax)')
        
        ax.set_xlabel('Hours Ahead', fontsize=PLOT_FONT_SIZES['label'])
        ax.set_ylabel('Magnitude (Mw)', fontsize=PLOT_FONT_SIZES['label'])
        ax.set_title(f'Sample {s_idx}, Node {node_idx}', fontsize=PLOT_FONT_SIZES['title'])
        ax.legend(fontsize=PLOT_FONT_SIZES['legend'], loc='upper left')
        ax.set_xticks(horizon_hours)
        ax.tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
        
    plt.suptitle('Aligned Multi-Horizon Forecast\n(Cummax should be monotonically increasing)', 
             fontsize=PLOT_FONT_SIZES['suptitle'], fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'aligned_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: aligned_timeline.png")
    
    # =========================================================================
    # 2. Metrics by Horizon (Cumulative Max)
    # =========================================================================
    metrics = []
    for h in horizon_hours:
        pred = np.array(results[h]['pred_cummax'])
        target = np.array(results[h]['target_cummax'])
        
        rmse = np.sqrt(mean_squared_error(target, pred))
        mae = mean_absolute_error(target, pred)
        r2 = r2_score(target, pred)
        
        metrics.append({
            'horizon_hours': h,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_pred': pred.mean(),
            'mean_target': target.mean()
        })
        
    metrics_df = pd.DataFrame(metrics)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = metrics_df['horizon_hours'].values
    
    axes[0].bar(x, metrics_df['rmse'], color='steelblue', width=3)
    axes[0].set_xlabel('Cumulative Horizon (hours)', fontsize=PLOT_FONT_SIZES['label'])
    axes[0].set_ylabel('RMSE', fontsize=PLOT_FONT_SIZES['label'])
    axes[0].set_title('RMSE by Cumulative Horizon', fontsize=PLOT_FONT_SIZES['title'])
    axes[0].set_xticks(x)
    axes[0].tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    
    axes[1].bar(x, metrics_df['r2'], color='coral', width=3)
    axes[1].set_xlabel('Cumulative Horizon (hours)', fontsize=PLOT_FONT_SIZES['label'])
    axes[1].set_ylabel('R²', fontsize=PLOT_FONT_SIZES['label'])
    axes[1].set_title('R² by Cumulative Horizon', fontsize=PLOT_FONT_SIZES['title'])
    axes[1].set_xticks(x)
    axes[1].tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    
    axes[2].bar(x - 1, metrics_df['mean_pred'], width=2, label='Predicted', color='steelblue')
    axes[2].bar(x + 1, metrics_df['mean_target'], width=2, label='Actual', color='coral')
    axes[2].set_xlabel('Cumulative Horizon (hours)', fontsize=PLOT_FONT_SIZES['label'])
    axes[2].set_ylabel('Mean Cumulative Max Magnitude', fontsize=PLOT_FONT_SIZES['label'])
    axes[2].set_title('Mean CumMax (MUST increase with horizon!)', fontsize=PLOT_FONT_SIZES['title'])
    axes[2].legend(fontsize=PLOT_FONT_SIZES['legend'])
    axes[2].set_xticks(x)
    axes[2].tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: cumulative_metrics.png")
    
    # =========================================================================
    # 3. Monotonic Verification
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(metrics_df['horizon_hours'], metrics_df['mean_target'], 'o-', 
            label='Mean Actual (cummax)', color='coral', linewidth=2, markersize=10)
    ax.plot(metrics_df['horizon_hours'], metrics_df['mean_pred'], 's--', 
            label='Mean Predicted (cummax)', color='steelblue', linewidth=2, markersize=10)
    
    ax.set_xlabel('Cumulative Horizon (hours)', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_ylabel('Mean Magnitude (Mw)', fontsize=PLOT_FONT_SIZES['label'])
    ax.set_title('Verification: Cumulative Max is Monotonically Increasing', fontsize=PLOT_FONT_SIZES['title'])
    ax.legend(fontsize=PLOT_FONT_SIZES['legend'])
    ax.grid(True, alpha=0.3)
    ax.set_xticks(horizon_hours)
    ax.tick_params(axis='both', labelsize=PLOT_FONT_SIZES['tick'])
    
    # Verify monotonicity
    target_vals = metrics_df['mean_target'].values
    is_monotonic = all(target_vals[i] <= target_vals[i+1] for i in range(len(target_vals)-1))
    status = "✓ VERIFIED" if is_monotonic else "✗ CHECK DATA"
    color = 'lightgreen' if is_monotonic else 'lightyellow'
    
    ax.text(0.02, 0.98, f"Monotonic: {status}", transform=ax.transAxes, 
            fontsize=PLOT_FONT_SIZES['label'], verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=color))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'monotonic_verification.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: monotonic_verification.png")
    
    # Save metrics
    metrics_df.to_csv(output_dir / 'aligned_metrics.csv', index=False)
    print(f"   Saved: aligned_metrics.csv")
    
    print("\n" + "=" * 70)
    print(" METRICS SUMMARY (Cumulative Max)")
    print("=" * 70)
    print(metrics_df.to_string(index=False))
    
    return metrics_df


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("\n" + "=" * 70)
    print(" ALIGNED MULTI-HORIZON VISUALIZATION")
    print(" (No training needed - using existing model)")
    print("=" * 70)
    
    output_dir = Path('outputs/aligned_comparison')
    
    # Load existing model
    data_dict = load_model_and_data()
    
    # Predict with cumulative max
    print("\n" + "=" * 70)
    print(" GENERATING PREDICTIONS")
    print("=" * 70)
    results, sample_data = predict_with_cumulative_max(data_dict)
    
    # Visualize
    print("\n" + "=" * 70)
    print(" GENERATING VISUALIZATIONS")
    print("=" * 70)
    visualize_aligned(results, sample_data, data_dict, output_dir)
    
    print("\n" + "=" * 70)
    print(" COMPLETE!")
    print(f" Output saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
