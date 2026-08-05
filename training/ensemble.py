# ==============================================================================
# DEEP_ENSEMBLE_MULTIRESOLUTION.PY - Deep Ensemble for All Resolutions
# ==============================================================================
"""
Train and evaluate Deep Ensembles for uncertainty estimation across all 
temporal resolutions (4h, 8h, 12h, 16h, 20h, 24h).

Usage:
    python deep_ensemble_multiresolution.py --mode train    # Train all ensembles
    python deep_ensemble_multiresolution.py --mode evaluate # Evaluate and visualize
    python deep_ensemble_multiresolution.py --mode all      # Train + Evaluate
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
import copy
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

warnings.filterwarnings('ignore')

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
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# MULTI-RESOLUTION DEEP ENSEMBLE
# ==============================================================================
class MultiResolutionDeepEnsemble:
    """
    Deep Ensemble trained separately for each temporal resolution.
    
    For each resolution (4h, 8h, ..., 24h):
        - Train N_MODELS models with different random seeds
        - Generate predictions with uncertainty estimates
    """
    
    def __init__(self, resolutions, base_config, device, n_models=5):
        """
        Args:
            resolutions: List of time_bin strings, e.g., ['4h', '8h', '12h', ...]
            base_config: Base configuration dictionary
            device: torch device
            n_models: Number of models per ensemble (default 5)
        """
        self.resolutions = resolutions
        self.base_config = base_config
        self.device = device
        self.n_models = n_models
        
        # Storage for each resolution
        self.ensembles = {}  # {resolution: [model1, model2, ...]}
        self.data_cache = {}
        self.adj_cache = {}
        self.canonical_split_timestamps = None
        
        # Random seeds for reproducibility
        self.seeds = [42, 123, 456, 789, 1024][:n_models]
        
    def prepare_data(self, filepath, resolution):
        """Prepare data for a specific resolution."""
        res_config = copy.deepcopy(self.base_config)
        res_config['time_bin'] = resolution
        res_config['horizon'] = 1  # Single step prediction
        resolution_hours = pd.Timedelta(resolution).total_seconds() / 3600.0
        res_config['window_size'] = max(1, int(round(
            res_config.get('history_hours', 96) / resolution_hours
        )))
        if self.canonical_split_timestamps is not None:
            res_config['split_timestamps'] = self.canonical_split_timestamps
        
        # Preprocess
        preprocessor = DataPreprocessor(res_config)
        data = preprocessor.process(filepath)
        if self.canonical_split_timestamps is None:
            self.canonical_split_timestamps = data['split_timestamps']
        
        # Build adjacency
        adj_builder = AdjacencyBuilder(res_config)
        adj_scipy = adj_builder.build_distance_weighted_adj(
            data['num_nodes'], 
            data['node_info'], 
            data['grid_params'],
            use_distance_weighting=True
        )
        adj_sparse = adj_builder.scipy_to_torch_sparse(adj_scipy, device=self.device)

        from utils.manifest import write_run_manifest
        write_run_manifest(
            Path(self.base_config.get('output_dir', 'outputs')) /
            'deep_ensemble_multiresolution' / resolution,
            res_config, data=data, data_path=filepath,
            adjacency=adj_scipy, stage=f'{resolution}_preprocessing_complete'
        )
        
        # Get target feature indices
        all_features = res_config['features']
        target_features = res_config.get('target_features', all_features)
        target_indices = [all_features.index(f) for f in target_features if f in all_features]
        
        # Create datasets
        train_dataset = SeismicDataset(
            data['train_data'],
            target_data=data['train_target_data'],
            window_size=res_config['window_size'],
            horizon=res_config['horizon'],
        )
        val_dataset = SeismicDataset(
            data['val_data'],
            target_data=data['val_target_data'],
            window_size=res_config['window_size'],
            horizon=res_config['horizon'],
        )
        test_dataset = SeismicDataset(
            data['test_data'],
            target_data=data['test_target_data'],
            window_size=res_config['window_size'],
            horizon=res_config['horizon'],
        )
        
        loader_generator = torch.Generator()
        loader_generator.manual_seed(res_config.get('seed', 42))
        self.data_cache[resolution] = {
            'train_loader': DataLoader(
                train_dataset, batch_size=res_config['batch_size'],
                shuffle=True, generator=loader_generator
            ),
            'val_loader': DataLoader(val_dataset, batch_size=res_config['batch_size'], shuffle=False),
            'test_loader': DataLoader(test_dataset, batch_size=res_config['batch_size'], shuffle=False),
            'num_nodes': data['num_nodes'],
            'in_features': data['train_data'].shape[-1],
            'n_targets': len(target_indices),
            'config': res_config,
            'feature_stats': data['feature_stats'],
            'target_stats': data['target_stats'],
            'split_timestamps': data['split_timestamps']
        }
        self.adj_cache[resolution] = adj_sparse
        
        return self.data_cache[resolution]
    
    def build_ensemble(self, resolution):
        """Build ensemble of models for a resolution."""
        data = self.data_cache[resolution]
        models = []
        
        for seed in self.seeds:
            set_seed(seed)
            model = STGAT(
                num_nodes=data['num_nodes'],
                in_features=data['in_features'],
                hidden_dim=self.base_config['hidden_dim'],
                out_features=data['n_targets'],
                horizon=1,
                num_gat_layers=self.base_config['num_gat_layers'],
                num_heads=self.base_config['num_heads'],
                dropout=self.base_config['dropout'],
                use_attention=True,
                use_multihead=True,
                use_skip=True
            ).to(self.device)
            models.append(model)
            
        self.ensembles[resolution] = models
        print(f"   Built {self.n_models} models for {resolution}")
        
    def train_ensemble(self, resolution, output_dir, epochs=50, patience=7):
        """Train all models in an ensemble for a resolution."""
        output_dir = Path(output_dir) / resolution
        output_dir.mkdir(parents=True, exist_ok=True)
        
        models = self.ensembles[resolution]
        data = self.data_cache[resolution]
        adj_sparse = self.adj_cache[resolution]
        
        for i, (model, seed) in enumerate(zip(models, self.seeds)):
            print(f"\n   Training model {i+1}/{self.n_models} (seed={seed})...")
            set_seed(seed)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.base_config['learning_rate'])
            criterion = torch.nn.MSELoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for batch in data['train_loader']:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(x, adj_sparse)
                    loss = criterion(outputs, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                train_loss /= len(data['train_loader'])
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in data['val_loader']:
                        x, y = batch
                        x = x.to(self.device)
                        y = y.to(self.device)
                        
                        outputs = model(x, adj_sparse)
                        loss = criterion(outputs, y)
                        val_loss += loss.item()
                        
                val_loss /= len(data['val_loader'])
                
                if (epoch + 1) % 10 == 0:
                    print(f"      Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), output_dir / f'ensemble_model_{i}.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"      Early stopping at epoch {epoch+1}")
                        break
                        
            # Load best model
            model.load_state_dict(torch.load(output_dir / f'ensemble_model_{i}.pth'))
            
    def load_ensemble(self, resolution, model_dir):
        """Load pre-trained ensemble for a resolution."""
        model_dir = Path(model_dir) / resolution
        
        for i, model in enumerate(self.ensembles[resolution]):
            model_path = model_dir / f'ensemble_model_{i}.pth'
            if model_path.exists():
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
            else:
                print(f"   WARNING: {model_path} not found!")
                
        print(f"   Loaded ensemble for {resolution}")
        
    def predict_with_uncertainty(self, resolution, split='test'):
        """
        Generate predictions with uncertainty for a resolution.
        
        Returns:
            dict with mean, std, ci_lower, ci_upper, targets
        """
        models = self.ensembles[resolution]
        data = self.data_cache[resolution]
        adj_sparse = self.adj_cache[resolution]
        
        # Find max_mw index
        res_config = data['config']
        target_features = res_config.get('target_features', res_config['features'])
        max_mw_idx = target_features.index('max_mw') if 'max_mw' in target_features else 0
        
        loader = data[f'{split}_loader']
        all_predictions = []  # (n_models, n_samples)
        all_targets = []
        
        for model in models:
            model.eval()
            model_preds = []
            model_targets = []
            
            with torch.no_grad():
                for batch in loader:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    outputs = model(x, adj_sparse)
                    
                    # Output shape: (B, 1, N, F)
                    pred = outputs[:, 0, :, max_mw_idx].cpu().numpy()
                    target = y[:, 0, :, max_mw_idx].cpu().numpy()
                    
                    model_preds.extend(pred.flatten().tolist())
                    model_targets.extend(target.flatten().tolist())
                    
            all_predictions.append(model_preds)
            if len(all_targets) == 0:
                all_targets = model_targets
                
        all_predictions = np.array(all_predictions)  # (n_models, n_samples)
        all_targets = np.array(all_targets)
        
        # Compute statistics
        mean_pred = all_predictions.mean(axis=0)
        std_pred = all_predictions.std(axis=0)
        ci_lower = mean_pred - 1.96 * std_pred
        ci_upper = mean_pred + 1.96 * std_pred
        
        target_stats = data['target_stats']
        target_mean = float(target_stats.get('offset', target_stats['mean'])[0])
        target_std = float(target_stats['std'][0])
        all_predictions = all_predictions * target_std + target_mean
        all_targets = all_targets * target_std + target_mean
        mean_pred = all_predictions.mean(axis=0)
        std_pred = all_predictions.std(axis=0)
        ci_lower = mean_pred - 1.96 * std_pred
        ci_upper = mean_pred + 1.96 * std_pred

        return {
            'mean': mean_pred,
            'std': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'targets': all_targets,
            'all_predictions': all_predictions
        }


# ==============================================================================
# CALIBRATION
# ==============================================================================
def calibrate_uncertainty(validation_results, test_results, target_coverage=95):
    """
    Calibrate uncertainty estimates using empirical scaling.
    
    Finds the optimal scaling factor such that the CI achieves target coverage.
    This is a post-hoc calibration technique similar to temperature scaling.
    
    Args:
        validation_results: dict with validation mean, std, targets
        test_results: dict with held-out test mean, std, targets
        target_coverage: target coverage percentage (default 95%)
        
    Returns:
        calibrated_results: dict with calibrated uncertainty
        calibration_factor: the scaling factor used
    """
    # Calibration factor is estimated from validation predictions only and
    # then frozen before the test set is inspected.
    mean = validation_results['mean']
    std = validation_results['std']
    targets = validation_results['targets']
    
    # Binary search for optimal scaling factor
    low, high = 0.1, 20.0
    target_frac = target_coverage / 100.0
    
    for _ in range(50):  # Binary search iterations
        mid = (low + high) / 2
        scaled_std = std * mid
        ci_lower = mean - 1.96 * scaled_std
        ci_upper = mean + 1.96 * scaled_std
        
        in_ci = (targets >= ci_lower) & (targets <= ci_upper)
        coverage = in_ci.mean()
        
        if coverage < target_frac:
            low = mid
        else:
            high = mid
            
        if abs(coverage - target_frac) < 0.001:
            break
    
    calibration_factor = mid
    calibrated_std = test_results['std'] * calibration_factor
    test_mean = test_results['mean']
    
    return {
        'mean': test_mean,
        'std': calibrated_std,
        'std_raw': test_results['std'],
        'ci_lower': test_mean - 1.96 * calibrated_std,
        'ci_upper': test_mean + 1.96 * calibrated_std,
        'targets': test_results['targets'],
        'all_predictions': test_results.get('all_predictions', None),
        'calibration_factor': calibration_factor
    }, calibration_factor


# ==============================================================================
# EVALUATION METRICS
# ==============================================================================
def compute_uncertainty_metrics(results, use_calibrated=True):
    """Compute uncertainty-specific metrics."""
    mean = results['mean']
    std = results['std']
    targets = results['targets']
    ci_lower = results['ci_lower']
    ci_upper = results['ci_upper']
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(targets, mean))
    mae = mean_absolute_error(targets, mean)
    r2 = r2_score(targets, mean)
    
    # Coverage: % of targets within 95% CI
    in_ci = (targets >= ci_lower) & (targets <= ci_upper)
    coverage = in_ci.mean() * 100
    
    # Sharpness: average width of CI (narrower = better)
    sharpness = (ci_upper - ci_lower).mean()
    
    # Mean uncertainty
    mean_std = std.mean()
    
    # Correlation between uncertainty and error
    errors = np.abs(targets - mean)
    # Use raw std for correlation if available (calibration doesn't affect correlation)
    std_for_corr = results.get('std_raw', std)
    corr = np.corrcoef(std_for_corr, errors)[0, 1] if len(std_for_corr) > 1 else 0
    
    # Calibration factor if available
    cal_factor = results.get('calibration_factor', 1.0)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'coverage': coverage,
        'sharpness': sharpness,
        'mean_uncertainty': mean_std,
        'uncertainty_error_corr': corr,
        'calibration_factor': cal_factor
    }


# ==============================================================================
# VISUALIZATION
# ==============================================================================
def visualize_uncertainty_analysis(all_results, output_dir):
    """Generate comprehensive uncertainty visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    resolutions = list(all_results.keys())
    
    # =========================================================================
    # 1. Metrics Summary Table
    # =========================================================================
    metrics_data = []
    for res in resolutions:
        metrics = compute_uncertainty_metrics(all_results[res])
        metrics['resolution'] = res
        metrics_data.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(output_dir / 'uncertainty_metrics.csv', index=False)
    
    # =========================================================================
    # 2. Coverage by Resolution
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Coverage bar chart
    coverages = [compute_uncertainty_metrics(all_results[r])['coverage'] for r in resolutions]
    colors = ['green' if 93 <= c <= 97 else 'orange' if 90 <= c <= 99 else 'red' for c in coverages]
    axes[0].bar(resolutions, coverages, color=colors)
    axes[0].axhline(y=95, color='red', linestyle='--', label='Target (95%)')
    axes[0].set_xlabel('Resolution')
    axes[0].set_ylabel('Coverage (%)')
    axes[0].set_title('95% CI Coverage by Resolution')
    axes[0].legend()
    axes[0].set_ylim([80, 100])
    
    # Sharpness
    sharpnesses = [compute_uncertainty_metrics(all_results[r])['sharpness'] for r in resolutions]
    axes[1].bar(resolutions, sharpnesses, color='steelblue')
    axes[1].set_xlabel('Resolution')
    axes[1].set_ylabel('Sharpness (CI Width)')
    axes[1].set_title('Prediction Interval Sharpness')
    
    # Mean uncertainty
    uncertainties = [compute_uncertainty_metrics(all_results[r])['mean_uncertainty'] for r in resolutions]
    axes[2].bar(resolutions, uncertainties, color='coral')
    axes[2].set_xlabel('Resolution')
    axes[2].set_ylabel('Mean Std Dev')
    axes[2].set_title('Average Prediction Uncertainty')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_by_resolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 3. Comprehensive Uncertainty Analysis (2x2 grid)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    
    # a) Uncertainty Distribution (all resolutions stacked)
    ax = axes[0, 0]
    for res in resolutions:
        ax.hist(all_results[res]['std'], bins=50, alpha=0.5, label=res)
    ax.set_xlabel('Uncertainty (Std Dev)')
    ax.set_ylabel('Frequency')
    ax.set_title('Uncertainty Distribution by Resolution')
    ax.legend()
    
    # b) Uncertainty vs Error Scatter
    ax = axes[0, 1]
    for res in resolutions[:3]:  # Plot first 3 for clarity
        errors = np.abs(all_results[res]['targets'] - all_results[res]['mean'])
        sample_idx = np.random.choice(len(errors), min(500, len(errors)), replace=False)
        ax.scatter(all_results[res]['std'][sample_idx], errors[sample_idx], 
                   alpha=0.3, s=10, label=res)
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Uncertainty vs Error Correlation')
    ax.legend()
    
    # c) Calibration Plot
    ax = axes[1, 0]
    expected_coverages = [50, 60, 70, 80, 90, 95]
    for res in resolutions[:3]:
        observed = []
        for exp in expected_coverages:
            z = {50: 0.674, 60: 0.842, 70: 1.036, 80: 1.282, 90: 1.645, 95: 1.96}[exp]
            lower = all_results[res]['mean'] - z * all_results[res]['std']
            upper = all_results[res]['mean'] + z * all_results[res]['std']
            in_interval = (all_results[res]['targets'] >= lower) & (all_results[res]['targets'] <= upper)
            observed.append(in_interval.mean() * 100)
        ax.plot(expected_coverages, observed, 'o-', label=res, markersize=8)
    ax.plot([50, 95], [50, 95], 'k--', label='Perfect Calibration')
    ax.set_xlabel('Expected Coverage (%)')
    ax.set_ylabel('Observed Coverage (%)')
    ax.set_title('Calibration Plot')
    ax.legend()
    
    # d) R² vs Coverage trade-off
    ax = axes[1, 1]
    r2_values = [compute_uncertainty_metrics(all_results[r])['r2'] for r in resolutions]
    ax.scatter(r2_values, coverages, c=range(len(resolutions)), cmap='viridis', s=100)
    for i, res in enumerate(resolutions):
        ax.annotate(res, (r2_values[i], coverages[i]), fontsize=12, ha='left')
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('R² Score')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Accuracy vs Calibration Trade-off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 4. Sample Predictions with Uncertainty Bands
    # =========================================================================
    n_samples = 100
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()
    
    for idx, res in enumerate(resolutions):
        ax = axes[idx]
        result = all_results[res]
        
        # Take first n_samples
        x = np.arange(n_samples)
        mean = result['mean'][:n_samples]
        std = result['std'][:n_samples]
        targets = result['targets'][:n_samples]
        
        ax.fill_between(x, mean - 1.96*std, mean + 1.96*std, alpha=0.3, color='steelblue', label='95% CI')
        ax.plot(x, mean, 'b-', linewidth=1.5, label='Prediction')
        ax.plot(x, targets, 'r.', markersize=4, alpha=0.7, label='Actual')
        
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Max Mw')
        ax.set_title(f'{res} Resolution')
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_bands.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n   Visualizations saved to {output_dir}")
    
    return metrics_df


# ==============================================================================
# MAIN
# ==============================================================================
def main(args):
    print("\n" + "=" * 70)
    print(" MULTI-RESOLUTION DEEP ENSEMBLE")
    print("=" * 70)
    
    filepath = Path(CONFIG['filename'])
    output_dir = Path('outputs/deep_ensemble_multiresolution')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define resolutions
    resolutions = ['1h', '2h', '4h', '6h', '12h', '24h']
    
    # Initialize ensemble manager
    ensemble = MultiResolutionDeepEnsemble(
        resolutions=resolutions,
        base_config=CONFIG,
        device=DEVICE,
        n_models=args.n_models
    )
    
    if args.mode in ['train', 'all']:
        print("\n" + "=" * 70)
        print(" TRAINING PHASE")
        print("=" * 70)
        
        for res in resolutions:
            print(f"\n{'='*60}")
            print(f" Processing Resolution: {res}")
            print('='*60)
            
            # Prepare data
            print(f"\n   Preparing data...")
            ensemble.prepare_data(filepath, res)
            
            # Build ensemble
            print(f"   Building ensemble...")
            ensemble.build_ensemble(res)
            
            # Train ensemble
            print(f"   Training {args.n_models} models...")
            ensemble.train_ensemble(res, output_dir / 'models', 
                                   epochs=args.epochs, patience=args.patience)
            
        print("\n   Training complete!")
        
    if args.mode in ['evaluate', 'all']:
        print("\n" + "=" * 70)
        print(" EVALUATION PHASE (with Calibration)")
        print("=" * 70)
        
        all_results = {}
        all_results_raw = {}  # Store raw results for comparison
        
        for res in resolutions:
            print(f"\n   Processing {res}...")
            
            # Prepare data if not already
            if res not in ensemble.data_cache:
                ensemble.prepare_data(filepath, res)
                ensemble.build_ensemble(res)
                
            # Load models
            ensemble.load_ensemble(res, output_dir / 'models')
            
            # Generate predictions with uncertainty
            validation_results = ensemble.predict_with_uncertainty(res, split='val')
            raw_results = ensemble.predict_with_uncertainty(res, split='test')
            all_results_raw[res] = raw_results
            
            # Apply calibration
            calibrated_results, cal_factor = calibrate_uncertainty(
                validation_results, raw_results, target_coverage=95
            )
            all_results[res] = calibrated_results
            
            # Compute metrics
            metrics = compute_uncertainty_metrics(calibrated_results)
            print(f"      R²: {metrics['r2']:.4f}, Coverage: {metrics['coverage']:.1f}%, "
                  f"Sharpness: {metrics['sharpness']:.4f}, Cal.Factor: {cal_factor:.2f}")
        
        # Visualize
        print("\n   Generating visualizations...")
        metrics_df = visualize_uncertainty_analysis(all_results, output_dir)
        
        # Print summary
        print("\n" + "=" * 70)
        print(" SUMMARY")
        print("=" * 70)
        print(metrics_df.to_string(index=False))
        
        # Save metrics
        metrics_df.to_csv(output_dir / 'uncertainty_metrics.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Resolution Deep Ensemble')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'evaluate', 'all'],
                        help='Mode: train, evaluate, or all')
    parser.add_argument('--n_models', type=int, default=5,
                        help='Number of models per ensemble')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)
