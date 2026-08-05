# ==============================================================================
# COMPARE_MR_MH.PY - Multi-Resolution vs Multi-Horizon Comparison
# ==============================================================================
"""
Compare predictions from:
1. Multi-Resolution Models: Separate models for different time resolutions
2. Multi-Horizon Model: Single model predicting multiple steps ahead

Usage:
    python compare_mr_mh.py --mode train   # Train both model types
    python -s compare_mr_mh.py --mode compare # Compare pretrained models
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import warnings
import json
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

set_seed(42)

# ==============================================================================
# MULTI-HORIZON MODEL (Single model predicting multiple steps)
# ==============================================================================
class MultiHorizonPredictor:
    """
    Single model that predicts multiple horizons at once.
    E.g., with time_bin=4h and horizon=6, predicts 4, 8, 12, 16, 20, 24 hours ahead.
    """
    
    def __init__(self, config, device):
        self.config = copy.deepcopy(config)
        self.device = device
        self.model = None
        self.data = None
        self.adj_sparse = None
        
    def prepare_data(self, filepath):
        """Prepare data with multi-horizon setup."""
        print("\n" + "=" * 70)
        print(" PREPARING DATA FOR MULTI-HORIZON MODEL")
        print("=" * 70)
        
        # Use base config with horizon=6
        mh_config = copy.deepcopy(self.config)
        mh_config['horizon'] = 6  # Predict 6 steps ahead
        mh_hours = pd.Timedelta(mh_config['time_bin']).total_seconds() / 3600.0
        mh_config['window_size'] = max(1, int(round(
            mh_config.get('history_hours', 96) / mh_hours
        )))
        
        # Preprocess using correct API
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
        self.adj_sparse = adj_builder.scipy_to_torch_sparse(adj_scipy, device=self.device)
        
        # Get target feature indices
        all_features = mh_config['features']
        target_features = mh_config.get('target_features', all_features)
        target_indices = [all_features.index(f) for f in target_features if f in all_features]
        
        # Create datasets
        train_dataset = SeismicDataset(
            data['train_data'],
            target_data=data['train_target_data'],
            window_size=mh_config['window_size'],
            horizon=mh_config['horizon'],
        )
        val_dataset = SeismicDataset(
            data['val_data'],
            target_data=data['val_target_data'],
            window_size=mh_config['window_size'],
            horizon=mh_config['horizon'],
        )
        test_dataset = SeismicDataset(
            data['test_data'],
            target_data=data['test_target_data'],
            window_size=mh_config['window_size'],
            horizon=mh_config['horizon'],
        )
        
        # Create data loaders
        self.data = {
            'train_loader': DataLoader(train_dataset, batch_size=mh_config['batch_size'], shuffle=True),
            'val_loader': DataLoader(val_dataset, batch_size=mh_config['batch_size'], shuffle=False),
            'test_loader': DataLoader(test_dataset, batch_size=mh_config['batch_size'], shuffle=False),
            'num_nodes': data['num_nodes'],
            'in_features': data['train_data'].shape[-1],  # (T, N, F)
            'n_targets': len(target_indices),
            'config': mh_config,
            'feature_stats': data['feature_stats'],
            'target_stats': data['target_stats'],
            'split_timestamps': data['split_timestamps']
        }
        
        print(f"   Num nodes: {self.data['num_nodes']}")
        print(f"   Input features: {self.data['in_features']}")
        print(f"   Target features: {self.data['n_targets']}")
        print(f"   Horizon: {mh_config['horizon']} steps = {mh_config['horizon'] * 4} hours")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
        
    def build_model(self):
        """Build multi-horizon model."""
        self.model = STGAT(
            num_nodes=self.data['num_nodes'],
            in_features=self.data['in_features'],
            hidden_dim=self.config['hidden_dim'],
            out_features=self.data['n_targets'],
            horizon=self.data['config']['horizon'],
            num_gat_layers=self.config['num_gat_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout'],
            use_attention=True,
            use_multihead=True,
            use_skip=True
        ).to(self.device)
        
        print(f"   Multi-Horizon Model built: {sum(p.numel() for p in self.model.parameters()):,} params")
        
    def train(self, output_dir, epochs=50, patience=7):
        """Train multi-horizon model."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        criterion = torch.nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in tqdm(self.data['train_loader'], desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x, self.adj_sparse)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(self.data['train_loader'])
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in self.data['val_loader']:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    outputs = self.model(x, self.adj_sparse)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    
            val_loss /= len(self.data['val_loader'])
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            print(f"   Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), output_dir / 'multi_horizon_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break
                    
        # Load best model
        self.model.load_state_dict(torch.load(output_dir / 'multi_horizon_model.pth'))
        
        return train_losses, val_losses
        
    def load(self, model_path):
        """Load pre-trained model."""
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"   Loaded multi-horizon model from {model_path}")
        else:
            print(f"   WARNING: {model_path} not found!")
        
    def predict(self):
        """
        Generate predictions for all horizons.
        
        Returns:
            dict: {horizon_idx: {'predictions': [], 'targets': []}}
        """
        self.model.eval()
        horizon = self.data['config']['horizon']
        results = {h: {'predictions': [], 'targets': []} for h in range(horizon)}
        
        # Find max_mw index in target features (usually index 1)
        target_features = self.data['config'].get('target_features', self.data['config']['features'])
        max_mw_idx = target_features.index('max_mw') if 'max_mw' in target_features else 0
        target_stats = self.data['target_stats']
        target_offset = float(target_stats.get('offset', target_stats['mean'])[0])
        target_std = float(target_stats['std'][0])
        
        with torch.no_grad():
            for batch in tqdm(self.data['test_loader'], desc="Multi-Horizon Predicting"):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Forward pass - output shape: (B, H, N, F)
                outputs = self.model(x, self.adj_sparse)
                
                # Store predictions for each horizon
                for h in range(horizon):
                    pred_h = outputs[:, h, :, max_mw_idx].cpu().numpy() * target_std + target_offset
                    target_h = y[:, h, :, max_mw_idx].cpu().numpy() * target_std + target_offset
                    
                    results[h]['predictions'].extend(pred_h.flatten().tolist())
                    results[h]['targets'].extend(target_h.flatten().tolist())
                    
        return results


# ==============================================================================
# MULTI-RESOLUTION MODEL (Separate models for each resolution)
# ==============================================================================
class MultiResolutionPredictor:
    """
    Separate models trained for different time resolutions.
    E.g., 4h model, 8h model, 12h model - each predicts 1 step ahead.
    """
    
    def __init__(self, resolutions, base_config, device):
        """
        Args:
            resolutions: List of time_bin strings, e.g., ['4h', '8h', '12h', '24h']
        """
        self.resolutions = resolutions
        self.base_config = base_config
        self.device = device
        self.models = {}
        self.data_cache = {}
        self.adj_cache = {}
        self.canonical_split_timestamps = None
        
    def prepare_data(self, filepath):
        """Prepare data for each resolution."""
        print("\n" + "=" * 70)
        print(" PREPARING DATA FOR MULTI-RESOLUTION MODELS")
        print("=" * 70)
        
        for res in self.resolutions:
            print(f"\n   Processing resolution: {res}")
            
            res_config = copy.deepcopy(self.base_config)
            res_config['time_bin'] = res
            res_config['horizon'] = 1  # Single step prediction
            res_hours = pd.Timedelta(res).total_seconds() / 3600.0
            res_config['window_size'] = max(1, int(round(
                res_config.get('history_hours', 96) / res_hours
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
            
            self.data_cache[res] = {
                'train_loader': DataLoader(train_dataset, batch_size=res_config['batch_size'], shuffle=True),
                'val_loader': DataLoader(val_dataset, batch_size=res_config['batch_size'], shuffle=False),
                'test_loader': DataLoader(test_dataset, batch_size=res_config['batch_size'], shuffle=False),
                'num_nodes': data['num_nodes'],
                'in_features': data['train_data'].shape[-1],
                'n_targets': len(target_indices),
                'config': res_config,
                'target_stats': data['target_stats'],
                'split_timestamps': data['split_timestamps']
            }
            self.adj_cache[res] = adj_sparse
            
            print(f"      Nodes: {data['num_nodes']}, Features: {data['train_data'].shape[-1]}, Test samples: {len(test_dataset)}")
            
    def build_models(self):
        """Build model for each resolution."""
        for res in self.resolutions:
            data = self.data_cache[res]
            
            model = STGAT(
                num_nodes=data['num_nodes'],
                in_features=data['in_features'],
                hidden_dim=self.base_config['hidden_dim'],
                out_features=data['n_targets'],
                horizon=1,  # Single step
                num_gat_layers=self.base_config['num_gat_layers'],
                num_heads=self.base_config['num_heads'],
                dropout=self.base_config['dropout'],
                use_attention=True,
                use_multihead=True,
                use_skip=True
            ).to(self.device)
            
            self.models[res] = model
            print(f"   Model {res}: {sum(p.numel() for p in model.parameters()):,} params")
            
    def train_all(self, output_dir, epochs=50, patience=7):
        """Train all resolution models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for res in self.resolutions:
            print(f"\n   Training {res} model...")
            
            model = self.models[res]
            data = self.data_cache[res]
            adj_sparse = self.adj_cache[res]
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.base_config['learning_rate'])
            criterion = torch.nn.MSELoss()
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0
                for batch in self.data_cache[res]['train_loader']:
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
                    
                train_loss /= len(self.data_cache[res]['train_loader'])
                
                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in self.data_cache[res]['val_loader']:
                        x, y = batch
                        x = x.to(self.device)
                        y = y.to(self.device)
                        
                        outputs = model(x, adj_sparse)
                        loss = criterion(outputs, y)
                        val_loss += loss.item()
                        
                val_loss /= len(self.data_cache[res]['val_loader'])
                
                if (epoch + 1) % 10 == 0:
                    print(f"      Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), output_dir / f'multi_res_{res}_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"      Early stopping at epoch {epoch+1}")
                        break
                        
            # Load best model
            model.load_state_dict(torch.load(output_dir / f'multi_res_{res}_model.pth'))
            
    def load_models(self, model_dir):
        """Load pre-trained models for each resolution."""
        model_dir = Path(model_dir)
        
        for res in self.resolutions:
            model_path = model_dir / f'multi_res_{res}_model.pth'
            if model_path.exists():
                self.models[res].load_state_dict(torch.load(model_path, map_location=self.device))
                self.models[res].eval()
                print(f"   Loaded {res} model from {model_path}")
            else:
                print(f"   WARNING: {model_path} not found!")
                
    def predict(self):
        """
        Generate predictions for each resolution.
        
        Returns:
            dict: {resolution: {'predictions': [], 'targets': []}}
        """
        results = {}
        
        for res in self.resolutions:
            model = self.models[res]
            model.eval()
            adj_sparse = self.adj_cache[res]
            
            # Find max_mw index
            res_config = self.data_cache[res]['config']
            target_features = res_config.get('target_features', res_config['features'])
            max_mw_idx = target_features.index('max_mw') if 'max_mw' in target_features else 0
            target_stats = self.data_cache[res]['target_stats']
            target_offset = float(target_stats.get('offset', target_stats['mean'])[0])
            target_std = float(target_stats['std'][0])
            
            predictions = []
            targets = []
            
            with torch.no_grad():
                for batch in tqdm(self.data_cache[res]['test_loader'], 
                                  desc=f"Multi-Res {res} Predicting"):
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    outputs = model(x, adj_sparse)
                    
                    # Output shape: (B, 1, N, F)
                    pred = outputs[:, 0, :, max_mw_idx].cpu().numpy() * target_std + target_offset
                    target = y[:, 0, :, max_mw_idx].cpu().numpy() * target_std + target_offset
                    
                    predictions.extend(pred.flatten().tolist())
                    targets.extend(target.flatten().tolist())
                    
            results[res] = {
                'predictions': predictions,
                'targets': targets
            }
            
        return results


# ==============================================================================
# COMPARISON FUNCTIONS
# ==============================================================================
def calculate_metrics(predictions, targets):
    """Calculate regression metrics."""
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Remove NaN
    mask = ~(np.isnan(predictions) | np.isnan(targets))
    predictions = predictions[mask]
    targets = targets[mask]
    
    if len(targets) == 0:
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
    
    return {
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions),
        'r2': r2_score(targets, predictions)
    }


def compare_at_same_horizons(mh_results, mr_results, time_bin_hours=4):
    """
    Compare multi-horizon and multi-resolution at same target times.
    
    Args:
        mh_results: Results from MultiHorizonPredictor
        mr_results: Results from MultiResolutionPredictor
        time_bin_hours: Time bin for multi-horizon model
        
    Returns:
        comparison_df: DataFrame with comparison metrics
    """
    comparisons = []
    
    # Map multi-horizon index to hours ahead
    # h=0 -> 4h, h=1 -> 8h, h=2 -> 12h, etc.
    
    for h_idx in range(len(mh_results)):
        target_hours = (h_idx + 1) * time_bin_hours
        target_res = f"{target_hours}h"
        
        # Multi-horizon metrics
        mh_metrics = calculate_metrics(
            mh_results[h_idx]['predictions'],
            mh_results[h_idx]['targets']
        )
        
        # Multi-resolution metrics (if available)
        if target_res in mr_results:
            mr_metrics = calculate_metrics(
                mr_results[target_res]['predictions'],
                mr_results[target_res]['targets']
            )
        else:
            mr_metrics = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}
            
        comparisons.append({
            'target_hours': target_hours,
            'mh_rmse': mh_metrics['rmse'],
            'mh_mae': mh_metrics['mae'],
            'mh_r2': mh_metrics['r2'],
            'mr_rmse': mr_metrics['rmse'],
            'mr_mae': mr_metrics['mae'],
            'mr_r2': mr_metrics['r2'],
            'rmse_diff': mr_metrics['rmse'] - mh_metrics['rmse'] if not np.isnan(mr_metrics['rmse']) else np.nan,
            'r2_diff': mh_metrics['r2'] - mr_metrics['r2'] if not np.isnan(mr_metrics['r2']) else np.nan
        })
        
    return pd.DataFrame(comparisons)


def visualize_comparison(comparison_df, mh_results, mr_results, output_dir):
    """Generate comparison visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set larger font sizes globally
    plt.rcParams.update({
        'font.size': 15,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20
    })
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # =========================================================================
    # 1. Bar Chart: Metrics Comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    
    x = comparison_df['target_hours'].values
    width = 0.35
    x_pos = np.arange(len(x))
    
    # RMSE
    axes[0].bar(x_pos - width/2, comparison_df['mh_rmse'], width, label='Multi-Horizon', color='steelblue')
    axes[0].bar(x_pos + width/2, comparison_df['mr_rmse'], width, label='Multi-Resolution', color='coral')
    axes[0].set_xlabel('Target Horizon (hours)')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('RMSE Comparison')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x)
    
    # MAE
    axes[1].bar(x_pos - width/2, comparison_df['mh_mae'], width, label='Multi-Horizon', color='steelblue')
    axes[1].bar(x_pos + width/2, comparison_df['mr_mae'], width, label='Multi-Resolution', color='coral')
    axes[1].set_xlabel('Target Horizon (hours)')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('MAE Comparison')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(x)
    
    # R²
    axes[2].bar(x_pos - width/2, comparison_df['mh_r2'], width, label='Multi-Horizon', color='steelblue')
    axes[2].bar(x_pos + width/2, comparison_df['mr_r2'], width, label='Multi-Resolution', color='coral')
    axes[2].set_xlabel('Target Horizon (hours)')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R² Comparison')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(x)

    # Use one shared legend to avoid overlap inside each subplot.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=True)

    plt.savefig(output_dir / 'metrics_comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 2. Line Chart: Performance Degradation
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(x, comparison_df['mh_r2'], 'o-', label='Multi-Horizon', color='steelblue', linewidth=2, markersize=8)
    axes[0].plot(x, comparison_df['mr_r2'], 's--', label='Multi-Resolution', color='coral', linewidth=2, markersize=8)
    axes[0].set_xlabel('Target Horizon (hours)')
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² Degradation by Horizon')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x, comparison_df['mh_rmse'], 'o-', label='Multi-Horizon', color='steelblue', linewidth=2, markersize=8)
    axes[1].plot(x, comparison_df['mr_rmse'], 's--', label='Multi-Resolution', color='coral', linewidth=2, markersize=8)
    axes[1].set_xlabel('Target Horizon (hours)')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE by Horizon')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_degradation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 3. Scatter Plots: Predicted vs Actual per Horizon
    # =========================================================================
    n_horizons = len(mh_results)
    n_cols = min(n_horizons, 3)
    n_rows = (n_horizons + n_cols - 1) // n_cols * 2  # 2 rows per group (MH and MR)
    
    fig, axes = plt.subplots(2, n_horizons, figsize=(4 * n_horizons, 8))
    if n_horizons == 1:
        axes = axes.reshape(2, 1)
    
    for h_idx in range(n_horizons):
        target_hours = (h_idx + 1) * 4
        target_res = f"{target_hours}h"
        
        # Multi-Horizon scatter
        ax_mh = axes[0, h_idx]
        mh_pred = np.array(mh_results[h_idx]['predictions'])
        mh_target = np.array(mh_results[h_idx]['targets'])
        ax_mh.scatter(mh_target, mh_pred, alpha=0.3, s=10, color='steelblue')
        ax_mh.plot([mh_target.min(), mh_target.max()], [mh_target.min(), mh_target.max()], 
                   'r--', linewidth=2)
        ax_mh.set_xlabel('Actual')
        ax_mh.set_ylabel('Predicted')
        mh_r2 = comparison_df.iloc[h_idx]["mh_r2"]
        ax_mh.set_title(f'Multi-Horizon +{target_hours}h\nR²={mh_r2:.3f}')
        
        # Multi-Resolution scatter (if available)
        ax_mr = axes[1, h_idx]
        if target_res in mr_results:
            mr_pred = np.array(mr_results[target_res]['predictions'])
            mr_target = np.array(mr_results[target_res]['targets'])
            ax_mr.scatter(mr_target, mr_pred, alpha=0.3, s=10, color='coral')
            ax_mr.plot([mr_target.min(), mr_target.max()], [mr_target.min(), mr_target.max()], 
                       'r--', linewidth=2)
            ax_mr.set_xlabel('Actual')
            ax_mr.set_ylabel('Predicted')
            mr_r2 = comparison_df.iloc[h_idx]["mr_r2"]
            ax_mr.set_title(f'Multi-Res {target_res}\nR²={mr_r2:.3f}')
        else:
            ax_mr.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20, transform=ax_mr.transAxes)
            ax_mr.set_title(f'Multi-Res {target_res}\n(Not Available)')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # 4. Difference Heatmap
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 3))
    
    diff_data = comparison_df[['r2_diff', 'rmse_diff']].T.values
    sns.heatmap(
        diff_data, 
        annot=True, 
        fmt='.3f',
        xticklabels=[f'+{h}h' for h in comparison_df['target_hours']],
        yticklabels=['R² (MH-MR)', 'RMSE (MR-MH)'],
        cmap='RdYlGn',
        center=0,
        ax=ax,
        annot_kws={'size': 13}
    )
    ax.set_title('Performance Difference (Positive = Multi-Horizon Better)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'difference_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n   Visualizations saved to {output_dir}")


# ==============================================================================
# MAIN
# ==============================================================================
def main(args):
    """Main function."""
    print("\n" + "=" * 70)
    print(" MULTI-RESOLUTION vs MULTI-HORIZON COMPARISON")
    print("=" * 70)
    
    filepath = Path(CONFIG['filename'])
    output_dir = Path('outputs/comparison_mr_mh')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define resolutions for multi-resolution (must align with multi-horizon steps)
    # Multi-Horizon: 4h time_bin, horizon=6 -> predicts 4, 8, 12, 16, 20, 24h ahead
    # Multi-Resolution: separate models for 4h, 8h, 12h, 16h, 20h, 24h
    resolutions = ['4h', '8h', '12h', '16h', '20h', '24h']
    
    # Initialize predictors
    mh_predictor = MultiHorizonPredictor(CONFIG, DEVICE)
    mr_predictor = MultiResolutionPredictor(resolutions, CONFIG, DEVICE)
    
    if args.mode == 'train':
        # Prepare and train both model types
        print("\n[1/4] Preparing Multi-Horizon Data...")
        mh_predictor.prepare_data(filepath)
        mh_predictor.build_model()
        
        print("\n[2/4] Training Multi-Horizon Model...")
        mh_predictor.train(output_dir / 'models', epochs=args.epochs, patience=args.patience)
        
        print("\n[3/4] Preparing Multi-Resolution Data...")
        mr_predictor.prepare_data(filepath)
        mr_predictor.build_models()
        
        print("\n[4/4] Training Multi-Resolution Models...")
        mr_predictor.train_all(output_dir / 'models', epochs=args.epochs, patience=args.patience)
        
        print("\n   Training complete!")
        
    elif args.mode == 'compare':
        # Load and compare pretrained models
        print("\n[1/4] Preparing Multi-Horizon Data...")
        mh_predictor.prepare_data(filepath)
        mh_predictor.build_model()
        
        model_path = output_dir / 'models' / 'multi_horizon_model.pth'
        mh_predictor.load(model_path)
        
        print("\n[2/4] Preparing Multi-Resolution Data...")
        mr_predictor.prepare_data(filepath)
        mr_predictor.build_models()
        mr_predictor.load_models(output_dir / 'models')
        
        print("\n[3/4] Generating Predictions...")
        mh_results = mh_predictor.predict()
        mr_results = mr_predictor.predict()
        
        print("\n[4/4] Comparing Results...")
        comparison_df = compare_at_same_horizons(mh_results, mr_results, time_bin_hours=4)
        
        # Save comparison metrics
        comparison_df.to_csv(output_dir / 'comparison_metrics.csv', index=False)
        print("\n" + "=" * 70)
        print(" COMPARISON RESULTS")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        
        # Visualize
        visualize_comparison(comparison_df, mh_results, mr_results, output_dir)
        
        # Summary
        print("\n" + "=" * 70)
        print(" SUMMARY")
        print("=" * 70)
        avg_mh_r2 = comparison_df['mh_r2'].mean()
        avg_mr_r2 = comparison_df['mr_r2'].dropna().mean()
        print(f"   Average R² (Multi-Horizon):   {avg_mh_r2:.4f}")
        print(f"   Average R² (Multi-Resolution): {avg_mr_r2:.4f}")
        if not np.isnan(avg_mr_r2):
            if avg_mh_r2 > avg_mr_r2:
                print(f"   Winner: Multi-Horizon (+{(avg_mh_r2 - avg_mr_r2):.4f})")
            else:
                print(f"   Winner: Multi-Resolution (+{(avg_mr_r2 - avg_mh_r2):.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Resolution vs Multi-Horizon Comparison')
    parser.add_argument('--mode', type=str, default='compare',
                        choices=['train', 'compare'],
                        help='Mode: train or compare')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)
