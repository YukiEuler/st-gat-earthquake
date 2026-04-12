# ==============================================================================
# CROSS_VALIDATION.PY - Time Series Cross-Validation
# ==============================================================================

import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import json
from pathlib import Path


class TimeSeriesCV:
    """Time Series Cross-Validation for temporal data."""
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.results = []
    
    def split(self, X):
        """
        Generate train/test indices for time series cross-validation.
        
        Args:
            X: Data array (T, N, F)
        
        Yields:
            train_indices, test_indices
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for train_idx, test_idx in tscv.split(X):
            yield train_idx, test_idx
    
    def run(self, X_norm, model_class, model_kwargs, criterion, config, 
            adj_sparse, device, run_cv=True):
        """
        Run cross-validation if enabled.
        
        Args:
            run_cv: If False, return single train/test split
        """
        if not run_cv:
            # Single split
            split_idx = int(len(X_norm) * config['train_ratio'])
            return {
                'train_data': X_norm[:split_idx],
                'test_data': X_norm[split_idx:],
                'cv_results': None,
            }
        
        from ..data.dataset import SeismicDataset
        from ..training.trainer import Trainer
        from ..evaluation.metrics import MetricsCalculator
        from torch.utils.data import DataLoader
        
        print("\n" + "=" * 70)
        print(f" Running {self.n_splits}-Fold Time Series Cross-Validation")
        print("=" * 70)
        
        for fold, (train_idx, test_idx) in enumerate(self.split(X_norm)):
            print(f"\n Fold {fold + 1}/{self.n_splits}")
            print(f"   Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
            
            train_data = X_norm[train_idx]
            test_data = X_norm[test_idx]
            
            # Create datasets
            train_dataset = SeismicDataset(
                train_data,
                window_size=config['window_size'],
                horizon=config['horizon']
            )
            test_dataset = SeismicDataset(
                test_data,
                window_size=config['window_size'],
                horizon=config['horizon']
            )
            
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
            
            # Train model
            model = model_class(**model_kwargs).to(device)
            trainer = Trainer(model, criterion, config, device)
            train_result = trainer.fit(train_loader, test_loader, adj_sparse)
            
            # Evaluate
            model.eval()
            predictions = []
            targets = []
            
            import torch
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    output = model(data, adj_sparse)
                    predictions.append(output.cpu().numpy())
                    targets.append(target.numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)
            
            metrics_calc = MetricsCalculator()
            metrics = metrics_calc.calculate_all_metrics(targets, predictions)
            
            self.results.append({
                'fold': fold,
                'train_result': train_result,
                'metrics': metrics,
            })
            
            print(f"   Fold {fold + 1} RMSE: {metrics['overall']['RMSE']:.6f}")
        
        # Aggregate results
        return self.get_summary()
    
    def get_summary(self):
        """Get summary statistics across all folds."""
        if not self.results:
            return None
        
        # Collect metrics across folds
        rmse_values = [r['metrics']['overall']['RMSE'] for r in self.results]
        mae_values = [r['metrics']['overall']['MAE'] for r in self.results]
        r2_values = [r['metrics']['overall']['R2'] for r in self.results]
        
        summary = {
            'n_folds': len(self.results),
            'rmse_mean': float(np.mean(rmse_values)),
            'rmse_std': float(np.std(rmse_values)),
            'mae_mean': float(np.mean(mae_values)),
            'mae_std': float(np.std(mae_values)),
            'r2_mean': float(np.mean(r2_values)),
            'r2_std': float(np.std(r2_values)),
            'per_fold': self.results,
        }
        
        print("\n" + "=" * 70)
        print(" Cross-Validation Summary")
        print("=" * 70)
        print(f"   RMSE: {summary['rmse_mean']:.6f}  {summary['rmse_std']:.6f}")
        print(f"   MAE:  {summary['mae_mean']:.6f}  {summary['mae_std']:.6f}")
        print(f"   R:   {summary['r2_mean']:.6f}  {summary['r2_std']:.6f}")
        print("=" * 70)
        
        return summary
    
    def save_results(self, save_path):
        """Save CV results to JSON."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2, default=str)
        
        print(f" CV results saved to {save_path}")
