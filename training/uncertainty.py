# ==============================================================================
# UNCERTAINTY.PY - Uncertainty Estimation with Deep Ensembles
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import copy


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty estimation.
    
    Better than MC Dropout because:
    - Better calibrated uncertainty estimates
    - Captures both epistemic and aleatoric uncertainty
    - Each model trained independently with different initialization
    """
    
    def __init__(self, model_class, model_kwargs, n_models=5, device='cuda'):
        self.n_models = n_models
        self.device = device
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        
        # Initialize ensemble members
        self.models = []
        for i in range(n_models):
            model = model_class(**model_kwargs).to(device)
            self.models.append(model)
        
        print(f" Deep Ensemble initialized with {n_models} models")
    
    def fit(self, train_loader, val_loader, adj_sparse, criterion, config, 
            save_dir=None):
        """Train all ensemble members."""
        from .trainer import Trainer
        
        results = []
        
        for i, model in enumerate(self.models):
            print(f"\n{'='*70}")
            print(f" Training Ensemble Model {i+1}/{self.n_models}")
            print('='*70)
            
            trainer = Trainer(model, criterion, config, self.device)
            result = trainer.fit(train_loader, val_loader, adj_sparse)
            results.append(result)
            
            if save_dir:
                save_path = Path(save_dir) / f"ensemble_model_{i}.pth"
                trainer.save_checkpoint(save_path)
        
        return results
    
    def predict_with_uncertainty(self, data, adj_sparse):
        """
        Generate predictions with uncertainty estimates.
        
        Returns:
            mean: Mean prediction across ensemble
            std: Standard deviation (uncertainty)
            ci_lower: Lower bound of 95% CI
            ci_upper: Upper bound of 95% CI
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(data, adj_sparse)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)  # (n_models, B, H, N, F)
        
        # Statistics
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        # 95% Confidence Interval
        ci_lower = mean - 1.96 * std
        ci_upper = mean + 1.96 * std
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_predictions': predictions,
        }
    
    def generate_predictions(self, test_loader, adj_sparse):
        """Generate predictions for entire test set."""
        all_means = []
        all_stds = []
        all_targets = []
        
        for data, target in tqdm(test_loader, desc="Generating predictions"):
            data = data.to(self.device)
            
            result = self.predict_with_uncertainty(data, adj_sparse)
            
            all_means.append(result['mean'])
            all_stds.append(result['std'])
            all_targets.append(target.numpy())
        
        return {
            'mean': np.concatenate(all_means, axis=0),
            'std': np.concatenate(all_stds, axis=0),
            'targets': np.concatenate(all_targets, axis=0),
        }
    
    def save(self, save_dir):
        """Save all ensemble models."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            path = save_dir / f"ensemble_model_{i}.pth"
            torch.save(model.state_dict(), path)
        
        print(f" Ensemble saved to {save_dir}")
    
    def load(self, save_dir):
        """Load all ensemble models."""
        save_dir = Path(save_dir)
        
        for i, model in enumerate(self.models):
            path = save_dir / f"ensemble_model_{i}.pth"
            model.load_state_dict(torch.load(path, map_location=self.device))
        
        print(f" Ensemble loaded from {save_dir}")


class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty estimation.
    Alternative to Deep Ensembles (simpler but less accurate).
    """
    
    def __init__(self, model, n_samples=50):
        self.model = model
        self.n_samples = n_samples
    
    def predict_with_uncertainty(self, data, adj_sparse):
        """Generate predictions with MC Dropout uncertainty."""
        self.model.train()  # Keep dropout active
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(data, adj_sparse)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)
        
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': mean - 1.96 * std,
            'ci_upper': mean + 1.96 * std,
        }
