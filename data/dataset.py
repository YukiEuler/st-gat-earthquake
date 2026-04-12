# ==============================================================================
# DATASET.PY - Seismic Dataset with Lazy Loading
# ==============================================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SeismicDataset(Dataset):
    """
    Custom Dataset dengan Lazy Loading untuk efisiensi memori.
    Mendukung multi-step forecasting dan feature selection.
    """
    
    def __init__(self, data, window_size=24, horizon=24, 
                 feature_indices=None, target_indices=None):
        """
        Args:
            data: numpy array (T, N, F) - normalized features
            window_size: input window size (hours)
            horizon: prediction horizon (hours) 
            feature_indices: indices of features to use as input
            target_indices: indices of features to predict
        """
        self.data = data
        self.window = window_size
        self.horizon = horizon
        self.feature_indices = feature_indices
        self.target_indices = target_indices
        
    def __len__(self):
        return len(self.data) - self.window - self.horizon + 1
    
    def __getitem__(self, idx):
        # Slice on-the-fly (lazy loading)
        x = self.data[idx : idx + self.window]  # (window, N, F)
        y = self.data[idx + self.window : idx + self.window + self.horizon]  # (horizon, N, F)
        
        # Feature selection for input
        if self.feature_indices is not None:
            x = x[:, :, self.feature_indices]
        
        # Target selection for output
        if self.target_indices is not None:
            y = y[:, :, self.target_indices]
        
        # Convert to tensor
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        
        return x, y


def create_dataloaders(train_data, test_data, config):
    """Create train and test dataloaders."""
    from ..config import CONFIG, get_feature_indices, FEATURE_DEFINITIONS
    
    # Get feature indices
    all_features = list(FEATURE_DEFINITIONS.keys())[:4]  # First 4 are default
    feature_indices = get_feature_indices(config.get('features', all_features), all_features)
    target_indices = get_feature_indices(config.get('target_features', ['max_mw']), all_features)
    
    # Create datasets
    train_dataset = SeismicDataset(
        train_data,
        window_size=config['window_size'],
        horizon=config['horizon'],
        feature_indices=feature_indices if feature_indices else None,
        target_indices=target_indices if target_indices else None
    )
    
    test_dataset = SeismicDataset(
        test_data,
        window_size=config['window_size'],
        horizon=config['horizon'],
        feature_indices=feature_indices if feature_indices else None,
        target_indices=target_indices if target_indices else None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader, {
        'feature_indices': feature_indices,
        'target_indices': target_indices,
        'n_input_features': len(feature_indices) if feature_indices else train_data.shape[-1],
        'n_target_features': len(target_indices) if target_indices else train_data.shape[-1],
    }


class HybridSeismicDataset(Dataset):
    """
    Hybrid Dataset untuk model classification + regression.
    
    Classification: 4 classes berdasarkan magnitude bins
        - Class 0: mw < 1
        - Class 1: 1 <= mw < 2
        - Class 2: 2 <= mw < 3
        - Class 3: mw >= 3
    
    Regression: count, log_energy, avg_depth
    """
    
    # Magnitude bins untuk classification (berdasarkan normalized values)
    # Data stats dari diagnostic: mean=2.01, std=0.57
    # Normalized bins: (raw - 2.01) / 0.57
    MAGNITUDE_BINS = {
        'raw_thresholds': [1.0, 2.0, 3.0],  # Raw magnitude thresholds
        'n_classes': 4,
        'class_names': ['M<1', 'M1-2', 'M2-3', 'M>=3']
    }
    
    def __init__(self, data, window_size=24, horizon=24,
                 magnitude_idx=1, regression_indices=None,
                 feature_stats=None):
        """
        Args:
            data: numpy array (T, N, F) - normalized features
            window_size: input window size
            horizon: prediction horizon
            magnitude_idx: index of max_mw in features
            regression_indices: indices for regression features
            feature_stats: dict with 'mean' and 'std' for denormalization
        """
        self.data = data
        self.window = window_size
        self.horizon = horizon
        self.magnitude_idx = magnitude_idx
        self.feature_stats = feature_stats
        
        # Default: predict count (0), log_energy (2), avg_depth (3)
        if regression_indices is None:
            self.regression_indices = [0, 2, 3]
        else:
            self.regression_indices = regression_indices
        
        # Calculate normalized thresholds if feature_stats provided
        if feature_stats is not None:
            mw_mean = feature_stats['mean'][magnitude_idx]
            mw_std = feature_stats['std'][magnitude_idx]
            # Convert raw thresholds to normalized space
            self.norm_thresholds = [(t - mw_mean) / mw_std for t in self.MAGNITUDE_BINS['raw_thresholds']]
        else:
            # Fallback: use approximate normalized values
            # Assuming mean~2, std~0.57 from diagnostic
            self.norm_thresholds = [-1.77, 0.0, 1.75]  # Approx for [1, 2, 3]
        
    def __len__(self):
        return len(self.data) - self.window - self.horizon + 1
    
    def _magnitude_to_class(self, magnitude):
        """Convert normalized magnitude to class label (0-3)."""
        # magnitude is normalized
        # Class 0: < threshold[0]
        # Class 1: threshold[0] <= x < threshold[1]
        # Class 2: threshold[1] <= x < threshold[2]
        # Class 3: >= threshold[2]
        
        classes = np.zeros_like(magnitude, dtype=np.int64)
        classes[magnitude >= self.norm_thresholds[0]] = 1
        classes[magnitude >= self.norm_thresholds[1]] = 2
        classes[magnitude >= self.norm_thresholds[2]] = 3
        
        return classes
    
    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx : idx + self.window]  # (window, N, F)
        
        # Target sequence
        y_full = self.data[idx + self.window : idx + self.window + self.horizon]  # (horizon, N, F)
        
        # Regression targets
        y_regression = y_full[:, :, self.regression_indices]  # (H, N, n_reg)
        
        # Classification targets (multiclass)
        y_magnitude = y_full[:, :, self.magnitude_idx]  # (H, N)
        y_classification = self._magnitude_to_class(y_magnitude)  # (H, N) - integer labels 0-3
        
        # Convert to tensors
        x = torch.from_numpy(x).float()
        y_regression = torch.from_numpy(y_regression).float()
        y_classification = torch.from_numpy(y_classification).long()  # Long for CrossEntropy
        
        return x, {
            'regression': y_regression,
            'classification': y_classification
        }

