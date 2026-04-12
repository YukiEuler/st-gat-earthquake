# ==============================================================================
# CONFIG.PY - Configuration for Earthquake Prediction
# ==============================================================================

import torch

# ==============================================================================
# MAIN CONFIGURATION
# ==============================================================================
CONFIG = {
    # Data
    'filename': 'Amatrice_CAT5.v20210504.csv',
    'grid_size': 0.015,           # ~1.11 km per cell
    'time_bin': '4h',           # Agregasi per jam
    
    # Spatial Bounds Filter (set to None to use full data extent)
    'lat_min': 42.5747,
    'lat_max': 42.9047,
    'lon_min': 13.1282,
    'lon_max': 13.3182,
    
    # Feature Selection - Pilih fitur yang akan digunakan
    'features': ['count', 'max_mw', 'log_energy', 'avg_depth', 'avg_ml', 'std_ml', 'avg_error', 'ml_n_sum'],
    'target_features': ['max_mw'],  # Only max_mw for evaluation
    
    # Preprocessing
    'use_active_nodes_only': True,
    'min_events_per_node': 1500,
    
    # Sequence
    'window_size': 24,          # Input: 24 jam ke belakang
    'horizon': 6,              # Output: 24 jam ke depan (multi-step)
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'mw_rolling_window': 3,
    
    # Rolling Aggregation for All Features
    'rolling_aggregation': {
        'enabled': True,
        'window': 3,  # 3 timesteps window
        'features': {
            'count': 'sum',
            'max_mw': 'max',
            'log_energy': 'sum',
            'avg_depth': 'min',
            'avg_ml': 'mean',
            'std_mw': 'max',
            'std_ml': 'max',
            'avg_error': 'max',
            'ml_n_sum': 'sum',
        }
    },
    
    # Adjacency Matrix
    'radius_km': 15.0,        # Radius konektivitas
    'sigma_km': 150.0,           # Gaussian decay parameter
    
    # Model
    'hidden_dim': 32,
    'num_gat_layers': 2,
    'num_heads': 2,
    'dropout': 0.1,
    'temporal_model': 'multiscale',   # Options: 'lstm', 'tft', or 'multiscale'
    'tft_layers': 3,            # Number of TFT encoder layers
    
    # Multi-Scale Model Options
    'multiscale_scales': [1, 2, 4],  # Temporal scales (1=full, 2=half, 4=quarter resolution)
    'multiscale_fusion': 'concat',   # Fusion type: 'concat', 'attention', or 'gate'
    
    # Training
    'batch_size': 4,
    'epochs': 50,
    'learning_rate': 5e-4,
    'early_stopping_patience': 7,
    'weight_decay': 1e-4,
    
    # Weighted Loss
    'active_weight': 20.0,
    'feature_weights': {
        'count': 1.0,
        'max_mw': 5.0,
        'log_energy': 2.0,
        'avg_depth': 1.0,
        'std_mw': 1.0,
        'min_depth': 1.0,
        'avg_ml': 1.0,
        'std_ml': 1.0,
        'avg_error': 0.5,
        'ml_n_sum': 0.5,
    },
    
    # Loss Function Options
    # Options: 'weighted_mse', 'asymmetric', 'active_only', 'sparse_aware', 'focal', 'multiscale'
    'loss_type': 'weighted_mse',  # Multi-scale loss for 6h, 12h, 24h predictions
    'asymmetric_alpha': 0.8,      # Alpha for asymmetric loss (>0.5 = penalize underpredict more)
    'underpredict_penalty': 5.0,  # Extra penalty for underprediction (sparse_aware only)
    'magnitude_idx': 0,           # Index of max_mw in target features (now 0 since only max_mw)
    'focal_gamma': 2.0,           # Focal loss gamma (higher = more focus on hard examples)
    
    # Multi-Scale Loss Options
    'multiscale_horizons': [1, 2, 4],      # Horizons in timesteps: 1=6h, 2=12h, 4=24h
    'multiscale_weights': [1.5, 1.75, 2.0], # Higher weight for longer horizons
    
    # Feature Transformation
    'transform_magnitude': True,       # Apply log transform to max_mw
    'add_running_features': True,      # Add running max/mean features
    'add_engineered_features': True,  # Add b-value, event_rate, time_since_last (slow)
    
    # Uncertainty Estimation (Deep Ensembles)
    'n_ensemble_models': 5,
    'uncertainty_enabled': True,
    
    # Optional Features
    'hyperparameter_tuning': False,  # Set True untuk Optuna tuning
    'cross_validation': False,       # Set True untuk Time Series CV
    'n_cv_splits': 5,
    
    # Output
    'output_dir': 'outputs',
    'save_figures': True,
    'save_attention': True,
    
    # Random Seed
    'seed': 42,
}

# ==============================================================================
# FEATURE DEFINITIONS
# ==============================================================================
FEATURE_DEFINITIONS = {
    'count': {
        'name': 'Event Count',
        'description': 'Jumlah event gempa dalam bin',
        'aggregation': 'count',
        'column': 'mw',
    },
    'max_mw': {
        'name': 'Max Magnitude',
        'description': 'Magnitudo maksimum dalam bin',
        'aggregation': 'max',
        'column': 'mw',
    },
    'log_energy': {
        'name': 'Log Total Energy',
        'description': 'Total energi seismik (log scale)',
        'aggregation': 'sum',
        'column': 'log_energy',
    },
    'avg_depth': {
        'name': 'Average Depth',
        'description': 'Rata-rata kedalaman gempa',
        'aggregation': 'mean',
        'column': 'dep',
    },
    'min_depth': {
        'name': 'Min Depth',
        'description': 'Kedalaman minimum',
        'aggregation': 'min',
        'column': 'dep',
    },
    'std_mw': {
        'name': 'Std Magnitude',
        'description': 'Standar deviasi magnitudo',
        'aggregation': 'std',
        'column': 'mw',
    },
    'avg_ml': {
        'name': 'Avg Local Magnitude',
        'description': 'Rata-rata local magnitude',
        'aggregation': 'mean',
        'column': 'ml_mean',
    },
    'std_ml': {
        'name': 'Std Local Magnitude',
        'description': 'Standar deviasi local magnitude',
        'aggregation': 'std',
        'column': 'ml_mean',
    },
    'avg_error': {
        'name': 'Avg Location Error',
        'description': 'Rata-rata horizontal error lokasi',
        'aggregation': 'mean',
        'column': 'avg_error',
    },
    'ml_n_sum': {
        'name': 'Total ML Readings',
        'description': 'Total jumlah ML readings',
        'aggregation': 'sum',
        'column': 'ml_n',
    },
}

# ==============================================================================
# ABLATION STUDY CONFIGURATIONS
# ==============================================================================
ABLATION_CONFIGS = {
    'full_stgat': {
        'name': 'Full ST-GAT',
        'use_attention': True,
        'use_multihead': True,
        'use_lstm': True,
        'use_distance_adj': True,
    },
    'no_multihead': {
        'name': 'ST-GAT (Single Head)',
        'use_attention': True,
        'use_multihead': False,
        'use_lstm': True,
        'use_distance_adj': True,
    },
    'no_distance_adj': {
        'name': 'ST-GAT (Binary Adj)',
        'use_attention': True,
        'use_multihead': True,
        'use_lstm': True,
        'use_distance_adj': False,
    },
    'gcn_lstm': {
        'name': 'GCN + LSTM',
        'use_attention': False,
        'use_multihead': False,
        'use_lstm': True,
        'use_distance_adj': True,
    },
    'lstm_only': {
        'name': 'LSTM Only',
        'use_attention': False,
        'use_multihead': False,
        'use_lstm': True,
        'use_distance_adj': False,
    },
}

# ==============================================================================
# DEVICE CONFIGURATION
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_indices(feature_list, all_features):
    """Get indices of selected features from all features."""
    return [all_features.index(f) for f in feature_list if f in all_features]

def print_config():
    """Print current configuration."""
    print("\n" + "=" * 70)
    print(" CONFIGURATION")
    print("=" * 70)
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    print("=" * 70)
