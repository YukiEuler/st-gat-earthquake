# ==============================================================================
# CONFIG.PY - Configuration for Earthquake Prediction
# ==============================================================================

import torch

# ==============================================================================
# MAIN CONFIGURATION
# ==============================================================================
CONFIG = {
    # Canonical rerun protocol (revision plan)
    'canonical_run_name': 'revision_canonical_v3_hurdle',
    'target_definition': 'raw_bin_max_mw',
    'auxiliary_target_definition': 'raw_bin_activity_indicator',
    'target_zero_preserving': True,
    'split_mode': 'timestamp',
    'history_hours': 96,
    'active_nodes_fit_on_train': True,
    'use_edge_weights': True,

    # Data
    'filename': 'Amatrice_CAT5.v20210504.csv',
    'grid_size': 0.015,           # ~1.67 km latitude per cell
    'time_bin': '4h',             # Four-hour aggregation bins
    
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
    
    # Sequence (4-hour bins)
    'window_size': 24,          # 24 bins = 96 hours of history
    'horizon': 6,               # 6 bins = 24 hours ahead
    'train_ratio': 0.7,
    'val_ratio': 0.15,
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
    
    # Adjacency Matrix. The previous 15 km / sigma 150 km graph connected
    # 64% of all node pairs and made distance weights almost uniform.
    'radius_km': 5.0,
    'sigma_km': 3.0,
    
    # Model
    'hidden_dim': 64,
    'num_gat_layers': 2,
    'num_heads': 4,
    'dropout': 0.2,
    'temporal_model': 'multiscale',   # Options: 'lstm', 'tft', or 'multiscale'
    'tft_layers': 3,            # Number of TFT encoder layers
    
    # Multi-Scale Model Options
    'multiscale_scales': [1, 2, 4],  # Temporal scales (1=full, 2=half, 4=quarter resolution)
    'multiscale_fusion': 'concat',   # Fusion type: 'concat', 'attention', or 'gate'
    
    # Training
    'batch_size': 8,
    'epochs': 100,
    'learning_rate': 3e-4,
    'early_stopping_patience': 15,
    'weight_decay': 1e-4,
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'min_learning_rate': 1e-6,
    'gradient_clip_norm': 1.0,
    
    # Weighted Loss
    # Legacy weighted losses. The canonical hurdle objective below does not
    # use dynamic active weighting because it biases every point forecast
    # upward when train/test activity prevalence differs.
    'active_weight': 1.0,
    'max_dynamic_active_weight': 60.0,
    # Additional non-cumulative weights for increasingly rare magnitudes.
    # These thresholds are interpreted in raw Mw units by SparseAwareLoss.
    'magnitude_event_thresholds': [1.0, 2.0, 3.0],
    'magnitude_event_weights': [2.0, 5.0, 10.0],
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
    # Options: 'hurdle', 'weighted_mse', 'asymmetric', 'active_only',
    #          'sparse_aware', 'focal', 'multiscale'
    'loss_type': 'hurdle',
    # The canonical model has one primary raw-Mw target and one auxiliary
    # activity logit. Primary regression metrics use the expected point
    # forecast P(activity) * E[Mw | activity].
    'model_output_features': 2,
    'hurdle_primary_prediction': 'expected',
    'hurdle_activity_threshold': 0.5,
    'hurdle_calibrate_activity_bias_on_validation': True,
    'hurdle_activity_loss_weight': 1.0,
    'hurdle_magnitude_loss_weight': 1.0,
    'hurdle_expected_value_loss_weight': 1.0,
    'hurdle_smooth_l1_beta': 0.5,
    # Keep None for probability calibration. Class-balancing the BCE would
    # change the probability scale and recreate the positive-forecast bias.
    'hurdle_activity_pos_weight': None,
    # Rare-event weighting is deliberately disabled for the primary run.
    # Threshold skill is reported separately instead of distorting RMSE/R2.
    'hurdle_magnitude_event_thresholds': [],
    'hurdle_magnitude_event_weights': [],
    'asymmetric_alpha': 0.8,      # Alpha for asymmetric loss (>0.5 = penalize underpredict more)
    'underpredict_penalty': 2.0,
    'magnitude_idx': 0,           # Index of max_mw in target features (now 0 since only max_mw)
    'focal_gamma': 2.0,           # Focal loss gamma (higher = more focus on hard examples)
    
    # Multi-Scale Loss Options
    'multiscale_horizons': [1, 3, 6],       # 4h, 12h, and 24h with 4-hour bins
    'multiscale_weights': [1.0, 1.5, 2.0],
    
    # Feature Transformation
    # These are input-only transformations. The forecast target is never
    # transformed and is stored separately as raw_bin_max_mw.
    'transform_magnitude': False,
    'add_activity_mask': True,
    'add_running_features': True,
    'running_feature_window': 6,
    # Optional exploratory features; disabled in the canonical rerun because
    # they are low-priority and substantially slow the catalog preprocessing.
    'add_engineered_features': False,
    
    # Uncertainty Estimation (Deep Ensembles)
    'n_ensemble_models': 5,
    'uncertainty_enabled': True,
    
    # Optional Features
    'hyperparameter_tuning': False,  # Set True untuk Optuna tuning
    'cross_validation': False,       # Set True untuk Time Series CV
    'n_cv_splits': 5,
    
    # Output
    'output_dir': 'outputs/revision_canonical_v3',
    'save_figures': True,
    'save_attention': True,
    
    # Random Seed
    'seed': 42,
}

# Keep an explicit activity indicator in the canonical input tensor.
if CONFIG['add_activity_mask'] and 'activity_mask' not in CONFIG['features']:
    CONFIG['features'] = CONFIG['features'] + ['activity_mask']

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
