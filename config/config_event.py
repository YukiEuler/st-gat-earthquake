# ==============================================================================
# CONFIG_EVENT.PY - Configuration for Event-based Approach
# ==============================================================================

CONFIG_EVENT = {
    # Data
    'filename': 'Amatrice_CAT5.v20210504.csv',
    'grid_size': 0.1,           # Spatial grid resolution
    
    # Spatial Bounds Filter (set to None to use full data extent)
    'lat_min': None,            # e.g., 42.5
    'lat_max': None,            # e.g., 43.0
    'lon_min': None,            # e.g., 13.0
    'lon_max': None,            # e.g., 13.5
    
    # Event Filtering
    'min_magnitude': 1.0,       # Minimum magnitude to include (filter small events)
    'max_events': None,         # Maximum events to use (None = all)
    
    # Event-based Preprocessing
    'aggregation_type': 'event',  # 'time' or 'event'
    'events_per_step': 10,        # Events to aggregate (if using aggregation)
    
    # Sequence Parameters
    'window_size': 50,          # Number of past events as input
    'horizon': 1,               # Number of future events to predict
    'use_node_sequences': False, # If True, use per-node sequences (for GAT)
    'train_ratio': 0.8,
    
    # Features
    'input_features': [
        'mw',           # Magnitude
        'dep',          # Depth
        'log_energy',   # Log of seismic energy
        'delta_t',      # Time since previous event
        'delta_dist',   # Distance from previous event
        'delta_mw',     # Change in magnitude
        'lat',          # Latitude
        'lon',          # Longitude
        'node_id',      # Grid cell ID
    ],
    
    'target_features': [
        'mw',           # Predict next event magnitude
        'dep',          # Predict next event depth
        'log_energy',   # Predict next event energy
        'delta_t',      # Predict time until next event
    ],
    
    # Adjacency Matrix
    'radius_km': 50.0,          # Radius for spatial connectivity
    'sigma_km': 15.0,           # Gaussian decay parameter
    
    # Model
    'model_type': 'sequence',   # 'sequence' (LSTM/TFT) or 'graph' (GAT+LSTM)
    'hidden_dim': 64,
    'num_layers': 2,
    'num_heads': 4,
    'dropout': 0.2,
    
    # Training
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'weight_decay': 1e-5,
    
    # Output
    'output_dir': 'outputs_event',
    'save_model': True,
}

# Feature descriptions
EVENT_FEATURE_DEFINITIONS = {
    'mw': {
        'name': 'Magnitude',
        'description': 'Earthquake magnitude (Mw)',
        'range': '0-6',
    },
    'dep': {
        'name': 'Depth',
        'description': 'Hypocenter depth in km',
        'range': '0-30',
    },
    'log_energy': {
        'name': 'Log Energy',
        'description': 'Log of seismic energy (Joules)',
        'range': '4.8-14',
    },
    'delta_t': {
        'name': 'Inter-event Time',
        'description': 'Hours since previous event',
        'range': '0-∞',
    },
    'delta_dist': {
        'name': 'Inter-event Distance',
        'description': 'Distance from previous event (km)',
        'range': '0-100+',
    },
    'delta_mw': {
        'name': 'Magnitude Change',
        'description': 'Change in magnitude from previous event',
        'range': '-6 to 6',
    },
}

# Print config
def print_event_config():
    print("\n" + "=" * 60)
    print(" EVENT-BASED CONFIGURATION")
    print("=" * 60)
    for k, v in CONFIG_EVENT.items():
        if not isinstance(v, list):
            print(f"   {k}: {v}")
    print("=" * 60)
