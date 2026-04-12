# ==============================================================================
# PREPROCESSING_EVENT.PY - Event-based Preprocessing
# ==============================================================================
# Hybrid approach: Grid for spatial structure + Event sequence per node
# 
# Differences from time-based:
# - Each step = N events (not 1 hour)
# - No zeros/sparse data (all steps have events)
# - Tracks inter-event features (delta_t, delta_dist)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class EventBasedPreprocessor:
    """
    Event-based preprocessing for seismic data.
    
    Instead of aggregating by time bins, we process events sequentially
    and create features based on event history per spatial node.
    """
    
    def __init__(self, grid_size=0.1, events_per_step=10, features=None,
                 min_magnitude=0.0, max_events=None,
                 lat_min=None, lat_max=None, lon_min=None, lon_max=None):
        """
        Args:
            grid_size: Spatial grid resolution in degrees
            events_per_step: Number of events to aggregate per step
            features: List of features to extract
            min_magnitude: Minimum magnitude to include (filter small events)
            max_events: Maximum number of events to use (None = all)
            lat_min/lat_max: Latitude bounds (None = use data extent)
            lon_min/lon_max: Longitude bounds (None = use data extent)
        """
        self.grid_size = grid_size
        self.events_per_step = events_per_step
        self.features = features or [
            'mw', 'dep', 'log_energy', 
            'delta_t', 'delta_dist', 'delta_mw'
        ]
        self.min_magnitude = min_magnitude
        self.max_events = max_events
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.feature_stats = None
        self.node_coords = None
    
    def load_data(self, filepath):
        """Load and parse earthquake catalog with optional filtering."""
        print(f" Loading data: {filepath}")
        df = pd.read_csv(filepath)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        original_count = len(df)
        
        # Filter by minimum magnitude
        if self.min_magnitude > 0:
            df = df[df['mw'] >= self.min_magnitude].reset_index(drop=True)
            print(f"   Filtered Mw >= {self.min_magnitude}: {original_count:,} → {len(df):,} events")
        
        # Limit number of events
        if self.max_events is not None and len(df) > self.max_events:
            df = df.tail(self.max_events).reset_index(drop=True)
            print(f"   Limited to last {self.max_events:,} events")
        
        # Calculate seismic energy (log scale)
        df['log_energy'] = 4.8 + 1.5 * df['mw']
        
        print(f"   Total Events: {len(df):,}")
        print(f"   Period: {df['datetime'].min()} - {df['datetime'].max()}")
        print(f"   Magnitude Range: {df['mw'].min():.2f} - {df['mw'].max():.2f}")
        
        return df
    
    def create_spatial_grid(self, df):
        """Create spatial grid and assign events to nodes."""
        print(" Creating spatial grid...")
        
        # Use config bounds if specified, otherwise use data extent
        lat_min = self.lat_min if self.lat_min is not None else (df['lat'].min() - self.grid_size)
        lat_max = self.lat_max if self.lat_max is not None else (df['lat'].max() + self.grid_size)
        lon_min = self.lon_min if self.lon_min is not None else (df['lon'].min() - self.grid_size)
        lon_max = self.lon_max if self.lon_max is not None else (df['lon'].max() + self.grid_size)
        
        # Filter data to bounds if config bounds are specified
        if self.lat_min is not None or self.lon_min is not None:
            original_len = len(df)
            df = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max) & 
                    (df['lon'] >= lon_min) & (df['lon'] <= lon_max)].reset_index(drop=True)
            print(f"   Filtered to bounds: {len(df):,} / {original_len:,} events ({len(df)/original_len*100:.1f}%)")
        
        print(f"   Bounds: lat [{lat_min:.4f}, {lat_max:.4f}], lon [{lon_min:.4f}, {lon_max:.4f}]")
        
        # Create grid
        lat_bins = np.arange(lat_min, lat_max + self.grid_size, self.grid_size)
        lon_bins = np.arange(lon_min, lon_max + self.grid_size, self.grid_size)
        
        # Assign events to grid cells
        df['lat_idx'] = np.digitize(df['lat'], lat_bins) - 1
        df['lon_idx'] = np.digitize(df['lon'], lon_bins) - 1
        df['node_id'] = df['lat_idx'] * (len(lon_bins) - 1) + df['lon_idx']
        
        # Store node coordinates (cell centers)
        self.node_coords = {}
        for node_id in df['node_id'].unique():
            node_events = df[df['node_id'] == node_id]
            self.node_coords[node_id] = (
                node_events['lat'].mean(),
                node_events['lon'].mean()
            )
        
        n_nodes = len(self.node_coords)
        print(f"   Grid: {len(lat_bins)-1} x {len(lon_bins)-1}")
        print(f"   Active nodes: {n_nodes}")
        
        return df, n_nodes
    
    def calculate_inter_event_features(self, df):
        """Calculate features based on event sequence."""
        print(" Calculating inter-event features...")
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Delta time (hours since previous event)
        df['delta_t'] = df['datetime'].diff().dt.total_seconds() / 3600
        df['delta_t'] = df['delta_t'].fillna(0)
        
        # Delta magnitude
        df['delta_mw'] = df['mw'].diff().fillna(0)
        
        # Delta distance (km from previous event)
        # Using Haversine approximation
        lat_prev = df['lat'].shift(1).fillna(df['lat'].iloc[0])
        lon_prev = df['lon'].shift(1).fillna(df['lon'].iloc[0])
        
        # Approximate km per degree
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(df['lat'].mean()))
        
        delta_lat_km = (df['lat'] - lat_prev) * km_per_deg_lat
        delta_lon_km = (df['lon'] - lon_prev) * km_per_deg_lon
        df['delta_dist'] = np.sqrt(delta_lat_km**2 + delta_lon_km**2)
        df['delta_dist'] = df['delta_dist'].fillna(0)
        
        # Log transform delta_t for better distribution
        df['log_delta_t'] = np.log1p(df['delta_t'])
        
        print(f"   Delta T range: {df['delta_t'].min():.2f} - {df['delta_t'].max():.2f} hours")
        print(f"   Delta Dist range: {df['delta_dist'].min():.2f} - {df['delta_dist'].max():.2f} km")
        
        return df
    
    def create_event_sequences(self, df, window_size=50, horizon=1):
        """
        Create event-based sequences for training.
        
        Args:
            df: DataFrame with events
            window_size: Number of past events to use as input
            horizon: Number of future events to predict
            
        Returns:
            X: Input sequences (samples, window_size, features)
            Y: Target sequences (samples, horizon, target_features)
        """
        print(f" Creating event sequences (window={window_size}, horizon={horizon})...")
        
        # Features to use
        feature_cols = ['mw', 'dep', 'log_energy', 'delta_t', 'delta_dist', 'delta_mw',
                        'lat', 'lon', 'node_id']
        
        # Target features
        target_cols = ['mw', 'dep', 'log_energy', 'delta_t']
        
        # Convert to numpy
        data = df[feature_cols].values.astype(np.float32)
        
        n_samples = len(data) - window_size - horizon + 1
        n_features = len(feature_cols)
        n_targets = len(target_cols)
        
        # Find indices of target columns
        target_indices = [feature_cols.index(col) for col in target_cols]
        
        # Create sequences
        X = np.zeros((n_samples, window_size, n_features), dtype=np.float32)
        Y = np.zeros((n_samples, horizon, n_targets), dtype=np.float32)
        
        for i in range(n_samples):
            X[i] = data[i:i + window_size]
            Y[i] = data[i + window_size:i + window_size + horizon][:, target_indices]
        
        print(f"   X shape: {X.shape}")
        print(f"   Y shape: {Y.shape}")
        print(f"   Feature cols: {feature_cols}")
        print(f"   Target cols: {target_cols}")
        
        return X, Y, feature_cols, target_cols
    
    def create_node_event_sequences(self, df, window_size=20, horizon=1):
        """
        Create event sequences per spatial node (Hybrid approach).
        
        Each node has its own sequence of events.
        Good for using GAT across spatial nodes.
        
        Returns:
            X: (samples, window_size, nodes, features)
            Y: (samples, horizon, nodes, features)
        """
        print(f" Creating node-based event sequences...")
        
        # Get unique nodes
        nodes = sorted(df['node_id'].unique())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)
        
        # Features per event
        feature_cols = ['mw', 'dep', 'log_energy', 'delta_t']
        n_features = len(feature_cols)
        
        # Group events by node
        node_events = {}
        for node_id in nodes:
            node_df = df[df['node_id'] == node_id].copy()
            node_events[node_id] = node_df[feature_cols].values.astype(np.float32)
        
        # Find minimum sequence length across all nodes
        min_len = min(len(v) for v in node_events.values())
        print(f"   Min events per node: {min_len}")
        
        # Determine number of samples
        n_samples = min_len - window_size - horizon + 1
        
        if n_samples < 10:
            # Fall back to global event sequence
            print("   Warning: Not enough events per node, using global sequence")
            return self.create_event_sequences(df, window_size, horizon)
        
        # Create arrays
        X = np.zeros((n_samples, window_size, n_nodes, n_features), dtype=np.float32)
        Y = np.zeros((n_samples, horizon, n_nodes, n_features), dtype=np.float32)
        
        for node_id, idx in node_to_idx.items():
            events = node_events[node_id]
            for i in range(n_samples):
                X[i, :, idx, :] = events[i:i + window_size]
                Y[i, :, idx, :] = events[i + window_size:i + window_size + horizon]
        
        print(f"   X shape: {X.shape}")
        print(f"   Y shape: {Y.shape}")
        print(f"   Nodes: {n_nodes}")
        
        return X, Y, feature_cols, feature_cols
    
    def normalize(self, X, Y=None):
        """Normalize features using Z-score."""
        print(" Normalizing features...")
        
        # Handle different input shapes
        if X.ndim == 3:  # (samples, window, features)
            self.feature_stats = {
                'mean': X.mean(axis=(0, 1)),
                'std': X.std(axis=(0, 1))
            }
        elif X.ndim == 4:  # (samples, window, nodes, features)
            self.feature_stats = {
                'mean': X.mean(axis=(0, 1, 2)),
                'std': X.std(axis=(0, 1, 2))
            }
        
        # Avoid division by zero
        self.feature_stats['std'] = np.where(
            self.feature_stats['std'] == 0, 1.0, self.feature_stats['std']
        )
        
        # Normalize
        X_norm = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        if Y is not None:
            Y_norm = (Y - self.feature_stats['mean'][:Y.shape[-1]]) / \
                     self.feature_stats['std'][:Y.shape[-1]]
            return X_norm, Y_norm
        
        return X_norm
    
    def build_adjacency_matrix(self, n_nodes, radius_km=50.0, sigma_km=15.0):
        """
        Build spatial adjacency matrix based on node coordinates.
        
        Uses Gaussian decay with distance.
        """
        print(" Building adjacency matrix...")
        
        if self.node_coords is None or len(self.node_coords) == 0:
            # Return identity if no coords
            return np.eye(n_nodes)
        
        # Get coordinates
        node_ids = sorted(self.node_coords.keys())
        coords = np.array([self.node_coords[n] for n in node_ids])
        
        # Compute pairwise distances (approximate km)
        km_per_deg = 111.0
        coords_km = coords * km_per_deg
        
        distances = cdist(coords_km, coords_km, metric='euclidean')
        
        # Gaussian decay
        adj = np.exp(-distances**2 / (2 * sigma_km**2))
        
        # Apply radius cutoff
        adj = np.where(distances <= radius_km, adj, 0)
        
        # Add self-loops
        np.fill_diagonal(adj, 1.0)
        
        # Row normalize
        row_sum = adj.sum(axis=1, keepdims=True)
        adj_norm = adj / np.where(row_sum > 0, row_sum, 1)
        
        print(f"   Adjacency shape: {adj_norm.shape}")
        print(f"   Avg connections: {(adj_norm > 0).sum(axis=1).mean():.1f}")
        
        return adj_norm.astype(np.float32)
    
    def preprocess(self, filepath, window_size=50, horizon=1, use_node_sequences=False):
        """
        Full preprocessing pipeline.
        
        Args:
            filepath: Path to earthquake CSV
            window_size: Number of events for input
            horizon: Number of events to predict
            use_node_sequences: If True, create per-node sequences (for GAT)
        
        Returns:
            Dictionary with all preprocessed data
        """
        # Load data
        df = self.load_data(filepath)
        
        # Create spatial grid
        df, n_nodes = self.create_spatial_grid(df)
        
        # Calculate inter-event features
        df = self.calculate_inter_event_features(df)
        
        # Create sequences
        if use_node_sequences and n_nodes > 1:
            X, Y, feature_names, target_names = self.create_node_event_sequences(
                df, window_size, horizon
            )
        else:
            X, Y, feature_names, target_names = self.create_event_sequences(
                df, window_size, horizon
            )
        
        # Build adjacency matrix
        adj = self.build_adjacency_matrix(n_nodes)
        
        # Normalize
        X_norm, Y_norm = self.normalize(X, Y)
        
        # Train/test split (80/20)
        split_idx = int(len(X_norm) * 0.8)
        
        result = {
            'train_X': X_norm[:split_idx],
            'train_Y': Y_norm[:split_idx],
            'test_X': X_norm[split_idx:],
            'test_Y': Y_norm[split_idx:],
            'adjacency': adj,
            'feature_names': feature_names,
            'target_names': target_names,
            'feature_stats': self.feature_stats,
            'n_nodes': n_nodes,
            'node_coords': self.node_coords,
            'raw_df': df,
        }
        
        print(f"\n Event-based Preprocessing Complete:")
        print(f"   Train samples: {len(result['train_X']):,}")
        print(f"   Test samples: {len(result['test_X']):,}")
        print(f"   Features: {feature_names}")
        print(f"   Targets: {target_names}")
        
        return result
