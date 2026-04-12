# ==============================================================================
# PREPROCESSING.PY - Data Loading and Preprocessing
# ==============================================================================

import pandas as pd
import numpy as np
from tqdm.auto import tqdm


class DataPreprocessor:
    """Handles data loading, gridding, and feature aggregation."""
    
    def __init__(self, config):
        self.config = config
        self.grid_size = config['grid_size']
        self.time_bin = config['time_bin']
        self.features = config.get('features', ['count', 'max_mw', 'log_energy', 'avg_depth'])
        
        # Will be set during preprocessing
        self.node_info = None
        self.feature_stats = None
        self.grid_params = None
        
    def load_data(self, filepath):
        """Load and parse earthquake catalog."""
        print(f" Loading data: {filepath}")
        df = pd.read_csv(filepath)
        
        # Parse datetime
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Calculate seismic energy (log scale)
        df['log_energy'] = 4.8 + 1.5 * df['mw']
        
        # Calculate average horizontal error (location quality indicator)
        # Handle NaN values!
        if 'EH1' in df.columns and 'EH2' in df.columns:
            df['avg_error'] = (df['EH1'].fillna(0) + df['EH2'].fillna(0)) / 2
        else:
            df['avg_error'] = 0.0
        
        # Use ml_mean if available, fill NaN with mw
        if 'ml_mean' in df.columns:
            df['ml_mean'] = df['ml_mean'].fillna(df['mw'])
        else:
            df['ml_mean'] = df['mw']
        
        # Use ml_n if available
        if 'ml_n' in df.columns:
            df['ml_n'] = df['ml_n'].fillna(1.0)
        else:
            df['ml_n'] = 1.0
        
        # Fill remaining NaN values for key columns
        df['mw'] = df['mw'].fillna(0)
        df['dep'] = df['dep'].fillna(df['dep'].median())
        
        print(f"   Total Events: {len(df):,}")
        print(f"   Period: {df['datetime'].min()} - {df['datetime'].max()}")
        print(f"   Magnitude Range: {df['mw'].min():.2f} - {df['mw'].max():.2f}")
        
        return df
    
    def create_spatial_grid(self, df):
        """Create spatial grid and map events to nodes."""
        print(" Creating spatial grid...")
        
        # Use config bounds if specified, otherwise use data extent
        lat_min = self.config.get('lat_min') or (df['lat'].min() - 0.01)
        lat_max = self.config.get('lat_max') or (df['lat'].max() + 0.01)
        lon_min = self.config.get('lon_min') or (df['lon'].min() - 0.01)
        lon_max = self.config.get('lon_max') or (df['lon'].max() + 0.01)
        
        # Filter data to bounds if config bounds are specified
        if self.config.get('lat_min') is not None or self.config.get('lon_min') is not None:
            original_len = len(df)
            df = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max) & 
                    (df['lon'] >= lon_min) & (df['lon'] <= lon_max)]
            print(f"   Filtered to bounds: {len(df):,} / {original_len:,} events ({len(df)/original_len*100:.1f}%)")
        
        print(f"   Bounds: lat [{lat_min:.4f}, {lat_max:.4f}], lon [{lon_min:.4f}, {lon_max:.4f}]")
        
        # Grid dimensions
        n_rows = int(np.ceil((lat_max - lat_min) / self.grid_size))
        n_cols = int(np.ceil((lon_max - lon_min) / self.grid_size))
        total_nodes = n_rows * n_cols
        
        # Map events to grid
        df['row_idx'] = ((df['lat'] - lat_min) / self.grid_size).astype(int).clip(0, n_rows - 1)
        df['col_idx'] = ((df['lon'] - lon_min) / self.grid_size).astype(int).clip(0, n_cols - 1)
        df['node_id_full'] = df['row_idx'] * n_cols + df['col_idx']
        
        # Active node filtering
        if self.config['use_active_nodes_only']:
            node_activity = df.groupby('node_id_full').size()
            active_nodes = node_activity[node_activity >= self.config['min_events_per_node']].index.values
            active_nodes = np.sort(active_nodes)
            
            full_to_reduced = {full_id: idx for idx, full_id in enumerate(active_nodes)}
            reduced_to_full = {idx: full_id for full_id, idx in full_to_reduced.items()}
            
            df['node_id'] = df['node_id_full'].map(full_to_reduced)
            df = df.dropna(subset=['node_id'])
            df['node_id'] = df['node_id'].astype(int)
            
            num_nodes = len(active_nodes)
            
            self.node_info = {
                'full_to_reduced': full_to_reduced,
                'reduced_to_full': reduced_to_full,
                'active_node_ids': active_nodes,
                'n_rows': n_rows,
                'n_cols': n_cols,
            }
            
            print(f"   Grid size: {self.grid_size}° ({self.grid_size * 111:.1f} km)")
            print(f"   Grid dimensions: {n_rows} x {n_cols} = {total_nodes} total cells")
            print(f"   Min events threshold: {self.config['min_events_per_node']}")
            print(f"   Nodes meeting threshold: {num_nodes} / {total_nodes}")
        else:
            df['node_id'] = df['node_id_full']
            num_nodes = total_nodes
            self.node_info = None
        
        self.grid_params = {
            'lat_min': lat_min, 'lat_max': lat_max,
            'lon_min': lon_min, 'lon_max': lon_max,
            'n_rows': n_rows, 'n_cols': n_cols,
            'num_nodes': num_nodes,
        }
        
        return df, num_nodes
    
    def create_temporal_features(self, df, num_nodes):
        """Aggregate events into spatio-temporal bins."""
        print(" Creating temporal features...")
        
        # Time range
        time_range = pd.date_range(
            start=df['datetime'].min().floor(self.time_bin),
            end=df['datetime'].max().ceil(self.time_bin),
            freq=self.time_bin
        )
        n_timesteps = len(time_range)
        
        # Map to time index
        time_bins = time_range.to_numpy()
        df['time_idx'] = np.searchsorted(time_bins, df['datetime'].values) - 1
        df = df[(df['time_idx'] >= 0) & (df['time_idx'] < n_timesteps)]
        
        # Determine number of features based on selection
        n_features = len(self.features)
        feature_names = self.features
        
        # Initialize tensor
        X = np.zeros((n_timesteps, num_nodes, n_features), dtype=np.float32)
        
        # Aggregate features - extended set
        grouped = df.groupby(['time_idx', 'node_id']).agg({
            'mw': ['count', 'max', 'std'],
            'log_energy': 'sum',
            'dep': ['mean', 'min'],
            'ml_mean': ['mean', 'std'],
            'avg_error': 'mean',
            'ml_n': 'sum'
        }).reset_index()
        
        grouped.columns = ['time_idx', 'node_id', 'count', 'max_mw', 'std_mw', 
                          'log_energy', 'avg_depth', 'min_depth',
                          'avg_ml', 'std_ml', 'avg_error', 'ml_n_sum']
        
        # Fill NaN values
        grouped['std_mw'] = grouped['std_mw'].fillna(0)
        grouped['std_ml'] = grouped['std_ml'].fillna(0)
        
        # ==================================================
        # FEATURE ENGINEERING FOR max_mw (Opsi 3)
        # ==================================================
        if self.config.get('transform_magnitude', False):
            # Log transform: log(mw + 1) to compress range and handle zeros
            # This helps model learn better when most values are small
            grouped['max_mw'] = np.log1p(grouped['max_mw'])
            print("   Applied log1p transform to max_mw")
        
        # Fill tensor based on selected features
        t_idx = grouped['time_idx'].values.astype(int)
        n_idx = grouped['node_id'].values.astype(int)
        
        feature_mapping = {
            'count': 'count',
            'max_mw': 'max_mw',
            'log_energy': 'log_energy',
            'avg_depth': 'avg_depth',
            'min_depth': 'min_depth',
            'std_mw': 'std_mw',
            'avg_ml': 'avg_ml',
            'std_ml': 'std_ml',
            'avg_error': 'avg_error',
            'ml_n_sum': 'ml_n_sum',
        }
        
        for i, feat in enumerate(self.features):
            if feat in feature_mapping:
                X[t_idx, n_idx, i] = grouped[feature_mapping[feat]].values
        
        # ==================================================
        # ROLLING AGGREGATION FOR ALL FEATURES
        # ==================================================
        rolling_config = self.config.get('rolling_aggregation', {})
        if rolling_config.get('enabled', False):
            rolling_window = rolling_config.get('window', 3)
            feature_agg_map = rolling_config.get('features', {})
            
            print(f"   Applying rolling aggregation (window={rolling_window})...")
            
            for feat_name, agg_type in feature_agg_map.items():
                if feat_name in self.features:
                    feat_idx = self.features.index(feat_name)
                    original_data = X[..., feat_idx].copy()
                    
                    for t in range(n_timesteps):
                        start_t = max(0, t - rolling_window + 1)
                        window_data = original_data[start_t:t+1, :]
                        
                        if agg_type == 'max':
                            X[t, :, feat_idx] = window_data.max(axis=0)
                        elif agg_type == 'sum':
                            X[t, :, feat_idx] = window_data.sum(axis=0)
                        elif agg_type == 'mean':
                            X[t, :, feat_idx] = window_data.mean(axis=0)
                        elif agg_type == 'min':
                            X[t, :, feat_idx] = window_data.min(axis=0)
                    
                    print(f"      {feat_name}: {agg_type}")
        else:
            # Fallback to old behavior: only max_mw rolling max
            rolling_window = self.config.get('mw_rolling_window', 3)
            if rolling_window > 1 and 'max_mw' in self.features:
                mw_idx = self.features.index('max_mw')
                original_mw = X[..., mw_idx].copy()
                
                for t in range(n_timesteps):
                    start_t = max(0, t - rolling_window + 1)
                    X[t, :, mw_idx] = original_mw[start_t:t+1, :].max(axis=0)
                
                print(f"   Applied rolling max (window={rolling_window}) to max_mw")
        
        # ==================================================
        # ENGINEERED FEATURES (b-value, rate, time since last)
        # ==================================================
        if self.config.get('add_engineered_features', False):
            print("   Computing engineered features...")
            
            # Find feature indices
            count_idx = self.features.index('count') if 'count' in self.features else None
            mw_idx = self.features.index('max_mw') if 'max_mw' in self.features else None
            
            # b-value estimation per node using rolling window
            # Gutenberg-Richter: log10(N) = a - b*M
            # We estimate b from recent events (higher b = fewer large quakes)
            if mw_idx is not None:
                b_values = np.zeros((n_timesteps, num_nodes), dtype=np.float32)
                event_rates = np.zeros((n_timesteps, num_nodes), dtype=np.float32)
                time_since_last = np.zeros((n_timesteps, num_nodes), dtype=np.float32)
                
                window_size = 24  # 24 timesteps for b-value calculation
                
                for node in tqdm(range(num_nodes), desc="   Eng. features", leave=False):
                    node_events = df[df['node_id'] == node].copy()
                    
                    for t in range(n_timesteps):
                        # Rolling window for b-value
                        start_t = max(0, t - window_size)
                        window_events = node_events[
                            (node_events['time_idx'] >= start_t) & 
                            (node_events['time_idx'] <= t)
                        ]
                        
                        # Event rate (events per timestep in window)
                        n_events = len(window_events)
                        event_rates[t, node] = n_events / (t - start_t + 1)
                        
                        # Time since last event at this node
                        past_events = node_events[node_events['time_idx'] < t]
                        if len(past_events) > 0:
                            last_event_t = past_events['time_idx'].max()
                            time_since_last[t, node] = t - last_event_t
                        else:
                            time_since_last[t, node] = t  # Max time if no prior event
                        
                        # b-value estimation (needs at least 5 events)
                        if n_events >= 5:
                            mags = window_events['mw'].values
                            mags = mags[mags > 0]  # Only positive magnitudes
                            if len(mags) >= 5:
                                # Maximum likelihood b-value: b = log10(e) / (mean(M) - Mmin)
                                m_min = mags.min()
                                m_mean = mags.mean()
                                if m_mean > m_min:
                                    b_values[t, node] = np.log10(np.e) / (m_mean - m_min + 0.1)
                                else:
                                    b_values[t, node] = 1.0  # Default b-value
                            else:
                                b_values[t, node] = 1.0
                        else:
                            b_values[t, node] = 1.0  # Default b-value
                
                # Normalize and clip values
                b_values = np.clip(b_values, 0.5, 2.5)  # Realistic b-value range
                event_rates = np.clip(event_rates, 0, 10)  # Cap at 10 events/timestep
                time_since_last = np.clip(time_since_last / 24, 0, 10)  # Normalize to days, cap at 10
                
                # Expand X to include new features if not already present
                # Store in separate arrays to add later
                self._engineered_features = {
                    'b_value': b_values,
                    'event_rate': event_rates,
                    'time_since_last': time_since_last
                }
                print(f"   Computed: b_value, event_rate, time_since_last")
        
        # ==================================================
        # RUNNING FEATURES (Opsi 3 - additional context)
        # ==================================================
        if self.config.get('add_running_features', False):
            # Find index of max_mw in features
            if 'max_mw' in self.features:
                mw_idx = self.features.index('max_mw')
                
                # Calculate running max over last 6 hours for better context
                window = 6
                running_max = np.zeros_like(X[..., mw_idx])
                
                for t in range(n_timesteps):
                    start_t = max(0, t - window + 1)
                    running_max[t] = np.maximum(running_max[max(0, t-1)] * 0.9, X[t, :, mw_idx])
                
                # Add running max as additional signal (scale and add to max_mw)
                # This gives the model memory of recent high magnitudes
                X[..., mw_idx] = X[..., mw_idx] + 0.3 * running_max
                print(f"   Added running max context (window={window}) to max_mw")
        
        # ==================================================
        # CONCATENATE ENGINEERED FEATURES TO X
        # ==================================================
        if self.config.get('add_engineered_features', False) and hasattr(self, '_engineered_features'):
            eng_feats = self._engineered_features
            
            # Stack engineered features: (T, N, 3)
            eng_array = np.stack([
                eng_feats['b_value'],
                eng_feats['event_rate'],
                eng_feats['time_since_last']
            ], axis=-1)
            
            # Concatenate to X: (T, N, F) -> (T, N, F+3)
            X = np.concatenate([X, eng_array], axis=-1)
            
            # Update feature names
            feature_names = list(feature_names) + ['b_value', 'event_rate', 'time_since_last']
            
            print(f"   Added engineered features: b_value, event_rate, time_since_last")
        
        print(f"   Timesteps: {n_timesteps:,}")
        print(f"   Features: {feature_names}")
        print(f"   Tensor shape: {X.shape}")
        
        return X, n_timesteps, feature_names
    
    def normalize(self, X):
        """Normalize features using Z-score (matches original working code)."""
        print(" Normalizing features...")
        
        # Handle any NaN in input tensor first
        X = np.nan_to_num(X, nan=0.0)
        
        self.feature_stats = {
            'mean': np.nanmean(X, axis=(0, 1)),
            'std': np.nanstd(X, axis=(0, 1))
        }
        
        # Avoid division by zero and handle NaN in stats
        self.feature_stats['std'] = np.nan_to_num(self.feature_stats['std'], nan=1.0)
        self.feature_stats['mean'] = np.nan_to_num(self.feature_stats['mean'], nan=0.0)
        self.feature_stats['std'][self.feature_stats['std'] == 0] = 1.0
        
        # Print stats for debugging
        print(f"   Feature means: {self.feature_stats['mean']}")
        print(f"   Feature stds: {self.feature_stats['std']}")
        
        X_norm = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        # Handle NaN/Inf
        X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_norm
    
    def denormalize(self, X_norm):
        """Denormalize features."""
        return X_norm * self.feature_stats['std'] + self.feature_stats['mean']
    
    def train_val_test_split(self, X_norm):
        """Time-based train/val/test split (no shuffle).
        
        Split proportions from config:
        - train_ratio: proportion for training (default 0.7)
        - val_ratio: proportion for validation (default 0.1)
        - test: remaining data (default 0.2)
        """
        train_ratio = self.config.get('train_ratio', 0.7)
        val_ratio = self.config.get('val_ratio', 0.1)
        
        n_samples = len(X_norm)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_data = X_norm[:train_end]
        val_data = X_norm[train_end:val_end]
        test_data = X_norm[val_end:]
        
        print(f"   Train samples: {len(train_data):,} ({train_ratio*100:.0f}%)")
        print(f"   Val samples: {len(val_data):,} ({val_ratio*100:.0f}%)")
        print(f"   Test samples: {len(test_data):,} ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return train_data, val_data, test_data
    
    def process(self, filepath):
        """Full preprocessing pipeline."""
        df = self.load_data(filepath)
        df, num_nodes = self.create_spatial_grid(df)
        X, n_timesteps, feature_names = self.create_temporal_features(df, num_nodes)
        X_norm = self.normalize(X)
        train_data, val_data, test_data = self.train_val_test_split(X_norm)
        
        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'X_raw': X,
            'num_nodes': num_nodes,
            'n_timesteps': n_timesteps,
            'feature_names': feature_names,
            'feature_stats': self.feature_stats,
            'node_info': self.node_info,
            'grid_params': self.grid_params,
            'df': df,
        }

