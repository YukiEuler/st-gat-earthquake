# ============================================================================
# PREPROCESSING.PY - Leakage-safe earthquake preprocessing
# ============================================================================

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


class DataPreprocessor:
    """Load a catalog and build leakage-safe spatio-temporal tensors.

    The input tensor and the forecast target are deliberately separate:
    ``X`` may contain historical/rolling context, while ``target_raw`` is
    always the raw value of the future bin (for example, the raw maximum Mw).
    All normalizers are fitted on the training period only.
    """

    BASE_FEATURES = [
        'count', 'max_mw', 'log_energy', 'avg_depth', 'min_depth',
        'std_mw', 'avg_ml', 'std_ml', 'avg_error', 'ml_n_sum'
    ]

    def __init__(self, config):
        self.config = config
        self.grid_size = config['grid_size']
        self.time_bin = config['time_bin']
        self.features = list(config.get('features', ['count', 'max_mw', 'log_energy', 'avg_depth']))
        self.target_features = list(config.get('target_features', ['max_mw']))

        self.node_info = None
        self.feature_stats = None
        self.target_stats = None
        self.grid_params = None
        self.split_timestamps = None

    def load_data(self, filepath):
        """Load and parse the earthquake catalog."""
        print(f" Loading data: {filepath}")
        df = pd.read_csv(filepath)
        required = {'year', 'month', 'day', 'hour', 'minute', 'second', 'lat', 'lon', 'mw', 'dep'}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Catalog is missing required columns: {missing}")

        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
        df = df.sort_values('datetime').reset_index(drop=True)
        df['mw'] = pd.to_numeric(df['mw'], errors='coerce').fillna(0.0)
        df['dep'] = pd.to_numeric(df['dep'], errors='coerce')
        df['dep'] = df['dep'].fillna(df['dep'].median()).fillna(0.0)

        # Event-level log10 energy. Temporal aggregation converts this back to
        # linear energy before taking log10, so it represents total energy.
        df['log_energy'] = 4.8 + 1.5 * df['mw']

        if 'EH1' in df.columns and 'EH2' in df.columns:
            df['avg_error'] = (
                pd.to_numeric(df['EH1'], errors='coerce').fillna(0.0) +
                pd.to_numeric(df['EH2'], errors='coerce').fillna(0.0)
            ) / 2.0
        else:
            df['avg_error'] = 0.0

        if 'ml_mean' in df.columns:
            df['ml_mean'] = pd.to_numeric(df['ml_mean'], errors='coerce').fillna(df['mw'])
        else:
            df['ml_mean'] = df['mw']
        if 'ml_n' in df.columns:
            df['ml_n'] = pd.to_numeric(df['ml_n'], errors='coerce').fillna(1.0)
        else:
            df['ml_n'] = 1.0

        print(f"   Total Events: {len(df):,}")
        print(f"   Period: {df['datetime'].min()} - {df['datetime'].max()}")
        print(f"   Magnitude Range: {df['mw'].min():.2f} - {df['mw'].max():.2f}")
        return df

    def _split_timestamps_for_catalog(self, df):
        """Return canonical calendar split boundaries for the catalog."""
        configured = self.config.get('split_timestamps')
        if configured:
            return {key: pd.Timestamp(value).isoformat() for key, value in configured.items()}

        start = pd.Timestamp(df['datetime'].min())
        end = pd.Timestamp(df['datetime'].max())
        duration = end - start
        train_end = start + duration * float(self.config.get('train_ratio', 0.7))
        val_end = start + duration * float(
            self.config.get('train_ratio', 0.7) + self.config.get('val_ratio', 0.15)
        )
        return {
            'start': start.isoformat(),
            'train_end': train_end.isoformat(),
            'val_end': val_end.isoformat(),
            'end': end.isoformat(),
        }

    def create_spatial_grid(self, df):
        """Create the grid and fit active-node selection on train data only."""
        print(" Creating spatial grid...")
        lat_min = self.config.get('lat_min')
        lat_max = self.config.get('lat_max')
        lon_min = self.config.get('lon_min')
        lon_max = self.config.get('lon_max')
        lat_min = lat_min if lat_min is not None else df['lat'].min() - 0.01
        lat_max = lat_max if lat_max is not None else df['lat'].max() + 0.01
        lon_min = lon_min if lon_min is not None else df['lon'].min() - 0.01
        lon_max = lon_max if lon_max is not None else df['lon'].max() + 0.01

        original_len = len(df)
        df = df[
            (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
            (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
        ].copy()
        if len(df) == 0:
            raise ValueError('No catalog events remain after spatial bounds filtering.')
        if len(df) != original_len:
            print(f"   Filtered to bounds: {len(df):,} / {original_len:,} events")

        n_rows = int(np.ceil((lat_max - lat_min) / self.grid_size))
        n_cols = int(np.ceil((lon_max - lon_min) / self.grid_size))
        total_nodes = n_rows * n_cols
        df['row_idx'] = ((df['lat'] - lat_min) / self.grid_size).astype(int).clip(0, n_rows - 1)
        df['col_idx'] = ((df['lon'] - lon_min) / self.grid_size).astype(int).clip(0, n_cols - 1)
        df['node_id_full'] = df['row_idx'] * n_cols + df['col_idx']

        split_timestamps = self._split_timestamps_for_catalog(df)
        self.split_timestamps = split_timestamps

        use_active = self.config.get('use_active_nodes_only', True)
        fit_on_train = self.config.get('active_nodes_fit_on_train', True)
        if use_active:
            if fit_on_train:
                train_cutoff = pd.Timestamp(split_timestamps['train_end'])
                fit_df = df[df['datetime'] < train_cutoff]
                selection_source = 'train_period_only'
            else:
                fit_df = df
                selection_source = 'full_catalog_legacy'

            node_activity = fit_df.groupby('node_id_full').size()
            active_nodes = np.sort(
                node_activity[node_activity >= self.config.get('min_events_per_node', 1)].index.values
            )
            if len(active_nodes) == 0:
                raise ValueError(
                    'No active nodes meet min_events_per_node in the training period. '
                    'Lower the threshold or check the spatial bounds.'
                )

            full_to_reduced = {int(full_id): idx for idx, full_id in enumerate(active_nodes)}
            reduced_to_full = {idx: int(full_id) for full_id, idx in full_to_reduced.items()}
            df['node_id'] = df['node_id_full'].map(full_to_reduced)
            df = df.dropna(subset=['node_id']).copy()
            df['node_id'] = df['node_id'].astype(int)
            num_nodes = len(active_nodes)
            self.node_info = {
                'full_to_reduced': full_to_reduced,
                'reduced_to_full': reduced_to_full,
                'active_node_ids': active_nodes.astype(int),
                'active_node_counts_train': {
                    str(int(node)): int(node_activity.get(node, 0)) for node in active_nodes
                },
                'active_nodes_fit_on_train': bool(fit_on_train),
                'node_selection_source': selection_source,
                'node_selection_cutoff': split_timestamps['train_end'],
                'n_rows': n_rows,
                'n_cols': n_cols,
            }
            print(f"   Grid dimensions: {n_rows} x {n_cols} = {total_nodes} total cells")
            print(f"   Active nodes (fit on {selection_source}): {num_nodes} / {total_nodes}")
        else:
            df['node_id'] = df['node_id_full'].astype(int)
            num_nodes = total_nodes
            self.node_info = {
                'active_nodes_fit_on_train': False,
                'node_selection_source': 'all_grid_nodes',
                'n_rows': n_rows,
                'n_cols': n_cols,
            }

        self.grid_params = {
            'lat_min': float(lat_min), 'lat_max': float(lat_max),
            'lon_min': float(lon_min), 'lon_max': float(lon_max),
            'n_rows': n_rows, 'n_cols': n_cols, 'num_nodes': num_nodes,
        }
        return df, num_nodes

    @staticmethod
    def _aggregate_events(df):
        """Aggregate one event table without applying input-only transforms."""
        if len(df) == 0:
            return pd.DataFrame(columns=[
                'time_idx', 'node_id', 'count', 'max_mw', 'std_mw', 'log_energy',
                'avg_depth', 'min_depth', 'avg_ml', 'std_ml', 'avg_error', 'ml_n_sum'
            ])

        work = df.copy()
        work['energy_linear'] = np.power(10.0, work['log_energy'].clip(upper=30.0))
        grouped = work.groupby(['time_idx', 'node_id']).agg({
            'mw': ['count', 'max', 'std'],
            'energy_linear': 'sum',
            'dep': ['mean', 'min'],
            'ml_mean': ['mean', 'std'],
            'avg_error': 'mean',
            'ml_n': 'sum'
        }).reset_index()
        grouped.columns = [
            'time_idx', 'node_id', 'count', 'max_mw', 'std_mw', 'energy_linear',
            'avg_depth', 'min_depth', 'avg_ml', 'std_ml', 'avg_error', 'ml_n_sum'
        ]
        grouped['log_energy'] = np.log10(grouped['energy_linear'].clip(lower=1e-12))
        grouped = grouped.drop(columns=['energy_linear'])
        for column in ['std_mw', 'std_ml']:
            grouped[column] = grouped[column].fillna(0.0)
        return grouped

    def create_temporal_features(self, df, num_nodes):
        """Build input features and an untransformed future-target tensor."""
        print(" Creating temporal features...")
        time_delta = pd.Timedelta(self.time_bin)
        time_start = pd.Timestamp(df['datetime'].min()).floor(self.time_bin)
        time_end = pd.Timestamp(df['datetime'].max())
        n_timesteps = int(np.floor((time_end - time_start) / time_delta)) + 1
        time_index = pd.date_range(start=time_start, periods=n_timesteps, freq=self.time_bin)

        work = df.copy()
        work['time_idx'] = ((work['datetime'] - time_start) / time_delta).astype(int)
        work = work[(work['time_idx'] >= 0) & (work['time_idx'] < n_timesteps)].copy()
        grouped = self._aggregate_events(work)

        feature_names = list(self.features)
        X = np.zeros((n_timesteps, num_nodes, len(feature_names)), dtype=np.float32)
        target_raw = np.zeros((n_timesteps, num_nodes, len(self.target_features)), dtype=np.float32)
        mapping = {
            'count': 'count', 'max_mw': 'max_mw', 'log_energy': 'log_energy',
            'avg_depth': 'avg_depth', 'min_depth': 'min_depth', 'std_mw': 'std_mw',
            'avg_ml': 'avg_ml', 'std_ml': 'std_ml', 'avg_error': 'avg_error',
            'ml_n_sum': 'ml_n_sum'
        }

        if len(grouped):
            t_idx = grouped['time_idx'].to_numpy(dtype=int)
            n_idx = grouped['node_id'].to_numpy(dtype=int)
            for i, feat in enumerate(self.features):
                if feat in mapping:
                    values = grouped[mapping[feat]].to_numpy(dtype=np.float32)
                    if feat == 'max_mw' and self.config.get('transform_magnitude', False):
                        values = np.log1p(values)
                    X[t_idx, n_idx, i] = values
            for i, target in enumerate(self.target_features):
                if target not in mapping:
                    raise ValueError(f"Unsupported target feature: {target}")
                # This is intentionally always the raw grouped value. It is
                # never affected by rolling, log1p, or running context.
                target_raw[t_idx, n_idx, i] = grouped[mapping[target]].to_numpy(dtype=np.float32)

        if 'activity_mask' in self.features:
            mask_idx = self.features.index('activity_mask')
            X[..., mask_idx] = (X[..., self.features.index('count')] > 0).astype(np.float32)

        rolling_config = self.config.get('rolling_aggregation', {})
        if rolling_config.get('enabled', False):
            rolling_window = int(rolling_config.get('window', 3))
            print(f"   Applying input rolling aggregation (window={rolling_window})...")
            for feat_name, agg_type in rolling_config.get('features', {}).items():
                if feat_name not in self.features:
                    continue
                feat_idx = self.features.index(feat_name)
                original = X[..., feat_idx].copy()
                for t in range(n_timesteps):
                    window_data = original[max(0, t - rolling_window + 1):t + 1]
                    if agg_type == 'max':
                        X[t, :, feat_idx] = window_data.max(axis=0)
                    elif agg_type == 'sum':
                        if feat_name == 'log_energy':
                            # log10(E1 + E2 + ...) rather than summing
                            # logarithms. Empty bins contribute zero energy.
                            linear = np.where(window_data > 0, np.power(10.0, window_data), 0.0)
                            total = linear.sum(axis=0)
                            X[t, :, feat_idx] = np.where(
                                total > 0, np.log10(np.maximum(total, 1e-12)), 0.0
                            )
                        else:
                            X[t, :, feat_idx] = window_data.sum(axis=0)
                    elif agg_type == 'mean':
                        X[t, :, feat_idx] = window_data.mean(axis=0)
                    elif agg_type == 'min':
                        X[t, :, feat_idx] = window_data.min(axis=0)
                print(f"      {feat_name}: {agg_type}")
        elif self.config.get('mw_rolling_window', 1) > 1 and 'max_mw' in self.features:
            mw_idx = self.features.index('max_mw')
            original = X[..., mw_idx].copy()
            window = int(self.config['mw_rolling_window'])
            for t in range(n_timesteps):
                X[t, :, mw_idx] = original[max(0, t - window + 1):t + 1].max(axis=0)
            print(f"   Applied input rolling max (window={window}) to max_mw")

        if self.config.get('add_running_features', False) and 'max_mw' in self.features:
            mw_idx = self.features.index('max_mw')
            window = int(self.config.get('running_feature_window', 6))
            running_max = np.zeros((n_timesteps, num_nodes), dtype=np.float32)
            raw_mw = X[..., mw_idx].copy()
            for t in range(n_timesteps):
                running_max[t] = raw_mw[max(0, t - window + 1):t + 1].max(axis=0)
            X = np.concatenate([X, running_max[..., None]], axis=-1)
            feature_names.append('running_max_mw')
            print(f"   Added separate running_max_mw feature (window={window})")

        if self.config.get('add_engineered_features', False):
            print("   Computing engineered features...")
            b_values = np.ones((n_timesteps, num_nodes), dtype=np.float32)
            event_rates = np.zeros((n_timesteps, num_nodes), dtype=np.float32)
            time_since_last = np.zeros((n_timesteps, num_nodes), dtype=np.float32)
            eng_window = int(self.config.get('engineered_window', 24))
            for node in tqdm(range(num_nodes), desc="   Eng. features", leave=False):
                node_events = work[work['node_id'] == node]
                for t in range(n_timesteps):
                    start_t = max(0, t - eng_window)
                    recent = node_events[(node_events['time_idx'] >= start_t) & (node_events['time_idx'] <= t)]
                    event_rates[t, node] = min(len(recent) / max(1, t - start_t + 1), 10.0)
                    past = node_events[node_events['time_idx'] < t]
                    time_since_last[t, node] = t - int(past['time_idx'].max()) if len(past) else t
                    mags = recent['mw'].to_numpy()
                    mags = mags[mags > 0]
                    if len(mags) >= 5 and mags.mean() > mags.min():
                        b_values[t, node] = np.log10(np.e) / (mags.mean() - mags.min() + 0.1)
            eng_array = np.stack([
                np.clip(b_values, 0.5, 2.5),
                np.clip(event_rates, 0, 10),
                np.clip(time_since_last / 24.0, 0, 10)
            ], axis=-1)
            X = np.concatenate([X, eng_array], axis=-1)
            feature_names.extend(['b_value', 'event_rate', 'time_since_last'])

        print(f"   Timesteps: {n_timesteps:,}")
        print(f"   Features: {feature_names}")
        print(f"   Input shape: {X.shape}; raw target shape: {target_raw.shape}")
        return X, target_raw, n_timesteps, feature_names, time_index

    @staticmethod
    def _fit_stats(values, preserve_zero=False):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean = np.nanmean(values, axis=(0, 1)).astype(np.float32)
        std = np.nanstd(values, axis=(0, 1)).astype(np.float32)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std[std == 0] = 1.0
        return {
            'mean': mean,
            'std': std,
            # Sparse targets use a zero-preserving scale so loss functions can
            # still identify inactive future bins after normalization.
            'offset': np.zeros_like(mean) if preserve_zero else mean.copy(),
            'zero_preserving': bool(preserve_zero),
        }

    @staticmethod
    def _normalize(values, stats):
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = (values - stats.get('offset', stats['mean'])) / stats['std']
        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def _split_indices(self, time_index):
        timestamps = self.split_timestamps or {}
        if self.config.get('split_mode', 'timestamp') == 'timestamp' and timestamps.get('train_end'):
            train_end = int(np.searchsorted(time_index.values, np.datetime64(timestamps['train_end']), side='left'))
            val_end = int(np.searchsorted(time_index.values, np.datetime64(timestamps['val_end']), side='left'))
        else:
            train_end = int(len(time_index) * self.config.get('train_ratio', 0.7))
            val_end = int(len(time_index) * (
                self.config.get('train_ratio', 0.7) + self.config.get('val_ratio', 0.15)
            ))
        train_end = max(1, min(train_end, len(time_index) - 2))
        val_end = max(train_end + 1, min(val_end, len(time_index) - 1))
        return train_end, val_end

    def train_val_test_split(self, values, time_index=None, return_metadata=False):
        """Split without shuffling; optionally use canonical timestamps."""
        if time_index is None:
            n = len(values)
            train_end = int(n * self.config.get('train_ratio', 0.7))
            val_end = int(n * (self.config.get('train_ratio', 0.7) + self.config.get('val_ratio', 0.15)))
        else:
            train_end, val_end = self._split_indices(time_index)
        parts = (values[:train_end], values[train_end:val_end], values[val_end:])
        if return_metadata:
            return parts + ({
                'train_start_idx': 0, 'train_end_idx': train_end,
                'val_end_idx': val_end,
                'train_start': pd.Timestamp(time_index[0]).isoformat() if time_index is not None else None,
                'train_end': pd.Timestamp(time_index[train_end]).isoformat() if time_index is not None else None,
                'val_end': pd.Timestamp(time_index[val_end]).isoformat() if time_index is not None else None,
                'test_end': pd.Timestamp(time_index[-1]).isoformat() if time_index is not None else None,
            },)
        return parts

    def process(self, filepath):
        """Run the complete preprocessing pipeline."""
        df = self.load_data(filepath)
        df, num_nodes = self.create_spatial_grid(df)
        X_raw, target_raw, n_timesteps, feature_names, time_index = self.create_temporal_features(df, num_nodes)

        train_end, val_end = self._split_indices(time_index)
        X_train_raw = X_raw[:train_end]
        target_train_raw = target_raw[:train_end]
        self.feature_stats = self._fit_stats(X_train_raw)
        self.target_stats = self._fit_stats(
            target_train_raw,
            preserve_zero=self.config.get('target_zero_preserving', True)
        )
        X_norm = self._normalize(X_raw, self.feature_stats)
        target_norm = self._normalize(target_raw, self.target_stats)

        train_data, val_data, test_data = X_norm[:train_end], X_norm[train_end:val_end], X_norm[val_end:]
        train_target, val_target, test_target = (
            target_norm[:train_end], target_norm[train_end:val_end], target_norm[val_end:]
        )
        target_activity_rates = {
            'train': float(np.mean(np.abs(target_raw[:train_end]) > 0)),
            'val': float(np.mean(np.abs(target_raw[train_end:val_end]) > 0)),
            'test': float(np.mean(np.abs(target_raw[val_end:]) > 0)),
        }
        magnitude_threshold_rates = {}
        if 'max_mw' in self.target_features:
            magnitude_idx = self.target_features.index('max_mw')
            split_targets = {
                'train': target_raw[:train_end, ..., magnitude_idx],
                'val': target_raw[train_end:val_end, ..., magnitude_idx],
                'test': target_raw[val_end:, ..., magnitude_idx],
            }
            for threshold in self.config.get('magnitude_event_thresholds', [1.0, 2.0, 3.0]):
                magnitude_threshold_rates[str(threshold)] = {
                    split: float(np.mean(values >= threshold))
                    for split, values in split_targets.items()
                }
        self.split_timestamps.update({
            'train_start': pd.Timestamp(time_index[0]).isoformat(),
            'train_end_bin': pd.Timestamp(time_index[train_end]).isoformat(),
            'val_end_bin': pd.Timestamp(time_index[val_end]).isoformat(),
            'test_end_bin': pd.Timestamp(time_index[-1]).isoformat(),
        })

        print(" Normalization fitted on train period only.")
        print(f"   Input train mean: {self.feature_stats['mean']}")
        print(f"   Target train mean: {self.target_stats['mean']}")
        print(
            "   Target active rates: "
            f"train={target_activity_rates['train']*100:.2f}%, "
            f"val={target_activity_rates['val']*100:.2f}%, "
            f"test={target_activity_rates['test']*100:.2f}%"
        )
        for threshold, rates in magnitude_threshold_rates.items():
            print(
                f"   Mw >= {threshold}: train={rates['train']*100:.3f}%, "
                f"val={rates['val']*100:.3f}%, test={rates['test']*100:.3f}%"
            )
        print(f"   Split sizes: train={len(train_data):,}, val={len(val_data):,}, test={len(test_data):,}")

        return {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_target_data': train_target,
            'val_target_data': val_target,
            'test_target_data': test_target,
            'X_raw': X_raw,
            'target_raw': target_raw,
            'train_target_raw': target_train_raw,
            'val_target_raw': target_raw[train_end:val_end],
            'test_target_raw': target_raw[val_end:],
            'num_nodes': num_nodes,
            'n_timesteps': n_timesteps,
            'time_index': time_index,
            'split_timestamps': self.split_timestamps,
            'train_end_idx': train_end,
            'val_end_idx': val_end,
            'feature_names': feature_names,
            'target_features': self.target_features,
            'feature_stats': self.feature_stats,
            'target_stats': self.target_stats,
            'target_activity_rates': target_activity_rates,
            'magnitude_threshold_rates': magnitude_threshold_rates,
            'node_info': self.node_info,
            'grid_params': self.grid_params,
            'df': df,
        }
