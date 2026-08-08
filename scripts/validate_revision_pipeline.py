"""Fast smoke test for the leakage-safe revision pipeline.

Run from the project root:
    python scripts/validate_revision_pipeline.py
"""

import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import DataPreprocessor
from data.dataset import SeismicDataset
from data.adjacency import AdjacencyBuilder
from models.baselines import NaiveBaseline
from models.gat_layer import SparseGATLayer
from training.losses import HurdleMagnitudeLoss
from training.hurdle import decode_hurdle_tensor, fit_activity_logit_bias


def make_catalog(path):
    rows = []
    start = pd.Timestamp('2020-01-01')
    # Node A is active throughout; node B appears only after the train period.
    for step in range(12):
        current = start + pd.Timedelta(hours=4 * step)
        rows.append({
            'year': current.year, 'month': current.month, 'day': current.day,
            'hour': current.hour, 'minute': 0, 'second': 0,
            'lat': 42.60, 'lon': 13.15,
            'mw': -0.2 if step == 0 else 1.0 + 0.1 * step, 'dep': 5.0,
            'EH1': 0.1, 'EH2': 0.1, 'ml_mean': 1.0, 'ml_n': 1,
        })
        if step >= 9:
            rows.append({
                'year': current.year, 'month': current.month, 'day': current.day,
                'hour': current.hour, 'minute': 30, 'second': 0,
                'lat': 42.75, 'lon': 13.25, 'mw': 3.5, 'dep': 6.0,
                'EH1': 0.1, 'EH2': 0.1, 'ml_mean': 3.5, 'ml_n': 1,
            })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        catalog = Path(tmp) / 'smoke.csv'
        make_catalog(catalog)
        config = {
            'grid_size': 0.05, 'time_bin': '4h',
            'lat_min': 42.55, 'lat_max': 42.85,
            'lon_min': 13.10, 'lon_max': 13.30,
            'features': ['count', 'max_mw', 'log_energy', 'avg_depth', 'activity_mask'],
            'target_features': ['max_mw'], 'target_definition': 'raw_bin_max_mw',
            'use_active_nodes_only': True, 'active_nodes_fit_on_train': True,
            'min_events_per_node': 1, 'train_ratio': 0.6, 'val_ratio': 0.2,
            'split_mode': 'timestamp', 'rolling_aggregation': {
                'enabled': True, 'window': 2, 'features': {'max_mw': 'max'}
            },
            'transform_magnitude': False, 'add_running_features': False,
            'add_engineered_features': False, 'target_zero_preserving': True,
        }
        data = DataPreprocessor(config).process(str(catalog))

        assert data['target_raw'].shape[-1] == 1
        assert data['target_activity_raw'].shape[-1] == 1
        assert data['target_activity_raw'][0, 0, 0] == 1.0
        assert data['target_raw'][0, 0, 0] < 0.0
        assert data['target_stats']['zero_preserving'] is True
        assert data['target_stats']['offset'][0] == 0
        assert data['node_info']['active_nodes_fit_on_train'] is True
        assert len(data['node_info']['active_node_ids']) == 1

        dataset = SeismicDataset(
            data['test_data'], target_data=data['test_target_data'],
            activity_data=data['test_target_activity'],
            window_size=1, horizon=1
        )
        x, y = dataset[0]
        assert x.shape[-1] == len(data['feature_names'])
        assert y.shape[-1] == 2

        # Hurdle loss must use the explicit activity channel and decode to a
        # one-channel point prediction without treating negative Mw as empty.
        raw_output = torch.zeros(*y.unsqueeze(0).shape[:-1], 2, requires_grad=True)
        criterion = HurdleMagnitudeLoss()
        loss = criterion(raw_output, y.unsqueeze(0))
        assert torch.isfinite(loss)
        loss.backward()
        decoded = decode_hurdle_tensor(raw_output)
        assert decoded['expected'].shape[-1] == 1
        calibration_targets = np.array([0, 0, 0, 1], dtype=np.float32)
        bias = fit_activity_logit_bias(
            np.zeros_like(calibration_targets), calibration_targets
        )
        calibrated_mean = 1.0 / (1.0 + np.exp(-bias))
        assert np.isclose(calibrated_mean, calibration_targets.mean(), atol=1e-6)

        # The baseline must select max_mw (input index 1), not the first input
        # channel (event count).
        baseline = NaiveBaseline(horizon=1, out_features=1, target_indices=[1])
        pred = baseline.predict(x.unsqueeze(0))
        assert pred.shape[-1] == 1

        # Weighted adjacency values must be accepted by the GAT layer.
        graph_cfg = dict(config, use_active_nodes_only=False, radius_km=100, sigma_km=100)
        graph_data = DataPreprocessor(graph_cfg).process(str(catalog))
        builder = AdjacencyBuilder(graph_cfg)
        adj = builder.build_distance_weighted_adj(
            graph_data['num_nodes'], graph_data['node_info'], graph_data['grid_params']
        )
        adj_t = builder.scipy_to_torch_sparse(adj)
        layer = SparseGATLayer(1, 2)
        out = layer(torch.zeros(1, graph_data['num_nodes'], 1), adj_t)
        assert out.shape == (1, graph_data['num_nodes'], 2)

    print('Revision pipeline smoke test: PASS')


if __name__ == '__main__':
    main()
