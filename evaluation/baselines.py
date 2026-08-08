"""Leakage-safe reference forecasts for the canonical regression pipeline."""

import csv
from pathlib import Path

import numpy as np

from .metrics import MetricsCalculator


def _denormalize_target_series(target_series, target_stats):
    target_series = np.asarray(target_series, dtype=np.float32)
    scale = np.asarray(target_stats['std'], dtype=np.float32).reshape(1, 1, -1)
    offset = np.asarray(
        target_stats.get('offset', target_stats['mean']), dtype=np.float32
    ).reshape(1, 1, -1)
    return target_series * scale + offset


def _repeat_forecast(value_by_sample, horizon):
    return np.repeat(value_by_sample[:, None, :, :], int(horizon), axis=1)


def generate_reference_forecasts(
        test_target_series, train_target_series, target_stats,
        window_size, horizon, recent_window=6,
        test_activity_series=None, train_activity_series=None):
    """Build online baselines using only information available at origin time.

    ``test_target_series`` and ``train_target_series`` are the normalized raw
    target streams produced by preprocessing, not rolling input features.
    Consequently the persistence baseline is truly the last raw target bin.
    """
    test_raw = _denormalize_target_series(test_target_series, target_stats)
    train_raw = _denormalize_target_series(train_target_series, target_stats)
    window_size = int(window_size)
    horizon = int(horizon)
    recent_window = max(1, min(int(recent_window), window_size))
    n_samples = test_raw.shape[0] - window_size - horizon + 1
    if n_samples <= 0:
        raise ValueError("Target series is too short for the requested window/horizon.")

    test_activity = None
    if test_activity_series is not None:
        test_activity = (
            np.asarray(test_activity_series, dtype=np.float32) >= 0.5
        ).astype(np.float32)
        if test_activity.shape != test_raw.shape:
            raise ValueError("Test activity and target series must have equal shapes.")

    train_activity = None
    if train_activity_series is not None:
        train_activity = (
            np.asarray(train_activity_series, dtype=np.float32) >= 0.5
        ).astype(np.float32)
        if train_activity.shape != train_raw.shape:
            raise ValueError("Train activity and target series must have equal shapes.")

    origin_indices = np.arange(n_samples) + window_size
    persistence_value = test_raw[origin_indices - 1]
    recent_value = np.stack([
        test_raw[origin - recent_window:origin].mean(axis=0)
        for origin in origin_indices
    ])

    zero_value = np.zeros_like(persistence_value)
    train_node_mean = train_raw.mean(axis=0, keepdims=True)
    train_node_mean = np.repeat(train_node_mean, n_samples, axis=0)

    forecasts = {
        'zero': {
            'kind': 'baseline',
            'description': 'All-zero raw-Mw forecast.',
            'prediction': _repeat_forecast(zero_value, horizon),
        },
        'persistence_last_bin': {
            'kind': 'baseline',
            'description': 'Repeat the last observed raw target bin.',
            'prediction': _repeat_forecast(persistence_value, horizon),
        },
        'recent_mean_24h': {
            'kind': 'baseline',
            'description': (
                f'Repeat the mean raw target over the last {recent_window} bins.'
            ),
            'prediction': _repeat_forecast(recent_value, horizon),
        },
        'train_node_climatology': {
            'kind': 'baseline',
            'description': 'Per-node expected raw Mw fitted on the train split only.',
            'prediction': _repeat_forecast(train_node_mean, horizon),
        },
    }

    if test_activity is not None:
        persistence_activity = test_activity[origin_indices - 1]
        recent_activity = np.stack([
            test_activity[origin - recent_window:origin].mean(axis=0)
            for origin in origin_indices
        ])
        forecasts['zero']['activity_probability'] = _repeat_forecast(
            np.zeros_like(persistence_activity), horizon
        )
        forecasts['persistence_last_bin']['activity_probability'] = (
            _repeat_forecast(persistence_activity, horizon)
        )
        forecasts['recent_mean_24h']['activity_probability'] = (
            _repeat_forecast(recent_activity, horizon)
        )

    if train_activity is not None:
        activity_rate = train_activity.mean(axis=0, keepdims=True)
        active_count = train_activity.sum(axis=0, keepdims=True)
        active_sum = (train_raw * train_activity).sum(axis=0, keepdims=True)
        conditional_mean = np.divide(
            active_sum,
            active_count,
            out=np.zeros_like(active_sum),
            where=active_count > 0,
        )
        hurdle_expected = activity_rate * conditional_mean
        hurdle_expected = np.repeat(hurdle_expected, n_samples, axis=0)
        forecasts['train_node_hurdle_climatology'] = {
            'kind': 'baseline',
            'description': (
                'Per-node train activity rate times train active-bin mean Mw.'
            ),
            'prediction': _repeat_forecast(hurdle_expected, horizon),
            'activity_probability': _repeat_forecast(
                np.repeat(activity_rate, n_samples, axis=0), horizon
            ),
        }

    return forecasts


def evaluate_forecast_comparison(y_true, forecasts, activity_target=None,
                                 feature_names=None, magnitude_idx=0,
                                 primary_name='stgat_expected'):
    """Evaluate model views and baselines with one consistent metric policy."""
    calculator = MetricsCalculator(feature_names=feature_names)
    results = {}
    for name, specification in forecasts.items():
        prediction = np.asarray(specification['prediction'])
        if prediction.shape != np.asarray(y_true).shape:
            raise ValueError(
                f"Forecast {name!r} shape {prediction.shape} does not match "
                f"target shape {np.asarray(y_true).shape}."
            )
        entry = {
            'kind': specification.get('kind', 'baseline'),
            'description': specification.get('description', ''),
            'regression': calculator.calculate_regression_metrics(
                y_true, prediction, activity_mask=activity_target
            ),
            'diagnostics': calculator.calculate_forecast_diagnostics(
                y_true, prediction
            ),
            'per_horizon': calculator.calculate_per_horizon_metrics(
                y_true, prediction, activity_mask=activity_target
            ),
            'classification_mw1': calculator.calculate_classification_metrics(
                y_true, prediction, magnitude_idx=magnitude_idx, threshold=1.0
            ),
            'peak_detection': calculator.calculate_peak_detection_metrics(
                y_true, prediction
            ),
        }
        if activity_target is not None and specification.get(
            'activity_probability'
        ) is not None:
            activity_threshold = float(
                specification.get('activity_threshold', 0.5)
            )
            entry['activity_detection'] = (
                calculator.calculate_activity_metrics(
                    activity_target,
                    specification['activity_probability'],
                    threshold=activity_threshold,
                )
            )
        results[name] = entry

    if primary_name not in results:
        raise ValueError(f"Primary forecast {primary_name!r} is missing.")
    model_mse = results[primary_name]['regression']['MSE']
    for entry in results.values():
        reference_mse = entry['regression']['MSE']
        entry['primary_model_skill_vs_forecast'] = (
            float(1.0 - model_mse / reference_mse)
            if reference_mse > 0 else float('nan')
        )
    return results


def save_forecast_comparison_csv(results, path):
    """Write a compact paper-ready table from comparison JSON metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'forecast', 'kind', 'description', 'MSE', 'RMSE', 'MAE', 'R2',
        'Pearson_r', 'Bias', 'skill_vs_zero_forecast',
        'primary_model_skill_vs_forecast', 'mw1_precision', 'mw1_recall',
        'mw1_f1', 'peak_precision', 'peak_recall', 'peak_f1',
        'activity_threshold', 'activity_pr_auc', 'activity_precision',
        'activity_recall', 'activity_f1',
    ]
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, entry in results.items():
            regression = entry['regression']
            diagnostics = entry['diagnostics']
            mw1 = entry['classification_mw1']
            peak = entry['peak_detection']
            activity = entry.get('activity_detection', {})
            writer.writerow({
                'forecast': name,
                'kind': entry['kind'],
                'description': entry['description'],
                'MSE': regression['MSE'],
                'RMSE': regression['RMSE'],
                'MAE': regression['MAE'],
                'R2': regression['R2'],
                'Pearson_r': regression['Pearson_r'],
                'Bias': regression['Bias'],
                'skill_vs_zero_forecast': diagnostics['skill_vs_zero_forecast'],
                'primary_model_skill_vs_forecast': entry[
                    'primary_model_skill_vs_forecast'
                ],
                'mw1_precision': mw1['precision'],
                'mw1_recall': mw1['recall'],
                'mw1_f1': mw1['f1_score'],
                'peak_precision': peak['peak_precision'],
                'peak_recall': peak['peak_recall'],
                'peak_f1': peak['peak_f1'],
                'activity_threshold': activity.get('threshold', ''),
                'activity_pr_auc': activity.get('pr_auc', ''),
                'activity_precision': activity.get('precision', ''),
                'activity_recall': activity.get('recall', ''),
                'activity_f1': activity.get('f1_score', ''),
            })
    return path
