# ==============================================================================
# METRICS.PY - Evaluation Metrics
# ==============================================================================

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from pathlib import Path


class MetricsCalculator:
    """Calculate various evaluation metrics for predictions."""
    
    def __init__(self, feature_names=None):
        self.feature_names = feature_names or ['count', 'max_mw', 'log_energy', 'avg_depth']
    
    def calculate_regression_metrics(self, y_true, y_pred):
        """
        Calculate standard regression metrics.
        
        Args:
            y_true: Ground truth (B, H, N, F) or flattened
            y_pred: Predictions (B, H, N, F) or flattened
        
        Returns:
            Dictionary of metrics
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # R² (handle zero variance)
        if np.std(y_true_flat) > 0:
            r2 = r2_score(y_true_flat, y_pred_flat)
        else:
            r2 = float('nan')
        
        # R² for ACTIVE (non-zero) values only - more meaningful for sparse seismic data
        active_mask = y_true_flat != 0
        if active_mask.sum() > 10 and np.std(y_true_flat[active_mask]) > 0:
            r2_active = r2_score(y_true_flat[active_mask], y_pred_flat[active_mask])
        else:
            r2_active = float('nan')
        
        # MAPE (avoid division by zero)
        mask = y_true_flat != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
        else:
            mape = float('nan')
        
        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'R2_active': float(r2_active),  # R² on non-zero values only
            'MAPE': float(mape),
        }
    
    def calculate_per_feature_metrics(self, y_true, y_pred):
        """Calculate metrics for each feature separately."""
        results = {}
        
        n_features = y_true.shape[-1]
        
        for i in range(n_features):
            feat_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            results[feat_name] = self.calculate_regression_metrics(
                y_true[..., i], y_pred[..., i]
            )
        
        return results
    
    def calculate_per_horizon_metrics(self, y_true, y_pred):
        """Calculate metrics for each prediction horizon separately."""
        results = {}
        
        horizon = y_true.shape[1]
        
        for h in range(horizon):
            results[f'h{h+1}'] = self.calculate_regression_metrics(
                y_true[:, h, :, :], y_pred[:, h, :, :]
            )
        
        return results
    
    def calculate_coverage_probability(self, y_true, ci_lower, ci_upper):
        """
        Calculate coverage probability for uncertainty estimates.
        
        A well-calibrated 95% CI should cover ~95% of true values.
        """
        within_ci = (y_true >= ci_lower) & (y_true <= ci_upper)
        coverage = within_ci.mean()
        return float(coverage)
    
    def calculate_sharpness(self, ci_lower, ci_upper):
        """
        Calculate sharpness (average width of confidence intervals).
        
        Smaller is better (tighter uncertainty bounds).
        """
        width = ci_upper - ci_lower
        return float(np.mean(width))
    
    def calculate_uncertainty_metrics(self, y_true, y_pred_mean, y_pred_std):
        """Calculate metrics for uncertainty estimation."""
        # 95% CI
        ci_lower = y_pred_mean - 1.96 * y_pred_std
        ci_upper = y_pred_mean + 1.96 * y_pred_std
        
        coverage = self.calculate_coverage_probability(y_true, ci_lower, ci_upper)
        sharpness = self.calculate_sharpness(ci_lower, ci_upper)
        
        # Calibration error (difference from ideal 95% coverage)
        calibration_error = abs(coverage - 0.95)
        
        return {
            'coverage_95': coverage,
            'sharpness': sharpness,
            'calibration_error': calibration_error,
        }
    
    def calculate_peak_detection_metrics(self, y_true, y_pred, percentile=90):
        """
        Calculate metrics for detecting peak events (large earthquakes).
        
        Args:
            percentile: Threshold for "peak" events (default: top 10%)
        """
        threshold = np.percentile(y_true, percentile)
        
        true_peaks = (y_true >= threshold).flatten().astype(int)
        pred_peaks = (y_pred >= threshold).flatten().astype(int)
        
        precision = precision_score(true_peaks, pred_peaks, zero_division=0)
        recall = recall_score(true_peaks, pred_peaks, zero_division=0)
        f1 = f1_score(true_peaks, pred_peaks, zero_division=0)
        
        return {
            'peak_threshold': float(threshold),
            'peak_precision': float(precision),
            'peak_recall': float(recall),
            'peak_f1': float(f1),
        }
    
    def calculate_classification_metrics(self, y_true, y_pred, magnitude_idx=0, threshold=1.0):
        """
        Calculate classification metrics for earthquake detection.
        
        Converts regression predictions to binary classification:
        - Class 1 (Significant): max_mw >= threshold
        - Class 0 (Not significant): max_mw < threshold
        
        Args:
            y_true: Ground truth (B, H, N, F)
            y_pred: Predictions (B, H, N, F)
            magnitude_idx: Index of max_mw feature (default 1)
            threshold: Magnitude threshold for classification (default 1.0 Mw)
        
        Returns:
            Dictionary with classification metrics
        """
        from sklearn.metrics import accuracy_score, confusion_matrix
        
        # Extract max_mw feature
        if y_true.ndim == 4:
            y_true_mw = y_true[..., magnitude_idx].flatten()
            y_pred_mw = y_pred[..., magnitude_idx].flatten()
        else:
            y_true_mw = y_true.flatten()
            y_pred_mw = y_pred.flatten()
        
        # Convert to binary classification
        y_true_class = (y_true_mw >= threshold).astype(int)
        y_pred_class = (y_pred_mw >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_class, y_pred_class)
        precision = precision_score(y_true_class, y_pred_class, zero_division=0)
        recall = recall_score(y_true_class, y_pred_class, zero_division=0)
        f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1]).ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        
        # Class distribution
        n_positive = y_true_class.sum()
        n_negative = len(y_true_class) - n_positive
        positive_rate = n_positive / len(y_true_class) if len(y_true_class) > 0 else 0
        
        return {
            'threshold_mw': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'n_positive_true': int(n_positive),
            'n_negative_true': int(n_negative),
            'positive_rate': float(positive_rate),
        }
    
    def calculate_window_classification_metrics(self, y_true, y_pred, magnitude_idx=0, 
                                                 threshold=1.0, window_size=4):
        """
        Calculate classification metrics using time-window aggregation.
        
        Instead of classifying each timestep individually, aggregate over windows:
        - If ANY timestep in window has max_mw >= threshold, classify as positive
        - More practical: "Will there be a significant earthquake in the next day?"
        
        Args:
            y_true: Ground truth (B, H, N, F) where H is horizon
            y_pred: Predictions (B, H, N, F)
            magnitude_idx: Index of max_mw feature (default 1)
            threshold: Magnitude threshold for classification (default 1.0 Mw)
            window_size: Number of timesteps to aggregate (4 = 1 day if 6h bins)
        
        Returns:
            Dictionary with window-aggregated classification metrics
        """
        from sklearn.metrics import accuracy_score, confusion_matrix
        
        # Extract max_mw feature: (B, H, N)
        if y_true.ndim == 4:
            y_true_mw = y_true[..., magnitude_idx]  # (B, H, N)
            y_pred_mw = y_pred[..., magnitude_idx]
        else:
            raise ValueError("Expected 4D input (B, H, N, F)")
        
        B, H, N = y_true_mw.shape
        
        # Number of complete windows
        n_windows = H // window_size
        if n_windows == 0:
            n_windows = 1
            window_size = H
        
        # Aggregate by taking MAX over each window
        # This answers: "Is the max magnitude in this window >= threshold?"
        window_results_true = []
        window_results_pred = []
        
        for b in range(B):
            for n in range(N):
                for w in range(n_windows):
                    start_h = w * window_size
                    end_h = min((w + 1) * window_size, H)
                    
                    # True: max magnitude in this window
                    true_max = y_true_mw[b, start_h:end_h, n].max()
                    pred_max = y_pred_mw[b, start_h:end_h, n].max()
                    
                    window_results_true.append(true_max)
                    window_results_pred.append(pred_max)
        
        y_true_windows = np.array(window_results_true)
        y_pred_windows = np.array(window_results_pred)
        
        # Binary classification based on threshold
        y_true_class = (y_true_windows >= threshold).astype(int)
        y_pred_class = (y_pred_windows >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_class, y_pred_class)
        precision = precision_score(y_true_class, y_pred_class, zero_division=0)
        recall = recall_score(y_true_class, y_pred_class, zero_division=0)
        f1 = f1_score(y_true_class, y_pred_class, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_class, y_pred_class, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Stats
        n_windows_total = len(y_true_class)
        n_positive = y_true_class.sum()
        positive_rate = n_positive / n_windows_total if n_windows_total > 0 else 0
        
        return {
            'window_size': int(window_size),
            'n_windows': int(n_windows_total),
            'threshold_mw': float(threshold),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'n_positive_windows': int(n_positive),
            'positive_rate': float(positive_rate),
        }
    
    def calculate_all_metrics(self, y_true, y_pred, y_pred_std=None, magnitude_idx=0):
        """Calculate all metrics."""
        results = {
            'overall': self.calculate_regression_metrics(y_true, y_pred),
            'per_feature': self.calculate_per_feature_metrics(y_true, y_pred),
            'per_horizon': self.calculate_per_horizon_metrics(y_true, y_pred),
            'peak_detection': self.calculate_peak_detection_metrics(y_true, y_pred),
        }
        
        # Classification metrics with different thresholds (per-timestep)
        results['classification_mw1'] = self.calculate_classification_metrics(
            y_true, y_pred, magnitude_idx=magnitude_idx, threshold=1.0
        )
        results['classification_mw2'] = self.calculate_classification_metrics(
            y_true, y_pred, magnitude_idx=magnitude_idx, threshold=2.0
        )
        results['classification_mw3'] = self.calculate_classification_metrics(
            y_true, y_pred, magnitude_idx=magnitude_idx, threshold=3.0
        )
        
        # Window-aggregated classification (daily window = 4 timesteps for 6h bins)
        # More practical: "Will there be a significant earthquake in the next day?"
        if y_true.ndim == 4:
            horizon = y_true.shape[1]
            # Use all timesteps as one window if horizon <= 4, else use 4
            window_size = min(4, horizon) if horizon <= 4 else 4
            
            results['window_cls_mw1'] = self.calculate_window_classification_metrics(
                y_true, y_pred, magnitude_idx=magnitude_idx, threshold=1.0, window_size=window_size
            )
            results['window_cls_mw2'] = self.calculate_window_classification_metrics(
                y_true, y_pred, magnitude_idx=magnitude_idx, threshold=2.0, window_size=window_size
            )
            results['window_cls_mw3'] = self.calculate_window_classification_metrics(
                y_true, y_pred, magnitude_idx=magnitude_idx, threshold=3.0, window_size=window_size
            )
        
        if y_pred_std is not None:
            results['uncertainty'] = self.calculate_uncertainty_metrics(
                y_true, y_pred, y_pred_std
            )
        
        return results
    
    def print_metrics(self, results):
        """Print metrics in a formatted way."""
        print("\n" + "=" * 70)
        print(" EVALUATION METRICS")
        print("=" * 70)
        
        print("\n Overall Metrics:")
        for metric, value in results['overall'].items():
            print(f"   {metric}: {value:.6f}")
        
        print("\n Per-Feature Metrics:")
        for feat, metrics in results['per_feature'].items():
            print(f"   {feat}:")
            for metric, value in metrics.items():
                print(f"      {metric}: {value:.6f}")
        
        if 'uncertainty' in results:
            print("\n Uncertainty Metrics:")
            for metric, value in results['uncertainty'].items():
                print(f"   {metric}: {value:.6f}")
        
        print("\n Peak Detection (Top 10%):")
        for metric, value in results['peak_detection'].items():
            print(f"   {metric}: {value:.6f}")
        
        # Classification metrics (per-timestep)
        for key in ['classification_mw1', 'classification_mw2', 'classification_mw3']:
            if key in results:
                cls = results[key]
                print(f"\n Classification Per-Timestep (Mw >= {cls['threshold_mw']}):")
                print(f"   Accuracy:    {cls['accuracy']:.4f}")
                print(f"   Precision:   {cls['precision']:.4f}")
                print(f"   Recall:      {cls['recall']:.4f}")
                print(f"   F1 Score:    {cls['f1_score']:.4f}")
                print(f"   Specificity: {cls['specificity']:.4f}")
                print(f"   TP: {cls['true_positives']:,} | TN: {cls['true_negatives']:,} | FP: {cls['false_positives']:,} | FN: {cls['false_negatives']:,}")
        
        # Window-aggregated classification (daily)
        for key in ['window_cls_mw1', 'window_cls_mw2', 'window_cls_mw3']:
            if key in results:
                wcls = results[key]
                print(f"\n Window Classification (Mw >= {wcls['threshold_mw']}, {wcls['window_size']} steps/window):")
                print(f"   Windows:     {wcls['n_windows']:,} ({wcls['n_positive_windows']:,} positive, {wcls['positive_rate']*100:.1f}%)")
                print(f"   Accuracy:    {wcls['accuracy']:.4f}")
                print(f"   Precision:   {wcls['precision']:.4f}")
                print(f"   Recall:      {wcls['recall']:.4f}")
                print(f"   F1 Score:    {wcls['f1_score']:.4f}")
                print(f"   TP: {wcls['true_positives']:,} | TN: {wcls['true_negatives']:,} | FP: {wcls['false_positives']:,} | FN: {wcls['false_negatives']:,}")
        
        print("=" * 70)
    
    def save_metrics(self, results, save_path):
        """Save metrics to JSON file."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f" Metrics saved to {save_path}")
