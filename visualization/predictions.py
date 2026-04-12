# ==============================================================================
# PREDICTIONS.PY - Prediction Visualization
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class PredictionVisualizer:
    """Visualize model predictions vs ground truth."""
    
    def __init__(self, feature_names=None, save_dir='outputs/figures'):
        self.feature_names = feature_names or ['Event Count', 'Max Magnitude', 'Log Energy', 'Avg Depth']
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, train_losses, val_losses, best_loss=None, 
                              save_name='training_history'):
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        
        if best_loss is not None:
            ax.axhline(y=best_loss, color='g', linestyle='--', 
                      label=f'Best: {best_loss:.6f}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
    
    def plot_scatter_pred_vs_actual(self, y_true, y_pred, save_name='scatter_pred_vs_actual'):
        """Plot scatter plots of predictions vs actual values."""
        n_features = y_true.shape[-1]
        n_cols = min(n_features, 4)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()
        
        for i in range(n_features):
            ax = axes[i]
            
            pred_flat = y_pred[..., i].flatten()
            target_flat = y_true[..., i].flatten()
            
            # Subsample for plotting
            n_samples = min(10000, len(pred_flat))
            idx = np.random.choice(len(pred_flat), n_samples, replace=False)
            
            ax.scatter(target_flat[idx], pred_flat[idx], alpha=0.3, s=10)
            
            # Perfect prediction line
            min_val = min(target_flat.min(), pred_flat.min())
            max_val = max(target_flat.max(), pred_flat.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            # Calculate R
            from sklearn.metrics import r2_score
            r2 = r2_score(target_flat, pred_flat) if np.std(target_flat) > 0 else 0
            
            feat_name = self.feature_names[i] if i < len(self.feature_names) else f'Feature {i}'
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{feat_name}\nR = {r2:.4f}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Prediction vs Actual', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
    
    def plot_timeseries_prediction(self, y_true, y_pred, node_idx=0, 
                                    horizon_idx=0, y_pred_std=None,
                                    save_name='timeseries_prediction'):
        """Plot time series prediction for a specific node."""
        n_features = y_true.shape[-1]
        
        fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features))
        
        if n_features == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            actual = y_true[:, horizon_idx, node_idx, i]
            predicted = y_pred[:, horizon_idx, node_idx, i]
            time_steps = np.arange(len(actual))
            
            ax.plot(time_steps, actual, 'b-', label='Actual', alpha=0.7, linewidth=1.5)
            ax.plot(time_steps, predicted, 'r--', label='Predicted', alpha=0.7, linewidth=1.5)
            
            # Plot uncertainty band if available
            if y_pred_std is not None:
                std = y_pred_std[:, horizon_idx, node_idx, i]
                ax.fill_between(time_steps, 
                               predicted - 1.96 * std, 
                               predicted + 1.96 * std,
                               alpha=0.2, color='red', label='95% CI')
            
            feat_name = self.feature_names[i] if i < len(self.feature_names) else f'Feature {i}'
            ax.set_xlabel('Time Step')
            ax.set_ylabel(feat_name)
            ax.set_title(f'{feat_name} - Node {node_idx}, Horizon h+{horizon_idx+1}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Time Series Prediction', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
    
    def plot_most_active_node(self, y_true, y_pred, horizon_idx=0,
                               save_name='most_active_node_prediction'):
        """
        Find and plot predictions for the most active node (highest event count).
        
        Args:
            y_true: Ground truth (B, H, N, F)
            y_pred: Predictions (B, H, N, F)
            horizon_idx: Which horizon to plot
            save_name: Output filename
        """
        # Find most active node based on total event count (first feature, typically 'count')
        # Sum across all samples and horizons for each node
        if y_true.ndim == 4:  # (B, H, N, F)
            node_activity = y_true[:, :, :, 0].sum(axis=(0, 1))  # Sum count over time
        else:  # (B, N, F)
            node_activity = y_true[:, :, 0].sum(axis=0)
        
        most_active_idx = np.argmax(node_activity)
        total_events = node_activity[most_active_idx]
        
        print(f"   Most active node: {most_active_idx} (Total events: {total_events:.0f})")
        
        n_features = y_true.shape[-1]
        
        # Create figure with 2 rows: first for main features, second for detailed view
        fig = plt.figure(figsize=(16, 4 * n_features))
        
        for i in range(n_features):
            ax = fig.add_subplot(n_features, 1, i + 1)
            
            if y_true.ndim == 4:
                actual = y_true[:, horizon_idx, most_active_idx, i]
                predicted = y_pred[:, horizon_idx, most_active_idx, i]
            else:
                actual = y_true[:, most_active_idx, i]
                predicted = y_pred[:, most_active_idx, i]
            
            time_steps = np.arange(len(actual))
            
            # Plot actual vs predicted
            ax.plot(time_steps, actual, 'b-', label='Actual', alpha=0.8, linewidth=1.5)
            ax.plot(time_steps, predicted, 'r--', label='Predicted', alpha=0.8, linewidth=1.5)
            
            # Calculate correlation for this feature
            if np.std(actual) > 0:
                from scipy.stats import pearsonr
                corr, _ = pearsonr(actual, predicted)
                corr_text = f'(r={corr:.3f})'
            else:
                corr_text = ''
            
            feat_name = self.feature_names[i] if i < len(self.feature_names) else f'Feature {i}'
            ax.set_xlabel('Time Step (hours)')
            ax.set_ylabel(feat_name)
            ax.set_title(f'{feat_name} {corr_text}')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Highlight peaks in actual data
            if i == 0:  # Only for count feature
                threshold = np.percentile(actual, 90)
                peak_mask = actual > threshold
                if peak_mask.sum() > 0:
                    ax.scatter(time_steps[peak_mask], actual[peak_mask], 
                              color='green', s=50, zorder=5, label='Peak Events')
        
        plt.suptitle(f'Predictions at Most Active Node (Node {most_active_idx}, Horizon h+{horizon_idx+1})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig, most_active_idx
    
    def plot_per_horizon_metrics(self, per_horizon_metrics, save_name='per_horizon_metrics'):
        """Plot metrics for each prediction horizon."""
        # Close any previous figures to prevent overlapping
        plt.close('all')
        
        horizons = sorted(per_horizon_metrics.keys())
        
        metrics_to_plot = ['RMSE', 'MAE', 'R2']
        n_metrics = len(metrics_to_plot)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            # Clear any previous data on this axis
            ax.clear()
            
            values = [per_horizon_metrics[h][metric] for h in horizons]
            horizon_nums = [int(h[1:]) for h in horizons]
            
            ax.plot(horizon_nums, values, 'o-', linewidth=2, markersize=6, color='steelblue')
            ax.set_xlabel('Forecast Horizon (hours)')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by Horizon')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Metrics by Prediction Horizon', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
    
    def plot_uncertainty_calibration(self, coverage_by_ci, save_name='uncertainty_calibration'):
        """Plot calibration curve for uncertainty estimates."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        expected = list(coverage_by_ci.keys())
        observed = list(coverage_by_ci.values())
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(expected, observed, 'o-', label='Model', linewidth=2, markersize=8)
        
        ax.set_xlabel('Expected Coverage')
        ax.set_ylabel('Observed Coverage')
        ax.set_title('Uncertainty Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
    
    def plot_ablation_comparison(self, ablation_df, save_name='ablation_comparison'):
        """Plot bar chart comparing ablation configurations."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        configs = ablation_df['Configuration']
        
        # RMSE comparison
        ax1 = axes[0]
        bars = ax1.barh(configs, ablation_df['RMSE'], color='steelblue')
        ax1.set_xlabel('RMSE')
        ax1.set_title('RMSE by Configuration')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # R2 comparison
        ax2 = axes[1]
        bars = ax2.barh(configs, ablation_df['R2'], color='forestgreen')
        ax2.set_xlabel('R²')
        ax2.set_title('R² by Configuration')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Ablation Study Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
