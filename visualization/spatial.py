# ==============================================================================
# SPATIAL.PY - Spatial Visualization
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SpatialVisualizer:
    """Visualize spatial predictions and errors."""
    
    def __init__(self, node_coords, save_dir='outputs/figures'):
        self.node_coords = node_coords  # {'lat': [...], 'lon': [...]}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_spatial_heatmap(self, y_true, y_pred, sample_idx=0, horizon_idx=0,
                             feature_idx=0, save_name='spatial_heatmap'):
        """Plot spatial heatmap comparing actual vs predicted."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        lats = self.node_coords['lat']
        lons = self.node_coords['lon']
        
        # Get values for specific sample, horizon, feature
        actual = y_true[sample_idx, horizon_idx, :, feature_idx]
        predicted = y_pred[sample_idx, horizon_idx, :, feature_idx]
        predicted = np.clip(predicted, 0, None)  # Clip negative predictions
        error = np.abs(actual - predicted)
        
        # Common vmax for actual and predicted
        vmax = max(actual.max(), predicted.max())
        
        # 1. Actual
        ax1 = axes[0]
        sc1 = ax1.scatter(lons, lats, c=actual, cmap='hot', s=30, 
                         vmin=0, vmax=vmax, edgecolors='none')
        plt.colorbar(sc1, ax=ax1)
        ax1.set_title('Actual')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # 2. Predicted
        ax2 = axes[1]
        sc2 = ax2.scatter(lons, lats, c=predicted, cmap='hot', s=30,
                         vmin=0, vmax=vmax, edgecolors='none')
        plt.colorbar(sc2, ax=ax2)
        ax2.set_title('Predicted')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        
        # 3. Error
        ax3 = axes[2]
        sc3 = ax3.scatter(lons, lats, c=error, cmap='coolwarm', s=30, 
                         edgecolors='none')
        plt.colorbar(sc3, ax=ax3, label='|Error|')
        ax3.set_title('Absolute Error')
        ax3.set_xlabel('Longitude')
        ax3.set_ylabel('Latitude')
        
        plt.suptitle(f'Spatial Comparison (Sample {sample_idx}, Horizon h+{horizon_idx+1})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
    
    def plot_uncertainty_spatial(self, y_pred_mean, y_pred_std, sample_idx=0,
                                  horizon_idx=0, feature_idx=0,
                                  save_name='uncertainty_spatial'):
        """Plot spatial distribution of uncertainty."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        lats = self.node_coords['lat']
        lons = self.node_coords['lon']
        
        mean = y_pred_mean[sample_idx, horizon_idx, :, feature_idx]
        std = y_pred_std[sample_idx, horizon_idx, :, feature_idx]
        
        # Mean prediction
        ax1 = axes[0]
        sc1 = ax1.scatter(lons, lats, c=mean, cmap='hot', s=30, edgecolors='none')
        plt.colorbar(sc1, ax=ax1, label='Mean Prediction')
        ax1.set_title('Mean Prediction')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # Uncertainty (std)
        ax2 = axes[1]
        sc2 = ax2.scatter(lons, lats, c=std, cmap='YlOrRd', s=30, edgecolors='none')
        plt.colorbar(sc2, ax=ax2, label='Uncertainty (Std)')
        ax2.set_title('Prediction Uncertainty')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        
        plt.suptitle(f'Uncertainty Visualization (Sample {sample_idx})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
    
    def plot_node_activity(self, y_true, save_name='node_activity'):
        """Plot cumulative activity per node."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        lats = self.node_coords['lat']
        lons = self.node_coords['lon']
        
        # Sum activity (feature 0 = event count) over all samples and horizons
        activity = y_true[:, :, :, 0].sum(axis=(0, 1))
        
        sc = ax.scatter(lons, lats, c=activity, cmap='hot', s=30 + activity * 2,
                       edgecolors='black', linewidths=0.5)
        plt.colorbar(sc, ax=ax, label='Total Events')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Cumulative Seismic Activity per Node')
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
    
    def plot_error_by_activity(self, y_true, y_pred, save_name='error_by_activity'):
        """Plot relationship between node activity and prediction error."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate per-node activity and error
        activity_per_node = y_true[:, :, :, 0].sum(axis=(0, 1))
        mse_per_node = ((y_true - y_pred) ** 2).mean(axis=(0, 1, 3))
        
        ax.scatter(activity_per_node, mse_per_node, alpha=0.5, s=20)
        
        # Trend line
        z = np.polyfit(activity_per_node, mse_per_node, 1)
        p = np.poly1d(z)
        x_line = np.linspace(activity_per_node.min(), activity_per_node.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
        
        ax.set_xlabel('Node Activity (Total Events)')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Prediction Error vs Node Activity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        save_path = self.save_dir / f'{save_name}.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Saved: {save_path}")
        
        return fig
