# ==============================================================================
# ATTENTION.PY - Attention Visualization
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import torch


class AttentionVisualizer:
    """Visualize attention weights from GAT layers."""
    
    def __init__(self, node_coords=None, save_dir='outputs/figures'):
        self.node_coords = node_coords  # {'lat': [...], 'lon': [...]}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_attention_weights(self, model, sample_data, adj_sparse, device):
        """Extract attention weights from a forward pass."""
        model.eval()
        
        if isinstance(sample_data, np.ndarray):
            sample_data = torch.from_numpy(sample_data).float()
        
        sample_data = sample_data.unsqueeze(0).to(device)  # Add batch dim
        
        with torch.no_grad():
            _ = model(sample_data, adj_sparse)
        
        # Get attention weights from all layers
        attention_weights = model.get_attention_weights()
        
        return attention_weights
    
    def plot_attention_heatmap(self, attention_weights, adj_sparse, 
                               layer_idx=0, head_idx=0, save_name=None):
        """
        Plot attention weights as a heatmap on the graph.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if attention_weights is None:
            ax.text(0.5, 0.5, 'No attention weights available', 
                   ha='center', va='center', fontsize=14)
            return fig
        
        # Get attention weights for specific layer and head
        layer_weights = attention_weights[layer_idx] if layer_idx < len(attention_weights) else attention_weights[-1]
        if isinstance(layer_weights, list):
            head_weights = layer_weights[head_idx] if head_idx < len(layer_weights) else layer_weights[0]
        else:
            head_weights = layer_weights
        
        # Convert to numpy if needed
        if isinstance(head_weights, torch.Tensor):
            head_weights = head_weights.cpu().numpy()
        
        # If batched, take first batch
        if head_weights.ndim > 1:
            head_weights = head_weights[0]
        
        # Get edge indices
        edge_index = adj_sparse.indices().cpu().numpy()
        src, dst = edge_index[0], edge_index[1]
        
        # Plot nodes
        if self.node_coords is not None:
            lats = self.node_coords['lat']
            lons = self.node_coords['lon']
            
            # Normalize attention weights for coloring
            weights_norm = (head_weights - head_weights.min()) / (head_weights.max() - head_weights.min() + 1e-8)
            
            # Plot edges with attention as color/width
            for i, (s, d) in enumerate(zip(src, dst)):
                if s != d:  # Skip self-loops
                    alpha = min(1.0, weights_norm[i] * 2)
                    width = 0.5 + weights_norm[i] * 2
                    ax.plot([lons[s], lons[d]], [lats[s], lats[d]], 
                           'b-', alpha=alpha, linewidth=width)
            
            # Plot nodes
            ax.scatter(lons, lats, c='red', s=30, zorder=5, edgecolors='white')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
        else:
            # Simple histogram of attention weights
            ax.hist(head_weights, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Attention Weight')
            ax.set_ylabel('Frequency')
        
        ax.set_title(f'Attention Weights (Layer {layer_idx+1}, Head {head_idx+1})')
        
        if save_name:
            save_path = self.save_dir / f'{save_name}.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved: {save_path}")
        
        return fig
    
    def plot_attention_distribution(self, attention_weights, save_name=None):
        """Plot distribution of attention weights across all layers and heads."""
        if attention_weights is None:
            return None
        
        n_layers = len(attention_weights)
        
        fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
        if n_layers == 1:
            axes = [axes]
        
        for layer_idx, layer_weights in enumerate(attention_weights):
            ax = axes[layer_idx]
            
            if isinstance(layer_weights, list):
                # Multiple heads
                for head_idx, head_weights in enumerate(layer_weights):
                    if isinstance(head_weights, torch.Tensor):
                        head_weights = head_weights.cpu().numpy()
                    if head_weights.ndim > 1:
                        head_weights = head_weights.flatten()
                    ax.hist(head_weights, bins=30, alpha=0.5, 
                           label=f'Head {head_idx+1}')
            else:
                if isinstance(layer_weights, torch.Tensor):
                    layer_weights = layer_weights.cpu().numpy()
                ax.hist(layer_weights.flatten(), bins=30, alpha=0.7)
            
            ax.set_xlabel('Attention Weight')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Layer {layer_idx + 1}')
            ax.legend()
        
        plt.suptitle('Attention Weight Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / f'{save_name}.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved: {save_path}")
        
        return fig
    
    def plot_top_attention_edges(self, attention_weights, adj_sparse, 
                                  top_k=50, save_name=None):
        """Plot the top-k strongest attention connections."""
        if attention_weights is None or self.node_coords is None:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get first layer, first head attention
        layer_weights = attention_weights[0]
        if isinstance(layer_weights, list):
            head_weights = layer_weights[0]
        else:
            head_weights = layer_weights
        
        if isinstance(head_weights, torch.Tensor):
            head_weights = head_weights.cpu().numpy()
        if head_weights.ndim > 1:
            head_weights = head_weights[0]
        
        # Get edge indices
        edge_index = adj_sparse.indices().cpu().numpy()
        src, dst = edge_index[0], edge_index[1]
        
        # Get top-k edges
        top_indices = np.argsort(head_weights)[-top_k:]
        
        lats = self.node_coords['lat']
        lons = self.node_coords['lon']
        
        # Plot all nodes as background
        ax.scatter(lons, lats, c='lightgray', s=20, alpha=0.5)
        
        # Plot top attention edges
        max_weight = head_weights[top_indices].max()
        for idx in top_indices:
            s, d = src[idx], dst[idx]
            if s != d:
                weight = head_weights[idx] / max_weight
                ax.plot([lons[s], lons[d]], [lats[s], lats[d]], 
                       'r-', alpha=weight, linewidth=1 + weight * 2)
        
        # Highlight nodes involved in top attention
        involved_nodes = np.unique(np.concatenate([src[top_indices], dst[top_indices]]))
        ax.scatter(lons[involved_nodes], lats[involved_nodes], 
                  c='red', s=50, edgecolors='black', zorder=5)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Top {top_k} Attention Connections')
        
        if save_name:
            save_path = self.save_dir / f'{save_name}.png'
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved: {save_path}")
        
        return fig
