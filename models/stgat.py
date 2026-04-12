# ==============================================================================
# STGAT.PY - Spatio-Temporal Graph Attention Network
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_layer import MultiHeadGATLayer


class STGAT(nn.Module):
    """
    Spatio-Temporal Graph Attention Network with Skip Connections and Node Embeddings.
    
    Architecture:
        - Node Embeddings: Learnable per-node vectors for local patterns
        - GAT Layers: Multi-head attention for spatial processing (with residual)
        - LSTM: Temporal sequence modeling
        - FC Output: Multi-step prediction with skip connection
    """
    
    def __init__(self, num_nodes, in_features, hidden_dim, out_features,
                 horizon=24, num_gat_layers=2, num_heads=4, dropout=0.2,
                 use_attention=True, use_multihead=True, use_skip=True,
                 node_embed_dim=16):
        super(STGAT, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads if use_multihead else 1
        self.horizon = horizon
        self.out_features = out_features
        self.use_attention = use_attention
        self.use_skip = use_skip
        self.node_embed_dim = node_embed_dim
        
        # Learnable node embeddings for capturing local patterns (optional)
        if node_embed_dim > 0:
            self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
            effective_in_features = in_features + node_embed_dim
        else:
            self.node_embedding = None
            effective_in_features = in_features
        
        # Input projection for skip connection
        self.input_proj = nn.Linear(effective_in_features, hidden_dim)
        
        # GAT Layers (Spatial)
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()  # Layer normalization for skip connections
        
        if use_attention:
            # First layer (input = features + node embeddings)
            self.gat_layers.append(
                MultiHeadGATLayer(
                    in_features=effective_in_features,
                    out_features=hidden_dim,
                    num_heads=self.num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
            
            # Additional layers with skip connections
            for _ in range(num_gat_layers - 2):
                self.gat_layers.append(
                    MultiHeadGATLayer(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        num_heads=self.num_heads,
                        dropout=dropout,
                        concat=True
                    )
                )
                self.gat_norms.append(nn.LayerNorm(hidden_dim))
            
            # Final GAT layer (average heads)
            if num_gat_layers > 1:
                self.gat_layers.append(
                    MultiHeadGATLayer(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        num_heads=self.num_heads,
                        dropout=dropout,
                        concat=False
                    )
                )
                self.gat_norms.append(nn.LayerNorm(hidden_dim))
        else:
            # Simple linear projection (no attention)
            self.gat_layers.append(nn.Linear(effective_in_features, hidden_dim))
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
            for _ in range(num_gat_layers - 1):
                self.gat_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.gat_norms.append(nn.LayerNorm(hidden_dim))
        
        # LSTM (Temporal)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if hidden_dim > 1 else 0,
            bidirectional=False
        )
        
        # Layer norm after LSTM
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        
        # Output layers (Multi-step) with skip from temporal features
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2 if use_skip else hidden_dim, hidden_dim),  # *2 for skip
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features * horizon)
        )
        
        # Skip connection projection for temporal (average pooling over time)
        if use_skip:
            self.skip_temporal_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, adj_sparse):
        """
        Args:
            x: Input tensor (B, T, N, F)
            adj_sparse: Coalesced sparse adjacency (N, N)
        
        Returns:
            Predictions (B, H, N, F') where H is horizon
        """
        batch_size, time_steps, nodes, features = x.size()
        
        # Reshape input
        x_reshaped = x.view(batch_size * time_steps, nodes, features)  # (B*T, N, F)
        
        # Add learnable node embeddings (if enabled)
        if self.node_embedding is not None:
            node_ids = torch.arange(nodes, device=x.device)
            node_embeds = self.node_embedding(node_ids)  # (N, embed_dim)
            # Expand to match (B*T, N, embed_dim)
            node_embeds = node_embeds.unsqueeze(0).expand(batch_size * time_steps, -1, -1)
            x_with_embed = torch.cat([x_reshaped, node_embeds], dim=-1)  # (B*T, N, F+embed_dim)
        else:
            x_with_embed = x_reshaped  # No embedding
        
        # Project input for skip connection
        x_skip = self.input_proj(x_with_embed)  # (B*T, N, H)
        
        gat_out = x_with_embed
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            if self.use_attention:
                gat_new = gat_layer(gat_out, adj_sparse)
            else:
                gat_new = F.relu(gat_layer(gat_out))
            
            # Skip connection (residual) for layers after first
            if self.use_skip and i > 0:
                gat_new = gat_new + gat_out  # Residual connection
            elif self.use_skip and i == 0:
                gat_new = gat_new + x_skip  # Skip from projected input
            
            gat_out = norm(gat_new)
            
            if i < len(self.gat_layers) - 1:
                gat_out = self.dropout(gat_out)
        
        # Reshape back: (B*T, N, H) -> (B, T, N, H)
        gat_out = gat_out.view(batch_size, time_steps, nodes, self.hidden_dim)
        
        # Save for temporal skip connection (mean over time)
        if self.use_skip:
            temporal_skip = gat_out.mean(dim=1)  # (B, N, H)
            temporal_skip = self.skip_temporal_proj(temporal_skip)  # (B, N, H)
        
        # Temporal processing (LSTM)
        # Reshape: (B, T, N, H) -> (B*N, T, H)
        gat_out = gat_out.permute(0, 2, 1, 3).contiguous()
        gat_out = gat_out.view(batch_size * nodes, time_steps, self.hidden_dim)
        
        lstm_out, _ = self.lstm(gat_out)
        last_out = lstm_out[:, -1, :]  # (B*N, H)
        last_out = self.lstm_norm(last_out)
        
        # Concatenate with temporal skip connection
        if self.use_skip:
            temporal_skip = temporal_skip.view(batch_size * nodes, self.hidden_dim)
            last_out = torch.cat([last_out, temporal_skip], dim=-1)  # (B*N, H*2)
        
        # Output prediction (Multi-step)
        out = self.fc_out(last_out)  # (B*N, F'*horizon)
        out = out.view(batch_size, nodes, self.horizon, self.out_features)
        
        # Reorder to (B, H, N, F')
        out = out.permute(0, 2, 1, 3)
        
        return out
    
    def get_attention_weights(self):
        """Get attention weights from all GAT layers for visualization."""
        if not self.use_attention:
            return None
        
        weights = []
        for layer in self.gat_layers:
            if hasattr(layer, 'get_attention_weights'):
                weights.append(layer.get_attention_weights())
        return weights

