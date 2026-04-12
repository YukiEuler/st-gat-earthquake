# ==============================================================================
# ST_TFT.PY - Spatio-Temporal Transformer (GAT + TFT)
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_layer import MultiHeadGATLayer
from .tft_layer import TemporalFusionEncoder


class STTFT(nn.Module):
    """
    Spatio-Temporal Transformer combining:
    - GAT (Graph Attention Network) for spatial dependencies
    - TFT (Temporal Fusion Transformer) for temporal dependencies
    
    This replaces LSTM with transformer-based temporal processing.
    """
    
    def __init__(self, num_nodes, in_features, hidden_dim, out_features,
                 horizon=24, num_gat_layers=2, num_heads=4, 
                 tft_layers=2, dropout=0.2,
                 use_attention=True, use_multihead=True):
        super(STTFT, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads if use_multihead else 1
        self.horizon = horizon
        self.out_features = out_features
        self.use_attention = use_attention
        
        # ==================================================
        # GAT Layers (Spatial Processing)
        # ==================================================
        self.gat_layers = nn.ModuleList()
        
        if use_attention:
            # First layer: input features -> hidden_dim
            self.gat_layers.append(
                MultiHeadGATLayer(
                    in_features=in_features,
                    out_features=hidden_dim,
                    num_heads=self.num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            
            # Additional layers
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
        else:
            # Simple linear projection (no attention)
            self.gat_layers.append(nn.Linear(in_features, hidden_dim))
            for _ in range(num_gat_layers - 1):
                self.gat_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # ==================================================
        # TFT Encoder (Temporal Processing)
        # ==================================================
        self.temporal_encoder = TemporalFusionEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_heads=num_heads,
            n_layers=tft_layers,
            dropout=dropout,
            max_seq_len=100
        )
        
        # ==================================================
        # Output Layers (Multi-step Prediction)
        # ==================================================
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features * horizon)
        )
        
        # Store temporal attention for interpretability
        self._temporal_attention = None
    
    def forward(self, x, adj_sparse):
        """
        Args:
            x: Input tensor (B, T, N, F)
            adj_sparse: Coalesced sparse adjacency (N, N)
        
        Returns:
            Predictions (B, H, N, F') where H is horizon
        """
        batch_size, time_steps, nodes, features = x.size()
        
        # ==================================================
        # SPATIAL PROCESSING (GAT)
        # ==================================================
        x_reshaped = x.view(batch_size * time_steps, nodes, features)
        
        gat_out = x_reshaped
        for i, gat_layer in enumerate(self.gat_layers):
            if self.use_attention:
                gat_out = gat_layer(gat_out, adj_sparse)
            else:
                gat_out = F.relu(gat_layer(gat_out))
            
            if i < len(self.gat_layers) - 1:
                gat_out = self.dropout(gat_out)
        
        # Reshape: (B*T, N, H) -> (B, T, N, H)
        gat_out = gat_out.view(batch_size, time_steps, nodes, self.hidden_dim)
        
        # ==================================================
        # TEMPORAL PROCESSING (TFT)
        # ==================================================
        # Reshape: (B, T, N, H) -> (B*N, T, H)
        gat_out = gat_out.permute(0, 2, 1, 3).contiguous()
        gat_out = gat_out.view(batch_size * nodes, time_steps, self.hidden_dim)
        
        # Apply TFT encoder
        tft_out, temporal_attn = self.temporal_encoder(gat_out)  # (B*N, H)
        
        # Store attention for interpretability
        self._temporal_attention = temporal_attn
        
        # ==================================================
        # OUTPUT PREDICTION (Multi-step)
        # ==================================================
        out = self.fc_out(tft_out)  # (B*N, F'*horizon)
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
    
    def get_temporal_attention(self):
        """Get temporal attention weights from TFT for interpretability."""
        return self._temporal_attention
