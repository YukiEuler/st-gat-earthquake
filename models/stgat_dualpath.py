# ==============================================================================
# STGAT_DUALPATH.PY - Dual-Path Spatio-Temporal Graph Attention Network
# ==============================================================================
"""
Architecture with two parallel paths:
  - Path 1 (Spatial-Temporal): GAT + LSTM - captures spatial dependencies via graph
  - Path 2 (Temporal-Only): LSTM only - captures pure temporal patterns per node

Both paths are merged at the end for final prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_layer import MultiHeadGATLayer


class STGATDualPath(nn.Module):
    """
    Dual-Path Spatio-Temporal Graph Attention Network.
    
    Architecture:
        Input (B, T, N, F)
              │
              ├───────────────────────────────┐
              │                               │
              ▼                               ▼
        ┌─────────────┐               ┌─────────────┐
        │   Path 1    │               │   Path 2    │
        │  GAT Layers │               │    Linear   │
        │     +       │               │ Projection  │
        │    LSTM     │               │     +       │
        │             │               │    LSTM     │
        └─────────────┘               └─────────────┘
              │                               │
              └─────────────┬─────────────────┘
                            │
                            ▼
                    ┌─────────────┐
                    │   Fusion    │
                    │  (Concat +  │
                    │    FC)      │
                    └─────────────┘
                            │
                            ▼
                    Output (B, H, N, F')
    """
    
    def __init__(self, num_nodes, in_features, hidden_dim, out_features,
                 horizon=24, num_gat_layers=2, num_heads=4, dropout=0.2,
                 use_attention=True, use_multihead=True, use_skip=True,
                 node_embed_dim=16, fusion_type='concat'):
        super(STGATDualPath, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads if use_multihead else 1
        self.horizon = horizon
        self.out_features = out_features
        self.use_attention = use_attention
        self.use_skip = use_skip
        self.node_embed_dim = node_embed_dim
        self.fusion_type = fusion_type  # 'concat', 'add', 'gate'
        
        # Effective input features (with optional node embeddings)
        if node_embed_dim > 0:
            self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
            effective_in_features = in_features + node_embed_dim
        else:
            self.node_embedding = None
            effective_in_features = in_features
        
        # ==================================================================
        # PATH 1: GAT + LSTM (Spatial-Temporal)
        # ==================================================================
        self.input_proj_p1 = nn.Linear(effective_in_features, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        
        if use_attention:
            # First GAT layer
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
            
            # Middle GAT layers
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
            
            # Final GAT layer
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
            # Simple linear layers if no attention
            self.gat_layers.append(nn.Linear(effective_in_features, hidden_dim))
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
            for _ in range(num_gat_layers - 1):
                self.gat_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.gat_norms.append(nn.LayerNorm(hidden_dim))
        
        # LSTM for Path 1 (after GAT)
        self.lstm_p1 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if hidden_dim > 1 else 0,
            bidirectional=False
        )
        self.lstm_norm_p1 = nn.LayerNorm(hidden_dim)
        
        # ==================================================================
        # PATH 2: LSTM Only (Pure Temporal)
        # ==================================================================
        # Input projection for Path 2
        self.input_proj_p2 = nn.Linear(effective_in_features, hidden_dim)
        
        # LSTM for Path 2 (directly on input features)
        self.lstm_p2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if hidden_dim > 1 else 0,
            bidirectional=False
        )
        self.lstm_norm_p2 = nn.LayerNorm(hidden_dim)
        
        # ==================================================================
        # FUSION LAYER
        # ==================================================================
        self.dropout_layer = nn.Dropout(dropout)
        
        if fusion_type == 'concat':
            # Concatenate both paths: hidden_dim * 2
            fusion_input_dim = hidden_dim * 2
        elif fusion_type == 'gate':
            # Gated fusion
            self.gate_fc = nn.Linear(hidden_dim * 2, hidden_dim)
            fusion_input_dim = hidden_dim
        else:  # 'add'
            fusion_input_dim = hidden_dim
        
        # Add skip connection dimension if enabled
        if use_skip:
            self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
            fusion_input_dim += hidden_dim
        
        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features * horizon)
        )
    
    def forward(self, x, adj_sparse):
        """
        Args:
            x: Input tensor (B, T, N, F)
            adj_sparse: Coalesced sparse adjacency (N, N)
        
        Returns:
            Predictions (B, H, N, F') where H is horizon
        """
        batch_size, time_steps, nodes, features = x.size()
        
        # Reshape input: (B, T, N, F) -> (B*T, N, F)
        x_reshaped = x.view(batch_size * time_steps, nodes, features)
        
        # Add learnable node embeddings (if enabled)
        if self.node_embedding is not None:
            node_ids = torch.arange(nodes, device=x.device)
            node_embeds = self.node_embedding(node_ids)  # (N, embed_dim)
            node_embeds = node_embeds.unsqueeze(0).expand(batch_size * time_steps, -1, -1)
            x_with_embed = torch.cat([x_reshaped, node_embeds], dim=-1)
        else:
            x_with_embed = x_reshaped
        
        # ==================================================================
        # PATH 1: GAT -> LSTM
        # ==================================================================
        # GAT layers with skip connections
        x_skip_p1 = self.input_proj_p1(x_with_embed)  # (B*T, N, H)
        
        gat_out = x_with_embed
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            if self.use_attention:
                gat_new = gat_layer(gat_out, adj_sparse)
            else:
                gat_new = F.relu(gat_layer(gat_out))
            
            # Skip connections
            if self.use_skip and i > 0:
                gat_new = gat_new + gat_out
            elif self.use_skip and i == 0:
                gat_new = gat_new + x_skip_p1
            
            gat_out = norm(gat_new)
            
            if i < len(self.gat_layers) - 1:
                gat_out = self.dropout_layer(gat_out)
        
        # Reshape for LSTM: (B*T, N, H) -> (B, T, N, H) -> (B*N, T, H)
        gat_out = gat_out.view(batch_size, time_steps, nodes, self.hidden_dim)
        
        # Save for skip connection
        if self.use_skip:
            skip_features = gat_out.mean(dim=1)  # (B, N, H)
            skip_features = self.skip_proj(skip_features)
        
        gat_out = gat_out.permute(0, 2, 1, 3).contiguous()
        gat_out = gat_out.view(batch_size * nodes, time_steps, self.hidden_dim)
        
        # LSTM Path 1
        lstm_out_p1, _ = self.lstm_p1(gat_out)
        path1_out = lstm_out_p1[:, -1, :]  # (B*N, H)
        path1_out = self.lstm_norm_p1(path1_out)
        
        # ==================================================================
        # PATH 2: LSTM Only (Direct Temporal)
        # ==================================================================
        # Project input and reshape for LSTM
        x_proj_p2 = self.input_proj_p2(x_with_embed)  # (B*T, N, H)
        x_proj_p2 = x_proj_p2.view(batch_size, time_steps, nodes, self.hidden_dim)
        x_proj_p2 = x_proj_p2.permute(0, 2, 1, 3).contiguous()
        x_proj_p2 = x_proj_p2.view(batch_size * nodes, time_steps, self.hidden_dim)
        
        # LSTM Path 2
        lstm_out_p2, _ = self.lstm_p2(x_proj_p2)
        path2_out = lstm_out_p2[:, -1, :]  # (B*N, H)
        path2_out = self.lstm_norm_p2(path2_out)
        
        # ==================================================================
        # FUSION
        # ==================================================================
        if self.fusion_type == 'concat':
            fused = torch.cat([path1_out, path2_out], dim=-1)  # (B*N, H*2)
        elif self.fusion_type == 'gate':
            # Learned gating mechanism
            gate_input = torch.cat([path1_out, path2_out], dim=-1)
            gate = torch.sigmoid(self.gate_fc(gate_input))  # (B*N, H)
            fused = gate * path1_out + (1 - gate) * path2_out  # (B*N, H)
        else:  # 'add'
            fused = path1_out + path2_out  # (B*N, H)
        
        # Add skip connection
        if self.use_skip:
            skip_features = skip_features.view(batch_size * nodes, self.hidden_dim)
            fused = torch.cat([fused, skip_features], dim=-1)
        
        fused = self.dropout_layer(fused)
        
        # ==================================================================
        # OUTPUT
        # ==================================================================
        out = self.fc_out(fused)  # (B*N, F'*horizon)
        out = out.view(batch_size, nodes, self.horizon, self.out_features)
        out = out.permute(0, 2, 1, 3)  # (B, H, N, F')
        
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
