# ==============================================================================
# STGAT_MULTISCALE.PY - Multi-Scale Spatio-Temporal Graph Attention Network
# ==============================================================================
"""
Multi-Scale ST-GAT processes temporal data at multiple resolutions:
- Scale 1x: Full resolution (fine-grained, good for short-term)
- Scale 2x: Half resolution (medium patterns)  
- Scale 4x: Quarter resolution (coarse, good for long-term)

Each scale captures different temporal patterns and they are fused
for horizon-aware prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_layer import MultiHeadGATLayer


class TemporalPooling(nn.Module):
    """Temporal pooling layer with learnable aggregation."""
    
    def __init__(self, pool_size, hidden_dim, pool_type='avg'):
        super().__init__()
        self.pool_size = pool_size
        self.pool_type = pool_type
        
        if pool_type == 'attention':
            # Learnable attention-based pooling
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, N, H)
        Returns:
            pooled: (B, T//pool_size, N, H)
        """
        B, T, N, H = x.shape
        
        # Pad if necessary
        if T % self.pool_size != 0:
            pad_size = self.pool_size - (T % self.pool_size)
            x = F.pad(x, (0, 0, 0, 0, 0, pad_size))
            T = T + pad_size
        
        # Reshape for pooling
        x = x.view(B, T // self.pool_size, self.pool_size, N, H)
        
        if self.pool_type == 'avg':
            return x.mean(dim=2)
        elif self.pool_type == 'max':
            return x.max(dim=2)[0]
        elif self.pool_type == 'attention':
            # Attention weights over pool window
            attn = self.attention(x)  # (B, T', pool, N, 1)
            attn = F.softmax(attn, dim=2)
            return (x * attn).sum(dim=2)
        else:
            return x.mean(dim=2)


class ScaleBranch(nn.Module):
    """Single temporal scale processing branch (LSTM-based)."""
    
    def __init__(self, hidden_dim, dropout=0.2, num_layers=2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B*N, T, H) - temporal sequence per node
        Returns:
            out: (B*N, H) - final hidden state
            all_states: (B*N, T, H) - all hidden states
        """
        lstm_out, (h_n, _) = self.lstm(x)
        
        # Return both last state and all states (for attention-based fusion)
        last_out = self.norm(lstm_out[:, -1, :])
        all_out = self.norm(lstm_out)
        
        return last_out, all_out


class MultiScaleFusion(nn.Module):
    """Fuse features from multiple temporal scales."""
    
    def __init__(self, hidden_dim, num_scales, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_scales = num_scales
        
        if fusion_type == 'concat':
            self.proj = nn.Linear(hidden_dim * num_scales, hidden_dim)
        elif fusion_type == 'attention':
            self.scale_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.proj = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_type == 'gate':
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * num_scales, hidden_dim),
                    nn.Sigmoid()
                ) for _ in range(num_scales)
            ])
            self.proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, scale_features):
        """
        Args:
            scale_features: list of (B*N, H) tensors, one per scale
        Returns:
            fused: (B*N, H)
        """
        if self.fusion_type == 'concat':
            concat = torch.cat(scale_features, dim=-1)
            return self.proj(concat)
        
        elif self.fusion_type == 'attention':
            # Stack scales: (B*N, num_scales, H)
            stacked = torch.stack(scale_features, dim=1)
            attn = self.scale_attention(stacked)  # (B*N, num_scales, 1)
            attn = F.softmax(attn, dim=1)
            fused = (stacked * attn).sum(dim=1)  # (B*N, H)
            return self.proj(fused)
        
        elif self.fusion_type == 'gate':
            concat = torch.cat(scale_features, dim=-1)
            gated = sum(
                gate(concat) * feat 
                for gate, feat in zip(self.gates, scale_features)
            )
            return self.proj(gated)


class HorizonAwareDecoder(nn.Module):
    """Decoder that generates horizon-specific predictions."""
    
    def __init__(self, hidden_dim, out_features, horizon, dropout=0.2):
        super().__init__()
        
        self.horizon = horizon
        self.out_features = out_features
        
        # Learnable horizon embeddings
        self.horizon_embed = nn.Embedding(horizon, hidden_dim)
        
        # Attention over multi-scale features for each horizon
        self.horizon_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features)
        )
    
    def forward(self, fused_features, scale_states=None):
        """
        Args:
            fused_features: (B*N, H) - fused multi-scale features
            scale_states: optional list of (B*N, T_i, H) for attention
        Returns:
            predictions: (B*N, horizon, out_features)
        """
        B_N, hidden_dim = fused_features.shape
        device = fused_features.device
        
        # Horizon queries
        h_ids = torch.arange(self.horizon, device=device)
        h_embed = self.horizon_embed(h_ids)  # (horizon, H)
        h_embed = h_embed.unsqueeze(0).expand(B_N, -1, -1)  # (B*N, horizon, H)
        
        # Use fused features as keys/values
        kv = fused_features.unsqueeze(1)  # (B*N, 1, H)
        
        # If we have scale states, concatenate them for richer context
        if scale_states is not None:
            # Concatenate all scale states
            kv = torch.cat(scale_states, dim=1)  # (B*N, sum(T_i), H)
        
        # Attention: horizon queries attend to temporal states
        attn_out, _ = self.horizon_attention(h_embed, kv, kv)  # (B*N, horizon, H)
        
        # Add residual from fused features
        attn_out = attn_out + fused_features.unsqueeze(1)
        
        # Project to output
        predictions = self.output_proj(attn_out)  # (B*N, horizon, out_features)
        
        return predictions


class STGATMultiScale(nn.Module):
    """
    Multi-Scale Spatio-Temporal Graph Attention Network.
    
    Processes temporal data at multiple resolutions (1x, 2x, 4x) and
    fuses them for improved multi-step prediction.
    
    Architecture:
        Input (B, T, N, F)
            ↓
        Node Embeddings (optional)
            ↓
        GAT Layers (Spatial Processing)
            ↓
        ┌─────────────┬─────────────┬─────────────┐
        │  Scale 1x   │  Scale 2x   │  Scale 4x   │
        │  (full)     │  (pool 2)   │  (pool 4)   │
        │    LSTM     │    LSTM     │    LSTM     │
        └─────────────┴─────────────┴─────────────┘
            ↓              ↓              ↓
        ─────────── Multi-Scale Fusion ───────────
                          ↓
              Horizon-Aware Decoder
                          ↓
              Output (B, horizon, N, F')
    """
    
    def __init__(self, num_nodes, in_features, hidden_dim, out_features,
                 horizon=6, num_gat_layers=2, num_heads=4, dropout=0.2,
                 scales=[1, 2, 4], fusion_type='concat',
                 use_attention=True, use_multihead=True, 
                 node_embed_dim=16, pool_type='avg'):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads if use_multihead else 1
        self.horizon = horizon
        self.out_features = out_features
        self.use_attention = use_attention
        self.scales = scales
        self.node_embed_dim = node_embed_dim
        
        # Learnable node embeddings
        if node_embed_dim > 0:
            self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
            effective_in_features = in_features + node_embed_dim
        else:
            self.node_embedding = None
            effective_in_features = in_features
        
        # Input projection
        self.input_proj = nn.Linear(effective_in_features, hidden_dim)
        
        # GAT Layers (Spatial)
        self.gat_layers = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        
        if use_attention:
            # First layer
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
            
            # Additional layers
            for i in range(num_gat_layers - 2):
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
            
            # Final layer
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
            # Simple linear (no attention)
            self.gat_layers.append(nn.Linear(effective_in_features, hidden_dim))
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
            for _ in range(num_gat_layers - 1):
                self.gat_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.gat_norms.append(nn.LayerNorm(hidden_dim))
        
        # Multi-Scale Temporal Branches
        self.temporal_pools = nn.ModuleList()
        self.temporal_branches = nn.ModuleList()
        
        for scale in scales:
            if scale > 1:
                self.temporal_pools.append(
                    TemporalPooling(scale, hidden_dim, pool_type=pool_type)
                )
            else:
                self.temporal_pools.append(None)  # No pooling for scale 1
            
            self.temporal_branches.append(
                ScaleBranch(hidden_dim, dropout=dropout, num_layers=2)
            )
        
        # Multi-Scale Fusion
        self.fusion = MultiScaleFusion(
            hidden_dim=hidden_dim,
            num_scales=len(scales),
            fusion_type=fusion_type
        )
        
        # Horizon-Aware Decoder
        self.decoder = HorizonAwareDecoder(
            hidden_dim=hidden_dim,
            out_features=out_features,
            horizon=horizon,
            dropout=dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_sparse):
        """
        Args:
            x: Input tensor (B, T, N, F)
            adj_sparse: Sparse adjacency matrix (N, N)
        
        Returns:
            predictions: (B, horizon, N, out_features)
        """
        batch_size, time_steps, nodes, features = x.size()
        
        # Reshape for GAT: (B*T, N, F)
        x_reshaped = x.view(batch_size * time_steps, nodes, features)
        
        # Add node embeddings
        if self.node_embedding is not None:
            node_ids = torch.arange(nodes, device=x.device)
            node_embeds = self.node_embedding(node_ids)
            node_embeds = node_embeds.unsqueeze(0).expand(batch_size * time_steps, -1, -1)
            x_with_embed = torch.cat([x_reshaped, node_embeds], dim=-1)
        else:
            x_with_embed = x_reshaped
        
        # Input projection for skip connection
        x_skip = self.input_proj(x_with_embed)
        
        # GAT layers with skip connections
        gat_out = x_with_embed
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            if self.use_attention:
                gat_new = gat_layer(gat_out, adj_sparse)
            else:
                gat_new = F.relu(gat_layer(gat_out))
            
            # Skip connection
            if i > 0:
                gat_new = gat_new + gat_out
            elif i == 0:
                gat_new = gat_new + x_skip
            
            gat_out = norm(gat_new)
            
            if i < len(self.gat_layers) - 1:
                gat_out = self.dropout(gat_out)
        
        # Reshape: (B*T, N, H) -> (B, T, N, H)
        gat_out = gat_out.view(batch_size, time_steps, nodes, self.hidden_dim)
        
        # Multi-Scale Temporal Processing
        scale_outputs = []  # Final states per scale
        scale_states = []   # All states per scale (for horizon attention)
        
        for pool, branch in zip(self.temporal_pools, self.temporal_branches):
            # Apply temporal pooling if needed
            if pool is not None:
                scaled = pool(gat_out)  # (B, T/scale, N, H)
            else:
                scaled = gat_out  # (B, T, N, H)
            
            # Reshape for LSTM: (B, T', N, H) -> (B*N, T', H)
            B, T_scaled, N, H = scaled.shape
            scaled = scaled.permute(0, 2, 1, 3).contiguous()
            scaled = scaled.view(B * N, T_scaled, H)
            
            # Process through LSTM
            last_out, all_out = branch(scaled)
            
            scale_outputs.append(last_out)  # (B*N, H)
            scale_states.append(all_out)    # (B*N, T', H)
        
        # Fuse multi-scale features
        fused = self.fusion(scale_outputs)  # (B*N, H)
        
        # Decode with horizon awareness
        predictions = self.decoder(fused, scale_states)  # (B*N, horizon, out_features)
        
        # Reshape: (B*N, horizon, F) -> (B, horizon, N, F)
        predictions = predictions.view(batch_size, nodes, self.horizon, self.out_features)
        predictions = predictions.permute(0, 2, 1, 3)  # (B, horizon, N, F)
        
        return predictions
    
    def get_attention_weights(self):
        """Get attention weights from GAT layers for visualization."""
        if not self.use_attention:
            return None
        
        weights = []
        for layer in self.gat_layers:
            if hasattr(layer, 'get_attention_weights'):
                weights.append(layer.get_attention_weights())
        return weights
    
    def get_scale_info(self):
        """Return information about temporal scales."""
        return {
            'scales': self.scales,
            'descriptions': [
                f"Scale {s}x: {'Full resolution' if s == 1 else f'Pool by {s}'}"
                for s in self.scales
            ]
        }
