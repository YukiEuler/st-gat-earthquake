# ==============================================================================
# STGAT_WARMSTART.PY - ST-GAT with Warm Start (Curriculum Learning)
# ==============================================================================
"""
ST-GAT with Warm Start Training Strategy:
  - Phase 1: Train LSTM only (bypass GAT completely)
  - Phase 2: Gradually blend in GAT contribution
  - Phase 3: Full GAT-LSTM training

This helps stabilize training by first learning temporal patterns,
then learning spatial relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_layer import MultiHeadGATLayer


class STGATWarmStart(nn.Module):
    """
    ST-GAT with Warm Start / Curriculum Learning.
    
    The model gradually increases GAT contribution:
    - Epoch < warmup_epochs: gat_weight = epoch / warmup_epochs
    - Epoch >= warmup_epochs: gat_weight = 1.0
    
    Output = gat_weight * GAT(x) + (1 - gat_weight) * Linear(x)
    """
    
    def __init__(self, num_nodes, in_features, hidden_dim, out_features,
                 horizon=24, num_gat_layers=2, num_heads=4, dropout=0.2,
                 use_attention=True, use_multihead=True, use_skip=True,
                 node_embed_dim=16, warmup_epochs=10):
        super(STGATWarmStart, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads if use_multihead else 1
        self.horizon = horizon
        self.out_features = out_features
        self.use_attention = use_attention
        self.use_skip = use_skip
        self.node_embed_dim = node_embed_dim
        self.warmup_epochs = warmup_epochs
        
        # Warm start state
        self._gat_weight = 0.0  # Start with GAT disabled
        self._current_epoch = 0
        
        # Node embeddings (optional)
        if node_embed_dim > 0:
            self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
            effective_in_features = in_features + node_embed_dim
        else:
            self.node_embedding = None
            effective_in_features = in_features
        
        # Input projection (for skip/bypass)
        self.input_proj = nn.Linear(effective_in_features, hidden_dim)
        
        # ==================================================================
        # GAT LAYERS (Spatial Path)
        # ==================================================================
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
            
            # Middle layers
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
            # Linear layers if no attention
            self.gat_layers.append(nn.Linear(effective_in_features, hidden_dim))
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
            for _ in range(num_gat_layers - 1):
                self.gat_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.gat_norms.append(nn.LayerNorm(hidden_dim))
        
        # ==================================================================
        # BYPASS PATH (Direct projection, no spatial)
        # ==================================================================
        self.bypass_layers = nn.Sequential(
            nn.Linear(effective_in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ==================================================================
        # LSTM (Temporal)
        # ==================================================================
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if hidden_dim > 1 else 0,
            bidirectional=False
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2 if use_skip else hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features * horizon)
        )
        
        if use_skip:
            self.skip_temporal_proj = nn.Linear(hidden_dim, hidden_dim)
    
    @property
    def gat_weight(self):
        """Current GAT contribution weight (0.0 to 1.0)."""
        return self._gat_weight
    
    def set_epoch(self, epoch):
        """
        Update the current epoch and recalculate GAT weight.
        
        Should be called at the start of each epoch during training.
        """
        self._current_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            self._gat_weight = epoch / self.warmup_epochs
        else:
            # Full GAT after warmup
            self._gat_weight = 1.0
        
        return self._gat_weight
    
    def set_gat_weight(self, weight):
        """Manually set GAT weight (for inference or testing)."""
        self._gat_weight = max(0.0, min(1.0, weight))
    
    def forward(self, x, adj_sparse):
        """
        Args:
            x: Input tensor (B, T, N, F)
            adj_sparse: Sparse adjacency (N, N)
        
        Returns:
            Predictions (B, H, N, F')
        """
        batch_size, time_steps, nodes, features = x.size()
        
        # Reshape input
        x_reshaped = x.view(batch_size * time_steps, nodes, features)
        
        # Add node embeddings
        if self.node_embedding is not None:
            node_ids = torch.arange(nodes, device=x.device)
            node_embeds = self.node_embedding(node_ids)
            node_embeds = node_embeds.unsqueeze(0).expand(batch_size * time_steps, -1, -1)
            x_with_embed = torch.cat([x_reshaped, node_embeds], dim=-1)
        else:
            x_with_embed = x_reshaped
        
        # ==================================================================
        # BYPASS PATH (always computed)
        # ==================================================================
        bypass_out = self.bypass_layers(x_with_embed)  # (B*T, N, H)
        
        # ==================================================================
        # GAT PATH (spatial processing)
        # ==================================================================
        if self._gat_weight > 0:
            x_skip = self.input_proj(x_with_embed)
            
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
                    gat_new = gat_new + x_skip
                
                gat_out = norm(gat_new)
                
                if i < len(self.gat_layers) - 1:
                    gat_out = self.dropout(gat_out)
            
            # Blend GAT and bypass based on warmup progress
            spatial_out = self._gat_weight * gat_out + (1 - self._gat_weight) * bypass_out
        else:
            # Pure bypass (epoch 0 or inference with gat_weight=0)
            spatial_out = bypass_out
        
        # Reshape for temporal processing
        spatial_out = spatial_out.view(batch_size, time_steps, nodes, self.hidden_dim)
        
        # Temporal skip connection
        if self.use_skip:
            temporal_skip = spatial_out.mean(dim=1)
            temporal_skip = self.skip_temporal_proj(temporal_skip)
        
        # LSTM processing
        spatial_out = spatial_out.permute(0, 2, 1, 3).contiguous()
        spatial_out = spatial_out.view(batch_size * nodes, time_steps, self.hidden_dim)
        
        lstm_out, _ = self.lstm(spatial_out)
        last_out = lstm_out[:, -1, :]
        last_out = self.lstm_norm(last_out)
        
        # Concatenate with skip
        if self.use_skip:
            temporal_skip = temporal_skip.view(batch_size * nodes, self.hidden_dim)
            last_out = torch.cat([last_out, temporal_skip], dim=-1)
        
        # Output
        out = self.fc_out(last_out)
        out = out.view(batch_size, nodes, self.horizon, self.out_features)
        out = out.permute(0, 2, 1, 3)
        
        return out
    
    def get_attention_weights(self):
        """Get attention weights from GAT layers."""
        if not self.use_attention:
            return None
        
        weights = []
        for layer in self.gat_layers:
            if hasattr(layer, 'get_attention_weights'):
                weights.append(layer.get_attention_weights())
        return weights
    
    def get_warmup_info(self):
        """Get current warmup status."""
        return {
            'current_epoch': self._current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'gat_weight': self._gat_weight,
            'phase': 'warmup' if self._current_epoch < self.warmup_epochs else 'full'
        }
