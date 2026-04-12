# ==============================================================================
# STGAT_LEARNABLE.PY - ST-GAT with Learnable Graph Structure
# ==============================================================================
"""
ST-GAT with Hybrid Learnable Graph:
  - Geographic Prior: Fixed adjacency based on spatial distance
  - Learnable Corrections: Model learns additional edge weights/connections

This allows discovering hidden spatial relationships while respecting 
geographic constraints (closer nodes still have higher prior weight).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_layer import MultiHeadGATLayer


class STGATLearnable(nn.Module):
    """
    ST-GAT with Hybrid Learnable Adjacency Matrix.
    
    Adjacency = α * Geo_Adj + (1-α) * Learned_Adj
    
    Where:
    - Geo_Adj: Fixed geographic distance-based adjacency
    - Learned_Adj: Learnable parameters capturing hidden relationships
    - α: Blend ratio (geographic_weight parameter)
    """
    
    def __init__(self, num_nodes, in_features, hidden_dim, out_features,
                 horizon=24, num_gat_layers=2, num_heads=4, dropout=0.2,
                 use_attention=True, use_multihead=True, use_skip=True,
                 node_embed_dim=16, geographic_weight=0.5, learn_full_graph=False):
        """
        Args:
            geographic_weight: Weight for geographic prior (0-1). 
                              Higher = more reliance on fixed geography.
            learn_full_graph: If True, learn all N*N connections.
                             If False, only learn corrections to existing edges.
        """
        super(STGATLearnable, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads if use_multihead else 1
        self.horizon = horizon
        self.out_features = out_features
        self.use_attention = use_attention
        self.use_skip = use_skip
        self.node_embed_dim = node_embed_dim
        self.geographic_weight = geographic_weight
        self.learn_full_graph = learn_full_graph
        
        # ==================================================================
        # LEARNABLE ADJACENCY COMPONENTS
        # ==================================================================
        
        # Node embeddings for computing learned adjacency
        self.node_source_embed = nn.Embedding(num_nodes, hidden_dim // 2)
        self.node_target_embed = nn.Embedding(num_nodes, hidden_dim // 2)
        
        # Learnable adjacency bias (direct parameterization)
        if learn_full_graph:
            # Learn full N x N matrix
            self.adj_bias = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        else:
            # Smaller parameter for edge-wise corrections
            self.adj_bias = None
        
        # Temperature for softmax (learnable)
        self.adj_temperature = nn.Parameter(torch.tensor(1.0))
        
        # ==================================================================
        # STANDARD ST-GAT COMPONENTS
        # ==================================================================
        
        # Learnable node embeddings for features
        if node_embed_dim > 0:
            self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
            effective_in_features = in_features + node_embed_dim
        else:
            self.node_embedding = None
            effective_in_features = in_features
        
        # Input projection for skip connection
        self.input_proj = nn.Linear(effective_in_features, hidden_dim)
        
        # GAT Layers
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
            # Simple linear projection
            self.gat_layers.append(nn.Linear(effective_in_features, hidden_dim))
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
            for _ in range(num_gat_layers - 1):
                self.gat_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.gat_norms.append(nn.LayerNorm(hidden_dim))
        
        # LSTM for temporal processing
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
    
    def compute_learned_adjacency(self, device):
        """
        Compute the learned adjacency matrix from node embeddings.
        
        Returns:
            learned_adj: (N, N) tensor with learned edge weights
        """
        node_ids = torch.arange(self.num_nodes, device=device)
        
        # Get source and target embeddings
        source_emb = self.node_source_embed(node_ids)  # (N, D/2)
        target_emb = self.node_target_embed(node_ids)  # (N, D/2)
        
        # Compute pairwise similarity: source_i * target_j
        # (N, D/2) @ (D/2, N) -> (N, N)
        similarity = torch.matmul(source_emb, target_emb.T)
        
        # Add bias if using full graph learning
        if self.adj_bias is not None:
            similarity = similarity + self.adj_bias
        
        # Apply temperature-scaled softmax for normalization
        # Each row sums to 1 (outgoing edge weights)
        learned_adj = F.softmax(similarity / self.adj_temperature.clamp(min=0.1), dim=-1)
        
        return learned_adj
    
    def get_hybrid_adjacency(self, geo_adj_sparse, device):
        """
        Combine geographic and learned adjacency matrices.
        
        Args:
            geo_adj_sparse: Sparse geographic adjacency (N, N)
            device: Torch device
        
        Returns:
            hybrid_adj: (N, N) dense tensor or sparse tensor
        """
        # Convert geographic adj to dense
        geo_adj_dense = geo_adj_sparse.to_dense()  # (N, N)
        
        # Normalize geographic adjacency (row-wise)
        geo_adj_sum = geo_adj_dense.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        geo_adj_norm = geo_adj_dense / geo_adj_sum
        
        # Compute learned adjacency
        learned_adj = self.compute_learned_adjacency(device)
        
        # Blend: α * geo + (1-α) * learned
        alpha = self.geographic_weight
        hybrid_adj = alpha * geo_adj_norm + (1 - alpha) * learned_adj
        
        # Convert back to sparse for efficiency
        hybrid_sparse = hybrid_adj.to_sparse()
        
        return hybrid_sparse
    
    def forward(self, x, adj_sparse):
        """
        Args:
            x: Input tensor (B, T, N, F)
            adj_sparse: Geographic sparse adjacency (N, N)
        
        Returns:
            Predictions (B, H, N, F')
        """
        batch_size, time_steps, nodes, features = x.size()
        device = x.device
        
        # Compute hybrid adjacency (geographic + learned)
        hybrid_adj = self.get_hybrid_adjacency(adj_sparse, device)
        
        # Reshape input
        x_reshaped = x.view(batch_size * time_steps, nodes, features)
        
        # Add learnable node embeddings
        if self.node_embedding is not None:
            node_ids = torch.arange(nodes, device=device)
            node_embeds = self.node_embedding(node_ids)
            node_embeds = node_embeds.unsqueeze(0).expand(batch_size * time_steps, -1, -1)
            x_with_embed = torch.cat([x_reshaped, node_embeds], dim=-1)
        else:
            x_with_embed = x_reshaped
        
        # Project input for skip connection
        x_skip = self.input_proj(x_with_embed)
        
        # GAT layers with hybrid adjacency
        gat_out = x_with_embed
        for i, (gat_layer, norm) in enumerate(zip(self.gat_layers, self.gat_norms)):
            if self.use_attention:
                gat_new = gat_layer(gat_out, hybrid_adj)
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
        
        # Reshape for temporal processing
        gat_out = gat_out.view(batch_size, time_steps, nodes, self.hidden_dim)
        
        # Temporal skip connection
        if self.use_skip:
            temporal_skip = gat_out.mean(dim=1)
            temporal_skip = self.skip_temporal_proj(temporal_skip)
        
        # LSTM processing
        gat_out = gat_out.permute(0, 2, 1, 3).contiguous()
        gat_out = gat_out.view(batch_size * nodes, time_steps, self.hidden_dim)
        
        lstm_out, _ = self.lstm(gat_out)
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
    
    def get_learned_adjacency(self, device='cpu'):
        """Get the learned adjacency matrix for visualization."""
        return self.compute_learned_adjacency(device).detach()
    
    def get_attention_weights(self):
        """Get attention weights from GAT layers."""
        if not self.use_attention:
            return None
        
        weights = []
        for layer in self.gat_layers:
            if hasattr(layer, 'get_attention_weights'):
                weights.append(layer.get_attention_weights())
        return weights
