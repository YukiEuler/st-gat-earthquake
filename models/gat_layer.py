# ==============================================================================
# GAT_LAYER.PY - Graph Attention Layers
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGATLayer(nn.Module):
    """
    Graph Attention Layer with explicit Query-Key-Value projections.
    Supports sparse adjacency matrices for memory efficiency.
    """
    
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2,
                 concat=True, use_edge_weights=True):
        super(SparseGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout
        self.alpha = alpha
        self.concat = concat
        self.use_edge_weights = use_edge_weights
        
        # Q, K, V projections (Transformer-style)
        self.linear_q = nn.Linear(in_features, out_features, bias=False)
        self.linear_k = nn.Linear(in_features, out_features, bias=False)
        self.linear_v = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism
        self.attention_layer = nn.Sequential(
            nn.Linear(2 * out_features, out_features, bias=True),
            nn.LeakyReLU(alpha),
            nn.Linear(out_features, 1, bias=False)
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        # For attention visualization
        self.last_attention_weights = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        for layer in self.attention_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x, adj_sparse):
        """
        Args:
            x: Node features (B, N, F) or (N, F)
            adj_sparse: Coalesced sparse adjacency tensor (N, N)
        
        Returns:
            Updated node features
        """
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)
        
        edge_index = adj_sparse.indices()
        edge_values = adj_sparse.values()
        src, dst = edge_index[0], edge_index[1]
        num_edges = src.size(0)
        
        if x.dim() == 3:
            # Batch mode: (B, N, F)
            batch_size, N, _ = Q.size()
            
            Q_src = Q[:, src, :]
            K_dst = K[:, dst, :]
            V_dst = V[:, dst, :]
            
            concat_qk = torch.cat([Q_src, K_dst], dim=2)
            concat_qk_flat = concat_qk.view(batch_size * num_edges, -1)
            attention_logits = self.attention_layer(concat_qk_flat)
            attention_logits = attention_logits.view(batch_size, num_edges)

            # The adjacency is not only a connectivity mask: its normalized
            # Gaussian values now influence the attention prior.
            if self.use_edge_weights:
                attention_logits = attention_logits + torch.log(edge_values.clamp_min(1e-12)).view(1, -1)
            
            # Numerical stability
            attention_logits = attention_logits - attention_logits.max(dim=1, keepdim=True)[0]
            attention_weights = torch.exp(attention_logits)
            
            # Normalize
            normalizer = torch.zeros(batch_size, N, device=x.device, dtype=x.dtype)
            normalizer.index_add_(1, dst, attention_weights)
            normalizer = normalizer[:, dst].clamp(min=1e-8)
            
            attention_weights = attention_weights / normalizer
            attention_weights = self.dropout_layer(attention_weights)
            
            # Store for visualization
            self.last_attention_weights = attention_weights.detach()
            
            # Aggregate
            out = torch.zeros(batch_size, N, self.out_features, device=x.device, dtype=x.dtype)
            weighted_values = attention_weights.unsqueeze(2) * V_dst
            out.index_add_(1, dst, weighted_values)
            
            return F.elu(out)
        else:
            # Single mode: (N, F)
            N = Q.size(0)
            
            Q_src = Q[src]
            K_dst = K[dst]
            V_dst = V[dst]
            
            concat_qk = torch.cat([Q_src, K_dst], dim=1)
            attention_logits = self.attention_layer(concat_qk).squeeze(1)

            if self.use_edge_weights:
                attention_logits = attention_logits + torch.log(edge_values.clamp_min(1e-12))
            
            attention_logits = attention_logits - attention_logits.max()
            attention_weights = torch.exp(attention_logits)
            
            normalizer = torch.zeros(N, device=x.device, dtype=x.dtype)
            normalizer.index_add_(0, dst, attention_weights)
            normalizer = normalizer[dst].clamp(min=1e-8)
            
            attention_weights = attention_weights / normalizer
            attention_weights = self.dropout_layer(attention_weights)
            
            self.last_attention_weights = attention_weights.detach()
            
            out = torch.zeros(N, self.out_features, device=x.device, dtype=x.dtype)
            weighted_values = attention_weights.unsqueeze(1) * V_dst
            out.index_add_(0, dst, weighted_values)
            
            return F.elu(out)


class MultiHeadGATLayer(nn.Module):
    """Multi-Head Graph Attention Layer."""
    
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2,
                 alpha=0.2, concat=True, use_edge_weights=True):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.out_features = out_features
        self.use_edge_weights = use_edge_weights
        
        if concat:
            assert out_features % num_heads == 0
            self.head_dim = out_features // num_heads
        else:
            self.head_dim = out_features
        
        self.heads = nn.ModuleList([
            SparseGATLayer(
                in_features, self.head_dim, dropout, alpha,
                concat=True, use_edge_weights=use_edge_weights
            )
            for _ in range(num_heads)
        ])
    
    def forward(self, x, adj_sparse):
        head_outputs = [head(x, adj_sparse) for head in self.heads]
        
        if self.concat:
            if x.dim() == 3:
                return torch.cat(head_outputs, dim=2)
            else:
                return torch.cat(head_outputs, dim=1)
        else:
            stacked = torch.stack(head_outputs, dim=0)
            return torch.mean(stacked, dim=0)
    
    def get_attention_weights(self):
        """Get attention weights from all heads for visualization."""
        return [head.last_attention_weights for head in self.heads]
