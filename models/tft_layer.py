# ==============================================================================
# TFT_LAYER.PY - Temporal Fusion Transformer Components
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Key component of TFT.
    
    Provides adaptive depth with gating mechanism.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        
        # Skip connection projection if dimensions differ
        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None
        
        # Main layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Context projection (optional)
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        
        # Gating layer
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.gate_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x, context=None):
        """
        Args:
            x: (B, T, input_dim) or (B, input_dim)
            context: Optional context tensor
        Returns:
            (B, T, output_dim) or (B, output_dim)
        """
        # Skip connection
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x
        
        # GRN computation
        hidden = self.fc1(x)
        
        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_proj(context)
        
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        
        # Gating mechanism (GLU-style)
        gate = torch.sigmoid(self.gate(self.elu(self.fc1(x))))
        gated = gate * hidden
        
        # Residual connection with layer norm
        output = self.gate_norm(skip + gated)
        
        return output


class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention from TFT paper.
    
    Uses additive attention and provides interpretable attention weights.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # For interpretability - shared weights across heads
        self.attn_combine = nn.Linear(n_heads, 1, bias=False)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (B, T, d_model)
            mask: Optional attention mask
        Returns:
            output: (B, T, d_model)
            attn_weights: (B, T, T) for interpretability
        """
        B, T, _ = query.size()
        
        # Linear projections
        Q = self.W_q(query).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, n_heads, T, T)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)  # (B, n_heads, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        output = self.W_o(context)
        
        # Combine attention weights for interpretability
        combined_attn = attn_weights.permute(0, 2, 3, 1)  # (B, T, T, n_heads)
        combined_attn = self.attn_combine(combined_attn).squeeze(-1)  # (B, T, T)
        
        return output, combined_attn


class TemporalFusionEncoder(nn.Module):
    """
    Simplified Temporal Fusion Transformer Encoder.
    
    Replaces LSTM with transformer-based temporal processing.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=4, 
                 n_layers=2, dropout=0.1, max_seq_len=100):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # GRN for variable selection (optional enrichment)
        self.variable_grn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout
        )
        
        # Stack of temporal self-attention layers
        self.temporal_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': InterpretableMultiHeadAttention(hidden_dim, n_heads, dropout),
                'grn': GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim),
            })
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, input_dim)
            mask: Optional causal mask
        Returns:
            output: (B, output_dim) - last timestep output
            temporal_attn: List of attention weights for interpretability
        """
        B, T, _ = x.size()
        
        # Project and add positional encoding
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        
        # Variable selection
        h = self.variable_grn(h)
        
        # Temporal self-attention layers
        temporal_attn_weights = []
        for layer in self.temporal_layers:
            # Self-attention with residual
            residual = h
            h_norm = layer['norm1'](h)
            attn_out, attn_weights = layer['self_attn'](h_norm, h_norm, h_norm, mask)
            h = residual + self.dropout(attn_out)
            temporal_attn_weights.append(attn_weights)
            
            # GRN with residual
            residual = h
            h_norm = layer['norm2'](h)
            h = residual + layer['grn'](h_norm)
        
        # Take last timestep output
        last_output = h[:, -1, :]  # (B, hidden_dim)
        
        # Project to output dimension
        output = self.output_proj(last_output)
        
        return output, temporal_attn_weights
    
    def get_temporal_attention(self):
        """Return stored temporal attention weights."""
        return self._last_temporal_attn if hasattr(self, '_last_temporal_attn') else None
