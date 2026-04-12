# ==============================================================================
# STGAT_HYBRID.PY - Hybrid Classification-Regression ST-GAT
# ==============================================================================
"""
Multi-task ST-GAT dengan:
- Classification head untuk max_mw (signifikan gempa detection)
- Regression head untuk count, log_energy, avg_depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_layer import MultiHeadGATLayer


class STGATHybrid(nn.Module):
    """
    Hybrid Spatio-Temporal GAT dengan dual output:
    - Classification: Apakah max_mw >= threshold (binary)
    - Regression: count, log_energy, avg_depth
    """
    
    def __init__(self, num_nodes, in_features, hidden_dim, 
                 n_regression_features=3,  # count, log_energy, avg_depth
                 n_classes=4,              # 4-class: M<1, M1-2, M2-3, M>=3
                 horizon=24, num_gat_layers=2, num_heads=4, dropout=0.2,
                 use_attention=True, use_multihead=True):
        super(STGATHybrid, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_gat_layers = num_gat_layers
        self.num_heads = num_heads if use_multihead else 1
        self.horizon = horizon
        self.n_regression_features = n_regression_features
        self.n_classes = n_classes
        self.use_attention = use_attention
        
        # ==================================================
        # SHARED BACKBONE: GAT Layers (Spatial)
        # ==================================================
        self.gat_layers = nn.ModuleList()
        
        if use_attention:
            self.gat_layers.append(
                MultiHeadGATLayer(
                    in_features=in_features,
                    out_features=hidden_dim,
                    num_heads=self.num_heads,
                    dropout=dropout,
                    concat=True
                )
            )
            
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
            self.gat_layers.append(nn.Linear(in_features, hidden_dim))
            for _ in range(num_gat_layers - 1):
                self.gat_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # ==================================================
        # SHARED BACKBONE: LSTM (Temporal)
        # ==================================================
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if hidden_dim > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # ==================================================
        # HEAD 1: REGRESSION (count, log_energy, avg_depth)
        # ==================================================
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_regression_features * horizon)
        )
        
        # ==================================================
        # HEAD 2: CLASSIFICATION (max_mw significant or not)
        # ==================================================
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes * horizon)  # logits for each class
        )
        
    def forward(self, x, adj_sparse):
        """
        Args:
            x: Input tensor (B, T, N, F)
            adj_sparse: Coalesced sparse adjacency (N, N)
        
        Returns:
            dict with:
                - 'regression': (B, H, N, n_regression_features)
                - 'classification': (B, H, N, n_classes) - logits
        """
        batch_size, time_steps, nodes, features = x.size()
        
        # ==================================================
        # SHARED BACKBONE: Spatial Processing (GAT)
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
        
        gat_out = gat_out.view(batch_size, time_steps, nodes, self.hidden_dim)
        
        # ==================================================
        # SHARED BACKBONE: Temporal Processing (LSTM)
        # ==================================================
        gat_out = gat_out.permute(0, 2, 1, 3).contiguous()
        gat_out = gat_out.view(batch_size * nodes, time_steps, self.hidden_dim)
        
        lstm_out, _ = self.lstm(gat_out)
        last_out = lstm_out[:, -1, :]  # (B*N, H)
        
        # ==================================================
        # HEAD 1: Regression Output
        # ==================================================
        reg_out = self.regression_head(last_out)  # (B*N, n_reg*horizon)
        reg_out = reg_out.view(batch_size, nodes, self.horizon, self.n_regression_features)
        reg_out = reg_out.permute(0, 2, 1, 3)  # (B, H, N, n_reg)
        
        # ==================================================
        # HEAD 2: Classification Output
        # ==================================================
        cls_out = self.classification_head(last_out)  # (B*N, n_cls*horizon)
        cls_out = cls_out.view(batch_size, nodes, self.horizon, self.n_classes)
        cls_out = cls_out.permute(0, 2, 1, 3)  # (B, H, N, n_cls)
        
        return {
            'regression': reg_out,
            'classification': cls_out,  # raw logits
            'classification_proba': F.softmax(cls_out, dim=-1),  # probabilities
        }
    
    def get_attention_weights(self):
        """Get attention weights from all GAT layers for visualization."""
        if not self.use_attention:
            return None
        
        weights = []
        for layer in self.gat_layers:
            if hasattr(layer, 'get_attention_weights'):
                weights.append(layer.get_attention_weights())
        return weights
