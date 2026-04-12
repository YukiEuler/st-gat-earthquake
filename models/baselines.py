# ==============================================================================
# BASELINES.PY - Baseline Models for Comparison
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NaiveBaseline:
    """Naive baseline: predict last observation."""
    
    def __init__(self, horizon=24, out_features=None):
        self.horizon = horizon
        self.out_features = out_features  # Number of target features to predict
        self.name = "Naive (Last Observation)"
    
    def predict(self, x):
        """
        Args:
            x: Input (B, T, N, F)
        Returns:
            Predictions (B, H, N, F') where F' = out_features or F if not specified
        """
        last_obs = x[:, -1:, :, :]  # (B, 1, N, F)
        
        # Only take the first out_features if specified
        if self.out_features is not None:
            last_obs = last_obs[..., :self.out_features]
        
        return last_obs.repeat(1, self.horizon, 1, 1)


class MovingAverageBaseline:
    """Moving average baseline."""
    
    def __init__(self, window=5, horizon=24, out_features=None):
        self.window = window
        self.horizon = horizon
        self.out_features = out_features
        self.name = f"Moving Average (w={window})"
    
    def predict(self, x):
        """
        Args:
            x: Input (B, T, N, F)
        Returns:
            Predictions (B, H, N, F') where F' = out_features or F if not specified
        """
        avg = x[:, -self.window:, :, :].mean(dim=1, keepdim=True)  # (B, 1, N, F)
        
        # Only take the first out_features if specified
        if self.out_features is not None:
            avg = avg[..., :self.out_features]
        
        return avg.repeat(1, self.horizon, 1, 1)


class LSTMOnly(nn.Module):
    """LSTM-only model (no spatial component)."""
    
    def __init__(self, num_nodes, in_features, hidden_dim, out_features, 
                 horizon=24, dropout=0.2):
        super(LSTMOnly, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.out_features = out_features
        self.name = "LSTM Only"
        
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features * horizon)
        )
    
    def forward(self, x, adj_sparse=None):
        """
        Args:
            x: Input (B, T, N, F)
            adj_sparse: Ignored (for API compatibility)
        Returns:
            Predictions (B, H, N, F')
        """
        batch_size, time_steps, nodes, features = x.size()
        
        # Reshape: (B, T, N, F) -> (B*N, T, F)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * nodes, time_steps, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # (B*N, H)
        
        # Output
        out = self.fc_out(last_out)  # (B*N, F'*horizon)
        out = out.view(batch_size, nodes, self.horizon, self.out_features)
        out = out.permute(0, 2, 1, 3)  # (B, H, N, F')
        
        return out


class GCNLayer(nn.Module):
    """Simple GCN layer (no attention)."""
    
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj_sparse):
        """
        Args:
            x: Node features (B, N, F) or (N, F)
            adj_sparse: Normalized adjacency matrix
        """
        # Linear transformation
        x = self.linear(x)
        
        # Graph convolution using sparse matrix multiplication
        if x.dim() == 3:
            batch_size, nodes, features = x.size()
            x = x.view(batch_size * nodes, features)
            
            # Sparse matmul
            adj_dense = adj_sparse.to_dense()
            x = x.view(batch_size, nodes, features)
            x = torch.bmm(adj_dense.unsqueeze(0).expand(batch_size, -1, -1), x)
        else:
            adj_dense = adj_sparse.to_dense()
            x = torch.mm(adj_dense, x)
        
        return F.relu(x)


class GCNLSTM(nn.Module):
    """GCN + LSTM model (no attention mechanism)."""
    
    def __init__(self, num_nodes, in_features, hidden_dim, out_features,
                 horizon=24, num_gcn_layers=2, dropout=0.2):
        super(GCNLSTM, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.out_features = out_features
        self.name = "GCN + LSTM"
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(in_features, hidden_dim))
        for _ in range(num_gcn_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Output
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features * horizon)
        )
    
    def forward(self, x, adj_sparse):
        batch_size, time_steps, nodes, features = x.size()
        
        # GCN processing
        x_reshaped = x.view(batch_size * time_steps, nodes, features)
        
        gcn_out = x_reshaped
        for i, gcn_layer in enumerate(self.gcn_layers):
            gcn_out = gcn_layer(gcn_out, adj_sparse)
            if i < len(self.gcn_layers) - 1:
                gcn_out = self.dropout(gcn_out)
        
        gcn_out = gcn_out.view(batch_size, time_steps, nodes, self.hidden_dim)
        
        # LSTM processing
        gcn_out = gcn_out.permute(0, 2, 1, 3).contiguous()
        gcn_out = gcn_out.view(batch_size * nodes, time_steps, self.hidden_dim)
        
        lstm_out, _ = self.lstm(gcn_out)
        last_out = lstm_out[:, -1, :]
        
        # Output
        out = self.fc_out(last_out)
        out = out.view(batch_size, nodes, self.horizon, self.out_features)
        out = out.permute(0, 2, 1, 3)
        
        return out


class TFTOnly(nn.Module):
    """
    Temporal Fusion Transformer Only (no spatial/graph component).
    
    Uses transformer-based temporal processing with:
    - Positional encoding
    - Multi-head self-attention
    - Gated Residual Networks (GRN)
    """
    
    def __init__(self, num_nodes, in_features, hidden_dim, out_features,
                 horizon=24, n_heads=4, n_layers=2, dropout=0.2):
        super(TFTOnly, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.out_features = out_features
        self.name = "TFT Only"
        
        # Import TFT components
        from .tft_layer import TemporalFusionEncoder
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # TFT Encoder (per node)
        self.tft_encoder = TemporalFusionEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Output layers
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features * horizon)
        )
    
    def forward(self, x, adj_sparse=None):
        """
        Args:
            x: Input (B, T, N, F)
            adj_sparse: Ignored (for API compatibility)
        Returns:
            Predictions (B, H, N, F')
        """
        batch_size, time_steps, nodes, features = x.size()
        
        # Reshape: (B, T, N, F) -> (B*N, T, F)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * nodes, time_steps, features)
        
        # Project input
        x = self.input_proj(x)  # (B*N, T, H)
        
        # TFT encoding
        tft_out, _ = self.tft_encoder(x)  # (B*N, H)
        
        # Output projection
        out = self.fc_out(tft_out)  # (B*N, F'*horizon)
        out = out.view(batch_size, nodes, self.horizon, self.out_features)
        out = out.permute(0, 2, 1, 3)  # (B, H, N, F')
        
        return out


class ETASBaseline:
    """
    ETAS-like (Epidemic Type Aftershock Sequence) Statistical Baseline.
    
    Simplified ETAS model that predicts based on:
    - Background seismicity rate (μ)
    - Triggered activity from recent events (Omori-Utsu decay)
    
    Formula: λ(t) = μ + Σ K * exp(α(M-Mc)) * (t - ti + c)^(-p)
    
    Note: This is a simplified version that uses empirical rates from data.
    """
    
    def __init__(self, horizon=24, out_features=None, decay_p=1.2, 
                 background_weight=0.3, triggered_weight=0.7):
        """
        Args:
            horizon: Prediction horizon (number of future time steps)
            out_features: Number of target features
            decay_p: Omori-Utsu decay exponent (typically 0.8-1.5)
            background_weight: Weight for background rate
            triggered_weight: Weight for triggered activity
        """
        self.horizon = horizon
        self.out_features = out_features
        self.decay_p = decay_p
        self.background_weight = background_weight
        self.triggered_weight = triggered_weight
        self.name = f"ETAS Baseline (p={decay_p})"
    
    def predict(self, x):
        """
        Args:
            x: Input (B, T, N, F) - historical seismic data
        Returns:
            Predictions (B, H, N, F')
        
        Prediction strategy:
        1. Background rate: Mean of all historical values
        2. Triggered rate: Omori-Utsu decay from most recent high activity
        """
        import torch
        
        batch_size, time_steps, nodes, features = x.size()
        
        # Get target features
        if self.out_features is not None:
            x_target = x[..., :self.out_features]
        else:
            x_target = x
            self.out_features = features
        
        # Calculate background rate (long-term mean per node)
        background_rate = x_target.mean(dim=1, keepdim=True)  # (B, 1, N, F')
        
        # Calculate triggered rate with Omori-Utsu decay
        # Recent activity contributes more than older activity
        triggered_rates = []
        for h in range(self.horizon):
            # Calculate decay weights for each historical timestep
            # Most recent timestep has highest weight
            decay_weights = []
            c = 0.1  # Small constant to avoid division by zero
            
            for t in range(time_steps):
                time_since = time_steps - t + h  # Time distance from prediction point
                weight = (time_since + c) ** (-self.decay_p)
                decay_weights.append(weight)
            
            # Normalize weights
            decay_weights = torch.tensor(decay_weights, device=x.device, dtype=x.dtype)
            decay_weights = decay_weights / decay_weights.sum()
            
            # Weighted sum of historical activity
            # (B, T, N, F') * (T,) -> weighted sum -> (B, N, F')
            weighted_sum = (x_target * decay_weights.view(1, -1, 1, 1)).sum(dim=1)
            triggered_rates.append(weighted_sum)
        
        # Stack triggered rates: (H, B, N, F') -> (B, H, N, F')
        triggered_rate = torch.stack(triggered_rates, dim=1)
        
        # Combine background and triggered rates
        prediction = (
            self.background_weight * background_rate.expand(-1, self.horizon, -1, -1) +
            self.triggered_weight * triggered_rate
        )
        
        return prediction
    
    def fit(self, train_data):
        """
        Optional: Fit ETAS parameters from training data.
        
        For simplicity, we use fixed parameters, but this could be extended
        to estimate optimal decay_p from the data.
        """
        # Could implement MLE estimation for decay_p here
        pass

