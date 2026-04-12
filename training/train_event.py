# ==============================================================================
# MAIN_EVENT.PY - Entry Point for Event-based Earthquake Prediction
# ==============================================================================

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from config import CONFIG_EVENT, print_event_config
from data.preprocessing_event import EventBasedPreprocessor
from data.dataset_event import create_event_dataloaders
from models.stgat import STGAT
from training.trainer import Trainer
from training.losses import WeightedMSELoss
from evaluation.metrics import MetricsCalculator
from visualization.predictions import PredictionVisualizer


# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EventSequenceModel(nn.Module):
    """
    Simple LSTM model for event sequence prediction.
    
    For global sequences (no spatial graph).
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_layers=2, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj=None):
        """
        Args:
            x: (batch, seq_len, features)
            adj: Not used (for compatibility)
        
        Returns:
            (batch, output_dim)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        
        # Project to output
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out


def train_event_model():
    """Train event-based model."""
    print("=" * 70)
    print(" EVENT-BASED EARTHQUAKE PREDICTION")
    print("=" * 70)
    print_event_config()
    print(f" Device: {DEVICE}")
    
    # Preprocessing
    preprocessor = EventBasedPreprocessor(
        grid_size=CONFIG_EVENT['grid_size'],
        min_magnitude=CONFIG_EVENT.get('min_magnitude', 0.0),
        max_events=CONFIG_EVENT.get('max_events', None),
        lat_min=CONFIG_EVENT.get('lat_min'),
        lat_max=CONFIG_EVENT.get('lat_max'),
        lon_min=CONFIG_EVENT.get('lon_min'),
        lon_max=CONFIG_EVENT.get('lon_max')
    )
    
    data = preprocessor.preprocess(
        filepath=CONFIG_EVENT['filename'],
        window_size=CONFIG_EVENT['window_size'],
        horizon=CONFIG_EVENT['horizon'],
        use_node_sequences=CONFIG_EVENT['use_node_sequences']
    )
    
    # Create dataloaders
    train_loader, test_loader = create_event_dataloaders(
        data,
        batch_size=CONFIG_EVENT['batch_size']
    )
    
    # Determine input/output dimensions
    sample_x, sample_y = next(iter(train_loader))
    print(f"\n Sample X shape: {sample_x.shape}")
    print(f" Sample Y shape: {sample_y.shape}")
    
    if sample_x.dim() == 3:  # (batch, seq, features)
        input_dim = sample_x.shape[-1]
        output_dim = sample_y.shape[-1]
        
        # Use simple sequence model
        model = EventSequenceModel(
            input_dim=input_dim,
            hidden_dim=CONFIG_EVENT['hidden_dim'],
            output_dim=output_dim,
            num_layers=CONFIG_EVENT['num_layers'],
            dropout=CONFIG_EVENT['dropout']
        ).to(DEVICE)
        
        adj_sparse = None
        
    elif sample_x.dim() == 4:  # (batch, seq, nodes, features)
        # Use ST-GAT
        n_nodes = sample_x.shape[2]
        in_features = sample_x.shape[-1]
        out_features = sample_y.shape[-1]
        
        model = STGAT(
            in_features=in_features,
            hidden_dim=CONFIG_EVENT['hidden_dim'],
            out_features=out_features,
            n_nodes=n_nodes,
            num_gat_layers=CONFIG_EVENT['num_layers'],
            num_heads=CONFIG_EVENT['num_heads'],
            dropout=CONFIG_EVENT['dropout'],
            horizon=CONFIG_EVENT['horizon']
        ).to(DEVICE)
        
        # Convert adjacency to sparse
        adj = torch.from_numpy(data['adjacency']).float()
        adj_sparse = adj.to_sparse().to(DEVICE)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n Model Parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG_EVENT['learning_rate'],
        weight_decay=CONFIG_EVENT['weight_decay']
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print(" Starting Training")
    print("=" * 70)
    
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    test_losses = []
    
    for epoch in range(CONFIG_EVENT['epochs']):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            
            if adj_sparse is not None:
                output = model(batch_x, adj_sparse)
            else:
                output = model(batch_x)
            
            # Squeeze target if needed
            if batch_y.dim() == 3 and batch_y.shape[1] == 1:
                batch_y = batch_y.squeeze(1)
            
            loss = criterion(output, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                if adj_sparse is not None:
                    output = model(batch_x, adj_sparse)
                else:
                    output = model(batch_x)
                
                if batch_y.dim() == 3 and batch_y.shape[1] == 1:
                    batch_y = batch_y.squeeze(1)
                
                loss = criterion(output, batch_y)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        # Check best
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_event_model.pth')
            status = " Best"
        else:
            patience_counter += 1
            status = f" ({patience_counter}/{CONFIG_EVENT['early_stopping_patience']})"
        
        print(f"Epoch {epoch+1:3d}/{CONFIG_EVENT['epochs']} | "
              f"Train: {train_loss:.6f} | Test: {test_loss:.6f}{status}")
        
        # Early stopping
        if patience_counter >= CONFIG_EVENT['early_stopping_patience']:
            print(f"\n Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_event_model.pth'))
    print(f"\n Loaded best model (Test Loss: {best_loss:.6f})")
    
    # Generate predictions
    print("\n Generating predictions...")
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            
            if adj_sparse is not None:
                output = model(batch_x, adj_sparse)
            else:
                output = model(batch_x)
            
            predictions.append(output.cpu().numpy())
            
            if batch_y.dim() == 3 and batch_y.shape[1] == 1:
                batch_y = batch_y.squeeze(1)
            targets.append(batch_y.numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Denormalize
    stats = data['feature_stats']
    n_targets = predictions.shape[-1]
    
    pred_denorm = predictions * stats['std'][:n_targets] + stats['mean'][:n_targets]
    target_denorm = targets * stats['std'][:n_targets] + stats['mean'][:n_targets]
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print(" EVALUATION METRICS")
    print("=" * 70)
    
    target_names = data['target_names']
    
    for i, name in enumerate(target_names):
        pred_flat = pred_denorm[..., i].flatten()
        target_flat = target_denorm[..., i].flatten()
        
        mse = np.mean((pred_flat - target_flat)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_flat - target_flat))
        
        # R²
        ss_res = np.sum((target_flat - pred_flat)**2)
        ss_tot = np.sum((target_flat - target_flat.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\n {name}:")
        print(f"   MSE:  {mse:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE:  {mae:.6f}")
        print(f"   R²:   {r2:.4f}")
    
    print("\n" + "=" * 70)
    print(" Training Complete!")
    print("=" * 70)
    
    return model, data, predictions, targets


if __name__ == '__main__':
    train_event_model()
