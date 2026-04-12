# ==============================================================================
# TRAINER.PY - Training Loop
# ==============================================================================

import torch
import torch.optim as optim
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path


class Trainer:
    """Handles model training with early stopping and checkpointing."""
    
    def __init__(self, model, criterion, config, device):
        self.model = model
        self.criterion = criterion
        self.config = config
        self.device = device
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
    def train_epoch(self, train_loader, adj_sparse):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        pbar = tqdm(train_loader, desc="Train", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data, adj_sparse)
            
            loss = self.criterion(output, target)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f" NaN/Inf loss at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            pbar.set_postfix({'loss': total_loss / max(1, total_batches)})
        
        return total_loss / max(1, total_batches)
    
    def evaluate(self, test_loader, adj_sparse):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        pbar = tqdm(test_loader, desc="Eval ", leave=False)
        
        with torch.no_grad():
            for data, target in pbar:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data, adj_sparse)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                total_batches += 1
                pbar.set_postfix({'loss': total_loss / max(1, total_batches)})
        
        return total_loss / max(1, total_batches)
    
    def fit(self, train_loader, val_loader, adj_sparse, epochs=None, patience=None):
        """Full training loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (used for early stopping)
            adj_sparse: Sparse adjacency matrix
            epochs: Number of epochs to train
            patience: Early stopping patience
        """
        epochs = epochs or self.config['epochs']
        patience = patience or self.config['early_stopping_patience']
        
        print("\n" + "=" * 70)
        print(" Starting Training")
        print("=" * 70)
        
        for epoch in range(epochs):
            # Warm start support: update GAT weight for curriculum learning
            if hasattr(self.model, 'set_epoch'):
                gat_weight = self.model.set_epoch(epoch)
                if epoch < getattr(self.model, 'warmup_epochs', 0):
                    print(f"  [WarmStart] GAT weight: {gat_weight:.2f}")
            
            train_loss = self.train_epoch(train_loader, adj_sparse)
            val_loss = self.evaluate(val_loader, adj_sparse)
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping check (based on validation loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                status = " Best"
            else:
                self.patience_counter += 1
                status = f"  ({self.patience_counter}/{patience})"
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e} {status}")
            
            if self.patience_counter >= patience:
                print(f"\n Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\n Loaded best model (Val Loss: {self.best_loss:.6f})")
        
        print("=" * 70)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
            'epochs_trained': len(self.train_losses),
        }
    
    def save_checkpoint(self, path, extra_info=None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        if extra_info:
            checkpoint.update(extra_info)
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f" Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f" Checkpoint loaded: {path}")
        return checkpoint
