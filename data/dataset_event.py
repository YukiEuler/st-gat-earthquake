# ==============================================================================
# DATASET_EVENT.PY - Event-based PyTorch Dataset
# ==============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class EventSequenceDataset(Dataset):
    """
    PyTorch Dataset for event-based earthquake sequences.
    
    Works with both:
    - Global sequences: (samples, window_size, features)
    - Node sequences: (samples, window_size, nodes, features)
    """
    
    def __init__(self, X, Y):
        """
        Args:
            X: Input sequences (numpy array)
            Y: Target sequences (numpy array)
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def create_event_dataloaders(data, batch_size=32, num_workers=0):
    """
    Create DataLoaders for event-based data.
    
    Args:
        data: Dictionary from EventBasedPreprocessor.preprocess()
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = EventSequenceDataset(data['train_X'], data['train_Y'])
    test_dataset = EventSequenceDataset(data['test_X'], data['test_Y'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f" DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, test_loader
