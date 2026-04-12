# Configuration Module

Centralized configuration for ST-GAT earthquake prediction models.

## Configuration Files

### Main Config: `base.py`
**Primary configuration dictionary `CONFIG`**

Contains all hyperparameters and settings:

**Data Configuration**
```python
'filename': 'Amatrice_CAT5.v20210504.csv'
'grid_size': 0.015                    # ~1.11 km per cell
'time_bin': '4h'                      # Temporal aggregation
'lat_min/max': [42.5747, 42.9047]    # Spatial bounds
'lon_min/max': [13.1282, 13.3182]
```

**Model Configuration**
```python
'window_size': 24          # Input sequence length (hours)
'horizon': 6               # Output prediction length
'num_layers': 3            # Number of STGAT layers
'hidden_dims': [64, 32]    # Hidden dimensions
```

**Training Configuration**
```python
'learning_rate': 0.001
'batch_size': 32
'num_epochs': 100
'dropout': 0.2
```

**Device Configuration**
```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Event-based Config: `config_event.py`
**Configuration for event-based training `CONFIG_EVENT`**

Different settings optimized for event prediction:
- Smaller batch sizes
- Adjusted thresholds
- Event-specific features
- Different loss weights

## Usage

### Import Configuration
```python
from config import CONFIG, DEVICE, print_config
from config import CONFIG_EVENT, print_event_config

# Print configuration
print_config()
print_event_config()

# Access values
window_size = CONFIG['window_size']
device = DEVICE
```

### Modifying Configuration

**Option 1: Direct modification (not recommended)**
```python
from config import CONFIG
CONFIG['learning_rate'] = 0.0001
```

**Option 2: Copy and modify (recommended)**
```python
from config import CONFIG
custom_config = CONFIG.copy()
custom_config['learning_rate'] = 0.0001
custom_config['num_epochs'] = 50
```

**Option 3: Create config file (best practice)**
Create a new config file:
```python
# my_config.py
from config import CONFIG

MY_CONFIG = CONFIG.copy()
MY_CONFIG.update({
    'learning_rate': 0.0001,
    'batch_size': 16,
    'dropout': 0.3,
})
```

Then import:
```python
from my_config import MY_CONFIG
```

## Configuration Groups

### Data Settings
- `filename` - Path to earthquake catalog
- `grid_size` - Spatial discretization
- `time_bin` - Temporal aggregation window
- Spatial bounds (lat/lon min/max)
- Feature selection

### Sequence Settings  
- `window_size` - Input temporal window
- `horizon` - Output prediction length
- `train_ratio`, `val_ratio` - Data splits
- `rolling_aggregation` - Feature aggregation method

### Model Architecture
- `num_layers` - Depth of GNN
- `hidden_dims` - Layer widths
- `num_heads` - Attention heads
- `dropout` - Regularization

### Training Parameters
- `learning_rate` - Optimizer step size
- `batch_size` - Training batch size
- `num_epochs` - Training iterations
- `device` - GPU/CPU selection

### Loss & Metrics
- Loss function weights
- Class weights for imbalanced data
- Evaluation metrics to track

## Environment Variables

Set custom paths via environment variables:
```bash
export STGAT_DATA_DIR=/path/to/data
export STGAT_OUTPUT_DIR=/path/to/outputs
export STGAT_DEVICE=cuda:0  # Specific GPU
```

## Validation

Check configuration validity:
```python
from config import CONFIG
assert CONFIG['horizon'] <= CONFIG['window_size']
assert 0 < CONFIG['train_ratio'] < 1
```

## Documentation

See main [README.md](../README.md) for full project documentation and config examples.
