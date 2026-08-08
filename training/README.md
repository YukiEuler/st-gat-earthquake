# Training Module

Contains all training scripts for ST-GAT earthquake prediction models.

## Entry Points

### main: `train.py`
**Default training pipeline with all features.**

Usage:
```bash
python main.py                           # Run from root
python -m training.train                 # Run as module
```

### Event-based training: `train_event.py`
**Training pipeline optimized for event-based prediction.**

Configuration: `config/config_event.py`

### Multi-resolution training: `train_multiresolution.py`
**Training with multiple spatial/temporal resolutions.**

### Hybrid training: `train_hybrid.py`
**Combined model training with multiple architectures.**

### Ensemble training: `ensemble.py`
**Deep Ensemble training for uncertainty quantification.**

## Supporting Modules

- **`trainer.py`** - Base Trainer class with training loop
- **`losses.py`** - Custom loss functions
- **`hurdle.py`** - Activity/magnitude decoding plus validation-only logit-bias and threshold fitting
- **`uncertainty.py`** - Uncertainty quantification methods
- **`cross_validation.py`** - K-fold and cross-validation utilities

## Configuration

All training scripts read from `config/` module:
- `CONFIG` from `config/base.py` - Main configuration
- `CONFIG_EVENT` from `config/config_event.py` - Event-based configuration

## Example

```python
from training.train import main
# or for multi-resolution:
from training.train_multiresolution import main
```

## Output

The canonical run is saved to `outputs/revision_canonical_v4/`. Its CSV keeps
the primary expected Mw forecast plus activity probability, conditional Mw,
and validation-thresholded event Mw. `metrics/forecast_comparison.csv` compares
the two model views with zero, persistence, recent-mean, and train-only
climatology baselines.
