# ST-GAT: Spatio-Temporal Graph Attention Networks for Microseismic Magnitude Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A deep learning framework for predicting microseismic magnitude sequences using spatio-temporal graph attention networks with calibrated uncertainty estimation.

## Overview

This project implements **ST-GAT**, a novel architecture that combines:
- **Graph Attention Networks (GAT)** for learning spatial interaction patterns in earthquake sequences
- **LSTM networks** for capturing temporal dynamics
- **Deep Ensembles** with post-hoc calibration for reliable uncertainty quantification

The model is evaluated on the **2016–2017 Amatrice aftershock sequence** (457,874 events, 39×28 km region) and achieves:
- **R² = 0.659** (average across 6 temporal resolutions: 1–24 hours)
- **F1 = 0.778** for significant-event detection (M_w ≥ 1.0)
- **95% calibrated coverage** for confidence intervals

## Key Features

✅ **Multi-Resolution Forecasting**: Separate models per temporal resolution (1, 2, 4, 6, 12, 24 hours)  
✅ **Spatial-Temporal Learning**: Graph attention + LSTM for inter-location coupling  
✅ **Calibrated Uncertainty**: Deep Ensembles with post-hoc scaling for trustworthy intervals  
✅ **Comprehensive Evaluation**: Regression metrics, event detection, ablation studies, attention analysis  
✅ **Interpretable Patterns**: Learned attention weights reveal geophysically meaningful stress-transfer pathways  

## Project Structure

```
st-gat-earthquake/
├── README.md                          # This file
├── main.py                            # Root entry point (mode selector)
├── requirements.txt                   # Python dependencies
├── Amatrice_CAT5.v20210504.csv        # Amatrice earthquake catalog
│
├── config/                            # All configuration files
│   ├── __init__.py
│   ├── base.py                        # Main configuration (CONFIG)
│   ├── config_event.py                # Event-based config (CONFIG_EVENT)
│   └── README.md
│
├── training/                          # Training entry points and utilities
│   ├── __init__.py
│   ├── train.py                       # Default training pipeline
│   ├── train_multiresolution.py       # Multi-resolution training
│   ├── train_event.py                 # Event-based training
│   ├── train_hybrid.py                # Hybrid training
│   ├── ensemble.py                    # Deep Ensemble training
│   ├── trainer.py                     # Trainer implementation
│   ├── losses.py                      # Custom losses
│   ├── uncertainty.py                 # Uncertainty utilities
│   ├── cross_validation.py            # Cross-validation tools
│   └── README.md
│
├── evaluation/                        # Metrics, calibration, ablation
│   ├── __init__.py
│   ├── evaluate.py                    # Evaluation entry point
│   ├── calibration.py                 # Ensemble calibration
│   ├── metrics.py                     # Metric computation
│   ├── ablation.py                    # Ablation studies
│   └── README.md
│
├── scripts/                           # Utility and comparison scripts
│   ├── __init__.py
│   ├── compare.py
│   ├── compare_mr_mh.py
│   ├── diagnose.py
│   ├── model_summary.py
│   ├── regenerate_predictions.py
│   ├── spatial_eda_flask.py
│   ├── generate_data_eda.py
│   ├── revision_stat_tests.py
│   ├── revision_uncertainty_compare.py
│   ├── revision_hazard_utility.py
│   └── README.md
│
├── analysis/                          # EDA scripts and notebook
│   ├── __init__.py
│   ├── eda.py
│   ├── eda_vis.py
│   ├── exploratory.ipynb
│   └── README.md
│
├── data/                              # Data processing modules
├── models/                            # Model definitions
├── notebooks/                         # Additional notebooks/scripts
├── eda/                               # Generated EDA outputs
├── outputs/                           # Experiment outputs
├── figures/                           # Publication figures
├── visualization/                     # Visualization helpers
└── utils/                             # Utility functions
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration, optional)

### Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd earthquake_prediction
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Key dependencies:**
   - PyTorch >= 2.0
   - PyTorch Geometric (for graph neural networks)
   - NumPy, SciPy, Pandas
   - scikit-learn
   - Matplotlib, Seaborn
   - Jupyter (for notebooks)

## Dataset

### Amatrice Earthquake Catalog

**Source:** Italian Seismic Instrumental Network (RSN) / INGV  
**Period:** August 15, 2016 – August 15, 2017  
**Total Events:** 900,050  
**Magnitude Range:** M_w –0.59 to 6.08  
**Study Area (cropped):** 42.60°–42.95°N, 13.10°–13.35°E (~39×28 km)  
**Retained Events:** 457,874 (50.9%)  
**Graph Representation:** 133 active nodes, 1,290 edges  

**File:** `Amatrice_CAT5.v20210504.csv`

Columns: Event ID, timestamp, latitude, longitude, depth, magnitude (M_w), etc.

### Data Preprocessing

The preprocessing pipeline includes:

1. **Spatial Cropping:** Focus on the main aftershock zone
2. **Spatial Gridding:** 22×13 grid cells; active nodes = cells with ≥1,500 events
3. **Temporal Aggregation:** Six resolutions (1, 2, 4, 6, 12, 24 hours)
4. **Feature Engineering:** Event count, max/mean/std magnitude, mean depth, cumulative seismic energy
5. **Graph Construction:** Gaussian kernel weighting based on inter-node distance (connectivity radius = 15 km)
6. **Normalization:** Z-score normalization on all features

See `data/preprocessing.py` for implementation.

## Quick Start

### 1. Train ST-GAT (Default)

```bash
python main.py --mode train
```

### 2. Train ST-GAT (Multi-Resolution)

```bash
python -m training.train_multiresolution \
    --config config/base.py \
    --resolution 1,2,4,6,12,24 \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.001
```

### 3. Evaluate Model

```bash
python -m evaluation.evaluate \
    --checkpoint outputs/model_state.pth \
    --data Amatrice_CAT5.v20210504.csv \
    --output_dir outputs/
```

### 4. Deep Ensemble (Uncertainty Estimation)

```bash
python -m training.ensemble \
    --n_models 5 \
    --config config/base.py \
    --calibration_method post_hoc \
    --target_coverage 0.95
```

### 5. Compare Multi-Resolution vs. Multi-Horizon

```bash
python -m scripts.compare_mr_mh \
    --mr_checkpoint outputs/mr_model.pth \
    --mh_checkpoint outputs/mh_model.pth \
    --output_dir outputs/comparison_mr_mh/
```

### 6. Interactive Analysis (Jupyter)

```bash
jupyter notebook analysis/exploratory.ipynb
```

## Model Architecture

### ST-GAT: Spatio-Temporal Graph Attention Network

**Spatial Encoder (Multi-Head GAT):**
- Input: Node features at time `t` (N nodes × F features)
- Two GAT layers with 4 attention heads each
- Query-Key-Value (QKV) projections for adaptive spatial weighting
- Skip connections and layer normalization
- Output: Spatially-encoded representations (N nodes × d hidden)

**Temporal Decoder (LSTM):**
- Input: Spatial encoder outputs over T time steps
- Two-layer LSTM with temporal skip connections
- Maintains both short-term and long-term memory
- Output mapping via MLP to prediction targets
- Multi-step predictions per node

**Loss Function:**
```
L_total = λ₁ × L_WMSE + λ₂ × L_Focal

L_WMSE = Weighted MSE (higher weight for larger magnitudes)
L_Focal = Focal MSE (emphasizes hard examples)
```

**Uncertainty Estimation:**
- Deep Ensembles (M=5 models, independent weight initialization)
- Ensemble mean: μ = (1/M) Σ f_m(x)
- Epistemic uncertainty: σ² = (1/M) Σ (f_m(x) - μ)²
- Post-hoc calibration: CI_95% = μ ± 1.96 × γ × σ
  - Scaling factor γ determined on validation set for target 95% coverage

## Configuration

Edit `config/base.py` to customize:

```python
# Model parameters
model_config = {
    'n_nodes': 133,
    'n_features': 6,
    'hidden_dim': 64,
    'n_gat_heads': 4,
    'n_gat_layers': 2,
    'n_lstm_layers': 2,
    'dropout': 0.1,
}

# Training parameters
train_config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'early_stopping_patience': 7,
    'gradient_clip': 1.0,
}

# Data parameters
data_config = {
    'temporal_resolutions': [1, 2, 4, 6, 12, 24],  # hours
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
}
```

## Results

### Overall Performance (Amatrice Dataset)

| Metric | Value |
|--------|-------|
| Average R² | 0.659 |
| Average RMSE | 0.190 |
| Average MAE | 0.122 |
| F1-score (event detection) | 0.778 |
| Precision | 87.4% |
| Recall | 69.2% |

### Multi-Resolution vs. Multi-Horizon

| Horizon | MR R² | MH R² | Improvement |
|---------|-------|-------|------------|
| 4h | 0.685 | 0.620 | +10.5% |
| 8h | 0.648 | 0.340 | +90.6% |
| 12h | 0.612 | 0.085 | +620% |
| 24h | 0.571 | 0.031 | 1,742% |
| **Average** | **0.616** | **0.197** | **213%** |

**Paired significance:** p ≈ 0.032 (RMSE, MAE, R²)

### Uncertainty Calibration

| Resolution | Coverage | Sharpness | Calibration Factor |
|------------|----------|-----------|-------------------|
| 1h | 95.0% | 1.51 | 17.5× |
| 6h | 95.0% | 2.59 | 13.2× |
| 24h | 94.9% | 2.76 | 7.9× |

See `outputs/SUMMARY_REPORT.txt` for detailed results.

## Ablation Studies

Comprehensive ablation with 28 architectural variants:

| Configuration | RMSE | R² | F1 |
|--------------|------|-----|-----|
| ST-GAT (Dropout 0.1) | 0.204 | 0.682 | 0.778 |
| LSTM Only | 0.207 | 0.674 | 0.735 |
| GCN + LSTM | 0.206 | 0.675 | 0.732 |
| Single Head | 0.206 | 0.677 | 0.747 |
| No Skip Connection | 0.207 | 0.674 | 0.742 |

**Key Findings:**
- Temporal dynamics provide dominant signal (LSTM-only: R² = 0.674)
- Spatial attention contributes +0.008 R², +0.043 F1
- Skip connections essential for gradient flow
- Multi-head attention marginal gain over single-head

See `outputs/ablation_checkpoints/` for checkpoints.

## Limitations

1. **Single-Region Study:** Evaluated on Amatrice only. Generalization to other tectonic regimes untested.
2. **Class Imbalance:** Low recall for high-magnitude events (M_w ≥ 3) due to data scarcity.
3. **Calibration Dependence:** Deep Ensembles require 7.9–20× post-hoc scaling; raw ensemble diversity insufficient.
4. **Lead-Time Constraints:** F1 drops sharply beyond 2 hours; limited early-warning value.
5. **Offline Evaluation:** Not tested in real-time deployment scenarios.

## Usage Examples

### Example 1: Predict Magnitudes for New Data

```python
import torch
from models.stgat import ST_GAT
from data.dataset import AmatriceMicroseismicDataset

# Load model
model = ST_GAT(n_nodes=133, hidden_dim=64)
model.load_state_dict(torch.load('outputs/best_event_model.pth'))
model.eval()

# Prepare data
dataset = AmatriceMicroseismicDataset('Amatrice_CAT5.v20210504.csv')
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

# Generate predictions
with torch.no_grad():
    predictions = []
    for X, adj in loader:
        pred = model(X, adj)
        predictions.append(pred.cpu().numpy())
```

### Example 2: Visualize Attention Patterns

```python
from analysis.attention import visualize_attention_weights
import matplotlib.pyplot as plt

# Extract and visualize top-50 attention edges
top_edges = visualize_attention_weights(model, n_top=50)
plt.savefig('figures/attention_patterns.png', dpi=300, bbox_inches='tight')
```

### Example 3: Generate Confidence Intervals

```python
from evaluation.metrics import calibrate_ensemble_predictions

# Load ensemble models
ensemble_models = [load_model(f'models/e{i}.pth') for i in range(5)]

# Generate calibrated predictions
mean_pred, ci_lower, ci_upper = calibrate_ensemble_predictions(
    ensemble_models, X, target_coverage=0.95
)

# Evaluate coverage
coverage = evaluate_calibration(y_test, ci_lower, ci_upper)
print(f"Achieved coverage: {coverage:.1%}")
```

## Comparison with Baselines

### Statistical Baselines

- **Naive (last observation):** R² = –0.025
- **Moving Average:** R² = –0.085
- **ETAS (p=1.5):** R² = –0.035

### Deep Learning Baselines

- **LSTM-Only:** R² = 0.674 (–0.008 vs ST-GAT)
- **GCN + LSTM:** R² = 0.675 (–0.007 vs ST-GAT)
- **Temporal Fusion Transformer:** R² = 0.638 (–0.044 vs ST-GAT)

**ST-GAT advantages:**
- 44% RMSE reduction vs. Naive
- Consistent spatial attention gains
- Superior multi-resolution performance

## Citation

If you use this code or models, please cite:

```bibtex
@article{widyanto2026stgat,
  title={Spatio-Temporal Graph Attention Networks for Microseismic Magnitude Prediction with Calibrated Uncertainty},
  author={Widyanto, Yesaya Rudolf Susanto and Wibowo, Adi},
  journal={Artificial Intelligence in Geosciences},
  year={2026},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgments

- **Data:** Italian Seismic Instrumental Network (RSN), INGV
- **Funding:** Directorate for Research and Community Service (DPPM) 2025, Ministry of Research, Technology, and Higher Education (Kemristekdikti), Indonesia
- **Frameworks:** PyTorch, PyTorch Geometric

## Contact

**Corresponding Author:**  
Yesaya Rudolf Susanto Widyanto  
Department of Informatics, Diponegoro University  
Email: ysyrudolf@gmail.com

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution:** Reduce batch size in `config/base.py`:
```python
train_config['batch_size'] = 16  # or lower
```

### Issue: Poor model convergence

**Solution:** Check data preprocessing and try different learning rates:
```python
train_config['learning_rate'] = 0.0005  # decrease
```

### Issue: Predictions all near mean

**Solution:** Verify loss function weights (λ₁, λ₂) and data normalization.

## References

- Veličković, P., et al. (2018). "Graph Attention Networks." ICLR.
- Lakshminarayanan, B., et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS.
- Ogata, Y. (1988). "Statistical Models for Earthquake Occurrences and Residual Analysis for Point Processes." JASA.
- Stein, R. S. (1999). "The Role of Stress Transfer in Earthquake Occurrence." Nature.

---

**Last Updated:** April 2026  
**Maintainers:** Department of Informatics, Diponegoro University
