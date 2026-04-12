# Scripts Module

Utility and analysis scripts for comparing models, diagnosing issues, and generating data.

## Analysis & Comparison Scripts

### Model Comparison: `compare.py`
**Compare predictions from different models**

Compare multiple trained models on test set:
```bash
python -m scripts.compare \
    --model1 outputs/stgat.pth \
    --model2 outputs/stgat_hybrid.pth \
    --output comparisons.csv
```

### Multi-Resolution vs Multi-Horizon: `compare_mr_mh.py`
**Compare multi-resolution vs multi-horizon architectures**

Benchmark different temporal/spatial configurations:
```bash
python -m scripts.compare_mr_mh
```

### Diagnostic Analysis: `diagnose.py`
**Diagnose model issues and analyze predictions**

Run comprehensive diagnostics:
- Check for NaN/Inf values
- Analyze prediction distributions  
- Identify outliers and anomalies
- Visualize error patterns

```bash
python -m scripts.diagnose --model outputs/best_model.pth
```

## Data & Visualization Scripts

### Model Summary: `model_summary.py`
**Print and analyze model architecture**

Get architecture statistics:
```bash
python -m scripts.model_summary --model stgat
```

### Regenerate Predictions: `regenerate_predictions.py`
**Regenerate predictions from trained models**

Re-run inference on test set:
```bash
python -m scripts.regenerate_predictions \
    --model outputs/trained_model.pth \
    --data data/test.csv
```

### Generate Data EDA: `generate_data_eda.py`
**Generate exploratory data analysis reports**

Create comprehensive EDA documentation

### Spatial EDA Flask: `spatial_eda_flask.py`
**Interactive spatial analysis visualization**

Launch web interface for spatial exploration:
```bash
python -m scripts.spatial_eda_flask
# Visit http://localhost:5000
```

## Statistical Tests

### Revision Statistical Tests: `revision_stat_tests.py`
**Statistical significance tests for model comparisons**

Compare models using:
- Paired t-tests
- Wilcoxon signed-rank tests
- Bootstrap confidence intervals

### Revision Uncertainty Compare: `revision_uncertainty_compare.py`
**Compare uncertainty quantification methods**

Analyze calibration across ensemble methods

### Revision Hazard Utility: `revision_hazard_utility.py`
**Earthquake hazard utility functions**

Convert predictions to hazard metrics

## Usage Examples

```python
# Compare models
from scripts.compare import compare_models
results = compare_models(model1, model2, test_data)

# Run diagnostics
from scripts.diagnose import run_diagnostics
diagnostics = run_diagnostics(predictions, ground_truth)

# Regenerate predictions
from scripts.regenerate_predictions import regenerate
preds = regenerate('outputs/model.pth', test_loader)
```
