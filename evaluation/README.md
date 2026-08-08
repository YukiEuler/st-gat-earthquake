# Evaluation Module

Contains evaluation, metrics, calibration, and ablation study utilities.

## Key Components

### Metrics: `metrics.py`
**Class: `MetricsCalculator`**

Computes comprehensive evaluation metrics:
- Regression metrics: MAE, MSE, RMSE, R², Pearson
- Classification metrics: F1, Precision, Recall
- Earthquake-specific: Event detection accuracy

Usage:
```python
from evaluation.metrics import MetricsCalculator

calc = MetricsCalculator()
metrics = calc.calculate_all_metrics(y_true, y_pred, activity_mask=activity)
```

### Calibration: `calibration.py`
**Class: `EnsembleCalibration`**

Calibrates ensemble predictions for uncertainty quantification:
- Confidence calibration
- Coverage analysis
- Expected calibration error (ECE)

### Ablation Studies: `ablation.py`  
**Class: `AblationStudy`**

Systematic ablation studies to analyze model components:
- Feature importance analysis
- Architecture component contribution
- Layer-wise significance

### Baseline Evaluation: `evaluate.py`
**Main evaluation functions**

Entry point for comprehensive model evaluation:
- Raw-Mw test metrics and activity-head diagnostics
- Prediction visualization
- Error analysis

`evaluate.py` does not invent uncertainty for a deterministic checkpoint.
Prediction intervals must come from the separately trained and
validation-calibrated ensemble pipeline.

## Typical Workflow

```python
from evaluation import MetricsCalculator, AblationStudy
from evaluation.calibration import EnsembleCalibration

# Compute metrics
metrics = MetricsCalculator().calculate_all_metrics(y_true, y_pred)

# Calibrate ensemble
calibrator = EnsembleCalibration()
calibrated_preds = calibrator.calibrate(ensemble_preds, y_true)

# Run ablation
ablation = AblationStudy()
ablation.run_ablation(model, dataset)
```

## Output

Results are saved to `outputs/` including:
- `metrics.json` - Computed metrics
- `calibration_curves.png` - Calibration plots
- `ablation_results.csv` - Ablation study results
