# Analysis Module

Exploratory data analysis (EDA) and interactive visualization tools.

## EDA Scripts

### Main EDA: `eda.py`
**Comprehensive exploratory data analysis**

Generates analysis reports:
- Dataset statistics and distributions
- Temporal autocorrelation analysis
- Spatial distribution of seismic events
- Feature correlations and relationships

Usage:
```bash
python -m analysis.eda
```

Output: `eda/` directory with text reports and figures

### EDA Visualizations: `eda_vis.py`
**Visualization functions for EDA**

Reusable plotting functions:
- Distribution plots
- Correlation matrices
- Spatial heatmaps
- Time series visualizations

```python
from analysis.eda_vis import plot_distributions, plot_spatial_map

plot_distributions(data, features=['count', 'max_mw'])
plot_spatial_map(data, title="Earthquake Density")
```

### Interactive Exploration: `exploratory.ipynb`
**Jupyter notebook for interactive data exploration**

Run in Jupyter:
```bash
jupyter notebook analysis/exploratory.ipynb
```

Features:
- Interactive plots with Plotly
- Custom data filtering
- Feature engineering examples
- Statistical hypothesis testing

## Analysis Outputs

Generated in `eda/` directory:

### Reports (Text)
- `data_summary.txt` - Dataset statistics
- `time_bin_summary.txt` - Temporal binning analysis
- `grid_size_summary.txt` - Spatial grid analysis  
- `graph_degree_summary.txt` - Network graph statistics
- `crop_justification.txt` - Spatial crop analysis

### Visualizations (Figures)
- Distribution histograms
- Correlation heatmaps
- Spatial density maps
- Temporal patterns
- Feature engineering results

### Tables
- Summary statistics CSV
- Feature correlations
- Binning analysis tables

## Example Workflow

```python
# Load and analyze data
from analysis.eda import run_eda
summary = run_eda(data_path='../data/Amatrice_CAT5.v20210504.csv')

# Create visualizations
from analysis.eda_vis import plot_all
plot_all(data, output_dir='../eda/figures')

# Statistical analysis
import pandas as pd
summary_stats = pd.read_csv('../eda/tables/summary_statistics.csv')
```

## Data Requirements

Analysis assumes Amatrice seismic catalog with columns:
- Latitude/Longitude coordinates
- Magnitude (various measures: Local, Moment)
- Depth
- Timestamp
- Uncertainty measures

## Integration

Results from EDA guide configuration:
- Optimal grid size → `CONFIG['grid_size']`
- Temporal bin width → `CONFIG['time_bin']`
- Spatial bounds → `CONFIG['lat_min']`, etc.
- Active node threshold → `CONFIG['min_events_per_node']`
