# ==============================================================================
# REGENERATE_PREDICTIONS.PY - Generate predictions from saved model
# ==============================================================================
"""
Regenerate predictions CSV and HTML viewer from saved hybrid model.
No training needed - just loads the model and generates outputs.

Usage:
    python regenerate_predictions.py
"""

import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, DEVICE
from data.preprocessing import DataPreprocessor
from data.adjacency import AdjacencyBuilder
from data.dataset import HybridSeismicDataset
from torch.utils.data import DataLoader

from models import STGATHybrid

# Import the prediction functions from main_hybrid
from main_hybrid import generate_predictions_csv, generate_html_viewer

HYBRID_CONFIG = {
    **CONFIG,
    'n_classes': 4,
}


def main():
    print("\n" + "=" * 70)
    print(" Regenerating Predictions from Saved Model")
    print("=" * 70)
    
    output_dir = Path(HYBRID_CONFIG['output_dir'])
    model_path = output_dir / 'hybrid_model.pt'
    
    if not model_path.exists():
        print(f" ERROR: Model not found at {model_path}")
        print(" Please run main_hybrid.py first to train the model.")
        return
    
    print(f"   Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    saved_config = checkpoint.get('config', HYBRID_CONFIG)
    
    # Load data
    print("   Loading data...")
    preprocessor = DataPreprocessor(saved_config)
    data = preprocessor.process(saved_config['filename'])
    
    adj_builder = AdjacencyBuilder(saved_config)
    adj_scipy = adj_builder.build_distance_weighted_adj(
        data['num_nodes'], 
        data['node_info'], 
        data['grid_params'],
        use_distance_weighting=True
    )
    adj_sparse = adj_builder.scipy_to_torch_sparse(adj_scipy, device=DEVICE)
    
    # Create test dataset
    test_dataset = HybridSeismicDataset(
        data['test_data'],
        window_size=saved_config['window_size'],
        horizon=saved_config['horizon'],
        magnitude_idx=1,
        regression_indices=[0, 2, 3],
        feature_stats=data['feature_stats']
    )
    test_loader = DataLoader(test_dataset, batch_size=saved_config['batch_size'], shuffle=False)
    
    # Create model and load weights
    print("   Creating model...")
    model = STGATHybrid(
        num_nodes=data['num_nodes'],
        in_features=4,
        hidden_dim=saved_config['hidden_dim'],
        n_regression_features=3,
        n_classes=saved_config.get('n_classes', 4),
        horizon=saved_config['horizon'],
        num_gat_layers=saved_config['num_gat_layers'],
        num_heads=saved_config['num_heads'],
        dropout=saved_config['dropout'],
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   Model loaded successfully!")
    
    # Generate predictions
    print("\n   Generating predictions (with denormalization)...")
    predictions_df = generate_predictions_csv(
        model, test_loader, adj_sparse, DEVICE,
        data['feature_stats'], saved_config.get('n_classes', 4)
    )
    
    csv_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(csv_path, index=False)
    print(f"   Predictions saved to {csv_path}")
    
    # Generate HTML viewer
    html_path = output_dir / 'predictions_viewer.html'
    generate_html_viewer(predictions_df, html_path, saved_config.get('n_classes', 4))
    print(f"   HTML viewer saved to {html_path}")
    
    print("\n" + "=" * 70)
    print(" Done! Open predictions_viewer.html in your browser.")
    print("=" * 70)


if __name__ == '__main__':
    main()
