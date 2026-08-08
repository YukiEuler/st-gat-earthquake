# ==============================================================================
# IO.PY - Input/Output Utilities
# ==============================================================================

import torch
import numpy as np
from scipy import sparse as sp
import json
from pathlib import Path
from datetime import datetime


def save_all_outputs(output_dir, model, config, metrics, adjacency, 
                     feature_stats, node_info=None, train_result=None,
                     target_stats=None):
    """Save all outputs to the output directory."""
    output_dir = Path(output_dir)
    
    # Create subdirectories
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics').mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    
    # 1. Save model
    model_path = output_dir / 'models' / 'stgat_best.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'feature_stats': feature_stats,
        'target_stats': target_stats,
        'timestamp': datetime.now().isoformat(),
    }, model_path)
    print(f" Model saved: {model_path}")
    
    # 2. Save adjacency matrix
    adj_path = output_dir / 'models' / 'adjacency_matrix.npz'
    sp.save_npz(adj_path, adjacency)
    print(f" Adjacency matrix saved: {adj_path}")
    
    # 3. Save node info
    if node_info is not None:
        node_path = output_dir / 'models' / 'node_info.npy'
        np.save(node_path, node_info, allow_pickle=True)
        print(f" Node info saved: {node_path}")
    
    # 4. Save metrics
    metrics_path = output_dir / 'metrics' / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
        }, f, indent=2, default=str)
    print(f" Metrics saved: {metrics_path}")
    
    # 5. Save training history
    if train_result:
        history_path = output_dir / 'metrics' / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': train_result.get('train_losses', []),
                'val_losses': train_result.get('val_losses', []),
                'best_loss': train_result.get('best_loss'),
                'epochs_trained': train_result.get('epochs_trained'),
            }, f, indent=2)
        print(f" Training history saved: {history_path}")
    
    print(f"\n All outputs saved to: {output_dir}")


def load_model(model_class, model_path, device='cuda'):
    """Load a saved model."""
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('config', {})
    
    # Reconstruct model
    model = model_class(
        num_nodes=config.get('num_nodes', 100),
        in_features=config.get('n_input_features', 4),
        hidden_dim=config.get('hidden_dim', 32),
        out_features=config.get('n_target_features', 4),
        horizon=config.get('horizon', 24),
        num_gat_layers=config.get('num_gat_layers', 2),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.2),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint


def generate_summary_report(output_dir, config, metrics, train_result=None):
    """Generate a text summary report."""
    output_dir = Path(output_dir)
    report_path = output_dir / 'SUMMARY_REPORT.txt'
    
    lines = [
        "=" * 70,
        " ST-GAT EARTHQUAKE PREDICTION - SUMMARY REPORT",
        "=" * 70,
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n" + "-" * 70,
        "CONFIGURATION",
        "-" * 70,
    ]
    
    for key, value in config.items():
        lines.append(f"  {key}: {value}")
    
    lines.extend([
        "\n" + "-" * 70,
        "RESULTS",
        "-" * 70,
    ])
    
    if train_result:
        lines.append(f"  Epochs Trained: {train_result.get('epochs_trained', 'N/A')}")
        lines.append(f"  Best Validation Loss: {train_result.get('best_loss', 'N/A'):.6f}")
    
    if 'overall' in metrics:
        lines.append("\n  Overall Metrics:")
        for metric, value in metrics['overall'].items():
            lines.append(f"    {metric}: {value:.6f}")
    
    if 'uncertainty' in metrics:
        lines.append("\n  Uncertainty Metrics:")
        for metric, value in metrics['uncertainty'].items():
            if isinstance(value, (int, float, np.number)):
                lines.append(f"    {metric}: {value:.6f}")
            else:
                lines.append(f"    {metric}: {value}")

    if 'activity_detection' in metrics:
        lines.append("\n  Activity Detection:")
        for metric in [
            'prevalence', 'mean_probability', 'brier_score', 'roc_auc',
            'pr_auc', 'precision', 'recall', 'f1_score', 'specificity'
        ]:
            lines.append(f"    {metric}: {metrics['activity_detection'][metric]:.6f}")

    if 'conditional_magnitude_active' in metrics:
        lines.append("\n  Conditional Magnitude (active bins only):")
        for metric, value in metrics['conditional_magnitude_active'].items():
            lines.append(f"    {metric}: {value:.6f}")

    if 'conditional_magnitude_active_diagnostics' in metrics:
        diagnostics = metrics['conditional_magnitude_active_diagnostics']
        lines.append("\n  Conditional Magnitude Dispersion:")
        for metric in [
            'target_std', 'prediction_std',
            'prediction_to_target_std_ratio',
        ]:
            lines.append(f"    {metric}: {diagnostics[metric]:.6f}")

    if 'forecast_views' in metrics:
        lines.append("\n  Explicit Forecast Views:")
        for name, view in metrics['forecast_views'].items():
            regression = view['regression']
            threshold = view.get('activity_threshold')
            suffix = f", threshold={threshold:.6f}" if threshold is not None else ''
            lines.append(f"    {name}{suffix}:")
            lines.append(
                f"      MSE={regression['MSE']:.6f}, "
                f"MAE={regression['MAE']:.6f}, R2={regression['R2']:.6f}"
            )

    if 'forecast_comparison' in metrics:
        lines.append("\n  Forecast/Baseline Comparison:")
        for name, result in metrics['forecast_comparison'].items():
            regression = result['regression']
            skill = result['primary_model_skill_vs_forecast']
            lines.append(
                f"    {name}: MSE={regression['MSE']:.6f}, "
                f"MAE={regression['MAE']:.6f}, R2={regression['R2']:.6f}, "
                f"primary_model_skill={skill:.6f}"
            )

    if 'diagnostics' in metrics:
        lines.append("\n  Forecast Diagnostics:")
        for metric, value in metrics['diagnostics'].items():
            lines.append(f"    {metric}: {value}")
    
    lines.extend([
        "\n" + "-" * 70,
        "OUTPUT FILES",
        "-" * 70,
        f"   {output_dir}/",
        "      models/",
        "         stgat_best.pth",
        "         adjacency_matrix.npz",
        "      metrics/",
        "         evaluation_metrics.json",
        "         forecast_comparison.csv",
        "      figures/",
        "          *.png",
        "\n" + "=" * 70,
    ])
    
    report = "\n".join(lines)
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\n Report saved: {report_path}")
    
    return report
