# ==============================================================================
# MAIN.PY - Entry Point for Earthquake Prediction
# ==============================================================================
"""
ST-GAT Earthquake Prediction Pipeline

Usage:
    python main.py --mode train          # Train model
    python main.py --mode eval           # Evaluate saved model
    python main.py --mode ablation       # Run ablation study
    python main.py --mode ensemble       # Train Deep Ensemble
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, DEVICE, print_config

# Set random seeds for full reproducibility
def set_seed(seed=42):
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # For CUDA >= 10.2, also set this for full determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # For PyTorch >= 1.8, enable deterministic algorithms globally
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations don't have deterministic implementations
            pass


def main(args):
    set_seed(CONFIG['seed'])
    
    print("\n" + "=" * 70)
    print(" ST-GAT EARTHQUAKE PREDICTION")
    print("=" * 70)
    print(f"   Device: {DEVICE}")
    print(f"   Mode: {args.mode}")
    print("=" * 70)
    
    if args.print_config:
        print_config()
    
    # ====================
    # DATA PREPARATION
    # ====================
    print("\n Loading and preprocessing data...")
    
    from data.preprocessing import DataPreprocessor
    from data.adjacency import AdjacencyBuilder
    from data.dataset import SeismicDataset, create_dataloaders
    from torch.utils.data import DataLoader
    
    preprocessor = DataPreprocessor(CONFIG)
    data = preprocessor.process(CONFIG['filename'])
    
    # Build adjacency matrix
    adj_builder = AdjacencyBuilder(CONFIG)
    adj_scipy = adj_builder.build_distance_weighted_adj(
        data['num_nodes'], 
        data['node_info'], 
        data['grid_params'],
        use_distance_weighting=True
    )
    adj_sparse = adj_builder.scipy_to_torch_sparse(adj_scipy, device=DEVICE)

    from utils.manifest import write_run_manifest
    output_dir = Path(CONFIG['output_dir'])
    write_run_manifest(
        output_dir, CONFIG, data=data, data_path=CONFIG['filename'],
        adjacency=adj_scipy, stage='preprocessing_complete'
    )
    
    # Get target feature indices
    all_features = CONFIG['features']
    target_features = CONFIG.get('target_features', all_features)
    target_indices = [all_features.index(f) for f in target_features if f in all_features]
    # The auxiliary activity channel belongs to the canonical primary model.
    # Legacy ablation/ensemble modes still compare one-output regressors and
    # therefore keep their original one-channel target tensors.
    use_hurdle = CONFIG.get('loss_type') == 'hurdle' and args.mode == 'train'
    
    # Create dataloaders
    train_dataset = SeismicDataset(
        data['train_data'],
        target_data=data['train_target_data'],
        activity_data=data['train_target_activity'] if use_hurdle else None,
        window_size=CONFIG['window_size'],
        horizon=CONFIG['horizon'],
    )
    val_dataset = SeismicDataset(
        data['val_data'],
        target_data=data['val_target_data'],
        activity_data=data['val_target_activity'] if use_hurdle else None,
        window_size=CONFIG['window_size'],
        horizon=CONFIG['horizon'],
    )
    test_dataset = SeismicDataset(
        data['test_data'],
        target_data=data['test_target_data'],
        activity_data=data['test_target_activity'] if use_hurdle else None,
        window_size=CONFIG['window_size'],
        horizon=CONFIG['horizon'],
    )
    
    # Create DataLoaders with reproducible shuffling
    g = torch.Generator()
    g.manual_seed(CONFIG['seed'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    data_info = {
        'num_nodes': data['num_nodes'],
        'n_input_features': data['train_data'].shape[-1], 
        'n_target_features': len(target_features),
        'n_model_outputs': (
            int(CONFIG.get('model_output_features', 2))
            if use_hurdle else len(target_features)
        ),
        'target_features': target_features,
        'feature_stats': data['feature_stats'],
        'target_stats': data['target_stats'],
        'target_indices': target_indices,         # Input-tensor indices for baselines
        'split_timestamps': data['split_timestamps'],
    }
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    # ====================
    # MODE: TRAIN
    # ====================
    if args.mode == 'train':
        from models import STGAT, STTFT, STGATWarmStart, STGATMultiScale
        from training import WeightedMSELoss, AsymmetricMSELoss, QuantileLoss, Trainer
        from evaluation import MetricsCalculator
        from visualization import PredictionVisualizer, AttentionVisualizer, SpatialVisualizer
        from utils import save_all_outputs
        from utils.io import generate_summary_report
        
        # Create model based on temporal_model config
        temporal_model = CONFIG.get('temporal_model', 'lstm')
        
        if temporal_model == 'tft':
            print(f"\n Using ST-TFT (GAT + Temporal Fusion Transformer)")
            model = STTFT(
                num_nodes=data['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=CONFIG['hidden_dim'],
                out_features=data_info['n_model_outputs'],
                horizon=CONFIG['horizon'],
                num_gat_layers=CONFIG['num_gat_layers'],
                num_heads=CONFIG['num_heads'],
                tft_layers=CONFIG.get('tft_layers', 2),
                dropout=CONFIG['dropout'],
            ).to(DEVICE)
        elif temporal_model == 'multiscale':
            print(f"\n Using ST-GAT Multi-Scale (GAT + Multi-Resolution LSTM)")
            model = STGATMultiScale(
                num_nodes=data['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=CONFIG['hidden_dim'],
                out_features=data_info['n_model_outputs'],
                horizon=CONFIG['horizon'],
                num_gat_layers=CONFIG['num_gat_layers'],
                num_heads=CONFIG['num_heads'],
                dropout=CONFIG['dropout'],
                scales=CONFIG.get('multiscale_scales', [1, 2, 4]),
                fusion_type=CONFIG.get('multiscale_fusion', 'concat'),
                node_embed_dim=16,
                pool_type='avg',
            ).to(DEVICE)
        else:
            print(f"\n Using ST-GAT (GAT + LSTM)")
            model = STGAT(
                        num_nodes=data['num_nodes'],
                        in_features=data_info['n_input_features'],
                        hidden_dim=CONFIG['hidden_dim'],
                        out_features=data_info['n_model_outputs'],
                        horizon=CONFIG['horizon'],
                        num_gat_layers=CONFIG['num_gat_layers'],
                        num_heads=CONFIG['num_heads'],
                        dropout=CONFIG['dropout'],
                    ).to(DEVICE)
                            
        print(f"\n Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Select loss function based on config (NEW)
        loss_type = CONFIG.get('loss_type', 'weighted_mse')
        print(f"\n Using Loss Function: {loss_type}")
        
        if loss_type == 'hurdle':
            from training.losses import HurdleMagnitudeLoss
            magnitude_idx = CONFIG.get('magnitude_idx', 0)
            target_scale = data['target_stats']['std'][magnitude_idx]
            target_offset = data['target_stats'].get(
                'offset', data['target_stats']['mean']
            )[magnitude_idx]
            criterion = HurdleMagnitudeLoss(
                activity_weight=CONFIG.get('hurdle_activity_loss_weight', 1.0),
                magnitude_weight=CONFIG.get('hurdle_magnitude_loss_weight', 1.0),
                expected_value_weight=CONFIG.get(
                    'hurdle_expected_value_loss_weight', 1.0
                ),
                smooth_l1_beta=CONFIG.get('hurdle_smooth_l1_beta', 0.5),
                activity_pos_weight=CONFIG.get('hurdle_activity_pos_weight'),
                magnitude_thresholds=CONFIG.get(
                    'hurdle_magnitude_event_thresholds', []
                ),
                magnitude_weights=CONFIG.get(
                    'hurdle_magnitude_event_weights', []
                ),
                target_scale=target_scale,
                target_offset=target_offset,
                normalize_magnitude_weights=CONFIG.get(
                    'hurdle_normalize_magnitude_weights', True
                ),
                max_magnitude_weight=CONFIG.get(
                    'hurdle_max_magnitude_weight'
                ),
                tail_underprediction_multiplier=CONFIG.get(
                    'hurdle_tail_underprediction_multiplier', 1.0
                ),
            )
        elif loss_type == 'asymmetric':
            criterion = AsymmetricMSELoss(
                alpha=CONFIG.get('asymmetric_alpha', 0.8),
                magnitude_idx=CONFIG.get('magnitude_idx', 1),
                active_weight=CONFIG['active_weight'],
                feature_weights=CONFIG.get('feature_weights')
            )
        elif loss_type == 'quantile':
            criterion = QuantileLoss(quantile=0.9)
        elif loss_type == 'active_only':
            from training.losses import ActiveOnlyMSELoss
            criterion = ActiveOnlyMSELoss(
                feature_weights=CONFIG.get('feature_weights')
            )
        elif loss_type == 'sparse_aware':
            from training.losses import SparseAwareLoss
            magnitude_idx = CONFIG.get('magnitude_idx', 0)
            target_scale = data['target_stats']['std'][magnitude_idx]
            target_offset = data['target_stats'].get(
                'offset', data['target_stats']['mean']
            )[magnitude_idx]
            criterion = SparseAwareLoss(
                active_loss_weight=CONFIG['active_weight'],
                underpredict_penalty=CONFIG.get('underpredict_penalty', 2.0),
                feature_weights=CONFIG.get('feature_weights'),
                max_active_weight=CONFIG.get('max_dynamic_active_weight', 60.0),
                feature_names=data_info['target_features'],
                magnitude_idx=magnitude_idx,
                magnitude_thresholds=CONFIG.get('magnitude_event_thresholds', []),
                magnitude_weights=CONFIG.get('magnitude_event_weights', []),
                target_scale=target_scale,
                target_offset=target_offset
            )
        elif loss_type == 'focal':
            from training.losses import FocalMSELoss
            criterion = FocalMSELoss(
                gamma=CONFIG.get('focal_gamma', 2.0),
                active_weight=CONFIG['active_weight'],
                feature_weights=CONFIG.get('feature_weights')
            )
        elif loss_type == 'multiscale':
            from training.losses import MultiScaleLoss
            criterion = MultiScaleLoss(
                scales=CONFIG.get('multiscale_horizons', [1, 2, 4]),  # 6h, 12h, 24h
                scale_weights=CONFIG.get('multiscale_weights', [1.0, 1.0, 1.0]),
                magnitude_idx=CONFIG.get('magnitude_idx', 1),
                active_weight=CONFIG['active_weight'],
                feature_weights=CONFIG.get('feature_weights')
            )
        else:  # default: weighted_mse
            criterion = WeightedMSELoss(
                active_weight=CONFIG['active_weight'],
                feature_weights=CONFIG.get('feature_weights')
            )
        
        trainer = Trainer(model, criterion, CONFIG, DEVICE)
        train_result = trainer.fit(train_loader, val_loader, adj_sparse)

        # Calibrate the activity head and select its decision threshold using
        # validation predictions only. Test labels are never consulted.
        activity_logit_bias = 0.0
        activity_threshold = float(CONFIG.get('hurdle_activity_threshold', 0.5))
        threshold_details = {
            'source': 'fixed_config',
            'threshold': activity_threshold,
        }
        threshold_mode = CONFIG.get(
            'hurdle_activity_threshold_mode', 'fixed'
        )
        needs_validation_outputs = use_hurdle and (
            CONFIG.get('hurdle_calibrate_activity_bias_on_validation', True)
            or threshold_mode != 'fixed'
        )
        validation_outputs = None
        validation_targets = None
        if needs_validation_outputs:
            validation_outputs = []
            validation_targets = []
            model.eval()
            with torch.no_grad():
                for batch_data, batch_target in val_loader:
                    batch_data = batch_data.to(DEVICE)
                    validation_outputs.append(
                        model(batch_data, adj_sparse).cpu().numpy()
                    )
                    validation_targets.append(batch_target.numpy())
            validation_outputs = np.concatenate(validation_outputs, axis=0)
            validation_targets = np.concatenate(validation_targets, axis=0)

        if use_hurdle and CONFIG.get(
            'hurdle_calibrate_activity_bias_on_validation', True
        ):
            from training.hurdle import fit_activity_logit_bias
            activity_logit_bias = fit_activity_logit_bias(
                validation_outputs[..., 1:2],
                validation_targets[..., 1:2],
            )
            print(
                "\n Validation-only activity calibration: "
                f"logit bias={activity_logit_bias:.6f}"
            )

        if use_hurdle and threshold_mode != 'fixed':
            if threshold_mode not in {'validation_f1', 'validation_balanced_accuracy'}:
                raise ValueError(
                    "hurdle_activity_threshold_mode must be 'fixed', "
                    "'validation_f1', or 'validation_balanced_accuracy'."
                )
            from training.hurdle import fit_activity_threshold
            validation_logit = (
                validation_outputs[..., 1:2] + activity_logit_bias
            )
            validation_probability = 1.0 / (
                1.0 + np.exp(-np.clip(validation_logit, -60.0, 60.0))
            )
            objective = CONFIG.get(
                'hurdle_activity_threshold_objective',
                'f1' if threshold_mode == 'validation_f1'
                else 'balanced_accuracy',
            )
            activity_threshold, threshold_details = fit_activity_threshold(
                validation_probability,
                validation_targets[..., 1:2],
                objective=objective,
                minimum=CONFIG.get('hurdle_activity_threshold_min', 0.05),
                maximum=CONFIG.get('hurdle_activity_threshold_max', 0.95),
            )
            print(
                " Validation-only activity threshold: "
                f"{activity_threshold:.6f} "
                f"({objective}={threshold_details['objective_score']:.4f}, "
                f"precision={threshold_details['precision']:.4f}, "
                f"recall={threshold_details['recall']:.4f})"
            )

        effective_config = dict(CONFIG)
        effective_config['fitted_activity_logit_bias'] = float(activity_logit_bias)
        effective_config['fitted_activity_threshold'] = float(activity_threshold)
        effective_config['fitted_activity_threshold_details'] = threshold_details
        
        # Generate predictions
        print("\n Generating predictions...")
        model.eval()
        raw_predictions = []
        packed_targets = []
        
        with torch.no_grad():
            for batch_data, batch_target in test_loader:
                batch_data = batch_data.to(DEVICE)
                output = model(batch_data, adj_sparse)
                raw_predictions.append(output.cpu().numpy())
                packed_targets.append(batch_target.numpy())
        
        raw_predictions = np.concatenate(raw_predictions, axis=0)
        packed_targets = np.concatenate(packed_targets, axis=0)

        hurdle_outputs = None
        activity_targets = None
        if use_hurdle:
            from training.hurdle import decode_hurdle_numpy, split_hurdle_targets
            targets, activity_targets = split_hurdle_targets(packed_targets)
            hurdle_outputs = decode_hurdle_numpy(
                raw_predictions,
                activity_threshold=activity_threshold,
                activity_logit_bias=activity_logit_bias,
            )
            primary_prediction = CONFIG.get(
                'hurdle_primary_prediction', 'expected'
            )
            if primary_prediction not in {'expected', 'thresholded'}:
                raise ValueError(
                    "hurdle_primary_prediction must be 'expected' or 'thresholded'."
                )
            predictions = hurdle_outputs[primary_prediction]
            print(
                "   Hurdle decoding: "
                f"primary={primary_prediction}, "
                f"activity threshold={activity_threshold:.4f}"
            )
        else:
            predictions = raw_predictions
            targets = packed_targets
        
        # Denormalize predictions and targets for proper metrics calculation
        print(" Denormalizing for metrics calculation...")
        target_stats = data['target_stats']
        target_mean = target_stats.get('offset', target_stats['mean'])
        target_std = target_stats['std']
        
        print(f"   Target features: {data_info['target_features']}")
        print(f"   Raw train mean: {target_stats['mean']}")
        print(f"   Denormalization offset: {target_mean}")
        print(f"   Std: {target_std}")
        print(f"   Raw model output shape: {raw_predictions.shape}")
        print(f"   Primary predictions shape: {predictions.shape}")
        print(f"   Targets shape: {targets.shape}")
        
        # Denormalize: x_denorm = x_norm * std + mean
        # predictions shape: (B, H, N, F) - need to broadcast correctly
        # stats shape: (F,) -> need to reshape for broadcasting
        n_dim = predictions.ndim
        if n_dim == 4:  # (B, H, N, F)
            target_mean = target_mean.reshape(1, 1, 1, -1)
            target_std = target_std.reshape(1, 1, 1, -1)
        elif n_dim == 3:  # (B, N, F)
            target_mean = target_mean.reshape(1, 1, -1)
            target_std = target_std.reshape(1, 1, -1)
        
        predictions_denorm = predictions * target_std + target_mean
        targets_denorm = targets * target_std + target_mean

        hurdle_denorm = None
        if hurdle_outputs is not None:
            hurdle_denorm = {
                'conditional': hurdle_outputs['conditional'] * target_std + target_mean,
                'expected': hurdle_outputs['expected'] * target_std + target_mean,
                'thresholded': hurdle_outputs['thresholded'] * target_std + target_mean,
                'activity_probability': hurdle_outputs['activity_probability'],
            }
        
        # Calculate metrics on DENORMALIZED data
        metrics_calc = MetricsCalculator(feature_names=data_info['target_features'])
        metrics = metrics_calc.calculate_all_metrics(
            targets_denorm,
            predictions_denorm,
            magnitude_idx=CONFIG.get('magnitude_idx', 0),
            activity_mask=activity_targets,
        )
        metrics['diagnostics'] = metrics_calc.calculate_forecast_diagnostics(
            targets_denorm, predictions_denorm
        )
        if hurdle_denorm is not None:
            metrics['activity_detection'] = metrics_calc.calculate_activity_metrics(
                activity_targets,
                hurdle_denorm['activity_probability'],
                threshold=activity_threshold,
            )
            active_entries = np.broadcast_to(
                activity_targets >= 0.5, targets_denorm.shape
            )
            metrics['conditional_magnitude_active'] = (
                metrics_calc.calculate_regression_metrics(
                    targets_denorm[active_entries],
                    hurdle_denorm['conditional'][active_entries],
                )
            )
            conditional_diagnostics = metrics_calc.calculate_forecast_diagnostics(
                targets_denorm[active_entries],
                hurdle_denorm['conditional'][active_entries],
            )
            metrics['conditional_magnitude_active_diagnostics'] = (
                conditional_diagnostics
            )
            expected_regression = metrics_calc.calculate_regression_metrics(
                targets_denorm,
                hurdle_denorm['expected'],
                activity_mask=activity_targets,
            )
            event_regression = metrics_calc.calculate_regression_metrics(
                targets_denorm,
                hurdle_denorm['thresholded'],
                activity_mask=activity_targets,
            )
            metrics['forecast_views'] = {
                'expected_magnitude': {
                    'interpretation': (
                        'P(activity) times conditional Mw; primary squared-error forecast.'
                    ),
                    'regression': expected_regression,
                    'diagnostics': metrics_calc.calculate_forecast_diagnostics(
                        targets_denorm, hurdle_denorm['expected']
                    ),
                },
                'event_magnitude': {
                    'interpretation': (
                        'Conditional Mw when validation-fitted activity decision is positive; '
                        'zero otherwise.'
                    ),
                    'activity_threshold': float(activity_threshold),
                    'regression': event_regression,
                    'diagnostics': metrics_calc.calculate_forecast_diagnostics(
                        targets_denorm, hurdle_denorm['thresholded']
                    ),
                },
            }
            # Backward-compatible alias used by older report scripts.
            metrics['thresholded_point_forecast'] = event_regression

            from evaluation.baselines import (
                evaluate_forecast_comparison,
                generate_reference_forecasts,
            )
            reference_forecasts = generate_reference_forecasts(
                data['test_target_data'],
                data['train_target_data'],
                data['target_stats'],
                window_size=CONFIG['window_size'],
                horizon=CONFIG['horizon'],
                recent_window=CONFIG.get('baseline_recent_window', 6),
                test_activity_series=data['test_target_activity'],
                train_activity_series=data['train_target_activity'],
            )
            forecast_set = {
                'stgat_expected': {
                    'kind': 'model_primary',
                    'description': 'Canonical ST-GAT expected-magnitude forecast.',
                    'prediction': hurdle_denorm['expected'],
                    'activity_probability': hurdle_denorm['activity_probability'],
                    'activity_threshold': float(activity_threshold),
                },
                'stgat_event_thresholded': {
                    'kind': 'model_event_view',
                    'description': (
                        'Canonical ST-GAT hard event forecast using validation threshold.'
                    ),
                    'prediction': hurdle_denorm['thresholded'],
                    'activity_probability': hurdle_denorm['activity_probability'],
                    'activity_threshold': float(activity_threshold),
                },
                **reference_forecasts,
            }
            metrics['forecast_comparison'] = evaluate_forecast_comparison(
                targets_denorm,
                forecast_set,
                activity_target=activity_targets,
                feature_names=data_info['target_features'],
                magnitude_idx=CONFIG.get('magnitude_idx', 0),
                primary_name='stgat_expected',
            )
        metrics_calc.print_metrics(metrics)
        
        # Visualizations
        output_dir = Path(CONFIG['output_dir'])
        
        pred_viz = PredictionVisualizer(
            feature_names=data_info['target_features'],
            save_dir=output_dir / 'figures'
        )
        pred_viz.plot_training_history(train_result['train_losses'], 
                                       train_result['val_losses'],
                                       train_result['best_loss'])
        pred_viz.plot_scatter_pred_vs_actual(targets_denorm, predictions_denorm)
        pred_viz.plot_timeseries_prediction(targets_denorm, predictions_denorm, 
                                            node_idx=0, horizon_idx=0)
        pred_viz.plot_per_horizon_metrics(metrics['per_horizon'])
        if hurdle_denorm is not None:
            pred_viz.plot_hurdle_activity(
                activity_targets,
                hurdle_denorm['activity_probability'],
                targets_denorm,
                hurdle_denorm['conditional'],
                threshold=activity_threshold,
            )
            pred_viz.plot_scatter_pred_vs_actual(
                targets_denorm,
                hurdle_denorm['thresholded'],
                save_name='scatter_event_pred_vs_actual',
            )
        
        # Plot predictions at most active node
        print("\n Visualizing most active node...")
        pred_viz.plot_most_active_node(targets_denorm, predictions_denorm,
                                       activity_mask=activity_targets,
                                       horizon_idx=0,
                                       save_name='most_active_node_prediction')
        if hurdle_denorm is not None:
            pred_viz.plot_most_active_node(
                targets_denorm,
                hurdle_denorm['thresholded'],
                activity_mask=activity_targets,
                horizon_idx=0,
                save_name='most_active_node_event_prediction',
            )
        
        # Attention visualization
        if CONFIG.get('save_attention', True):
            att_viz = AttentionVisualizer(
                node_coords=adj_builder.coords,
                save_dir=output_dir / 'figures'
            )
            sample_data = data['test_data'][:CONFIG['window_size']]
            attention_weights = att_viz.extract_attention_weights(
                model, sample_data, adj_sparse, DEVICE
            )
            att_viz.plot_attention_distribution(attention_weights, 
                                                save_name='attention_distribution')
            att_viz.plot_top_attention_edges(attention_weights, adj_sparse,
                                            save_name='top_attention_edges')
        
        # Spatial visualization
        spatial_viz = SpatialVisualizer(
            node_coords=adj_builder.coords,
            save_dir=output_dir / 'figures'
        )
        spatial_viz.plot_spatial_heatmap(targets, predictions)
        spatial_viz.plot_node_activity(targets)

        if metrics.get('forecast_comparison'):
            from evaluation.baselines import save_forecast_comparison_csv
            comparison_path = save_forecast_comparison_csv(
                metrics['forecast_comparison'],
                output_dir / 'metrics' / 'forecast_comparison.csv',
            )
            print(f" Forecast comparison saved: {comparison_path}")
        
        # Save outputs
        save_all_outputs(
            output_dir, model, effective_config, metrics, adj_scipy,
            data['feature_stats'], data['node_info'], train_result,
            target_stats=data['target_stats']
        )
        write_run_manifest(
            output_dir, effective_config, data=data, data_path=CONFIG['filename'],
            adjacency=adj_scipy,
            checkpoint_paths=[output_dir / 'models' / 'stgat_best.pth'],
            stage='training_complete'
        )
        
        # ====================
        # SAVE PREDICTIONS TO CSV & HTML VIEWER
        # ====================
        print("\n Saving predictions to CSV and generating HTML viewer...")
        
        predictions_df = generate_regression_predictions_csv(
            predictions_denorm, targets_denorm,
            data_info['target_features'], output_dir,
            activity_targets=activity_targets,
            hurdle_outputs=hurdle_denorm,
            activity_threshold=activity_threshold,
        )
        
        html_path = output_dir / 'predictions_viewer.html'
        generate_regression_html_viewer(predictions_df, html_path, data_info['target_features'])
        print(f"   HTML viewer saved to {html_path}")
        generate_summary_report(
            output_dir, effective_config, metrics, train_result
        )
        
    # ====================
    # MODE: ENSEMBLE
    # ====================
    elif args.mode == 'ensemble':
        from models import STGAT
        from training import WeightedMSELoss, DeepEnsemble
        from evaluation import MetricsCalculator
        from visualization import PredictionVisualizer, SpatialVisualizer
        
        print("\n Training Deep Ensemble for Uncertainty Estimation...")
        
        model_kwargs = {
            'num_nodes': data['num_nodes'],
            'in_features': data_info['n_input_features'],
            'hidden_dim': CONFIG['hidden_dim'],
            'out_features': data_info['n_target_features'],
            'horizon': CONFIG['horizon'],
            'num_gat_layers': CONFIG['num_gat_layers'],
            'num_heads': CONFIG['num_heads'],
            'dropout': CONFIG['dropout'],
        }
        
        criterion = WeightedMSELoss(
            active_weight=CONFIG['active_weight'],
            feature_weights=CONFIG.get('feature_weights')
        )
        
        ensemble = DeepEnsemble(
            model_class=STGAT,
            model_kwargs=model_kwargs,
            n_models=CONFIG.get('n_ensemble_models', 5),
            device=DEVICE
        )
        
        output_dir = Path(CONFIG['output_dir'])
        ensemble.fit(train_loader, val_loader, adj_sparse, criterion, CONFIG,
                    save_dir=output_dir / 'models')
        
        # Estimate the calibration factor on validation only, then freeze it
        # before evaluating the test set.
        val_results = ensemble.generate_predictions(val_loader, adj_sparse)
        results = ensemble.generate_predictions(test_loader, adj_sparse)
        low, high = 0.1, 20.0
        for _ in range(50):
            factor = (low + high) / 2.0
            val_lower = val_results['mean'] - 1.96 * val_results['std'] * factor
            val_upper = val_results['mean'] + 1.96 * val_results['std'] * factor
            coverage = np.mean((val_results['targets'] >= val_lower) & (val_results['targets'] <= val_upper))
            if coverage < 0.95:
                low = factor
            else:
                high = factor
        results['calibration_factor'] = factor
        results['std_raw'] = results['std'].copy()
        results['std'] = results['std'] * factor

        target_offset = data['target_stats'].get('offset', data['target_stats']['mean'])
        target_std = data['target_stats']['std']
        target_offset = target_offset.reshape(1, 1, 1, -1)
        target_std = target_std.reshape(1, 1, 1, -1)
        for result in (val_results, results):
            result['mean'] = result['mean'] * target_std + target_offset
            result['targets'] = result['targets'] * target_std + target_offset
            result['std'] = result['std'] * target_std
        results['ci_lower'] = results['mean'] - 1.96 * results['std']
        results['ci_upper'] = results['mean'] + 1.96 * results['std']
        
        # Calculate metrics
        metrics_calc = MetricsCalculator(feature_names=data_info['target_features'])
        metrics = metrics_calc.calculate_all_metrics(
            results['targets'], results['mean'], results['std']
        )
        metrics_calc.print_metrics(metrics)
        
        # Visualizations
        pred_viz = PredictionVisualizer(
            feature_names=data_info['target_features'],
            save_dir=output_dir / 'figures'
        )
        pred_viz.plot_scatter_pred_vs_actual(results['targets'], results['mean'])
        pred_viz.plot_timeseries_prediction(
            results['targets'], results['mean'],
            node_idx=0, horizon_idx=0, y_pred_std=results['std']
        )
        
        # Spatial uncertainty
        spatial_viz = SpatialVisualizer(
            node_coords=adj_builder.coords,
            save_dir=output_dir / 'figures'
        )
        spatial_viz.plot_uncertainty_spatial(results['mean'], results['std'])
        
        # Save
        metrics_calc.save_metrics(metrics, output_dir / 'metrics' / 'ensemble_metrics.json')
        
    # ====================
    # MODE: ABLATION
    # ====================
    elif args.mode == 'ablation':
        from evaluation import AblationStudy
        from visualization import PredictionVisualizer
        
        print("\n Running Ablation Study...")
        
        ablation = AblationStudy(CONFIG, DEVICE)
        
        # Run all ablation configurations
        # Architecture: full_stgat, single_head, stgat_8heads, st_tft, gcn_lstm, stgat_no_skip, stgat_no_embed
        # Dual-Path: dualpath_concat, dualpath_gate, dualpath_add
        # Learnable Graph: learnable_geo30, learnable_geo50, learnable_geo70
        # Loss: loss_mse, loss_focal, loss_multiscale, loss_asymmetric
        # Regularization: dropout_low, dropout_high
        # Baselines: lstm_only, tft_only, etas, etas_fast, naive, moving_avg
        results = ablation.run_all(
            train_loader, val_loader, test_loader, adj_sparse, data_info,
            configs_to_run=[
                # Architecture
                'full_stgat', 'single_head', 'stgat_8heads', #'st_tft', 
                'gcn_lstm', 'stgat_no_skip', 'stgat_no_embed',
                # Dual-Path
                'dualpath_concat', 'dualpath_gate', 'dualpath_add',
                # Learnable Graph
                'learnable_geo30', 'learnable_geo50', 'learnable_geo70',
                # Warm Start (Curriculum Learning)
                'warmstart_5', 'warmstart_10', 'warmstart_20',
                # Loss functions
                'loss_mse', 'loss_focal', 'loss_multiscale', 'loss_asymmetric',
                # Regularization
                'dropout_low', 'dropout_high',
                # Baselines (Neural)
                'lstm_only', 'tft_only',
                # Baselines (Statistical)
                'etas', 'etas_fast', 'naive', 'moving_avg'
            ]
        )
        
        df = ablation.save_results(Path(CONFIG['output_dir']) / 'metrics')
        print("\n Ablation Results:")
        print(df.to_string())
        
        # Visualization
        pred_viz = PredictionVisualizer(save_dir=Path(CONFIG['output_dir']) / 'figures')
        pred_viz.plot_ablation_comparison(df)
    
    print("\n Done!")


def generate_regression_predictions_csv(
        predictions, targets, feature_names, output_dir,
        activity_targets=None, hurdle_outputs=None, activity_threshold=None):
    """Generate predictions CSV for regression model."""
    import pandas as pd
    
    all_rows = []
    B, H, N, F = predictions.shape
    
    sample_idx = 0
    for b in range(B):
        for h in range(H):
            for n in range(N):
                row = {
                    'sample_idx': sample_idx,
                    'horizon': h + 1,
                    'node_id': n,
                }
                for f_idx, fname in enumerate(feature_names):
                    row[f'{fname}_target'] = float(targets[b, h, n, f_idx])
                    row[f'{fname}_pred'] = float(predictions[b, h, n, f_idx])
                if activity_targets is not None:
                    row['activity_target'] = int(
                        activity_targets[b, h, n, 0] >= 0.5
                    )
                if hurdle_outputs is not None:
                    probability = float(
                        hurdle_outputs['activity_probability'][b, h, n, 0]
                    )
                    row['activity_probability'] = probability
                    if activity_threshold is not None:
                        row['activity_decision'] = int(
                            probability >= float(activity_threshold)
                        )
                    row['max_mw_pred_conditional'] = float(
                        hurdle_outputs['conditional'][b, h, n, 0]
                    )
                    row['max_mw_pred_expected'] = float(
                        hurdle_outputs['expected'][b, h, n, 0]
                    )
                    row['max_mw_pred_thresholded'] = float(
                        hurdle_outputs['thresholded'][b, h, n, 0]
                    )
                all_rows.append(row)
        sample_idx += 1
    
    df = pd.DataFrame(all_rows)
    csv_path = output_dir / 'predictions.csv'
    df.to_csv(csv_path, index=False)
    print(f"   Predictions saved to {csv_path}")
    return df


def generate_regression_html_viewer(df, output_path, feature_names):
    """Generate simple HTML viewer with node selector for regression predictions."""
    import json
    import numpy as np
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    nodes = sorted(df['node_id'].unique())
    
    # Prepare data per node
    node_data = {}
    for node in nodes:
        agg_dict = {'sample_idx': 'first'}
        for fname in feature_names:
            agg_dict[f'{fname}_target'] = 'mean'
            agg_dict[f'{fname}_pred'] = 'mean'
        
        node_df = df[df['node_id'] == node].groupby('sample_idx').agg(agg_dict).reset_index(drop=True)
        node_df['sample_idx'] = range(len(node_df))
        node_data[int(node)] = node_df.to_dict('records')
    
    data_json = json.dumps(node_data, cls=NumpyEncoder)
    feature_names_json = json.dumps(feature_names)
    
    # Generate chart sections
    chart_divs = ""
    for fname in feature_names:
        chart_divs += f'''
    <div class="chart-container">
        <div class="chart-title">{fname}: Actual vs Predicted</div>
        <canvas id="chart_{fname}"></canvas>
    </div>'''
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Earthquake Prediction Viewer</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .controls {{
            background: #fff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .controls label {{
            margin-right: 10px;
            font-weight: bold;
        }}
        .controls select, .controls input {{
            padding: 8px;
            margin-right: 20px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }}
        .chart-container {{
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }}
        canvas {{
            max-height: 250px;
        }}
    </style>
</head>
<body>
    <h1>Earthquake Prediction Viewer (Regression)</h1>
    
    <div class="controls">
        <label>Node:</label>
        <select id="nodeSelect">
            {" ".join([f'<option value="{n}">Node {n}</option>' for n in nodes])}
        </select>
        
        <label>Time Range:</label>
        <input type="range" id="rangeSlider" min="20" max="300" value="100">
        <span id="rangeValue">100</span>
        
        <label>Start:</label>
        <input type="range" id="startSlider" min="0" max="500" value="0">
        <span id="startValue">0</span>
    </div>
    {chart_divs}
    
    <script>
        const allData = {data_json};
        const featureNames = {feature_names_json};
        const charts = {{}};
        
        function initCharts() {{
            featureNames.forEach(fname => {{
                const ctx = document.getElementById('chart_' + fname);
                if (ctx) {{
                    charts[fname] = new Chart(ctx, {{
                        type: 'line',
                        data: {{ labels: [], datasets: [] }},
                        options: {{ responsive: true }}
                    }});
                }}
            }});
        }}
        
        function updateCharts() {{
            const node = parseInt(document.getElementById('nodeSelect').value);
            const range = parseInt(document.getElementById('rangeSlider').value);
            const start = parseInt(document.getElementById('startSlider').value);
            
            const nodeData = allData[node] || [];
            const end = Math.min(start + range, nodeData.length);
            const data = nodeData.slice(start, end);
            
            document.getElementById('rangeValue').textContent = range;
            document.getElementById('startValue').textContent = start;
            document.getElementById('startSlider').max = Math.max(0, nodeData.length - range);
            
            const labels = data.map((d, i) => start + i);
            
            featureNames.forEach(fname => {{
                if (charts[fname]) {{
                    charts[fname].data.labels = labels;
                    charts[fname].data.datasets = [
                        {{ label: 'Actual', data: data.map(d => d[fname + '_target']), borderColor: '#333', borderWidth: 2, fill: false, pointRadius: 1 }},
                        {{ label: 'Predicted', data: data.map(d => d[fname + '_pred']), borderColor: '#3498db', borderWidth: 2, fill: false, pointRadius: 1, borderDash: [5, 5] }}
                    ];
                    charts[fname].update();
                }}
            }});
        }}
        
        document.getElementById('nodeSelect').addEventListener('change', updateCharts);
        document.getElementById('rangeSlider').addEventListener('input', updateCharts);
        document.getElementById('startSlider').addEventListener('input', updateCharts);
        
        initCharts();
        updateCharts();
    </script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ST-GAT Earthquake Prediction')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'ensemble', 'ablation'],
                       help='Mode: train, eval, ensemble, or ablation')
    parser.add_argument('--print_config', action='store_true',
                       help='Print configuration')
    
    args = parser.parse_args()
    main(args)
