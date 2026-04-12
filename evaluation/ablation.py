# ==============================================================================
# ABLATION.PY - Ablation Study
# ==============================================================================

import torch
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import json


class AblationStudy:
    """Run ablation study comparing different model configurations."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.results = {}
    
    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def get_ablation_configs(self):
        """Define ablation configurations."""
        return {
            # ======================
            # MODEL ARCHITECTURE
            # ======================
            'full_stgat': {
                'name': 'Full ST-GAT',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 4,
            },
            'single_head': {
                'name': 'ST-GAT (Single Head)',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 1,
            },
            'stgat_8heads': {
                'name': 'ST-GAT (8 Heads)',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 8,
            },
            'st_tft': {
                'name': 'ST-TFT (Temp. Fusion)',
                'model_type': 'tft',
                'num_heads': 4,
            },
            'gcn_lstm': {
                'name': 'GCN + LSTM',
                'use_attention': False,
                'use_multihead': False,
                'num_heads': 1,
            },
            'stgat_no_skip': {
                'name': 'ST-GAT (No Skip)',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 4,
                'use_skip': False,
            },
            'stgat_no_embed': {
                'name': 'ST-GAT (No Node Embed)',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 4,
                'node_embed_dim': 0,
            },
            
            # ======================
            # DUAL-PATH ARCHITECTURE
            # ======================
            'dualpath_concat': {
                'name': 'ST-GAT Dual-Path (Concat)',
                'model_type': 'dualpath',
                'fusion_type': 'concat',
                'num_heads': 4,
            },
            'dualpath_gate': {
                'name': 'ST-GAT Dual-Path (Gate)',
                'model_type': 'dualpath',
                'fusion_type': 'gate',
                'num_heads': 4,
            },
            'dualpath_add': {
                'name': 'ST-GAT Dual-Path (Add)',
                'model_type': 'dualpath',
                'fusion_type': 'add',
                'num_heads': 4,
            },
            
            # ======================
            # LEARNABLE GRAPH
            # ======================
            'learnable_geo50': {
                'name': 'ST-GAT Learnable (Geo 50%)',
                'model_type': 'learnable',
                'geographic_weight': 0.5,  # 50% geo, 50% learned
                'num_heads': 4,
            },
            'learnable_geo30': {
                'name': 'ST-GAT Learnable (Geo 30%)',
                'model_type': 'learnable',
                'geographic_weight': 0.3,  # 30% geo, 70% learned
                'num_heads': 4,
            },
            'learnable_geo70': {
                'name': 'ST-GAT Learnable (Geo 70%)',
                'model_type': 'learnable',
                'geographic_weight': 0.7,  # 70% geo, 30% learned
                'num_heads': 4,
            },
            
            # ======================
            # WARM START (Curriculum Learning)
            # ======================
            'warmstart_10': {
                'name': 'ST-GAT WarmStart (10 epochs)',
                'model_type': 'warmstart',
                'warmup_epochs': 10,
                'num_heads': 4,
            },
            'warmstart_20': {
                'name': 'ST-GAT WarmStart (20 epochs)',
                'model_type': 'warmstart',
                'warmup_epochs': 20,
                'num_heads': 4,
            },
            'warmstart_5': {
                'name': 'ST-GAT WarmStart (5 epochs)',
                'model_type': 'warmstart',
                'warmup_epochs': 5,
                'num_heads': 4,
            },
            
            # ======================
            # LOSS FUNCTIONS
            # ======================
            'loss_mse': {
                'name': 'Loss: Weighted MSE',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 4,
                'loss_type': 'weighted_mse',
            },
            'loss_focal': {
                'name': 'Loss: Focal',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 4,
                'loss_type': 'focal',
            },
            'loss_multiscale': {
                'name': 'Loss: Multi-Scale',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 4,
                'loss_type': 'multiscale',
            },
            'loss_asymmetric': {
                'name': 'Loss: Asymmetric',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 4,
                'loss_type': 'asymmetric',
            },
            
            # ======================
            # REGULARIZATION
            # ======================
            'dropout_low': {
                'name': 'Dropout 0.1',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 4,
                'dropout': 0.1,
            },
            'dropout_high': {
                'name': 'Dropout 0.5',
                'use_attention': True,
                'use_multihead': True,
                'num_heads': 4,
                'dropout': 0.5,
            },
            
            # ======================
            # MULTI-SCALE TEMPORAL
            # ======================
            'multiscale_124': {
                'name': 'ST-GAT Multi-Scale (1,2,4)',
                'model_type': 'multiscale',
                'scales': [1, 2, 4],
                'fusion_type': 'concat',
                'num_heads': 4,
            },
            'multiscale_1248': {
                'name': 'ST-GAT Multi-Scale (1,2,4,8)',
                'model_type': 'multiscale',
                'scales': [1, 2, 4, 8],
                'fusion_type': 'concat',
                'num_heads': 4,
            },
            'multiscale_attn': {
                'name': 'ST-GAT Multi-Scale (Attn Fusion)',
                'model_type': 'multiscale',
                'scales': [1, 2, 4],
                'fusion_type': 'attention',
                'num_heads': 4,
            },
            'multiscale_gate': {
                'name': 'ST-GAT Multi-Scale (Gate Fusion)',
                'model_type': 'multiscale',
                'scales': [1, 2, 4],
                'fusion_type': 'gate',
                'num_heads': 4,
            },
            
            # ======================
            # BASELINES
            # ======================
            'lstm_only': {
                'name': 'LSTM Only',
                'model_type': 'lstm_only',
            },
            'tft_only': {
                'name': 'TFT Only',
                'model_type': 'tft_only',
                'n_heads': 4,
                'n_layers': 2,
            },
            'etas': {
                'name': 'ETAS Baseline',
                'model_type': 'etas',
                'decay_p': 1.2,
            },
            'etas_fast': {
                'name': 'ETAS (Fast Decay)',
                'model_type': 'etas',
                'decay_p': 1.5,
            },
            'naive': {
                'name': 'Naive (Last Obs)',
                'model_type': 'naive',
            },
            'moving_avg': {
                'name': 'Moving Average',
                'model_type': 'moving_avg',
            },
        }

    
    def run_single_config(self, config_name, ablation_config, train_loader, 
                         val_loader, test_loader, adj_sparse, data_info):
        """Run experiment for a single ablation configuration."""
        from models import STGAT, STGATDualPath, LSTMOnly, TFTOnly, NaiveBaseline, MovingAverageBaseline, ETASBaseline
        from training import WeightedMSELoss, Trainer
        from evaluation.metrics import MetricsCalculator
        
        # Reset seed for reproducibility (same seed for all configs)
        self._set_seed(self.config.get('seed', 42))
        
        print(f"\n{'='*70}")
        print(f" Running: {ablation_config['name']}")
        print('='*70)
        
        # Handle non-trainable baselines
        if ablation_config.get('model_type') == 'naive':
            baseline = NaiveBaseline(
                horizon=self.config['horizon'],
                out_features=data_info['n_target_features']
            )
            return self._evaluate_baseline(baseline, test_loader, data_info)
        
        elif ablation_config.get('model_type') == 'moving_avg':
            baseline = MovingAverageBaseline(
                window=5, 
                horizon=self.config['horizon'],
                out_features=data_info['n_target_features']
            )
            return self._evaluate_baseline(baseline, test_loader, data_info)
        
        elif ablation_config.get('model_type') == 'etas':
            baseline = ETASBaseline(
                horizon=self.config['horizon'],
                out_features=data_info['n_target_features'],
                decay_p=ablation_config.get('decay_p', 1.2)
            )
            return self._evaluate_baseline(baseline, test_loader, data_info)
        
        elif ablation_config.get('model_type') == 'lstm_only':
            model = LSTMOnly(
                num_nodes=data_info['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=self.config['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=self.config['horizon'],
                dropout=self.config['dropout']
            ).to(self.device)
        
        elif ablation_config.get('model_type') == 'tft_only':
            model = TFTOnly(
                num_nodes=data_info['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=self.config['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=self.config['horizon'],
                n_heads=ablation_config.get('n_heads', 4),
                n_layers=ablation_config.get('n_layers', 2),
                dropout=self.config['dropout']
            ).to(self.device)
        
        elif ablation_config.get('model_type') == 'learnable':
            from models import STGATLearnable
            model = STGATLearnable(
                num_nodes=data_info['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=self.config['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=self.config['horizon'],
                num_gat_layers=self.config['num_gat_layers'],
                num_heads=ablation_config.get('num_heads', 4),
                dropout=self.config['dropout'],
                geographic_weight=ablation_config.get('geographic_weight', 0.5),
                use_attention=True,
                use_multihead=True,
                use_skip=True,
                node_embed_dim=16,
            ).to(self.device)
        
        elif ablation_config.get('model_type') == 'warmstart':
            from models import STGATWarmStart
            model = STGATWarmStart(
                num_nodes=data_info['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=self.config['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=self.config['horizon'],
                num_gat_layers=self.config['num_gat_layers'],
                num_heads=ablation_config.get('num_heads', 4),
                dropout=self.config['dropout'],
                use_attention=True,
                use_multihead=True,
                use_skip=True,
                node_embed_dim=16,
                warmup_epochs=ablation_config.get('warmup_epochs', 10),
            ).to(self.device)
        
        elif ablation_config.get('model_type') == 'tft':
            from models import STTFT
            model = STTFT(
                num_nodes=data_info['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=self.config['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=self.config['horizon'],
                num_gat_layers=self.config['num_gat_layers'],
                num_heads=ablation_config.get('num_heads', 4),
                tft_layers=self.config.get('tft_layers', 2),
                dropout=self.config['dropout'],
            ).to(self.device)
        
        elif ablation_config.get('model_type') == 'dualpath':
            # Dual-Path ST-GAT: GAT+LSTM and LSTM-only paths merged
            model = STGATDualPath(
                num_nodes=data_info['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=self.config['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=self.config['horizon'],
                num_gat_layers=self.config['num_gat_layers'],
                num_heads=ablation_config.get('num_heads', 4),
                dropout=ablation_config.get('dropout', self.config['dropout']),
                use_attention=True,
                use_multihead=True,
                use_skip=True,
                node_embed_dim=16,
                fusion_type=ablation_config.get('fusion_type', 'concat'),
            ).to(self.device)
        
        elif ablation_config.get('model_type') == 'multiscale':
            from models import STGATMultiScale
            model = STGATMultiScale(
                num_nodes=data_info['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=self.config['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=self.config['horizon'],
                num_gat_layers=self.config['num_gat_layers'],
                num_heads=ablation_config.get('num_heads', 4),
                dropout=self.config['dropout'],
                scales=ablation_config.get('scales', [1, 2, 4]),
                fusion_type=ablation_config.get('fusion_type', 'concat'),
                use_attention=True,
                use_multihead=True,
                node_embed_dim=16,
                pool_type='avg',
            ).to(self.device)
        
        else:
            # ST-GAT variants - use ablation_config values if provided, else defaults
            model = STGAT(
                num_nodes=data_info['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=self.config['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=self.config['horizon'],
                num_gat_layers=self.config['num_gat_layers'],
                num_heads=ablation_config.get('num_heads', 4),
                dropout=ablation_config.get('dropout', self.config['dropout']),
                use_attention=ablation_config.get('use_attention', True),
                use_multihead=ablation_config.get('use_multihead', True),
                use_skip=ablation_config.get('use_skip', True),
                node_embed_dim=ablation_config.get('node_embed_dim', 16),
            ).to(self.device)
        
        # Train - use ablation_config loss_type if provided, else global config
        loss_type = ablation_config.get('loss_type', self.config.get('loss_type', 'weighted_mse'))
        
        if loss_type == 'asymmetric':
            from training.losses import AsymmetricMSELoss
            criterion = AsymmetricMSELoss(
                alpha=self.config.get('asymmetric_alpha', 0.8),
                magnitude_idx=self.config.get('magnitude_idx', 1),
                active_weight=self.config['active_weight'],
                feature_weights=self.config.get('feature_weights')
            )
        elif loss_type == 'active_only':
            from training.losses import ActiveOnlyMSELoss
            criterion = ActiveOnlyMSELoss(
                feature_weights=self.config.get('feature_weights')
            )
        elif loss_type == 'sparse_aware':
            from training.losses import SparseAwareLoss
            criterion = SparseAwareLoss(
                active_loss_weight=self.config['active_weight'],
                underpredict_penalty=self.config.get('underpredict_penalty', 2.0),
                feature_weights=self.config.get('feature_weights')
            )
        elif loss_type == 'focal':
            from training.losses import FocalMSELoss
            criterion = FocalMSELoss(
                gamma=self.config.get('focal_gamma', 2.0),
                active_weight=self.config['active_weight'],
                feature_weights=self.config.get('feature_weights')
            )
        elif loss_type == 'multiscale':
            from training.losses import MultiScaleLoss
            criterion = MultiScaleLoss(
                scales=self.config.get('multiscale_horizons', [1, 2, 4]),
                scale_weights=self.config.get('multiscale_weights', [1.0, 1.0, 1.0]),
                magnitude_idx=self.config.get('magnitude_idx', 1),
                active_weight=self.config['active_weight'],
                feature_weights=self.config.get('feature_weights')
            )
        else:  # default: weighted_mse
            criterion = WeightedMSELoss(
                active_weight=self.config['active_weight'],
                feature_weights=self.config.get('feature_weights')
            )
        
        trainer = Trainer(model, criterion, self.config, self.device)
        train_result = trainer.fit(train_loader, val_loader, adj_sparse)
        
        # Evaluate
        predictions, targets = self._generate_predictions(model, test_loader, adj_sparse)
        
        # Denormalize predictions and targets for proper metrics
        predictions, targets = self._denormalize(predictions, targets, data_info)
        
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_all_metrics(
            targets, predictions, 
            magnitude_idx=self.config.get('magnitude_idx', 0)
        )
        
        return {
            'name': ablation_config['name'],
            'train_result': train_result,
            'metrics': metrics,
            'n_params': sum(p.numel() for p in model.parameters()),
        }
    
    def _evaluate_baseline(self, baseline, test_loader, data_info):
        """Evaluate non-trainable baseline."""
        from evaluation.metrics import MetricsCalculator
        
        predictions = []
        targets = []
        
        for data, target in tqdm(test_loader, desc="Evaluating"):
            pred = baseline.predict(data)
            predictions.append(pred.numpy())
            targets.append(target.numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Denormalize predictions and targets for proper metrics
        predictions, targets = self._denormalize(predictions, targets, data_info)
        
        metrics_calc = MetricsCalculator()
        metrics = metrics_calc.calculate_all_metrics(
            targets, predictions,
            magnitude_idx=self.config.get('magnitude_idx', 0)
        )
        
        return {
            'name': baseline.name,
            'train_result': None,
            'metrics': metrics,
            'n_params': 0,
        }
    
    def _generate_predictions(self, model, test_loader, adj_sparse):
        """Generate predictions for a model."""
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Predicting"):
                data = data.to(self.device)
                output = model(data, adj_sparse)
                predictions.append(output.cpu().numpy())
                targets.append(target.numpy())
        
        return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)
    
    def _denormalize(self, predictions, targets, data_info):
        """Denormalize predictions and targets to original scale."""
        feature_stats = data_info.get('feature_stats')
        target_indices = data_info.get('target_indices')
        
        if feature_stats is None or target_indices is None:
            print("   Warning: feature_stats not available, using normalized metrics")
            return predictions, targets
        
        # Get target feature stats
        target_mean = feature_stats['mean'][target_indices]
        target_std = feature_stats['std'][target_indices]
        
        # Reshape for broadcasting: predictions shape is (B, H, N, F)
        n_dim = predictions.ndim
        if n_dim == 4:  # (B, H, N, F)
            target_mean = target_mean.reshape(1, 1, 1, -1)
            target_std = target_std.reshape(1, 1, 1, -1)
        elif n_dim == 3:  # (B, N, F)
            target_mean = target_mean.reshape(1, 1, -1)
            target_std = target_std.reshape(1, 1, -1)
        
        predictions_denorm = predictions * target_std + target_mean
        targets_denorm = targets * target_std + target_mean
        
        return predictions_denorm, targets_denorm
    
    def run_all(self, train_loader, val_loader, test_loader, adj_sparse, data_info, 
               configs_to_run=None, checkpoint_dir=None):
        """Run all ablation configurations with checkpointing."""
        all_configs = self.get_ablation_configs()
        
        if configs_to_run:
            all_configs = {k: v for k, v in all_configs.items() if k in configs_to_run}
        
        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path(self.config.get('output_dir', 'outputs')) / 'ablation_checkpoints'
        else:
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing results
        checkpoint_file = checkpoint_dir / 'ablation_results.json'
        if checkpoint_file.exists():
            try:
                import json
                with open(checkpoint_file, 'r') as f:
                    saved_results = json.load(f)
                self.results = saved_results
                print(f"\n Loaded {len(saved_results)} cached ablation results")
            except:
                pass
        
        for config_name, ablation_config in all_configs.items():
            # Skip if already computed
            if config_name in self.results:
                print(f"\n Skipping {ablation_config['name']} (cached)")
                continue
            
            try:
                result = self.run_single_config(
                    config_name, ablation_config, 
                    train_loader, val_loader, test_loader, adj_sparse, data_info
                )
                self.results[config_name] = result
                
                # Save checkpoint after each successful run
                self._save_checkpoint(checkpoint_file)
                print(f"   Checkpoint saved: {config_name}")
                
            except Exception as e:
                print(f"\n ERROR in {ablation_config['name']}: {str(e)}")
                print(f"   Skipping and continuing...")
                # Save partial results
                self._save_checkpoint(checkpoint_file)
                continue
        
        return self.results
    
    def _save_checkpoint(self, checkpoint_file):
        """Save current results to checkpoint file."""
        import json
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for k, v in self.results.items():
            # Skip None entries (failed or incomplete runs)
            if v is None:
                continue
            
            # Handle train_result being None (e.g., for baselines)
            train_result = v.get('train_result') or {}
            
            serializable_results[k] = {
                'name': v['name'],
                'n_params': v['n_params'],
                'metrics': v['metrics'],
                'train_epochs': train_result.get('epochs', 0),
                'best_loss': train_result.get('best_loss', 0),
            }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

    
    def get_comparison_table(self):
        """Create comparison table of all configurations."""
        rows = []
        
        for config_name, result in self.results.items():
            row = {
                'Configuration': result['name'],
                'Parameters': result['n_params'],
            }
            
            # Add overall metrics
            overall = result['metrics']['overall']
            row.update({
                'MSE': overall['MSE'],
                'RMSE': overall['RMSE'],
                'MAE': overall['MAE'],
                'R2': overall['R2'],
            })
            
            # Add classification metrics (Mw >= 1.0)
            if 'classification_mw1' in result['metrics']:
                cls = result['metrics']['classification_mw1']
                row.update({
                    'Cls_Accuracy': cls['accuracy'],
                    'Cls_Precision': cls['precision'],
                    'Cls_Recall': cls['recall'],
                    'Cls_F1': cls['f1_score'],
                })
            
            # Add classification metrics (Mw >= 3.0)
            if 'classification_mw3' in result['metrics']:
                cls3 = result['metrics']['classification_mw3']
                row.update({
                    'Cls3_Precision': cls3['precision'],
                    'Cls3_Recall': cls3['recall'],
                    'Cls3_F1': cls3['f1_score'],
                })
            
            # Add uncertainty if available
            if 'uncertainty' in result['metrics']:
                unc = result['metrics']['uncertainty']
                row['Coverage'] = unc['coverage_95']
                row['Sharpness'] = unc['sharpness']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def save_results(self, save_dir):
        """Save ablation results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        json_path = save_dir / 'ablation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save comparison table as CSV
        df = self.get_comparison_table()
        csv_path = save_dir / 'ablation_comparison.csv'
        df.to_csv(csv_path, index=False)
        
        print(f" Ablation results saved to {save_dir}")
        
        return df


# Need numpy for baseline evaluation
import numpy as np
