# ==============================================================================
# MAIN_MULTIRESOLUTION.PY - Multi-Resolution Ensemble Training (High-Vis Edition)
# ==============================================================================
"""
Multi-Resolution Ensemble Approach for Earthquake Prediction
Updated with increased font sizes for all visualizations.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import json
import copy
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

from config import CONFIG, DEVICE


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    import random
    import os
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            pass


class MultiResolutionEnsemble:
    """
    Ensemble of models trained at different temporal resolutions.
    Each model predicts 1 step ahead at its resolution.
    """
    
    def __init__(self, resolutions, base_config, device):
        self.resolutions = resolutions
        self.base_config = base_config
        self.device = device
        self.models = {}
        self.data_cache = {}
        self.feature_stats = {}
        self.target_stats = {}
        self.canonical_split_timestamps = None
        
    def prepare_data(self, filepath):
        from data.preprocessing import DataPreprocessor
        from data.adjacency import AdjacencyBuilder
        from data.dataset import SeismicDataset
        from torch.utils.data import DataLoader
        
        print("\n" + "=" * 70)
        print(" PREPARING DATA FOR EACH RESOLUTION")
        print("=" * 70)
        
        for res in self.resolutions:
            print(f"\n Processing resolution: {res}")
            res_config = copy.deepcopy(self.base_config)
            res_config['time_bin'] = res
            res_config['horizon'] = 1  
            resolution_hours = pd.Timedelta(res).total_seconds() / 3600.0
            res_config['window_size'] = max(1, int(round(
                res_config.get('history_hours', 96) / resolution_hours
            )))
            if self.canonical_split_timestamps is not None:
                res_config['split_timestamps'] = self.canonical_split_timestamps
            
            preprocessor = DataPreprocessor(res_config)
            data = preprocessor.process(filepath)
            if self.canonical_split_timestamps is None:
                self.canonical_split_timestamps = data['split_timestamps']
            
            adj_builder = AdjacencyBuilder(res_config)
            adj_scipy = adj_builder.build_distance_weighted_adj(
                data['num_nodes'],
                data['node_info'],
                data['grid_params'],
                use_distance_weighting=True
            )
            adj_sparse = adj_builder.scipy_to_torch_sparse(adj_scipy, device=self.device)
            
            all_features = res_config['features']
            target_features = res_config.get('target_features', all_features)
            target_indices = [all_features.index(f) for f in target_features if f in all_features]
            
            g = torch.Generator()
            g.manual_seed(res_config['seed'])
            
            train_dataset = SeismicDataset(data['train_data'], target_data=data['train_target_data'], window_size=res_config['window_size'], horizon=1)
            val_dataset = SeismicDataset(data['val_data'], target_data=data['val_target_data'], window_size=res_config['window_size'], horizon=1)
            test_dataset = SeismicDataset(data['test_data'], target_data=data['test_target_data'], window_size=res_config['window_size'], horizon=1)
            
            train_loader = DataLoader(train_dataset, batch_size=res_config['batch_size'], shuffle=True, generator=g)
            val_loader = DataLoader(val_dataset, batch_size=res_config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=res_config['batch_size'], shuffle=False)
            
            data_info = {
                'num_nodes': data['num_nodes'],
                'n_input_features': data['train_data'].shape[-1],
                'n_target_features': len(target_features),
                'target_features': target_features,
                'feature_stats': data['feature_stats'],
                'target_stats': data['target_stats'],
                'target_indices': target_indices,
                'split_timestamps': data['split_timestamps'],
            }
            
            self.data_cache[res] = {
                'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
                'adj_sparse': adj_sparse, 'adj_scipy': adj_scipy, 'data_info': data_info,
                'config': res_config, 'coords': adj_builder.coords,
            }
            self.feature_stats[res] = data['feature_stats']
            self.target_stats[res] = data['target_stats']

    def train_all(self, output_dir):
        from models import STGAT
        from training import WeightedMSELoss, Trainer
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "=" * 70)
        print(" TRAINING MODELS FOR EACH RESOLUTION")
        print("=" * 70)
        
        results = {}
        for res in self.resolutions:
            print(f"\n Training model for resolution: {res}")
            set_seed(self.base_config['seed'])
            cache = self.data_cache[res]
            config = cache['config']
            data_info = cache['data_info']
            
            model = STGAT(
                num_nodes=data_info['num_nodes'],
                in_features=data_info['n_input_features'],
                hidden_dim=config['hidden_dim'],
                out_features=data_info['n_target_features'],
                horizon=1,
                num_gat_layers=config['num_gat_layers'],
                num_heads=config['num_heads'],
                dropout=config['dropout'],
                use_attention=True, use_multihead=True, use_skip=True, node_embed_dim=16,
            ).to(self.device)
            
            criterion = WeightedMSELoss(active_weight=config['active_weight'], feature_weights=config.get('feature_weights'))
            trainer = Trainer(model, criterion, config, self.device)
            train_result = trainer.fit(cache['train_loader'], cache['val_loader'], cache['adj_sparse'])
            
            self.models[res] = model
            results[res] = train_result
            
            model_path = output_dir / 'models' / f'model_{res}.pt'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'config': config, 'train_result': train_result}, model_path)
            
        return results

    def evaluate_ensemble(self, output_dir):
        from evaluation.metrics import MetricsCalculator
        output_dir = Path(output_dir)
        
        all_predictions, all_targets = {}, {}
        
        for res in self.resolutions:
            cache = self.data_cache[res]
            model = self.models[res]
            model.eval()
            
            preds, targs = [], []
            with torch.no_grad():
                for data, target in tqdm(cache['test_loader'], desc=f" Eval {res}"):
                    data = data.to(self.device)
                    output = model(data, cache['adj_sparse'])
                    preds.append(output.cpu().numpy())
                    targs.append(target.numpy())
            
            preds, targs = np.concatenate(preds, axis=0), np.concatenate(targs, axis=0)
            
            t_stats = self.target_stats[res]
            t_mean = t_stats.get('offset', t_stats['mean']).reshape(1, 1, 1, -1)
            t_std = t_stats['std'].reshape(1, 1, 1, -1)
            
            all_predictions[res] = preds * t_std + t_mean
            all_targets[res] = targs * t_std + t_mean
            
        return all_predictions, all_targets

    def load_models(self, model_dir):
        from models import STGAT
        model_dir = Path(model_dir)
        for res in self.resolutions:
            model_path = model_dir / f'model_{res}.pt'
            if not model_path.exists(): continue
            checkpoint = torch.load(model_path, map_location=self.device)
            data_info = self.data_cache[res]['data_info']
            model = STGAT(
                num_nodes=data_info['num_nodes'], in_features=data_info['n_input_features'],
                hidden_dim=checkpoint['config']['hidden_dim'], out_features=data_info['n_target_features'],
                horizon=1, num_gat_layers=checkpoint['config']['num_gat_layers'],
                num_heads=checkpoint['config']['num_heads'], dropout=checkpoint['config']['dropout'],
                use_attention=True, use_multihead=True, use_skip=True, node_embed_dim=16
            ).to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.models[res] = model

    def generate_combined_forecast(self, output_dir):
        res_to_hours = {'1h': 1, '2h': 2, '3h': 3, '4h': 4, '5h': 5, '6h': 6, '8h': 8, '12h': 12, '24h': 24}
        combined_predictions, combined_targets = {}, {}
        
        for res in self.resolutions:
            if res not in res_to_hours: continue
            h_ahead = res_to_hours[res]
            cache = self.data_cache[res]
            model = self.models[res]
            model.eval()
            
            preds, targs = [], []
            with torch.no_grad():
                for data, target in cache['test_loader']:
                    output = model(data.to(self.device), cache['adj_sparse'])
                    preds.append(output.cpu().numpy())
                    targs.append(target.numpy())
            
            preds, targs = np.concatenate(preds, axis=0), np.concatenate(targs, axis=0)
            t_stats = self.target_stats[res]
            t_mean = t_stats.get('offset', t_stats['mean']).reshape(1, 1, 1, -1)
            t_std = t_stats['std'].reshape(1, 1, 1, -1)
            
            combined_predictions[h_ahead] = (preds * t_std + t_mean).squeeze(1)
            combined_targets[h_ahead] = (targs * t_std + t_mean).squeeze(1)
            
        np.savez(Path(output_dir) / 'combined_forecast.npz', **{f'p_{h}': combined_predictions[h] for h in combined_predictions})
        return combined_predictions, combined_targets

    def visualize_results(self, all_predictions, all_targets, output_dir):
        import matplotlib.pyplot as plt
        from evaluation.metrics import MetricsCalculator
        import pandas as pd

        # --- FONT SIZE UPDATES ---
        plt.rcParams.update({
            'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
            'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 14,
            'figure.titlesize': 22
        })
        
        output_dir = Path(output_dir)
        fig_dir = output_dir / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        all_metrics = {}
        for res in self.resolutions:
            metrics_calc = MetricsCalculator(feature_names=self.data_cache[res]['data_info']['target_features'])
            all_metrics[res] = metrics_calc.calculate_all_metrics(
                all_targets[res], all_predictions[res], 
                magnitude_idx=self.base_config.get('magnitude_idx', 0)
            )

        # 1. METRICS COMPARISON
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        resolutions = list(all_metrics.keys())
        x = np.arange(len(resolutions))
        
        # RMSE plot example
        rmse_vals = [all_metrics[r]['overall']['RMSE'] for r in resolutions]
        axes[0, 0].bar(x, rmse_vals, color='steelblue', edgecolor='black')
        axes[0, 0].set_title('Root Mean Square Error')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(resolutions)
        for i, v in enumerate(rmse_vals):
            axes[0, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=13, fontweight='bold')

        # ... (MAE, R2, F1 plots follow same logic) ...
        # (Sisa plot kode bar tetap sama, hanya sesuaikan penempatan teks)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fig_dir / 'metrics_comparison.png', dpi=150)
        
        # 6. SUMMARY TABLE (Enhanced Size)
        summary_data = []
        for res in resolutions:
            m = all_metrics[res]
            summary_data.append({'Resolution': res, 'RMSE': m['overall']['RMSE'], 'MAE': m['overall']['MAE'], 'R²': m['overall']['R2']})
        
        df = pd.DataFrame(summary_data)
        fig, ax = plt.subplots(figsize=(14, len(resolutions) + 3))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(16)
        table.scale(1.2, 2.2) # Menambah tinggi baris agar teks besar tidak sesak
        
        plt.savefig(fig_dir / 'metrics_summary_table.png', dpi=150, bbox_inches='tight')
        plt.close('all')
        return all_metrics

    def visualize_combined_forecast(self, combined_predictions, combined_targets, output_dir):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        # --- FONT SIZE UPDATES ---
        plt.rcParams.update({
            'font.size': 14, 'axes.titlesize': 18, 'axes.labelsize': 16,
            'xtick.labelsize': 12, 'ytick.labelsize': 12, 'figure.titlesize': 22
        })

        output_dir = Path(output_dir)
        fig_dir = output_dir / 'figures'
        
        hours = sorted(combined_predictions.keys())
        mag_idx = self.base_config.get('magnitude_idx', 0)
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(hours)))
        
        # ERROR BY HORIZON
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        mae_vals, rmse_vals = [], []
        
        for h in hours:
            pred, true = combined_predictions[h][..., mag_idx].flatten(), combined_targets[h][..., mag_idx].flatten()
            mae_vals.append(np.mean(np.abs(pred - true)))
            rmse_vals.append(np.sqrt(np.mean((pred - true) ** 2)))

        axes[0].bar(hours, mae_vals, color='steelblue', edgecolor='black')
        axes[0].set_title('MAE by Horizon')
        for i, v in enumerate(mae_vals):
            axes[0].text(hours[i], v + 0.005, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')

        axes[1].bar(hours, rmse_vals, color='coral', edgecolor='black')
        axes[1].set_title('RMSE by Horizon')
        for i, v in enumerate(rmse_vals):
            axes[1].text(hours[i], v + 0.005, f'{v:.3f}', ha='center', fontsize=12, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(fig_dir / 'error_by_horizon.png', dpi=150)
        plt.close('all')


def main(args):
    set_seed(CONFIG['seed'])
    print(f"\n MULTI-RESOLUTION ENSEMBLE | MODE: {args.mode}")
    
    resolutions = [r.strip() for r in args.resolutions.split(',')]
    ensemble = MultiResolutionEnsemble(resolutions, CONFIG, DEVICE)
    ensemble.prepare_data(CONFIG['filename'])
    
    output_dir = Path(CONFIG['output_dir']) / 'multiresolution'
    
    if args.mode == 'train':
        ensemble.train_all(output_dir)
    else:
        ensemble.load_models(output_dir / 'models')
        
    predictions, targets = ensemble.evaluate_ensemble(output_dir)
    ensemble.visualize_results(predictions, targets, output_dir)
    
    comb_pred, comb_targ = ensemble.generate_combined_forecast(output_dir)
    ensemble.visualize_combined_forecast(comb_pred, comb_targ, output_dir)
    print("\n All processes completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--resolutions', type=str, default='1h,2h,4h,6h')
    main(parser.parse_args())
