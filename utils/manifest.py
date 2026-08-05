"""Run provenance manifest for reproducible canonical reruns."""

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root):
    def run(*args):
        try:
            return subprocess.check_output(
                ['git', *args], cwd=project_root, text=True,
                stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    return {
        'commit': run('rev-parse', 'HEAD'),
        'status_short': run('status', '--short'),
    }


def write_run_manifest(output_dir, config, data=None, data_path=None,
                       adjacency=None, checkpoint_paths=None, stage='run'):
    """Write a JSON manifest that records every decision needed to rerun.

    The manifest is intentionally written at each major entry point. A later
    call can add checkpoint paths without changing the data-processing record.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]

    manifest = {
        'manifest_version': 1,
        'stage': stage,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'project_root': str(project_root),
        'python': sys.version,
        'platform': platform.platform(),
        'git': _git(project_root),
        'config': config,
        'seed': config.get('seed'),
        'target_definition': config.get('target_definition', 'configured_target_features'),
        'target_features': config.get('target_features', []),
        'checkpoint_paths': [str(path) for path in (checkpoint_paths or [])],
    }

    if data_path:
        path = Path(data_path)
        if not path.is_absolute():
            path = project_root / path
        manifest['data'] = {
            'path': str(path),
            'sha256': _sha256(path) if path.exists() else None,
            'size_bytes': path.stat().st_size if path.exists() else None,
        }

    if data is not None:
        manifest['preprocessing'] = {
            'feature_names': data.get('feature_names', []),
            'target_features': data.get('target_features', config.get('target_features', [])),
            'input_shape_full': list(getattr(data.get('X_raw'), 'shape', [])),
            'target_shape_full': list(getattr(data.get('target_raw'), 'shape', [])),
            'train_shape': list(getattr(data.get('train_data'), 'shape', [])),
            'val_shape': list(getattr(data.get('val_data'), 'shape', [])),
            'test_shape': list(getattr(data.get('test_data'), 'shape', [])),
            'split_timestamps': data.get('split_timestamps'),
            'train_end_idx': data.get('train_end_idx'),
            'val_end_idx': data.get('val_end_idx'),
            'feature_stats': data.get('feature_stats'),
            'target_stats': data.get('target_stats'),
            'grid_params': data.get('grid_params'),
            'node_info': data.get('node_info'),
        }

    if adjacency is not None:
        manifest['graph'] = {
            'shape': list(adjacency.shape),
            'nnz': int(adjacency.nnz),
            'weighted': bool(config.get('use_edge_weights', True)),
            'radius_km': config.get('radius_km'),
            'sigma_km': config.get('sigma_km'),
        }

    path = output_dir / 'run_manifest.json'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, default=str)
    print(f" Run manifest saved: {path}")
    return path
