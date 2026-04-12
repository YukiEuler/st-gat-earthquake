# ==============================================================================
# MODEL_SUMMARY.PY - Generate TensorFlow-like Model Summary for ST-GAT
# ==============================================================================
"""
Generate a detailed model summary showing:
- Layer name and type
- Output shape
- Parameter count
- Connected from (input layer)

Usage:
    python model_summary.py
"""

import torch
import torch.nn as nn
from tabulate import tabulate
from collections import OrderedDict

from config import CONFIG, DEVICE
from models.stgat import STGAT


def count_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def format_shape(shape):
    """Format tensor shape as string."""
    return f"({', '.join(str(s) for s in shape)})"


def get_layer_info(model, input_shape, adj_shape):
    """
    Trace through the model and collect layer information.
    Returns a list of dicts with layer info.
    """
    layers_info = []
    
    B, T, N, F = input_shape
    
    # =========================================================================
    # 1. Input Layer
    # =========================================================================
    layers_info.append({
        'layer_name': 'Input',
        'layer_type': 'InputLayer',
        'output_shape': format_shape([B, T, N, F]),
        'params': 0,
        'connected_from': '-'
    })
    
    # =========================================================================
    # 2. Node Embedding (if exists)
    # =========================================================================
    if model.node_embedding is not None:
        embed_dim = model.node_embed_dim
        layers_info.append({
            'layer_name': 'node_embedding',
            'layer_type': 'Embedding',
            'output_shape': format_shape([B, T, N, embed_dim]),
            'params': count_parameters(model.node_embedding),
            'connected_from': 'node_indices'
        })
        
        # Concatenated output
        effective_features = F + embed_dim
        layers_info.append({
            'layer_name': 'concat_features',
            'layer_type': 'Concatenate',
            'output_shape': format_shape([B, T, N, effective_features]),
            'params': 0,
            'connected_from': 'Input, node_embedding'
        })
    else:
        effective_features = F
        
    # =========================================================================
    # 3. Input Projection (for skip connection)
    # =========================================================================
    layers_info.append({
        'layer_name': 'input_proj',
        'layer_type': 'Linear',
        'output_shape': format_shape([B, T, N, model.hidden_dim]),
        'params': count_parameters(model.input_proj),
        'connected_from': 'concat_features' if model.node_embedding else 'Input'
    })
    
    # =========================================================================
    # 4. GAT Layers (Spatial Encoder)
    # =========================================================================
    prev_layer = 'concat_features' if model.node_embedding else 'Input'
    
    for i, (gat_layer, norm_layer) in enumerate(zip(model.gat_layers, model.gat_norms)):
        layer_num = i + 1
        
        # GAT Layer
        gat_type = 'MultiHeadGAT' if model.use_attention else 'Linear'
        layers_info.append({
            'layer_name': f'gat_layer_{layer_num}',
            'layer_type': gat_type,
            'output_shape': format_shape([B, T, N, model.hidden_dim]),
            'params': count_parameters(gat_layer),
            'connected_from': f'{prev_layer}, adj_matrix'
        })
        
        # Skip Connection (residual)
        if model.use_skip and i > 0:
            layers_info.append({
                'layer_name': f'skip_add_{layer_num}',
                'layer_type': 'Add (Residual)',
                'output_shape': format_shape([B, T, N, model.hidden_dim]),
                'params': 0,
                'connected_from': f'gat_layer_{layer_num}, gat_norm_{layer_num-1}'
            })
        
        # Layer Normalization
        layers_info.append({
            'layer_name': f'gat_norm_{layer_num}',
            'layer_type': 'LayerNorm',
            'output_shape': format_shape([B, T, N, model.hidden_dim]),
            'params': count_parameters(norm_layer),
            'connected_from': f'skip_add_{layer_num}' if (model.use_skip and i > 0) else f'gat_layer_{layer_num}'
        })
        
        prev_layer = f'gat_norm_{layer_num}'
    
    # =========================================================================
    # 5. Reshape for LSTM
    # =========================================================================
    layers_info.append({
        'layer_name': 'reshape_for_lstm',
        'layer_type': 'Reshape',
        'output_shape': format_shape([B * N, T, model.hidden_dim]),
        'params': 0,
        'connected_from': prev_layer
    })
    
    # =========================================================================
    # 6. LSTM Layers (Temporal Encoder)
    # =========================================================================
    lstm_params = count_parameters(model.lstm)
    layers_info.append({
        'layer_name': 'lstm',
        'layer_type': f'LSTM (2-layer, {model.hidden_dim} units)',
        'output_shape': format_shape([B * N, T, model.hidden_dim]),
        'params': lstm_params,
        'connected_from': 'reshape_for_lstm'
    })
    
    # Take last timestep
    layers_info.append({
        'layer_name': 'lstm_last_step',
        'layer_type': 'Select[:, -1, :]',
        'output_shape': format_shape([B * N, model.hidden_dim]),
        'params': 0,
        'connected_from': 'lstm'
    })
    
    # Layer Norm after LSTM
    layers_info.append({
        'layer_name': 'lstm_norm',
        'layer_type': 'LayerNorm',
        'output_shape': format_shape([B * N, model.hidden_dim]),
        'params': count_parameters(model.lstm_norm),
        'connected_from': 'lstm_last_step'
    })
    
    # =========================================================================
    # 7. Skip Connection (Temporal)
    # =========================================================================
    if model.use_skip:
        layers_info.append({
            'layer_name': 'skip_temporal_pool',
            'layer_type': 'MeanPool (over T)',
            'output_shape': format_shape([B, N, model.hidden_dim]),
            'params': 0,
            'connected_from': 'input_proj'
        })
        
        layers_info.append({
            'layer_name': 'skip_temporal_proj',
            'layer_type': 'Linear',
            'output_shape': format_shape([B, N, model.hidden_dim]),
            'params': count_parameters(model.skip_temporal_proj),
            'connected_from': 'skip_temporal_pool'
        })
        
        layers_info.append({
            'layer_name': 'concat_skip',
            'layer_type': 'Concatenate',
            'output_shape': format_shape([B, N, model.hidden_dim * 2]),
            'params': 0,
            'connected_from': 'lstm_norm, skip_temporal_proj'
        })
        fc_input = 'concat_skip'
    else:
        fc_input = 'lstm_norm'
    
    # =========================================================================
    # 8. Output Layers
    # =========================================================================
    # Dropout
    layers_info.append({
        'layer_name': 'dropout',
        'layer_type': f'Dropout({model.dropout.p})',
        'output_shape': format_shape([B, N, model.hidden_dim * 2 if model.use_skip else model.hidden_dim]),
        'params': 0,
        'connected_from': fc_input
    })
    
    # FC layers (from Sequential)
    fc_modules = list(model.fc_out.children())
    prev = 'dropout'
    for j, fc_mod in enumerate(fc_modules):
        if isinstance(fc_mod, nn.Linear):
            out_dim = fc_mod.out_features
            layers_info.append({
                'layer_name': f'fc_out_{j}',
                'layer_type': 'Linear',
                'output_shape': format_shape([B, N, out_dim]),
                'params': count_parameters(fc_mod),
                'connected_from': prev
            })
            prev = f'fc_out_{j}'
        elif isinstance(fc_mod, nn.ReLU):
            layers_info.append({
                'layer_name': f'relu_{j}',
                'layer_type': 'ReLU',
                'output_shape': format_shape([B, N, out_dim]),
                'params': 0,
                'connected_from': prev
            })
            prev = f'relu_{j}'
        elif isinstance(fc_mod, nn.Dropout):
            layers_info.append({
                'layer_name': f'dropout_{j}',
                'layer_type': f'Dropout({fc_mod.p})',
                'output_shape': format_shape([B, N, out_dim]),
                'params': 0,
                'connected_from': prev
            })
            prev = f'dropout_{j}'
    
    # Final reshape
    layers_info.append({
        'layer_name': 'reshape_output',
        'layer_type': 'Reshape',
        'output_shape': format_shape([B, model.horizon, N, model.out_features]),
        'params': 0,
        'connected_from': prev
    })
    
    return layers_info


def print_model_summary(model, input_shape=(4, 24, 133, 8), adj_shape=(133, 133)):
    """Print a TensorFlow-style model summary."""
    
    print("\n" + "=" * 100)
    print(f" Model: ST-GAT (Spatio-Temporal Graph Attention Network)")
    print("=" * 100)
    
    # Get layer info
    layers_info = get_layer_info(model, input_shape, adj_shape)
    
    # Print as table
    headers = ['Layer (name)', 'Type', 'Output Shape', 'Param #', 'Connected From']
    rows = []
    for info in layers_info:
        rows.append([
            info['layer_name'],
            info['layer_type'],
            info['output_shape'],
            f"{info['params']:,}",
            info['connected_from']
        ])
    
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    
    # Summary
    total_params = sum(info['params'] for info in layers_info)
    trainable_params = count_parameters(model)
    
    print("\n" + "=" * 100)
    print(f" Total params: {total_params:,}")
    print(f" Trainable params: {trainable_params:,}")
    print(f" Non-trainable params: {total_params - trainable_params:,}")
    print("=" * 100)
    
    # Model configuration
    print("\n" + "=" * 100)
    print(" Model Configuration")
    print("=" * 100)
    config_info = [
        ['num_nodes', model.num_nodes],
        ['hidden_dim', model.hidden_dim],
        ['num_gat_layers', model.num_gat_layers],
        ['num_heads', model.num_heads],
        ['horizon', model.horizon],
        ['out_features', model.out_features],
        ['use_attention', model.use_attention],
        ['use_skip', model.use_skip],
        ['node_embed_dim', model.node_embed_dim],
    ]
    print(tabulate(config_info, headers=['Parameter', 'Value'], tablefmt='grid'))
    
    return layers_info, total_params


def export_to_latex(layers_info, output_file='model_summary.tex'):
    """Export layer info to LaTeX table."""
    
    latex = """\\begin{table}[h]
    \\centering
    \\caption{Arsitektur Model ST-GAT}
    \\label{tab:model_architecture}
    \\scriptsize
    \\begin{tabular}{|l|l|l|r|l|}
        \\hline
        \\textbf{Layer} & \\textbf{Tipe} & \\textbf{Output Shape} & \\textbf{Param} & \\textbf{Terhubung dari} \\\\
        \\hline
        \\hline
"""
    
    for info in layers_info:
        # Escape underscores for LaTeX
        name = info['layer_name'].replace('_', '\\_')
        ltype = info['layer_type'].replace('_', '\\_')
        connected = info['connected_from'].replace('_', '\\_')
        
        latex += f"        {name} & {ltype} & {info['output_shape']} & {info['params']:,} & {connected} \\\\\n"
        latex += "        \\hline\n"
    
    total_params = sum(info['params'] for info in layers_info)
    latex += f"""        \\hline
        \\multicolumn{{3}}{{|l|}}{{\\textbf{{Total Parameter}}}} & \\multicolumn{{2}}{{r|}}{{\\textbf{{{total_params:,}}}}} \\\\
        \\hline
    \\end{{tabular}}
\\end{{table}}
"""
    
    with open(output_file, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table exported to: {output_file}")
    return latex


def main():
    # Create model with typical configuration
    model = STGAT(
        num_nodes=133,
        in_features=8,
        hidden_dim=CONFIG['hidden_dim'],
        out_features=3,
        horizon=1,
        num_gat_layers=CONFIG['num_gat_layers'],
        num_heads=CONFIG['num_heads'],
        dropout=0.1,
        use_attention=True,
        use_multihead=True,
        use_skip=True,
        node_embed_dim=16
    ).to(DEVICE)
    
    # Print summary
    layers_info, total_params = print_model_summary(
        model,
        input_shape=(4, 24, 133, 8),  # B, T, N, F
        adj_shape=(133, 133)
    )
    
    # Export to LaTeX
    export_to_latex(layers_info, 'outputs/model_summary.tex')
    
    # Also print the actual PyTorch module structure
    print("\n" + "=" * 100)
    print(" PyTorch Module Structure")
    print("=" * 100)
    print(model)


if __name__ == '__main__':
    main()
