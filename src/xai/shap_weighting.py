"""SHAP-Guided Feature Weighting Script

Computes DeepSHAP values on a trained NIDS model and derives per-field importance
weights. These weights can be used to initialize the ShapAttention layer for
XAI-guided model improvement.

Usage:
    python3 shap_weighting.py \
        --model-path ../results/intra_edge-iot_scratch/models/ \
        --data-path ../data/uniform_label/edge-iiot_dwn10p.parquet \
        --fields PL IAT DIR WIN \
        --num-pkts 10 \
        --output-path ../results/shap_weights.npy \
        --num-samples 500
"""

import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def load_model(model_path, num_pkts, num_fields):
    """Load a trained model from disk."""
    from networks.lopez17cnn import Lopez17CNN
    from networks.network import LLL_Net

    # Find the model file
    if os.path.isdir(model_path):
        model_files = []
        for pattern in ('*.ckpt', '*.pth', '*.pt'):
            model_files.extend(glob(os.path.join(model_path, pattern)))
        model_files = sorted(model_files)
        if not model_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {model_path}. Expected one of: *.ckpt, *.pth, *.pt"
            )
        model_file = model_files[-1]  # use latest
    else:
        model_file = model_path

    print(f'Loading model from: {model_file}')

    # Create model architecture
    init_model = Lopez17CNN(num_pkts=num_pkts, num_fields=num_fields)
    model = LLL_Net(init_model)

    # Load state dict and infer number of heads/classes
    checkpoint = torch.load(model_file, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    # Count heads from state_dict
    head_keys = [k for k in state_dict.keys() if k.startswith('heads.')]
    num_heads = max(int(k.split('.')[1]) for k in head_keys) + 1 if head_keys else 1

    for h in range(num_heads):
        weight_key = f'heads.{h}.weight'
        if weight_key in state_dict:
            num_classes = state_dict[weight_key].shape[0]
            model.add_head(num_classes)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_data(data_path, fields, num_pkts, num_samples=500, seed=1):
    """Load and prepare data samples for SHAP computation."""
    import pandas as pd
    from datasets.dataset_config import min_max_config

    def normalize(x, field):
        min_ = min_max_config[field][0]
        max_ = min_max_config[field][1]
        x_n = (x - min_) / (max_ - min_)
        x_n[x_n < 0] = 0
        x_n[x_n > 1] = 1
        return x_n

    df = pd.read_parquet(data_path)

    scaled_cols = [f'SCALED_{field}' for field in fields]
    raw_cols = list(fields)

    if all(col in df.columns for col in scaled_cols):
        print(f'Loading preprocessed data from: {data_path}')
    elif all(col in df.columns for col in raw_cols):
        print(f'Loading raw data from: {data_path}')
        for field in fields:
            df[f'SCALED_{field}'] = df[field].apply(lambda x: normalize(x, field))
    else:
        raise KeyError(
            f"Could not find either raw columns {raw_cols} or preprocessed columns {scaled_cols} in {data_path}"
        )

    # Sample data
    if num_samples > 0 and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=seed)

    # Format as model input
    samples = []
    for _, row in df.iterrows():
        field_data = []
        for field in fields:
            col = f'SCALED_{field}'
            vals = row[col][:num_pkts] if len(row[col]) >= num_pkts else \
                np.concatenate([row[col], np.zeros(num_pkts - len(row[col]))])
            field_data.append(vals)
        sample = np.array(field_data).reshape(1, len(fields), num_pkts).transpose(0, 2, 1).astype('float32')
        samples.append(sample)

    data = np.array(samples)
    print(f'Loaded {len(data)} samples, shape: {data.shape}')
    return data


def compute_shap_weights(model, data, fields):
    """Compute DeepSHAP values and aggregate per-field importance."""
    try:
        import shap
    except ImportError:
        print("ERROR: 'shap' package not installed. Install with: pip install shap")
        sys.exit(1)

    print(f'Computing DeepSHAP values on {len(data)} samples...')

    # Use a subset as background for DeepSHAP
    num_background = min(100, len(data))
    background = torch.tensor(data[:num_background]).float()

    # Create explainer
    # Wrap model to output concatenated logits
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            outputs = self.model(x)
            return torch.cat(outputs, dim=1)

    wrapped = ModelWrapper(model)

    try:
        explainer = shap.DeepExplainer(wrapped, background)

        # Compute SHAP values
        test_data = torch.tensor(data).float()
        shap_values = explainer.shap_values(test_data)
    except Exception as e:
        print(f'DeepSHAP failed ({e}), falling back to GradientExplainer...')
        explainer = shap.GradientExplainer(wrapped, background)
        test_data = torch.tensor(data).float()
        shap_values = explainer.shap_values(test_data)

    def _reduce_to_fields(values):
        values = np.asarray(values)
        values = np.abs(values)

        if values.ndim < 2:
            raise ValueError(f'Unexpected SHAP value shape: {values.shape}')

        field_axes = [axis for axis, size in enumerate(values.shape) if size == len(fields)]
        if not field_axes:
            raise ValueError(f'Could not locate field axis in SHAP value shape: {values.shape}')

        field_axis = field_axes[0]
        reduce_axes = tuple(axis for axis in range(values.ndim) if axis != field_axis)
        reduced = values.mean(axis=reduce_axes)
        return np.asarray(reduced).reshape(-1)

    # shap_values may be a list (one array per class/output) or a single ndarray.
    # Reduce everything down to one scalar per input field.
    if isinstance(shap_values, list):
        per_field_importance = np.mean([_reduce_to_fields(sv) for sv in shap_values], axis=0)
    else:
        per_field_importance = _reduce_to_fields(shap_values)

    print(f'\nPer-field SHAP importance:')
    for i, field in enumerate(fields):
        print(f'  {field}: {per_field_importance[i]:.6f}')

    # Normalize to sum to 1
    per_field_importance = per_field_importance / per_field_importance.sum()

    print(f'\nNormalized weights:')
    for i, field in enumerate(fields):
        print(f'  {field}: {per_field_importance[i]:.4f}')

    return per_field_importance


def main():
    parser = argparse.ArgumentParser(description='Compute SHAP-guided feature weights for NIDS')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (directory or .pth file)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset parquet file')
    parser.add_argument('--fields', nargs='+', default=['PL', 'IAT', 'DIR', 'WIN'],
                        help='Fields to use (default: PL IAT DIR WIN)')
    parser.add_argument('--num-pkts', type=int, default=10,
                        help='Number of packets per biflow (default: 10)')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Path to save the weights (.npy file)')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of samples for SHAP computation (default: 500)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    args = parser.parse_args()

    # Load model and data
    model = load_model(args.model_path, args.num_pkts, len(args.fields))
    data = load_data(args.data_path, args.fields, args.num_pkts,
                     num_samples=args.num_samples, seed=args.seed)

    # Compute SHAP weights
    weights = compute_shap_weights(model, data, args.fields)

    # Save
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    np.save(args.output_path, weights)
    print(f'\nSHAP weights saved to: {args.output_path}')


if __name__ == '__main__':
    main()
