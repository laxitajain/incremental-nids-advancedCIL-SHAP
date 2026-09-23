import torch
import torch.nn as nn
import numpy as np


class ShapAttention(nn.Module):
    """Learnable per-field attention layer initialized with SHAP-derived importance scores.

    This module applies channel-wise (per-field) attention weights to the input tensor
    before it is passed to the CNN backbone. The attention weights are initialized using
    global SHAP importance values, which encode the relative contribution of each
    packet-level field (PL, IAT, DIR, WIN) to the classification decision.

    The attention weights are learnable, allowing the model to refine the initial
    SHAP-based weighting during fine-tuning.

    Input shape:  (batch_size, 1, num_pkts, num_fields)
    Output shape: (batch_size, 1, num_pkts, num_fields)
    """

    def __init__(self, num_fields, shap_weights=None):
        """
        Args:
            num_fields (int): Number of input fields (channels), e.g. 4 for PL, IAT, DIR, WIN.
            shap_weights (np.ndarray or torch.Tensor, optional): SHAP importance weights
                of shape (num_fields,). If None, weights are initialized uniformly to 1.0.
        """
        super(ShapAttention, self).__init__()

        if shap_weights is not None:
            if isinstance(shap_weights, np.ndarray):
                shap_weights = torch.from_numpy(shap_weights).float()
            assert len(shap_weights) == num_fields, \
                f"SHAP weights length ({len(shap_weights)}) != num_fields ({num_fields})"
            # Normalize to mean=1 so initial attention doesn't change scale drastically
            shap_weights = shap_weights / shap_weights.mean()
            init_weights = shap_weights
        else:
            init_weights = torch.ones(num_fields)

        # Learnable attention weights (one per field)
        self.attention_weights = nn.Parameter(init_weights)

        # Optional temperature parameter for softmax-based attention
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """Apply per-field attention to input.

        Args:
            x: Input tensor of shape (batch_size, 1, num_pkts, num_fields)

        Returns:
            Attention-weighted input of same shape
        """
        # Compute soft attention via softmax (keeps weights positive and bounded)
        attn = torch.softmax(self.attention_weights / self.temperature, dim=0)
        # Scale to maintain the overall magnitude (multiply by num_fields)
        attn = attn * len(self.attention_weights)

        # Apply attention: broadcast over batch and packet dimensions
        # x shape: (B, 1, num_pkts, num_fields)
        # attn shape: (num_fields,) -> (1, 1, 1, num_fields)
        attn = attn.view(1, 1, 1, -1)
        return x * attn

    def get_attention_values(self):
        """Return the current attention weights as a numpy array for analysis."""
        with torch.no_grad():
            attn = torch.softmax(self.attention_weights / self.temperature, dim=0)
            attn = attn * len(self.attention_weights)
            return attn.cpu().numpy()
