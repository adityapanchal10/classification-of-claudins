import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients


def compute_ig_attributions(model, inputs, baseline, target_class, n_steps=50, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    inputs = inputs.to(device).requires_grad_(True)
    baseline = torch.zeros_like(inputs) if baseline is None else baseline.to(device)

    def forward_fn(inp):
        return model(inp)

    ig = IntegratedGradients(forward_fn)
    attributions, delta = ig.attribute(
        inputs,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
        method='gausslegendre',
    )
    residue_attrs = torch.sum(attributions, dim=2)
    return residue_attrs.detach().cpu(), delta.detach().cpu()


def residue_importance_dataframe(sequence: str, scores: np.ndarray) -> pd.DataFrame:
    max_abs = np.max(np.abs(scores)) if len(scores) else 1.0
    norm = np.zeros_like(scores) if max_abs == 0 else scores / max_abs
    return pd.DataFrame({
        "position": np.arange(1, len(sequence) + 1),
        "residue": list(sequence),
        "score": scores,
        "normalized_score": norm,
    })


def attention_dataframe(sequence: str, scores: np.ndarray) -> pd.DataFrame:
    vmax = np.max(scores) if len(scores) else 1.0
    norm = scores / vmax if vmax > 0 else np.zeros_like(scores)
    return pd.DataFrame({
        "position": np.arange(1, len(sequence) + 1),
        "residue": list(sequence),
        "attention": scores,
        "normalized_attention": norm,
    })
