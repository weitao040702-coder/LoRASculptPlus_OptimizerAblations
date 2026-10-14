"""Scaled Stiefel retractions for LoRA A/B factors.

The retraction is deliberately applied to the actual LoRA factors instead of
the frozen pretrained weight.  It preserves each factor's Frobenius norm, so
it changes the update direction without changing the LoRA scale convention.
SRR is applied by the trainer after this retraction; the reported residuals
therefore describe the masked factors that are actually used for training.
"""

from __future__ import annotations

from typing import Dict

import torch
from peft.tuners.lora import LoraLayer


def _scaled_stiefel(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a scaled orthogonal factor with the input Frobenius norm."""
    if matrix.ndim != 2:
        return matrix

    original_dtype = matrix.dtype
    work = matrix.detach().float()
    norm = torch.linalg.vector_norm(work)
    if not torch.isfinite(norm) or norm <= eps:
        return matrix.detach().clone()

    rows, cols = work.shape
    # Polar retraction through the small Gram matrix avoids a QR/SVD over the
    # full hidden dimension for every LoRA layer at every optimizer step.
    if rows >= cols:
        gram = work.transpose(0, 1) @ work
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        inv_sqrt = eigenvectors @ torch.diag(torch.rsqrt(eigenvalues.clamp_min(eps))) @ eigenvectors.transpose(0, 1)
        projected = work @ inv_sqrt * (norm / (cols ** 0.5))
    else:
        gram = work @ work.transpose(0, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        inv_sqrt = eigenvectors @ torch.diag(torch.rsqrt(eigenvalues.clamp_min(eps))) @ eigenvectors.transpose(0, 1)
        projected = inv_sqrt @ work * (norm / (rows ** 0.5))
    return projected.to(dtype=original_dtype)


def _factor_residual(factor: torch.Tensor, orientation: str, eps: float = 1e-8) -> float:
    """Measure the scaled Stiefel residual for a B or A factor."""
    work = factor.detach().float()
    if orientation == "B":
        gram = work.transpose(0, 1) @ work
        rank = work.shape[1]
    else:
        gram = work @ work.transpose(0, 1)
        rank = work.shape[0]
    c = torch.linalg.vector_norm(work).square() / max(rank, 1)
    target = torch.eye(rank, device=work.device, dtype=work.dtype) * c
    residual = torch.linalg.matrix_norm(gram - target, ord="fro")
    return float(residual.detach().cpu()) if torch.isfinite(residual) else float("inf")


def retract_lora_module(module: LoraLayer, adapter_name: str = "default") -> Dict[str, float]:
    """Retract one PEFT LoRA layer and return diagnostics."""
    if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
        return {}

    A = module.lora_A[adapter_name].weight
    B = module.lora_B[adapter_name].weight
    with torch.no_grad():
        before_a = float(torch.linalg.vector_norm(A.detach().float()).cpu())
        before_b = float(torch.linalg.vector_norm(B.detach().float()).cpu())
        projected_a = _scaled_stiefel(A.transpose(0, 1)).transpose(0, 1)
        projected_b = _scaled_stiefel(B)
        A.copy_(projected_a)
        B.copy_(projected_b)
        after_a = float(torch.linalg.vector_norm(A.detach().float()).cpu())
        after_b = float(torch.linalg.vector_norm(B.detach().float()).cpu())

    return {
        "a_norm_before": before_a,
        "a_norm_after_projection": after_a,
        "b_norm_before": before_b,
        "b_norm_after_projection": after_b,
        "a_residual_before_mask": _factor_residual(A, "A"),
        "b_residual_before_mask": _factor_residual(B, "B"),
    }


def retract_lora_model(model: torch.nn.Module, adapter_name: str = "default") -> Dict[str, float]:
    """Retract all LLM and connector LoRA factors in ``model``."""
    values = []
    for module in model.modules():
        if isinstance(module, LoraLayer):
            stats = retract_lora_module(module, adapter_name=adapter_name)
            if stats:
                values.append(stats)
    if not values:
        raise RuntimeError("AB-manifold retraction found no LoRA factors")

    return {
        "layers": float(len(values)),
        "a_norm_before": sum(item["a_norm_before"] for item in values) / len(values),
        "a_norm_after_projection": sum(item["a_norm_after_projection"] for item in values) / len(values),
        "b_norm_before": sum(item["b_norm_before"] for item in values) / len(values),
        "b_norm_after_projection": sum(item["b_norm_after_projection"] for item in values) / len(values),
        "a_residual_before_mask": max(item["a_residual_before_mask"] for item in values),
        "b_residual_before_mask": max(item["b_residual_before_mask"] for item in values),
    }


def lora_factor_residuals(model: torch.nn.Module, adapter_name: str = "default") -> Dict[str, float]:
    """Return maximum residuals and exact nonzero ratios after SRR masking."""
    a_residuals = []
    b_residuals = []
    a_ratios = []
    b_ratios = []
    for module in model.modules():
        if not isinstance(module, LoraLayer):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue
        A = module.lora_A[adapter_name].weight.detach()
        B = module.lora_B[adapter_name].weight.detach()
        a_residuals.append(_factor_residual(A, "A"))
        b_residuals.append(_factor_residual(B, "B"))
        a_ratios.append(float(torch.count_nonzero(A).cpu()) / max(A.numel(), 1))
        b_ratios.append(float(torch.count_nonzero(B).cpu()) / max(B.numel(), 1))
    if not a_residuals:
        return {}
    return {
        "a_residual_after_mask": max(a_residuals),
        "b_residual_after_mask": max(b_residuals),
        "a_nonzero_ratio": sum(a_ratios) / len(a_ratios),
        "b_nonzero_ratio": sum(b_ratios) / len(b_ratios),
    }
