"""Focused tests for the two optimizer-ablation mechanisms."""

from __future__ import annotations

import torch
from peft.tuners.lora import Linear

from llava.model.ab_manifold import lora_factor_residuals, retract_lora_module
from llava.optim.muon_optimizer import MuonWithAdamW


def test_scaled_retraction_preserves_norm_and_makes_factors_orthogonal():
    layer = Linear("default", 12, 10, r=4, lora_alpha=4, lora_dropout=0.0)
    layer.lora_A["default"].weight.data.normal_(0, 0.2)
    layer.lora_B["default"].weight.data.normal_(0, 0.2)
    before = (
        torch.linalg.vector_norm(layer.lora_A["default"].weight.detach().float()),
        torch.linalg.vector_norm(layer.lora_B["default"].weight.detach().float()),
    )
    retract_lora_module(layer)
    after = (
        torch.linalg.vector_norm(layer.lora_A["default"].weight.detach().float()),
        torch.linalg.vector_norm(layer.lora_B["default"].weight.detach().float()),
    )
    residuals = lora_factor_residuals(layer)
    assert torch.allclose(before[0], after[0], rtol=1e-4, atol=1e-5)
    assert torch.allclose(before[1], after[1], rtol=1e-4, atol=1e-5)
    assert residuals["a_residual_after_mask"] < 1e-4
    assert residuals["b_residual_after_mask"] < 1e-4


def test_zero_initialized_B_is_left_unchanged():
    layer = Linear("default", 8, 6, r=4, lora_alpha=4, lora_dropout=0.0)
    before = layer.lora_B["default"].weight.detach().clone()
    retract_lora_module(layer)
    assert torch.equal(before, layer.lora_B["default"].weight.detach())


def test_muon_updates_matrix_and_adamw_fallback_updates_vector():
    matrix = torch.nn.Parameter(torch.randn(8, 4))
    vector = torch.nn.Parameter(torch.randn(4))
    optimizer = MuonWithAdamW(
        [
            {"params": [matrix], "lr": 1e-2},
            {"params": [vector], "lr": 1e-2},
        ],
        momentum=0.95,
        ns_steps=3,
        nesterov=True,
        weight_decay=0.0,
    )
    matrix_before = matrix.detach().clone()
    vector_before = vector.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        loss = matrix.square().mean() + vector.square().mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        assert torch.isfinite(matrix).all()
        assert torch.isfinite(vector).all()
    assert not torch.equal(matrix_before, matrix.detach())
    assert not torch.equal(vector_before, vector.detach())
    assert optimizer.state[matrix]["step"] == 2
    assert optimizer.state[vector]["step"] == 2
