#!/usr/bin/env python
"""Two-step CPU smoke for the new mathematical components."""

import torch
from peft.tuners.lora import Linear

from llava.model.ab_manifold import lora_factor_residuals, retract_lora_module
from llava.optim.muon_optimizer import MuonWithAdamW


def main():
    layer = Linear("default", 8, 6, r=4, lora_alpha=4, lora_dropout=0.0)
    layer.lora_B["default"].weight.data.normal_(0, 0.1)
    stats = retract_lora_module(layer)
    assert stats["a_norm_after_projection"] >= 0
    assert stats["b_norm_after_projection"] >= 0
    residual = lora_factor_residuals(layer)
    assert torch.isfinite(torch.tensor(list(residual.values()))).all()

    parameter = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = MuonWithAdamW([parameter], lr=1e-2, ns_steps=3)
    before = parameter.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        loss = parameter.square().mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(parameter).all()
        assert torch.isfinite(loss)
    assert not torch.equal(before, parameter.detach())
    print("AB_MANIFOLD_SMOKE=PASS")
    print("MUON_SMOKE=PASS")


if __name__ == "__main__":
    main()
