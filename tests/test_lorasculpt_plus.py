import torch
from torch import nn

from llava.model.lorasculpt_plus import add_connector_lora_adapters
from llava.train.LoRASculptPlus_Trainer import LoRASculptPlus


def _linear(adapter_name="default", in_features=8, out_features=6, rank=4):
    from peft.tuners.lora import Linear

    return Linear(adapter_name, in_features, out_features, r=rank, lora_alpha=rank, lora_dropout=0.0)


def test_connector_injection_is_rank_128_and_freezes_base():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.mm_projector = nn.Sequential(nn.Linear(8, 6), nn.GELU(), nn.Linear(6, 8))

    toy = Toy()
    names = add_connector_lora_adapters(toy, rank=5, alpha=10, dropout=0.0)
    assert names == ["mm_projector.0", "mm_projector.2"]
    for name, module in toy.named_modules():
        if "mm_projector" not in name or not hasattr(module, "r"):
            continue
        assert module.r["default"] == 5
        assert module.weight.requires_grad is False
        assert module.lora_A["default"].weight.requires_grad is True
        assert module.lora_B["default"].weight.requires_grad is True


def test_orthogonality_penalty_matches_zero_and_nonzero_cases():
    orthogonal = torch.eye(3)
    repeated = torch.ones(3, 3)
    assert torch.equal(LoRASculptPlus._off_diagonal_gram_penalty(orthogonal), torch.zeros(()))
    assert LoRASculptPlus._off_diagonal_gram_penalty(repeated).item() > 0


def test_srr_only_masks_llm_factors_and_keeps_exact_fraction():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _linear(in_features=10, out_features=10, rank=10)
            self.mm_projector = _linear(in_features=10, out_features=10, rank=10)

    trainer = LoRASculptPlus.__new__(LoRASculptPlus)
    trainer.AB_masks = {}
    trainer.state = type("State", (), {"global_step": 1})()
    old_threshold = __import__("llava.train.LoRASculptPlus_Trainer", fromlist=["STEP_THRESHOLD"]).STEP_THRESHOLD
    old_ratio = __import__("llava.train.LoRASculptPlus_Trainer", fromlist=["AB_PRESERVE_RATIO"]).AB_PRESERVE_RATIO
    trainer_module = __import__("llava.train.LoRASculptPlus_Trainer", fromlist=["STEP_THRESHOLD"])
    trainer_module.STEP_THRESHOLD = 1
    trainer_module.AB_PRESERVE_RATIO = 0.1
    try:
        model = Toy()
        trainer._initialize_srr_masks(model)
        assert trainer.AB_masks
        assert all("mm_projector" not in name for name in trainer.AB_masks)
        for mask in trainer.AB_masks.values():
            assert int(mask.sum()) == max(1, int(mask.numel() * 0.1))
        model.q_proj.lora_A["default"].weight.data.fill_(1.0)
        trainer._apply_srr_masks(model)
        assert torch.count_nonzero(model.q_proj.lora_A["default"].weight).item() == 10
        assert torch.count_nonzero(model.mm_projector.lora_A["default"].weight).item() == 100
    finally:
        trainer_module.STEP_THRESHOLD = old_threshold
        trainer_module.AB_PRESERVE_RATIO = old_ratio


def test_retention_mask_is_finite_and_bounded():
    mask = LoRASculptPlus._retention_mask(nn.Parameter(torch.tensor([[1.0, 0.5], [0.1, -0.2]])))
    assert torch.isfinite(mask).all()
    assert torch.all(mask >= 0)
    assert torch.all(mask <= 1)


def test_srr_gradient_hook_masks_before_optimizer_step():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _linear(in_features=4, out_features=4, rank=2)

    trainer = LoRASculptPlus.__new__(LoRASculptPlus)
    model = Toy()
    parameter_name = "q_proj.lora_A.default.weight"
    parameter = dict(model.named_parameters())[parameter_name]
    mask = torch.zeros_like(parameter, dtype=torch.bool)
    mask[0, 0] = True
    trainer.AB_masks = {parameter_name: mask}
    trainer._srr_gradient_hooks = {}
    trainer._register_srr_gradient_hooks(model)

    parameter.sum().backward()
    expected = mask.to(dtype=parameter.grad.dtype)
    assert torch.equal(parameter.grad, expected)
