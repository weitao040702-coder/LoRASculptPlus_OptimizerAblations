"""Mixed-rank LoRA support used by the LoRASculpt+ reproduction.

PEFT 0.4.0, which is pinned by this project, does not expose ``rank_pattern``.
LoRASculpt+ needs one active LoRA adapter with different ranks for the LLM and
the multimodal connector.  This module keeps PEFT's standard LLM injection and
wraps the connector's linear layers with the same PEFT LoRA layer class.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear as PeftLoraLinear
from peft.tuners.lora import LoraLayer


LORASCULPT_PLUS_CONFIG_NAME = "lorasculpt_plus_config.json"
DEFAULT_ADAPTER_NAME = "default"
_MULTIMODAL_KEYWORDS = ("mm_projector", "vision_tower", "vision_resampler")


def find_llm_linear_names(model: nn.Module) -> List[str]:
    """Return the LLaVA LLM linear-module suffixes, excluding multimodal parts."""
    names = set()
    for name, module in model.named_modules():
        if any(keyword in name for keyword in _MULTIMODAL_KEYWORDS):
            continue
        if isinstance(module, nn.Linear):
            names.add(name.split(".")[-1])
    names.discard("lm_head")
    return sorted(names)


def make_llm_lora_config(
    *,
    rank: int,
    alpha: int,
    dropout: float,
    bias: str,
    target_modules: Iterable[str],
) -> LoraConfig:
    """Build the one PEFT configuration shared by LLM and connector layers."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias=bias,
        task_type="CAUSAL_LM",
    )


def add_llm_lora_adapters(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    bias: str,
    target_modules: Optional[Iterable[str]] = None,
):
    """Apply standard PEFT LoRA to all LLM linear modules."""
    if target_modules is None:
        target_modules = find_llm_linear_names(model)
    config = make_llm_lora_config(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        bias=bias,
        target_modules=target_modules,
    )
    return get_peft_model(model, config), list(config.target_modules)


def _parent_and_child(module: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parent = module
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def _replace_connector_linear(
    root: nn.Module,
    qualified_name: str,
    old_module: nn.Linear,
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> None:
    parent, child_name = _parent_and_child(root, qualified_name)
    replacement = PeftLoraLinear(
        DEFAULT_ADAPTER_NAME,
        old_module.in_features,
        old_module.out_features,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=old_module.bias is not None,
    )
    replacement.weight = old_module.weight
    if old_module.bias is not None:
        replacement.bias = old_module.bias
    if getattr(old_module, "state", None) is not None:
        replacement.state = old_module.state

    replacement.weight.requires_grad_(False)
    if replacement.bias is not None:
        replacement.bias.requires_grad_(False)
    replacement.to(device=old_module.weight.device)
    setattr(parent, child_name, replacement)


def freeze_non_lora_parameters(model: nn.Module) -> None:
    """LoRASculpt+ trains only LoRA factors, including connector factors."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name)


def add_connector_lora_adapters(
    peft_model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> List[str]:
    """Replace every connector linear layer with a rank-specific PEFT LoRA layer."""
    connector_linears = [
        (name, module)
        for name, module in peft_model.named_modules()
        if "mm_projector" in name and isinstance(module, nn.Linear) and not isinstance(module, LoraLayer)
    ]
    if not connector_linears:
        raise RuntimeError("LoRASculpt+ expected at least one nn.Linear under mm_projector.")

    for name, module in connector_linears:
        _replace_connector_linear(
            peft_model,
            name,
            module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    freeze_non_lora_parameters(peft_model)
    return [name for name, _ in connector_linears]


def validate_lorasculpt_plus_model(
    model: nn.Module,
    *,
    llm_rank: int,
    connector_rank: int,
) -> Dict[str, int]:
    """Validate the mixed-rank layout before a costly distributed run."""
    llm_layers = 0
    connector_layers = 0
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        if DEFAULT_ADAPTER_NAME not in module.r:
            continue
        actual_rank = module.r[DEFAULT_ADAPTER_NAME]
        if "mm_projector" in name:
            if actual_rank != connector_rank:
                raise RuntimeError(
                    f"Connector LoRA rank mismatch at {name}: {actual_rank} != {connector_rank}."
                )
            connector_layers += 1
        else:
            if actual_rank != llm_rank:
                raise RuntimeError(f"LLM LoRA rank mismatch at {name}: {actual_rank} != {llm_rank}.")
            llm_layers += 1

    if llm_layers == 0 or connector_layers == 0:
        raise RuntimeError(
            f"Incomplete LoRASculpt+ layout: {llm_layers} LLM and {connector_layers} connector LoRA layers."
        )

    non_lora_trainables = [name for name, param in model.named_parameters() if param.requires_grad and "lora_" not in name]
    if non_lora_trainables:
        raise RuntimeError("Unexpected non-LoRA trainables: " + ", ".join(non_lora_trainables[:8]))

    return {"llm_lora_layers": llm_layers, "connector_lora_layers": connector_layers}


def configure_lorasculpt_plus_adapters(
    model: nn.Module,
    *,
    llm_rank: int,
    llm_alpha: int,
    connector_rank: int,
    connector_alpha: int,
    dropout: float,
    bias: str,
    llm_target_modules: Optional[Iterable[str]] = None,
):
    """Create all LoRASculpt+ adapters and return serializable layout metadata."""
    peft_model, target_modules = add_llm_lora_adapters(
        model,
        rank=llm_rank,
        alpha=llm_alpha,
        dropout=dropout,
        bias=bias,
        target_modules=llm_target_modules,
    )
    connector_modules = add_connector_lora_adapters(
        peft_model,
        rank=connector_rank,
        alpha=connector_alpha,
        dropout=dropout,
    )
    counts = validate_lorasculpt_plus_model(
        peft_model,
        llm_rank=llm_rank,
        connector_rank=connector_rank,
    )
    metadata = {
        "format_version": 1,
        "adapter_name": DEFAULT_ADAPTER_NAME,
        "llm_rank": llm_rank,
        "llm_alpha": llm_alpha,
        "connector_rank": connector_rank,
        "connector_alpha": connector_alpha,
        "lora_dropout": dropout,
        "lora_bias": bias,
        "llm_target_modules": target_modules,
        "connector_modules": connector_modules,
        **counts,
    }
    return peft_model, metadata


def write_lorasculpt_plus_config(output_dir: str, metadata: Dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, LORASCULPT_PLUS_CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def read_lorasculpt_plus_config(model_path: str) -> Dict:
    path = os.path.join(model_path, LORASCULPT_PLUS_CONFIG_NAME)
    with open(path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {
        "llm_rank",
        "llm_alpha",
        "connector_rank",
        "connector_alpha",
        "lora_dropout",
        "lora_bias",
        "llm_target_modules",
    }
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"Invalid LoRASculpt+ checkpoint metadata; missing {missing}.")
    return metadata
