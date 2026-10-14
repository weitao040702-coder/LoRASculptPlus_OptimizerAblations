"""Trainer variants for the LoRASculpt+ optimizer ablation study.

Both variants inherit SRR and CMR from the local LoRASculpt+ trainer while
setting ORTHO_LAMBDA to zero. AB-manifold performs a post-step retraction;
Muon replaces the optimizer with a real Muon/AdamW hybrid.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import torch
from peft.tuners.lora import LoraLayer

from llava.model.ab_manifold import lora_factor_residuals, retract_lora_model
from llava.optim.muon_optimizer import MuonWithAdamW
from llava.train.progress import update_run_progress
from llava.train.LoRASculptPlus_Trainer import LoRASculptPlus
from transformers.trainer import ALL_LAYERNORM_LAYERS, get_parameter_names, is_sagemaker_mp_enabled
from transformers.trainer_utils import speed_metrics
from transformers.utils import logging


logger = logging.get_logger(__name__)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class LoRASculptPlusOptimizerTrainer(LoRASculptPlus):
    """Shared no-orthogonal-loss behavior and progress reporting."""

    optimizer_label = "unspecified"

    def __init__(self, *args, **kwargs):
        os.environ["ORTHO_LAMBDA"] = "0"
        super().__init__(*args, **kwargs)
        self._ab_last_stats = {}
        self._last_progress_write = 0.0
        logger.info("Optimizer ablation: ORTHO_LAMBDA=0; optimizer=%s", self.optimizer_label)

    def _is_llm_lora(self, name):
        return "lora_" in name and "mm_projector" not in name

    def compute_custom_regularizers(self, model):
        """Compute only the two original CMR terms; the OCE term is removed."""
        llm_cmr_terms = []
        connector_cmr_terms = []
        for name, module in model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            if "default" not in module.lora_A or "default" not in module.lora_B:
                continue
            A = module.lora_A["default"].weight
            B = module.lora_B["default"].weight
            delta = B @ A
            if "mm_projector" in name:
                retention = self._retention_mask(module.weight).to(dtype=delta.dtype, device=delta.device)
                connector_cmr_terms.append(torch.linalg.vector_norm(retention * delta, ord=1))
            elif any(proj in name for proj in ("q_proj", "k_proj", "v_proj")):
                retention = self._retention_mask(module.weight).to(dtype=delta.dtype, device=delta.device)
                llm_cmr_terms.append(torch.linalg.matrix_norm(retention * delta, ord="fro"))

        device = next(model.parameters()).device
        zero = torch.zeros((), device=device)
        llm_cmr = torch.stack(llm_cmr_terms).sum() if llm_cmr_terms else zero
        connector_cmr = torch.stack(connector_cmr_terms).sum() if connector_cmr_terms else zero
        return (
            float(os.environ.get("CMR_LLM_LAMBDA", 1e-3)) * llm_cmr,
            float(os.environ.get("CMR_CONNECTOR_LAMBDA", 1e-5)) * connector_cmr,
            zero,
        )

    def _write_progress(self, status: str = "TRAINING", force: bool = False, **extra) -> None:
        progress_file = os.environ.get("OPTAB_PROGRESS_FILE")
        if not progress_file:
            return
        now = time.time()
        if not force and now - self._last_progress_write < 10:
            return
        self._last_progress_write = now
        payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "method": os.environ.get("OPTAB_METHOD_LABEL", os.environ.get("OPTAB_METHOD", self.optimizer_label)),
            "rank": int(os.environ.get("OPTAB_RANK", "0")),
            "gpu": os.environ.get("OPTAB_GPU", "unknown"),
            "status": status,
            "current_step": int(getattr(self.state, "global_step", 0)) + 1,
            "total_steps": int(getattr(self.state, "max_steps", 0)),
            "optimizer": self.optimizer_label,
            "orthogonal_loss": False,
            "cmr": True,
            "srr": bool(getattr(self, "AB_masks", {})),
            "log": os.environ.get("OPTAB_LOG", ""),
            "checkpoint": os.environ.get("OPTAB_CHECKPOINT", ""),
        }
        payload.update(extra)
        update_run_progress(Path(progress_file), payload)

    def _post_optimizer_step(self, model):
        """Default order for the Muon variant: optimizer then SRR."""
        self._apply_srr_masks(model)
        self._write_progress()

    def create_optimizer(self):
        raise NotImplementedError


class ABManifoldLoRASculptPlus(LoRASculptPlusOptimizerTrainer):
    optimizer_label = "AdamW + masked AB-manifold retraction"

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.optimizer is None:
            decay_names = set(get_parameter_names(self.model, ALL_LAYERNORM_LAYERS))
            decay_names = {name for name in decay_names if "bias" not in name}
            projector = {name for name, _ in self.model.named_parameters() if "mm_projector" in name}
            grouped = [
                {
                    "params": [p for n, p in self.model.named_parameters() if p.requires_grad and n not in projector and n in decay_names],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if p.requires_grad and n not in projector and n not in decay_names],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if p.requires_grad and n in projector],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_projector_lr,
                },
            ]
            grouped = [group for group in grouped if group["params"]]
            self.optimizer = torch.optim.AdamW(grouped, betas=(0.9, 0.999), eps=1e-8)
            logger.info("Actual optimizer class: %s", self.optimizer.__class__.__name__)
            print("Actual optimizer class: AdamW; AB-manifold retraction enabled; ORTHO_LAMBDA=0", flush=True)
        return self.optimizer

    def _post_optimizer_step(self, model):
        stats = retract_lora_model(model)
        self._ab_last_stats = stats
        # SRR is deliberately applied after retraction. The residuals written
        # after this call describe the masked factors used by the next batch.
        self._apply_srr_masks(model)
        stats.update(lora_factor_residuals(model))
        self._write_progress(ab_stats=stats)
        logger.info(
            "AB_MANIFOLD step=%d residual_a=%.6g residual_b=%.6g masked_a=%.4f masked_b=%.4f",
            self.state.global_step + 1,
            stats.get("a_residual_after_mask", 0.0),
            stats.get("b_residual_after_mask", 0.0),
            stats.get("a_nonzero_ratio", 1.0),
            stats.get("b_nonzero_ratio", 1.0),
        )


class MuonLoRASculptPlus(LoRASculptPlusOptimizerTrainer):
    optimizer_label = "MuonWithAdamW"

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.optimizer is None:
            llm_matrix = []
            connector_matrix = []
            vectors = []
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.ndim == 2 and "lora_" in name:
                    (connector_matrix if "mm_projector" in name else llm_matrix).append(parameter)
                else:
                    vectors.append(parameter)
            groups = []
            if llm_matrix:
                groups.append({"params": llm_matrix, "lr": self.args.learning_rate, "weight_decay": 0.0})
            if connector_matrix:
                groups.append({"params": connector_matrix, "lr": self.args.mm_projector_lr, "weight_decay": 0.0})
            if vectors:
                groups.append({"params": vectors, "lr": self.args.learning_rate, "weight_decay": 0.0})
            self.optimizer = MuonWithAdamW(
                groups,
                lr=self.args.learning_rate,
                momentum=0.95,
                ns_steps=5,
                nesterov=True,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
            logger.info(
                "Actual optimizer class: %s; momentum=0.95 ns_steps=5 nesterov=True; llm_2d=%d connector_2d=%d vectors=%d",
                self.optimizer.__class__.__name__, len(llm_matrix), len(connector_matrix), len(vectors),
            )
            print(
                "Actual optimizer class: MuonWithAdamW; momentum=0.95 ns_steps=5 nesterov=True; "
                "Newton-Schulz steps=5; ORTHO_LAMBDA=0",
                flush=True,
            )
        return self.optimizer
