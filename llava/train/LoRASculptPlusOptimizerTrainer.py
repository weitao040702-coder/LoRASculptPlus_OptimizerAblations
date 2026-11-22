"""Shared training behavior used by the OrbitStiefel trainer."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import torch
from peft.tuners.lora import LoraLayer

from llava.train.progress import update_run_progress
from llava.train.LoRASculptPlus_Trainer import LoRASculptPlus
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
        logger.info("OrbitStiefel: ORTHO_LAMBDA=0; optimizer=%s", self.optimizer_label)

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
        """Apply masks and write progress after an optimizer step."""
        self._apply_srr_masks(model)
        self._write_progress()

    def create_optimizer(self):
        raise NotImplementedError
