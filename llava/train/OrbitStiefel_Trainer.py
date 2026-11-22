"""New default method: OrbitSparse + spectral protection + sparse Stiefel LoRA."""
import json
import os
from pathlib import Path

from llava.model.orbit_stiefel import (
    exported_state, initialize_orbit, layers, load_training_state,
    post_step, regularizers, save_training_state,
)
from llava.model.lorasculpt_plus import write_lorasculpt_plus_config
from llava.train.LoRASculptPlusOptimizerTrainer import LoRASculptPlusOptimizerTrainer


class OrbitStiefelLoRASculptPlus(LoRASculptPlusOptimizerTrainer):
    optimizer_label = 'AdamW + QR Stiefel directions + learned spectral scales'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.deepspeed or self.args.fsdp:
            raise ValueError('OrbitStiefel supports replicated DDP, not DeepSpeed/FSDP')
        if self.args.weight_decay != 0:
            raise ValueError('OrbitStiefel v1 requires weight_decay=0')
        self._orbit_reports = []

    def create_optimizer(self):
        import torch
        if self.optimizer is None:
            groups = []
            for connector, lr in [(False, self.args.learning_rate),
                                  (True, self.args.mm_projector_lr)]:
                parameters = [p for name, p in self.model.named_parameters()
                              if p.requires_grad and ('mm_projector' in name) == connector]
                if parameters:
                    groups.append(dict(params=parameters, lr=lr, weight_decay=0.0))
            self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
            print('OrbitStiefel: AdamW; old CMR/SRR/OCE disabled; sparse QR generators and learned scales enabled', flush=True)
        return self.optimizer

    def _initialize_srr_masks(self, model):
        import torch
        dist = torch.distributed
        distributed = dist.is_available() and dist.is_initialized()
        pending = self.state.global_step >= self.model.orbit_stiefel_config['warmup_steps'] and any(
            not m._orbit_connector and not bool(m.lora_orbit_pruned) for _, m in layers(self.model))
        if not pending:
            return
        # Choose coordinates/support once, then share exactly the same state.
        reports, error = [], None
        if not distributed or dist.get_rank() == 0:
            try:
                reports = initialize_orbit(self.model, self.optimizer, self.state.global_step)
            except Exception as exc:
                error = repr(exc)
        if distributed:
            message = [reports, error]
            dist.broadcast_object_list(message, src=0)
            reports, error = message
        if error:
            raise RuntimeError('Orbit pruning failed: ' + error)
        if distributed:
            with torch.no_grad():
                for name, tensor in self.model.state_dict().items():
                    if 'lora_' in name:
                        # QR/SVD outputs can be non-contiguous (notably the
                        # transposed A factors). NCCL broadcast requires a
                        # contiguous storage view on every replica.
                        buffer = tensor.contiguous()
                        dist.broadcast(buffer, src=0)
                        if buffer.data_ptr() != tensor.data_ptr():
                            tensor.copy_(buffer)
            # Reset the same parameter states on every replica.
            for _, layer in layers(self.model):
                if layer._orbit_connector:
                    continue
                for p in (layer.lora_A['default'].weight, layer.lora_B['default'].weight,
                          layer.lora_spectral_scale):
                    self.optimizer.state.pop(p, None)
        self._orbit_reports.extend(reports)
        if reports and self.is_world_process_zero():
            path = Path(self.args.output_dir) / 'orbit_pruning_report.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._orbit_reports, indent=2) + '\n')

    # Masks live in model buffers and enter QR before the forward computation.
    # Their gradients are explicitly masked below; no legacy SRR hooks are used.
    def _apply_srr_masks(self, model):
        pass

    def _register_srr_gradient_hooks(self, model):
        pass

    def _mask_srr_gradients(self, model):
        for _, layer in layers(self.model):
            for key in ['A', 'B']:
                p = getattr(layer, 'lora_' + key)['default'].weight
                if p.grad is not None:
                    p.grad.mul_(getattr(layer, 'lora_mask_' + key))

    def compute_custom_regularizers(self, model):
        protected, connector_l1, zero = regularizers(self.model)
        self._orbit_connector_l1 = float(connector_l1.detach())
        # Connector L1 is optimized by post-step proximal shrinkage, not also
        # differentiated here (which would apply the same penalty twice).
        return protected, zero, zero

    def _post_optimizer_step(self, model):
        stats = post_step(self.model, self.optimizer, self.args.mm_projector_lr)
        if not self.is_world_process_zero():
            return
        self._write_progress(cmr=False, srr=False, orbit_sparse=True,
                             spectral_protection=True, stiefel_parameterization=True,
                             connector_sparse_penalty=getattr(self, '_orbit_connector_l1', None),
                             new_method_stats=stats)

    def _save(self, output_dir=None, state_dict=None):
        if not self.is_world_process_zero():
            return
        output_dir = output_dir or self.args.output_dir
        # Export ordinary PEFT A/B for the existing mixed-rank evaluation loader.
        import torch
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, state_dict=exported_state(self.model),
                                   safe_serialization=False)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_args.bin'))
        save_training_state(self.model, output_dir)
        write_lorasculpt_plus_config(output_dir, self.model.orbit_layout_metadata)
        self.model.config.save_pretrained(output_dir)
        Path(output_dir, 'orbit_pruning_report.json').write_text(json.dumps(self._orbit_reports, indent=2) + '\n')
        import torch
        torch.save({}, os.path.join(output_dir, 'non_lora_trainables.bin'))

    def _load_rng_state(self, checkpoint):
        # Transformers 4.37 RNG files include NumPy arrays. PyTorch >=2.6
        # defaults to weights_only=True; allow only their required NumPy types.
        import torch
        import numpy as np
        if hasattr(torch.serialization, 'safe_globals'):
            allowed = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                       type(np.dtype(np.uint32))]
            with torch.serialization.safe_globals(allowed):
                return super()._load_rng_state(checkpoint)
        return super()._load_rng_state(checkpoint)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        load_training_state(self.model, resume_from_checkpoint)
        report = Path(resume_from_checkpoint) / 'orbit_pruning_report.json'
        if report.exists():
            self._orbit_reports = json.loads(report.read_text())

    def _load_srr_masks(self, resume_from_checkpoint):
        # The legacy inner loop may call checkpoint loading on a second path.
        if resume_from_checkpoint:
            load_training_state(self.model, resume_from_checkpoint)
