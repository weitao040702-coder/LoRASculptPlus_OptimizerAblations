"""Instrumentation for the local paper-equation LoRASculpt+ baseline.

The objective and SRR order are inherited unchanged. This module adds seed
initialization, rank-consistent masks, progress and durable checkpoint markers.
"""
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from transformers import TrainerCallback, set_seed
from llava.train.LoRASculptPlus_Trainer import LoRASculptPlus


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


class Progress(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        request = os.environ.get('BASELINE_RECONFIG_FILE')
        requested = bool(request and Path(request).exists())
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor(int(requested), device=args.device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            requested = bool(flag.item())
        if requested:
            control.should_save = True
        if state.is_world_process_zero:
            atomic(Path(args.output_dir) / 'baseline-progress.json', dict(
                stage='TRAINING', step=state.global_step, total_steps=state.max_steps,
                epoch=state.epoch, updated_at=time.time(),
                gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gradient_checkpointing=args.gradient_checkpointing,
                allocated_mib=torch.cuda.memory_allocated() // 1048576 if torch.cuda.is_available() else 0,
                reserved_mib=torch.cuda.memory_reserved() // 1048576 if torch.cuda.is_available() else 0,
                peak_allocated_mib=torch.cuda.max_memory_allocated() // 1048576 if torch.cuda.is_available() else 0,
                peak_reserved_mib=torch.cuda.max_memory_reserved() // 1048576 if torch.cuda.is_available() else 0,
                global_batch=args.train_batch_size * args.gradient_accumulation_steps * args.world_size))


class OriginalBaselineTrainer(LoRASculptPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(Progress())

    def _load_rng_state(self, checkpoint):
        # The experiment created these local RNG files. Load their NumPy/Python
        # state explicitly; old Transformers predates PyTorch's new default.
        if checkpoint is None:
            return
        import random
        import numpy as np
        filename = f'rng_state_{self.args.process_index}.pth' if self.args.world_size > 1 else 'rng_state.pth'
        state = torch.load(Path(checkpoint) / filename, map_location='cpu', weights_only=False)
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.random.set_rng_state(state['cpu'])
        if torch.cuda.is_available() and 'cuda' in state:
            cuda = state['cuda']
            if isinstance(cuda, (list, tuple)):
                torch.cuda.random.set_rng_state(cuda[min(torch.cuda.current_device(), len(cuda) - 1)])
            else:
                torch.cuda.random.set_rng_state(cuda)

    def _initialize_srr_masks(self, model):
        had_masks = bool(self.AB_masks)
        super()._initialize_srr_masks(model)
        if not had_masks and self.AB_masks and dist.is_available() and dist.is_initialized():
            for name in sorted(self.AB_masks):
                mask = self.AB_masks[name].to(self.args.device)
                dist.broadcast(mask, src=0)
                self.AB_masks[name] = mask.cpu()

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.is_world_process_zero():
            directory = Path(self._get_output_dir(trial)) / f'checkpoint-{self.state.global_step}'
            atomic(directory / 'baseline-complete.json', dict(
                step=self.state.global_step, world_size=self.args.world_size,
                per_device_batch=self.args.per_device_train_batch_size,
                accumulation=self.args.gradient_accumulation_steps,
                srr_masks=len(self.AB_masks), created_at=time.time()))
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        request = os.environ.get('BASELINE_RECONFIG_FILE')
        if request and Path(request).exists():
            raise SystemExit(75)
        stop = int(os.environ.get('BASELINE_STOP_AFTER_SAVE', '0'))
        if stop and self.state.global_step >= stop:
            raise SystemExit(0)


def main():
    # train.py injects adapters before Trainer is initialized.
    set_seed(int(os.environ.get('OPTAB_SEED', '42')))
    import llava.train as trainers
    from llava.train.train import train
    trainers.names['LoRASculptPlus'] = OriginalBaselineTrainer
    print('ORIGINAL_BASELINE: inherited CMR(LLM,connector) + OCE + SRR; no manifold/OrbitSparse', flush=True)
    train()


if __name__ == '__main__':
    main()
