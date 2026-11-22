"""Two-replica trainer test across pruning, export and checkpoint resume."""
import os
import faulthandler
faulthandler.dump_traceback_later(45, repeat=True)
from pathlib import Path
import sys
import importlib
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent))
source = vars(importlib.import_module('test_orbit_stiefel_integration'))
Args, Data, make = (source[x] for x in ['Args', 'Data', 'make'])
Trainer = source['OrbitStiefelLoRASculptPlus']
exported_state = source['exported_state']
layers = source['layers']

def run():
    torch.set_num_threads(1)
    root = Path(os.environ['ORBIT_DDP_TEST_ROOT'])
    args = Args(output_dir=str(root/'full'), use_cpu=os.environ.get('ORBIT_TEST_CPU') == '1',
                max_steps=4, per_device_train_batch_size=2, gradient_accumulation_steps=2,
                learning_rate=.001, weight_decay=0, save_steps=2, logging_steps=1,
                report_to=[], seed=123, disable_tqdm=True,
                ddp_find_unused_parameters=False, ddp_broadcast_buffers=False)
    model, _ = make()
    trainer = Trainer(model=model, args=args, train_dataset=Data())
    trainer.train(); trainer.save_model()
    assert trainer.state.global_step == 4
    assert all(bool(m.lora_orbit_pruned) for _,m in layers(model) if not m._orbit_connector)
    for name, value in model.state_dict().items():
        if 'lora_' not in name: continue
        expected = value.clone()
        dist.broadcast(expected, src=0)
        torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7, msg=name)
    dist.barrier()
    resumed, _ = make()
    args.output_dir = str(root/'resume')
    other = Trainer(model=resumed, args=args, train_dataset=Data())
    other.train(resume_from_checkpoint=str(root/'full/checkpoint-2'))
    for k, v in exported_state(model).items():
        torch.testing.assert_close(v, exported_state(resumed)[k], rtol=2e-5, atol=2e-6, msg=k)
    if dist.get_rank() == 0: print('DDP_PRUNE_SYNC_EXPORT_RESUME=PASS', flush=True)
    dist.barrier(); dist.destroy_process_group()
if __name__ == '__main__': run()
