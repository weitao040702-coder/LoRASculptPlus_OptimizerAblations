"""Two-GPU adapter generation check; never produces formal accuracy scores.

Uses two original questions per available upstream dataset. Missing OCRVQA
author questions remain explicitly blocked rather than synthesized.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = ROOT / 'automation/orbit-paper-seed42/technical-eval-smoke'


def worker(checkpoint, shard):
    import torch
    from llava.eval import model_vqa_loader as loader
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.constants import IMAGE_TOKEN_INDEX
    disable_torch_init()
    loader.args = argparse.Namespace(conv_mode='vicuna_v1')
    tokenizer, model, processor, _ = load_pretrained_model(
        str(checkpoint), str(ROOT/'assets/models/llava-v1.5-7b'),
        get_model_name_from_path(str(checkpoint)), device_map=None,
        low_cpu_mem_usage=False)
    model = model.to('cuda').eval()
    cases = [
        ('okvqa', 'okvqa/okvqa_val.jsonl', 'okvqa/val2014'),
        ('ocrvqa', 'ocrvqa/sampled_ocrvqa_test.jsonl', 'ocrvqa/sampled_images'),
        ('gqa', 'gqa/llava_gqa_testdev_balanced.jsonl', 'gqa/data/images'),
        ('textvqa', 'textvqa/llava_textvqa_val_v051_ocr.jsonl', 'textvqa/train_images'),
    ]
    result = dict(gpu=shard, checkpoint=str(checkpoint), formal_scores=False, datasets={})
    for name, question_file, images in cases:
        path = ROOT/'assets/data/eval'/question_file
        if not path.exists():
            result['datasets'][name] = dict(status='BLOCKED_MISSING_AUTHOR_QUESTIONS')
            continue
        with path.open() as handle:
            rows = [json.loads(next(handle)) for _ in range(2)]
        row = rows[shard]
        data = loader.CustomDataset([row], str(ROOT/'assets/data/eval'/images),
                                    tokenizer, processor, model.config)
        ids, image, image_size = data[0]
        with torch.inference_mode():
            answer = model.generate(ids.unsqueeze(0).to('cuda'),
                images=image.unsqueeze(0).to(device='cuda', dtype=torch.float16),
                image_sizes=[image_size], do_sample=False, max_new_tokens=32,
                use_cache=True)
        answer = tokenizer.batch_decode(answer, skip_special_tokens=True)[0].strip()
        if not answer:
            raise RuntimeError('Empty generated answer: '+name)
        result['datasets'][name] = dict(status='PASS', question_id=row['question_id'], answer=answer)
        print(name, row['question_id'], repr(answer), flush=True)
    (WORK/f'gpu{shard}.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        worker(args.checkpoint.resolve(), args.worker)
    else:
        processes = []
        handles = []
        for shard in [0, 1]:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
            handle = (WORK/f'gpu{shard}.log').open('w'); handles.append(handle)
            processes.append(subprocess.Popen([sys.executable, __file__, '--checkpoint',
                str(args.checkpoint.resolve()), '--worker', str(shard)], env=env,
                stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT))
        codes = [p.wait() for p in processes]
        for handle in handles:
            handle.close()
        if any(codes):
            raise SystemExit('Technical generation check failed: '+str(codes))
        print('AVAILABLE_DATASET_TWO_GPU_GENERATION=PASS; see per-GPU JSON for blocked datasets')
