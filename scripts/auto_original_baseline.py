"""Paper-equation LoRASculpt+ control on exactly the OrbitStiefel data.
Prefer two GPUs; recover on one GPU at the same global batch after OOM.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'automation/original-paper-seed42'
LOG = ROOT / 'logs/original-paper'
PYTHON = ROOT / '.venv_new_scheme/bin/python'
TRAIN = ROOT / 'assets/data/iconqa/iconqa_txt-train-10000-seed42-paper.json'
ACTIVE = None
COOLDOWN = {}
MAX_BATCH = {0: 4, 1: 4}
DATASETS = ['okvqa', 'ocrvqa', 'gqa', 'textvqa']
sys.path.insert(0, str(ROOT / 'scripts'))
from auto_paper_pipeline import eval_ready


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def status(stage, **extra):
    value = dict(stage=stage, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 method='LoRASculpt+ paper-equation reproduction', seed=42, ranks=[16, 32, 64], **extra)
    atomic(WORK / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    render()


def memory():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    return {int(a): int(b) for a, b in (line.split(',') for line in out.splitlines()) if int(a) in [0, 1]}


def select_gpus(training=True):
    minimum = 58000 if training else 25000
    while True:
        free = memory()
        eligible = sorted((g for g in free if free[g] >= minimum and COOLDOWN.get(g, 0) <= time.time()), key=lambda g: -free[g])
        if eligible:
            return sorted(eligible[:2])
        status('WAITING_GPU_MEMORY', free_mib=free, minimum_mib=minimum, task='training' if training else 'evaluation')
        time.sleep(20)


def env_for(gpus, rank, output, batch=1, smoke=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('OPTAB_', 'ORBIT_', 'BASELINE_'))}
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), LORASCULPT_ROOT=str(ROOT),
        PYTHONPATH=str(ROOT), PATH=str(PYTHON.parent) + ':' + env['PATH'],
        OPTAB_METHOD='original', OPTAB_RANK=str(rank), OPTAB_SEED='42',
        OPTAB_OUTPUT=str(output), OPTAB_PROGRESS_FILE=str(output / 'baseline-progress.json'),
        OPTAB_PYTHON=str(PYTHON), OPTAB_WORLD_SIZE=str(len(gpus)), OPTAB_TRAIN_JSON=str(TRAIN),
        OPTAB_BATCH_SIZE=str(batch), OPTAB_GRAD_ACCUM=str(16 // (batch * len(gpus))),
        OPTAB_SAVE_STEPS='100', OPTAB_SAVE_TOTAL_LIMIT='2', OPTAB_GRAD_CHECKPOINTING='True',
        STEP_THRESHOLD='1' if smoke else '100', AB_PRESERVE_RATIO='0.1', OMEGA='1.0',
        CMR_LLM_LAMBDA='1e-3', CMR_CONNECTOR_LAMBDA='1e-5', ORTHO_LAMBDA='1e-5',
        TORCHDYNAMO_DISABLE='1', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_SHM_DISABLE='1',
        NCCL_SOCKET_IFNAME='lo', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        HF_HOME=str(ROOT / '.cache/huggingface'), TORCH_HOME=str(ROOT / '.cache/torch'),
        TRITON_CACHE_DIR=str(ROOT / '.cache/triton'), TMPDIR=str(ROOT / '.cache/tmp'),
        BASELINE_RECONFIG_FILE=str(output / '.request-reconfigure'))
    return env


def latest(output):
    complete = []
    for p in output.glob('checkpoint-*'):
        if not p.name[11:].isdigit():
            continue
        required = ['baseline-complete.json', 'trainer_state.json', 'optimizer.pt', 'scheduler.pt']
        weights = any((p / name).is_file() and (p / name).stat().st_size for name in ['adapter_model.bin', 'adapter_model.safetensors'])
        if weights and all((p / name).is_file() and (p / name).stat().st_size for name in required):
            state = json.loads((p / 'baseline-complete.json').read_text())
            if state['step'] != int(p.name[11:]) or json.loads((p / 'trainer_state.json').read_text())['global_step'] != state['step']:
                continue
            rngs = ['rng_state.pth'] if state['world_size'] == 1 else [f'rng_state_{i}.pth' for i in range(state['world_size'])]
            if not all((p / name).is_file() for name in rngs):
                continue
            if state['step'] > (1 if 'smoke' in str(output) else 100) and not (p / 'lorasculpt_plus_srr_masks.pt').is_file():
                continue
            complete.append(p)
    return max(complete, key=lambda p: int(p.name[11:])) if complete else None


def prepare_resume(output, gpus):
    saved = latest(output)
    for p in output.glob('checkpoint-*'):
        if not p.name[11:].isdigit():
            continue
        if saved is None or int(p.name[11:]) > int(saved.name[11:]):
            archive = WORK / 'incomplete' / (output.name + '-' + p.name + '-' + str(time.time_ns()))
            archive.parent.mkdir(parents=True, exist_ok=True)
            p.rename(archive)
    if saved is None:
        return None
    prior = json.loads((saved / 'baseline-complete.json').read_text())
    rng = saved / ('rng_state.pth' if prior['world_size'] == 1 else 'rng_state_0.pth')
    assert rng.is_file(), 'Missing saved random-number state'
    # Preserve rank0 state across topology changes; additional workers start from
    # rank0's saved stream. This is recovery, not bitwise-identical DDP replay.
    names = ['rng_state.pth'] if len(gpus) == 1 else [f'rng_state_{i}.pth' for i in range(len(gpus))]
    for name in names:
        target = saved / name
        if not target.exists():
            shutil.copy2(rng, target)
    if prior['world_size'] != len(gpus):
        atomic(WORK / f'recovery-{output.name}-{time.time_ns()}.json', dict(
            checkpoint=str(saved), previous_world_size=prior['world_size'], new_gpus=gpus,
            global_batch=16, rng_policy='preserve existing streams, initialize absent streams from rank0',
            limitation='Topology changes preserve optimizer/scheduler/SRR/step; sample ordering is not bitwise identical.'))
    return saved


def launch(command, env, logfile, gpus, training=False):
    global ACTIVE
    offset = logfile.stat().st_size if logfile.exists() else 0
    logfile.parent.mkdir(parents=True, exist_ok=True)
    request = Path(env['BASELINE_RECONFIG_FILE'])
    request.unlink(missing_ok=True)
    atomic(WORK / 'active.json', dict(command=[str(x) for x in command], gpus=gpus,
        global_batch=16 if training else None, output=env['OPTAB_OUTPUT'], logfile=str(logfile), started_at=time.time()))
    with logfile.open('a') as out:
        ACTIVE = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        low_since = None
        while ACTIVE.poll() is None:
            time.sleep(5)
            free = memory()
            low = [g for g in gpus if free.get(g, 0) < 1500]
            if low:
                low_since = low_since or time.time()
                if time.time() - low_since > 15:
                    if training:
                        request.touch()
                        # Ask every worker to save at the next optimizer boundary.
                        if time.time() - low_since < 90:
                            continue
                    os.killpg(ACTIVE.pid, signal.SIGTERM)
                    try:
                        ACTIVE.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(ACTIVE.pid, signal.SIGKILL)
                    break
            else:
                low_since = None
            render()
        code = ACTIVE.wait()
        # Reap lingering dataloader/shard workers before a restart.
        try:
            os.killpg(ACTIVE.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        ACTIVE = None
    with logfile.open('rb') as f:
        f.seek(offset)
        tail = f.read().decode(errors='replace')[-20000:]
    return code, tail, bool(low_since)


def recoverable(tail, pressure):
    return pressure or any(x in tail.lower() for x in ['out of memory', 'cuda error', 'nccl', 'sigsegv', 'sigbus', 'exitcode: -9'])


def penalize(gpus):
    free = memory()
    # Keep the freer card usable immediately after a failed two-card run.
    worst = min(gpus, key=lambda g: free.get(g, 0))
    COOLDOWN[worst] = time.time() + 600


def reduce_capacity(gpus, batch):
    if batch > 1:
        for gpu in gpus:
            MAX_BATCH[gpu] = 1
    else:
        penalize(gpus)


def smoke(gpus, batch=4):
    topology = len(gpus)
    marker = WORK / f'smoke-{topology}gpu-b{batch}-gc.json'
    if marker.exists():
        return
    output = ROOT / 'checkpoints' / f'Original-paper-seed42-smoke-{topology}gpu-b{batch}-gc'
    for end in [2, 4]:
        saved = prepare_resume(output, gpus)
        if saved and int(saved.name[11:]) >= end:
            continue
        env = env_for(gpus, 16, output, batch=batch, smoke=True)
        env.update(OPTAB_MAX_STEPS='4', OPTAB_SAVE_STEPS='2', BASELINE_STOP_AFTER_SAVE=str(end))
        status('VALIDATING_SAVE_RESUME', gpus=gpus, target_step=end)
        code, tail, pressure = launch(['bash', 'scripts/train_original_baseline.sh'], env,
            LOG / f'smoke-{topology}gpu-b{batch}-gc.log', gpus, training=True)
        if code:
            if recoverable(tail, pressure):
                reduce_capacity(gpus, batch)
                raise MemoryError('Smoke GPU capacity changed; select available GPU(s)')
            raise RuntimeError(f'Smoke failed ({code}); see {LOG}/smoke-{topology}gpu.log: {tail[-1000:]}')
    saved = latest(output)
    assert saved and int(saved.name[11:]) == 4
    atomic(marker, dict(passed=True, save_step=2, resume_step=4, srr_activated=True,
        gpus=gpus, per_device_batch=batch, weights_reused_for_formal_training=False, step_threshold_for_test=1))


def train(rank):
    output = ROOT / 'checkpoints' / f'Original-paper-seed42-r{rank}'
    marker = WORK / f'trained-r{rank}.json'
    if marker.exists():
        assert digest(output / 'adapter_model.bin') == json.loads(marker.read_text())['adapter_sha256']
        return output
    while True:
        gpus = select_gpus()
        batch = min(MAX_BATCH[g] for g in gpus)
        try:
            smoke(gpus, batch=batch)
        except MemoryError:
            continue
        saved = prepare_resume(output, gpus)
        env = env_for(gpus, rank, output, batch=batch)
        status('TRAINING', rank=rank, gpus=gpus, resume_step=int(saved.name[11:]) if saved else 0,
               global_batch=16, per_device_batch=batch, accumulation=16 // (batch * len(gpus)))
        code, tail, pressure = launch(['bash', 'scripts/train_original_baseline.sh'], env,
            LOG / f'r{rank}-train.log', gpus, training=True)
        if code:
            if code == 75 or recoverable(tail, pressure):
                status('RECOVERING_GPU_MEMORY', rank=rank, previous_gpus=gpus, exit_code=code,
                       checkpoint=str(latest(output)))
                reduce_capacity(gpus, batch)
                time.sleep(5)
                continue
            raise RuntimeError(f'rank{rank} training exited {code}: {tail[-1500:]}')
        state = json.loads((output / 'trainer_state.json').read_text())
        assert state['global_step'] == 1875
        assert (output / 'lorasculpt_plus_config.json').is_file()
        atomic(marker, dict(checkpoint=str(output), global_step=1875, adapter_sha256=digest(output / 'adapter_model.bin')))
        return output


def evaluate(rank, output):
    results = output / 'eval_results_matched'
    for dataset in DATASETS:
        marker = WORK / f'evaluated-r{rank}-{dataset}.json'
        if marker.exists():
            evidence = json.loads(marker.read_text())
            assert digest(evidence['result']) == evidence['result_sha256']
            continue
        while True:
            gpus = select_gpus(training=False)
            env = env_for(gpus, rank, output)
            status('EVALUATING', rank=rank, dataset=dataset, gpus=gpus)
            code, tail, pressure = launch(['bash', f'scripts/v1_5/eval/upstream/eval_{dataset}.sh',
                output, results, results / 'summary.txt'], env, LOG / f'r{rank}-{dataset}.log', gpus)
            if code:
                if recoverable(tail, pressure):
                    penalize(gpus)
                    status('RECOVERING_EVALUATION', rank=rank, dataset=dataset, previous_gpus=gpus)
                    time.sleep(5)
                    continue
                raise RuntimeError(f'rank{rank} {dataset} exited {code}: {tail[-1500:]}')
            score = results / dataset / 'llava-v1.5-7b' / f'result-{dataset}.txt'
            assert score.is_file() and score.stat().st_size
            atomic(marker, dict(result=str(score), result_sha256=digest(score), gpus=gpus))
            break
    atomic(results / 'protocol.json', json.loads((ROOT / 'configs/paper_eval_provenance.json').read_text()))


def render():
    if not WORK.exists():
        return
    rows = []
    for rank in [16, 32, 64]:
        output = ROOT / 'checkpoints' / f'Original-paper-seed42-r{rank}'
        p = output / 'baseline-progress.json'
        step = json.loads(p.read_text()).get('step', 0) if p.exists() else 0
        if (WORK / f'trained-r{rank}.json').exists():
            step = 1875
        scores = {}
        for d in DATASETS:
            p = output / 'eval_results_matched' / d / 'llava-v1.5-7b' / f'result-{d}.txt'
            evidence = WORK / f'evaluated-r{rank}-{d}.json'
            match = re.search(r'Accuracy:\s*([\d.]+)', p.read_text()) if evidence.exists() and p.exists() else None
            scores[d] = float(match.group(1)) if match else None
        rows.append(dict(rank=rank, step=step, scores=scores,
            source_mean=sum(scores.values()) / 4 if all(v is not None for v in scores.values()) else None))
    state = json.loads((WORK / 'status.json').read_text()) if (WORK / 'status.json').exists() else {}
    lines = ['# 原版 LoRASculpt+ 同数据对照', '', f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"阶段：{state.get('stage', 'PREPARING')}；GPU：{state.get('gpus', '自动选择')}。", '',
        '| Rank | 训练步数 | OKVQA | OCRVQA | GQA | TextVQA | 四项平均 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        vals = ['—' if v is None else f'{v:.2f}' for v in row['scores'].values()]
        mean = '—' if row['source_mean'] is None else f"{row['source_mean']:.2f}"
        lines.append(f"| {row['rank']} | {row['step']}/1875 | " + ' | '.join(vals) + f' | {mean} |')
    lines += ['', '与新方案使用同一份 seed42 IconQA 10000条训练文件和同一份四项评测集（OCRVQA 2000条）。',
        '训练全局batch16；优先双卡；显存不足从最新完整断点单卡恢复；不承诺跨拓扑逐位一致。',
        '原版为本地论文公式复现；目标任务IconQA尚不包含在本次四项上游评测中。']
    (WORK / 'live.md').write_text('\n'.join(lines) + '\n')
    atomic(WORK / 'results.json', rows)


def manifest():
    files = ['scripts/auto_original_baseline.py', 'scripts/original_baseline_runtime.py',
        'scripts/train_original_baseline.sh', 'llava/train/LoRASculptPlus_Trainer.py',
        'llava/train/llava_trainer.py', 'llava/train/train.py', 'llava/model/lorasculpt_plus.py',
        'scripts/eval_serial_guard.sh', 'configs/paper_training_provenance.json', 'configs/paper_eval_provenance.json']
    files += [f'scripts/v1_5/eval/upstream/eval_{d}.sh' for d in DATASETS]
    files += ['scripts/score_gqa_accuracy.py', 'llava/eval/eval_ocrvqa.py',
        'llava/eval/eval_okvqa.py', 'llava/eval/eval_textvqa.py', 'llava/eval/model_vqa_loader.py', 'scripts/convert_gqa_for_eval.py',
        'assets/data/eval/okvqa/okvqa_val.jsonl',
        'assets/data/eval/okvqa/mscoco_val2014_annotations.json',
        'assets/data/eval/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'assets/data/eval/ocrvqa/sampled_ocrvqa_test.jsonl',
        'assets/data/eval/gqa/llava_gqa_testdev_balanced.jsonl',
        'assets/data/eval/gqa/data/testdev_balanced_questions.json',
        'assets/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl',
        'assets/data/eval/textvqa/TextVQA_0.5.1_val.json']
    provenance = json.loads((ROOT / 'configs/paper_training_provenance.json').read_text())
    assert digest(TRAIN) == provenance['sha256']
    assert len(json.loads(TRAIN.read_text())) == 10000
    ready, reason = eval_ready()
    assert ready, reason
    return dict(method='Local LoRASculpt+ paper-equation reproduction',
        seed=42, ranks=[16, 32, 64], train_json=str(TRAIN), train_sha256=digest(TRAIN),
        epochs=3, global_batch=16, per_device_batch='4 preferred, 1 on memory fallback', llm_lr=2e-4, connector_lr=2e-5,
        connector_rank=128, alpha=64, dropout=.05, weight_decay=0, warmup_ratio=.03,
        bf16=True, tf32=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False, 'preserve_rng_state': True}, max_length=2048,
        srr_keep=.1, srr_warmup=100, cmr_llm=1e-3, cmr_connector=1e-5, orthogonal=1e-5,
        hardware='Prefer 2 GPUs; fallback to 1; accumulation adjusted to preserve global batch16',
        limitation='GPU count/microbatch and recovery can affect sample ordering and numerical trajectory.',
        files={name: digest(ROOT / name) for name in files},
        data=json.loads((ROOT / 'configs/paper_eval_provenance.json').read_text()))


def stop(signum, frame):
    if ACTIVE is not None:
        try:
            os.killpg(ACTIVE.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    status('STOPPED', signal=signum)
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    lock = (WORK / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    current = manifest()
    if args.check:
        print(json.dumps(current, ensure_ascii=False, indent=2))
        return
    p = WORK / 'run-manifest.json'
    if p.exists():
        assert json.loads(p.read_text()) == current, 'Inputs changed; preserve prior run and start a versioned experiment.'
    else:
        atomic(p, current)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status('STARTING')
    for rank in [16, 32, 64]:
        output = train(rank)
        evaluate(rank, output)
    status('COMPLETED', evaluations=12)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        print('Original baseline pipeline already running.', file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        status('FAILED', error=repr(exc))
        raise
