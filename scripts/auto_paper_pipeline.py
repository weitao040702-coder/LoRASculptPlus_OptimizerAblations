"""One seed, ranks 16/32/64: wait for data, DDP smoke/train, strict eval.

Record whether evaluation follows a reconstructed protocol or identical author IDs.
Run under tmux; status and immutable run manifest live under automation/.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
WORK = Path(os.environ.get('PAPER_PIPELINE_WORK', ROOT / 'automation/orbit-paper-seed42'))
PYTHON = ROOT / '.venv_new_scheme/bin/python'
PROTOCOL = ROOT / 'configs/paper_training_protocol.json'
PROVENANCE = ROOT / 'configs/paper_eval_provenance.json'
TRAIN = ROOT / 'assets/data/iconqa/iconqa_txt-train-10000-seed42-paper.json'
ICON = ROOT / 'assets/data/iconqa/raw/data/iconqa_data'
LOG = Path(os.environ.get('PAPER_PIPELINE_LOG', ROOT / 'logs/auto-paper'))
ACTIVE = None
GPU_IDS = [int(x) for x in os.environ.get('PAPER_PIPELINE_GPUS', '0').split(',') if x]
WORLD_SIZE = len(GPU_IDS)
EVAL_FOLDER = os.environ.get('PAPER_PIPELINE_EVAL_FOLDER', 'eval_results_paper_gpu0')
TAG_PREFIX = os.environ.get('PAPER_PIPELINE_TAG_PREFIX', 'OrbitStiefel-paper-seed42')
METHOD_CONFIG = Path(os.environ.get('PAPER_PIPELINE_METHOD_CONFIG', ROOT / 'configs/orbit_stiefel.json'))
METHOD_LABEL = os.environ.get('PAPER_PIPELINE_METHOD_LABEL', 'OrbitStiefel')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def status(stage, **extra):
    value = dict(stage=stage, updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 seed=42, ranks=[16, 32, 64], gpus=GPU_IDS, **extra)
    atomic(WORK / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith('OPTAB_')}
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, GPU_IDS)), LORASCULPT_ROOT=str(ROOT),
               PYTHONPATH=str(ROOT), PATH=str(PYTHON.parent) + ':' + env['PATH'],
               OPTAB_PYTHON=str(PYTHON), OPTAB_WORLD_SIZE=str(WORLD_SIZE), OPTAB_SEED='42',
               OPTAB_TRAIN_JSON=str(TRAIN),
               OPTAB_BATCH_SIZE='4', OPTAB_GRAD_ACCUM=str(16 // (4 * WORLD_SIZE)), OPTAB_SAVE_STEPS='100', OPTAB_GRAD_CHECKPOINTING='False', TORCHDYNAMO_DISABLE='1',
               ORBIT_STIEFEL_CONFIG=str(METHOD_CONFIG),
               NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_SHM_DISABLE='1',
               NCCL_SOCKET_IFNAME='lo', OPTAB_DIAGNOSTICS='0',
               OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def run(command, logfile, env=None):
    global ACTIVE
    with logfile.open('a') as out:
        print('RUN', command, 'log=', str(logfile), flush=True)
        ACTIVE = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
                                  stdout=out, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
        code = ACTIVE.wait()
        ACTIVE = None
    if code:
        raise RuntimeError(f'Command exited {code}; see {logfile}')


def process_running(fragment):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            if any(x.endswith(fragment.encode()) for x in argv):
                return True
        except OSError:
            pass
    return False


def prepare_icon(protocol):
    archive = ROOT / 'assets/data/upstream/downloads/iconqa_data.zip'
    if not archive.with_name(archive.name + '.complete').is_file():
        if not process_running('download_iconqa_ranges.py'):
            status('RESUMING_ICONQA')
            run([PYTHON, '-u', ROOT/'scripts/download_iconqa_ranges.py'], LOG/'iconqa-download.log')
        return False
    marker = WORK / 'iconqa-extracted.json'
    if not marker.exists():
        # Concurrent unzip -o is avoided; the recovery worker owns extraction
        # until it exits or has explicitly logged this archive's completion.
        recovery = ROOT/'logs/public-assets-recovery.out'
        recovery_log = recovery.read_text() if recovery.exists() else ''
        if process_running('resume_interrupted_datasets.py') and (
            'RECOVERED iconqa_data.zip' not in recovery_log and 'RECOVERY_FAILED' not in recovery_log):
            return False
        status('EXTRACTING_ICONQA')
        ICON.parent.mkdir(parents=True, exist_ok=True)
        run(['unzip', '-q', '-o', archive, '-d', ICON.parent], LOG/'iconqa-extract.log')
        atomic(marker, dict(archive_size=archive.stat().st_size))
    from prepare_iconqa_paper import build_records
    from PIL import Image
    records = build_records(ICON, json.loads((ICON/'pid_splits.json').read_text()),
                            json.loads((ICON/'problems.json').read_text()),
                            protocol['train_samples'], protocol['sampling_seed'])
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    assert hashlib.sha256(payload.encode()).hexdigest() == protocol['train_sha256'], 'IconQA fingerprint mismatch'
    assert len({x['question_id'] for x in records}) == 10000
    for row in records:
        with Image.open(ICON/row['image']) as image:
            image.verify()
    TRAIN.parent.mkdir(parents=True, exist_ok=True)
    if TRAIN.exists():
        assert digest(TRAIN) == protocol['train_sha256'], 'Existing training file differs'
    else:
        TRAIN.write_text(payload)
    return True


def wait_gpus():
    required_mib = int(os.environ.get('PAPER_PIPELINE_MIN_FREE_MIB', '48000'))
    while True:
        text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                        '--format=csv,noheader,nounits'], text=True)
        free = {int(a): int(b) for a, b in (line.split(',') for line in text.splitlines())}
        if all(free.get(i, 0) >= required_mib for i in GPU_IDS):
            return
        status('WAITING_GPU_MEMORY', free_mib=free, required_mib_per_gpu=required_mib)
        time.sleep(30)


def manifest(protocol):
    files = ['scripts/train_new_scheme.sh', 'scripts/train_optimizer_config.sh',
             'llava/train/train.py', 'llava/train/OrbitStiefel_Trainer.py',
             'llava/model/orbit_stiefel.py', str(METHOD_CONFIG.relative_to(ROOT)),
             'configs/paper_training_protocol.json', 'scripts/auto_paper_pipeline.py',
             'scripts/prepare_iconqa_paper.py', 'configs/paper_training_provenance.json']
    return dict(protocol=protocol, execution=dict(gpus=GPU_IDS, world_size=WORLD_SIZE, per_device_batch=4, accumulation=16//(4*WORLD_SIZE), note='two-GPU replicated DDP training and sharded evaluation'), files={p: digest(ROOT/p) for p in files},
                training_sha256=digest(TRAIN),
                ddp_validation='logs/new-scheme-ddp-gpu-validation-socket.txt',
                gpu_info=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                                  '--format=csv'], text=True))


def train_rank(rank, smoke_only=False):
    tag = f'{TAG_PREFIX}-r{rank}'
    checkpoint = ROOT/'checkpoints'/tag
    marker = WORK/f'trained-r{rank}.json'
    if marker.exists():
        assert (checkpoint/'adapter_model.bin').is_file(), 'Completed checkpoint missing'
        return checkpoint
    env = environment()
    env.update(OPTAB_RANK=str(rank), OPTAB_TAG=tag)
    # Fresh four-step 7B smoke exercises the actual pruning transition on each
    # rank. Its parameters/config/checkpoints never initialize the formal run.
    smoke = WORK/f'smoke-gpu0-r{rank}.json'
    smoke_tag = tag+'-smoke-gpu0'
    if smoke.exists():
        assert json.loads(smoke.read_text()).get('training_sha256') == digest(TRAIN), 'Smoke training data changed'
    if not smoke.exists():
        wait_gpus()
        smoke_config = json.loads(METHOD_CONFIG.read_text())
        smoke_config['warmup_steps'] = 3
        atomic(WORK/f'smoke-config-r{rank}.json', smoke_config)
        smoke_env = dict(env, OPTAB_TAG=smoke_tag, OPTAB_MAX_STEPS='4',
                         ORBIT_STIEFEL_CONFIG=str(WORK/f'smoke-config-r{rank}.json'))
        status('GPU_SMOKE', rank=rank)
        run(['bash', 'scripts/train_new_scheme.sh'], LOG/f'r{rank}-smoke.log', smoke_env)
        resume_env = dict(smoke_env, OPTAB_RESUME='1', OPTAB_MAX_STEPS='5')
        run(['bash', 'scripts/train_new_scheme.sh'], LOG/f'r{rank}-smoke.log', resume_env)
        smoke_state = json.loads((ROOT/'checkpoints'/smoke_tag/'trainer_state.json').read_text())
        assert smoke_state['global_step'] == 5, 'GPU0 checkpoint resume failed'
        atomic(smoke, dict(passed=True, checkpoint_resume_passed=True, formal_initialization_reused=False,
                          training_sha256=digest(TRAIN)))
    if smoke_only:
        return checkpoint
    wait_gpus()
    status('TRAINING', rank=rank, expected_optimizer_steps=1875)
    for attempt in range(3):
        saved = []
        for path in checkpoint.glob('checkpoint-*'):
            required = ['orbit_stiefel_state.pt', 'orbit_stiefel_config.json', 'trainer_state.json',
                        'optimizer.pt', 'scheduler.pt'] + (['rng_state.pth'] if WORLD_SIZE == 1 else [f'rng_state_{i}.pth' for i in range(WORLD_SIZE)])
            if all((path/name).is_file() for name in required):
                saved.append(path)
            else:
                path.rename(WORK/(f'incomplete-r{rank}-'+path.name+'-'+str(time.time_ns())))
        if saved:
            env['OPTAB_RESUME'] = '1'
        elif checkpoint.exists():
            raise RuntimeError('Existing output without a complete checkpoint; preserve and explicitly restart')
        try:
            run(['bash', 'scripts/train_new_scheme.sh'], LOG/f'r{rank}-train.log', env)
            break
        except RuntimeError:
            if attempt == 2 or not list(checkpoint.glob('checkpoint-*')):
                raise
            status('RETRYING_FROM_CHECKPOINT', rank=rank, attempt=attempt+1)
            time.sleep(10)
    trainer = json.loads((checkpoint/'trainer_state.json').read_text())
    assert trainer['global_step'] == 1875 and abs(trainer['epoch']-3) < 1e-6
    assert (checkpoint/'adapter_model.bin').is_file()
    assert (checkpoint/'orbit_pruning_report.json').is_file()
    atomic(marker, dict(checkpoint=str(checkpoint), global_step=1875,
                        adapter_sha256=digest(checkpoint/'adapter_model.bin')))
    return checkpoint


def eval_ready():
    provenance = json.loads(PROVENANCE.read_text())
    evidence = provenance['ocrvqa']
    ocr = ROOT/'assets/data/eval/ocrvqa/sampled_ocrvqa_test.jsonl'
    if not evidence['verified'] or not evidence['source'] or not evidence['sha256']:
        return False, 'OCRVQA test protocol/source not verified; arbitrary subsets are blocked'
    if not ocr.exists() or digest(ocr) != evidence['sha256']:
        return False, 'OCRVQA fixed test subset is missing or differs from verified fingerprint'
    if sum(bool(x.strip()) for x in ocr.read_text().splitlines()) != evidence['count']:
        return False, 'OCRVQA fixed test subset count mismatch'
    gqa = ROOT/'assets/data/eval/gqa/data'
    fixed = {
        'okvqa/okvqa_val.jsonl': ('daa1e91a0934a0f7ef4f6aa5790c93a26b8868e4eead987e1012abed9def53e7', 5046),
        'gqa/llava_gqa_testdev_balanced.jsonl': ('515aab3142889269480c71212d3b5b8118704d8da6637d767a0cb5bdf80a3d35', 12578),
        'textvqa/llava_textvqa_val_v051_ocr.jsonl': ('9f60db6fb89e369263ed676dc26b48d00d4a462e73f0930c09d23394e9dcf3a3', 5000),
    }
    for name, (sha, count) in fixed.items():
        path = ROOT/'assets/data/eval'/name
        if not path.exists() or digest(path) != sha or len(path.read_text().splitlines()) != count:
            return False, 'Fixed evaluation questions missing or changed: '+name
    # Paper §4.1 reports Accuracy, not GQA validity/plausibility/grounding.
    # Those auxiliary metrics require scene graphs/choices; Accuracy does not.
    questions = gqa/'testdev_balanced_questions.json'
    if not questions.is_file():
        return False, 'GQA official testdev-balanced answer metadata missing'
    metadata = json.loads(questions.read_text())
    ids = {json.loads(x)['question_id'] for x in (ROOT/'assets/data/eval/gqa/llava_gqa_testdev_balanced.jsonl').read_text().splitlines()}
    if set(metadata) != ids or not all(q['isBalanced'] for q in metadata.values()):
        return False, 'GQA official answer/question ID mismatch'
    result = subprocess.run([str(PYTHON), 'scripts/verify_upstream_eval.py',
                             '--eval-root', str(ROOT/'assets/data/eval')], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (LOG/'eval-data-validation.log').write_text(result.stdout)
    return result.returncode == 0, ('Evaluation data verified' if result.returncode == 0 else 'Evaluation images/annotations not ready')


def read_training_provenance():
    path = ROOT/'configs/paper_training_provenance.json'
    if not path.exists():
        return False
    evidence = json.loads(path.read_text())
    return bool(evidence.get('verified') and evidence.get('source')
                and evidence.get('sha256') == digest(TRAIN))


def summarize(checkpoints):
    from summarize_eval_results import read_score
    lines = ['| Method | Seed | LLM rank | OKVQA | OCRVQA | GQA | TextVQA | Source mean |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    rows = []
    for rank, checkpoint in checkpoints.items():
        scores = {}
        for name in ['okvqa', 'ocrvqa', 'gqa', 'textvqa']:
            path = checkpoint/EVAL_FOLDER/name/'llava-v1.5-7b'/f'result-{name}.txt'
            score, _ = read_score(path)
            if score is None:
                raise RuntimeError('Missing/unparseable score: '+str(path))
            scores[name] = score
        mean = sum(scores.values())/4
        rows.append(dict(method=METHOD_LABEL, seed=42, rank=rank, scores=scores, source_mean=mean))
        values = ' | '.join(f'{v:.2f}' for v in scores.values())
        lines.append(f'| {METHOD_LABEL} | 42 | {rank} | {values} | {mean:.2f} |')
    lines += ['', 'Only the four upstream metrics are evaluated. IconQA Target and paper Avg are not measured.',
              'Training follows paper settings and author-cited IconQA formatting, with recorded local seed/sample IDs; the author training subset is not claimed to be identical.']
    lines += ['', 'OCRVQA uses a recorded seed42 reconstruction of the author random-2000 test protocol; original author subset IDs are unavailable. Scores are not an identical-subset reproduction of the paper.']
    (WORK/'results.md').write_text('\n'.join(lines)+'\n')
    atomic(WORK/'results.json', rows)


def evaluate(rank, checkpoint):
    results = checkpoint/EVAL_FOLDER
    results.mkdir(exist_ok=True)
    summary = results/'summary.txt'
    env = environment()
    for dataset in ['okvqa', 'ocrvqa', 'gqa', 'textvqa']:
        marker = WORK/f'evaluated-r{rank}-{dataset}.json'
        if marker.exists():
            continue
        ready, reason = eval_ready()
        if not ready:
            raise RuntimeError('Strict evaluation gate changed: '+reason)
        wait_gpus()
        status('EVALUATING', rank=rank, dataset=dataset, shards=WORLD_SIZE)
        run(['bash', f'scripts/v1_5/eval/upstream/eval_{dataset}.sh', checkpoint, results, summary],
            LOG/f'r{rank}-{dataset}.log', env)
        score = results/dataset/'llava-v1.5-7b'/f'result-{dataset}.txt'
        assert score.is_file() and score.stat().st_size
        atomic(marker, dict(result=str(score), result_sha256=digest(score)))
    atomic(results/'protocol.json', json.loads(PROVENANCE.read_text()))


def main(check=False):
    WORK.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    lock = (WORK/'pipeline.lock').open('w')
    if not check:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol['ranks'] == [16, 32, 64] and protocol['seed'] == 42
    assert protocol['per_device_train_batch_size'] * protocol['gradient_accumulation_steps'] * 2 == 16
    assert protocol['sampling_seed'] == 42
    validation = ROOT/'logs/new-scheme-ddp-gpu-validation-socket.txt'
    assert 'DDP_PRUNE_SYNC_EXPORT_RESUME=PASS' in validation.read_text(), 'Actual two-GPU test must pass first'
    if check:
        print('CONFIG_AND_TWO_GPU_VALIDATION=PASS; eval gate:', eval_ready(), flush=True)
        return
    while not prepare_icon(protocol):
        status('WAITING_ICONQA')
        time.sleep(30)
    current = manifest(protocol)
    path = WORK/'run-manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != current:
            if os.environ.get('PAPER_PIPELINE_ALLOW_MANIFEST_UPDATE') != '1':
                raise AssertionError('Run inputs/code changed; create a new version directory')
            path.rename(WORK/'run-manifest-pre-resume.json')
            atomic(path, current)
    else:
        atomic(path, current)
    for rank in protocol['ranks']:
        if not (WORK/f'trained-r{rank}.json').exists():
            train_rank(rank, smoke_only=True)
    assert read_training_provenance(), 'Paper training protocol/author template reconstruction must be verified'
    checkpoints = {rank: train_rank(rank) for rank in protocol['ranks']}
    while True:
        ready, reason = eval_ready()
        if ready:
            break
        status('TRAINED_WAITING_STRICT_EVAL', trained_ranks=protocol['ranks'], reason=reason)
        time.sleep(30)
    for rank, checkpoint in checkpoints.items():
        evaluate(rank, checkpoint)
    summarize(checkpoints)
    status('COMPLETED', trained_ranks=protocol['ranks'], evaluations=12)


def terminate(signum, frame):
    if ACTIVE is not None:
        os.killpg(ACTIVE.pid, signal.SIGTERM)
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    options = parser.parse_args()
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        main(options.check)
    except BlockingIOError:
        print('Pipeline already running; duplicate launch refused.', file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        if not options.check:
            status('FAILED', error=repr(exc))
        raise
