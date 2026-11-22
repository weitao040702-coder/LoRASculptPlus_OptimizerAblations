import json
from pathlib import Path

from scripts import auto_original_baseline as job


def checkpoint(root, step, world=1, complete=True, masks=True):
    p = root / f'checkpoint-{step}'
    p.mkdir(parents=True)
    for name in ['adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt']:
        (p / name).write_bytes(b'nonempty')
    (p / 'trainer_state.json').write_text(json.dumps({'global_step': step}))
    if complete:
        (p / 'baseline-complete.json').write_text(json.dumps({'step': step, 'world_size': world}))
    if masks:
        (p / 'lorasculpt_plus_srr_masks.pt').write_bytes(b'masks')
    for name in (['rng_state.pth'] if world == 1 else ['rng_state_0.pth', 'rng_state_1.pth']):
        (p / name).write_bytes(b'random-state')
    return p


def test_latest_rejects_partial_save_and_accepts_safetensors(tmp_path):
    good = checkpoint(tmp_path, 100)
    checkpoint(tmp_path, 200, complete=False)
    assert job.latest(tmp_path) == good


def test_latest_rejects_missing_masks_or_rng(tmp_path):
    good = checkpoint(tmp_path, 100)
    checkpoint(tmp_path, 200, masks=False)
    p = checkpoint(tmp_path, 300)
    (p / 'rng_state.pth').unlink()
    assert job.latest(tmp_path) == good


def test_single_to_two_gpu_resume_preserves_prior_and_archives_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(job, 'WORK', tmp_path / 'state')
    output = tmp_path / 'formal'
    good = checkpoint(output, 100)
    checkpoint(output, 200, complete=False)
    assert job.prepare_resume(output, [0, 1]) == good
    assert not (output / 'checkpoint-200').exists()
    assert (good / 'rng_state_1.pth').read_bytes() == (good / 'rng_state.pth').read_bytes()
    assert list((tmp_path / 'state/incomplete').iterdir())
    assert list((tmp_path / 'state').glob('recovery-*.json'))


def test_two_to_single_gpu_resume_preserves_saved_rank0(tmp_path, monkeypatch):
    monkeypatch.setattr(job, 'WORK', tmp_path / 'state')
    good = checkpoint(tmp_path / 'formal', 100, world=2)
    job.prepare_resume(good.parent, [1])
    assert (good / 'rng_state.pth').read_bytes() == (good / 'rng_state_0.pth').read_bytes()


def test_gpu_selection_uses_one_when_other_full_and_prefers_two(monkeypatch):
    monkeypatch.setattr(job, 'COOLDOWN', {})
    monkeypatch.setattr(job, 'memory', lambda: {0: 70000, 1: 2000})
    assert job.select_gpus() == [0]
    monkeypatch.setattr(job, 'memory', lambda: {0: 70000, 1: 65000})
    assert job.select_gpus() == [0, 1]
    job.penalize([0, 1])
    assert job.select_gpus() == [0]


def test_global_batch_remains_16_on_each_topology(tmp_path):
    for gpus in [[0], [1], [0, 1]]:
        env = job.env_for(gpus, 32, tmp_path)
        assert int(env['OPTAB_BATCH_SIZE']) * int(env['OPTAB_GRAD_ACCUM']) * len(gpus) == 16
        assert env['OPTAB_TRAIN_JSON'] == str(job.TRAIN)
        assert env['ORTHO_LAMBDA'] == '1e-5'
        assert env['CMR_LLM_LAMBDA'] == '1e-3'
