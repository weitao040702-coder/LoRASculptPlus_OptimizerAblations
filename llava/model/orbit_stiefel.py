"""Sparse generators, Stiefel directions and independent signed spectral scales.

The masks constrain generators, not the dense QR directions or merged delta.
Inference is exported to ordinary mixed-rank PEFT factors; training state is
saved separately, including masks, rotations and protected pretrained bases.
"""
from __future__ import annotations

import json
import math
import os
import types
from pathlib import Path

import torch
import torch.nn.functional as F

STATE_NAME = 'orbit_stiefel_state.pt'
CONFIG_NAME = 'orbit_stiefel_config.json'
DEFAULTS = dict(warmup_steps=100, keep_ratio=0.1, orbit_iterations=10,
                protection_rank=8, protection_lambda=1e-3,
                connector_sparse_lambda=1e-5, seed=42)


def qr_frame(x):
    # QR backward needs full column rank; sparse candidates are checked below.
    with torch.autocast(device_type=x.device.type, enabled=False):
        q, r = torch.linalg.qr(x.float(), mode='reduced')
        sign = torch.where(r.diagonal() < 0, -1., 1.)
        return q * sign, sign[:, None] * r


def topk_mask(x, ratio):
    if not 0 < ratio <= 1:
        raise ValueError('keep_ratio must be in (0, 1]')
    k = max(1, int(x.numel() * ratio))
    # Stable sort fixes tie-breaking by original index, including exact zeros.
    indices = torch.argsort(x.abs().flatten(), descending=True, stable=True)[:k]
    mask = torch.zeros(x.numel(), dtype=torch.bool, device=x.device)
    mask[indices] = True
    return mask.view_as(x)


def product_error(b, a, sb, sa):
    # FP64 small Gram contraction avoids cancellation in candidate ranking.
    b, a, sb, sa = [x.double() for x in (b, a, sb, sa)]
    value = ((sb.T @ sb) * (sa @ sa.T).T).sum()
    value += ((b.T @ b) * (a @ a.T).T).sum()
    value -= 2 * ((sb.T @ b) * (a @ sa.T).T).sum()
    return value.clamp_min(0)


def balanced_factors(b, a):
    qb, rb = qr_frame(b)
    qa, ra = qr_frame(a.T)
    u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
    return (qb @ u) * s.sqrt(), s.sqrt()[:, None] * (vh @ qa.T)


def full_rank(x):
    singular = torch.linalg.svdvals(x.float())
    return bool(singular[-1] > singular[0].clamp_min(1e-30) * 1e-6)


@torch.no_grad()
def orbit_prune(b, a, ratio, iterations, seed=42):
    """Choose the least-error full-rank candidate, including direct pruning.

    Full-rank generators are required by differentiable QR. The fallback bound
    holds relative to direct pruning only when that candidate is admissible.
    """
    if iterations < 0:
        raise ValueError('orbit_iterations must be nonnegative')
    # This one-time search consists of thin SVDs and FP64 contractions.
    # CPU LAPACK avoids extremely slow tall-matrix cuSOLVER calls on the
    # deployed GPUs; only the selected sparse factors return to the device.
    device = b.device
    b, a = b.float().cpu(), a.float().cpu()
    bc, ac = balanced_factors(b, a)
    best = None
    candidates = []

    def consider(x, y, label):
        nonlocal best
        mb, ma = topk_mask(x, ratio), topk_mask(y, ratio)
        sb, sa = x * mb, y * ma
        admissible = full_rank(sb) and full_rank(sa.T)
        error = float(product_error(b, a, sb, sa))
        candidates.append(dict(label=label, error=error, admissible=admissible))
        if admissible and (best is None or error < best[0]):
            best = (error, sb, sa, mb, ma, label)
        return sb, sa

    consider(b, a, 'original')
    rotation = torch.eye(b.shape[1], device=b.device)
    for i in range(iterations + 1):
        sb, sa = consider(bc @ rotation, rotation.T @ ac, f'orbit-{i}')
        if i < iterations:
            u, _, vh = torch.linalg.svd(bc.T @ sb + ac @ sa.T, full_matrices=False)
            rotation = u @ vh
    # Identity-start alternating search can leave a weak coordinate entirely
    # outside the top-k mask. Try equivalent dense orthogonal coordinates
    # only when all original candidates are rank deficient. This keeps the
    # exact same element budget and never perturbs the pre-pruning product.
    if best is None:
        generator = torch.Generator(device=b.device).manual_seed(seed)
        for i in range(max(1, iterations)):
            rotation, _ = qr_frame(torch.randn(b.shape[1], b.shape[1],
                                               generator=generator, device=b.device))
            consider(bc @ rotation, rotation.T @ ac, f'mixed-fallback-{i}')
    if best is None:
        raise RuntimeError('No full-rank sparse generator candidate; increase keep_ratio')
    _, sb, sa, mb, ma, label = best
    return sb.to(device), sa.to(device), mb.to(device), ma.to(device), dict(selected=label, error=best[0], candidates=candidates)


@torch.no_grad()
def principal_bases(weight, rank, seed=42):
    """Deterministic randomized leading singular subspaces, two power passes.

    This is an approximation to top-k SVD, not an exact full-weight SVD.
    """
    w = weight.detach().float()
    k = min(rank, *w.shape)
    if k < 1:
        raise ValueError('protection_rank must be positive')
    gen = torch.Generator(device=w.device).manual_seed(seed)
    width = min(k + 8, *w.shape)
    omega = torch.randn(w.shape[1], width, generator=gen, device=w.device)
    q, _ = qr_frame(w @ omega)
    for _ in range(2):
        v, _ = qr_frame(w.T @ q)
        q, _ = qr_frame(w @ v)
    u, _, vh = torch.linalg.svd(q.T @ w, full_matrices=False)
    return q @ u[:, :k], vh[:k].T


def directions(layer):
    qb, _ = qr_frame(layer.lora_B['default'].weight * layer.lora_mask_B)
    qa, _ = qr_frame((layer.lora_A['default'].weight * layer.lora_mask_A).T)
    return qb @ layer.lora_rotation_B.float(), qa @ layer.lora_rotation_A.float()


def effective_factors(layer):
    qb, qa = directions(layer)
    return qb * layer.lora_spectral_scale.float(), qa.T


def _forward(layer, x):
    if getattr(layer, 'disable_adapters', False) or getattr(layer, 'merged', False):
        raise RuntimeError('Use exported standard adapter for merging/disabled-adapter inference')
    if getattr(layer, 'active_adapter', 'default') != 'default':
        raise RuntimeError('OrbitStiefel supports only the default adapter')
    weight = layer.weight.T if getattr(layer, 'fan_in_fan_out', False) else layer.weight
    base = F.linear(x, weight, layer.bias)
    b, a = effective_factors(layer)
    # Keep QR in FP32; matmuls follow autocast and return the base output dtype.
    dropped = layer.lora_dropout['default'](x)
    update = F.linear(F.linear(dropped.to(a.dtype), a), b)
    return base + (update * layer.scaling['default']).to(base.dtype)


def _reject_training_merge(layer, *args, **kwargs):
    raise RuntimeError('Export a standard adapter before merge; raw generators are not effective A/B')


def layers(model):
    for name, layer in model.named_modules():
        if hasattr(layer, 'lora_spectral_scale'):
            yield name, layer


def install(model, config=None):
    if config and set(config) - set(DEFAULTS):
        raise ValueError('Unknown method configuration keys')
    config = {**DEFAULTS, **(config or {})}
    if not 0 < config['keep_ratio'] <= 1 or config['orbit_iterations'] < 0:
        raise ValueError('Invalid sparsity/search configuration')
    for key in ['protection_lambda', 'connector_sparse_lambda']:
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError('Regularization coefficients must be finite and nonnegative')
    if config['warmup_steps'] < 0 or config['protection_rank'] < 1:
        raise ValueError('Invalid warmup/protection rank')
    for name, layer in list(model.named_modules()):
        if not hasattr(layer, 'lora_A') or 'default' not in layer.lora_A:
            continue
        if hasattr(layer, 'lora_spectral_scale'):
            continue
        a, b = layer.lora_A['default'].weight, layer.lora_B['default'].weight
        rank = a.shape[0]
        if rank > min(a.shape[1], b.shape[0]):
            raise ValueError(f'Rank exceeds Stiefel dimension: {name}')
        # Install on fresh zero-B adapters only, never silently discard weights.
        if torch.count_nonzero(b.detach()):
            raise ValueError('Install before loading or training nonzero adapters')
        with torch.no_grad():
            qb, _ = qr_frame(torch.randn_like(b, dtype=torch.float32))
            qa, _ = qr_frame(a.T)
            b.copy_(qb); a.copy_(qa.T)
        layer.register_parameter('lora_spectral_scale', torch.nn.Parameter(torch.zeros(rank, device=a.device)))
        for key in ['A', 'B']:
            layer.register_buffer('lora_rotation_' + key, torch.eye(rank, device=a.device))
            value = a if key == 'A' else b
            layer.register_buffer('lora_mask_' + key, torch.ones_like(value, dtype=torch.bool))
        layer.register_buffer('lora_orbit_pruned', torch.tensor(False, device=a.device))
        layer._orbit_connector = 'mm_projector' in name
        # Preserve the historical LLM protection target scope, q/k/v only.
        if not layer._orbit_connector and any(t in name for t in ('q_proj', 'k_proj', 'v_proj')):
            u, v = principal_bases(layer.weight, config['protection_rank'], config['seed'])
            layer.register_buffer('lora_protected_U', u)
            layer.register_buffer('lora_protected_V', v)
        layer.forward = types.MethodType(_forward, layer)
        layer.merge = types.MethodType(_reject_training_merge, layer)
    if not list(layers(model)):
        raise RuntimeError('No LoRA layers found')
    model.orbit_stiefel_config = config
    return model


@torch.no_grad()
def initialize_orbit(model, optimizer, step):
    config = model.orbit_stiefel_config
    if step < config['warmup_steps']:
        return []
    reports = []
    for name, layer in layers(model):
        if layer._orbit_connector or bool(layer.lora_orbit_pruned):
            continue
        print('ORBIT_PRUNE_LAYER', name, 'step', step, flush=True)
        b, a = effective_factors(layer)
        # Balanced split avoids the signed scale residing only in B.
        b, a = balanced_factors(b, a)
        try:
            sb, sa, mb, ma, report = orbit_prune(b, a, config['keep_ratio'], config['orbit_iterations'], config['seed'])
        except RuntimeError as exc:
            directory = os.environ.get('OPTAB_OUTPUT')
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                torch.save(dict(layer=name, b=b.cpu(), a=a.cpu(),
                                scales=layer.lora_spectral_scale.detach().cpu(),
                                generator_b=layer.lora_B['default'].weight.detach().cpu(),
                                generator_a=layer.lora_A['default'].weight.detach().cpu(),
                                config=config), Path(directory)/'orbit_pruning_failure.pt')
            raise RuntimeError(f'{name}: {exc}') from exc
        _, rb = qr_frame(sb); _, ra = qr_frame(sa.T)
        u, s, vh = torch.linalg.svd(rb @ ra.T, full_matrices=False)
        layer.lora_B['default'].weight.copy_(sb)
        layer.lora_A['default'].weight.copy_(sa)
        layer.lora_rotation_B.copy_(u); layer.lora_rotation_A.copy_(vh.T)
        layer.lora_spectral_scale.copy_(s)
        layer.lora_mask_B.copy_(mb); layer.lora_mask_A.copy_(ma)
        layer.lora_orbit_pruned.fill_(True)
        for p in (layer.lora_B['default'].weight, layer.lora_A['default'].weight, layer.lora_spectral_scale):
            if optimizer is not None:
                optimizer.state.pop(p, None)
        reports.append(dict(layer=name, step=step, optimizer_state_reset=True, **report))
    return reports


def regularizers(model):
    config = model.orbit_stiefel_config
    protection = []; sparse = []
    for _, layer in layers(model):
        scaling = layer.scaling['default']
        if layer._orbit_connector:
            sparse.append(scaling * layer.lora_spectral_scale.float().abs().sum())
        elif hasattr(layer, 'lora_protected_U'):
            b, a = effective_factors(layer)
            # ||U.T delta||_F^2 + ||delta V||_F^2 using low-rank contractions.
            left = layer.lora_protected_U.float().T @ b
            right = a @ layer.lora_protected_V.float()
            protection.append(scaling**2 * (((left.T @ left) * (a @ a.T).T).sum()
                                            + ((b.T @ b) * (right @ right.T).T).sum()))
    zero = next(model.parameters()).new_zeros((), dtype=torch.float32)
    return (config['protection_lambda'] * sum(protection, zero),
            config['connector_sparse_lambda'] * sum(sparse, zero), zero)


@torch.no_grad()
def post_step(model, optimizer, connector_lr):
    """Support maintenance plus proximal connector spectral sparsity."""
    stats = dict(layers=0, connector_active_directions=0)
    for _, layer in layers(model):
        stats['layers'] += 1
        for key in ['A', 'B']:
            p = getattr(layer, 'lora_' + key)['default'].weight
            mask = getattr(layer, 'lora_mask_' + key)
            p.mul_(mask)
            if optimizer is not None:
                for value in optimizer.state.get(p, {}).values():
                    if torch.is_tensor(value) and value.shape == p.shape:
                        value.mul_(mask)
        if layer._orbit_connector:
            # Adam diagonal-metric proximal step for lambda * scaling * |s|.
            p = layer.lora_spectral_scale
            group = next(g for g in optimizer.param_groups if any(v is p for v in g['params']))
            state = optimizer.state.get(p, {})
            threshold = group['lr'] * model.orbit_stiefel_config['connector_sparse_lambda'] * layer.scaling['default']
            if 'exp_avg_sq' in state:
                beta2 = group['betas'][1]; t = float(state['step'])
                variance = state.get('max_exp_avg_sq', state['exp_avg_sq']) / (1 - beta2**t)
                threshold = threshold / (variance.sqrt() + group['eps'])
            p.copy_(p.sign() * (p.abs() - threshold).clamp_min(0))
            stats['connector_active_directions'] += int(torch.count_nonzero(p))
    return stats


@torch.no_grad()
def exported_state(model):
    """Materialize ordinary A/B factors, omitting training-only state."""
    state = {}
    for name, layer in layers(model):
        b, a = effective_factors(layer)
        prefix = name + '.' if name else ''
        state[prefix + 'lora_A.default.weight'] = a.to(layer.lora_A['default'].weight.dtype).cpu()
        state[prefix + 'lora_B.default.weight'] = b.to(layer.lora_B['default'].weight.dtype).cpu()
    return state


def save_training_state(model, directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if 'lora_' in k}
    torch.save(state, directory / STATE_NAME)
    (directory / CONFIG_NAME).write_text(json.dumps(model.orbit_stiefel_config, indent=2) + '\n')


def load_training_state(model, directory):
    directory = Path(directory)
    config = json.loads((directory / CONFIG_NAME).read_text())
    if config != model.orbit_stiefel_config:
        raise ValueError('Resume configuration differs from saved method configuration')
    state = torch.load(directory / STATE_NAME, map_location='cpu', weights_only=True)
    expected = {k for k in model.state_dict() if 'lora_' in k}
    if set(state) != expected:
        raise ValueError('Incomplete or incompatible OrbitStiefel training state')
    model.load_state_dict(state, strict=False)
