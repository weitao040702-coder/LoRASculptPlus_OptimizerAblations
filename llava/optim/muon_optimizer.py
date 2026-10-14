"""A small, explicit Muon optimizer for LoRA matrices.

Two-dimensional parameters use momentum followed by Newton-Schulz
orthogonalization. One-dimensional parameters use the AdamW update. The
optimizer keeps both paths in one state_dict so checkpoints remain portable.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch


def newton_schulz_orthogonalize(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal factor of a matrix using Newton-Schulz."""
    if matrix.ndim != 2:
        raise ValueError(f"Muon expects a matrix, got shape={tuple(matrix.shape)}")
    original_dtype = matrix.dtype
    work = matrix.float()
    transposed = work.shape[0] < work.shape[1]
    if transposed:
        work = work.transpose(0, 1)
    norm = torch.linalg.vector_norm(work)
    if not torch.isfinite(norm) or norm <= eps:
        return torch.zeros_like(matrix)
    work = work / norm.clamp_min(eps)
    # These coefficients are the stable fifth-order polynomial used by the
    # standard Muon update. The result is scaled below to match matrix size.
    a, b, c = 3.4445, -4.7750, 2.0315
    # Use the smaller Gram matrix. LoRA factors are usually [d, r] with
    # d >> r, so this keeps the auxiliary matrix at r x r.
    identity = torch.eye(work.shape[1], device=work.device, dtype=work.dtype)
    for _ in range(steps):
        gram = work.transpose(0, 1) @ work
        polynomial = a * identity + b * gram + c * (gram @ gram)
        work = work @ polynomial
    work = work * ((max(work.shape) / min(work.shape)) ** 0.5)
    if transposed:
        work = work.transpose(0, 1)
    return work.to(dtype=original_dtype)


class MuonWithAdamW(torch.optim.Optimizer):
    """Muon for matrices and AdamW for vector parameters."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 2e-4,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid Muon momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid Newton-Schulz steps: {ns_steps}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            nesterov=nesterov,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("MuonWithAdamW does not support sparse gradients")
                state = self.state[parameter]

                if parameter.ndim == 2:
                    buffer = state.get("momentum_buffer")
                    grad_fp32 = grad.float()
                    if buffer is None:
                        buffer = state["momentum_buffer"] = torch.zeros_like(parameter, dtype=torch.float32)
                    buffer.mul_(momentum).add_(grad_fp32)
                    update = grad_fp32.add(buffer, alpha=momentum) if nesterov else buffer
                    update = newton_schulz_orthogonalize(update, steps=ns_steps)
                    if weight_decay:
                        parameter.mul_(1.0 - lr * weight_decay)
                    parameter.add_(update.to(dtype=parameter.dtype), alpha=-lr)
                    state["step"] = state.get("step", 0) + 1
                    continue

                # AdamW fallback for one-dimensional or otherwise non-matrix
                # trainables. This path is expected to be small in LoRA runs.
                step = state.get("step", 0) + 1
                state["step"] = step
                grad_fp32 = grad.float()
                exp_avg = state.setdefault("exp_avg", torch.zeros_like(parameter, dtype=torch.float32))
                exp_avg_sq = state.setdefault("exp_avg_sq", torch.zeros_like(parameter, dtype=torch.float32))
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1.0 - beta2)
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt() / (bias_correction2**0.5)
                denom.add_(eps)
                parameter.addcdiv_(exp_avg.to(dtype=parameter.dtype), denom.to(dtype=parameter.dtype), value=-step_size)

        return loss
