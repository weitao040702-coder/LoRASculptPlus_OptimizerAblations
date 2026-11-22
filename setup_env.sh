#!/usr/bin/env bash
# LoRASculpt runtime environment for this checkout.
# Usage: source setup_env.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LORASCULPT_ROOT="$PROJECT_ROOT"
LORASCULPT_ENV="${LORASCULPT_ENV:-$PROJECT_ROOT/.venv_new_scheme}"
if [[ ! -x "$LORASCULPT_ENV/bin/python" ]]; then
  echo "Missing Python environment: $LORASCULPT_ENV/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi
export PATH="$LORASCULPT_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Keep all writable caches out of the read-only home directory.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/.cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/.cache/wandb}"
export TMPDIR="${TMPDIR:-$PROJECT_ROOT/.cache/tmp}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TMPDIR"

echo "LoRASculpt environment ready: $LORASCULPT_ENV"
echo "Python: $(command -v python)"
