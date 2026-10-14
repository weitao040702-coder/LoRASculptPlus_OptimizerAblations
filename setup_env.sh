#!/usr/bin/env bash
# LoRASculpt runtime environment for the WFU cluster.
# Usage: source setup_env.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LORASCULPT_ROOT="$PROJECT_ROOT"
export PATH="$PROJECT_ROOT/.conda/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The login shell may not have the cluster CUDA module loaded yet.
if type module >/dev/null 2>&1; then
  module load nvidia/cuda11/cuda/11.8.0 >/dev/null 2>&1 || true
fi

# Keep all writable caches out of the read-only home directory.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PROJECT_ROOT/.cache/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/.cache/wandb}"
export TMPDIR="${TMPDIR:-/tmp/qianr-lorasculpt}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" \
  "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TMPDIR"

echo "LoRASculpt environment ready: $PROJECT_ROOT/.conda"
echo "Python: $(command -v python)"
