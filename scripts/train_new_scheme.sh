#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LORASCULPT_ROOT="$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPTAB_PYTHON="${OPTAB_PYTHON:-$ROOT/.venv_new_scheme/bin/python}"
export OPTAB_METHOD=orbit_stiefel
export OPTAB_METHOD_LABEL=OrbitStiefel
export OPTAB_RANK="${OPTAB_RANK:-16}"
export OPTAB_SEED="${OPTAB_SEED:-42}"
TAG="${OPTAB_TAG:-OrbitStiefel-v1-seed${OPTAB_SEED}-r${OPTAB_RANK}}"
export OPTAB_OUTPUT="${OPTAB_OUTPUT:-$ROOT/checkpoints/$TAG}"
export OPTAB_PROGRESS_FILE="${OPTAB_PROGRESS_FILE:-$ROOT/progress/$TAG.json}"
export OPTAB_CHECKPOINT="$OPTAB_OUTPUT"
export OPTAB_LOG="${OPTAB_LOG:-$ROOT/logs/$TAG.out}"
export ORBIT_STIEFEL_CONFIG="${ORBIT_STIEFEL_CONFIG:-$ROOT/configs/orbit_stiefel.json}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/.cache/triton}"
export TMPDIR="${TMPDIR:-$ROOT/.cache/tmp}"
export OPTAB_SAVE_STEPS="${OPTAB_SAVE_STEPS:-10000}"
if [[ -e "$OPTAB_OUTPUT" && "${OPTAB_RESUME:-0}" != 1 ]]; then
  echo "Output exists: $OPTAB_OUTPUT; set OPTAB_RESUME=1 only for a matching saved run" >&2
  exit 1
fi
if [[ "${OPTAB_RESUME:-0}" == 1 ]] && ! compgen -G "$OPTAB_OUTPUT/checkpoint-*/orbit_stiefel_state.pt" > /dev/null; then
  echo "Resume requested without a complete OrbitStiefel checkpoint" >&2
  exit 1
fi
if [[ ! -s "$ORBIT_STIEFEL_CONFIG" ]]; then
  echo "Missing new method configuration" >&2; exit 1
fi
mkdir -p "$(dirname "$OPTAB_LOG")" "$TRITON_CACHE_DIR" "$TMPDIR"
cd "$ROOT"
# Replicated DDP: keep the paper's global batch size at 16.
export OPTAB_WORLD_SIZE="${OPTAB_WORLD_SIZE:-1}"
"$OPTAB_PYTHON" -c 'import os, torch; assert torch.cuda.device_count() == int(os.environ["OPTAB_WORLD_SIZE"]), "Visible GPUs must match OPTAB_WORLD_SIZE"'
if [[ "$OPTAB_WORLD_SIZE" == 2 ]]; then
  export OPTAB_BATCH_SIZE="${OPTAB_BATCH_SIZE:-4}"
  export OPTAB_GRAD_ACCUM="${OPTAB_GRAD_ACCUM:-2}"
fi
if (( ${OPTAB_BATCH_SIZE:-1} * ${OPTAB_GRAD_ACCUM:-16} * OPTAB_WORLD_SIZE != 16 )); then
  echo "Paper protocol requires global batch size 16" >&2; exit 1
fi
bash scripts/train_optimizer_config.sh 2>&1 | tee -a "$OPTAB_LOG"
