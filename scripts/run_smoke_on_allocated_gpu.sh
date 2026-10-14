#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"
source ./setup_env.sh
: "${OPTAB_METHOD:?OPTAB_METHOD is required}"
: "${OPTAB_SMOKE_SLOT:?OPTAB_SMOKE_SLOT is required}"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not provide CUDA_VISIBLE_DEVICES}"

IFS=',' read -r -a ALLOCATED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
if (( OPTAB_SMOKE_SLOT >= ${#ALLOCATED_GPUS[@]} )); then
  echo "Invalid smoke slot=$OPTAB_SMOKE_SLOT for CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi
export OPTAB_GPU="$OPTAB_SMOKE_SLOT"
export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[$OPTAB_SMOKE_SLOT]}"
bash scripts/run_smoke.sh
