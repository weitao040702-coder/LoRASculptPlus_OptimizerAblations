#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source ./setup_env.sh

METHOD="${1:?method is required}"
case "$METHOD" in
  ab_manifold) METHOD_LABEL=ABManifold ;;
  muon) METHOD_LABEL=Muon ;;
  *) echo "Unsupported method: $METHOD" >&2; exit 2 ;;
esac

: "${CUDA_VISIBLE_DEVICES:?Slurm did not provide CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${OPTAB_GPU_COUNT:-${#ALLOCATED_GPUS[@]}}"
if (( GPU_COUNT < 1 || GPU_COUNT > ${#ALLOCATED_GPUS[@]} )); then
  echo "Invalid GPU_COUNT=$GPU_COUNT for CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 3
fi

echo "PARALLEL_TRAIN_START method=$METHOD_LABEL gpu_count=$GPU_COUNT cuda_visible=$CUDA_VISIBLE_DEVICES host=$(hostname)"
RANKS=(16 32 64)
for ((batch_start=0; batch_start<${#RANKS[@]}; batch_start+=GPU_COUNT)); do
  pids=()
  batch_end=$((batch_start + GPU_COUNT))
  for ((rank_index=batch_start; rank_index<batch_end && rank_index<${#RANKS[@]}; rank_index++)); do
    rank="${RANKS[$rank_index]}"
    slot=$((rank_index % GPU_COUNT))
    (
      export OPTAB_METHOD="$METHOD"
      export OPTAB_METHOD_LABEL="$METHOD_LABEL"
      export OPTAB_RANK="$rank"
      export OPTAB_GPU="$slot"
      # Select one element from Slurm's allocation; no physical GPU id is hardcoded.
      export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[$slot]}"
      bash scripts/run_one_optimizer_experiment.sh
    ) &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if (( status != 0 )); then
    echo "PARALLEL_TRAIN_FAILED method=$METHOD_LABEL batch_start=$batch_start" >&2
    exit "$status"
  fi
done
echo "PARALLEL_TRAIN_COMPLETE method=$METHOD_LABEL"
