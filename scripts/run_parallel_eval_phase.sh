#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source ./setup_env.sh

METHOD_DIR="${1:?method directory is required}"
case "$METHOD_DIR" in
  ABManifold|Muon) ;;
  *) echo "Unsupported method directory: $METHOD_DIR" >&2; exit 2 ;;
esac

: "${CUDA_VISIBLE_DEVICES:?Slurm did not provide CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${OPTAB_GPU_COUNT:-${#ALLOCATED_GPUS[@]}}"
if (( GPU_COUNT < 1 || GPU_COUNT > ${#ALLOCATED_GPUS[@]} )); then
  echo "Invalid GPU_COUNT=$GPU_COUNT for CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 3
fi

echo "PARALLEL_EVAL_START method=$METHOD_DIR gpu_count=$GPU_COUNT cuda_visible=$CUDA_VISIBLE_DEVICES host=$(hostname)"
RANKS=(16 32 64)
for ((batch_start=0; batch_start<${#RANKS[@]}; batch_start+=GPU_COUNT)); do
  pids=()
  batch_end=$((batch_start + GPU_COUNT))
  for ((rank_index=batch_start; rank_index<batch_end && rank_index<${#RANKS[@]}; rank_index++)); do
    rank="${RANKS[$rank_index]}"
    slot=$((rank_index % GPU_COUNT))
    (
      export OPTAB_GPU="$slot"
      # Each evaluator sees exactly one allocated device.
      export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[$slot]}"
      bash scripts/eval_one_upstream.sh \
        "$ROOT/checkpoints/$METHOD_DIR-r$rank" "$METHOD_DIR" "$rank" "$slot"
    ) &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if (( status != 0 )); then
    echo "PARALLEL_EVAL_FAILED method=$METHOD_DIR batch_start=$batch_start" >&2
    exit "$status"
  fi
done
echo "PARALLEL_EVAL_COMPLETE method=$METHOD_DIR"
