#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source ./setup_env.sh

: "${CUDA_VISIBLE_DEVICES:?Slurm did not provide CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$CUDA_VISIBLE_DEVICES"
GPU_COUNT="${OPTAB_GPU_COUNT:-${#ALLOCATED_GPUS[@]}}"
if (( GPU_COUNT < 1 || GPU_COUNT > ${#ALLOCATED_GPUS[@]} )); then
  echo "Invalid GPU_COUNT=$GPU_COUNT for CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

is_training_complete() {
  local method_dir="$1" rank="$2"
  [[ -s "$ROOT/checkpoints/${method_dir}-r${rank}/adapter_model.bin" &&
     -s "$ROOT/checkpoints/${method_dir}-r${rank}/adapter_config.json" ]]
}

is_eval_complete() {
  local method_dir="$1" rank="$2" result_root
  result_root="$ROOT/checkpoints/${method_dir}-r${rank}/eval_results"
  for result in \
    "$result_root/okvqa/llava-v1.5-7b/result-okvqa.txt" \
    "$result_root/ocrvqa/llava-v1.5-7b/result-ocrvqa.txt" \
    "$result_root/gqa/llava-v1.5-7b/result-gqa.txt" \
    "$result_root/textvqa/llava-v1.5-7b/result-textvqa.txt"; do
    [[ -s "$result" ]] || return 1
  done
}

render() {
  python scripts/render_progress.py \
    --progress-dir "$ROOT/progress" \
    --allocation-job "${SLURM_JOB_ID:-resume}" \
    --node "$(hostname)" || true
}

run_pending_training() {
  local method="$1" method_dir pending rank slot
  case "$method" in
    ab_manifold) method_dir=ABManifold ;;
    muon) method_dir=Muon ;;
    *) echo "unsupported method: $method" >&2; return 2 ;;
  esac
  pending=()
  for rank in 16 32 64; do
    if ! is_training_complete "$method_dir" "$rank"; then
      pending+=("$rank")
    else
      echo "SKIP_TRAIN method=$method_dir rank=$rank reason=checkpoint-present"
    fi
  done
  ((${#pending[@]})) || { render; return 0; }
  echo "PENDING_TRAIN method=$method_dir ranks=${pending[*]} gpu_count=$GPU_COUNT"
  local pids=()
  for ((slot=0; slot<${#pending[@]}; slot++)); do
    rank="${pending[$slot]}"
    if (( slot >= GPU_COUNT )); then
      echo "ERROR: pending ranks exceed allocated GPUs" >&2
      return 3
    fi
    (
      export OPTAB_METHOD="$method"
      export OPTAB_METHOD_LABEL="$method_dir"
      export OPTAB_RANK="$rank"
      export OPTAB_GPU="$slot"
      export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[$slot]}"
      bash scripts/run_one_optimizer_experiment.sh
    ) &
    pids+=("$!")
  done
  local status=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  ((status == 0)) || return "$status"
  render
}

run_pending_evaluation() {
  local method_dir="$1" pending rank slot
  pending=()
  for rank in 16 32 64; do
    if ! is_eval_complete "$method_dir" "$rank"; then
      pending+=("$rank")
    else
      echo "SKIP_EVAL method=$method_dir rank=$rank reason=all-four-results-present"
    fi
  done
  ((${#pending[@]})) || { render; return 0; }
  echo "PENDING_EVAL method=$method_dir ranks=${pending[*]} gpu_count=$GPU_COUNT"
  local pids=()
  for ((slot=0; slot<${#pending[@]}; slot++)); do
    rank="${pending[$slot]}"
    if (( slot >= GPU_COUNT )); then
      echo "ERROR: pending ranks exceed allocated GPUs" >&2
      return 3
    fi
    (
      export OPTAB_GPU="$slot"
      export CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[$slot]}"
      bash scripts/eval_one_upstream.sh \
        "$ROOT/checkpoints/$method_dir-r$rank" "$method_dir" "$rank" "$slot"
    ) &
    pids+=("$!")
  done
  local status=0 pid
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  ((status == 0)) || return "$status"
  render
}

echo "RESUME_START allocation=${SLURM_JOB_ID:-unknown} host=$(hostname) time=$(date -Is)"
echo "GPU_COUNT=$GPU_COUNT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
render

for method in ab_manifold muon; do
  run_pending_training "$method"
  case "$method" in
    ab_manifold) run_pending_evaluation ABManifold ;;
    muon) run_pending_evaluation Muon ;;
  esac
done

python scripts/summarize_optimizer_ablation_final.py --root "$ROOT"
render
echo "RESUME_COMPLETE allocation=${SLURM_JOB_ID:-unknown} time=$(date -Is)"
