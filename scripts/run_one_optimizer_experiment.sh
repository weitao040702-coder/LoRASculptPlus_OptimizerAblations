#!/usr/bin/env bash
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source ./setup_env.sh

: "${OPTAB_METHOD:?OPTAB_METHOD is required}"
: "${OPTAB_RANK:?OPTAB_RANK is required}"
: "${OPTAB_GPU:?OPTAB_GPU is required}"

case "$OPTAB_METHOD" in
  ab_manifold) METHOD_DIR=ABManifold; OPTAB_METHOD_LABEL=ABManifold ;;
  muon) METHOD_DIR=Muon; OPTAB_METHOD_LABEL=Muon ;;
  *) echo "unsupported method: $OPTAB_METHOD" >&2; exit 2 ;;
esac

ABLATION_ROOT="$LORASCULPT_ROOT"
OPTAB_OUTPUT="$ABLATION_ROOT/checkpoints/${METHOD_DIR}-r${OPTAB_RANK}"
OPTAB_LOG="$ABLATION_ROOT/logs/${METHOD_DIR}-r${OPTAB_RANK}.out"
OPTAB_PROGRESS_FILE="$ABLATION_ROOT/progress/${METHOD_DIR}-r${OPTAB_RANK}.json"
export OPTAB_OUTPUT OPTAB_LOG OPTAB_PROGRESS_FILE
export OPTAB_CHECKPOINT="$OPTAB_OUTPUT"

mkdir -p "$ABLATION_ROOT/logs" "$ABLATION_ROOT/progress"
exec > >(tee -a "$OPTAB_LOG") 2>&1
trap 'code=$?; python scripts/mark_failed.py --progress "$OPTAB_PROGRESS_FILE" --log "$OPTAB_LOG" --exit-code "$code" || true; exit "$code"' ERR

echo "START method=$OPTAB_METHOD rank=$OPTAB_RANK gpu=$OPTAB_GPU host=$(hostname)"
echo "ORTHO_LAMBDA=0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
train_start_epoch="$(date +%s)"
echo "TRAINING_START epoch=$train_start_epoch"

export ORTHO_LAMBDA=0
export CMR_LLM_LAMBDA=1e-3
export CMR_CONNECTOR_LAMBDA=1e-5
export STEP_THRESHOLD=100
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export OPTAB_METHOD OPTAB_METHOD_LABEL OPTAB_RANK OPTAB_GPU

monitor_file="$ABLATION_ROOT/logs/${METHOD_DIR}-r${OPTAB_RANK}.gpu.csv"
monitor_gpu() {
  while :; do
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$monitor_file" 2>/dev/null || true
    sleep 10
  done
}
printf '%s\n' 'index,name,memory.used [MiB],memory.total [MiB],utilization.gpu [%],power.draw [W]' > "$monitor_file"
monitor_gpu &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

python -c 'import json, os, time; from pathlib import Path; p=Path(os.environ["OPTAB_PROGRESS_FILE"]); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps({"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": os.environ["OPTAB_METHOD_LABEL"], "rank": int(os.environ["OPTAB_RANK"]), "gpu": os.environ["OPTAB_GPU"], "status": "TRAINING", "current_step": 0, "total_steps": 1875, "orthogonal_loss": False, "checkpoint": os.environ["OPTAB_OUTPUT"], "log": os.environ["OPTAB_LOG"]}, indent=2) + "\n", encoding="utf-8")'

bash scripts/train_optimizer_config.sh

test -f "$OPTAB_OUTPUT/adapter_model.bin"
test -f "$OPTAB_OUTPUT/adapter_config.json"
train_end_epoch="$(date +%s)"
python scripts/record_training_complete.py \
  --progress "$OPTAB_PROGRESS_FILE" \
  --gpu-log "${OPTAB_LOG%.out}.gpu.csv" \
  --start "$train_start_epoch" \
  --end "$train_end_epoch"
echo "TRAINING_COMPLETE method=$OPTAB_METHOD rank=$OPTAB_RANK"
echo "TRAINING_END epoch=$train_end_epoch"
