#!/usr/bin/env bash
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source ./setup_env.sh
: "${OPTAB_METHOD:?OPTAB_METHOD is required}"
: "${OPTAB_GPU:?OPTAB_GPU is required}"

case "$OPTAB_METHOD" in
  ab_manifold) METHOD_DIR=ABManifold ;;
  muon) METHOD_DIR=Muon ;;
  *) echo "unsupported method: $OPTAB_METHOD" >&2; exit 2 ;;
esac

export OPTAB_RANK=32
export OPTAB_METHOD_LABEL="$METHOD_DIR"
export OPTAB_OUTPUT="$LORASCULPT_ROOT/checkpoints/smoke-${METHOD_DIR}-r32"
export OPTAB_LOG="$LORASCULPT_ROOT/logs/smoke-${METHOD_DIR}-r32.out"
export OPTAB_PROGRESS_FILE="$LORASCULPT_ROOT/progress/smoke/${METHOD_DIR}-r32.json"
export OPTAB_CHECKPOINT="$OPTAB_OUTPUT"
export OPTAB_MAX_STEPS=2
export ORTHO_LAMBDA=0 CMR_LLM_LAMBDA=1e-3 CMR_CONNECTOR_LAMBDA=1e-5
export STEP_THRESHOLD=100 AB_PRESERVE_RATIO=0.1 OMEGA=1.0

mkdir -p "$LORASCULPT_ROOT/checkpoints" "$LORASCULPT_ROOT/logs" "$LORASCULPT_ROOT/progress/smoke"
exec > >(tee -a "$OPTAB_LOG") 2>&1
trap 'code=$?; python scripts/mark_failed.py --progress "$OPTAB_PROGRESS_FILE" --log "$OPTAB_LOG" --exit-code "$code" || true; exit "$code"' ERR
echo "SMOKE_START method=$OPTAB_METHOD rank=32 gpu=$OPTAB_GPU host=$(hostname)"
bash scripts/train_optimizer_config.sh
test -f "$OPTAB_OUTPUT/adapter_model.bin"
python -c 'import json, os, time; from pathlib import Path; from llava.train.progress import update_run_progress; p=Path(os.environ["OPTAB_PROGRESS_FILE"]); d=json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}; d.update({"updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "method": os.environ["OPTAB_METHOD_LABEL"], "rank": 32, "gpu": os.environ["OPTAB_GPU"], "smoke": "PASS", "status": "WAITING", "current_step": 0}); update_run_progress(p, d)'
echo "SMOKE_COMPLETE method=$OPTAB_METHOD"
