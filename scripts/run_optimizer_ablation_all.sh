#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source ./setup_env.sh

ALLOC_JOB="${1:-${SLURM_JOB_ID:-}}"
if [[ -z "$ALLOC_JOB" ]]; then
  echo "Usage: $0 <existing A100 allocation job id>" >&2
  exit 2
fi

if ! scontrol ping >/dev/null 2>&1; then
  echo "Slurm controller is unavailable; no experiment was started." >&2
  exit 3
fi

JOB_INFO="$(scontrol show job "$ALLOC_JOB" -o)"
echo "$JOB_INFO"
case "$JOB_INFO" in
  *JobState=RUNNING*|*JobState=CONFIGURING*) ;;
  *) echo "Allocation $ALLOC_JOB is not running/configuring" >&2; exit 4 ;;
esac

export OPTAB_ALLOCATION_JOB="$ALLOC_JOB"
ALLOC_GPU_COUNT="$(sed -n 's/.*TresPerNode=gres\/gpu:[Aa]100_[0-9][0-9]:\([0-9][0-9]*\).*/\1/p' <<< "$JOB_INFO" | head -1)"
if [[ -z "$ALLOC_GPU_COUNT" ]]; then
  echo "Could not determine A100 count from allocation: $JOB_INFO" >&2
  exit 5
fi
export OPTAB_GPU_COUNT="${OPTAB_GPU_COUNT:-$ALLOC_GPU_COUNT}"
export OPTAB_NODE="$(sed -n 's/.*NodeList=\([^ ]*\).*/\1/p' <<< "$JOB_INFO" | head -1)"
python scripts/init_progress.py --root "$ROOT" --allocation-job "$ALLOC_JOB" --node "$OPTAB_NODE"

run_smoke() {
  local method="$1" slot="$2"
  echo "SMOKE_SUBMIT method=$method slot=$slot"
  srun --jobid="$ALLOC_JOB" --overlap --ntasks=1 --gres="gpu:$OPTAB_GPU_COUNT" \
    --export="ALL,OPTAB_METHOD=$method,OPTAB_GPU=$slot,OPTAB_GPU_COUNT=$OPTAB_GPU_COUNT,OPTAB_ALLOCATION_JOB=$ALLOC_JOB,OPTAB_NODE=$OPTAB_NODE,OPTAB_SMOKE_SLOT=$slot" \
    bash scripts/run_smoke_on_allocated_gpu.sh
}

run_training_phase() {
  local method="$1"
  echo "TRAIN_SUBMIT method=$method ranks=16,32,64"
  srun --jobid="$ALLOC_JOB" --overlap --ntasks=1 --gres="gpu:$OPTAB_GPU_COUNT" \
    --export="ALL,OPTAB_METHOD=$method,OPTAB_ALLOCATION_JOB=$ALLOC_JOB,OPTAB_NODE=$OPTAB_NODE,OPTAB_GPU_COUNT=$OPTAB_GPU_COUNT" \
    bash scripts/run_parallel_train_phase.sh "$method"
}

run_evaluation_phase() {
  local method="$1"
  echo "EVAL_SUBMIT method=$method ranks=16,32,64"
  srun --jobid="$ALLOC_JOB" --overlap --ntasks=1 --gres="gpu:$OPTAB_GPU_COUNT" \
    --export="ALL,OPTAB_ALLOCATION_JOB=$ALLOC_JOB,OPTAB_NODE=$OPTAB_NODE,OPTAB_GPU_COUNT=$OPTAB_GPU_COUNT" \
    bash scripts/run_parallel_eval_phase.sh "$method"
}

run_smoke ab_manifold 0
run_smoke muon 0
run_training_phase ab_manifold
run_evaluation_phase ABManifold
run_training_phase muon
run_evaluation_phase Muon

python scripts/summarize_optimizer_ablation_final.py --root "$ROOT"
python scripts/render_progress.py --progress-dir "$ROOT/progress" --allocation-job "$ALLOC_JOB" --node "$OPTAB_NODE"
echo "ALL_EXPERIMENTS_COMPLETE allocation=$ALLOC_JOB"
echo "GPU allocation remains reserved; release only after reviewing results.md and progress/progress.md."
