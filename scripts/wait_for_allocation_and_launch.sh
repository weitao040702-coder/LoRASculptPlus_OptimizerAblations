#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source ./setup_env.sh

ALLOC_JOB="${1:-${SLURM_JOB_ID:-8375599}}"
WAIT_LOG="$ROOT/logs/allocation-${ALLOC_JOB}.out"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$WAIT_LOG") 2>&1

echo "WAIT_START allocation=$ALLOC_JOB time=$(date -Is) host=$(hostname)"

while ! timeout 8s scontrol ping >/dev/null 2>&1; do
  echo "SLURM_CONTROLLER_UNAVAILABLE time=$(date -Is)"
  sleep 30
done

echo "SLURM_CONTROLLER_AVAILABLE time=$(date -Is)"
while :; do
  JOB_INFO="$(scontrol show job "$ALLOC_JOB" -o)"
  echo "$JOB_INFO"
  case "$JOB_INFO" in
    *JobState=RUNNING*|*JobState=CONFIGURING*) break ;;
    *JobState=PENDING*)
      echo "ALLOCATION_PENDING allocation=$ALLOC_JOB time=$(date -Is)"
      sleep 30
      ;;
    *)
      echo "ALLOCATION_NOT_ACTIVE allocation=$ALLOC_JOB" >&2
      exit 4
      ;;
  esac
done

echo "REQUEST_ALLOCATION_EXTENSION allocation=$ALLOC_JOB time_limit=3-00:00:00"
if scontrol update JobId="$ALLOC_JOB" TimeLimit=3-00:00:00; then
  echo "ALLOCATION_EXTENSION_REQUESTED allocation=$ALLOC_JOB"
else
  echo "ALLOCATION_EXTENSION_FAILED allocation=$ALLOC_JOB"
fi

# Do not launch a chain that cannot finish before the allocation expires.  The
# current experiment is expected to take several hours; require a conservative
# twelve-hour window after any optional extension.
JOB_INFO_AFTER="$(scontrol show job "$ALLOC_JOB" -o)"
END_TIME="$(sed -n 's/.*EndTime=\([^ ]*\).*/\1/p' <<< "$JOB_INFO_AFTER")"
if [[ -z "$END_TIME" || "$END_TIME" == "Unknown" ]]; then
  echo "ALLOCATION_END_TIME_UNKNOWN allocation=$ALLOC_JOB" >&2
  exit 6
fi
END_EPOCH="$(date -d "$END_TIME" +%s)"
NOW_EPOCH="$(date +%s)"
REMAINING=$((END_EPOCH - NOW_EPOCH))
if (( REMAINING < 8 * 3600 )); then
  echo "ALLOCATION_WINDOW_TOO_SHORT allocation=$ALLOC_JOB remaining_seconds=$REMAINING" >&2
  exit 7
fi
echo "ALLOCATION_WINDOW_OK allocation=$ALLOC_JOB remaining_seconds=$REMAINING end=$END_TIME"

echo "LAUNCH_EXPERIMENT allocation=$ALLOC_JOB time=$(date -Is)"
if bash scripts/run_optimizer_ablation_all.sh "$ALLOC_JOB"; then
  echo "EXPERIMENT_CHAIN_COMPLETE allocation=$ALLOC_JOB time=$(date -Is)"
  python scripts/validate_optimizer_ablation.py --root "$ROOT"
  echo "RELEASING_ALLOCATION allocation=$ALLOC_JOB time=$(date -Is)"
  scancel "$ALLOC_JOB"
  echo "ALLOCATION_RELEASED allocation=$ALLOC_JOB time=$(date -Is)"
else
  echo "EXPERIMENT_CHAIN_FAILED allocation=$ALLOC_JOB time=$(date -Is)"
  exit 5
fi
