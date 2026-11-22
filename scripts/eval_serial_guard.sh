#!/usr/bin/env bash
# Source from formal evaluators; serialize inference work.
EVAL_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DATASET="$(basename "$0" .sh)"
EVAL_RESULT_ROOT="${2:?result directory is required}"
mkdir -p "$EVAL_RESULT_ROOT" "$EVAL_PROJECT_ROOT/automation/orbit-paper-seed42"
# Serialize identical model/dataset work, including calls from the main queue.
exec 9>"$EVAL_RESULT_ROOT/.${EVAL_DATASET}.lock"
flock -x 9
EVAL_SUCCESS="$EVAL_RESULT_ROOT/.${EVAL_DATASET}.completed"
EVAL_SPLIT="${EVAL_DATASET#eval_}"
EVAL_SCORE="$EVAL_RESULT_ROOT/$EVAL_SPLIT/llava-v1.5-7b/result-$EVAL_SPLIT.txt"
if [[ -s "$EVAL_SUCCESS" && -s "$EVAL_SCORE" ]]; then
  echo "Verified evaluator previously completed: $EVAL_SUCCESS"
  exit 0
fi
# New evaluations serialize on slot zero; paused rank32 retains its old slot.
exec 8>"$EVAL_PROJECT_ROOT/automation/orbit-paper-seed42/gpu-evaluation.lock"
flock -x 8
# Interrupted inference must never create a success marker.
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
trap 'eval_exit=$?; if [[ "$eval_exit" == 0 && -s "$EVAL_SCORE" ]]; then date -u +%FT%TZ > "$EVAL_SUCCESS"; fi' EXIT
