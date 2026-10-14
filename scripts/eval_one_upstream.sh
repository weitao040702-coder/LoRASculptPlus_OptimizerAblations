#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${LORASCULPT_ROOT:?LORASCULPT_ROOT is required}"
CHECKPOINT="${1:?checkpoint is required}"
METHOD_DIR="${2:?method directory is required}"
RANK="${3:?rank is required}"
GPU="${4:?gpu is required}"
RESULT_DIR="$ROOT/checkpoints/$METHOD_DIR-r$RANK/eval_results"
LOG="$ROOT/logs/${METHOD_DIR}-r${RANK}-eval.out"
SUMMARY="$ROOT/checkpoints/$METHOD_DIR-r$RANK/evaluation_metadata.txt"
GPU_LOG="$ROOT/logs/${METHOD_DIR}-r${RANK}-eval.gpu.csv"

mkdir -p "$RESULT_DIR" "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; python scripts/mark_failed.py --progress "$ROOT/progress/${METHOD_DIR}-r${RANK}.json" --log "$LOG" --exit-code "$code" || true; exit "$code"' ERR

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
eval_start_epoch="$(date +%s)"
echo "EVAL_START method=$METHOD_DIR rank=$RANK gpu=$GPU cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} host=$(hostname) start=$(date -Is) epoch=$eval_start_epoch"
echo "checkpoint=$CHECKPOINT"

monitor_gpu() {
  while :; do
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits >> "$GPU_LOG" 2>/dev/null || true
    sleep 10
  done
}
printf 'index,name,memory.used [MiB],memory.total [MiB],utilization.gpu [%%],power.draw [W]\n' > "$GPU_LOG"
monitor_gpu &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

python scripts/mark_eval_progress.py --root "$ROOT" --method "$METHOD_DIR" --rank "$RANK" --gpu "$GPU" --status EVALUATING

python scripts/verify_upstream_eval.py --eval-root "$ROOT/assets/data/eval"
printf 'method=%s\nrank=%s\ncheckpoint=%s\ngpu=%s\nstart=%s\n' \
  "$METHOD_DIR" "$RANK" "$CHECKPOINT" "$GPU" "$(date -Is)" > "$SUMMARY"
printf 'protocol=OKVQA 5046; OCRVQA deterministic 5000; GQA testdev-balanced; TextVQA 5000\n' >> "$SUMMARY"

for spec in \
  "okvqa scripts/v1_5/eval/upstream/eval_okvqa.sh result-okvqa.txt" \
  "ocrvqa scripts/v1_5/eval/upstream/eval_ocrvqa.sh result-ocrvqa.txt" \
  "gqa scripts/v1_5/eval/upstream/eval_gqa.sh result-gqa.txt" \
  "textvqa scripts/v1_5/eval/upstream/eval_textvqa.sh result-textvqa.txt"; do
  read -r dataset script result <<< "$spec"
  echo "DATASET_START dataset=$dataset time=$(date -Is)"
  bash "$script" "$CHECKPOINT" "$RESULT_DIR" "$SUMMARY"
  result_path="$RESULT_DIR/$dataset/llava-v1.5-7b/$result"
  test -s "$result_path"
  python scripts/mark_eval_progress.py --root "$ROOT" --method "$METHOD_DIR" --rank "$RANK" --gpu "$GPU" --dataset "$(tr '[:lower:]' '[:upper:]' <<< "$dataset" | sed 's/OCRVQA/OCRVQA/; s/TEXTVQA/TextVQA/; s/OKVQA/OKVQA/; s/GQA/GQA/')" --result "$result_path" --status EVALUATING
  echo "DATASET_DONE dataset=$dataset result=$result_path time=$(date -Is)"
done

printf 'end=%s\n' "$(date -Is)" >> "$SUMMARY"
eval_end_epoch="$(date +%s)"
python scripts/record_eval_complete.py \
  --progress "$ROOT/progress/${METHOD_DIR}-r${RANK}.json" \
  --result-root "$RESULT_DIR" \
  --gpu-log "$GPU_LOG" \
  --start "$eval_start_epoch" \
  --end "$eval_end_epoch"
echo "EVAL_COMPLETE method=$METHOD_DIR rank=$RANK"
