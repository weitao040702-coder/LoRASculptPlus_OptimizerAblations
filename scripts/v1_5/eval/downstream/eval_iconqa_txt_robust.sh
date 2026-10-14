#!/usr/bin/env bash
set -Eeuo pipefail

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

ROOT="${LORASCULPT_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
MODEL_PATH="${1:?checkpoint path is required}"
RESULT_DIR="${2:?result directory is required}"
SUMMARY_OUTPUT_DIR="${3:-None}"
MODEL_BASE="$ROOT/assets/models/llava-v1.5-7b"
ICONQA_IMAGE_ROOT="${ICONQA_IMAGE_ROOT:-$ROOT/assets/data/iconqa/raw/data/iconqa_data}"
OUTPUT_DIR="$RESULT_DIR/iconqa/llava-v1.5-7b"
QUESTION_FILE="$ROOT/assets/data/iconqa/fixed_iconqa_txt-test.jsonl"

mkdir -p "$OUTPUT_DIR"
test -f "$QUESTION_FILE"
test -d "$ICONQA_IMAGE_ROOT"

declare -a pids=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    shard="$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl"
    : > "$shard"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path "$MODEL_PATH" \
        --model-base "$MODEL_BASE" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$ICONQA_IMAGE_ROOT" \
        --answers-file "$shard" \
        --num-chunks "$CHUNKS" \
        --chunk-idx "$IDX" \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "IconQA inference failed" >&2
    exit "$status"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    test -s "$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" || {
        echo "Missing or empty IconQA shard: $OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" >&2
        exit 1
    }
done

output_file="$OUTPUT_DIR/llava-v1.5-7b-iconqa.jsonl"
: > "$output_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done

expected_count=$(wc -l < "$QUESTION_FILE")
actual_count=$(wc -l < "$output_file")
test "$actual_count" -eq "$expected_count" || {
    echo "IconQA prediction count mismatch: expected=$expected_count actual=$actual_count" >&2
    exit 1
}

python -m llava.eval.eval_iconqa \
    --annotation-file "$QUESTION_FILE" \
    --result-file "$output_file" \
    --output-dir "$OUTPUT_DIR" \
    --summary-output-dir "$SUMMARY_OUTPUT_DIR"
