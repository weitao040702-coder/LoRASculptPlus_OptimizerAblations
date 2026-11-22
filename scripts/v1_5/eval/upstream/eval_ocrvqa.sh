#!/bin/bash
set -e
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/eval_serial_guard.sh"

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


MODEL_PATH=""
ROOT="${LORASCULPT_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
MODEL_BASE="$ROOT/assets/models/llava-v1.5-7b"
CKPT="llava-v1.5-7b"
SPLIT="ocrvqa"
RESULT_DIR=""


if [ ! -n "$1" ] ;then
    MODEL_PATH=$MODEL_PATH
else
    MODEL_PATH=$1
fi

if [ ! -n "$2" ] ;then
    RESULT_DIR=$RESULT_DIR
else
    RESULT_DIR=$2
fi

if [ ! -n "$3" ] ;then
    SUMMARY_OUTPUT_DIR="None"
else
    SUMMARY_OUTPUT_DIR=$3
fi

OUTPUT_DIR="$RESULT_DIR/$SPLIT/$CKPT"
mkdir -p "$OUTPUT_DIR"





declare -a pids=()
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --model-base $MODEL_BASE \
        --question-file "$ROOT/assets/data/eval/ocrvqa/sampled_ocrvqa_test.jsonl" \
        --image-folder "$ROOT/assets/data/eval/ocrvqa/sampled_images" \
        --answers-file "$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    pids+=("$!")
done





status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if [ "$status" -ne 0 ]; then
    echo "OCR-VQA inference failed" >&2
    exit "$status"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    test -s "$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" || {
        echo "Missing or empty OCR-VQA shard: $OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" >&2
        exit 1
    }
done



output_file="$OUTPUT_DIR/llava-v1.5-7b-ocrvqa.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done
expected_count=$(wc -l < "$ROOT/assets/data/eval/ocrvqa/sampled_ocrvqa_test.jsonl")
actual_count=$(wc -l < "$output_file")
test "$actual_count" -eq "$expected_count" || {
    echo "OCR-VQA prediction count mismatch: expected=$expected_count actual=$actual_count" >&2
    exit 1
}


python -m llava.eval.eval_ocrvqa \
    --annotation-file "$ROOT/assets/data/eval/ocrvqa/sampled_ocrvqa_test.jsonl" \
    --result-file "$output_file" \
    --output-dir "$OUTPUT_DIR" \
    --summary-output-dir "$SUMMARY_OUTPUT_DIR"
