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
SPLIT="gqa"
RESULT_DIR=""
GQADIR="$ROOT/assets/data/eval/gqa/data"


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
        --question-file "$ROOT/assets/data/eval/gqa/llava_gqa_testdev_balanced.jsonl" \
        --image-folder "$GQADIR/images" \
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
    echo "GQA inference failed" >&2
    exit "$status"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
    test -s "$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" || {
        echo "Missing or empty GQA shard: $OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" >&2
        exit 1
    }
done

output_file="$OUTPUT_DIR/llava-v1.5-7b-gqa.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "$OUTPUT_DIR/${CHUNKS}_${IDX}.jsonl" >> "$output_file"
done
expected_count=$(wc -l < "$ROOT/assets/data/eval/gqa/llava_gqa_testdev_balanced.jsonl")
actual_count=$(wc -l < "$output_file")
test "$actual_count" -eq "$expected_count" || {
    echo "GQA prediction count mismatch: expected=$expected_count actual=$actual_count" >&2
    exit 1
}

GQA_PREDICTIONS="$OUTPUT_DIR/testdev_balanced_predictions.json"
python scripts/convert_gqa_for_eval.py --src "$output_file" --dst "$GQA_PREDICTIONS"

if [ -f "$GQADIR/eval/1_eval.py" ] && [ -f "$GQADIR/testdev_balanced_sceneGraphs.json" ] && [ -f "$GQADIR/testdev_balanced_choices.json" ]; then
    EVAL_SCRIPT="$GQADIR/eval/1_eval.py"
elif [ -f "$GQADIR/eval.py" ] && [ -f "$GQADIR/testdev_balanced_sceneGraphs.json" ] && [ -f "$GQADIR/testdev_balanced_choices.json" ]; then
    EVAL_SCRIPT="$GQADIR/eval.py"
else
    python scripts/score_gqa_accuracy.py \
        --questions "$GQADIR/testdev_balanced_questions.json" \
        --predictions "$GQA_PREDICTIONS" \
        --output "$OUTPUT_DIR/result-gqa.txt"
    if [ "$SUMMARY_OUTPUT_DIR" != "None" ]; then
        printf 'GQA protocol: testdev-balanced accuracy only (scene-graph evaluator unavailable)\n' >> "$SUMMARY_OUTPUT_DIR"
        grep '^Accuracy:' "$OUTPUT_DIR/result-gqa.txt" | sed 's/^/GQA /' >> "$SUMMARY_OUTPUT_DIR"
    fi
    exit 0
fi
python "$ROOT/scripts/run_gqa_official.py" \
    --data-root "$GQADIR" \
    --evaluator "$EVAL_SCRIPT" \
    --predictions "$GQA_PREDICTIONS" \
    --output-dir "$OUTPUT_DIR" \
    --summary-output-dir "$SUMMARY_OUTPUT_DIR"
