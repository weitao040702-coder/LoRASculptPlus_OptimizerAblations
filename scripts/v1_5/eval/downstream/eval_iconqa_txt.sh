#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


MODEL_PATH=""
MODEL_BASE="${LORASCULPT_ROOT:-$(pwd)}/assets/models/llava-v1.5-7b"
ICONQA_IMAGE_ROOT="${ICONQA_IMAGE_ROOT:-${LORASCULPT_ROOT:-$(pwd)}/assets/data/iconqa/raw/data/iconqa_data}"
CKPT="llava-v1.5-7b"
SPLIT="iconqa"
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

mkdir -p $RESULT_DIR





for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --model-base $MODEL_BASE \
        --question-file "${LORASCULPT_ROOT:-$(pwd)}/assets/data/iconqa/fixed_iconqa_txt-test.jsonl" \
        --image-folder "$ICONQA_IMAGE_ROOT" \
        --answers-file $RESULT_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done



wait



output_file=$RESULT_DIR/$SPLIT/$CKPT/llava-v1.5-7b-iconqa.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


python -m llava.eval.eval_iconqa \
    --annotation-file "${LORASCULPT_ROOT:-$(pwd)}/assets/data/iconqa/fixed_iconqa_txt-test.jsonl" \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$SPLIT/$CKPT \
    --summary-output-dir $SUMMARY_OUTPUT_DIR
