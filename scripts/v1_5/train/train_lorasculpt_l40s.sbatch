#!/usr/bin/env bash
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-iconqa
#SBATCH --output=logs/lorasculpt-%j.out
#SBATCH --error=logs/lorasculpt-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
source ./setup_env.sh
mkdir -p logs

echo "job=${SLURM_JOB_ID} host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

MODEL_ROOT="$LORASCULPT_ROOT/assets/models"
test -f "$MODEL_ROOT/llava-v1.5-7b/config.json"
test -f "$MODEL_ROOT/clip-vit-large-patch14-336/config.json"
test -f "$LORASCULPT_ROOT/assets/data/iconqa/iconqa_data.zip"
test -f "$LORASCULPT_ROOT/assets/data/iconqa/iconqa_txt-train.json"

LOCAL_DATA="${TMPDIR:-/tmp}/lorasculpt-iconqa-${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DATA"
if [[ ! -f "$LOCAL_DATA/iconqa_data/problems.json" ]]; then
  unzip -q "$LORASCULPT_ROOT/assets/data/iconqa/iconqa_data.zip" -d "$LOCAL_DATA"
fi
export ICONQA_IMAGE_ROOT="$LOCAL_DATA/iconqa_data"
test -f "$ICONQA_IMAGE_ROOT/iconqa/train/choose_txt/10/image.png"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DEVICE="localhost:0,1,2,3"
export DATASET_NAME=iconqa_txt
export TRAINER_NAME=LoRASculpt
export MODEL_NAME_OR_PATH="$MODEL_ROOT/llava-v1.5-7b"
export LORA_RANK=32
export LORA_ALPHA=64
export NUM_TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export LEARNING_RATE=2e-4
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LAMBDA=1e-3
export DEEPSPEED_ZEROFILE="$LORASCULPT_ROOT/scripts/zero2.json"

bash ./scripts/v1_5/train/ours-train.sh
