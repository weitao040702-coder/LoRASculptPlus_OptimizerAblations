#!/usr/bin/env bash
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-iconqa10k
#SBATCH --output=logs/lorasculpt-iconqa10k-%j.out
#SBATCH --error=logs/lorasculpt-iconqa10k-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh
mkdir -p logs

DATA_ROOT="$LORASCULPT_ROOT/assets/data/iconqa/raw/data/iconqa_data"
TRAIN_JSON="$LORASCULPT_ROOT/assets/data/iconqa/iconqa_txt-train-10000k-seed0.json"
if [[ ! -f "$TRAIN_JSON" ]]; then
  python scripts/prepare_iconqa_lora.py \
    --data-root "$DATA_ROOT" \
    --output-root "$LORASCULPT_ROOT/assets/data/iconqa" \
    --train-sample-size 10000 \
    --seed 0
fi

export DATASET_NAME=iconqa_txt
export TRAINER_NAME=LoRASculpt
export LORA_RANK=32
export LORA_ALPHA=64
export NUM_TRAIN_EPOCHS=3
export ICONQA_TRAIN_DATA="$TRAIN_JSON"
export ICONQA_IMAGE_ROOT="$DATA_ROOT"
export OUTPUT_DIR="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculpt-10k-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1"
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LAMBDA=1e-3

bash scripts/v1_5/train/trainconfig_lora.sh
