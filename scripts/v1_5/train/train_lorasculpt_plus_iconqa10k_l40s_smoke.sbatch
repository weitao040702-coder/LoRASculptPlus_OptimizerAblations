#!/usr/bin/env bash
# Short four-GPU launch validation. This creates a separate smoke checkpoint.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=00:30:00
#SBATCH --job-name=lorasculpt-plus-smoke
#SBATCH --output=logs/lorasculpt-plus-smoke-%j.out
#SBATCH --error=logs/lorasculpt-plus-smoke-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh

export DATASET_NAME=iconqa_txt
export TRAINER_NAME=LoRASculptPlus
export LORA_RANK=32
export LORA_ALPHA=64
export CONNECTOR_LORA_RANK=128
export CONNECTOR_LORA_ALPHA=64
export NUM_TRAIN_EPOCHS=1
# Cross the 100-step SRR activation point so the smoke run validates the
# ZeRO-compatible gradient masking path used by the formal experiments.
export MAX_STEPS=105
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export ICONQA_TRAIN_DATA="$LORASCULPT_ROOT/assets/data/iconqa/iconqa_txt-train-10000k-seed0.json"
export ICONQA_IMAGE_ROOT="$LORASCULPT_ROOT/assets/data/iconqa/raw/data/iconqa_data"
export STEP_THRESHOLD=100
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LLM_LAMBDA=1e-3
export CMR_CONNECTOR_LAMBDA=1e-5
export ORTHO_LAMBDA=1e-5
export OUTPUT_DIR="$LORASCULPT_ROOT/checkpoints/lorasculpt-plus-smoke-r32-connector-r128-srr105"

bash scripts/v1_5/train/trainconfig_lora.sh
