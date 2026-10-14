#!/usr/bin/env bash
# LoRASculpt+ main reproduction: LLaVA-v1.5-7B fine-tuned on 10k IconQA samples.
# Paper protocol: 3 epochs, global batch size 16, LLM rank 32, connector rank 128.
# The four L40S devices are allocated by Slurm; torch.distributed.run preserves
# the allocation's CUDA_VISIBLE_DEVICES mapping on this cluster.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-plus-iconqa
#SBATCH --output=logs/lorasculpt-plus-iconqa-%j.out
#SBATCH --error=logs/lorasculpt-plus-iconqa-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh
mkdir -p logs checkpoints

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
export TRAINER_NAME=LoRASculptPlus
export LORA_RANK=32
export LORA_ALPHA=64
export CONNECTOR_LORA_RANK=128
export CONNECTOR_LORA_ALPHA=64
export NUM_TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export ICONQA_TRAIN_DATA="$TRAIN_JSON"
export ICONQA_IMAGE_ROOT="$DATA_ROOT"

# LoRASculpt+ Eq. (18) and Algorithm 1.
export STEP_THRESHOLD=100
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LLM_LAMBDA=1e-3
export CMR_CONNECTOR_LAMBDA=1e-5
export ORTHO_LAMBDA=1e-5

export OUTPUT_DIR="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculptPlus-10k-lora-r32-a64-connector-r128-ca64-e3-srr0.1-cmr1e-3-1e-5-ortho1e-5"

bash scripts/v1_5/train/trainconfig_lora.sh
