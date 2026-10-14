#!/usr/bin/env bash
# One LoRASculpt+ paper run for a selected LLM LoRA rank (16, 32, or 64).
# Each run uses four L40S GPUs, global batch 16, and connector rank 128.
# The submit helper chains rank 16 -> 32 -> 64 to avoid concurrent allocations.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-plus-rank
#SBATCH --output=logs/lorasculpt-plus-rank-%j.out
#SBATCH --error=logs/lorasculpt-plus-rank-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
source ./setup_env.sh
mkdir -p logs checkpoints

LLM_RANK="${PLUS_LLM_RANK:?PLUS_LLM_RANK must be 16, 32, or 64}"
case "$LLM_RANK" in
  16|32|64) ;;
  *) echo "Unsupported PLUS_LLM_RANK=$LLM_RANK" >&2; exit 2 ;;
esac

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
export LORA_RANK="$LLM_RANK"
export LORA_ALPHA=64
export CONNECTOR_LORA_RANK=128
export CONNECTOR_LORA_ALPHA=64
export NUM_TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
export ICONQA_TRAIN_DATA="$TRAIN_JSON"
export ICONQA_IMAGE_ROOT="$DATA_ROOT"

# The TPAMI paper leaves the numeric SRR warm-up implicit; the local
# conference implementation uses 100 steps, which is retained here.
export STEP_THRESHOLD=100
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LLM_LAMBDA=1e-3
export CMR_CONNECTOR_LAMBDA=1e-5
export ORTHO_LAMBDA=1e-5

export OUTPUT_DIR="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculptPlus-10k-lora-r${LLM_RANK}-a64-connector-r128-ca64-e3-srr0.1-cmr1e-3-1e-5-ortho1e-5"

echo "job=$SLURM_JOB_ID host=$(hostname) rank=$LLM_RANK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "output=$OUTPUT_DIR"
bash scripts/v1_5/train/trainconfig_lora.sh
