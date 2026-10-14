#!/usr/bin/env bash
# Two-GPU fallback while the remaining L40S cards are allocated elsewhere.
# The effective global batch remains 16 via gradient accumulation.
# This is not the paper's exact four-GPU hardware configuration.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:2
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-iconqa10k-2gpu
#SBATCH --output=logs/lorasculpt-iconqa10k-2gpu-%j.out
#SBATCH --error=logs/lorasculpt-iconqa10k-2gpu-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh
mkdir -p logs

DATA_ROOT="$LORASCULPT_ROOT/assets/data/iconqa/raw/data/iconqa_data"
TRAIN_JSON="$LORASCULPT_ROOT/assets/data/iconqa/iconqa_txt-train-10000k-seed0.json"

python - "$TRAIN_JSON" "$DATA_ROOT" <<'PY'
import json
import os
import sys

train_json, image_root = sys.argv[1:]
rows = json.load(open(train_json, encoding="utf-8"))
missing = [row["image"] for row in rows if not os.path.isfile(os.path.join(image_root, row["image"]))]
if missing:
    raise SystemExit(f"missing {len(missing)} training images; first={missing[0]}")
print(f"validated {len(rows)} IconQA samples")
PY

export DEVICE=localhost:0,1
export DATASET_NAME=iconqa_txt
export TRAINER_NAME=LoRASculpt
export LORA_RANK=32
export LORA_ALPHA=64
export NUM_TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=2
export ICONQA_TRAIN_DATA="$TRAIN_JSON"
export ICONQA_IMAGE_ROOT="$DATA_ROOT"
export OUTPUT_DIR="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculpt-10k-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1"
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LAMBDA=1e-3

bash scripts/v1_5/train/trainconfig_lora.sh
