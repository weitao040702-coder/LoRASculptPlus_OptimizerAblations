#!/usr/bin/env bash
# Train one LoRASculpt rank on two L40S GPUs.
# Submit twice with --export=ALL,LORA_RANK=16 and LORA_RANK=64.
# Two simultaneous jobs use all four GPUs while preserving effective batch 16.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:2
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-iconqa-rank
#SBATCH --output=logs/lorasculpt-iconqa-rank-%j.out
#SBATCH --error=logs/lorasculpt-iconqa-rank-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh
mkdir -p logs

: "${LORA_RANK:=32}"
case "$LORA_RANK" in
  16|64) ;;
  *) echo "LORA_RANK must be 16 or 64, got $LORA_RANK" >&2; exit 2 ;;
esac

DATA_ROOT="$LORASCULPT_ROOT/assets/data/iconqa/raw/data/iconqa_data"
TRAIN_JSON="$LORASCULPT_ROOT/assets/data/iconqa/iconqa_txt-train-10000k-seed0.json"

python - "$TRAIN_JSON" "$DATA_ROOT" <<'PY'
import json
import os
import sys

train_json, image_root = sys.argv[1:]
with open(train_json, encoding="utf-8") as handle:
    rows = json.load(handle)
missing = [row["image"] for row in rows if not os.path.isfile(os.path.join(image_root, row["image"]))]
if missing:
    raise SystemExit(f"missing {len(missing)} training images; first={missing[0]}")
print(f"validated {len(rows)} IconQA samples")
PY

# Keep Slurm's CUDA_VISIBLE_DEVICES mapping. The training config deliberately
# does not pass DeepSpeed --include when running under Slurm.
export DEVICE=localhost:0,1
export MASTER_PORT="${MASTER_PORT:-$((29600 + LORA_RANK))}"
export DATASET_NAME=iconqa_txt
export TRAINER_NAME=LoRASculpt
export LORA_RANK
export LORA_ALPHA=64
export NUM_TRAIN_EPOCHS=3
export PER_DEVICE_TRAIN_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=2
export ICONQA_TRAIN_DATA="$TRAIN_JSON"
export ICONQA_IMAGE_ROOT="$DATA_ROOT"
export OUTPUT_DIR="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculpt-10k-lora-r${LORA_RANK}-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1"
export AB_PRESERVE_RATIO=0.1
export OMEGA=1.0
export CMR_LAMBDA=1e-3

echo "rank=$LORA_RANK gpus=2 effective_global_batch=16 master_port=$MASTER_PORT"
echo "output=$OUTPUT_DIR"
bash scripts/v1_5/train/trainconfig_lora.sh
