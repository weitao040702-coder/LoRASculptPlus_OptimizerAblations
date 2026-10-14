#!/usr/bin/env bash
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=12:00:00
#SBATCH --job-name=lorasculpt-iconqa-eval
#SBATCH --output=logs/lorasculpt-iconqa-eval-%j.out
#SBATCH --error=logs/lorasculpt-iconqa-eval-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
source ./setup_env.sh
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CHECKPOINT="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1"
RESULT_DIR="$CHECKPOINT/eval_results"
SUMMARY_OUTPUT_DIR="$CHECKPOINT/SummaryResult_iconqa_txt-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1"

test -f "$CHECKPOINT/adapter_model.bin"
test -f "$LORASCULPT_ROOT/assets/models/llava-v1.5-7b/config.json"
test -f "$LORASCULPT_ROOT/assets/data/iconqa/fixed_iconqa_txt-test.jsonl"
test -f "$LORASCULPT_ROOT/assets/data/iconqa/raw/data/iconqa_data/iconqa/test/choose_txt/7/image.png"

echo "job=$SLURM_JOB_ID host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "checkpoint=$CHECKPOINT"

bash ./scripts/v1_5/eval/downstream/eval_iconqa_txt.sh \
  "$CHECKPOINT" \
  "$RESULT_DIR" \
  "$SUMMARY_OUTPUT_DIR"
