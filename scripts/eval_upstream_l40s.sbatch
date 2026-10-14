#!/usr/bin/env bash
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-upstream-eval
#SBATCH --output=logs/lorasculpt-upstream-eval-%j.out
#SBATCH --error=logs/lorasculpt-upstream-eval-%j.err
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
python scripts/verify_upstream_eval.py --eval-root "$LORASCULPT_ROOT/assets/data/eval"

for script in \
  scripts/v1_5/eval/upstream/eval_okvqa.sh \
  scripts/v1_5/eval/upstream/eval_ocrvqa.sh \
  scripts/v1_5/eval/upstream/eval_gqa.sh \
  scripts/v1_5/eval/upstream/eval_textvqa.sh; do
  echo "===== START $script $(date) ====="
  bash "$script" "$CHECKPOINT" "$RESULT_DIR" "$SUMMARY_OUTPUT_DIR"
  echo "===== END $script $(date) ====="
done

echo "===== SUMMARY ====="
cat "$SUMMARY_OUTPUT_DIR"
