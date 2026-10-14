#!/usr/bin/env bash
# One-question evaluator smoke test for the rank-32 checkpoint.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=00:10:00
#SBATCH --job-name=lorasculpt-eval-smoke
#SBATCH --output=logs/lorasculpt-eval-smoke-%j.out
#SBATCH --error=logs/lorasculpt-eval-smoke-%j.err

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh
CHECKPOINT="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculpt-10k-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1"
python -m llava.eval.model_vqa_loader \
  --model-path "$CHECKPOINT" \
  --model-base "$LORASCULPT_ROOT/assets/models/llava-v1.5-7b" \
  --question-file "$LORASCULPT_ROOT/assets/data/eval/okvqa/okvqa_val.jsonl" \
  --image-folder "$LORASCULPT_ROOT/assets/data/eval/okvqa/val2014" \
  --answers-file "$CHECKPOINT/smoke_rank32.jsonl" \
  --num-chunks 5046 \
  --chunk-idx 0 \
  --temperature 0 \
  --conv-mode vicuna_v1
test "$(wc -l < "$CHECKPOINT/smoke_rank32.jsonl")" -eq 1
echo "SMOKE_OK"
