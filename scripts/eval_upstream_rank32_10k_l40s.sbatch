#!/usr/bin/env bash
# Evaluate the corrected 10k IconQA LoRASculpt rank-32 checkpoint.
# OCR-VQA uses the prepared deterministic 5,000-example local subset.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-r32-upstream
#SBATCH --output=logs/lorasculpt-r32-upstream-%j.out
#SBATCH --error=logs/lorasculpt-r32-upstream-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CHECKPOINT="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculpt-10k-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1"
RESULT_DIR="$CHECKPOINT/eval_results_upstream_rank32_10k"
SUMMARY_OUTPUT_DIR="$CHECKPOINT/SummaryResult_upstream_rank32_10k"

test -f "$CHECKPOINT/adapter_model.bin"
python scripts/verify_upstream_eval.py --eval-root "$LORASCULPT_ROOT/assets/data/eval"
mkdir -p "$RESULT_DIR"
printf 'LoRASculpt rank32, IconQA 10k, checkpoint=%s\n' "$CHECKPOINT" > "$SUMMARY_OUTPUT_DIR"
printf 'OCR-VQA protocol: deterministic local 5,000-example subset\n' >> "$SUMMARY_OUTPUT_DIR"

for script in \
  scripts/v1_5/eval/upstream/eval_okvqa.sh \
  scripts/v1_5/eval/upstream/eval_ocrvqa.sh \
  scripts/v1_5/eval/upstream/eval_gqa.sh \
  scripts/v1_5/eval/upstream/eval_textvqa.sh; do
  echo "===== START $script $(date) ====="
  bash "$script" "$CHECKPOINT" "$RESULT_DIR" "$SUMMARY_OUTPUT_DIR"
  echo "===== END $script $(date) ====="
done

python scripts/summarize_eval_results.py \
  --checkpoint "$CHECKPOINT" \
  --results-dir "$RESULT_DIR" \
  --output "$CHECKPOINT/paper_style_results_rank32_10k.md"
cat "$SUMMARY_OUTPUT_DIR"
