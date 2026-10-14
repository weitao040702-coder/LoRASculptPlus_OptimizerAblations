#!/usr/bin/env bash
# Evaluate the LoRASculpt+ checkpoint on the four upstream datasets used by the paper.
# This deliberately uses the same local prepared protocol as the existing
# reproduction scripts, including the deterministic OCR-VQA 5k subset.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-plus-upstream
#SBATCH --output=logs/lorasculpt-plus-upstream-%j.out
#SBATCH --error=logs/lorasculpt-plus-upstream-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CHECKPOINT="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculptPlus-10k-lora-r32-a64-connector-r128-ca64-e3-srr0.1-cmr1e-3-1e-5-ortho1e-5"
RESULT_DIR="$CHECKPOINT/eval_results_upstream"
SUMMARY_OUTPUT_DIR="$CHECKPOINT/SummaryResult_upstream"

test -f "$CHECKPOINT/adapter_model.bin"
test -f "$CHECKPOINT/lorasculpt_plus_config.json"
python scripts/verify_upstream_eval.py --eval-root "$LORASCULPT_ROOT/assets/data/eval"
mkdir -p "$RESULT_DIR"
printf 'LoRASculpt+ LLM rank 32 / connector rank 128, IconQA 10k\n' > "$SUMMARY_OUTPUT_DIR"
printf 'checkpoint=%s\n' "$CHECKPOINT" >> "$SUMMARY_OUTPUT_DIR"
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
  --rank 32 \
  --output "$CHECKPOINT/paper_style_results_lorasculpt_plus_upstream.md"
cat "$SUMMARY_OUTPUT_DIR"
