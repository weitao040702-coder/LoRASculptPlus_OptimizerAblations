#!/usr/bin/env bash
# Evaluate one completed LoRASculpt+ rank on the paper target and four source datasets.
# The local OCR-VQA and GQA protocol is documented in README_REPRODUCTION.md.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-plus-eval
#SBATCH --output=logs/lorasculpt-plus-eval-%j.out
#SBATCH --error=logs/lorasculpt-plus-eval-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
source ./setup_env.sh
mkdir -p logs

LLM_RANK="${PLUS_LLM_RANK:?PLUS_LLM_RANK must be 16, 32, or 64}"
case "$LLM_RANK" in
  16|32|64) ;;
  *) echo "Unsupported PLUS_LLM_RANK=$LLM_RANK" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CHECKPOINT="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculptPlus-10k-lora-r${LLM_RANK}-a64-connector-r128-ca64-e3-srr0.1-cmr1e-3-1e-5-ortho1e-5"
RESULT_DIR="$CHECKPOINT/eval_results"
SUMMARY_OUTPUT_DIR="$CHECKPOINT/SummaryResult_lorasculpt_plus_rank${LLM_RANK}"

test -f "$CHECKPOINT/adapter_model.bin"
test -f "$CHECKPOINT/lorasculpt_plus_config.json"
python scripts/verify_upstream_eval.py --eval-root "$LORASCULPT_ROOT/assets/data/eval"
mkdir -p "$RESULT_DIR"
printf 'LoRASculpt+ rank=%s, connector rank=128, IconQA 10k\n' "$LLM_RANK" > "$SUMMARY_OUTPUT_DIR"
printf 'checkpoint=%s\n' "$CHECKPOINT" >> "$SUMMARY_OUTPUT_DIR"
printf 'OCR-VQA protocol: deterministic local 5,000-example subset\n' >> "$SUMMARY_OUTPUT_DIR"

echo "===== START IconQA target rank=$LLM_RANK $(date) ====="
bash scripts/v1_5/eval/downstream/eval_iconqa_txt_robust.sh \
  "$CHECKPOINT" "$RESULT_DIR" "$SUMMARY_OUTPUT_DIR"
echo "===== END IconQA target rank=$LLM_RANK $(date) ====="

for script in \
  scripts/v1_5/eval/upstream/eval_okvqa.sh \
  scripts/v1_5/eval/upstream/eval_ocrvqa.sh \
  scripts/v1_5/eval/upstream/eval_gqa.sh \
  scripts/v1_5/eval/upstream/eval_textvqa.sh; do
  echo "===== START $script rank=$LLM_RANK $(date) ====="
  bash "$script" "$CHECKPOINT" "$RESULT_DIR" "$SUMMARY_OUTPUT_DIR"
  echo "===== END $script rank=$LLM_RANK $(date) ====="
done

echo "rank=$LLM_RANK evaluation complete"
