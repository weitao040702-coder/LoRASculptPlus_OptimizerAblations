#!/usr/bin/env bash
# Continue after OK-VQA inference; its four shard outputs already exist.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:L40S:4
#SBATCH --time=72:00:00
#SBATCH --job-name=lorasculpt-upstream-cont
#SBATCH --output=logs/lorasculpt-upstream-cont-%j.out
#SBATCH --error=logs/lorasculpt-upstream-cont-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CHECKPOINT="$LORASCULPT_ROOT/checkpoints/llava-v1.5-7b-iconqa_txt-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1"
RESULT_DIR="$CHECKPOINT/eval_results"
SUMMARY_OUTPUT_DIR="$CHECKPOINT/SummaryResult_iconqa_txt-LoRASculpt-lora-r32-a64-e3-CMRLAMBDA1e-3-OMEGA1.0-RATIO0.1"

python -m llava.eval.eval_okvqa \
  --annotation-file "$LORASCULPT_ROOT/assets/data/eval/okvqa/mscoco_val2014_annotations.json" \
  --question-file "$LORASCULPT_ROOT/assets/data/eval/okvqa/OpenEnded_mscoco_val2014_questions.json" \
  --result-file "$RESULT_DIR/okvqa/llava-v1.5-7b/llava-v1.5-7b-okvqa.jsonl" \
  --output-dir "$RESULT_DIR/okvqa/llava-v1.5-7b" \
  --summary-output-dir "$SUMMARY_OUTPUT_DIR"

GQA_RESULT_DIR="$RESULT_DIR/gqa/llava-v1.5-7b"
GQA_OUTPUT="$GQA_RESULT_DIR/llava-v1.5-7b-gqa.jsonl"
python scripts/convert_gqa_for_eval.py \
  --src "$GQA_OUTPUT" \
  --dst "$LORASCULPT_ROOT/assets/data/eval/gqa/data/testdev_balanced_predictions.json"
if [[ -f "$LORASCULPT_ROOT/assets/data/eval/gqa/data/testdev_balanced_sceneGraphs.json" && \
      -f "$LORASCULPT_ROOT/assets/data/eval/gqa/data/testdev_balanced_all_questions.json" ]]; then
  python "$LORASCULPT_ROOT/assets/data/eval/gqa/data/eval.py" \
    --tier testdev_balanced \
    --scenes "$LORASCULPT_ROOT/assets/data/eval/gqa/data/{tier}_sceneGraphs.json" \
    --questions "$LORASCULPT_ROOT/assets/data/eval/gqa/data/{tier}_all_questions.json" \
    --choices "$LORASCULPT_ROOT/assets/data/eval/gqa/data/{tier}_choices.json" \
    --predictions "$LORASCULPT_ROOT/assets/data/eval/gqa/data/{tier}_predictions.json"
else
  echo "GQA scoring deferred: official testdev scene/question files are missing."
fi

for script in scripts/v1_5/eval/upstream/eval_textvqa.sh; do
  echo "===== START $script $(date) ====="
  bash "$script" "$CHECKPOINT" "$RESULT_DIR" "$SUMMARY_OUTPUT_DIR"
  echo "===== END $script $(date) ====="
done

python scripts/summarize_eval_results.py --checkpoint "$CHECKPOINT"
