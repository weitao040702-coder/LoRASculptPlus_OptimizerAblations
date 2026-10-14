#!/usr/bin/env bash
# CPU-only final aggregation after rank 16/32/64 evaluations.
#SBATCH --account=yanggrp
#SBATCH --partition=yangGrp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=lorasculpt-plus-summary
#SBATCH --output=logs/lorasculpt-plus-summary-%j.out
#SBATCH --error=logs/lorasculpt-plus-summary-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}"
source ./setup_env.sh
python scripts/summarize_lorasculpt_plus_ranks.py \
  --root "$LORASCULPT_ROOT" \
  --output "$LORASCULPT_ROOT/rank16_32_64_paper_style_results.md"
