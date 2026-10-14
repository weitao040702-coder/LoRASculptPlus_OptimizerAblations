#!/usr/bin/env bash
# Resume the optimizer ablations with enough host memory for model loading and Muon.
# The pending runner skips completed checkpoints and evaluation results.
#SBATCH --account=yanggrp
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-a100-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
# The cluster registers this GRES as gpu:A100_80 (case-sensitive).
# gpu-a100-02 currently has one unrelated GPU allocation; use the three
# remaining A100 80GB GPUs for all rank-parallel work.
#SBATCH --gres=gpu:A100_80:3
# 14 days covers the remaining experiments while fitting before the node's
# scheduled reservation later in the month.
#SBATCH --time=14-00:00:00
#SBATCH --job-name=qianr-optab-resume-200g
#SBATCH --output=logs/optab-resume-200g-%j.out
#SBATCH --error=logs/optab-resume-200g-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR:?}"
source ./setup_env.sh
export OPTAB_GPU_COUNT=3

echo "RESUME_ALLOCATION job=${SLURM_JOB_ID} node=$(hostname)"
echo "GRES=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

bash scripts/run_pending_optimizer_ablation.sh

echo "RESUME_JOB_COMPLETE job=${SLURM_JOB_ID} time=$(date -Is)"
