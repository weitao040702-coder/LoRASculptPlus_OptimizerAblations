#!/usr/bin/env bash
# Hold the three A100 GPUs after the optimizer ablation completes.
# This job is submitted with an afterok dependency on the experiment job.
# It intentionally performs no computation; it only keeps the GRES allocation.
#SBATCH --account=yanggrp
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-a100-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:A100_80:3
#SBATCH --time=14-00:00:00
#SBATCH --job-name=qianr-optab-hold
#SBATCH --output=logs/optab-hold-%j.out
#SBATCH --error=logs/optab-hold-%j.err
#SBATCH --open-mode=append

set -Eeuo pipefail

hold_until_epoch="$(date -d '2026-10-22 01:10:00 EDT' +%s)"
echo "HOLD_START job=${SLURM_JOB_ID} node=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset} until=$(date -d "@${hold_until_epoch}" -Is)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

while :; do
    now_epoch="$(date +%s)"
    (( now_epoch >= hold_until_epoch )) && break
    sleep 60
done

echo "HOLD_END job=${SLURM_JOB_ID} time=$(date -Is)"
