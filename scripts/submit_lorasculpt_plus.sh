#!/usr/bin/env bash
# Submit the LoRASculpt+ rank-16/32/64 reproduction chain.
# Each train/eval pair requests four L40S GPUs; jobs are serialized so the same
# four devices are reused and no rank is evaluated before its own checkpoint exists.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! scontrol ping >/dev/null 2>&1; then
  echo "Slurm controller is unavailable; no job was submitted." >&2
  exit 2
fi

common_export="ALL"
smoke_job="$(sbatch --parsable --export="${common_export},PLUS_LLM_RANK=32" scripts/v1_5/train/train_lorasculpt_plus_iconqa10k_l40s_smoke.sbatch)"

train16="$(sbatch --parsable --dependency="afterok:${smoke_job}" --export="${common_export},PLUS_LLM_RANK=16" scripts/v1_5/train/train_lorasculpt_plus_iconqa10k_l40s_rank.sbatch)"
eval16="$(sbatch --parsable --dependency="afterok:${train16}" --export="${common_export},PLUS_LLM_RANK=16" scripts/v1_5/eval/eval_lorasculpt_plus_rank_l40s.sbatch)"

train32="$(sbatch --parsable --dependency="afterok:${eval16}" --export="${common_export},PLUS_LLM_RANK=32" scripts/v1_5/train/train_lorasculpt_plus_iconqa10k_l40s_rank.sbatch)"
eval32="$(sbatch --parsable --dependency="afterok:${train32}" --export="${common_export},PLUS_LLM_RANK=32" scripts/v1_5/eval/eval_lorasculpt_plus_rank_l40s.sbatch)"

train64="$(sbatch --parsable --dependency="afterok:${eval32}" --export="${common_export},PLUS_LLM_RANK=64" scripts/v1_5/train/train_lorasculpt_plus_iconqa10k_l40s_rank.sbatch)"
eval64="$(sbatch --parsable --dependency="afterok:${train64}" --export="${common_export},PLUS_LLM_RANK=64" scripts/v1_5/eval/eval_lorasculpt_plus_rank_l40s.sbatch)"

summary="$(sbatch --parsable --dependency="afterok:${eval64}" scripts/v1_5/eval/summarize_lorasculpt_plus_ranks.sbatch)"

printf 'smoke_job=%s\n' "$smoke_job"
printf 'train16=%s (afterok:%s)\n' "$train16" "$smoke_job"
printf 'eval16=%s (afterok:%s)\n' "$eval16" "$train16"
printf 'train32=%s (afterok:%s)\n' "$train32" "$eval16"
printf 'eval32=%s (afterok:%s)\n' "$eval32" "$train32"
printf 'train64=%s (afterok:%s)\n' "$train64" "$eval32"
printf 'eval64=%s (afterok:%s)\n' "$eval64" "$train64"
printf 'summary=%s (afterok:%s)\n' "$summary" "$eval64"
