#!/usr/bin/env bash
set -Eeuo pipefail

: "${OPTAB_METHOD:?OPTAB_METHOD is required}"
: "${OPTAB_RANK:?OPTAB_RANK is required}"
: "${OPTAB_OUTPUT:?OPTAB_OUTPUT is required}"
: "${OPTAB_PROGRESS_FILE:?OPTAB_PROGRESS_FILE is required}"

ROOT="${LORASCULPT_ROOT:?LORASCULPT_ROOT is required}"
DATA_ROOT="$ROOT/assets/data/iconqa/raw/data/iconqa_data"
TRAIN_JSON="$ROOT/assets/data/iconqa/iconqa_txt-train-10000k-seed0.json"

case "$OPTAB_METHOD" in
  ab_manifold) TRAINER_NAME=ABManifoldLoRASculptPlus ;;
  muon) TRAINER_NAME=MuonLoRASculptPlus ;;
  *) echo "Unsupported OPTAB_METHOD=$OPTAB_METHOD" >&2; exit 2 ;;
esac

mkdir -p "$OPTAB_OUTPUT" "$(dirname "$OPTAB_PROGRESS_FILE")"

EXTRA_TRAIN_ARGS=()
if [[ -n "${OPTAB_MAX_STEPS:-}" ]]; then
  EXTRA_TRAIN_ARGS+=(--max_steps "$OPTAB_MAX_STEPS" --save_steps "$OPTAB_MAX_STEPS")
fi

python -m llava.train.train_mem \
  --lora_enable True \
  --lora_r "$OPTAB_RANK" \
  --lora_alpha 64 \
  --connector_lora_r 128 \
  --connector_lora_alpha 64 \
  --mm_projector_lr 2e-5 \
  --model_name_or_path "$ROOT/assets/models/llava-v1.5-7b" \
  --version v1 \
  --data_path "$TRAIN_JSON" \
  --image_folder "$DATA_ROOT" \
  --vision_tower "$ROOT/assets/models/clip-vit-large-patch14-336" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "$OPTAB_OUTPUT" \
  --overwrite_output_dir True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 10000 \
  --save_total_limit 15 \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none \
  --seed 42 \
  --data_seed 42 \
  --trainer_name "$TRAINER_NAME" \
  "${EXTRA_TRAIN_ARGS[@]}"
