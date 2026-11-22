#!/usr/bin/env bash
set -Eeuo pipefail

: "${OPTAB_METHOD:?OPTAB_METHOD is required}"
: "${OPTAB_RANK:?OPTAB_RANK is required}"
: "${OPTAB_OUTPUT:?OPTAB_OUTPUT is required}"
: "${OPTAB_PROGRESS_FILE:?OPTAB_PROGRESS_FILE is required}"

ROOT="${LORASCULPT_ROOT:?LORASCULPT_ROOT is required}"
DATA_ROOT="$ROOT/assets/data/iconqa/raw/data/iconqa_data"
TRAIN_JSON="${OPTAB_TRAIN_JSON:-$ROOT/assets/data/iconqa/iconqa_txt-train-10000k-seed0.json}"

TRAIN_MODULE=llava.train.train_mem
SAVE_STEPS_DEFAULT=10000
case "$OPTAB_METHOD" in
  orbit_stiefel) TRAINER_NAME=OrbitStiefelLoRASculptPlus; TRAIN_MODULE=llava.train.train; SAVE_STEPS_DEFAULT=10000
    TRAIN_JSON="${OPTAB_TRAIN_JSON:-$ROOT/assets/data/iconqa/iconqa_txt-train-10000-seed42-paper.json}" ;;
  *) echo "Unsupported OPTAB_METHOD=$OPTAB_METHOD" >&2; exit 2 ;;
esac

mkdir -p "$OPTAB_OUTPUT" "$(dirname "$OPTAB_PROGRESS_FILE")"

EXTRA_TRAIN_ARGS=()
if [[ -n "${OPTAB_MAX_STEPS:-}" ]]; then
  EXTRA_TRAIN_ARGS+=(--max_steps "$OPTAB_MAX_STEPS" --save_steps "$OPTAB_MAX_STEPS")
fi

LAUNCH=("${OPTAB_PYTHON:-python}" -m "$TRAIN_MODULE")
if [[ "${OPTAB_WORLD_SIZE:-1}" == 2 ]]; then
  LAUNCH=("${OPTAB_PYTHON:-python}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m "$TRAIN_MODULE")
  EXTRA_TRAIN_ARGS+=(--ddp_find_unused_parameters False --ddp_broadcast_buffers False)
fi
"${LAUNCH[@]}" \
  --lora_enable True \
  --lora_r "$OPTAB_RANK" \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
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
  --per_device_train_batch_size "${OPTAB_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "${OPTAB_GRAD_ACCUM:-16}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps "${OPTAB_SAVE_STEPS:-$SAVE_STEPS_DEFAULT}" \
  --save_total_limit "${OPTAB_SAVE_TOTAL_LIMIT:-2}" \
  --learning_rate 2e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing "${OPTAB_GRAD_CHECKPOINTING:-True}" \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none \
  --seed "${OPTAB_SEED:-42}" \
  --data_seed "${OPTAB_SEED:-42}" \
  --trainer_name "$TRAINER_NAME" \
  "${EXTRA_TRAIN_ARGS[@]}"
