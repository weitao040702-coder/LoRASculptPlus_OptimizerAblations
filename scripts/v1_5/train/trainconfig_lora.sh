#!/bin/bash


# 默认参数
: ${DEVICE:=localhost:0,1,2,3}
: ${MASTER_PORT:=29600}
: ${LORA_RANK:=32}
: ${LORA_ALPHA:=64}
: ${CONNECTOR_LORA_RANK:=128}
: ${CONNECTOR_LORA_ALPHA:=64}
: ${PER_DEVICE_TRAIN_BATCH_SIZE:=4}
: ${NUM_TRAIN_EPOCHS:=3}
: ${TRAINER_NAME:="LoRASculpt"}
: ${DATASET_NAME:=""}
: ${LEARNING_RATE:=2e-4}
: ${OUTPUT_DIR:=""}

: ${GRADIENT_ACCUMULATION_STEPS:=1}
: ${MAX_STEPS:=}
: ${MODEL_NAME_OR_PATH:="${LORASCULPT_ROOT:-$(pwd)}/assets/models/llava-v1.5-7b"}
: ${DEEPSPEED_ZEROFILE:="${LORASCULPT_ROOT:-$(pwd)}/scripts/zero2.json"}


if [ "$DATASET_NAME" == "iconqa_txt" ]; then
    data_path="${ICONQA_TRAIN_DATA:-${LORASCULPT_ROOT:-$(pwd)}/assets/data/iconqa/iconqa_txt-train.json}"
    image_folder="${ICONQA_IMAGE_ROOT:-${LORASCULPT_ROOT:-$(pwd)}/assets/data/iconqa/raw/data}"
elif [ "$DATASET_NAME" == "coco" ]; then
    data_path="${LORASCULPT_ROOT:-$(pwd)}/assets/data/coco/coco-train.json"
    image_folder="${LORASCULPT_ROOT:-$(pwd)}/assets/data/coco"
else
    echo "Unsupported DATASET_NAME: $DATASET_NAME"
    exit 1
fi



# DeepSpeed's launcher interprets Slurm's physical CUDA_VISIBLE_DEVICES values
# as host slots. On this cluster, an allocation such as 2,3 therefore fails.
# torch.distributed.run preserves the Slurm mapping and assigns local ranks
# relative to the allocated devices.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    CUDA_DEVICE_LIST="${CUDA_VISIBLE_DEVICES:-}"
    if [[ -z "$CUDA_DEVICE_LIST" ]]; then
        echo "CUDA_VISIBLE_DEVICES is empty inside Slurm job" >&2
        exit 1
    fi
    IFS=',' read -r -a CUDA_DEVICE_IDS <<< "$CUDA_DEVICE_LIST"
    GPU_COUNT=${#CUDA_DEVICE_IDS[@]}
    TRAIN_LAUNCHER=(python -m torch.distributed.run --nproc_per_node="$GPU_COUNT" --master_addr=127.0.0.1 --master_port="$MASTER_PORT")
    echo "launcher=torch.distributed.run nproc_per_node=$GPU_COUNT CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_LIST"
else
    TRAIN_LAUNCHER=(deepspeed --include "$DEVICE" --master_port "$MASTER_PORT")
    echo "launcher=deepspeed include=$DEVICE"
fi

TRAIN_ARGS=(
    --lora_enable True
    --lora_r "$LORA_RANK"
    --lora_alpha "$LORA_ALPHA"
    --connector_lora_r "$CONNECTOR_LORA_RANK"
    --connector_lora_alpha "$CONNECTOR_LORA_ALPHA"
    --mm_projector_lr 2e-5
    --deepspeed "$DEEPSPEED_ZEROFILE"
    --model_name_or_path "$MODEL_NAME_OR_PATH"
    --version v1
    --data_path "$data_path"
    --image_folder "$image_folder"
    --vision_tower "${LORASCULPT_ROOT:-$(pwd)}/assets/models/clip-vit-large-patch14-336"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 True
    --output_dir "$OUTPUT_DIR"
    --overwrite_output_dir True
    --num_train_epochs "$NUM_TRAIN_EPOCHS"
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
    --evaluation_strategy no
    --save_strategy steps
    --save_steps 10000
    --save_total_limit 15
    --learning_rate "$LEARNING_RATE"
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --tf32 True
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to none
    --trainer_name "$TRAINER_NAME"
)

if [[ -n "$MAX_STEPS" ]]; then
    TRAIN_ARGS+=(--max_steps "$MAX_STEPS")
fi

"${TRAIN_LAUNCHER[@]}" llava/train/train_mem.py "${TRAIN_ARGS[@]}"
