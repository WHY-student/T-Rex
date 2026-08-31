#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/data/why/foldVLA/T-Rex
cd "${PROJECT_ROOT}/scripts"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export WANDB_MODE=offline
unset CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29523}
GPU_IDS=${GPU_IDS:-2,3}
SAVE_STEPS=${SAVE_STEPS:-1000}
TRAIN_BSZ_PER_GPU=${TRAIN_BSZ_PER_GPU:-16}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
MAX_STEPS=${MAX_STEPS:-0}
RESUME_GLOBAL_STEP=${RESUME_GLOBAL_STEP:-0}
ACTION_LORA_RANK=${ACTION_LORA_RANK:-16}
ACTION_LORA_ALPHA=${ACTION_LORA_ALPHA:-32}
ACTION_LORA_DROPOUT=${ACTION_LORA_DROPOUT:-0.05}
M3_ENABLE=${M3_ENABLE:-0}
M3_VISION_MASK_PROB=${M3_VISION_MASK_PROB:-0.5}
M3_LANGUAGE_MASK_PROB=${M3_LANGUAGE_MASK_PROB:-0.1}
M3_QUERY_MASK_PROB=${M3_QUERY_MASK_PROB:-0.1}
M3_TACTILE_MASK_PROB=${M3_TACTILE_MASK_PROB:-0.1}

ORIGIN_MODEL_PATH=/data/why/foldVLA/checkpoints/Qwen3-VL-2B-Instruct
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-/data/why/foldVLA/checkpoints/T-Rex-midTrain}
DEFORM_ENCODER_PATH=/data/why/foldVLA/checkpoints/T-Rex-midTrain/deform_encoder_from_model_pt.pth
LEROBOT_ROOT=${LEROBOT_ROOT:-/data/why/foldVLA/dataset/Robotic_Origami_Challenge}
OUTPUT_ROOT=/data/why/foldVLA/checkpoints/T-Rex-origami-posttrain

EXPERIMENT_NAME=t-rex_origami_65d_freeze_vlm
RUN_NAME=${RUN_NAME:-"${EXPERIMENT_NAME}_2x4090_3epoch_$(date +%m%d_%H%M%S)"}

conda run --no-capture-output -n trex accelerate launch \
    --config_file ../config/sft_qwen.yaml \
    --num_processes 2 \
    --gpu_ids "${GPU_IDS}" \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    --deepspeed_multinode_launcher standard \
    train_origami_freeze_vlm.py \
    --model_path "${ORIGIN_MODEL_PATH}" \
    --data_format lerobot \
    --lerobot_root "${LEROBOT_ROOT}" \
    --lerobot_repo_id origami/local \
    --n_epochs 3 \
    --max_steps "${MAX_STEPS}" \
    --save_freq 1 \
    --save_steps "${SAVE_STEPS}" \
    --max_ckpts 5 \
    --action_dim 65 \
    --action_chunk 16 \
    --train_bsz_per_gpu "${TRAIN_BSZ_PER_GPU}" \
    --learning_rate 1e-4 \
    --min_lr_ratio 0 \
    --warmup_rates 0.03 \
    --weight_decay 0 \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --gradient_checkpointing 1 \
    --action_lora 1 \
    --action_lora_rank "${ACTION_LORA_RANK}" \
    --action_lora_alpha "${ACTION_LORA_ALPHA}" \
    --action_lora_dropout "${ACTION_LORA_DROPOUT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --log_dir "${OUTPUT_ROOT}" \
    --experiment_name "${EXPERIMENT_NAME}" \
    --run_name "${RUN_NAME}" \
    --freeze_vlm 1 \
    --use_robot_state 1 \
    --use_tactile_vec 1 \
    --use_tactile_deform 1 \
    --use_tactile_vqvae 1 \
    --deform_encoder_ckpt "${DEFORM_ENCODER_PATH}" \
    --tactile_intermediate_size 1536 \
    --training_stage 2 \
    --tactile_loss_weight 1.0 \
    --cascaded_total_steps 10 \
    --cascaded_split_step 6 \
    --cascaded_tactile_dropout 0.1 \
    --cascaded_loss_weight 1.0 \
    --m3_enable "${M3_ENABLE}" \
    --m3_vision_mask_prob "${M3_VISION_MASK_PROB}" \
    --m3_language_mask_prob "${M3_LANGUAGE_MASK_PROB}" \
    --m3_query_mask_prob "${M3_QUERY_MASK_PROB}" \
    --m3_tactile_mask_prob "${M3_TACTILE_MASK_PROB}" \
    --resume_checkpoint "${RESUME_CHECKPOINT}" \
    --resume_global_step "${RESUME_GLOBAL_STEP}" \
    --resume_source midtrain \
    --use_flare 1 \
    --n_flare_tokens_per_frame 4 \
    --n_flare_steps 8 \
    --flare_loss_weight 0.5 \
    --flare_frame_stride 4 \
    --flare_layer_index -1 \
    --image_size 384 288 \
    --val_ratio 0.01 \
    --val_freq 1000 \
    --max_val_batches 10
