#!/usr/bin/env bash
# Portable single-node T-Rex Origami post-training launcher.
#
# The dataset, base model, resume checkpoint and output directory are external
# mounts. The image only contains the code and the trex/CUDA 12.4 environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT="${TREX_DATASET_ROOT:-}"
MODEL_PATH="${TREX_MODEL_PATH:-/mnt/models/Qwen3-VL-2B-Instruct}"
RESUME_CHECKPOINT="${TREX_RESUME_CHECKPOINT:-/mnt/checkpoints/T-Rex-midTrain}"
DEFORM_ENCODER_PATH="${TREX_DEFORM_ENCODER_PATH:-}"
VQVAE_CKPT="${TREX_VQVAE_CKPT:-}"
OUTPUT_ROOT="${TREX_CHECKPOINT_DIR:-/mnt/checkpoints/T-Rex-origami-posttrain}"
LOG_ROOT="${TREX_LOG_DIR:-${OUTPUT_ROOT}/logs}"
LOG_ROOT_SET=0
[[ -n "${TREX_LOG_DIR:-}" ]] && LOG_ROOT_SET=1
GPU_IDS="${TREX_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
BATCH_SIZE="${TREX_BATCH_SIZE:-1}"
STEPS="${TREX_STEPS:-50000}"
EPOCHS="${TREX_EPOCHS:-3}"
EPOCHS_SET=0
[[ -n "${TREX_EPOCHS:-}" ]] && EPOCHS_SET=1
SAVE_STEPS="${TREX_SAVE_STEPS:-1000}"
MAX_CKPTS="${TREX_MAX_CKPTS:-5}"
GRAD_ACCUM="${TREX_GRAD_ACCUM:-1}"
NUM_WORKERS="${TREX_NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${TREX_VAL_NUM_WORKERS:-2}"
VAL_RATIO="${TREX_VAL_RATIO:-0.01}"
VAL_FREQ="${TREX_VAL_FREQ:-1000}"
MAX_VAL_BATCHES="${TREX_MAX_VAL_BATCHES:-10}"
FREEZE_VLM="${TREX_FREEZE_VLM:-1}"
ACTION_LORA="${TREX_ACTION_LORA:-1}"
ACTION_LORA_RANK="${TREX_ACTION_LORA_RANK:-16}"
ACTION_LORA_ALPHA="${TREX_ACTION_LORA_ALPHA:-32}"
ACTION_LORA_DROPOUT="${TREX_ACTION_LORA_DROPOUT:-0.05}"
OFFLOAD_OPTIMIZER_DEVICE="${TREX_OFFLOAD_OPTIMIZER_DEVICE:-cpu}"
EXPERIMENT_NAME="${TREX_EXPERIMENT_NAME:-t-rex_origami_65d_freeze_vlm}"
RUN_NAME="${TREX_RUN_NAME:-}"
RESUME_GLOBAL_STEP="${TREX_RESUME_GLOBAL_STEP:-0}"
SKIP_PREFLIGHT=0
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage:
  train_origami_docker.sh --dataset-root DATASET_ROOT [options]

Required:
  --dataset-root PATH       Root containing season_*/lerobot3.0

Common options:
  --model-path PATH         Local Qwen3-VL-2B-Instruct directory
  --resume-checkpoint PATH  T-Rex mid-train directory or model.pt
  --deform-encoder PATH     Defaults to RESUME_CHECKPOINT/deform_encoder_from_model_pt.pth
  --checkpoint-dir PATH     Base output directory for checkpoints
  --gpus IDS                all or a list visible inside the process, e.g. 0,1
  --batch-size N             Per-GPU micro-batch size
  --steps N                  Optimizer steps; 0 disables step limit
  --epochs N                 Epoch limit when --steps 0 (or an explicit upper bound)
  --save-steps N             Checkpoint interval
  --grad-accum N             Gradient accumulation steps
  --full-vlm                 Train the VLM instead of freeze+Action-LoRA
  --skip-preflight           Skip dataset/video preflight checks
  --dry-run                  Print the final Accelerate command and exit
USAGE
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root|--data-root) need_value "$1" "$2"; DATASET_ROOT="$2"; shift 2 ;;
        --model-path) need_value "$1" "$2"; MODEL_PATH="$2"; shift 2 ;;
        --resume-checkpoint) need_value "$1" "$2"; RESUME_CHECKPOINT="$2"; shift 2 ;;
        --deform-encoder) need_value "$1" "$2"; DEFORM_ENCODER_PATH="$2"; shift 2 ;;
        --vqvae-ckpt) need_value "$1" "$2"; VQVAE_CKPT="$2"; shift 2 ;;
        --checkpoint-dir|--output-dir) need_value "$1" "$2"; OUTPUT_ROOT="$2"; shift 2 ;;
        --log-dir) need_value "$1" "$2"; LOG_ROOT="$2"; LOG_ROOT_SET=1; shift 2 ;;
        --gpus|--visible-gpus) need_value "$1" "$2"; GPU_IDS="$2"; shift 2 ;;
        --batch-size) need_value "$1" "$2"; BATCH_SIZE="$2"; shift 2 ;;
        --steps) need_value "$1" "$2"; STEPS="$2"; shift 2 ;;
        --epochs) need_value "$1" "$2"; EPOCHS="$2"; EPOCHS_SET=1; shift 2 ;;
        --save-steps) need_value "$1" "$2"; SAVE_STEPS="$2"; shift 2 ;;
        --max-ckpts) need_value "$1" "$2"; MAX_CKPTS="$2"; shift 2 ;;
        --grad-accum) need_value "$1" "$2"; GRAD_ACCUM="$2"; shift 2 ;;
        --num-workers) need_value "$1" "$2"; NUM_WORKERS="$2"; shift 2 ;;
        --val-num-workers) need_value "$1" "$2"; VAL_NUM_WORKERS="$2"; shift 2 ;;
        --val-ratio) need_value "$1" "$2"; VAL_RATIO="$2"; shift 2 ;;
        --val-freq) need_value "$1" "$2"; VAL_FREQ="$2"; shift 2 ;;
        --max-val-batches) need_value "$1" "$2"; MAX_VAL_BATCHES="$2"; shift 2 ;;
        --freeze-vlm) need_value "$1" "$2"; FREEZE_VLM="$2"; shift 2 ;;
        --full-vlm) FREEZE_VLM=0; ACTION_LORA=0; shift ;;
        --action-lora) need_value "$1" "$2"; ACTION_LORA="$2"; shift 2 ;;
        --action-lora-rank) need_value "$1" "$2"; ACTION_LORA_RANK="$2"; shift 2 ;;
        --action-lora-alpha) need_value "$1" "$2"; ACTION_LORA_ALPHA="$2"; shift 2 ;;
        --action-lora-dropout) need_value "$1" "$2"; ACTION_LORA_DROPOUT="$2"; shift 2 ;;
        --offload-optimizer-device) need_value "$1" "$2"; OFFLOAD_OPTIMIZER_DEVICE="$2"; shift 2 ;;
        --experiment-name) need_value "$1" "$2"; EXPERIMENT_NAME="$2"; shift 2 ;;
        --run-name) need_value "$1" "$2"; RUN_NAME="$2"; shift 2 ;;
        --resume-global-step) need_value "$1" "$2"; RESUME_GLOBAL_STEP="$2"; shift 2 ;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done

[[ -n "$DATASET_ROOT" ]] || die "--dataset-root is required"
if (( LOG_ROOT_SET == 0 )); then
    LOG_ROOT="${OUTPUT_ROOT}/logs"
fi

is_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
for value_name in BATCH_SIZE STEPS EPOCHS SAVE_STEPS MAX_CKPTS GRAD_ACCUM NUM_WORKERS VAL_NUM_WORKERS VAL_FREQ MAX_VAL_BATCHES RESUME_GLOBAL_STEP; do
    value="${!value_name}"
    is_integer "$value" || die "$value_name must be a non-negative integer, got '$value'"
done
(( BATCH_SIZE > 0 )) || die "batch size must be > 0"
(( GRAD_ACCUM > 0 )) || die "gradient accumulation must be > 0"
(( NUM_WORKERS >= 0 && VAL_NUM_WORKERS >= 0 )) || die "worker counts must be >= 0"
(( EPOCHS_SET == 1 || STEPS > 0 )) || die "set --steps > 0 or explicitly provide --epochs"

if [[ -z "$DEFORM_ENCODER_PATH" && -n "$RESUME_CHECKPOINT" ]]; then
    DEFORM_ENCODER_PATH="${RESUME_CHECKPOINT%/}/deform_encoder_from_model_pt.pth"
fi
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="${EXPERIMENT_NAME}_$(date +%m%d_%H%M%S)"
fi
# When the user controls the run by optimizer steps, do not let a small epoch
# value terminate the loop before max_steps is reached.
if (( STEPS > 0 && EPOCHS_SET == 0 )); then
    EPOCHS=1000000
fi

if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" == "trex" ]]; then
    TREX_EXEC=()
elif command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | awk '{print $1}' | grep -qx trex; then
    TREX_EXEC=(conda run --no-capture-output -n trex)
else
    die "the Python environment must be the conda environment named 'trex'"
fi

if [[ "$GPU_IDS" == "all" ]]; then
    unset CUDA_VISIBLE_DEVICES
    ACCELERATE_GPU_IDS=all
    GPU_COUNT=$("${TREX_EXEC[@]}" python -c 'import torch; print(torch.cuda.device_count())')
else
    [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--gpus must be 'all' or a comma-separated list such as 0,1"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    ACCELERATE_GPU_IDS=all
    IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
    GPU_COUNT="${#GPU_ARRAY[@]}"
fi
(( GPU_COUNT > 0 )) || die "no visible CUDA devices"

TORCH_GPU_COUNT=$("${TREX_EXEC[@]}" python -c 'import torch; print(torch.cuda.device_count())')
(( TORCH_GPU_COUNT == GPU_COUNT )) || die "requested $GPU_COUNT GPU process(es), but trex sees $TORCH_GPU_COUNT CUDA device(s)"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/lerobot/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export TREX_VIDEO_BACKEND="${TREX_VIDEO_BACKEND:-pyav}"
export TREX_NUM_WORKERS="$NUM_WORKERS"
export TREX_VAL_NUM_WORKERS="$VAL_NUM_WORKERS"
export HF_HOME="${HF_HOME:-/mnt/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/cache/torch_extensions}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29523}"

for directory in "$OUTPUT_ROOT" "$LOG_ROOT" "$HF_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR"; do
    mkdir -p "$directory"
done

[[ -d "$DATASET_ROOT" ]] || die "dataset root does not exist: $DATASET_ROOT"
[[ -d "$MODEL_PATH" && -f "$MODEL_PATH/config.json" ]] || die "model path must contain config.json: $MODEL_PATH"
if ! compgen -G "$MODEL_PATH/*.safetensors" >/dev/null && [[ ! -f "$MODEL_PATH/pytorch_model.bin" ]]; then
    die "model path has no safetensors or pytorch_model.bin weights: $MODEL_PATH"
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    if [[ -d "$RESUME_CHECKPOINT" ]]; then
        [[ -f "$RESUME_CHECKPOINT/model.pt" ]] || die "resume directory lacks model.pt: $RESUME_CHECKPOINT"
    elif [[ ! -f "$RESUME_CHECKPOINT" ]]; then
        die "resume checkpoint does not exist: $RESUME_CHECKPOINT"
    fi
fi
[[ "$FREEZE_VLM" == "0" || "$FREEZE_VLM" == "1" ]] || die "--freeze-vlm must be 0 or 1"
[[ "$ACTION_LORA" == "0" || "$ACTION_LORA" == "1" ]] || die "--action-lora must be 0 or 1"
if [[ -n "$DEFORM_ENCODER_PATH" && ! -f "$DEFORM_ENCODER_PATH" ]]; then
    die "deform encoder checkpoint does not exist: $DEFORM_ENCODER_PATH"
fi
if [[ -n "$VQVAE_CKPT" && ! -f "$VQVAE_CKPT" ]]; then
    die "VQ-VAE checkpoint does not exist: $VQVAE_CKPT"
fi

if (( SKIP_PREFLIGHT == 0 )); then
    "${TREX_EXEC[@]}" python "$PROJECT_ROOT/scripts/check_trex_dataset.py" \
        --dataset-root "$DATASET_ROOT" --check-video
fi

TRAIN_ARGS=(
    --model_path "$MODEL_PATH"
    --data_format lerobot
    --lerobot_root "$DATASET_ROOT"
    --lerobot_repo_id origami/local
    --n_epochs "$EPOCHS"
    --max_steps "$STEPS"
    --save_freq 1
    --save_steps "$SAVE_STEPS"
    --max_ckpts "$MAX_CKPTS"
    --action_dim 65
    --action_chunk 16
    --train_bsz_per_gpu "$BATCH_SIZE"
    --learning_rate 1e-4
    --min_lr_ratio 0
    --warmup_rates 0.03
    --weight_decay 0
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --gradient_checkpointing 1
    --offload_optimizer_device "$OFFLOAD_OPTIMIZER_DEVICE"
    --output_dir "$OUTPUT_ROOT"
    --log_dir "$LOG_ROOT"
    --experiment_name "$EXPERIMENT_NAME"
    --run_name "$RUN_NAME"
    --freeze_vlm "$FREEZE_VLM"
    --action_lora "$ACTION_LORA"
    --action_lora_rank "$ACTION_LORA_RANK"
    --action_lora_alpha "$ACTION_LORA_ALPHA"
    --action_lora_dropout "$ACTION_LORA_DROPOUT"
    --use_robot_state 1
    --use_tactile_vec 1
    --use_tactile_deform 1
    --use_tactile_vqvae 1
    --tactile_intermediate_size 1536
    --training_stage 2
    --tactile_loss_weight 1.0
    --cascaded_total_steps 10
    --cascaded_split_step 6
    --cascaded_tactile_dropout 0.1
    --cascaded_loss_weight 1.0
    --resume_source midtrain
    --resume_global_step "$RESUME_GLOBAL_STEP"
    --use_flare 1
    --n_flare_tokens_per_frame 4
    --n_flare_steps 8
    --flare_loss_weight 0.5
    --flare_frame_stride 4
    --flare_layer_index -1
    --image_size 384 288
    --video_backend pyav
    --video_tolerance_s 0.001
    --num_workers "$NUM_WORKERS"
    --val_num_workers "$VAL_NUM_WORKERS"
    --val_ratio "$VAL_RATIO"
    --val_freq "$VAL_FREQ"
    --max_val_batches "$MAX_VAL_BATCHES"
)
if [[ -n "$DEFORM_ENCODER_PATH" ]]; then
    TRAIN_ARGS+=(--deform_encoder_ckpt "$DEFORM_ENCODER_PATH")
fi
if [[ -n "$VQVAE_CKPT" ]]; then
    TRAIN_ARGS+=(--vqvae_ckpt "$VQVAE_CKPT")
fi
if [[ -n "$RESUME_CHECKPOINT" ]]; then
    TRAIN_ARGS+=(--resume_checkpoint "$RESUME_CHECKPOINT")
else
    [[ -n "$VQVAE_CKPT" ]] || die "no resume checkpoint: provide --vqvae-ckpt"
fi

CMD=(
    "${TREX_EXEC[@]}" accelerate launch
    --config_file "$PROJECT_ROOT/config/sft_qwen.yaml"
    --num_processes "$GPU_COUNT"
    --gpu_ids "$ACCELERATE_GPU_IDS"
    --num_machines 1
    --machine_rank 0
    --main_process_ip "$MASTER_ADDR"
    --main_process_port "$MASTER_PORT"
    --deepspeed_multinode_launcher standard
    "$PROJECT_ROOT/scripts/train_origami_freeze_vlm.py"
    "${TRAIN_ARGS[@]}"
)

echo "T-Rex launch summary:"
echo "  dataset:      $DATASET_ROOT (season_*/lerobot3.0 only)"
echo "  model:        $MODEL_PATH"
echo "  resume:       ${RESUME_CHECKPOINT:-<none>}"
echo "  GPUs:         $GPU_IDS ($GPU_COUNT process(es))"
echo "  batch/GPU:    $BATCH_SIZE; grad_accum: $GRAD_ACCUM"
echo "  steps/epochs: $STEPS / $EPOCHS"
echo "  checkpoints:  $OUTPUT_ROOT/$EXPERIMENT_NAME/$RUN_NAME"
echo "  video:        pyav"

if (( DRY_RUN == 1 )); then
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
