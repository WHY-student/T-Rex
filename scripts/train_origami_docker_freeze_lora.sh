#!/usr/bin/env bash
# T-Rex Origami post-training: train VLM-LoRA + Action-LoRA with frozen bases.
# This is a mode-specific wrapper around train_origami_docker.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${TREX_CHECKPOINT_DIR:=/mnt/checkpoints/T-Rex-origami-m3-vlm-action-lora}"
: "${TREX_EXPERIMENT_NAME:=t-rex_origami_65d_vlm_action_lora}"
: "${TREX_BATCH_SIZE:=8}"
: "${TREX_STEPS:=10000}"
: "${TREX_EPOCHS:=1000000}"
: "${TREX_RESUME_CHECKPOINT:=/mnt/checkpoints/T-Rex-origami-posttrain/checkpoint-0-50000}"
: "${TREX_RESUME_GLOBAL_STEP:=50000}"
export TREX_CHECKPOINT_DIR TREX_EXPERIMENT_NAME TREX_BATCH_SIZE TREX_STEPS TREX_EPOCHS
export TREX_RESUME_CHECKPOINT TREX_RESUME_GLOBAL_STEP
export TREX_FREEZE_VLM=1
export TREX_ACTION_LORA=1
export TREX_VLM_LORA=1

exec "${SCRIPT_DIR}/train_origami_docker.sh" "$@"
