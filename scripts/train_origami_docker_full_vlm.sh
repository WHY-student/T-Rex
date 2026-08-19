#!/usr/bin/env bash
# T-Rex Origami post-training: full VLM fine-tuning.
# This is a mode-specific wrapper around train_origami_docker.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${TREX_CHECKPOINT_DIR:=/mnt/checkpoints/T-Rex-origami-full-vlm}"
: "${TREX_EXPERIMENT_NAME:=t-rex_origami_65d_full_vlm}"
: "${TREX_BATCH_SIZE:=8}"
: "${TREX_STEPS:=0}"
: "${TREX_EPOCHS:=3}"
: "${TREX_OFFLOAD_OPTIMIZER_DEVICE:=cpu}"
export TREX_CHECKPOINT_DIR TREX_EXPERIMENT_NAME TREX_BATCH_SIZE TREX_STEPS TREX_EPOCHS TREX_OFFLOAD_OPTIMIZER_DEVICE
export TREX_FREEZE_VLM=0
export TREX_ACTION_LORA=0

exec "${SCRIPT_DIR}/train_origami_docker.sh" --full-vlm "$@"
