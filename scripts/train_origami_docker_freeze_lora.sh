#!/usr/bin/env bash
# T-Rex Origami post-training: freeze the VLM and train Action-LoRA.
# This is a mode-specific wrapper around train_origami_docker.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${TREX_CHECKPOINT_DIR:=/mnt/checkpoints/T-Rex-origami-freeze-lora}"
: "${TREX_EXPERIMENT_NAME:=t-rex_origami_65d_freeze_vlm}"
: "${TREX_BATCH_SIZE:=1}"
export TREX_CHECKPOINT_DIR TREX_EXPERIMENT_NAME TREX_BATCH_SIZE
export TREX_FREEZE_VLM=1
export TREX_ACTION_LORA=1

exec "${SCRIPT_DIR}/train_origami_docker.sh" "$@"
