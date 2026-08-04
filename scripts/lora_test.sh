#!/usr/bin/env bash
# Strict launcher for the local 65-D Origami Action-LoRA checkpoint.
set -euo pipefail

FOLDVLA_ROOT="${FOLDVLA_ROOT:-/data/why/foldVLA}"
TREX_PYTHON="${TREX_PYTHON:-/data/why/.conda/envs/trex/bin/python}"
TREX_LORA_CHECKPOINT="${TREX_LORA_CHECKPOINT:-${FOLDVLA_ROOT}/checkpoints/T-Rex-origami-posttrain/t-rex_origami_65d_freeze_vlm/t-rex_origami_65d_freeze_vlm_2x4090_3epoch_0702_145407/checkpoint-0-60000}"
TREX_LEROBOT_ROOT="${TREX_LEROBOT_ROOT:-${FOLDVLA_ROOT}/dataset/lerobot3.0}"
TREX_CUDA_INDEX="${TREX_CUDA_INDEX:-0}"
TREX_SERVER_PORT="${TREX_SERVER_PORT:-5678}"

exec "${TREX_PYTHON}" "${FOLDVLA_ROOT}/T-Rex/scripts/lora_test.py" \
    --checkpoint_path "${TREX_LORA_CHECKPOINT}" \
    --lerobot_root "${TREX_LEROBOT_ROOT}" \
    --cuda "${TREX_CUDA_INDEX}" \
    --port "${TREX_SERVER_PORT}" \
    "$@"
