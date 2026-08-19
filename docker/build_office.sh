#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLDVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME="${1:-fold-the-world/origami-policy:submission}"
TREX_ROOT="${TREX_ROOT:-${FOLDVLA_ROOT}/T-Rex}"
TREX_CHECKPOINT="${TREX_CHECKPOINT:-${FOLDVLA_ROOT}/.trex-build-checkpoint-50000/checkpoint-0-50000}"
TREX_BASE_MODEL="${TREX_BASE_MODEL:-${FOLDVLA_ROOT}/checkpoints/Qwen3-VL-2B-Instruct}"
TREX_STATS_FILE="${TREX_STATS_FILE:-${FOLDVLA_ROOT}/dataset/Robotic_Origami_Challenge/lerobot3.0/meta/stats.json}"

for required_path in "${TREX_ROOT}" "${TREX_CHECKPOINT}" "${TREX_BASE_MODEL}" "${TREX_STATS_FILE}"; do
    if [[ ! -e "${required_path}" ]]; then
        printf 'Missing build input: %s\n' "${required_path}" >&2
        exit 1
    fi
done

if [[ ! -f "${TREX_CHECKPOINT}/model.pt" || ! -f "${TREX_CHECKPOINT}/training_args.json" ]]; then
    printf 'Checkpoint is not a T-Rex checkpoint directory: %s\n' "${TREX_CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${TREX_BASE_MODEL}/config.json" || ! -f "${TREX_BASE_MODEL}/model.safetensors" ]]; then
    printf 'Base model is not a complete Qwen3-VL directory: %s\n' "${TREX_BASE_MODEL}" >&2
    exit 1
fi

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

docker build \
    --file "${SCRIPT_DIR}/Dockerfile.office" \
    --tag "${IMAGE_NAME}" \
    --build-context "trex=${TREX_ROOT}" \
    --build-context "base_model=${TREX_BASE_MODEL}" \
    --build-context "checkpoint=${TREX_CHECKPOINT}" \
    --build-context "stats=$(dirname "${TREX_STATS_FILE}")" \
    "${SCRIPT_DIR}"
