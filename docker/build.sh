#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${1:-trex:cuda12.4-vlm-lora}"

docker build \
    --file "${SCRIPT_DIR}/Dockerfile" \
    --tag "${IMAGE_NAME}" \
    "${PROJECT_ROOT}"
