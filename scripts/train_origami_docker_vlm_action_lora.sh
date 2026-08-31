#!/usr/bin/env bash
# Explicit name for the default Origami VLM-LoRA + Action-LoRA recipe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/train_origami_docker_freeze_lora.sh" "$@"
