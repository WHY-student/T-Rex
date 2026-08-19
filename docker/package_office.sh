#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-fold-the-world/origami-policy:submission}"
ARCHIVE="${2:-fold-the-world-origami-policy-submission.tar.zst}"

if [[ "${ARCHIVE}" != *.tar.zst ]]; then
    printf 'Archive name must end with .tar.zst: %s\n' "${ARCHIVE}" >&2
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    printf 'docker command not found\n' >&2
    exit 1
fi
if ! command -v zstd >/dev/null 2>&1; then
    printf 'zstd command not found\n' >&2
    exit 1
fi
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    printf 'Docker image not found: %s\n' "${IMAGE_NAME}" >&2
    exit 1
fi

PARTIAL="${ARCHIVE}.partial"

# docker save preserves the image configuration, entrypoint, and layer metadata.
docker save "${IMAGE_NAME}" \
    | zstd -T0 -3 -f -o "${PARTIAL}"
mv -- "${PARTIAL}" "${ARCHIVE}"

zstd -t "${ARCHIVE}"
sha256sum "${ARCHIVE}" | tee "${ARCHIVE}.sha256"

printf 'Image ID: '
docker image inspect --format '{{.Id}}' "${IMAGE_NAME}"
printf 'Archive: %s\n' "${ARCHIVE}"
