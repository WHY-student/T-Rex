# T-Rex office-inference submission image

`Dockerfile.office` builds the self-contained `origami-zenoh-v1` policy image
for the T-Rex checkpoint:

```text
checkpoints/T-Rex-origami-posttrain/
  t-rex_origami_65d_freeze_vlm/
  t-rex_origami_65d_freeze_vlm_2x4090_3epoch_resume60000_0729_092738/
  checkpoint-0-73000/
```

The image uses the same Python 3.10 `trex` conda environment and pinned
PyTorch 2.6.0/cu124 dependencies as the T-Rex training image. It contains the
Qwen3-VL-2B base weights, the post-train checkpoint, tokenizer/processor,
normalization statistics, T-Rex inference code, Zenoh, and the fixed entrypoint.
Nothing is mounted at evaluation time.

## Build

Run from the `T-Rex` directory or use the absolute script path:

```bash
../submission/build_office.sh fold-the-world/origami-policy:submission
```

Export the built office image as the submission archive. The default output
name follows the complete participant guide and includes the `-submission`
suffix:

```bash
../submission/package_office.sh \
  fold-the-world/origami-policy:submission
```

The script runs `docker save`, tests the compressed archive, and writes the
matching `.sha256` checksum file. Each run creates a timestamped directory under
the repository's `submissions/` directory and also writes the generated submission
Markdown and `submission-manifest.json` there.

The build script uses BuildKit named contexts so the large base model and
checkpoint are copied into the image without making the whole workspace the
Docker build context. Override `TREX_CHECKPOINT`, `TREX_BASE_MODEL`, or
`TREX_STATS_FILE` when building on another machine.

The fixed policy horizon is 16, matching `action_chunk=16` in this checkpoint.

## Run and validate

The organizer supplies only the two protocol variables. No port is exposed:

```bash
docker run --rm --gpus all \
  --network origami-eval \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --shm-size 8g \
  -e ORIGAMI_ZENOH_ENDPOINT=tcp/origami-router:7447 \
  -e ORIGAMI_SESSION_ID=local-test \
  fold-the-world/origami-policy:submission
```

With the public test router running, validate the image using the office SDK:

```bash
cd ../office_inference/sharpa_north_ces_lite_sdk-main
python examples/check_zenoh_policy.py \
  --endpoint tcp/127.0.0.1:7447 \
  --session-id local-test \
  --timeout 180 \
  --requests 3 \
  --expected-horizon 16
```

T-Rex was trained with `head_left`, `wrist_right`, and `wrist_left`. The
adapter accepts the complete office observation and uses those three streams
in the same order; `head_right`, torque, and optional raw tactile pixels are
validated at the protocol boundary but are not consumed by this checkpoint.

The policy adapter uses a protocol-transparent approximate slow/fast schedule:
the first request and every fourth `infer` request run the six-step visual/action
slow segment followed by the four-step tactile segment; the three intervening
requests reuse the cached slow state and run only the tactile segment. This does
not add a public protocol field or queryable. Set `TREX_APPROX_ASYNC=0` to run
the full slow-plus-fast path on every request, or override
`TREX_SLOW_EVERY_N` when the organizer's request cadence is known.

The expected runtime is one CUDA GPU with enough memory for the Qwen3-VL
backbone and the T-Rex checkpoint. The first startup performs model loading and
one warm-up inference before declaring Zenoh queryables ready.
