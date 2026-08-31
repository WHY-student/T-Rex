# T-Rex CUDA 12.4 training image

This image packages the T-Rex code, the pinned LeRobot reader, and a conda
environment named `trex` with Python 3.10 and PyTorch 2.6.0/cu124. The dataset,
Qwen base model, resume checkpoint, and post-train outputs stay outside the
image and are mounted at runtime.

## Build on the new machine

The build context must be the `T-Rex` directory so the pinned
`third_party/lerobot` checkout is included:

```bash
cd /path/to/foldVLA/T-Rex
docker pull nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 # 先拉去对应nvidia镜像
docker build -f docker/Dockerfile -t trex:cuda12.4 .
# or: ./docker/build.sh trex:cuda12.4
```

If the image is built on a connected machine and transferred to the new host:

```bash
docker save trex:cuda12.4 | gzip > trex-cuda12.4.tar.gz
# on the new host:
gzip -dc trex-cuda12.4.tar.gz | docker load
```

The host needs a working NVIDIA driver, Docker, and NVIDIA Container Toolkit.
Check the runtime before building/training:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

The image creates the `trex` environment from the `conda-forge` channel, so
the build does not require accepting Anaconda `pkgs/main` or `pkgs/r` channel
terms.

## Build the office-inference submission image

For the self-contained `origami-zenoh-v1` policy image, use the production
Dockerfile and build script below. It embeds the Qwen base model, the selected
T-Rex checkpoint, tokenizer/processor, normalization statistics, and the
office Zenoh entrypoint; it does not use runtime source or checkpoint mounts.

```bash
../submission/build_office.sh fold-the-world/origami-policy:submission
```

The image uses CUDA 12.4 and the same Python 3.10 `trex` environment. Its
fixed action horizon is 16, matching the checkpoint's `action_chunk=16`.
See [`README_office_inference.md`](README_office_inference.md) for the
runtime and black-box validator commands.

## Run Origami training

The current Origami training entrypoint is
`scripts/train_origami_freeze_vlm.py`. The dataset argument can point either
to the merged LeRobot directory itself (`.../lerobot3.0`, containing
`meta/info.json`) or to its parent collection root. The loader opens each
season independently and discards complete episodes longer than 260 seconds:

```bash
docker run --rm -it \
  --gpus '"device=0,1"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /new-machine/data/Robotic_Origami_Challenge:/mnt/data/Robotic_Origami_Challenge:ro \
  -v /new-machine/models:/mnt/models:ro \
  -v /new-machine/checkpoints:/mnt/checkpoints \
  -v /new-machine/cache:/mnt/cache \
  trex:cuda12.4-vlm-lora \
  /opt/trex/scripts/train_origami_docker_vlm_action_lora.sh \
    --dataset-root /mnt/data/Robotic_Origami_Challenge \
    --gpus 0,1
```

`--gpus` is interpreted inside the container. If Docker is started with
`--gpus '"device=2,3"'`, those two cards are renumbered as `0,1` inside the
container. If all host cards are passed through, use `--gpus all` for both
Docker and the launcher.

The recommended mode-specific launcher trains both latent/text VLM-LoRA and
Action-LoRA. The VLM and action base weights stay frozen; only the adapters and
the existing T-Rex task heads/projections are updated. It defaults to 10000
optimizer steps, M3 masking, and the checkpoint used for this Origami run:

```bash
# VLM-LoRA + Action-LoRA (default batch size: 8 per GPU, 10000 steps)
/opt/trex/scripts/train_origami_docker_freeze_lora.sh \
  --dataset-root /mnt/data/Robotic_Origami_Challenge \
  --gpus 0,1

# The explicit alias has the same configuration:
/opt/trex/scripts/train_origami_docker_vlm_action_lora.sh \
  --dataset-root /mnt/data/Robotic_Origami_Challenge \
  --gpus 0,1

# Optional full-VLM fine-tuning (no LoRA; separate high-memory mode)
/opt/trex/scripts/train_origami_docker_full_vlm.sh \
  --dataset-root /mnt/data/Robotic_Origami_Challenge \
  --gpus 0,1
```

The generic `train_origami_docker.sh` remains available for custom settings.
The resume path can always be supplied at runtime with
`--resume-checkpoint PATH` (or `TREX_RESUME_CHECKPOINT`); it is not baked into
the image.

The default mounted paths are:

```text
/mnt/models/Qwen3-VL-2B-Instruct
/mnt/checkpoints/T-Rex-origami-posttrain/checkpoint-0-50000
/mnt/checkpoints/T-Rex-origami-m3-vlm-action-lora
```

Override them with `--model-path`, `--resume-checkpoint`, and
`--checkpoint-dir`. The deform encoder defaults to
`<resume-checkpoint>/deform_encoder_from_model_pt.pth`. The embedded VQ-VAE is
auto-detected from the mid-train checkpoint; use `--vqvae-ckpt` only when
starting without a merged/resume checkpoint.

Useful controls:

```text
--batch-size N       micro-batch per GPU, not the global batch
--steps N            optimizer steps; 0 means use --epochs (default: 10000)
--epochs N           epoch limit when --steps is 0 (default: 3)
--grad-accum N       effective batch = N × GPU count × grad-accum
--save-steps N       checkpoint interval
--max-ckpts N        number of retained checkpoint directories
--num-workers N      video DataLoader workers per GPU process
--vlm-lora N         train latent/text VLM-LoRA adapters (default: 1)
--vlm-lora-rank N    VLM-LoRA rank (default: 16)
--action-lora N      train Action-LoRA adapters (default: 1)
--full-vlm           disable both LoRA modes and train the full VLM
--resume-global-step N  starting step for LR/dataloader progress (default wrapper: 50000)
--dry-run            print the resolved Accelerate command
```

`--resume-checkpoint` defaults to
`/mnt/checkpoints/T-Rex-origami-posttrain/checkpoint-0-50000` in the Docker
recipe, but an arbitrary checkpoint directory or `model.pt` can be passed at
runtime. `--resume-global-step 50000` aligns the new run's LR/dataloader
progress with `checkpoint-0-50000`; weight-only checkpoints do not restore
optimizer momentum.

For the second physical RTX 4090 on this host, pass only host GPU 3 to Docker
and use GPU 0 inside the container:

```bash
docker run --rm -it --gpus '"device=3"' --ipc=host \
  -v /data/why/foldVLA/dataset/Robotic_Origami_Challenge:/mnt/data/Robotic_Origami_Challenge:ro \
  -v /data/why/foldVLA/checkpoints:/mnt/checkpoints \
  -v /data/why/foldVLA/.cache/trex-docker:/mnt/cache \
  trex:cuda12.4-vlm-lora \
  /opt/trex/scripts/train_origami_docker_vlm_action_lora.sh \
    --dataset-root /mnt/data/Robotic_Origami_Challenge \
    --model-path /mnt/checkpoints/Qwen3-VL-2B-Instruct \
    --resume-checkpoint /mnt/checkpoints/T-Rex-origami-posttrain/checkpoint-0-50000 \
    --gpus 0
```

## Preflight without starting training

```bash
docker run --rm -it --gpus '"device=0"' \
  -v /new-machine/data/Robotic_Origami_Challenge:/mnt/data/Robotic_Origami_Challenge:ro \
  trex:cuda12.4 \
  python /opt/trex/scripts/check_trex_dataset.py \
    --dataset-root /mnt/data/Robotic_Origami_Challenge/lerobot3.0 --check-video
```

The check verifies every selected season's v3.0 metadata, feature dimensions
(65-D state/action and 60-D tactile), statistics, one parquet file, and one
PyAV-decodable video. Git-LFS pointer files are rejected explicitly.

## New-machine issues already handled

- The old scripts used absolute paths from another host. The container launcher
  takes all paths as arguments or mounted defaults.
- A collection root containing many seasons is now supported. Each season is
  opened as an independent LeRobot dataset, so episode/video indices cannot
  collide. Normalization statistics are aggregated across seasons and saved in
  each checkpoint's `stats_data.json` for later inference.
- Training uses `pyav` explicitly. `torchcodec` is not installed in the image;
  the old host's torchcodec could not load because of an FFmpeg/torch ABI
  mismatch.
- The final partial training batch is dropped so DeepSpeed's configured
  micro-batch size remains fixed. Video worker counts are configurable; set
  `--num-workers 0` if a particular host has a video-decoder/fork issue.
- `--ipc=host` (or a sufficiently large `--shm-size`) is important for several
  video DataLoader workers. A busy `MASTER_PORT` can be changed with
  `-e MASTER_PORT=29524`.
- Checkpoint files are weight-only. Resuming a post-training run does not
  restore optimizer momentum; pass `--resume-global-step` when continuing a
  run so the LR schedule and dataloader position are aligned.
- Do not bake the dataset or multi-gigabyte checkpoints into the image. Keep
  enough space for the mounted output, PyTorch/DeepSpeed extension cache, and
  any retained checkpoints.
- If outputs are created as root and the host needs a different owner, run the
  container with `--user "$(id -u):$(id -g)"` and mount writable cache/output
  directories with matching permissions.
