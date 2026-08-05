# T-Rex CUDA 12.4 training image

This image packages the T-Rex code, the pinned LeRobot reader, and a conda
environment named `trex` with Python 3.10 and PyTorch 2.6.0/cu124. The dataset,
Qwen base model, mid-train checkpoint, and post-train outputs stay outside the
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

## Run training

The dataset argument can point to the collection root. The launcher scans only
the immediate `season_*/lerobot3.0` directories and concatenates them without
copying data:

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
  trex:cuda12.4 \
  /opt/trex/scripts/train_origami_docker.sh \
    --dataset-root /mnt/data/Robotic_Origami_Challenge \
    --gpus 0,1 \
    --batch-size 8 \
    --steps 50000 \
    --save-steps 1000 \
    --checkpoint-dir /mnt/checkpoints/T-Rex-origami-posttrain
```

`--gpus` is interpreted inside the container. If Docker is started with
`--gpus '"device=2,3"'`, those two cards are renumbered as `0,1` inside the
container. If all host cards are passed through, use `--gpus all` for both
Docker and the launcher.

There are two mode-specific launchers. They use the same image and dataset
mounts, but write to different default checkpoint directories:

```bash
# Freeze VLM + Action-LoRA (default batch size: 1 per GPU)
/opt/trex/scripts/train_origami_docker_freeze_lora.sh \
  --dataset-root /mnt/data/Robotic_Origami_Challenge \
  --gpus 0,1 --batch-size 8 --steps 50000

# Full VLM fine-tuning (default batch size: 1 per GPU)
/opt/trex/scripts/train_origami_docker_full_vlm.sh \
  --dataset-root /mnt/data/Robotic_Origami_Challenge \
  --gpus 0,1 --batch-size 1 --steps 50000
```

The generic `train_origami_docker.sh` remains available when a custom
combination of `--freeze-vlm` and `--action-lora` is needed.

The default mounted paths are:

```text
/mnt/models/Qwen3-VL-2B-Instruct
/mnt/checkpoints/T-Rex-midTrain
/mnt/checkpoints/T-Rex-origami-posttrain
```

Override them with `--model-path`, `--resume-checkpoint`, and
`--checkpoint-dir`. The deform encoder defaults to
`<resume-checkpoint>/deform_encoder_from_model_pt.pth`. The embedded VQ-VAE is
auto-detected from the mid-train checkpoint; use `--vqvae-ckpt` only when
starting without a merged/resume checkpoint.

Useful controls:

```text
--batch-size N       micro-batch per GPU, not the global batch
--steps N            optimizer steps; 0 means use --epochs
--grad-accum N       effective batch = N × GPU count × grad-accum
--save-steps N       checkpoint interval
--max-ckpts N        number of retained checkpoint directories
--num-workers N      video DataLoader workers per GPU process
--full-vlm           disable freeze+Action-LoRA and train the VLM backbone
--dry-run            print the resolved Accelerate command
```

The freeze+Action-LoRA mode is the safer starting point for 24-GB cards.
Full-VLM training normally requires a smaller per-GPU batch and more VRAM;
the separate launcher starts conservatively with batch size 1.

## Preflight without starting training

```bash
docker run --rm -it --gpus '"device=0"' \
  -v /new-machine/data/Robotic_Origami_Challenge:/mnt/data/Robotic_Origami_Challenge:ro \
  trex:cuda12.4 \
  python /opt/trex/scripts/check_trex_dataset.py \
    --dataset-root /mnt/data/Robotic_Origami_Challenge --check-video
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
