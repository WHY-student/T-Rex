FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MINICONDA_INSTALLER=Miniconda3-py310_25.11.1-1-Linux-x86_64.sh

ENV TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_DIR=/opt/conda \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TREX_VIDEO_BACKEND=pyav

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ninja-build \
        pkg-config \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://repo.anaconda.com/miniconda/${MINICONDA_INSTALLER}" \
        -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && "${CONDA_DIR}/bin/conda" config --system --set auto_update_conda false \
    && "${CONDA_DIR}/bin/conda" create -y -n trex \
        --override-channels --channel conda-forge python=3.10 \
    && "${CONDA_DIR}/bin/conda" clean -afy

ENV PATH=/opt/conda/envs/trex/bin:/opt/conda/bin:${PATH} \
    CONDA_DEFAULT_ENV=trex \
    CONDA_PREFIX=/opt/conda/envs/trex

WORKDIR /opt/trex
COPY docker/requirements-cu124.txt /tmp/requirements-cu124.txt

# Install the CUDA 12.4 PyTorch wheels explicitly before the project
# dependencies. The project manifest is installed with --no-deps below so a
# later resolver cannot silently replace these wheels with CPU builds.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        torch==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install -r /tmp/requirements-cu124.txt \
    && rm -f /tmp/requirements-cu124.txt

COPY . /opt/trex

# LeRobot is vendored as a pinned git submodule and is imported through
# PYTHONPATH below. This avoids its package metadata trying to upgrade the
# older, pinned T-Rex accelerate/wandb combination. In particular, torchcodec
# is omitted because the training loader is forced to use PyAV.
RUN python -m pip install -e . --no-deps \
    && chmod +x \
        scripts/train_origami_docker.sh \
        scripts/train_origami_docker_freeze_lora.sh \
        scripts/train_origami_docker_vlm_action_lora.sh \
        scripts/train_origami_docker_full_vlm.sh \
        scripts/check_trex_dataset.py

# Keep the image build tied to the current Origami training path.
RUN python -m py_compile \
        qwen_vla/origami_lerobot_dataset.py \
        qwen_vla/m3_masking.py \
        qwen_vla/checkpoint_restore.py \
        scripts/train_origami_freeze_vlm.py \
        scripts/check_trex_dataset.py

ENV PYTHONPATH=/opt/trex:/opt/trex/third_party/lerobot/src \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
    HF_HOME=/mnt/cache/huggingface \
    TORCH_HOME=/mnt/cache/torch \
    TORCH_EXTENSIONS_DIR=/mnt/cache/torch_extensions

RUN python - <<'PY'
import torch
import torchvision
import transformers
import accelerate
import av

assert torch.version.cuda == "12.4", torch.version.cuda
assert torch.__version__.startswith("2.6.0+cu124"), torch.__version__
assert torchvision.__version__.startswith("0.21.0+cu124"), torchvision.__version__
print("T-Rex image check:", torch.__version__, torchvision.__version__, transformers.__version__, accelerate.__version__, av.__version__)
PY

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
