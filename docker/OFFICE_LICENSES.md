# T-Rex office image license notices

The image contains the following model and source components. The corresponding
Python package metadata and source notices are also retained in the image under
`/opt/conda/envs/trex/lib/python3.10/site-packages/*.dist-info` where supplied
by the package.

- T-Rex source and model adapter: MIT License, `/opt/trex/LICENSE`.
- Qwen3-VL-2B-Instruct base model: Apache License 2.0, as identified by the
  bundled base-model README and the Qwen model repository.
- Eclipse Zenoh Python binding: Eclipse Public License 2.0 / Apache License
  2.0.
- MessagePack Python binding: Apache License 2.0.
- NumPy: BSD 3-Clause License.
- PyTorch and torchvision: BSD-style licenses as included in their package
  metadata.
- Transformers, safetensors, tokenizers, and related Hugging Face packages:
  Apache License 2.0 or the license recorded in each package's metadata.
- Pillow: HPND License.
- OpenCV Python bindings: Apache License 2.0.

The NVIDIA CUDA base image and its runtime libraries retain the licensing terms
of the upstream `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` image. This image
does not contain North SDK code, robot credentials, robot topic names, or
remote-observation credentials.
