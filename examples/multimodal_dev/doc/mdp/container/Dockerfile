# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Container for the Qwen3.5-VL 397B-A17B 32K recipe. The image is aarch64 only, so build it
# on an aarch64 host or with an aarch64 builder:
#   docker build -t qwen35vl-397b-gb300 examples/multimodal_dev/doc/mdp/container
FROM nvcr.io/nvidia/pytorch:26.07-py3

# Parallel compile jobs for Transformer Engine and DeepEP.
ARG MAX_JOBS=32
COPY install_deps.sh /tmp/install_deps.sh
RUN bash /tmp/install_deps.sh && rm /tmp/install_deps.sh
