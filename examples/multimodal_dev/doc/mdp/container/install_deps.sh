#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Installs the software stack for the Qwen3.5-VL 397B-A17B 32K recipe on top of
# nvcr.io/nvidia/pytorch:26.07-py3 (aarch64). Run as root inside that image.
# The build does not need a GPU. rdma-core and DeepEP sources stay under BUILD_DIR,
# because DeepEP loads libraries from rdma-core and JIT-compiles HybridEP kernels at runtime.
set -euxo pipefail

BUILD_DIR="${BUILD_DIR:-/opt/build}"
export MAX_JOBS="${MAX_JOBS:-32}"
mkdir -p "$BUILD_DIR"

# --no-deps everywhere keeps the torch stack shipped in the base image unchanged.
# The base image's pip constraints would reject the newer pins below.
export PIP_CONSTRAINT=
PIP="pip install --no-cache-dir --no-deps"
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# 1. Python packages pinned to the validated versions.
$PIP \
    nvidia-cudnn-cu13==9.25.1.1 \
    nvidia-cudnn-frontend==1.28.0 \
    nvidia-cublas==13.6.1.10 \
    nvidia-cuda-nvrtc==13.3.33 \
    nvidia-cuda-nvdisasm==13.3.73 \
    nvidia-mathdx==25.1.1 \
    nvidia-cutlass-dsl==4.7.0 \
    nvidia-cutlass-dsl-libs-base==4.7.0 \
    nvidia-cutlass-dsl-libs-core==4.7.0 \
    nvidia-cutlass-dsl-libs-cu12==4.7.0 \
    nvidia-cutlass-dsl-libs-cu13==4.7.0 \
    flash-attn-4==4.0.0b29 \
    quack-kernels==0.6.4 \
    torch_c_dlpack_ext==0.1.5 \
    transformers==4.57.1 \
    tokenizers==0.22.2 \
    huggingface_hub==0.36.2 \
    protobuf==6.33.6 \
    grpcio-tools==1.81.0 \
    sentencepiece==0.2.2 \
    tiktoken==0.14.0 \
    ftfy==6.3.1 \
    nltk==3.10.3 \
    joblib==1.5.3 \
    wrapt==2.3.0 \
    pynvml==13.0.1 \
    dm-tree==0.1.10 \
    hydra-core==1.3.5 \
    modelcards==0.1.6 \
    pytz==2026.3.post1 \
    toml==0.10.2 \
    tomlkit==0.15.1 \
    Flask==3.1.3 \
    Flask-RESTful==0.3.10 \
    aniso8601==10.0.1 \
    blinker==1.9.0 \
    itsdangerous==2.2.0 \
    py-spy==0.4.2 \
    torch-tb-profiler==0.4.3

# 2. Packages built from source. TE is built against the PyPI cuDNN installed above.
$PIP "git+https://github.com/fla-org/flash-linear-attention.git@6b6f546f9bd10a97be1e8cf2f0120de70cfca482"
$PIP --no-build-isolation causal_conv1d==1.7.0
$PIP --no-build-isolation "git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.3.1"
$PIP --no-build-isolation "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0"

git clone https://github.com/NVIDIA/TransformerEngine.git "$BUILD_DIR/TransformerEngine"
git -C "$BUILD_DIR/TransformerEngine" checkout 5f6105b900778068144f1b24e25a2e84066983cb
git -C "$BUILD_DIR/TransformerEngine" submodule update --init --recursive
(
    cd "$BUILD_DIR/TransformerEngine"
    NVTE_FRAMEWORK=pytorch \
    NVTE_CUDA_ARCHS="100a;103a" \
    CUDNN_PATH="$SITE_PACKAGES/nvidia/cudnn" \
    $PIP --no-build-isolation .
)
rm -rf "$BUILD_DIR/TransformerEngine"

# 3. rdma-core v60.0 and DeepEP with the HybridEP multinode backend.
apt-get update
apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config \
    libudev-dev libnl-3-dev libnl-route-3-dev python3-docutils
git clone --branch v60.0 --depth 1 https://github.com/linux-rdma/rdma-core.git "$BUILD_DIR/rdma-core"
(cd "$BUILD_DIR/rdma-core" && bash build.sh)

git clone https://github.com/deepseek-ai/DeepEP.git "$BUILD_DIR/DeepEP"
git -C "$BUILD_DIR/DeepEP" checkout 58397eac493b075b0baea5861ac6e9793d8d7a21
(
    cd "$BUILD_DIR/DeepEP"
    HYBRID_EP_MULTINODE=1 \
    RDMA_CORE_HOME="$BUILD_DIR/rdma-core/build" \
    TORCH_CUDA_ARCH_LIST="10.0;10.3" \
    $PIP --no-build-isolation .
)

rm -rf /var/lib/apt/lists/* /root/.cache
