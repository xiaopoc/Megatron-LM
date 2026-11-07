#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Quick check of the container on one GPU node: versions, cuDNN selection and a small MXFP8 GEMM.
set -euo pipefail
export CUDNN_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn
export LD_LIBRARY_PATH="$CUDNN_HOME/lib:${LD_LIBRARY_PATH:-}"

pip list 2>/dev/null | grep -iE "^(torch |transformer.engine|nvidia-cudnn|flash|deep.ep|causal|transformers|emerging|fast.hadamard|nvidia-cutlass-dsl )"

python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling
layer = te.Linear(4096, 4096, params_dtype=torch.bfloat16).cuda()
x = torch.randn(8192, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
    y = layer(x)
y.sum().backward()
print("TE MXFP8 linear fwd/bwd OK", tuple(y.shape))

import cudnn
print("cudnn frontend", cudnn.__version__, "backend", cudnn.backend_version())

import deep_ep, hybrid_ep_cpp
print("deep_ep and hybrid_ep_cpp import OK")

import fla, causal_conv1d
import flash_attn.cute
print("fla", fla.__version__, "causal_conv1d and flash_attn.cute import OK")
EOF

for lib in /usr/local/lib/python3.12/dist-packages/hybrid_ep_cpp*.so \
           /usr/local/lib/python3.12/dist-packages/transformer_engine/libtransformer_engine.so; do
    echo "== $lib"
    ldd "$lib" | grep -iE "cudnn|ibverbs|mlx5" || true
done
