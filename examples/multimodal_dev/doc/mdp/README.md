# Qwen3.5-VL 397B-A17B: 28-layer proxy at 32K on 64 GB300 GPUs

This recipe trains the Qwen3.5-VL 397B-A17B vision-language model with Modality-Decoupled
Parallelism (MDP) at a 32K sequence length on a single rack of 64 GB300 GPUs. To fit one rack,
the language model is reduced from 60 to 28 layers. All other dimensions match the full model,
and the vision encoder keeps all 27 layers.

| File | Purpose |
|---|---|
| [`recipes/qwen35_vl_397b_a17b/s32k_pp2ep32_mdp_28L_64gpu.sh`](recipes/qwen35_vl_397b_a17b/s32k_pp2ep32_mdp_28L_64gpu.sh) | Full training command and environment variables |
| [`recipes/qwen35_vl_397b_a17b/launch_s32k_pp2ep32_mdp_28L_64gpu.sbatch`](recipes/qwen35_vl_397b_a17b/launch_s32k_pp2ep32_mdp_28L_64gpu.sbatch) | Example Slurm launcher |
| [`recipes/data/mock_fixed_32768.json`](recipes/data/mock_fixed_32768.json) | Synthetic data configuration |

A second recipe runs the full 60-layer model at a 16K sequence length on 128 GB300 GPUs. See
[Full model at 16K on 128 GB300 GPUs](#full-model-at-16k-on-128-gb300-gpus).

## Reference results

Measured over 20 iterations without a profiler, taking the median of iterations 9 to 20.

| Metric | Value |
|---|---|
| Throughput | about 866 TFLOP/s/GPU |
| Step time | about 41.4 s |
| Tokens | about 12.7K tokens/s/GPU |
| Peak device memory, first pipeline stage | about 259 GiB of 277 GiB |

Tokens per second per GPU is computed as global batch size × sequence length ÷ step time ÷ number
of GPUs. TFLOP/s/GPU is the value Megatron logs as `throughput per GPU`. It counts model FLOPs,
including the vision encoder, and excludes recompute and padding. Iterations 1 to 3 are slower
because of warmup and CUDA graph capture.

## Configuration

- **Parallelism:** TP1, PP2, EP32, CP1, ETP1, with pipeline layout `Et*15|t*13,m,L`. The first
  stage holds the vision encoder, the embedding and 15 decoder layers. The second stage holds
  13 decoder layers, the MTP layer and the output layer.
- **Batch:** global batch size 1024, micro-batch size 1, sequence length 32768.
- **Precision:** MXFP8 for the language model's linear layers, including the expert GEMMs, with
  FP8 parameter all-gather. The vision encoder runs entirely in BF16. The optimizer is
  precision-aware, with FP32 main parameters and gradients and BF16 Adam moments.
- **MoE:** HybridEP token dispatcher, grouped GEMM, router and permutation fusion.
- **Kernels:** cuDNN Gated DeltaNet kernels, TE partial CUDA graphs for `attn`, `moe_router` and
  `moe_preprocess`.
- **Recompute:** selective recompute of `moe_act` and `gdn` in the language model, and whole
  recompute of the vision encoder, which MDP replays during the backward pass.
- **Data:** THD packing. Each micro-batch holds one 32,768-token synthetic sample.

## Requirements

### Hardware

64 GB300 GPUs: 16 nodes with 4 GPUs each. All 16 nodes must be in the same NVL72 NVLink domain,
so that EP32 communication stays on NVLink. Multi-node NVLink must be enabled on the cluster.

### Container

The recipe was validated with the following software stack:

| Component | Version |
|---|---|
| Base image | `nvcr.io/nvidia/pytorch:26.07-py3` |
| PyTorch | 2.13.0a0, CUDA 13.3, Python 3.12 |
| Transformer Engine | 2.20.0.dev0, main branch at commit `5f6105b9` |
| cuDNN | 9.25.1.1, `nvidia-cudnn-cu13` from PyPI |
| cuDNN frontend | 1.28.0 |
| FlashAttention-4 | 4.0.0b29 |
| flash-linear-attention | 0.6.0 |
| DeepEP, HybridEP | 1.2.1 |
| causal-conv1d | 1.7.0 |
| transformers | 4.57.1 |
| nvidia-resiliency-ext | 0.6.0 |

The recipe sets `CUDNN_HOME` and `LD_LIBRARY_PATH` to the cuDNN installed from PyPI, so that this
version is used even if the base image also ships a system cuDNN.

#### Building the container

[`container/install_deps.sh`](container/install_deps.sh) installs the stack above on top of the
base image, and [`container/Dockerfile`](container/Dockerfile) wraps it. The image is aarch64
only. Build it on an aarch64 host, for example a GB300 node:

```bash
cd examples/multimodal_dev/doc/mdp/container
docker build --build-arg MAX_JOBS=64 -t qwen35vl-397b-gb300 .
```

The build needs internet access to PyPI, GitHub and the Ubuntu package mirrors, but no GPU. It
took about 25 minutes on a GB300 node, most of it compiling Transformer Engine. With enroot and
pyxis instead of Docker, run `install_deps.sh` as root inside the base image and save the result
with `--container-save`.

To check the image, run [`container/check_env.sh`](container/check_env.sh) on one GPU inside the
container. It prints the installed versions, runs an MXFP8 linear layer forward and backward
through Transformer Engine, imports DeepEP, HybridEP, flash-linear-attention, causal-conv1d and
FlashAttention-4, and shows which cuDNN and RDMA libraries they link against.

### Hugging Face files

The recipe reads the tokenizer and processor of `Qwen/Qwen3.5-397B-A17B` with `HF_HUB_OFFLINE=1`.
Download them in advance, keeping the standard Hugging Face cache layout, and point `HF_HOME` at
that cache. Model weights are not needed.

## Running

1. Check out this branch.
2. In the example launcher, replace the `<...>` placeholders: account, partition, checkout path,
   container image, Hugging Face cache and container mounts.
3. Bind each rank to the CPU cores and NUMA node closest to its GPU, and give each rank enough
   cores. Set `RANK_LAUNCHER` to the cluster's affinity wrapper followed by `python`.
4. Submit with `sbatch launch_s32k_pp2ep32_mdp_28L_64gpu.sbatch`.

The recipe script does not allocate nodes. It can also be called with `torchrun` instead of
`srun`, once on each of the 16 nodes inside the container, with the launcher as its arguments:

```bash
bash examples/multimodal_dev/doc/mdp/recipes/qwen35_vl_397b_a17b/s32k_pp2ep32_mdp_28L_64gpu.sh \
  torchrun --nnodes=16 --nproc_per_node=4 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
```

This form does not apply CPU and NUMA binding, which still has to be configured separately.

The output directory can be set with `MCORE_BENCH_OUTPUT`.

## Full model at 16K on 128 GB300 GPUs

| File | Purpose |
|---|---|
| [`recipes/qwen35_vl_397b_a17b/s16k_pp2ep64_mdp_60L_128gpu.sh`](recipes/qwen35_vl_397b_a17b/s16k_pp2ep64_mdp_60L_128gpu.sh) | Full training command and environment variables |
| [`recipes/qwen35_vl_397b_a17b/launch_s16k_pp2ep64_mdp_60L_128gpu.sbatch`](recipes/qwen35_vl_397b_a17b/launch_s16k_pp2ep64_mdp_60L_128gpu.sbatch) | Example Slurm launcher |
| [`recipes/data/mock_fixed_16384.json`](recipes/data/mock_fixed_16384.json) | Synthetic data configuration |

This recipe uses all 60 language model layers. Compared with the 32K recipe, only these settings
change:

- **Parallelism:** EP64 instead of EP32, with pipeline layout `Et*32|t*28,m,L`.
- **Batch:** global batch size 2048, sequence length 16384.
- **Hardware:** 128 GB300 GPUs, 32 nodes with 4 GPUs each. Each pipeline stage spans 16 nodes,
  and those 16 nodes must be in the same NVL72 NVLink domain.

The container, Hugging Face files and launch steps are the same as for the 32K recipe.

Measured over 20 iterations, median of iterations 9 to 20:

| Metric | Value |
|---|---|
| Throughput | about 842 TFLOP/s/GPU |
| Step time | about 36.9 s |
| Tokens | about 7.1K tokens/s/GPU |
| Peak device memory | about 277 GiB of 277 GiB |

These numbers were measured with an earlier build of the same software stack and the same
training configuration. Memory is nearly full on the second pipeline stage, so any change that
adds memory, such as heavier vision data, will need more recompute.

## Limitations

- **Proxy model:** in the 32K recipe the language model has 28 of its 60 layers, so those results
  do not represent the full model's step time or memory.
- **Synthetic data:** about 200 merged vision tokens per sample, or about 0.6% of the sequence.
  Workloads with more images or video will put more load and memory on the vision encoder.
- **Forced load balancing:** `--moe-router-force-load-balancing` spreads tokens evenly across
  experts. Throughput with real routing can be lower.
- **Memory headroom:** about 18 GiB remains on the first pipeline stage. Heavier vision workloads
  may need more recompute or a smaller `--mdp-encoder-max-payload-rows`.
- **Experimental settings:** `MDP_ALLOW_OVERLAP_WITH_CUDA_GRAPHS=1` enables a combination of MDP
  window overlap and CUDA graphs that has been tested with this recipe but is still experimental.
- **Not supported with MDP:** context parallelism and activation offloading.
- **Not validated:** checkpoint saving and loading, and convergence.
