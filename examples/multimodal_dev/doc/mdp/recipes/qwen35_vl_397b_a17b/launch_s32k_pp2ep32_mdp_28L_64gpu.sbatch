#!/bin/bash
# Example Slurm launcher for s32k_pp2ep32_mdp_28L_64gpu.sh on 64 GB300 GPUs.
# Requires Slurm with the pyxis/enroot container plugin. Replace every <...> placeholder before submitting.
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<PARTITION>
#SBATCH --time=0:50:00
#SBATCH --job-name=qwen35vl-397b-28L-32k-64gpu
#SBATCH --output=qwen35vl-397b-28L-32k-64gpu_%j.log
#SBATCH --exclusive
# All 16 nodes must share one NVL72 NVLink domain so the EP32 HybridEP traffic stays on NVLink.
# --segment is the option for that on clusters configured with Slurm block topology; use the site equivalent otherwise.
#SBATCH --segment=16

set -euo pipefail

# Megatron-LM checkout at this branch.
export MCORE_ROOT="<PATH_TO_MEGATRON_LM>"
# Container image built for this recipe.
CONTAINER_IMAGE="<PATH_TO_CONTAINER_SQSH>"
# Hugging Face cache that already contains the Qwen/Qwen3.5-397B-A17B tokenizer and processor files.
# No model weights are needed. The recipe runs with HF_HUB_OFFLINE=1.
export HF_HOME="<PATH_TO_HF_HOME>"
export MCORE_BENCH_OUTPUT="${MCORE_BENCH_OUTPUT:-$PWD/output/s32k_pp2ep32_mdp_28L_64gpu}"
# Each rank should be bound to the CPU cores and NUMA node closest to its GPU, with enough cores per rank.
# Set RANK_LAUNCHER to a site affinity wrapper followed by python, for example "/path/to/affinity_wrapper python".
RANK_LAUNCHER="${RANK_LAUNCHER:-python}"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${SLURM_JOB_ID}}"
mkdir -p "$MCORE_BENCH_OUTPUT"

RECIPE="$MCORE_ROOT/examples/multimodal_dev/doc/mdp/recipes/qwen35_vl_397b_a17b/s32k_pp2ep32_mdp_28L_64gpu.sh"

# Mount every filesystem that holds MCORE_ROOT, HF_HOME and MCORE_BENCH_OUTPUT.
srun \
    --mpi=pmix -l \
    --export=ALL \
    --no-container-mount-home \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="<MOUNTS, e.g. /data:/data>" \
    --container-workdir="$MCORE_ROOT" \
    bash "$RECIPE" $RANK_LAUNCHER
