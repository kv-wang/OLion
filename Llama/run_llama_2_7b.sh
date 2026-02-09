#!/bin/sh
export WANDB_BASE_URL=https://api.bandw.top

export WANDB_API_KEY="add your wandb api key here"
conda activate torchtitan
# export NCCL_TIMEOUT=300000  # 300 秒

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export WANDB_OFFICIAL=1

set -ex

# libUV is a scalable backend for TCPStore which is used in processGroup
# rendezvous. This is the recommended backend for distributed training.
export USE_LIBUV=1

export HF_DATASETS_CACHE="/root/autodl-tmp/huggingface_cache"
export TRANSFORMERS_CACHE="/root/autodl-tmp/huggingface_cache"

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_2_7b_nofsdp.sh

NGPU=${NGPU:-"8"}
NNODES=${NNODES:-"1"}

# by default log just rank 0 output,
LOG_RANK=${LOG_RANK:-0}

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama2_7b.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# Disable PyTorch Dynamo compilation to avoid duplicate template name error
export TORCH_COMPILE_DISABLED=1
export TORCHDYNAMO_DISABLE=1

export NCCL_P2P_LEVEL=NVL
# Increase NCCL timeout to avoid communication timeouts during checkpoint saving
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=10800  # 1 hour
export NCCL_P2P_DISABLE=1  # Disable P2P to avoid some communication issues
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0  # Disable async error handling
export TORCH_DISTRIBUTED_DETAIL=DEBUG  # Enable debug logging for distributed operations

# Call train_nofsdp.py for multi-GPU training without FSDP
HF_ENDPOINT=https://hf-mirror.com torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides
