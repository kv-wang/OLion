#!/bin/bash

export WANDB_BASE_URL=https://api.bandw.top
export WANDB_API_KEY="2eaf5d3e15da1d68fbce32137184e1eaba001ff6"
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_TIMEOUT=10800

# 训练脚本：使用FSDP、最小模型(SiT-B/2)和OLion优化器
accelerate launch train.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-B/2" \
  --enc-type="dinov2-vit-b" \
  --proj-coeff=0.5 \
  --encoder-depth=4 \
  --output-dir="depth-4" \
  --exp-name="checkpointing" \
  --data-dir=/root/autodl-tmp/imagenet-vae \
  --use-fsdp \
  --fsdp-sharding-strategy="FULL_SHARD" \
  --fsdp-backward-prefetch="BACKWARD_PRE" \
  --optimizer="olion" \
  --momentum=0.95 \
  --olion-momentum-2=0.98 \
  --rms-scale \
  --nesterov \
  --ns-steps=5 \
  --olion-eps=1e-8 \
  --olion-use-scale \
  --learning-rate=1e-4 \
  --adam-weight-decay=0.1 \
  --adam-beta1=0.9 \
  --adam-beta2=0.95 \
  --sampling-steps=10000 \
  --checkpointing-steps=10000 \



