#!/bin/bash
export WANDB_BASE_URL=https://api.bandw.top
export WANDB_API_KEY="add your wandb api key here"

# basic config
BASE_BATCH_SIZE=60
BASE_MAX_ITERS=20000
BASE_WEIGHT_DECAY=1e-1
BASE_EVAL_INTERVAL=200
WANDB_PROJECT=GPT2
# model config
BASE_N_LAYER=12
BASE_N_HEAD=12
BASE_N_EMBD=768
# AdamW beta2 parameter
BASE_BETA2=0.95

# create output directory
mkdir -p out

echo "Starting training with multiple optimizers..."
echo "=================================================="
echo "Model config: n_layer=$BASE_N_LAYER, n_head=$BASE_N_HEAD, n_embd=$BASE_N_EMBD"
echo "Training config: batch_size=$BASE_BATCH_SIZE, max_iters=$BASE_MAX_ITERS"
echo "=================================================="

# define train function
train_with_config() {
    local OPTIMIZER=$1
    local LR=$2
    
    echo ""
    echo "--------------------------------------------------"
    echo "Training $OPTIMIZER with learning rate: $LR"
    echo "--------------------------------------------------"
    
    # create output directory name (replace dot with underscore in learning rate)
    LR_STR=$(echo $LR | sed 's/\./_/g' | sed 's/-/_/g')
    OUT_DIR="out/${OPTIMIZER}_lr_${LR_STR}"
    
    # build train command
    TRAIN_CMD="torchrun --standalone --nproc_per_node=4 train.py \
        --opt=$OPTIMIZER \
        --wandb_project=$WANDB_PROJECT \
        --learning_rate=$LR \
        --batch_size=$BASE_BATCH_SIZE \
        --max_iters=$BASE_MAX_ITERS \
        --weight_decay=$BASE_WEIGHT_DECAY \
        --eval_interval=$BASE_EVAL_INTERVAL \
        --n_layer=$BASE_N_LAYER \
        --n_head=$BASE_N_HEAD \
        --n_embd=$BASE_N_EMBD \
        --beta2=$BASE_BETA2 \
        --out_dir=$OUT_DIR"
    # execute train
    eval $TRAIN_CMD
    
    echo "Training completed: $OPTIMIZER with lr=$LR"
    echo "Checkpoint saved to: $OUT_DIR"
}

# e.g: train muon with lr=6e-4
train_with_config "olion" "6e-4"

