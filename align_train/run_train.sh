#!/bin/bash
# run_train.sh - Launch VLA-GDINO alignment training

# Default values
NUM_GPUS=${NUM_GPUS:-1}
DATASET=${DATASET:-"libero_spatial_no_noops"}
BATCH_SIZE=${BATCH_SIZE:-8}
LR=${LR:-5e-4}
MAX_STEPS=${MAX_STEPS:-50000}

# V100 compatible by default (use_bf16=False)
# Set USE_BF16=1 for A100/H100
USE_BF16=${USE_BF16:-0}

echo "============================================"
echo "VLA-GDINO Feature Alignment Training"
echo "============================================"
echo "GPUs: $NUM_GPUS"
echo "Dataset: $DATASET"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "Max steps: $MAX_STEPS"
echo "Use BF16: $USE_BF16 (set USE_BF16=1 for A100/H100)"
echo "============================================"

# Change to project root
cd "$(dirname "$0")/.."

# Build extra args
EXTRA_ARGS=""
if [ "$USE_BF16" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use_bf16"
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training
    torchrun --nproc_per_node=$NUM_GPUS \
        align_train/train_align.py \
        --dataset_name $DATASET \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --max_steps $MAX_STEPS \
        $EXTRA_ARGS \
        "$@"
else
    # Single GPU training
    python align_train/train_align.py \
        --dataset_name $DATASET \
        --batch_size $BATCH_SIZE \
        --learning_rate $LR \
        --max_steps $MAX_STEPS \
        $EXTRA_ARGS \
        "$@"
fi

