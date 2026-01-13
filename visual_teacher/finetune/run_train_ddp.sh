#!/bin/bash
# Grounding DINO Fine-tuning with DDP (8 GPUs)
#
# Usage:
#   bash run_train_ddp.sh                    # Use all 8 GPUs
#   bash run_train_ddp.sh 4                  # Use 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_train_ddp.sh 4  # Use specific GPUs

# Number of GPUs (default: 8)
NUM_GPUS=${1:-8}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Training parameters
BATCH_SIZE=2          # Per GPU batch size
EPOCHS=30
LR=1e-4
FREEZE_BACKBONE=""    # Add --freeze-backbone for small datasets

echo "============================================"
echo "Grounding DINO DDP Training"
echo "============================================"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LR}"
echo "============================================"

# Run with torchrun
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    ${SCRIPT_DIR}/train_grounding_dino.py \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --num-workers 4 \
    ${FREEZE_BACKBONE}

echo "Training completed!"


