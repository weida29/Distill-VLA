#!/bin/bash
# Grounding DINO Parameter-Efficient Fine-tuning with DDP (8 GPUs)
#
# Strategy: Freeze most layers (backbone, text encoder, encoder, most decoder layers)
#           Only train bbox_embed and last 2 decoder layers
#
# Usage:
#   bash run_train_ddp.sh                    # Use all 8 GPUs
#   bash run_train_ddp.sh 4                  # Use 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_train_ddp.sh 4  # Use specific GPUs

# Number of GPUs (default: 8)
NUM_GPUS=${1:-8}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Paths (adjust as needed)
TRAIN_JSONL="${PROJECT_ROOT}/data_processed/grounding_dino_dataset/train.jsonl"
VAL_JSONL="${PROJECT_ROOT}/data_processed/grounding_dino_dataset/val.jsonl"
IMAGE_DIR="${PROJECT_ROOT}/data_processed/grounding_dino_dataset"
CONFIG="${SCRIPT_DIR}/../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
PRETRAINED="${SCRIPT_DIR}/../pretrained_ckpt/groundingdino_swint_ogc.pth"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/grounding_dino_finetuned"

# Training parameters - Fine-tuning with contrastive loss
BATCH_SIZE=4              # Per GPU batch size
EPOCHS=25                 # More epochs for learning text-visual alignment
LR=5e-6                   # Slightly larger lr for contrastive learning
UNFREEZE_LAYERS=3         # Unfreeze last 3 decoder layers for better adaptation
LOSS_CLASS=1.0            # Enable token-level contrastive loss for text-visual alignment

echo "============================================"
echo "Grounding DINO Fine-tuning with Contrastive Loss"
echo "============================================"
echo "Strategy: Train bbox_embed + last ${UNFREEZE_LAYERS} decoder layers + token-level contrastive"
echo ""
echo "Number of GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LR}"
echo "Classification loss weight: ${LOSS_CLASS}"
echo ""
echo "Train data: ${TRAIN_JSONL}"
echo "Val data: ${VAL_JSONL}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run with torchrun
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29500 \
    ${SCRIPT_DIR}/train_grounding_dino.py \
    --train-jsonl ${TRAIN_JSONL} \
    --val-jsonl ${VAL_JSONL} \
    --image-dir ${IMAGE_DIR} \
    --config ${CONFIG} \
    --pretrained ${PRETRAINED} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --loss-coef-class ${LOSS_CLASS} \
    --efficient-finetune \
    --unfreeze-decoder-layers ${UNFREEZE_LAYERS} \
    --freeze-text-encoder \
    --num-workers 4

echo ""
echo "Training completed!"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
