#!/bin/bash
# =============================================================================
# VLA-GDINO Feature Alignment Training Script
# =============================================================================
#
# Multi-task training: Action prediction + Feature alignment with GDINO teacher.
#
# Usage:
#   bash align_train/run_align_train.sh [NUM_GPUS] [BATCH_SIZE] [MAX_STEPS]
#   
# Examples:
#   bash align_train/run_align_train.sh         # 1 GPU, default settings
#   bash align_train/run_align_train.sh 4       # 4 GPUs, default settings
#   bash align_train/run_align_train.sh 4 2 50000  # 4 GPUs, batch=2, 50k steps
#
# Environment variables:
#   GDINO_CKPT   - Path to finetuned GDINO checkpoint
#   VLM_PATH     - Path to VLM pretrained model
#   DATA_ROOT    - Path to RLDS data directory
#
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_ROOT}"

# ============================================
# Configuration
# ============================================
NUM_GPUS=${1:-1}
BATCH_SIZE=${2:-4}
MAX_STEPS=${3:-50000}

# Model paths (override with environment variables)
VLM_PATH=${VLM_PATH:-"pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"}
GDINO_CKPT=${GDINO_CKPT:-"checkpoints/open_gdino_finetuned/checkpoint_best_regular.pth"}

# Data paths
DATA_ROOT=${DATA_ROOT:-"datasets/rlds"}
DATASET_NAME=${DATASET_NAME:-"libero_spatial_no_noops"}

# Training
LEARNING_RATE=${LEARNING_RATE:-5e-4}
LORA_RANK=${LORA_RANK:-32}
GRAD_ACCUM=${GRAD_ACCUM:-1}

# Loss weights
ACTION_WEIGHT=${ACTION_WEIGHT:-1.0}
HS_WEIGHT=${HS_WEIGHT:-1.0}
REF_WEIGHT=${REF_WEIGHT:-1.0}

# Output
RUN_ID=${RUN_ID:-"align_$(date +%Y%m%d_%H%M%S)"}

# ============================================
# Display Configuration
# ============================================
echo "============================================"
echo "VLA-GDINO Feature Alignment Training"
echo "============================================"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM))"
echo "Max steps: ${MAX_STEPS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "LoRA rank: ${LORA_RANK}"
echo ""
echo "VLM path: ${VLM_PATH}"
echo "GDINO checkpoint: ${GDINO_CKPT}"
echo "Data root: ${DATA_ROOT}"
echo "Dataset: ${DATASET_NAME}"
echo ""
echo "Loss weights: action=${ACTION_WEIGHT}, hs=${HS_WEIGHT}, ref=${REF_WEIGHT}"
echo "Run ID: ${RUN_ID}"
echo "============================================"

# ============================================
# Check Prerequisites
# ============================================
if [ ! -d "${VLM_PATH}" ]; then
    echo "ERROR: VLM model not found at ${VLM_PATH}"
    exit 1
fi

if [ ! -f "${GDINO_CKPT}" ]; then
    echo "WARNING: GDINO checkpoint not found at ${GDINO_CKPT}"
    echo "Training will fail if checkpoint is required."
fi

# ============================================
# Run Training
# ============================================
export TOKENIZERS_PARALLELISM=false

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Running distributed training on ${NUM_GPUS} GPUs..."
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29502 \
        align_train/train_align.py \
        --vlm_path "${VLM_PATH}" \
        --gdino_checkpoint "${GDINO_CKPT}" \
        --dataset_name "${DATASET_NAME}" \
        --data_root_dir "${DATA_ROOT}" \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --max_steps ${MAX_STEPS} \
        --grad_accumulation_steps ${GRAD_ACCUM} \
        --lora_rank ${LORA_RANK} \
        --action_loss_weight ${ACTION_WEIGHT} \
        --hs_weight ${HS_WEIGHT} \
        --ref_weight ${REF_WEIGHT} \
        --run_id "${RUN_ID}"
else
    echo "Running single GPU training..."
    python align_train/train_align.py \
        --vlm_path "${VLM_PATH}" \
        --gdino_checkpoint "${GDINO_CKPT}" \
        --dataset_name "${DATASET_NAME}" \
        --data_root_dir "${DATA_ROOT}" \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --max_steps ${MAX_STEPS} \
        --grad_accumulation_steps ${GRAD_ACCUM} \
        --lora_rank ${LORA_RANK} \
        --action_loss_weight ${ACTION_WEIGHT} \
        --hs_weight ${HS_WEIGHT} \
        --ref_weight ${REF_WEIGHT} \
        --run_id "${RUN_ID}"
fi

echo ""
echo "Training completed!"
echo "Checkpoints saved to: runs/align_train/${RUN_ID}"

