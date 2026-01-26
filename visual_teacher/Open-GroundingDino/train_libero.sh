#!/bin/bash
# Open-GroundingDino Fine-tuning on Libero Dataset
#
# Prerequisites:
# 1. Download BERT weights to: weights/bert-base-uncased/
# 2. Download pretrained model to: weights/groundingdino_swint_ogc.pth
# 3. Prepare data in: data_processed/open_gdino_dataset/
#
# Usage:
#   bash train_libero.sh                    # Use all available GPUs
#   bash train_libero.sh 4                  # Use 4 GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash train_libero.sh 2  # Use specific GPUs

set -e

# Number of GPUs (default: all available or 8)
NUM_GPUS=${1:-8}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# ============================================
# Paths Configuration - MODIFY THESE
# ============================================

# Data paths
DATA_ROOT="${PROJECT_ROOT}/data_processed/open_gdino_dataset"
IMAGE_ROOT="${DATA_ROOT}/images"

# Model weights
BERT_PATH="${SCRIPT_DIR}/weights/bert-base-uncased"
PRETRAINED_MODEL="${SCRIPT_DIR}/weights/groundingdino_swint_ogc.pth"

# Config files
CONFIG="${SCRIPT_DIR}/config/cfg_odvg.py"
DATASETS="${SCRIPT_DIR}/config/datasets_libero.json"

# Output directory
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/open_gdino_finetuned"

# ============================================
# Training Hyperparameters
# ============================================

BATCH_SIZE=4              # Per GPU batch size
EPOCHS=25                 # Training epochs
LR=0.0001                 # Base learning rate

# ============================================
# Validation
# ============================================

echo "============================================"
echo "Open-GroundingDino Fine-tuning on Libero"
echo "============================================"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Batch size per GPU: ${BATCH_SIZE}"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Epochs: ${EPOCHS}"
echo "Learning rate: ${LR}"
echo ""
echo "Data root: ${DATA_ROOT}"
echo "BERT path: ${BERT_PATH}"
echo "Pretrained: ${PRETRAINED_MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================"

# Check prerequisites
if [ ! -d "${BERT_PATH}" ]; then
    echo "Error: BERT weights not found at ${BERT_PATH}"
    echo "Please download from: https://huggingface.co/bert-base-uncased"
    echo "Required files: config.json, pytorch_model.bin, vocab.txt"
    exit 1
fi

if [ ! -f "${PRETRAINED_MODEL}" ]; then
    echo "Error: Pretrained model not found at ${PRETRAINED_MODEL}"
    echo "Please download from: https://github.com/IDEA-Research/GroundingDINO/releases"
    exit 1
fi

if [ ! -f "${DATA_ROOT}/train.jsonl" ]; then
    echo "Error: Training data not found at ${DATA_ROOT}/train.jsonl"
    echo "Please run the data conversion scripts first."
    exit 1
fi

# ============================================
# Update config with BERT path
# ============================================

# Create a temporary config with correct BERT path
TMP_CONFIG="${SCRIPT_DIR}/config/cfg_libero_train.py"
cp "${CONFIG}" "${TMP_CONFIG}"

# Replace bert path in config
sed -i "s|text_encoder_type = \"bert-base-uncased\"|text_encoder_type = \"${BERT_PATH}\"|g" "${TMP_CONFIG}"

# Update datasets config with actual paths
TMP_DATASETS="${SCRIPT_DIR}/config/datasets_libero_train.json"
cat > "${TMP_DATASETS}" << EOF
{
  "train": [
    {
      "root": "${IMAGE_ROOT}/",
      "anno": "${DATA_ROOT}/train.jsonl",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "${IMAGE_ROOT}/",
      "anno": "${DATA_ROOT}/val_coco.json",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
EOF

echo "Created temporary config files..."

# ============================================
# Create output directory
# ============================================

mkdir -p "${OUTPUT_DIR}"

# ============================================
# Run Training
# ============================================

echo ""
echo "Starting training..."
echo ""

cd "${SCRIPT_DIR}"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=29501 \
    main.py \
    --config_file "${TMP_CONFIG}" \
    --datasets "${TMP_DATASETS}" \
    --pretrain_model_path "${PRETRAINED_MODEL}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --options "text_encoder_type=${BERT_PATH}"

echo ""
echo "============================================"
echo "Training completed!"
echo "============================================"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo ""

# Cleanup temporary configs
rm -f "${TMP_CONFIG}" "${TMP_DATASETS}"

