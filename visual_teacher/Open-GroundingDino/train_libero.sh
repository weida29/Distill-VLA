#!/bin/bash
# Open-GroundingDino Fine-tuning on Libero Dataset
#
# Prerequisites:
# 1. Download weights: bash download_weights.sh
#    (or manually download BERT to: weights/bert-base-uncased/
#     and pretrained model to: weights/groundingdino_swint_ogc.pth)
# 2. Prepare data in: data_processed/open_gdino_dataset/
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
BATCH_SIZE=4          # Batch size per GPU
EPOCHS=15             # Total training epochs
LR=0.0001             # Base learning rate

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

# Set use_coco_eval to False for custom dataset (required for non-COCO validation)
sed -i "s|use_coco_eval = True|use_coco_eval = False|g" "${TMP_CONFIG}"

# Extract unique labels from train.jsonl and add label_list to config
echo "Extracting unique labels from training data..."
LABEL_INFO=$(python3 << EOF
import json
import sys

labels = set()
train_file = "${DATA_ROOT}/train.jsonl"

try:
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if 'grounding' in data and 'regions' in data['grounding']:
                    for region in data['grounding']['regions']:
                        if 'phrase' in region:
                            labels.add(region['phrase'])
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    print(f"Error: {train_file} not found", file=sys.stderr)
    sys.exit(1)

# Sort labels for consistent output
sorted_labels = sorted(labels)
label_list_str = "label_list = [" + ", ".join([f"'{label}'" for label in sorted_labels]) + "]"
# Output: label_list_str|count (separated by |)
print(f"{label_list_str}|{len(sorted_labels)}")
EOF
)

if [ $? -eq 0 ] && [ -n "${LABEL_INFO}" ]; then
    # Split label_list and count
    LABEL_LIST=$(echo "${LABEL_INFO}" | cut -d'|' -f1)
    LABEL_COUNT=$(echo "${LABEL_INFO}" | cut -d'|' -f2)
    
    # Append label_list to the end of config file
    echo "" >> "${TMP_CONFIG}"
    echo "${LABEL_LIST}" >> "${TMP_CONFIG}"
    echo "Added label_list with ${LABEL_COUNT} unique labels to config"
else
    echo "Warning: Failed to extract labels, continuing without label_list"
fi

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
    --options "text_encoder_type=${BERT_PATH}"

echo ""
echo "============================================"
echo "Training completed!"
echo "============================================"
echo "Checkpoints saved to: ${OUTPUT_DIR}"
echo ""

# Cleanup temporary configs
rm -f "${TMP_CONFIG}" "${TMP_DATASETS}"


