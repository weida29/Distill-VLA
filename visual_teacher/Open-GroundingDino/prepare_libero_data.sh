#!/bin/bash
# Prepare Libero dataset for Open-GroundingDino training
#
# This script:
# 1. Converts val.jsonl to COCO format (required for validation)
# 2. Creates symlinks for images
#
# Usage:
#   bash prepare_libero_data.sh

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Paths
DATA_ROOT="${PROJECT_ROOT}/data_processed/open_gdino_dataset"
ORIGINAL_IMAGE_DIR="${PROJECT_ROOT}/data_processed/grounding_dino_dataset/images"

echo "============================================"
echo "Preparing Libero Data for Open-GroundingDino"
echo "============================================"
echo "Data root: ${DATA_ROOT}"
echo "Original images: ${ORIGINAL_IMAGE_DIR}"
echo "============================================"

# Check if data exists
if [ ! -f "${DATA_ROOT}/train.jsonl" ]; then
    echo "Error: ${DATA_ROOT}/train.jsonl not found!"
    echo "Please run convert_to_open_gdino.py first."
    exit 1
fi

# ============================================
# Step 1: Create symlink to images
# ============================================

echo ""
echo "Step 1: Creating symlink to images..."

if [ -L "${DATA_ROOT}/images" ]; then
    echo "  Symlink already exists, removing..."
    rm "${DATA_ROOT}/images"
fi

if [ -d "${DATA_ROOT}/images" ]; then
    echo "  Images directory already exists."
else
    ln -s "${ORIGINAL_IMAGE_DIR}" "${DATA_ROOT}/images"
    echo "  Created symlink: ${DATA_ROOT}/images -> ${ORIGINAL_IMAGE_DIR}"
fi

# ============================================
# Step 2: Convert val.jsonl to COCO format
# ============================================

echo ""
echo "Step 2: Converting val.jsonl to COCO format..."

if [ -f "${DATA_ROOT}/val.jsonl" ]; then
    python "${PROJECT_ROOT}/vla-scripts/data_process/convert_vg_to_coco.py" \
        --input "${DATA_ROOT}/val.jsonl" \
        --output "${DATA_ROOT}/val_coco.json" \
        --label_map "${DATA_ROOT}/label_map.json"
else
    echo "  Warning: val.jsonl not found, skipping validation set conversion."
    echo "  Training will work but validation metrics won't be available."
fi

# ============================================
# Step 3: Verify data
# ============================================

echo ""
echo "Step 3: Verifying data..."

echo "  Checking train.jsonl..."
TRAIN_COUNT=$(wc -l < "${DATA_ROOT}/train.jsonl")
echo "    Train samples: ${TRAIN_COUNT}"

if [ -f "${DATA_ROOT}/val_coco.json" ]; then
    echo "  Checking val_coco.json..."
    VAL_IMAGES=$(python -c "import json; print(len(json.load(open('${DATA_ROOT}/val_coco.json'))['images']))")
    VAL_ANNS=$(python -c "import json; print(len(json.load(open('${DATA_ROOT}/val_coco.json'))['annotations']))")
    echo "    Val images: ${VAL_IMAGES}"
    echo "    Val annotations: ${VAL_ANNS}"
fi

echo "  Checking images directory..."
if [ -d "${DATA_ROOT}/images" ]; then
    IMAGE_COUNT=$(find "${DATA_ROOT}/images" -name "*.jpg" 2>/dev/null | wc -l)
    echo "    Found ${IMAGE_COUNT} images"
else
    echo "    Warning: Images directory not found!"
fi

# ============================================
# Done
# ============================================

echo ""
echo "============================================"
echo "Data preparation complete!"
echo "============================================"
echo ""
echo "Output files:"
echo "  - ${DATA_ROOT}/train.jsonl (training data)"
echo "  - ${DATA_ROOT}/val_coco.json (validation data in COCO format)"
echo "  - ${DATA_ROOT}/label_map.json (category mapping)"
echo "  - ${DATA_ROOT}/images/ (symlink to images)"
echo ""
echo "Next steps:"
echo "  1. Download BERT weights to: ${SCRIPT_DIR}/weights/bert-base-uncased/"
echo "  2. Download pretrained model to: ${SCRIPT_DIR}/weights/groundingdino_swint_ogc.pth"
echo "  3. Run: bash train_libero.sh"
echo "============================================"





