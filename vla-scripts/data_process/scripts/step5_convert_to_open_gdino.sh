#!/bin/bash
# Step 5: Convert grounding DINO dataset to Open-GroundingDino format
# Converts bbox from normalized coordinates to pixel coordinates
#
# Usage:
#   bash step5_convert_to_open_gdino.sh

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source config
if [ -f "${SCRIPT_DIR}/config.sh" ]; then
    source "${SCRIPT_DIR}/config.sh"
fi

# Default paths (can be overridden by config.sh)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
INPUT_DIR="${PROJECT_ROOT}/data_processed/grounding_dino_dataset"
OUTPUT_DIR="${PROJECT_ROOT}/data_processed/open_gdino_dataset"
IMAGE_DIR="${PROJECT_ROOT}/data_processed/grounding_dino_dataset/images"

# Visualization settings
NUM_VIS_SAMPLES=20
SEED=42

echo "============================================"
echo "Step 5: Convert to Open-GroundingDino Format"
echo "============================================"
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Images: ${IMAGE_DIR}"
echo "============================================"

# Check if input exists
if [ ! -f "${INPUT_DIR}/train.jsonl" ]; then
    echo "Error: ${INPUT_DIR}/train.jsonl not found!"
    echo "Please run step 4 first to generate the grounding DINO dataset."
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run conversion with visualization
python "${SCRIPT_DIR}/../convert_to_open_gdino.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --image_dir "${IMAGE_DIR}" \
    --image_root_prefix "images/" \
    --visualize \
    --num_vis ${NUM_VIS_SAMPLES} \
    --seed ${SEED}

echo ""
echo "============================================"
echo "Conversion complete!"
echo "============================================"
echo "Output files:"
echo "  - ${OUTPUT_DIR}/train.jsonl"
echo "  - ${OUTPUT_DIR}/val.jsonl (if exists)"
echo ""
echo "Visualization samples:"
echo "  - ${OUTPUT_DIR}/visualization/"
echo ""
echo "Please check the visualization images to verify bbox correctness!"
echo "============================================"

