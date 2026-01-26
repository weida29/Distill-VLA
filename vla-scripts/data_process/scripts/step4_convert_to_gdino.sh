#!/bin/bash
# ============================================================
# Step 4: Convert to Grounding DINO Format
# ============================================================
# This script converts bbox.json to ODVG format for Grounding DINO training.
#
# Input:  data_processed/bbox.json
# Output: data_processed/grounding_dino_dataset/
#         ├── images/
#         ├── train.jsonl
#         ├── val.jsonl (if val_ratio > 0)
#         └── meta.json

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "============================================================"
echo "Step 4: Convert to Grounding DINO Format"
echo "============================================================"
echo "BBox File:      ${BBOX_OUTPUT_FILE}"
echo "Keyframes Dir:  ${KEYFRAMES_DIR}"
echo "Output Dir:     ${GDINO_DATASET_DIR}"
echo "Val Ratio:      ${VAL_RATIO}"
echo "Random Seed:    ${SEED}"
echo "============================================================"

# Check if bbox file exists
if [ ! -f "${BBOX_OUTPUT_FILE}" ]; then
    echo "Error: BBox file not found: ${BBOX_OUTPUT_FILE}"
    echo "Please run step3_vlm_annotation.sh first"
    exit 1
fi

# Run conversion
python ${PROJECT_ROOT}/vla-scripts/data_process/convert_to_grounding_dino.py \
    --bbox-json ${BBOX_OUTPUT_FILE} \
    --keyframes-dir ${KEYFRAMES_DIR} \
    --output ${GDINO_DATASET_DIR} \
    --val-ratio ${VAL_RATIO} \
    --seed ${SEED}

echo ""
echo "Step 4 completed! Output saved to: ${GDINO_DATASET_DIR}"
echo ""
echo "============================================================"
echo "Data processing pipeline completed!"
echo "============================================================"
echo "Dataset ready for Grounding DINO training at:"
echo "  ${GDINO_DATASET_DIR}"
echo ""
echo "To train Grounding DINO, run:"
echo "  cd visual_teacher/GroundingDINO"
echo "  ./finetune/run_train_ddp.sh"

