#!/bin/bash
# ============================================================
# Step 2: Prepare VLM Input
# ============================================================
# This script encodes keyframe images to base64 and generates JSONL
# file for VLM annotation.
#
# Input:  data_processed/keyframes/
# Output: data_processed/vlm_input.jsonl

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "============================================================"
echo "Step 2: Prepare VLM Input"
echo "============================================================"
echo "Keyframes Dir:  ${KEYFRAMES_DIR}"
echo "Output File:    ${VLM_INPUT_FILE}"
echo "Sample Frames:  ${VLM_FRAME_NUM}"
echo "============================================================"

# Check if keyframes exist
if [ ! -d "${KEYFRAMES_DIR}" ]; then
    echo "Error: Keyframes directory not found: ${KEYFRAMES_DIR}"
    echo "Please run step1_extract_keyframes.sh first"
    exit 1
fi

# Run
python ${PROJECT_ROOT}/vla-scripts/data_process/prepare_vlm_input.py \
    --input_dir ${KEYFRAMES_DIR} \
    --output_file ${VLM_INPUT_FILE} \
    --sample_num ${VLM_FRAME_NUM} \
    --max_workers 8

echo ""
echo "Step 2 completed! Output saved to: ${VLM_INPUT_FILE}"
echo "Next step: Run step3_vlm_annotation.sh (requires VLM server)"

