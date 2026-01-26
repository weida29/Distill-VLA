#!/bin/bash
# ============================================================
# Step 3: VLM Annotation
# ============================================================
# This script uses VLM to detect objects and generate bounding boxes.
#
# Prerequisites:
#   1. Start VLM server: vllm serve Qwen3-VL-30B-A3B-Instruct --port 18000 -tp 8
#   2. Ensure OPENAI_API_BASE_URL is correctly set in config.sh
#
# Input:  data_processed/vlm_input.jsonl
# Output: data_processed/bbox.json

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "============================================================"
echo "Step 3: VLM Annotation"
echo "============================================================"
echo "VLM API URL:    ${OPENAI_API_BASE_URL}"
echo "VLM Model:      ${OPENAI_MODEL}"
echo "Input File:     ${VLM_INPUT_FILE}"
echo "Output File:    ${BBOX_OUTPUT_FILE}"
echo "Frame Num:      ${VLM_FRAME_NUM}"
echo "Max Workers:    ${VLM_MAX_WORKERS}"
echo "============================================================"

# Check if VLM input exists
if [ ! -f "${VLM_INPUT_FILE}" ]; then
    echo "Error: VLM input file not found: ${VLM_INPUT_FILE}"
    echo "Please run step2_prepare_vlm_input.sh first"
    exit 1
fi

# Check if VLM server is running
echo "Checking VLM server..."
if ! curl -s "${OPENAI_API_BASE_URL}/models" > /dev/null 2>&1; then
    echo "Warning: VLM server may not be running at ${OPENAI_API_BASE_URL}"
    echo "Please start the VLM server first:"
    echo "  vllm serve ${OPENAI_MODEL} --port 18000 -tp 8"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run annotation
python ${PROJECT_ROOT}/vla-scripts/data_process/vlm_annotator/vlm_annotation.py \
    --input_file ${VLM_INPUT_FILE} \
    --output_file ${BBOX_OUTPUT_FILE} \
    --task tracking \
    --frame_num ${VLM_FRAME_NUM} \
    --max_workers ${VLM_MAX_WORKERS}

echo ""
echo "Step 3 completed! Output saved to: ${BBOX_OUTPUT_FILE}"
echo "Next step: Run step4_convert_to_gdino.sh"


