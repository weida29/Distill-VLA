#!/bin/bash
# Open-GroundingDino Inference Script for Libero Dataset
#
# Usage:
#   bash infer_libero.sh <checkpoint_path> <image_path> <text_prompt>
#   bash infer_libero.sh checkpoints/open_gdino_finetuned/checkpoint.pth test_image.jpg "pick up the bowl"
#
# Optional arguments:
#   --box_threshold: Box confidence threshold (default: 0.3)
#   --text_threshold: Text matching threshold (default: 0.25)
#   --output_dir: Output directory (default: outputs)

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# ============================================
# Parse Arguments
# ============================================

if [ $# -lt 3 ]; then
    echo "Usage: $0 <checkpoint_path> <image_path> <text_prompt> [--box_threshold 0.3] [--text_threshold 0.25] [--output_dir outputs]"
    echo ""
    echo "Example:"
    echo "  $0 checkpoints/open_gdino_finetuned/checkpoint.pth test_image.jpg \"pick up the bowl\""
    echo "  $0 checkpoints/open_gdino_finetuned/checkpoint.pth test_image.jpg \"basket\" --box_threshold 0.4 --output_dir my_outputs"
    exit 1
fi

CHECKPOINT_PATH="$1"
IMAGE_PATH="$2"
TEXT_PROMPT="$3"

# Default values
BOX_THRESHOLD=0.3
TEXT_THRESHOLD=0.25
OUTPUT_DIR="outputs"

# Parse optional arguments
shift 3
while [[ $# -gt 0 ]]; do
    case $1 in
        --box_threshold)
            BOX_THRESHOLD="$2"
            shift 2
            ;;
        --text_threshold)
            TEXT_THRESHOLD="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ============================================
# Paths Configuration
# ============================================

# Config file (use the same config as training)
CONFIG="${SCRIPT_DIR}/config/cfg_odvg.py"

# If checkpoint path is relative, make it relative to project root
if [[ ! "$CHECKPOINT_PATH" = /* ]]; then
    CHECKPOINT_PATH="${PROJECT_ROOT}/${CHECKPOINT_PATH}"
fi

# If image path is relative, make it relative to project root
if [[ ! "$IMAGE_PATH" = /* ]]; then
    IMAGE_PATH="${PROJECT_ROOT}/${IMAGE_PATH}"
fi

# Output directory relative to project root
OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"

# ============================================
# Validation
# ============================================

echo "============================================"
echo "Open-GroundingDino Inference"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Image:      ${IMAGE_PATH}"
echo "Prompt:     ${TEXT_PROMPT}"
echo "Box threshold: ${BOX_THRESHOLD}"
echo "Text threshold: ${TEXT_THRESHOLD}"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================"

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "Error: Checkpoint not found at ${CHECKPOINT_PATH}"
    echo "Please provide a valid checkpoint path."
    exit 1
fi

# Check if image exists
if [ ! -f "${IMAGE_PATH}" ]; then
    echo "Error: Image not found at ${IMAGE_PATH}"
    echo "Please provide a valid image path."
    exit 1
fi

# Check if config exists
if [ ! -f "${CONFIG}" ]; then
    echo "Error: Config file not found at ${CONFIG}"
    exit 1
fi

# ============================================
# Run Inference
# ============================================

echo ""
echo "Running inference..."
echo ""

cd "${SCRIPT_DIR}"

python tools/inference_on_a_image.py \
    --config_file "${CONFIG}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --image_path "${IMAGE_PATH}" \
    --text_prompt "${TEXT_PROMPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --box_threshold ${BOX_THRESHOLD} \
    --text_threshold ${TEXT_THRESHOLD}

echo ""
echo "============================================"
echo "Inference completed!"
echo "============================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - raw_image.jpg: Original image"
echo "  - pred.jpg: Image with bounding boxes"
echo ""

