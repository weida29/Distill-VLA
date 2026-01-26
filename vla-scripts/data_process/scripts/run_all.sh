#!/bin/bash
# ============================================================
# Run All Data Processing Steps
# ============================================================
# This script runs the complete data processing pipeline.
#
# Usage:
#   ./run_all.sh              # Run all steps
#   ./run_all.sh --skip-vlm   # Skip VLM annotation (use existing bbox.json)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
SKIP_VLM=false
for arg in "$@"; do
    case $arg in
        --skip-vlm)
            SKIP_VLM=true
            shift
            ;;
    esac
done

echo "============================================================"
echo "LIBERO Data Processing Pipeline"
echo "============================================================"
echo ""

# Step 1: Extract keyframes
echo "[1/4] Extracting keyframes..."
bash "${SCRIPT_DIR}/step1_extract_keyframes.sh"
echo ""

# Step 2: Prepare VLM input
echo "[2/4] Preparing VLM input..."
bash "${SCRIPT_DIR}/step2_prepare_vlm_input.sh"
echo ""

# Step 3: VLM annotation
if [ "$SKIP_VLM" = true ]; then
    echo "[3/4] Skipping VLM annotation (--skip-vlm flag)"
else
    echo "[3/4] Running VLM annotation..."
    bash "${SCRIPT_DIR}/step3_vlm_annotation.sh"
fi
echo ""

# Step 4: Convert to Grounding DINO format
echo "[4/4] Converting to Grounding DINO format..."
bash "${SCRIPT_DIR}/step4_convert_to_gdino.sh"
echo ""

echo "============================================================"
echo "All steps completed successfully!"
echo "============================================================"


