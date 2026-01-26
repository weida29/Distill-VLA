#!/bin/bash
# ============================================================
# Step 1: Extract Keyframes from RLDS Dataset
# ============================================================
# This script extracts keyframe images and prompts from LIBERO RLDS dataset.
#
# Input:  RLDS dataset (e.g., data/modified_libero_rlds/)
# Output: data_processed/keyframes/
#         ├── libero_spatial_no_noops/
#         │   ├── episode_00000/
#         │   │   ├── prompt.txt
#         │   │   ├── frame_00000.jpg
#         │   │   └── ...
#         └── metadata.json

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "============================================================"
echo "Step 1: Extract Keyframes from RLDS Dataset"
echo "============================================================"
echo "RLDS Data Dir:  ${RLDS_DATA_DIR}"
echo "Output Dir:     ${KEYFRAMES_DIR}"
echo "Sample Rate:    ${SAMPLE_RATE}"
echo "Subsets:        ${SUBSETS}"
echo "Max Episodes:   ${MAX_EPISODES:-all}"
echo "============================================================"

# Build command
CMD="python ${PROJECT_ROOT}/vla-scripts/data_process/extract_keyframes.py \
    --data_dir ${RLDS_DATA_DIR} \
    --output_dir ${KEYFRAMES_DIR} \
    --sample_rate ${SAMPLE_RATE} \
    --subsets ${SUBSETS}"

# Add optional arguments
if [ -n "${MAX_EPISODES}" ]; then
    CMD="${CMD} --max_episodes ${MAX_EPISODES}"
fi

# Run
echo "Running: ${CMD}"
eval ${CMD}

echo ""
echo "Step 1 completed! Output saved to: ${KEYFRAMES_DIR}"
echo "Next step: Run step2_prepare_vlm_input.sh"


