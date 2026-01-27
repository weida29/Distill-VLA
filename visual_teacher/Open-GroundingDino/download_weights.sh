#!/bin/bash
# Download pretrained weights for Open-GroundingDino training
#
# Prerequisites:
#   - hfd.sh script available at /storage/v-xiangxizheng/cache/hfd.sh
#   - aria2c installed for parallel downloading
#
# Usage:
#   bash download_weights.sh
#   HFD_SCRIPT=/path/to/hfd.sh bash download_weights.sh

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# HFD script path (default to user's example path)
HFD_SCRIPT=${HFD_SCRIPT:-"/storage/v-xiangxizheng/cache/hfd.sh"}

# Weights directory (as specified in train_libero.sh)
WEIGHTS_DIR="${SCRIPT_DIR}/weights"
mkdir -p "${WEIGHTS_DIR}"

# Download parameters
ARIA2C_THREADS=${ARIA2C_THREADS:-10}

echo "============================================"
echo "Downloading Open-GroundingDino Weights"
echo "============================================"
echo "HFD Script: ${HFD_SCRIPT}"
echo "Weights Directory: ${WEIGHTS_DIR}"
echo "Aria2c Threads: ${ARIA2C_THREADS}"
echo "============================================"
echo ""

# Check if hfd.sh exists
if [ ! -f "${HFD_SCRIPT}" ]; then
    echo "Error: hfd.sh not found at ${HFD_SCRIPT}"
    echo "Please set HFD_SCRIPT environment variable or install hfd.sh"
    exit 1
fi

# ============================================
# 1. Download BERT weights
# ============================================
echo "Step 1/2: Downloading BERT weights (bert-base-uncased)..."
BERT_DIR="${WEIGHTS_DIR}/bert-base-uncased"

if [ -d "${BERT_DIR}" ] && [ -f "${BERT_DIR}/pytorch_model.bin" ]; then
    echo "  ✓ BERT weights already exist at ${BERT_DIR}"
else
    echo "  Downloading from HuggingFace..."
    # Use the exact format from user's example
    "${HFD_SCRIPT}" \
        bert-base-uncased \
        --ocal-dir "${BERT_DIR}" \
        --tool aria2c \
        -x${ARIA2C_THREADS}
    
    # Verify download
    if [ ! -f "${BERT_DIR}/pytorch_model.bin" ] && [ ! -f "${BERT_DIR}/model.safetensors" ]; then
        echo "  Error: BERT download failed or incomplete"
        exit 1
    fi
    echo "  ✓ BERT weights downloaded successfully"
fi
echo ""

# ============================================
# 2. Download GroundingDINO pretrained model
# ============================================
echo "Step 2/2: Downloading GroundingDINO pretrained model..."
PRETRAINED_MODEL="${WEIGHTS_DIR}/groundingdino_swint_ogc.pth"
PRETRAINED_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

if [ -f "${PRETRAINED_MODEL}" ]; then
    echo "  ✓ Pretrained model already exists at ${PRETRAINED_MODEL}"
else
    echo "  Downloading from GitHub releases..."
    echo "  URL: ${PRETRAINED_URL}"
    
    # Use aria2c directly for GitHub releases (hfd.sh is for HuggingFace)
    if command -v aria2c &> /dev/null; then
        aria2c \
            --dir="${WEIGHTS_DIR}" \
            --out="groundingdino_swint_ogc.pth" \
            --max-connection-per-server=16 \
            --split=${ARIA2C_THREADS} \
            --min-split-size=1M \
            "${PRETRAINED_URL}"
    elif command -v wget &> /dev/null; then
        wget -O "${PRETRAINED_MODEL}" "${PRETRAINED_URL}"
    elif command -v curl &> /dev/null; then
        curl -L -o "${PRETRAINED_MODEL}" "${PRETRAINED_URL}"
    else
        echo "  Error: No download tool found (aria2c/wget/curl)"
        echo "  Please install aria2c, wget, or curl, or download manually:"
        echo "    ${PRETRAINED_URL}"
        exit 1
    fi
    
    # Verify download
    if [ ! -f "${PRETRAINED_MODEL}" ]; then
        echo "  Error: Pretrained model download failed"
        exit 1
    fi
    echo "  ✓ Pretrained model downloaded successfully"
fi
echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "Download Complete!"
echo "============================================"
echo "BERT weights: ${BERT_DIR}"
echo "Pretrained model: ${PRETRAINED_MODEL}"
echo ""
echo "You can now run training with:"
echo "  bash train_libero.sh"
echo "============================================"

