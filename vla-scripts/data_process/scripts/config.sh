#!/bin/bash
# ============================================================
# Data Processing Pipeline - Configuration
# ============================================================
# Modify these paths according to your environment

# ============== Path Configuration ==============
# Project root directory
export PROJECT_ROOT="/tmp/Distill-VLA"

# RLDS dataset directory (contains libero_spatial_no_noops, libero_goal_no_noops, etc.)
export RLDS_DATA_DIR="${PROJECT_ROOT}/data/libero"

# Output directory for processed data
export OUTPUT_DIR="${PROJECT_ROOT}/data_processed"

# ============== Dataset Configuration ==============
# Subsets to process (space-separated)
export SUBSETS="libero_spatial_no_noops libero_object_no_noops libero_goal_no_noops libero_10_no_noops"

# Frame sampling rate (extract every N frames)
export SAMPLE_RATE=5

# Max episodes per subset (empty for all)
export MAX_EPISODES=""

# ============== VLM Configuration ==============
# VLM model path (local checkpoint)
export VLM_MODEL_PATH="${PROJECT_ROOT}/ckpt/qwen3-vl-30b-a3b"
export VLM_MODEL_NAME="Qwen3-VL-30B-A3B-Instruct"

# VLM API endpoint (vLLM server)
export OPENAI_API_BASE_URL="http://127.0.0.1:18000/v1"
export OPENAI_API_KEY="sk-placeholder"
export OPENAI_MODEL="${VLM_MODEL_NAME}"

# vLLM server settings
export VLLM_TP=8                    # Tensor parallelism (number of GPUs)
export VLLM_PORT=18000              # Server port

# Number of frames for VLM inference
export VLM_FRAME_NUM=6

# Number of parallel workers
export VLM_MAX_WORKERS=8

# ============== Grounding DINO Configuration ==============
# Validation set ratio (0.0 for no split)
export VAL_RATIO=0.1

# Random seed for train/val split
export SEED=42

# ============================================================
# Don't modify below unless you know what you're doing
# ============================================================
export KEYFRAMES_DIR="${OUTPUT_DIR}/keyframes"
export VLM_INPUT_FILE="${OUTPUT_DIR}/vlm_input.jsonl"
export BBOX_OUTPUT_FILE="${OUTPUT_DIR}/bbox.json"
export GDINO_DATASET_DIR="${OUTPUT_DIR}/grounding_dino_dataset"

