#!/bin/bash
# ============================================================
# Start vLLM Server for VLM Annotation
# ============================================================
# This script starts a vLLM server for the VLM annotation pipeline.
#
# Usage:
#   ./start_vllm_server.sh                  # Default: 8 GPUs, port 18000
#   ./start_vllm_server.sh --tp 4           # Use 4 GPUs
#   ./start_vllm_server.sh --port 8000      # Use port 8000
#   ./start_vllm_server.sh --tp 4 --port 8000

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# ============== vLLM Configuration ==============
# Use values from config.sh with fallback defaults
MODEL_PATH="${VLM_MODEL_PATH:-${PROJECT_ROOT}/ckpt/qwen3-vl-30b-a3b}"
MODEL_NAME="${VLM_MODEL_NAME:-Qwen3-VL-30B-A3B-Instruct}"

# Default settings (can be overridden by config.sh or command line)
TP=${VLLM_TP:-8}           # Tensor parallelism (number of GPUs)
PORT=${VLLM_PORT:-18000}   # Server port
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}  # Max context length
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
DTYPE=${DTYPE:-half}       # Data type (half, float, auto)

# ============== Parse Arguments ==============
while [[ $# -gt 0 ]]; do
    case $1 in
        --tp|-tp)
            TP="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tp N                      Tensor parallelism (default: 8)"
            echo "  --port PORT                 Server port (default: 18000)"
            echo "  --max-model-len LEN         Max context length (default: 65536)"
            echo "  --gpu-memory-utilization R  GPU memory utilization (default: 0.7)"
            echo "  --dtype TYPE                Data type: half, float, auto (default: half)"
            echo "  -h, --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============== Validation ==============
if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model path not found: ${MODEL_PATH}"
    echo "Please download the model first."
    exit 1
fi

echo "============================================================"
echo "Starting vLLM Server"
echo "============================================================"
echo "Model Path:     ${MODEL_PATH}"
echo "Model Name:     ${MODEL_NAME}"
echo "Tensor Parallel: ${TP} GPUs"
echo "Port:           ${PORT}"
echo "Max Model Len:  ${MAX_MODEL_LEN}"
echo "GPU Memory:     ${GPU_MEMORY_UTILIZATION}"
echo "Dtype:          ${DTYPE}"
echo "============================================================"
echo ""
echo "API Endpoint:   http://127.0.0.1:${PORT}/v1"
echo "Test with:      curl http://127.0.0.1:${PORT}/v1/models"
echo ""
echo "To use with VLM annotation, set in config.sh:"
echo "  export OPENAI_API_BASE_URL=\"http://127.0.0.1:${PORT}/v1\""
echo "============================================================"
echo ""

# ============== Start vLLM Server ==============
vllm serve "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --port ${PORT} \
    -tp ${TP} \
    --max-model-len ${MAX_MODEL_LEN} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --dtype ${DTYPE} \
    --trust-remote-code

