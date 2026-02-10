#!/bin/bash

# ============================================================================
# Evaluation Script for VLA-Adapter Models on LIBERO Benchmark
# ============================================================================
# This script runs evaluation on all four LIBERO task suites:
# - LIBERO-Spatial
# - LIBERO-Object
# - LIBERO-Goal
# - LIBERO-Long (LIBERO-10)
#
# All configurations are hardcoded in this script.
# ============================================================================

set -e

# ============ Configuration (Hardcoded) ============

# Task suite to evaluate: spatial, object, goal, long, or all
TASK_SUITE="all"

# Checkpoint paths for each task suite
CHECKPOINT_SPATIAL="outputs/LIBERO-Spatial-Pro"
CHECKPOINT_OBJECT="outputs/LIBERO-Object-Pro"
CHECKPOINT_GOAL="outputs/LIBERO-Goal-Pro"
CHECKPOINT_LONG="outputs/LIBERO-long-Pro"

# GPU ID to use
GPU_ID=0

# Common evaluation parameters
USE_PROPRIO=True
NUM_IMAGES_IN_INPUT=2
USE_FILM=False
USE_PRO_VERSION=True

# Task suite names
TASK_SPATIAL="libero_spatial"
TASK_OBJECT="libero_object"
TASK_GOAL="libero_goal"
TASK_LONG="libero_10"

# Log directory
LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"

# ============ Helper Functions ============

print_usage() {
    echo "Usage: bash run_libero_eval.sh"
    echo ""
    echo "This script evaluates VLA-Adapter models on LIBERO benchmarks."
    echo "All configurations are hardcoded in the script."
    echo ""
    echo "To change configuration, edit the script directly:"
    echo "  - TASK_SUITE: Task suite to evaluate (spatial, object, goal, long, or all)"
    echo "  - CHECKPOINT_*: Checkpoint paths for each task suite"
    echo "  - GPU_ID: GPU ID to use"
    echo ""
    echo "Current configuration:"
    echo "  Task Suite: $TASK_SUITE"
    echo "  GPU ID: $GPU_ID"
    echo "  Checkpoint Spatial: $CHECKPOINT_SPATIAL"
    echo "  Checkpoint Object: $CHECKPOINT_OBJECT"
    echo "  Checkpoint Goal: $CHECKPOINT_GOAL"
    echo "  Checkpoint Long: $CHECKPOINT_LONG"
}

run_eval() {
    local task_name=$1
    local task_suite_name=$2
    local checkpoint_path=$3
    local gpu_id=$4
    
    echo "============================================================================"
    echo "Evaluating: $task_name"
    echo "============================================================================"
    echo "Task Suite: $task_suite_name"
    echo "Checkpoint: $checkpoint_path"
    echo "GPU: $gpu_id"
    echo "Log: $LOG_DIR/$task_name--chkpt.log"
    echo ""
    
    # Check if checkpoint exists
    if [ ! -d "$checkpoint_path" ]; then
        echo "Error: Checkpoint directory not found: $checkpoint_path"
        echo "Skipping $task_name evaluation..."
        return 1
    fi
    
    # Run evaluation
    CUDA_VISIBLE_DEVICES=$gpu_id python experiments/robot/libero/run_libero_eval.py \
        --use_proprio $USE_PROPRIO \
        --num_images_in_input $NUM_IMAGES_IN_INPUT \
        --use_film $USE_FILM \
        --pretrained_checkpoint "$checkpoint_path" \
        --task_suite_name "$task_suite_name" \
        --use_pro_version $USE_PRO_VERSION \
        > "$LOG_DIR/$task_name--chkpt.log" 2>&1 &
    
    local pid=$!
    echo "Started evaluation process (PID: $pid)"
    echo "You can monitor progress with: tail -f $LOG_DIR/$task_name--chkpt.log"
    echo ""
    
    return 0
}

# ============ Parse Arguments ============

# Show usage if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    print_usage
    exit 0
fi

# ============ Run Evaluation ============

echo "============================================================================"
echo "VLA-Adapter Evaluation Script"
echo "============================================================================"
echo "Task Suite: $TASK_SUITE"
echo "GPU ID: $GPU_ID"
echo "Log Directory: $LOG_DIR"
echo "============================================================================"
echo ""

# Evaluate specific task suite or all
case "$TASK_SUITE" in
    spatial)
        run_eval "Spatial" "$TASK_SPATIAL" "$CHECKPOINT_SPATIAL" "$GPU_ID"
        ;;
    
    object)
        run_eval "Object" "$TASK_OBJECT" "$CHECKPOINT_OBJECT" "$GPU_ID"
        ;;
    
    goal)
        run_eval "Goal" "$TASK_GOAL" "$CHECKPOINT_GOAL" "$GPU_ID"
        ;;
    
    long)
        run_eval "Long" "$TASK_LONG" "$CHECKPOINT_LONG" "$GPU_ID"
        ;;
    
    all)
        echo "Evaluating all four task suites..."
        echo ""
        
        # Evaluate all four task suites
        run_eval "Spatial" "$TASK_SPATIAL" "$CHECKPOINT_SPATIAL" "$GPU_ID"
        run_eval "Object" "$TASK_OBJECT" "$CHECKPOINT_OBJECT" "$GPU_ID"
        run_eval "Goal" "$TASK_GOAL" "$CHECKPOINT_GOAL" "$GPU_ID"
        run_eval "Long" "$TASK_LONG" "$CHECKPOINT_LONG" "$GPU_ID"
        
        echo ""
        echo "============================================================================"
        echo "All evaluations started in background!"
        echo "============================================================================"
        echo "You can monitor progress with:"
        echo "  tail -f $LOG_DIR/Spatial--chkpt.log"
        echo "  tail -f $LOG_DIR/Object--chkpt.log"
        echo "  tail -f $LOG_DIR/Goal--chkpt.log"
        echo "  tail -f $LOG_DIR/Long--chkpt.log"
        echo ""
        echo "To check if evaluations are still running:"
        echo "  ps aux | grep run_libero_eval.py"
        echo ""
        ;;
    
    *)
        echo "Error: Invalid task suite '$TASK_SUITE'"
        echo ""
        print_usage
        exit 1
        ;;
esac

echo "Done!"
