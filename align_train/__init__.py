"""
align_train - VLA Feature Alignment with Grounding DINO

This module implements feature alignment training between VLA backbone 
and Grounding DINO visual teacher.

Key components:
- GroundingFeatureAdapter: Adapts VLA features to GDINO decoder input format
- GroundingDistillLoss: Combined loss (KL + BBox + Class)
- AlignTrainConfig: Training configuration

Usage:
    # Single GPU
    python align_train/train_align.py --dataset_name libero_spatial_no_noops
    
    # Multi-GPU
    torchrun --nproc_per_node=8 align_train/train_align.py
"""

from .grounding_adapter import GroundingFeatureAdapter, GroundingFeatureAdapterV2
from .losses import GroundingDistillLoss, HungarianMatcher
from .config import AlignTrainConfig

__all__ = [
    "GroundingFeatureAdapter",
    "GroundingFeatureAdapterV2", 
    "GroundingDistillLoss",
    "HungarianMatcher",
    "AlignTrainConfig",
]

