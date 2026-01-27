"""
config.py

Configuration dataclass for VLA-GDINO alignment training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class AlignTrainConfig:
    """Configuration for VLA-GDINO feature alignment training."""
    
    # ============ Model Paths ============
    # VLA model
    vla_config_path: str = "pretrained_models/configs/config.json"
    vla_checkpoint: Optional[str] = None  # Path to pretrained VLA checkpoint
    vlm_path: str = "pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
    
    # Grounding DINO teacher (Open-GroundingDino)
    gdino_config: str = "visual_teacher/Open-GroundingDino/config/cfg_odvg.py"
    gdino_checkpoint: str = "checkpoints/open_gdino_finetuned/checkpoint_best_regular.pth"
    bert_path: str = "pretrained_models/bert-base-uncased"  # BERT for text encoding
    
    # ============ Data Paths ============
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "libero_spatial_no_noops"
    bbox_annotation_file: str = "data_processed/bbox.json"
    shuffle_buffer_size: int = 100_000
    
    # ============ Output ============
    run_root_dir: Path = Path("runs/align_train")
    run_id: Optional[str] = None
    run_id_note: Optional[str] = None
    
    # ============ Training ============
    batch_size: int = 8
    learning_rate: float = 5e-4  # Match finetune.py default
    adapter_lr: float = 1e-4  # Learning rate for grounding adapter
    weight_decay: float = 0.01
    max_steps: int = 50000
    warmup_steps: int = 1000
    lr_warmup_ratio: float = 0.1  # Warmup from 10% to 100%
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    num_steps_before_decay: int = 100000  # LR decay milestone
    
    # Checkpointing
    save_freq: int = 5000
    eval_freq: int = 1000
    log_freq: int = 10
    
    # ============ Loss Weights ============
    # Action prediction loss (original VLA task, L1 regression)
    action_loss_weight: float = 1.0
    
    # Feature alignment losses (MSE: student vs teacher)
    hs_weight: float = 1.0              # Hidden state alignment weight
    ref_weight: float = 1.0             # Reference point alignment weight
    
    # Total loss = action_loss_weight * action_loss + hs_weight * loss_hs + ref_weight * loss_ref
    
    # ============ Model Architecture ============
    # Grounding Module
    num_grounding_queries: int = 900  # Number of grounding queries (same as GDINO)
    gdino_dim: int = 256              # Grounding DINO hidden dimension
    grounding_dropout: float = 0.1    # Dropout in grounding module
    
    # VLA dimensions
    llm_dim: int = 896                # LLM hidden dimension (Qwen2.5-0.5B)
    vla_dim: int = 1152               # Vision backbone output (DINOv2(768) + SigLIP(384))
    
    # ============ Training Mode ============
    # What to train
    train_vla_backbone: bool = False      # Train VLA vision backbone
    train_vla_projector: bool = False     # Train VLA projector
    train_vla_llm: bool = False           # Train VLA LLM backbone
    train_vla_action_head: bool = True    # Train VLA action head
    train_grounding_adapter: bool = True  # Train grounding feature adapter
    
    # Use LoRA for VLA
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    
    # ============ Data Augmentation ============
    image_aug: bool = True
    
    # ============ Hardware ============
    device: str = "cuda"
    num_workers: int = 0  # Important: Set to 0 if using RLDS (uses its own parallelism)
    mixed_precision: bool = True
    use_bf16: bool = False  # V100 doesn't support bf16, set True only for A100/H100
    
    # ============ Logging ============
    wandb_project: str = "vla-gdino-align"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True
    
    # ============ Resume ============
    resume: bool = False
    resume_checkpoint: Optional[str] = None
    resume_step: Optional[int] = None
    
    def __post_init__(self):
        """Validate and process config after initialization."""
        # Convert paths
        self.data_root_dir = Path(self.data_root_dir)
        self.run_root_dir = Path(self.run_root_dir)
        
        # Generate run ID if not provided
        if self.run_id is None:
            self.run_id = f"align+{self.dataset_name}+b{self.batch_size}+lr{self.learning_rate}"
            if self.run_id_note:
                self.run_id += f"--{self.run_id_note}"


@dataclass  
class GDINOConfig:
    """Configuration for Open-GroundingDino model (for reference)."""
    
    # Model architecture
    hidden_dim: int = 256
    num_queries: int = 900
    num_feature_levels: int = 4
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    
    # Text encoder
    text_encoder_type: str = "bert-base-uncased"
    max_text_len: int = 256
    
    # Detection head
    dec_pred_bbox_embed_share: bool = True
    two_stage_type: str = "standard"
    
    # Open-GroundingDino specific
    use_coco_eval: bool = False
    nms_iou_threshold: float = 0.7


# Default configuration
DEFAULT_CONFIG = AlignTrainConfig()

