#!/usr/bin/env python3
"""
train_align.py

VLA-GDINO Feature Alignment Training.

Multi-task training:
1. Action prediction (L1 regression) - original VLA task
2. Feature alignment (MSE) - align VLA grounding features with GDINO teacher

Architecture:
- VLA (Prismatic VLM) + Grounding Queries → LLM → Hidden states
- Grounding Module: Project grounding query hidden states to GDINO space (hs, ref)
- GDINO Teacher: Frozen, provides target features
- Loss: action_loss + alignment_loss (hs + ref)

Usage:
    python align_train/train_align.py
    torchrun --nproc_per_node=8 align_train/train_align.py
"""

import json
import os
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForVision2Seq
from transformers.modeling_outputs import CausalLMOutputWithPast

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from align_train.config import AlignTrainConfig
from align_train.models import GroundingModule, GDINOTeacher
from align_train.losses import FeatureAlignmentLoss

# Import from existing VLA code
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_TOKENS
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.models import load


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# Utility Functions
# ============================================================================

def get_torch_dtype(use_bf16: bool) -> torch.dtype:
    """Get appropriate torch dtype. V100 doesn't support bf16."""
    if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def setup_distributed():
    """Initialize distributed training."""
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    return distributed_state, device_id


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    """Wrap module with DistributedDataParallel."""
    return DDP(module, device_ids=[device_id], find_unused_parameters=find_unused, gradient_as_bucket_view=True)


def get_base_model(model):
    """Get base model from DDP/LoRA wrapped model."""
    if hasattr(model, 'module'):
        model = model.module
    if hasattr(model, 'base_model'):
        model = model.base_model
    if hasattr(model, 'model'):
        model = model.model
    return model


# ============================================================================
# Model Loading
# ============================================================================

def load_vla_model(cfg: AlignTrainConfig, device, dtype: torch.dtype) -> Tuple[nn.Module, dict]:
    """Load VLA model following finetune.py pattern."""
    print("Loading VLA model...")
    
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    
    # Register OpenVLA model to HF Auto Classes (required for local model)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    
    config = AutoConfig.from_pretrained(str(PROJECT_ROOT / cfg.vla_config_path))
    vla = AutoModelForVision2Seq.from_config(config, torch_dtype=dtype)
    
    raw_state_dict = {}
    
    if cfg.vla_checkpoint and os.path.exists(cfg.vla_checkpoint):
        print(f"Loading VLA checkpoint from: {cfg.vla_checkpoint}")
        checkpoint = torch.load(cfg.vla_checkpoint, map_location='cpu', weights_only=True)
        vla.load_state_dict(checkpoint.get('model', checkpoint), strict=False)
    elif cfg.vlm_path:
        print(f"Loading VLM weights from: {cfg.vlm_path}")
        vlm = load(str(PROJECT_ROOT / cfg.vlm_path), hf_token='', load_for_training=True)
        
        replace_map = [
            ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
            ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
            ("llm_backbone.llm", "language_model"),
            ("projector.projector.0", "projector.fc1"),
            ("projector.projector.2", "projector.fc2"),
            ("projector.projector.4", "projector.fc3"),
            ("gamma", "scale_factor"),
        ]
        
        old_state_dict = vlm.state_dict()
        new_state_dict = {}
        for k, v in old_state_dict.items():
            new_k = k
            for old, new in replace_map:
                if old in new_k:
                    new_k = new_k.replace(old, new)
            new_state_dict[new_k] = v
        
        raw_state_dict = new_state_dict
        missing, unexpected = vla.load_state_dict(new_state_dict, strict=False)
        print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        del old_state_dict, vlm
    
    vla = vla.to(device)
    
    # Apply LoRA if configured
    if cfg.use_lora:
        print(f"Applying LoRA with rank={cfg.lora_rank}")
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=2 * cfg.lora_rank,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        for name, param in vla.named_parameters():
            if "action_queries" in name:
                param.requires_grad = True
        vla.print_trainable_parameters()
    
    total_params = sum(p.numel() for p in vla.parameters())
    trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    print(f"VLA loaded: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")
    
    return vla, raw_state_dict


# ============================================================================
# Training Step
# ============================================================================

def train_step(
    batch: Dict,
    vla: nn.Module,
    action_head: nn.Module,
    grounding_module: GroundingModule,
    gdino_teacher: GDINOTeacher,
    align_criterion: FeatureAlignmentLoss,
    cfg: AlignTrainConfig,
    device,
    dtype: torch.dtype,
    num_patches: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Single training step with action loss + alignment loss.
    
    Following finetune.py pattern for action prediction.
    """
    metrics = {}
    batch_size = batch["input_ids"].shape[0]
    
    # Get ground-truth actions
    ground_truth_actions = batch["actions"].to(device).to(dtype)
    
    # ============ 1. VLA Forward Pass ============
    with torch.autocast("cuda", dtype=dtype):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(dtype).to(device),
            labels=batch["labels"],
            output_hidden_states=True,
        )
    
    # ============ 2. Action Loss (L1 Regression) ============
    # Following finetune.py pattern
    ground_truth_token_ids = batch["labels"][:, 1:].to(device)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    
    # Get multi-layer hidden states for action head
    multi_layer_hidden_states = []
    for item in output.hidden_states[0:]:
        text_hidden_states = item[:, num_patches:-1]
        actions_hidden_states = text_hidden_states[current_action_mask | next_actions_mask].reshape(
            batch_size, 1, NUM_TOKENS, -1
        ).to(dtype)
        task_latten_states = item[:, :num_patches].reshape(batch_size, 1, num_patches, -1)
        all_hidden_states = torch.cat((task_latten_states, actions_hidden_states), 2)
        multi_layer_hidden_states.append(all_hidden_states)
    multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)
    
    # Action head prediction
    action_head_module = action_head.module if hasattr(action_head, 'module') else action_head
    predicted_actions = action_head_module.predict_action(
        multi_layer_hidden_states,
        proprio=None,
        proprio_projector=None,
        phase="Training",
    )
    
    action_loss = F.l1_loss(predicted_actions, ground_truth_actions)
    
    # Action metrics
    curr_action_l1 = F.l1_loss(predicted_actions[:, 0], ground_truth_actions[:, 0])
    next_actions_l1 = F.l1_loss(predicted_actions[:, 1:], ground_truth_actions[:, 1:])
    metrics["curr_action_l1"] = curr_action_l1.item()
    metrics["next_actions_l1"] = next_actions_l1.item()
    
    # ============ 3. Grounding Feature Alignment ============
    # Get grounding query embeddings and append to LLM
    base_vla = get_base_model(vla)
    grounding_module_unwrapped = grounding_module.module if hasattr(grounding_module, 'module') else grounding_module
    
    with torch.autocast("cuda", dtype=dtype):
        # Get visual features
        visual_features = base_vla.vision_backbone(batch["pixel_values"].to(dtype).to(device))
        projected_visual = base_vla.projector(visual_features)
        
        # Get text embeddings
        text_embeds = base_vla.language_model.get_input_embeddings()(batch["input_ids"].to(device))
        
        # Get grounding query embeddings
        grounding_embeds = grounding_module_unwrapped.get_query_embeddings(batch_size, device)
        num_grounding_queries = grounding_embeds.shape[1]
        
        # Concatenate: [visual | text | grounding_queries]
        combined_embeds = torch.cat([projected_visual, text_embeds, grounding_embeds.to(dtype)], dim=1)
        
        # Create combined attention mask
        visual_mask = torch.ones(batch_size, projected_visual.shape[1], dtype=torch.long, device=device)
        grounding_mask = torch.ones(batch_size, num_grounding_queries, dtype=torch.long, device=device)
        combined_attention_mask = torch.cat([
            visual_mask,
            batch["attention_mask"].to(device),
            grounding_mask
        ], dim=1)
        
        # LLM forward with grounding queries
        llm_output = base_vla.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Extract grounding query hidden states (at the end of sequence)
        last_hidden_state = llm_output.hidden_states[-1]
        grounding_hidden_states = last_hidden_state[:, -num_grounding_queries:, :]
        
        # Project to GDINO space
        student_hs, student_ref = grounding_module_unwrapped(grounding_hidden_states)
    
    # ============ 4. Teacher Forward (Frozen) ============
    # Get task description as caption for GDINO
    captions = batch.get("task_description", None)
    if captions is None:
        # Fallback: generic caption
        captions = ["pick up the object"] * batch_size
    
    with torch.no_grad():
        teacher_outputs = gdino_teacher(
            batch["pixel_values"].to(dtype).to(device),
            captions
        )
        teacher_hs = teacher_outputs["teacher_hs"]
        teacher_ref = teacher_outputs["teacher_ref"]
    
    # ============ 5. Alignment Loss (MSE) ============
    align_losses = align_criterion(
        student_hs=student_hs,
        teacher_hs=teacher_hs.to(student_hs.dtype),
        student_ref=student_ref,
        teacher_ref=teacher_ref.to(student_ref.dtype),
    )
    
    # ============ 6. Total Loss ============
    total_loss = (
        cfg.action_loss_weight * action_loss +
        align_losses["loss_align"]
    )
    
    # Record metrics
    metrics["loss_action"] = action_loss.item()
    metrics["loss_hs"] = align_losses["loss_hs"].item()
    metrics["loss_ref"] = align_losses["loss_ref"].item()
    metrics["loss_align"] = align_losses["loss_align"].item()
    metrics["loss_total"] = total_loss.item()
    
    return total_loss, metrics


# ============================================================================
# Checkpoint Saving
# ============================================================================

def save_checkpoint(
    run_dir: Path,
    log_step: int,
    vla: nn.Module,
    action_head: nn.Module,
    grounding_module: GroundingModule,
    optimizer: torch.optim.Optimizer,
    scheduler,
    processor,
    train_dataset,
    cfg: AlignTrainConfig,
    distributed_state,
    is_final: bool = False,
) -> None:
    """Save training checkpoint."""
    checkpoint_dir = run_dir / ("final" if is_final else f"checkpoint_{log_step}")
    
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"\nSaving checkpoint for Step {log_step}")
    
    if distributed_state.num_processes > 1:
        dist.barrier()
    
    if distributed_state.is_main_process:
        # Save processor
        processor.save_pretrained(checkpoint_dir)
        
        # Save VLA
        vla_module = vla.module if hasattr(vla, 'module') else vla
        vla_module.save_pretrained(checkpoint_dir / "vla")
        
        # Save action head
        action_head_module = action_head.module if hasattr(action_head, 'module') else action_head
        torch.save(action_head_module.state_dict(), checkpoint_dir / f"action_head_{log_step}.pt")
        
        # Save grounding module
        grounding_module_unwrapped = grounding_module.module if hasattr(grounding_module, 'module') else grounding_module
        torch.save(grounding_module_unwrapped.state_dict(), checkpoint_dir / f"grounding_module_{log_step}.pt")
        
        # Save optimizer and scheduler
        torch.save({
            "step": log_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, checkpoint_dir / "training_state.pt")
        
        print(f"Saved checkpoint to {checkpoint_dir}")
    
    if distributed_state.num_processes > 1:
        dist.barrier()


# ============================================================================
# Main Training Function
# ============================================================================

def main(cfg: AlignTrainConfig):
    """Main training function."""
    
    # Setup
    distributed_state, device_id = setup_distributed()
    device = torch.device(f"cuda:{device_id}")
    is_main = distributed_state.is_main_process
    
    dtype = get_torch_dtype(cfg.use_bf16)
    print(f"Using dtype: {dtype}")
    
    # Create run directory
    run_dir = cfg.run_root_dir / cfg.run_id
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)
    
    if distributed_state.num_processes > 1:
        dist.barrier()
    
    print("=" * 60)
    print("VLA-GDINO Feature Alignment Training")
    print("=" * 60)
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Run ID: {cfg.run_id}")
    print(f"Loss weights: action={cfg.action_loss_weight}, hs={cfg.hs_weight}, ref={cfg.ref_weight}")
    print("=" * 60)
    
    # ============ Load Models ============
    
    # Load GDINO teacher (frozen)
    print("\nLoading GDINO Teacher...")
    gdino_teacher = GDINOTeacher(
        config_path=cfg.gdino_config,
        checkpoint_path=cfg.gdino_checkpoint,
        device=str(device),
        freeze=True,
    )
    
    # Load VLA
    vla, raw_state_dict = load_vla_model(cfg, device, dtype)
    
    # Get LLM dim before wrapping
    base_vla = get_base_model(vla)
    llm_dim = base_vla.llm_dim
    num_patches = base_vla.vision_backbone.get_num_patches()
    print(f"LLM dim: {llm_dim}, Num patches: {num_patches}")
    
    # Create action head
    action_head = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=ACTION_DIM,
        use_pro_version=True,
    ).to(device).to(dtype)
    print(f"Action Head: {sum(p.numel() for p in action_head.parameters()) / 1e6:.2f}M params")
    
    # Create grounding module
    grounding_module = GroundingModule(
        num_queries=cfg.num_grounding_queries,
        llm_dim=llm_dim,
        gdino_dim=cfg.gdino_dim,
        dropout=cfg.grounding_dropout,
    ).to(device).to(dtype)
    print(f"Grounding Module: {sum(p.numel() for p in grounding_module.parameters()) / 1e6:.2f}M params")
    
    # Wrap with DDP if distributed
    if distributed_state.num_processes > 1:
        vla = wrap_ddp(vla, device_id, find_unused=True)
        action_head = wrap_ddp(action_head, device_id)
        grounding_module = wrap_ddp(grounding_module, device_id)
    
    # ============ Loss & Optimizer ============
    
    align_criterion = FeatureAlignmentLoss(
        hs_weight=cfg.hs_weight,
        ref_weight=cfg.ref_weight,
    )
    
    # Collect trainable parameters
    trainable_params = [p for p in vla.parameters() if p.requires_grad]
    trainable_params += list(action_head.parameters())
    trainable_params += list(grounding_module.parameters())
    
    print(f"Total trainable params: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)
    
    # ============ Data ============
    
    processor = PrismaticProcessor.from_pretrained(
        str(PROJECT_ROOT / cfg.vla_config_path.replace("config.json", "")),
        trust_remote_code=True,
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=False,
        use_proprio=False,
        use_minivlm=True,
    )
    
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    print(f"Dataloader created with batch_size={cfg.batch_size}")
    
    if is_main:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    
    # ============ Training Loop ============
    
    print("\nStarting training...")
    
    recent_metrics = {
        "loss_total": deque(maxlen=cfg.grad_accumulation_steps),
        "loss_action": deque(maxlen=cfg.grad_accumulation_steps),
        "loss_hs": deque(maxlen=cfg.grad_accumulation_steps),
        "loss_ref": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1": deque(maxlen=cfg.grad_accumulation_steps),
    }
    
    if is_main and cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=f"align+{cfg.run_id}",
                mode="offline",
            )
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")
            cfg.use_wandb = False
    
    vla.train()
    action_head.train()
    grounding_module.train()
    optimizer.zero_grad()
    
    log_step = 0
    with tqdm(total=cfg.max_steps, leave=False, disable=not is_main) as progress:
        for batch_idx, batch in enumerate(dataloader):
            
            loss, metrics = train_step(
                batch=batch,
                vla=vla,
                action_head=action_head,
                grounding_module=grounding_module,
                gdino_teacher=gdino_teacher,
                align_criterion=align_criterion,
                cfg=cfg,
                device=device,
                dtype=dtype,
                num_patches=num_patches,
            )
            
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()
            
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)
            
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            smoothened_metrics = {k: sum(v) / len(v) for k, v in recent_metrics.items() if len(v) > 0}
            
            log_step = gradient_step_idx if not cfg.resume else cfg.resume_step + gradient_step_idx
            
            if is_main and log_step % cfg.log_freq == 0:
                log_str = f"Step {log_step}: "
                log_str += " | ".join([f"{k}: {v:.4f}" for k, v in smoothened_metrics.items()])
                progress.set_description(log_str)
                
                if cfg.use_wandb:
                    import wandb
                    wandb.log(smoothened_metrics, step=log_step)
                    wandb.log({"lr": scheduler.get_last_lr()[0]}, step=log_step)
            
            # LR warmup
            if cfg.warmup_steps > 0 and gradient_step_idx < cfg.warmup_steps:
                lr_progress = min((gradient_step_idx + 1) / cfg.warmup_steps, 1.0)
                current_lr = original_lr * (cfg.lr_warmup_ratio + (1 - cfg.lr_warmup_ratio) * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            
            # Optimizer step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()
            
            # Save checkpoint
            if gradient_step_idx > 0 and log_step % cfg.save_freq == 0 and is_main:
                save_checkpoint(
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    action_head=action_head,
                    grounding_module=grounding_module,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    processor=processor,
                    train_dataset=train_dataset,
                    cfg=cfg,
                    distributed_state=distributed_state,
                )
            
            if log_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached!")
                break
    
    # Final save
    if is_main:
        save_checkpoint(
            run_dir=run_dir,
            log_step=log_step,
            vla=vla,
            action_head=action_head,
            grounding_module=grounding_module,
            optimizer=optimizer,
            scheduler=scheduler,
            processor=processor,
            train_dataset=train_dataset,
            cfg=cfg,
            distributed_state=distributed_state,
            is_final=True,
        )
        
        if cfg.use_wandb:
            import wandb
            wandb.finish()
    
    print("\nTraining completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="VLA-GDINO Feature Alignment Training")
    
    parser.add_argument("--vlm_path", type=str, default=None)
    parser.add_argument("--vla_checkpoint", type=str, default=None)
    parser.add_argument("--gdino_checkpoint", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--grad_accumulation_steps", type=int, default=None)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--action_loss_weight", type=float, default=None)
    parser.add_argument("--hs_weight", type=float, default=None)
    parser.add_argument("--ref_weight", type=float, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--run_id_note", type=str, default=None)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=None)
    
    args = parser.parse_args()
    
    cfg = AlignTrainConfig()
    
    # Override from command line
    if args.vlm_path:
        cfg.vlm_path = args.vlm_path
    if args.vla_checkpoint:
        cfg.vla_checkpoint = args.vla_checkpoint
    if args.gdino_checkpoint:
        cfg.gdino_checkpoint = args.gdino_checkpoint
    if args.dataset_name:
        cfg.dataset_name = args.dataset_name
    if args.data_root_dir:
        cfg.data_root_dir = Path(args.data_root_dir)
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.learning_rate:
        cfg.learning_rate = args.learning_rate
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.grad_accumulation_steps:
        cfg.grad_accumulation_steps = args.grad_accumulation_steps
    if args.use_bf16:
        cfg.use_bf16 = True
    if args.action_loss_weight:
        cfg.action_loss_weight = args.action_loss_weight
    if args.hs_weight:
        cfg.hs_weight = args.hs_weight
    if args.ref_weight:
        cfg.ref_weight = args.ref_weight
    if args.run_id:
        cfg.run_id = args.run_id
    if args.run_id_note:
        cfg.run_id_note = args.run_id_note
    if args.no_lora:
        cfg.use_lora = False
    if args.lora_rank:
        cfg.lora_rank = args.lora_rank
    
    cfg.__post_init__()
    
    main(cfg)
