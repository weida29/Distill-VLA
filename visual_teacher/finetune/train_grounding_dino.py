#!/usr/bin/env python3
"""
Grounding DINO Fine-tuning Training Script with DDP support.

Usage:
    # Single GPU
    python train_grounding_dino.py
    
    # Multi-GPU (8 GPUs)
    torchrun --nproc_per_node=8 train_grounding_dino.py
    
    # Or use the launch script
    bash run_train_ddp.sh
"""

# Disable tokenizers parallelism warning in multiprocessing
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Add GroundingDINO to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "GroundingDINO"))

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.misc import NestedTensor
from groundingdino.util import box_ops

from odvg_dataset import build_odvg_dataset, collate_fn


# ============== Distributed Utils ==============

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):
    """Disable printing when not in master process."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ['SLURM_NTASKS'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        return

    args.distributed = True
    
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    
    print(f'| distributed init (rank {args.rank}, local_rank {args.local_rank}): ', flush=True)
    
    dist.init_process_group(
        backend=args.dist_backend,
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def reduce_dict(input_dict, average=True):
    """Reduce dict values across all processes."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


# ============== Model and Loss ==============

class GroundingDINOCriterion(nn.Module):
    """
    Loss function for Grounding DINO fine-tuning.
    Combines box regression loss and IoU-based contrastive alignment loss.
    """
    
    def __init__(
        self,
        num_classes: int = 256,
        loss_coef_bbox: float = 5.0,
        loss_coef_giou: float = 2.0,
        loss_coef_class: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.loss_coef_bbox = loss_coef_bbox
        self.loss_coef_giou = loss_coef_giou
        self.loss_coef_class = loss_coef_class
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def sigmoid_focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """Focal loss for classification."""
        p = torch.sigmoid(inputs)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()
    
    def _ensure_valid_boxes(self, boxes_xyxy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Ensure boxes have valid coordinates (x2 > x1, y2 > y1)."""
        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]
        
        x2_valid = torch.max(x2, x1 + eps)
        y2_valid = torch.max(y2, y1 + eps)
        
        boxes = torch.stack([x1, y1, x2_valid, y2_valid], dim=-1)
        boxes = boxes.clamp(min=0, max=1)
        return boxes
    
    def forward(self, outputs: dict, targets: list) -> dict:
        """
        Compute losses with token-level contrastive alignment.
        
        For Grounding DINO:
        - pred_logits: [B, num_queries, max_text_len] - logits for each token position
        - Each bbox should have high logits at its corresponding phrase's token positions
        """
        pred_logits = outputs['pred_logits']  # [B, num_queries, max_text_len]
        pred_boxes = outputs['pred_boxes']    # [B, num_queries, 4]
        
        batch_size = pred_logits.shape[0]
        device = pred_logits.device
        
        total_loss_bbox = torch.tensor(0.0, device=device)
        total_loss_giou = torch.tensor(0.0, device=device)
        total_loss_class = torch.tensor(0.0, device=device)
        num_boxes = 0
        
        for b in range(batch_size):
            tgt_boxes = targets[b]['boxes'].to(device)
            tokens_positive = targets[b].get('tokens_positive', None)  # List of [start, end] for each box
            
            if len(tgt_boxes) == 0:
                continue
            
            pred_boxes_b = pred_boxes[b]      # [num_queries, 4]
            pred_logits_b = pred_logits[b]    # [num_queries, max_text_len]
            
            # Match predictions to targets using IoU
            pred_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes_b)
            tgt_boxes_xyxy = box_ops.box_cxcywh_to_xyxy(tgt_boxes)
            
            iou_matrix, _ = box_ops.box_iou(pred_boxes_xyxy, tgt_boxes_xyxy)
            
            # Greedy matching
            matched_pred_indices = []
            iou_matrix_copy = iou_matrix.clone()
            for i in range(len(tgt_boxes)):
                best_idx = iou_matrix_copy[:, i].argmax().item()
                matched_pred_indices.append(best_idx)
                iou_matrix_copy[best_idx, :] = -1
            
            matched_pred_indices = torch.tensor(matched_pred_indices, device=device)
            matched_pred_boxes = pred_boxes_b[matched_pred_indices]
            
            # L1 loss
            loss_bbox = torch.nn.functional.l1_loss(matched_pred_boxes, tgt_boxes, reduction='sum')
            total_loss_bbox = total_loss_bbox + loss_bbox
            
            # GIoU loss
            matched_pred_xyxy = box_ops.box_cxcywh_to_xyxy(matched_pred_boxes)
            matched_pred_xyxy = self._ensure_valid_boxes(matched_pred_xyxy)
            tgt_boxes_xyxy_valid = self._ensure_valid_boxes(tgt_boxes_xyxy)
            
            giou = box_ops.generalized_box_iou(matched_pred_xyxy, tgt_boxes_xyxy_valid)
            loss_giou = 1 - torch.diag(giou)
            total_loss_giou = total_loss_giou + loss_giou.sum()
            
            # Classification loss: Token-level contrastive alignment
            # Each matched query should have high logits at its corresponding phrase's token positions
            num_queries = pred_logits_b.shape[0]
            max_text_len = pred_logits_b.shape[1]
            
            # Build token-level target using tokens_positive
            # target[query_i, token_j] = 1 if query_i should align with token_j
            target_classes = torch.zeros(num_queries, max_text_len, device=device)
            
            if tokens_positive is not None and len(tokens_positive) > 0:
                # For each matched (pred_idx, gt_idx) pair, set target tokens
                for pred_idx, gt_idx in zip(matched_pred_indices.tolist(), range(len(tgt_boxes))):
                    if gt_idx < len(tokens_positive):
                        token_span = tokens_positive[gt_idx]  # [start, end]
                        if isinstance(token_span, (list, tuple)) and len(token_span) == 2:
                            start, end = token_span
                            # Set target to 1 for tokens in this phrase
                            # Note: end is exclusive in Python slicing
                            target_classes[pred_idx, start:end] = 1.0
            
            # Apply focal loss for token-level classification
            # This helps with class imbalance (most tokens should be 0)
            loss_class = self.sigmoid_focal_loss(
                pred_logits_b,
                target_classes,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma
            )
            total_loss_class = total_loss_class + loss_class
            
            num_boxes += len(tgt_boxes)
        
        num_boxes = max(num_boxes, 1)
        
        losses = {
            'loss_bbox': self.loss_coef_bbox * total_loss_bbox / num_boxes,
            'loss_giou': self.loss_coef_giou * total_loss_giou / num_boxes,
            'loss_class': self.loss_coef_class * total_loss_class / num_boxes,
        }
        losses['loss_total'] = losses['loss_bbox'] + losses['loss_giou'] + losses['loss_class']
        
        return losses


def load_model(config_path: str, pretrained_path: str = None, device: str = 'cuda') -> nn.Module:
    """Load Grounding DINO model."""
    args = SLConfig.fromfile(config_path)
    args.device = device
    
    model = build_model(args)
    
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    
    model = model.to(device)
    return model


def freeze_model_for_finetuning(model, unfreeze_decoder_layers: int = 2, freeze_text_encoder: bool = True):
    """
    Freeze model parameters for efficient fine-tuning.
    
    Strategy: Only train bbox_embed and last few decoder layers.
    This preserves the pretrained visual-language alignment while
    allowing the model to adapt bbox predictions to the new domain.
    
    Args:
        model: Grounding DINO model
        unfreeze_decoder_layers: Number of decoder layers to unfreeze from the end (default 2)
        freeze_text_encoder: Whether to freeze BERT text encoder (default True)
    
    Returns:
        Number of trainable parameters
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze bbox_embed (the bbox regression head)
    if hasattr(model, 'bbox_embed'):
        for param in model.bbox_embed.parameters():
            param.requires_grad = True
        print("Unfroze: bbox_embed")
    
    # Unfreeze the last N decoder layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'decoder'):
        decoder = model.transformer.decoder
        num_layers = len(decoder.layers)
        start_layer = max(0, num_layers - unfreeze_decoder_layers)
        
        for i in range(start_layer, num_layers):
            for param in decoder.layers[i].parameters():
                param.requires_grad = True
            print(f"Unfroze: decoder.layers[{i}]")
        
        # Also unfreeze decoder norm if exists
        if hasattr(decoder, 'norm') and decoder.norm is not None:
            for param in decoder.norm.parameters():
                param.requires_grad = True
            print("Unfroze: decoder.norm")
    
    # Optionally unfreeze text encoder (not recommended for preserving generalization)
    if not freeze_text_encoder and hasattr(model, 'bert'):
        for param in model.bert.parameters():
            param.requires_grad = True
        print("Unfroze: bert (text encoder)")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return trainable_params


# ============== Training ==============

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    max_norm: float = 0.1,
    writer: SummaryWriter = None,
    log_freq: int = 10,
) -> dict:
    """Train for one epoch."""
    model.train()
    criterion.train()
    
    total_losses = {}
    num_batches = len(dataloader)
    
    for batch_idx, (samples, targets) in enumerate(dataloader):
        samples = samples.to(device)
        
        outputs = model(samples, targets=targets)
        losses = criterion(outputs, targets)
        
        optimizer.zero_grad()
        losses['loss_total'].backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
        
        # Reduce losses across processes
        loss_dict_reduced = reduce_dict({k: v.detach() for k, v in losses.items()})
        
        for k, v in loss_dict_reduced.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
        
        # Log progress (only main process)
        if is_main_process() and batch_idx % log_freq == 0:
            loss_str = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict_reduced.items()])
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{num_batches}] {loss_str}")
        
        if writer is not None and is_main_process():
            global_step = epoch * num_batches + batch_idx
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'train/{k}', v.item(), global_step)
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    criterion.eval()
    
    total_losses = {}
    num_batches = len(dataloader)
    
    for samples, targets in dataloader:
        samples = samples.to(device)
        
        outputs = model(samples, targets=targets)
        losses = criterion(outputs, targets)
        
        loss_dict_reduced = reduce_dict({k: v.detach() for k, v in losses.items()})
        
        for k, v in loss_dict_reduced.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    return avg_losses


# ============== Main ==============

def main(args):
    """Main training function."""
    
    # Initialize distributed mode
    init_distributed_mode(args)
    
    # Setup device
    if args.distributed:
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Setup paths
    project_root = SCRIPT_DIR.parent.parent
    
    if args.train_jsonl is None:
        args.train_jsonl = str(project_root / "data_processed" / "grounding_dino_dataset" / "train.jsonl")
    if args.val_jsonl is None:
        val_path = project_root / "data_processed" / "grounding_dino_dataset" / "val.jsonl"
        args.val_jsonl = str(val_path) if val_path.exists() else None
    if args.image_dir is None:
        args.image_dir = str(project_root / "data_processed" / "grounding_dino_dataset")
    if args.config is None:
        args.config = str(SCRIPT_DIR.parent / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
    if args.pretrained is None:
        args.pretrained = str(SCRIPT_DIR.parent / "pretrained_ckpt" / "groundingdino_swint_ogc.pth")
    if args.output_dir is None:
        args.output_dir = str(project_root / "checkpoints" / "grounding_dino_finetuned")
    
    # Create output directory (only main process)
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.distributed:
        dist.barrier()
    
    # Setup TensorBoard (only main process)
    writer = None
    if is_main_process():
        writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    
    print("=" * 60)
    print("Grounding DINO Fine-tuning (DDP)")
    print("=" * 60)
    print(f"World size: {args.world_size}")
    print(f"Rank: {args.rank}")
    print(f"Config: {args.config}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Train JSONL: {args.train_jsonl}")
    print(f"Val JSONL: {args.val_jsonl}")
    print(f"Image dir: {args.image_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Effective batch size: {args.batch_size * args.world_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Build datasets
    print("\nLoading datasets...")
    train_dataset = build_odvg_dataset(
        jsonl_path=args.train_jsonl,
        image_dir=args.image_dir,
        image_size=args.image_size,
        is_train=True,
    )
    
    val_dataset = None
    if args.val_jsonl and os.path.exists(args.val_jsonl):
        val_dataset = build_odvg_dataset(
            jsonl_path=args.val_jsonl,
            image_dir=args.image_dir,
            image_size=args.image_size,
            is_train=False,
        )
    
    # Build samplers for DDP
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if val_dataset else None
    else:
        train_sampler = None
        val_sampler = None
    
    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val samples: {len(val_dataset)}")
    
    # Build model
    print("\nLoading model...")
    model = load_model(
        config_path=args.config,
        pretrained_path=args.pretrained,
        device=device,
    )
    
    # Wrap model with DDP
    # Note: Grounding DINO uses torch.utils.checkpoint which requires static_graph=True
    if args.distributed:
        model = DDP(
            model, 
            device_ids=[args.local_rank], 
            find_unused_parameters=False,  # Must be False when using static_graph
            gradient_as_bucket_view=True,
        )
        # Set static graph for compatibility with gradient checkpointing
        model._set_static_graph()
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Apply parameter-efficient fine-tuning if specified
    if args.efficient_finetune:
        print("\n" + "=" * 50)
        print("Applying parameter-efficient fine-tuning strategy")
        print("=" * 50)
        freeze_model_for_finetuning(
            model_without_ddp,
            unfreeze_decoder_layers=args.unfreeze_decoder_layers,
            freeze_text_encoder=args.freeze_text_encoder
        )
    elif args.freeze_backbone:
        # Legacy option: only freeze backbone
        print("Freezing backbone...")
        for name, param in model_without_ddp.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Build criterion
    criterion = GroundingDINOCriterion(
        loss_coef_bbox=args.loss_coef_bbox,
        loss_coef_giou=args.loss_coef_giou,
        loss_coef_class=args.loss_coef_class,
    )
    
    # Build optimizer - use even smaller lr for backbone
    param_groups = [
        {'params': [p for n, p in model_without_ddp.named_parameters() if 'backbone' not in n and p.requires_grad]},
        {'params': [p for n, p in model_without_ddp.named_parameters() if 'backbone' in n and p.requires_grad], 'lr': args.lr * 0.01},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # Linear warmup
        else:
            # Step decay after warmup
            decay_epochs = [args.lr_drop - warmup_epochs]
            factor = 1.0
            for de in decay_epochs:
                if epoch - warmup_epochs >= de:
                    factor *= 0.1
            return factor
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for sampler
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'=' * 40}")
        
        # Train
        train_losses = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=args.clip_max_norm,
            writer=writer,
        )
        
        lr_scheduler.step()
        
        # Log training losses
        train_loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_losses.items()])
        print(f"Train - {train_loss_str}")
        
        # Evaluate
        if val_loader:
            val_losses = evaluate(model, criterion, val_loader, device)
            val_loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_losses.items()])
            print(f"Val   - {val_loss_str}")
            
            if writer is not None:
                for k, v in val_losses.items():
                    writer.add_scalar(f'val/{k}', v, epoch)
            
            # Save best model (only main process)
            if is_main_process() and val_losses['loss_total'] < best_val_loss:
                best_val_loss = val_losses['loss_total']
                save_path = output_dir / "best_model.pth"
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                }, save_path)
                print(f"Saved best model to {save_path}")
        
        # Save checkpoint (only main process)
        if is_main_process() and (epoch + 1) % args.save_freq == 0:
            save_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
    
    # Save final model (only main process)
    if is_main_process():
        save_path = output_dir / "final_model.pth"
        torch.save({
            'model': model_without_ddp.state_dict(),
            'epoch': args.epochs - 1,
        }, save_path)
        print(f"\nSaved final model to {save_path}")
        
        if writer is not None:
            writer.close()
    
    # Cleanup
    if args.distributed:
        dist.destroy_process_group()
    
    print("\nTraining completed!")


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding DINO Fine-tuning with DDP")
    
    # Data
    parser.add_argument('--train-jsonl', type=str, default=None)
    parser.add_argument('--val-jsonl', type=str, default=None)
    parser.add_argument('--image-dir', type=str, default=None)
    parser.add_argument('--image-size', type=int, default=800)
    
    # Model
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--freeze-backbone', action='store_true')
    
    # Parameter-efficient fine-tuning options
    parser.add_argument('--efficient-finetune', action='store_true', 
                        help='Enable parameter-efficient fine-tuning (freeze most layers)')
    parser.add_argument('--unfreeze-decoder-layers', type=int, default=2,
                        help='Number of decoder layers to unfreeze from the end (default: 2)')
    parser.add_argument('--freeze-text-encoder', action='store_true', default=True,
                        help='Freeze BERT text encoder (default: True)')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=15)  # Fewer epochs for parameter-efficient finetuning
    parser.add_argument('--lr', type=float, default=1e-6)  # Very small lr to prevent catastrophic forgetting
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lr-drop', type=int, default=10)  # Adjusted for fewer epochs
    parser.add_argument('--clip-max-norm', type=float, default=0.5)
    
    # Loss
    parser.add_argument('--loss-coef-bbox', type=float, default=5.0)
    parser.add_argument('--loss-coef-giou', type=float, default=2.0)
    parser.add_argument('--loss-coef-class', type=float, default=0.0)  # Disable class loss to preserve confidence distribution
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--save-freq', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    
    # Misc
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Distributed (auto-detected, usually don't need to set)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--dist-url', default='env://')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)
