"""
losses.py

Loss functions for VLA-GDINO feature alignment training.
Combines KL distillation loss with bbox and classification supervision.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format
        boxes2: [M, 4] in (x1, y1, x2, y2) format
        
    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU between two sets of boxes.
    
    GIoU = IoU - (C - Union) / C
    where C is the smallest enclosing box.
    """
    # Ensure valid boxes
    boxes1 = boxes1.clamp(min=0, max=1)
    boxes2 = boxes2.clamp(min=0, max=1)
    
    # IoU
    iou = box_iou(boxes1, boxes2)
    
    # Enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]
    
    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2 - iou * (area1[:, None] + area2) / (iou + 1e-6)
    
    giou = iou - (area_c - union) / (area_c + 1e-6)
    
    return giou


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for optimal bipartite matching between predictions and targets.
    
    This is used to match predicted boxes with ground truth boxes for loss computation.
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching.
        
        Args:
            outputs: dict with 'pred_logits' [B, num_queries, text_len] 
                     and 'pred_boxes' [B, num_queries, 4]
            targets: list of dicts with 'boxes' [num_gt, 4] and 'labels' [num_gt]
            
        Returns:
            List of (pred_indices, gt_indices) tuples for each batch element
        """
        B, num_queries = outputs["pred_boxes"].shape[:2]
        
        # Flatten predictions for easier matching
        out_prob = outputs["pred_logits"].sigmoid()  # [B, num_queries, text_len]
        out_bbox = outputs["pred_boxes"]  # [B, num_queries, 4]
        
        indices = []
        for b in range(B):
            tgt_boxes = targets[b]["boxes"]  # [num_gt, 4]
            num_gt = tgt_boxes.shape[0]
            
            if num_gt == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=out_bbox.device),
                    torch.tensor([], dtype=torch.long, device=out_bbox.device),
                ))
                continue
            
            # Get max confidence for each query as "class" score
            out_prob_b = out_prob[b].max(dim=-1).values  # [num_queries]
            
            # Compute costs
            # Classification cost: negative confidence for matched targets
            cost_class = -out_prob_b.unsqueeze(1).expand(-1, num_gt)  # [num_queries, num_gt]
            
            # L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox[b], tgt_boxes, p=1)  # [num_queries, num_gt]
            
            # GIoU cost
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_boxes),
            )  # [num_queries, num_gt]
            
            # Total cost
            C = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )
            
            # Hungarian matching
            C_np = C.cpu().numpy()
            pred_idx, gt_idx = linear_sum_assignment(C_np)
            
            indices.append((
                torch.tensor(pred_idx, dtype=torch.long, device=out_bbox.device),
                torch.tensor(gt_idx, dtype=torch.long, device=out_bbox.device),
            ))
        
        return indices


class GroundingDistillLoss(nn.Module):
    """
    Combined loss for VLA-GDINO feature alignment.
    
    Loss = λ_kl * KL_Loss + λ_bbox * (L1 + GIoU) + λ_class * Class_Loss
    
    Components:
    1. KL Loss: Align student logits with teacher logits (soft distillation)
    2. BBox Loss: L1 + GIoU for box regression (hard supervision with GT)
    3. Class Loss: Focal loss for object-text alignment (hard supervision with GT)
    """
    
    def __init__(
        self,
        kl_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        class_weight: float = 1.0,
        temperature: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_hungarian: bool = True,
    ):
        super().__init__()
        
        self.kl_weight = kl_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.class_weight = class_weight
        self.temperature = temperature
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_hungarian = use_hungarian
        
        if use_hungarian:
            self.matcher = HungarianMatcher(
                cost_class=class_weight,
                cost_bbox=bbox_weight,
                cost_giou=giou_weight,
            )
    
    def forward(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Optional[Dict[str, torch.Tensor]] = None,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            student_output: VLA features through GDINO head
                - 'pred_logits': [B, num_queries, text_len]
                - 'pred_boxes': [B, num_queries, 4]
            teacher_output: Original GDINO output (for KL distillation)
                - Same structure as student_output
            targets: List of dicts with GT annotations
                - 'boxes': [num_gt, 4] in cxcywh format, normalized
                - 'labels': [num_gt] or 'tokens_positive': list of [start, end]
                
        Returns:
            Dict of losses
        """
        losses = {}
        device = student_output["pred_boxes"].device
        
        # ============ 1. KL Loss: Soft distillation ============
        if teacher_output is not None and self.kl_weight > 0:
            loss_kl = self._compute_kl_loss(
                student_output["pred_logits"],
                teacher_output["pred_logits"].detach(),
            )
            losses["loss_kl"] = self.kl_weight * loss_kl
        else:
            losses["loss_kl"] = torch.tensor(0.0, device=device)
        
        # ============ 2 & 3. BBox + Class Loss: Hard supervision ============
        if targets is not None:
            # Match predictions with targets
            if self.use_hungarian:
                indices = self.matcher(student_output, targets)
            else:
                indices = self._simple_iou_matching(student_output, targets)
            
            # BBox losses
            loss_bbox, loss_giou = self._compute_bbox_loss(
                student_output["pred_boxes"],
                targets,
                indices,
            )
            losses["loss_bbox"] = self.bbox_weight * loss_bbox
            losses["loss_giou"] = self.giou_weight * loss_giou
            
            # Class loss
            loss_class = self._compute_class_loss(
                student_output["pred_logits"],
                targets,
                indices,
            )
            losses["loss_class"] = self.class_weight * loss_class
        else:
            losses["loss_bbox"] = torch.tensor(0.0, device=device)
            losses["loss_giou"] = torch.tensor(0.0, device=device)
            losses["loss_class"] = torch.tensor(0.0, device=device)
        
        # ============ Total Loss ============
        losses["loss_total"] = sum(v for k, v in losses.items() if k != "loss_total")
        
        return losses
    
    def _compute_kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss between student and teacher logits.
        
        Uses temperature scaling for softer probability distributions.
        """
        # Apply temperature
        student_scaled = student_logits / self.temperature
        teacher_scaled = teacher_logits / self.temperature
        
        # KL divergence
        loss = F.kl_div(
            F.log_softmax(student_scaled, dim=-1),
            F.softmax(teacher_scaled, dim=-1),
            reduction='batchmean',
        )
        
        # Temperature compensation
        loss = loss * (self.temperature ** 2)
        
        return loss
    
    def _compute_bbox_loss(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute L1 and GIoU losses for matched boxes.
        """
        device = pred_boxes.device
        
        # Gather matched predictions and targets
        src_boxes = []
        tgt_boxes = []
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            src_boxes.append(pred_boxes[b, pred_idx])
            tgt_boxes.append(targets[b]["boxes"][gt_idx])
        
        if len(src_boxes) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        src_boxes = torch.cat(src_boxes, dim=0)
        tgt_boxes = torch.cat(tgt_boxes, dim=0).to(device)
        
        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='mean')
        
        # GIoU loss
        src_boxes_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_boxes_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou = generalized_box_iou(src_boxes_xyxy, tgt_boxes_xyxy)
        loss_giou = 1 - torch.diag(giou).mean()
        
        return loss_bbox, loss_giou
    
    def _compute_class_loss(
        self,
        pred_logits: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute focal loss for classification.
        
        For matched queries: should have high confidence at corresponding text positions
        For unmatched queries: should have low confidence (background)
        """
        B, num_queries, text_len = pred_logits.shape
        device = pred_logits.device
        
        # Build target tensor
        target_classes = torch.zeros(B, num_queries, device=device)
        
        for b, (pred_idx, gt_idx) in enumerate(indices):
            target_classes[b, pred_idx] = 1.0  # Matched queries should be positive
        
        # Use max logit as confidence
        pred_conf = pred_logits.max(dim=-1).values  # [B, num_queries]
        
        # Focal loss
        loss = self._sigmoid_focal_loss(pred_conf, target_classes)
        
        return loss
    
    def _sigmoid_focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute sigmoid focal loss.
        
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)
        
        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()
    
    def _simple_iou_matching(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Simple IoU-based matching (alternative to Hungarian).
        Each GT is matched with the prediction having highest IoU.
        """
        indices = []
        
        for b in range(len(targets)):
            pred_boxes = outputs["pred_boxes"][b]  # [num_queries, 4]
            tgt_boxes = targets[b]["boxes"]  # [num_gt, 4]
            
            if len(tgt_boxes) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long, device=pred_boxes.device),
                    torch.tensor([], dtype=torch.long, device=pred_boxes.device),
                ))
                continue
            
            # Compute IoU
            iou = box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(tgt_boxes.to(pred_boxes.device)),
            )  # [num_queries, num_gt]
            
            # For each GT, find best matching prediction
            gt_idx = torch.arange(len(tgt_boxes), device=pred_boxes.device)
            pred_idx = iou.argmax(dim=0)  # [num_gt]
            
            indices.append((pred_idx, gt_idx))
        
        return indices



