"""
Visualize Action Query Alignment: Extract features from VLA and perform grounding.

This script loads:
1. VLA model (from checkpoint)
2. ActionQueryAlignmentHead (from checkpoint)
3. GDINO Teacher (for comparison)

Then performs grounding on input images using both student and teacher features.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from align_train.models.action_query_alignment_head import ActionQueryAlignmentHead
from align_train.models.gdino_teacher import GDINOTeacher
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticProcessor
from prismatic.vla.constants import NUM_TOKENS


def load_models(
    vla_checkpoint_path: str,
    action_query_alignment_head_path: Optional[str] = None,
    gdino_config_path: str = "/tmp/Distill-VLA/visual_teacher/Open-GroundingDino/config/cfg_odvg.py",
    gdino_checkpoint_path: str = "/tmp/Distill-VLA/checkpoints/open_gdino_finetuned/checkpoint_best_regular.pth",
    device: str = "cuda:0",
) -> Tuple[OpenVLAForActionPrediction, Optional[ActionQueryAlignmentHead], GDINOTeacher]:
    """
    Load VLA, ActionQueryAlignmentHead, and GDINO Teacher models.
    
    Args:
        vla_checkpoint_path: Path to VLA checkpoint directory
        action_query_alignment_head_path: Path to ActionQueryAlignmentHead checkpoint (optional)
        gdino_config_path: Path to GDINO config file
        gdino_checkpoint_path: Path to GDINO checkpoint file
        device: Device to load models on
        
    Returns:
        vla: VLA model
        action_query_alignment_head: ActionQueryAlignmentHead (or None)
        gdino_teacher: GDINO Teacher model
    """
    device = torch.device(device)
    
    print("=" * 80)
    print("Loading Models")
    print("=" * 80)
    
    # ============ Load VLA Model ============
    print(f"\n[1/3] Loading VLA model from: {vla_checkpoint_path}")
    vla = OpenVLAForActionPrediction.from_pretrained(vla_checkpoint_path)
    vla = vla.to(device).eval()
    llm_dim = vla.llm_dim
    print(f"  ? VLA loaded successfully (llm_dim={llm_dim})")
    
    # ============ Load ActionQueryAlignmentHead ============
    action_query_alignment_head = None
    if action_query_alignment_head_path and os.path.exists(action_query_alignment_head_path):
        print(f"\n[2/3] Loading ActionQueryAlignmentHead from: {action_query_alignment_head_path}")
        action_query_alignment_head = ActionQueryAlignmentHead(
            input_dim=llm_dim,
            gdino_dim=256,
            dropout=0.1,
        ).to(torch.float32).to(device).eval()
        
        state_dict = torch.load(action_query_alignment_head_path, weights_only=True, map_location=device)
        action_query_alignment_head.load_state_dict(state_dict)
        print(f"  ? ActionQueryAlignmentHead loaded successfully")
    else:
        print(f"\n[2/3] ActionQueryAlignmentHead not found or not specified, skipping...")
    
    # ============ Load GDINO Teacher ============
    print(f"\n[3/3] Loading GDINO Teacher...")
    gdino_teacher = GDINOTeacher(
        config_path=gdino_config_path,
        checkpoint_path=gdino_checkpoint_path,
        device=device,
    )
    print(f"  ? GDINO Teacher loaded successfully")
    
    print("\n" + "=" * 80)
    print("All models loaded successfully!")
    print("=" * 80)
    
    return vla, action_query_alignment_head, gdino_teacher


def extract_action_queries(
    vla: OpenVLAForActionPrediction,
    processor: PrismaticProcessor,
    image: Image.Image,
    task_description: str = "pick up the red block",
    device: str = "cuda:0",
) -> torch.Tensor:
    """
    Extract action queries from VLA model.
    
    Args:
        vla: VLA model
        processor: PrismaticProcessor
        image: Input image (PIL Image)
        task_description: Task description text
        device: Device to run on
        
    Returns:
        action_queries: Action query hidden states [B, 64, 896]
    """
    device = torch.device(device)
    
    # Prepare inputs
    inputs = processor(
        images=image,
        text=task_description,
        return_tensors="pt",
    )
    
    # Move to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = vla(**inputs)
    
    # Extract action queries from LLM output
    # VLA output structure: hidden_states[-1] has shape [B, seq_len, 896]
    last_hidden = outputs.hidden_states[-1]  # [B, seq_len, 896]
    
    # Sequence structure: [CLS] + [vision_patches] + [text_tokens] + [action_tokens]
    # We need to extract action tokens (last 64 tokens)
    action_queries = last_hidden[:, -NUM_TOKENS:, :]  # [B, 64, 896]
    
    return action_queries


def perform_grounding_student(
    action_queries: torch.Tensor,
    action_query_alignment_head: ActionQueryAlignmentHead,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform grounding using student features (via ActionQueryAlignmentHead).
    
    Args:
        action_queries: Action query hidden states [B, 64, 896]
        action_query_alignment_head: ActionQueryAlignmentHead
        
    Returns:
        student_hs: Student hidden states [B, 900, 256]
        student_ref: Student reference points (bboxes) [B, 900, 4]
    """
    with torch.no_grad():
        student_hs, student_ref = action_query_alignment_head(action_queries)
    
    return student_hs, student_ref


def perform_grounding_teacher(
    gdino_teacher: GDINOTeacher,
    image: Image.Image,
    task_description: str = "pick up the red block",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform grounding using teacher (GDINO).
    
    Args:
        gdino_teacher: GDINO Teacher model
        image: Input image (PIL Image)
        task_description: Task description text
        
    Returns:
        teacher_hs: Teacher hidden states [B, 900, 256]
        teacher_ref: Teacher reference points (bboxes) [B, 900, 4]
    """
    with torch.no_grad():
        outputs = gdino_teacher.forward_with_intermediates(
            images=image,
            captions=[task_description],
        )
    
    teacher_hs = outputs["hs"]
    teacher_ref = outputs["ref"]
    
    return teacher_hs, teacher_ref


def visualize_grounding_results(
    image: Image.Image,
    student_ref: torch.Tensor,
    teacher_ref: torch.Tensor,
    task_description: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize grounding results from student and teacher.
    
    Args:
        image: Original image
        student_ref: Student reference points [B, 900, 4]
        teacher_ref: Teacher reference points [B, 900, 4]
        task_description: Task description
        save_path: Path to save visualization (optional)
    """
    # Convert to numpy
    image_np = np.array(image)
    height, width = image_np.shape[:2]
    
    # Convert reference points to bboxes [x, y, w, h] in pixel coordinates
    student_ref_np = student_ref[0].cpu().numpy()  # [900, 4]
    teacher_ref_np = teacher_ref[0].cpu().numpy()  # [900, 4]
    
    # Convert from normalized [cx, cy, w, h] to pixel coordinates
    student_bboxes = student_ref_np.copy()
    student_bboxes[:, 0] *= width  # cx
    student_bboxes[:, 1] *= height  # cy
    student_bboxes[:, 2] *= width  # w
    student_bboxes[:, 3] *= height  # h
    
    teacher_bboxes = teacher_ref_np.copy()
    teacher_bboxes[:, 0] *= width  # cx
    teacher_bboxes[:, 1] *= height  # cy
    teacher_bboxes[:, 2] *= width  # w
    teacher_bboxes[:, 3] *= height  # h
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot 2: Student bboxes (from ActionQueryAlignmentHead)
    axes[1].imshow(image_np)
    axes[1].set_title(f"Student (ActionQueryAlignmentHead)\n{task_description}", fontsize=14, fontweight='bold')
    
    # Draw top-20 student bboxes with highest confidence
    for i in range(min(20, len(student_bboxes))):
        cx, cy, w, h = student_bboxes[i]
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    axes[1].imshow(image_np)
    axes[1].axis('off')
    
    # Plot 3: Teacher bboxes (from GDINO)
    axes[2].imshow(image_np)
    axes[2].set_title(f"Teacher (GDINO)\n{task_description}", fontsize=14, fontweight='bold')
    
    # Draw top-20 teacher bboxes with highest confidence
    for i in range(min(20, len(teacher_bboxes))):
        cx, cy, w, h = teacher_bboxes[i]
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    axes[2].imshow(image_np)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ? Visualization saved to: {save_path}")
    
    plt.show()


def compute_feature_similarity(
    student_hs: torch.Tensor,
    teacher_hs: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute similarity metrics between student and teacher features.
    
    Args:
        student_hs: Student hidden states [B, 900, 256]
        teacher_hs: Teacher hidden states [B, 900, 256]
        
    Returns:
        Dict with similarity metrics
    """
    # Flatten features
    student_flat = student_hs.flatten()  # [B*900*256]
    teacher_flat = teacher_hs.flatten()  # [B*900*256]
    
    # Cosine similarity
    student_norm = F.normalize(student_flat, p=2, dim=0)
    teacher_norm = F.normalize(teacher_flat, p=2, dim=0)
    cosine_sim = torch.dot(student_norm, teacher_norm).item()
    
    # MSE
    mse = F.mse_loss(student_hs, teacher_hs).item()
    
    # L1
    l1 = F.l1_loss(student_hs, teacher_hs).item()
    
    return {
        "cosine_similarity": cosine_sim,
        "mse": mse,
        "l1": l1,
    }


def main():
    """Main function to run action query alignment visualization."""
    
    # ============ Configuration ============
    vla_checkpoint_path = "outputs/VLA-Adapter--libero_spatial_no_noops--20250210_120000/"
    action_query_alignment_head_path = "outputs/VLA-Adapter--libero_spatial_no_noops--20250210_120000/action_query_alignment_head--120000_checkpoint.pt"
    
    # Image path (replace with your image)
    image_path = "data/libero/libero_spatial_no_noops/1.0.0/example_image.png"  # Replace with actual image
    
    # Task description
    task_description = "pick up the red block"
    
    device = "cuda:0"
    
    # ============ Load Models ============
    vla, action_query_alignment_head, gdino_teacher = load_models(
        vla_checkpoint_path=vla_checkpoint_path,
        action_query_alignment_head_path=action_query_alignment_head_path,
        device=device,
    )
    
    # Load processor
    processor = PrismaticProcessor.from_pretrained(vla_checkpoint_path)
    
    # ============ Load Image ============
    print(f"\nLoading image from: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"  ? Image loaded: {image.size}")
    
    # ============ Extract Action Queries from VLA ============
    print(f"\nExtracting action queries from VLA...")
    action_queries = extract_action_queries(
        vla=vla,
        processor=processor,
        image=image,
        task_description=task_description,
        device=device,
    )
    print(f"  ? Action queries extracted: {action_queries.shape}")
    
    # ============ Perform Grounding with Student ============
    if action_query_alignment_head is not None:
        print(f"\nPerforming grounding with student (ActionQueryAlignmentHead)...")
        student_hs, student_ref = perform_grounding_student(
            action_queries=action_queries,
            action_query_alignment_head=action_query_alignment_head,
        )
        print(f"  ? Student features: hs={student_hs.shape}, ref={student_ref.shape}")
    else:
        print("\nSkipping student grounding (ActionQueryAlignmentHead not loaded)")
        student_hs, student_ref = None, None
    
    # ============ Perform Grounding with Teacher ============
    print(f"\nPerforming grounding with teacher (GDINO)...")
    teacher_hs, teacher_ref = perform_grounding_teacher(
        gdino_teacher=gdino_teacher,
        image=image,
        task_description=task_description,
    )
    print(f"  ? Teacher features: hs={teacher_hs.shape}, ref={teacher_ref.shape}")
    
    # ============ Compute Feature Similarity ============
    if student_hs is not None:
        print(f"\nComputing feature similarity...")
        similarity_metrics = compute_feature_similarity(
            student_hs=student_hs,
            teacher_hs=teacher_hs,
        )
        print(f"  ? Similarity metrics:")
        for key, value in similarity_metrics.items():
            print(f"      {key}: {value:.4f}")
    
    # ============ Visualize Results ============
    if student_ref is not None:
        print(f"\nVisualizing grounding results...")
        visualize_grounding_results(
            image=image,
            student_ref=student_ref,
            teacher_ref=teacher_ref,
            task_description=task_description,
            save_path="action_query_alignment_visualization.png",
        )
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
