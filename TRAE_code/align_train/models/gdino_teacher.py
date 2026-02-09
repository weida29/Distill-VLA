"""
gdino_teacher.py

GDINO Teacher loader for VLA alignment training.
Loads pretrained/finetuned GDINO as a frozen teacher model.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn

# Add Open-GroundingDino to path
# gdino_teacher.py is at TRAE_code/align_train/models/, so we need 4 levels up to reach project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
GDINO_PATH = PROJECT_ROOT / "visual_teacher" / "Open-GroundingDino"
sys.path.insert(0, str(GDINO_PATH))


class GDINOTeacher(nn.Module):
    """
    Wrapper for frozen Grounding DINO teacher model.
    
    Provides:
    - Forward pass to get teacher_hs and teacher_ref
    - Access to frozen bbox_embed and class_embed heads for student prediction
    - Text encoding via BERT
    """
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        freeze: bool = True,
    ):
        super().__init__()
        
        self.device = device
        self.freeze = freeze
        
        # Load GDINO model
        self.model = self._load_model(config_path, checkpoint_path)
        
        if freeze:
            self._freeze_model()
        
        # Store references to key components
        self.tokenizer = self.model.tokenizer
        self.bert = self.model.bert
        self.feat_map = self.model.feat_map
        self.bbox_embed = self.model.bbox_embed
        self.class_embed = self.model.class_embed
        self.num_queries = self.model.num_queries
        self.max_text_len = self.model.max_text_len
    
    def _load_model(self, config_path: str, checkpoint_path: str) -> nn.Module:
        """Load GDINO model from config and checkpoint."""
        from util.slconfig import SLConfig
        from models.registry import MODULE_BUILD_FUNCS
        
        # Load config - use path directly (can be absolute or relative like ../visual_teacher/...)
        args = SLConfig.fromfile(config_path)
        args.device = self.device
        
        # Set required attributes for PostProcess (even though we won't use it)
        args.use_coco_eval = False
        args.label_list = ["object"]  # Dummy label list for PostProcess
        
        # Build model using registry
        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, criterion, postprocessors = build_func(args)
        
        # Load checkpoint - use path directly (can be absolute or relative)
        if os.path.exists(checkpoint_path):
            print(f"Loading GDINO teacher from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            state_dict = checkpoint.get('model', checkpoint)
            # Remove 'module.' prefix if present (from DDP training)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            raise FileNotFoundError(f"GDINO checkpoint not found: {checkpoint_path}")
        
        model = model.to(self.device)
        print(f"GDINO Teacher loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
        
        return model
    
    def _freeze_model(self):
        """Freeze all model parameters."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print("GDINO Teacher frozen")
    
    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        captions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GDINO to get teacher outputs.
        
        Args:
            images: Input images [B, 3, H, W]
            captions: List of text captions
            
        Returns:
            Dict containing:
            - teacher_hs: Decoder hidden states [B, num_queries, 256]
            - teacher_ref: Reference points [B, num_queries, 4]
            - text_dict: Text encoding for class_embed
            - pred_boxes: Teacher's predicted boxes
            - pred_logits: Teacher's predicted logits
        """
        # Get full model output
        outputs = self.model(images, captions=captions)
        
        # The model's transformer returns hs and references
        # We need to get intermediate outputs
        # Run again to extract hs/ref (or modify model to return them)
        
        return self._forward_with_intermediates(images, captions)
    
    def _forward_with_intermediates(
        self,
        images: torch.Tensor,
        captions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass that returns intermediate decoder outputs.
        """
        from groundingdino.util.misc import NestedTensor
        from groundingdino.util.utils import get_phrases_from_posmap
        import torch.nn.functional as F
        
        device = images.device
        
        # ============ Text Encoding ============
        tokenized = self.tokenizer(
            captions, 
            padding="longest", 
            return_tensors="pt"
        ).to(device)
        
        # Handle max text length
        if tokenized.input_ids.shape[1] > self.max_text_len:
            tokenized["input_ids"] = tokenized["input_ids"][:, :self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :self.max_text_len]
        
        bert_output = self.bert(**tokenized)
        encoded_text = self.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()
        
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, :self.max_text_len, :]
            text_token_mask = text_token_mask[:, :self.max_text_len]
        
        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
        }
        
        # ============ Image Encoding ============
        # Convert to NestedTensor if needed
        if isinstance(images, torch.Tensor):
            from groundingdino.util.misc import nested_tensor_from_tensor_list
            samples = nested_tensor_from_tensor_list(images)
        else:
            samples = images
        
        # Backbone features
        features, poss = self.model.backbone(samples)
        
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.model.input_proj[l](src))
            masks.append(mask)
        
        # Handle additional feature levels
        if self.model.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.model.num_feature_levels):
                if l == _len_srcs:
                    src = self.model.input_proj[l](features[-1].tensors)
                else:
                    src = self.model.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.model.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        
        # ============ Transformer Forward ============
        # Add position_ids and text_self_attention_masks to text_dict
        text_dict["position_ids"] = torch.arange(
            encoded_text.shape[1], device=device
        ).unsqueeze(0).expand(images.shape[0], -1)
        
        text_dict["text_self_attention_masks"] = text_token_mask.unsqueeze(1).expand(
            -1, encoded_text.shape[1], -1
        )
        
        hs, references, hs_enc, ref_enc, init_box_proposal = self.model.transformer(
            srcs, masks, None, poss, None, None, text_dict
        )
        
        # ============ Get Final Outputs ============
        # hs: [num_decoder_layers, B, num_queries, hidden_dim]
        # references: [num_decoder_layers+1, B, num_queries, 4]
        
        teacher_hs = hs[-1]  # Last decoder layer: [B, num_queries, 256]
        teacher_ref = references[-1]  # Last reference points: [B, num_queries, 4]
        
        # Get predictions using heads
        pred_boxes = self.bbox_embed[-1](teacher_hs)
        pred_boxes = (pred_boxes + self._inverse_sigmoid(teacher_ref)).sigmoid()
        pred_logits = self.class_embed[-1](teacher_hs, text_dict)
        
        return {
            "teacher_hs": teacher_hs,
            "teacher_ref": teacher_ref,
            "text_dict": text_dict,
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "hs_all_layers": hs,
            "references_all_layers": references,
        }
    
    def _inverse_sigmoid(self, x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        """Inverse sigmoid function."""
        x = x.clamp(min=0, max=1)
        x1 = x.clamp(min=eps)
        x2 = (1 - x).clamp(min=eps)
        return torch.log(x1 / x2)
    
    def predict_with_student_features(
        self,
        student_hs: torch.Tensor,
        student_ref: torch.Tensor,
        text_dict: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use frozen GDINO heads to predict from student features.
        
        Args:
            student_hs: Student hidden states [B, num_queries, 256]
            student_ref: Student reference points [B, num_queries, 4]
            text_dict: Text encoding from teacher
            
        Returns:
            pred_boxes: Predicted boxes [B, num_queries, 4]
            pred_logits: Predicted logits [B, num_queries, text_len]
        """
        # Apply bbox_embed
        delta_boxes = self.bbox_embed[-1](student_hs)
        pred_boxes = (delta_boxes + self._inverse_sigmoid(student_ref)).sigmoid()
        
        # Apply class_embed
        pred_logits = self.class_embed[-1](student_hs, text_dict)
        
        return pred_boxes, pred_logits
    
    def encode_text(self, captions: List[str], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Encode text captions using BERT.
        
        Args:
            captions: List of text captions
            device: Device
            
        Returns:
            text_dict: Dict with encoded_text, text_token_mask, etc.
        """
        tokenized = self.tokenizer(
            captions, 
            padding="longest", 
            return_tensors="pt"
        ).to(device)
        
        if tokenized.input_ids.shape[1] > self.max_text_len:
            tokenized["input_ids"] = tokenized["input_ids"][:, :self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, :self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :self.max_text_len]
        
        with torch.no_grad():
            bert_output = self.bert(**tokenized)
        
        encoded_text = self.feat_map(bert_output["last_hidden_state"])
        text_token_mask = tokenized.attention_mask.bool()
        
        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, :self.max_text_len, :]
            text_token_mask = text_token_mask[:, :self.max_text_len]
        
        text_dict = {
            "encoded_text": encoded_text,
            "text_token_mask": text_token_mask,
            "position_ids": torch.arange(
                encoded_text.shape[1], device=device
            ).unsqueeze(0).expand(len(captions), -1),
        }
        
        return text_dict


def load_gdino_teacher(
    config_path: str = "visual_teacher/Open-GroundingDino/config/cfg_odvg.py",
    checkpoint_path: str = "checkpoints/open_gdino_finetuned/checkpoint_best_regular.pth",
    device: str = "cuda",
) -> GDINOTeacher:
    """
    Convenience function to load GDINO teacher.
    
    Args:
        config_path: Path to GDINO config (relative to project root)
        checkpoint_path: Path to checkpoint (relative to project root)
        device: Device to load model on
        
    Returns:
        GDINOTeacher instance
    """
    return GDINOTeacher(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        freeze=True,
    )

