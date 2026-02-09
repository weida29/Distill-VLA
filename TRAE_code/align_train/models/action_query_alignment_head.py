"""
action_query_alignment_head.py

Action Query Alignment Head for mapping VLA action queries to GDINO space.
"""

import torch
import torch.nn as nn


class ActionQueryAlignmentHead(nn.Module):
    """
    Alignment head that maps VLA action queries to GDINO space.
    
    Architecture:
    - Input: Action query hidden states [B, 64, 896]
    - Output: Student hs [B, 64, 256] and student ref [B, 64, 4]
    """
    
    def __init__(
        self,
        input_dim: int = 896,
        gdino_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.gdino_dim = gdino_dim
        
        # HS Projection: LLM hidden -> GDINO decoder space
        self.hs_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, gdino_dim),
            nn.LayerNorm(gdino_dim),
        )
        
        # Reference Point Projection: LLM hidden -> bbox coords
        self.ref_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, 4),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.hs_proj, self.ref_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        action_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            action_queries: Action query hidden states [B, 64, 896]
        
        Returns:
            student_hs: Projected hidden states [B, 64, 256]
            student_ref: Predicted reference points [B, 64, 4] (sigmoid applied)
        """
        student_hs = self.hs_proj(action_queries)
        student_ref = self.ref_proj(action_queries).sigmoid()
        
        return student_hs, student_ref
