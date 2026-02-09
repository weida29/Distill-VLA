"""
action_query_alignment_head.py

Action Query Alignment Head for mapping VLA action queries to GDINO space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionQueryAlignmentHead(nn.Module):
    """
    Alignment head that maps VLA action queries to GDINO space.
    
    Architecture:
    - Input: Action query hidden states [B, 64, 896]
    - Output: Student hs [B, 900, 256] and student ref [B, 900, 4]
    
    Expands 64 action queries to 900 to match GDINO's object queries.
    """
    
    def __init__(
        self,
        input_dim: int = 896,
        gdino_dim: int = 256,
        dropout: float = 0.1,
        num_action_queries: int = 64,
        num_gdino_queries: int = 900,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.gdino_dim = gdino_dim
        self.num_action_queries = num_action_queries
        self.num_gdino_queries = num_gdino_queries
        
        # HS Projection: LLM hidden -> GDINO decoder space
        self.hs_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, gdino_dim),
            nn.LayerNorm(gdino_dim),
        )
        
        # Expansion layer: 64 -> 900 queries
        self.query_expansion = nn.Linear(
            num_action_queries,
            num_gdino_queries,
            bias=False,
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
        for module in [self.hs_proj, self.query_expansion, self.ref_proj]:
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
            student_hs: Projected hidden states [B, 900, 256]
            student_ref: Predicted reference points [B, 900, 4] (sigmoid applied)
        """
        # Project action queries to GDINO space [B, 64, 256]
        action_hs_64 = self.hs_proj(action_queries)
        action_ref_64 = self.ref_proj(action_queries).sigmoid()
        
        # Expand from 64 to 900 queries
        # Use learnable expansion weights
        expansion_weights = self.query_expansion.weight  # [900, 64]
        # Normalize to sum to 1 for each source query
        expansion_weights = F.softmax(expansion_weights, dim=1)  # [900, 64]
        
        # Expand: [B, 900, 256] = [B, 900, 64] @ [64, 256]
        student_hs = torch.bmm(expansion_weights.unsqueeze(0), action_hs_64)
        
        # Expand reference points: [B, 900, 4] = [B, 900, 64] @ [64, 4]
        student_ref = torch.bmm(expansion_weights.unsqueeze(0), action_ref_64)
        
        return student_hs, student_ref
