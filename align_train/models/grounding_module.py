"""
grounding_module.py

Grounding Module for VLA-GDINO alignment.
Adds learnable grounding queries to VLA's LLM and projects outputs to GDINO space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GroundingModule(nn.Module):
    """
    Grounding Module that adds learnable grounding queries to VLA.
    
    Architecture:
    1. Learnable grounding_queries [num_queries, llm_dim] - appended to LLM input
    2. After LLM forward, extract hidden states at grounding query positions
    3. Project to GDINO decoder space (hs) and reference points (ref)
    
    The grounding queries can attend to:
    - Visual patches (main + wrist view)
    - Text tokens (prompt/instruction)
    - Other grounding queries
    
    This enables the model to learn object grounding through the VLM's reasoning.
    """
    
    def __init__(
        self,
        num_queries: int = 900,
        llm_dim: int = 896,           # Qwen2.5-0.5B hidden dim
        gdino_dim: int = 256,         # GDINO decoder hidden dim
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.llm_dim = llm_dim
        self.gdino_dim = gdino_dim
        
        # ============ Learnable Grounding Queries ============
        # These are appended to LLM input sequence
        self.grounding_queries = nn.Embedding(num_queries, llm_dim)
        
        # ============ HS Projection: LLM hidden → GDINO decoder space ============
        self.hs_proj = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim // 2, gdino_dim),
            nn.LayerNorm(gdino_dim),
        )
        
        # ============ Reference Point Projection: LLM hidden → bbox coords ============
        # Predicts (cx, cy, w, h) in [0, 1]
        self.refpoint_proj = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim // 4, 4),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize grounding queries with small random values
        nn.init.normal_(self.grounding_queries.weight, mean=0, std=0.02)
        
        # Initialize projection layers
        for module in [self.hs_proj, self.refpoint_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0)
        
        # Initialize refpoint bias to reasonable default (center of image)
        # This helps training converge faster
        if hasattr(self.refpoint_proj[-1], 'bias'):
            # Initialize to predict center (0.5, 0.5) with small size (0.1, 0.1)
            self.refpoint_proj[-1].bias.data = torch.tensor([0.0, 0.0, -2.0, -2.0])
    
    def get_query_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get grounding query embeddings for appending to LLM input.
        
        Args:
            batch_size: Batch size
            device: Device to place tensors on
            
        Returns:
            query_embeds: [B, num_queries, llm_dim]
        """
        # Expand queries for batch
        query_embeds = self.grounding_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return query_embeds.to(device)
    
    def forward(
        self,
        grounding_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project LLM hidden states at grounding query positions to GDINO space.
        
        Args:
            grounding_hidden_states: Hidden states at grounding query positions
                                    [B, num_queries, llm_dim]
        
        Returns:
            student_hs: Projected hidden states [B, num_queries, gdino_dim]
            student_ref: Predicted reference points [B, num_queries, 4] (sigmoid applied)
        """
        # Project to GDINO decoder hidden dimension
        student_hs = self.hs_proj(grounding_hidden_states)  # [B, num_queries, gdino_dim]
        
        # Project to reference points (apply sigmoid for [0, 1] range)
        student_ref = self.refpoint_proj(grounding_hidden_states).sigmoid()  # [B, num_queries, 4]
        
        return student_hs, student_ref
    
    def get_attention_mask_extension(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bool,
    ) -> torch.Tensor:
        """
        Get attention mask for grounding queries (all True = attend to all).
        
        Args:
            batch_size: Batch size
            device: Device
            dtype: Data type for mask
            
        Returns:
            mask: [B, num_queries] attention mask (True = attend)
        """
        return torch.ones(batch_size, self.num_queries, dtype=dtype, device=device)


class GroundingModuleV2(nn.Module):
    """
    Alternative version with multi-layer projection for better feature alignment.
    Uses a small transformer decoder for cross-attention to visual features.
    """
    
    def __init__(
        self,
        num_queries: int = 900,
        llm_dim: int = 896,
        gdino_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_queries = num_queries
        self.llm_dim = llm_dim
        self.gdino_dim = gdino_dim
        
        # Learnable grounding queries
        self.grounding_queries = nn.Embedding(num_queries, llm_dim)
        
        # Optional: Additional cross-attention refinement
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=llm_dim,
            nhead=num_heads,
            dim_feedforward=llm_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.refine_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Projections
        self.hs_proj = nn.Sequential(
            nn.Linear(llm_dim, gdino_dim),
            nn.LayerNorm(gdino_dim),
        )
        
        self.refpoint_proj = nn.Sequential(
            nn.Linear(llm_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.grounding_queries.weight, mean=0, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def get_query_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.grounding_queries.weight.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    
    def forward(
        self,
        grounding_hidden_states: torch.Tensor,
        visual_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward with optional cross-attention refinement.
        
        Args:
            grounding_hidden_states: [B, num_queries, llm_dim]
            visual_memory: Optional visual features for cross-attention [B, num_patches, llm_dim]
        """
        # Optional refinement with cross-attention to visual features
        if visual_memory is not None:
            grounding_hidden_states = self.refine_decoder(
                grounding_hidden_states,
                visual_memory,
            )
        
        student_hs = self.hs_proj(grounding_hidden_states)
        student_ref = self.refpoint_proj(grounding_hidden_states).sigmoid()
        
        return student_hs, student_ref

