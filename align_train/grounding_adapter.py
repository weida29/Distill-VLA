"""
grounding_adapter.py

Feature adapter that converts VLA backbone features to Grounding DINO decoder input format.
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundingFeatureAdapter(nn.Module):
    """
    Adapts VLA vision backbone features to Grounding DINO decoder input format.
    
    VLA backbone (DINOv2 + SigLIP) outputs: [B, num_patches, vla_dim]
    GDINO decoder expects: srcs (multi-scale), masks, pos_embeds
    
    This adapter:
    1. Projects VLA features from vla_dim to gdino_dim
    2. Reshapes to spatial format (H, W)
    3. Generates position embeddings
    4. Optionally creates multi-scale features
    """
    
    def __init__(
        self,
        vla_dim: int = 1152,        # DINOv2(768) + SigLIP(384) = 1152
        gdino_dim: int = 256,        # Grounding DINO hidden_dim
        num_patches: int = 256,      # 16x16 patches for 224px image with 14px patch
        num_feature_levels: int = 1, # Number of feature levels (1 = single scale)
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vla_dim = vla_dim
        self.gdino_dim = gdino_dim
        self.num_patches = num_patches
        self.num_feature_levels = num_feature_levels
        
        # Calculate spatial dimensions
        self.spatial_size = int(math.sqrt(num_patches))
        assert self.spatial_size ** 2 == num_patches, \
            f"num_patches ({num_patches}) must be a perfect square"
        
        # Feature projection: vla_dim -> gdino_dim
        self.proj = nn.Sequential(
            nn.Linear(vla_dim, gdino_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gdino_dim * 2, gdino_dim),
            nn.LayerNorm(gdino_dim),
        )
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, gdino_dim) * 0.02
        )
        
        # Level embedding for multi-scale (if needed)
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(
                torch.randn(num_feature_levels, gdino_dim)
            )
            # Additional projection for different scales
            self.scale_projs = nn.ModuleList([
                nn.Conv2d(gdino_dim, gdino_dim, kernel_size=3, stride=2, padding=1)
                for _ in range(num_feature_levels - 1)
            ])
        
        # Input projection to match GDINO format (conv instead of linear for spatial features)
        self.input_proj = nn.Sequential(
            nn.Conv2d(gdino_dim, gdino_dim, kernel_size=1),
            nn.GroupNorm(32, gdino_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        vla_features: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Convert VLA features to GDINO decoder input format.
        
        Args:
            vla_features: [B, num_patches, vla_dim] from VLA vision backbone
            
        Returns:
            srcs: List of feature maps [B, C, H, W]
            masks: List of masks [B, H, W] (False = valid)
            pos_embeds: List of position embeddings [B, C, H, W]
        """
        B = vla_features.shape[0]
        device = vla_features.device
        dtype = vla_features.dtype
        
        # 1. Project features: [B, num_patches, vla_dim] -> [B, num_patches, gdino_dim]
        projected = self.proj(vla_features)
        
        # 2. Reshape to spatial format: [B, num_patches, C] -> [B, C, H, W]
        H = W = self.spatial_size
        spatial_features = projected.transpose(1, 2).reshape(B, self.gdino_dim, H, W)
        
        # 3. Apply input projection
        spatial_features = self.input_proj(spatial_features)
        
        # 4. Generate position embeddings
        pos = self.pos_embed.expand(B, -1, -1)  # [B, num_patches, C]
        pos = pos.transpose(1, 2).reshape(B, self.gdino_dim, H, W)
        
        # 5. Create mask (all False = all valid)
        mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
        
        # 6. Build output lists
        srcs = [spatial_features]
        masks = [mask]
        pos_embeds = [pos]
        
        # 7. Create multi-scale features if needed
        if self.num_feature_levels > 1:
            current_feat = spatial_features
            for i, scale_proj in enumerate(self.scale_projs):
                # Downsample features
                current_feat = scale_proj(current_feat)
                
                # Update spatial dimensions
                new_H, new_W = current_feat.shape[-2:]
                
                # Create mask for this scale
                scale_mask = torch.zeros(B, new_H, new_W, dtype=torch.bool, device=device)
                
                # Create position embedding for this scale
                scale_pos = F.interpolate(
                    pos, size=(new_H, new_W), mode='bilinear', align_corners=False
                )
                
                # Add level embedding
                level_embed = self.level_embed[i + 1].view(1, -1, 1, 1)
                current_feat = current_feat + level_embed
                
                srcs.append(current_feat)
                masks.append(scale_mask)
                pos_embeds.append(scale_pos)
        
        return srcs, masks, pos_embeds
    
    def get_reference_points(
        self,
        spatial_shapes: torch.Tensor,
        valid_ratios: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate reference points for deformable attention.
        
        This is needed by GDINO's deformable attention mechanism.
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing='ij'
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class GroundingFeatureAdapterV2(nn.Module):
    """
    Alternative adapter with cross-attention for better feature alignment.
    
    Uses learnable queries to attend to VLA features, producing
    a fixed number of output tokens suitable for GDINO decoder.
    """
    
    def __init__(
        self,
        vla_dim: int = 1152,
        gdino_dim: int = 256,
        num_queries: int = 900,     # Match GDINO num_queries
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.vla_dim = vla_dim
        self.gdino_dim = gdino_dim
        self.num_queries = num_queries
        
        # Input projection
        self.input_proj = nn.Linear(vla_dim, gdino_dim)
        
        # Learnable queries (like DETR object queries)
        self.query_embed = nn.Embedding(num_queries, gdino_dim)
        
        # Cross-attention layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=gdino_dim,
            nhead=num_heads,
            dim_feedforward=gdino_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(gdino_dim),
            nn.Linear(gdino_dim, gdino_dim),
        )
        
    def forward(self, vla_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vla_features: [B, num_patches, vla_dim]
            
        Returns:
            query_features: [B, num_queries, gdino_dim]
        """
        B = vla_features.shape[0]
        
        # Project input features
        memory = self.input_proj(vla_features)  # [B, num_patches, gdino_dim]
        
        # Get query embeddings
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-attention: queries attend to VLA features
        output = self.decoder(queries, memory)  # [B, num_queries, gdino_dim]
        
        # Output projection
        output = self.output_proj(output)
        
        return output



