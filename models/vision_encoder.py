"""
Vision encoder using DinoV2 for visual embedding extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class VisionEncoder(nn.Module):
    """
    Vision encoder that extracts embeddings from images using DinoV2.
    
    Features:
    - Uses DinoV2-large as the base encoder
    - Supports cached feature mode to skip DinoV2 computation
    - Applies transformer adapter and projection layers
    - Handles both raw images and pre-computed features
    """

    def __init__(self, embedding_dim: int = 256, use_cached_features: bool = False):
        """
        Initialize VisionEncoder.
        
        Args:
            embedding_dim: Final embedding dimension
            use_cached_features: If True, skip loading DinoV2 and expect cached features
        """
        super().__init__()
        self.use_cached_features = use_cached_features
        
        if not use_cached_features:
            # Load DinoV2 model
            self.model = AutoModel.from_pretrained('facebook/dinov2-large')
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            hidden_size = self.model.config.hidden_size
            print("VisionEncoder: Using DinoV2-large")
        else:
            # Skip loading DinoV2 when using cached features
            self.model = None
            hidden_size = 1024  # DinoV2-large hidden size
            print("VisionEncoder: Using cached features (DinoV2 not loaded)")
        
        # Transformer adapter for feature refinement
        self.adapter = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Projection to final embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.Linear(256, embedding_dim)
        )
    
    def forward_cached_features(self, cached_features: torch.Tensor) -> torch.Tensor:
        """
        Process cached DinoV2 features through adapter and projection.
        
        Args:
            cached_features: Pre-computed DinoV2 features (B, N+1, hidden_size)
            
        Returns:
            Normalized patch embeddings (B, N, embedding_dim)
        """
        cached_features = cached_features.detach() 
        adapted = self.adapter(cached_features)  # (B, N+1, hidden_size)
        patches = adapted[:, 1:]  # (B, N, hidden_size) - skip CLS token
        patch_embeds = self.projection(patches)  # (B, N, embedding_dim)
        patch_embeds = F.normalize(patch_embeds, dim=-1)
        return patch_embeds
        
    def forward(self, x: torch.Tensor = None, cached_features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass - processes either raw images or cached features.
        
        Args:
            x: Raw image tensors (B, 3, H, W) - required if not using cached features
            cached_features: Pre-computed DinoV2 features (B, N+1, hidden_size) - optional
            
        Returns:
            Normalized patch embeddings (B, N, embedding_dim)
            
        Raises:
            ValueError: If neither x nor cached_features are provided, or if misconfigured
        """
        if cached_features is not None:
            return self.forward_cached_features(cached_features)
            
        if self.use_cached_features:
            raise ValueError("VisionEncoder configured for cached features but none provided")
            
        if x is None:
            raise ValueError("Either x or cached_features must be provided")
            
        # Process raw images through DinoV2
        with torch.no_grad():
            features = self.model(x).last_hidden_state  # [B, N+1, hidden_size]

        # Apply adapter and projection
        adapted = self.adapter(features)  # [B, N+1, hidden_size]
        patches = adapted[:, 1:]  # [B, N, hidden_size] - skip CLS token
        patch_embeds = self.projection(patches)  # [B, N, embedding_dim]
        patch_embeds = F.normalize(patch_embeds, dim=-1)
        
        return patch_embeds