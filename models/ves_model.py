"""
VeS (Vision-Speech) model for cross-modal representation learning.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Union

from .audio_encoder import AudioEmbedder
from .vision_encoder import VisionEncoder
from .losses import VeSLossComputer


class VeS(nn.Module):
    """
    Vision-Speech (VeS) model for cross-modal alignment.
    
    The model learns joint representations of audio and visual content
    through contrastive learning with various loss formulations.
    """
    
    def __init__(self, 
                 loss_type: str = "dense",
                 use_cached_visual_features: bool = False,
                 embedding_dim: int = 256,
                 hubert_name: str = "ntu-spml/distilhubert",
                 device: str = "auto"):
        """
        Initialize VeS model.
        
        Args:
            loss_type: Type of loss to use ("dense", "global", "dense_global")
            use_cached_visual_features: Whether to use pre-computed visual features
            embedding_dim: Dimension of the joint embedding space
            hubert_name: HuBERT model name for audio encoding
            device: Device to place models on
        """
        super().__init__()

        # Initialize encoders
        self.visual_embedder = VisionEncoder(
            embedding_dim=embedding_dim,
            use_cached_features=use_cached_visual_features
        )
        self.audio_embedder = AudioEmbedder(
            embedding_dim=embedding_dim,
            hubert_name=hubert_name,
            device=device
        )
        
        # Model configuration
        self.use_cached_visual_features = use_cached_visual_features
        self.loss_type = loss_type
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        
        # Loss computer
        self.loss_computer = VeSLossComputer(loss_type=loss_type, tv_weight=0.1)
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"VeS model initialized with loss_type='{loss_type}', embedding_dim={embedding_dim}")

    def forward(self, 
                audio_input: torch.Tensor,
                images: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                cached_visual_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of VeS model.
        
        Args:
            audio_input: Raw audio waveform (B, T) at 16kHz
            images: Preprocessed image tensors (B, 3, H, W) - optional if using cached features
            attention_mask: Audio attention mask (B, T)
            cached_visual_features: Pre-computed DinoV2 features (B, N+1, 768) - optional
            
        Returns:
            Dictionary containing:
                - loss: Total training loss
                - clip_sims: Clip-level similarity matrix (B, B)
                - audio_feats: Audio feature embeddings (B, Na, D)
                - visual_feats: Visual feature embeddings (B, Nv, D)
                - audio_attention_mask: Processed audio attention mask
                - token_sims: Token-level similarities (B, B, Na, Nv)
                - l_nonneg: Non-negativity regularization loss
                - l_tv: Temporal variation regularization loss
                - dense_loss: Dense contrastive loss component
                - global_loss: Global contrastive loss component
        """
        # Extract audio features
        audio_feats, audio_attention_mask = self.audio_embedder(audio_input, attention_mask)
        
        # Extract visual features
        if cached_visual_features is not None:
            visual_feats = self.visual_embedder(cached_features=cached_visual_features)
        elif images is not None:
            visual_feats = self.visual_embedder(images)
        else:
            raise ValueError("Either images or cached_visual_features must be provided")
        
        # Compute similarities
        clip_sims, token_sims = self.loss_computer.compute_all_similarities_tv(
            audio_feats, 
            visual_feats, 
            audio_attention_mask,
            self.logit_scale
        )

        # Compute loss based on loss type
        if self.loss_type == "dense":
            loss, l_nonneg, l_tv, dense_loss, global_loss = self.loss_computer.compute_contrastive_loss(
                clip_sims, 
                token_sims, 
                audio_attention_mask,
                audio_feats=None,     
                visual_feats=None,    
                logit_scale=None,
                global_weight=0.0             
            )
        elif self.loss_type == "dense_global":
            loss, l_nonneg, l_tv, dense_loss, global_loss = self.loss_computer.compute_contrastive_loss(
                clip_sims, 
                token_sims, 
                audio_attention_mask,
                audio_feats=audio_feats,     
                visual_feats=visual_feats,
                logit_scale=self.logit_scale,
                global_weight=0.30             
            )
        elif self.loss_type == "global":
            loss, l_nonneg, l_tv, dense_loss, global_loss = self.loss_computer.compute_contrastive_loss(
                clip_sims, 
                token_sims, 
                audio_attention_mask,
                audio_feats=audio_feats,     
                visual_feats=visual_feats,
                logit_scale=self.logit_scale,
                global_weight=1.0             
            )
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        
        return {
            'loss': loss,
            'clip_sims': clip_sims.detach(),
            'audio_feats': audio_feats.detach(),
            'visual_feats': visual_feats.detach(),
            'audio_attention_mask': audio_attention_mask.detach(),
            'token_sims': token_sims.detach(),
            'l_nonneg': l_nonneg.detach(),
            'l_tv': l_tv.detach(),
            'dense_loss': dense_loss.detach() if isinstance(dense_loss, torch.Tensor) else dense_loss,  
            'global_loss': global_loss.detach() if isinstance(global_loss, torch.Tensor) else global_loss
        }

    def unfreeze_hubert(self):
        """Unfreeze HuBERT encoder for fine-tuning."""
        self.audio_embedder.unfreeze_hubert()
        
    def get_embeddings(self,
                      audio_input: torch.Tensor,
                      images: Optional[torch.Tensor] = None,
                      attention_mask: Optional[torch.Tensor] = None,
                      cached_visual_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings without computing loss (for inference).
        
        Args:
            audio_input: Raw audio waveform (B, T)
            images: Image tensors (B, 3, H, W) - optional if using cached features
            attention_mask: Audio attention mask (B, T)
            cached_visual_features: Pre-computed visual features - optional
            
        Returns:
            Dictionary with audio_feats, visual_feats, and attention_mask
        """
        with torch.no_grad():
            audio_feats, audio_attention_mask = self.audio_embedder(audio_input, attention_mask)
            
            if cached_visual_features is not None:
                visual_feats = self.visual_embedder(cached_features=cached_visual_features)
            elif images is not None:
                visual_feats = self.visual_embedder(images)
            else:
                raise ValueError("Either images or cached_visual_features must be provided")
                
            return {
                'audio_feats': audio_feats,
                'visual_feats': visual_feats,
                'audio_attention_mask': audio_attention_mask
            }


def create_dummy_inputs(batch_size: int = 2, 
                       audio_duration: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create dummy inputs for testing the VeS model.
    
    Args:
        batch_size: Number of samples in batch
        audio_duration: Audio duration in seconds
        
    Returns:
        Tuple of (audio_input, images)
    """
    audio_seq_len = 16000 * audio_duration  # 16kHz sampling rate
    audio_input = torch.randn(batch_size, audio_seq_len)
    images = torch.randn(batch_size, 3, 224, 224)
    
    return audio_input, images