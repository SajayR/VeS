"""
Loss computation utilities for VeS model training.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class VeSLossComputer:
    """
    Handles all loss computations for the VeS model.
    
    Supports different loss types:
    - "dense": Token-level contrastive loss with regularization
    - "global": Global mean-pooled contrastive loss
    - "dense_global": Combination of both dense and global losses
    """
    
    def __init__(self, loss_type: str = "dense", tv_weight: float = 0.1):
        """
        Initialize loss computer.
        
        Args:
            loss_type: Type of loss ("dense", "global", "dense_global")
            tv_weight: Weight for temporal variation regularization
        """
        self.loss_type = loss_type
        self.tv_weight = tv_weight
        assert loss_type in ["dense", "dense_global", "global"], f"Invalid loss type: {loss_type}"
    
    def compute_similarity_matrix(self, feats1: torch.Tensor, feats2: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        """
        Compute token-level cosine similarity between two feature sets.
        
        Args:
            feats1: First feature set (B, N1, D)
            feats2: Second feature set (B, N2, D)
            logit_scale: Learnable temperature parameter
            
        Returns:
            Similarity matrix (B, N1, N2)
        """
        sim = torch.bmm(feats1, feats2.transpose(1, 2))
        sim = sim * logit_scale.exp().clamp(max=100)
        return sim

    def compute_all_similarities_tv(self, 
                                  audio_feats: torch.Tensor,
                                  visual_feats: torch.Tensor,
                                  attention_mask: torch.Tensor,
                                  logit_scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross-modal similarities using hard max over visual patches and mean over audio tokens.
        
        Args:
            audio_feats: Audio embeddings (B, Na, D)
            visual_feats: Visual embeddings (B, Nv, D)
            attention_mask: Audio attention mask (B, Na)
            logit_scale: Learnable temperature parameter
            
        Returns:
            Tuple of:
                - clip_sims: Clip-level similarity matrix (B, B)
                - token_sims: Token-level similarity matrix (B, B, Na, Nv)
        """
        B = audio_feats.size(0)
        Na_audio = audio_feats.size(1)
        Na_mask = attention_mask.size(1)

        if Na_mask != Na_audio:
            raise ValueError(f"Attention mask and audio features have mismatched sequence lengths: {Na_mask} vs {Na_audio}")

        # Compute token-level similarities
        token_sims = torch.einsum(
            'bnd, mvd -> bmnv',
            audio_feats, visual_feats
        ) 

        # Apply attention mask
        mask = attention_mask[:, None, :, None]  # (B,1,Na,1) for broadcasting
        masked_token_sims = torch.where(mask.bool(), token_sims, float('-inf'))

        # Audio-to-visual: max over patches, mean over tokens
        a2v_max = masked_token_sims.max(dim=3).values  # (B, B, Na)
        
        # Replace -inf with 0 to avoid NaN
        a2v_max = torch.where(torch.isinf(a2v_max), torch.zeros_like(a2v_max), a2v_max)
        
        # Compute masked mean
        a_mask_2d = attention_mask.unsqueeze(1).float().expand(-1, B, -1)
        a2v_sum = (a2v_max * a_mask_2d).sum(dim=2)  # (B, B)
        valid_a = a_mask_2d.sum(dim=2).clamp(min=1e-5)
        a2v_clip = a2v_sum / valid_a  # (B, B)

        # Apply temperature scaling
        clip_sims = a2v_clip * logit_scale.exp().clamp(max=100)

        return clip_sims, token_sims

    def compute_regularization_losses_tv(self, 
                                       token_sims: torch.Tensor,
                                       attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute regularization losses: non-negativity and temporal variation.
        
        Args:
            token_sims: Token-level similarities (B, B, Na, Nv)
            attn_mask: Attention mask (B, Na) or None
            
        Returns:
            Tuple of:
                - total_reg_loss: Combined regularization loss
                - l_nonneg: Non-negativity loss
                - l_tv: Temporal variation loss (weighted)
        """
        # Non-negativity regularization
        neg_sims = token_sims.clamp(min=-20.0, max=0.0)
        l_nonneg = neg_sims.pow(2).mean()

        # Early exit if temporal smoothing is disabled
        if self.tv_weight == 0.0:
            return l_nonneg, l_nonneg, torch.zeros_like(l_nonneg)
        
        # Temporal variation regularization on diagonal elements
        B = token_sims.size(0)
        device = token_sims.device

        # Extract max similarities for positive pairs (diagonal)
        a2v_max = token_sims.max(dim=3).values  # (B, B, Na)
        pos_trace = a2v_max[torch.arange(B, device=device), torch.arange(B, device=device)]  # (B, Na)

        if attn_mask is not None:
            m_valid = attn_mask.float().to(device)  # (B, Na)
            neighbour = m_valid[:, 1:] * m_valid[:, :-1]  # (B, Na-1)

            # Scale-invariant temporal variation: normalize by local magnitude
            eps = 1e-6
            denominators = torch.maximum(
                pos_trace[:, :-1].abs(), 
                pos_trace[:, 1:].abs()
            ) + eps  # (B, Na-1)
            
            # Compute relative differences
            relative_diffs = ((pos_trace[:, 1:] - pos_trace[:, :-1]) / denominators).pow(2)
            l_tv = (relative_diffs * neighbour).sum() / neighbour.sum().clamp_min(1.0)
        else:
            # Fallback without attention mask
            eps = 1e-6
            denominators = torch.maximum(
                pos_trace[:, :-1].abs(), 
                pos_trace[:, 1:].abs()
            ) + eps  # (B, Na-1)
            
            relative_diffs = ((pos_trace[:, 1:] - pos_trace[:, :-1]) / denominators).pow(2)
            l_tv = relative_diffs.mean()

        l_tv_weighted = self.tv_weight * l_tv
        total_reg_loss = l_nonneg + l_tv_weighted

        return total_reg_loss, l_nonneg, l_tv_weighted
    
    def compute_contrastive_loss(self, 
                               clip_sims: torch.Tensor,
                               token_sims: torch.Tensor,
                               attention_mask: torch.Tensor,
                               audio_feats: Optional[torch.Tensor] = None,
                               visual_feats: Optional[torch.Tensor] = None,
                               logit_scale: Optional[torch.Tensor] = None,
                               global_weight: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute contrastive loss with regularization.
        
        Args:
            clip_sims: Clip-level similarities (B, B)
            token_sims: Token-level similarities (B, B, Na, Nv)
            attention_mask: Audio attention mask (B, Na)
            audio_feats: Audio features for global loss (B, Na, D)
            visual_feats: Visual features for global loss (B, Nv, D)
            logit_scale: Temperature parameter for global loss
            global_weight: Weight for combining dense and global losses
            
        Returns:
            Tuple of:
                - total_loss: Combined loss
                - l_nonneg: Non-negativity regularization
                - l_tv: Temporal variation regularization
                - dense_loss: Dense contrastive loss
                - global_loss: Global contrastive loss
        """
        batch_size = clip_sims.shape[0]
        labels = torch.arange(batch_size).to(clip_sims.device)
        
        # Dense token-level loss
        if self.loss_type == "dense" or self.loss_type == "dense_global":
            # Audio-to-visual direction
            log_prob_a2v = F.log_softmax(clip_sims, dim=1)
            losses_a2v = -log_prob_a2v[torch.arange(batch_size), labels]
            
            # Visual-to-audio direction  
            log_prob_v2a = F.log_softmax(clip_sims.t(), dim=1)
            losses_v2a = -log_prob_v2a[torch.arange(batch_size), labels]
            
            # Average both directions
            token_contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        else:
            token_contrastive_loss = torch.tensor(0.0, device=clip_sims.device)
            
        # Global mean-pooled loss
        global_contrastive_loss = torch.tensor(0.0, device=clip_sims.device)
        if self.loss_type == "global" or self.loss_type == "dense_global":
            if audio_feats is None or visual_feats is None or logit_scale is None:
                raise ValueError("audio_feats, visual_feats, and logit_scale required for global loss")
                
            # Compute global representations via masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, Na, 1]
            audio_global = (audio_feats * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-6)  # [B, D]
            
            visual_global = visual_feats.mean(dim=1)  # [B, D]
            global_sims = torch.matmul(audio_global, visual_global.t()) * logit_scale.exp().clamp(max=100)  # [B, B]
            
            # Global InfoNCE loss
            log_prob_a2v_global = F.log_softmax(global_sims, dim=1)
            losses_a2v_global = -log_prob_a2v_global[torch.arange(batch_size), labels]
            log_prob_v2a_global = F.log_softmax(global_sims.t(), dim=1)
            losses_v2a_global = -log_prob_v2a_global[torch.arange(batch_size), labels]
            
            global_contrastive_loss = (losses_a2v_global + losses_v2a_global).mean() / 2
        
        # Combine dense and global losses
        contrastive_loss = (1 - global_weight) * token_contrastive_loss + global_weight * global_contrastive_loss
        
        # Add regularization
        reg_loss, l_nonneg, l_tv = self.compute_regularization_losses_tv(token_sims, attention_mask)    
        total_loss = contrastive_loss + reg_loss
        
        return total_loss, l_nonneg, l_tv, token_contrastive_loss, global_contrastive_loss