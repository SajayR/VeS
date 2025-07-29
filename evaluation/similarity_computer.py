"""
Similarity computation utilities for retrieval evaluation.
"""

import torch
import torch.nn.functional as F
from typing import Literal


class SimilarityComputer:
    """Handles different similarity computation methods for retrieval evaluation."""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize similarity computer.
        
        Args:
            device: Device to run computations on
        """
        self.device = device
    
    def compute_max_mean_similarity(self,
                                  audio_feats: torch.Tensor,
                                  visual_feats: torch.Tensor,
                                  attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute max-mean aggregated similarity (matching VeS model's approach).
        
        This computes:
        1. Token-level similarities between all audio-visual pairs
        2. For each audio token, max over visual patches 
        3. Mean over valid audio tokens (using attention mask)
        
        Args:
            audio_feats: Audio embeddings (B1, Na, D)
            visual_feats: Visual embeddings (B2, Nv, D)
            attention_mask: Audio attention mask (B1, Na)
            
        Returns:
            Similarity matrix (B1, B2)
        """
        B1, Na, D = audio_feats.shape
        B2, Nv, _ = visual_feats.shape
        
        # Compute token-level similarities: audio tokens vs visual patches
        token_sims = torch.einsum('bnd,mvd->bmnv', audio_feats, visual_feats)  # (B1, B2, Na, Nv)
        
        # Apply attention mask to ignore padded audio tokens
        mask = attention_mask[:, None, :, None]  # (B1, 1, Na, 1) for broadcasting
        masked_sims = torch.where(mask.bool(), token_sims, float('-inf'))
        
        # Audio-to-Visual aggregation: max over visual patches, mean over audio tokens
        a2v_max = masked_sims.max(dim=3).values  # (B1, B2, Na) - max over patches
        
        # Replace -inf with 0 to avoid NaN in subsequent operations
        a2v_max = torch.where(torch.isinf(a2v_max), torch.zeros_like(a2v_max), a2v_max)
        
        # Compute masked mean over audio tokens
        a_mask_2d = attention_mask.unsqueeze(1).float()  # (B1, 1, Na)
        a2v_sum = (a2v_max * a_mask_2d).sum(dim=2)  # (B1, B2)
        valid_tokens = a_mask_2d.sum(dim=2).clamp(min=1e-5)  # (B1, 1)
        clip_sims = a2v_sum / valid_tokens  # (B1, B2)
        
        return clip_sims
    
    def compute_mean_pooled_similarity(self,
                                     audio_feats: torch.Tensor,
                                     visual_feats: torch.Tensor,
                                     attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute mean-pooled similarity for global comparison.
        
        Args:
            audio_feats: Audio embeddings (B1, Na, D)
            visual_feats: Visual embeddings (B2, Nv, D)
            attention_mask: Audio attention mask (B1, Na)
            
        Returns:
            Similarity matrix (B1, B2)
        """
        # Mean pool audio features with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B1, Na, 1)
        audio_global = (audio_feats * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-6)  # (B1, D)
        
        # Mean pool visual features
        visual_global = visual_feats.mean(dim=1)  # (B2, D)
        
        # Compute cosine similarities
        clip_sims = torch.matmul(audio_global, visual_global.t())  # (B1, B2)
        
        return clip_sims
    
    def compute_similarity_matrix_chunked(self,
                                        audio_embeds: torch.Tensor,
                                        visual_embeds: torch.Tensor,
                                        attention_masks: torch.Tensor,
                                        aggregation: Literal["max_mean", "mean"] = "max_mean",
                                        chunk_size: int = 100) -> torch.Tensor:
        """
        Compute similarity matrix in chunks to avoid OOM.
        
        Args:
            audio_embeds: Audio embeddings (N, Na, D)
            visual_embeds: Visual embeddings (N, Nv, D)
            attention_masks: Audio attention masks (N, Na)
            aggregation: Similarity aggregation method
            chunk_size: Number of samples to process at once
            
        Returns:
            Similarity matrix (N, N)
        """
        N = audio_embeds.shape[0]
        sim_matrix = torch.zeros(N, N, dtype=torch.float32)
        
        # Process in chunks to avoid memory explosion
        for i in range(0, N, chunk_size):
            i_end = min(i + chunk_size, N)
            
            for j in range(0, N, chunk_size):
                j_end = min(j + chunk_size, N)
                
                # Move chunks to GPU
                audio_chunk = audio_embeds[i:i_end].to(self.device)
                visual_chunk = visual_embeds[j:j_end].to(self.device)
                mask_chunk = attention_masks[i:i_end].to(self.device)
                
                # Compute similarities for this chunk
                if aggregation == "max_mean":
                    chunk_sim = self.compute_max_mean_similarity(
                        audio_chunk, visual_chunk, mask_chunk
                    )
                else:  # mean pooling
                    chunk_sim = self.compute_mean_pooled_similarity(
                        audio_chunk, visual_chunk, mask_chunk
                    )
                
                # Store result back to CPU
                sim_matrix[i:i_end, j:j_end] = chunk_sim.cpu()
                
                # Clear GPU memory
                del audio_chunk, visual_chunk, mask_chunk, chunk_sim
                torch.cuda.empty_cache()
        
        return sim_matrix