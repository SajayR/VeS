"""
Main retrieval evaluation class for VeS model.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Optional
import wandb

from .metrics import RetrievalMetrics, compute_retrieval_metrics_from_ranks
from .similarity_computer import SimilarityComputer
from .embedding_extractor import EmbeddingExtractor


class RetrievalEvaluator:
    """
    Comprehensive retrieval evaluation for VeS model.
    
    Handles embedding extraction, similarity computation, and metric calculation
    for both audio-to-visual and visual-to-audio retrieval tasks.
    """
    
    def __init__(self, 
                 val_dataloader,
                 device: str = "cuda",
                 batch_size: int = 32,
                 logger: Optional[logging.Logger] = None,
                 use_cached_embeddings: bool = False,
                 cache_dir: Optional[Path] = None):
        """
        Initialize retrieval evaluator.
        
        Args:
            val_dataloader: Validation data loader
            device: Device for computation
            batch_size: Batch size for evaluation
            logger: Logger instance
            use_cached_embeddings: Whether to cache embeddings
            cache_dir: Directory for caching embeddings
        """
        self.val_dataloader = val_dataloader
        self.device = device
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.embedding_extractor = EmbeddingExtractor(
            device=device,
            logger=self.logger,
            use_cached_embeddings=use_cached_embeddings,
            cache_dir=cache_dir
        )
        
        self.similarity_computer = SimilarityComputer(device=device)
    
    def compute_retrieval_metrics(self, sim_matrix: torch.Tensor) -> Dict[str, RetrievalMetrics]:
        """
        Compute retrieval metrics from similarity matrix.
        
        Args:
            sim_matrix: Similarity matrix (N, N) where sim_matrix[i,j] is 
                       similarity between audio i and visual j
            
        Returns:
            Dictionary with 'audio_to_visual' and 'visual_to_audio' metrics
        """
        N = sim_matrix.shape[0]
        
        # Audio-to-Visual retrieval: for each audio, rank all visuals
        a2v_ranks = []
        for i in range(N):
            similarities = sim_matrix[i]  # Similarities for audio i to all visuals
            # Rank of correct match (diagonal element i)
            rank = (similarities > similarities[i]).sum().item() + 1
            a2v_ranks.append(rank)
        
        # Visual-to-Audio retrieval: for each visual, rank all audios
        v2a_ranks = []
        for j in range(N):
            similarities = sim_matrix[:, j]  # Similarities for visual j to all audios
            # Rank of correct match (diagonal element j)
            rank = (similarities > similarities[j]).sum().item() + 1
            v2a_ranks.append(rank)
        
        return {
            'audio_to_visual': compute_retrieval_metrics_from_ranks(np.array(a2v_ranks)),
            'visual_to_audio': compute_retrieval_metrics_from_ranks(np.array(v2a_ranks))
        }
    
    @torch.no_grad()
    def evaluate(self,
                model,
                max_samples: Optional[int] = None,
                log_to_wandb: bool = True,
                global_step: Optional[int] = None) -> Dict[str, Dict[str, RetrievalMetrics]]:
        """
        Run complete retrieval evaluation on validation set.
        
        Args:
            model: VeS model to evaluate
            max_samples: Maximum number of samples to evaluate
            log_to_wandb: Whether to log results to Weights & Biases
            global_step: Global training step for logging
            
        Returns:
            Dictionary with results for different aggregation methods:
            {
                'max_mean': {'audio_to_visual': metrics, 'visual_to_audio': metrics},
                'mean_pooled': {'audio_to_visual': metrics, 'visual_to_audio': metrics}
            }
        """
        model.eval()
        
        # Extract embeddings from validation set
        self.logger.info("Extracting embeddings from validation set...")
        audio_embeds, visual_embeds, attention_masks = self.embedding_extractor.extract_embeddings(
            model, self.val_dataloader, max_samples
        )
        
        self.logger.info(f"Extracted embeddings: audio {audio_embeds.shape}, visual {visual_embeds.shape}")
        
        results = {}
        
        # Evaluate with max-mean aggregation (model's training method)
        self.logger.info("Computing max-mean aggregated similarities...")
        sim_matrix_maxmean = self.similarity_computer.compute_similarity_matrix_chunked(
            audio_embeds, visual_embeds, attention_masks, 
            aggregation="max_mean"
        )
        results['max_mean'] = self.compute_retrieval_metrics(sim_matrix_maxmean)
        
        # Evaluate with mean pooling (global comparison)
        self.logger.info("Computing mean-pooled similarities...")
        sim_matrix_mean = self.similarity_computer.compute_similarity_matrix_chunked(
            audio_embeds, visual_embeds, attention_masks,
            aggregation="mean"
        )
        results['mean_pooled'] = self.compute_retrieval_metrics(sim_matrix_mean)
        
        # Log results
        self._log_results(results, log_to_wandb, global_step)
        
        # Clean up GPU memory
        del audio_embeds, visual_embeds, attention_masks, sim_matrix_maxmean, sim_matrix_mean
        torch.cuda.empty_cache()
        
        return results
    
    def _log_results(self,
                    results: Dict[str, Dict[str, RetrievalMetrics]], 
                    log_to_wandb: bool,
                    global_step: Optional[int]):
        """Log evaluation results to console and wandb."""
        # Console logging
        for method, method_results in results.items():
            self.logger.info(f"\n{method.upper().replace('_', ' ')} Aggregation Results:")
            for direction, metrics in method_results.items():
                self.logger.info(f"  {direction.replace('_', ' ').title()}: {metrics}")
        
        # Weights & Biases logging
        if log_to_wandb and wandb.run is not None:
            wandb_dict = {}
            for method, method_results in results.items():
                for direction, metrics in method_results.items():
                    prefix = f"val/{method}/{direction}/"
                    wandb_dict.update(metrics.to_dict(prefix))
            
            if global_step is not None:
                wandb.log(wandb_dict, step=global_step)
            else:
                wandb.log(wandb_dict)