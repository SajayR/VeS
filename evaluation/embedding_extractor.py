"""
Embedding extraction utilities for evaluation.
"""

import torch
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Tuple, Optional


class EmbeddingExtractor:
    """Handles extraction and caching of embeddings from validation data."""
    
    def __init__(self, 
                 device: str = "cuda",
                 logger: Optional[logging.Logger] = None,
                 use_cached_embeddings: bool = False,
                 cache_dir: Optional[Path] = None):
        """
        Initialize embedding extractor.
        
        Args:
            device: Device to run extraction on
            logger: Logger instance
            use_cached_embeddings: Whether to cache embeddings to disk
            cache_dir: Directory to store cached embeddings
        """
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.use_cached_embeddings = use_cached_embeddings
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.use_cached_embeddings and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, model_step: int) -> Path:
        """Get cache file path for given model step."""
        return self.cache_dir / f"val_embeddings_step{model_step}.pt"
    
    def _load_cached_embeddings(self, model_step: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Load cached embeddings if available."""
        if not (self.use_cached_embeddings and self.cache_dir):
            return None
            
        cache_file = self._get_cache_path(model_step)
        if cache_file.exists():
            self.logger.info(f"Loading cached embeddings from {cache_file}")
            cached = torch.load(cache_file, map_location='cpu')
            return cached['audio'], cached['visual'], cached['masks']
        
        return None
    
    def _save_cached_embeddings(self, 
                               audio_embeds: torch.Tensor,
                               visual_embeds: torch.Tensor, 
                               attention_masks: torch.Tensor,
                               model_step: int):
        """Save embeddings to cache."""
        if not (self.use_cached_embeddings and self.cache_dir):
            return
            
        cache_file = self._get_cache_path(model_step)
        torch.save({
            'audio': audio_embeds,
            'visual': visual_embeds,
            'masks': attention_masks,
            'step': model_step
        }, cache_file)
        self.logger.info(f"Cached embeddings to {cache_file}")
    
    @torch.no_grad()
    def extract_embeddings(self, 
                          model,
                          dataloader,
                          max_samples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from validation set in batches to avoid OOM.
        
        Args:
            model: VeS model to extract embeddings from
            dataloader: Validation dataloader
            max_samples: Maximum number of samples to process
            
        Returns:
            Tuple of:
                - audio_embeds: (N, Na, D) audio embeddings
                - visual_embeds: (N, Nv, D) visual embeddings  
                - attention_masks: (N, Na) attention masks
        """
        model.eval()
        
        # Try to load from cache first
        model_step = getattr(model, 'global_step', 0)
        cached_result = self._load_cached_embeddings(model_step)
        if cached_result is not None:
            return cached_result
        
        # Extract embeddings
        audio_embeds_list = []
        visual_embeds_list = []
        attention_masks_list = []
        num_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
            if max_samples and num_samples >= max_samples:
                break
            
            # Move batch to device
            audio = batch["audio"].to(self.device, non_blocking=True)
            attention_mask = batch["audio_attention_mask"].to(self.device, non_blocking=True)
            
            # Handle visual input (cached features or raw images)
            if "cached_visual_features" in batch:
                cached_features = batch["cached_visual_features"].to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = model(audio, attention_mask=attention_mask, cached_visual_features=cached_features)
            else:
                images = batch["image"].to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = model(audio, images=images, attention_mask=attention_mask)
            
            # Store embeddings on CPU to save GPU memory
            audio_embeds_list.append(outputs['audio_feats'].cpu())
            visual_embeds_list.append(outputs['visual_feats'].cpu())
            attention_masks_list.append(outputs['audio_attention_mask'].cpu())
            
            num_samples += audio.shape[0]
            
            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Concatenate all embeddings
        audio_embeds = torch.cat(audio_embeds_list, dim=0)
        visual_embeds = torch.cat(visual_embeds_list, dim=0)
        attention_masks = torch.cat(attention_masks_list, dim=0)
        
        # Cache embeddings if requested
        self._save_cached_embeddings(audio_embeds, visual_embeds, attention_masks, model_step)
        
        return audio_embeds, visual_embeds, attention_masks