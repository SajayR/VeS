"""
Main VeS visualization class - refactored for modularity.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image as PILImage
import torchvision.transforms as transforms

from .video_encoder import VideoEncoder
from .image_utils import ImageUtils


class VeSVisualizer:
    """
    Main visualization class for VeS model attention patterns.
    
    Creates MP4 videos with heatmap overlays showing token-level attention
    over visual patches, synchronized with audio tokens at ~25 FPS.
    """
    
    def __init__(self,
                 out_dir: str = "visualizations",
                 token_hz: int = 25,
                 alpha: float = 0.9,
                 max_samples_per_call: int = 4,
                 reduction: int = 2,
                 side_by_side: bool = True,
                 separator_width: int = 4,
                 label_height: int = 30):
        """
        Initialize VeS visualizer.
        
        Args:
            out_dir: Output directory for videos and plots
            token_hz: Token frequency (25 Hz = 40ms per token after reduction)
            alpha: Heatmap opacity
            max_samples_per_call: Maximum samples to process per batch
            reduction: Audio downsampling reduction factor
            side_by_side: Enable side-by-side original/heatmap layout
            separator_width: Width of separator between frames
            label_height: Height for text labels
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_samples_per_call = max_samples_per_call
        
        # Initialize components
        self.video_encoder = VideoEncoder(
            fps=token_hz,
            side_by_side=side_by_side,
            separator_width=separator_width,
            label_height=label_height
        )
        
        self.image_utils = ImageUtils()
    
    def visualize_batch(self, batch: Dict[str, torch.Tensor], 
                       outputs: Dict[str, torch.Tensor], step: int) -> List[Tuple[str, plt.Figure]]:
        """
        Visualize a batch of samples from training.
        
        Args:
            batch: Batch dictionary from DataLoader
            outputs: Model outputs containing token_sims
            step: Training step number
            
        Returns:
            List of (basename, matplotlib_figure) tuples for logging
        """
        # Process crop info
        crop_infos = self._process_crop_info(batch)
        
        # Handle images (load from paths if using cached features)
        if "image" not in batch:
            images = self._load_images_from_paths(batch["image_path"], crop_infos)
        else:
            images = batch["image"].cpu()
        
        # Extract other data
        audio = batch["audio"].cpu().numpy()
        sample_rates = self._normalize_sample_rates(batch["sampling_rate"], images.size(0))
        attention_masks = outputs["audio_attention_mask"].cpu()
        token_similarities = outputs["token_sims"].cpu()
        
        # Generate visualizations
        matplotlib_figures = []
        num_samples = min(images.size(0), self.max_samples_per_call)
        
        for i in range(num_samples):
            basename = f"step{step}_idx{i}"
            output_path = self.out_dir / f"{basename}.mp4"
            
            # Generate video and collect frames
            frames, timestamps = self.video_encoder.encode_sample(
                image_tensor=images[i],
                token_similarities=token_similarities[i, i],  # Diagonal for self-similarity
                audio_array=audio[i],
                sample_rate=sample_rates[i],
                output_path=output_path,
                attention_mask=attention_masks[i],
                crop_info=crop_infos[i] if i < len(crop_infos) else None,
                sample_idx=i
            )
            
            # Create matplotlib figure for logging
            if frames:
                fig = self._create_frame_plot(frames, timestamps, basename)
                if fig is not None:
                    matplotlib_figures.append((basename, fig))
        
        return matplotlib_figures
    
    def _process_crop_info(self, batch: Dict[str, torch.Tensor]) -> List[Optional[Dict]]:
        """Process crop information from batch."""
        if "crop_info" not in batch:
            batch_size = len(batch.get("image_path", batch.get("image", [])))
            return [None] * batch_size
        
        raw_crop_infos = batch["crop_info"]
        
        if isinstance(raw_crop_infos, list):
            return raw_crop_infos
        elif isinstance(raw_crop_infos, dict):
            # Handle batched dictionary
            batch_size = len(batch.get("image_path", batch.get("image", [])))
            crop_infos = []
            
            for i in range(batch_size):
                try:
                    sample_crop_info = {}
                    for k, v in raw_crop_infos.items():
                        if isinstance(v, (list, tuple)):
                            sample_crop_info[k] = v[i]
                        elif hasattr(v, '__getitem__') and hasattr(v, 'shape') and len(v.shape) > 0:
                            sample_crop_info[k] = v[i]
                        else:
                            sample_crop_info[k] = v
                    crop_infos.append(sample_crop_info)
                except (IndexError, KeyError):
                    crop_infos.append(None)
            
            return crop_infos
        else:
            batch_size = len(batch.get("image_path", batch.get("image", [])))
            return [None] * batch_size
    
    def _load_images_from_paths(self, image_paths: List[str], 
                               crop_infos: List[Optional[Dict]]) -> torch.Tensor:
        """Load images from file paths with consistent preprocessing."""
        images = []
        
        for i, img_path in enumerate(image_paths):
            image = PILImage.open(img_path).convert('RGB')
            crop_info = crop_infos[i] if i < len(crop_infos) and crop_infos[i] is not None else {}
            
            # Extract crop parameters
            crop_strategy = self._extract_crop_strategy(crop_info)
            target_size = self._extract_target_size(crop_info)
            
            # Apply consistent preprocessing (no augmentations for visualization)
            if crop_strategy == "pad_square":
                image = self._apply_pad_square(image, target_size)
            
            # Convert to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            images.append(transform(image))
        
        return torch.stack(images)
    
    def _extract_crop_strategy(self, crop_info: Dict) -> str:
        """Extract crop strategy from crop info."""
        crop_strategy = crop_info.get("crop_strategy", "pad_square")
        
        if hasattr(crop_strategy, 'item'):
            crop_strategy = crop_strategy.item()
        elif isinstance(crop_strategy, bytes):
            crop_strategy = crop_strategy.decode('utf-8')
        
        return str(crop_strategy)
    
    def _extract_target_size(self, crop_info: Dict) -> int:
        """Extract target size from crop info."""
        target_size = crop_info.get("target_size", 224)
        
        if hasattr(target_size, 'item'):
            target_size = int(target_size.item())
        else:
            target_size = int(target_size)
        
        return target_size
    
    def _apply_pad_square(self, image: PILImage.Image, target_size: int) -> PILImage.Image:
        """Apply pad_square preprocessing to image."""
        width, height = image.size
        max_dim = max(width, height)
        new_image = PILImage.new('RGB', (max_dim, max_dim), color=(0, 0, 0))
        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        new_image.paste(image, (paste_x, paste_y))
        return new_image.resize((target_size, target_size), PILImage.LANCZOS)
    
    def _normalize_sample_rates(self, sample_rates, batch_size: int) -> List[int]:
        """Normalize sample rates to per-sample list."""
        if isinstance(sample_rates, int):
            return [sample_rates] * batch_size
        return list(sample_rates)
    
    def _create_frame_plot(self, frames: List[np.ndarray], timestamps: List[float], 
                          basename: str) -> Optional[plt.Figure]:
        """Create matplotlib figure showing selected frames."""
        if not frames:
            return None
        
        n_frames = len(frames)
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.1)
        
        for i, (frame, ts) in enumerate(zip(frames, timestamps)):
            if i >= 6:  # Limit to 6 frames
                break
                
            ax = fig.add_subplot(gs[i // 3, i % 3])
            ax.imshow(frame)
            ax.set_title(f"t = {ts:.2f}s", fontsize=12)
            ax.axis('off')
        
        fig.suptitle(f"Attention Visualization: {basename}", fontsize=16, y=0.98)
        plt.tight_layout()
        
        return fig