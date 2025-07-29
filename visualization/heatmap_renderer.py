"""
Heatmap rendering utilities for attention visualization.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple


class HeatmapRenderer:
    """Handles creation and blending of attention heatmaps."""
    
    def __init__(self, alpha: float = 0.9):
        """
        Initialize heatmap renderer.
        
        Args:
            alpha: Heatmap opacity for blending
        """
        self.alpha = alpha
        self._inferno_cmap = self._create_inferno_with_alpha()
    
    def _create_inferno_with_alpha(self) -> mcolors.ListedColormap:
        """Create inferno colormap with variable alpha channel."""
        inferno = plt.cm.inferno(np.linspace(0, 1, 256))
        alphas = np.linspace(0, 1, 256)
        inferno_with_alpha = np.zeros((256, 4))
        inferno_with_alpha[:, 0:3] = inferno[:, 0:3]
        inferno_with_alpha[:, 3] = alphas
        return mcolors.ListedColormap(inferno_with_alpha)
    
    def similarity_to_heatmap(self, sim_row: torch.Tensor, grid: int = 16, 
                            size: int = 224) -> np.ndarray:
        """
        Convert similarity vector to RGBA heatmap.
        
        Args:
            sim_row: Similarity vector (Nv,) where Nv = grid^2
            grid: Grid dimension (usually 16 for 16x16 patches)
            size: Output heatmap size
            
        Returns:
            RGBA heatmap array (size, size, 4)
        """
        expected_size = grid * grid
        assert sim_row.shape == torch.Size([expected_size]), \
            f"Expected sim_row shape [{expected_size}], got {sim_row.shape}"
        
        # Reshape to grid and normalize
        arr = sim_row.view(grid, grid).float().cpu()
        arr = arr.clamp_min(0)
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-6)
        
        # Resize to target size
        arr_resized = cv2.resize(arr.numpy(), (size, size), interpolation=cv2.INTER_CUBIC)
        
        # Apply inferno colormap with alpha
        heat_rgba = self._inferno_cmap(arr_resized)  # (size, size, 4)
        heat_rgba = (heat_rgba * 255).astype(np.uint8)
        
        return heat_rgba
    
    def blend_heatmap(self, rgb_base: np.ndarray, heat_rgba: np.ndarray) -> np.ndarray:
        """
        Blend RGB image with RGBA heatmap using alpha compositing.
        
        Args:
            rgb_base: Base RGB image (H, W, 3)
            heat_rgba: RGBA heatmap (H, W, 4)
            
        Returns:
            Blended RGB image (H, W, 3)
        """
        # Extract RGB and alpha from heatmap
        heat_rgb = heat_rgba[:, :, :3].astype(np.float32) / 255.0
        heat_alpha = heat_rgba[:, :, 3:4].astype(np.float32) / 255.0
        
        # Convert base image to float
        rgb_base_float = rgb_base.astype(np.float32) / 255.0
        
        # Alpha blend: result = heat * alpha + base * (1 - alpha)
        alpha_3ch = np.repeat(heat_alpha, 3, axis=2)
        result = heat_rgb * alpha_3ch + rgb_base_float * (1 - alpha_3ch)
        
        # Convert back to uint8
        return (result * 255).astype(np.uint8)