"""
Image processing utilities for visualization.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Any, Optional


class ImageUtils:
    """Utilities for image processing in visualization pipeline."""
    
    def __init__(self):
        # ImageNet normalization constants
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def unnormalize_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert normalized tensor to uint8 RGB array.
        
        Args:
            img_tensor: Normalized image tensor (3, H, W)
            
        Returns:
            RGB array (H, W, 3) in uint8 format
        """
        img = img_tensor.cpu() * self._std + self._mean
        img = (img.clamp(0, 1) * 255).byte()
        return img.permute(1, 2, 0).numpy()
    
    def calculate_crop_region(self, crop_info: Optional[Dict[str, Any]], 
                            video_size: int = 448, sample_idx: int = 0) -> Tuple[int, int, int, int]:
        """
        Calculate crop region to remove black padding from pad_square strategy.
        
        Args:
            crop_info: Dictionary with original dimensions and crop strategy
            video_size: Current video frame size
            sample_idx: Sample index for batched data
            
        Returns:
            Crop coordinates (x1, y1, x2, y2)
        """
        if crop_info is None:
            return (0, 0, video_size, video_size)
        
        # Extract crop strategy (handle batched data)
        crop_strategy = self._extract_batch_value(crop_info.get("crop_strategy"), sample_idx)
        
        # Convert bytes to string if needed
        if isinstance(crop_strategy, bytes):
            crop_strategy = crop_strategy.decode('utf-8')
        
        if crop_strategy != "pad_square":
            return (0, 0, video_size, video_size)
        
        # Extract original dimensions and target size
        orig_w = self._extract_batch_value(crop_info.get("original_width"), sample_idx)
        orig_h = self._extract_batch_value(crop_info.get("original_height"), sample_idx)
        target_size = self._extract_batch_value(crop_info.get("target_size", 224), sample_idx)
        
        # Convert to scalars
        orig_w = self._to_scalar(orig_w)
        orig_h = self._to_scalar(orig_h)
        target_size = self._to_scalar(target_size)
        
        # Calculate content dimensions in current video size
        max_dim = max(orig_w, orig_h)
        content_w_ratio = orig_w / max_dim
        content_h_ratio = orig_h / max_dim
        
        content_w_pixels = int(content_w_ratio * video_size)
        content_h_pixels = int(content_h_ratio * video_size)
        
        # Ensure even dimensions for video encoding
        content_w_pixels = content_w_pixels - (content_w_pixels % 2)
        content_h_pixels = content_h_pixels - (content_h_pixels % 2)
        
        # Calculate centered crop coordinates
        x1 = (video_size - content_w_pixels) // 2
        y1 = (video_size - content_h_pixels) // 2
        x2 = x1 + content_w_pixels
        y2 = y1 + content_h_pixels
        
        return (x1, y1, x2, y2)
    
    def _extract_batch_value(self, value: Any, sample_idx: int) -> Any:
        """Extract value from potentially batched tensor/list."""
        if value is None:
            return None
        
        # Handle tensors and lists
        if hasattr(value, '__getitem__') and hasattr(value, 'shape') and len(value.shape) > 0:
            return value[sample_idx]
        elif isinstance(value, (list, tuple)) and len(value) > sample_idx:
            return value[sample_idx]
        else:
            return value
    
    def _to_scalar(self, value: Any) -> Any:
        """Convert tensor to scalar if needed."""
        if hasattr(value, 'item'):
            return value.item()
        return value
    
    def apply_crop(self, frame: np.ndarray, crop_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply cropping to frame.
        
        Args:
            frame: Input frame array
            crop_coords: Crop coordinates (x1, y1, x2, y2)
            
        Returns:
            Cropped frame
        """
        x1, y1, x2, y2 = crop_coords
        return frame[y1:y2, x1:x2]
    
    def add_timestamp_overlay(self, frame: np.ndarray, timestamp: float, 
                            position: str = "bottom-left") -> np.ndarray:
        """
        Add timestamp overlay to frame.
        
        Args:
            frame: Input frame
            timestamp: Timestamp in seconds
            position: Position for overlay ("bottom-left", "top-left", etc.)
            
        Returns:
            Frame with timestamp overlay
        """
        frame_copy = frame.copy()
        
        # Configure text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        color = (255, 255, 255)
        thickness = 1
        
        # Determine position
        text = f"{timestamp:5.2f}s"
        if position == "bottom-left":
            position_coords = (10, frame.shape[0] - 10)
        elif position == "top-left":
            position_coords = (10, 25)
        else:
            position_coords = (10, frame.shape[0] - 10)  # Default to bottom-left
        
        cv2.putText(frame_copy, text, position_coords, font, font_scale, 
                   color, thickness, cv2.LINE_AA)
        
        return frame_copy
    
    def create_side_by_side_frame(self, left_frame: np.ndarray, right_frame: np.ndarray,
                                separator_width: int = 4, label_height: int = 30) -> np.ndarray:
        """
        Create side-by-side frame with labels.
        
        Args:
            left_frame: Left image frame
            right_frame: Right image frame
            separator_width: Width of separator line
            label_height: Height for text labels
            
        Returns:
            Combined frame with labels and separator
        """
        H, W, C = left_frame.shape
        
        # Create canvas
        total_width = W * 2 + separator_width
        total_height = H + label_height
        canvas = np.zeros((total_height, total_width, C), dtype=np.uint8)
        
        # Add frames
        canvas[label_height:, :W] = left_frame
        canvas[label_height:, W + separator_width:] = right_frame
        
        # Add separator
        if separator_width > 0:
            separator_color = (128, 128, 128)
            canvas[label_height:, W:W + separator_width] = separator_color
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        font_thickness = 2
        
        # "Original" label
        text_size = cv2.getTextSize("Original", font, font_scale, font_thickness)[0]
        text_x = (W - text_size[0]) // 2
        text_y = label_height - 8
        cv2.putText(canvas, "Original", (text_x, text_y), font, font_scale, 
                   font_color, font_thickness, cv2.LINE_AA)
        
        # "Heatmap" label
        text_size = cv2.getTextSize("Heatmap", font, font_scale, font_thickness)[0]
        text_x = W + separator_width + (W - text_size[0]) // 2
        cv2.putText(canvas, "Heatmap", (text_x, text_y), font, font_scale, 
                   font_color, font_thickness, cv2.LINE_AA)
        
        return canvas