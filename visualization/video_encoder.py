"""
Video encoding utilities for attention visualization.
"""

import av
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from .heatmap_renderer import HeatmapRenderer
from .image_utils import ImageUtils


class VideoEncoder:
    """Handles MP4 video encoding with heatmap overlays."""
    
    def __init__(self, fps: int = 25, side_by_side: bool = True, 
                 separator_width: int = 4, label_height: int = 30):
        """
        Initialize video encoder.
        
        Args:
            fps: Video frame rate
            side_by_side: Enable side-by-side layout
            separator_width: Width of separator between frames
            label_height: Height for text labels
        """
        self.fps = fps
        self.side_by_side = side_by_side
        self.separator_width = separator_width
        self.label_height = label_height
        
        # Initialize helper components
        self.heatmap_renderer = HeatmapRenderer()
        self.image_utils = ImageUtils()
        
        # Audio processing constants
        self.samples_per_token = 320 * 2  # Reduction factor of 2
    
    def encode_sample(self, 
                     image_tensor: torch.Tensor,
                     token_similarities: torch.Tensor,
                     audio_array: np.ndarray,
                     sample_rate: int,
                     output_path: Path,
                     attention_mask: torch.Tensor,
                     crop_info: Optional[Dict[str, Any]] = None,
                     sample_idx: int = 0) -> Tuple[List[np.ndarray], List[float]]:
        """
        Encode a single sample to MP4 video.
        
        Args:
            image_tensor: Normalized image tensor (3, 224, 224)
            token_similarities: Token-level similarities (Na, Nv)
            audio_array: Audio waveform array
            sample_rate: Audio sample rate
            output_path: Output video file path
            attention_mask: Audio attention mask (Na,)
            crop_info: Image crop information
            sample_idx: Sample index for batched data
            
        Returns:
            Tuple of (collected frames, timestamps) for matplotlib visualization
        """
        # Prepare base image
        rgb_base = self.image_utils.unnormalize_image(image_tensor)
        rgb_base = self._resize_image(rgb_base, 448)  # Double size for better quality
        
        # Calculate crop region and apply cropping
        crop_coords = self.image_utils.calculate_crop_region(crop_info, 448, sample_idx)
        rgb_base_cropped = self.image_utils.apply_crop(rgb_base, crop_coords)
        H, W, _ = rgb_base_cropped.shape
        
        # Determine valid token count
        valid_tokens = self._calculate_valid_tokens(attention_mask, token_similarities)
        token_similarities = token_similarities[:valid_tokens]
        
        # Trim audio to match token count
        expected_samples = valid_tokens * self.samples_per_token
        audio_array = audio_array[:expected_samples]
        
        # Collect frames for matplotlib visualization
        frame_indices = np.linspace(0, valid_tokens - 1, min(6, valid_tokens), dtype=int)
        collected_frames = []
        collected_timestamps = []
        
        # Setup video encoding
        container, video_stream, audio_stream = self._setup_video_container(
            output_path, sample_rate, H, W
        )
        
        try:
            # Generate and encode video frames
            for t in range(valid_tokens):
                timestamp = t / self.fps
                
                # Create heatmap overlay
                heat_rgba = self.heatmap_renderer.similarity_to_heatmap(
                    token_similarities[t], size=448
                )
                frame_with_heatmap = self.heatmap_renderer.blend_heatmap(rgb_base, heat_rgba)
                frame_with_heatmap = self.image_utils.apply_crop(frame_with_heatmap, crop_coords)
                
                # Add timestamp overlay
                frame_with_heatmap = self.image_utils.add_timestamp_overlay(
                    frame_with_heatmap, timestamp
                )
                
                # Collect frames for matplotlib
                if t in frame_indices:
                    collected_frames.append(frame_with_heatmap.copy())
                    collected_timestamps.append(timestamp)
                
                # Create final video frame
                if self.side_by_side:
                    original_cropped = self.image_utils.apply_crop(rgb_base, crop_coords)
                    video_frame = self.image_utils.create_side_by_side_frame(
                        original_cropped, frame_with_heatmap,
                        self.separator_width, self.label_height
                    )
                else:
                    video_frame = frame_with_heatmap
                
                # Encode video frame
                self._encode_video_frame(video_stream, video_frame, container)
            
            # Finalize video encoding
            self._finalize_video_stream(video_stream, container)
            
            # Encode audio
            self._encode_audio(audio_stream, audio_array, sample_rate, container)
            
        finally:
            container.close()
        
        return collected_frames, collected_timestamps
    
    def _resize_image(self, image: np.ndarray, size: int) -> np.ndarray:
        """Resize image using cubic interpolation."""
        import cv2
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    
    def _calculate_valid_tokens(self, attention_mask: torch.Tensor, 
                              token_similarities: torch.Tensor) -> int:
        """Calculate number of valid tokens based on attention mask."""
        mask_tokens = int(attention_mask.round().sum().item())
        sim_tokens = token_similarities.size(0)
        return min(mask_tokens, sim_tokens)
    
    def _setup_video_container(self, output_path: Path, sample_rate: int, 
                             height: int, width: int) -> Tuple[av.container.Container, av.VideoStream, av.AudioStream]:
        """Setup AV container and streams."""
        container = av.open(str(output_path), mode="w")
        
        # Video stream
        video_stream = container.add_stream("libx264", rate=self.fps)
        video_stream.pix_fmt = "yuv420p"
        
        if self.side_by_side:
            video_stream.width = width * 2 + self.separator_width
            video_stream.height = height + self.label_height
        else:
            video_stream.width = width
            video_stream.height = height
        
        # Audio stream
        audio_stream = container.add_stream("aac", rate=int(sample_rate))
        audio_stream.layout = "mono"
        
        return container, video_stream, audio_stream
    
    def _encode_video_frame(self, stream: av.VideoStream, frame: np.ndarray, 
                          container: av.container.Container):
        """Encode a single video frame."""
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    
    def _finalize_video_stream(self, stream: av.VideoStream, container: av.container.Container):
        """Flush any pending video frames."""
        for packet in stream.encode():
            container.mux(packet)
    
    def _encode_audio(self, stream: av.AudioStream, audio_array: np.ndarray, 
                     sample_rate: int, container: av.container.Container):
        """Encode audio stream."""
        # Convert to 16-bit integer
        if audio_array.dtype.kind == "f":
            audio16 = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
        else:
            audio16 = audio_array.astype(np.int16)
        
        # Reshape for mono audio
        audio16 = audio16.reshape(1, -1)
        
        # Create audio frame and encode
        audio_frame = av.AudioFrame.from_ndarray(audio16, format="s16", layout="mono")
        audio_frame.sample_rate = sample_rate
        
        for packet in stream.encode(audio_frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)