"""
Audio processing utilities for the VeS dataset.
"""

import random
from typing import Tuple, Optional
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoProcessor


class AudioProcessor:
    """Handles audio loading, preprocessing, and feature extraction."""
    
    def __init__(self, sampling_rate: int = 16000, max_duration: float = 5.0,
                 processor_name: str = "facebook/hubert-large-ls960-ft"):
        """
        Initialize audio processor.
        
        Args:
            sampling_rate: Target sampling rate
            max_duration: Maximum audio duration in seconds
            processor_name: HuggingFace processor name
        """
        self.sampling_rate = sampling_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sampling_rate)
        
        # Initialize HuggingFace processor
        self.processor = AutoProcessor.from_pretrained(processor_name)
        
        # Resampler will be created on-demand
        self.resampler: Optional[T.Resample] = None
        
        # Set audio backend
        self._setup_audio_backend()
    
    def _setup_audio_backend(self):
        """Setup the best available audio backend."""
        available_backends = torchaudio.list_audio_backends()
        
        if 'sox_io' in available_backends:
            torchaudio.set_audio_backend('sox_io')
        elif 'soundfile' in available_backends:
            torchaudio.set_audio_backend('soundfile')
    
    def load_and_resample(self, audio_path: str) -> torch.Tensor:
        """
        Load audio file and resample if necessary.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Resampled audio tensor (1D)
        """
        waveform, original_sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if original_sr != self.sampling_rate:
            if self.resampler is None or self.resampler.orig_freq != original_sr:
                self.resampler = T.Resample(original_sr, self.sampling_rate)
            waveform = self.resampler(waveform)
        
        return waveform.squeeze(0)
    
    def apply_duration_constraint(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply maximum duration constraint with random cropping.
        
        Args:
            audio_tensor: Input audio tensor
            
        Returns:
            Duration-constrained audio tensor
        """
        if audio_tensor.shape[0] > self.max_samples:
            start_idx = random.randint(0, audio_tensor.shape[0] - self.max_samples)
            audio_tensor = audio_tensor[start_idx:start_idx + self.max_samples]
        
        return audio_tensor
    
    def extract_features(self, audio_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features using HuggingFace processor.
        
        Args:
            audio_tensor: Input audio tensor
            
        Returns:
            Tuple of (input_values, attention_mask)
        """
        processed = self.processor(
            audio_tensor.numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_samples,
            truncation=True,
            return_attention_mask=True,
        )
        
        return processed.input_values.squeeze(0), processed.attention_mask.squeeze(0)
    
    def process_audio(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete audio processing pipeline.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (processed audio, attention mask)
        """
        # Load and resample
        audio_tensor = self.load_and_resample(audio_path)
        
        # Apply duration constraint
        audio_tensor = self.apply_duration_constraint(audio_tensor)
        
        # Extract features
        input_values, attention_mask = self.extract_features(audio_tensor)
        
        return input_values, attention_mask