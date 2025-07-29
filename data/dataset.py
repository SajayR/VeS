"""
VeS dataset implementation with modular processing.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .audio_processing import AudioProcessor
from .transforms import ImageProcessor


class VAAPairedDataset(Dataset):
    """
    Video-Audio-Aligned Paired Dataset for VeS training.
    
    Supports both cached visual features and on-the-fly image processing.
    """
    
    def __init__(self, 
                 json_dir_path: str = "/workspace/vaani_jsons",
                 data_base_path: str = "/workspace/vaani_data",
                 crop_strategy: str = "pad_square",
                 target_size: int = 224,
                 max_audio_duration: float = 5.0,
                 sampling_rate: int = 16000,
                 cached_features_base_path: Optional[str] = "/workspace/cached_features/dinov2_large",
                 cached_image_tensors_path: Optional[str] = "/workspace/cached_tensors/images",
                 is_validation: bool = False,
                 debug: bool = False):
        """
        Initialize dataset.
        
        Args:
            json_dir_path: Directory containing JSON mapping files
            data_base_path: Base directory for audio/image files
            crop_strategy: Image cropping strategy
            target_size: Target image size
            max_audio_duration: Maximum audio duration in seconds
            sampling_rate: Audio sampling rate
            cached_features_base_path: Path to cached visual features
            cached_image_tensors_path: Path to cached image tensors
            is_validation: Whether this is validation dataset
            debug: Enable debug mode
        """
        super().__init__()
        
        self.data_base_path = Path(data_base_path)
        self.cached_features_base_path = Path(cached_features_base_path) if cached_features_base_path else None
        self.cached_image_tensors_path = Path(cached_image_tensors_path) if cached_image_tensors_path else None
        self.is_validation = is_validation
        self.debug = debug
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sampling_rate=sampling_rate,
            max_duration=max_audio_duration
        )
        self.image_processor = ImageProcessor(target_size=target_size)
        self.crop_strategy = crop_strategy
        
        # Load dataset mappings
        self.all_mappings = self._load_mappings(json_dir_path)
        
        # Log cached features info
        self._log_cache_info()
    
    def _load_mappings(self, json_dir_path: str) -> List[Dict[str, str]]:
        """Load dataset mappings from JSON files."""
        json_dir = Path(json_dir_path)
        
        if self.is_validation:
            json_files = [json_dir / "validation_set.json"]
            json_files = [f for f in json_files if f.exists()]
        else:
            json_files = sorted(json_dir.glob("train_*.json"))
        
        print(f"Found {len(json_files)} JSON files to load")
        
        all_mappings = []
        for json_path in tqdm(json_files, desc="Loading JSON mappings"):
            with open(json_path, 'r') as f:
                data = json.load(f)
                all_mappings.extend(data)
                print(f"  {json_path.name}: {len(data)} entries")
        
        print(f"Total mappings loaded: {len(all_mappings)}")
        return all_mappings
    
    def _log_cache_info(self):
        """Log information about cached features."""
        if self.cached_features_base_path and self.cached_features_base_path.exists():
            pt_files = list(self.cached_features_base_path.rglob("*.pt"))
            print(f"Using cached visual features from: {self.cached_features_base_path}")
            print(f"Found {len(pt_files)} cached .pt files")
        
        if self.cached_image_tensors_path and self.cached_image_tensors_path.exists():
            print(f"Using cached image tensors from: {self.cached_image_tensors_path}")
    
    def _load_cached_features(self, image_file: str) -> Optional[torch.Tensor]:
        """Load cached visual features if available."""
        if not self.cached_features_base_path:
            return None
            
        pt_filename = self._convert_to_pt_filename(image_file)
        pt_path = self.cached_features_base_path / pt_filename
        
        if pt_path.exists():
            try:
                return torch.load(pt_path, map_location='cpu')
            except Exception as e:
                if self.debug:
                    print(f"Warning: Failed to load cached features from {pt_path}: {e}")
        
        return None
    
    def _load_cached_crop_info(self, image_file: str) -> Optional[Dict[str, Any]]:
        """Load cached crop info if available."""
        if not self.cached_image_tensors_path:
            return None
            
        tensor_file = self._convert_to_pt_filename(image_file)
        tensor_path = self.cached_image_tensors_path / tensor_file
        
        if tensor_path.exists():
            try:
                cached_data = torch.load(tensor_path, map_location='cpu')
                return cached_data.get('crop_info')
            except Exception as e:
                if self.debug:
                    print(f"Warning: Failed to load crop info from {tensor_path}: {e}")
        
        return None
    
    def _convert_to_pt_filename(self, image_file: str) -> str:
        """Convert image filename to .pt filename."""
        return image_file.replace('.jpg', '.pt').replace('.jpeg', '.pt').replace('.png', '.pt')
    
    def _process_sample(self, mapping: Dict[str, str]) -> Dict[str, Any]:
        """Process a single sample."""
        audio_file = mapping['audioFileName'].lstrip('/')
        image_file = mapping['imageFileName'].lstrip('/')
        
        # Construct paths
        audio_path = self.data_base_path / audio_file
        image_path = self.data_base_path / image_file
        
        # Verify files exist
        if not audio_path.exists() or not image_path.exists():
            if self.debug:
                print(f"Warning: Missing files - audio: {audio_path.exists()}, image: {image_path.exists()}")
            return None
        
        # Process audio
        audio_tensor, attention_mask = self.audio_processor.process_audio(str(audio_path))
        
        # Try to load cached features and crop info
        cached_features = self._load_cached_features(image_file)
        crop_info = self._load_cached_crop_info(image_file)
        
        # Build result dictionary
        result = {
            "audio": audio_tensor,
            "audio_attention_mask": attention_mask,
            "sampling_rate": self.audio_processor.sampling_rate,
            "image_path": str(image_path),
            "audio_path": str(audio_path),
            "crop_info": crop_info,
        }
        
        if cached_features is not None:
            # Using cached features
            result.update({
                "using_cached_features": True,
                "cached_visual_features": cached_features
            })
        else:
            # Process image on-the-fly
            if crop_info is None:
                if self.debug:
                    print(f"Warning: Missing crop info for {image_file}")
                return None
                    
            image_tensor, new_crop_info = self.image_processor.process_image(
                str(image_path), 
                self.crop_strategy,
                use_augmentations=True
            )
            
            result.update({
                "image": image_tensor,
                "using_cached_features": False,
                "crop_info": new_crop_info
            })
        
        return result
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.all_mappings)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        mapping = self.all_mappings[idx]
        
        try:
            result = self._process_sample(mapping)
            if result is None:
                # Fallback to next sample
                return self.__getitem__((idx + 1) % len(self))
            return result
            
        except Exception as e:
            if self.debug:
                print(f"Error processing sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))