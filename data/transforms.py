"""
Image preprocessing and transformation utilities.
"""

import random
from typing import Tuple, Dict, Any
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage


class ImageProcessor:
    """Handles image preprocessing with various cropping strategies."""
    
    def __init__(self, target_size: int = 224):
        """
        Initialize image processor.
        
        Args:
            target_size: Target image size for model input
        """
        self.target_size = target_size
        
        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def get_transform(self, use_augmentations: bool = True) -> transforms.Compose:
        """
        Get image transformation pipeline.
        
        Args:
            use_augmentations: Whether to apply data augmentations
            
        Returns:
            Composed transforms
        """
        if use_augmentations:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                self.normalize
            ])
    
    def apply_crop_strategy(self, image: PILImage.Image, strategy: str) -> PILImage.Image:
        """
        Apply specified cropping strategy to image.
        
        Args:
            image: PIL Image
            strategy: Cropping strategy ("stretch", "center_crop", "random_crop", "pad_square")
            
        Returns:
            Processed PIL Image
        """
        if strategy == "stretch":
            return image.resize((self.target_size, self.target_size), PILImage.LANCZOS)
            
        elif strategy == "center_crop":
            width, height = image.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            image = image.crop((left, top, right, bottom))
            return image.resize((self.target_size, self.target_size), PILImage.LANCZOS)
            
        elif strategy == "random_crop":
            width, height = image.size
            min_dim = min(width, height)
            max_left = width - min_dim
            max_top = height - min_dim
            left = random.randint(0, max_left) if max_left > 0 else 0
            top = random.randint(0, max_top) if max_top > 0 else 0
            right = left + min_dim
            bottom = top + min_dim
            image = image.crop((left, top, right, bottom))
            return image.resize((self.target_size, self.target_size), PILImage.LANCZOS)
            
        elif strategy == "pad_square":
            width, height = image.size
            max_dim = max(width, height)
            new_image = PILImage.new('RGB', (max_dim, max_dim), color=(0, 0, 0))
            paste_x = (max_dim - width) // 2
            paste_y = (max_dim - height) // 2
            new_image.paste(image, (paste_x, paste_y))
            return new_image.resize((self.target_size, self.target_size), PILImage.LANCZOS)
        
        else:
            raise ValueError(f"Unknown crop_strategy: {strategy}")
    
    def process_image(self, image_path: str, crop_strategy: str = "pad_square", 
                     use_augmentations: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete image processing pipeline.
        
        Args:
            image_path: Path to image file
            crop_strategy: Cropping strategy to apply
            use_augmentations: Whether to apply data augmentations
            
        Returns:
            Tuple of (processed tensor, crop info dict)
        """
        # Load image
        image = PILImage.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        # Apply cropping strategy
        image = self.apply_crop_strategy(image, crop_strategy)
        
        # Create crop info
        crop_info = {
            "original_width": original_width,
            "original_height": original_height,
            "crop_strategy": crop_strategy,
            "target_size": self.target_size,
            "augmentations_used": use_augmentations
        }
        
        # Apply transforms
        transform = self.get_transform(use_augmentations)
        tensor = transform(image)
        
        return tensor, crop_info


# Legacy function for backward compatibility
def process_image(image_path: str, crop_strategy: str = "pad_square", 
                 target_size: int = 224, use_augmentations: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Legacy function - use ImageProcessor class instead."""
    processor = ImageProcessor(target_size)
    return processor.process_image(image_path, crop_strategy, use_augmentations)