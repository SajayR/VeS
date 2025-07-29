"""
Data processing and dataset utilities.
"""

from .dataset import VAAPairedDataset
from .audio_processing import AudioProcessor
from .transforms import ImageProcessor, process_image

__all__ = [
    'VAAPairedDataset',
    'AudioProcessor', 
    'ImageProcessor',
    'process_image'  # Legacy function
]