"""
Data loading and management utilities.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from typing import Dict, Any
from data import VAAPairedDataset


class DataManager:
    """Handles data loading and deterministic shuffling."""
    
    def __init__(self, config: Dict[str, Any], cached_features_base_path: str = None):
        """
        Initialize data manager.
        
        Args:
            config: Training configuration
            cached_features_base_path: Path to cached visual features
        """
        self.config = config
        self.cached_features_base_path = cached_features_base_path
        
        # Training config
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training'].get('num_workers', 8)
        self.data_seed = config['training'].get('data_seed', 42)
        self.eval_batch_size = config['training'].get('eval_batch_size', 32)
        
        # Set up datasets and data loaders
        self._setup_datasets()
        self._setup_data_generator()
    
    def _setup_datasets(self):
        """Initialize training and validation datasets."""
        # Training dataset
        self.train_dataset = VAAPairedDataset(
            cached_features_base_path=self.cached_features_base_path
        )
        
        # Validation dataset
        self.val_dataset = VAAPairedDataset(
            is_validation=True,
            cached_features_base_path=self.cached_features_base_path
        )
        
        # Validation dataloader (fixed, no shuffling needed)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )
    
    def _setup_data_generator(self):
        """Set up deterministic data generator."""
        self.data_generator = torch.Generator()
        self.data_generator.manual_seed(self.data_seed)
        self.set_seeds(self.data_seed)
    
    def set_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"Set all random seeds to {seed} for deterministic training")
    
    def get_training_dataloader(self, epoch: int = None) -> DataLoader:
        """
        Get training dataloader with epoch-specific deterministic shuffling.
        
        Args:
            epoch: Current epoch number for deterministic shuffling
            
        Returns:
            DataLoader for training
        """
        # Set epoch-specific seed for deterministic but varied shuffling
        if epoch is not None:
            epoch_seed = self.data_seed + epoch * 1000
            self.data_generator.manual_seed(epoch_seed)
            print(f"Epoch {epoch}: Using deterministic seed {epoch_seed} for data shuffling")
            
            def epoch_worker_init_fn(worker_id):
                worker_seed = epoch_seed + worker_id
                np.random.seed(worker_seed)
                random.seed(worker_seed)
                torch.manual_seed(worker_seed)
        else:
            def epoch_worker_init_fn(worker_id):
                worker_seed = self.data_seed + worker_id
                np.random.seed(worker_seed)
                random.seed(worker_seed)
                torch.manual_seed(worker_seed)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
            shuffle=True,
            generator=self.data_generator,
            worker_init_fn=epoch_worker_init_fn,
        )
    
    def get_validation_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return self.val_dataloader
    
    def get_steps_per_epoch(self) -> int:
        """Get number of training steps per epoch."""
        return len(self.get_training_dataloader())