"""
Checkpoint management utilities.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import random
import numpy as np
from typing import Optional, Dict, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class CheckpointManager:
    """Handles model checkpointing and resumption."""
    
    def __init__(self, output_dir: str, logger: logging.Logger):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory to save checkpoints
            logger: Logger instance
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    def _ckpt_path(self, epoch: int, step: int) -> Path:
        """Generate checkpoint file path."""
        return self.output_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the checkpoint with the highest step number.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        if not self.output_dir.exists():
            return None
        
        checkpoint_files = list(self.output_dir.glob("checkpoint_epoch*_step*.pt"))
        if not checkpoint_files:
            return None
        
        # Extract step numbers and find the maximum
        latest_step = -1
        latest_checkpoint = None
        
        for ckpt_file in checkpoint_files:
            try:
                filename = ckpt_file.stem  # Remove .pt extension
                parts = filename.split('_')
                step_part = next(part for part in parts if part.startswith('step'))
                step_num = int(step_part[4:])  # Remove 'step' prefix
                
                if step_num > latest_step:
                    latest_step = step_num
                    latest_checkpoint = ckpt_file
            except (ValueError, StopIteration, IndexError):
                # Skip malformed filenames
                continue
        
        return latest_checkpoint
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: Optimizer,
                       scheduler: LRScheduler,
                       epoch: int,
                       step: int,
                       epoch_step: int,
                       global_step: int,
                       best_loss: float,
                       data_generator: torch.Generator,
                       data_seed: int) -> None:
        """
        Save model checkpoint with full training state.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            step: Current step
            epoch_step: Step within current epoch
            global_step: Global training step
            best_loss: Best validation loss so far
            data_generator: Data loader generator for reproducibility
            data_seed: Random seed used for data loading
        """
        ckpt = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "epoch_step": epoch_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_loss": best_loss,
            # Save random states for exact resumption
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "data_generator_state": data_generator.get_state(),
            "data_seed": data_seed,
        }
        
        # Atomic save using temporary file
        temp_path = self._ckpt_path(epoch, step).with_suffix(".temp.pt")
        torch.save(ckpt, temp_path)
        temp_path.rename(self._ckpt_path(epoch, step))
        
        self.logger.info(f"Saved checkpoint – epoch {epoch}, step {step}, epoch_step {epoch_step}")
    
    def load_checkpoint(self,
                       checkpoint_path: Path,
                       model: nn.Module,
                       optimizer: Optimizer,
                       scheduler: LRScheduler,
                       data_generator: torch.Generator,
                       device: torch.device) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            data_generator: Data generator to restore state
            device: Device to load checkpoint on
            
        Returns:
            Dictionary with restored training state information
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Restore model and training states
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        
        # Restore random states for deterministic continuation
        self._restore_random_states(ckpt, data_generator)
        
        # Return training state info
        state_info = {
            "global_step": ckpt.get("global_step", ckpt["step"]),
            "current_epoch": ckpt["epoch"],
            "epoch_step": ckpt.get("epoch_step", 0),
            "best_loss": ckpt.get("best_loss", float("inf"))
        }
        
        print(f"Resumed from epoch {state_info['current_epoch']}, "
              f"step {state_info['global_step']}, "
              f"epoch_step {state_info['epoch_step']}")
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return state_info
    
    def _restore_random_states(self, ckpt: Dict[str, Any], data_generator: torch.Generator) -> None:
        """Restore random number generator states."""
        # Restore torch RNG state
        try:
            if "torch_rng_state" in ckpt:
                rng_state = ckpt["torch_rng_state"]
                if hasattr(rng_state, 'cpu'):
                    rng_state = rng_state.cpu()
                if not isinstance(rng_state, torch.ByteTensor):
                    if hasattr(rng_state, 'byte'):
                        rng_state = rng_state.byte()
                torch.set_rng_state(rng_state)
        except Exception as e:
            print(f"Warning: Could not restore torch RNG state: {e}")
        
        # Restore numpy and python RNG states
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        
        # Restore data generator state
        try:
            if "data_generator_state" in ckpt:
                generator_state = ckpt["data_generator_state"]
                if hasattr(generator_state, 'cpu'):
                    generator_state = generator_state.cpu()
                if not isinstance(generator_state, torch.ByteTensor):
                    if hasattr(generator_state, 'byte'):
                        generator_state = generator_state.byte()
                data_generator.set_state(generator_state)
        except Exception as e:
            print(f"Warning: Could not restore data generator RNG state: {e}")
            # Fallback: Reset data generator with the original seed
            data_seed = ckpt.get("data_seed", 42)
            data_generator.manual_seed(data_seed)
    
    def auto_resume_if_available(self,
                                model: nn.Module,
                                optimizer: Optimizer,
                                scheduler: LRScheduler,
                                data_generator: torch.Generator,
                                device: torch.device) -> Optional[Dict[str, Any]]:
        """
        Automatically find and load the latest checkpoint if available.
        
        Returns:
            Training state info if checkpoint was loaded, None otherwise
        """
        latest_checkpoint = self.find_latest_checkpoint()
        if latest_checkpoint:
            print(f"Found latest checkpoint: {latest_checkpoint}")
            return self.load_checkpoint(latest_checkpoint, model, optimizer, scheduler, data_generator, device)
        else:
            print("No existing checkpoints found, starting training from scratch")
            return None