"""
Main VeS trainer class.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
import numpy as np
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
import bitsandbytes as bnb
from typing import Dict, Any, Optional
import gc

from models import VeS
from evaluation import RetrievalEvaluator
from visualization import VeSVisualizer
from .config_loader import ConfigLoader
from .checkpoint_manager import CheckpointManager
from .data_manager import DataManager


class VeSTrainer:
    """
    Main trainer class for the VeS model.
    
    Handles model initialization, training loop, evaluation, and logging.
    """

    def __init__(self, config: Dict[str, Any], use_cached_visual_features: bool = False, 
                 cached_features_base_path: Optional[str] = None):
        """
        Initialize VeS trainer.
        
        Args:
            config: Training configuration dictionary
            use_cached_visual_features: Whether to use pre-computed visual features
            cached_features_base_path: Path to cached visual features
        """
        self.config = config
        self.use_cached_visual_features = use_cached_visual_features
        self.cached_features_base_path = cached_features_base_path

        # Extract configuration sections
        self.cfg_train = config.get("training", {})
        self.cfg_model = config.get("model", {})
        self.cfg_wandb = config.get("wandb", {})
        self.cfg_logging = config.get("logging", {})

        # Setup core components
        self._setup_device_and_paths()
        self._setup_logging()
        self._setup_data_manager()
        self._setup_model()
        self._setup_optimizer_and_scheduler()
        self._setup_checkpoint_manager()
        self._setup_evaluation()
        self._setup_visualization()
        self._setup_wandb()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.epoch_step = 0
        self.best_loss = float("inf")

    def _setup_device_and_paths(self):
        """Setup device and output directories."""
        self.device = torch.device(
            self.cfg_train.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.output_dir = Path(self.cfg_train.get("output_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training hyperparameters
        self.batch_size = self.cfg_train.get("batch_size", 64)
        self.num_epochs = self.cfg_train.get("num_epochs", 1)
        self.gradient_accumulation = self.cfg_train.get("gradient_accumulation_steps", 1)
        self.checkpoint_every_steps = self.cfg_train.get("checkpoint_every_steps", 2000)
        self.learning_rate = self.cfg_train.get("learning_rate", 1e-4)
        self.visualize_every_steps = self.cfg_train.get("viz_every_steps", 10)
        self.eval_every_steps = self.cfg_train.get("eval_every_steps", 20000)

    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            filename=str(self.output_dir / self.cfg_logging.get("log_file", "training.log")),
            level=getattr(logging, self.cfg_logging.get("level", "INFO")),
            format="%(asctime)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _setup_data_manager(self):
        """Setup data manager."""
        self.data_manager = DataManager(self.config, self.cached_features_base_path)
        self.steps_per_epoch = self.data_manager.get_steps_per_epoch()

    def _setup_model(self):
        """Setup VeS model."""
        self.model = VeS(
            loss_type=self.cfg_model.get("loss_type", "dense"),
            use_cached_visual_features=self.use_cached_visual_features,
            embedding_dim=self.cfg_model.get("embedding_dim", 256),
            hubert_name=self.cfg_model.get("hubert_name", "ntu-spml/distilhubert"),
            device="auto"
        ).to(self.device)
        
        # Optional: compile model for optimization
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        self.model.train()

    def _setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        optimizer_type = self.cfg_train.get("optimizer", "adam8bit")
        
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=float(self.learning_rate), 
                fused=True
            )
        elif optimizer_type == "adam8bit":
            self.optimizer = bnb.optim.Adam8bit(
                self.model.parameters(), 
                lr=float(self.learning_rate)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Learning rate scheduler
        total_steps = self.steps_per_epoch * self.num_epochs // self.gradient_accumulation
        warmup_steps = int(self.cfg_train.get("warmup_ratio", 0.1) * total_steps)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
            last_epoch=-1
        )
        
        print(f"Scheduler setup: {total_steps} total steps, {warmup_steps} warmup steps")

    def _setup_checkpoint_manager(self):
        """Setup checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(str(self.output_dir), self.logger)

    def _setup_evaluation(self):
        """Setup evaluation components."""
        val_dataloader = self.data_manager.get_validation_dataloader()
        self.retrieval_evaluator = RetrievalEvaluator(
            val_dataloader,
            device=str(self.device),
            batch_size=self.cfg_train.get("eval_batch_size", 32),
            logger=self.logger,
            use_cached_embeddings=False,
            cache_dir=self.output_dir / "eval_cache"
        )

    def _setup_visualization(self):
        """Setup visualization."""
        REDUCTION = 2  # From audio encoder
        self.visualizer = VeSVisualizer(
            out_dir=self.output_dir / "viz",
            token_hz=25,
            max_samples_per_call=20,
            reduction=REDUCTION,
        )

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        self.use_wandb = self.cfg_wandb.get("enabled", False)
        if self.use_wandb:
            config_to_log = {
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "gradient_accumulation_steps": self.gradient_accumulation,
                "device": str(self.device),
                "model_config": self.cfg_model,
                "data_seed": self.data_manager.data_seed,
            }
            
            wandb.init(
                project=self.cfg_wandb.get("project", "ves-training"),
                name=self.cfg_wandb.get("name"),
                tags=self.cfg_wandb.get("tags", []),
                notes=self.cfg_wandb.get("notes", ""),
                config=config_to_log,
            )
            
            if self.cfg_wandb.get("watch_model", False):
                wandb.watch(self.model, log="all", log_freq=self.cfg_wandb.get("log_freq", 10))
            
            self.logger.info("Initialized wandb logging")

    def compute_gradient_norm(self) -> tuple[float, int]:
        """Compute gradient norm for monitoring training stability."""
        total_norm = 0.0
        param_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2) if param_count > 0 else 0.0
        return total_norm, param_count

    def auto_resume_if_available(self) -> bool:
        """Attempt to resume from latest checkpoint."""
        state_info = self.checkpoint_manager.auto_resume_if_available(
            self.model, self.optimizer, self.scheduler, 
            self.data_manager.data_generator, self.device
        )
        
        if state_info:
            self.global_step = state_info["global_step"]
            self.current_epoch = state_info["current_epoch"]
            self.epoch_step = state_info["epoch_step"]
            self.best_loss = state_info["best_loss"]
            return True
        return False

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Mean loss for the epoch
        """
        epoch_losses = []
        dataloader = self.data_manager.get_training_dataloader(epoch)
        
        # Handle mid-epoch resumption
        steps_to_skip = 0
        if epoch == self.current_epoch and self.epoch_step > 0:
            steps_to_skip = min(self.epoch_step, 40000)  # Safety limit
            print(f"Resuming from epoch {epoch}, skipping first {steps_to_skip} steps")
        else:
            self.epoch_step = 0

        pbar = tqdm(enumerate(dataloader), total=self.steps_per_epoch, desc=f"Epoch {epoch}")
        accumulation_counter = 0

        for step, batch in pbar:
            # Skip already processed steps when resuming
            if step < steps_to_skip:
                pbar.set_postfix({"status": f"Skipping step {step}/{steps_to_skip-1}"})
                continue
                
            if step >= self.steps_per_epoch:
                break

            # Forward pass
            loss = self._forward_step(batch)
            loss = loss / self.gradient_accumulation
            loss.backward()
            
            accumulation_counter += 1

            # Optimizer step
            if accumulation_counter % self.gradient_accumulation == 0:
                grad_norm, param_count = self.compute_gradient_norm()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Log gradient norm
                if self.use_wandb and self.global_step % self.cfg_wandb.get("log_freq", 10) == 0:
                    wandb.log({
                        "gradients/grad_norm": grad_norm,
                        "gradients/param_count": param_count,
                    }, step=self.global_step)

            # Periodic tasks
            self._handle_periodic_tasks(batch, step, epoch)
            
            # Logging
            loss_val = loss.item() * self.gradient_accumulation
            epoch_losses.append(loss_val)
            pbar.set_postfix({"loss": f"{loss_val:.4f}"})
            
            self.global_step += 1
            self.epoch_step = step + 1

        return float(np.mean(epoch_losses)) if epoch_losses else 0.0

    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Execute forward pass for a batch."""
        audio = batch["audio"].to(self.device, non_blocking=True)
        attention_mask = batch["audio_attention_mask"].to(self.device, non_blocking=True)
        
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            if "cached_visual_features" in batch:
                cached_features = batch["cached_visual_features"].to(self.device, non_blocking=False)
                outputs = self.model(audio, attention_mask=attention_mask, cached_visual_features=cached_features)
            else:
                images = batch["image"].to(self.device, non_blocking=True)
                outputs = self.model(audio, images=images, attention_mask=attention_mask)
        
        # Log to wandb
        if self.use_wandb and self.global_step % self.cfg_wandb.get("log_freq", 10) == 0:
            self._log_training_metrics(outputs)
        
        return outputs["loss"]

    def _handle_periodic_tasks(self, batch: Dict[str, torch.Tensor], step: int, epoch: int):
        """Handle visualization, checkpointing, and evaluation."""
        # Visualization
        if self.global_step % self.visualize_every_steps == 0 and self.global_step != 0:
            self._run_visualization(batch)
        
        # Checkpointing
        if self.global_step % self.checkpoint_every_steps == 0 and self.global_step != 0:
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, self.global_step, step + 1, self.global_step,
                self.best_loss, self.data_manager.data_generator,
                self.data_manager.data_seed
            )
        
        # Evaluation
        if self.global_step % self.eval_every_steps == 0 and self.global_step != 0:
            self._run_evaluation()

    def _run_visualization(self, batch: Dict[str, torch.Tensor]):
        """Run batch visualization."""
        with torch.no_grad():
            outputs = self._forward_step(batch)
            matplotlib_figures = self.visualizer.visualize_batch(
                batch, {"loss": outputs}, step=self.global_step
            )
            
            if self.use_wandb and matplotlib_figures:
                wandb_images = {}
                for basename, fig in matplotlib_figures:
                    wandb_images[f"heatmaps/{basename}"] = wandb.Image(fig)
                    plt.close(fig)

    def _run_evaluation(self):
        """Run retrieval evaluation."""
        self.logger.info(f"🔍 Running retrieval eval @ step {self.global_step}...")
        
        eval_results = self.retrieval_evaluator.evaluate(
            self.model,
            max_samples=None,
            log_to_wandb=self.use_wandb,
            global_step=self.global_step,
        )
        
        # Log results
        for agg_name, dirs in eval_results.items():
            a2v = dirs["audio_to_visual"]
            v2a = dirs["visual_to_audio"]
            self.logger.info(
                f"[{agg_name}] A2V-R@1 {a2v.r1:4.1f} V2A-R@1 {v2a.r1:4.1f}"
            )
        
        torch.cuda.empty_cache()
        self.model.train()

    def _log_training_metrics(self, outputs: Dict[str, torch.Tensor]):
        """Log training metrics to wandb."""
        current_lr = self.scheduler.get_last_lr()[0]
        to_log = {
            "train/loss": outputs["loss"].item(),
            "train/learning_rate": current_lr,
            "train/epoch": self.current_epoch,
            "train/step": self.global_step,
            "train/l_nonneg": outputs["l_nonneg"].item(),
            "train/l_tv": outputs["l_tv"].item(),
            "train/logit_scale": self.model.logit_scale.exp().item(),
            "train/clip_sims": outputs["clip_sims"].mean().item(),
            "train/clip_diagonal_sims": outputs["clip_sims"].diagonal().mean().item(),
        }
        
        if "dense_loss" in outputs:
            to_log["train/dense_loss"] = outputs["dense_loss"].item() if isinstance(outputs["dense_loss"], torch.Tensor) else outputs["dense_loss"]
        if "global_loss" in outputs:
            to_log["train/global_loss"] = outputs["global_loss"].item() if isinstance(outputs["global_loss"], torch.Tensor) else outputs["global_loss"]
        
        wandb.log(to_log, step=self.global_step)

    def train(self):
        """Main training loop."""
        print("Starting VeS training...")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            mean_loss = self.train_epoch(epoch)
            
            # End of epoch tasks
            self.current_epoch = epoch + 1
            self.logger.info(f"Epoch {epoch} – mean loss {mean_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "epoch/mean_loss": mean_loss,
                    "epoch/epoch": epoch,
                }, step=self.global_step)
            
            # Save end-of-epoch checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, self.global_step, self.epoch_step, self.global_step,
                self.best_loss, self.data_manager.data_generator,
                self.data_manager.data_seed
            )

        print("Training completed!")
        
        if self.use_wandb:
            wandb.finish()


def create_trainer_from_config(config_path: str, 
                             use_cached_visual_features: bool = False,
                             cached_features_base_path: str = None) -> VeSTrainer:
    """
    Create trainer from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        use_cached_visual_features: Whether to use cached visual features
        cached_features_base_path: Path to cached features
        
    Returns:
        Configured VeSTrainer instance
    """
    config = ConfigLoader.load_config(config_path)
    ConfigLoader.validate_config(config)
    
    # Override config with provided parameters
    if use_cached_visual_features:
        config['model']['use_cached_visual_features'] = True
        if cached_features_base_path:
            config['model']['cached_features_base_path'] = cached_features_base_path
    
    return VeSTrainer(
        config, 
        use_cached_visual_features=use_cached_visual_features,
        cached_features_base_path=cached_features_base_path
    )