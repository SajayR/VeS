#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from training import create_trainer_from_config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train VeS model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/train_config.yaml",
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--cached-features-path",
        type=str,
        default=None,
        help="Path to cached visual features directory"
    )
    parser.add_argument(
        "--no-cached-features",
        action="store_true",
        help="Disable cached visual features and compute from images"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true", 
        help="Start training from scratch, ignoring existing checkpoints"
    )
    
    args = parser.parse_args()
    
    # Determine cached features usage
    use_cached_features = not args.no_cached_features
    cached_features_path = args.cached_features_path
    
    # Load configuration and create trainer
    print(f"Loading configuration from: {args.config}")
    trainer = create_trainer_from_config(
        config_path=args.config,
        use_cached_visual_features=use_cached_features,
        cached_features_base_path=cached_features_path
    )
    
    print("VeS Trainer initialized successfully")
    
    # Auto-resume from checkpoint unless disabled
    if not args.no_resume:
        resumed = trainer.auto_resume_if_available()
        if resumed:
            print("✓ Resumed training from checkpoint")
        else:
            print("✓ Starting fresh training")
    else:
        print("✓ Starting fresh training (resume disabled)")
    
    # Start training
    try:
        trainer.train()
        print("🎉 Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()