#!/usr/bin/env python3
"""
Standalone evaluation script for VeS model retrieval performance.
"""

import argparse
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data import VAAPairedDataset
from models import VeS
from evaluation import RetrievalEvaluator


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate VeS model retrieval performance")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--max-samples", 
        type=int, 
        default=500, 
        help="Maximum validation samples to use (None for all)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--use-cached-features", 
        action="store_true", 
        help="Use cached visual features"
    )
    parser.add_argument(
        "--cached-features-path", 
        type=str,
        default="/workspace/cached_features/dinov2_large",
        help="Path to cached visual features"
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="global",
        choices=["dense", "global", "dense_global"],
        help="Loss type used during training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on"
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting VeS retrieval evaluation...")
    
    # Create validation dataset
    print("📊 Loading validation dataset...")
    val_dataset = VAAPairedDataset(
        is_validation=True,
        cached_features_base_path=args.cached_features_path if args.use_cached_features else None
    )
    print(f"✅ Loaded {len(val_dataset)} validation samples")
    
    # Create dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
        shuffle=False,
    )
    
    # Initialize model
    print("🤖 Initializing VeS model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = VeS(
        loss_type=args.loss_type,
        use_cached_visual_features=args.use_cached_features,
        device=args.device
    ).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"📁 Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"✅ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        print("⚠️  No checkpoint provided, using random weights")
    
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Create evaluator
    print("\n🔍 Creating retrieval evaluator...")
    evaluator = RetrievalEvaluator(
        val_dataloader,
        device=str(device),
        batch_size=32,
        use_cached_embeddings=False,  # Don't cache during standalone evaluation
    )
    
    # Run evaluation
    print(f"\n🏃 Running evaluation on {args.max_samples or 'all'} samples...")
    print("=" * 80)
    
    try:
        results = evaluator.evaluate(
            model,
            max_samples=args.max_samples,
            log_to_wandb=False,  # No wandb for standalone evaluation
        )
        
        # Print detailed results
        print_detailed_results(results)
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        print("\n🧹 Cleaned up GPU memory")
    
    return 0


def print_detailed_results(results):
    """Print detailed evaluation results."""
    print("\n" + "="*80)
    print("📊 RETRIEVAL EVALUATION RESULTS")
    print("="*80)
    
    for method, method_results in results.items():
        print(f"\n🔧 {method.upper().replace('_', ' ')} Aggregation:")
        print("-" * 50)
        
        for direction, metrics in method_results.items():
            print(f"\n  📍 {direction.replace('_', ' ').title()}:")
            print(f"     R@1:  {metrics.r1:6.2f}%")
            print(f"     R@5:  {metrics.r5:6.2f}%")
            print(f"     R@10: {metrics.r10:6.2f}%")
            print(f"     R@50: {metrics.r50:6.2f}%")
            print(f"     Mean Rank:   {metrics.mean_rank:6.1f}")
            print(f"     Median Rank: {metrics.median_rank:6.1f}")
    
    # Print averaged metrics
    print("\n" + "="*80)
    print("📈 AVERAGED METRICS (both directions)")
    print("="*80)
    
    for method in results:
        a2v = results[method]['audio_to_visual']
        v2a = results[method]['visual_to_audio']
        print(f"\n🔧 {method.upper().replace('_', ' ')}:")
        print(f"   R@1:  {(a2v.r1 + v2a.r1) / 2:6.2f}%")
        print(f"   R@5:  {(a2v.r5 + v2a.r5) / 2:6.2f}%")
        print(f"   R@10: {(a2v.r10 + v2a.r10) / 2:6.2f}%")


def test_similarity_computation():
    """Quick test to verify similarity computation works correctly."""
    print("\n🧪 Testing similarity computation...")
    
    from evaluation import SimilarityComputer
    
    # Create dummy data
    B, Na, Nv, D = 4, 10, 16*16, 256
    audio_feats = torch.randn(B, Na, D).cuda()
    visual_feats = torch.randn(B, Nv, D).cuda()
    attention_mask = torch.ones(B, Na).cuda()
    attention_mask[:, 7:] = 0  # Mask out last 3 tokens
    
    # Create similarity computer
    sim_computer = SimilarityComputer(device="cuda")
    
    # Test max-mean similarity
    sim_maxmean = sim_computer.compute_max_mean_similarity(
        audio_feats, visual_feats, attention_mask
    )
    print(f"Max-mean similarity shape: {sim_maxmean.shape}")
    print(f"Max-mean similarity range: [{sim_maxmean.min():.3f}, {sim_maxmean.max():.3f}]")
    
    # Test mean-pooled similarity
    sim_mean = sim_computer.compute_mean_pooled_similarity(
        audio_feats, visual_feats, attention_mask
    )
    print(f"Mean-pooled similarity shape: {sim_mean.shape}")
    print(f"Mean-pooled similarity range: [{sim_mean.min():.3f}, {sim_mean.max():.3f}]")
    
    print("✅ Similarity computation test passed!")


if __name__ == "__main__":
    if "--test-sim" in sys.argv:
        test_similarity_computation()
    else:
        exit_code = main()
        sys.exit(exit_code)