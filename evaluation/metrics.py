"""
Evaluation metrics for retrieval tasks.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""
    r1: float
    r5: float
    r10: float
    r50: float
    mean_rank: float
    median_rank: float
    
    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        """
        Convert metrics to dictionary for logging.
        
        Args:
            prefix: Prefix to add to metric names
            
        Returns:
            Dictionary of metric name -> value
        """
        return {
            f"{prefix}R@1": self.r1,
            f"{prefix}R@5": self.r5,
            f"{prefix}R@10": self.r10,
            f"{prefix}R@50": self.r50,
            f"{prefix}mean_rank": self.mean_rank,
            f"{prefix}median_rank": self.median_rank,
        }
    
    def __str__(self) -> str:
        """String representation of metrics."""
        return (f"R@1: {self.r1:.2f}%, R@5: {self.r5:.2f}%, R@10: {self.r10:.2f}%, "
                f"Mean Rank: {self.mean_rank:.1f}, Median Rank: {self.median_rank:.1f}")


def compute_retrieval_metrics_from_ranks(ranks: np.ndarray) -> RetrievalMetrics:
    """
    Compute retrieval metrics from rank array.
    
    Args:
        ranks: Array of ranks for each query
        
    Returns:
        RetrievalMetrics object
    """
    return RetrievalMetrics(
        r1=(ranks <= 1).mean() * 100,
        r5=(ranks <= 5).mean() * 100,
        r10=(ranks <= 10).mean() * 100,
        r50=(ranks <= 50).mean() * 100,
        mean_rank=ranks.mean(),
        median_rank=np.median(ranks)
    )