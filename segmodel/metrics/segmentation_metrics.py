"""
Standard text segmentation metrics: WindowDiff and Pk.

These metrics are commonly used in text segmentation research for evaluating
boundary detection quality with more tolerance than exact boundary matching.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SegmentationMetrics:
    """Container for text segmentation metrics results."""
    window_diff: float
    pk_metric: float
    num_boundaries_predicted: int
    num_boundaries_actual: int
    num_sequences: int


def compute_window_diff(predictions: List[int], targets: List[int], window_size: int = None) -> float:
    """
    Compute WindowDiff metric for text segmentation.
    
    WindowDiff measures the proportion of windows where the predicted and actual
    segmentations disagree on whether there is a boundary within the window.
    
    Args:
        predictions: Predicted label sequence
        targets: True label sequence
        window_size: Window size (default: half the average segment length)
        
    Returns:
        WindowDiff score (lower is better, 0 is perfect)
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) < 2:
        return 0.0  # Cannot compute for sequences shorter than 2
    
    # Convert labels to binary boundary indicators
    pred_boundaries = [0] + [1 if predictions[i] != predictions[i-1] else 0 for i in range(1, len(predictions))]
    true_boundaries = [0] + [1 if targets[i] != targets[i-1] else 0 for i in range(1, len(targets))]
    
    # Compute window size if not provided
    if window_size is None:
        # Use half the average segment length as window size
        num_segments = sum(true_boundaries) + 1
        if num_segments > 0:
            avg_segment_length = len(targets) / num_segments
            window_size = max(1, int(avg_segment_length / 2))
        else:
            window_size = max(1, min(len(targets) // 4, 8))  # Fallback, capped at 8 lines
    
    # Ensure window size doesn't exceed sequence length
    window_size = min(window_size, len(predictions))
    
    # Count disagreements
    disagreements = 0
    total_windows = 0
    
    for i in range(len(predictions) - window_size + 1):
        # Count boundaries in window for predicted and actual
        pred_boundaries_in_window = sum(pred_boundaries[i:i + window_size])
        true_boundaries_in_window = sum(true_boundaries[i:i + window_size])
        
        # Check if they disagree on presence of boundaries
        pred_has_boundary = pred_boundaries_in_window > 0
        true_has_boundary = true_boundaries_in_window > 0
        
        if pred_has_boundary != true_has_boundary:
            disagreements += 1
        
        total_windows += 1
    
    return disagreements / total_windows if total_windows > 0 else 0.0


def compute_pk_metric(predictions: List[int], targets: List[int], window_size: int = None) -> float:
    """
    Compute Pk metric for text segmentation.
    
    Pk is the probability that a randomly chosen pair of units at distance k
    is inconsistently classified (one in same segment, other in different segment).
    
    Args:
        predictions: Predicted label sequence
        targets: True label sequence
        window_size: Distance k (default: half the average segment length)
        
    Returns:
        Pk score (lower is better, 0 is perfect)
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if len(predictions) < 2:
        return 0.0
    
    # Compute window size if not provided
    if window_size is None:
        # Count segments in true segmentation
        num_boundaries = sum(1 for i in range(1, len(targets)) if targets[i] != targets[i-1])
        num_segments = num_boundaries + 1
        if num_segments > 0:
            avg_segment_length = len(targets) / num_segments
            window_size = max(1, int(avg_segment_length / 2))
        else:
            window_size = max(1, min(len(targets) // 4, 8))  # Fallback, capped at 8 lines
    
    # Ensure window size doesn't exceed sequence length
    window_size = min(window_size, len(predictions) - 1)
    
    # Count inconsistencies
    inconsistencies = 0
    total_pairs = 0
    
    for i in range(len(predictions) - window_size):
        # Check positions i and i+window_size
        pred_same_segment = predictions[i] == predictions[i + window_size]
        true_same_segment = targets[i] == targets[i + window_size]
        
        # Inconsistency if they disagree
        if pred_same_segment != true_same_segment:
            inconsistencies += 1
        
        total_pairs += 1
    
    return inconsistencies / total_pairs if total_pairs > 0 else 0.0


def compute_segmentation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> SegmentationMetrics:
    """
    Compute text segmentation metrics for batch of sequences.
    
    NOTE: Current implementation processes sequences individually for clarity.
    For large batches, could be optimized with vectorized operations:
    
    def compute_segmentation_metrics_vectorized(predictions, targets, mask):
        # Vectorized boundary detection across entire batch
        # Only implement if current version becomes a bottleneck
    
    Args:
        predictions: Predicted labels (batch_size, seq_len)
        targets: True labels (batch_size, seq_len)
        mask: Valid positions mask (batch_size, seq_len)
        
    Returns:
        SegmentationMetrics with WindowDiff and Pk scores
    """
    batch_size = predictions.size(0)
    
    window_diff_scores = []
    pk_scores = []
    total_pred_boundaries = 0
    total_true_boundaries = 0
    
    for seq_idx in range(batch_size):
        # Get valid sequence (remove padding)
        seq_length = mask[seq_idx].sum().item()
        if seq_length < 2:
            continue
        
        pred_seq = predictions[seq_idx][:seq_length].cpu().numpy().tolist()
        true_seq = targets[seq_idx][:seq_length].cpu().numpy().tolist()
        
        # Compute metrics for this sequence
        try:
            window_diff = compute_window_diff(pred_seq, true_seq)
            pk_metric = compute_pk_metric(pred_seq, true_seq)
            
            window_diff_scores.append(window_diff)
            pk_scores.append(pk_metric)
            
            # Count boundaries
            pred_boundaries = sum(1 for i in range(1, len(pred_seq)) if pred_seq[i] != pred_seq[i-1])
            true_boundaries = sum(1 for i in range(1, len(true_seq)) if true_seq[i] != true_seq[i-1])
            
            total_pred_boundaries += pred_boundaries
            total_true_boundaries += true_boundaries
            
        except Exception as e:
            # Skip sequences with errors
            print(f"Warning: Error computing segmentation metrics for sequence {seq_idx}: {e}")
            continue
    
    # Aggregate results
    avg_window_diff = np.mean(window_diff_scores) if window_diff_scores else 1.0
    avg_pk = np.mean(pk_scores) if pk_scores else 1.0
    
    return SegmentationMetrics(
        window_diff=avg_window_diff,
        pk_metric=avg_pk,
        num_boundaries_predicted=total_pred_boundaries,
        num_boundaries_actual=total_true_boundaries,
        num_sequences=len(window_diff_scores)
    )


def format_segmentation_metrics_report(metrics: SegmentationMetrics) -> str:
    """
    Format segmentation metrics for human-readable display.
    
    Args:
        metrics: SegmentationMetrics object
        
    Returns:
        Formatted report string
    """
    return (
        f"📏 Text Segmentation Metrics:\n"
        f"   WindowDiff: {metrics.window_diff:.3f} (lower better)\n"
        f"   Pk: {metrics.pk_metric:.3f} (lower better)\n"
        f"   Boundaries: {metrics.num_boundaries_predicted} pred, {metrics.num_boundaries_actual} actual\n"
        f"   Sequences: {metrics.num_sequences}"
    )


# Test functions for validation
if __name__ == "__main__":
    # Test WindowDiff and Pk with simple examples
    print("🧪 Testing segmentation metrics...")
    
    # Perfect match
    pred = [0, 0, 1, 1, 0, 0]
    true = [0, 0, 1, 1, 0, 0]
    wd = compute_window_diff(pred, true)
    pk = compute_pk_metric(pred, true)
    print(f"Perfect match - WindowDiff: {wd:.3f}, Pk: {pk:.3f}")
    
    # Off by one boundary
    pred = [0, 0, 0, 1, 1, 0]
    true = [0, 0, 1, 1, 0, 0]
    wd = compute_window_diff(pred, true)
    pk = compute_pk_metric(pred, true)
    print(f"Off by one - WindowDiff: {wd:.3f}, Pk: {pk:.3f}")
    
    # Completely wrong
    pred = [0, 1, 0, 1, 0, 1]
    true = [0, 0, 0, 1, 1, 1]
    wd = compute_window_diff(pred, true)
    pk = compute_pk_metric(pred, true)
    print(f"Completely wrong - WindowDiff: {wd:.3f}, Pk: {pk:.3f}")
    
    print("✅ Segmentation metrics tests completed")
