"""
Boundary-aware metrics for lyrics segmentation evaluation.
Measures boundary detection, segment quality, and transition accuracy.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class BoundaryMetrics:
    """Container for boundary detection metrics."""
    boundary_precision: float
    boundary_recall: float 
    boundary_f1: float
    total_boundaries_predicted: int
    total_boundaries_actual: int
    correct_boundaries: int


@dataclass 
class SegmentMetrics:
    """Container for segment-level metrics."""
    complete_segments_detected: float  # Ratio of perfectly detected segments
    partial_segments_detected: float   # Ratio of partially detected segments
    avg_segment_overlap: float         # Average IoU between predicted/actual segments
    avg_segment_length_error: float    # Average error in segment lengths
    total_segments: int
    perfect_segments: int


@dataclass
class TransitionMetrics:
    """Container for transition-specific metrics."""
    verse_to_chorus_accuracy: float    # Accuracy of verse→chorus transitions
    chorus_to_verse_accuracy: float    # Accuracy of chorus→verse transitions  
    verse_to_chorus_count: int         # Number of verse→chorus transitions
    chorus_to_verse_count: int         # Number of chorus→verse transitions
    verse_to_chorus_correct: int       # Correct verse→chorus predictions
    chorus_to_verse_correct: int       # Correct chorus→verse predictions


def detect_boundaries(labels: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
    """
    Detect boundary positions (transitions) in label sequences.
    
    Args:
        labels: Label tensor (batch_size, seq_len) 
        mask: Boolean mask (batch_size, seq_len)
    
    Returns:
        List of boundary positions for each sequence in batch
    """
    boundaries_batch = []
    
    for seq_idx in range(labels.size(0)):
        seq_labels = labels[seq_idx]
        seq_mask = mask[seq_idx]
        
        # Get valid sequence length
        valid_length = seq_mask.sum().item()
        if valid_length <= 1:
            boundaries_batch.append([])
            continue
            
        # Extract valid labels  
        valid_labels = seq_labels[:valid_length]
        
        # Find transitions: where label[i] != label[i+1] 
        boundaries = []
        for i in range(len(valid_labels) - 1):
            if valid_labels[i] != valid_labels[i + 1]:
                boundaries.append(i + 1)  # Boundary after position i
                
        boundaries_batch.append(boundaries)
    
    return boundaries_batch


def detect_segments(labels: torch.Tensor, mask: torch.Tensor) -> List[List[Tuple[int, int, int]]]:
    """
    Detect segments (contiguous sections of same label) in sequences.
    
    Args:
        labels: Label tensor (batch_size, seq_len)
        mask: Boolean mask (batch_size, seq_len)
        
    Returns:
        List of segments for each sequence: [(start, end, label), ...]
    """
    segments_batch = []
    
    for seq_idx in range(labels.size(0)):
        seq_labels = labels[seq_idx]
        seq_mask = mask[seq_idx]
        
        # Get valid sequence length
        valid_length = seq_mask.sum().item()
        if valid_length == 0:
            segments_batch.append([])
            continue
            
        # Extract valid labels
        valid_labels = seq_labels[:valid_length].tolist()
        
        # Find contiguous segments
        segments = []
        if valid_labels:
            current_label = valid_labels[0]
            start_pos = 0
            
            for i in range(1, len(valid_labels)):
                if valid_labels[i] != current_label:
                    # End of current segment
                    segments.append((start_pos, i - 1, current_label))
                    # Start new segment
                    current_label = valid_labels[i]
                    start_pos = i
            
            # Add final segment
            segments.append((start_pos, len(valid_labels) - 1, current_label))
        
        segments_batch.append(segments)
    
    return segments_batch


def compute_boundary_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor, 
    mask: torch.Tensor
) -> BoundaryMetrics:
    """
    Compute boundary detection metrics.
    
    Args:
        predictions: Predicted labels (batch_size, seq_len)
        targets: True labels (batch_size, seq_len)
        mask: Boolean mask (batch_size, seq_len)
        
    Returns:
        BoundaryMetrics with precision, recall, F1
    """
    # Detect boundaries in predictions and targets
    pred_boundaries_batch = detect_boundaries(predictions, mask)
    true_boundaries_batch = detect_boundaries(targets, mask)
    
    total_pred_boundaries = 0
    total_true_boundaries = 0
    correct_boundaries = 0
    
    for pred_boundaries, true_boundaries in zip(pred_boundaries_batch, true_boundaries_batch):
        total_pred_boundaries += len(pred_boundaries)
        total_true_boundaries += len(true_boundaries)
        
        # Count correct boundaries (exact position match)
        pred_set = set(pred_boundaries)
        true_set = set(true_boundaries)
        correct_boundaries += len(pred_set & true_set)
    
    # Compute metrics
    precision = correct_boundaries / total_pred_boundaries if total_pred_boundaries > 0 else 0.0
    recall = correct_boundaries / total_true_boundaries if total_true_boundaries > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return BoundaryMetrics(
        boundary_precision=precision,
        boundary_recall=recall,
        boundary_f1=f1,
        total_boundaries_predicted=total_pred_boundaries,
        total_boundaries_actual=total_true_boundaries,
        correct_boundaries=correct_boundaries
    )


def compute_segment_overlap(pred_segment: Tuple[int, int, int], true_segment: Tuple[int, int, int]) -> float:
    """
    Compute IoU (Intersection over Union) between two segments.
    
    Args:
        pred_segment: (start, end, label) for predicted segment
        true_segment: (start, end, label) for true segment
        
    Returns:
        IoU score (0.0 to 1.0)
    """
    pred_start, pred_end, pred_label = pred_segment
    true_start, true_end, true_label = true_segment
    
    # Only compute overlap for same label
    if pred_label != true_label:
        return 0.0
    
    # Compute intersection
    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    
    if intersection_start <= intersection_end:
        intersection = intersection_end - intersection_start + 1
    else:
        intersection = 0
    
    # Compute union
    pred_length = pred_end - pred_start + 1
    true_length = true_end - true_start + 1
    union = pred_length + true_length - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_segment_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> SegmentMetrics:
    """
    Compute segment-level detection metrics.
    
    Args:
        predictions: Predicted labels (batch_size, seq_len)
        targets: True labels (batch_size, seq_len) 
        mask: Boolean mask (batch_size, seq_len)
        
    Returns:
        SegmentMetrics with segment detection quality
    """
    pred_segments_batch = detect_segments(predictions, mask)
    true_segments_batch = detect_segments(targets, mask)
    
    total_segments = 0
    perfect_segments = 0
    partial_segments = 0
    all_overlaps = []
    all_length_errors = []
    
    for pred_segments, true_segments in zip(pred_segments_batch, true_segments_batch):
        total_segments += len(true_segments)
        
        # For each true segment, find best matching predicted segment
        for true_seg in true_segments:
            best_overlap = 0.0
            best_pred_seg = None
            
            # Find predicted segment with highest overlap
            for pred_seg in pred_segments:
                overlap = compute_segment_overlap(pred_seg, true_seg)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pred_seg = pred_seg
            
            all_overlaps.append(best_overlap)
            
            # Count perfect vs partial segments
            if best_overlap >= 0.99:  # Nearly perfect match
                perfect_segments += 1
            elif best_overlap >= 0.5:  # Significant overlap
                partial_segments += 1
                
            # Compute length error if there's a match
            if best_pred_seg is not None:
                true_length = true_seg[1] - true_seg[0] + 1
                pred_length = best_pred_seg[1] - best_pred_seg[0] + 1
                length_error = abs(true_length - pred_length) / true_length
                all_length_errors.append(length_error)
    
    # Compute metrics
    complete_ratio = perfect_segments / total_segments if total_segments > 0 else 0.0
    partial_ratio = partial_segments / total_segments if total_segments > 0 else 0.0
    avg_overlap = np.mean(all_overlaps) if all_overlaps else 0.0
    avg_length_error = np.mean(all_length_errors) if all_length_errors else 0.0
    
    return SegmentMetrics(
        complete_segments_detected=complete_ratio,
        partial_segments_detected=partial_ratio,
        avg_segment_overlap=avg_overlap,
        avg_segment_length_error=avg_length_error,
        total_segments=total_segments,
        perfect_segments=perfect_segments
    )


def compute_transition_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> TransitionMetrics:
    """
    Compute transition-specific accuracy metrics.
    
    Args:
        predictions: Predicted labels (batch_size, seq_len)
        targets: True labels (batch_size, seq_len)
        mask: Boolean mask (batch_size, seq_len)
        
    Returns:
        TransitionMetrics with transition-specific accuracies
    """
    v2c_total = 0  # verse to chorus transitions
    c2v_total = 0  # chorus to verse transitions  
    v2c_correct = 0
    c2v_correct = 0
    
    for seq_idx in range(targets.size(0)):
        seq_targets = targets[seq_idx]
        seq_preds = predictions[seq_idx]
        seq_mask = mask[seq_idx]
        
        # Get valid sequence length
        valid_length = seq_mask.sum().item()
        if valid_length <= 1:
            continue
            
        # Check each transition position
        for i in range(valid_length - 1):
            true_curr = seq_targets[i].item()
            true_next = seq_targets[i + 1].item()
            pred_curr = seq_preds[i].item()
            pred_next = seq_preds[i + 1].item()
            
            # Identify transition type in ground truth
            if true_curr == 0 and true_next == 1:  # verse → chorus
                v2c_total += 1
                if pred_curr == 0 and pred_next == 1:
                    v2c_correct += 1
                    
            elif true_curr == 1 and true_next == 0:  # chorus → verse
                c2v_total += 1
                if pred_curr == 1 and pred_next == 0:
                    c2v_correct += 1
    
    # Compute accuracies
    v2c_accuracy = v2c_correct / v2c_total if v2c_total > 0 else 0.0
    c2v_accuracy = c2v_correct / c2v_total if c2v_total > 0 else 0.0
    
    return TransitionMetrics(
        verse_to_chorus_accuracy=v2c_accuracy,
        chorus_to_verse_accuracy=c2v_accuracy,
        verse_to_chorus_count=v2c_total,
        chorus_to_verse_count=c2v_total,
        verse_to_chorus_correct=v2c_correct,
        chorus_to_verse_correct=c2v_correct
    )


def format_boundary_metrics_report(
    boundary_metrics: BoundaryMetrics,
    segment_metrics: SegmentMetrics,
    transition_metrics: TransitionMetrics,
    indent: str = ""
) -> str:
    """
    Format boundary metrics into a readable report.
    
    Args:
        boundary_metrics: Boundary detection results
        segment_metrics: Segment detection results  
        transition_metrics: Transition accuracy results
        indent: String to prepend to each line
        
    Returns:
        Formatted metrics report
    """
    lines = []
    
    # Boundary Detection Metrics
    lines.append(f"{indent}📏 Boundary Detection:")
    lines.append(f"{indent}   Precision: {boundary_metrics.boundary_precision:.3f}")
    lines.append(f"{indent}   Recall: {boundary_metrics.boundary_recall:.3f}")
    lines.append(f"{indent}   F1: {boundary_metrics.boundary_f1:.3f}")
    lines.append(f"{indent}   Boundaries: {boundary_metrics.correct_boundaries}/{boundary_metrics.total_boundaries_actual}")
    
    # Segment Detection Metrics
    lines.append(f"{indent}🎯 Segment Detection:")
    lines.append(f"{indent}   Complete segments: {segment_metrics.complete_segments_detected:.1%}")
    lines.append(f"{indent}   Partial segments: {segment_metrics.partial_segments_detected:.1%}")
    lines.append(f"{indent}   Avg overlap (IoU): {segment_metrics.avg_segment_overlap:.3f}")
    lines.append(f"{indent}   Avg length error: {segment_metrics.avg_segment_length_error:.1%}")
    
    # Transition Accuracy
    lines.append(f"{indent}🔄 Transition Accuracy:")
    lines.append(f"{indent}   Verse→Chorus: {transition_metrics.verse_to_chorus_accuracy:.1%} ({transition_metrics.verse_to_chorus_correct}/{transition_metrics.verse_to_chorus_count})")
    lines.append(f"{indent}   Chorus→Verse: {transition_metrics.chorus_to_verse_accuracy:.1%} ({transition_metrics.chorus_to_verse_correct}/{transition_metrics.chorus_to_verse_count})")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the boundary metrics
    print("🧪 Testing boundary metrics...")
    
    # Create test data: Simple sequence with one verse→chorus transition
    batch_size, seq_len = 2, 8
    
    # True labels: verse(0), verse(0), chorus(1), chorus(1), verse(0), verse(0), chorus(1), chorus(1)
    targets = torch.tensor([
        [0, 0, 1, 1, 0, 0, 1, 1],  # 3 boundaries: at pos 2, 4, 6
        [1, 1, 0, 0, 1, 1, 0, 0]   # 3 boundaries: at pos 2, 4, 6
    ], dtype=torch.long)
    
    # Predicted labels: mostly correct with some errors
    predictions = torch.tensor([
        [0, 0, 1, 1, 0, 1, 1, 1],  # Missed boundary at pos 5, extra at pos 5
        [1, 1, 0, 0, 1, 1, 0, 1]   # Missed boundary at pos 7
    ], dtype=torch.long)
    
    mask = torch.ones_like(targets, dtype=torch.bool)
    
    # Test boundary detection
    boundary_metrics = compute_boundary_metrics(predictions, targets, mask)
    print(f"✅ Boundary metrics computed:")
    print(f"   Precision: {boundary_metrics.boundary_precision:.3f}")
    print(f"   Recall: {boundary_metrics.boundary_recall:.3f}")  
    print(f"   F1: {boundary_metrics.boundary_f1:.3f}")
    
    # Test segment detection
    segment_metrics = compute_segment_metrics(predictions, targets, mask)
    print(f"✅ Segment metrics computed:")
    print(f"   Complete segments: {segment_metrics.complete_segments_detected:.1%}")
    print(f"   Avg overlap: {segment_metrics.avg_segment_overlap:.3f}")
    
    # Test transition metrics  
    transition_metrics = compute_transition_metrics(predictions, targets, mask)
    print(f"✅ Transition metrics computed:")
    print(f"   V→C accuracy: {transition_metrics.verse_to_chorus_accuracy:.1%}")
    print(f"   C→V accuracy: {transition_metrics.chorus_to_verse_accuracy:.1%}")
    
    # Test full report
    print(f"\n📊 Full Report:")
    report = format_boundary_metrics_report(boundary_metrics, segment_metrics, transition_metrics, indent="   ")
    print(report)
    
    print("\n✅ Boundary metrics test completed!")
