"""
Calibration metrics for confidence evaluation.

Implements Expected Calibration Error (ECE) and reliability curve calculation
to evaluate the quality of model confidence calibration.
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Union, Optional
from dataclasses import dataclass


@dataclass
class ReliabilityDiagram:
    """Container for reliability diagram data."""
    bin_confidences: np.ndarray  # Mean confidence per bin
    bin_accuracies: np.ndarray   # Mean accuracy per bin
    bin_counts: np.ndarray       # Number of samples per bin
    ece: float                   # Expected Calibration Error


def ece(
    probs: Union[torch.Tensor, np.ndarray], 
    labels: Union[torch.Tensor, np.ndarray], 
    n_bins: int = 15,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy across bins.
    Lower values indicate better calibration (0 is perfect).
    
    Args:
        probs: Predicted probabilities (batch_size, seq_len) or (batch_size, seq_len, num_classes)
        labels: True labels (batch_size, seq_len)
        n_bins: Number of bins for confidence histogram
        mask: Boolean mask for valid positions (batch_size, seq_len)
        
    Returns:
        ece_score: Expected Calibration Error (0-1, lower is better)
    """
    # to numpy
    if isinstance(probs, torch.Tensor): probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor): labels = labels.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()

    # --- SHAPE HANDLING ---
    # 3-D: (B, T, C) -> multiclass per token
    if probs.ndim == 3 and probs.shape[-1] > 1:
        confidences = probs.max(axis=-1)         # (B, T)
        predictions = probs.argmax(axis=-1)      # (B, T)
    # 2-D: (N, C) -> multiclass flat
    elif probs.ndim == 2 and probs.shape[-1] > 1:
        confidences = probs.max(axis=-1)         # (N,)
        predictions = probs.argmax(axis=-1)      # (N,)
    # 2-D or 1-D: already confidences (binary)
    else:
        confidences = probs
        predictions = (confidences >= 0.5).astype(np.int32)

    # --- MASK & FLATTEN ---
    if mask is not None and mask.size != 0:
        confidences = confidences[mask]
        predictions = predictions[mask]
        labels = labels[mask]
    else:
        confidences = confidences.reshape(-1)
        predictions = predictions.reshape(-1)
        labels = labels.reshape(-1)

    if confidences.size == 0:
        return 0.0

    # --- BINNING ---
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:-1], right=False)  # 0..n_bins-1

    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_cnt = np.zeros(n_bins)

    for b in range(n_bins):
        m = (bin_ids == b)
        if m.any():
            bin_acc[b] = np.mean(predictions[m] == labels[m])
            bin_conf[b] = np.mean(confidences[m])
            bin_cnt[b] = m.sum()

    ece_score = np.sum(bin_cnt * np.abs(bin_acc - bin_conf)) / max(1, bin_cnt.sum())
    return float(ece_score)


def reliability_curve(
    probs: Union[torch.Tensor, np.ndarray], 
    labels: Union[torch.Tensor, np.ndarray], 
    n_bins: int = 15,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> ReliabilityDiagram:
    """
    Compute reliability curve data (confidence vs. accuracy).
    
    Args:
        probs: Predicted probabilities (batch_size, seq_len) or (batch_size, seq_len, num_classes)
        labels: True labels (batch_size, seq_len)
        n_bins: Number of bins for confidence histogram
        mask: Boolean mask for valid positions (batch_size, seq_len)
        
    Returns:
        ReliabilityDiagram with bin data and ECE score
    """
    # Convert to numpy if needed
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor) and mask is not None:
        mask = mask.detach().cpu().numpy()
    
    # Handle different probability shapes
    if probs.ndim == 3:  # (batch_size, seq_len, num_classes)
        # Get max probability and corresponding class
        confidences = np.max(probs, axis=-1)  # (batch_size, seq_len)
        predictions = np.argmax(probs, axis=-1)  # (batch_size, seq_len)
    elif probs.ndim == 2 and probs.shape[1] > 1:  # (N, num_classes)
        # Get max probability and corresponding class
        confidences = np.max(probs, axis=-1)  # (N,)
        predictions = np.argmax(probs, axis=-1)  # (N,)
    else:  # (batch_size, seq_len) or (N,) - already confidence values
        confidences = probs
        predictions = (confidences >= 0.5).astype(np.int32)  # Binary case
    
    # Apply mask if provided
    if mask is not None:
        confidences = confidences[mask]
        predictions = predictions[mask]
        labels = labels[mask]
    else:
        # Flatten if no mask
        confidences = confidences.flatten()
        predictions = predictions.flatten()
        labels = labels.flatten()
    
    # Skip if no valid samples
    if len(confidences) == 0:
        return ReliabilityDiagram(
            bin_confidences=np.zeros(n_bins),
            bin_accuracies=np.zeros(n_bins),
            bin_counts=np.zeros(n_bins),
            ece=0.0
        )
    
    # Create bins and compute bin statistics
    # Fix edge case: ensure confidence=1.0 falls into the last bin
    bin_edges = np.linspace(0, 1, n_bins+1)
    
    # Use vectorized operations instead of loop for better performance and to avoid scalar comparison
    # First use digitize for all values (returns 1-indexed values)
    bin_indices = np.digitize(confidences, bin_edges[:-1])
    
    # Special case for confidence=1.0 values
    bin_indices[confidences >= 1.0] = n_bins
    
    # Ensure all values are within bounds
    bin_indices = np.clip(bin_indices, 1, n_bins)
    
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Compute accuracy and confidence for each bin
    for bin_idx in range(1, n_bins+1):
        bin_mask = (bin_indices == bin_idx)
        if np.sum(bin_mask) > 0:
            bin_accuracies[bin_idx-1] = np.mean(predictions[bin_mask] == labels[bin_mask])
            bin_confidences[bin_idx-1] = np.mean(confidences[bin_mask])
            bin_counts[bin_idx-1] = np.sum(bin_mask)
    
    # Compute ECE
    ece_score = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / np.sum(bin_counts)
    
    return ReliabilityDiagram(
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        ece=float(ece_score)
    )


def plot_reliability_diagram(
    reliability_data: ReliabilityDiagram,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None
) -> None:
    """
    Plot reliability diagram using matplotlib.
    
    Args:
        reliability_data: ReliabilityDiagram with bin data
        title: Plot title
        save_path: Path to save the figure (if None, just displays)
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Plot reliability curve
        bin_centers = np.linspace(1/(2*len(reliability_data.bin_confidences)), 
                                 1-1/(2*len(reliability_data.bin_confidences)), 
                                 len(reliability_data.bin_confidences))
        
        # Plot points with size proportional to bin count
        max_cnt = max(1, int(reliability_data.bin_counts.max()))
        sizes = 20 + 100 * reliability_data.bin_counts / max_cnt
        ax.scatter(reliability_data.bin_confidences, reliability_data.bin_accuracies, 
                  s=sizes, alpha=0.7, label='Calibration bins')
        
        # Connect points with lines
        ax.plot(reliability_data.bin_confidences, reliability_data.bin_accuracies, 
               'b-', alpha=0.5)
        
        # Add ECE to title
        title = f"{title}\nECE: {reliability_data.ece:.4f}"
        
        # Formatting
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        # Save or display
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
            
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")


def print_reliability_stats(reliability_data: ReliabilityDiagram) -> None:
    """
    Print detailed reliability statistics.
    
    Args:
        reliability_data: ReliabilityDiagram with bin data
    """
    print("\n📊 Reliability Statistics:")
    print(f"   ECE: {reliability_data.ece:.4f}")
    
    # Print bin statistics
    print("\n   Bin Statistics:")
    print("   -------------------------")
    print("   Bin | Confidence | Accuracy | Samples")
    print("   -------------------------")
    
    for i, (conf, acc, count) in enumerate(zip(
        reliability_data.bin_confidences,
        reliability_data.bin_accuracies,
        reliability_data.bin_counts
    )):
        if count > 0:  # Only print bins with samples
            print(f"   {i+1:2d} | {conf:.4f} | {acc:.4f} | {int(count):6d}")
    
    # Print summary statistics
    total_samples = np.sum(reliability_data.bin_counts)
    weighted_acc = np.sum(reliability_data.bin_accuracies * reliability_data.bin_counts) / total_samples
    weighted_conf = np.sum(reliability_data.bin_confidences * reliability_data.bin_counts) / total_samples
    
    print("\n   Summary:")
    print(f"   Total samples: {int(total_samples)}")
    print(f"   Average accuracy: {weighted_acc:.4f}")
    print(f"   Average confidence: {weighted_conf:.4f}")
    print(f"   Confidence-accuracy gap: {weighted_conf - weighted_acc:.4f}")


if __name__ == "__main__":
    print("🧪 Running calibration metrics tests...")
    
    # Create synthetic data for testing
    # Case 1: Perfect calibration
    print("\n🔍 Test Case 1: Perfect Calibration")
    n_samples = 1000
    confidences = np.linspace(0.1, 1.0, n_samples)
    # For perfect calibration, accuracy = confidence
    labels = np.random.binomial(1, confidences)
    
    # Calculate ECE
    perfect_ece = ece(confidences, labels)
    print(f"   ECE (should be close to 0): {perfect_ece:.4f}")
    
    # Calculate reliability curve
    perfect_rel = reliability_curve(confidences, labels)
    print_reliability_stats(perfect_rel)
    
    # Case 2: Overconfident model
    print("\n🔍 Test Case 2: Overconfident Model")
    # Confidences are higher than actual accuracy
    overconf = np.clip(confidences + 0.2, 0, 1.0)
    
    # Calculate ECE
    overconf_ece = ece(overconf, labels)
    print(f"   ECE (should be higher): {overconf_ece:.4f}")
    
    # Calculate reliability curve
    overconf_rel = reliability_curve(overconf, labels)
    print_reliability_stats(overconf_rel)
    
    # Case 3: Underconfident model
    print("\n🔍 Test Case 3: Underconfident Model")
    # Confidences are lower than actual accuracy
    underconf = np.clip(confidences - 0.2, 0, 1.0)
    
    # Calculate ECE
    underconf_ece = ece(underconf, labels)
    print(f"   ECE (should be higher): {underconf_ece:.4f}")
    
    # Calculate reliability curve
    underconf_rel = reliability_curve(underconf, labels)
    print_reliability_stats(underconf_rel)
    
    # Case 4: Edge case with confidence=1.0
    print("\n🔍 Test Case 4: Edge Case with Confidence=1.0")
    edge_conf = np.ones(100)  # All confidences are 1.0
    edge_labels = np.ones(100)  # All correct
    
    # Calculate ECE
    edge_ece = ece(edge_conf, edge_labels)
    print(f"   ECE (should be 0): {edge_ece:.4f}")
    
    # Calculate reliability curve
    edge_rel = reliability_curve(edge_conf, edge_labels)
    print_reliability_stats(edge_rel)
    
    print("\n✅ All calibration metric tests completed!")
    
    # Try to plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        print("\n📊 Generating reliability plots...")
        
        # Plot reliability diagrams
        plot_reliability_diagram(perfect_rel, title="Perfect Calibration")
        plot_reliability_diagram(overconf_rel, title="Overconfident Model")
        plot_reliability_diagram(underconf_rel, title="Underconfident Model")
        
        print("   Plots displayed. Close windows to continue.")
    except ImportError:
        print("\n⚠️ Matplotlib not available. Skipping plot generation.")