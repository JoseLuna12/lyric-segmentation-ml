"""
Cross-entropy loss with anti-collapse features.
Implements label smoothing and class weighting to prevent overconfidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class CrossEntropyWithLabelSmoothing(nn.Module):
    """
    Cross-entropy loss with label smoothing to prevent overconfidence.
    
    Implements principled weighting under label smoothing and robust handling
    of ignore_index. Based on lessons learned from architecture knowledge 
    about overconfidence collapse.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        label_smoothing: float = 0.2,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        entropy_lambda: float = 0.0
    ):
        """
        Initialize the loss function.
        
        Args:
            num_classes: Number of output classes
            label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.2 = 20% smoothing)
            class_weights: Per-class weights for handling imbalance
            ignore_index: Index to ignore in loss computation (for padding)
            entropy_lambda: Optional entropy regularization weight (0.0 = disabled)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.entropy_lambda = entropy_lambda
        
        # Store class weights as buffer (moves with model to GPU)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss with principled label smoothing and class weighting.
        
        Args:
            logits: Model predictions (batch_size, seq_len, num_classes)
            targets: True labels (batch_size, seq_len)
            mask: Boolean mask for valid positions (batch_size, seq_len)
            
        Returns:
            loss: Scalar loss value
        """
        batch_size, seq_len, num_classes = logits.shape
        
        # Reshape for loss computation (use reshape instead of view for safety)
        logits_flat = logits.reshape(-1, num_classes)  # (N, num_classes)
        targets_flat = targets.reshape(-1)  # (N,)
        mask_flat = mask.reshape(-1)  # (N,)
        
        # Unified valid mask: combines ignore_index and explicit mask
        valid = (targets_flat != self.ignore_index) & mask_flat
        
        # Handle edge case: no valid tokens
        if valid.sum() == 0:
            return logits.new_zeros((), requires_grad=True)
        
        # Apply label smoothing if enabled
        if self.label_smoothing > 0.0:
            loss = self._compute_label_smoothed_loss(logits_flat, targets_flat, valid)
        else:
            loss = self._compute_standard_loss(logits_flat, targets_flat, valid)
        
        # Optional entropy regularization (anti-peaking escape hatch)
        if self.entropy_lambda > 0.0:
            probs = F.softmax(logits_flat, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
            entropy_bonus = entropy[valid].mean()
            loss = loss - self.entropy_lambda * entropy_bonus
        
        return loss
    
    def _compute_standard_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor
    ) -> torch.Tensor:
        """Compute standard cross-entropy loss with unified masking."""
        # Use CrossEntropyLoss with ignore_index (safer than manual indexing)
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='none'
        )
        
        per_token_loss = loss_fn(logits, targets)  # (N,)
        
        # Average only over valid positions
        num_valid = valid.float().sum().clamp_min(1.0)
        return (per_token_loss * valid.float()).sum() / num_valid
    
    def _compute_label_smoothed_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss with principled weighting.
        
        Uses expected class weights under the smoothed target distribution
        rather than just the hard target's weight.
        """
        N, C = logits.shape
        
        # Build smoothed targets only for valid positions (avoids ignore_index bug)
        smoothed_targets = torch.zeros(N, C, device=logits.device, dtype=logits.dtype)
        
        if valid.sum() > 0:
            # Get valid targets and create smoothed distribution
            valid_targets = targets[valid]  # Only valid targets (no -100)
            
            # One-hot encode valid targets
            num_valid = valid_targets.numel()
            smoothed_valid = torch.zeros(num_valid, C, device=logits.device, dtype=logits.dtype)
            smoothed_valid.scatter_(1, valid_targets.unsqueeze(1), 1.0)
            
            # Apply label smoothing: (1-ε) * one_hot + ε/C
            eps = self.label_smoothing
            smoothed_valid = smoothed_valid * (1.0 - eps) + eps / C
            
            # Place back into full tensor
            smoothed_targets[valid] = smoothed_valid
        
        # Compute log probabilities and loss
        log_probs = F.log_softmax(logits, dim=-1)
        token_loss = -(smoothed_targets * log_probs).sum(dim=-1)  # (N,)
        
        # Apply class weights using expected weight under smoothed distribution
        if self.class_weights is not None:
            # Expected weight = smoothed_distribution @ class_weights
            token_weights = smoothed_targets @ self.class_weights  # (N,)
            token_loss = token_loss * token_weights
        
        # Average over valid positions
        num_valid = valid.float().sum().clamp_min(1.0)
        return token_loss[valid].sum() / num_valid
    
def create_loss_function(
    num_classes: int = 2,
    label_smoothing: float = 0.2,
    class_weights: Optional[torch.Tensor] = None,
    entropy_lambda: float = 0.0
) -> CrossEntropyWithLabelSmoothing:
    """
    Factory function to create improved loss function.
    
    Args:
        num_classes: Number of classes
        label_smoothing: Label smoothing factor
        class_weights: Class weights for imbalance handling
        entropy_lambda: Entropy regularization weight (0.0 = disabled)
        
    Returns:
        Configured loss function with all improvements
    """
    return CrossEntropyWithLabelSmoothing(
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        entropy_lambda=entropy_lambda
    )


# =============================================================================
# TRAINING METRICS AND GUARDRAILS
# =============================================================================

def batch_guardrails(logits: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Compute batch-level guardrail metrics to detect overconfidence.
    
    Based on lessons from architecture knowledge about monitoring training.
    
    Args:
        logits: Model predictions (batch_size, seq_len, num_classes)
        mask: Boolean mask (batch_size, seq_len)
        
    Returns:
        Dictionary of guardrail metrics
    """
    with torch.no_grad():
        # Get probabilities and predictions
        probs = F.softmax(logits, dim=-1)
        max_probs, predictions = torch.max(probs, dim=-1)
        
        # Apply mask
        valid_mask = mask
        valid_predictions = predictions[valid_mask]
        valid_max_probs = max_probs[valid_mask]
        
        if len(valid_predictions) == 0:
            return {
                'chorus_rate': 0.0,
                'max_prob_mean': 0.0,
                'confidence_over_90': 0.0,
                'confidence_over_95': 0.0,
            }
        
        # Compute metrics
        chorus_rate = (valid_predictions == 1).float().mean().item()
        max_prob_mean = valid_max_probs.mean().item()
        conf_over_90 = (valid_max_probs > 0.9).float().mean().item()
        conf_over_95 = (valid_max_probs > 0.95).float().mean().item()
        
        return {
            'chorus_rate': float(chorus_rate),
            'max_prob_mean': float(max_prob_mean),
            'confidence_over_90': float(conf_over_90),
            'confidence_over_95': float(conf_over_95),
        }


def sequence_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Compute F1 scores for sequence labeling.
    
    Args:
        predictions: Predicted labels (batch_size, seq_len)
        targets: True labels (batch_size, seq_len)
        mask: Boolean mask (batch_size, seq_len)
        num_classes: Number of classes
        
    Returns:
        Dictionary with F1 scores
    """
    with torch.no_grad():
        # Get valid predictions and targets
        valid_mask = mask & (targets != -100)
        
        if valid_mask.sum() == 0:
            return {'macro_f1': 0.0, 'verse_f1': 0.0, 'chorus_f1': 0.0}
        
        pred_valid = predictions[valid_mask].cpu().numpy()
        true_valid = targets[valid_mask].cpu().numpy()
        
        # Compute per-class metrics
        f1_scores = {}
        class_names = ['verse', 'chorus'] if num_classes == 2 else [f'class_{i}' for i in range(num_classes)]
        
        for class_idx, class_name in enumerate(class_names):
            # True positives, false positives, false negatives
            tp = ((pred_valid == class_idx) & (true_valid == class_idx)).sum()
            fp = ((pred_valid == class_idx) & (true_valid != class_idx)).sum()
            fn = ((pred_valid != class_idx) & (true_valid == class_idx)).sum()
            
            # F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores[f'{class_name}_f1'] = float(f1)
        
        # Macro F1
        f1_scores['macro_f1'] = float(sum(f1_scores.values()) / len(f1_scores))
        
        return f1_scores


if __name__ == "__main__":
    # Test the loss function
    print("🧪 Testing CrossEntropy with Label Smoothing...")
    
    # Create test data
    batch_size, seq_len, num_classes = 2, 4, 2
    logits = torch.randn(batch_size, seq_len, num_classes)
    targets = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 0]], dtype=torch.long)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[1, -1] = False  # Mask last position
    
    # Test class weights
    class_weights = torch.tensor([1.0, 1.5])  # Slightly favor chorus class
    
    # Create loss functions
    loss_fn_smooth = create_loss_function(
        num_classes=num_classes,
        label_smoothing=0.2,
        class_weights=class_weights
    )
    
    loss_fn_standard = create_loss_function(
        num_classes=num_classes,
        label_smoothing=0.0,
        class_weights=class_weights
    )
    
    # Compute losses
    loss_smooth = loss_fn_smooth(logits, targets, mask)
    loss_standard = loss_fn_standard(logits, targets, mask)
    
    print(f"\n📊 Loss comparison:")
    print(f"   Standard CE loss: {loss_standard:.4f}")
    print(f"   Label-smoothed loss: {loss_smooth:.4f}")
    
    # Test guardrails
    guardrails = batch_guardrails(logits, mask)
    print(f"\n🛡️  Batch guardrails:")
    for metric, value in guardrails.items():
        print(f"   {metric}: {value:.3f}")
    
    # Test F1 computation
    predictions = torch.argmax(logits, dim=-1)
    f1_metrics = sequence_f1_score(predictions, targets, mask)
    print(f"\n📈 F1 scores:")
    for metric, value in f1_metrics.items():
        print(f"   {metric}: {value:.3f}")
    
    print("✅ Loss function test completed!")
