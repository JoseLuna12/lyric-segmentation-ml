# Boundary-Aware Cross-Entropy Loss Implementation

## Overview

This document describes the implementation of the **Boundary-Aware Cross-Entropy Loss**, which evolves the original token-level cross-entropy loss into a segmentation-aware loss function following the established roadmap.

## Roadmap Implementation Status

### ✅ Phase 1 — Baseline Stability (Foundation)
- **Cross-entropy with label smoothing** → combats overconfidence
- **Class weights** → balance chorus vs verse (chorus rate ~0.32)
- **Entropy regularization (`entropy_lambda`)** → avoids confidence collapse

### ✅ Phase 2 — Boundary Awareness
- **Goal**: Encourage the model to care about transition points
- **Implementation**: 
  - Detect ground-truth boundaries: `boundaries = (targets[:, 1:] != targets[:, :-1])`
  - Apply `boundary_weight` (e.g. ×2.0) multiplier to loss at boundary positions
  - **Expected effect**: Improves `val_boundary_f1`, `val_window_diff`, and `val_pk_metric`

### ✅ Phase 3 — Structural Regularization
- **Goal**: Move from token-level to segment-level learning
- **Implementation**: Segment consistency loss that minimizes variance of logits within true segments
- **Weight**: `segment_consistency_lambda = 0.05` (tunable)
- **Expected effect**: Reduces fragmentation, increases `val_complete_segments` and `val_avg_segment_overlap`

### ✅ Phase 4 — Calibration & Confidence Control
- **Goal**: Align probabilities with reality (avoid overconfident wrong boundaries)
- **Implementation**: Confidence penalty for predictions above threshold (default 0.95)
- **Weight**: `conf_penalty_lambda = 0.01` (tunable)
- **Expected effect**: Reduces overconfidence spikes, improves calibration

### 🔮 Phase 5 — Differentiable Segmentation Surrogates (Future)
- **Goal**: Directly approximate `val_window_diff` / `val_pk_metric`
- **Status**: Reserved for future experimentation after validating Phases 2-4

## Usage

### Basic Usage (Backward Compatible)
```python
from segmodel.losses import create_loss_function

# Works exactly like the original loss function
loss_fn = create_loss_function(
    num_classes=2,
    label_smoothing=0.2,
    class_weights=class_weights,
    entropy_lambda=0.01
)

# Forward pass (backward compatible)
loss = loss_fn(logits, targets, mask)
```

### Advanced Usage (With Metrics)
```python
# Get detailed loss component metrics
loss, metrics = loss_fn(logits, targets, mask, return_metrics=True)

print(f"Total loss: {loss:.4f}")
print(f"Boundary loss: {metrics['boundary_loss']:.4f}")
print(f"Consistency loss: {metrics['consistency_loss']:.4f}")
print(f"Confidence penalty: {metrics['confidence_penalty']:.4f}")
```

### Full Configuration
```python
loss_fn = create_loss_function(
    num_classes=2,
    label_smoothing=0.2,
    class_weights=class_weights,
    entropy_lambda=0.01,
    # Phase 2: Boundary awareness
    boundary_weight=2.0,
    # Phase 3: Segment consistency  
    segment_consistency_lambda=0.05,
    # Phase 4: Confidence control
    conf_penalty_lambda=0.01,
    conf_threshold=0.95
)
```

## Key Features

### 1. Boundary Detection
- Automatically detects transition points between different segment types
- Applies higher weight to boundary positions during training
- Encourages the model to focus on segmentation quality

### 2. Segment Consistency
- Encourages predictions within the same true segment to be similar
- Reduces fragmentation and improves segment integrity
- Uses variance minimization within segments

### 3. Confidence Control
- Penalizes overconfident predictions to improve calibration
- Helps prevent the model from being too certain about wrong boundaries
- Configurable confidence threshold

### 4. Backward Compatibility
- Seamlessly replaces existing loss function
- No changes needed in existing training code
- Optional enhanced metrics for monitoring

## Expected Benefits

1. **Improved Boundary Detection**: Higher `val_boundary_f1` scores
2. **Better Segmentation Quality**: Improved `val_window_diff` and `val_pk_metric`
3. **Reduced Fragmentation**: Higher `val_complete_segments` and `val_avg_segment_overlap`
4. **Better Calibration**: More reliable confidence estimates
5. **Stable Training**: Maintains macro F1 stability while improving segmentation

## Monitoring Plan

When training with this loss, monitor these metrics:

### Loss Components
- `base_loss`: Base cross-entropy loss
- `boundary_loss`: Boundary-weighted loss  
- `consistency_loss`: Segment consistency regularization
- `confidence_penalty`: Overconfidence penalty
- `entropy_bonus`: Anti-collapse regularization

### Key Validation Metrics
- `val_macro_f1`: Overall token-level accuracy (should remain stable)
- `val_boundary_f1`: Segmentation accuracy (should improve)
- `val_complete_segments`: Segment integrity (should improve)
- `chorus_rate`: Should stay realistic (~0.32, no collapse)

## Files Modified

1. **`segmodel/losses/boundary_aware_cross_entropy.py`**: New loss implementation
2. **`segmodel/losses/__init__.py`**: Updated exports for backward compatibility

## Testing

The implementation includes comprehensive tests that verify:
- Boundary detection accuracy
- Loss component calculations
- Backward compatibility
- Metric collection
- Different configuration options

Run tests with: `python segmodel/losses/boundary_aware_cross_entropy.py`

## Implementation Notes

### Memory Efficiency
- Boundary detection is computed efficiently using tensor operations
- Segment consistency uses batched processing
- No significant memory overhead compared to original loss

### Numerical Stability
- All computations include proper handling of edge cases
- Gradient flow is preserved through all components
- Robust handling of empty sequences and masked positions

### Hyperparameter Tuning
The loss function provides several tunable hyperparameters:
- `boundary_weight`: How much to emphasize boundary positions (default: 2.0)
- `segment_consistency_lambda`: Weight for consistency regularization (default: 0.05)  
- `conf_penalty_lambda`: Weight for confidence penalty (default: 0.01)
- `conf_threshold`: Confidence threshold for penalty (default: 0.95)

Start with default values and adjust based on validation metrics.
