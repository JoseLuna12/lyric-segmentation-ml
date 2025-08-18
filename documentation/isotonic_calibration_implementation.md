# Isotonic Calibration Implementation

## Overview

Isotonic calibration has been successfully implemented as the third calibration method alongside temperature scaling and Platt scaling. This provides a non-parametric approach to calibration that can capture complex, non-linear miscalibration patterns.

## What is Isotonic Calibration?

Isotonic calibration learns a **monotone, stepwise-increasing function** that maps confidence scores to empirical accuracy. Unlike temperature scaling (single parameter) or Platt scaling (sigmoid with 2 parameters), isotonic regression can model arbitrary monotone relationships between predicted confidence and actual accuracy.

## When to Use It

### ✅ **Use isotonic when:**
- Temperature/Platt scaling show remaining calibration errors
- Reliability plots show non-linear miscalibration patterns (U-shaped, S-shaped curves)
- You have sufficient validation data (hundreds to thousands of samples)
- You want the most flexible calibration approach

### ⚠️ **Be cautious when:**
- Validation set is very small (< 20 samples) - may not have enough diversity
- Model is already well-calibrated - may not provide significant improvement
- You need fast inference - isotonic requires lookup in fitted points

**Note**: The implementation now uses adaptive thresholds, so datasets with 75+ validation samples should work well.

## Implementation Details

### Class: `IsotonicCalibrator`

```python
calibrator = IsotonicCalibrator(
    out_of_bounds="clip",  # How to handle extrapolation
    min_samples=50         # Adaptive: min(50, dataset_size//2)
)
```

### Key Features

1. **Top-1 Probability Calibration**: Calibrates the maximum probability and rescales others proportionally
2. **Shape Preservation**: Non-top classes maintain their relative proportions
3. **Adaptive Thresholds**: Automatically adjusts requirements for small datasets (like 75 validation songs)
4. **Robust Fallback**: Uses identity mapping if insufficient data
5. **Extrapolation Handling**: Clips predictions outside training range

### Algorithm Steps

1. Extract top-1 probabilities and correctness from validation data
2. Fit isotonic regression: `g(p_top) → P(correct | p_top)`
3. At inference: apply `g` to top-1 prob, rescale others to sum to 1

## Configuration

Add `'isotonic'` to the calibration methods in your training config:

```yaml
calibration:
  methods: ['temperature', 'platt', 'isotonic']
  enabled: true
```

## Expected Performance

Based on the implementation:
- **Temperature**: Often good baseline, especially for overconfident models
- **Platt**: Better for margin-based miscalibration patterns  
- **Isotonic**: Best for complex, non-linear miscalibration patterns

The system automatically selects the method with the lowest ECE on validation data.

## Technical Notes

- Uses `sklearn.isotonic.IsotonicRegression` with monotonicity constraints
- Handles edge cases (insufficient data, degenerate confidences)
- Preserves probability simplex constraints
- Compatible with existing temperature/Platt infrastructure

## Files Modified

1. `segmodel/train/calibration.py` - Added `IsotonicCalibrator` class
2. `configs/training/debug.yaml` - Added isotonic to methods list
3. `configs/training/better_2layer_training.yaml` - Added isotonic to methods list

The implementation is now ready for use in training sessions!
