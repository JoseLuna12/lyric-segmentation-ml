# CNN Hyperparameter Optimization - Quick Start Guide

## Overview
This script (`scripts/hyperopt_cnn.py`) provides fast hyperparameter optimization specifically for CNN models with the following optimizations:

### Fast Testing Configuration
- **Reduced parameter space**: Minimal ranges for quick trials
- **Short epochs**: Maximum 3-5 epochs per trial
- **Fixed architecture**: Single hidden dimension (128), limited layers (1-2)
- **Conservative ranges**: Narrow parameter ranges for proven values
- **Feature stability**: Disabled feature toggles for consistency

### CNN-Specific Optimizations
- **Kernel sizes**: Only 2 configurations ([3,5,7] and [3,5])
- **Dilation rates**: Simple progressions ([1,2,4] and [1,2])
- **Residual connections**: Always enabled for stability
- **Attention**: Fixed to boundary_aware type with 4 heads
- **Validation**: CNN-composite strategy only

### Quick Start Commands

#### Fast Testing (10 trials, ~30 minutes)
```bash
python scripts/hyperopt_cnn.py --study-name cnn_fast_test --n-trials 10 --timeout 1800
```

#### Production Run (50 trials, ~3 hours)
```bash
python scripts/hyperopt_cnn.py --study-name cnn_production_v1 --n-trials 50 --timeout 10800
```

#### Resume Previous Study
```bash
python scripts/hyperopt_cnn.py --study-name cnn_production_v1 --resume --session-dir results/2025-08-27/timestamp_cnn_production_v1 --n-trials 25
```

### Expected Trial Duration
- **Fast mode**: ~3-5 minutes per trial (5 epochs max)
- **Total for 10 trials**: ~30-50 minutes
- **Total for 50 trials**: ~2.5-4 hours

### Output Structure
```
results/YYYY-MM-DD/HH-MM-SS_study_name/
├── trials/          # Individual trial results
├── configs/         # Best configuration files  
├── logs/           # Training and optimization logs
├── study.db        # SQLite database with all trials
├── best_config.yaml # Best configuration for reuse
└── summary_report.md # Optimization summary
```

### Configuration Files
- **Base config**: `configs/training/cnn_hyperopt_fast.yaml`
- **Model used**: CNN with boundary-aware attention
- **Loss function**: Boundary-aware cross-entropy
- **Features**: Minimal set (head_ssm, tail_ssm, phonetic_ssm, syllable_pattern_ssm)

### Key Differences from BiLSTM Hyperopt
1. **CNN-specific parameters**: kernel_sizes, dilation_rates, residual_connections
2. **Faster convergence**: CNNs typically converge in fewer epochs
3. **Batch efficiency**: CNNs can handle larger batches (multiplier up to 1.5x)
4. **Conservative monitoring**: More strict overconfidence thresholds
5. **Boundary focus**: Enhanced boundary detection capabilities

### Performance Expectations
- **Boundary F1**: Target improvement of 15-25% over baseline
- **Macro F1**: Competitive with BiLSTM while being faster
- **Training speed**: 2-3x faster than equivalent BiLSTM
- **Memory usage**: More efficient due to parallel processing

### Monitoring and Safety
- **Emergency monitoring**: CNN-tuned thresholds prevent overconfidence
- **Early stopping**: Aggressive (patience=3) for fast decisions
- **Gradient stability**: Monitored for CNN-specific patterns
- **Confidence calibration**: Temperature and Platt scaling

This fast configuration is designed for quick experimentation and validation of CNN approaches while maintaining the quality of the optimization process.
