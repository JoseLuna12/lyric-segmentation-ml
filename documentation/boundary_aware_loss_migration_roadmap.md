# 🚀 Boundary-Aware Cross-Entropy Migration Roadmap

## 📋 **Current State Analysis**

### Existing Loss Configuration Structure (from `anti_collapse` section)
```yaml
anti_collapse:
  label_smoothing: 0.16           # 0.1-0.2 range in configs
  weighted_sampling: true         # Essential for class balance
  entropy_lambda: 0.08           # 0.0-0.08 range, usually disabled
```

### Current Training Entry Points
1. **`train_with_config.py`** - Main training script
2. **`segmodel/train/trainer.py`** - Core training loop
3. **Config files** - YAML-based configuration system

### Current Metrics & Monitoring
- Loss curves (train/val)
- F1 scores (macro, verse, chorus)
- Confidence metrics (max_prob, conf_over_95)
- Chorus rate monitoring
- Boundary F1 (recent addition)

---

## 🎯 **Proposed Migration Strategy**

### Phase 1: Configuration Evolution (Non-Breaking)

#### 1.1 Enhanced Loss Configuration
```yaml
# NEW: Enhanced anti-collapse system with boundary awareness
loss:
  type: "boundary_aware_cross_entropy"
  
  # Phase 1: Existing settings (backward compatible)
  label_smoothing: 0.16
  entropy_lambda: 0.0          # Keep disabled by default
  
  # Phase 2: Boundary awareness (NEW)
  boundary_weight: 2.0         # 1.5-3.0 range
  
  # Phase 3: Segment consistency (NEW) 
  segment_consistency_lambda: 0.03   # 0.0-0.1 range, start conservative
  
  # Phase 4: Confidence control (NEW)
  conf_penalty_lambda: 0.005   # 0.0-0.02 range, start conservative
  conf_threshold: 0.95         # 0.90-0.98 range
  
  # Advanced (optional)
  use_boundary_as_primary: true
```

#### 1.2 Migration Timeline
**Immediate (Existing Settings)**:
- ✅ `label_smoothing`: Direct mapping from `anti_collapse.label_smoothing`
- ✅ `entropy_lambda`: Direct mapping from `anti_collapse.entropy_lambda` 
- ✅ `class_weights`: Auto-computed from dataset (no config needed)

**New Settings (Conservative Defaults)**:
- 🆕 `boundary_weight: 2.0` (proven effective in tests)
- 🆕 `segment_consistency_lambda: 0.03` (start conservative, tune up)
- 🆕 `conf_penalty_lambda: 0.005` (start conservative, tune up)

### Phase 2: Training Code Integration

#### 2.1 Training Script Updates
```python
# train_with_config.py - MINIMAL CHANGE
loss_function = create_loss_function(
    num_classes=config.num_classes,
    label_smoothing=config.loss.label_smoothing,  # Moved from anti_collapse
    class_weights=class_weights,
    boundary_weight=getattr(config.loss, 'boundary_weight', 2.0),
    segment_consistency_lambda=getattr(config.loss, 'segment_consistency_lambda', 0.03),
    conf_penalty_lambda=getattr(config.loss, 'conf_penalty_lambda', 0.005),
    conf_threshold=getattr(config.loss, 'conf_threshold', 0.95),
    entropy_lambda=getattr(config.loss, 'entropy_lambda', 0.0)
)
```

#### 2.2 Metrics Enhancement  
```python
# Enhanced metrics collection in trainer
if hasattr(loss_function, 'forward') and return_metrics:
    loss, loss_metrics = loss_function(logits, labels, mask, return_metrics=True)
    # Log detailed loss breakdown
    self.log_loss_components(loss_metrics)
else:
    # Backward compatibility
    loss = loss_function(logits, labels, mask)
```

### Phase 3: Analysis & Visualization Updates

#### 3.1 New Metrics to Track
```python
# Enhanced metrics collection
enhanced_metrics = {
    # Existing metrics
    'train_loss', 'val_loss', 'val_macro_f1', 'val_boundary_f1',
    
    # NEW: Loss component breakdown
    'boundary_loss_contribution',      # Phase 2 impact
    'consistency_loss_contribution',   # Phase 3 impact  
    'confidence_penalty_contribution', # Phase 4 impact
    'entropy_contribution',           # Phase 1 regularization
    
    # NEW: Loss architecture tracking
    'boundary_weight',                # Hyperparameter tracking
    'segment_consistency_lambda',     # Hyperparameter tracking
    'conf_penalty_lambda',           # Hyperparameter tracking
}
```

#### 3.2 Enhanced Visualizations
```python
# New charts for analyze_training.py
def plot_loss_breakdown(df):
    """Plot loss component contributions over time"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw components
    axes[0,0].plot(df['boundary_loss'], label='Boundary Loss')
    axes[0,0].plot(df['consistency_loss'], label='Consistency Loss')
    axes[0,0].plot(df['confidence_penalty'], label='Confidence Penalty')
    
    # Weighted contributions
    axes[0,1].plot(df['boundary_loss_contribution'], label='Boundary')
    axes[0,1].plot(df['consistency_loss_contribution'], label='Consistency')
    axes[0,1].plot(df['confidence_penalty_contribution'], label='Confidence')
    
    # Hyperparameter evolution
    axes[1,0].plot(df['boundary_weight'], label='Boundary Weight')
    axes[1,0].plot(df['segment_consistency_lambda'], label='Consistency λ')
    
    # Architecture monitoring
    axes[1,1].bar(['Boundary', 'Consistency', 'Confidence'], 
                  [df['boundary_loss_contribution'].iloc[-1],
                   df['consistency_loss_contribution'].iloc[-1], 
                   df['confidence_penalty_contribution'].iloc[-1]])
```

---

## 🔧 **Implementation Priority**

### Critical Path (Must Have)
1. ✅ **Config migration**: `anti_collapse.label_smoothing` → `loss.label_smoothing`
2. 🎯 **Training integration**: Update `train_with_config.py` loss creation
3. 📊 **Basic metrics**: Add loss component tracking to trainer
4. 📈 **Visualization**: Update `analyze_training.py` for new metrics

### Enhanced Features (Nice to Have)
1. 🔍 **Advanced debugging**: Detailed loss breakdown charts
2. ⚖️ **Hyperparameter tracking**: Evolution of loss weights over training
3. 🎛️ **Dynamic tuning**: Adaptive loss component weights
4. 📋 **Config validation**: Automatic range checking for new parameters

---

## 📊 **Backward Compatibility Strategy**

### Configuration Migration
```yaml
# OLD FORMAT (still supported)
anti_collapse:
  label_smoothing: 0.16
  entropy_lambda: 0.08

# NEW FORMAT (recommended)
loss:
  type: "boundary_aware_cross_entropy"
  label_smoothing: 0.16
  entropy_lambda: 0.08
  boundary_weight: 2.0
  segment_consistency_lambda: 0.03
```

### Code Compatibility
```python
# Gradual migration approach
def get_loss_config(config):
    """Extract loss configuration with backward compatibility"""
    if hasattr(config, 'loss'):
        return config.loss
    else:
        # Backward compatibility: map from anti_collapse
        return SimpleNamespace(
            label_smoothing=getattr(config, 'label_smoothing', 0.2),
            entropy_lambda=getattr(config, 'entropy_lambda', 0.0),
            boundary_weight=2.0,  # Default for legacy configs
            segment_consistency_lambda=0.03,
            conf_penalty_lambda=0.005
        )
```

---

## 🚦 **Rollout Plan**

### Week 1: Foundation
- [ ] Update config schema to support `loss` section
- [ ] Implement backward compatibility helpers
- [ ] Test with existing configs (should work unchanged)

### Week 2: Integration  
- [ ] Update training entry points
- [ ] Add enhanced metrics collection
- [ ] Test boundary-aware loss with conservative settings

### Week 3: Visualization
- [ ] Update analysis scripts for new metrics
- [ ] Add loss breakdown visualizations
- [ ] Validate charts with real training data

### Week 4: Optimization
- [ ] Tune default hyperparameters based on training results
- [ ] Add config validation and recommendations
- [ ] Document best practices

---

## 🎯 **Expected Impact**

### Training Improvements
- **Better `val_boundary_f1`**: Direct boundary optimization
- **Improved `val_window_diff`**: Better segmentation quality
- **Higher `val_complete_segments`**: Reduced fragmentation
- **Maintained `val_macro_f1`**: Token-level accuracy preserved

### Monitoring Enhancements
- **Loss transparency**: See exactly what drives training
- **Debugging capability**: Identify problematic loss components  
- **Hyperparameter insights**: Track optimal settings over time
- **Reproducibility**: Complete loss configuration documentation

---

## ⚠️ **Migration Risks & Mitigations**

### Risk: Configuration Breaking Changes
**Mitigation**: Full backward compatibility with automatic config migration

### Risk: Performance Regression  
**Mitigation**: Conservative defaults + gradual parameter tuning

### Risk: Analysis Script Failures
**Mitigation**: Graceful handling of missing metrics + fallback to existing charts

### Risk: Training Loop Disruption
**Mitigation**: Thorough testing with existing configs before rollout

---

## 📝 **Recommended Conservative Configuration**

```yaml
# Proven starter configuration
loss:
  type: "boundary_aware_cross_entropy"
  
  # Phase 1: Proven settings
  label_smoothing: 0.16           # From successful configs
  entropy_lambda: 0.0             # Keep disabled initially
  
  # Phase 2: Conservative boundary focus  
  boundary_weight: 2.0            # Tested effective range
  
  # Phase 3: Gentle consistency regularization
  segment_consistency_lambda: 0.03  # Start conservative
  
  # Phase 4: Light confidence control
  conf_penalty_lambda: 0.005      # Start conservative
  conf_threshold: 0.95            # Standard threshold
  
  # Architecture
  use_boundary_as_primary: true   # Recommended
```

This configuration should provide immediate benefits while maintaining training stability and enabling gradual optimization.
