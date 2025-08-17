# 🛠️ BiLSTM Developer Guide

This guide covers how to extend, modify, and maintain the BiLSTM text segmentation system.

---

## 🧩 **Adding New Features**

### Step 1: Create Feature Extractor

**File to create:** `segmodel/features/your_feature_ssm.py`

```python
class YourFeatureSSMExtractor:
    def __init__(self, output_dim=12, custom_param=0.5):
        self.output_dim = output_dim
        self.custom_param = custom_param
        
    def extract_features(self, lines: List[str]) -> torch.Tensor:
        # Your feature extraction logic here
        features = torch.zeros(len(lines), self.output_dim)
        # ... implement feature extraction ...
        return features
```

### Step 2: Update Feature Extractor

**File to modify:** `segmodel/features/extractor.py`

Add to `_setup_extractors()` method:
```python
# Your custom feature
if self.feature_config.get('your_feature_ssm', {}).get('enabled', False):
    config = self.feature_config['your_feature_ssm']
    output_dim = config.get('output_dim', 12)
    custom_param = config.get('custom_param', 0.5)
    
    self.extractors['your_feature_ssm'] = YourFeatureSSMExtractor(
        output_dim=output_dim,
        custom_param=custom_param
    )
    self.total_dim += output_dim
    enabled_features.append(f"your_feature_ssm({output_dim}D)")
```

### Step 3: Update Training Configuration

**Files to modify:** 
- `configs/training/aggressive_config.yaml` (or your config file)
- `segmodel/utils/config_loader.py`

**In config YAML:**
```yaml
features:
  your_feature_ssm:
    enabled: true
    dimension: 12
    custom_param: 0.5
```

**In TrainingConfig dataclass:**
```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    
    # Your feature parameters  
    your_feature_ssm_enabled: bool = False
    your_feature_ssm_dimension: int = 12
    your_feature_ssm_custom_param: float = 0.5
```

**In `flatten_config()` function:**
```python
def flatten_config(config_dict: Dict) -> Dict:
    # ... existing code ...
    
    # Your feature config
    features = config_dict.get('features', {})
    your_feature = features.get('your_feature_ssm', {})
    flattened['your_feature_ssm_enabled'] = your_feature.get('enabled', False)
    flattened['your_feature_ssm_dimension'] = your_feature.get('dimension', 12)
    flattened['your_feature_ssm_custom_param'] = your_feature.get('custom_param', 0.5)
    
    return flattened
```

### Step 4: Update Prediction Configuration

**Files to modify:**
- `configs/prediction/default.yaml`
- `segmodel/utils/prediction_config.py`

**In prediction config YAML:**
```yaml
features:
  your_feature_ssm:
    dim: 12
    enabled: true
    custom_param: 0.5
```

**In PredictionConfig dataclass:**
```python
@dataclass  
class PredictionConfig:
    features: Dict[str, Any] = field(default_factory=lambda: {
        # ... existing features ...
        'your_feature_ssm': {'dim': 12, 'enabled': True, 'custom_param': 0.5}
    })
```

### ✅ **Validation Checklist for New Features**

- [ ] Feature extractor returns correct tensor shape `(seq_len, feature_dim)`
- [ ] Feature dimension matches config specification
- [ ] Training config loads without errors
- [ ] Prediction config loads without errors  
- [ ] End-to-end training works with new feature
- [ ] Prediction system recognizes new feature
- [ ] Feature appears in training output logs
- [ ] Total feature dimension is calculated correctly

---

## ⚙️ **Adding New Configuration Parameters**

### Step 1: Update YAML Configuration

**File to modify:** `configs/training/your_config.yaml`

Add new section or extend existing:
```yaml
your_new_section:
  param1: value1
  param2: value2
  
# Or extend existing section
training:
  # ... existing params ...
  your_new_param: new_value
```

### Step 2: Update TrainingConfig Dataclass

**File to modify:** `segmodel/utils/config_loader.py`

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    
    # New parameters
    your_new_param: str = "default_value"
    your_numeric_param: float = 1.0
    your_boolean_param: bool = False
```

### Step 3: Update Config Flattening

**File to modify:** `segmodel/utils/config_loader.py` in `flatten_config()`

```python
def flatten_config(config_dict: Dict) -> Dict:
    # ... existing flattening ...
    
    # Your new section
    your_section = config_dict.get('your_new_section', {})
    flattened['your_new_param'] = your_section.get('param1', 'default')
    
    # Or from existing section
    training = config_dict.get('training', {})
    flattened['your_new_param'] = training.get('your_new_param', 'default')
    
    return flattened
```

### Step 4: Use Configuration in Code

**File to modify:** `segmodel/train/trainer.py` (or relevant file)

```python
def __init__(self, config: TrainingConfig):
    # ... existing initialization ...
    
    # Use your new parameters
    self.your_param = config.your_new_param
    if config.your_boolean_param:
        # Do something
        pass
```

### ✅ **Validation Checklist for New Configuration**

- [ ] YAML syntax is valid (no parsing errors)
- [ ] Dataclass fields have correct types and defaults
- [ ] Flattening function handles all new parameters
- [ ] Configuration loads without validation errors
- [ ] Parameters are accessible in code as `config.param_name`
- [ ] Default values work when parameter is omitted  
- [ ] Configuration appears in training output logs

---

## 🎯 **Adding New Evaluation Metrics (Phase 2 System)**

### Step 1: Create Metric Implementation

**File to create:** `segmodel/metrics/your_metric.py`

```python
import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass 
class YourMetricResults:
    """Container for your metric results."""
    metric_value: float
    additional_info: Dict[str, float]
    
def compute_your_metric(
    predictions: torch.Tensor,
    targets: torch.Tensor, 
    mask: torch.Tensor
) -> YourMetricResults:
    """
    Compute your custom metric.
    
    Args:
        predictions: Predicted labels (batch_size, seq_len)
        targets: True labels (batch_size, seq_len)
        mask: Boolean mask (batch_size, seq_len)
        
    Returns:
        YourMetricResults with computed values
    """
    # Your metric computation here
    metric_value = 0.0  # Replace with actual computation
    
    return YourMetricResults(
        metric_value=metric_value,
        additional_info={"extra_stat": 0.0}
    )
```

### Step 2: Update Metrics Module

**File to modify:** `segmodel/metrics/__init__.py`

```python
from .your_metric import (
    compute_your_metric,
    YourMetricResults
)

__all__ = [
    # ... existing exports ...
    'compute_your_metric',
    'YourMetricResults'
]
```

### Step 3: Integrate into Trainer

**File to modify:** `segmodel/train/trainer.py`

```python
# Add import
from ..metrics import (
    # ... existing imports ...
    compute_your_metric
)

# In TrainingMetrics dataclass
@dataclass
class TrainingMetrics:
    # ... existing fields ...
    val_your_metric: float = 0.0

# In evaluate() method
def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
    # ... existing code ...
    
    # Compute your metric
    your_metric_results = compute_your_metric(all_pred_batch, all_targ_batch, all_mask_batch)
    
    # Add to metrics dict
    metrics = {
        # ... existing metrics ...
        'your_metric': your_metric_results.metric_value,
    }
    
    return metrics

# In training loop metric creation
metrics = TrainingMetrics(
    # ... existing fields ...
    val_your_metric=val_metrics['your_metric'],
)
```

### Step 4: Update Training Display

**File to modify:** `segmodel/train/trainer.py` in epoch summary

```python
print(f"  🆕 Your Metric: {metrics.val_your_metric:.3f}")
```

### ✅ **Validation Checklist for New Metrics**

- [ ] Metric handles variable sequence lengths correctly
- [ ] Proper tensor masking implemented
- [ ] Cross-batch aggregation works properly  
- [ ] Metric appears in training logs
- [ ] Values are saved to training_metrics.json
- [ ] Metric provides meaningful signal for model improvement
- [ ] Edge cases handled (empty sequences, all same label, etc.)

---

## 📊 **Understanding Boundary Metrics Output (Phase 2)**

### **Training Session Output Files**

After training, you'll find these boundary-related files:

```
training_sessions/session_*/
├── boundary_metrics_summary.json    # 🆕 Detailed structural analysis
├── training_metrics.json           # Enhanced with boundary metrics per epoch  
└── final_results.txt               # Human-readable summary
```

### **Interpreting Boundary Metrics**

**Boundary Detection Scores:**
- **F1 > 0.8**: Excellent boundary detection
- **F1 0.6-0.8**: Good structural understanding  
- **F1 0.4-0.6**: Moderate boundary detection
- **F1 0.2-0.4**: Poor structural understanding
- **F1 < 0.2**: Catastrophic structural failure ⚠️

**Segment Quality Scores:**
- **Complete Segments > 80%**: Excellent section detection
- **Complete Segments 60-80%**: Good section quality
- **Complete Segments 40-60%**: Moderate fragmentation
- **Complete Segments < 40%**: Severe fragmentation ⚠️

**Transition Accuracy:**
- **> 80%**: Excellent transition detection
- **60-80%**: Good transition handling
- **40-60%**: Moderate transition issues
- **< 40%**: Major transition problems ⚠️

### **Using Analysis Insights**

The `boundary_metrics_summary.json` contains clean boundary metrics data:

```json
{
  "boundary_metrics": {
    "boundary_detection": {
      "f1": 0.040,
      "precision": 0.105,
      "recall": 0.025
    },
    "segment_quality": {
      "complete_segments_detected": 0.036,
      "avg_segment_overlap_iou": 0.207
    },
    "transition_accuracy": {
      "verse_to_chorus": 0.031,
      "chorus_to_verse": 0.009
    }
  }
}
```

**Boundary Metrics Interpretation Guidelines:**

| Metric Range | Boundary F1 | Segment Complete | Interpretation |
|--------------|-------------|------------------|----------------|
| **Excellent** | > 0.8 | > 80% | Good structural understanding |
| **Good** | 0.6-0.8 | 60-80% | Reasonable boundary detection |
| **Moderate** | 0.4-0.6 | 40-60% | Some structural issues |
| **Poor** | 0.2-0.4 | 20-40% | Significant boundary problems |
| **Failure** | < 0.2 | < 20% | Catastrophic structural failure |

### **Historical Analysis**

Use the `historical_progression` data for:

```python
import json
import matplotlib.pyplot as plt

# Load training data
with open('training_sessions/session_*/boundary_metrics_summary.json') as f:
    data = json.load(f)

# Plot boundary F1 evolution
epochs = [m['epoch'] for m in data['historical_progression']]
boundary_f1 = [m['boundary_f1'] for m in data['historical_progression']]

plt.plot(epochs, boundary_f1)
plt.title('Boundary F1 Evolution During Training')
plt.ylabel('Boundary F1')
plt.xlabel('Epoch')
plt.show()
```

### **Comparative Model Analysis**

Compare different models using boundary metrics:

```python
def compare_models(session_dirs):
    results = []
    for session_dir in session_dirs:
        with open(f'{session_dir}/boundary_metrics_summary.json') as f:
            data = json.load(f)
            results.append({
                'model': session_dir,
                'boundary_f1': data['boundary_metrics']['boundary_detection']['f1'],
                'complete_segments': data['boundary_metrics']['segment_quality']['complete_segments_detected'],
                'line_f1': data['boundary_metrics']['line_level_metrics']['macro_f1']
            })
    
    # Sort by boundary F1 (the most important metric)
    results.sort(key=lambda x: x['boundary_f1'], reverse=True)
    return results
```

This analysis helps identify which models truly understand structure vs just memorizing patterns.

---

## 📝 **Editing Existing Configuration**

### Safe Configuration Editing Process

1. **Backup existing configs:**
   ```bash
   cp configs/training/aggressive_config.yaml configs/training/aggressive_config.yaml.bak
   ```

2. **Edit configuration file:**
   ```bash
   # Use your preferred editor
   nano configs/training/aggressive_config.yaml
   ```

3. **Validate syntax:**
   ```bash
   python -c "
   import yaml
   with open('configs/training/aggressive_config.yaml') as f:
       config = yaml.safe_load(f)
   print('✅ YAML syntax valid')
   "
   ```

4. **Test configuration loading:**
   ```bash
   python -c "
   from segmodel.utils.config_loader import load_training_config
   config = load_training_config('configs/training/aggressive_config.yaml')
   print(f'✅ Config loaded: {config.experiment_name}')
   "
   ```

5. **Test with training (dry run):**
   ```bash
   python train_with_config.py configs/training/aggressive_config.yaml --epochs 1
   ```

### Common Configuration Changes

**Change Learning Rate:**
```yaml
training:
  learning_rate: 0.002  # Change from 0.001 to 0.002
```

**Enable/Disable Features:**
```yaml
features:
  pos_ssm:
    enabled: false  # Disable POS-SSM features
```

**Adjust Model Architecture:**
```yaml
model:
  hidden_dim: 256    # Reduce from 512 to 256
  dropout: 0.3       # Increase from 0.2 to 0.3
```

**Modify Emergency Monitoring:**
```yaml
emergency_monitoring:
  max_confidence_threshold: 0.90  # More strict
  min_chorus_rate: 0.10           # Lower bound
```

---

## 🔧 **System Update Procedures**

### When Adding New Schedulers

**Files to update:**
1. `segmodel/train/trainer.py` - Add to `create_scheduler()` function
2. `configs/training/*.yaml` - Add scheduler config examples
3. `segmodel/utils/config_loader.py` - Add scheduler parameters to dataclass
4. `README.md` - Document new scheduler options

**Update Process:**
```python
# In trainer.py create_scheduler() function
elif config.scheduler == 'your_new_scheduler':
    from torch.optim.lr_scheduler import YourScheduler
    scheduler = YourScheduler(
        optimizer,
        param1=getattr(config, 'scheduler_param1', default_value),
        param2=getattr(config, 'scheduler_param2', default_value)
    )
```

### When Updating Model Architecture

**Files to update:**
1. `segmodel/models/blstm_tagger.py` - Model definition
2. `configs/training/*.yaml` - New architecture parameters  
3. `segmodel/utils/config_loader.py` - New model parameters
4. `predict_baseline.py` - Update model loading logic (if needed)

**Validation Steps:**
- [ ] Model loads correctly with new parameters
- [ ] Forward pass works with expected input/output shapes
- [ ] Training script recognizes new architecture
- [ ] Prediction script handles new model format
- [ ] Saved models are backward compatible (if required)

### When Modifying Feature System

**Files to update:**
1. `segmodel/features/extractor.py` - Feature extraction logic
2. `segmodel/features/your_feature.py` - Individual feature modules
3. `configs/training/*.yaml` - Feature configuration
4. `configs/prediction/*.yaml` - Prediction feature config
5. `segmodel/utils/prediction_config.py` - Prediction config defaults

**Critical Checks:**
- [ ] Feature dimensions remain consistent
- [ ] Total feature dimension calculation is correct
- [ ] Both training and prediction configs are updated
- [ ] Feature compatibility is maintained across train/predict
- [ ] No regression in existing feature extractors

---

## 🧪 **Testing & Validation Procedures**

### Complete System Test Workflow

```bash
# 1. Test session-based prediction (recommended)
./predict.sh

# 2. Test session-based prediction with custom input
./predict.sh my_lyrics.txt

# 3. Test configuration loading
python -c "
from segmodel.utils.config_loader import load_training_config
config = load_training_config('configs/training/aggressive_config.yaml')
print(f'✅ Training config: {config.experiment_name}')
"

# 4. Test session-based config extraction
python -c "
from segmodel.utils.prediction_config import create_prediction_config_from_training_session
config, model_path = create_prediction_config_from_training_session('training_sessions/session_20250817_024332_Aggressive_Maximum_Performance_v1')
print(f'✅ Session config: {model_path}')
"

# 5. Test feature compatibility
python -c "
from segmodel.features.extractor import FeatureExtractor
config = {'head_ssm': {'enabled': True, 'output_dim': 12}}
extractor = FeatureExtractor(config)
print(f'✅ Feature extractor: {extractor.total_dim}D')
"

# 6. Test training (short run)
python train_with_config.py configs/training/aggressive_config.yaml --epochs 1

# 7. Test various prediction methods
# Session-based (recommended)
python predict_baseline.py --session training_sessions/session_20250817_024332_Aggressive_Maximum_Performance_v1 --quiet

# Training config source of truth
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --train-config-file configs/training/aggressive_config.yaml \
    --quiet

# Custom prediction config
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --prediction-config configs/prediction/default.yaml \
    --quiet
```

### Regression Testing Checklist

When making any system changes:

- [ ] **Config Loading**: All existing configs load without errors
- [ ] **Training Pipeline**: Can train models successfully  
- [ ] **Prediction Pipeline**: Can make predictions with trained models
- [ ] **Feature Compatibility**: Feature dimensions match between train/predict
- [ ] **Backward Compatibility**: Existing models still work
- [ ] **File Structure**: All expected output files are created
- [ ] **Performance**: No significant slowdown in training/prediction
- [ ] **Documentation**: All changes are documented

### Emergency Rollback Procedure

If something breaks after updates:

```bash
# 1. Restore config backups
cp configs/training/aggressive_config.yaml.bak configs/training/aggressive_config.yaml

# 2. Check git status and revert if needed
git status
git checkout HEAD -- segmodel/train/trainer.py  # Revert specific file

# 3. Test basic functionality
python -c "from segmodel.utils.config_loader import load_training_config; print('Basic import works')"

# 4. Run minimal test
python train_with_config.py configs/training/quick_test.yaml --epochs 1
```

---

## 📚 **Key Files Reference**

### Training Session Directories (Self-Contained Packages)
- `training_sessions/session_*/best_model.pt` - Trained models
- `training_sessions/session_*/training_config_snapshot.yaml` - Exact training parameters
- `training_sessions/session_*/final_results.txt` - Performance metrics
- `training_sessions/session_*/training_metrics.json` - Training history

### Configuration Files
- `configs/training/*.yaml` - Training configurations for new experiments
- `configs/prediction/*.yaml` - Prediction configurations for custom inference

### Prediction Scripts
- `predict.sh` - Simple prediction script (uses best model automatically)
- `predict_baseline.py` - Full prediction system with multiple config options

### Core System Files
- `segmodel/utils/config_loader.py` - Training config system
- `segmodel/utils/prediction_config.py` - Prediction config system (includes session-based loading)
- `segmodel/train/trainer.py` - Training logic
- `train_with_config.py` - Training script

### Feature System Files
- `segmodel/features/extractor.py` - Main feature extractor
- `segmodel/features/*_ssm.py` - Individual feature modules

### Model Files  
- `segmodel/models/blstm_tagger.py` - Model architecture

---

## 🎯 **Best Practices**

### Configuration Management
1. **Use session-based configs** for inference whenever possible (zero configuration drift)
2. **Keep training sessions intact** - they're self-contained packages with everything needed
3. **Update both training AND prediction configs** when adding features (for manual configs)
4. **Validate feature dimensions** match between training and prediction

### Development Workflow
5. **Always test changes** with a quick training run before full experiments
6. **Keep config backups** when making significant changes  
7. **Test with the simple script first**: `./predict.sh` before complex configurations
8. **Document new parameters** with clear descriptions and defaults

### System Reliability  
9. **Use descriptive config names** that indicate their purpose
10. **Test backward compatibility** with existing models when possible
11. **Follow the established patterns** for consistency
12. **Prefer training sessions over manual config hunting**

### Session-Based Workflow (New Recommended Approach)
- **Training produces**: Complete session directory with model + config + results
- **Prediction uses**: Session directory as single source of truth
- **Benefits**: Zero drift, perfect compatibility, complete reproducibility
- **Fallback**: Manual configs still supported for advanced use cases

---

## 📖 **Command Documentation Maintenance**

### Critical Responsibility: Keep Documentation Updated

**When adding new command-line arguments, you MUST update:**

1. **README.md** - Complete Command Reference section
2. **DEVELOPER_GUIDE.md** - This section
3. **Script help text** - The `--help` output in the scripts

### Adding New Training Arguments

**Files to update:**
- `train_with_config.py` - Add argument to parser + help text
- `README.md` - Add to Training Commands section with examples
- `documentation/DEVELOPER_GUIDE.md` - Update this documentation section

**Example Process:**
```bash
# 1. Add to train_with_config.py
parser.add_argument('--your-new-param', help='Description of new parameter')

# 2. Test help output
python train_with_config.py --help

# 3. Update README.md with new parameter and examples
# 4. Update DEVELOPER_GUIDE.md with maintenance notes
```

### Adding New Prediction Arguments

**Files to update:**
- `predict_baseline.py` - Add argument to parser + help text  
- `README.md` - Add to Prediction Commands section with examples
- `documentation/DEVELOPER_GUIDE.md` - Update this documentation section

### Documentation Testing Checklist

When modifying command-line interfaces:

- [ ] **Help text is accurate**: `python script.py --help` shows correct info
- [ ] **README examples work**: All examples in README can be copy-pasted and run
- [ ] **Parameter descriptions are clear**: Each parameter has clear purpose and usage
- [ ] **Examples cover common use cases**: Most frequent usage patterns are documented
- [ ] **Advanced options are explained**: Complex parameters have sufficient detail

### Automated Documentation Checks

**Recommended workflow:**
```bash
# Generate help text and compare with documentation
python train_with_config.py --help > current_train_help.txt
python predict_baseline.py --help > current_predict_help.txt

# Check if documentation needs updates
# (Compare with examples in README.md)
```

**Warning Signs Documentation is Outdated:**
- Help text shows parameters not mentioned in README
- README examples fail to run
- Users report confusion about parameter usage
- New features added but not documented

This developer guide should help you safely extend and maintain the BiLSTM system! 🚀

**Remember: Good documentation is as important as good code!**
