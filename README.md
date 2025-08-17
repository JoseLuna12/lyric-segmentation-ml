# 🎵 BLSTM Baseline for Lyrics Segmentation

Clean, modular implementation of BiLSTM for verse/chorus classification with anti-collapse features.

## 🏗️ Architecture

```
├── configs/
│   ├── training_config.yaml     # Default production settings
│   ├── quick_test.yaml          # Fast testing configuration
│   └── aggressive_training.yaml # High-performance settings
├── segmodel/                    # Modular package
│   ├── data/                    # Dataset loading with weighted sampling
│   ├── features/                # Head-SSM & Tail-SSM feature extraction
│   ├── models/                  # Simple BiLSTM model
│   ├── losses/                  # Cross-entropy with label smoothing
│   ├── train/                   # Training with guardrails
│   └── utils/                   # Utilities
├── train_baseline.py            # Main training script
└── predict_baseline.py          # Inference script
```

## 🚀 Quick Start

### Training

**Simple Script (Recommended):**
```bash
# Train with best configuration
./train.sh
```

**With YAML Configuration:**
```bash
# Train with aggressive configuration (best performance)
python train_with_config.py configs/training/aggressive_config.yaml

# Quick test run
python train_with_config.py configs/training/quick_test.yaml

# With command line overrides
python train_with_config.py configs/training/aggressive_config.yaml \
    --batch-size 32 \
    --learning-rate 0.0005 \
    --epochs 100 \
    --disable-emergency-monitoring
```

### Prediction (Simple!)

**Easy Prediction Script (Recommended):**
```bash
# Use default config (everything configured in YAML)
./predict.sh
```

**Advanced Prediction Options:**
```bash
# Use session directory (contains model + exact config)
python predict_baseline.py --session training_sessions/session_20250817_024332_Aggressive_Maximum_Performance_v1

# Use custom prediction config
python predict_baseline.py --prediction-config configs/prediction/production.yaml
```

## 📚 Complete Command Reference

### Training Commands

#### Using the Training Script
```bash
# Simple training with best configuration
./train.sh
```

#### Advanced Training Options
```bash
python train_with_config.py <config_file> [OPTIONS]

# Required:
<config_file>                    Path to YAML configuration file

# Optional overrides:
--batch-size BATCH_SIZE          Override batch size from config
--learning-rate LEARNING_RATE    Override learning rate from config  
--epochs EPOCHS                  Override max epochs from config
--label-smoothing LABEL_SMOOTHING Override label smoothing from config
--entropy-lambda ENTROPY_LAMBDA  Override entropy lambda from config
--disable-emergency-monitoring   Disable emergency monitoring
--train TRAIN                    Override training data file
--val VAL                        Override validation data file
--test TEST                      Override test data file
```

#### Training Examples
```bash
# Use default aggressive configuration
python train_with_config.py configs/training/aggressive_config.yaml

# Quick test run (1 epoch)
python train_with_config.py configs/training/quick_test.yaml

# Override specific parameters
python train_with_config.py configs/training/aggressive_config.yaml \
    --batch-size 64 \
    --learning-rate 0.002 \
    --epochs 80

# Disable safety monitoring for maximum performance
python train_with_config.py configs/training/aggressive_config.yaml \
    --disable-emergency-monitoring

# Use different dataset files
python train_with_config.py configs/training/aggressive_config.yaml \
    --train data/my_train.jsonl \
    --val data/my_val.jsonl \
    --test data/my_test.jsonl
```

### Prediction Commands  

#### Using the Prediction Script (Recommended)
```bash
# Simple prediction with default configuration
./predict.sh
```

#### Advanced Prediction Options
```bash
python predict_baseline.py [MODEL_OPTIONS] [INPUT_OPTIONS] [CONFIG_OPTIONS] [OTHER_OPTIONS]

# Model Options (choose one):
--session SESSION                Path to training session directory (recommended)
--model MODEL                    Path to trained model file

# Configuration Options (choose one):
--prediction-config CONFIG       Path to prediction config file
--train-config-file CONFIG       Path to training config file (legacy)

# Input Options (choose one):
--text TEXT                      Text file with lyrics
--lines LINE1 LINE2 ...          Lyrics lines as command arguments
--lyrics "MULTI_LINE_STRING"     Multi-line lyrics as single string
--stdin                          Read lyrics from stdin/pipe

# Other Options:
--temperature TEMPERATURE        Override temperature scaling
--quiet                          Quiet mode (no terminal output)
```

#### Prediction Examples
```bash
# Use session directory (everything in one place)
python predict_baseline.py --session training_sessions/session_20250817_024332_Aggressive_Maximum_Performance_v1

# Use custom prediction config
python predict_baseline.py --prediction-config configs/prediction/production.yaml

# Use training config as source of truth
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --train-config-file configs/training/aggressive_config.yaml

# Different input methods
python predict_baseline.py --session training_sessions/session_*/ --text my_lyrics.txt
python predict_baseline.py --session training_sessions/session_*/ --lines "First line" "Second line"
python predict_baseline.py --session training_sessions/session_*/ --lyrics "Line 1\nLine 2\nLine 3"
echo "Line 1\nLine 2" | python predict_baseline.py --session training_sessions/session_*/ --stdin

# Override temperature scaling
python predict_baseline.py --session training_sessions/session_*/ --temperature 2.0

# Silent operation (for scripts)
python predict_baseline.py --session training_sessions/session_*/ --quiet
```

### Configuration-Based Workflow (Recommended)

**Why Use Session-Based Prediction:**
- ✅ **Zero configuration drift** - Uses exact parameters from training
- ✅ **Everything in one place** - No hunting for matching configs  
- ✅ **Perfect compatibility** - Guaranteed feature dimension match
- ✅ **Complete reproducibility** - Full context in one directory

**Configuration Files:**
- `configs/prediction/default.yaml` - General purpose prediction
- `configs/prediction/production.yaml` - Production deployment (quiet mode)
- `configs/prediction/debug.yaml` - Development debugging (verbose, CPU)

---

### Analysis & Monitoring

**Compare training sessions:**
```bash
python scripts/compare_sessions.py
```

**Detailed session analysis:**
```bash
python scripts/compare_sessions.py --detail session_20250816_123456
```

**Visualize training progress:**
```bash
python scripts/analyze_training.py training_sessions/session_20250816_123456/training_metrics.json
```

### ✅ **Updated Inference**

**Multiple input methods supported with config compatibility:**

```bash
# ✅ RECOMMENDED: Use training session config for feature compatibility
python predict_baseline.py \
    --model training_sessions/session_20250816_123456/best_model.pt \
    --config training_sessions/session_20250816_123456/ \
    --lyrics "Walking down the street tonight
Looking at the stars above
This is our song we sing
Dancing to the beat of love"

# ✅ Use specific config file
python predict_baseline.py \
    --model best_model.pt \
    --config configs/aggressive_config.yaml \
    --text lyrics.txt

# ⚠️ Fallback: Comprehensive default features (if no config specified)
python predict_baseline.py \
    --model best_model.pt \
    --lines "Walking down the street" "Thinking of you" "Dancing tonight"
```

### 🎯 **Clean Configuration System** 

**Explicit configuration (no hidden defaults):**

```bash
# 1. RECOMMENDED: Use prediction config templates
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --prediction-config configs/prediction/default.yaml

# 2. Production deployment (quiet mode)
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --prediction-config configs/prediction/production.yaml

# 3. Custom training config extraction
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --train-config-file configs/training/aggressive_config.yaml \
    --temperature 1.5

# 4. No config = Clear error (no hidden fallbacks)
python predict_baseline.py --model training_sessions/session_*/best_model.pt
# ❌ Error: "No prediction config found! Please provide..."
```

**Alternative Input Methods:**
```bash
# Multi-line string
python predict_baseline.py \
    --model [model] \
    --prediction-config configs/prediction/default.yaml \
    --lyrics "Your lyrics here..."

# Command line arguments  
python predict_baseline.py \
    --model [model] \
    --prediction-config configs/prediction/default.yaml \
    --lines "Line 1" "Line 2" "Line 3"
```

# From pipe/stdin
echo "Line 1\nLine 2" | python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt --stdin
```

### 📁 **Organized Output Structure**

Results are automatically organized by model name (perfect for AI-less machines):

```
prediction_results/
├── predict_lyric.txt                                    # Input file (place here)
└── 20250817-024332-Aggressive-Maximum-Performance-v1/  # Auto-generated model folder
    ├── predictions.txt      # 📄 Plain text output (no colors)
    ├── predictions.json     # 📊 Detailed JSON data
    └── parameters.json      # ⚙️  Run parameters & metadata
```

**predictions.txt** format (perfect for scripts):
```
Lyrics Structure Prediction
==================================================

VERSE   (0.894) | When you were here before, couldn't look you in the eye  
VERSE   (0.902) | You're just like an angel, your skin makes me cry
CHORUS  (0.693) | But I'm a creep, I'm a weirdo
CHORUS  (0.795) | What the hell am I doin' here?
```

**parameters.json** tracks everything:
```json
{
  "timestamp": "2025-08-17T15:23:28.518156",
  "model_path": "training_sessions/session_.../best_model.pt", 
  "config_path": "configs/aggressive_config.yaml",
  "temperature": 1.5,
  "input_file": "predict_lyric.txt",
  "device": "mps"
}
```

## 🧩 Features

- **Modular Design**: Easy to extend with new features
- **Anti-Collapse**: Prevents overconfidence with label smoothing + class weights
- **Weighted Sampling**: Ensures balanced training despite class imbalance  
- **✅ Advanced LR Scheduling**: Cosine annealing, warm restarts, step decay, plateau
- **✅ Optimized Training**: 4x larger batch sizes (8→32) with proper LR scaling
- **✅ Configurable Monitoring**: No more magic numbers - all thresholds configurable
- **Temperature Calibration**: Post-hoc confidence calibration with configurable grid
- **Emergency Monitoring**: Real-time training guardrails with full parameter control
- **Head-SSM Features**: Line-beginning similarity for chorus detection
- **Tail-SSM Features**: Line-ending similarity for rhyme pattern detection
- **String-SSM Features**: Overall textual similarity with multiple algorithms
- **Comprehensive Metrics**: Incremental saving of training progress with analysis tools
- **YAML Configuration**: Flexible configuration system with command-line overrides and reproducibility snapshots

## 📊 Expected Performance

**With Head-SSM + Tail-SSM + String-SSM Features:**
- **Accuracy**: 75-85% (improved with triple features)
- **F1 Macro**: 0.70-0.80
- **Verse F1**: 0.80-0.90 (high precision)
- **Chorus F1**: 0.60-0.75 (challenging due to variety)
- **Confidence**: Well-calibrated (avg ~0.70-0.80)
- **Training Time**: ✅ **Improved** ~35-50 minutes on modern hardware (faster with larger batches)
- **Feature Dimension**: 36D (12D Head-SSM + 12D Tail-SSM + 12D String-SSM)

## 🔧 Configuration

### YAML Configuration Files

The recommended way to manage training parameters is through YAML configuration files:

**Available Configurations:**
- `configs/training_config.yaml` - Default production settings
- `configs/quick_test.yaml` - Fast testing (5 epochs, smaller model)
- `configs/aggressive_config.yaml` - ✅ **Updated**: High-performance with advanced scheduling

### ✅ **New: Advanced Learning Rate Scheduling**

The training system now supports multiple sophisticated learning rate schedulers:

**Available Schedulers:**
- **`cosine`** (default): Cosine annealing - smooth decay from initial LR to min_lr
- **`cosine_restarts`**: Cosine annealing with warm restarts - periodic LR cycling
- **`step`**: Step decay - reduce LR by factor every N epochs
- **`plateau`**: Reduce LR when validation F1 plateaus (legacy default)
- **`warmup_cosine`**: Linear warmup + cosine decay - best for large models

**Example Scheduler Configuration:**
```yaml
training:
  batch_size: 32            # ✅ Optimized from 8 to 32 
  learning_rate: 0.001      # ✅ Scaled accordingly
  max_epochs: 120
  patience: 20
  
  # ✅ NEW: Advanced Learning Rate Scheduling
  scheduler: "cosine"              # Cosine annealing (recommended)
  min_lr: 1e-6                    # Minimum learning rate
  cosine_t_max: 120               # Period for cosine (should match max_epochs)
  
  # Alternative scheduler options
  # scheduler: "cosine_restarts"   # For periodic restarts
  # cosine_t0: 10                 # Initial restart period
  # cosine_t_mult: 2              # Period multiplier after each restart
  
  # scheduler: "step"             # For step decay
  # step_size: 30                 # Reduce LR every 30 epochs
  # step_gamma: 0.5               # Multiply LR by 0.5
  
  # scheduler: "warmup_cosine"    # For warmup + cosine
  # warmup_epochs: 5              # Warmup for first 5 epochs
```

### ✅ **New: Configurable Emergency Monitoring**

All magic numbers have been moved to configuration for full control:

```yaml
# ✅ NEW: Fully Configurable Emergency Monitoring
emergency_monitoring:
  enabled: true
  
  # Batch-level thresholds
  max_confidence_threshold: 0.95      # Max avg confidence per batch
  min_chorus_rate: 0.05              # Min chorus prediction rate
  max_chorus_rate: 0.85              # Max chorus prediction rate  
  max_conf_over_95_ratio: 0.1        # Max ratio of predictions >95% confidence
  
  # Epoch-level thresholds
  val_overconf_threshold: 0.96       # Validation overconfidence warning
  val_f1_collapse_threshold: 0.1     # F1 collapse detection
  emergency_overconf_threshold: 0.98 # Emergency stop threshold
  emergency_conf95_ratio: 0.8        # Emergency high-confidence ratio
  emergency_f1_threshold: 0.05       # Emergency F1 collapse threshold
  
  # Timing parameters  
  skip_batches: 50                   # Skip monitoring first N batches
  skip_epochs: 3                     # Skip monitoring first N epochs
  print_batch_every: 10              # Print batch info every N batches
```

### ✅ **New: Temperature Calibration Configuration**

```yaml
# ✅ NEW: Configurable Temperature Calibration
temperature_calibration:
  temperature_grid: [0.8, 1.0, 1.2, 1.5, 1.7, 2.0]  # Grid search values
  default_temperature: 1.0          # Fallback if calibration fails
```

**Complete Example Configuration:**
```yaml
# Training parameters - IMPROVED
training:
  batch_size: 32         # ✅ INCREASED from 8 for stable gradients
  learning_rate: 0.001   # ✅ SCALED UP (sqrt scaling) for larger batch
  weight_decay: 0.005
  max_epochs: 120
  patience: 20
  gradient_clip_norm: 0.5
  
  # ✅ ADVANCED LEARNING RATE SCHEDULING
  scheduler: "cosine"         # Cosine annealing (recommended for BiLSTM)
  min_lr: 1e-6               # Minimum learning rate
  cosine_t_max: 120          # Should match max_epochs

# Anti-collapse settings  
anti_collapse:
  label_smoothing: 0.15      # Reduced for aggressive training
  weighted_sampling: true
  entropy_lambda: 0.0

# ✅ CONFIGURABLE EMERGENCY MONITORING
emergency_monitoring:
  enabled: true
  max_confidence_threshold: 0.95
  min_chorus_rate: 0.05
  max_chorus_rate: 0.85
  skip_batches: 50
  skip_epochs: 3
  print_batch_every: 10

# ✅ CONFIGURABLE TEMPERATURE CALIBRATION  
temperature_calibration:
  temperature_grid: [0.8, 1.0, 1.2, 1.5, 1.7, 2.0]
  default_temperature: 1.0
  
# Experiment metadata
experiment:
  name: "improved_training_v2"
  description: "Advanced scheduling + configurable monitoring"
  tags: ["improved", "cosine_scheduling", "larger_batch"]
```

### Command Line Overrides

All YAML parameters can be overridden from the command line:

```bash
python train_with_config.py configs/training_config.yaml \
    --batch-size 32 \
    --learning-rate 0.0005 \
    --epochs 100 \
    --entropy-lambda 0.003 \
    --disable-emergency-monitoring
```

## 🧩 Feature System Architecture

### How Features Work

The BLSTM model doesn't work directly with raw text. Instead, each lyric line is converted into a **numerical feature vector** that captures structural patterns useful for verse/chorus classification.

**Feature Pipeline:**
```
Raw Lyrics → Feature Extractors → Numerical Vectors → BLSTM → Classification
```

**Example transformation:**
```python
# Input: List of lyric lines
lines = [
    "Walking down this street tonight",    # verse
    "Thinking of you every day",           # chorus  
    "Dancing in the pale moonlight",       # verse
    "Thinking of you every day"            # chorus (repeated)
]

# Output: Feature matrix (4 lines × 36 dimensions)
features = torch.tensor([
    [0.33, 1.00, 0.47, ..., 0.51, 0.94, 0.33, ...],  # Line 0: 36D feature vector
    [0.25, 1.00, 0.50, ..., 0.45, 0.78, 0.25, ...],  # Line 1: 36D feature vector  
    [0.25, 1.00, 0.50, ..., 0.48, 0.94, 0.22, ...],  # Line 2: 36D feature vector
    [0.25, 1.00, 0.50, ..., 0.52, 0.78, 0.31, ...]   # Line 3: 36D feature vector
])
```

### Current Feature Types

#### 1. **Head-SSM Features** (12 dimensions)
**Purpose**: Detect chorus repetitions by analyzing line beginnings.

**How it works:**
- Extracts first 2 words from each line: `"Walking down this street"` → `"walking down"`
- Computes similarity matrix between all line beginnings
- Generates 12 statistical features per line:

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `mean_similarity` | Average similarity to all lines | 0.33 |
| `max_similarity` | Highest similarity to any line | 1.00 |
| `high_sim_ratio` | Fraction of high similarities (≥0.8) | 0.25 |
| `prev_similarity` | Similarity to previous line | 0.00 |
| `position` | Normalized position in song [0,1] | 0.25 |
| ... | 7 more features | ... |

**Why it works**: Choruses often start with the same words (`"Thinking of you"`, `"I will always"`), creating high similarity scores that the BLSTM learns to associate with chorus labels.

#### 2. **Tail-SSM Features** (12 dimensions)  
**Purpose**: Detect rhyme patterns by analyzing line endings.

**How it works:**
- Extracts last 2 words from each line: `"street at night"` → `"at night"`
- Removes punctuation for better rhyme detection
- Computes similarity matrix between all line endings
- Generates 12 statistical features per line (same structure as Head-SSM)

**Why it works**: Verses and choruses often have different rhyme schemes. Choruses may rhyme more consistently (`"day"/"way"/"say"`), while verses may have more varied endings.

#### 3. **String-SSM Features** (12 dimensions)
**Purpose**: Detect overall textual similarity between lines using normalized Levenshtein distance.

**How it works:**
- Normalizes text by removing punctuation and converting to lowercase (configurable)
- Computes Levenshtein distance between all pairs of lines
- Normalizes distance by dividing by the length of the longer string
- Converts to similarity score: `similarity = 1 - (distance / max_length)`
- Generates 12 statistical features per line (same structure as Head-SSM and Tail-SSM)

| Feature | Description | Example Value |
|---------|-------------|---------------|
| `string_mean_similarity` | Average similarity to all lines | 0.45 |
| `string_max_similarity` | Highest similarity to any line | 0.94 |
| `string_high_sim_ratio` | Fraction of high similarities (≥0.7) | 0.33 |
| `string_prev_similarity` | Similarity to previous line | 0.22 |
| ... | 8 more features | ... |

**Configuration options:**
- `case_sensitive`: Whether to preserve case during comparison
- `remove_punctuation`: Whether to remove punctuation before comparison  
- `similarity_threshold`: Minimum similarity threshold (values below are set to 0)
- `similarity_method`: Algorithm to use (`"word_overlap"`, `"jaccard"`, `"levenshtein"`)

**Similarity Methods:**

| Method | Speed | Accuracy | Best For | Trade-offs |
|--------|-------|----------|----------|------------|
| `word_overlap` | **Fastest** (0.24s/500 lines) | Medium | Repeated phrases, chorus detection | Ignores word order; simple word counting |
| `jaccard` | Medium (0.51s/500 lines) | Medium-High | Content similarity, topic overlap | Set-based; good for similar themes |
| `levenshtein` | Slowest (89s/500 lines) | **Highest** | Exact repetitions, typo detection | Character-level; very expensive |

**Example Comparison:**
```
Line A: "thinking of you every single day"
Line B: "thinking of you every day"

word_overlap:  0.833  (5 matches / 6 words)
jaccard:       0.833  (5 shared / 6 total unique words)  
levenshtein:   0.781  (7 character edits / 32 chars)
```

**When to Use Each Method:**
- **`word_overlap`** (default): General training, fast iterations, chorus detection
- **`jaccard`**: When you want better semantic similarity than word_overlap
- **`levenshtein`**: Research/analysis only, when you need exact character-level similarity

**Performance Impact:**
```
For a typical song (50 lines):
word_overlap:  ~0.002 seconds  ✅ 
jaccard:       ~0.005 seconds  ✅
levenshtein:   ~2.5 seconds    ⚠️  (would cause training hangs)

Training time difference (1000 songs):
word_overlap:  ~2 seconds   
jaccard:       ~5 seconds   
levenshtein:   ~42 minutes  (unusable for training)
```

**Recommendation**: Use `"word_overlap"` (default) for training - it's 373x faster than Levenshtein while capturing the key repetition patterns needed for verse/chorus classification.

**Why it works**: Captures general textual repetition patterns that complement the specific head/tail patterns. Useful for detecting exact or near-exact line repetitions, variations of the same lyrical content, and overall structural similarity.

#### 4. **Combined Features** (36 dimensions total)
When multiple extractors are enabled, features are concatenated:
```python
combined_features = torch.cat([head_features, tail_features, string_features], dim=-1)
# Shape: (num_lines, 36) = (num_lines, 12 + 12 + 12)
```

### Feature Configuration

Features are controlled via YAML configuration:

```yaml
features:
  head_ssm:
    enabled: true
    output_dim: 12
    head_words: 2      # Number of words from line start
  tail_ssm:
    enabled: true  
    output_dim: 12
    tail_words: 2      # Number of words from line end
  string_ssm:
    enabled: true
    output_dim: 12
    case_sensitive: false        # Case-insensitive comparison
    remove_punctuation: true     # Remove punctuation before comparison
    similarity_threshold: 0.0    # No minimum threshold
    similarity_method: "word_overlap"  # Fast method: word_overlap, jaccard, levenshtein
  # Future features...
  text_embeddings:
    enabled: false     # Not implemented yet
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    output_dim: 384
```

### Adding New Features

Follow this step-by-step guide to add a new feature extractor:

#### Step 1: Create Feature Module

Create `segmodel/features/your_feature.py`:

```python
import torch
import numpy as np
from typing import List

def extract_your_features(lines: List[str]) -> torch.Tensor:
    """
    Extract your custom features from lines.
    
    Args:
        lines: List of text lines
        
    Returns:
        Feature tensor (seq_len, feature_dim)
    """
    features = []
    
    for line in lines:
        # Your feature extraction logic here
        line_features = [
            len(line),                    # Length feature
            line.count(' '),              # Word count  
            line.count('!'),              # Exclamation count
            # ... add more features ...
        ]
        features.append(line_features)
    
    return torch.tensor(features, dtype=torch.float32)

class YourFeatureExtractor:
    """Modular extractor for your custom features."""
    
    def __init__(self, output_dim: int = 4):
        self.output_dim = output_dim
        
    def __call__(self, lines: List[str]) -> torch.Tensor:
        return extract_your_features(lines)
    
    def get_feature_names(self) -> List[str]:
        return ['line_length', 'word_count', 'exclamation_count', 'etc']
    
    def describe_features(self):
        print("🧩 Your Custom Features:")
        names = self.get_feature_names()
        for i, name in enumerate(names):
            print(f"   {i:2d}. {name}")
```

#### Step 2: Update Feature Extractor

Add your feature to `segmodel/features/extractor.py`:

```python
# Add import at the top
from .your_feature import YourFeatureExtractor

# Add to _setup_extractors method
def _setup_extractors(self):
    # ... existing head_ssm and tail_ssm code ...
    
    # Your feature
    if self.feature_config.get('your_feature', {}).get('enabled', False):
        your_config = self.feature_config['your_feature']
        output_dim = your_config.get('output_dim', 4)
        
        self.extractors['your_feature'] = YourFeatureExtractor(
            output_dim=output_dim
        )
        self.total_dim += output_dim
        enabled_features.append(f"your_feature({output_dim}D)")
```

#### Step 3: Update Configuration System

Add to `segmodel/utils/config_loader.py`:

```python
# Add to TrainingConfig dataclass
@dataclass
class TrainingConfig:
    # ... existing fields ...
    your_feature_enabled: bool = False
    your_feature_dimension: int = 4

# Add to flatten_config function  
def flatten_config(config: Dict[str, Any]) -> TrainingConfig:
    # ... existing code ...
    your_feature = features.get('your_feature', {})
    
    return TrainingConfig(
        # ... existing fields ...
        your_feature_enabled=your_feature.get('enabled', False),
        your_feature_dimension=your_feature.get('output_dim', 4),
    )
```

#### Step 4: Update Training Script

Add to `train_with_config.py`:

```python
# Update feature configuration creation
feature_config = {
    'head_ssm': {
        'enabled': config.head_ssm_enabled,
        'head_words': 2,
        'output_dim': config.head_ssm_dimension
    },
    'tail_ssm': {
        'enabled': config.tail_ssm_enabled,
        'tail_words': 2,
        'output_dim': config.tail_ssm_dimension
    },
    'your_feature': {
        'enabled': config.your_feature_enabled,
        'output_dim': config.your_feature_dimension
    }
}
```

#### Step 5: Update Module Exports

Add to `segmodel/features/__init__.py`:

```python
from .your_feature import (
    YourFeatureExtractor,
    extract_your_features
)

__all__ = [
    # ... existing exports ...
    'YourFeatureExtractor',
    'extract_your_features'
]
```

#### Step 6: Add Configuration

Add to your YAML config file:

```yaml
features:
  head_ssm:
    enabled: true
    output_dim: 12
  tail_ssm:
    enabled: true
    output_dim: 12
  your_feature:
    enabled: true
    output_dim: 4
```

#### Step 7: Test Your Feature

Create a test script:

```python
# Test your feature independently
python -m segmodel.features.your_feature

# Test combined feature extraction  
python -m segmodel.features.extractor

# Test full training pipeline
python train_with_config.py configs/your_config.yaml
```

### Best Practices for New Features

1. **Fixed Dimensions**: Each feature type should have a fixed output dimension for model compatibility
2. **Robust Handling**: Handle edge cases (empty lines, special characters, etc.)
3. **Meaningful Features**: Extract features that capture structural differences between verses and choruses
4. **Documentation**: Include clear descriptions of what each feature dimension represents
5. **Testing**: Test your feature independently before integration
6. **Configuration**: Make your feature optional via YAML configuration

### Feature Ideas for Future Implementation

- **Text Embeddings**: Semantic similarity using sentence transformers
- **Positional Features**: Song structure patterns (intro, verse, chorus, bridge, outro)
- **Rhythmic Features**: Syllable count, stress patterns, meter detection  
- **Semantic Features**: Sentiment, emotion, topic modeling
- **Structural Features**: Line length patterns, punctuation usage
- **Phonetic Features**: Alliteration, assonance, consonance detection

## 📁 Data Format

Expected JSONL format:
```json
{
  "id": "song_001",
  "lines": ["verse line 1", "chorus line 1", "verse line 2"],
  "labels": [0, 1, 0]
}
```

Where: `0 = verse, 1 = chorus`

## 🛡️ Anti-Collapse Features

Based on lessons learned from previous overconfidence issues:

1. **Label Smoothing (0.2)**: Prevents extreme confidence
2. **Weighted Sampling**: Balances chorus exposure per batch
3. **Class Weights**: Handles verse/chorus imbalance
4. **Emergency Monitoring**: Auto-stops if overconfidence detected
5. **Temperature Scaling**: Post-hoc calibration for inference
6. **Gradient Clipping**: Prevents exploding gradients

## 🧪 Testing

Run individual module tests:
```bash
cd segmodel
python -m features.head_ssm      # Test Head-SSM feature extraction
python -m features.tail_ssm      # Test Tail-SSM feature extraction
python -m features.extractor     # Test combined feature extraction
python -m models.blstm_tagger    # Test model architecture  
python -m losses.cross_entropy   # Test loss function
python -m train.trainer          # Test training components
```

## 🎯 Next Steps

### ✅ Completed
- ✅ Head-SSM features for chorus detection
- ✅ Tail-SSM features for rhyme pattern detection  
- ✅ String-SSM features for overall textual similarity
- ✅ Modular feature architecture
- ✅ YAML configuration system
- ✅ Anti-collapse measures
- ✅ **NEW**: Advanced learning rate scheduling (cosine, restarts, step, warmup)
- ✅ **NEW**: Optimized batch size (8→32) with proper LR scaling  
- ✅ **NEW**: Configurable emergency monitoring (no magic numbers)
- ✅ **NEW**: Configurable temperature calibration

### 🚧 In Progress
- Text embeddings (sentence-BERT) integration
- Positional features for song structure
- Advanced rhyme detection algorithms

### 📋 Planned Features
1. **Semantic Features**: Sentence transformers for semantic similarity
2. **Positional Features**: Song section patterns (intro/verse/chorus/bridge/outro)
3. **Rhythmic Features**: Syllable count, stress patterns, meter detection
4. **Advanced Architecture**: Attention mechanisms, Transformer layers
5. **Sequence Modeling**: CRF layers for sequence constraints
6. **Multi-scale Features**: Combine line-level and section-level features

### 🎯 Research Directions
- Unsupervised pre-training on large lyrics corpora
- Multi-task learning (structure + sentiment + topic)
- Few-shot learning for new music genres
- Cross-lingual lyrics segmentation

---

## 🛠️ **Configuration System Guide**

### Overview

The BLSTM system uses a **3-layer configuration architecture** for maximum flexibility and maintainability:

1. **YAML Configuration Files** (`configs/*.yaml`) - Human-readable parameter definitions
2. **TrainingConfig Dataclass** (`segmodel/utils/config_loader.py`) - Structured configuration object
3. **Component Integration** (`trainer.py`, `predict_baseline.py`) - Direct parameter access

### ✅ **How to Add New Configuration Parameters**

Follow this step-by-step process to add any new configurable parameter:

#### Step 1: Add Parameter to YAML Configuration

**File:** `configs/your_config.yaml`

```yaml
# Example: Adding a new advanced scheduler
training:
  batch_size: 32
  learning_rate: 0.001
  
  # ✅ NEW: Your new parameters
  my_new_parameter: true
  my_numeric_parameter: 0.5
  my_list_parameter: [1.0, 2.0, 3.0]

# ✅ NEW: Entire new configuration section
my_new_section:
  enabled: true
  threshold: 0.8
  mode: "advanced"
```

#### Step 2: Add to TrainingConfig Dataclass

**File:** `segmodel/utils/config_loader.py`

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...
    
    # ✅ NEW: Add your parameters with defaults
    my_new_parameter: bool = False
    my_numeric_parameter: float = 0.3
    my_list_parameter: list = None
    
    # New section parameters
    my_new_section_enabled: bool = False
    my_new_section_threshold: float = 0.5
    my_new_section_mode: str = "basic"
    
    def __post_init__(self):
        # ... existing post_init ...
        if self.my_list_parameter is None:
            self.my_list_parameter = [0.5, 1.0, 1.5]
```

#### Step 3: Update Config Loading Function

**File:** `segmodel/utils/config_loader.py`

```python
def flatten_config(config: Dict[str, Any]) -> TrainingConfig:
    # ... existing extractions ...
    training = config.get('training', {})
    my_new_section = config.get('my_new_section', {})
    
    return TrainingConfig(
        # ... existing parameters ...
        
        # ✅ NEW: Extract your parameters
        my_new_parameter=training.get('my_new_parameter', False),
        my_numeric_parameter=training.get('my_numeric_parameter', 0.3),
        my_list_parameter=training.get('my_list_parameter', [0.5, 1.0, 1.5]),
        
        # New section
        my_new_section_enabled=my_new_section.get('enabled', False),
        my_new_section_threshold=my_new_section.get('threshold', 0.5),
        my_new_section_mode=my_new_section.get('mode', 'basic'),
    )
```

#### Step 4: Use Parameters in Components

**File:** `segmodel/train/trainer.py` or other components

```python
class Trainer:
    def __init__(self, config, ...):
        # ✅ Access flattened parameters directly
        if config.my_new_section_enabled:
            threshold = config.my_new_section_threshold
            mode = config.my_new_section_mode
            
            # Use your parameters
            self.my_component = MyNewComponent(
                threshold=threshold,
                mode=mode,
                parameters=config.my_list_parameter
            )
```

### ✅ **How to Add New Scheduler Types**

The system supports adding custom learning rate schedulers:

#### Step 1: Add Scheduler to Factory Function

**File:** `segmodel/train/trainer.py`

```python
def create_scheduler(optimizer, config, total_steps: int = None):
    scheduler_name = getattr(config, 'scheduler', 'plateau')
    
    # ✅ NEW: Add your custom scheduler
    elif scheduler_name == 'my_custom_scheduler':
        my_param1 = getattr(config, 'my_scheduler_param1', 0.1)
        my_param2 = getattr(config, 'my_scheduler_param2', 10)
        
        scheduler = MyCustomScheduler(
            optimizer,
            param1=my_param1,
            param2=my_param2,
            eta_min=float(getattr(config, 'min_lr', 1e-6))
        )
        return scheduler, 'epoch'
    
    # ... existing schedulers ...
```

#### Step 2: Add Parameters to Config

**File:** `configs/your_config.yaml`

```yaml
training:
  scheduler: "my_custom_scheduler"  # ✅ NEW: Your scheduler name
  min_lr: 1e-6
  
  # ✅ NEW: Scheduler-specific parameters
  my_scheduler_param1: 0.05
  my_scheduler_param2: 20
```

### ✅ **How to Update Inference Script**

When adding new features that affect inference:

#### Step 1: Update Feature Configuration

**File:** `predict_baseline.py`

```python
def create_feature_extractor_from_config(config_or_path: str = None):
    if config_or_path is None:
        # ✅ Update default configuration
        feature_config = {
            'existing_feature': { ... },
            
            # ✅ NEW: Add your new feature
            'my_new_feature': {
                'enabled': True,
                'output_dim': 12,
                'my_parameter': 'default_value'
            }
        }
    else:
        # ✅ Extract from training config
        if config.my_new_feature_enabled:
            feature_config['my_new_feature'] = {
                'enabled': True,
                'output_dim': config.my_new_feature_dimension,
                'my_parameter': config.my_new_feature_parameter
            }
```

### ✅ **Configuration Best Practices**

1. **Always Add Defaults**: Every parameter should have a sensible default value
2. **Maintain Backward Compatibility**: New parameters should not break existing configs
3. **Use Descriptive Names**: `batch_size` not `bs`, `learning_rate` not `lr`
4. **Group Related Parameters**: Use sections like `training:`, `emergency_monitoring:`
5. **Document Parameters**: Add comments explaining what each parameter does
6. **Validate Ranges**: Add validation in `validate_config()` for numeric ranges

### ✅ **Testing Configuration Changes**

Always test your configuration changes:

```bash
# Test config loading
python -c "
from segmodel.utils.config_loader import load_training_config
config = load_training_config('configs/your_new_config.yaml')
print(f'New parameter: {config.my_new_parameter}')
"

# Test trainer initialization
python -c "
from segmodel.utils.config_loader import load_training_config
from segmodel.train.trainer import create_scheduler
import torch

config = load_training_config('configs/your_new_config.yaml')
optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=0.001)
scheduler, scheduler_type = create_scheduler(optimizer, config)
print(f'Scheduler: {type(scheduler).__name__}')
"

# Test inference compatibility
python predict_baseline.py \
    --model dummy_model.pt \
    --config configs/your_new_config.yaml \
    --lines "Test line"
```

### ✅ **Common Configuration Patterns**

**Toggle Features:**
```yaml
my_feature:
  enabled: true  # Always include enabled flag
  parameter1: value1
  parameter2: value2
```

**Threshold Parameters:**
```yaml
monitoring:
  threshold: 0.8          # Main threshold
  min_threshold: 0.1      # Minimum bound
  max_threshold: 0.95     # Maximum bound
```

**Mode Selection:**
```yaml
component:
  mode: "advanced"        # Options: "basic", "advanced", "experimental"
  advanced_parameter: 0.5 # Only used when mode="advanced"
```

**List Parameters:**
```yaml
search:
  grid: [0.1, 0.5, 1.0, 2.0]  # Use lists for parameter grids
  default_value: 1.0           # Single default from the grid
```

### ⚠️ **Configuration Gotchas**

1. **YAML Type Conversion**: `1e-6` stays float, but config loader may need `float()` conversion
2. **Nested vs Flattened**: Trainer accesses `config.parameter`, not `config.section.parameter`
3. **Default Handling**: Use `getattr(config, 'param', default)` for optional parameters
4. **List Initialization**: Initialize list parameters in `__post_init__` method
5. **Backward Compatibility**: New parameters should work with old configs (use defaults)

### 📁 **Configuration File Organization**

```
configs/
├── training_config.yaml          # Default production settings
├── quick_test.yaml               # Fast testing (5 epochs)
├── aggressive_config.yaml        # High-performance (our improved version)
├── experimental_*.yaml           # Your experimental configurations
└── production/                   # Production-ready configs
    ├── small_model.yaml          # For limited compute
    ├── large_model.yaml          # For high compute
    └── balanced.yaml             # CPU/GPU balanced
```

This configuration system provides **complete control** over every aspect of training while maintaining **backward compatibility** and **ease of use**. 🚀

---

## 🧪 **Testing the Prediction System**

### Quick Start: Test Your Trained Model

After training a model, you can immediately test it on new lyrics:

```bash
# ✅ RECOMMENDED: Test with training session config (perfect feature compatibility)
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --config training_sessions/session_*/ \
    --text your_lyrics.txt \
    --output results.json

# ✅ Test with specific config file
python predict_baseline.py \
    --model best_model.pt \
    --config configs/aggressive_config.yaml \
    --text data/predict_lyric.txt \
    --temperature 1.5 \
    --output prediction_results.json
```

### Multiple Input Methods

```bash
# From text file
python predict_baseline.py --model best_model.pt --text lyrics.txt

# Interactive input (type lyrics directly)
python predict_baseline.py --model best_model.pt

# Command line arguments
python predict_baseline.py --model best_model.pt --lines "Line 1" "Line 2" "Line 3"

# Multi-line string
python predict_baseline.py --model best_model.pt --lyrics "Walking down the street
Thinking of you today
This is our favorite song"

# From stdin (for scripting)
echo -e "Line 1\nLine 2" | python predict_baseline.py --model best_model.pt --stdin
```

### Understanding the Output

**Terminal Output Example:**
```
🎵 Lyrics Structure Prediction
==================================================
VERSE   (0.894) | [Narrative descriptive content]
CHORUS  (0.693) | [Repetitive emotional hook]
VERSE   (0.831) | [Story-telling content]
CHORUS  (0.803) | [Repeated phrase or emotional line]

📊 Summary:
   Total lines: 24
   Chorus lines: 12 (50.0%)
   Verse lines: 12 (50.0%)
   Avg confidence: 0.765
```

**Key Metrics:**
- **Labels**: VERSE (narrative/descriptive) vs CHORUS (repetitive/emotional hooks)
- **Confidence**: 0.0-1.0 scale, higher = more certain
- **Balance**: Percentage split between verse and chorus sections
- **Calibration**: Well-calibrated models show appropriate uncertainty

### Output File Formats

**JSON Output (--output results.json):**
```json
{
  "model_path": "path/to/model.pt",
  "temperature": 1.5,
  "total_lines": 24,
  "predictions": [
    {
      "line_number": 1,
      "line": "[content]",
      "predicted_label": "verse",
      "confidence": 0.894
    }
  ]
}
```

**Text Summary (generated automatically):**
```
BLSTM Lyrics Structure Prediction Results
Total Lines: 24
Verse Lines: 12 (50.0%)  
Chorus Lines: 12 (50.0%)
Average Confidence: 0.765
```

### Configuration Compatibility

**⚠️ Important**: Always use the same configuration that was used for training!

```bash
# ✅ CORRECT: Use training session config
python predict_baseline.py \
    --model training_sessions/session_20250817_024332_Aggressive_Maximum_Performance_v1/best_model.pt \
    --config training_sessions/session_20250817_024332_Aggressive_Maximum_Performance_v1/ \
    --text test_lyrics.txt

# ✅ CORRECT: Use original config file
python predict_baseline.py \
    --model my_model.pt \
    --config configs/aggressive_config.yaml \
    --text test_lyrics.txt

# ⚠️ FALLBACK: Uses comprehensive default features (may not match exactly)
python predict_baseline.py \
    --model my_model.pt \
    --text test_lyrics.txt
```

### Feature Compatibility Check

The system automatically validates feature compatibility:

```bash
# This will show feature dimension matching
python predict_baseline.py --model my_model.pt --config my_config.yaml --lines "test"
```

**Expected Output:**
```
🔧 Detected model architecture:
   Input size: 60           # Model expects 60D features
   Hidden size: 512
   Classes: 2

🧩 Initialized feature extractor:
   Total dimension: 60      # ✅ Perfect match!
```

**If dimensions don't match:**
- Check your config file matches the training setup
- Ensure all feature extractors are enabled/disabled correctly
- Use the training session directory as config source

### Temperature Calibration

Adjust confidence calibration with the `--temperature` parameter:

```bash
# Default: T=1.5 (recommended for most cases)
python predict_baseline.py --model my_model.pt --text lyrics.txt

# Lower temperature: More confident predictions
python predict_baseline.py --model my_model.pt --text lyrics.txt --temperature 1.0

# Higher temperature: More uncertain, conservative predictions  
python predict_baseline.py --model my_model.pt --text lyrics.txt --temperature 2.0
```

**Temperature Effects:**
- **T < 1.0**: Sharper, more confident predictions (use cautiously)
- **T = 1.0**: Raw model predictions (no calibration)
- **T = 1.5**: Recommended default (good calibration balance)
- **T > 2.0**: More uncertain, conservative predictions

### Troubleshooting

**Common Issues and Solutions:**

1. **Dimension Mismatch Error**
   ```bash
   RuntimeError: mat1 and mat2 shapes cannot be multiplied
   ```
   **Solution**: Feature extractor dimension doesn't match model. Use correct config file.

2. **Missing Features**
   ```bash
   ModuleNotFoundError: No module named 'some_feature'
   ```
   **Solution**: Ensure all required feature dependencies are installed.

3. **Config Validation Error**
   ```bash
   ValueError: Missing required config section: data
   ```
   **Solution**: Use training config file, not training snapshot, or use fallback mode.

4. **Low Confidence Predictions**
   - **Normal**: Model appropriately uncertain about ambiguous content
   - **Fix**: Try different temperature values or retrain with more data

5. **All Predictions Same Label**
   - **Issue**: Model collapse or incorrect features
   - **Fix**: Check feature extraction, try different model checkpoint

### Performance Benchmarking

Test your model's performance:

```bash
# Time a prediction run
time python predict_baseline.py --model my_model.pt --text large_file.txt

# Memory usage monitoring
python -m memory_profiler predict_baseline.py --model my_model.pt --text test.txt
```

**Expected Performance:**
- **Speed**: ~10-50 lines/second (depending on hardware)
- **Memory**: <100MB for inference
- **Startup**: 2-5 seconds (loading model + features)

### Example Test Workflow

```bash
# 1. Train a model
python train_with_config.py configs/aggressive_config.yaml

# 2. Create organized results directory
mkdir -p prediction_results

# 3. Test on sample lyrics with organized output
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --config configs/aggressive_config.yaml \
    --text data/predict_lyric.txt \
    --output prediction_results/prediction_results.json \
    --temperature 1.5

# 4. Create summary report
cat > prediction_results/prediction_results_summary.txt << EOF
BLSTM Lyrics Structure Prediction Results
=========================================
Model: $(basename training_sessions/session_*/best_model.pt)
Config: configs/aggressive_config.yaml
Date: $(date +%Y-%m-%d)
Lines Processed: $(wc -l < data/predict_lyric.txt)
EOF

# 5. Review organized results
ls -la prediction_results/
# Expected files:
# - prediction_results.json      (detailed predictions)
# - prediction_results_summary.txt (human-readable summary)
# - predict_lyric.txt            (input lyrics)
# - predict_lyric_answer.txt     (reference structure)

# 6. Analyze performance
python -c "
import json
with open('prediction_results/prediction_results.json') as f:
    data = json.load(f)
predictions = data['predictions']
avg_conf = sum(p['confidence'] for p in predictions) / len(predictions)
chorus_count = sum(1 for p in predictions if p['predicted_label'] == 'chorus')
verse_count = len(predictions) - chorus_count

print(f'📊 Prediction Analysis:')
print(f'   Total lines: {len(predictions)}')
print(f'   Verse lines: {verse_count} ({verse_count/len(predictions):.1%})')
print(f'   Chorus lines: {chorus_count} ({chorus_count/len(predictions):.1%})')
print(f'   Avg confidence: {avg_conf:.3f}')
print(f'   High confidence (>0.8): {sum(1 for p in predictions if p[\"confidence\"] > 0.8)}')
print(f'   Low confidence (<0.6): {sum(1 for p in predictions if p[\"confidence\"] < 0.6)}')
"
```

### Results Organization

Your prediction results should be organized like this:

```
prediction_results/
├── prediction_results.json         # Detailed JSON output
├── prediction_results_summary.txt  # Human-readable summary
├── predict_lyric.txt               # Input lyrics
└── predict_lyric_answer.txt        # Reference structure (if available)
```

This organized structure allows for:
- ✅ **Easy Comparison**: Input, output, and reference in one place
- ✅ **Version Control**: Track prediction improvements over time
- ✅ **Analysis**: Both machine-readable (JSON) and human-readable formats
- ✅ **Documentation**: Clear summary of model performance and configuration

This testing system provides **comprehensive validation** of your trained models with **flexible input methods** and **detailed output analysis**. 🎯

---

## 🛠️ **Developer Guides**

### 📖 **Complete Developer Documentation**

For comprehensive guides on extending and maintaining this system, see:
**[📋 Developer Guide](documentation/DEVELOPER_GUIDE.md)**

**Quick Reference:**
- 🧩 **Adding New Features**: Step-by-step guide for feature extractors
- ⚙️ **Adding Configuration Parameters**: YAML → Dataclass → Code integration  
- 📝 **Editing Configurations**: Safe editing and validation procedures
- 🔧 **System Updates**: Schedulers, models, feature system changes
- 🧪 **Testing Procedures**: Complete validation workflows
- 🚨 **Emergency Rollback**: When something breaks

### 🎯 **Quick Start for Developers**

```bash
# 1. Test your changes don't break configs
python -c "from segmodel.utils.config_loader import load_training_config; load_training_config('configs/training/aggressive_config.yaml')"

# 2. Quick training test (2 epochs)  
python train_with_config.py configs/training/aggressive_config.yaml --epochs 2

# 3. Test prediction compatibility
python predict_baseline.py --model training_sessions/session_*/best_model.pt --prediction-config configs/prediction/default.yaml --quiet

# 4. Full system validation
bash scripts/test_full_system.sh  # (if available)
```

### 📋 **Training Output Configuration Reporting**

The training system now displays comprehensive configuration details in:

**Final Results File:** `training_sessions/session_*/final_results.txt`
```
Configuration: configs/training/aggressive_config.yaml
Training time: 1.9 minutes
Feature dimension: 60

Feature Configuration:
-------------------------
  Head-SSM: Enabled (12D)
  Tail-SSM: Enabled (12D) 
  Phonetic-SSM: Enabled (12D)
    Mode: rhyme, Similarity: binary, Threshold: 0.4
  POS-SSM: Enabled (12D)  
    Tagset: simplified, Similarity: combined, Threshold: 0.3
  String-SSM: Enabled (12D)
    Threshold: 0.1

Model Architecture:
------------------
  Hidden dimension: 512, Dropout: 0.2
  Batch size: 32, Learning rate: 0.001
  Label smoothing: 0.15, Weighted sampling: True

Test Results:
-------------
  Macro F1: 0.6921, Verse F1: 0.8103, Chorus F1: 0.5740
  Confidence: 0.625, Chorus rate: 31.32%
  Optimal Temperature: 0.80
```
