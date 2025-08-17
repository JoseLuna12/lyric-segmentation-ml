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

**With YAML Configuration (Recommended):**
```bash
# Use predefined configurations
python train_with_config.py configs/training_config.yaml
python train_with_config.py configs/quick_test.yaml
python train_with_config.py configs/aggressive_training.yaml

# With command line overrides
python train_with_config.py configs/training_config.yaml \
    --batch-size 32 \
    --learning-rate 0.0005 \
    --epochs 100 \
    --disable-emergency-monitoring
```

**Legacy Command Line (Still supported):**
```bash
python train_baseline.py \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --test data/test.jsonl \
    --epochs 60 \
    --batch-size 8 \
    --lr 1e-3
```

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

### Inference

**Multiple input methods supported:**

```bash
# Multi-line string (most convenient)
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --lyrics "Walking down the street tonight
Looking at the stars above
This is our song we sing
Dancing to the beat of love"

# From text file
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --text lyrics.txt

# Interactive input (default - just run without input args)
python predict_baseline.py --model training_sessions/session_*/best_model.pt

# Command line arguments
python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt \
    --lines "Walking down the street" "Thinking of you" "Dancing tonight"

# From pipe/stdin
echo "Line 1\nLine 2" | python predict_baseline.py \
    --model training_sessions/session_*/best_model.pt --stdin
```

## 🧩 Features

- **Modular Design**: Easy to extend with new features
- **Anti-Collapse**: Prevents overconfidence with label smoothing + class weights
- **Weighted Sampling**: Ensures balanced training despite class imbalance  
- **Temperature Calibration**: Post-hoc confidence calibration
- **Emergency Monitoring**: Real-time training guardrails
- **Head-SSM Features**: Line-beginning similarity for chorus detection
- **Tail-SSM Features**: Line-ending similarity for rhyme pattern detection
- **Comprehensive Metrics**: Incremental saving of training progress with analysis tools
- **YAML Configuration**: Flexible configuration system with command-line overrides and reproducibility snapshots

## 📊 Expected Performance

**With Head-SSM + Tail-SSM + String-SSM Features:**
- **Accuracy**: 75-85% (improved with triple features)
- **F1 Macro**: 0.70-0.80
- **Verse F1**: 0.80-0.90 (high precision)
- **Chorus F1**: 0.60-0.75 (challenging due to variety)
- **Confidence**: Well-calibrated (avg ~0.70-0.80)
- **Training Time**: ~60-120 minutes on modern hardware
- **Feature Dimension**: 36D (12D Head-SSM + 12D Tail-SSM + 12D String-SSM)

## 🔧 Configuration

### YAML Configuration Files

The recommended way to manage training parameters is through YAML configuration files:

**Available Configurations:**
- `configs/training_config.yaml` - Default production settings
- `configs/quick_test.yaml` - Fast testing (5 epochs, smaller model)
- `configs/aggressive_training.yaml` - High-performance with strict anti-collapse

**Example YAML config:**
```yaml
# Training parameters
training:
  batch_size: 16
  learning_rate: 0.001
  max_epochs: 60
  patience: 8

# Anti-collapse settings  
anti_collapse:
  label_smoothing: 0.2
  weighted_sampling: true
  entropy_lambda: 0.0

# Emergency monitoring
emergency_monitoring:
  enabled: true
  max_confidence_threshold: 0.95
  
# Experiment metadata
experiment:
  name: "my_experiment"
  description: "Testing new architecture"
  tags: ["baseline", "test"]
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
