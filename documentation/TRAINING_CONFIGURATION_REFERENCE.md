# 🎛️ Training Configuration Reference

**BiLSTM Text Segmentation System - Complete Configuration Documentation**

This document provides factual documentation of all available training configuration options.

---

## 📋 **Configuration Structure**

Configuration files use YAML format with the following main sections:

- `data` - Dataset file paths
- `model` - Model architecture parameters
- `training` - Training process parameters  
- `anti_collapse` - Anti-collapse system settings
- `emergency_monitoring` - Real-time monitoring thresholds
- `temperature_calibration` - Post-training calibration settings (deprecated)
- `calibration` - Modern calibration system with multiple methods
- `features` - Feature extractor configurations
- `output` - Output and saving settings
- `system` - Runtime and platform settings
- `experiment` - Metadata and tracking

---

## 📂 **Data Configuration**

### Required Parameters
```yaml
data:
  train_file: "path/to/train.jsonl"
  val_file: "path/to/val.jsonl"  
  test_file: "path/to/test.jsonl"
```

**Parameter Details:**
- `train_file` - Path to training dataset file
- `val_file` - Path to validation dataset file
- `test_file` - Path to test dataset file

**Expected Format:** JSONL with structure: `{"id": "song_id", "lines": ["line1", "line2"], "labels": [0, 1]}`
**Labels:** `0 = verse, 1 = chorus`

---

## 🏗️ **Model Architecture**

```yaml
model:
  hidden_dim: 256
  num_layers: 2
  num_classes: 2
  dropout: 0.2
  layer_dropout: 0.3
  
  # Attention Mechanism (Optional)
  attention_enabled: true
  attention_type: "self"  # "self", "localized", "boundary_aware"
  attention_heads: 8
  attention_dropout: 0.15
  attention_dim: 256  # If null, uses BiLSTM output dimension
  positional_encoding: true
  max_seq_length: 1000
  window_size: 7  # For localized attention
  boundary_temperature: 2.0  # For boundary-aware attention
```

**Core Parameters:**
- `hidden_dim` - BiLSTM hidden state dimension (integer, recommended: 128-512)
- `num_layers` - Number of BiLSTM layers (integer, 1-3, default: 2)
- `num_classes` - Number of output classes (integer, fixed at 2 for verse/chorus)
- `dropout` - Output dropout probability for regularization (float 0.0-1.0)
- `layer_dropout` - Inter-layer dropout for multi-layer models (float 0.0-1.0)

**Attention Parameters (Optional):**
- `attention_enabled` - Whether to use attention mechanism (boolean)
- `attention_type` - Type of attention mechanism:
  - `"self"` - Standard self-attention (global context)
  - `"localized"` - Window-based local attention (efficient for long sequences)
  - `"boundary_aware"` - Temperature-scaled attention for boundary detection
- `attention_heads` - Number of attention heads (integer, 1-16, default: 8)
- `attention_dropout` - Attention dropout probability (float 0.0-1.0)
- `attention_dim` - Attention dimension (integer, if null uses BiLSTM output dimension)
- `positional_encoding` - Whether to use positional encoding (boolean, recommended: true)
- `max_seq_length` - Maximum sequence length for positional encoding (integer, 500-2000)
- `window_size` - Window size for localized attention (integer, 3-15, default: 7)
- `boundary_temperature` - Temperature for boundary-aware attention (float, 1.0-5.0, default: 2.0)

**Architecture Notes:**
- Multi-layer BiLSTM (num_layers > 1) improves capacity but increases parameters
- Attention mechanism adds ~10-15% parameters but often improves performance
- Positional encoding helps with long sequences and structure detection

---

## 🎯 **Training Parameters**

```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.005
  max_epochs: 120
  patience: 25
  gradient_clip_norm: 0.5
```

**Basic Parameters:**
- `batch_size` - Number of samples per batch (integer)
- `learning_rate` - Initial learning rate (float)
- `weight_decay` - L2 regularization coefficient (float)
- `max_epochs` - Maximum number of training epochs (integer)
- `patience` - Early stopping patience in epochs (integer)
- `gradient_clip_norm` - Gradient clipping threshold (float)

### Learning Rate Scheduling

```yaml
training:
  scheduler: "cosine"
  min_lr: 1e-6
  cosine_t_max: 120
  warmup_epochs: 5
  lr_factor: 0.5
  lr_patience: 10
  step_size: 30
  step_gamma: 0.5
  cosine_t0: 10
  cosine_t_mult: 2
```

**Scheduler Options:**
- `scheduler` - Scheduler type: `"plateau"`, `"cosine"`, `"cosine_restarts"`, `"step"`, `"warmup_cosine"`
- `min_lr` - Minimum learning rate for schedulers (float)

**Plateau Scheduler:**
- `lr_factor` - Factor to multiply LR when plateau detected (float)
- `lr_patience` - Epochs to wait before LR reduction (integer)

**Cosine Scheduler:**
- `cosine_t_max` - Period length for cosine annealing (integer)

**Cosine Restarts Scheduler:**
- `cosine_t0` - Initial restart period (integer)
- `cosine_t_mult` - Period multiplier after each restart (integer)

**Step Scheduler:**
- `step_size` - Reduce LR every N epochs (integer)
- `step_gamma` - Factor to multiply LR (float)

**Warmup Cosine Scheduler:**
- `warmup_epochs` - Linear warmup period (integer)

---

## 🛡️ **Anti-Collapse System**

```yaml
anti_collapse:
  label_smoothing: 0.15
  weighted_sampling: true
  entropy_lambda: 0.0
```

**Parameters:**
- `label_smoothing` - Label smoothing factor (float 0.0-1.0)
- `weighted_sampling` - Enable balanced sampling per batch (boolean)
- `entropy_lambda` - Entropy regularization weight (float)

---

## 🚨 **Emergency Monitoring**

```yaml
emergency_monitoring:
  enabled: true
  
  # Batch-level thresholds
  max_confidence_threshold: 0.95
  min_chorus_rate: 0.05
  max_chorus_rate: 0.85
  max_conf_over_95_ratio: 0.15
  
  # Epoch-level thresholds
  val_overconf_threshold: 0.97
  val_f1_collapse_threshold: 0.05
  emergency_overconf_threshold: 0.98
  emergency_conf95_ratio: 0.9
  emergency_f1_threshold: 0.02
  
  # Timing
  skip_batches: 50
  skip_epochs: 5
  print_batch_every: 20
```

**Control Parameters:**
- `enabled` - Enable/disable monitoring system (boolean)

**Batch-Level Monitoring:**
- `max_confidence_threshold` - Maximum average confidence per batch (float)
- `min_chorus_rate` - Minimum chorus prediction rate per batch (float)
- `max_chorus_rate` - Maximum chorus prediction rate per batch (float)  
- `max_conf_over_95_ratio` - Maximum ratio of predictions >95% confident (float)

**Epoch-Level Monitoring:**
- `val_overconf_threshold` - Validation overconfidence warning threshold (float)
- `val_f1_collapse_threshold` - F1 collapse detection threshold (float)
- `emergency_overconf_threshold` - Emergency stop overconfidence threshold (float)
- `emergency_conf95_ratio` - Emergency stop confidence ratio threshold (float)
- `emergency_f1_threshold` - Emergency stop F1 threshold (float)

**Timing Parameters:**
- `skip_batches` - Skip monitoring for first N batches (integer)
- `skip_epochs` - Skip monitoring for first N epochs (integer)
- `print_batch_every` - Print batch info frequency (integer)

---

## 🌡️ **Calibration System**

### Modern Calibration (Recommended)
```yaml
calibration:
  methods: ['temperature', 'platt', 'isotonic']  # List of methods to try
  enabled: true                                  # Enable calibration fitting
```

**Parameters:**
- `methods` - List of calibration methods to evaluate (array of strings)
  - `'temperature'` - Single temperature parameter scaling
  - `'platt'` - Sigmoid-based Platt scaling (2 parameters)
  - `'isotonic'` - Non-parametric isotonic regression 
- `enabled` - Whether to fit calibration on validation data (boolean)

**Behavior:**
- All specified methods are fitted on validation data
- Best method (lowest ECE) is automatically selected
- Results saved to `calibration.json` in session directory

### Legacy Temperature Calibration (Deprecated)
```yaml
temperature_calibration:
  temperature_grid: [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2]
  default_temperature: 1.0
```

**Note:** Use the modern `calibration` system instead.

---

## 🎯 **Validation Strategy (Phase 3)**

```yaml
# Simple validation strategy selection
validation_strategy: "boundary_f1"
```

**Available Strategies:**
- `"line_f1"` - Line-level macro F1 score (original method)
- `"boundary_f1"` - **Boundary detection F1 (RECOMMENDED for structural understanding)**
- `"windowdiff"` - WindowDiff metric (forgiving boundary evaluation)
- `"pk"` - Pk metric (penalty-based boundary evaluation)
- `"segment_iou"` - Segment IoU (complete segment quality focus)
- `"composite"` - Balanced combination (boundary=40%, line=25%, segment=25%, window=10%)

**Recommendations:**
- **For verse/chorus segmentation**: Use `"boundary_f1"` (detects section boundaries)
- **For research comparison**: Use `"windowdiff"` or `"pk"` (standard text segmentation)
- **For balanced approach**: Use `"composite"` (optimizes multiple objectives)
- **For backward compatibility**: Use `"line_f1"` (original behavior)

---

## 🧩 **Feature System**

The system supports 9 feature extractors with configurable dimensions and parameters:

1. **Head-SSM** - Analyzes line beginnings
2. **Tail-SSM** - Analyzes line endings
3. **Phonetic-SSM** - Phonetic similarity patterns
4. **POS-SSM** - Part-of-speech patterns
5. **String-SSM** - String similarity patterns
6. **SyllablePattern-SSM** - Syllable pattern similarity (NEW)
7. **LineSyllable-SSM** - Line-level syllable analysis (NEW)
8. **Word2Vec Embeddings** - Pre-trained word embeddings
9. **Contextual Embeddings** - Contextual sentence embeddings

### 1. Head-SSM Features

```yaml
features:
  head_ssm:
    enabled: true
    dimension: 12
    head_words: 2
```

**Parameters:**
- `enabled` - Enable/disable feature extractor (boolean)
- `dimension` - Output feature dimensions (integer)
- `head_words` - Number of words from line start to analyze (integer, 1-5)

### 2. Tail-SSM Features

```yaml
features:
  tail_ssm:
    enabled: true
    dimension: 12
    tail_words: 2
```

**Parameters:**
- `enabled` - Enable/disable feature extractor (boolean)
- `dimension` - Output feature dimensions (integer)
- `tail_words` - Number of words from line end to analyze (integer, 1-5)

### 3. Phonetic-SSM Features

```yaml
features:
  phonetic_ssm:
    enabled: true
    dimension: 12
    mode: "rhyme"
    similarity_method: "binary"
    normalize: false
    normalize_method: "zscore"
    high_sim_threshold: 0.4
```

**Parameters:**
- `enabled` - Enable/disable feature extractor (boolean)
- `dimension` - Output feature dimensions (integer)
- `mode` - Analysis mode: `"rhyme"` or `"alliteration"`
- `similarity_method` - Computation method: `"binary"` or `"weighted"`
- `normalize` - Enable feature normalization (boolean)
- `normalize_method` - Normalization method: `"zscore"` or `"minmax"`
- `high_sim_threshold` - Threshold for high similarity classification (float, 0.0-1.0)

### 4. POS-SSM Features

```yaml
features:
  pos_ssm:
    enabled: true
    dimension: 12
    tagset: "simplified"
    similarity_method: "combined"
    high_sim_threshold: 0.3
```

**Parameters:**
- `enabled` - Enable/disable feature extractor (boolean)
- `dimension` - Output feature dimensions (integer)
- `tagset` - POS tagset type: `"simplified"` or `"full"`
- `similarity_method` - Similarity computation: `"exact"`, `"similarity"`, or `"combined"`
- `high_sim_threshold` - Threshold for high similarity classification (float, 0.0-1.0)

### 5. String-SSM Features

```yaml
features:
  string_ssm:
    enabled: true
    dimension: 12
    case_sensitive: false
    remove_punctuation: true
    similarity_threshold: 0.1
    similarity_method: "word_overlap"
```

**Parameters:**
- `enabled` - Enable/disable feature extractor (boolean)
- `dimension` - Output feature dimensions (integer)
- `case_sensitive` - Enable case-sensitive comparison (boolean)
- `remove_punctuation` - Remove punctuation before comparison (boolean)
- `similarity_threshold` - Minimum similarity threshold (float, 0.0-1.0)
- `similarity_method` - Computation method: `"word_overlap"`, `"jaccard"`, or `"levenshtein"`

### 6. SyllablePattern-SSM Features (NEW)

```yaml
features:
  syllable_pattern_ssm:
    enabled: true
    dimension: 12
    similarity_method: "cosine"
    levenshtein_weight: 0.7
    cosine_weight: 0.3
    normalize: false
    normalize_method: "zscore"
```

**Parameters:**
- `enabled` - Enable/disable feature extractor (boolean)
- `dimension` - Output feature dimensions (integer)
- `similarity_method` - Method: `"levenshtein"`, `"cosine"`, or `"combined"`
- `levenshtein_weight` - Weight for Levenshtein distance in combined method (float, 0.0-1.0)
- `cosine_weight` - Weight for cosine similarity in combined method (float, 0.0-1.0)
- `normalize` - Enable feature normalization (boolean)
- `normalize_method` - Normalization method: `"zscore"` or `"minmax"`

**Description:** Analyzes syllable pattern similarity between lines using phonetic syllable counts and stress patterns.

### 7. LineSyllable-SSM Features (NEW)

```yaml
features:
  line_syllable_ssm:
    enabled: true
    dimension: 12
    similarity_method: "cosine"
    ratio_threshold: 0.1
    normalize: false
    normalize_method: "zscore"
```

**Parameters:**
- `enabled` - Enable/disable feature extractor (boolean)
- `dimension` - Output feature dimensions (integer)
- `similarity_method` - Method: `"ratio"`, `"cosine"`, or `"combined"`
- `ratio_threshold` - Threshold for syllable count ratio similarity (float, 0.0-1.0)
- `normalize` - Enable feature normalization (boolean)
- `normalize_method` - Normalization method: `"zscore"` or `"minmax"`

**Description:** Analyzes line-level syllable count patterns and relationships for verse/chorus structure detection.

### 8. Word2Vec Embeddings

```yaml
features:
  word2vec:
    enabled: true
    model: "word2vec-google-news-300"
    mode: "complete"
    normalize: true
    similarity_metric: "cosine"
    high_sim_threshold: 0.8
```

**Parameters:**
- `enabled` - Enable/disable feature extractor (boolean)
- `model` - Model name: `"word2vec-google-news-300"` (300D vectors)
- `mode` - Output mode: `"complete"` (300D) or `"summary"` (12D statistical features)
- `normalize` - Normalize embeddings before processing (boolean)
- `similarity_metric` - Similarity computation: `"cosine"` or `"euclidean"`
- `high_sim_threshold` - Threshold for high similarity detection (float, 0.0-1.0)

**Output Dimensions:**
- `complete` mode: 300D (full word2vec embeddings)
- `summary` mode: 12D (statistical summary features)

### 9. Contextual Embeddings

```yaml
features:
  contextual:
    enabled: true
    model: "all-MiniLM-L6-v2"
    mode: "complete"
    normalize: true
    similarity_metric: "cosine"
    high_sim_threshold: 0.7
```

**Parameters:**
- `enabled` - Enable/disable feature extractor (boolean)
- `model` - SentenceTransformer model: `"all-MiniLM-L6-v2"` (384D) or `"all-mpnet-base-v2"` (768D)
- `mode` - Output mode: `"complete"` (full embeddings) or `"summary"` (12D statistical features)
- `normalize` - Normalize embeddings before processing (boolean)
- `similarity_metric` - Similarity computation: `"cosine"` or `"euclidean"`
- `high_sim_threshold` - Threshold for high similarity detection (float, 0.0-1.0)

**Output Dimensions:**
- `complete` mode: 384D (all-MiniLM-L6-v2) or 768D (all-mpnet-base-v2)
- `summary` mode: 12D (statistical summary features)

**Supported Models:**
- `"all-MiniLM-L6-v2"` - Lightweight, fast (384D)
- `"all-mpnet-base-v2"` - Higher quality, slower (768D)

### Feature Dimension Calculation

The total feature dimension is the sum of all enabled features:

```
Total = Head-SSM + Tail-SSM + Phonetic-SSM + POS-SSM + String-SSM + 
        SyllablePattern-SSM + LineSyllable-SSM + Word2Vec + Contextual

Example (all features enabled):
12 + 12 + 12 + 12 + 12 + 12 + 12 + 300 + 384 = 768D
```

---

## 📁 **Output Configuration**

```yaml
output:
  base_dir: "training_sessions"
  save_best_model: true
  save_final_model: true
  save_training_metrics: true
  save_config_snapshot: true
```

**Parameters:**
- `base_dir` - Base directory for training outputs (string)
- `save_best_model` - Save best performing model (boolean)
- `save_final_model` - Save final epoch model (boolean)
- `save_training_metrics` - Save training history JSON (boolean)
- `save_config_snapshot` - Save configuration snapshot (boolean)

**Output Structure:**
Each training session creates a directory: `training_sessions/session_YYYYMMDD_HHMMSS_ExperimentName/`

**Generated Files:**
- `best_model.pt` - Best performing model checkpoint
- `final_model.pt` - Final epoch model checkpoint
- `training_config_snapshot.yaml` - Exact configuration used
- `training_metrics.json` - Complete training history data
- `boundary_metrics_summary.json` - Boundary analysis metrics
- `final_results.txt` - Human-readable results summary

---

## ⚙️ **System Settings**

```yaml
system:
  seed: 42
  device: "auto"
  num_workers: 0
  deterministic: true
```

**Parameters:**
- `seed` - Random seed for reproducibility (integer)
- `device` - Device selection: `"auto"`, `"cpu"`, `"cuda"`, or `"mps"`
- `num_workers` - DataLoader worker processes (integer)
- `deterministic` - Enable deterministic operations (boolean)

**Device Selection:**
- `"auto"` - Automatically selects best available device (MPS > CUDA > CPU)
- `"cpu"` - Force CPU usage
- `"cuda"` - Force CUDA GPU usage (if available)
- `"mps"` - Force Apple Silicon GPU usage (if available)

---

## 🏷️ **Experiment Metadata**

```yaml
experiment:
  name: "experiment_name"
  description: "Experiment description"
  tags: ["tag1", "tag2"]
  notes: "Additional notes"
```

**Parameters:**
- `name` - Experiment identifier (string)
- `description` - Brief experiment description (string)  
- `tags` - Array of searchable tags (array of strings)
- `notes` - Detailed notes and comments (string)

---

## 📊 **Evaluation Metrics**

The system automatically computes and exports the following metrics:

### Line-Level Metrics
- `val_macro_f1` - Macro-averaged F1 score
- `val_verse_f1` - Verse class F1 score
- `val_chorus_f1` - Chorus class F1 score

### Boundary-Aware Metrics (Phase 2)
- `val_boundary_f1` - Boundary detection F1 score
- `val_boundary_precision` - Boundary detection precision
- `val_boundary_recall` - Boundary detection recall
- `val_complete_segments` - Complete segment detection rate
- `val_avg_segment_overlap` - Average segment overlap (IoU)
- `val_verse_to_chorus_acc` - Verse-to-chorus transition accuracy
- `val_chorus_to_verse_acc` - Chorus-to-verse transition accuracy

### Segmentation Metrics (Phase 3)
- `val_window_diff` - WindowDiff metric (lower is better)
- `val_pk_metric` - Pk metric (lower is better)

### Confidence Metrics
- `val_chorus_rate` - Validation chorus prediction rate
- `val_max_prob` - Maximum prediction confidence
- `val_conf_over_90` - Ratio of predictions >90% confident
- `val_conf_over_95` - Ratio of predictions >95% confident

---

## 📋 **Complete Configuration Example**

```yaml
# Data Configuration
data:
  train_file: "data/train.jsonl"
  val_file: "data/val.jsonl"
  test_file: "data/test.jsonl"

# Model Architecture
model:
  hidden_dim: 512
  num_classes: 2
  dropout: 0.2

# Training Parameters
training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.005
  max_epochs: 120
  patience: 25
  gradient_clip_norm: 0.5
  scheduler: "cosine"
  min_lr: 1e-6
  cosine_t_max: 120

# Anti-Collapse System
anti_collapse:
  label_smoothing: 0.15
  weighted_sampling: true
  entropy_lambda: 0.0

# Emergency Monitoring
emergency_monitoring:
  enabled: true
  max_confidence_threshold: 0.95
  min_chorus_rate: 0.05
  max_chorus_rate: 0.85
  max_conf_over_95_ratio: 0.15
  val_overconf_threshold: 0.97
  val_f1_collapse_threshold: 0.05
  emergency_overconf_threshold: 0.98
  emergency_conf95_ratio: 0.9
  emergency_f1_threshold: 0.02
  skip_batches: 50
  skip_epochs: 5
  print_batch_every: 20

# Temperature Calibration
temperature_calibration:
  temperature_grid: [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2]
  default_temperature: 1.0

# Validation Strategy (Phase 3) - Simple selection
validation_strategy: "boundary_f1"

# Feature Configuration
features:
  head_ssm:
    enabled: true
    dimension: 12
    head_words: 2
  tail_ssm:
    enabled: true
    dimension: 12
    tail_words: 2
  phonetic_ssm:
    enabled: true
    dimension: 12
    mode: "rhyme"
    similarity_method: "binary"
    normalize: false
    normalize_method: "zscore"
    high_sim_threshold: 0.4
  pos_ssm:
    enabled: true
    dimension: 12
    tagset: "simplified"
    similarity_method: "combined"
    high_sim_threshold: 0.3
  string_ssm:
    enabled: true
    dimension: 12
    case_sensitive: false
    remove_punctuation: true
    similarity_threshold: 0.1
    similarity_method: "word_overlap"
  syllable_pattern_ssm:
    enabled: true
    dimension: 12
    similarity_method: "cosine"
    levenshtein_weight: 0.7
    cosine_weight: 0.3
    normalize: false
    normalize_method: "zscore"
  line_syllable_ssm:
    enabled: true
    dimension: 12
    similarity_method: "cosine"
    ratio_threshold: 0.1
    normalize: false
    normalize_method: "zscore"
  word2vec:
    enabled: true
    model: "word2vec-google-news-300"
    mode: "complete"
    normalize: true
    similarity_metric: "cosine"
    high_sim_threshold: 0.8
  contextual:
    enabled: true
    model: "all-MiniLM-L6-v2"
    mode: "complete"
    normalize: true
    similarity_metric: "cosine"
    high_sim_threshold: 0.7

# Output Configuration
output:
  base_dir: "training_sessions"
  save_best_model: true
  save_final_model: true
  save_training_metrics: true
  save_config_snapshot: true

# System Settings
system:
  seed: 42
  device: "auto"
  num_workers: 0
  deterministic: true

# Experiment Metadata
experiment:
  name: "blstm_experiment"
  description: "BiLSTM training experiment"
  tags: ["blstm", "verse-chorus", "60d-features"]
  notes: "Complete configuration example"
```

---

## 🔧 **Configuration Loading**

Configuration files are loaded via the `TrainingConfig` dataclass in `segmodel/utils/config_loader.py`.

**Usage:**
```bash
python train_with_config.py path/to/config.yaml
```

**Command Line Overrides:**
```bash
python train_with_config.py config.yaml --batch-size 64 --learning-rate 0.002
```

**Available Overrides:**
- `--batch-size` - Override batch size
- `--learning-rate` - Override learning rate
- `--epochs` - Override max epochs
- `--label-smoothing` - Override label smoothing
- `--entropy-lambda` - Override entropy lambda
- `--disable-emergency-monitoring` - Disable emergency monitoring
- `--train` - Override training file path
- `--val` - Override validation file path
- `--test` - Override test file path

---

This document provides complete factual reference for all configuration options available in the BiLSTM training system.
