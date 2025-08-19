# 🎯 Attention Mechanism Integration Guide

**BiLSTM Text Segmentation System - Phase 2 Implementation**

This document explains how to use the newly implemented attention mechanism in the BiLSTM text segmentation system.

---

## 📋 **Overview**

Phase 2 adds a comprehensive multi-head self-attention mechanism to the BiLSTM architecture, enabling the model to better capture long-range dependencies and improve sequence understanding.

### Key Features

- **Multi-head self-attention** with configurable head count (4-16 heads)
- **Optional positional encoding** for sequence position awareness
- **Residual connections** for stable training
- **Attention weight analysis** for interpretability
- **Complete backward compatibility** with existing configurations

---

## 🚀 **Quick Start**

### Enable Attention in Your Configuration

Add these parameters to your model configuration:

```yaml
model:
  hidden_dim: 256
  num_layers: 2
  dropout: 0.2
  
  # NEW: Attention Configuration
  attention_enabled: true          # Enable attention mechanism
  attention_heads: 8               # Number of attention heads
  attention_dropout: 0.1           # Attention dropout rate
  attention_dim: null              # Use LSTM output dimension
  positional_encoding: true        # Enable positional encoding
  max_seq_length: 1000            # Maximum sequence length
```

### Training Configuration Adjustments

When using attention, consider these training adjustments:

```yaml
training:
  batch_size: 24           # Slightly reduced due to attention memory overhead
  learning_rate: 0.0006    # Slightly lower for attention stability
  weight_decay: 0.012      # Increased regularization for more parameters
  patience: 15             # Increased patience for attention convergence
  gradient_clip_norm: 0.5  # Tighter gradient clipping
```

---

## 📊 **Configuration Parameters**

### Core Attention Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention_enabled` | bool | `false` | Enable/disable attention mechanism |
| `attention_heads` | int | `8` | Number of attention heads (4-16 recommended) |
| `attention_dropout` | float | `0.1` | Dropout rate for attention weights |
| `attention_dim` | int/null | `null` | Attention dimension (null = auto from LSTM) |
| `positional_encoding` | bool | `true` | Enable sinusoidal positional encoding |
| `max_seq_length` | int | `1000` | Maximum sequence length for positional encoding |

### Parameter Impact

| Model Configuration | Parameters | Memory Impact |
|-------------------|------------|---------------|
| BiLSTM only (256D, 2-layer) | ~2.3M | Baseline |
| + Attention (8 heads) | ~2.8M | +20% |
| + Attention (16 heads) | ~3.2M | +40% |

---

## 🧪 **Example Configurations**

### 1. Small Attention Model (Quick Testing)

```yaml
# configs/training/attention_small.yaml
model:
  hidden_dim: 128
  num_layers: 1
  attention_enabled: true
  attention_heads: 4
  attention_dropout: 0.1

training:
  batch_size: 16
  learning_rate: 0.001
  max_epochs: 50
```

### 2. Production Attention Model

```yaml
# configs/training/attention_production.yaml  
model:
  hidden_dim: 256
  num_layers: 2
  layer_dropout: 0.3
  attention_enabled: true
  attention_heads: 8
  attention_dropout: 0.1
  positional_encoding: true

training:
  batch_size: 24
  learning_rate: 0.0006
  weight_decay: 0.012
  max_epochs: 120
  patience: 15
```

### 3. Large Attention Model (High Performance)

```yaml
# configs/training/attention_large.yaml
model:
  hidden_dim: 512
  num_layers: 3
  layer_dropout: 0.35
  attention_enabled: true
  attention_heads: 16
  attention_dropout: 0.15
  positional_encoding: true

training:
  batch_size: 16
  learning_rate: 0.0004
  weight_decay: 0.015
  max_epochs: 150
  patience: 20
```

---

## 🔍 **Monitoring Attention**

### Model Architecture Display

When attention is enabled, the model info shows:

```
🤖 BiLSTM + Attention Model Architecture:
   🎯 Attention enabled:
      Attention heads: 8
      Attention dropout: 0.1
      Positional encoding: True
      Attention parameters: 66,304
   📊 Parameter breakdown:
      LSTM parameters: 52,224
      Classifier parameters: 258
      Attention parameters: 66,304
      Total parameters: 118,786
```

### Attention Statistics

Access attention analysis during training:

```python
# Get attention weights from last forward pass
attention_weights = model.get_last_attention_weights()

# Get comprehensive attention statistics
attention_stats = model.get_attention_statistics(features, mask)
print(f"Mean attention entropy: {attention_stats['overall']['mean_entropy']:.3f}")
print(f"Mean max attention: {attention_stats['overall']['mean_max_attention']:.3f}")
```

---

## 📈 **Performance Guidelines**

### When to Use Attention

✅ **Use attention when:**
- Working with longer sequences (>20 lines)
- Need better long-range dependency modeling
- Have sufficient computational resources
- Training on larger datasets

❌ **Skip attention when:**
- Working with very short sequences (<10 lines)
- Limited computational resources
- Quick prototyping/testing phases
- Memory constraints

### Training Tips

1. **Start Small**: Begin with 4-8 attention heads
2. **Monitor Memory**: Watch GPU memory usage during training
3. **Adjust Batch Size**: Reduce batch size by 25-50% when adding attention
4. **Increase Patience**: Attention models may need more epochs to converge
5. **Learning Rate**: Slightly reduce learning rate (0.001 → 0.0006)

---

## 🔧 **Troubleshooting**

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Reduce attention heads
   - Use attention_dim < lstm_output_dim

2. **Slow Convergence**
   - Increase patience
   - Lower learning rate
   - Check gradient clipping

3. **Attention Collapse**
   - Increase attention dropout
   - Add stronger regularization
   - Check positional encoding

### Configuration Validation

```bash
# Test configuration loading
python -c "
from segmodel.utils.config_loader import load_training_config
config = load_training_config('your_config.yaml')
print(f'Attention enabled: {config.attention_enabled}')
"

# Test model creation
python -c "
from segmodel.models.blstm_tagger import BLSTMTagger
model = BLSTMTagger(feat_dim=60, attention_enabled=True)
model.print_model_info()
"
```

---

## 🚀 **Migration from Non-Attention Models**

### Existing Configurations

All existing configurations continue to work without changes:

```yaml
# This still works exactly as before
model:
  hidden_dim: 256
  num_layers: 2
  dropout: 0.2
  # No attention parameters = attention disabled
```

### Gradual Migration

1. **Test with existing config** (attention disabled by default)
2. **Add attention_enabled: true** to enable with defaults
3. **Fine-tune attention parameters** for your use case
4. **Adjust training parameters** for optimal performance

---

## 📝 **Next Steps**

After implementing attention, consider:

1. **Experiment with different head counts** (4, 8, 16)
2. **Try different attention dimensions** 
3. **Analyze attention patterns** on your specific data
4. **Compare performance** with and without attention
5. **Optimize training hyperparameters** for attention models

---

## 🤖 **Technical Details**

### Architecture

```
Input Features (60D)
    ↓
BiLSTM Layers (256D → 512D output)
    ↓
Multi-Head Self-Attention (optional)
    ↓
Dropout + Residual Connection
    ↓
Linear Classifier (2 classes)
```

### Implementation

- **Attention Type**: Multi-head self-attention with scaled dot-product
- **Positional Encoding**: Sinusoidal encoding (optional)
- **Normalization**: Layer normalization with residual connections
- **Memory Efficiency**: Proper sequence masking for variable lengths

For more technical details, see:
- `segmodel/models/attention.py` - Core attention implementation
- `segmodel/models/blstm_tagger.py` - BiLSTM + attention integration
