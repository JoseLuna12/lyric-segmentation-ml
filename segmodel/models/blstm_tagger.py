"""
Simple BiLSTM model for verse/chorus segmentation.
Focuses on stability and avoiding overconfidence collapse.
Now supports optional attention mechanism for enhanced performance.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

# Handle both relative and absolute imports
try:
    from .attention import AttentionModule
except ImportError:
    from attention import AttentionModule


class BLSTMTagger(nn.Module):
    """
    Bidirectional LSTM tagger for sequence labeling.
    
    Simple architecture focused on stability:
    - BiLSTM for sequence processing
    - Single classification head (not multi-head to avoid amplification)
    - Proper initialization to prevent overconfidence
    """
    
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.4,
        layer_dropout: float = 0.0,
        # NEW: Attention parameters
        attention_enabled: bool = False,
        attention_type: str = 'self',
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        attention_dim: int = None,
        positional_encoding: bool = True,
        max_seq_length: int = 1000,
        window_size: int = 7,
        boundary_temperature: float = 2.0
    ):
        """
        Initialize the BiLSTM tagger.
        
        Args:
            feat_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes (2 for verse/chorus)
            dropout: Dropout probability for output layer
            layer_dropout: Inter-layer dropout for LSTM (only applied if num_layers > 1)
            attention_enabled: Whether to use attention mechanism
            attention_type: Type of attention ('self', 'localized', 'boundary_aware')
            attention_heads: Number of attention heads
            attention_dropout: Dropout probability for attention
            attention_dim: Attention dimension (if None, uses LSTM output dimension)
            positional_encoding: Whether to use positional encoding in attention
            max_seq_length: Maximum sequence length for positional encoding
            window_size: Window size for localized attention
            boundary_temperature: Temperature for boundary-aware attention
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.layer_dropout_p = layer_dropout
        
        # NEW: Attention configuration
        self.attention_enabled = attention_enabled
        self.attention_type = attention_type
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.attention_dim = attention_dim
        self.positional_encoding = positional_encoding
        self.max_seq_length = max_seq_length
        self.window_size = window_size
        self.boundary_temperature = boundary_temperature
        
        # BiLSTM output dimension (bidirectional)
        self.lstm_output_dim = hidden_dim * 2
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=layer_dropout if num_layers > 1 else 0.0  # Inter-layer dropout
        )
        
        # NEW: Optional attention mechanism
        if self.attention_enabled:
            self.attention = AttentionModule(
                input_dim=self.lstm_output_dim,
                attention_dim=self.attention_dim,
                attention_type=self.attention_type,
                num_heads=self.attention_heads,
                dropout=self.attention_dropout,
                positional_encoding=self.positional_encoding,
                max_seq_length=self.max_seq_length,
                use_projection=True,
                window_size=self.window_size,
                boundary_temperature=self.boundary_temperature
            )
        else:
            self.attention = None
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Single classification head (avoid multi-head amplification issues)
        # Input dimension depends on whether attention is enabled
        classifier_input_dim = self.lstm_output_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)
        
        # Initialize parameters properly to prevent overconfidence
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initialize parameters to prevent overconfidence.
        Based on lessons from architecture knowledge.
        """
        # Initialize LSTM parameters with Xavier/Glorot initialization
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
        
        # Initialize classifier with small weights and proper bias
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.1)  # Small gain for stability
        
        # Initialize bias to log prior probabilities
        # Assume balanced classes for initialization (will be adjusted by loss weights)
        with torch.no_grad():
            self.classifier.bias.fill_(0.0)  # Start neutral, let class weights handle imbalance
    
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Input features (batch_size, seq_len, feat_dim)
            mask: Boolean mask (batch_size, seq_len)
            
        Returns:
            logits: Output logits (batch_size, seq_len, num_classes)
        """
        batch_size, seq_len = features.shape[:2]
        
        # Compute sequence lengths for packing
        lengths = mask.sum(dim=1).cpu()  # (batch_size,)
        
        # Pack sequences for efficient LSTM processing
        packed_input = nn.utils.rnn.pack_padded_sequence(
            features,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # BiLSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Unpack sequences
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len
        )
        
        # NEW: Apply attention if enabled
        if self.attention_enabled and self.attention is not None:
            lstm_output, attention_weights = self.attention(lstm_output, mask)
            # Store attention weights for potential visualization
            self._last_attention_weights = attention_weights
        else:
            self._last_attention_weights = None
        
        # Apply dropout
        lstm_output = self.dropout(lstm_output)
        
        # Classification
        logits = self.classifier(lstm_output)  # (batch_size, seq_len, num_classes)
        
        return logits
    
    def predict_with_temperature(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with temperature scaling for calibration.
        
        Args:
            features: Input features
            mask: Boolean mask
            temperature: Temperature for softmax calibration
            
        Returns:
            predictions: Predicted class indices (batch_size, seq_len)
            confidences: Prediction confidences (batch_size, seq_len)
        """
        with torch.no_grad():
            logits = self.forward(features, mask)
            
            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Get predictions and confidences
            probs = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)
            
            # Apply mask (set padded positions to 0)
            predictions = torch.where(mask, predictions, torch.zeros_like(predictions))
            confidences = torch.where(mask, confidences, torch.zeros_like(confidences))
            
            return predictions, confidences
    
    def get_last_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from the last forward pass.
        
        Returns:
            Attention weights if attention is enabled and available, None otherwise
        """
        if hasattr(self, '_last_attention_weights'):
            return self._last_attention_weights
        return None
    
    def get_attention_statistics(self, features: torch.Tensor, mask: torch.Tensor) -> Optional[dict]:
        """
        Get attention statistics for analysis.
        
        Args:
            features: Input features
            mask: Boolean mask
            
        Returns:
            Dictionary with attention statistics if attention is enabled
        """
        if not self.attention_enabled or self.attention is None:
            return None
        
        with torch.no_grad():
            # Run forward pass to get attention weights
            self.forward(features, mask)
            attention_weights = self.get_last_attention_weights()
            
            if attention_weights is None:
                return None
            
            # Calculate statistics
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Apply mask to get valid sequence lengths
            valid_lengths = mask.sum(dim=1)  # (batch_size,)
            
            stats = {}
            for i in range(batch_size):
                seq_len_i = valid_lengths[i].item()
                weights_i = attention_weights[i, :, :seq_len_i, :seq_len_i]  # (num_heads, seq_len_i, seq_len_i)
                
                # Attention entropy (diversity measure)
                entropy = -(weights_i * torch.log(weights_i + 1e-9)).sum(dim=-1)  # (num_heads, seq_len_i)
                
                # Attention concentration (how focused the attention is)
                max_attention = weights_i.max(dim=-1)[0]  # (num_heads, seq_len_i)
                
                stats[f'batch_{i}'] = {
                    'sequence_length': seq_len_i,
                    'mean_entropy': entropy.mean().item(),
                    'std_entropy': entropy.std().item(),
                    'mean_max_attention': max_attention.mean().item(),
                    'std_max_attention': max_attention.std().item(),
                }
            
            # Overall statistics
            all_entropies = []
            all_max_attentions = []
            
            for i in range(batch_size):
                seq_len_i = valid_lengths[i].item()
                weights_i = attention_weights[i, :, :seq_len_i, :seq_len_i]
                entropy = -(weights_i * torch.log(weights_i + 1e-9)).sum(dim=-1)
                max_attention = weights_i.max(dim=-1)[0]
                
                all_entropies.append(entropy.flatten())
                all_max_attentions.append(max_attention.flatten())
            
            all_entropies = torch.cat(all_entropies)
            all_max_attentions = torch.cat(all_max_attentions)
            
            stats['overall'] = {
                'mean_entropy': all_entropies.mean().item(),
                'std_entropy': all_entropies.std().item(),
                'mean_max_attention': all_max_attentions.mean().item(),
                'std_max_attention': all_max_attentions.std().item(),
                'total_attention_elements': len(all_entropies),
            }
            
            return stats
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Calculate parameter breakdown
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        attention_params = 0
        if self.attention is not None:
            attention_params = sum(p.numel() for p in self.attention.parameters())
        
        info = {
            'architecture': 'BiLSTM' + (' + Attention' if self.attention_enabled else ''),
            'feature_dim': self.feat_dim,
            'hidden_dim': self.hidden_dim,
            'lstm_output_dim': self.lstm_output_dim,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout_p,
            'layer_dropout': self.layer_dropout_p,
            'bidirectional': True,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'lstm_params': lstm_params,
            'classifier_params': classifier_params,
            'attention_enabled': self.attention_enabled,
        }
        
        # Add attention-specific information
        if self.attention_enabled and self.attention is not None:
            attention_info = self.attention.get_module_info()
            info.update({
                'attention_type': self.attention_type,
                'attention_heads': self.attention_heads,
                'attention_dropout': self.attention_dropout,
                'attention_dim': self.attention_dim,
                'positional_encoding': self.positional_encoding,
                'max_seq_length': self.max_seq_length,
                'window_size': self.window_size,
                'boundary_temperature': self.boundary_temperature,
                'attention_params': attention_params,
                'attention_details': attention_info,
            })
        
        return info
    
    def print_model_info(self):
        """Print model architecture information."""
        info = self.get_model_info()
        print(f"🤖 {info['architecture']} Model Architecture:")
        print(f"   Input dimension: {info['feature_dim']}")
        print(f"   Hidden dimension: {info['hidden_dim']}")
        print(f"   LSTM output dimension: {info['lstm_output_dim']}")
        print(f"   LSTM layers: {info['num_layers']}")
        if info['num_layers'] > 1 and info['layer_dropout'] > 0:
            print(f"   ✅ Multi-layer with inter-layer dropout: {info['layer_dropout']}")
        print(f"   Output classes: {info['num_classes']}")
        print(f"   Dropout: {info['dropout']}")
        print(f"   Bidirectional: {info['bidirectional']}")
        
        # NEW: Attention information
        if info['attention_enabled']:
            print(f"   🎯 Attention enabled:")
            print(f"      Attention heads: {info['attention_heads']}")
            print(f"      Attention dropout: {info['attention_dropout']}")
            print(f"      Attention dimension: {info.get('attention_dim', 'same as LSTM')}")
            print(f"      Positional encoding: {info['positional_encoding']}")
            print(f"      Max sequence length: {info['max_seq_length']}")
            print(f"      Attention parameters: {info['attention_params']:,}")
        else:
            print(f"   🎯 Attention: disabled")
        
        # Parameter breakdown
        print(f"   📊 Parameter breakdown:")
        print(f"      LSTM parameters: {info['lstm_params']:,}")
        print(f"      Classifier parameters: {info['classifier_params']:,}")
        if info['attention_enabled']:
            print(f"      Attention parameters: {info['attention_params']:,}")
        print(f"      Total parameters: {info['total_params']:,}")
        print(f"      Trainable parameters: {info['trainable_params']:,}")


def create_model(config) -> BLSTMTagger:
    """
    Factory function to create model from configuration.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        Initialized BiLSTM model (with optional attention)
    """
    # Extract attention parameters with safe defaults
    attention_enabled = getattr(config, 'attention_enabled', False)
    attention_type = getattr(config, 'attention_type', 'self')
    attention_heads = getattr(config, 'attention_heads', 8)
    attention_dropout = getattr(config, 'attention_dropout', 0.1)
    attention_dim = getattr(config, 'attention_dim', None)
    positional_encoding = getattr(config, 'positional_encoding', True)
    max_seq_length = getattr(config, 'max_seq_length', 1000)
    window_size = getattr(config, 'window_size', 7)
    boundary_temperature = getattr(config, 'boundary_temperature', 2.0)
    
    model = BLSTMTagger(
        feat_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        layer_dropout=config.layer_dropout,
        # NEW: Attention parameters
        attention_enabled=attention_enabled,
        attention_type=attention_type,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        attention_dim=attention_dim,
        positional_encoding=positional_encoding,
        max_seq_length=max_seq_length,
        window_size=window_size,
        boundary_temperature=boundary_temperature
    )
    
    model.print_model_info()
    return model


if __name__ == "__main__":
    # Test the model with and without attention
    print("🧪 Testing BiLSTM model...")
    
    # Create test data
    batch_size, seq_len, feat_dim = 2, 8, 12
    features = torch.randn(batch_size, seq_len, feat_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 6:] = False  # Mask last 2 positions of first sequence
    mask[1, 5:] = False  # Mask last 3 positions of second sequence
    
    print(f"\n📊 Test Setup:")
    print(f"   Input shape: {features.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Sequence lengths: {mask.sum(dim=1).tolist()}")
    
    # Test 1: Model without attention
    print(f"\n🧪 Test 1: BiLSTM without attention")
    model_no_attention = BLSTMTagger(
        feat_dim=feat_dim, 
        hidden_dim=32, 
        num_classes=2,
        attention_enabled=False
    )
    model_no_attention.print_model_info()
    
    with torch.no_grad():
        logits1 = model_no_attention(features, mask)
        predictions1, confidences1 = model_no_attention.predict_with_temperature(features, mask, temperature=1.5)
    
    print(f"   Logits shape: {logits1.shape}")
    print(f"   Predictions shape: {predictions1.shape}")
    print(f"   Sample predictions: {predictions1[0]}")
    
    # Test 2: Model with attention
    print(f"\n🧪 Test 2: BiLSTM with attention")
    model_with_attention = BLSTMTagger(
        feat_dim=feat_dim, 
        hidden_dim=32, 
        num_classes=2,
        attention_enabled=True,
        attention_heads=4,
        attention_dropout=0.1,
        positional_encoding=True
    )
    model_with_attention.print_model_info()
    
    with torch.no_grad():
        logits2 = model_with_attention(features, mask)
        predictions2, confidences2 = model_with_attention.predict_with_temperature(features, mask, temperature=1.5)
    
    print(f"   Logits shape: {logits2.shape}")
    print(f"   Predictions shape: {predictions2.shape}")
    print(f"   Sample predictions: {predictions2[0]}")
    
    # Test attention weights
    attention_weights = model_with_attention.get_last_attention_weights()
    if attention_weights is not None:
        print(f"   Attention weights shape: {attention_weights.shape}")
    
    # Test attention statistics
    print(f"\n� Attention Statistics:")
    attention_stats = model_with_attention.get_attention_statistics(features, mask)
    if attention_stats is not None:
        overall_stats = attention_stats['overall']
        print(f"   Mean attention entropy: {overall_stats['mean_entropy']:.3f}")
        print(f"   Mean max attention: {overall_stats['mean_max_attention']:.3f}")
        print(f"   Total attention elements: {overall_stats['total_attention_elements']}")
    
    # Compare parameter counts
    params_no_attention = sum(p.numel() for p in model_no_attention.parameters())
    params_with_attention = sum(p.numel() for p in model_with_attention.parameters())
    
    print(f"\n📊 Parameter Comparison:")
    print(f"   Without attention: {params_no_attention:,} parameters")
    print(f"   With attention: {params_with_attention:,} parameters")
    print(f"   Attention overhead: {params_with_attention - params_no_attention:,} parameters")
    print(f"   Relative increase: {((params_with_attention / params_no_attention - 1) * 100):.1f}%")
    
    # Test temperature scaling
    _, conf_t1 = model_with_attention.predict_with_temperature(features, mask, temperature=1.0)
    _, conf_t2 = model_with_attention.predict_with_temperature(features, mask, temperature=2.0)
    
    print(f"\n🌡️  Temperature scaling test (with attention):")
    print(f"   T=1.0 mean confidence: {conf_t1.mean():.3f}")
    print(f"   T=2.0 mean confidence: {conf_t2.mean():.3f}")
    
    if conf_t2.mean() < conf_t1.mean():
        print("✅ Temperature scaling working correctly")
    else:
        print("❌ Temperature scaling issue")
    
    print("✅ BiLSTM model tests completed!")
