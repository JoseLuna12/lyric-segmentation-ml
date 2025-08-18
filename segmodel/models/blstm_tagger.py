"""
Simple BiLSTM model for verse/chorus segmentation.
Focuses on stability and avoiding overconfidence collapse.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


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
        layer_dropout: float = 0.0
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
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.layer_dropout_p = layer_dropout
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=layer_dropout if num_layers > 1 else 0.0  # Inter-layer dropout
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Single classification head (avoid multi-head amplification issues)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
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
    
    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'BiLSTM',
            'feature_dim': self.feat_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout_p,
            'layer_dropout': self.layer_dropout_p,
            'bidirectional': True,
            'total_params': total_params,
            'trainable_params': trainable_params,
        }
    
    def print_model_info(self):
        """Print model architecture information."""
        info = self.get_model_info()
        print(f"🤖 BiLSTM Model Architecture:")
        print(f"   Input dimension: {info['feature_dim']}")
        print(f"   Hidden dimension: {info['hidden_dim']}")
        print(f"   LSTM layers: {info['num_layers']}")
        if info['num_layers'] > 1 and info['layer_dropout'] > 0:
            print(f"   ✅ Multi-layer with inter-layer dropout: {info['layer_dropout']}")
        print(f"   Output classes: {info['num_classes']}")
        print(f"   Dropout: {info['dropout']}")
        print(f"   Bidirectional: {info['bidirectional']}")
        print(f"   Total parameters: {info['total_params']:,}")
        print(f"   Trainable parameters: {info['trainable_params']:,}")


def create_model(config) -> BLSTMTagger:
    """
    Factory function to create model from configuration.
    
    Args:
        config: ModelConfig instance
        
    Returns:
        Initialized BiLSTM model
    """
    model = BLSTMTagger(
        feat_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        layer_dropout=config.layer_dropout
    )
    
    model.print_model_info()
    return model


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing BiLSTM model...")
    
    # Create test data
    batch_size, seq_len, feat_dim = 2, 5, 12
    features = torch.randn(batch_size, seq_len, feat_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 4] = False  # Mask last position of first sequence
    mask[1, 3:] = False  # Mask last 2 positions of second sequence
    
    # Create model
    model = BLSTMTagger(feat_dim=feat_dim, hidden_dim=64, num_classes=2)
    model.print_model_info()
    
    print(f"\n🧪 Testing forward pass:")
    print(f"   Input shape: {features.shape}")
    print(f"   Mask shape: {mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(features, mask)
        predictions, confidences = model.predict_with_temperature(features, mask, temperature=1.5)
    
    print(f"   Logits shape: {logits.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Confidences shape: {confidences.shape}")
    
    print(f"\n🔍 Sample predictions:")
    print(f"   Predictions: {predictions[0]}")
    print(f"   Confidences: {confidences[0].round(3)}")
    print(f"   Mask: {mask[0]}")
    
    # Test temperature scaling
    _, conf_t1 = model.predict_with_temperature(features, mask, temperature=1.0)
    _, conf_t2 = model.predict_with_temperature(features, mask, temperature=2.0)
    
    print(f"\n🌡️  Temperature scaling test:")
    print(f"   T=1.0 mean confidence: {conf_t1.mean():.3f}")
    print(f"   T=2.0 mean confidence: {conf_t2.mean():.3f}")
    
    if conf_t2.mean() < conf_t1.mean():
        print("✅ Temperature scaling working correctly (higher T = lower confidence)")
    else:
        print("❌ Temperature scaling issue")
    
    print("✅ BiLSTM model test completed!")
