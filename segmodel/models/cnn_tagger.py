"""
CNN model for verse/chorus segmentation.
Focuses on stability and avoiding overconfidence collapse.
Designed as a drop-in replacement for BiLSTM with the same interface.
Supports optional attention mechanism for enhanced performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

# Handle both relative and absolute imports
try:
    from .attention import AttentionModule
except ImportError:
    from attention import AttentionModule


class CNNTagger(nn.Module):
    """
    CNN tagger for sequence labeling.
    
    Architecture focused on stability and local pattern detection:
    - Multiple dilated convolution layers for multi-scale feature extraction
    - Residual connections for gradient flow
    - Optional attention mechanism for enhanced performance
    - Same interface as BiLSTM for drop-in replacement
    """
    
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.4,
        layer_dropout: float = 0.0,
        attention_enabled: bool = False,
        attention_type: str = 'self',
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        attention_dim: int = None,
        positional_encoding: bool = True,
        max_seq_length: int = 1000,
        window_size: int = 7,
        boundary_temperature: float = 2.0,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dilation_rates: Tuple[int, ...] = (1, 2, 4),
        use_residual: bool = True
    ):
        """
        Initialize the CNN tagger.
        
        Args:
            feat_dim: Input feature dimension
            hidden_dim: Hidden dimension for CNN layers
            num_layers: Number of CNN layer blocks
            num_classes: Number of output classes (2 for verse/chorus)
            dropout: Dropout probability for output layer
            layer_dropout: Inter-layer dropout (applied between CNN blocks)
            attention_enabled: Whether to use attention mechanism
            attention_type: Type of attention ('self', 'localized', 'boundary_aware')
            attention_heads: Number of attention heads
            attention_dropout: Dropout probability for attention
            attention_dim: Attention dimension (if None, uses CNN output dimension)
            positional_encoding: Whether to use positional encoding in attention
            max_seq_length: Maximum sequence length for positional encoding
            window_size: Window size for localized attention
            boundary_temperature: Temperature for boundary-aware attention
            kernel_sizes: Kernel sizes for parallel convolutions
            dilation_rates: Dilation rates for different layers
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.layer_dropout_p = layer_dropout
        
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        self.use_residual = use_residual
        
        self.attention_enabled = attention_enabled
        self.attention_type = attention_type
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.attention_dim = attention_dim
        self.positional_encoding = positional_encoding
        self.max_seq_length = max_seq_length
        self.window_size = window_size
        self.boundary_temperature = boundary_temperature
        
        self.cnn_output_dim = hidden_dim
        
        self.input_projection = nn.Linear(feat_dim, hidden_dim)
        
        self.cnn_blocks = nn.ModuleList()
        for layer_idx in range(num_layers):
            block = CNNBlock(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                kernel_sizes=kernel_sizes,
                dilation_rate=dilation_rates[min(layer_idx, len(dilation_rates) - 1)],
                dropout=layer_dropout if layer_idx < num_layers - 1 else 0.0,
                use_residual=use_residual
            )
            self.cnn_blocks.append(block)
        
        # Optional attention mechanism
        if self.attention_enabled:
            self.attention = AttentionModule(
                input_dim=self.cnn_output_dim,
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
        
        # Single classification head
        classifier_input_dim = self.cnn_output_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initialize parameters with much smaller values to prevent NaN and collapse.
        Uses consistent normal initialization with small std values.
        """
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.input_projection.bias, 0.01)
        
        for block in self.cnn_blocks:
            block._initialize_parameters()
        
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        
        with torch.no_grad():
            self.classifier.bias.fill_(0.0) 
            
    
    def check_tensor(self, x: torch.Tensor, name: str) -> bool:
        """
        Comprehensive tensor checking with detailed statistics.
        
        Args:
            x: Tensor to check
            name: Name of the tensor for logging
            
        Returns:
            bool: True if NaN/Inf detected, False otherwise
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            try:
                min_val = x.min().item() if not torch.isnan(x).all() else "all NaN"
                max_val = x.max().item() if not torch.isnan(x).all() else "all NaN"
                mean_val = x.mean().item() if not torch.isnan(x).all() else "all NaN"
                std_val = x.std().item() if not torch.isnan(x).all() else "all NaN"
            except:
                min_val = max_val = mean_val = std_val = "error"
                
            print(f"⚠️ NaN/Inf detected in {name}: nan={nan_count}, inf={inf_count}")
            print(f"  Shape: {x.shape}, Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")
            return True
        return False
    
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with comprehensive NaN detection and gradient stabilization.
        
        Args:
            features: Input features (batch_size, seq_len, feat_dim)
            mask: Boolean mask (batch_size, seq_len)
            
        Returns:
            logits: Output logits (batch_size, seq_len, num_classes)
        """
        batch_size, seq_len = features.shape[:2]
        
        if self.check_tensor(features, "input_features"):
            # Replace NaN/Inf with zeros to prevent propagation
            features = torch.where(torch.isnan(features) | torch.isinf(features), 
                                torch.zeros_like(features), features)
            print("⚠️ Input features fixed - replaced NaN/Inf with zeros")
            
        x = self.input_projection(features)
        
        if self.check_tensor(x, "input_projection_output"):
            x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
            
        x = torch.clamp(x, min=-10.0, max=10.0)
        
        for i, block in enumerate(self.cnn_blocks):
            x_prev = x
            x = block(x, mask)
            
            if self.check_tensor(x, f"cnn_block_{i}_output"):
                print(f"⚠️ NaN/Inf detected in CNN block {i}, falling back to previous block output")
                x = x_prev
                x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
                break
                
        if self.attention_enabled and self.attention is not None:
            try:
                x_prev = x
                x, attention_weights = self.attention(x, mask)
                
                if self.check_tensor(x, "attention_output"):
                    print("⚠️ NaN/Inf detected in attention output, disabling attention")
                    x = x_prev
                    attention_weights = None
                    x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
                
                self._last_attention_weights = attention_weights
            except Exception as e:
                print(f"⚠️ Attention error: {e}, falling back to non-attention path")
                x = x_prev
                self._last_attention_weights = None
        else:
            self._last_attention_weights = None
        
        x = self.dropout(x)
        
        if self.check_tensor(x, "pre_classification"):
            x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        
        x = torch.clamp(x, min=-5.0, max=5.0)
        
        logits = self.classifier(x)
        
        if self.check_tensor(logits, "output_logits"):
            logits = torch.zeros_like(logits)
            logits[:, :, 1] = 0.1
            print("⚠️ Returning safe logits after NaN detection")

        if self.training:
            logits.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
            
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
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
            self.forward(features, mask)
            attention_weights = self.get_last_attention_weights()
            
            if attention_weights is None:
                return None
            
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            valid_lengths = mask.sum(dim=1)
            
            stats = {}
            for i in range(batch_size):
                seq_len_i = valid_lengths[i].item()
                weights_i = attention_weights[i, :, :seq_len_i, :seq_len_i]
                
                entropy = -(weights_i * torch.log(weights_i + 1e-9)).sum(dim=-1)
                
                max_attention = weights_i.max(dim=-1)[0]
                
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
        input_proj_params = sum(p.numel() for p in self.input_projection.parameters())
        cnn_params = sum(sum(p.numel() for p in block.parameters()) for block in self.cnn_blocks)
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        attention_params = 0
        if self.attention is not None:
            attention_params = sum(p.numel() for p in self.attention.parameters())
        
        info = {
            'architecture': 'CNN' + (' + Attention' if self.attention_enabled else ''),
            'feature_dim': self.feat_dim,
            'hidden_dim': self.hidden_dim,
            'cnn_output_dim': self.cnn_output_dim,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout_p,
            'layer_dropout': self.layer_dropout_p,
            'kernel_sizes': self.kernel_sizes,
            'dilation_rates': self.dilation_rates,
            'use_residual': self.use_residual,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_projection_params': input_proj_params,
            'cnn_params': cnn_params,
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
        print(f"   CNN output dimension: {info['cnn_output_dim']}")
        print(f"   CNN layers: {info['num_layers']}")
        print(f"   Kernel sizes: {info['kernel_sizes']}")
        print(f"   Dilation rates: {info['dilation_rates']}")
        print(f"   Residual connections: {info['use_residual']}")
        if info['num_layers'] > 1 and info['layer_dropout'] > 0:
            print(f"   ✅ Multi-layer with inter-layer dropout: {info['layer_dropout']}")
        print(f"   Output classes: {info['num_classes']}")
        print(f"   Dropout: {info['dropout']}")
        
        # Attention information
        if info['attention_enabled']:
            print(f"   🎯 Attention enabled:")
            print(f"      Attention type: {info['attention_type']}")
            print(f"      Attention heads: {info['attention_heads']}")
            print(f"      Attention dropout: {info['attention_dropout']}")
            print(f"      Attention dimension: {info.get('attention_dim', 'same as CNN')}")
            print(f"      Positional encoding: {info['positional_encoding']}")
            print(f"      Max sequence length: {info['max_seq_length']}")
            print(f"      Attention parameters: {info['attention_params']:,}")
        else:
            print(f"   🎯 Attention: disabled")
        
        # Parameter breakdown
        print(f"   📊 Parameter breakdown:")
        print(f"      Input projection parameters: {info['input_projection_params']:,}")
        print(f"      CNN parameters: {info['cnn_params']:,}")
        print(f"      Classifier parameters: {info['classifier_params']:,}")
        if info['attention_enabled']:
            print(f"      Attention parameters: {info['attention_params']:,}")
        print(f"      Total parameters: {info['total_params']:,}")
        print(f"      Trainable parameters: {info['trainable_params']:,}")


class CNNBlock(nn.Module):
    """
    A single CNN block with multiple parallel convolutions and optional residual connection.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dilation_rate: int = 1,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.dilation_rate = dilation_rate
        self.use_residual = use_residual
        
        # Parallel convolutions with different kernel sizes
        self.convs = nn.ModuleList()
        conv_output_dim = hidden_dim // len(kernel_sizes)
        
        for kernel_size in kernel_sizes:
            # Proper padding calculation for odd and even kernel sizes
            padding = (kernel_size - 1) * dilation_rate // 2
            # Ensure odd kernel sizes work correctly
            if (kernel_size - 1) * dilation_rate % 2 != 0:
                padding += 1
                
            conv = nn.Conv1d(
                in_channels=input_dim,
                out_channels=conv_output_dim,
                kernel_size=kernel_size,
                dilation=dilation_rate,
                padding=padding
            )
            self.convs.append(conv)
        
        # Adjust for potential dimension mismatch
        total_conv_dim = conv_output_dim * len(kernel_sizes)
        if total_conv_dim != hidden_dim:
            self.conv_projection = nn.Linear(total_conv_dim, hidden_dim)
        else:
            self.conv_projection = None
        
        # Residual connection projection if needed
        if use_residual and input_dim != hidden_dim:
            self.residual_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_projection = None
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def _initialize_parameters(self):
        """Initialize CNN block parameters with much smaller values for stability."""
        for conv in self.convs:
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.constant_(conv.bias, 0.01)
        
        if self.conv_projection is not None:
            nn.init.normal_(self.conv_projection.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.conv_projection.bias, 0.01)
        
        if self.residual_projection is not None:
            nn.init.normal_(self.residual_projection.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.residual_projection.bias, 0.01)
            
        if hasattr(self.layer_norm, 'weight') and self.layer_norm.weight is not None:
            nn.init.ones_(self.layer_norm.weight)
        if hasattr(self.layer_norm, 'bias') and self.layer_norm.bias is not None:
            nn.init.zeros_(self.layer_norm.bias)
    
    def check_tensor(self, x: torch.Tensor, name: str) -> bool:
        """
        Comprehensive tensor checking with detailed statistics.
        
        Args:
            x: Tensor to check
            name: Name of the tensor for logging
            
        Returns:
            bool: True if NaN/Inf detected, False otherwise
        """
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_count = torch.isnan(x).sum().item()
            inf_count = torch.isinf(x).sum().item()
            try:
                min_val = x.min().item() if not torch.isnan(x).all() else "all NaN"
                max_val = x.max().item() if not torch.isnan(x).all() else "all NaN"
                mean_val = x.mean().item() if not torch.isnan(x).all() else "all NaN"
                std_val = x.std().item() if not torch.isnan(x).all() else "all NaN"
            except:
                min_val = max_val = mean_val = std_val = "error"
                
            print(f"⚠️ Block NaN/Inf in {name}: nan={nan_count}, inf={inf_count}")
            print(f"  Shape: {x.shape}, Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")
            return True
        return False
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN block with comprehensive stability safeguards.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mask: Boolean mask (batch_size, seq_len)
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        if self.check_tensor(x, "cnn_block_input"):
            x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
            print("⚠️ NaN/Inf in CNN block input fixed")
        
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
        
        x_conv = x.transpose(1, 2)
        
        conv_outputs = []
        for i, conv in enumerate(self.convs):
            try:
                conv_out = conv(x_conv)
            
                if self.check_tensor(conv_out, f"conv_{i}_output"):
                    conv_out = torch.zeros_like(conv_out)
                
                conv_out = torch.clamp(conv_out, min=-5.0, max=5.0)
                
                conv_outputs.append(conv_out)
            except Exception as e:
                print(f"⚠️ Exception in convolution {i}: {e}")
                safe_shape = (x_conv.shape[0], self.hidden_dim // len(self.convs), x_conv.shape[2])
                conv_outputs.append(torch.zeros(safe_shape, device=x_conv.device))
        
        x_conv = torch.cat(conv_outputs, dim=1)
        
        if self.check_tensor(x_conv, "concatenated_conv_output"):
            x_conv = torch.nan_to_num(x_conv, nan=0.0, posinf=5.0, neginf=-5.0)
        
        x = x_conv.transpose(1, 2)
        
        if self.conv_projection is not None:
            try:
                x_proj = self.conv_projection(x)
                
                if self.check_tensor(x_proj, "projection_output"):
                    print("⚠️ Using alternative projection")
                    x = torch.nn.functional.linear(x, 
                                             torch.ones((self.hidden_dim, x.shape[-1]), device=x.device) / x.shape[-1])
                else:
                    x = x_proj
            except Exception as e:
                print(f"⚠️ Exception in projection: {e}")
                x = torch.nn.functional.linear(x, 
                                        torch.ones((self.hidden_dim, x.shape[-1]), device=x.device) / x.shape[-1])
            
            x = torch.clamp(x, min=-5.0, max=5.0)
        
        x = self.activation(x)
        
        if self.use_residual:
            try:
                if self.residual_projection is not None:
                    residual = self.residual_projection(residual)
                    
                    if self.check_tensor(residual, "residual_projection"):
                        residual = torch.zeros_like(x)
                
                x = x + residual
            except Exception as e:
                print(f"⚠️ Exception in residual connection: {e}")
                pass
        
        try:
            x = self.layer_norm(x)
            
        
            if self.check_tensor(x, "layer_norm_output"):
                x = x - x.mean(dim=-1, keepdim=True)
                x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
        except Exception as e:
            print(f"⚠️ Exception in layer norm: {e}")
            x = x - x.mean(dim=-1, keepdim=True)
            x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
        
        x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
        
        if self.check_tensor(x, "cnn_block_final_output"):
            x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        
        x = torch.clamp(x, min=-5.0, max=5.0)
        
        # Dropout
        x = self.dropout(x)
        
        # Add gradient clipping hook during training
        if self.training:
            x.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))
        
        return x


def create_model(config) -> CNNTagger:
    """
    Factory function to create CNN model from configuration.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        Initialized CNN model (with optional attention)
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
    
    kernel_sizes = getattr(config, 'kernel_sizes', (3, 5, 7))
    dilation_rates = getattr(config, 'dilation_rates', (1, 2, 4))
    use_residual = getattr(config, 'use_residual', True)
    
    model = CNNTagger(
        feat_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        layer_dropout=config.layer_dropout,
        # Attention parameters
        attention_enabled=attention_enabled,
        attention_type=attention_type,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        attention_dim=attention_dim,
        positional_encoding=positional_encoding,
        max_seq_length=max_seq_length,
        window_size=window_size,
        boundary_temperature=boundary_temperature,
        # CNN-specific parameters
        kernel_sizes=kernel_sizes,
        dilation_rates=dilation_rates,
        use_residual=use_residual
    )
    
    model.print_model_info()
    return model


if __name__ == "__main__":

    print("🧪 Testing CNN model...")
    
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
    
    # Test 1: CNN without attention
    print(f"\n🧪 Test 1: CNN without attention")
    model_no_attention = CNNTagger(
        feat_dim=feat_dim, 
        hidden_dim=32, 
        num_layers=2,
        num_classes=2,
        attention_enabled=False,
        kernel_sizes=(3, 5),
        dilation_rates=(1, 2)
    )
    model_no_attention.print_model_info()
    
    with torch.no_grad():
        logits1 = model_no_attention(features, mask)
        predictions1, confidences1 = model_no_attention.predict_with_temperature(features, mask, temperature=1.5)
    
    print(f"   Logits shape: {logits1.shape}")
    print(f"   Predictions shape: {predictions1.shape}")
    print(f"   Sample predictions: {predictions1[0]}")
    
    # Test 2: CNN with attention
    print(f"\n🧪 Test 2: CNN with attention")
    model_with_attention = CNNTagger(
        feat_dim=feat_dim, 
        hidden_dim=32, 
        num_layers=2,
        num_classes=2,
        attention_enabled=True,
        attention_heads=4,
        attention_dropout=0.1,
        positional_encoding=True,
        kernel_sizes=(3, 5),
        dilation_rates=(1, 2)
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
    print(f"\n📈 Attention Statistics:")
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
    
    print("✅ CNN model tests completed!")
