"""
Attention mechanism for BiLSTM text segmentation.
Implements configurable self-attention with optional positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List


class PositionalEncoding(nn.Module):
    """
    Positional encoding for attention mechanism.
    Uses sinusoidal encoding as in "Attention Is All You Need".
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 1000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension (should match attention dimension)
            max_seq_length: Maximum sequence length to support
        """
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_seq_length, 1, d_model)
        
        # Register as buffer (not a parameter, but part of the model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (seq_length, batch_size, d_model)
            
        Returns:
            Input with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism for sequence processing.
    Designed specifically for BiLSTM output enhancement.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        positional_encoding: bool = True,
        max_seq_length: int = 1000
    ):
        """
        Initialize multi-head self-attention.
        
        Args:
            d_model: Input dimension (should match BiLSTM output dimension)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
            positional_encoding: Whether to use positional encoding
            max_seq_length: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout for attention weights
        self.attention_dropout = nn.Dropout(dropout)
        
        # Layer normalization for residual connection
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Optional positional encoding
        self.use_positional_encoding = positional_encoding
        if positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Initialize parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize attention parameters with Xavier initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor (batch_size, seq_length, d_model)
            mask: Optional attention mask (batch_size, seq_length)
            
        Returns:
            output: Attention output (batch_size, seq_length, d_model)
            attention_weights: Attention weights (batch_size, num_heads, seq_length, seq_length)
        """
        batch_size, seq_length, d_model = x.shape
        
        # Store residual connection
        residual = x
        
        # Apply positional encoding if enabled
        if self.use_positional_encoding:
            # Convert to (seq_length, batch_size, d_model) for positional encoding
            x_pe = x.transpose(0, 1)
            x_pe = self.positional_encoding(x_pe)
            x = x_pe.transpose(0, 1)  # Back to (batch_size, seq_length, d_model)
        
        # Linear projections
        Q = self.w_q(x)  # (batch_size, seq_length, d_model)
        K = self.w_k(x)  # (batch_size, seq_length, d_model)
        V = self.w_v(x)  # (batch_size, seq_length, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_length, d_k)
        K = K.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_length, d_k)
        V = V.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, seq_length, d_k)
        
        # Compute attention
        attention_output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model
        )  # (batch_size, seq_length, d_model)
        
        # Output projection
        output = self.w_o(attention_output)
        
        # Residual connection + layer normalization
        output = self.layer_norm(residual + output)
        
        return output, attention_weights
    
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query tensor (batch_size, num_heads, seq_length, d_k)
            K: Key tensor (batch_size, num_heads, seq_length, d_k)
            V: Value tensor (batch_size, num_heads, seq_length, d_k)
            mask: Optional mask (batch_size, seq_length)
            
        Returns:
            output: Attention output (batch_size, num_heads, seq_length, d_k)
            attention_weights: Attention weights (batch_size, num_heads, seq_length, seq_length)
        """
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        # scores: (batch_size, num_heads, seq_length, seq_length)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for all heads: (batch_size, 1, seq_length)
            mask_expanded = mask.unsqueeze(1)
            # Create attention mask: (batch_size, 1, seq_length, seq_length)
            attention_mask = mask_expanded.unsqueeze(-1) * mask_expanded.unsqueeze(-2)
            # Apply mask (set masked positions to large negative value)
            scores = scores.masked_fill(~attention_mask, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def get_attention_info(self) -> dict:
        """Get attention mechanism information."""
        return {
            'type': 'MultiHeadSelfAttention',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_k': self.d_k,
            'positional_encoding': self.use_positional_encoding,
            'total_params': sum(p.numel() for p in self.parameters()),
        }


class AttentionModule(nn.Module):
    """
    Complete attention module for BiLSTM enhancement.
    Combines multi-head self-attention with optional projection layers.
    Supports multiple attention types: 'self', 'localized', 'boundary_aware'.
    """
    
    def __init__(
        self,
        input_dim: int,
        attention_dim: int = None,
        attention_type: str = 'self',
        num_heads: int = 8,
        dropout: float = 0.1,
        positional_encoding: bool = True,
        max_seq_length: int = 1000,
        use_projection: bool = True,
        window_size: int = 7,
        boundary_temperature: float = 2.0
    ):
        """
        Initialize attention module.
        
        Args:
            input_dim: Input dimension (BiLSTM output dimension)
            attention_dim: Attention dimension (if None, uses input_dim)
            attention_type: Type of attention ('self', 'localized', 'boundary_aware')
            num_heads: Number of attention heads
            dropout: Dropout probability
            positional_encoding: Whether to use positional encoding
            max_seq_length: Maximum sequence length
            use_projection: Whether to use input/output projections
            window_size: Window size for localized attention (only used if attention_type='localized')
            boundary_temperature: Temperature for boundary-aware attention (only used if attention_type='boundary_aware')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim if attention_dim is not None else input_dim
        self.use_projection = use_projection
        self.attention_type = attention_type
        
        # Input projection (if needed)
        if use_projection and input_dim != self.attention_dim:
            self.input_projection = nn.Linear(input_dim, self.attention_dim)
        else:
            self.input_projection = None
        
        # Choose attention mechanism based on type
        if attention_type == 'self':
            # Your original multi-head self-attention
            self.attention = MultiHeadSelfAttention(
                d_model=self.attention_dim,
                num_heads=num_heads,
                dropout=dropout,
                positional_encoding=positional_encoding,
                max_seq_length=max_seq_length
            )
        elif attention_type == 'localized':
            self.attention = LocalizedAttention(
                d_model=self.attention_dim,
                num_heads=num_heads,
                dropout=dropout,
                positional_encoding=positional_encoding,
                max_seq_length=max_seq_length,
                window_size=window_size
            )
        elif attention_type == 'boundary_aware':
            self.attention = BoundaryAwareAttention(
                d_model=self.attention_dim,
                num_heads=num_heads,
                dropout=dropout,
                positional_encoding=positional_encoding,
                max_seq_length=max_seq_length,
                boundary_temperature=boundary_temperature
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}. Supported types: 'self', 'localized', 'boundary_aware'")
        
        # Output projection (if needed)
        if use_projection and self.attention_dim != input_dim:
            self.output_projection = nn.Linear(self.attention_dim, input_dim)
        else:
            self.output_projection = None
        
        # Final dropout
        self.output_dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of attention module.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_dim)
            mask: Optional attention mask (batch_size, seq_length)
            
        Returns:
            output: Enhanced output (batch_size, seq_length, input_dim)
            attention_weights: Attention weights (batch_size, num_heads, seq_length, seq_length)
        """
        # Store original input for residual connection
        original_input = x
        
        # Input projection if needed
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Apply attention
        attention_output, attention_weights = self.attention(x, mask)
        
        # Output projection if needed
        if self.output_projection is not None:
            attention_output = self.output_projection(attention_output)
        
        # Apply dropout
        attention_output = self.output_dropout(attention_output)
        
        # Residual connection with original input
        if attention_output.shape == original_input.shape:
            output = original_input + attention_output
        else:
            # If dimensions don't match, skip residual connection
            output = attention_output
        
        return output, attention_weights
    
    def get_module_info(self) -> dict:
        """Get attention module information."""
        attention_info = self.attention.get_attention_info()
        
        total_params = sum(p.numel() for p in self.parameters())
        projection_params = 0
        
        if self.input_projection is not None:
            projection_params += sum(p.numel() for p in self.input_projection.parameters())
        if self.output_projection is not None:
            projection_params += sum(p.numel() for p in self.output_projection.parameters())
        
        return {
            'type': 'AttentionModule',
            'attention_type': self.attention_type,
            'input_dim': self.input_dim,
            'attention_dim': self.attention_dim,
            'use_projection': self.use_projection,
            'projection_params': projection_params,
            'attention_info': attention_info,
            'total_params': total_params,
        }


class LocalizedAttention(nn.Module):
    """
    Localized attention that focuses on nearby tokens for boundary detection.
    Compatible with your existing AttentionModule interface.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        positional_encoding: bool = True,
        max_seq_length: int = 1000,
        window_size: int = 7
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.window_size = window_size
        
        # Linear projections for Q, K, V (same as your implementation)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout and layer norm (same as your implementation)
        self.attention_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Optional positional encoding (same as your implementation)
        self.use_positional_encoding = positional_encoding
        if positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Initialize parameters (same as your implementation)
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize attention parameters with Xavier initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - same interface as your MultiHeadSelfAttention.
        """
        batch_size, seq_length, d_model = x.shape
        residual = x
        
        # Apply positional encoding if enabled (same as your code)
        if self.use_positional_encoding:
            x_pe = x.transpose(0, 1)
            x_pe = self.positional_encoding(x_pe)
            x = x_pe.transpose(0, 1)
        
        # Linear projections (same as your code)
        Q = self.w_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        # Create localized attention mask
        local_mask = self._create_local_mask(seq_length, self.window_size, x.device)
        
        # Compute attention with local masking
        attention_output, attention_weights = self._scaled_dot_product_attention(Q, K, V, mask, local_mask)
        
        # Concatenate heads (same as your code)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model
        )
        
        # Output projection and residual connection (same as your code)
        output = self.w_o(attention_output)
        output = self.layer_norm(residual + output)
        
        return output, attention_weights
    
    def _create_local_mask(self, seq_length: int, window_size: int, device: torch.device) -> torch.Tensor:
        """Create mask for localized attention"""
        mask = torch.zeros(seq_length, seq_length, dtype=torch.bool, device=device)
        half_window = window_size // 2
        
        for i in range(seq_length):
            start = max(0, i - half_window)
            end = min(seq_length, i + half_window + 1)
            mask[i, start:end] = True
        
        return mask
    
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        local_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified version of your _scaled_dot_product_attention with local masking.
        """
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply local mask first
        if local_mask is not None:
            scores = scores.masked_fill(~local_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        # Apply input mask if provided (same as your code)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            attention_mask = mask_expanded.unsqueeze(-1) * mask_expanded.unsqueeze(-2)
            scores = scores.masked_fill(~attention_mask, -1e9)
        
        # Apply softmax and dropout (same as your code)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def get_attention_info(self) -> dict:
        """Get attention mechanism information (same interface as yours)."""
        return {
            'type': 'LocalizedAttention',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_k': self.d_k,
            'window_size': self.window_size,
            'positional_encoding': self.use_positional_encoding,
            'total_params': sum(p.numel() for p in self.parameters()),
        }


class BoundaryAwareAttention(nn.Module):
    """
    Boundary-aware attention that uses auxiliary boundary prediction.
    Compatible with your existing AttentionModule interface.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        positional_encoding: bool = True,
        max_seq_length: int = 1000,
        boundary_temperature: float = 2.0
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.boundary_temperature = boundary_temperature
        
        # Standard attention components (same as your implementation)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Boundary prediction components
        self.boundary_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Optional positional encoding (same as your implementation)
        self.use_positional_encoding = positional_encoding
        if positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Initialize parameters (same as your implementation)
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """Initialize attention parameters with Xavier initialization."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - same interface as your MultiHeadSelfAttention.
        """
        batch_size, seq_length, d_model = x.shape
        residual = x
        
        # Apply positional encoding if enabled (same as your code)
        if self.use_positional_encoding:
            x_pe = x.transpose(0, 1)
            x_pe = self.positional_encoding(x_pe)
            x = x_pe.transpose(0, 1)
        
        # Predict boundary probabilities
        boundary_logits = self.boundary_predictor(x)  # (batch, seq, 1)
        boundary_probs = torch.sigmoid(boundary_logits / self.boundary_temperature)
        
        # Linear projections (same as your code)
        Q = self.w_q(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
        # Create boundary bias
        boundary_bias = self._compute_boundary_bias(boundary_probs, seq_length)
        
        # Compute attention with boundary bias
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask, boundary_bias
        )
        
        # Concatenate heads (same as your code)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.d_model
        )
        
        # Output projection and residual connection (same as your code)
        output = self.w_o(attention_output)
        output = self.layer_norm(residual + output)
        
        return output, attention_weights
    
    def _compute_boundary_bias(self, boundary_probs: torch.Tensor, seq_length: int) -> torch.Tensor:
        """Compute attention bias based on boundary probabilities - OPTIMIZED VERSION"""
        boundary_scores = boundary_probs.squeeze(-1)  # (batch, seq)
        
        # Vectorized computation instead of nested loops (much faster!)
        scores_i = boundary_scores.unsqueeze(-1)  # (batch, seq, 1)
        scores_j = boundary_scores.unsqueeze(-2)  # (batch, 1, seq)
        bias = (scores_i + scores_j) * 0.5  # (batch, seq, seq)
        
        return bias
    
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        boundary_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modified version of your _scaled_dot_product_attention with boundary bias.
        """
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply boundary bias
        if boundary_bias is not None:
            scores = scores + boundary_bias.unsqueeze(1)  # Add bias to all heads
        
        # Apply input mask if provided (same as your code)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            attention_mask = mask_expanded.unsqueeze(-1) * mask_expanded.unsqueeze(-2)
            scores = scores.masked_fill(~attention_mask, -1e9)
        
        # Apply softmax and dropout (same as your code)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def get_attention_info(self) -> dict:
        """Get attention mechanism information (same interface as yours)."""
        return {
            'type': 'BoundaryAwareAttention',
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_k': self.d_k,
            'boundary_temperature': self.boundary_temperature,
            'positional_encoding': self.use_positional_encoding,
            'total_params': sum(p.numel() for p in self.parameters()),
        }


if __name__ == "__main__":
    # Test the attention mechanisms
    print("🧪 Testing Attention Mechanisms...")
    
    # Test parameters
    batch_size, seq_length, input_dim = 2, 10, 128
    
    # Create test data
    x = torch.randn(batch_size, seq_length, input_dim)
    mask = torch.ones(batch_size, seq_length, dtype=torch.bool)
    mask[0, 8:] = False  # Mask last 2 positions of first sequence
    mask[1, 6:] = False  # Mask last 4 positions of second sequence
    
    print(f"📊 Test Setup:")
    print(f"   Input shape: {x.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Sequence lengths: {mask.sum(dim=1).tolist()}")
    
    # Test 1: MultiHeadSelfAttention
    print(f"\n🧪 Test 1: MultiHeadSelfAttention")
    attention = MultiHeadSelfAttention(
        d_model=input_dim,
        num_heads=8,
        dropout=0.1,
        positional_encoding=True
    )
    
    with torch.no_grad():
        output, weights = attention(x, mask)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    print(f"   Parameters: {attention.get_attention_info()['total_params']:,}")
    
    # Test 2: AttentionModule with projection
    print(f"\n🧪 Test 2: AttentionModule (with projection)")
    attention_module = AttentionModule(
        input_dim=input_dim,
        attention_dim=64,  # Different dimension to test projection
        num_heads=4,
        dropout=0.1,
        use_projection=True
    )
    
    with torch.no_grad():
        output2, weights2 = attention_module(x, mask)
    
    print(f"   Output shape: {output2.shape}")
    print(f"   Attention weights shape: {weights2.shape}")
    
    module_info = attention_module.get_module_info()
    print(f"   Total parameters: {module_info['total_params']:,}")
    print(f"   Projection parameters: {module_info['projection_params']:,}")
    
    # Test 3: AttentionModule without projection
    print(f"\n🧪 Test 3: AttentionModule (no projection)")
    attention_simple = AttentionModule(
        input_dim=input_dim,
        attention_dim=input_dim,  # Same dimension
        num_heads=8,
        dropout=0.1,
        use_projection=False
    )
    
    with torch.no_grad():
        output3, weights3 = attention_simple(x, mask)
    
    print(f"   Output shape: {output3.shape}")
    print(f"   Attention weights shape: {weights3.shape}")
    print(f"   Parameters: {attention_simple.get_module_info()['total_params']:,}")
    
    # Test attention weight statistics
    print(f"\n📈 Attention Weight Analysis:")
    with torch.no_grad():
        # Get attention weights for first sequence
        first_seq_weights = weights[0, :, :mask[0].sum(), :mask[0].sum()]  # (num_heads, seq_len, seq_len)
        
        # Check if attention is properly normalized
        weight_sums = first_seq_weights.sum(dim=-1)  # Should be all 1.0
        print(f"   Attention weight sums (should be ~1.0): {weight_sums[0, :3].tolist()}")
        
        # Check attention diversity (entropy)
        attention_entropy = -(first_seq_weights * torch.log(first_seq_weights + 1e-9)).sum(dim=-1)
        print(f"   Attention entropy (higher = more diverse): {attention_entropy[0, :3].tolist()}")
    
    # Test 4: LocalizedAttention
    print(f"\n🧪 Test 4: LocalizedAttention")
    localized_module = AttentionModule(
        input_dim=input_dim,
        attention_type='localized',
        num_heads=8,
        dropout=0.1,
        window_size=5
    )
    
    with torch.no_grad():
        output4, weights4 = localized_module(x, mask)
    
    print(f"   Output shape: {output4.shape}")
    print(f"   Attention weights shape: {weights4.shape}")
    
    localized_info = localized_module.get_module_info()
    print(f"   Total parameters: {localized_info['total_params']:,}")
    print(f"   Window size: {localized_info['attention_info']['window_size']}")
    
    # Test 5: BoundaryAwareAttention
    print(f"\n🧪 Test 5: BoundaryAwareAttention")
    boundary_module = AttentionModule(
        input_dim=input_dim,
        attention_type='boundary_aware',
        num_heads=8,
        dropout=0.1,
        boundary_temperature=1.5
    )
    
    with torch.no_grad():
        output5, weights5 = boundary_module(x, mask)
    
    print(f"   Output shape: {output5.shape}")
    print(f"   Attention weights shape: {weights5.shape}")
    
    boundary_info = boundary_module.get_module_info()
    print(f"   Total parameters: {boundary_info['total_params']:,}")
    print(f"   Boundary temperature: {boundary_info['attention_info']['boundary_temperature']}")
    
    print("✅ Extended attention mechanism tests completed!")
