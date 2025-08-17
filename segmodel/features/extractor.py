"""
Modular feature extraction system.
Combines multiple feature types based on configuration.
"""

import torch
from typing import List, Dict, Any, Callable
from .head_ssm import HeadSSMExtractor
from .tail_ssm import TailSSMExtractor
from .phonetic_ssm import PhoneticSSMExtractor
from .pos_ssm import POSSSMExtractor
from .string_ssm import StringSSMExtractor


class FeatureExtractor:
    """
    Modular feature extractor that combines multiple feature types.
    
    Features are controlled by configuration and can be easily enabled/disabled
    without changing the core training code.
    """
    
    def __init__(self, feature_config: Dict[str, Any]):
        """
        Initialize with feature configuration.
        
        Args:
            feature_config: Dictionary with feature settings
        """
        self.feature_config = feature_config
        self.extractors = {}
        self.total_dim = 0
        
        self._setup_extractors()
    
    def _setup_extractors(self):
        """Set up individual feature extractors based on configuration."""
        enabled_features = []
        
        # Head-SSM features
        if self.feature_config.get('head_ssm', {}).get('enabled', False):
            head_config = self.feature_config['head_ssm']
            head_words = head_config.get('head_words', 2)
            output_dim = head_config.get('output_dim', 12)
            
            self.extractors['head_ssm'] = HeadSSMExtractor(
                head_words=head_words,
                output_dim=output_dim
            )
            self.total_dim += output_dim
            enabled_features.append(f"head_ssm({output_dim}D)")
        
        # Tail-SSM features (rhyme detection)
        if self.feature_config.get('tail_ssm', {}).get('enabled', False):
            tail_config = self.feature_config['tail_ssm']
            tail_words = tail_config.get('tail_words', 2)
            output_dim = tail_config.get('output_dim', 12)
            
            self.extractors['tail_ssm'] = TailSSMExtractor(
                tail_words=tail_words,
                output_dim=output_dim
            )
            self.total_dim += output_dim
            enabled_features.append(f"tail_ssm({output_dim}D)")
        
        # Phonetic-SSM features (true rhyme/alliteration detection)
        if self.feature_config.get('phonetic_ssm', {}).get('enabled', False):
            phonetic_config = self.feature_config['phonetic_ssm']
            mode = phonetic_config.get('mode', 'rhyme')
            output_dim = phonetic_config.get('output_dim', 12)
            similarity_method = phonetic_config.get('similarity_method', 'binary')
            normalize = phonetic_config.get('normalize', False)
            normalize_method = phonetic_config.get('normalize_method', 'zscore')
            high_sim_threshold = phonetic_config.get('high_sim_threshold', 0.8)
            
            # Initialize phonetic processor early to show loading message once
            from .phonetic_ssm import initialize_phonetic_processor
            initialize_phonetic_processor()
            
            self.extractors['phonetic_ssm'] = PhoneticSSMExtractor(
                mode=mode,
                output_dim=output_dim,
                similarity_method=similarity_method,
                normalize=normalize,
                normalize_method=normalize_method,
                high_sim_threshold=high_sim_threshold
            )
            self.total_dim += output_dim
            sim_desc = similarity_method if similarity_method != 'binary' else 'binary'
            norm_desc = f",norm" if normalize else ""
            enabled_features.append(f"phonetic_ssm({output_dim}D,{mode},{sim_desc}{norm_desc})")
        
        # POS-SSM features (grammatical structure detection)
        if self.feature_config.get('pos_ssm', {}).get('enabled', False):
            pos_config = self.feature_config['pos_ssm']
            tagset = pos_config.get('tagset', 'simplified')
            similarity_method = pos_config.get('similarity_method', 'combined')
            high_sim_threshold = pos_config.get('high_sim_threshold', 0.7)
            output_dim = pos_config.get('output_dim', 12)
            
            self.extractors['pos_ssm'] = POSSSMExtractor(
                tagset=tagset,
                similarity_method=similarity_method,
                high_sim_threshold=high_sim_threshold,
                output_dim=output_dim
            )
            self.total_dim += output_dim
            sim_short = similarity_method if similarity_method != 'combined' else 'comb'
            enabled_features.append(f"pos_ssm({output_dim}D,{tagset},{sim_short},th={high_sim_threshold})")
        
        # String-SSM features (overall textual similarity using Levenshtein distance)
        if self.feature_config.get('string_ssm', {}).get('enabled', False):
            string_config = self.feature_config['string_ssm']
            case_sensitive = string_config.get('case_sensitive', False)
            remove_punctuation = string_config.get('remove_punctuation', True)
            similarity_threshold = string_config.get('similarity_threshold', 0.0)
            similarity_method = string_config.get('similarity_method', 'word_overlap')
            output_dim = string_config.get('output_dim', 12)
            
            self.extractors['string_ssm'] = StringSSMExtractor(
                case_sensitive=case_sensitive,
                remove_punctuation=remove_punctuation,
                similarity_threshold=similarity_threshold,
                similarity_method=similarity_method,
                output_dim=output_dim
            )
            self.total_dim += output_dim
            desc_parts = [f"string_ssm({output_dim}D,{similarity_method})"]
            if case_sensitive:
                desc_parts.append("case_sens")
            if not remove_punctuation:
                desc_parts.append("keep_punct")
            if similarity_threshold > 0:
                desc_parts.append(f"th={similarity_threshold}")
            enabled_features.append(",".join(desc_parts))
        
        # Text embeddings (placeholder for future)
        if self.feature_config.get('text_embeddings', {}).get('enabled', False):
            print("⚠️  Text embeddings not implemented yet, skipping...")
        
        # Positional features (placeholder for future)
        if self.feature_config.get('positional_features', {}).get('enabled', False):
            print("⚠️  Positional features not implemented yet, skipping...")
        
        if not self.extractors:
            raise ValueError("No feature extractors enabled! Check feature configuration.")
        
        print(f"🧩 Initialized feature extractor:")
        print(f"   Enabled features: {enabled_features}")
        print(f"   Total dimension: {self.total_dim}")
    
    def __call__(self, lines: List[str]) -> torch.Tensor:
        """
        Extract features from lines.
        
        Args:
            lines: List of text lines
            
        Returns:
            Combined feature tensor (seq_len, total_dim)
        """
        if not lines:
            return torch.zeros(0, self.total_dim, dtype=torch.float32)
        
        feature_tensors = []
        
        # Extract each enabled feature type
        for feature_name, extractor in self.extractors.items():
            try:
                features = extractor(lines)
                feature_tensors.append(features)
                
            except Exception as e:
                print(f"⚠️  Error extracting {feature_name}: {e}")
                # Create zero features as fallback
                expected_dim = self.feature_config[feature_name]['output_dim']
                fallback = torch.zeros(len(lines), expected_dim, dtype=torch.float32)
                feature_tensors.append(fallback)
        
        # Concatenate all features
        if len(feature_tensors) == 1:
            combined = feature_tensors[0]
        else:
            combined = torch.cat(feature_tensors, dim=-1)
        
        # Validate output shape
        expected_shape = (len(lines), self.total_dim)
        if combined.shape != expected_shape:
            raise ValueError(f"Feature shape mismatch: got {combined.shape}, expected {expected_shape}")
        
        return combined
    
    def get_feature_dimension(self) -> int:
        """Get total feature dimension."""
        return self.total_dim
    
    def describe_features(self):
        """Print description of all enabled features."""
        print(f"🧩 Feature Extractor Configuration:")
        print(f"   Total dimension: {self.total_dim}")
        
        for feature_name, extractor in self.extractors.items():
            config = self.feature_config[feature_name]
            print(f"\n   {feature_name.upper()}:")
            print(f"      Dimension: {config['output_dim']}")
            print(f"      Description: {config.get('description', 'No description')}")
            
            # Feature-specific descriptions
            if hasattr(extractor, 'describe_features'):
                extractor.describe_features()


def create_feature_extractor(feature_config: Dict[str, Any]) -> FeatureExtractor:
    """
    Factory function to create feature extractor from configuration.
    
    Args:
        feature_config: Feature configuration dictionary
        
    Returns:
        Configured FeatureExtractor instance
    """
    return FeatureExtractor(feature_config)


def validate_feature_config(feature_config: Dict[str, Any]):
    """
    Validate feature configuration.
    
    Args:
        feature_config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    enabled_count = 0
    total_dim = 0
    
    for feature_name, config in feature_config.items():
        if not isinstance(config, dict):
            raise ValueError(f"Feature config for '{feature_name}' must be a dictionary")
        
        if config.get('enabled', False):
            enabled_count += 1
            
            output_dim = config.get('output_dim', 0)
            if output_dim <= 0:
                raise ValueError(f"Invalid output_dim for '{feature_name}': {output_dim}")
            
            total_dim += output_dim
    
    if enabled_count == 0:
        raise ValueError("At least one feature must be enabled")
    
    print(f"✅ Feature configuration valid:")
    print(f"   Enabled features: {enabled_count}")
    print(f"   Total dimension: {total_dim}")


if __name__ == "__main__":
    # Test the modular feature extractor
    test_feature_config = {
        'head_ssm': {
            'enabled': True,
            'output_dim': 12
        },
        'tail_ssm': {
            'enabled': True,
            'output_dim': 12
        },
        'phonetic_ssm': {
            'enabled': True,
            'output_dim': 12,
            'mode': 'rhyme',
            'similarity_method': 'edit_distance',  # Use soft similarity 
            'normalize': True,                     # Enable per-song normalization
            'normalize_method': 'zscore',
            'high_sim_threshold': 0.7              # Custom threshold for testing
        },
        'pos_ssm': {
            'enabled': True,
            'output_dim': 12,
            'tagset': 'universal',
            'similarity_method': 'lcs',
            'high_sim_threshold': 0.8
        },
        'string_ssm': {
            'enabled': True,
            'output_dim': 12,
            'case_sensitive': False,
            'remove_punctuation': True,
            'similarity_threshold': 0.2
        }
    }
    
    print("🧪 Testing modular feature extractor...")
    
    # Validate config
    validate_feature_config(test_feature_config)
    
    # Create extractor
    extractor = FeatureExtractor(test_feature_config)
    extractor.describe_features()
    
    # Test with sample lines
    test_lines = [
        "Walking down this street tonight",
        "Thinking of you every day", 
        "Walking down this street tonight",
        "Tomorrow is another day"
    ]
    
    print(f"\n🧪 Testing feature extraction:")
    print(f"   Input: {len(test_lines)} lines")
    
    features = extractor(test_lines)
    print(f"   Output: {features.shape}")
    print(f"   Expected: ({len(test_lines)}, {extractor.get_feature_dimension()})")
    
    if features.shape == (len(test_lines), extractor.get_feature_dimension()):
        print("✅ Feature extraction test passed!")
    else:
        print("❌ Feature extraction test failed!")
        print(f"   Shape mismatch: {features.shape}")
    
    print(f"\n🔍 Sample features (first line):")
    print(f"   Values: {features[0][:5]}... (showing first 5)")
    print(f"   Range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"   Mean: {features.mean():.3f}, Std: {features.std():.3f}")
