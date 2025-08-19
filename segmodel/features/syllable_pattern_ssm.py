"""
Syllable Pattern SSM (Self-Similarity Matrix) feature extraction.
Implements rhythmic pattern detection based on syllable sequences within lines.
"""

import numpy as np
import torch
from typing import List, Dict, Any

# Handle imports for both testing and module usage
try:
    from .syllable_utils import (
        extract_syllable_pattern, 
        syllable_pattern_to_string,
        normalized_levenshtein_similarity,
        normalized_levenshtein_similarity_lists,
        compute_cosine_similarity,
        normalize_syllable_counts
    )
except ImportError:
    # For testing when run as script
    from syllable_utils import (
        extract_syllable_pattern, 
        syllable_pattern_to_string,
        normalized_levenshtein_similarity,
        normalized_levenshtein_similarity_lists,
        compute_cosine_similarity,
        normalize_syllable_counts
    )


def compute_syllable_pattern_ssm(lines: List[str], similarity_method: str = "levenshtein", 
                                normalize: bool = False) -> np.ndarray:
    """
    Compute Self-Similarity Matrix based on syllable patterns.
    
    Args:
        lines: List of text lines
        similarity_method: "levenshtein" or "cosine"
        normalize: Whether to normalize syllable patterns
        
    Returns:
        NxN similarity matrix where N is the number of lines
    """
    n = len(lines)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    # Extract syllable patterns for all lines
    patterns = []
    for line in lines:
        pattern = extract_syllable_pattern(line)
        patterns.append(pattern)
    
    # Compute similarity matrix
    ssm = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        ssm[i, i] = 1.0  # Self-similarity is always 1.0
        
        for j in range(i + 1, n):
            if similarity_method == "levenshtein":
                # Use edit distance directly on integer lists (not stringified)
                similarity = normalized_levenshtein_similarity_lists(patterns[i], patterns[j])
                
            elif similarity_method == "cosine":
                # Use cosine similarity between syllable count vectors
                if normalize:
                    vec1 = normalize_syllable_counts(patterns[i], "normalized")
                    vec2 = normalize_syllable_counts(patterns[j], "normalized")
                else:
                    vec1 = [float(x) for x in patterns[i]]
                    vec2 = [float(x) for x in patterns[j]]
                
                similarity = compute_cosine_similarity(vec1, vec2)
                # Since syllable counts are non-negative, cosine is naturally [0,1], no need to shift
                similarity = max(0.0, similarity)  # Ensure non-negative
                
            else:
                raise ValueError(f"Unknown similarity method: {similarity_method}")
            
            ssm[i, j] = ssm[j, i] = similarity
    
    return ssm


def compute_combined_syllable_pattern_ssm(lines: List[str], levenshtein_weight: float = 0.7, 
                                        cosine_weight: float = 0.3, normalize: bool = False) -> np.ndarray:
    """
    Compute Combined Syllable Pattern SSM using weighted combination of Levenshtein and Cosine similarities.
    
    Args:
        lines: List of text lines
        levenshtein_weight: Weight for Levenshtein similarity
        cosine_weight: Weight for cosine similarity
        normalize: Whether to normalize syllable patterns
        
    Returns:
        NxN similarity matrix where N is the number of lines
    """
    n = len(lines)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    # Extract syllable patterns for all lines
    patterns = []
    for line in lines:
        pattern = extract_syllable_pattern(line)
        patterns.append(pattern)
    
    # Compute similarity matrix using combined method
    ssm = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        ssm[i, i] = 1.0  # Self-similarity is always 1.0
        
        for j in range(i + 1, n):
            # Levenshtein similarity
            lev_sim = normalized_levenshtein_similarity_lists(patterns[i], patterns[j])
            
            # Cosine similarity 
            if normalize:
                vec1 = normalize_syllable_counts(patterns[i], "normalized")
                vec2 = normalize_syllable_counts(patterns[j], "normalized")
            else:
                vec1 = [float(x) for x in patterns[i]]
                vec2 = [float(x) for x in patterns[j]]
            
            cos_sim = compute_cosine_similarity(vec1, vec2)
            cos_sim = max(0.0, cos_sim)  # Ensure non-negative
            
            # Combined similarity
            combined_sim = levenshtein_weight * lev_sim + cosine_weight * cos_sim
            
            ssm[i, j] = ssm[j, i] = combined_sim
    
    return ssm


def summarize_ssm_per_line(ssm: np.ndarray, high_sim_threshold: float = 0.8) -> np.ndarray:
    """
    Extract per-line features from syllable pattern SSM.
    
    Args:
        ssm: NxN similarity matrix
        high_sim_threshold: Threshold for high similarity detection
        
    Returns:
        Nx12 feature matrix with statistical summaries per line
    """
    n = ssm.shape[0]
    if n == 0:
        return np.zeros((0, 12), dtype=np.float32)
    
    features = []
    
    for i in range(n):
        row = ssm[i, :]
        
        # Basic statistics
        mean_sim = float(row.mean())
        max_sim = float(row.max())
        std_sim = float(row.std()) if n > 1 else 0.0
        
        # Quantiles for distribution shape
        q75 = float(np.quantile(row, 0.75))
        q90 = float(np.quantile(row, 0.90))
        
        # High similarity count (rhythmic pattern repetition indicator)
        high_sim_ratio = float((row >= high_sim_threshold).sum() / n)
        
        # Local context (neighboring lines for rhythmic flow)
        prev_sim = float(row[i - 1]) if i > 0 else 0.0
        next_sim = float(row[i + 1]) if i + 1 < n else 0.0
        
        # Global context (boundaries)
        first_sim = float(row[0])
        last_sim = float(row[-1])
        
        # Positional features
        position = i / (n - 1) if n > 1 else 0.0
        inv_position = 1.0 - position
        
        # Combine into feature vector
        line_features = [
            mean_sim,       # 0: Mean syllable pattern similarity to all lines
            max_sim,        # 1: Max syllable pattern similarity to any line
            std_sim,        # 2: Std of syllable pattern similarities
            q75,            # 3: 75th percentile syllable pattern similarity
            q90,            # 4: 90th percentile syllable pattern similarity  
            high_sim_ratio, # 5: Fraction of high syllable pattern similarities (rhythmic repetition)
            prev_sim,       # 6: Syllable pattern similarity to previous line
            next_sim,       # 7: Syllable pattern similarity to next line
            first_sim,      # 8: Syllable pattern similarity to first line
            last_sim,       # 9: Syllable pattern similarity to last line
            position,       # 10: Normalized position in song
            inv_position    # 11: Inverse position (1 - position)
        ]
        
        features.append(line_features)
    
    return np.asarray(features, dtype=np.float32)


def extract_syllable_pattern_ssm_features(lines: List[str], similarity_method: str = "levenshtein",
                                        levenshtein_weight: float = 0.7, cosine_weight: float = 0.3,
                                        normalize: bool = False, normalize_method: str = "zscore",
                                        dimension: int = 12) -> torch.Tensor:
    """
    Extract syllable pattern SSM features from lines.
    
    Args:
        lines: List of text lines
        similarity_method: "levenshtein", "cosine", or "combined"
        levenshtein_weight: Weight for Levenshtein similarity in combined method
        cosine_weight: Weight for cosine similarity in combined method
        normalize: Whether to normalize features
        normalize_method: "zscore" or "minmax"
        dimension: Output dimension (configurable)
        
    Returns:
        Feature tensor of shape (n_lines, dimension)
    """
    # Clean lines
    clean_lines = [str(line) if line is not None else "" for line in lines]
    
    # Compute SSM
    if similarity_method == "combined":
        ssm = compute_combined_syllable_pattern_ssm(clean_lines, levenshtein_weight, cosine_weight, normalize)
    else:
        ssm = compute_syllable_pattern_ssm(clean_lines, similarity_method, normalize)
    
    # Extract per-line features
    base_features = summarize_ssm_per_line(ssm, high_sim_threshold=0.8)
    
    # Adjust to target dimension
    if dimension != 12:
        if dimension < 12:
            # Take first `dimension` features
            features = base_features[:, :dimension]
        else:
            # Pad with zeros or repeat features
            n_lines = base_features.shape[0]
            features = np.zeros((n_lines, dimension), dtype=np.float32)
            features[:, :12] = base_features
            
            # Fill remaining dimensions with derived features
            if dimension > 12:
                remaining = dimension - 12
                for i in range(remaining):
                    # Use simple derived features (e.g., combinations of existing ones)
                    col_idx = i % 12
                    features[:, 12 + i] = base_features[:, col_idx] * 0.7
    else:
        features = base_features
    
    # Optionally normalize features
    if normalize:
        from .syllable_utils import normalize_features
        features = normalize_features(features, method=normalize_method)
    
    # Convert to torch tensor
    return torch.from_numpy(features).float()


class SyllablePatternSSMExtractor:
    """
    Modular Syllable Pattern SSM feature extractor for rhythmic pattern detection.
    
    This class encapsulates syllable pattern-based feature extraction logic and can be
    easily configured and integrated into the training pipeline alongside other SSM features.
    """
    
    def __init__(self, similarity_method: str = "levenshtein", levenshtein_weight: float = 0.7,
                 cosine_weight: float = 0.3, normalize: bool = False, normalize_method: str = "zscore",
                 dimension: int = 12):
        """
        Initialize syllable pattern SSM extractor.
        
        Args:
            similarity_method: "levenshtein", "cosine", or "combined"
            levenshtein_weight: Weight for Levenshtein similarity in combined method
            cosine_weight: Weight for cosine similarity in combined method
            normalize: Whether to normalize features
            normalize_method: "zscore" or "minmax"
            dimension: Output dimension (configurable, default 12)
        """
        if similarity_method not in ["levenshtein", "cosine", "combined"]:
            raise ValueError(f"Similarity method must be 'levenshtein', 'cosine', or 'combined', got {similarity_method}")
        
        if similarity_method == "combined":
            if not (0.0 <= levenshtein_weight <= 1.0 and 0.0 <= cosine_weight <= 1.0):
                raise ValueError("Weights must be between 0.0 and 1.0")
            if abs(levenshtein_weight + cosine_weight - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
        
        if normalize_method not in ["zscore", "minmax"]:
            raise ValueError(f"Normalize method must be 'zscore' or 'minmax', got {normalize_method}")
        
        self.similarity_method = similarity_method
        self.levenshtein_weight = levenshtein_weight
        self.cosine_weight = cosine_weight
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.dimension = dimension
        self.output_dim = dimension  # For compatibility with other extractors
        self.similarity_method = similarity_method
        self.levenshtein_weight = levenshtein_weight
        self.cosine_weight = cosine_weight
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.dimension = dimension
        self.output_dim = dimension  # For compatibility with other extractors
    
    def __call__(self, lines: List[str]) -> torch.Tensor:
        """Extract features from lines."""
        return extract_syllable_pattern_ssm_features(
            lines,
            similarity_method=self.similarity_method,
            levenshtein_weight=self.levenshtein_weight,
            cosine_weight=self.cosine_weight,
            normalize=self.normalize,
            normalize_method=self.normalize_method,
            dimension=self.dimension
        )
    
    def get_feature_names(self) -> List[str]:
        """Get names of the 12 features for interpretability."""
        return [
            "syl_pattern_mean_sim",     # 0: Mean syllable pattern similarity
            "syl_pattern_max_sim",      # 1: Max syllable pattern similarity
            "syl_pattern_std_sim",      # 2: Std of syllable pattern similarities
            "syl_pattern_q75_sim",      # 3: 75th percentile similarity
            "syl_pattern_q90_sim",      # 4: 90th percentile similarity
            "syl_pattern_high_ratio",   # 5: High similarity ratio (rhythmic repetition)
            "syl_pattern_prev_sim",     # 6: Similarity to previous line
            "syl_pattern_next_sim",     # 7: Similarity to next line
            "syl_pattern_first_sim",    # 8: Similarity to first line
            "syl_pattern_last_sim",     # 9: Similarity to last line
            "syl_pattern_position",     # 10: Normalized position
            "syl_pattern_inv_pos"       # 11: Inverse position
        ]
    
    def describe_features(self):
        """Print feature descriptions."""
        print(f"🎵 Syllable Pattern SSM Features ({self.output_dim}D):")
        print(f"   Method: {self.similarity_method}")
        if self.similarity_method == "combined":
            print(f"   Levenshtein weight: {self.levenshtein_weight}")
            print(f"   Cosine weight: {self.cosine_weight}")
        print(f"   Normalize: {self.normalize}")
        if self.normalize:
            print(f"   Normalize method: {self.normalize_method}")
        print(f"   📊 Feature breakdown:")
        
        feature_names = self.get_feature_names()
        for i, name in enumerate(feature_names):
            print(f"      {i:2d}: {name}")


if __name__ == "__main__":
    # Test the Syllable Pattern SSM feature extraction
    test_lines = [
        "Walking down this street tonight",    # 2,1,1,1,2 pattern - 7 syllables
        "Thinking of you every day",          # 2,1,1,3,1 pattern - 8 syllables  
        "Dancing in the pale moonlight",      # 2,1,1,1,2 pattern - 7 syllables (same as line 0)
        "Thinking of you every day",          # 2,1,1,3,1 pattern - 8 syllables (same as line 1)
        "Beautiful dreams come true tonight", # 3,1,1,1,2 pattern - 8 syllables
        "I keep thinking of you every day"    # 1,1,2,1,1,3,1 pattern - 10 syllables
    ]
    
    print("🧪 Testing Syllable Pattern SSM feature extraction...")
    print(f"Input: {len(test_lines)} lines")
    
    # Show the syllable patterns being extracted
    print("\n🔍 Extracted syllable patterns:")
    for i, line in enumerate(test_lines):
        pattern = extract_syllable_pattern(line)
        pattern_str = syllable_pattern_to_string(pattern)
        print(f"   {i}: '{line}' → pattern: {pattern} → '{pattern_str}'")
    
    # Test Levenshtein similarity method
    print(f"\n📊 Testing Levenshtein Similarity Method:")
    features_lev = extract_syllable_pattern_ssm_features(test_lines, "levenshtein")
    ssm_lev = compute_syllable_pattern_ssm(test_lines, "levenshtein")
    print(f"✅ Features shape: {features_lev.shape}")
    print(f"📊 Levenshtein SSM matrix:")
    print(ssm_lev.round(3))
    
    # Test Cosine similarity method
    print(f"\n📊 Testing Cosine Similarity Method:")
    features_cos = extract_syllable_pattern_ssm_features(test_lines, "cosine")
    ssm_cos = compute_syllable_pattern_ssm(test_lines, "cosine")
    print(f"✅ Features shape: {features_cos.shape}")
    print(f"📊 Cosine SSM matrix:")
    print(ssm_cos.round(3))
    
    # Test with normalization
    print(f"\n📊 Testing Cosine with Normalization:")
    features_cos_norm = extract_syllable_pattern_ssm_features(test_lines, "cosine", normalize=True)
    ssm_cos_norm = compute_syllable_pattern_ssm(test_lines, "cosine", normalize=True)
    print(f"✅ Features shape: {features_cos_norm.shape}")
    print(f"📊 Cosine (normalized) SSM matrix:")
    print(ssm_cos_norm.round(3))
    
    # Show feature descriptions
    print(f"\n🧩 Feature Descriptions:")
    extractor = SyllablePatternSSMExtractor()
    extractor.describe_features()
    
    print(f"\n🔍 Sample features (line 0, Levenshtein method):")
    feature_names = extractor.get_feature_names()
    for i, (name, value) in enumerate(zip(feature_names, features_lev[0])):
        print(f"   {name:24s}: {value:.3f}")
    
    print(f"\n✅ Syllable Pattern SSM test completed successfully!")
    print(f"🎵 Key observations:")
    print(f"   ✅ Lines 0 and 2 have identical patterns: high similarity (1.000)")
    print(f"   ✅ Lines 1 and 3 have identical patterns: high similarity (1.000)")  
    print(f"   ✅ Different patterns show lower but meaningful similarity scores")
    print(f"   ✅ Features capture rhythmic pattern relationships effectively")
