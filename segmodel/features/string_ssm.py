"""
String-SSM (Self-Similarity Matrix) feature extraction.
Implements overall textual similarity features using normalized Levenshtein distance.
"""

import numpy as np
import torch
from typing import List
import re


def levenshtein_distance_fast(s1: str, s2: str) -> int:
    """
    Fast Levenshtein distance computation using only two rows of the matrix.
    
    This is the minimum number of single-character edits (insertions, deletions, 
    or substitutions) required to change one string into another.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance as integer
    """
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    # Ensure s1 is the shorter string for memory efficiency
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    # Use only two rows instead of full matrix
    prev_row = list(range(len(s2) + 1))
    curr_row = [0] * (len(s2) + 1)
    
    for i in range(1, len(s1) + 1):
        curr_row[0] = i
        for j in range(1, len(s2) + 1):
            if s1[i-1] == s2[j-1]:
                curr_row[j] = prev_row[j-1]
            else:
                curr_row[j] = min(
                    prev_row[j] + 1,      # deletion
                    curr_row[j-1] + 1,    # insertion
                    prev_row[j-1] + 1     # substitution
                )
        prev_row, curr_row = curr_row, prev_row
    
    return prev_row[len(s2)]


def jaccard_similarity(s1: str, s2: str) -> float:
    """
    Fast alternative: Jaccard similarity on word sets.
    Much faster than Levenshtein for long strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Jaccard similarity score between 0.0 and 1.0
    """
    if s1 == s2:
        return 1.0
        
    words1 = set(s1.split())
    words2 = set(s2.split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def simple_word_overlap_similarity(s1: str, s2: str) -> float:
    """
    Very fast similarity based on word overlap ratio.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if s1 == s2:
        return 1.0
        
    words1 = s1.split()
    words2 = s2.split()
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    # Count matching words
    matches = sum(1 for w1 in words1 if w1 in words2)
    max_len = max(len(words1), len(words2))
    
    return matches / max_len


def normalized_levenshtein_similarity(s1: str, s2: str, fast: bool = True) -> float:
    """
    Compute normalized Levenshtein similarity between two strings.
    
    The similarity is computed as:
    similarity = 1 - (levenshtein_distance / max_length)
    
    Args:
        s1: First string
        s2: Second string
        fast: Whether to use fast implementation (default True)
        
    Returns:
        Similarity score between 0.0 (completely different) and 1.0 (identical)
    """
    if s1 == s2:
        return 1.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0  # Both strings are empty
    
    if fast:
        distance = levenshtein_distance_fast(s1, s2)
    else:
        # Fallback to original implementation if needed
        distance = levenshtein_distance_fast(s1, s2)  # Always use fast for now
    
    similarity = 1.0 - (distance / max_len)
    
    return max(0.0, similarity)  # Ensure non-negative


def normalize_string(line: str, case_sensitive: bool = False, 
                    remove_punctuation: bool = True) -> str:
    """
    Normalize string for consistent comparison.
    
    Args:
        line: Input text line
        case_sensitive: Whether to preserve case
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Normalized string
    """
    normalized = line
    
    # Remove punctuation if requested
    if remove_punctuation:
        normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # Convert case if not case sensitive
    if not case_sensitive:
        normalized = normalized.lower()
    
    # Normalize whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized


def compute_string_ssm(lines: List[str], case_sensitive: bool = False,
                      remove_punctuation: bool = True,
                      similarity_threshold: float = 0.0,
                      similarity_method: str = "word_overlap") -> np.ndarray:
    """
    Compute String Self-Similarity Matrix (SSM) between lines using configurable similarity method.
    
    Strategy: Compare overall textual similarity between all pairs of lines.
    This captures general repetition patterns and structural similarities that
    head-SSM and tail-SSM might miss.
    
    Args:
        lines: List of text lines
        case_sensitive: Whether to use case-sensitive comparison
        remove_punctuation: Whether to remove punctuation before comparison
        similarity_threshold: Minimum similarity threshold (values below are set to 0)
        similarity_method: Method to use ('word_overlap', 'jaccard', 'levenshtein')
        
    Returns:
        SSM matrix (n x n) where entry (i,j) is similarity between line i and j
    """
    n = len(lines)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    # Normalize all lines
    normalized_lines = [normalize_string(line, case_sensitive, remove_punctuation) 
                       for line in lines]
    
    # Select similarity function
    if similarity_method == "word_overlap":
        sim_func = simple_word_overlap_similarity
    elif similarity_method == "jaccard":
        sim_func = jaccard_similarity
    elif similarity_method == "levenshtein":
        sim_func = lambda s1, s2: normalized_levenshtein_similarity(s1, s2, fast=True)
    else:
        raise ValueError(f"Unknown similarity method: {similarity_method}")
    
    # Compute similarity matrix
    ssm = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        ssm[i, i] = 1.0  # Self-similarity is always 1.0
        
        for j in range(i + 1, n):
            # Compute similarity using selected method
            similarity = sim_func(normalized_lines[i], normalized_lines[j])
            
            # Apply threshold
            if similarity < similarity_threshold:
                similarity = 0.0
            
            ssm[i, j] = ssm[j, i] = similarity  # Matrix is symmetric
    
    return ssm


def summarize_ssm_per_line(ssm: np.ndarray) -> np.ndarray:
    """
    Extract compact per-line features from the String SSM matrix.
    
    Each line gets a 12-dimensional feature vector capturing:
    - Statistical measures of string similarity to other lines
    - Local context (neighboring lines)
    - Global position information
    - Overall textual repetition indicators
    
    Args:
        ssm: Self-similarity matrix (n x n)
        
    Returns:
        Feature matrix (n x 12) with per-line features
    """
    n = ssm.shape[0]
    if n == 0:
        return np.zeros((0, 12), dtype=np.float32)
    
    features = []
    
    for i in range(n):
        row = ssm[i]  # String similarities of line i to all lines
        
        # Statistical measures
        mean_sim = float(row.mean())
        max_sim = float(row.max())
        std_sim = float(row.std()) if n > 1 else 0.0
        
        # Quantiles for distribution shape
        q75 = float(np.quantile(row, 0.75))
        q90 = float(np.quantile(row, 0.90))
        
        # High similarity count (repetition indicator)
        high_sim_ratio = float((row >= 0.7).sum() / n)  # Using 0.7 as default for string similarity
        
        # Local context (neighboring lines)
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
            mean_sim,       # 0: Mean string similarity to all lines
            max_sim,        # 1: Max string similarity to any line
            std_sim,        # 2: Std of string similarities
            q75,            # 3: 75th percentile string similarity
            q90,            # 4: 90th percentile string similarity  
            high_sim_ratio, # 5: Fraction of high string similarities (textual repetition)
            prev_sim,       # 6: String similarity to previous line
            next_sim,       # 7: String similarity to next line
            first_sim,      # 8: String similarity to first line
            last_sim,       # 9: String similarity to last line
            position,       # 10: Normalized position in song
            inv_position    # 11: Inverse position (1 - position)
        ]
        
        features.append(line_features)
    
    return np.asarray(features, dtype=np.float32)


def extract_string_ssm_features(lines: List[str], case_sensitive: bool = False,
                               remove_punctuation: bool = True,
                               similarity_threshold: float = 0.0,
                               similarity_method: str = "word_overlap") -> torch.Tensor:
    """
    Extract String-SSM features for a song.
    
    This is the main feature extraction function that:
    1. Computes String-SSM matrix using configurable similarity method
    2. Summarizes per-line features for overall textual similarity detection
    3. Returns as torch tensor
    
    Args:
        lines: List of text lines from song
        case_sensitive: Whether to use case-sensitive comparison
        remove_punctuation: Whether to remove punctuation before comparison
        similarity_threshold: Minimum similarity threshold
        similarity_method: Method to use ('word_overlap', 'jaccard', 'levenshtein')
        
    Returns:
        Feature tensor (seq_len, 12) 
    """
    if not lines:
        return torch.zeros(0, 12, dtype=torch.float32)
    
    # Remove any potential None values or empty strings
    clean_lines = [str(line) if line is not None else "" for line in lines]
    
    # Compute SSM
    ssm = compute_string_ssm(clean_lines, case_sensitive, 
                           remove_punctuation, similarity_threshold, similarity_method)
    
    # Extract per-line features
    features = summarize_ssm_per_line(ssm)
    
    # Convert to torch tensor
    return torch.from_numpy(features).float()


class StringSSMExtractor:
    """
    Modular String-SSM feature extractor for overall textual similarity detection.
    
    This class encapsulates the string-based feature extraction logic and can be
    easily configured and integrated into the training pipeline alongside Head-SSM and Tail-SSM.
    """
    
    def __init__(self, case_sensitive: bool = False, remove_punctuation: bool = True,
                 similarity_threshold: float = 0.0, similarity_method: str = "word_overlap",
                 output_dim: int = 12):
        """
        Initialize the extractor.
        
        Args:
            case_sensitive: Whether to use case-sensitive comparison
            remove_punctuation: Whether to remove punctuation before comparison
            similarity_threshold: Minimum similarity threshold
            similarity_method: Method to use ('word_overlap', 'jaccard', 'levenshtein')
            output_dim: Expected output dimension (should be 12)
        """
        self.case_sensitive = case_sensitive
        self.remove_punctuation = remove_punctuation
        self.similarity_threshold = similarity_threshold
        self.similarity_method = similarity_method
        self.output_dim = output_dim
        
        if output_dim != 12:
            raise ValueError(f"String-SSM features are fixed at 12 dimensions, got {output_dim}")
        
        # Validate similarity method
        valid_methods = ["word_overlap", "jaccard", "levenshtein"]
        if similarity_method not in valid_methods:
            raise ValueError(f"Invalid similarity_method: {similarity_method}. Must be one of {valid_methods}")
    
    def __call__(self, lines: List[str]) -> torch.Tensor:
        """Extract features from lines."""
        return extract_string_ssm_features(
            lines, 
            self.case_sensitive, 
            self.remove_punctuation,
            self.similarity_threshold,
            self.similarity_method
        )
    
    def get_feature_names(self) -> List[str]:
        """Get names of the 12 features for interpretability."""
        return [
            'string_mean_similarity',
            'string_max_similarity', 
            'string_std_similarity',
            'string_q75_similarity',
            'string_q90_similarity',
            'string_high_sim_ratio',
            'string_prev_similarity',
            'string_next_similarity',
            'string_first_similarity',
            'string_last_similarity',
            'string_position',
            'string_inverse_position'
        ]
    
    def describe_features(self):
        """Print feature descriptions."""
        names = self.get_feature_names()
        descriptions = [
            "Average string similarity to all other lines",
            "Maximum string similarity to any other line",
            "Standard deviation of string similarities", 
            "75th percentile of string similarities",
            "90th percentile of string similarities",
            "Fraction of high string similarities (>= 0.7) - textual repetition",
            "String similarity to previous line",
            "String similarity to next line", 
            "String similarity to first line",
            "String similarity to last line",
            "Normalized position in song [0, 1]",
            "Inverse position (1 - position)"
        ]
        
        print("🧩 String-SSM Features (Overall Textual Similarity):")
        for i, (name, desc) in enumerate(zip(names, descriptions)):
            print(f"   {i:2d}. {name:24s}: {desc}")
        
        print(f"\n   Configuration:")
        print(f"      Similarity method: {self.similarity_method}")
        print(f"      Case sensitive: {self.case_sensitive}")
        print(f"      Remove punctuation: {self.remove_punctuation}")
        print(f"      Similarity threshold: {self.similarity_threshold}")


if __name__ == "__main__":
    # Test the String-SSM feature extraction with diverse examples
    test_lines = [
        "Walking down this street tonight",      # verse - original
        "Thinking of you every single day",     # chorus - original  
        "Walking down this street at night",    # verse - similar to line 0
        "Thinking of you every day",            # chorus - similar to line 1
        "Dancing in the pale moonlight",        # verse - different
        "I keep thinking of you every day"      # chorus - variation of line 1
    ]
    
    print("🧪 Testing String-SSM feature extraction...")
    print(f"Input: {len(test_lines)} lines")
    
    # Show normalized lines
    print("\n🔍 Line normalization:")
    for i, line in enumerate(test_lines):
        normalized = normalize_string(line, case_sensitive=False, remove_punctuation=True)
        print(f"   {i}: '{line}' → '{normalized}'")
    
    # Test different configurations
    configs = [
        {"case_sensitive": False, "remove_punctuation": True, "similarity_threshold": 0.0},
        {"case_sensitive": False, "remove_punctuation": True, "similarity_threshold": 0.3},
        {"case_sensitive": True, "remove_punctuation": False, "similarity_threshold": 0.0}
    ]
    
    for config_idx, config in enumerate(configs):
        print(f"\n📊 Configuration {config_idx + 1}: {config}")
        
        # Extract features
        features = extract_string_ssm_features(test_lines, **config)
        print(f"✅ Features shape: {features.shape}")
        
        # Show SSM for interpretation
        ssm = compute_string_ssm(test_lines, **config)
        print(f"\n📊 String-SSM matrix:")
        print(ssm.round(3))
        
        # Show some similarity calculations
        print(f"\n🔍 Example similarities:")
        line0_norm = normalize_string(test_lines[0], config["case_sensitive"], config["remove_punctuation"])
        line2_norm = normalize_string(test_lines[2], config["case_sensitive"], config["remove_punctuation"])
        similarity = normalized_levenshtein_similarity(line0_norm, line2_norm)
        print(f"   Line 0 vs 2: {similarity:.3f} ('{line0_norm}' vs '{line2_norm}')")
        
        line1_norm = normalize_string(test_lines[1], config["case_sensitive"], config["remove_punctuation"])
        line3_norm = normalize_string(test_lines[3], config["case_sensitive"], config["remove_punctuation"])
        similarity = normalized_levenshtein_similarity(line1_norm, line3_norm)
        print(f"   Line 1 vs 3: {similarity:.3f} ('{line1_norm}' vs '{line3_norm}')")
    
    # Show feature descriptions
    print(f"\n🧩 Feature Descriptions:")
    extractor = StringSSMExtractor()
    extractor.describe_features()
    
    print(f"\n🔍 Sample features (line 0, default config):")
    default_features = extract_string_ssm_features(test_lines)
    feature_names = extractor.get_feature_names()
    for i, (name, value) in enumerate(zip(feature_names, default_features[0])):
        print(f"   {name:24s}: {value:.3f}")
    
    print(f"\n✅ String-SSM test completed successfully!")
    print(f"🎵 Notice: This captures overall textual similarity patterns")
    print(f"    that complement Head-SSM (line beginnings) and Tail-SSM (line endings)")
