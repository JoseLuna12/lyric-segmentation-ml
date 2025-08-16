"""
Head-SSM (Self-Similarity Matrix) feature extraction.
Implements the baseline structural feature for verse/chorus detection.
"""

import numpy as np
import torch
from typing import List


def extract_head(line: str, k: int = 2) -> str:
    """
    Extract the 'head' of a line (first k words).
    
    Args:
        line: Input text line
        k: Number of words to extract
        
    Returns:
        Head string (lowercase, space-separated)
    """
    words = line.lower().split()
    head_words = words[:k] if len(words) >= k else words
    return " ".join(head_words)


def compute_head_ssm(lines: List[str], head_words: int = 2) -> np.ndarray:
    """
    Compute Head Self-Similarity Matrix (SSM) between lines.
    
    Strategy: Compare first k words of each line to detect repetition patterns.
    This is particularly effective for detecting chorus repetitions.
    
    Args:
        lines: List of text lines
        head_words: Number of words to use for head comparison
        
    Returns:
        SSM matrix (n x n) where entry (i,j) is similarity between line i and j
    """
    n = len(lines)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    # Extract heads for all lines
    heads = [extract_head(line, head_words) for line in lines]
    
    # Compute similarity matrix
    ssm = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        ssm[i, i] = 1.0  # Self-similarity is always 1.0
        
        for j in range(i + 1, n):
            # Simple binary similarity: 1.0 if heads match, 0.0 otherwise
            # This is stable and interpretable for the baseline
            if heads[i] == heads[j] and heads[i] != "":
                similarity = 1.0
            else:
                similarity = 0.0
            
            ssm[i, j] = ssm[j, i] = similarity  # Matrix is symmetric
    
    return ssm


def summarize_ssm_per_line(ssm: np.ndarray) -> np.ndarray:
    """
    Extract compact per-line features from the SSM matrix.
    
    Each line gets a 12-dimensional feature vector capturing:
    - Statistical measures of similarity to other lines
    - Local context (neighbors)
    - Global position information
    - Repetition indicators
    
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
        row = ssm[i]  # Similarities of line i to all lines
        
        # Statistical measures
        mean_sim = float(row.mean())
        max_sim = float(row.max())
        std_sim = float(row.std()) if n > 1 else 0.0
        
        # Quantiles for distribution shape
        q75 = float(np.quantile(row, 0.75))
        q90 = float(np.quantile(row, 0.90))
        
        # High similarity count (repetition indicator)
        high_sim_ratio = float((row >= 0.8).sum() / n)
        
        # Local context (neighbors)
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
            mean_sim,      # 0: Mean similarity to all lines
            max_sim,       # 1: Max similarity to any line
            std_sim,       # 2: Std of similarities
            q75,           # 3: 75th percentile similarity
            q90,           # 4: 90th percentile similarity  
            high_sim_ratio, # 5: Fraction of high similarities (repetition)
            prev_sim,      # 6: Similarity to previous line
            next_sim,      # 7: Similarity to next line
            first_sim,     # 8: Similarity to first line
            last_sim,      # 9: Similarity to last line
            position,      # 10: Normalized position in song
            inv_position   # 11: Inverse position (1 - position)
        ]
        
        features.append(line_features)
    
    return np.asarray(features, dtype=np.float32)


def extract_head_ssm_features(lines: List[str], head_words: int = 2) -> torch.Tensor:
    """
    Extract Head-SSM features for a song.
    
    This is the main feature extraction function that:
    1. Computes Head-SSM matrix
    2. Summarizes per-line features
    3. Returns as torch tensor
    
    Args:
        lines: List of text lines from song
        head_words: Number of words for head comparison
        
    Returns:
        Feature tensor (seq_len, 12) 
    """
    if not lines:
        return torch.zeros(0, 12, dtype=torch.float32)
    
    # Remove any potential None values or empty strings
    clean_lines = [str(line) if line is not None else "" for line in lines]
    
    # Compute SSM
    ssm = compute_head_ssm(clean_lines, head_words)
    
    # Extract per-line features
    features = summarize_ssm_per_line(ssm)
    
    # Convert to torch tensor
    return torch.from_numpy(features).float()


class HeadSSMExtractor:
    """
    Modular Head-SSM feature extractor.
    
    This class encapsulates the feature extraction logic and can be
    easily configured and integrated into the training pipeline.
    """
    
    def __init__(self, head_words: int = 2, output_dim: int = 12):
        """
        Initialize the extractor.
        
        Args:
            head_words: Number of words for head comparison
            output_dim: Expected output dimension (should be 12)
        """
        self.head_words = head_words
        self.output_dim = output_dim
        
        if output_dim != 12:
            raise ValueError(f"Head-SSM features are fixed at 12 dimensions, got {output_dim}")
    
    def __call__(self, lines: List[str]) -> torch.Tensor:
        """Extract features from lines."""
        return extract_head_ssm_features(lines, self.head_words)
    
    def get_feature_names(self) -> List[str]:
        """Get names of the 12 features for interpretability."""
        return [
            'mean_similarity',
            'max_similarity', 
            'std_similarity',
            'q75_similarity',
            'q90_similarity',
            'high_sim_ratio',
            'prev_similarity',
            'next_similarity',
            'first_similarity',
            'last_similarity',
            'position',
            'inverse_position'
        ]
    
    def describe_features(self):
        """Print feature descriptions."""
        names = self.get_feature_names()
        descriptions = [
            "Average similarity to all other lines",
            "Maximum similarity to any other line",
            "Standard deviation of similarities", 
            "75th percentile of similarities",
            "90th percentile of similarities",
            "Fraction of high similarities (>= 0.8)",
            "Similarity to previous line",
            "Similarity to next line", 
            "Similarity to first line",
            "Similarity to last line",
            "Normalized position in song [0, 1]",
            "Inverse position (1 - position)"
        ]
        
        print("🧩 Head-SSM Features:")
        for i, (name, desc) in enumerate(zip(names, descriptions)):
            print(f"   {i:2d}. {name:18s}: {desc}")


if __name__ == "__main__":
    # Test the Head-SSM feature extraction
    test_lines = [
        "Walking down this street tonight",  # verse
        "Thinking of you every day",         # chorus
        "Dancing in the moonlight",          # verse  
        "Thinking of you every day",         # chorus (repeated)
        "Walking down this street tonight",  # verse (repeated)
        "Tomorrow is another day"            # bridge
    ]
    
    print("🧪 Testing Head-SSM feature extraction...")
    print(f"Input: {len(test_lines)} lines")
    
    # Extract features
    features = extract_head_ssm_features(test_lines)
    print(f"✅ Features shape: {features.shape}")
    
    # Show SSM for interpretation
    ssm = compute_head_ssm(test_lines)
    print(f"\n📊 Head-SSM matrix:")
    print(ssm.round(1))
    
    # Show some features
    extractor = HeadSSMExtractor()
    extractor.describe_features()
    
    print(f"\n🔍 Sample features (line 0):")
    feature_names = extractor.get_feature_names()
    for i, (name, value) in enumerate(zip(feature_names, features[0])):
        print(f"   {name:18s}: {value:.3f}")
    
    print(f"\n✅ Head-SSM test completed successfully!")
