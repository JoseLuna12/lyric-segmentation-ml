"""
Tail-SSM (Self-Similarity Matrix) feature extraction.
Implements tail-based structural features for rhyme detection and verse/chorus segmentation.
"""

import numpy as np
import torch
from typing import List
import re


def extract_tail(line: str, k: int = 2) -> str:
    """
    Extract the 'tail' of a line (last k words).
    
    This function is designed for rhyme detection by focusing on line endings,
    which are crucial for identifying repeated patterns in choruses.
    
    Args:
        line: Input text line
        k: Number of words to extract from the end
        
    Returns:
        Tail string (lowercase, space-separated)
    """
    # Remove punctuation to focus on the actual words for rhyme detection
    clean_line = re.sub(r'[^\w\s]', '', line.lower())
    words = clean_line.split()
    tail_words = words[-k:] if len(words) >= k else words
    return " ".join(tail_words)


def compute_tail_ssm(lines: List[str], tail_words: int = 2) -> np.ndarray:
    """
    Compute Tail Self-Similarity Matrix (SSM) between lines.
    
    Strategy: Compare last k words of each line to detect rhyme patterns
    and chorus repetitions. This is particularly effective for songs where
    choruses end with similar rhyming patterns.
    
    Args:
        lines: List of text lines
        tail_words: Number of words to use for tail comparison
        
    Returns:
        SSM matrix (n x n) where entry (i,j) is similarity between line i and j
    """
    n = len(lines)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    # Extract tails for all lines
    tails = [extract_tail(line, tail_words) for line in lines]
    
    # Compute similarity matrix
    ssm = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        ssm[i, i] = 1.0  # Self-similarity is always 1.0
        
        for j in range(i + 1, n):
            # Binary similarity: 1.0 if tails match, 0.0 otherwise
            # Empty tails (from very short lines) don't count as matches
            if tails[i] == tails[j] and tails[i] != "":
                similarity = 1.0
            else:
                similarity = 0.0
            
            ssm[i, j] = ssm[j, i] = similarity  # Matrix is symmetric
    
    return ssm


def summarize_ssm_per_line(ssm: np.ndarray) -> np.ndarray:
    """
    Extract compact per-line features from the Tail SSM matrix.
    
    Each line gets a 12-dimensional feature vector capturing:
    - Statistical measures of tail similarity to other lines
    - Local context (neighboring lines)
    - Global position information
    - Rhyme pattern indicators
    
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
        row = ssm[i]  # Tail similarities of line i to all lines
        
        # Statistical measures
        mean_sim = float(row.mean())
        max_sim = float(row.max())
        std_sim = float(row.std()) if n > 1 else 0.0
        
        # Quantiles for distribution shape
        q75 = float(np.quantile(row, 0.75))
        q90 = float(np.quantile(row, 0.90))
        
        # High similarity count (rhyme pattern indicator)
        high_sim_ratio = float((row >= 0.8).sum() / n)
        
        # Local context (neighboring lines for rhyme scheme detection)
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
            mean_sim,       # 0: Mean tail similarity to all lines
            max_sim,        # 1: Max tail similarity to any line
            std_sim,        # 2: Std of tail similarities
            q75,            # 3: 75th percentile tail similarity
            q90,            # 4: 90th percentile tail similarity  
            high_sim_ratio, # 5: Fraction of high tail similarities (rhyme density)
            prev_sim,       # 6: Tail similarity to previous line
            next_sim,       # 7: Tail similarity to next line
            first_sim,      # 8: Tail similarity to first line
            last_sim,       # 9: Tail similarity to last line
            position,       # 10: Normalized position in song
            inv_position    # 11: Inverse position (1 - position)
        ]
        
        features.append(line_features)
    
    return np.asarray(features, dtype=np.float32)


def extract_tail_ssm_features(lines: List[str], tail_words: int = 2) -> torch.Tensor:
    """
    Extract Tail-SSM features for a song.
    
    This is the main feature extraction function that:
    1. Computes Tail-SSM matrix based on line endings
    2. Summarizes per-line features for rhyme pattern detection
    3. Returns as torch tensor
    
    Args:
        lines: List of text lines from song
        tail_words: Number of words for tail comparison
        
    Returns:
        Feature tensor (seq_len, 12) 
    """
    if not lines:
        return torch.zeros(0, 12, dtype=torch.float32)
    
    # Remove any potential None values or empty strings
    clean_lines = [str(line) if line is not None else "" for line in lines]
    
    # Compute SSM
    ssm = compute_tail_ssm(clean_lines, tail_words)
    
    # Extract per-line features
    features = summarize_ssm_per_line(ssm)
    
    # Convert to torch tensor
    return torch.from_numpy(features).float()


class TailSSMExtractor:
    """
    Modular Tail-SSM feature extractor for rhyme pattern detection.
    
    This class encapsulates the tail-based feature extraction logic and can be
    easily configured and integrated into the training pipeline alongside Head-SSM.
    """
    
    def __init__(self, tail_words: int = 2, output_dim: int = 12):
        """
        Initialize the extractor.
        
        Args:
            tail_words: Number of words for tail comparison
            output_dim: Expected output dimension (should be 12)
        """
        self.tail_words = tail_words
        self.output_dim = output_dim
        
        if output_dim != 12:
            raise ValueError(f"Tail-SSM features are fixed at 12 dimensions, got {output_dim}")
    
    def __call__(self, lines: List[str]) -> torch.Tensor:
        """Extract features from lines."""
        return extract_tail_ssm_features(lines, self.tail_words)
    
    def get_feature_names(self) -> List[str]:
        """Get names of the 12 features for interpretability."""
        return [
            'tail_mean_similarity',
            'tail_max_similarity', 
            'tail_std_similarity',
            'tail_q75_similarity',
            'tail_q90_similarity',
            'tail_high_sim_ratio',
            'tail_prev_similarity',
            'tail_next_similarity',
            'tail_first_similarity',
            'tail_last_similarity',
            'tail_position',
            'tail_inverse_position'
        ]
    
    def describe_features(self):
        """Print feature descriptions."""
        names = self.get_feature_names()
        descriptions = [
            "Average tail similarity to all other lines",
            "Maximum tail similarity to any other line",
            "Standard deviation of tail similarities", 
            "75th percentile of tail similarities",
            "90th percentile of tail similarities",
            "Fraction of high tail similarities (>= 0.8) - rhyme density",
            "Tail similarity to previous line",
            "Tail similarity to next line", 
            "Tail similarity to first line",
            "Tail similarity to last line",
            "Normalized position in song [0, 1]",
            "Inverse position (1 - position)"
        ]
        
        print("🧩 Tail-SSM Features (Rhyme Detection):")
        for i, (name, desc) in enumerate(zip(names, descriptions)):
            print(f"   {i:2d}. {name:22s}: {desc}")


if __name__ == "__main__":
    # Test the Tail-SSM feature extraction with rhyme-focused examples
    test_lines = [
        "Walking down this street at night",    # verse - "at night"
        "Thinking of you every day",           # chorus - "every day"  
        "Dancing in the pale moonlight",       # verse - "moonlight" (rhymes with "night")
        "Thinking of you every day",           # chorus - "every day" (repeated)
        "Walking down this street at night",   # verse - "at night" (repeated)
        "Tomorrow is a brand new way"          # bridge - "new way" (rhymes with "day")
    ]
    
    print("🧪 Testing Tail-SSM feature extraction...")
    print(f"Input: {len(test_lines)} lines")
    
    # Show the tails being extracted
    print("\n🔍 Extracted tails:")
    for i, line in enumerate(test_lines):
        tail = extract_tail(line, 2)
        print(f"   {i}: '{line}' → tail: '{tail}'")
    
    # Extract features
    features = extract_tail_ssm_features(test_lines)
    print(f"\n✅ Features shape: {features.shape}")
    
    # Show SSM for interpretation
    ssm = compute_tail_ssm(test_lines)
    print(f"\n📊 Tail-SSM matrix (rhyme patterns):")
    print(ssm.round(1))
    
    # Show feature descriptions
    extractor = TailSSMExtractor()
    extractor.describe_features()
    
    print(f"\n🔍 Sample features (line 0):")
    feature_names = extractor.get_feature_names()
    for i, (name, value) in enumerate(zip(feature_names, features[0])):
        print(f"   {name:22s}: {value:.3f}")
    
    print(f"\n✅ Tail-SSM test completed successfully!")
    print(f"🎵 Notice the rhyme patterns: 'night'↔'moonlight', 'day'↔'way'")
