"""
Part-of-Speech Self-Similarity Matrix (POS SSM) feature extraction.
Measures grammatical structure similarity by comparing POS tag sequences.
"""

import numpy as np
import torch
from typing import List
import nltk
from nltk import word_tokenize, pos_tag
import re
from collections import Counter

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    print("📦 Downloading NLTK data for POS tagging...")
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('universal_tagset', quiet=True)


def extract_pos_sequence(line: str, tagset: str = 'simplified') -> str:
    """
    Extract Part-of-Speech tag sequence from a line.
    
    Args:
        line: Input text line
        tagset: POS tagset to use - 'simplified', 'universal', or 'penn'
        
    Returns:
        POS sequence string (space-separated tags)
    """
    if not line.strip():
        return ""
    
    try:
        # Tokenize and get POS tags
        tokens = word_tokenize(line.lower())
        
        if tagset == 'universal':
            # Use Universal POS tags (more standard)
            pos_tags = pos_tag(tokens, tagset='universal')
            return " ".join([tag for _, tag in pos_tags])
            
        elif tagset == 'penn':
            # Use original Penn Treebank POS tags
            pos_tags = pos_tag(tokens)
            return " ".join([tag for _, tag in pos_tags])
            
        else:  # tagset == 'simplified' (default)
            # Simplify POS tags to major categories
            pos_tags = pos_tag(tokens)
            simplified_tags = []
            for word, tag in pos_tags:
                if tag.startswith('N'):       # NN, NNS, NNP, NNPS
                    simplified_tags.append('NOUN')
                elif tag.startswith('V'):     # VB, VBD, VBG, VBN, VBP, VBZ
                    simplified_tags.append('VERB')
                elif tag.startswith('J'):     # JJ, JJR, JJS
                    simplified_tags.append('ADJ')
                elif tag.startswith('R'):     # RB, RBR, RBS
                    simplified_tags.append('ADV')
                elif tag in ['DT', 'WDT']:    # Determiners
                    simplified_tags.append('DET')
                elif tag in ['IN', 'TO']:     # Prepositions
                    simplified_tags.append('PREP')
                elif tag in ['PRP', 'PRP$']:  # Pronouns
                    simplified_tags.append('PRON')
                elif tag in ['CC']:           # Coordinating conjunction
                    simplified_tags.append('CONJ')
                elif tag.startswith('W'):     # WH-words (what, who, where, etc.)
                    simplified_tags.append('WH')
                else:
                    simplified_tags.append('OTHER')
            
            return " ".join(simplified_tags)
            
    except Exception:
        # Fallback for any tokenization errors
        return ""


def compute_lcs_length(seq1: list, seq2: list) -> int:
    """
    Compute the length of the Longest Common Subsequence (LCS).
    
    Uses dynamic programming to find LCS length efficiently.
    Handles insertions/deletions while preserving order.
    
    Args:
        seq1: First sequence as list
        seq2: Second sequence as list
        
    Returns:
        Length of LCS
    """
    m, n = len(seq1), len(seq2)
    
    # DP table: dp[i][j] = LCS length for seq1[:i] and seq2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def compute_lcs_similarity(tags1: list, tags2: list) -> float:
    """
    Compute normalized LCS similarity between two tag sequences.
    
    This method handles insertions, deletions, and reorderings
    better than position-wise matching.
    
    Args:
        tags1: First tag sequence
        tags2: Second tag sequence
        
    Returns:
        LCS similarity score between 0.0 and 1.0
    """
    if not tags1 and not tags2:
        return 1.0
    if not tags1 or not tags2:
        return 0.0
    
    lcs_length = compute_lcs_length(tags1, tags2)
    max_length = max(len(tags1), len(tags2))
    
    return lcs_length / max_length if max_length > 0 else 0.0


def compute_position_similarity(tags1: list, tags2: list) -> float:
    """
    Compute position-wise similarity (original method).
    
    Args:
        tags1: First tag sequence
        tags2: Second tag sequence
        
    Returns:
        Position similarity score between 0.0 and 1.0
    """
    if not tags1 and not tags2:
        return 1.0
    if not tags1 or not tags2:
        return 0.0
    
    min_len = min(len(tags1), len(tags2))
    if min_len == 0:
        return 0.0
    
    matches = sum(1 for i in range(min_len) if tags1[i] == tags2[i])
    return matches / min_len


def compute_pos_similarity(pos_seq1: str, pos_seq2: str, method: str = 'combined') -> float:
    """
    Compute similarity between two POS sequences using specified method.
    
    Args:
        pos_seq1: First POS sequence
        pos_seq2: Second POS sequence
        method: Similarity method - 'combined', 'lcs', 'position', 'jaccard'
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not pos_seq1 or not pos_seq2:
        return 0.0 if pos_seq1 != pos_seq2 else 1.0
    
    tags1 = pos_seq1.split()
    tags2 = pos_seq2.split()
    
    if not tags1 or not tags2:
        return 0.0
    
    # Exact match gets perfect score
    if pos_seq1 == pos_seq2:
        return 1.0
    
    if method == 'lcs':
        # Use normalized LCS similarity (handles insertions/deletions)
        return compute_lcs_similarity(tags1, tags2)
        
    elif method == 'position':
        # Use position-wise similarity (original method)
        return compute_position_similarity(tags1, tags2)
        
    elif method == 'jaccard':
        # Use pure Jaccard similarity (tag overlap only)
        set1 = set(tags1)
        set2 = set(tags2)
        return len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0.0
        
    else:  # method == 'combined' (default)
        # Original combined approach with multiple similarity measures
        
        # Jaccard similarity (tag overlap)
        set1 = set(tags1)
        set2 = set(tags2)
        jaccard = len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0.0
        
        # Length-normalized overlap (accounts for repetition patterns)
        counter1 = Counter(tags1)
        counter2 = Counter(tags2)
        
        # Common tags with minimum counts
        overlap_count = sum((counter1 & counter2).values())
        total_count = max(len(tags1), len(tags2))
        overlap_ratio = overlap_count / total_count if total_count > 0 else 0.0
        
        # Position-wise similarity (order preservation)
        position_similarity = compute_position_similarity(tags1, tags2)
        
        # Combine different similarity measures
        # Weight: 40% jaccard, 40% overlap, 20% position
        final_similarity = 0.4 * jaccard + 0.4 * overlap_ratio + 0.2 * position_similarity
        
        return float(np.clip(final_similarity, 0.0, 1.0))


def compute_pos_ssm(lines: List[str], tagset: str = 'simplified', similarity_method: str = 'combined') -> np.ndarray:
    """
    Compute Part-of-Speech Self-Similarity Matrix (SSM) between lines.
    
    Strategy: Compare POS tag sequences to detect grammatical structure patterns.
    This is particularly effective for detecting structural repetitions in choruses
    that may use different words but similar grammatical patterns.
    
    Args:
        lines: List of text lines
        tagset: POS tagset to use - 'simplified', 'universal', or 'penn'
        similarity_method: Method for computing similarity - 'combined', 'lcs', 'position', 'jaccard'
        
    Returns:
        SSM matrix (n x n) where entry (i,j) is POS similarity between line i and j
    """
    n = len(lines)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    # Extract POS sequences for all lines
    pos_sequences = [extract_pos_sequence(line, tagset) for line in lines]
    
    # Compute similarity matrix
    ssm = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        ssm[i, i] = 1.0  # Self-similarity is always 1.0
        
        for j in range(i + 1, n):
            similarity = compute_pos_similarity(pos_sequences[i], pos_sequences[j], similarity_method)
            ssm[i, j] = ssm[j, i] = similarity  # Matrix is symmetric
    
    return ssm


def summarize_pos_ssm_per_line(ssm: np.ndarray, high_sim_threshold: float = 0.7) -> np.ndarray:
    """
    Extract compact per-line features from the POS SSM matrix.
    
    Each line gets a 12-dimensional feature vector capturing:
    - Statistical measures of grammatical similarity to other lines
    - Local context (neighboring lines)
    - Global position information
    - Grammatical pattern indicators
    
    Args:
        ssm: Self-similarity matrix (n x n)
        high_sim_threshold: Threshold for considering similarity "high" (configurable)
        
    Returns:
        Feature matrix (n x 12) with per-line features
    """
    n = ssm.shape[0]
    if n == 0:
        return np.zeros((0, 12), dtype=np.float32)
    
    features = []
    
    for i in range(n):
        row = ssm[i]  # POS similarities of line i to all lines
        
        # Statistical measures
        mean_sim = float(row.mean())
        max_sim = float(row.max())
        std_sim = float(row.std()) if n > 1 else 0.0
        
        # Quantiles for distribution shape
        q75 = float(np.quantile(row, 0.75))
        q90 = float(np.quantile(row, 0.90))
        
        # High similarity count (grammatical pattern repetition indicator)
        # Use configurable threshold instead of hardcoded 0.7
        high_sim_ratio = float((row >= high_sim_threshold).sum() / n)
        
        # Local context (neighboring lines for grammatical flow detection)
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
            mean_sim,       # 0: Mean POS similarity to all lines
            max_sim,        # 1: Max POS similarity to any line
            std_sim,        # 2: Std of POS similarities
            q75,            # 3: 75th percentile POS similarity
            q90,            # 4: 90th percentile POS similarity  
            high_sim_ratio, # 5: Fraction of high POS similarities (grammatical repetition)
            prev_sim,       # 6: POS similarity to previous line
            next_sim,       # 7: POS similarity to next line
            first_sim,      # 8: POS similarity to first line
            last_sim,       # 9: POS similarity to last line
            position,       # 10: Normalized position in song
            inv_position    # 11: Inverse position (1 - position)
        ]
        
        features.append(line_features)
    
    return np.asarray(features, dtype=np.float32)


def extract_pos_ssm_features(lines: List[str], tagset: str = 'simplified', 
                            similarity_method: str = 'combined', 
                            high_sim_threshold: float = 0.7) -> torch.Tensor:
    """
    Extract POS-SSM features for a song.
    
    This is the main feature extraction function that:
    1. Computes POS-SSM matrix based on grammatical structure
    2. Summarizes per-line features for pattern detection
    3. Returns as torch tensor
    
    Args:
        lines: List of text lines from song
        tagset: POS tagset to use - 'simplified', 'universal', or 'penn'
        similarity_method: Method for computing similarity - 'combined', 'lcs', 'position', 'jaccard'
        high_sim_threshold: Threshold for considering similarity "high"
        
    Returns:
        Feature tensor (seq_len, 12) 
    """
    if not lines:
        return torch.zeros(0, 12, dtype=torch.float32)
    
    # Remove any potential None values or empty strings
    clean_lines = [str(line) if line is not None else "" for line in lines]
    
    # Compute POS SSM
    ssm = compute_pos_ssm(clean_lines, tagset, similarity_method)
    
    # Extract per-line features
    features = summarize_pos_ssm_per_line(ssm, high_sim_threshold)
    
    # Convert to torch tensor
    return torch.from_numpy(features).float()


class POSSSMExtractor:
    """
    Modular POS-SSM feature extractor for grammatical structure analysis.
    
    This class encapsulates the POS-based feature extraction logic and can be
    easily configured and integrated into the training pipeline.
    """
    
    def __init__(self, tagset: str = 'simplified', similarity_method: str = 'combined',
                 high_sim_threshold: float = 0.7, output_dim: int = 12):
        """
        Initialize the extractor.
        
        Args:
            tagset: POS tagset to use - 'simplified', 'universal', or 'penn'
            similarity_method: Method for computing similarity - 'combined', 'lcs', 'position', 'jaccard'
            high_sim_threshold: Threshold for considering similarity "high"
            output_dim: Expected output dimension (should be 12)
        """
        self.tagset = tagset
        self.similarity_method = similarity_method
        self.high_sim_threshold = high_sim_threshold
        self.output_dim = output_dim
        
        if output_dim != 12:
            raise ValueError(f"POS-SSM features are fixed at 12 dimensions, got {output_dim}")
    
    def __call__(self, lines: List[str]) -> torch.Tensor:
        """Extract features from lines."""
        return extract_pos_ssm_features(lines, self.tagset, self.similarity_method, self.high_sim_threshold)
    
    def get_feature_names(self) -> List[str]:
        """Get names of the 12 features for interpretability."""
        return [
            'pos_mean_similarity',
            'pos_max_similarity', 
            'pos_std_similarity',
            'pos_q75_similarity',
            'pos_q90_similarity',
            'pos_high_sim_ratio',
            'pos_prev_similarity',
            'pos_next_similarity',
            'pos_first_similarity',
            'pos_last_similarity',
            'pos_position',
            'pos_inverse_position'
        ]
    
    def describe_features(self):
        """Print feature descriptions."""
        names = self.get_feature_names()
        descriptions = [
            "Average POS similarity to all other lines",
            "Maximum POS similarity to any other line",
            "Standard deviation of POS similarities", 
            "75th percentile of POS similarities",
            "90th percentile of POS similarities",
            f"Fraction of high POS similarities (>= {self.high_sim_threshold}) - grammatical repetition",
            "POS similarity to previous line",
            "POS similarity to next line", 
            "POS similarity to first line",
            "POS similarity to last line",
            "Normalized position in song [0, 1]",
            "Inverse position (1 - position)"
        ]
        
        print(f"🧩 POS-SSM Features (Grammatical Structure):")
        print(f"   Tagset: {self.tagset}")
        print(f"   Similarity method: {self.similarity_method}")
        print(f"   High similarity threshold: {self.high_sim_threshold}")
        print(f"   Features:")
        for i, (name, desc) in enumerate(zip(names, descriptions)):
            print(f"     {i:2d}. {name:22s}: {desc}")


if __name__ == "__main__":
    # Test the enhanced POS-SSM feature extraction
    test_lines = [
        "I am walking down the street",           # Test insertions/deletions
        "I will always love you",                 # Test reordering patterns
        "I will love you always",                 # Reordered version  
        "She is dancing in the moonlight",       # Different but similar structure
        "The beautiful cat sleeps peacefully",   # Different structure
        "I am walking down the street"           # Exact repeat
    ]
    
    print("🧪 Testing Enhanced POS-SSM feature extraction...")
    print(f"Input: {len(test_lines)} lines")
    
    # Test different tagsets
    print("\n🏷️ Testing different POS tagsets:")
    for tagset in ['simplified', 'universal', 'penn']:
        print(f"\n{tagset.upper()} tags:")
        for i, line in enumerate(test_lines[:3]):  # Just first 3 for brevity
            pos_seq = extract_pos_sequence(line, tagset=tagset)
            print(f"   {i}: '{line}' → {pos_seq}")
    
    # Test different similarity methods
    print("\n🔧 Testing similarity methods:")
    pos1 = extract_pos_sequence(test_lines[1], 'simplified')  # "I will always love you"
    pos2 = extract_pos_sequence(test_lines[2], 'simplified')  # "I will love you always"
    
    for method in ['combined', 'lcs', 'position', 'jaccard']:
        sim = compute_pos_similarity(pos1, pos2, method=method)
        print(f"   {method:10s}: {sim:.3f}")
    
    print(f"\n💡 LCS handles reordering better than position-wise matching!")
    
    # Test configurable thresholds
    print(f"\n📊 Testing configurable thresholds:")
    
    # Extract features with different configurations
    configs = [
        ('simplified', 'combined', 0.7),
        ('universal', 'lcs', 0.6),
        ('simplified', 'lcs', 0.8)
    ]
    
    for tagset, method, threshold in configs:
        features = extract_pos_ssm_features(test_lines, tagset, method, threshold)
        extractor = POSSSMExtractor(tagset, method, threshold)
        
        print(f"\nConfig: {tagset}, {method}, threshold={threshold}")
        print(f"Features shape: {features.shape}")
        extractor.describe_features()
        
        # Show high similarity ratio for first line
        high_sim_ratio = features[0][5]  # Index 5 is high_sim_ratio
        print(f"High similarity ratio (line 0): {high_sim_ratio:.3f}")
        
        break  # Just show one detailed config for brevity
    
    # Show SSM comparison
    print(f"\n📊 SSM comparison (LCS vs Position):")
    ssm_lcs = compute_pos_ssm(test_lines, 'simplified', 'lcs')
    ssm_pos = compute_pos_ssm(test_lines, 'simplified', 'position')
    
    print(f"LCS similarity (lines 1-2): {ssm_lcs[1, 2]:.3f}")
    print(f"Position similarity (lines 1-2): {ssm_pos[1, 2]:.3f}")
    
    print(f"\n✅ Enhanced POS-SSM test completed successfully!")
    print(f"🎯 Key improvements:")
    print(f"   • Universal POS support for better standardization")
    print(f"   • LCS similarity handles insertions/deletions")
    print(f"   • Configurable thresholds for different feature types")
    print(f"   • Multiple similarity methods for different use cases")
