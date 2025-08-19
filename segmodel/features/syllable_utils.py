"""
Syllable processing utilities for syllable-based SSM features.
Provides functions for counting syllables and extracting syllable patterns.
"""

import re
from typing import List, Dict
import string


def count_syllables_in_word(word: str) -> int:
    """
    Count syllables in a single word using heuristic rules.
    
    This is a simplified syllable counter that works reasonably well for English.
    For more accuracy, we could integrate with a phonetic dictionary, but this
    heuristic approach should be sufficient for rhythm pattern detection.
    
    Args:
        word: Input word to count syllables
        
    Returns:
        Number of syllables (minimum 1)
    """
    if not word or not word.strip():
        return 0
    
    # Clean the word - remove punctuation and convert to lowercase
    word = word.lower().strip(string.punctuation)
    
    if not word:
        return 0
    
    # Handle common exceptions and contractions
    word = re.sub(r"'", "", word)  # Remove apostrophes
    
    # Count vowel groups (consecutive vowels count as one syllable)
    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False
    
    for i, char in enumerate(word):
        is_vowel = char in vowels
        
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        
        prev_was_vowel = is_vowel
    
    # Handle silent 'e' at the end
    if word.endswith('e') and syllable_count > 1:
        syllable_count -= 1
    
    # Handle 'le' ending (like "apple", "simple")
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        syllable_count += 1
    
    # Every word has at least one syllable
    return max(1, syllable_count)


def count_syllables_in_line(line: str) -> int:
    """
    Count total syllables in a line of text.
    
    Args:
        line: Input line of text
        
    Returns:
        Total syllable count for the line
    """
    if not line or not line.strip():
        return 0
    
    # Split into words and count syllables for each
    words = re.findall(r'\b\w+\b', line.lower())
    total_syllables = sum(count_syllables_in_word(word) for word in words)
    
    return total_syllables


def extract_syllable_pattern(line: str) -> List[int]:
    """
    Extract syllable count pattern for each word in a line.
    
    Args:
        line: Input line of text
        
    Returns:
        List of syllable counts per word
    """
    if not line or not line.strip():
        return []
    
    # Split into words and get syllable count for each
    words = re.findall(r'\b\w+\b', line.lower())
    pattern = [count_syllables_in_word(word) for word in words]
    
    return pattern


def normalize_syllable_counts(counts: List[int], mode: str = "raw") -> List[float]:
    """
    Normalize syllable counts based on the specified mode.
    
    Args:
        counts: List of syllable counts
        mode: Normalization mode - "raw" or "normalized"
        
    Returns:
        Normalized counts as floats
    """
    if not counts:
        return []
    
    if mode == "raw":
        # Return as-is but as floats
        return [float(count) for count in counts]
    
    elif mode == "normalized":
        # Normalize by total length (syllables per word ratio)
        total_syllables = sum(counts)
        num_words = len(counts)
        
        if num_words == 0:
            return []
        
        # Return syllables per word for each position
        syllables_per_word = total_syllables / num_words
        return [count / syllables_per_word if syllables_per_word > 0 else 0.0 for count in counts]
    
    else:
        raise ValueError(f"Unknown normalization mode: {mode}. Use 'raw' or 'normalized'")


def syllable_pattern_to_string(pattern: List[int]) -> str:
    """
    Convert syllable pattern to string representation for similarity comparison.
    
    Args:
        pattern: List of syllable counts per word
        
    Returns:
        String representation of the pattern
    """
    return ",".join(map(str, pattern))


def compute_levenshtein_distance_lists(list1: List[int], list2: List[int]) -> int:
    """
    Compute Levenshtein (edit) distance between two lists of integers.
    
    Args:
        list1, list2: Input lists of integers
        
    Returns:
        Edit distance between the lists
    """
    if len(list1) < len(list2):
        return compute_levenshtein_distance_lists(list2, list1)
    
    if len(list2) == 0:
        return len(list1)
    
    previous_row = list(range(len(list2) + 1))
    
    for i, item1 in enumerate(list1):
        current_row = [i + 1]
        
        for j, item2 in enumerate(list2):
            # Cost of insertions, deletions, and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (item1 != item2)
            
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]


def normalized_levenshtein_similarity_lists(list1: List[int], list2: List[int]) -> float:
    """
    Compute normalized Levenshtein similarity between two lists of integers.
    
    Args:
        list1, list2: Input lists of integers
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not list1 and not list2:
        return 1.0
    
    max_len = max(len(list1), len(list2))
    if max_len == 0:
        return 1.0
    
    distance = compute_levenshtein_distance_lists(list1, list2)
    similarity = 1.0 - (distance / max_len)
    
    return max(0.0, similarity)


def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.
    
    Args:
        s1, s2: Input strings
        
    Returns:
        Edit distance between the strings
    """
    if len(s1) < len(s2):
        return compute_levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, and substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            
            current_row.append(min(insertions, deletions, substitutions))
        
        previous_row = current_row
    
    return previous_row[-1]


def normalized_levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity between two strings.
    
    Args:
        s1, s2: Input strings
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not s1 and not s2:
        return 1.0
    
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    
    distance = compute_levenshtein_distance(s1, s2)
    similarity = 1.0 - (distance / max_len)
    
    return max(0.0, similarity)


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1, vec2: Input vectors
        
    Returns:
        Cosine similarity score between -1.0 and 1.0
    """
    if not vec1 or not vec2:
        return 0.0
    
    # Pad vectors to same length with zeros
    max_len = max(len(vec1), len(vec2))
    vec1_padded = vec1 + [0.0] * (max_len - len(vec1))
    vec2_padded = vec2 + [0.0] * (max_len - len(vec2))
    
    # Compute dot product
    dot_product = sum(a * b for a, b in zip(vec1_padded, vec2_padded))
    
    # Compute magnitudes
    magnitude1 = sum(a * a for a in vec1_padded) ** 0.5
    magnitude2 = sum(b * b for b in vec2_padded) ** 0.5
    
    # Avoid division by zero
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


if __name__ == "__main__":
    # Test the syllable processing utilities
    print("🧪 Testing Syllable Processing Utilities...")
    
    # Test syllable counting
    test_words = [
        "hello", "world", "beautiful", "rhythm", "pattern",
        "apple", "simple", "create", "love", "thinking"
    ]
    
    print(f"\n🔍 Syllable counting test:")
    for word in test_words:
        count = count_syllables_in_word(word)
        print(f"   '{word}': {count} syllables")
    
    # Test line processing
    test_lines = [
        "Walking down this street tonight",      # 7 syllables
        "Thinking of you every day",           # 6 syllables  
        "Dancing in the pale moonlight",       # 7 syllables
        "Beautiful dreams come true"           # 6 syllables
    ]
    
    print(f"\n🔍 Line syllable counting test:")
    for line in test_lines:
        total = count_syllables_in_line(line)
        pattern = extract_syllable_pattern(line)
        pattern_str = syllable_pattern_to_string(pattern)
        print(f"   '{line}'")
        print(f"     Total: {total} syllables")
        print(f"     Pattern: {pattern} -> '{pattern_str}'")
    
    # Test similarity functions
    print(f"\n🔍 Similarity computation test:")
    pattern1 = syllable_pattern_to_string([2, 1, 1, 1, 2])  # "Walking down this street tonight"
    pattern2 = syllable_pattern_to_string([2, 1, 1, 1, 2])  # "Dancing in the pale moonlight"
    pattern3 = syllable_pattern_to_string([2, 1, 1, 2, 1])  # "Thinking of you every day"
    
    lev_sim_same = normalized_levenshtein_similarity(pattern1, pattern2)
    lev_sim_diff = normalized_levenshtein_similarity(pattern1, pattern3)
    
    print(f"   Pattern 1: '{pattern1}'")
    print(f"   Pattern 2: '{pattern2}'")
    print(f"   Pattern 3: '{pattern3}'")
    print(f"   Levenshtein similarity (1-2): {lev_sim_same:.3f}")
    print(f"   Levenshtein similarity (1-3): {lev_sim_diff:.3f}")
    
    # Test cosine similarity
    vec1 = [2.0, 1.0, 1.0, 1.0, 2.0]
    vec2 = [2.0, 1.0, 1.0, 1.0, 2.0]
    vec3 = [2.0, 1.0, 1.0, 2.0, 1.0]
    
    cos_sim_same = compute_cosine_similarity(vec1, vec2)
    cos_sim_diff = compute_cosine_similarity(vec1, vec3)
    
    print(f"   Cosine similarity (1-2): {cos_sim_same:.3f}")
    print(f"   Cosine similarity (1-3): {cos_sim_diff:.3f}")
    
    print(f"\n✅ Syllable processing utilities test completed!")


def normalize_features(features: list, method: str = "zscore") -> list:
    """
    Normalize feature matrix using specified method.
    
    Args:
        features: Feature matrix (numpy array or list)
        method: "zscore" or "minmax"
        
    Returns:
        Normalized features
    """
    import numpy as np
    
    features = np.asarray(features, dtype=np.float32)
    
    if method == "zscore":
        # Z-score normalization
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        normalized = (features - mean) / std
        
    elif method == "minmax":
        # Min-max normalization to [0, 1]
        min_vals = features.min(axis=0, keepdims=True)
        max_vals = features.max(axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        # Avoid division by zero
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        normalized = (features - min_vals) / range_vals
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized
