"""
Word2Vec Embeddings feature extraction.
Implements Word2Vec-based features using pre-trained Google News vectors.
Follows established SSM pattern with configurable modes and similarity metrics.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union
import logging
import warnings
from pathlib import Path

# Global singleton for Word2Vec model
_word2vec_models = {}
_word2vec_model_loading = {}

def get_word2vec_model(model_name: str = "word2vec-google-news-300"):
    """
    Singleton pattern for Word2Vec model loading.
    Load Word2Vec model once per session to avoid repeated loading overhead.
    
    Args:
        model_name: Name of the Word2Vec model to load
    """
    global _word2vec_models, _word2vec_model_loading
    
    if model_name in _word2vec_models:
        return _word2vec_models[model_name]
    
    if model_name in _word2vec_model_loading and _word2vec_model_loading[model_name]:
        # Another thread is loading, wait for it
        import time
        while _word2vec_model_loading.get(model_name, False) and model_name not in _word2vec_models:
            time.sleep(0.1)
        return _word2vec_models.get(model_name)
    
    _word2vec_model_loading[model_name] = True
    
    try:
        import gensim.downloader as api
        print(f"🔤 Loading Word2Vec model '{model_name}' (this may take a moment)...")
        
        # Check if model is already downloaded
        try:
            _word2vec_models[model_name] = api.load(model_name)
            print(f"✅ Word2Vec model '{model_name}' loaded successfully!")
        except Exception as e:
            print(f"⚠️  Error loading Word2Vec model '{model_name}': {e}")
            print(f"🔄 Attempting to download Word2Vec model '{model_name}' (this is a one-time setup)...")
            _word2vec_models[model_name] = api.load(model_name)
            print(f"✅ Word2Vec model '{model_name}' downloaded and loaded successfully!")
            
    except ImportError:
        raise ImportError("gensim is required for Word2Vec embeddings. Install with: pip install gensim")
    except Exception as e:
        print(f"❌ Failed to load Word2Vec model '{model_name}': {e}")
        _word2vec_models[model_name] = None
    finally:
        _word2vec_model_loading[model_name] = False
    
    return _word2vec_models.get(model_name)


def text_to_word2vec_embedding(text: str, model, normalize: bool = True) -> Optional[np.ndarray]:
    """
    Convert text to Word2Vec embedding by averaging word vectors.
    
    Args:
        text: Input text line
        model: Word2Vec model
        normalize: Whether to normalize the final embedding
        
    Returns:
        Averaged embedding vector (300D) or None if no words found
    """
    if not text.strip():
        return None
    
    words = text.lower().split()
    vectors = []
    
    for word in words:
        if word in model:
            vectors.append(model[word])
    
    if not vectors:
        return None
    
    # Average the word vectors
    embedding = np.mean(vectors, axis=0)
    
    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
    
    return embedding


def compute_similarity(vec1: np.ndarray, vec2: np.ndarray, metric: str = "cosine") -> float:
    """
    Compute similarity between two vectors.
    
    Args:
        vec1, vec2: Input vectors
        metric: Similarity metric ("cosine" or "dot")
        
    Returns:
        Similarity score
    """
    if vec1 is None or vec2 is None:
        return 0.0
    
    if metric == "cosine":
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    elif metric == "dot":
        return float(np.dot(vec1, vec2))
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


def compute_word2vec_ssm(lines: List[str], model, normalize: bool = True, 
                        similarity_metric: str = "cosine") -> np.ndarray:
    """
    Compute Word2Vec Self-Similarity Matrix (SSM) between lines.
    
    Args:
        lines: List of text lines
        model: Word2Vec model
        normalize: Whether to normalize embeddings
        similarity_metric: Similarity metric ("cosine" or "dot")
        
    Returns:
        SSM matrix (n x n) where entry (i,j) is similarity between line i and j
    """
    n = len(lines)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    
    # Get embeddings for all lines
    embeddings = []
    for line in lines:
        embedding = text_to_word2vec_embedding(line, model, normalize)
        embeddings.append(embedding)
    
    # Compute similarity matrix
    ssm = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(n):
            similarity = compute_similarity(embeddings[i], embeddings[j], similarity_metric)
            ssm[i, j] = similarity
    
    return ssm


def extract_word2vec_summary_features(lines: List[str], model, normalize: bool = True,
                                     similarity_metric: str = "cosine", 
                                     high_sim_threshold: float = 0.8) -> torch.Tensor:
    """
    Extract 12D Word2Vec summary features following SSM pattern.
    
    Features extracted:
    1. Mean embedding magnitude
    2. Max embedding magnitude  
    3. Std embedding magnitude
    4. Similarity to previous line (cosine or dot)
    5. Similarity to next line (cosine or dot)
    6. Similarity to first line (cosine or dot)
    7. Similarity to last line (cosine or dot)
    8. High similarity count ratio
    9. Position-weighted similarity
    10. Inverse position-weighted similarity
    11. Line position (0-1)
    12. Inverse line position (1-0)
    
    Args:
        lines: List of text lines
        model: Word2Vec model
        normalize: Whether to normalize embeddings
        similarity_metric: Similarity metric ("cosine" or "dot")
        high_sim_threshold: Threshold for high similarity detection
        
    Returns:
        Feature tensor (seq_len, 12)
    """
    n = len(lines)
    if n == 0:
        return torch.zeros(0, 12, dtype=torch.float32)
    
    # Get embeddings for all lines
    embeddings = []
    magnitudes = []
    
    for line in lines:
        embedding = text_to_word2vec_embedding(line, model, normalize)
        embeddings.append(embedding)
        if embedding is not None:
            magnitude = np.linalg.norm(embedding)
            magnitudes.append(magnitude)
        else:
            magnitudes.append(0.0)
    
    # Compute SSM for similarity-based features
    ssm = compute_word2vec_ssm(lines, model, normalize, similarity_metric)
    
    # Extract features for each line
    features = []
    
    for i in range(n):
        line_features = []
        
        # Features 1-3: Embedding magnitude statistics
        if magnitudes:
            line_features.append(magnitudes[i])  # 1. Current magnitude
            line_features.append(max(magnitudes))  # 2. Max magnitude
            line_features.append(np.std(magnitudes) if len(magnitudes) > 1 else 0.0)  # 3. Std magnitude
        else:
            line_features.extend([0.0, 0.0, 0.0])
        
        # Features 4-7: Directional similarities
        line_features.append(ssm[i, i-1] if i > 0 else 0.0)  # 4. Previous line
        line_features.append(ssm[i, i+1] if i < n-1 else 0.0)  # 5. Next line
        line_features.append(ssm[i, 0])  # 6. First line
        line_features.append(ssm[i, n-1])  # 7. Last line
        
        # Feature 8: High similarity count ratio
        high_sim_count = np.sum(ssm[i, :] >= high_sim_threshold)
        line_features.append(high_sim_count / n if n > 0 else 0.0)
        
        # Features 9-10: Position-weighted similarities
        positions = np.arange(n)
        weights = positions + 1  # 1 to n
        inv_weights = n - positions  # n to 1
        
        position_weighted_sim = np.sum(ssm[i, :] * weights) / np.sum(weights) if np.sum(weights) > 0 else 0.0
        inv_position_weighted_sim = np.sum(ssm[i, :] * inv_weights) / np.sum(inv_weights) if np.sum(inv_weights) > 0 else 0.0
        
        line_features.append(position_weighted_sim)  # 9. Position-weighted similarity
        line_features.append(inv_position_weighted_sim)  # 10. Inverse position-weighted similarity
        
        # Features 11-12: Position indicators
        line_features.append(i / (n - 1) if n > 1 else 0.0)  # 11. Line position (0-1)
        line_features.append((n - 1 - i) / (n - 1) if n > 1 else 0.0)  # 12. Inverse line position (1-0)
        
        features.append(line_features)
    
    return torch.tensor(features, dtype=torch.float32)


class Word2VecEmbeddingsExtractor:
    """
    Word2Vec embeddings feature extractor following established SSM patterns.
    
    Supports both summary (12D) and complete (300D) modes with configurable similarity metrics.
    """
    
    def __init__(self, 
                 model: str = "word2vec-google-news-300",
                 mode: str = "summary",
                 normalize: bool = True,
                 similarity_metric: str = "cosine",
                 high_sim_threshold: float = 0.8):
        """
        Initialize Word2Vec embeddings extractor.
        
        Args:
            model: Name of the Word2Vec model to use
            mode: "summary" (12D) or "complete" (300D)
            normalize: Whether to normalize embeddings
            similarity_metric: "cosine" or "dot" similarity
            high_sim_threshold: Threshold for high similarity detection
        """
        self.model_name = model
        self.mode = mode
        self.normalize = normalize
        self.similarity_metric = similarity_metric
        self.high_sim_threshold = high_sim_threshold
        
        # Set dimension automatically based on mode (no config override)
        if mode == "summary":
            self.dimension = 12  # Always 12D for SSM consistency
        elif mode == "complete":
            self.dimension = 300  # Full Word2Vec dimension
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'summary' or 'complete'")
        
        # Validate similarity metric
        if similarity_metric not in ["cosine", "dot"]:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}. Must be 'cosine' or 'dot'")
        
        # Load model (singleton pattern)
        self.model = get_word2vec_model(self.model_name)
        if self.model is None:
            raise RuntimeError(f"Failed to load Word2Vec model: {self.model_name}")
        
        print(f"🔤 Word2Vec extractor initialized: model='{self.model_name}', {mode} mode ({self.dimension}D), {similarity_metric} similarity")
    
    def __call__(self, lines: List[str]) -> torch.Tensor:
        """
        Extract Word2Vec features from lines.
        
        Args:
            lines: List of text lines
            
        Returns:
            Feature tensor (seq_len, dimension)
        """
        if not lines:
            return torch.zeros(0, self.dimension, dtype=torch.float32)
        
        if self.mode == "summary":
            return extract_word2vec_summary_features(
                lines, self.model, self.normalize, 
                self.similarity_metric, self.high_sim_threshold
            )
        elif self.mode == "complete":
            # Extract complete 300D embeddings
            embeddings = []
            for line in lines:
                embedding = text_to_word2vec_embedding(line, self.model, self.normalize)
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Use zero vector for lines without embeddings
                    embeddings.append(np.zeros(300, dtype=np.float32))
            
            # Convert to numpy array first, then to tensor (more efficient)
            embeddings_array = np.array(embeddings, dtype=np.float32)
            return torch.tensor(embeddings_array, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
