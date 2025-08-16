"""
Data loading and dataset classes for BLSTM verse/chorus segmentation.
Includes anti-collapse sampling strategies based on architecture knowledge.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path


@dataclass
class Song:
    """Represents a single song with lyrics and labels."""
    id: str
    lines: List[str]
    labels: List[int]  # 0=verse, 1=chorus
    
    def __post_init__(self):
        """Validate song data."""
        if len(self.lines) != len(self.labels):
            raise ValueError(f"Length mismatch in song {self.id}: "
                           f"{len(self.lines)} lines vs {len(self.labels)} labels")
        
        # Validate labels are 0 or 1
        unique_labels = set(self.labels)
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Invalid labels in song {self.id}: {unique_labels}")


class SongsDataset(Dataset):
    """
    Dataset for loading songs from JSONL format.
    
    Expected JSONL format:
    {
      "id": "song_001",
      "lines": ["line 1", "line 2", ...],
      "labels": [0, 1, 1, 0, ...]  # 0=verse, 1=chorus
    }
    """
    
    def __init__(self, jsonl_path: str):
        self.jsonl_path = Path(jsonl_path)
        self.songs: List[Song] = []
        
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
        
        self._load_songs()
        self._compute_stats()
    
    def _load_songs(self):
        """Load songs from JSONL file."""
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    song_id = obj.get('id', f"song_{len(self.songs)}")
                    lines = obj['lines']
                    labels = obj['labels']
                    
                    song = Song(song_id, lines, labels)
                    self.songs.append(song)
                
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"⚠️  Warning: Skipping invalid line {line_num}: {e}")
                    continue
        
        if not self.songs:
            raise ValueError(f"No valid songs loaded from {self.jsonl_path}")
        
        print(f"✅ Loaded {len(self.songs)} songs from {self.jsonl_path}")
    
    def _compute_stats(self):
        """Compute dataset statistics for monitoring."""
        total_lines = sum(len(song.labels) for song in self.songs)
        total_chorus = sum(sum(song.labels) for song in self.songs)
        total_verse = total_lines - total_chorus
        
        self.stats = {
            'num_songs': len(self.songs),
            'total_lines': total_lines,
            'chorus_lines': total_chorus,
            'verse_lines': total_verse,
            'chorus_ratio': total_chorus / total_lines if total_lines > 0 else 0.0,
            'avg_lines_per_song': total_lines / len(self.songs) if self.songs else 0.0,
        }
        
        print(f"📊 Dataset stats:")
        print(f"   Songs: {self.stats['num_songs']}")
        print(f"   Total lines: {self.stats['total_lines']}")
        print(f"   Chorus: {self.stats['chorus_lines']} ({self.stats['chorus_ratio']:.2%})")
        print(f"   Verse: {self.stats['verse_lines']} ({1-self.stats['chorus_ratio']:.2%})")
        print(f"   Avg lines/song: {self.stats['avg_lines_per_song']:.1f}")
    
    def __len__(self) -> int:
        return len(self.songs)
    
    def __getitem__(self, idx: int) -> Tuple[str, List[str], List[int]]:
        """Return song data for feature extraction."""
        song = self.songs[idx]
        return song.id, song.lines, song.labels
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for balanced loss.
        Based on inverse frequency, capped to prevent extreme weighting.
        """
        verse_count = self.stats['verse_lines']
        chorus_count = self.stats['chorus_lines']
        
        # Prevent division by zero
        verse_count = max(verse_count, 1)
        chorus_count = max(chorus_count, 1)
        
        # Compute inverse frequency weights
        verse_weight = 1.0 / verse_count
        chorus_weight = 1.0 / chorus_count
        
        # Normalize so weights sum to 2.0 (for 2 classes)
        total_weight = verse_weight + chorus_weight
        verse_weight = (verse_weight / total_weight) * 2.0
        chorus_weight = (chorus_weight / total_weight) * 2.0
        
        # Cap maximum ratio to prevent extreme weighting (from architecture knowledge)
        max_ratio = 2.0  # From ANTI_COLLAPSE_CONFIG
        if chorus_weight / verse_weight > max_ratio:
            chorus_weight = verse_weight * max_ratio
        elif verse_weight / chorus_weight > max_ratio:
            verse_weight = chorus_weight * max_ratio
        
        weights = torch.tensor([verse_weight, chorus_weight], dtype=torch.float32)
        
        print(f"🔧 Class weights: verse={verse_weight:.3f}, chorus={chorus_weight:.3f}")
        print(f"   Ratio: {max(verse_weight, chorus_weight) / min(verse_weight, chorus_weight):.2f}")
        
        return weights


def create_weighted_sampler(dataset: SongsDataset) -> WeightedRandomSampler:
    """
    Create WeightedRandomSampler to encourage chorus exposure.
    
    Strategy: Weight songs by their chorus rarity to ensure diverse batches.
    Songs with fewer choruses get higher sampling probability.
    """
    song_weights = []
    
    for i in range(len(dataset)):
        _, _, labels = dataset[i]
        
        if not labels:
            weight = 1.0  # Fallback for empty songs
        else:
            chorus_ratio = sum(labels) / len(labels)
            # Higher weight for chorus-poor songs (encourage chorus exposure)
            # Weight formula: 0.5 + 0.5 * (1 - chorus_ratio)
            # Result: songs with 0% chorus get weight 1.0, songs with 100% chorus get weight 0.5
            weight = 0.5 + 0.5 * (1.0 - chorus_ratio)
        
        song_weights.append(weight)
    
    weights_tensor = torch.tensor(song_weights, dtype=torch.double)
    
    # Print sampling statistics
    avg_weight = weights_tensor.mean().item()
    max_weight = weights_tensor.max().item()
    min_weight = weights_tensor.min().item()
    
    print(f"🎲 Weighted sampler stats:")
    print(f"   Avg weight: {avg_weight:.3f}")
    print(f"   Weight range: [{min_weight:.3f}, {max_weight:.3f}]")
    print(f"   Ratio: {max_weight/min_weight:.2f}")
    
    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True
    )


# =============================================================================
# COLLATE FUNCTIONS FOR BATCHING
# =============================================================================

@dataclass
class Batch:
    """Represents a batch of processed songs."""
    song_ids: List[str]
    features: torch.Tensor  # (batch_size, max_seq_len, feature_dim)
    labels: torch.Tensor    # (batch_size, max_seq_len) with -100 for padding
    mask: torch.Tensor      # (batch_size, max_seq_len) boolean mask


def pad_sequences(sequences: List[torch.Tensor], padding_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length sequences to same length.
    
    Returns:
        padded_sequences: (batch_size, max_len, feature_dim)
        mask: (batch_size, max_len) boolean mask indicating valid positions
    """
    if not sequences:
        raise ValueError("Empty sequence list")
    
    max_len = max(seq.shape[0] for seq in sequences)
    batch_size = len(sequences)
    feature_dim = sequences[0].shape[-1]
    
    # Create padded tensor
    padded = torch.full(
        (batch_size, max_len, feature_dim),
        padding_value,
        dtype=sequences[0].dtype
    )
    
    # Create mask
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    
    # Fill in sequences
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded[i, :seq_len] = seq
        mask[i, :seq_len] = True
    
    return padded, mask


def pad_labels(label_sequences: List[List[int]], ignore_index: int = -100) -> torch.Tensor:
    """Pad label sequences with ignore_index for loss computation."""
    if not label_sequences:
        raise ValueError("Empty label sequence list")
    
    max_len = max(len(labels) for labels in label_sequences)
    batch_size = len(label_sequences)
    
    padded_labels = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)
    
    for i, labels in enumerate(label_sequences):
        seq_len = len(labels)
        padded_labels[i, :seq_len] = torch.tensor(labels, dtype=torch.long)
    
    return padded_labels


def create_dataloader(
    dataset: SongsDataset,
    feature_extractor,
    batch_size: int = 8,
    shuffle: bool = True,
    use_weighted_sampling: bool = True
) -> DataLoader:
    """
    Create DataLoader with proper collation and sampling.
    
    Args:
        dataset: SongsDataset instance
        feature_extractor: Function that extracts features from raw text
        batch_size: Batch size
        shuffle: Whether to shuffle (ignored if using weighted sampling)
        use_weighted_sampling: Whether to use anti-collapse weighted sampling
    """
    
    def collate_fn(batch_items):
        """Collate function that processes raw data through feature extractor."""
        song_ids, all_lines, all_labels = zip(*batch_items)
        
        # Extract features for each song
        features_list = []
        for lines in all_lines:
            features = feature_extractor(lines)  # Should return (seq_len, feature_dim)
            features_list.append(features)
        
        # Pad sequences
        padded_features, mask = pad_sequences(features_list)
        padded_labels = pad_labels(all_labels)
        
        return Batch(
            song_ids=list(song_ids),
            features=padded_features.float(),
            labels=padded_labels.long(),
            mask=mask
        )
    
    # Set up sampling strategy
    if use_weighted_sampling:
        sampler = create_weighted_sampler(dataset)
        shuffle = False  # Sampler handles randomization
    else:
        sampler = None
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for now
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    # Test the dataset loading
    import tempfile
    import os
    
    # Create a test JSONL file
    test_data = [
        {"id": "song_1", "lines": ["verse line 1", "chorus line 1", "chorus line 2"], "labels": [0, 1, 1]},
        {"id": "song_2", "lines": ["verse line 1", "verse line 2", "chorus line 1"], "labels": [0, 0, 1]},
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            json.dump(item, f)
            f.write('\n')
        temp_path = f.name
    
    try:
        # Test dataset loading
        dataset = SongsDataset(temp_path)
        weights = dataset.get_class_weights()
        sampler = create_weighted_sampler(dataset)
        
        print(f"\n✅ Dataset test passed!")
        print(f"   Loaded {len(dataset)} songs")
        print(f"   Class weights: {weights}")
        
    finally:
        os.unlink(temp_path)
