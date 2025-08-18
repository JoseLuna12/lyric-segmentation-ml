#!/usr/bin/env python3
"""
Data transformation script to convert input data to JSONL format for BiLSTM training.

This script handles:
1. training_data.json - Single JSON file with all songs (stanzas structure)
2. assetsJson directory - Multiple JSON files, one per song (stanzas structure)

Output format: {"id": "...", "lines": [...], "labels": [...]}
Where labels: 0 = verse, 1 = chorus

Features:
- Proper handling of special characters and JSON escaping
- Enhanced data validation and cleaning
- No filtering of single-class songs (keeps all valid songs)
"""

import json
import os
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import re
import unicodedata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text by handling special characters and normalizing unicode.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Remove or replace problematic characters
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    
    # Replace smart quotes with regular quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Replace other problematic unicode characters
    text = text.replace('…', '...')
    text = text.replace('–', '-').replace('—', '-')
    
    # Strip whitespace and ensure single line
    text = text.strip()
    
    return text


def extract_song_data(song_data: Dict[str, Any], song_id: str) -> Dict[str, Any]:
    """
    Extract lines and labels from a song's stanzas structure.
    
    Args:
        song_data: Dictionary containing song data with 'stanzas' key
        song_id: Unique identifier for the song
        
    Returns:
        Dictionary with id, lines, and labels, or None if invalid
    """
    lines = []
    labels = []
    
    stanzas = song_data.get('stanzas', [])
    if not stanzas:
        logger.warning(f"No stanzas found for song {song_id}")
        return None
    
    for stanza in stanzas:
        stanza_lines = stanza.get('lines', [])
        for line_data in stanza_lines:
            if isinstance(line_data, dict):
                text = clean_text(line_data.get('text', ''))
                if text and len(text.strip()) > 0:  # Only add non-empty lines
                    lines.append(text)
                    # Convert label to binary: chorus=1, verse=0
                    label_str = line_data.get('label', 'verse').lower()
                    label = 1 if label_str in ['chorus', 'hook', 'refrain'] else 0
                    labels.append(label)
            elif isinstance(line_data, str):
                # Fallback for simple string format
                text = clean_text(line_data)
                if text and len(text.strip()) > 0:
                    lines.append(text)
                    labels.append(0)  # Default to verse
    
    if not lines:
        logger.warning(f"No valid lines found for song {song_id}")
        return None
    
    # Validate that we have at least some meaningful content
    if len(lines) < 2:
        logger.warning(f"Song {song_id} has too few lines ({len(lines)}), skipping")
        return None
    
    return {
        'id': song_id,
        'lines': lines,
        'labels': labels
    }


def process_training_data_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Process training_data.json format.
    Expected format: Single JSON file with all songs.
    
    Args:
        file_path: Path to training_data.json
        
    Returns:
        List of processed song dictionaries
    """
    logger.info(f"Processing training_data.json: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []
    
    processed_songs = []
    
    # Handle different possible structures
    if isinstance(data, list):
        songs = data
    elif isinstance(data, dict):
        # Try common keys for song collections
        songs = data.get('songs', data.get('data', data.get('tracks', [])))
        if not songs:
            # If no common keys found, try to use the whole dict as a single song
            songs = [data]
    else:
        logger.error(f"Unexpected data format in {file_path}")
        return []
    
    total_songs = len(songs)
    skipped_count = 0
    
    for i, song in enumerate(songs):
        # Generate song ID from filename and index
        base_name = Path(file_path).stem
        song_id = f"{base_name}_{i:04d}"
        
        processed_song = extract_song_data(song, song_id)
        if processed_song:
            processed_songs.append(processed_song)
        else:
            skipped_count += 1
    
    logger.info(f"Processed {len(processed_songs)} songs from {file_path}")
    logger.info(f"Skipped {skipped_count} songs (invalid data)")
    
    return processed_songs


def process_assets_json_directory(dir_path: str) -> List[Dict[str, Any]]:
    """
    Process assetsJson directory format.
    Expected format: Directory with multiple JSON files, one per song.
    
    Args:
        dir_path: Path to assetsJson directory
        
    Returns:
        List of processed song dictionaries
    """
    logger.info(f"Processing assetsJson directory: {dir_path}")
    
    if not os.path.isdir(dir_path):
        logger.error(f"Directory not found: {dir_path}")
        return []
    
    processed_songs = []
    json_files = list(Path(dir_path).glob("*.json"))
    total_files = len(json_files)
    skipped_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                song_data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {json_file}: {e}")
            skipped_count += 1
            continue
        
        # Use filename (without extension) as song ID
        song_id = json_file.stem
        
        processed_song = extract_song_data(song_data, song_id)
        if processed_song:
            processed_songs.append(processed_song)
        else:
            skipped_count += 1
    
    logger.info(f"Processed {len(processed_songs)} songs from {dir_path}")
    logger.info(f"Skipped {skipped_count} files (invalid data)")
    
    return processed_songs


def split_data(songs: List[Dict[str, Any]], train_ratio: float = 0.8) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split songs into training and testing sets.
    
    Args:
        songs: List of song dictionaries
        train_ratio: Fraction of data to use for training
        
    Returns:
        Tuple of (train_songs, test_songs)
    """
    random.shuffle(songs)
    split_idx = int(len(songs) * train_ratio)
    return songs[:split_idx], songs[split_idx:]


def write_jsonl(songs: List[Dict[str, Any]], output_path: str) -> None:
    """
    Write songs to JSONL format with proper JSON escaping.
    
    Args:
        songs: List of song dictionaries
        output_path: Path to output JSONL file
    """
    logger.info(f"Writing {len(songs)} songs to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for song in songs:
            # Use json.dumps to ensure proper escaping of special characters
            json_line = json.dumps(song, ensure_ascii=False, separators=(',', ':'))
            f.write(json_line + '\n')


def analyze_data(songs: List[Dict[str, Any]], name: str) -> None:
    """
    Analyze and print statistics about the processed data.
    
    Args:
        songs: List of song dictionaries
        name: Name of the dataset (for logging)
    """
    if not songs:
        logger.warning(f"No songs in {name} dataset")
        return
    
    total_lines = sum(len(song['lines']) for song in songs)
    total_chorus_lines = sum(sum(song['labels']) for song in songs)
    total_verse_lines = total_lines - total_chorus_lines
    
    chorus_ratio = total_chorus_lines / total_lines if total_lines > 0 else 0
    
    # Count songs with only verse (class 0) or only chorus (class 1)
    verse_only_songs = sum(1 for song in songs if all(label == 0 for label in song['labels']))
    chorus_only_songs = sum(1 for song in songs if all(label == 1 for label in song['labels']))
    mixed_songs = len(songs) - verse_only_songs - chorus_only_songs
    
    logger.info(f"{name} dataset statistics:")
    logger.info(f"  Songs: {len(songs)}")
    logger.info(f"  - Mixed (verse+chorus): {mixed_songs}")
    logger.info(f"  - Verse only: {verse_only_songs}")
    logger.info(f"  - Chorus only: {chorus_only_songs}")
    logger.info(f"  Total lines: {total_lines}")
    logger.info(f"  Verse lines: {total_verse_lines}")
    logger.info(f"  Chorus lines: {total_chorus_lines}")
    logger.info(f"  Chorus ratio: {chorus_ratio:.2%}")


def main():
    parser = argparse.ArgumentParser(description='Transform data to JSONL format for BiLSTM training')
    parser.add_argument('--training-data', required=True, help='Path to training_data.json')
    parser.add_argument('--assets-json', required=True, help='Path to assetsJson directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for JSONL files')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio for train/test split (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process training data
    training_songs = process_training_data_json(args.training_data)
    
    # Process assets data  
    assets_songs = process_assets_json_directory(args.assets_json)
    
    # Split training data into train/test
    if training_songs:
        train_songs, test_songs = split_data(training_songs, args.train_ratio)
    else:
        train_songs, test_songs = [], []
    
    # Validation set is all assets data
    val_songs = assets_songs
    
    # Write output files
    train_path = os.path.join(args.output_dir, 'train.jsonl')
    test_path = os.path.join(args.output_dir, 'test.jsonl')
    val_path = os.path.join(args.output_dir, 'val.jsonl')
    
    write_jsonl(train_songs, train_path)
    write_jsonl(test_songs, test_path)
    write_jsonl(val_songs, val_path)
    
    # Analyze results
    analyze_data(train_songs, "Training")
    analyze_data(test_songs, "Testing")
    analyze_data(val_songs, "Validation")
    
    logger.info("Data transformation completed successfully!")


if __name__ == "__main__":
    main()
