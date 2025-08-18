#!/usr/bin/env python3
"""
Quick data transformation utility with example usage.
"""

import json
import os
import sys
from pathlib import Path

def quick_transform():
    """Quick transformation with default paths"""
    
    # Default paths
    training_data_path = "/Users/joseluna/master/Tesis/webscrapper/train_preprocess/training_data.json"
    assets_json_path = "/Users/joseluna/master/Tesis/assets_decode/assetsJson"
    output_dir = "../data"  # Go up one level from scripts to blstm_text, then to data
    
    print("🎵 BiLSTM Data Transformation Utility")
    print("=" * 50)
    
    # Check what files exist
    training_exists = os.path.exists(training_data_path)
    assets_exists = os.path.exists(assets_json_path)
    
    print(f"Training data file: {training_data_path}")
    print(f"  Status: {'✅ Found' if training_exists else '❌ Not found'}")
    
    print(f"Assets JSON directory: {assets_json_path}")
    print(f"  Status: {'✅ Found' if assets_exists else '❌ Not found'}")
    
    if not training_exists and not assets_exists:
        print("\n❌ No input files found!")
        print("Please check the paths or provide custom paths:")
        print("python transform_data.py --training-data /path/to/training_data.json --assets-json /path/to/assetsJson")
        return
    
    # Build command
    cmd_parts = ["python", "transform_data.py"]
    
    if training_exists:
        cmd_parts.extend(["--training-data", training_data_path])
    
    if assets_exists:
        cmd_parts.extend(["--assets-json", assets_json_path])
    
    cmd_parts.extend(["--output-dir", output_dir])
    
    print(f"\n🚀 Running transformation command:")
    print(" ".join(cmd_parts))
    print()
    
    # Execute the transformation
    os.system(" ".join(cmd_parts))


def analyze_current_data():
    """Analyze the current data files"""
    
    data_dir = Path("../data")  # Go up one level from scripts
    
    print("📊 Current Data Analysis")
    print("=" * 30)
    
    for file in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        filepath = data_dir / file
        if filepath.exists():
            count = 0
            total_lines = 0
            total_chorus = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    count += 1
                    total_lines += len(data['lines'])
                    total_chorus += sum(data['labels'])
            
            chorus_ratio = total_chorus / total_lines if total_lines > 0 else 0
            
            print(f"{file}:")
            print(f"  Songs: {count}")
            print(f"  Total lines: {total_lines}")
            print(f"  Chorus lines: {total_chorus}")
            print(f"  Chorus ratio: {chorus_ratio:.2%}")
            print()
        else:
            print(f"{file}: Not found")


def show_sample():
    """Show sample from current data"""
    
    train_file = Path("../data/train.jsonl")  # Go up one level from scripts
    
    if not train_file.exists():
        print("❌ No training data found. Run transformation first.")
        return
    
    print("🎼 Sample Training Data")
    print("=" * 30)
    
    with open(train_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line:
            data = json.loads(first_line)
            
            print(f"Song ID: {data['id']}")
            print(f"Total lines: {len(data['lines'])}")
            print()
            
            print("First 10 lines with labels:")
            for i in range(min(10, len(data['lines']))):
                label = "CHORUS" if data['labels'][i] == 1 else "VERSE"
                print(f"[{label:6}] {data['lines'][i]}")
            
            if len(data['lines']) > 10:
                print("...")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            analyze_current_data()
        elif sys.argv[1] == "sample":
            show_sample()
        elif sys.argv[1] == "quick":
            quick_transform()
        else:
            print("Usage: python quick_transform.py [quick|analyze|sample]")
    else:
        quick_transform()
