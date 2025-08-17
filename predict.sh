#!/bin/bash

# BLSTM Text Segmentation Prediction Script
# Simple script that uses a single prediction config file
# Usage: ./predict.sh

set -e  # Exit on any error

# Config file location (everything configured here)
CONFIG="configs/prediction/default.yaml"

echo "Starting BLSTM Prediction..."
echo "============================="
echo

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "❌ Error: Config file not found: $CONFIG"
    exit 1
fi

echo "📋 Using config: $CONFIG"
echo

# Run prediction with the config (input file specified in YAML)
python predict_baseline.py --prediction-config "$CONFIG" --temperature 0.6

echo
echo "✅ Prediction completed!"
