#!/bin/bash

# BLSTM Text Segmentation Training Script
# Usage: ./train.sh

set -e  # Exit on any error

# Configuration file path
CONFIG_FILE="configs/training/training.yaml"

echo "Starting BLSTM Training with $(basename $CONFIG_FILE)..."
echo "=================================================="
echo

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: $CONFIG_FILE not found!"
    exit 1
fi

echo "Using configuration: $CONFIG_FILE"
echo "Starting training..."
echo

# Main training command with hardcoded arguments
python train_with_config.py "$CONFIG_FILE"