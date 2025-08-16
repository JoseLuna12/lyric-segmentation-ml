#!/bin/bash

# BLSTM Text Segmentation Training Script
# Hardcoded arguments for training_config.yaml
# Usage: ./train.sh

set -e  # Exit on any error

echo "Starting BLSTM Training with training_config.yaml..."
echo "=================================================="
echo

# Check if training_config.yaml exists
if [ ! -f "configs/training_config.yaml" ]; then
    echo "❌ Error: configs/training_config.yaml not found!"
    exit 1
fi

echo "Starting training..."
echo

# Main training command with hardcoded arguments
python train_with_config.py configs/training_config.yaml