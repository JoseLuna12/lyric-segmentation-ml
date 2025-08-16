#!/usr/bin/env python3
"""
Inference script for BLSTM baseline model.
Predicts verse/chorus structure for new lyrics.
"""

import argparse
import torch
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add segmodel to path
sys.path.append(str(Path(__file__).parent))

from segmodel.features import FeatureExtractor
from segmodel.models import BLSTMTagger
from segmodel.utils import load_training_config


def load_model(model_path: str, device: torch.device) -> BLSTMTagger:
    """Load trained model from file."""
    print(f"📦 Loading model from {model_path}...")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer model dimensions from state dict
    # LSTM layer: 'lstm.weight_ih_l0' has shape (4*hidden_size, input_size)
    # Classifier: 'classifier.weight' has shape (num_classes, 2*hidden_size) for bidirectional
    
    lstm_weight = state_dict['lstm.weight_ih_l0']
    classifier_weight = state_dict['classifier.weight']
    
    input_size = lstm_weight.shape[1]
    lstm_4h = lstm_weight.shape[0]  # 4 * hidden_size
    hidden_size = lstm_4h // 4      # Extract hidden_size
    num_classes = classifier_weight.shape[0]
    
    print(f"🔧 Detected model architecture:")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Classes: {num_classes}")
    
    # Create model
    model = BLSTMTagger(
        feat_dim=input_size,
        hidden_dim=hidden_size,
        num_classes=num_classes,
        dropout=0.0  # No dropout during inference
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def create_default_feature_extractor() -> FeatureExtractor:
    """Create feature extractor with default configuration for inference."""
    # Standard feature configuration for prediction
    # This should match what was used during training
    feature_config = {
        'head_ssm': {
            'enabled': True,
            'dimension': 12
        }
    }
    
    extractor = FeatureExtractor(feature_config)
    print(f"🧩 Initialized feature extractor:")
    print(f"   Enabled features: {list(feature_config.keys())}")
    print(f"   Total dimension: {extractor.get_feature_dimension()}")
    
    return extractor


def predict_lyrics_structure(
    lines: List[str],
    model: BLSTMTagger,
    feature_extractor,
    device: torch.device,
    temperature: float = 1.5
) -> List[Dict[str, Any]]:
    """
    Predict structure labels for a list of lyric lines.
    
    Args:
        lines: List of lyric lines
        model: Trained model
        feature_extractor: Feature extraction function
        device: Device to run on
        temperature: Temperature for calibrated predictions
        
    Returns:
        List of prediction dictionaries
    """
    if not lines:
        return []
    
    # Extract features
    features = feature_extractor(lines)  # (seq_len, feature_dim)
    features = features.unsqueeze(0).to(device)  # (1, seq_len, feature_dim)
    
    # Create mask (all positions valid for inference)
    mask = torch.ones(1, len(lines), dtype=torch.bool, device=device)
    
    # Get predictions with temperature scaling
    with torch.no_grad():
        predictions, confidences = model.predict_with_temperature(
            features, mask, temperature=temperature
        )
    
    # Convert to results
    predictions = predictions.squeeze(0).cpu().numpy()  # (seq_len,)
    confidences = confidences.squeeze(0).cpu().numpy()  # (seq_len,)
    
    # Map predictions to labels
    label_map = {0: 'verse', 1: 'chorus'}
    
    results = []
    for i, (line, pred, conf) in enumerate(zip(lines, predictions, confidences)):
        results.append({
            'line_number': i + 1,
            'line': line.strip(),
            'predicted_label': label_map[pred],
            'confidence': float(conf)
        })
    
    return results


def load_lines_from_file(filepath: str) -> List[str]:
    """Load lines from text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def parse_lyrics_input(text: str) -> List[str]:
    """
    Parse lyrics text into individual lines.
    Handles multi-line strings, removes empty lines, and normalizes spacing.
    """
    # Split by newlines and clean up
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line:  # Skip empty lines
            lines.append(line)
    
    return lines


def load_lyrics_from_stdin() -> List[str]:
    """Load lyrics from stdin (for piping)."""
    import sys
    
    print("📝 Enter lyrics (press Ctrl+D when finished):")
    try:
        text = sys.stdin.read()
        return parse_lyrics_input(text)
    except KeyboardInterrupt:
        print("\n❌ Input cancelled")
        return []


def print_predictions(predictions: List[Dict[str, Any]]):
    """Print predictions in a nice format."""
    print(f"\n🎵 Lyrics Structure Prediction")
    print("=" * 50)
    
    for pred in predictions:
        label = pred['predicted_label'].upper()
        conf = pred['confidence']
        line = pred['line']
        
        # Color coding for terminal
        if label == 'CHORUS':
            color_start = '\033[92m'  # Green
        else:
            color_start = '\033[94m'  # Blue
        color_end = '\033[0m'
        
        print(f"{color_start}{label:<7}{color_end} ({conf:.3f}) | {line}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Predict verse/chorus structure for lyrics",
        epilog="""
Examples:
  # From text file
  python predict_baseline.py --model best_model.pt --text song.txt
  
  # Multi-line string input  
  python predict_baseline.py --model best_model.pt --lyrics "Walking down the street
  Thinking of you
  Dancing tonight"
  
  # Command line arguments
  python predict_baseline.py --model best_model.pt --lines "Line 1" "Line 2" "Line 3"
  
  # Interactive input (default)
  python predict_baseline.py --model best_model.pt
  
  # Piped input
  echo "Line 1\nLine 2" | python predict_baseline.py --model best_model.pt --stdin
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--temperature', type=float, default=1.5,
                       help='Temperature for calibrated predictions')
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--text', help='Text file with lyrics (one line per line)')
    input_group.add_argument('--lines', nargs='+', help='Lyrics lines as command line arguments')
    input_group.add_argument('--lyrics', help='Multi-line lyrics as a single string')
    input_group.add_argument('--stdin', action='store_true', help='Read lyrics from stdin (interactive or piped)')
    
    # Output arguments
    parser.add_argument('--output', help='Output JSON file (optional)')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (no terminal output)')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🎵 BLSTM Baseline Inference")
        print("=" * 30)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    
    if not args.quiet:
        print(f"🔧 Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    
    # Create feature extractor
    feature_extractor = create_default_feature_extractor()
    
    # Get input lines
    lines = []
    
    if args.text:
        lines = load_lines_from_file(args.text)
        if not args.quiet:
            print(f"📄 Loaded {len(lines)} lines from {args.text}")
            
    elif args.lines:
        lines = args.lines
        if not args.quiet:
            print(f"📝 Processing {len(lines)} lines from command line")
            
    elif args.lyrics:
        lines = parse_lyrics_input(args.lyrics)
        if not args.quiet:
            print(f"🎵 Parsed {len(lines)} lines from lyrics string")
            
    elif args.stdin:
        lines = load_lyrics_from_stdin()
        if not args.quiet and lines:
            print(f"⌨️  Loaded {len(lines)} lines from input")
            
    else:
        # Default to stdin if no input method specified
        lines = load_lyrics_from_stdin()
        if not args.quiet and lines:
            print(f"⌨️  Loaded {len(lines)} lines from input")
    
    if not lines:
        print("❌ No input lines provided")
        return
    
    # Make predictions
    if not args.quiet:
        print(f"🔮 Making predictions...")
    
    predictions = predict_lyrics_structure(
        lines=lines,
        model=model,
        feature_extractor=feature_extractor,
        device=device,
        temperature=args.temperature
    )
    
    # Print results
    if not args.quiet:
        print_predictions(predictions)
        
        # Summary statistics
        total_lines = len(predictions)
        chorus_lines = sum(1 for p in predictions if p['predicted_label'] == 'chorus')
        verse_lines = total_lines - chorus_lines
        avg_confidence = sum(p['confidence'] for p in predictions) / total_lines
        
        print(f"📊 Summary:")
        print(f"   Total lines: {total_lines}")
        print(f"   Chorus lines: {chorus_lines} ({chorus_lines/total_lines:.1%})")
        print(f"   Verse lines: {verse_lines} ({verse_lines/total_lines:.1%})")
        print(f"   Avg confidence: {avg_confidence:.3f}")
    
    # Save output if requested
    if args.output:
        output_data = {
            'model_path': args.model,
            'temperature': args.temperature,
            'total_lines': len(predictions),
            'predictions': predictions
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        if not args.quiet:
            print(f"💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
