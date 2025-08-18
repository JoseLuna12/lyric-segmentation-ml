#!/usr/bin/env python3
"""
Inference script for BLSTM baseline model.
Predicts verse/chorus structure for new lyrics.
"""

import argparse
import torch
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add segmodel to path
sys.path.append(str(Path(__file__).parent))

from segmodel.features import FeatureExtractor
from segmodel.models import BLSTMTagger
from segmodel.utils import load_training_config
from segmodel.utils.prediction_config import (
    PredictionConfig, 
    load_prediction_config, 
    auto_detect_prediction_config,
    create_prediction_config_from_training_config,
    create_prediction_config_from_training_session,
    get_feature_extractor_from_config
)



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
    # Count LSTM layers by looking for weight keys
    forward_layer_keys = [key for key in state_dict.keys() 
                         if key.startswith('lstm.weight_ih_l') and '_reverse' not in key]
    num_layers = len(forward_layer_keys)
    
    print(f"🔧 Detected model architecture:")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Classes: {num_classes}")
    print(f"   Layers: {num_layers}")
    
    # Create model (num_layers defaults to 1 in BLSTMTagger)
    model = BLSTMTagger(
        feat_dim=input_size,
        hidden_dim=hidden_size,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=0.0  # No dropout during inference
    )
    
    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def create_feature_extractor_from_training_config(config_path: str) -> FeatureExtractor:
    """
    Create feature extractor from training configuration file.
    
    Args:
        config_path: Path to training config YAML file
        
    Returns:
        FeatureExtractor configured to match training setup
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config file not found: {config_path}")
    
    print(f"📋 Loading training config from: {config_path}")
    config = load_training_config(config_path)
    
    # Build feature config from training config
    feature_config = {}
    
    if config.head_ssm_enabled:
        feature_config['head_ssm'] = {
            'enabled': True,
            'output_dim': config.head_ssm_dimension,
            'head_words': config.head_ssm_words
        }
        
    if config.tail_ssm_enabled:
        feature_config['tail_ssm'] = {
            'enabled': True,
            'output_dim': config.tail_ssm_dimension,
            'tail_words': config.tail_ssm_words
        }
        
    if config.phonetic_ssm_enabled:
        feature_config['phonetic_ssm'] = {
            'enabled': True,
            'output_dim': config.phonetic_ssm_dimension,
            'mode': config.phonetic_ssm_mode,
            'similarity_method': config.phonetic_ssm_similarity_method,
            'normalize': config.phonetic_ssm_normalize,
            'normalize_method': config.phonetic_ssm_normalize_method,
            'high_sim_threshold': config.phonetic_ssm_high_sim_threshold
        }
        
    if config.pos_ssm_enabled:
        feature_config['pos_ssm'] = {
            'enabled': True,
            'output_dim': config.pos_ssm_dimension,
            'tagset': config.pos_ssm_tagset,
            'similarity_method': config.pos_ssm_similarity_method,
            'high_sim_threshold': config.pos_ssm_high_sim_threshold
        }
        
    if config.string_ssm_enabled:
        feature_config['string_ssm'] = {
            'enabled': True,
            'output_dim': config.string_ssm_dimension,
            'case_sensitive': config.string_ssm_case_sensitive,
            'remove_punctuation': config.string_ssm_remove_punctuation,
            'similarity_threshold': config.string_ssm_similarity_threshold,
            'similarity_method': config.string_ssm_similarity_method
        }
    
    if not feature_config:
        raise ValueError(f"No features enabled in training config: {config_path}")
    
    extractor = FeatureExtractor(feature_config)
    print(f"🧩 Initialized feature extractor from training config:")
    enabled_features = [name for name, config in feature_config.items() if config.get('enabled', False)]
    print(f"   Enabled features: {enabled_features}")
    print(f"   Total dimension: {extractor.get_feature_dimension()}")
    
    return extractor


def load_calibration_from_session(session_dir: str) -> Dict[str, Any]:
    """
    Load calibration parameters from training session.
    
    Args:
        session_dir: Path to training session directory
        
    Returns:
        Calibration info dictionary, or None if not found
    """
    session_path = Path(session_dir)
    calibration_file = session_path / "calibration.json"
    
    if not calibration_file.exists():
        print(f"⚠️  No calibration file found: {calibration_file}")
        return None
    
    try:
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        
        print(f"📊 Loaded calibration from: {calibration_file}")
        print(f"   Method: {calibration_data.get('method', 'unknown')}")
        if 'params' in calibration_data:
            params = calibration_data['params']
            if 'temperature' in params:
                print(f"   Temperature: {params['temperature']:.3f}")
            if 'A' in params and 'B' in params:
                print(f"   Platt coefficients: A={params['A']:.3f}, B={params['B']:.3f}")
        
        return calibration_data
        
    except Exception as e:
        print(f"⚠️  Error loading calibration: {e}")
        return None


def select_calibration_method(
    calibration_info=None,
    config_method="auto", 
    config_temp=1.0,
    config_platt_A=1.0, 
    config_platt_B=0.0,
    cli_method=None,
    cli_temp=None,
    cli_platt_A=None,
    cli_platt_B=None,
    quiet=False
):
    """
    Select calibration method and parameters based on priority:
    1. CLI overrides (highest priority)
    2. Config settings (medium priority) 
    3. Auto-selection from calibration_info (lowest priority, but recommended)
    
    Returns:
        method: Selected calibration method ('temperature', 'platt', 'none')
        params: Dictionary with method-specific parameters
    """
    
    # Priority 1: CLI overrides everything
    if cli_method is not None:
        method = cli_method
        if method == 'temperature':
            temp = cli_temp if cli_temp is not None else config_temp
            params = {'temperature': temp}
            if not quiet:
                print(f"🔧 Using CLI calibration method: {method} (temperature: {temp:.3f})")
        elif method == 'platt':
            A = cli_platt_A if cli_platt_A is not None else config_platt_A
            B = cli_platt_B if cli_platt_B is not None else config_platt_B
            params = {'A': A, 'B': B}
            if not quiet:
                print(f"🔧 Using CLI calibration method: {method} (A: {A:.3f}, B: {B:.3f})")
        elif method == 'none':
            params = {}
            if not quiet:
                print(f"🔧 Using CLI calibration method: none (no calibration)")
        else:
            # Fall back to auto if invalid CLI method
            method = 'auto'
            params = {}
    else:
        method = config_method
        params = {}
    
    # Priority 2: Handle auto mode and config fallbacks
    if method == 'auto':
        if calibration_info and 'all_results' in calibration_info:
            # Auto-select best method from calibration.json (lowest ECE_after)
            all_results = calibration_info['all_results']
            if all_results:
                best_result = min(all_results, key=lambda r: r.get('ece_after', float('inf')))
                auto_method = best_result['method']
                auto_params = best_result['params']
                
                if not quiet:
                    print(f"📊 Auto-selected calibration method: {auto_method} (ECE: {best_result.get('ece_after', 'N/A'):.4f})")
                    if auto_method == 'temperature':
                        print(f"🎯 Using calibrated temperature: {auto_params['temperature']:.3f}")
                    elif auto_method == 'platt':
                        print(f"🎯 Using calibrated Platt scaling: A={auto_params['A']:.3f}, B={auto_params['B']:.3f}")
                
                return auto_method, auto_params
        
        # No calibration info available, fall back to temperature with config values
        method = 'temperature'
        params = {'temperature': config_temp}
        if not quiet:
            print(f"⚠️  No calibration file found, using config temperature: {config_temp}")
    
    elif method == 'temperature':
        temp = cli_temp if cli_temp is not None else config_temp
        params = {'temperature': temp}
        if not quiet and cli_temp is None:
            print(f"🎯 Using config temperature: {temp:.3f}")
            
    elif method == 'platt':
        A = cli_platt_A if cli_platt_A is not None else config_platt_A
        B = cli_platt_B if cli_platt_B is not None else config_platt_B
        params = {'A': A, 'B': B}
        if not quiet and cli_platt_A is None and cli_platt_B is None:
            print(f"🎯 Using config Platt scaling: A={A:.3f}, B={B:.3f}")
    
    elif method == 'none':
        params = {}
        if not quiet:
            print(f"🎯 No calibration applied")
    
    return method, params


def predict_lyrics_structure(
    lines: List[str],
    model: BLSTMTagger,
    feature_extractor,
    device: torch.device,
    calibration_method: str = "none",
    temperature: float = 1.0,
    platt_A: float = 1.0,
    platt_B: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Predict structure labels for a list of lyric lines.
    
    Args:
        lines: List of lyric lines
        model: Trained model
        feature_extractor: Feature extraction function
        device: Device to run on
        calibration_method: Calibration method ('temperature', 'platt', 'none')
        temperature: Temperature for temperature scaling
        platt_A: Platt scaling A coefficient
        platt_B: Platt scaling B coefficient
        
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
    
    # Get raw model predictions
    with torch.no_grad():
        if calibration_method == 'temperature':
            # Use temperature scaling
            predictions, confidences = model.predict_with_temperature(
                features, mask, temperature=temperature
            )
        elif calibration_method == 'platt':
            # Get raw predictions and apply Platt scaling
            logits = model(features, mask)  # (1, seq_len, num_classes)
            probs = torch.softmax(logits, dim=-1)
            max_probs, predictions = torch.max(probs, dim=-1)
            
            # Apply Platt scaling: sigmoid(A * confidence + B)
            calibrated_confidences = torch.sigmoid(platt_A * max_probs + platt_B)
            
            predictions = predictions.squeeze(0)  # (seq_len,)
            confidences = calibrated_confidences.squeeze(0)  # (seq_len,)
        else:
            # No calibration, use temperature=1.0
            predictions, confidences = model.predict_with_temperature(
                features, mask, temperature=1.0
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


def create_plain_text_output(predictions: List[Dict[str, Any]]) -> str:
    """Create plain text output without colors (for AI-less machines)."""
    lines = []
    lines.append("Lyrics Structure Prediction")
    lines.append("=" * 50)
    lines.append("")
    
    for pred in predictions:
        label = pred['predicted_label'].upper()
        conf = pred['confidence']
        line = pred['line']
        lines.append(f"{label:<7} ({conf:.3f}) | {line}")
    
    return '\n'.join(lines)


def get_model_name(model_path: str) -> str:
    """Extract model name from path for folder naming."""
    # Get the parent directory name (session name)
    session_dir = os.path.basename(os.path.dirname(model_path))
    # Clean up the session name for folder naming
    model_name = session_dir.replace('session_', '').replace('_', '-')
    return model_name


def save_parameters(output_dir: str, args, pred_config: PredictionConfig):
    """Save the parameters used for this prediction run."""
    import json
    import datetime
    
    params = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_path': args.model,
        'prediction_config_path': getattr(args, 'prediction_config', None),
        'temperature': pred_config.temperature,
        'input_method': 'file' if args.text else 'default',
        'input_file': pred_config.input_file,
        'output_dir': pred_config.output_dir,
        'device': pred_config.device,
        'quiet_mode': pred_config.quiet,
        'features_enabled': list(pred_config.features.keys()),
        'total_feature_dim': sum(f.get('dim', 0) for f in pred_config.features.values())
    }
    
    params_file = os.path.join(output_dir, 'parameters.json')
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    return params_file


def main():
    parser = argparse.ArgumentParser(
        description="Predict verse/chorus structure for lyrics",
        epilog="""
Examples:
  # From text file (with training session config for feature compatibility)
  python predict_baseline.py --model training_sessions/session_*/best_model.pt --config training_sessions/session_*/ --text song.txt
  
  # Multi-line string input with specific config
  python predict_baseline.py --model best_model.pt --config configs/aggressive_config.yaml --lyrics "Walking down the street
  Thinking of you
  Dancing tonight"
  
  # Command line arguments (uses default comprehensive features if no config)
  python predict_baseline.py --model best_model.pt --lines "Line 1" "Line 2" "Line 3"
  
  # Interactive input with training session config
  python predict_baseline.py --model training_sessions/session_*/best_model.pt --config training_sessions/session_*/
  
  # Piped input
  echo "Line 1\nLine 2" | python predict_baseline.py --model best_model.pt --stdin
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', help='Path to trained model (or use --session)')
    parser.add_argument('--session', help='Path to training session directory (contains model + config)')
    parser.add_argument('--prediction-config', help='Path to custom prediction config file')
    parser.add_argument('--train-config-file', help='Path to training config file (legacy)')
    
    # Calibration arguments
    parser.add_argument('--calibration-method', choices=['auto', 'temperature', 'platt', 'none'], 
                       help='Calibration method (overrides config)')
    parser.add_argument('--temperature', type=float, help='Temperature for temperature scaling (overrides config)')
    parser.add_argument('--platt-A', type=float, help='Platt scaling A coefficient (overrides config)')
    parser.add_argument('--platt-B', type=float, help='Platt scaling B coefficient (overrides config)')
    
    # Input arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--text', help='Text file with lyrics (default: predict_lyric.txt)')
    input_group.add_argument('--lines', nargs='+', help='Lyrics lines as command line arguments')
    input_group.add_argument('--lyrics', help='Multi-line lyrics as a single string')
    input_group.add_argument('--stdin', action='store_true', help='Read lyrics from stdin (interactive or piped)')
    
    # Output arguments
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
    
    # Load configuration and create feature extractor
    feature_extractor = None
    pred_config = None
    
    # Determine model path
    model_path = None
    calibration_info = None
    
    if args.session:
        # Use training session directory (PREFERRED - everything in one place)
        pred_config, model_path = create_prediction_config_from_training_session(args.session)
        feature_extractor = get_feature_extractor_from_config(pred_config)
        
        # Load calibration from session
        calibration_info = load_calibration_from_session(args.session)
        
        if not args.quiet:
            print(f"🎯 Using training session as complete source: {args.session}")
    elif args.train_config_file:
        # Use training config file as single source of truth for features
        pred_config = create_prediction_config_from_training_config(args.train_config_file)
        feature_extractor = get_feature_extractor_from_config(pred_config)
        model_path = args.model
        if not args.quiet:
            print(f"🔧 Using training config as source of truth: {args.train_config_file}")
    elif args.prediction_config:
        # Use specified custom prediction config (may include training session)
        config_result = load_prediction_config(args.prediction_config)
        if isinstance(config_result, tuple):
            pred_config, config_model_path = config_result
            model_path = config_model_path or args.model  # Use config model or fallback to args
        else:
            pred_config = config_result
            model_path = args.model
        
        # Check if config references a training session for calibration
        if hasattr(pred_config, 'training_session') and pred_config.training_session:
            calibration_info = load_calibration_from_session(pred_config.training_session)
        
        feature_extractor = get_feature_extractor_from_config(pred_config)
        if not args.quiet:
            print(f"📋 Using prediction config: {args.prediction_config}")
    else:
        # Try to auto-detect from available sources
        if args.model:
            model_path = args.model
            pred_config = auto_detect_prediction_config(model_path=model_path)
        else:
            pred_config = auto_detect_prediction_config()
            
        if not pred_config:
            print("❌ No configuration found!")
            print("   Please provide one of:")
            print("   --session training_sessions/session_*/")
            print("   --model [path] --prediction-config configs/prediction/default.yaml")
            print("   --model [path] --train-config-file configs/training/aggressive_config.yaml")
            return
        
        feature_extractor = get_feature_extractor_from_config(pred_config)
        model_path = model_path or args.model
        if not args.quiet:
            print(f"🔧 Auto-detected configuration")
    
    # Validate we have everything we need
    if not model_path:
        print("❌ No model path specified!")
        print("   Use --model [path] or --session [session_dir]")
        return
        
    if not feature_extractor:
        print("❌ Failed to create feature extractor!")
        return
        
    if not pred_config:
        print("❌ Failed to load prediction configuration!")
        return
    
    # Select calibration method and parameters using the new system
    calibration_method, calibration_params = select_calibration_method(
        calibration_info=calibration_info,
        config_method=getattr(pred_config, 'calibration_method', 'auto'),
        config_temp=pred_config.temperature,
        config_platt_A=getattr(pred_config, 'platt_A', 1.0),
        config_platt_B=getattr(pred_config, 'platt_B', 0.0),
        cli_method=getattr(args, 'calibration_method', None),
        cli_temp=args.temperature,
        cli_platt_A=getattr(args, 'platt_A', None),
        cli_platt_B=getattr(args, 'platt_B', None),
        quiet=args.quiet
    )
    
    # Extract calibration parameters
    final_temperature = calibration_params.get('temperature', 1.0)
    final_platt_A = calibration_params.get('A', 1.0)
    final_platt_B = calibration_params.get('B', 0.0)

    # Override config with command line arguments
    
    if args.quiet:
        pred_config.quiet = True
    
    # Load model (now that we have model_path)
    model = load_model(model_path, device)
    
    # Get input lines
    lines = []
    input_file = None
    
    if args.lines:
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
        # Use specified file or default from config
        input_file = args.text or pred_config.input_file
        
        try:
            lines = load_lines_from_file(input_file)
            if not pred_config.quiet:
                print(f"📄 Loaded {len(lines)} lines from {input_file}")
        except FileNotFoundError:
            print(f"❌ Input file not found: {input_file}")
            print("   Put your lyrics in the specified location or use --text [filepath]")
            return
    
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
        calibration_method=calibration_method,
        temperature=final_temperature,
        platt_A=final_platt_A,
        platt_B=final_platt_B
    )
    
    # Print results
    if not pred_config.quiet:
        print_predictions(predictions)
    
    # Create organized output directory structure
    model_name = get_model_name(model_path)
    output_dir = f"{pred_config.output_dir}/{model_name}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save plain text output (for AI-less machines)
    plain_text_output = create_plain_text_output(predictions)
    plain_text_file = os.path.join(output_dir, 'predictions.txt')
    with open(plain_text_file, 'w') as f:
        f.write(plain_text_output)
    
    # Save detailed JSON output
    output_data = {
        'model_path': model_path,
        'model_name': model_name,
        'input_file': input_file,
        'temperature': pred_config.temperature,
        'total_lines': len(predictions),
        'predictions': predictions
    }
    
    json_file = os.path.join(output_dir, 'predictions.json')
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save parameters used
    params_file = save_parameters(output_dir, args, pred_config)
    
    if not pred_config.quiet:
        print(f"💾 Results saved to {output_dir}/:")
        print(f"   📄 predictions.txt (plain text)")
        print(f"   📄 predictions.json (detailed data)")  
        print(f"   📄 parameters.json (run parameters)")
    
    # Return output directory for external use
    return output_dir


if __name__ == "__main__":
    main()
