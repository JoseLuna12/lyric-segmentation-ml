#!/usr/bin/env python3
"""
Inference script for CNN model.
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
from segmodel.models import CNNTagger
from segmodel.utils import load_training_config
from segmodel.utils.prediction_config import (
    PredictionConfig, 
    load_prediction_config, 
    auto_detect_prediction_config,
    create_prediction_config_from_training_config,
    create_prediction_config_from_training_session,
    get_feature_extractor_from_config
)



def load_model(model_path: str, device: torch.device, training_config_path: str = None) -> CNNTagger:
    """Load trained CNN model from file."""
    print(f"📦 Loading CNN model from {model_path}...")
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)

    
    classifier_weight = state_dict['classifier.weight']
    hidden_dim = classifier_weight.shape[1]
    num_classes = classifier_weight.shape[0]
    
    # Infer input dimension from input projection layer
    if 'input_projection.weight' in state_dict:
        input_projection_weight = state_dict['input_projection.weight']
        input_size = input_projection_weight.shape[1]  # Input features dimension
    else:
        # Fallback: try to find first CNN block
        first_conv_key = None
        for key in state_dict.keys():
            if 'cnn_blocks.0.convs.0.weight' in key:
                first_conv_key = key
                break
        
        if first_conv_key:
            first_conv_weight = state_dict[first_conv_key]
            input_size = first_conv_weight.shape[1]  # in_channels
        else:
            # try to find any conv weight
            conv_keys = [k for k in state_dict.keys() if 'conv' in k and 'weight' in k]
            if conv_keys:
                first_conv_weight = state_dict[conv_keys[0]]
                input_size = first_conv_weight.shape[1]
            else:
                raise ValueError("Could not determine input size from CNN state dict")
    
    # Count CNN blocks
    cnn_block_keys = [key for key in state_dict.keys() if key.startswith('cnn_blocks.') and '.conv.weight' in key]
    num_cnn_blocks = len(set(key.split('.')[1] for key in cnn_block_keys))
    
    # Detect CNN-specific parameters
    kernel_sizes = [3, 5, 7] 
    dilation_rates = [1, 2, 4] 
    use_residual = True 
    use_attention = any(key.startswith('attention.') for key in state_dict.keys())
    attention_type = 'self' 
    attention_heads = 8 
    attention_dropout = 0.1 
    positional_encoding = False 
    attention_dim = None
    boundary_temperature = 2.0 
    
    config_loaded = False
    
    if training_config_path and os.path.exists(training_config_path):
        try:
            print(f"📋 Loading CNN config from training config: {training_config_path}")
            
            import yaml
            with open(training_config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            def get_config_value(key, config_dict):
                if key in config_dict:
                    return config_dict[key]
                if 'model' in config_dict and key in config_dict['model']:
                    return config_dict['model'][key]
                if 'cnn' in config_dict and key in config_dict['cnn']:
                    return config_dict['cnn'][key]
                return None
            
            val = get_config_value('hidden_dim', config_dict)
            if val is not None:
                hidden_dim = val
            val = get_config_value('num_layers', config_dict)
            if val is not None:
                num_cnn_blocks = val
            val = get_config_value('kernel_sizes', config_dict)
            if val is not None:
                kernel_sizes = val
            val = get_config_value('dilation_rates', config_dict)
            if val is not None:
                dilation_rates = val
            val = get_config_value('use_residual', config_dict)
            if val is not None:
                use_residual = val
            
            val = get_config_value('attention_enabled', config_dict)
            if val is not None:
                use_attention = val
            val = get_config_value('attention_type', config_dict)
            if val is not None:
                attention_type = val
            val = get_config_value('attention_heads', config_dict)
            if val is not None:
                attention_heads = val
            val = get_config_value('attention_dropout', config_dict)
            if val is not None:
                attention_dropout = val
            val = get_config_value('attention_dim', config_dict)
            if val is not None:
                attention_dim = val
            val = get_config_value('positional_encoding', config_dict)
            if val is not None:
                positional_encoding = val
            val = get_config_value('boundary_temperature', config_dict)
            if val is not None:
                boundary_temperature = val
                
            config_loaded = True
            print(f"🎯 Loaded CNN config from training config:")
            print(f"   Hidden dim: {hidden_dim}")
            print(f"   CNN blocks: {num_cnn_blocks}")
            print(f"   Kernel sizes: {kernel_sizes}")
            print(f"   Dilation rates: {dilation_rates}")
            print(f"   Use residual: {use_residual}")
            print(f"   Attention enabled: {use_attention}")
            if use_attention:
                print(f"   Attention type: {attention_type}")
                print(f"   Attention heads: {attention_heads}")
                print(f"   Attention dimension: {attention_dim}")
                print(f"   Attention dropout: {attention_dropout}")
                print(f"   Positional encoding: {positional_encoding}")
                if attention_type == 'boundary_aware':
                    print(f"   Boundary temperature: {boundary_temperature}")
            
        except Exception as e:
            print(f"⚠️  Could not load training config, falling back to state dict detection: {e}")
    
    if not config_loaded and attention_dim is None and use_attention:

        try:
            if 'attention.attention.w_q.weight' in state_dict:
                saved_attention_dim = state_dict['attention.attention.w_q.weight'].shape[0]
                attention_dim = saved_attention_dim
                
                print(f"   ⚠️  Could not read from config, using default attention_heads: {attention_heads}")
            
            positional_encoding = 'attention.attention.positional_encoding.pe' in state_dict
            
            print(f"🎯 Detected CNN attention mechanism from state dict:")
            print(f"   Attention enabled: {use_attention}")
            print(f"   Attention heads: {attention_heads} (default - could not read from config)")
            print(f"   Attention dimension: {attention_dim} (from state dict)")
            print(f"   Positional encoding: {positional_encoding}")
        except Exception as e:
            print(f"⚠️  Could not fully determine CNN attention config, using defaults: {e}")
            attention_dim = None
    
    print(f"🔧 Detected CNN model architecture:")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {hidden_dim}")
    print(f"   Classes: {num_classes}")
    print(f"   CNN blocks: {num_cnn_blocks}")
    if use_attention:
        print(f"   🎯 Attention: enabled ({attention_heads} heads)")
    else:
        print(f"   🎯 Attention: disabled")
    
    model = CNNTagger(
        feat_dim=input_size,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_cnn_blocks,
        dropout=0.0,
        layer_dropout=0.0,
        kernel_sizes=kernel_sizes,
        dilation_rates=dilation_rates,
        use_residual=use_residual,
        attention_enabled=use_attention,
        attention_type=attention_type,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        attention_dim=attention_dim,
        positional_encoding=positional_encoding,
        window_size=7,
        boundary_temperature=boundary_temperature
    )
    
    # Load weights
    try:
        model.load_state_dict(state_dict)
        print("✅ CNN model loaded successfully (including attention weights)")
    except Exception as e:
        print(f"❌ Error loading CNN model weights: {e}")
        print("🔧 This might be due to architecture mismatch or missing CNN parameters")
        raise
    
    model.to(device)
    model.eval()
    
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
    
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = Config(config_dict)
    
    print(f"🔧 Debug - syllable_pattern_ssm_enabled: {getattr(config, 'syllable_pattern_ssm_enabled', 'NOT_FOUND')}")
    print(f"🔧 Debug - line_syllable_ssm_enabled: {getattr(config, 'line_syllable_ssm_enabled', 'NOT_FOUND')}")
    
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
    
    # Syllable SSM features
    if config.syllable_pattern_ssm_enabled:
        print(f"🔧 Adding syllable_pattern_ssm feature (dimension: {config.syllable_pattern_ssm_dimension})")
        feature_config['syllable_pattern_ssm'] = {
            'enabled': True,
            'dimension': config.syllable_pattern_ssm_dimension,
            'similarity_method': config.syllable_pattern_ssm_similarity_method,
            'levenshtein_weight': config.syllable_pattern_ssm_levenshtein_weight,
            'cosine_weight': config.syllable_pattern_ssm_cosine_weight,
            'normalize': config.syllable_pattern_ssm_normalize,
            'normalize_method': config.syllable_pattern_ssm_normalize_method
        }
    
    if config.line_syllable_ssm_enabled:
        print(f"🔧 Adding line_syllable_ssm feature (dimension: {config.line_syllable_ssm_dimension})")
        feature_config['line_syllable_ssm'] = {
            'enabled': True,
            'dimension': config.line_syllable_ssm_dimension,
            'similarity_method': config.line_syllable_ssm_similarity_method,
            'ratio_threshold': config.line_syllable_ssm_ratio_threshold,
            'normalize': config.line_syllable_ssm_normalize,
            'normalize_method': config.line_syllable_ssm_normalize_method
        }
    
    # Embeddings - converted to flattened format FeatureExtractor expects
    if config.word2vec_enabled:
        feature_config['word2vec_enabled'] = True
        feature_config['word2vec_model'] = config.word2vec_model
        feature_config['word2vec_mode'] = config.word2vec_mode
        feature_config['word2vec_normalize'] = config.word2vec_normalize
        feature_config['word2vec_similarity_metric'] = config.word2vec_similarity_metric
        feature_config['word2vec_high_sim_threshold'] = config.word2vec_high_sim_threshold
    
    if config.contextual_enabled:
        feature_config['contextual_enabled'] = True
        feature_config['contextual_model'] = config.contextual_model
        feature_config['contextual_mode'] = config.contextual_mode
        feature_config['contextual_normalize'] = config.contextual_normalize
        feature_config['contextual_similarity_metric'] = config.contextual_similarity_metric
        feature_config['contextual_high_sim_threshold'] = config.contextual_high_sim_threshold
    
    if not feature_config:
        raise ValueError(f"No features enabled in training config: {config_path}")
    
    extractor = FeatureExtractor(feature_config)
    print(f"🧩 Initialized feature extractor from training config:")
    enabled_features = [name for name, config in feature_config.items() if isinstance(config, dict) and config.get('enabled', False)]
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
    config_isotonic_knots=0,
    cli_method=None,
    cli_temp=None,
    cli_platt_A=None,
    cli_platt_B=None,
    cli_isotonic_knots=None,
    quiet=False
):
    """
    Select calibration method and parameters based on priority:
    1. CLI overrides (highest priority)
    2. Config settings (medium priority) 
    3. Auto-selection from calibration_info (lowest priority, but recommended)
    
    Returns:
        method: Selected calibration method ('temperature', 'platt', 'isotonic', 'none')
        params: Dictionary with method-specific parameters
    """
    
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
        elif method == 'isotonic':
            knots = cli_isotonic_knots if cli_isotonic_knots is not None else config_isotonic_knots
            params = {'knots': knots}
            if not quiet:
                print(f"🔧 Using CLI calibration method: {method} (knots: {knots})")
        elif method == 'none':
            params = {}
            if not quiet:
                print(f"🔧 Using CLI calibration method: none (no calibration)")
        else:
            method = 'auto'
            params = {}
    else:
        method = config_method
        params = {}
    
    if method == 'auto':
        if calibration_info and 'method' in calibration_info:
            auto_method = calibration_info['method']
            auto_params = calibration_info['params']
            
            if not quiet:
                ece_after = calibration_info.get('ece_after', 'N/A')
                print(f"📊 Auto-selected calibration method: {auto_method} (ECE: {ece_after:.4f})")
                if auto_method == 'temperature':
                    print(f"🎯 Using calibrated temperature: {auto_params['temperature']:.3f}")
                elif auto_method == 'platt':
                    print(f"🎯 Using calibrated Platt scaling: A={auto_params['A']:.3f}, B={auto_params['B']:.3f}")
                elif auto_method == 'isotonic':
                    print(f"🎯 Using isotonic calibration: knots={auto_params.get('knots', 0)}")
            
            return auto_method, auto_params
        
        if not quiet:
            print(f"⚠️  'auto' method requires calibration.json but none found")
            print(f"    Falling back to 'none' (no calibration applied)")
        method = 'none'
        params = {}
    
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
    
    elif method == 'isotonic':
        knots = cli_isotonic_knots if cli_isotonic_knots is not None else config_isotonic_knots
        params = {'knots': knots}
        if not quiet and cli_isotonic_knots is None:
            print(f"🎯 Using config isotonic calibration: knots={knots}")
    
    elif method == 'none':
        params = {}
        if not quiet:
            print(f"🎯 No calibration applied")
    
    return method, params


def predict_lyrics_structure(
    lines: List[str],
    model: CNNTagger,
    feature_extractor,
    device: torch.device,
    calibration_method: str = "none",
    temperature: float = 1.0,
    platt_A: float = 1.0,
    platt_B: float = 0.0,
    isotonic_knots: int = 0
) -> List[Dict[str, Any]]:
    """
    Predict structure labels for a list of lyric lines using CNN model.
    
    Args:
        lines: List of lyric lines
        model: Trained CNN model
        feature_extractor: Feature extraction function
        device: Device to run on
        calibration_method: Calibration method ('temperature', 'platt', 'isotonic', 'none')
        temperature: Temperature for temperature scaling
        platt_A: Platt scaling A coefficient
        platt_B: Platt scaling B coefficient
        isotonic_knots: Number of knots for isotonic calibration (informational)
        
    Returns:
        List of prediction dictionaries
    """
    if not lines:
        return []
    
    features = feature_extractor(lines)
    features = features.unsqueeze(0).to(device)
    
    mask = torch.ones(1, len(lines), dtype=torch.bool, device=device)
    
    with torch.no_grad():
        if calibration_method == 'temperature':
            predictions, confidences = model.predict_with_temperature(
                features, mask, temperature=temperature
            )
        elif calibration_method == 'platt':
            logits = model(features, mask)
            probs = torch.softmax(logits, dim=-1)
            max_probs, predictions = torch.max(probs, dim=-1)
            
            calibrated_confidences = torch.sigmoid(platt_A * max_probs + platt_B)
            
            predictions = predictions.squeeze(0) 
            confidences = calibrated_confidences.squeeze(0)
        elif calibration_method == 'isotonic':
            print("⚠️ Isotonic calibration requires fitted model (not available in inference)")
            print("   Falling back to uncalibrated predictions")
            predictions, confidences = model.predict_with_temperature(
                features, mask, temperature=1.0
            )
        else:
            # No calibration, use temperature=1.0
            predictions, confidences = model.predict_with_temperature(
                features, mask, temperature=1.0
            )
    
    predictions = predictions.squeeze(0).cpu().numpy()
    confidences = confidences.squeeze(0).cpu().numpy()
    
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
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line:
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
    print(f"\n🎵 CNN Lyrics Structure Prediction")
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
    lines.append("CNN Lyrics Structure Prediction")
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
    session_dir = os.path.basename(os.path.dirname(model_path))
    model_name = f"cnn-{session_dir.replace('session_', '').replace('_', '-')}"
    return model_name


def save_parameters(output_dir: str, args, pred_config: PredictionConfig):
    """Save the parameters used for this prediction run."""
    import json
    import datetime
    
    params = {
        'timestamp': datetime.datetime.now().isoformat(),
        'model_type': 'CNN',
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
    
    params_file = os.path.join(output_dir, 'cnn_parameters.json')
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    
    return params_file


def main():
    parser = argparse.ArgumentParser(
        description="Predict verse/chorus structure for lyrics using CNN model",
        epilog="""
Examples:
  # From text file (with training session config for feature compatibility)
  python predict_cnn.py --model training_sessions/session_*/best_cnn_model.pt --config training_sessions/session_*/ --text song.txt
  
  # Multi-line string input with specific config
  python predict_cnn.py --model best_cnn_model.pt --config configs/cnn_config.yaml --lyrics "Walking down the street
  Thinking of you
  Dancing tonight"
  
  # Command line arguments (uses default comprehensive features if no config)
  python predict_cnn.py --model best_cnn_model.pt --lines "Line 1" "Line 2" "Line 3"
  
  # Interactive input with training session config
  python predict_cnn.py --model training_sessions/session_*/best_cnn_model.pt --config training_sessions/session_*/
  
  # Piped input
  echo "Line 1\nLine 2" | python predict_cnn.py --model best_cnn_model.pt --stdin
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model arguments
    parser.add_argument('--model', help='Path to trained CNN model (or use --session)')
    parser.add_argument('--session', help='Path to training session directory (contains model + config)')
    parser.add_argument('--prediction-config', help='Path to custom prediction config file')
    parser.add_argument('--train-config-file', help='Path to training config file (legacy)')
    
    # Calibration arguments
    parser.add_argument('--calibration-method', choices=['auto', 'temperature', 'platt', 'isotonic', 'none'], 
                       help='Calibration method (overrides config)')
    parser.add_argument('--temperature', type=float, help='Temperature for temperature scaling (overrides config)')
    parser.add_argument('--platt-A', type=float, help='Platt scaling A coefficient (overrides config)')
    parser.add_argument('--platt-B', type=float, help='Platt scaling B coefficient (overrides config)')
    parser.add_argument('--isotonic-knots', type=int, help='Isotonic calibration knots (informational, overrides config)')
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--text', help='Text file with lyrics (default: predict_lyric.txt)')
    input_group.add_argument('--lines', nargs='+', help='Lyrics lines as command line arguments')
    input_group.add_argument('--lyrics', help='Multi-line lyrics as a single string')
    input_group.add_argument('--stdin', action='store_true', help='Read lyrics from stdin (interactive or piped)')
    
    # Output arguments
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (no terminal output)')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🧠 CNN Model Inference")
        print("=" * 30)
    
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    
    if not args.quiet:
        print(f"🔧 Using device: {device}")
    
    feature_extractor = None
    pred_config = None
    
    model_path = None
    calibration_info = None
    
    if args.session:
        session_path = Path(args.session)
        
        cnn_model_candidates = [
            session_path / "best_cnn_model.pt",
            session_path / "final_cnn_model.pt",
            session_path / "best_model.pt",
            session_path / "final_model.pt"
        ]
        
        model_path = None
        for candidate in cnn_model_candidates:
            if candidate.exists():
                model_path = str(candidate)
                break
        
        if not model_path:
            print(f"❌ No CNN model found in session directory: {args.session}")
            print(f"   Looked for: {[c.name for c in cnn_model_candidates]}")
            return
        
        try:
            pred_config, _ = create_prediction_config_from_training_session(args.session)
            feature_extractor = get_feature_extractor_from_config(pred_config)
        except (FileNotFoundError, AttributeError):
            class BasicPredictionConfig:
                def __init__(self):
                    self.input_file = "predict_lyric.txt"
                    self.output_dir = "prediction_results"
                    self.temperature = 1.0
                    self.quiet = False
                    self.device = str(device)
                    self.features = {}
            
            pred_config = BasicPredictionConfig()
            feature_extractor = None  
        
        calibration_info = load_calibration_from_session(args.session)
        
        if not args.quiet:
            print(f"🎯 Using training session as complete source: {args.session}")
            print(f"🎯 Found CNN model: {os.path.basename(model_path)}")
    elif args.train_config_file:
        pred_config = create_prediction_config_from_training_config(args.train_config_file)
        feature_extractor = get_feature_extractor_from_config(pred_config)
        model_path = args.model
        if not args.quiet:
            print(f"🔧 Using training config as source of truth: {args.train_config_file}")
    elif args.prediction_config:
        config_result = load_prediction_config(args.prediction_config)
        if isinstance(config_result, tuple):
            pred_config, config_model_path = config_result
            model_path = config_model_path or args.model 
        else:
            pred_config = config_result
            model_path = args.model
        
        if hasattr(pred_config, 'training_session') and pred_config.training_session:
            calibration_info = load_calibration_from_session(pred_config.training_session)
        
        feature_extractor = get_feature_extractor_from_config(pred_config)
        if not args.quiet:
            print(f"📋 Using prediction config: {args.prediction_config}")
    else:
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
            print("   --model [path] --train-config-file configs/cnn_config.yaml")
            return
        
        feature_extractor = get_feature_extractor_from_config(pred_config)
        model_path = model_path or args.model
        if not args.quiet:
            print(f"🔧 Auto-detected configuration")
    
    if not model_path:
        print("❌ No CNN model path specified!")
        print("   Use --model [path] or --session [session_dir]")
        return

    training_config_path = None
    
    if args.session:
        # Session directory should contain training_config_snapshot.yaml
        training_config_path = os.path.join(args.session, 'training_config_snapshot.yaml')
        if not os.path.exists(training_config_path):
            training_config_path = None
    elif args.train_config_file:
        # Direct training config file path
        training_config_path = args.train_config_file
    elif hasattr(pred_config, 'training_session') and pred_config.training_session:
        # Training session referenced in prediction config
        training_config_path = os.path.join(pred_config.training_session, 'training_config_snapshot.yaml')
        if not os.path.exists(training_config_path):
            training_config_path = None

    if training_config_path and os.path.exists(training_config_path):
        if not args.quiet:
            print(f"🔧 Using feature configuration from training session")
        feature_extractor = create_feature_extractor_from_training_config(training_config_path)

    model = load_model(model_path, device, training_config_path)
        
    if not feature_extractor:
        print("❌ Failed to create feature extractor!")
        return
        
    if not pred_config:
        print("❌ Failed to load prediction configuration!")
        return
    
    calibration_method, calibration_params = select_calibration_method(
        calibration_info=calibration_info,
        config_method=getattr(pred_config, 'calibration_method', 'auto'),
        config_temp=pred_config.temperature,
        config_platt_A=getattr(pred_config, 'platt_A', 1.0),
        config_platt_B=getattr(pred_config, 'platt_B', 0.0),
        config_isotonic_knots=getattr(pred_config, 'isotonic_knots', 0),
        cli_method=getattr(args, 'calibration_method', None),
        cli_temp=args.temperature,
        cli_platt_A=getattr(args, 'platt_A', None),
        cli_platt_B=getattr(args, 'platt_B', None),
        cli_isotonic_knots=getattr(args, 'isotonic_knots', None),
        quiet=args.quiet
    )
    
    final_temperature = calibration_params.get('temperature', 1.0)
    final_platt_A = calibration_params.get('A', 1.0)
    final_platt_B = calibration_params.get('B', 0.0)
    final_isotonic_knots = calibration_params.get('knots', 0)

    
    if args.quiet:
        pred_config.quiet = True
    
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
        print(f"🔮 Making CNN predictions...")
    
    predictions = predict_lyrics_structure(
        lines=lines,
        model=model,
        feature_extractor=feature_extractor,
        device=device,
        calibration_method=calibration_method,
        temperature=final_temperature,
        platt_A=final_platt_A,
        platt_B=final_platt_B,
        isotonic_knots=final_isotonic_knots
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
    plain_text_file = os.path.join(output_dir, 'cnn_predictions.txt')
    with open(plain_text_file, 'w') as f:
        f.write(plain_text_output)
    
    # Save detailed JSON output
    output_data = {
        'model_type': 'CNN',
        'model_path': model_path,
        'model_name': model_name,
        'input_file': input_file,
        'temperature': pred_config.temperature,
        'total_lines': len(predictions),
        'predictions': predictions
    }
    
    json_file = os.path.join(output_dir, 'cnn_predictions.json')
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save parameters used
    params_file = save_parameters(output_dir, args, pred_config)
    
    if not pred_config.quiet:
        print(f"💾 CNN Results saved to {output_dir}/:")
        print(f"   📄 cnn_predictions.txt (plain text)")
        print(f"   📄 cnn_predictions.json (detailed data)")  
        print(f"   📄 cnn_parameters.json (run parameters)")
    
    # Return output directory for external use
    return output_dir


if __name__ == "__main__":
    main()
