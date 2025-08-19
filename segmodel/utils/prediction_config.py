#!/usr/bin/env python3
"""
Prediction configuration loader and dataclass.
Lightweight configuration system focused only on prediction parameters.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class PredictionConfig:
    """Configuration for prediction/inference operations."""
    
    # Model parameters
    model_path: str = ""
    
    # Training session reference (for calibration loading)
    training_session: str = ""
    
    # Calibration configuration
    calibration_method: str = "auto"  # auto, temperature, platt, isotonic, none
    temperature: float = 1.0          # Temperature scaling parameter
    platt_A: float = 1.0             # Platt scaling A coefficient  
    platt_B: float = 0.0             # Platt scaling B coefficient
    isotonic_knots: int = 0          # Isotonic calibration knots (informational)
    
    # Input/Output parameters  
    input_file: str = "prediction_results/predict_lyric.txt"  # Single default location
    output_dir: str = "prediction_results"
    quiet: bool = False
    
    # Feature configuration (lightweight)
    features: Dict[str, Any] = field(default_factory=lambda: {
        'head_ssm': {'dim': 12, 'enabled': True},
        'tail_ssm': {'dim': 12, 'enabled': True}, 
        'phonetic_ssm': {'dim': 12, 'enabled': True, 'mode': 'rhyme', 'binary': True},
        'pos_ssm': {'dim': 12, 'enabled': True, 'mode': 'simplified', 'combined': True, 'threshold': 0.3},
        'string_ssm': {'dim': 12, 'enabled': True, 'threshold': 0.1},
        # NEW: Embedding features
        'word2vec_embeddings': {'enabled': False, 'model': 'word2vec-google-news-300', 'mode': 'summary', 'normalize': True, 'similarity_metric': 'cosine', 'high_sim_threshold': 0.8},
        'contextual_embeddings': {'enabled': False, 'model': 'all-MiniLM-L6-v2', 'mode': 'summary', 'normalize': True, 'similarity_metric': 'cosine', 'high_sim_threshold': 0.7}
    })
    
    # Device configuration
    device: str = "auto"  # auto, cpu, cuda, mps
    
    def __post_init__(self):
        """Validate and setup configuration after initialization."""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps" 
            else:
                self.device = "cpu"


def load_prediction_config(config_path: str) -> tuple[PredictionConfig, str]:
    """
    Load prediction configuration from YAML file.
    If training_session is specified, extract features and model from that session.
    
    Returns:
        Tuple of (PredictionConfig, model_path)
    """
    """
    Load prediction configuration from YAML file.
    
    Args:
        config_path: Path to prediction config YAML file
        
    Returns:
        PredictionConfig object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Prediction config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Flatten nested structure if present
    flattened = {}
    
    # Extract prediction-specific sections
    if 'prediction' in config_dict:
        pred_section = config_dict['prediction']
        
        # Handle calibration configuration
        if 'calibration' in pred_section:
            calib_config = pred_section['calibration']
            if 'method' in calib_config:
                flattened['calibration_method'] = calib_config['method']
            if 'temperature' in calib_config:
                flattened['temperature'] = calib_config['temperature']
            if 'platt_A' in calib_config:
                flattened['platt_A'] = calib_config['platt_A']
            if 'platt_B' in calib_config:
                flattened['platt_B'] = calib_config['platt_B']
            if 'isotonic_knots' in calib_config:
                flattened['isotonic_knots'] = calib_config['isotonic_knots']
        
        # Handle other prediction parameters
        for key in ['input_file', 'output_dir', 'quiet']:
            if key in pred_section:
                flattened[key] = pred_section[key]
        
        # Handle legacy temperature field (backward compatibility)
        if 'temperature' in pred_section and 'temperature' not in flattened:
            flattened['temperature'] = pred_section['temperature']
    
    if 'model' in config_dict:
        flattened.update({f"model_{k}": v for k, v in config_dict['model'].items()})
        
    if 'features' in config_dict:
        flattened['features'] = config_dict['features']
        
    if 'device' in config_dict:
        flattened['device'] = config_dict['device']
    
    # Check if training_session is specified (preferred approach)
    if 'training_session' in config_dict:
        training_session_dir = config_dict['training_session']
        if os.path.exists(training_session_dir):
            # Load from training session (features + model)
            session_config, model_path = create_prediction_config_from_training_session(training_session_dir)
            
            # Override with prediction-specific settings from config file
            prediction_params = config_dict.get('prediction', {})
            
            # Handle calibration configuration
            if 'calibration' in prediction_params:
                calib_config = prediction_params['calibration']
                if calib_config.get('method') is not None:
                    session_config.calibration_method = calib_config['method']
                if calib_config.get('temperature') is not None:
                    session_config.temperature = calib_config['temperature']
                if calib_config.get('platt_A') is not None:
                    session_config.platt_A = calib_config['platt_A']
                if calib_config.get('platt_B') is not None:
                    session_config.platt_B = calib_config['platt_B']
                if calib_config.get('isotonic_knots') is not None:
                    session_config.isotonic_knots = calib_config['isotonic_knots']
            
            # Handle legacy temperature field (backward compatibility)
            if prediction_params.get('temperature') is not None:
                session_config.temperature = prediction_params['temperature']
                
            if prediction_params.get('input_file') is not None:
                session_config.input_file = prediction_params['input_file']
            if prediction_params.get('output_dir') is not None:
                session_config.output_dir = prediction_params['output_dir']
            if prediction_params.get('quiet') is not None:
                session_config.quiet = prediction_params['quiet']
            
            # Override device if specified
            if 'device' in config_dict:
                session_config.device = config_dict['device']
            
            return session_config, model_path
        else:
            raise FileNotFoundError(f"Training session directory not found: {training_session_dir}")
    
    # Handle direct top-level keys (for simple configs - backward compatibility)
    for key in ['temperature', 'input_file', 'output_dir', 'quiet', 'calibration_method', 'platt_A', 'platt_B', 'isotonic_knots']:
        if key in config_dict:
            flattened[key] = config_dict[key]
    
    # Create config object with available parameters
    config_params = {}
    for field_name in PredictionConfig.__dataclass_fields__.keys():
        if field_name in flattened:
            config_params[field_name] = flattened[field_name]
    
    return PredictionConfig(**config_params), None  # No model path for manual configs


def create_default_prediction_config() -> PredictionConfig:
    """Create a default prediction configuration."""
    return PredictionConfig()


def save_prediction_config(config: PredictionConfig, output_path: str):
    """
    Save prediction configuration to YAML file.
    
    Args:
        config: PredictionConfig object to save
        output_path: Path where to save the config
    """
    config_dict = {
        'prediction': {
            'temperature': config.temperature,
            'input_file': config.input_file,
            'input_search_paths': config.input_search_paths,
            'output_dir': config.output_dir,
            'quiet': config.quiet
        },
        'features': config.features,
        'device': config.device
    }
    
    # Add model_path if specified
    if config.model_path:
        config_dict['model'] = {'path': config.model_path}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def create_prediction_config_from_training_session(session_dir: str) -> tuple[PredictionConfig, str]:
    """
    Create prediction configuration from a training session directory.
    This is the preferred method - training session contains everything needed.
    
    Args:
        session_dir: Path to training session directory
        
    Returns:
        Tuple of (PredictionConfig, model_path)
    """
    if not os.path.exists(session_dir):
        raise FileNotFoundError(f"Training session directory not found: {session_dir}")
    
    # Look for training config snapshot in session directory
    config_snapshot_path = os.path.join(session_dir, 'training_config_snapshot.yaml')
    if not os.path.exists(config_snapshot_path):
        raise FileNotFoundError(f"No training config snapshot found in {session_dir}")
    
    # Look for trained model in session directory
    model_path = os.path.join(session_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found in {session_dir}")
    
    # Load training configuration from snapshot (this has the exact parameters used)
    import yaml
    with open(config_snapshot_path, 'r') as f:
        train_config_dict = yaml.safe_load(f)
    
    # Extract feature configuration from training config snapshot
    features = {}
    
    if train_config_dict.get('head_ssm_enabled', False):
        features['head_ssm'] = {
            'dim': train_config_dict.get('head_ssm_dimension', 12),
            'enabled': True,
            'head_words': train_config_dict.get('head_ssm_words', 2)
        }
        
    if train_config_dict.get('tail_ssm_enabled', False):
        features['tail_ssm'] = {
            'dim': train_config_dict.get('tail_ssm_dimension', 12),
            'enabled': True,
            'tail_words': train_config_dict.get('tail_ssm_words', 2)
        }
        
    if train_config_dict.get('phonetic_ssm_enabled', False):
        features['phonetic_ssm'] = {
            'dim': train_config_dict.get('phonetic_ssm_dimension', 12),
            'enabled': True,
            'mode': train_config_dict.get('phonetic_ssm_mode', 'rhyme'),
            'binary': train_config_dict.get('phonetic_ssm_similarity_method', 'binary') == 'binary',
            'threshold': train_config_dict.get('phonetic_ssm_high_sim_threshold', 0.8)
        }
        
    if train_config_dict.get('pos_ssm_enabled', False):
        features['pos_ssm'] = {
            'dim': train_config_dict.get('pos_ssm_dimension', 12),
            'enabled': True,
            'mode': train_config_dict.get('pos_ssm_tagset', 'simplified'),
            'combined': train_config_dict.get('pos_ssm_similarity_method', 'combined') == 'combined',
            'threshold': train_config_dict.get('pos_ssm_high_sim_threshold', 0.7)
        }
        
    if train_config_dict.get('string_ssm_enabled', False):
        features['string_ssm'] = {
            'dim': train_config_dict.get('string_ssm_dimension', 12),
            'enabled': True,
            'threshold': train_config_dict.get('string_ssm_similarity_threshold', 0.0)
        }
    
    # NEW: Extract embedding feature configurations from training session
    if train_config_dict.get('word2vec_enabled', False):
        features['word2vec_embeddings'] = {
            'enabled': True,
            'model': train_config_dict.get('word2vec_model', 'word2vec-google-news-300'),
            'mode': train_config_dict.get('word2vec_mode', 'summary'),
            'normalize': train_config_dict.get('word2vec_normalize', True),
            'similarity_metric': train_config_dict.get('word2vec_similarity_metric', 'cosine'),
            'high_sim_threshold': train_config_dict.get('word2vec_high_sim_threshold', 0.8)
        }
        
    if train_config_dict.get('contextual_enabled', False):
        features['contextual_embeddings'] = {
            'enabled': True,
            'model': train_config_dict.get('contextual_model', 'all-MiniLM-L6-v2'),
            'mode': train_config_dict.get('contextual_mode', 'summary'),
            'normalize': train_config_dict.get('contextual_normalize', True),
            'similarity_metric': train_config_dict.get('contextual_similarity_metric', 'cosine'),
            'high_sim_threshold': train_config_dict.get('contextual_high_sim_threshold', 0.7)
        }
    
    # Create prediction config with extracted features
    pred_config = PredictionConfig(features=features)
    pred_config.model_path = model_path
    pred_config.training_session = session_dir  # Set the training session reference
    
    return pred_config, model_path


def create_prediction_config_from_training_config(training_config_path: str) -> PredictionConfig:
    """
    Create prediction configuration by extracting feature settings from training config.
    This is the preferred method - training config is the single source of truth.
    
    Args:
        training_config_path: Path to training config YAML file
        
    Returns:
        PredictionConfig object with features extracted from training config
    """
    from .config_loader import load_training_config
    
    if not os.path.exists(training_config_path):
        raise FileNotFoundError(f"Training config not found: {training_config_path}")
    
    # Load training configuration
    train_config = load_training_config(training_config_path)
    
    # Extract feature configuration from training config
    features = {}
    
    if train_config.head_ssm_enabled:
        features['head_ssm'] = {
            'dim': train_config.head_ssm_dimension,
            'enabled': True,
            'head_words': train_config.head_ssm_words
        }
        
    if train_config.tail_ssm_enabled:
        features['tail_ssm'] = {
            'dim': train_config.tail_ssm_dimension,
            'enabled': True,
            'tail_words': train_config.tail_ssm_words
        }
        
    if train_config.phonetic_ssm_enabled:
        features['phonetic_ssm'] = {
            'dim': train_config.phonetic_ssm_dimension,
            'enabled': True,
            'mode': train_config.phonetic_ssm_mode,
            'binary': train_config.phonetic_ssm_similarity_method == 'binary',
            'threshold': train_config.phonetic_ssm_high_sim_threshold
        }
        
    if train_config.pos_ssm_enabled:
        features['pos_ssm'] = {
            'dim': train_config.pos_ssm_dimension,
            'enabled': True,
            'mode': train_config.pos_ssm_tagset,
            'combined': train_config.pos_ssm_similarity_method == 'combined',
            'threshold': train_config.pos_ssm_high_sim_threshold
        }
        
    if train_config.string_ssm_enabled:
        features['string_ssm'] = {
            'dim': train_config.string_ssm_dimension,
            'enabled': True,
            'threshold': train_config.string_ssm_similarity_threshold
        }
    
    # NEW: Extract embedding feature configurations
    if train_config.word2vec_enabled:
        features['word2vec_embeddings'] = {
            'enabled': True,
            'model': train_config.word2vec_model,
            'mode': train_config.word2vec_mode,
            'normalize': train_config.word2vec_normalize,
            'similarity_metric': train_config.word2vec_similarity_metric,
            'high_sim_threshold': train_config.word2vec_high_sim_threshold
        }
        
    if train_config.contextual_enabled:
        features['contextual_embeddings'] = {
            'enabled': True,
            'model': train_config.contextual_model,
            'mode': train_config.contextual_mode,
            'normalize': train_config.contextual_normalize,
            'similarity_metric': train_config.contextual_similarity_metric,
            'high_sim_threshold': train_config.contextual_high_sim_threshold
        }
    
    # Create prediction config with extracted features
    pred_config = PredictionConfig(features=features)
    return pred_config


def auto_detect_prediction_config(model_path: str = None) -> PredictionConfig:
    """
    Auto-detect prediction configuration from available sources.
    Priority: Training sessions > Training configs > Prediction configs
    """
    
    # PRIORITY 1: Look for default/best training session first
    default_session = "training_sessions/session_20250817_024332_Aggressive_Maximum_Performance_v1"
    if os.path.exists(default_session):
        try:
            pred_config, _ = create_prediction_config_from_training_session(default_session)
            return pred_config
        except:
            pass
    
    # PRIORITY 2: Try to create from training configs
    training_configs = [
        "configs/training/aggressive_config.yaml",
        "configs/training/training_config.yaml"
    ]
    
    for config_path in training_configs:
        if os.path.exists(config_path):
            try:
                pred_config = create_prediction_config_from_training_config(config_path)
                return pred_config
            except:
                continue
    
    # PRIORITY 3: Try explicit prediction configs
    prediction_configs = [
        "configs/prediction/default.yaml",
        "configs/prediction/production.yaml"
    ]
    
    for config_path in prediction_configs:
        if os.path.exists(config_path):
            try:
                config_result = load_prediction_config(config_path)
                if isinstance(config_result, tuple):
                    pred_config, _ = config_result
                else:
                    pred_config = config_result
                return pred_config
            except:
                continue
    
    return None


def get_feature_extractor_from_config(config: PredictionConfig):
    """
    Create feature extractor from prediction configuration.
    
    Args:
        config: PredictionConfig object
        
    Returns:
        FeatureExtractor object
    """
    from segmodel.features.extractor import FeatureExtractor
    
    # Convert prediction config features to extractor format
    # The FeatureExtractor expects a flattened config structure
    feature_config = {}
    
    for feature_name, feature_data in config.features.items():
        if not feature_data.get('enabled', False):
            continue
            
        if feature_name == 'word2vec_embeddings':
            # Convert to flattened structure expected by FeatureExtractor
            feature_config['word2vec_enabled'] = True
            feature_config['word2vec_model'] = feature_data.get('model', 'word2vec-google-news-300')
            feature_config['word2vec_mode'] = feature_data.get('mode', 'summary')
            feature_config['word2vec_normalize'] = feature_data.get('normalize', True)
            feature_config['word2vec_similarity_metric'] = feature_data.get('similarity_metric', 'cosine')
            feature_config['word2vec_high_sim_threshold'] = feature_data.get('high_sim_threshold', 0.8)
            
        elif feature_name == 'contextual_embeddings':
            # Convert to flattened structure expected by FeatureExtractor
            feature_config['contextual_enabled'] = True
            feature_config['contextual_model'] = feature_data.get('model', 'all-MiniLM-L6-v2')
            feature_config['contextual_mode'] = feature_data.get('mode', 'summary')
            feature_config['contextual_normalize'] = feature_data.get('normalize', True)
            feature_config['contextual_similarity_metric'] = feature_data.get('similarity_metric', 'cosine')
            feature_config['contextual_high_sim_threshold'] = feature_data.get('high_sim_threshold', 0.7)
            
        else:
            # Handle traditional features (they already have the right structure)
            feature_config[feature_name] = feature_data
    
    return FeatureExtractor(feature_config=feature_config)


if __name__ == "__main__":
    # Test the configuration system
    print("🧪 Testing prediction config system...")
    
    # Create default config
    config = create_default_prediction_config()
    print(f"✅ Default config created: {config.temperature}")
    
    # Test auto-detection
    detected_config = auto_detect_prediction_config()
    print(f"✅ Auto-detection works: {detected_config.device}")
    
    print("🎉 Prediction config system ready!")
