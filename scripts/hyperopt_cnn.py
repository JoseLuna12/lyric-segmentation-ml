#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization for CNN Text Segmentation.

This script optimizes the complex parameter space of CNN models for verse/chorus segmentation,
including CNN-specific architecture parameters, attention mechanisms, and boundary-aware loss function.
Adapted for CNN models with convolutional layers, dilated convolutions, and residual connections.
"""

import optuna
import json
import yaml
import torch
import argparse
import traceback
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from train_cnn_with_config import setup_data_loaders, setup_cnn_model_and_training
from segmodel.train.cnn_trainer import CNNTrainer
from segmodel.utils import TrainingConfig
from segmodel.features import FeatureExtractor


# =============================================================================
# CNN-SPECIFIC HYPERPARAMETER SEARCH SPACE CONFIGURATION
# =============================================================================

class CNNHyperparameterSpace:
    """
    Centralized configuration for CNN hyperparameter search space.
    Optimized for CNN architecture with convolutional layers and attention mechanisms.
    """
    
    def __init__(self, config_path=None):
        """Initialize with CNN-optimized comprehensive configuration like BLSTM."""
        print("🧠 Using CNN-COMPREHENSIVE search space configuration (like BLSTM)")
        self._load_cnn_defaults()
    
    def _load_cnn_defaults(self):
        """Load CNN-optimized parameter ranges for comprehensive optimization - LIKE BLSTM VERSION."""
        # ================================================================
        # CNN ARCHITECTURE PARAMETERS - Reduced for speed
        # ================================================================
        self.HIDDEN_DIM = [128]                  # Removed 256 for memory safety
        self.NUM_LAYERS = [1, 2]                 # Removed 3 layers for memory safety
        self.DROPOUT_RANGE = (0.25, 0.40)        # Unchanged
        self.LAYER_DROPOUT_RANGE = (0.10, 0.25)  # Unchanged
        
        # CNN-specific architecture parameters - more aggressive (convert to strings for Optuna)
        # Note: Optuna requires categorical options to be primitive types (str, int, float, bool, None)
        # Lists cause warnings, so we store as comma-separated strings and convert back to lists
        self.CNN_KERNEL_SIZES_OPTIONS = [
            "3,5",            # Simple dual-scale (MEMORY SAFE)
            "3,5,7",          # Standard multi-scale (MODERATE)
        ]
        self.CNN_DILATION_RATES_OPTIONS = [
            "1,2",            # Simple progression (MEMORY SAFE)
            "1,2,4",          # Standard exponential dilation (MODERATE)
        ]
        self.CNN_USE_RESIDUAL_OPTIONS = [True]    # Always use residual for stability
        self.CNN_BATCH_MULTIPLIER_RANGE = (1.0, 1.5)  # Conservative range
        
        # ================================================================
        # ATTENTION MECHANISMS - CONSERVATIVE for memory safety
        # ================================================================
        self.ATTENTION_ENABLED_OPTIONS = [True, False]  # Test both attention and no-attention
        self.ATTENTION_TYPE_OPTIONS = ['self', 'localized', 'boundary_aware']  # All three attention types
        self.ATTENTION_HEADS = [4, 8]             # Safe options: 128÷4=32, 256÷8=32
        self.ATTENTION_DIM = [128, 256]           # Will be matched to hidden_dim in sampling
        self.ATTENTION_DROPOUT_RANGE = (0.10, 0.30)  # Around proven 0.20
        self.WINDOW_SIZE_OPTIONS = [5, 7, 9]      # Window size for localized attention
        
        # ================================================================
        # CNN-OPTIMIZED TRAINING PARAMETERS - CONSERVATIVE for memory safety
        # ================================================================
        self.MAX_EPOCHS_OPTIONS = [8, 12, 15]      # Speed-optimized but multiple options
        self.PATIENCE = 4                        # Aggressive early stopping for speed
        self.BATCH_SIZE = [32, 64]               # Focus on faster batches
        self.LEARNING_RATE_RANGE = (1e-4, 1e-3)  # Around proven range
        self.WEIGHT_DECAY_RANGE = (0.005, 0.025) # Around proven range
        self.SCHEDULER_OPTIONS = ['cosine', 'onecycle']  # Multiple proven schedulers
        
        # CNN-specific training parameters - expanded ranges
        self.MIN_DELTA_RANGE = (1e-5, 1e-3)     # Full range like BLSTM
        self.GRADIENT_CLIP_RANGE = (0.3, 1.0)   # Around proven 0.5
        self.CONVERGENCE_WINDOW_RANGE = (3, 5)  # Short window range
        
        # CNN validation strategies - test all like BLSTM
        self.VALIDATION_STRATEGY_OPTIONS = ['cnn_composite', 'boundary_f1', 'line_f1']  # Equal weight for all strategies
        
        # ================================================================
        # CNN LOSS PARAMETERS - Conservative ranges for speed
        # ================================================================
        self.BOUNDARY_WEIGHT_RANGE = (2.0, 2.5)       # Narrow proven range
        self.LABEL_SMOOTHING_RANGE = (0.15, 0.20)     # Narrow range
        self.SEGMENT_CONSISTENCY_RANGE = (0.015, 0.025) # Minimal range
        self.CONF_PENALTY_RANGE = (0.005, 0.010)      # Conservative
        self.CONF_THRESHOLD_RANGE = (0.92, 0.95)      # Proven range
        self.ENTROPY_LAMBDA_RANGE = (0.02, 0.04)      # Narrow beneficial range
        
        # ================================================================
        # CNN EMERGENCY MONITORING - Fixed conservative values
        # ================================================================
        self.MAX_CONFIDENCE_THRESHOLD_RANGE = (0.90, 0.92)  # Very narrow
        self.MIN_CHORUS_RATE_RANGE = (0.05, 0.07)           # Tight range
        self.MAX_CHORUS_RATE_RANGE = (0.82, 0.88)           # Tight range
        self.MAX_CONF_OVER_95_RATIO_RANGE = (0.05, 0.08)    # Narrow range
        
        # ================================================================
        # FEATURE PARAMETERS - REDUCED for memory safety
        # ================================================================
        self.OPTIMIZE_FEATURE_TOGGLES = True   # Enable feature optimization like BLSTM
        self.HEAD_TAIL_WORDS_OPTIONS = [2, 3, 4]  # Test different word counts like BLSTM
        
        # Feature toggles - REDUCED for memory efficiency
        self.FEATURE_TOGGLES = {
            'head_ssm_enabled': [True, False],
            'tail_ssm_enabled': [True, False],
            'phonetic_ssm_enabled': [True, False],
            'pos_ssm_enabled': [True, False],
            'string_ssm_enabled': [True, False],
            'syllable_pattern_ssm_enabled': [True, False],
            'line_syllable_ssm_enabled': [True, False],
            'word2vec_enabled': [True, False],
            'contextual_enabled': [True, False]
        }
        
        # Feature-specific parameters like BLSTM
        self.PHONETIC_MODE_OPTIONS = ["rhyme", "alliteration", "combined"]
        self.PHONETIC_SIMILARITY_OPTIONS = ["binary", "edit_distance", "sequence_match"]
        self.PHONETIC_THRESHOLD_RANGE = (0.25, 0.35)     # Like BLSTM
        
        self.POS_TAGSET_OPTIONS = ["simplified", "universal", "penn"]
        self.POS_SIMILARITY_OPTIONS = ["combined", "lcs", "position", "jaccard"]
        self.POS_THRESHOLD_RANGE = (0.22, 0.32)        # Like BLSTM
        
        self.STRING_SIMILARITY_OPTIONS = ["word_overlap", "jaccard", "levenshtein"]
        self.STRING_THRESHOLD_RANGE = (0.045, 0.065)   # Like BLSTM
        
        # ================================================================
        # FAST TESTING SUMMARY -> COMPREHENSIVE OPTIMIZATION
        # ================================================================
        print("✅ CNN COMPREHENSIVE Hyperparameter Space Loaded")
        print(f"   🏗️  Architecture: {len(self.HIDDEN_DIM)} hidden_dim × {len(self.NUM_LAYERS)} layers")
        print(f"   🔧 CNN Kernels: {len(self.CNN_KERNEL_SIZES_OPTIONS)} configurations")
        print(f"   🎯 Attention: {len(self.ATTENTION_ENABLED_OPTIONS)} enable × {len(self.ATTENTION_TYPE_OPTIONS)} types × {len(self.ATTENTION_HEADS)} heads")
        print(f"   ⚡ Training: {len(self.MAX_EPOCHS_OPTIONS)} epochs × {len(self.BATCH_SIZE)} batch × {len(self.SCHEDULER_OPTIONS)} schedulers")
        print(f"   📊 Validation: {len(self.VALIDATION_STRATEGY_OPTIONS)} strategies")
        print(f"   🎵 Features: {len(self.HEAD_TAIL_WORDS_OPTIONS)} head/tail × {'ON/OFF toggles' if self.OPTIMIZE_FEATURE_TOGGLES else 'static'}")
        print(f"   🚀 COMPREHENSIVE OPTIMIZATION: Full parameter space like BLSTM")
        
        # ================================================================
        # DEFAULT CONFIGURATION
        # ================================================================
        self._load_default_feature_params()


    def _load_default_feature_params(self):
        """Load default feature configuration optimized for CNNs."""
        self.default_config = {
            # Data files
            'train_file': 'data/train.jsonl',
            'val_file': 'data/val.jsonl', 
            'test_file': 'data/test.jsonl',
            
            # Model defaults
            'num_classes': 2,
            'max_seq_length': 1000,
            'window_size': 7,
            'boundary_temperature': 2.0,
            'positional_encoding': True,
            
            # Training defaults
            'min_lr': 1e-6,
            'warmup_epochs': 5,
            'lr_factor': 0.3,
            'lr_patience': 2,
            'step_size': 0,
            'step_gamma': 0.5,
            'cosine_t0': 10,
            'cosine_t_mult': 2,
            'num_workers': 2,
            'weighted_sampling': True,
            
            # Emergency monitoring (CNN-tuned)
            'emergency_monitoring_enabled': True,
            'emergency_overconf_threshold': 0.94,  # More conservative
            'emergency_conf95_ratio': 0.65,
            'emergency_f1_threshold': 0.05,
            'val_overconf_threshold': 0.92,        # More conservative
            'val_f1_collapse_threshold': 0.10,
            'skip_batches': 15,                     # Shorter for CNNs
            'skip_epochs': 2,                       # Shorter for CNNs
            
            # Output and device
            'output_base_dir': 'results',
            'device': 'mps',  # Change to 'cuda' or 'cpu' as needed
            'seed': 42,
            'print_batch_every': 10,
            'save_best_model': True,
            'save_final_model': True,
            'save_training_metrics': True,
            
            # Calibration
            'calibration_enabled': True,
            'calibration_methods': ['temperature', 'platt']
        }


# Global instance (will be initialized in main)
CNN_HYPERPARAMS = None

# Session directory for organized results
SESSION_DIR = None


def create_cnn_session_directory(study_name: str) -> Path:
    """
    Create a CNN session directory organized by date and study name.
    
    Structure: results/YYYY-MM-DD/HH-MM-SS_cnn_study_name/
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    
    # Create the session directory path
    session_dir = Path("cnn_optuna_results") / date_str / f"{time_str}_cnn_{study_name}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    (session_dir / "trials").mkdir(exist_ok=True)
    (session_dir / "configs").mkdir(exist_ok=True)
    (session_dir / "logs").mkdir(exist_ok=True)
    
    # Create a session info file
    session_info = {
        "session_start": now.isoformat(),
        "study_name": study_name,
        "session_directory": str(session_dir),
        "script_version": "cnn_hyperopt_v1.0",
        "model_type": "CNN",
        "architecture": "CNNTagger",
        "subdirectories": {
            "trials": "Individual CNN trial results and configurations",
            "configs": "Best CNN configuration files",
            "logs": "CNN training and optimization logs"
        }
    }
    
    with open(session_dir / "session_info.json", 'w') as f:
        json.dump(session_info, f, indent=2)
    
    print(f"📁 CNN Session directory created: {session_dir}")
    print(f"   🧠 CNN Results will be saved to: {session_dir.absolute()}")
    print(f"   📂 Subdirectories: trials/, configs/, logs/")
    
    return session_dir


def get_optimal_device():
    """Get the best available compute device with proper MPS support."""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🚀 Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print(f"🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        device = 'cpu'
        print(f"💻 Using CPU")
    
    return device


def flatten_cnn_config_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested CNN config dictionary for TrainingConfig initialization.
    
    The TrainingConfig expects a flat dictionary with all parameters at the top level.
    This function flattens the nested YAML structure for CNN models.
    """
    flat_config = {}
    
    # Data section
    if 'data' in config_dict:
        flat_config.update({
            'train_file': config_dict['data'].get('train_file', 'data/train.jsonl'),
            'val_file': config_dict['data'].get('val_file', 'data/val.jsonl'),
            'test_file': config_dict['data'].get('test_file', 'data/test.jsonl')
        })
    
    # Model section (CNN-specific)
    if 'model' in config_dict:
        model_config = config_dict['model']
        flat_config.update({
            'hidden_dim': model_config.get('hidden_dim', 128),
            'num_layers': model_config.get('num_layers', 2),
            'layer_dropout': model_config.get('layer_dropout', 0.1),
            'num_classes': model_config.get('num_classes', 2),
            'dropout': model_config.get('dropout', 0.3),
            'attention_enabled': model_config.get('attention_enabled', True),
            'attention_type': model_config.get('attention_type', 'boundary_aware'),
            'attention_heads': model_config.get('attention_heads', 4),
            'attention_dropout': model_config.get('attention_dropout', 0.1),
            'attention_dim': model_config.get('attention_dim', None),
            'positional_encoding': model_config.get('positional_encoding', True),
            'max_seq_length': model_config.get('max_seq_length', 1000),
            'window_size': model_config.get('window_size', 7),
            'boundary_temperature': model_config.get('boundary_temperature', 2.0)
        })
    
    # Training section (CNN-optimized)
    if 'training' in config_dict:
        training_config = config_dict['training']
        flat_config.update({
            'batch_size': training_config.get('batch_size', 32),
            'learning_rate': training_config.get('learning_rate', 0.002),
            'weight_decay': training_config.get('weight_decay', 0.005),
            'max_epochs': training_config.get('max_epochs', 15),
            'patience': training_config.get('patience', 5),
            'gradient_clip_norm': training_config.get('gradient_clip_norm', 2.0),
            'scheduler': training_config.get('scheduler', 'onecycle'),
            'min_lr': training_config.get('min_lr', 1e-6),
            'warmup_epochs': training_config.get('warmup_epochs', 5),
            'lr_factor': training_config.get('lr_factor', 0.3),
            'lr_patience': training_config.get('lr_patience', 2),
            'validation_strategy': training_config.get('validation_strategy', 'cnn_composite'),
            
            # CNN-specific training parameters  
            'cnn_kernel_sizes': training_config.get('cnn_kernel_sizes', [3, 5, 7]),
            'cnn_dilation_rates': training_config.get('cnn_dilation_rates', [1, 2, 4]),
            'cnn_use_residual': training_config.get('cnn_use_residual', True),
            'cnn_batch_multiplier': training_config.get('cnn_batch_multiplier', 1.5),
            'cnn_weight_decay': training_config.get('cnn_weight_decay', 0.005),
        })
    
    # CNN section (dedicated CNN parameters)
    if 'cnn' in config_dict:
        cnn_config = config_dict['cnn']
        flat_config.update({
            'cnn_kernel_sizes': cnn_config.get('kernel_sizes', [3, 5, 7]),
            'cnn_dilation_rates': cnn_config.get('dilation_rates', [1, 2, 4]),
            'cnn_use_residual': cnn_config.get('use_residual', True),
            'cnn_batch_multiplier': cnn_config.get('batch_multiplier', 1.5),
            'cnn_weight_decay': cnn_config.get('weight_decay', 0.005),
        })
    
    # Loss configuration (keep as nested for proper handling)
    if 'loss' in config_dict:
        flat_config['loss'] = config_dict['loss']
    
    # ========================================================================
    # FEATURES SECTION - Flatten nested features to flat keys
    # ========================================================================
    if 'features' in config_dict:
        features_config = config_dict['features']
        
        # Head SSM
        if 'head_ssm' in features_config:
            head_config = features_config['head_ssm']
            flat_config.update({
                'head_ssm_enabled': head_config.get('enabled', True),
                'head_ssm_words': head_config.get('head_words', 3),
                'head_ssm_dimension': head_config.get('dimension', 12)
            })
        
        # Tail SSM
        if 'tail_ssm' in features_config:
            tail_config = features_config['tail_ssm']
            flat_config.update({
                'tail_ssm_enabled': tail_config.get('enabled', True),
                'tail_ssm_words': tail_config.get('tail_words', 3),
                'tail_ssm_dimension': tail_config.get('dimension', 12)
            })
        
        # Phonetic SSM
        if 'phonetic_ssm' in features_config:
            phonetic_config = features_config['phonetic_ssm']
            flat_config.update({
                'phonetic_ssm_enabled': phonetic_config.get('enabled', True),
                'phonetic_ssm_mode': phonetic_config.get('mode', 'rhyme'),
                'phonetic_ssm_dimension': phonetic_config.get('dimension', 12),
                'phonetic_ssm_similarity_method': phonetic_config.get('similarity_method', 'binary'),
                'phonetic_ssm_normalize': phonetic_config.get('normalize', True),
                'phonetic_ssm_normalize_method': phonetic_config.get('normalize_method', 'zscore'),
                'phonetic_ssm_high_sim_threshold': phonetic_config.get('high_sim_threshold', 0.7)
            })
        
        # Add other feature sections as needed...
        # (POS SSM, String SSM, etc.)
    
    return flat_config


def extract_cnn_feature_config(config: TrainingConfig) -> Dict[str, Any]:
    """Extract feature configuration for CNN FeatureExtractor."""
    return {
        'head_ssm': {
            'enabled': getattr(config, 'head_ssm_enabled', True),
            'head_words': getattr(config, 'head_ssm_words', 3),
            'output_dim': getattr(config, 'head_ssm_dimension', 12)
        },
        'tail_ssm': {
            'enabled': getattr(config, 'tail_ssm_enabled', True),
            'tail_words': getattr(config, 'tail_ssm_words', 3),
            'output_dim': getattr(config, 'tail_ssm_dimension', 12)
        },
        'phonetic_ssm': {
            'enabled': getattr(config, 'phonetic_ssm_enabled', True),
            'mode': getattr(config, 'phonetic_ssm_mode', 'rhyme'),
            'output_dim': getattr(config, 'phonetic_ssm_dimension', 12),
            'similarity_method': getattr(config, 'phonetic_ssm_similarity_method', 'binary'),
            'normalize': getattr(config, 'phonetic_ssm_normalize', True),
            'normalize_method': getattr(config, 'phonetic_ssm_normalize_method', 'zscore'),
            'high_sim_threshold': getattr(config, 'phonetic_ssm_high_sim_threshold', 0.7)
        },
        # Add more feature configurations as needed
        'word2vec_enabled': getattr(config, 'word2vec_enabled', False),
        'contextual_enabled': getattr(config, 'contextual_enabled', False)
    }


def create_cnn_trial_config(trial, base_config_path=None, hyperparams=None):
    """
    Create CNN trial config with LOCAL configuration only.
    NO external config file loading - build everything locally for CNN.
    """
    
    # Use global CNN_HYPERPARAMS if not provided
    if hyperparams is None:
        global CNN_HYPERPARAMS
        hyperparams = CNN_HYPERPARAMS
    
    if hyperparams is None:
        raise ValueError("CNN hyperparameters not initialized")
    
    print(f"   🧠 Sampling CNN parameters for trial {trial.number}...")
    
    # ========================================================================
    # CNN ARCHITECTURE - OPTIMIZED FOR CONVOLUTION
    # ========================================================================
    hidden_dim = trial.suggest_categorical('hidden_dim', hyperparams.HIDDEN_DIM)
    num_layers = trial.suggest_categorical('num_layers', hyperparams.NUM_LAYERS)
    dropout = trial.suggest_float('dropout', *hyperparams.DROPOUT_RANGE)
    
    # Layer dropout only if num_layers > 1
    if num_layers > 1:
        layer_dropout = trial.suggest_float('layer_dropout', *hyperparams.LAYER_DROPOUT_RANGE)
    else:
        layer_dropout = 0.0
    
    # CNN-specific architecture parameters
    kernel_sizes_str = trial.suggest_categorical('cnn_kernel_sizes', hyperparams.CNN_KERNEL_SIZES_OPTIONS)
    dilation_rates_str = trial.suggest_categorical('cnn_dilation_rates', hyperparams.CNN_DILATION_RATES_OPTIONS)
    
    # Convert string representations back to lists of integers
    kernel_sizes = [int(x) for x in kernel_sizes_str.split(',')]
    dilation_rates = [int(x) for x in dilation_rates_str.split(',')]
    
    use_residual = trial.suggest_categorical('cnn_use_residual', hyperparams.CNN_USE_RESIDUAL_OPTIONS)
    batch_multiplier = trial.suggest_float('cnn_batch_multiplier', *hyperparams.CNN_BATCH_MULTIPLIER_RANGE)
    
    # ========================================================================
    # ATTENTION ON/OFF TOGGLE FOR CNN
    # ========================================================================
    attention_enabled = trial.suggest_categorical('attention_enabled', hyperparams.ATTENTION_ENABLED_OPTIONS)
    
    if attention_enabled:
        attention_type = trial.suggest_categorical('attention_type', hyperparams.ATTENTION_TYPE_OPTIONS)
        attention_heads = trial.suggest_categorical('attention_heads', hyperparams.ATTENTION_HEADS)
        attention_dropout = trial.suggest_float('attention_dropout', *hyperparams.ATTENTION_DROPOUT_RANGE)
        # Match attention dim to hidden dim for compatibility
        attention_dim = hidden_dim
        
        # Window size for localized attention
        if attention_type == 'localized':
            window_size = trial.suggest_categorical('window_size', hyperparams.WINDOW_SIZE_OPTIONS)
        else:
            window_size = 7  # Default value for other attention types
            
        # Validate attention heads compatibility
        if hidden_dim % attention_heads != 0:
            # Force compatible heads
            compatible_heads = [h for h in hyperparams.ATTENTION_HEADS if hidden_dim % h == 0]
            if compatible_heads:
                attention_heads = compatible_heads[0]  # Use first compatible option
            else:
                attention_heads = 4  # Safe fallback
    else:
        attention_type = 'self'
        attention_heads = 4
        attention_dropout = 0.1
        attention_dim = None
        window_size = 7
    
    # ========================================================================
    # CNN-OPTIMIZED TRAINING PARAMETERS
    # ========================================================================
    max_epochs = trial.suggest_categorical('max_epochs', hyperparams.MAX_EPOCHS_OPTIONS)
    cosine_t_max = max_epochs  # FORCE EQUALITY for scheduler
    
    batch_size = trial.suggest_categorical('batch_size', hyperparams.BATCH_SIZE)
    learning_rate = trial.suggest_float('learning_rate', *hyperparams.LEARNING_RATE_RANGE, log=True)
    weight_decay = trial.suggest_float('weight_decay', *hyperparams.WEIGHT_DECAY_RANGE, log=True)
    scheduler = trial.suggest_categorical('scheduler', hyperparams.SCHEDULER_OPTIONS)
    
    # CNN-specific training parameters
    patience = hyperparams.PATIENCE
    min_delta = trial.suggest_float('min_delta', *hyperparams.MIN_DELTA_RANGE)
    gradient_clip_norm = trial.suggest_float('gradient_clip_norm', *hyperparams.GRADIENT_CLIP_RANGE)
    
    # CNN VALIDATION STRATEGY
    validation_strategy = trial.suggest_categorical('validation_strategy', hyperparams.VALIDATION_STRATEGY_OPTIONS)
    
    # ========================================================================
    # CNN LOSS PARAMETERS (adjusted for CNN)
    # ========================================================================
    boundary_weight = trial.suggest_float('boundary_weight', *hyperparams.BOUNDARY_WEIGHT_RANGE)
    label_smoothing = trial.suggest_float('label_smoothing', *hyperparams.LABEL_SMOOTHING_RANGE)
    segment_consistency_lambda = trial.suggest_float('segment_consistency_lambda', *hyperparams.SEGMENT_CONSISTENCY_RANGE)
    conf_penalty_lambda = trial.suggest_float('conf_penalty_lambda', *hyperparams.CONF_PENALTY_RANGE)
    conf_threshold = trial.suggest_float('conf_threshold', *hyperparams.CONF_THRESHOLD_RANGE)
    entropy_lambda = trial.suggest_float('entropy_lambda', *hyperparams.ENTROPY_LAMBDA_RANGE)
    
    # ========================================================================
    # CNN EMERGENCY MONITORING PARAMETERS
    # ========================================================================
    max_confidence_threshold = trial.suggest_float('max_confidence_threshold', *hyperparams.MAX_CONFIDENCE_THRESHOLD_RANGE)
    min_chorus_rate = trial.suggest_float('min_chorus_rate', *hyperparams.MIN_CHORUS_RATE_RANGE)
    max_chorus_rate = trial.suggest_float('max_chorus_rate', *hyperparams.MAX_CHORUS_RATE_RANGE)
    max_conf_over_95_ratio = trial.suggest_float('max_conf_over_95_ratio', *hyperparams.MAX_CONF_OVER_95_RATIO_RANGE)
    
    # ========================================================================
    # FEATURE TOGGLES FOR CNN - Full optimization like BLSTM
    # ========================================================================
    feature_toggles = {}
    if hyperparams.OPTIMIZE_FEATURE_TOGGLES:
        # Feature toggles (ON/OFF like BLSTM)
        for feature_name, options in hyperparams.FEATURE_TOGGLES.items():
            feature_toggles[feature_name] = trial.suggest_categorical(feature_name, options)
        
        # Head/tail words variation
        head_tail_words = trial.suggest_categorical('head_tail_words', hyperparams.HEAD_TAIL_WORDS_OPTIONS)
        
        # Feature-specific parameters like BLSTM
        phonetic_mode = trial.suggest_categorical('phonetic_mode', hyperparams.PHONETIC_MODE_OPTIONS)
        phonetic_similarity = trial.suggest_categorical('phonetic_similarity', hyperparams.PHONETIC_SIMILARITY_OPTIONS)
        phonetic_threshold = trial.suggest_float('phonetic_threshold', *hyperparams.PHONETIC_THRESHOLD_RANGE)
        
        pos_tagset = trial.suggest_categorical('pos_tagset', hyperparams.POS_TAGSET_OPTIONS)
        pos_similarity = trial.suggest_categorical('pos_similarity', hyperparams.POS_SIMILARITY_OPTIONS)
        pos_threshold = trial.suggest_float('pos_threshold', *hyperparams.POS_THRESHOLD_RANGE)
        
        string_similarity = trial.suggest_categorical('string_similarity', hyperparams.STRING_SIMILARITY_OPTIONS)
        string_threshold = trial.suggest_float('string_threshold', *hyperparams.STRING_THRESHOLD_RANGE)
        
        feature_toggles.update({
            'head_tail_words': head_tail_words,
            'phonetic_mode': phonetic_mode,
            'phonetic_similarity': phonetic_similarity,
            'phonetic_threshold': phonetic_threshold,
            'pos_tagset': pos_tagset,
            'pos_similarity': pos_similarity,
            'pos_threshold': pos_threshold,
            'string_similarity': string_similarity,
            'string_threshold': string_threshold
        })
    else:
        # Use defaults (fallback)
        feature_toggles = {
            'head_ssm_enabled': True,
            'tail_ssm_enabled': True,
            'phonetic_ssm_enabled': True,
            'pos_ssm_enabled': True,
            'string_ssm_enabled': True,
            'syllable_pattern_ssm_enabled': True,
            'line_syllable_ssm_enabled': True,
            'word2vec_enabled': True,
            'contextual_enabled': True,
            'head_tail_words': 3,
            'phonetic_mode': 'rhyme',
            'phonetic_similarity': 'binary',
            'phonetic_threshold': 0.7,
            'pos_tagset': 'universal',
            'pos_similarity': 'combined',
            'pos_threshold': 0.27,
            'string_similarity': 'word_overlap',
            'string_threshold': 0.055
        }
    
    # ========================================================================
    # BUILD CNN CONFIGURATION
    # ========================================================================
    
    # Start with defaults
    config_dict = hyperparams.default_config.copy()
    
    # Update with sampled parameters
    config_dict.update({
        # CNN Model architecture
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'layer_dropout': layer_dropout,
        
        # Attention parameters
        'attention_enabled': attention_enabled,
        'attention_type': attention_type,
        'attention_heads': attention_heads,
        'attention_dropout': attention_dropout,
        'attention_dim': attention_dim,
        'window_size': window_size,
        
        # CNN Training parameters
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'max_epochs': max_epochs,
        'patience': patience,
        'min_delta': min_delta,
        'gradient_clip_norm': gradient_clip_norm,
        'scheduler': scheduler,
        'cosine_t_max': cosine_t_max,
        'validation_strategy': validation_strategy,
        
        # CNN Emergency monitoring
        'max_confidence_threshold': max_confidence_threshold,
        'min_chorus_rate': min_chorus_rate,
        'max_chorus_rate': max_chorus_rate,
        'max_conf_over_95_ratio': max_conf_over_95_ratio,
        
        # Feature parameters - comprehensive like BLSTM
        'head_ssm_enabled': feature_toggles.get('head_ssm_enabled', True),
        'head_ssm_words': feature_toggles['head_tail_words'],
        'head_ssm_dimension': 12,
        
        'tail_ssm_enabled': feature_toggles.get('tail_ssm_enabled', True),
        'tail_ssm_words': feature_toggles['head_tail_words'],
        'tail_ssm_dimension': 12,
        
        'phonetic_ssm_enabled': feature_toggles.get('phonetic_ssm_enabled', True),
        'phonetic_ssm_mode': feature_toggles['phonetic_mode'],
        'phonetic_ssm_dimension': 12,
        'phonetic_ssm_similarity_method': feature_toggles['phonetic_similarity'],
        'phonetic_ssm_normalize': True,
        'phonetic_ssm_normalize_method': 'zscore',
        'phonetic_ssm_high_sim_threshold': feature_toggles['phonetic_threshold'],
        
        'pos_ssm_enabled': feature_toggles.get('pos_ssm_enabled', True),
        'pos_ssm_tagset': feature_toggles.get('pos_tagset', 'universal'),
        'pos_ssm_similarity_method': feature_toggles.get('pos_similarity', 'combined'),
        'pos_ssm_high_sim_threshold': feature_toggles.get('pos_threshold', 0.27),
        'pos_ssm_dimension': 12,
        
        'string_ssm_enabled': feature_toggles.get('string_ssm_enabled', True),
        'string_ssm_case_sensitive': False,
        'string_ssm_remove_punctuation': True,
        'string_ssm_similarity_threshold': feature_toggles.get('string_threshold', 0.055),
        'string_ssm_similarity_method': feature_toggles.get('string_similarity', 'word_overlap'),
        'string_ssm_dimension': 12,
        
        'syllable_pattern_ssm_enabled': feature_toggles.get('syllable_pattern_ssm_enabled', True),
        'syllable_pattern_ssm_similarity_method': 'cosine',
        'syllable_pattern_ssm_dimension': 12,
        
        'line_syllable_ssm_enabled': feature_toggles.get('line_syllable_ssm_enabled', True),
        'line_syllable_ssm_similarity_method': 'cosine',
        'line_syllable_ssm_dimension': 12,
        
        'word2vec_enabled': feature_toggles.get('word2vec_enabled', True),
        'word2vec_model': 'word2vec-google-news-300',
        'word2vec_mode': 'complete',
        'word2vec_normalize': True,
        'word2vec_similarity_metric': 'cosine',
        'word2vec_high_sim_threshold': 0.82,
        
        'contextual_enabled': feature_toggles.get('contextual_enabled', True),
        'contextual_model': 'all-MiniLM-L6-v2',
        'contextual_mode': 'complete',
        'contextual_normalize': True,
        'contextual_similarity_metric': 'cosine',
        'contextual_high_sim_threshold': 0.72,
        
        # Loss configuration
        'loss': {
            'type': 'boundary_aware_cross_entropy',
            'label_smoothing': label_smoothing,
            'boundary_weight': boundary_weight,
            'segment_consistency_lambda': segment_consistency_lambda,
            'conf_penalty_lambda': conf_penalty_lambda,
            'conf_threshold': conf_threshold,
            'entropy_lambda': entropy_lambda,
            'use_boundary_as_primary': True
        }
    })
    
    # Create TrainingConfig from flattened dict
    flat_config = flatten_cnn_config_dict({'root': config_dict})
    flat_config.update(config_dict)  # Add root level items
    
    try:
        config = TrainingConfig(**flat_config)
        
        # Add CNN-specific parameters as attributes (not part of TrainingConfig)
        config.cnn_kernel_sizes = kernel_sizes
        config.cnn_dilation_rates = dilation_rates  
        config.cnn_use_residual = use_residual
        config.cnn_batch_multiplier = batch_multiplier
        config.cnn_weight_decay = weight_decay
        
        # Log CNN trial info
        print(f"   🧠 CNN Trial {trial.number} configuration:")
        print(f"      Architecture: CNN {num_layers} blocks, {hidden_dim}D")
        print(f"      Kernels: {kernel_sizes}, Dilations: {dilation_rates}")
        print(f"      Residual: {use_residual}")
        if attention_enabled:
            print(f"      Attention: {attention_type} ({attention_heads} heads)")
            if attention_type == 'localized':
                print(f"      Window size: {window_size}")
        else:
            print(f"      Attention: DISABLED")
        print(f"      Training: {batch_size} batch, {max_epochs} epochs, {scheduler}")
        print(f"      Validation: {validation_strategy}")
        if hyperparams.OPTIMIZE_FEATURE_TOGGLES:
            enabled_features = [k for k, v in feature_toggles.items() if k.endswith('_enabled') and v]
            print(f"      Features: {len(enabled_features)} enabled")
        
        return config
        
    except Exception as e:
        print(f"❌ Error creating CNN TrainingConfig: {e}")
        print(f"   Config keys: {list(flat_config.keys())}")
        raise


def set_all_seeds(seed: int):
    """Set all random seeds for reproducible CNN training."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS doesn't have deterministic operations yet
        pass


def cnn_objective(trial):
    """
    CNN objective function for Optuna optimization.
    
    This function:
    1. Creates a CNN trial configuration
    2. Sets up CNN data loaders and model
    3. Trains the CNN model with boundary-aware loss
    4. Returns the CNN validation score for optimization
    """
    trial_id = f"cnn_trial_{trial.number:04d}"
    
    try:
        print(f"\n🧠 Starting CNN Trial {trial.number}")
        print("=" * 60)
        
        # Set seeds for reproducibility
        set_all_seeds(42)
        
        # Create CNN trial configuration
        config = create_cnn_trial_config(trial)
        
        # Set device
        device = get_optimal_device()
        config.device = device
        
        print(f"📊 Setting up CNN data and model...")
        
        # Extract feature configuration for CNN
        feature_config = extract_cnn_feature_config(config)
        
        # Create feature extractor for CNN
        feature_extractor = FeatureExtractor(feature_config)
        print(f"✅ CNN Feature extractor created with {feature_extractor.total_dim}D features")
        
        # Setup CNN data loaders
        train_loader, val_loader, test_loader, train_dataset = setup_data_loaders(
            config, feature_extractor
        )
        print(f"✅ CNN Data loaders created")
        
        # Setup CNN model and training components
        model, loss_function, optimizer = setup_cnn_model_and_training(
            config, train_dataset, device, feature_extractor.total_dim
        )
        print(f"✅ CNN Model and training setup complete")
        
        # Create CNN trainer - save to session trials directory
        trial_output_dir = SESSION_DIR / "trials" / trial_id
        trial_output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = CNNTrainer(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            config=config,
            device=device,
            output_dir=trial_output_dir
        )
        print(f"✅ CNN Trainer created")
        
        # Train CNN model
        print(f"🚀 Starting CNN training...")
        trained_model, calibration_info = trainer.train(train_loader, val_loader)
        
        # Extract validation metrics from trainer
        # The trainer stores the best metrics in training_metrics and best_val_score
        if trainer.training_metrics:
            # Use the last epoch metrics since trainer loads best model at the end
            best_metrics = trainer.training_metrics[-1]
            
            # Extract validation metrics - build dictionary for compatibility
            val_metrics = {
                'line_f1': getattr(best_metrics, 'val_macro_f1', 0.0),
                'boundary_f1': getattr(best_metrics, 'val_boundary_f1', 0.0),
                'macro_f1': getattr(best_metrics, 'val_macro_f1', 0.0),
                'verse_f1': getattr(best_metrics, 'val_verse_f1', 0.0),
                'chorus_f1': getattr(best_metrics, 'val_chorus_f1', 0.0),
                'boundary_precision': getattr(best_metrics, 'val_boundary_precision', 0.0),
                'boundary_recall': getattr(best_metrics, 'val_boundary_recall', 0.0),
                'window_diff': getattr(best_metrics, 'val_window_diff', 1.0),
                'pk_metric': getattr(best_metrics, 'val_pk_metric', 1.0),
                'confidence': getattr(best_metrics, 'val_max_prob', 0.0),
                'val_loss': getattr(best_metrics, 'val_loss', 1.0),
            }
            
            # Debug print to verify we're getting real values
            print(f"🔍 Debug: Best epoch metrics extracted:")
            print(f"   val_macro_f1: {getattr(best_metrics, 'val_macro_f1', 'MISSING')}")
            print(f"   val_boundary_f1: {getattr(best_metrics, 'val_boundary_f1', 'MISSING')}")
            print(f"   val_verse_f1: {getattr(best_metrics, 'val_verse_f1', 'MISSING')}")
            print(f"   val_chorus_f1: {getattr(best_metrics, 'val_chorus_f1', 'MISSING')}")
            
        else:
            # Fallback values if no metrics available
            val_metrics = {
                'line_f1': 0.0,
                'boundary_f1': 0.0,
                'macro_f1': 0.0,
                'verse_f1': 0.0,
                'chorus_f1': 0.0,
                'boundary_precision': 0.0,
                'boundary_recall': 0.0,
                'window_diff': 1.0,
                'pk_metric': 1.0,
                'confidence': 0.0,
                'val_loss': 1.0,
            }
            print(f"⚠️  Warning: No training metrics found, using fallback values")
        
        # CNN-specific score computation for training
        validation_score = compute_cnn_validation_score(val_metrics, config)
        
        # But for Optuna optimization, ALWAYS use boundary_f1 as objective
        optuna_objective = val_metrics.get('boundary_f1', 0.0)
        
        # Log CNN trial results
        print(f"\n📊 CNN Trial {trial.number} Results:")
        print(f"   Training Score ({getattr(config, 'validation_strategy', 'unknown')}): {validation_score:.4f}")
        print(f"   Optuna Objective (boundary_f1): {optuna_objective:.4f}")
        print(f"   Line F1: {val_metrics.get('line_f1', 0):.4f}")
        print(f"   Boundary F1: {val_metrics.get('boundary_f1', 0):.4f}")
        print(f"   Macro F1: {val_metrics.get('macro_f1', 0):.4f}")
        print(f"   Verse F1: {val_metrics.get('verse_f1', 0):.4f}")
        print(f"   Chorus F1: {val_metrics.get('chorus_f1', 0):.4f}")
        print(f"   Window Diff: {val_metrics.get('window_diff', 1.0):.4f}")
        print(f"   Pk Metric: {val_metrics.get('pk_metric', 1.0):.4f}")
        print(f"   Confidence: {val_metrics.get('confidence', 0):.3f}")
        
        # Save CNN trial results
        trial_dir = SESSION_DIR / "trials" / trial_id
        trial_dir.mkdir(exist_ok=True)
        
        # Save CNN trial configuration
        with open(trial_dir / "cnn_config.yaml", 'w') as f:
            # Convert config to dict for YAML serialization
            config_dict = {}
            for key, value in config.__dict__.items():
                if not key.startswith('_'):
                    config_dict[key] = value
            yaml.dump(config_dict, f, indent=2)
        
        # Save CNN trial results
        trial_results = {
            'trial_number': trial.number,
            'trial_id': trial_id,
            'model_type': 'CNN',
            'training_score': validation_score,  # Score used during training
            'optuna_objective': optuna_objective,  # Boundary F1 (what Optuna maximizes)
            'validation_metrics': val_metrics,
            'all_results': val_metrics,  # Store the validation metrics
            'parameters': trial.params,
            'duration_minutes': 0  # Training time not tracked in CNN trainer
        }
        
        with open(trial_dir / "cnn_results.json", 'w') as f:
            json.dump(trial_results, f, indent=2, default=str)
        
        # Log all metrics to Optuna for visualization and analysis
        trial.set_user_attr("val_macro_f1", val_metrics.get('macro_f1', 0.0))
        trial.set_user_attr("val_boundary_f1", val_metrics.get('boundary_f1', 0.0))
        trial.set_user_attr("val_verse_f1", val_metrics.get('verse_f1', 0.0))
        trial.set_user_attr("val_chorus_f1", val_metrics.get('chorus_f1', 0.0))
        trial.set_user_attr("val_line_f1", val_metrics.get('line_f1', 0.0))
        trial.set_user_attr("val_boundary_precision", val_metrics.get('boundary_precision', 0.0))
        trial.set_user_attr("val_boundary_recall", val_metrics.get('boundary_recall', 0.0))
        
        # Segmentation metrics (as-is, lower is better)
        trial.set_user_attr("val_window_diff", val_metrics.get('window_diff', 1.0))
        trial.set_user_attr("val_pk_metric", val_metrics.get('pk_metric', 1.0))
        
        trial.set_user_attr("val_confidence", val_metrics.get('confidence', 0.0))
        trial.set_user_attr("val_loss", val_metrics.get('val_loss', 1.0))
        trial.set_user_attr("training_score", validation_score)  # Score used during training
        trial.set_user_attr("optuna_objective", optuna_objective)  # Boundary F1 (what Optuna maximizes)
        trial.set_user_attr("model_type", "CNN")
        trial.set_user_attr("trial_id", trial_id)
        
        # Log key hyperparameters for easy analysis
        trial.set_user_attr("hidden_dim", getattr(config, 'hidden_dim', 0))
        trial.set_user_attr("cnn_layers", getattr(config, 'cnn_layers', 0))
        trial.set_user_attr("kernel_sizes", str(getattr(config, 'kernel_sizes', [])))
        trial.set_user_attr("dilation_rates", str(getattr(config, 'dilation_rates', [])))
        trial.set_user_attr("attention_enabled", getattr(config, 'attention_enabled', False))
        trial.set_user_attr("attention_type", getattr(config, 'attention_type', 'none'))
        trial.set_user_attr("validation_strategy", getattr(config, 'validation_strategy', 'unknown'))
        
        print(f"✅ CNN Trial {trial.number} completed successfully")
        
        return optuna_objective  # Always return boundary_f1 for Optuna to maximize
        
    except Exception as e:
        print(f"❌ CNN Trial {trial.number} failed: {str(e)}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        
        # Return a poor score to indicate failure
        return 0.0


def compute_cnn_validation_score(metrics, config: Any) -> float:
    """
    CNN-optimized validation score computation.
    Emphasizes boundary detection which CNNs excel at.
    """
    
    strategy = getattr(config, 'validation_strategy', 'cnn_composite')
    
    if strategy == 'cnn_composite':
        # CNN-optimized composite score
        # Weight boundary metrics more heavily since CNNs excel at local patterns
        line_f1 = metrics.get('line_f1', 0)
        boundary_f1 = metrics.get('boundary_f1', 0)
        macro_f1 = metrics.get('macro_f1', 0)
        confidence = metrics.get('confidence', 0)
        
        # CNN composite: emphasize boundary detection
        score = (
            0.35 * line_f1 +           # Line-level performance
            0.40 * boundary_f1 +       # Boundary detection (CNN strength)
            0.20 * macro_f1 +          # Overall class balance
            0.05 * confidence          # Confidence calibration
        )
        
    elif strategy == 'boundary_f1':
        # Pure boundary F1 (CNN should excel here)
        score = metrics.get('boundary_f1', 0)
        
    elif strategy == 'line_f1':
        # Line-level F1 score
        score = metrics.get('line_f1', 0)
        
    else:
        # Default: line F1
        score = metrics.get('line_f1', 0)
    
    return score


def export_cnn_results(study, session_dir: Path):
    """Export CNN optimization results to various formats."""
    
    print(f"📄 Exporting CNN results...")
    
    # Create results summary
    results = []
    for trial in study.trials:
        if trial.value is not None:
            result = {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            results.append(result)
    
    # Sort by value (descending)
    results.sort(key=lambda x: x['value'], reverse=True)
    
    # Export to JSON
    with open(session_dir / "cnn_optimization_results.json", 'w') as f:
        json.dump({
            'study_name': study.study_name,
            'model_type': 'CNN',
            'optimization_completed': datetime.now().isoformat(),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'all_results': results
        }, f, indent=2)
    
    # Export best parameters to YAML
    with open(session_dir / "configs" / "best_cnn_params.yaml", 'w') as f:
        yaml.dump({
            'best_cnn_validation_score': study.best_value,
            'best_cnn_parameters': study.best_params,
            'optimization_info': {
                'study_name': study.study_name,
                'model_type': 'CNN',
                'n_trials': len(study.trials),
                'optimization_date': datetime.now().isoformat()
            }
        }, f, indent=2)
    
    print(f"✅ CNN Results exported to {session_dir}")


def create_best_cnn_config(study, session_dir: Path):
    """Create a complete CNN training configuration from the best trial."""
    
    print(f"🏆 Creating best CNN configuration...")
    
    try:
        # Get best parameters
        best_params = study.best_params
        
        # Use CNN hyperparameters to create the config
        global CNN_HYPERPARAMS
        
        # Build full configuration
        config_dict = CNN_HYPERPARAMS.default_config.copy()
        
        # Update with best parameters
        config_dict.update({
            # CNN Architecture
            'hidden_dim': best_params['hidden_dim'],
            'num_layers': best_params['num_layers'],
            'dropout': best_params['dropout'],
            'layer_dropout': best_params.get('layer_dropout', 0.0),
            
            # CNN-specific parameters - convert string representations back to lists
            'cnn_kernel_sizes': [int(x) for x in best_params['cnn_kernel_sizes'].split(',')],
            'cnn_dilation_rates': [int(x) for x in best_params['cnn_dilation_rates'].split(',')],
            'cnn_use_residual': best_params['cnn_use_residual'],
            'cnn_batch_multiplier': best_params['cnn_batch_multiplier'],
            'cnn_weight_decay': best_params.get('weight_decay', 0.01),
            
            # Attention
            'attention_enabled': best_params['attention_enabled'],
            'attention_type': best_params.get('attention_type', 'self'),
            'attention_heads': best_params.get('attention_heads', 4),
            'attention_dropout': best_params.get('attention_dropout', 0.1),
            'attention_dim': None,  # Will be auto-sized
            'positional_encoding': True,
            'max_seq_length': 1000,
            'window_size': best_params.get('window_size', 7),
            'boundary_temperature': 2.0,
            
            # Training
            'batch_size': best_params['batch_size'],
            'learning_rate': best_params['learning_rate'],
            'weight_decay': best_params['weight_decay'],
            'max_epochs': best_params['max_epochs'],
            'patience': CNN_HYPERPARAMS.PATIENCE,  # Fixed: get from hyperparams
            'min_delta': best_params.get('min_delta', 0.001),
            'gradient_clip_norm': best_params['gradient_clip_norm'],
            'scheduler': best_params['scheduler'],
            'validation_strategy': best_params['validation_strategy'],
            'num_workers': 2,
            'weighted_sampling': True,
            
            # Loss configuration
            'loss': {
                'type': 'boundary_aware_cross_entropy',
                'label_smoothing': best_params['label_smoothing'],
                'boundary_weight': best_params['boundary_weight'],
                'segment_consistency_lambda': best_params['segment_consistency_lambda'],
                'conf_penalty_lambda': best_params['conf_penalty_lambda'],
                'conf_threshold': best_params['conf_threshold'],
                'entropy_lambda': best_params['entropy_lambda'],
                'use_boundary_as_primary': True
            },
            
            # Feature parameters
            'head_ssm_words': best_params.get('head_tail_words', 3),
            'tail_ssm_words': best_params.get('head_tail_words', 3),
            'phonetic_ssm_mode': best_params.get('phonetic_mode', 'rhyme'),
            'phonetic_ssm_similarity_method': best_params.get('phonetic_similarity', 'binary'),
            'phonetic_ssm_high_sim_threshold': best_params.get('phonetic_threshold', 0.7),
            
            # Experiment metadata
            'experiment_name': f"cnn_optimized_best",
            'experiment_description': f"Best CNN configuration from hyperparameter optimization",
            'experiment_tags': ["CNN", "hyperopt", "boundary_aware", "optimized"],
            'experiment_notes': f"Generated from trial {study.best_trial.number} with score {study.best_value:.4f}",
            
            # Other required parameters
            'num_classes': 2,
            'device': 'mps',  # or 'cuda'/'cpu'
            'seed': 42
        })
        
        # Create structured YAML
        structured_config = {
            'experiment': {
                'name': config_dict['experiment_name'],
                'description': config_dict['experiment_description'],
                'tags': config_dict['experiment_tags'],
                'notes': config_dict['experiment_notes']
            },
            'data': {
                'train_file': config_dict['train_file'],
                'val_file': config_dict['val_file'],
                'test_file': config_dict['test_file']
            },
            'model': {
                'hidden_dim': config_dict['hidden_dim'],
                'num_layers': config_dict['num_layers'],
                'num_classes': config_dict['num_classes'],
                'dropout': config_dict['dropout'],
                'layer_dropout': config_dict['layer_dropout'],
                'attention_enabled': config_dict['attention_enabled'],
                'attention_type': config_dict['attention_type'],
                'attention_heads': config_dict['attention_heads'],
                'attention_dropout': config_dict['attention_dropout'],
                'positional_encoding': config_dict['positional_encoding'],
                'max_seq_length': config_dict['max_seq_length'],
                'window_size': config_dict['window_size'],
                'boundary_temperature': config_dict['boundary_temperature']
            },
            'training': {
                'batch_size': config_dict['batch_size'],
                'learning_rate': config_dict['learning_rate'],
                'weight_decay': config_dict['weight_decay'],
                'max_epochs': config_dict['max_epochs'],
                'patience': config_dict['patience'],
                'min_delta': config_dict.get('min_delta', 0.001),
                'gradient_clip_norm': config_dict['gradient_clip_norm'],
                'scheduler': config_dict['scheduler'],
                'validation_strategy': config_dict['validation_strategy'],
                'num_workers': config_dict['num_workers'],
                'weighted_sampling': config_dict['weighted_sampling']
            },
            'cnn': {
                'kernel_sizes': config_dict['cnn_kernel_sizes'],
                'dilation_rates': config_dict['cnn_dilation_rates'],
                'use_residual': config_dict['cnn_use_residual'],
                'batch_multiplier': config_dict['cnn_batch_multiplier'],
                'weight_decay': config_dict['cnn_weight_decay']
            },
            'loss': config_dict['loss'],
            'features': {
                'head_ssm': {
                    'enabled': True,
                    'head_words': config_dict['head_ssm_words'],
                    'dimension': 12
                },
                'tail_ssm': {
                    'enabled': True,
                    'tail_words': config_dict['tail_ssm_words'],
                    'dimension': 12
                },
                'phonetic_ssm': {
                    'enabled': True,
                    'mode': config_dict['phonetic_ssm_mode'],
                    'dimension': 12,
                    'similarity_method': config_dict['phonetic_ssm_similarity_method'],
                    'normalize': True,
                    'normalize_method': 'zscore',
                    'high_sim_threshold': config_dict['phonetic_ssm_high_sim_threshold']
                }
            },
            'calibration': {
                'methods': config_dict['calibration_methods']
            },
            'output_base_dir': config_dict['output_base_dir'],
            'device': config_dict['device'],
            'seed': config_dict['seed']
        }
        
        # Save the complete CNN configuration
        config_path = session_dir / "configs" / "best_cnn_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(structured_config, f, indent=2, default_flow_style=False)
        
        print(f"✅ Best CNN configuration saved to: {config_path}")
        print(f"🧠 CNN Architecture: {config_dict['num_layers']} blocks, {config_dict['hidden_dim']}D")
        print(f"🔧 CNN Kernels: {config_dict['cnn_kernel_sizes']}")
        print(f"🎯 Validation Score: {study.best_value:.4f}")
        
        return config_path
        
    except Exception as e:
        print(f"❌ Error creating best CNN config: {e}")
        traceback.print_exc()
        return None


def create_cnn_summary_report(study, session_dir: Path):
    """Create a comprehensive CNN optimization summary report."""
    
    print(f"📋 Creating CNN optimization summary...")
    
    try:
        # Get trials data
        completed_trials = [t for t in study.trials if t.value is not None]
        
        if not completed_trials:
            print("⚠️ No completed CNN trials found")
            return
        
        # Sort by value
        completed_trials.sort(key=lambda t: t.value, reverse=True)
        
        # Create summary
        summary = {
            'optimization_summary': {
                'study_name': study.study_name,
                'model_type': 'CNN',
                'completion_date': datetime.now().isoformat(),
                'total_trials': len(study.trials),
                'completed_trials': len(completed_trials),
                'best_score': study.best_value,
                'optimization_duration_hours': sum(
                    (t.duration.total_seconds() / 3600) for t in completed_trials if t.duration
                )
            },
            'best_trial': {
                'trial_number': study.best_trial.number,
                'validation_score': study.best_value,
                'parameters': study.best_params,
                'duration_minutes': study.best_trial.duration.total_seconds() / 60 if study.best_trial.duration else None
            },
            'top_10_trials': []
        }
        
        # Add top 10 trials
        for i, trial in enumerate(completed_trials[:10]):
            trial_info = {
                'rank': i + 1,
                'trial_number': trial.number,
                'validation_score': trial.value,
                'key_parameters': {
                    'hidden_dim': trial.params.get('hidden_dim'),
                    'num_layers': trial.params.get('num_layers'),
                    'cnn_kernel_sizes': trial.params.get('cnn_kernel_sizes'),
                    'attention_enabled': trial.params.get('attention_enabled'),
                    'learning_rate': trial.params.get('learning_rate'),
                    'validation_strategy': trial.params.get('validation_strategy')
                }
            }
            summary['top_10_trials'].append(trial_info)
        
        # Parameter analysis
        summary['parameter_analysis'] = {}
        
        # Analyze categorical parameters
        categorical_params = ['hidden_dim', 'num_layers', 'attention_enabled', 'validation_strategy']
        for param in categorical_params:
            if param in completed_trials[0].params:
                param_analysis = {}
                param_values = {}
                
                for trial in completed_trials:
                    value = trial.params.get(param)
                    if value not in param_values:
                        param_values[value] = []
                    param_values[value].append(trial.value)
                
                # Calculate statistics for each value
                for value, scores in param_values.items():
                    param_analysis[str(value)] = {
                        'count': len(scores),
                        'mean_score': sum(scores) / len(scores),
                        'max_score': max(scores),
                        'min_score': min(scores)
                    }
                
                summary['parameter_analysis'][param] = param_analysis
        
        # Save summary
        with open(session_dir / "cnn_optimization_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Create readable report
        report_lines = [
            "🧠 CNN Hyperparameter Optimization Summary",
            "=" * 60,
            "",
            f"📊 Study: {study.study_name}",
            f"🏗️ Model: CNN (Convolutional Neural Network)",
            f"📅 Completed: {summary['optimization_summary']['completion_date']}",
            f"🔢 Total Trials: {summary['optimization_summary']['total_trials']}",
            f"✅ Completed: {summary['optimization_summary']['completed_trials']}",
            f"🏆 Best Score: {summary['optimization_summary']['best_score']:.4f}",
            f"⏱️ Total Time: {summary['optimization_summary']['optimization_duration_hours']:.1f}h",
            "",
            "🥇 Best CNN Configuration:",
            f"   Trial: {summary['best_trial']['trial_number']}",
            f"   Score: {summary['best_trial']['validation_score']:.4f}",
            f"   Duration: {summary['best_trial']['duration_minutes']:.1f}m",
            "",
            "🔧 Best CNN Parameters:",
        ]
        
        for param, value in summary['best_trial']['parameters'].items():
            report_lines.append(f"   {param}: {value}")
        
        report_lines.extend([
            "",
            "🔝 Top 10 CNN Trials:",
            ""
        ])
        
        for trial in summary['top_10_trials']:
            report_lines.append(
                f"   #{trial['rank']:2d} | Trial {trial['trial_number']:3d} | "
                f"Score: {trial['validation_score']:.4f} | "
                f"Arch: {trial['key_parameters']['num_layers']}×{trial['key_parameters']['hidden_dim']}D | "
                f"Attn: {trial['key_parameters']['attention_enabled']}"
            )
        
        with open(session_dir / "cnn_optimization_report.txt", 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ CNN Summary report created")
        print(f"🏆 Best CNN score: {study.best_value:.4f}")
        
    except Exception as e:
        print(f"❌ Error creating CNN summary: {e}")
        traceback.print_exc()


def create_resumable_cnn_study(args, session_dir):
    """Create or resume a CNN-specific Optuna study."""
    
    storage_url = f"sqlite:///{session_dir / 'cnn_optuna_study.db'}"
    study_name = f"cnn_{args.study_name}"
    
    # Create or load study
    try:
        if args.resume and args.session_dir:
            # Resume existing study
            print(f"🔄 Resuming CNN study: {study_name}")
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            print(f"📈 Loaded {len(study.trials)} existing CNN trials")
        else:
            # Create new study
            print(f"🆕 Creating new CNN study: {study_name}")
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42)
            )
        
        return study
        
    except Exception as e:
        print(f"❌ Error with CNN study: {e}")
        raise


def main():
    """Main CNN hyperparameter optimization function."""
    
    parser = argparse.ArgumentParser(
        description="CNN Hyperparameter Optimization for Text Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CNN Hyperopt Examples:
  # Start new CNN optimization
  python scripts/hyperopt_cnn.py --study-name cnn_optimization_v1 --n-trials 50
  
  # Resume CNN optimization  
  python scripts/hyperopt_cnn.py --study-name cnn_optimization_v1 --resume --session-dir results/2025-08-27/14-30-00_cnn_optimization_v1 --n-trials 25
  
  # Quick CNN test
  python scripts/hyperopt_cnn.py --study-name cnn_quick_test --n-trials 10 --timeout 3600
        """
    )
    
    parser.add_argument('--study-name', required=True,
                       help='Name of the CNN optimization study')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of CNN trials to run (default: 50)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds for CNN optimization')
    parser.add_argument('--resume', action='store_true',
                       help='Resume existing CNN study')
    parser.add_argument('--session-dir', type=Path,
                       help='Existing session directory to resume from')
    
    args = parser.parse_args()
    
    try:
        print(f"🧠 CNN Hyperparameter Optimization")
        print("=" * 50)
        print(f"🔬 Study: {args.study_name}")
        print(f"🎯 Trials: {args.n_trials}")
        print(f"⏱️ Timeout: {args.timeout}s" if args.timeout else "⏱️ No timeout")
        print(f"🔄 Resume: {args.resume}")
        
        # Initialize global CNN hyperparameters
        global CNN_HYPERPARAMS, SESSION_DIR
        CNN_HYPERPARAMS = CNNHyperparameterSpace()
        
        # Create or use existing session directory
        if args.resume and args.session_dir:
            SESSION_DIR = Path(args.session_dir)
            print(f"📁 Using existing session: {SESSION_DIR}")
        else:
            SESSION_DIR = create_cnn_session_directory(args.study_name)
        
        # Create or resume study
        study = create_resumable_cnn_study(args, SESSION_DIR)
        
        print(f"\n🚀 Starting CNN optimization...")
        print(f"📊 Target metric: CNN validation score (higher is better)")
        
        # Run optimization
        study.optimize(
            cnn_objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True
        )
        
        print(f"\n🎉 CNN optimization completed!")
        print(f"🏆 Best CNN score: {study.best_value:.4f}")
        print(f"🧠 Best CNN trial: {study.best_trial.number}")
        
        # Export results
        export_cnn_results(study, SESSION_DIR)
        create_best_cnn_config(study, SESSION_DIR)
        create_cnn_summary_report(study, SESSION_DIR)
        
        print(f"\n📁 All CNN results saved to: {SESSION_DIR}")
        print(f"✅ CNN hyperparameter optimization complete!")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️ CNN optimization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ CNN optimization failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
