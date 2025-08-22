#!/usr/bin/env python3
"""
Advanced Hyperparameter Optimization for Boundary-Aware BiLSTM Text Segmentation.

This script optimizes the complex parameter space of the boundary-aware BiLSTM model
including attention mechanisms, loss function parameters, and model architecture.
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

from train_with_config import setup_data_loaders, setup_model_and_training
from segmodel.train.trainer import Trainer
from segmodel.utils import TrainingConfig
from segmodel.features import FeatureExtractor


# =============================================================================
# HYPERPARAMETER SEARCH SPACE CONFIGURATION
# =============================================================================

class HyperparameterSpace:
    """
    Centralized configuration for hyperparameter search space.
    NO EXTERNAL CONFIG LOADING - only local configuration.
    """
    
    def __init__(self, config_path=None):
        """Initialize with local configuration only."""
        print("📋 Using LOCAL search space configuration (no external files)")
        self._load_defaults()
    
    def _load_defaults(self):
        """Load optimized parameter ranges for boundary F1 optimization."""
        # ================================================================
        # MODEL ARCHITECTURE - COMPATIBLE COMBINATIONS ONLY
        # ================================================================
        self.HIDDEN_DIM = [128, 256, 512]        # Core dimensions
        self.NUM_LAYERS = [1, 2]                 # Limited as requested
        self.DROPOUT_RANGE = (0.1, 0.4)          # General dropout
        self.LAYER_DROPOUT_RANGE = (0.1, 0.3)    # Between layers
        
        # Attention - COMPATIBILITY ENFORCED
        # Heads must divide hidden_dim evenly and attention_dim should match hidden_dim
        self.ATTENTION_HEADS = [4, 8]            # Safe options: 128÷4=32, 256÷8=32, 512÷8=64
        self.ATTENTION_DIM = [128, 256, 512]     # Will be matched to hidden_dim in sampling
        self.ATTENTION_DROPOUT_RANGE = (0.1, 0.25)
        
        # ================================================================
        # TRAINING PARAMETERS - EPOCHS = COSINE_T_MAX
        # ================================================================
        self.MAX_EPOCHS_OPTIONS = [25, 27, 30]   # Used for BOTH max_epochs AND cosine_t_max (25-30 range)
        self.BATCH_SIZE = [16, 32, 64]
        self.LEARNING_RATE_RANGE = (1e-5, 1e-2)  # Log scale
        self.WEIGHT_DECAY_RANGE = (1e-6, 1e-3)
        self.SCHEDULER_OPTIONS = ['plateau', 'cosine', 'cosine_restarts']  # Fixed scheduler names
        
        # Early stopping and optimization
        self.PATIENCE_OPTIONS = [5, 8, 12]
        self.MIN_DELTA_RANGE = (1e-5, 1e-3)
        self.GRADIENT_CLIP_RANGE = (0.5, 2.0)
        
        # VALIDATION STRATEGY OPTIMIZATION (corrected to match trainer)
        self.VALIDATION_STRATEGY_OPTIONS = ['boundary_f1', 'line_f1', 'composite']
        
        # ================================================================
        # HEAD/TAIL WORD VARIATIONS (as requested)
        # ================================================================
        self.HEAD_TAIL_WORDS_OPTIONS = [2, 3, 4]  # Test different word counts
        
        # ================================================================
        # BOUNDARY-AWARE LOSS PARAMETERS
        # ================================================================
        self.BOUNDARY_WEIGHT_RANGE = (1.0, 3.0)
        self.LABEL_SMOOTHING_RANGE = (0.1, 0.3)
        self.SEGMENT_CONSISTENCY_RANGE = (0.01, 0.05)
        self.CONF_PENALTY_RANGE = (0.005, 0.02)
        self.CONF_THRESHOLD_RANGE = (0.90, 0.95)
        self.ENTROPY_LAMBDA_RANGE = (0.0, 0.08)
        
        # ================================================================
        # EMERGENCY MONITORING PARAMETERS
        # ================================================================
        self.MAX_CONFIDENCE_THRESHOLD_RANGE = (0.90, 0.98)
        self.MIN_CHORUS_RATE_RANGE = (0.05, 0.15)
        self.MAX_CHORUS_RATE_RANGE = (0.70, 0.90)
        self.MAX_CONF_OVER_95_RATIO_RANGE = (0.05, 0.20)
        
        # ================================================================
        # FEATURE CONFIGURATION - NO EMBEDDINGS DIMENSIONS/MODELS
        # ================================================================
        self.OPTIMIZE_FEATURE_TOGGLES = True
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
        self.OPTIMIZE_FEATURE_DIMS = False  # Keep dimensions static
        
        # Feature-specific parameters (NO embedding dimensions/models)
        self._load_default_feature_params()
        
        # ================================================================
        # CALIBRATION - ALWAYS ENABLED (as requested)
        # ================================================================
        self.CALIBRATION_METHODS = ['temperature', 'platt']  # Always use both
        
    def _load_default_feature_params(self):
        """Load feature-specific parameters (NO embedding models/dimensions)."""
        # Phonetic SSM Parameters
        self.PHONETIC_MODE_OPTIONS = ["rhyme", "alliteration", "combined"]
        self.PHONETIC_SIMILARITY_METHOD = ["binary", "edit_distance", "sequence_match"]
        self.PHONETIC_NORMALIZE_OPTIONS = [True, False]
        self.PHONETIC_NORMALIZE_METHOD = ["zscore", "minmax"]
        self.PHONETIC_HIGH_SIM_THRESHOLD_RANGE = (0.25, 0.35)
        
        # POS SSM Parameters
        self.POS_TAGSET_OPTIONS = ["simplified", "universal", "penn"]
        self.POS_SIMILARITY_METHOD = ["combined", "lcs", "position", "jaccard"]
        self.POS_HIGH_SIM_THRESHOLD_RANGE = (0.22, 0.32)
        
        # String SSM Parameters
        self.STRING_CASE_SENSITIVE_OPTIONS = [True, False]
        self.STRING_REMOVE_PUNCTUATION_OPTIONS = [True, False]
        self.STRING_SIMILARITY_THRESHOLD_RANGE = (0.045, 0.065)
        self.STRING_SIMILARITY_METHOD = ["word_overlap", "jaccard", "levenshtein"]
        
        # Syllable Pattern SSM Parameters
        self.SYLLABLE_PATTERN_SIMILARITY_METHOD = ["levenshtein", "cosine", "combined"]
        self.SYLLABLE_PATTERN_LEVENSHTEIN_WEIGHT_RANGE = (0.6, 0.8)
        self.SYLLABLE_PATTERN_COSINE_WEIGHT_RANGE = (0.2, 0.4)
        self.SYLLABLE_PATTERN_NORMALIZE_OPTIONS = [True, False]
        self.SYLLABLE_PATTERN_NORMALIZE_METHOD = ["zscore", "minmax"]
        
        # Line Syllable SSM Parameters
        self.LINE_SYLLABLE_SIMILARITY_METHOD = ["cosine"]  # Only cosine is supported
        self.LINE_SYLLABLE_RATIO_THRESHOLD_RANGE = (0.07, 0.11)
        self.LINE_SYLLABLE_NORMALIZE_OPTIONS = [True, False]
        self.LINE_SYLLABLE_NORMALIZE_METHOD = ["zscore", "minmax"]
        
        # Word2Vec Parameters (NO model/dimension changes)
        self.WORD2VEC_MODE_OPTIONS = ["complete", "summary"]
        self.WORD2VEC_NORMALIZE_OPTIONS = [True, False]
        self.WORD2VEC_SIMILARITY_METRIC = ["cosine", "dot"]  # Only cosine and dot supported
        self.WORD2VEC_HIGH_SIM_THRESHOLD_RANGE = (0.75, 0.85)
        
        # Contextual Embeddings Parameters (NO model/dimension changes)
        self.CONTEXTUAL_MODE_OPTIONS = ["complete", "summary"]
        self.CONTEXTUAL_NORMALIZE_OPTIONS = [True, False]
        self.CONTEXTUAL_SIMILARITY_METRIC = ["cosine", "dot"]  # Only cosine and dot supported
        self.CONTEXTUAL_HIGH_SIM_THRESHOLD_RANGE = (0.65, 0.75)
        
        # ================================================================
        # FEATURE DIMENSIONS (Optional)
        # ================================================================
        self.OPTIMIZE_FEATURE_DIMS = False
        self.SSM_DIMENSION_OPTIONS = [8, 12, 16]


# Global instance (will be initialized in main)
HYPERPARAMS = None

# Session directory for organized results
SESSION_DIR = None


def create_session_directory(study_name: str) -> Path:
    """
    Create a session directory organized by date and study name.
    
    Structure: results/YYYY-MM-DD/HH-MM-SS_study_name/
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    
    # Create the session directory path
    session_dir = Path("results") / date_str / f"{time_str}_{study_name}"
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
        "script_version": "hyperopt_organized_v1.0",
        "subdirectories": {
            "trials": "Individual trial results and configurations",
            "configs": "Best configuration files",
            "logs": "Training and optimization logs"
        }
    }
    
    with open(session_dir / "session_info.json", 'w') as f:
        json.dump(session_info, f, indent=2)
    
    print(f"📁 Session directory created: {session_dir}")
    print(f"   📊 Results will be saved to: {session_dir.absolute()}")
    print(f"   📂 Subdirectories: trials/, configs/, logs/")
    
    return session_dir


def get_optimal_device():
    """Get the best available compute device with proper MPS support."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print(f"💻 Using CPU (consider using GPU for faster training)")
    
    return device


def flatten_config_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested config dictionary for TrainingConfig initialization.
    
    The TrainingConfig expects a flat dictionary with all parameters at the top level.
    This function flattens the nested YAML structure.
    """
    flat_config = {}
    
    # Data section
    if 'data' in config_dict:
        flat_config.update({
            'train_file': config_dict['data'].get('train_file', 'data/train.jsonl'),
            'val_file': config_dict['data'].get('val_file', 'data/val.jsonl'),
            'test_file': config_dict['data'].get('test_file', 'data/test.jsonl')
        })
    
    # Model section
    if 'model' in config_dict:
        model_config = config_dict['model']
        flat_config.update({
            'hidden_dim': model_config.get('hidden_dim', 256),
            'num_layers': model_config.get('num_layers', 1),
            'layer_dropout': model_config.get('layer_dropout', 0.0),
            'num_classes': model_config.get('num_classes', 2),
            'dropout': model_config.get('dropout', 0.25),
            'attention_enabled': model_config.get('attention_enabled', False),
            'attention_type': model_config.get('attention_type', 'self'),
            'attention_heads': model_config.get('attention_heads', 8),
            'attention_dropout': model_config.get('attention_dropout', 0.1),
            'attention_dim': model_config.get('attention_dim', None),
            'positional_encoding': model_config.get('positional_encoding', True),
            'max_seq_length': model_config.get('max_seq_length', 1000),
            'window_size': model_config.get('window_size', 7),
            'boundary_temperature': model_config.get('boundary_temperature', 2.0)
        })
    
    # Training section
    if 'training' in config_dict:
        training_config = config_dict['training']
        flat_config.update({
            'batch_size': training_config.get('batch_size', 32),
            'learning_rate': training_config.get('learning_rate', 0.001),
            'weight_decay': training_config.get('weight_decay', 0.01),
            'max_epochs': training_config.get('max_epochs', 60),
            'patience': training_config.get('patience', 8),
            'gradient_clip_norm': training_config.get('gradient_clip_norm', 1.0),
            'scheduler': training_config.get('scheduler', 'plateau'),
            'min_lr': training_config.get('min_lr', 1e-6),
            'cosine_t_max': training_config.get('cosine_t_max', 60),
            'warmup_epochs': training_config.get('warmup_epochs', 5),
            'lr_factor': training_config.get('lr_factor', 0.5),
            'lr_patience': training_config.get('lr_patience', 10),
            'step_size': training_config.get('step_size', 30),
            'step_gamma': training_config.get('step_gamma', 0.5),
            'cosine_t0': training_config.get('cosine_t0', 10),
            'cosine_t_mult': training_config.get('cosine_t_mult', 2)
        })
    
    # Anti-collapse section
    if 'anti_collapse' in config_dict:
        flat_config.update({
            'weighted_sampling': config_dict['anti_collapse'].get('weighted_sampling', True)
        })
    
    # Emergency monitoring section
    if 'emergency_monitoring' in config_dict:
        emergency_config = config_dict['emergency_monitoring']
        flat_config.update({
            'emergency_monitoring_enabled': emergency_config.get('enabled', True),
            'max_confidence_threshold': emergency_config.get('max_confidence_threshold', 0.95),
            'min_chorus_rate': emergency_config.get('min_chorus_rate', 0.05),
            'max_chorus_rate': emergency_config.get('max_chorus_rate', 0.85),
            'max_conf_over_95_ratio': emergency_config.get('max_conf_over_95_ratio', 0.1),
            'val_overconf_threshold': emergency_config.get('val_overconf_threshold', 0.96),
            'val_f1_collapse_threshold': emergency_config.get('val_f1_collapse_threshold', 0.1),
            'emergency_overconf_threshold': emergency_config.get('emergency_overconf_threshold', 0.98),
            'emergency_conf95_ratio': emergency_config.get('emergency_conf95_ratio', 0.8),
            'skip_batches': emergency_config.get('skip_batches', 30),
            'skip_epochs': emergency_config.get('skip_epochs', 3),
            'print_batch_every': emergency_config.get('print_batch_every', 10)
        })
    
    # Loss configuration (keep as nested for proper handling)
    if 'loss' in config_dict:
        flat_config['loss'] = config_dict['loss']
    
    # ========================================================================
    # FEATURES SECTION - Flatten nested features to flat keys
    # ========================================================================
    if 'features' in config_dict:
        features = config_dict['features']
        
        # Head SSM
        if 'head_ssm' in features:
            head_ssm = features['head_ssm']
            flat_config.update({
                'head_ssm_enabled': head_ssm.get('enabled', True),
                'head_ssm_words': head_ssm.get('head_words', 2),
                'head_ssm_dimension': head_ssm.get('dimension', 12)
            })
        
        # Tail SSM
        if 'tail_ssm' in features:
            tail_ssm = features['tail_ssm']
            flat_config.update({
                'tail_ssm_enabled': tail_ssm.get('enabled', True),
                'tail_ssm_words': tail_ssm.get('tail_words', 2),
                'tail_ssm_dimension': tail_ssm.get('dimension', 12)
            })
        
        # Phonetic SSM
        if 'phonetic_ssm' in features:
            phonetic_ssm = features['phonetic_ssm']
            flat_config.update({
                'phonetic_ssm_enabled': phonetic_ssm.get('enabled', True),
                'phonetic_ssm_mode': phonetic_ssm.get('mode', 'rhyme'),
                'phonetic_ssm_dimension': phonetic_ssm.get('dimension', 12),
                'phonetic_ssm_similarity_method': phonetic_ssm.get('similarity_method', 'binary'),
                'phonetic_ssm_normalize': phonetic_ssm.get('normalize', False),
                'phonetic_ssm_normalize_method': phonetic_ssm.get('normalize_method', 'zscore'),
                'phonetic_ssm_high_sim_threshold': phonetic_ssm.get('high_sim_threshold', 0.31)
            })
        
        # POS SSM
        if 'pos_ssm' in features:
            pos_ssm = features['pos_ssm']
            flat_config.update({
                'pos_ssm_enabled': pos_ssm.get('enabled', True),
                'pos_ssm_tagset': pos_ssm.get('tagset', 'simplified'),
                'pos_ssm_similarity_method': pos_ssm.get('similarity_method', 'combined'),
                'pos_ssm_high_sim_threshold': pos_ssm.get('high_sim_threshold', 0.27),
                'pos_ssm_dimension': pos_ssm.get('dimension', 12)
            })
        
        # String SSM
        if 'string_ssm' in features:
            string_ssm = features['string_ssm']
            flat_config.update({
                'string_ssm_enabled': string_ssm.get('enabled', True),
                'string_ssm_case_sensitive': string_ssm.get('case_sensitive', False),
                'string_ssm_remove_punctuation': string_ssm.get('remove_punctuation', True),
                'string_ssm_similarity_threshold': string_ssm.get('similarity_threshold', 0.055),
                'string_ssm_similarity_method': string_ssm.get('similarity_method', 'word_overlap'),
                'string_ssm_dimension': string_ssm.get('dimension', 12)
            })
        
        # Syllable Pattern SSM
        if 'syllable_pattern_ssm' in features:
            syllable_pattern_ssm = features['syllable_pattern_ssm']
            flat_config.update({
                'syllable_pattern_ssm_enabled': syllable_pattern_ssm.get('enabled', True),
                'syllable_pattern_ssm_similarity_method': syllable_pattern_ssm.get('similarity_method', 'cosine'),
                'syllable_pattern_ssm_levenshtein_weight': syllable_pattern_ssm.get('levenshtein_weight', 0.7),
                'syllable_pattern_ssm_cosine_weight': syllable_pattern_ssm.get('cosine_weight', 0.3),
                'syllable_pattern_ssm_normalize': syllable_pattern_ssm.get('normalize', False),
                'syllable_pattern_ssm_normalize_method': syllable_pattern_ssm.get('normalize_method', 'zscore'),
                'syllable_pattern_ssm_dimension': syllable_pattern_ssm.get('dimension', 12)
            })
        
        # Line Syllable SSM
        if 'line_syllable_ssm' in features:
            line_syllable_ssm = features['line_syllable_ssm']
            flat_config.update({
                'line_syllable_ssm_enabled': line_syllable_ssm.get('enabled', True),
                'line_syllable_ssm_similarity_method': line_syllable_ssm.get('similarity_method', 'cosine'),
                'line_syllable_ssm_ratio_threshold': line_syllable_ssm.get('ratio_threshold', 0.09),
                'line_syllable_ssm_normalize': line_syllable_ssm.get('normalize', False),
                'line_syllable_ssm_normalize_method': line_syllable_ssm.get('normalize_method', 'minmax'),
                'line_syllable_ssm_dimension': line_syllable_ssm.get('dimension', 12)
            })
    
    # ========================================================================
    # EMBEDDINGS SECTION - Flatten nested embeddings to flat keys
    # ========================================================================
    if 'embeddings' in config_dict:
        embeddings = config_dict['embeddings']
        
        # Word2Vec Embeddings
        if 'word2vec' in embeddings:
            word2vec = embeddings['word2vec']
            flat_config.update({
                'word2vec_enabled': word2vec.get('enabled', True),
                'word2vec_model': word2vec.get('model', 'word2vec-google-news-300'),
                'word2vec_mode': word2vec.get('mode', 'complete'),
                'word2vec_normalize': word2vec.get('normalize', True),
                'word2vec_similarity_metric': word2vec.get('similarity_metric', 'cosine'),
                'word2vec_high_sim_threshold': word2vec.get('high_sim_threshold', 0.82)
            })
        
        # Contextual Embeddings
        if 'contextual' in embeddings:
            contextual = embeddings['contextual']
            flat_config.update({
                'contextual_enabled': contextual.get('enabled', True),
                'contextual_model': contextual.get('model', 'all-MiniLM-L6-v2'),
                'contextual_mode': contextual.get('mode', 'complete'),
                'contextual_normalize': contextual.get('normalize', True),
                'contextual_similarity_metric': contextual.get('similarity_metric', 'cosine'),
                'contextual_high_sim_threshold': contextual.get('high_sim_threshold', 0.72)
            })
    
    return flat_config


def extract_feature_config(config: TrainingConfig) -> Dict[str, Any]:
    """Extract feature configuration for FeatureExtractor."""
    return {
        'head_ssm': {
            'enabled': getattr(config, 'head_ssm_enabled', True),
            'head_words': getattr(config, 'head_ssm_words', 2),
            'output_dim': getattr(config, 'head_ssm_dimension', 12)
        },
        'tail_ssm': {
            'enabled': getattr(config, 'tail_ssm_enabled', True),
            'tail_words': getattr(config, 'tail_ssm_words', 2),
            'output_dim': getattr(config, 'tail_ssm_dimension', 12)
        },
        'phonetic_ssm': {
            'enabled': getattr(config, 'phonetic_ssm_enabled', True),
            'mode': getattr(config, 'phonetic_ssm_mode', 'rhyme'),
            'output_dim': getattr(config, 'phonetic_ssm_dimension', 12),
            'similarity_method': getattr(config, 'phonetic_ssm_similarity_method', 'binary'),
            'normalize': getattr(config, 'phonetic_ssm_normalize', False),
            'normalize_method': getattr(config, 'phonetic_ssm_normalize_method', 'none'),
            'high_sim_threshold': getattr(config, 'phonetic_ssm_high_sim_threshold', 0.31)
        },
        'pos_ssm': {
            'enabled': getattr(config, 'pos_ssm_enabled', True),
            'tagset': getattr(config, 'pos_ssm_tagset', 'simplified'),
            'similarity_method': getattr(config, 'pos_ssm_similarity_method', 'combined'),
            'high_sim_threshold': getattr(config, 'pos_ssm_high_sim_threshold', 0.27),
            'output_dim': getattr(config, 'pos_ssm_dimension', 12)
        },
        'string_ssm': {
            'enabled': getattr(config, 'string_ssm_enabled', True),
            'case_sensitive': getattr(config, 'string_ssm_case_sensitive', False),
            'remove_punctuation': getattr(config, 'string_ssm_remove_punctuation', True),
            'similarity_threshold': getattr(config, 'string_ssm_similarity_threshold', 0.055),
            'similarity_method': getattr(config, 'string_ssm_similarity_method', 'word_overlap'),
            'output_dim': getattr(config, 'string_ssm_dimension', 12)
        },
        'syllable_pattern_ssm': {
            'enabled': getattr(config, 'syllable_pattern_ssm_enabled', True),
            'similarity_method': getattr(config, 'syllable_pattern_ssm_similarity_method', 'cosine'),
            'levenshtein_weight': getattr(config, 'syllable_pattern_ssm_levenshtein_weight', 0.5),
            'cosine_weight': getattr(config, 'syllable_pattern_ssm_cosine_weight', 0.5),
            'normalize': getattr(config, 'syllable_pattern_ssm_normalize', False),
            'normalize_method': getattr(config, 'syllable_pattern_ssm_normalize_method', 'none'),
            'dimension': getattr(config, 'syllable_pattern_ssm_dimension', 12)
        },
        'line_syllable_ssm': {
            'enabled': getattr(config, 'line_syllable_ssm_enabled', True),
            'similarity_method': getattr(config, 'line_syllable_ssm_similarity_method', 'cosine'),
            'ratio_threshold': getattr(config, 'line_syllable_ssm_ratio_threshold', 0.09),
            'normalize': getattr(config, 'line_syllable_ssm_normalize', False),
            'normalize_method': getattr(config, 'line_syllable_ssm_normalize_method', 'none'),
            'dimension': getattr(config, 'line_syllable_ssm_dimension', 12)
        },
        'word2vec_enabled': getattr(config, 'word2vec_enabled', True),
        'word2vec_model': getattr(config, 'word2vec_model', 'word2vec-google-news-300'),
        'word2vec_mode': getattr(config, 'word2vec_mode', 'complete'),
        'word2vec_normalize': getattr(config, 'word2vec_normalize', True),
        'word2vec_similarity_metric': getattr(config, 'word2vec_similarity_metric', 'cosine'),
        'word2vec_high_sim_threshold': getattr(config, 'word2vec_high_sim_threshold', 0.82),
        'contextual_enabled': getattr(config, 'contextual_enabled', True),
        'contextual_model': getattr(config, 'contextual_model', 'all-MiniLM-L6-v2'),
        'contextual_mode': getattr(config, 'contextual_mode', 'complete'),
        'contextual_normalize': getattr(config, 'contextual_normalize', True),
        'contextual_similarity_metric': getattr(config, 'contextual_similarity_metric', 'cosine'),
        'contextual_high_sim_threshold': getattr(config, 'contextual_high_sim_threshold', 0.72)
    }


def create_trial_config(trial, base_config_path=None, hyperparams=None):
    """
    Create trial config with LOCAL configuration only.
    NO external config file loading - build everything locally.
    """
    
    # Use global HYPERPARAMS if not provided
    if hyperparams is None:
        hyperparams = HYPERPARAMS
    
    if hyperparams is None:
        raise ValueError("No hyperparameters configuration provided. Either pass hyperparams or initialize global HYPERPARAMS.")
    
    print(f"   🎲 Sampling parameters for trial {trial.number}...")
    
    # ========================================================================
    # MODEL ARCHITECTURE - COMPATIBILITY ENFORCED
    # ========================================================================
    hidden_dim = trial.suggest_categorical('hidden_dim', hyperparams.HIDDEN_DIM)
    num_layers = trial.suggest_categorical('num_layers', hyperparams.NUM_LAYERS)
    dropout = trial.suggest_float('dropout', *hyperparams.DROPOUT_RANGE)
    
    # Layer dropout only if num_layers > 1
    if num_layers > 1:
        layer_dropout = trial.suggest_float('layer_dropout', *hyperparams.LAYER_DROPOUT_RANGE)
    else:
        layer_dropout = 0.0
    
    # COMPATIBILITY CHECK: attention_dim should match hidden_dim for optimal performance
    attention_heads = trial.suggest_categorical('attention_heads', hyperparams.ATTENTION_HEADS)
    attention_dim = hidden_dim  # FORCE COMPATIBILITY
    attention_dropout = trial.suggest_float('attention_dropout', *hyperparams.ATTENTION_DROPOUT_RANGE)
    
    # Validate heads compatibility
    if hidden_dim % attention_heads != 0:
        # Force compatible heads
        compatible_heads = [h for h in hyperparams.ATTENTION_HEADS if hidden_dim % h == 0]
        if compatible_heads:
            attention_heads = trial.suggest_categorical('attention_heads_compat', compatible_heads)
        else:
            attention_heads = 4  # Safe fallback
    
    # ========================================================================
    # TRAINING PARAMETERS - EPOCHS = COSINE_T_MAX + OOM PREVENTION
    # ========================================================================
    max_epochs = trial.suggest_categorical('max_epochs', hyperparams.MAX_EPOCHS_OPTIONS)
    cosine_t_max = max_epochs  # FORCE EQUALITY as requested
    
    # OOM Prevention: Reduce batch size for large models
    if hidden_dim >= 512 and num_layers >= 2:
        batch_size = trial.suggest_categorical('batch_size_large', [16, 32])  # Smaller batches for large models
        print(f"   ⚠️  Large model detected ({hidden_dim}D, {num_layers}L) - limiting batch size to {[16, 32]}")
    else:
        batch_size = trial.suggest_categorical('batch_size', hyperparams.BATCH_SIZE)
    learning_rate = trial.suggest_float('learning_rate', *hyperparams.LEARNING_RATE_RANGE, log=True)
    weight_decay = trial.suggest_float('weight_decay', *hyperparams.WEIGHT_DECAY_RANGE, log=True)
    scheduler = trial.suggest_categorical('scheduler', hyperparams.SCHEDULER_OPTIONS)
    
    # Additional training parameters
    patience = trial.suggest_categorical('patience', hyperparams.PATIENCE_OPTIONS)
    min_delta = trial.suggest_float('min_delta', *hyperparams.MIN_DELTA_RANGE)
    gradient_clip_norm = trial.suggest_float('gradient_clip_norm', *hyperparams.GRADIENT_CLIP_RANGE)
    
    # VALIDATION STRATEGY OPTIMIZATION
    validation_strategy = trial.suggest_categorical('validation_strategy', hyperparams.VALIDATION_STRATEGY_OPTIONS)
    
    # ========================================================================
    # HEAD/TAIL WORD VARIATIONS (as requested)
    # ========================================================================
    head_tail_words = trial.suggest_categorical('head_tail_words', hyperparams.HEAD_TAIL_WORDS_OPTIONS)
    
    # ========================================================================
    # BOUNDARY-AWARE LOSS PARAMETERS
    # ========================================================================
    boundary_weight = trial.suggest_float('boundary_weight', *hyperparams.BOUNDARY_WEIGHT_RANGE)
    label_smoothing = trial.suggest_float('label_smoothing', *hyperparams.LABEL_SMOOTHING_RANGE)
    segment_consistency_lambda = trial.suggest_float('segment_consistency_lambda', *hyperparams.SEGMENT_CONSISTENCY_RANGE)
    conf_penalty_lambda = trial.suggest_float('conf_penalty_lambda', *hyperparams.CONF_PENALTY_RANGE)
    conf_threshold = trial.suggest_float('conf_threshold', *hyperparams.CONF_THRESHOLD_RANGE)
    entropy_lambda = trial.suggest_float('entropy_lambda', *hyperparams.ENTROPY_LAMBDA_RANGE)
    
    # ========================================================================
    # EMERGENCY MONITORING PARAMETERS
    # ========================================================================
    max_confidence_threshold = trial.suggest_float('max_confidence_threshold', *hyperparams.MAX_CONFIDENCE_THRESHOLD_RANGE)
    min_chorus_rate = trial.suggest_float('min_chorus_rate', *hyperparams.MIN_CHORUS_RATE_RANGE)
    max_chorus_rate = trial.suggest_float('max_chorus_rate', *hyperparams.MAX_CHORUS_RATE_RANGE)
    max_conf_over_95_ratio = trial.suggest_float('max_conf_over_95_ratio', *hyperparams.MAX_CONF_OVER_95_RATIO_RANGE)
    
    # ========================================================================
    # FEATURE TOGGLES (ON/OFF as requested)
    # ========================================================================
    feature_toggles = {}
    if hyperparams.OPTIMIZE_FEATURE_TOGGLES:
        for feature_name, options in hyperparams.FEATURE_TOGGLES.items():
            feature_toggles[feature_name] = trial.suggest_categorical(feature_name, options)
    else:
        # Default all features to enabled
        for feature_name in hyperparams.FEATURE_TOGGLES.keys():
            feature_toggles[feature_name] = True
    
    # ========================================================================
    # FEATURE-SPECIFIC PARAMETERS (NO EMBEDDING MODELS/DIMENSIONS)
    # ========================================================================
    
    # Phonetic SSM Parameters
    phonetic_mode = trial.suggest_categorical('phonetic_mode', hyperparams.PHONETIC_MODE_OPTIONS)
    phonetic_similarity_method = trial.suggest_categorical('phonetic_similarity_method', hyperparams.PHONETIC_SIMILARITY_METHOD)
    phonetic_normalize = trial.suggest_categorical('phonetic_normalize', hyperparams.PHONETIC_NORMALIZE_OPTIONS)
    phonetic_normalize_method = trial.suggest_categorical('phonetic_normalize_method', hyperparams.PHONETIC_NORMALIZE_METHOD)
    phonetic_high_sim_threshold = trial.suggest_float('phonetic_high_sim_threshold', *hyperparams.PHONETIC_HIGH_SIM_THRESHOLD_RANGE)
    
    # POS SSM Parameters
    pos_tagset = trial.suggest_categorical('pos_tagset', hyperparams.POS_TAGSET_OPTIONS)
    pos_similarity_method = trial.suggest_categorical('pos_similarity_method', hyperparams.POS_SIMILARITY_METHOD)
    pos_high_sim_threshold = trial.suggest_float('pos_high_sim_threshold', *hyperparams.POS_HIGH_SIM_THRESHOLD_RANGE)
    
    # String SSM Parameters
    string_case_sensitive = trial.suggest_categorical('string_case_sensitive', hyperparams.STRING_CASE_SENSITIVE_OPTIONS)
    string_remove_punctuation = trial.suggest_categorical('string_remove_punctuation', hyperparams.STRING_REMOVE_PUNCTUATION_OPTIONS)
    string_similarity_threshold = trial.suggest_float('string_similarity_threshold', *hyperparams.STRING_SIMILARITY_THRESHOLD_RANGE)
    string_similarity_method = trial.suggest_categorical('string_similarity_method', hyperparams.STRING_SIMILARITY_METHOD)
    
    # Syllable Pattern SSM Parameters
    syllable_pattern_similarity_method = trial.suggest_categorical('syllable_pattern_similarity_method', hyperparams.SYLLABLE_PATTERN_SIMILARITY_METHOD)
    syllable_pattern_levenshtein_weight = trial.suggest_float('syllable_pattern_levenshtein_weight', *hyperparams.SYLLABLE_PATTERN_LEVENSHTEIN_WEIGHT_RANGE)
    syllable_pattern_cosine_weight = trial.suggest_float('syllable_pattern_cosine_weight', *hyperparams.SYLLABLE_PATTERN_COSINE_WEIGHT_RANGE)
    
    # Normalize weights to sum to 1.0
    total_weight = syllable_pattern_levenshtein_weight + syllable_pattern_cosine_weight
    if total_weight > 0:
        syllable_pattern_levenshtein_weight = syllable_pattern_levenshtein_weight / total_weight
        syllable_pattern_cosine_weight = syllable_pattern_cosine_weight / total_weight
    else:
        syllable_pattern_levenshtein_weight = 0.7
        syllable_pattern_cosine_weight = 0.3
    
    syllable_pattern_normalize = trial.suggest_categorical('syllable_pattern_normalize', hyperparams.SYLLABLE_PATTERN_NORMALIZE_OPTIONS)
    syllable_pattern_normalize_method = trial.suggest_categorical('syllable_pattern_normalize_method', hyperparams.SYLLABLE_PATTERN_NORMALIZE_METHOD)
    
    # Line Syllable SSM Parameters
    line_syllable_similarity_method = trial.suggest_categorical('line_syllable_similarity_method', hyperparams.LINE_SYLLABLE_SIMILARITY_METHOD)
    line_syllable_ratio_threshold = trial.suggest_float('line_syllable_ratio_threshold', *hyperparams.LINE_SYLLABLE_RATIO_THRESHOLD_RANGE)
    line_syllable_normalize = trial.suggest_categorical('line_syllable_normalize', hyperparams.LINE_SYLLABLE_NORMALIZE_OPTIONS)
    line_syllable_normalize_method = trial.suggest_categorical('line_syllable_normalize_method', hyperparams.LINE_SYLLABLE_NORMALIZE_METHOD)
    
    # Embedding Parameters (NO model/dimension changes - only mode and similarity settings)
    word2vec_mode = trial.suggest_categorical('word2vec_mode', hyperparams.WORD2VEC_MODE_OPTIONS)
    word2vec_normalize = trial.suggest_categorical('word2vec_normalize', hyperparams.WORD2VEC_NORMALIZE_OPTIONS)
    word2vec_similarity_metric = trial.suggest_categorical('word2vec_similarity_metric', hyperparams.WORD2VEC_SIMILARITY_METRIC)
    word2vec_high_sim_threshold = trial.suggest_float('word2vec_high_sim_threshold', *hyperparams.WORD2VEC_HIGH_SIM_THRESHOLD_RANGE)
    
    contextual_mode = trial.suggest_categorical('contextual_mode', hyperparams.CONTEXTUAL_MODE_OPTIONS)
    contextual_normalize = trial.suggest_categorical('contextual_normalize', hyperparams.CONTEXTUAL_NORMALIZE_OPTIONS)
    contextual_similarity_metric = trial.suggest_categorical('contextual_similarity_metric', hyperparams.CONTEXTUAL_SIMILARITY_METRIC)
    contextual_high_sim_threshold = trial.suggest_float('contextual_high_sim_threshold', *hyperparams.CONTEXTUAL_HIGH_SIM_THRESHOLD_RANGE)
    
    # ========================================================================
    # BUILD LOCAL CONFIG DICTIONARY (NO EXTERNAL LOADING)
    # ========================================================================
    
    config_dict = {
        # Model Configuration
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'layer_dropout': layer_dropout,
        'num_classes': 2,
        'dropout': dropout,
        'attention_enabled': True,  # Always enabled
        'attention_type': 'boundary_aware',
        'attention_heads': attention_heads,
        'attention_dropout': attention_dropout,
        'attention_dim': attention_dim,
        'positional_encoding': True,
        'max_seq_length': 1000,
        'window_size': 7,
        'boundary_temperature': 2.0,
        
        # Training Configuration - EPOCHS = COSINE_T_MAX
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'max_epochs': max_epochs,
        'patience': patience,
        'gradient_clip_norm': gradient_clip_norm,
        'scheduler': scheduler,
        'min_lr': 1e-6,
        'cosine_t_max': cosine_t_max,  # MATCHES max_epochs
        'warmup_epochs': 5,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'step_size': 30,
        'step_gamma': 0.5,
        'cosine_t0': 10,
        'cosine_t_mult': 2,
        'min_delta': min_delta,
        
        # Data Configuration  
        'train_file': 'data/train.jsonl',
        'val_file': 'data/val.jsonl',
        'test_file': 'data/test.jsonl',
        
        # Anti-collapse
        'weighted_sampling': True,
        
        # Emergency monitoring
        'emergency_monitoring_enabled': True,
        'max_confidence_threshold': max_confidence_threshold,
        'min_chorus_rate': min_chorus_rate,
        'max_chorus_rate': max_chorus_rate,
        'max_conf_over_95_ratio': max_conf_over_95_ratio,
        'val_overconf_threshold': 0.96,
        'val_f1_collapse_threshold': 0.1,
        
        # Loss Configuration (as nested dict for TrainingConfig)
        'loss': {
            'loss_type': 'boundary_aware',
            'boundary_weight': boundary_weight,
            'label_smoothing': label_smoothing,
            'segment_consistency_lambda': segment_consistency_lambda,
            'conf_penalty_lambda': conf_penalty_lambda,
            'conf_threshold': conf_threshold,
            'entropy_lambda': entropy_lambda,
        },
        
        # CALIBRATION - ALWAYS ENABLED (as requested)
        'calibration_enabled': True,
        'calibration_methods': ['temperature', 'platt'],  # Both methods
        
        # VALIDATION STRATEGY
        'validation_strategy': validation_strategy,
        
        # Features Configuration - HEAD/TAIL WORDS VARIATION
        'head_ssm_enabled': feature_toggles.get('head_ssm_enabled', True),
        'head_ssm_words': head_tail_words,  # 2-4 words as requested
        'head_ssm_dimension': 12,  # Static dimension
        
        'tail_ssm_enabled': feature_toggles.get('tail_ssm_enabled', True),
        'tail_ssm_words': head_tail_words,  # Same as head_ssm
        'tail_ssm_dimension': 12,  # Static dimension
        
        'phonetic_ssm_enabled': feature_toggles.get('phonetic_ssm_enabled', True),
        'phonetic_ssm_mode': phonetic_mode,
        'phonetic_ssm_dimension': 12,  # Static dimension
        'phonetic_ssm_similarity_method': phonetic_similarity_method,
        'phonetic_ssm_normalize': phonetic_normalize,
        'phonetic_ssm_normalize_method': phonetic_normalize_method,
        'phonetic_ssm_high_sim_threshold': phonetic_high_sim_threshold,
        
        'pos_ssm_enabled': feature_toggles.get('pos_ssm_enabled', True),
        'pos_ssm_tagset': pos_tagset,
        'pos_ssm_similarity_method': pos_similarity_method,
        'pos_ssm_high_sim_threshold': pos_high_sim_threshold,
        'pos_ssm_dimension': 12,  # Static dimension
        
        'string_ssm_enabled': feature_toggles.get('string_ssm_enabled', True),
        'string_ssm_case_sensitive': string_case_sensitive,
        'string_ssm_remove_punctuation': string_remove_punctuation,
        'string_ssm_similarity_threshold': string_similarity_threshold,
        'string_ssm_similarity_method': string_similarity_method,
        'string_ssm_dimension': 12,  # Static dimension
        
        'syllable_pattern_ssm_enabled': feature_toggles.get('syllable_pattern_ssm_enabled', True),
        'syllable_pattern_ssm_similarity_method': syllable_pattern_similarity_method,
        'syllable_pattern_ssm_levenshtein_weight': syllable_pattern_levenshtein_weight,
        'syllable_pattern_ssm_cosine_weight': syllable_pattern_cosine_weight,
        'syllable_pattern_ssm_normalize': syllable_pattern_normalize,
        'syllable_pattern_ssm_normalize_method': syllable_pattern_normalize_method,
        'syllable_pattern_ssm_dimension': 12,  # Static dimension
        
        'line_syllable_ssm_enabled': feature_toggles.get('line_syllable_ssm_enabled', True),
        'line_syllable_ssm_similarity_method': line_syllable_similarity_method,
        'line_syllable_ssm_ratio_threshold': line_syllable_ratio_threshold,
        'line_syllable_ssm_normalize': line_syllable_normalize,
        'line_syllable_ssm_normalize_method': line_syllable_normalize_method,
        'line_syllable_ssm_dimension': 12,  # Static dimension
        
        # Embeddings Configuration (NO model/dimension changes)
        'word2vec_enabled': feature_toggles.get('word2vec_enabled', True),
        'word2vec_model': 'word2vec-google-news-300',  # STATIC model
        'word2vec_mode': word2vec_mode,
        'word2vec_normalize': word2vec_normalize,
        'word2vec_similarity_metric': word2vec_similarity_metric,
        'word2vec_high_sim_threshold': word2vec_high_sim_threshold,
        
        'contextual_enabled': feature_toggles.get('contextual_enabled', True),
        'contextual_model': 'all-MiniLM-L6-v2',  # STATIC model
        'contextual_mode': contextual_mode,
        'contextual_normalize': contextual_normalize,
        'contextual_similarity_metric': contextual_similarity_metric,
        'contextual_high_sim_threshold': contextual_high_sim_threshold,
    }
    
    # ========================================================================
    # COMPATIBILITY VALIDATION (AUTO-FIX, NO RAISING)
    # ========================================================================
    
    # Validate attention configuration
    if config_dict['attention_enabled']:
        if config_dict['hidden_dim'] % config_dict['attention_heads'] != 0:
            print(f"⚠️  Auto-fixing: Hidden dim {config_dict['hidden_dim']} not divisible by attention heads {config_dict['attention_heads']}")
            # Find compatible heads
            compatible_heads = [h for h in hyperparams.ATTENTION_HEADS if config_dict['hidden_dim'] % h == 0]
            if compatible_heads:
                config_dict['attention_heads'] = compatible_heads[0]
                print(f"   → Fixed to {config_dict['attention_heads']} heads")
            else:
                config_dict['attention_heads'] = 4  # Safe fallback
                print(f"   → Fallback to {config_dict['attention_heads']} heads")
        
        if config_dict['attention_dim'] != config_dict['hidden_dim']:
            print(f"⚠️  Auto-fixing: Attention dim {config_dict['attention_dim']} != hidden dim {config_dict['hidden_dim']}")
            config_dict['attention_dim'] = config_dict['hidden_dim']
            print(f"   → Fixed to match hidden_dim: {config_dict['attention_dim']}")
    
    # Validate epochs = cosine_t_max
    if config_dict['scheduler'] == 'cosine' and config_dict['max_epochs'] != config_dict['cosine_t_max']:
        print(f"⚠️  Auto-fixing: max_epochs {config_dict['max_epochs']} != cosine_t_max {config_dict['cosine_t_max']}")
        config_dict['cosine_t_max'] = config_dict['max_epochs']
        print(f"   → Fixed cosine_t_max to match: {config_dict['cosine_t_max']}")
    
    # Validate calibration is enabled
    if not config_dict['calibration_enabled']:
        print("⚠️  Auto-fixing: Calibration was disabled, forcing enabled as requested")
        config_dict['calibration_enabled'] = True
        print("   → Calibration now enabled")
    
    print(f"   ✅ Trial {trial.number} config created with auto-fixes applied")
    
    return config_dict, config_dict


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def objective(trial):
    """
    Objective function for hyperparameter optimization.
    NO external config loading - everything is local.
    """
    from optuna.exceptions import TrialPruned
    
    try:
        print(f"\n🔬 Starting Trial {trial.number}")
        
        # Create trial config (completely local, no external files)
        config_dict, _ = create_trial_config(trial, hyperparams=HYPERPARAMS)
        
        # Create training config object directly from local config dict
        config = TrainingConfig(**config_dict)
        
        # Set device and other essentials
        config.device = get_optimal_device()
        config.experiment_name = f"hyperopt_trial_{trial.number:03d}"
        config.seed = 42  # Fixed seed for reproducibility
        
        # Set all seeds for reproducibility
        set_all_seeds(config.seed)
        
        print(f"   📋 Trial {trial.number} Configuration:")
        print(f"      Model: {config.hidden_dim}D hidden, {config.num_layers} layers, {config.attention_heads} heads")
        print(f"      Training: lr={config.learning_rate:.2e}, batch={config.batch_size}, {config.scheduler} scheduler")
        print(f"      Loss: boundary_wt={config.loss['boundary_weight']:.2f}, label_smooth={config.loss['label_smoothing']:.2f}")
        print(f"      Features: head/tail={config.head_ssm_words} words, validation={config.validation_strategy}")
        
        # Show feature toggles
        if HYPERPARAMS.OPTIMIZE_FEATURE_TOGGLES:
            enabled_features = []
            disabled_features = []
            for feature_name in HYPERPARAMS.FEATURE_TOGGLES.keys():
                if trial.params.get(feature_name, True):
                    feature_short = feature_name.replace('_enabled', '').replace('_ssm', '')
                    enabled_features.append(feature_short)
                else:
                    feature_short = feature_name.replace('_enabled', '').replace('_ssm', '')
                    disabled_features.append(feature_short)
            
            if enabled_features:
                print(f"      Features ON: {', '.join(enabled_features)}")
            if disabled_features:
                print(f"      Features OFF: {', '.join(disabled_features)}")
        else:
            print(f"      Features: All enabled (not optimizing feature selection)")
        
        # Create trial directory in session directory
        trial_dir = SESSION_DIR / "trials" / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trial config for reference
        with open(trial_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # ====================================================================
        # TRAINING EXECUTION
        # ====================================================================
        
        # Create feature extractor
        feature_config = extract_feature_config(config)
        feature_extractor = FeatureExtractor(feature_config)
        
        # Get feature dimension
        feature_dim = feature_extractor.get_feature_dimension()
        
        # Setup model and training components
        train_loader, val_loader, test_loader, train_dataset = setup_data_loaders(config, feature_extractor)
        model, loss_function, optimizer = setup_model_and_training(
            config, train_dataset, config.device, feature_dim
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            device=config.device,
            config=config,
            output_dir=trial_dir
        )
        
        # Set data loaders separately
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        trainer.test_loader = test_loader
        
        # Add trial for per-epoch pruning
        trainer.optuna_trial = trial  # Add trial for per-epoch pruning
        
        # Train model (pass data loaders as arguments)
        print(f"   🚀 Training trial {trial.number}...")
        model, train_metrics = trainer.train(train_loader, val_loader)
        
        # ====================================================================
        # EVALUATION - BOUNDARY F1 AS PRIMARY METRIC
        # ====================================================================
        
        print(f"   📊 Evaluating trial {trial.number}...")
        val_metrics = trainer.evaluate(val_loader)
        
        # Extract boundary F1 (our primary optimization target)
        boundary_f1 = val_metrics.get('boundary_f1', 0.0)
        
        # Log comprehensive results
        print(f"   📈 Trial {trial.number} Results:")
        print(f"      Boundary F1: {boundary_f1:.4f}")
        print(f"      Validation Strategy: {config.validation_strategy}")
        print(f"      Overall Accuracy: {val_metrics.get('accuracy', 0.0):.4f}")
        print(f"      Weighted F1: {val_metrics.get('weighted_f1', 0.0):.4f}")
        
        print(f"✅ Trial {trial.number} completed: boundary_f1 = {boundary_f1:.4f}")
        
        # Final report for pruning
        trial.report(boundary_f1, step=999)  # Final step
        
        # Save trial results
        trial_results = {
            'trial_number': trial.number,
            'boundary_f1': boundary_f1,
            'validation_strategy': config.validation_strategy,
            'all_metrics': val_metrics,
            'parameters': trial.params,
            'config': config_dict
        }
        
        with open(trial_dir / "results.json", 'w') as f:
            json.dump(trial_results, f, indent=2)
        
        return boundary_f1
        
    except TrialPruned:
        print(f"🔪 Trial {trial.number} pruned.")
        raise
        
    except Exception as e:
        print(f"❌ Trial {trial.number} failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 0.0  # Return worst possible score


def export_results(study, session_dir: Path):
    """Export study results to JSON format in the session directory."""
    results = {
        'study_name': study.study_name,
        'best_score': study.best_value,
        'best_params': study.best_params,
        'optimization_target': 'boundary_f1',
        'n_trials': len(study.trials),
        'session_directory': str(session_dir),
        'export_timestamp': datetime.now().isoformat(),
        'all_trials': []
    }
    
    for trial in study.trials:
        trial_data = {
            'trial': trial.number,
            'boundary_f1': trial.value,
            'params': trial.params,
            'status': str(trial.state),
            'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None
        }
        results['all_trials'].append(trial_data)
    
    # Save in session directory
    results_file = session_dir / 'hyperopt_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results exported to {results_file}")


def export_to_sql(study, session_dir: Path):
    """Export study results to SQL database for dashboard analysis in the session directory."""
    
    # Create SQL database in the session directory
    db_path = session_dir / 'hyperopt_results.db'
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS studies (
            study_name TEXT PRIMARY KEY,
            best_score REAL,
            optimization_target TEXT,
            n_trials INTEGER,
            created_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER,
            study_name TEXT,
            boundary_f1 REAL,
            status TEXT,
            duration_seconds REAL,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (study_name) REFERENCES studies (study_name)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trial_parameters (
            trial_id INTEGER,
            study_name TEXT,
            param_name TEXT,
            param_value TEXT,
            param_type TEXT,
            FOREIGN KEY (study_name) REFERENCES studies (study_name)
        )
    ''')
    
    # Insert study information
    cursor.execute('''
        INSERT OR REPLACE INTO studies 
        (study_name, best_score, optimization_target, n_trials, created_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        study.study_name,
        study.best_value,
        'boundary_f1',
        len(study.trials),
        datetime.now().isoformat(),
        datetime.now().isoformat()
    ))
    
    # Insert trial information
    for trial in study.trials:
        # Trial data
        cursor.execute('''
            INSERT OR REPLACE INTO trials 
            (trial_id, study_name, boundary_f1, status, duration_seconds, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            trial.number,
            study.study_name,
            trial.value,
            str(trial.state),
            (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None,
            trial.datetime_start.isoformat() if trial.datetime_start else None,
            trial.datetime_complete.isoformat() if trial.datetime_complete else None
        ))
        
        # Trial parameters
        for param_name, param_value in trial.params.items():
            param_type = type(param_value).__name__
            cursor.execute('''
                INSERT OR REPLACE INTO trial_parameters 
                (trial_id, study_name, param_name, param_value, param_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                trial.number,
                study.study_name,
                param_name,
                str(param_value),
                param_type
            ))
    
    # Create useful views for dashboard analysis
    cursor.execute('''
        CREATE VIEW IF NOT EXISTS trial_summary AS
        SELECT 
            t.trial_id,
            t.study_name,
            t.boundary_f1,
            t.status,
            t.duration_seconds,
            t.started_at,
            t.completed_at,
            CASE 
                WHEN t.boundary_f1 = (SELECT MAX(boundary_f1) FROM trials WHERE study_name = t.study_name) 
                THEN 'best' 
                ELSE 'regular' 
            END as trial_rank
        FROM trials t
    ''')
    
    cursor.execute('''
        CREATE VIEW IF NOT EXISTS param_analysis AS
        SELECT 
            tp.param_name,
            tp.param_value,
            tp.param_type,
            COUNT(*) as frequency,
            AVG(t.boundary_f1) as avg_boundary_f1,
            MAX(t.boundary_f1) as max_boundary_f1,
            MIN(t.boundary_f1) as min_boundary_f1
        FROM trial_parameters tp
        JOIN trials t ON tp.trial_id = t.trial_id AND tp.study_name = t.study_name
        GROUP BY tp.param_name, tp.param_value
        ORDER BY tp.param_name, avg_boundary_f1 DESC
    ''')
    
    # Commit and close
    conn.commit()
    conn.close()
    
    print(f"📊 SQL database exported to {db_path}")
    print(f"   Tables: studies, trials, trial_parameters")
    print(f"   Views: trial_summary, param_analysis")
    print(f"   Ready for dashboard analysis!")


def create_best_config_local(study, session_dir: Path):
    """Create best configuration YAML file using LOCAL config only (no external file loading)."""
    
    best_params = study.best_params
    
    # Build the best config dict from local configuration
    best_config = {
        'experiment_name': f'best_hyperopt_config_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        
        # Model Configuration
        'model': {
            'hidden_dim': best_params.get('hidden_dim', 256),
            'num_layers': best_params.get('num_layers', 2),
            'dropout': best_params.get('dropout', 0.25),
            'layer_dropout': best_params.get('layer_dropout', 0.0),
            'num_classes': 2,
            'attention_enabled': True,
            'attention_type': 'boundary_aware',
            'attention_heads': best_params.get('attention_heads', 8),
            'attention_dropout': best_params.get('attention_dropout', 0.2),
            'attention_dim': best_params.get('hidden_dim', 256),  # Match hidden_dim
            'positional_encoding': True,
            'max_seq_length': 1000,
            'window_size': 7,
            'boundary_temperature': 2.0
        },
        
        # Training Configuration
        'training': {
            'batch_size': best_params.get('batch_size', 32),
            'learning_rate': best_params.get('learning_rate', 0.0005),
            'weight_decay': best_params.get('weight_decay', 0.015),
            'max_epochs': best_params.get('max_epochs', 45),
            'patience': best_params.get('patience', 8),
            'gradient_clip_norm': best_params.get('gradient_clip_norm', 1.0),
            'scheduler': best_params.get('scheduler', 'cosine'),
            'min_lr': 1e-6,
            'cosine_t_max': best_params.get('max_epochs', 45),  # Match max_epochs
            'min_delta': best_params.get('min_delta', 0.005),
            'warmup_epochs': 5,
            'lr_factor': 0.5,
            'lr_patience': 10,
            'step_size': 30,
            'step_gamma': 0.5,
            'cosine_t0': 10,
            'cosine_t_mult': 2
        },
        
        # Data Configuration
        'data': {
            'train_file': 'data/train.jsonl',
            'val_file': 'data/val.jsonl',
            'test_file': 'data/test.jsonl'
        },
        
        # Loss Configuration
        'loss': {
            'type': 'boundary_aware',
            'boundary_weight': best_params.get('boundary_weight', 1.8),
            'label_smoothing': best_params.get('label_smoothing', 0.20),
            'segment_consistency_lambda': best_params.get('segment_consistency_lambda', 0.02),
            'conf_penalty_lambda': best_params.get('conf_penalty_lambda', 0.010),
            'conf_threshold': best_params.get('conf_threshold', 0.93),
            'entropy_lambda': best_params.get('entropy_lambda', 0.04)
        },
        
        # Anti-collapse Configuration
        'anti_collapse': {
            'weighted_sampling': True
        },
        
        # Emergency Monitoring
        'emergency_monitoring': {
            'enabled': True,
            'max_confidence_threshold': best_params.get('max_confidence_threshold', 0.95),
            'min_chorus_rate': best_params.get('min_chorus_rate', 0.05),
            'max_chorus_rate': best_params.get('max_chorus_rate', 0.85),
            'max_conf_over_95_ratio': best_params.get('max_conf_over_95_ratio', 0.1),
            'val_overconf_threshold': 0.96,
            'val_f1_collapse_threshold': 0.1
        },
        
        # Calibration (ALWAYS ENABLED)
        'calibration': {
            'enabled': True,
            'methods': ['temperature', 'platt']
        },
        
        # Validation Strategy
        'validation': {
            'strategy': best_params.get('validation_strategy', 'boundary_f1')
        },
        
        # Features Configuration
        'features': {
            'head_ssm': {
                'enabled': best_params.get('head_ssm_enabled', True),
                'words': best_params.get('head_tail_words', 2),
                'dimension': 12
            },
            'tail_ssm': {
                'enabled': best_params.get('tail_ssm_enabled', True),
                'words': best_params.get('head_tail_words', 2),
                'dimension': 12
            },
            'phonetic_ssm': {
                'enabled': best_params.get('phonetic_ssm_enabled', True),
                'mode': best_params.get('phonetic_mode', 'rhyme'),
                'similarity_method': best_params.get('phonetic_similarity_method', 'binary'),
                'normalize': best_params.get('phonetic_normalize', False),
                'normalize_method': best_params.get('phonetic_normalize_method', 'zscore'),
                'high_sim_threshold': best_params.get('phonetic_high_sim_threshold', 0.31),
                'dimension': 12
            },
            'pos_ssm': {
                'enabled': best_params.get('pos_ssm_enabled', True),
                'tagset': best_params.get('pos_tagset', 'simplified'),
                'similarity_method': best_params.get('pos_similarity_method', 'combined'),
                'high_sim_threshold': best_params.get('pos_high_sim_threshold', 0.27),
                'dimension': 12
            },
            'string_ssm': {
                'enabled': best_params.get('string_ssm_enabled', True),
                'case_sensitive': best_params.get('string_case_sensitive', False),
                'remove_punctuation': best_params.get('string_remove_punctuation', True),
                'similarity_threshold': best_params.get('string_similarity_threshold', 0.055),
                'similarity_method': best_params.get('string_similarity_method', 'word_overlap'),
                'dimension': 12
            },
            'syllable_pattern_ssm': {
                'enabled': best_params.get('syllable_pattern_ssm_enabled', True),
                'similarity_method': best_params.get('syllable_pattern_similarity_method', 'cosine'),
                'levenshtein_weight': best_params.get('syllable_pattern_levenshtein_weight', 0.7),
                'cosine_weight': best_params.get('syllable_pattern_cosine_weight', 0.3),
                'normalize': best_params.get('syllable_pattern_normalize', False),
                'normalize_method': best_params.get('syllable_pattern_normalize_method', 'zscore'),
                'dimension': 12
            },
            'line_syllable_ssm': {
                'enabled': best_params.get('line_syllable_ssm_enabled', True),
                'similarity_method': best_params.get('line_syllable_similarity_method', 'cosine'),
                'ratio_threshold': best_params.get('line_syllable_ratio_threshold', 0.09),
                'normalize': best_params.get('line_syllable_normalize', False),
                'normalize_method': best_params.get('line_syllable_normalize_method', 'minmax'),
                'dimension': 12
            }
        },
        
        # Embeddings Configuration (STATIC models)
        'embeddings': {
            'word2vec': {
                'enabled': best_params.get('word2vec_enabled', True),
                'model': 'word2vec-google-news-300',  # STATIC
                'mode': best_params.get('word2vec_mode', 'complete'),
                'normalize': best_params.get('word2vec_normalize', True),
                'similarity_metric': best_params.get('word2vec_similarity_metric', 'cosine'),
                'high_sim_threshold': best_params.get('word2vec_high_sim_threshold', 0.82)
            },
            'contextual': {
                'enabled': best_params.get('contextual_enabled', True),
                'model': 'all-MiniLM-L6-v2',  # STATIC
                'mode': best_params.get('contextual_mode', 'complete'),
                'normalize': best_params.get('contextual_normalize', True),
                'similarity_metric': best_params.get('contextual_similarity_metric', 'cosine'),
                'high_sim_threshold': best_params.get('contextual_high_sim_threshold', 0.72)
            }
        },
        
        # Optimization metadata
        'optimization_info': {
            'optimized_with_optuna': True,
            'optimization_date': datetime.now().isoformat(),
            'best_boundary_f1': float(study.best_value),
            'n_trials': len(study.trials),
            'study_name': study.study_name,
            'config_mode': 'LOCAL (no external files)',
            'compatibility_enforced': True,
            'calibration_always_enabled': True,
            'static_embedding_models': True
        }
    }
    
    # Save in session directory
    config_file = session_dir / 'best_config_local.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"🏆 Best LOCAL config saved to {config_file}")
    print(f"   🎯 Boundary F1: {study.best_value:.4f}")
    print(f"   ⚙️  Compatibility: ENFORCED")
    print(f"   📏  Calibration: ENABLED")
    print(f"   🎵  Embeddings: STATIC models")


def create_best_config(study):
    """Legacy function - redirects to create_best_config_local."""
    create_best_config_local(study)


def _fmt(v, fmt="{:.3f}"):
    """Safe formatting helper to avoid errors with missing values."""
    try: 
        return fmt.format(v)
    except Exception: 
        return str(v)


def create_summary_report(study, session_dir: Path):
    """Create human-readable summary report in the session directory."""
    best_params = study.best_params
    
    summary = f"""Boundary-Aware BiLSTM Hyperparameter Optimization Results
========================================================
Optimization completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Study name: {study.study_name}
Session directory: {session_dir}

PERFORMANCE
-----------
Best boundary F1: {study.best_value:.4f}
Total trials completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}
Total trials: {len(study.trials)}

OPTIMAL PARAMETERS
------------------
Model Architecture:
  - hidden_dim: {best_params.get('hidden_dim', 'N/A')}
  - num_layers: {best_params.get('num_layers', 'N/A')}
  - dropout: {_fmt(best_params.get('dropout', 'N/A'))}
  - layer_dropout: {_fmt(best_params.get('layer_dropout', 'N/A'))}

Attention Mechanism:
  - attention_heads: {best_params.get('attention_heads', 'N/A')}
  - attention_dropout: {_fmt(best_params.get('attention_dropout', 'N/A'))}
  - attention_dim: {best_params.get('attention_dim', 'N/A')}

Training Configuration:
  - learning_rate: {_fmt(best_params.get('learning_rate', 'N/A'), '{:.4f}')}
  - batch_size: {best_params.get('batch_size', 'N/A')}
  - weight_decay: {_fmt(best_params.get('weight_decay', 'N/A'))}
  - scheduler: {best_params.get('scheduler', 'N/A')}

Boundary-Aware Loss:
  - boundary_weight: {_fmt(best_params.get('boundary_weight', 'N/A'), '{:.2f}')}
  - label_smoothing: {_fmt(best_params.get('label_smoothing', 'N/A'), '{:.2f}')}
  - segment_consistency_lambda: {_fmt(best_params.get('segment_consistency_lambda', 'N/A'))}
  - conf_penalty_lambda: {_fmt(best_params.get('conf_penalty_lambda', 'N/A'))}
  - entropy_lambda: {_fmt(best_params.get('entropy_lambda', 'N/A'))}

OPTIMIZATION STATISTICS
-----------------------
"""
    
    # Add trial performance distribution
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if completed_trials:
        values = [t.value for t in completed_trials]
        summary += f"Boundary F1 distribution:\n"
        summary += f"  - Best: {_fmt(max(values), '{:.4f}')}\n"
        summary += f"  - Worst: {_fmt(min(values), '{:.4f}')}\n"
        summary += f"  - Mean: {_fmt(sum(values)/len(values), '{:.4f}')}\n"
        summary += f"  - Median: {_fmt(sorted(values)[len(values)//2], '{:.4f}')}\n"
    
    # Add failed trial info
    failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    if failed_trials > 0:
        summary += f"\nFailed trials: {failed_trials}\n"
    
    summary += f"\nNext steps:\n"
    summary += f"1. Review {session_dir}/best_config_local.yaml\n"
    summary += f"2. Run full training with optimized parameters\n"
    summary += f"3. Compare against previous manual configurations\n"
    summary += f"4. Analyze SQL database: {session_dir}/hyperopt_results.db\n"
    
    # Save in session directory
    summary_file = session_dir / 'hyperopt_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(summary)
    
    print(f"📄 Summary report saved to {summary_file}")


def main():
    """Main hyperparameter optimization with LOCAL configuration only."""
    global HYPERPARAMS, SESSION_DIR
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for BiLSTM (LOCAL CONFIG ONLY)")
    parser.add_argument('--n-trials', type=int, default=50, help='Number of trials to run')
    parser.add_argument('--timeout', type=int, default=72*3600, help='Timeout in seconds (default: 72 hours)')
    parser.add_argument('--study-name', default='boundary_aware_hyperopt_local', help='Study name')
    parser.add_argument('--resume', action='store_true', help='Resume existing study')
    
    args = parser.parse_args()
    
    # Initialize hyperparameter space (LOCAL ONLY - no external config files)
    HYPERPARAMS = HyperparameterSpace()
    
    # Create session directory with date/time organization
    SESSION_DIR = create_session_directory(args.study_name)
    
    print("🔬 Boundary-Aware BiLSTM Hyperparameter Optimization (LOCAL CONFIG)")
    print("=" * 70)
    print(f"📊 Configuration:")
    print(f"   Config mode: LOCAL only (no external files)")
    print(f"   Max trials: {args.n_trials}")
    print(f"   Timeout: {args.timeout/3600:.1f} hours")
    print(f"   Study name: {args.study_name}")
    print(f"   Session directory: {SESSION_DIR}")
    
    # Show device info
    device = get_optimal_device()
    
    print(f"\n📋 Search Space Summary:")
    print(f"   🏗️  Architecture: hidden_dim {HYPERPARAMS.HIDDEN_DIM}, layers {HYPERPARAMS.NUM_LAYERS}")
    print(f"       Attention: heads {HYPERPARAMS.ATTENTION_HEADS}, dims matched to hidden_dim")
    print(f"       Compatibility: ENFORCED (heads divide hidden_dim, attention_dim = hidden_dim)")
    print(f"   🎯  Training: epochs = cosine_t_max {HYPERPARAMS.MAX_EPOCHS_OPTIONS}")
    print(f"       Validation strategy: {HYPERPARAMS.VALIDATION_STRATEGY_OPTIONS}")
    print(f"       Calibration: ALWAYS ENABLED (temperature + platt)")
    print(f"   🎵  Features: head/tail words {HYPERPARAMS.HEAD_TAIL_WORDS_OPTIONS}")
    if HYPERPARAMS.OPTIMIZE_FEATURE_TOGGLES:
        print(f"       Feature toggles: {len(HYPERPARAMS.FEATURE_TOGGLES)} ON/OFF switches")
    else:
        print(f"       Feature toggles: All enabled (not optimizing)")
    print(f"   📊  Embeddings: STATIC models (word2vec-google-news-300, all-MiniLM-L6-v2)")
    print(f"       Only mode/similarity optimized, NO dimension/model changes")
    print(f"   📈  Target metric: boundary_f1 (primary optimization goal)")
    
    # Show parameter counts
    arch_combinations = len(HYPERPARAMS.HIDDEN_DIM) * len(HYPERPARAMS.NUM_LAYERS)
    attention_combinations = len(HYPERPARAMS.ATTENTION_HEADS)  # attention_dim matches hidden_dim
    training_combinations = len(HYPERPARAMS.MAX_EPOCHS_OPTIONS) * len(HYPERPARAMS.BATCH_SIZE) * len(HYPERPARAMS.SCHEDULER_OPTIONS)
    validation_combinations = len(HYPERPARAMS.VALIDATION_STRATEGY_OPTIONS)
    head_tail_combinations = len(HYPERPARAMS.HEAD_TAIL_WORDS_OPTIONS)
    
    print(f"\n🔢 Parameter Combinations:")
    print(f"   Architecture: {arch_combinations} combinations")
    print(f"   Attention: {attention_combinations} head options (compatible with all hidden_dims)")
    print(f"   Training: {training_combinations} combinations")
    print(f"   Validation: {validation_combinations} strategies")
    print(f"   Head/Tail words: {head_tail_combinations} options")
    if HYPERPARAMS.OPTIMIZE_FEATURE_TOGGLES:
        feature_combinations = 2 ** len(HYPERPARAMS.FEATURE_TOGGLES)
        print(f"   Feature toggles: {feature_combinations} combinations")
    print(f"   Continuous parameters: ~100 combinations (loss, thresholds, etc.)")
    
    # Create study with SQLite persistence in session directory
    study_db_path = SESSION_DIR / "optuna_study.db"
    study = optuna.create_study(
        direction='maximize',
        storage=f'sqlite:///{study_db_path}',
        study_name=args.study_name,
        load_if_exists=args.resume,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    if args.resume and len(study.trials) > 0:
        print(f"\n📈 Resuming study with {len(study.trials)} existing trials")
        print(f"   Best so far: {study.best_value:.4f}")
    
    print(f"\n🚀 Starting optimization...")
    print(f"   ⚙️  Compatibility checks: ENABLED")
    print(f"   📏  Calibration: ALWAYS ON")
    print(f"   🎵  Embedding models: STATIC")
    print(f"   📊  Epochs = cosine_t_max: ENFORCED")
    
    try:
        # Optimize
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
        
        print(f"\n✅ Optimization complete!")
        print(f"   🏆 Best boundary F1: {study.best_value:.4f}")
        print(f"   📊 Total trials: {len(study.trials)}")
        
        # Show best parameters summary
        if len(study.trials) > 0:
            best_params = study.best_params
            print(f"\n🎯 Best Configuration Summary:")
            print(f"   Model: {best_params.get('hidden_dim')}D hidden, {best_params.get('num_layers')} layers")
            print(f"   Attention: {best_params.get('attention_heads')} heads")
            print(f"   Training: lr={best_params.get('learning_rate', 0):.2e}, batch={best_params.get('batch_size')}")
            print(f"   Epochs: {best_params.get('max_epochs')} (= cosine_t_max)")
            print(f"   Validation: {best_params.get('validation_strategy')}")
            print(f"   Head/Tail words: {best_params.get('head_tail_words')}")
        
        # Export results to session directory
        export_results(study, SESSION_DIR)
        export_to_sql(study, SESSION_DIR)
        create_best_config_local(study, SESSION_DIR)
        create_summary_report(study, SESSION_DIR)
        
        print(f"\n📁 All results saved to session directory: {SESSION_DIR}")
        print(f"   🏆 Best config: {SESSION_DIR}/best_config_local.yaml")
        print(f"   📊 Summary: {SESSION_DIR}/hyperopt_summary.txt")
        print(f"   📈 Full results: {SESSION_DIR}/hyperopt_results.json")
        print(f"   🗄️  SQL database: {SESSION_DIR}/hyperopt_results.db")
        print(f"   📂 Trial details: {SESSION_DIR}/trials/")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Optimization interrupted by user")
        if len(study.trials) > 0:
            print(f"   Partial results available with {len(study.trials)} trials")
            export_results(study, SESSION_DIR)
            export_to_sql(study, SESSION_DIR)
            create_best_config_local(study, SESSION_DIR)
            create_summary_report(study, SESSION_DIR)
            create_best_config_local(study)
            create_summary_report(study)
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        print(traceback.format_exc())


def _fmt(value, format_str='{:.4f}'):
    """Safe formatting helper that handles None and NaN values."""
    if value is None or (hasattr(value, '__iter__') and len(value) == 0):
        return 'N/A'
    try:
        if isinstance(value, (int, float)) and (value != value or value == float('inf') or value == float('-inf')):  # NaN or inf check
            return 'N/A'
        return format_str.format(value)
    except (TypeError, ValueError):
        return str(value)


def print_study_summary(study):
    """Print a comprehensive study summary with safe formatting."""
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    
    summary = f"\n📊 OPTIMIZATION SUMMARY\n"
    summary += f"{'='*50}\n"
    summary += f"Study name: {study.study_name}\n"
    summary += f"Total trials: {len(study.trials)}\n"
    summary += f"  - Completed: {len(completed_trials)}\n"
    summary += f"  - Pruned: {len(pruned_trials)}\n"
    summary += f"  - Failed: {len(failed_trials)}\n"
    
    if completed_trials:
        values = [t.value for t in completed_trials]
        summary += f"Boundary F1 distribution:\n"
        summary += f"  - Best: {_fmt(max(values), '{:.4f}')}\n"
        summary += f"  - Worst: {_fmt(min(values), '{:.4f}')}\n"
        summary += f"  - Mean: {_fmt(sum(values)/len(values), '{:.4f}')}\n"
        summary += f"  - Median: {_fmt(sorted(values)[len(values)//2], '{:.4f}')}\n"
        
        if study.best_trial:
            best_params = study.best_params
            summary += f"\n🏆 BEST CONFIGURATION:\n"
            summary += f"  Trial: {study.best_trial.number}\n"
            summary += f"  Boundary F1: {_fmt(study.best_value, '{:.4f}')}\n"
            summary += f"  Model: {best_params.get('hidden_dim')}D hidden, {best_params.get('num_layers')} layers\n"
            summary += f"  Attention: {best_params.get('attention_heads')} heads\n"
            summary += f"  Training: lr={_fmt(best_params.get('learning_rate', 0), '{:.2e}')}, batch={best_params.get('batch_size')}\n"
            summary += f"  Scheduler: {best_params.get('scheduler')}\n"
            summary += f"  Validation: {best_params.get('validation_strategy')}\n"
            summary += f"  Head/Tail words: {best_params.get('head_tail_words')}\n"
    
    print(summary)



if __name__ == "__main__":
    main()
