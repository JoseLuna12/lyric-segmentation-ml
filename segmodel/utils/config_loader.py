"""
Configuration loading and validation utilities.
"""

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import torch


@dataclass
class TrainingConfig:
    """Complete training configuration dataclass."""
    
    # Data paths
    train_file: str
    val_file: str  
    test_file: str
    
    # Model architecture
    hidden_dim: int = 128
    num_classes: int = 2
    dropout: float = 0.4
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_epochs: int = 60
    patience: int = 8
    gradient_clip_norm: float = 1.0
    
    # Advanced Learning Rate Scheduling
    scheduler: str = "plateau"
    min_lr: float = 1e-6
    cosine_t_max: int = 60
    warmup_epochs: int = 5
    lr_factor: float = 0.5
    lr_patience: int = 10
    step_size: int = 30
    step_gamma: float = 0.5
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    
    # Anti-collapse settings
    label_smoothing: float = 0.2
    weighted_sampling: bool = True
    entropy_lambda: float = 0.0
    
    # Enhanced Emergency Monitoring (all configurable)
    emergency_monitoring_enabled: bool = True
    # Batch-level monitoring thresholds
    max_confidence_threshold: float = 0.95
    min_chorus_rate: float = 0.05
    max_chorus_rate: float = 0.85
    max_conf_over_95_ratio: float = 0.1
    # Epoch-level monitoring thresholds
    val_overconf_threshold: float = 0.96
    val_f1_collapse_threshold: float = 0.1
    emergency_overconf_threshold: float = 0.98
    emergency_conf95_ratio: float = 0.8
    emergency_f1_threshold: float = 0.05
    # Timing parameters
    skip_batches: int = 50
    skip_epochs: int = 3
    print_batch_every: int = 10
    
    # Temperature Calibration
    temperature_grid: list = None
    default_temperature: float = 1.0
    
    # Validation Strategy (Phase 3) - Simplified
    validation_strategy: str = "line_f1"  # Simple strategy selection
    
    # Features
    head_ssm_enabled: bool = True
    head_ssm_dimension: int = 12
    head_ssm_words: int = 2
    tail_ssm_enabled: bool = True
    tail_ssm_dimension: int = 12
    tail_ssm_words: int = 2
    phonetic_ssm_enabled: bool = True
    phonetic_ssm_dimension: int = 12
    phonetic_ssm_mode: str = "rhyme"
    phonetic_ssm_similarity_method: str = "binary"
    phonetic_ssm_normalize: bool = False
    phonetic_ssm_normalize_method: str = "zscore"
    phonetic_ssm_high_sim_threshold: float = 0.8
    pos_ssm_enabled: bool = False
    pos_ssm_dimension: int = 12
    pos_ssm_tagset: str = "simplified"
    pos_ssm_similarity_method: str = "combined"
    pos_ssm_high_sim_threshold: float = 0.7
    string_ssm_enabled: bool = True
    string_ssm_dimension: int = 12
    string_ssm_case_sensitive: bool = False
    string_ssm_remove_punctuation: bool = True
    string_ssm_similarity_threshold: float = 0.0
    string_ssm_similarity_method: str = "word_overlap"
    
    # Output settings
    output_base_dir: str = "training_sessions"
    save_best_model: bool = True
    save_final_model: bool = True
    save_training_metrics: bool = True
    
    # System settings
    seed: int = 42
    device: str = "auto"
    num_workers: int = 4
    
    # Experiment metadata
    experiment_name: str = "blstm_experiment"
    experiment_description: str = "BiLSTM training experiment"
    experiment_tags: list = None
    experiment_notes: str = ""
    
    def __post_init__(self):
        if self.experiment_tags is None:
            self.experiment_tags = []
        if self.temperature_grid is None:
            self.temperature_grid = [0.8, 1.0, 1.2, 1.5, 1.7, 2.0]


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded config from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {e}")
    
    except Exception as e:
        raise RuntimeError(f"Error loading config file: {e}")


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize configuration."""
    
    # Check required sections
    required_sections = ['data', 'training']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Check required data fields
    required_data_fields = ['train_file', 'val_file', 'test_file']
    for field in required_data_fields:
        if field not in config['data']:
            raise ValueError(f"Missing required data field: {field}")
    
    # Validate file paths exist
    for field in required_data_fields:
        file_path = Path(config['data'][field])
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {config['data'][field]}")
    
    # Validate numeric ranges
    training = config.get('training', {})
    
    if training.get('batch_size', 1) < 1:
        raise ValueError("batch_size must be >= 1")
    
    if training.get('learning_rate', 0.001) <= 0:
        raise ValueError("learning_rate must be > 0")
    
    if training.get('max_epochs', 1) < 1:
        raise ValueError("max_epochs must be >= 1")
    
    # Validate anti-collapse settings
    anti_collapse = config.get('anti_collapse', {})
    label_smoothing = anti_collapse.get('label_smoothing', 0.0)
    
    if not (0.0 <= label_smoothing <= 1.0):
        raise ValueError("label_smoothing must be between 0.0 and 1.0")
    
    entropy_lambda = anti_collapse.get('entropy_lambda', 0.0)
    if entropy_lambda < 0:
        raise ValueError("entropy_lambda must be >= 0")
    
    print("✅ Configuration validation passed")
    return config


def flatten_config(config: Dict[str, Any]) -> TrainingConfig:
    """Convert nested YAML config to flat TrainingConfig object."""
    
    # Extract nested values with defaults
    data = config.get('data', {})
    model = config.get('model', {})
    training = config.get('training', {})
    anti_collapse = config.get('anti_collapse', {})
    emergency = config.get('emergency_monitoring', {})
    temperature_calib = config.get('temperature_calibration', {})
    features = config.get('features', {})
    head_ssm = features.get('head_ssm', {})
    tail_ssm = features.get('tail_ssm', {})
    phonetic_ssm = features.get('phonetic_ssm', {})
    pos_ssm = features.get('pos_ssm', {})
    string_ssm = features.get('string_ssm', {})
    output = config.get('output', {})
    system = config.get('system', {})
    experiment = config.get('experiment', {})
    
    return TrainingConfig(
        # Data
        train_file=data['train_file'],
        val_file=data['val_file'],
        test_file=data['test_file'],
        
        # Model
        hidden_dim=model.get('hidden_dim', 128),
        num_classes=model.get('num_classes', 2),
        dropout=model.get('dropout', 0.4),
        
        # Training
        batch_size=training.get('batch_size', 16),
        learning_rate=training.get('learning_rate', 0.001),
        weight_decay=training.get('weight_decay', 0.01),
        max_epochs=training.get('max_epochs', 60),
        patience=training.get('patience', 8),
        gradient_clip_norm=training.get('gradient_clip_norm', 1.0),
        
        # ✅ NEW: Advanced Learning Rate Scheduling
        scheduler=training.get('scheduler', 'plateau'),
        min_lr=training.get('min_lr', 1e-6),
        cosine_t_max=training.get('cosine_t_max', training.get('max_epochs', 60)),
        warmup_epochs=training.get('warmup_epochs', 5),
        lr_factor=training.get('lr_factor', 0.5),
        lr_patience=training.get('lr_patience', training.get('patience', 8) // 2),
        step_size=training.get('step_size', training.get('max_epochs', 60) // 4),
        step_gamma=training.get('step_gamma', 0.5),
        cosine_t0=training.get('cosine_t0', 10),
        cosine_t_mult=training.get('cosine_t_mult', 2),
        
        # Anti-collapse
        label_smoothing=anti_collapse.get('label_smoothing', 0.2),
        weighted_sampling=anti_collapse.get('weighted_sampling', True),
        entropy_lambda=anti_collapse.get('entropy_lambda', 0.0),
        
        # ✅ UPDATED: Enhanced Emergency Monitoring
        emergency_monitoring_enabled=emergency.get('enabled', True),
        # Batch-level monitoring thresholds
        max_confidence_threshold=emergency.get('max_confidence_threshold', 0.95),
        min_chorus_rate=emergency.get('min_chorus_rate', 0.05),
        max_chorus_rate=emergency.get('max_chorus_rate', 0.85),
        max_conf_over_95_ratio=emergency.get('max_conf_over_95_ratio', 0.1),
        # Epoch-level monitoring thresholds
        val_overconf_threshold=emergency.get('val_overconf_threshold', 0.96),
        val_f1_collapse_threshold=emergency.get('val_f1_collapse_threshold', 0.1),
        emergency_overconf_threshold=emergency.get('emergency_overconf_threshold', 0.98),
        emergency_conf95_ratio=emergency.get('emergency_conf95_ratio', 0.8),
        emergency_f1_threshold=emergency.get('emergency_f1_threshold', 0.05),
        # Timing parameters
        skip_batches=emergency.get('skip_batches', 50),
        skip_epochs=emergency.get('skip_epochs', 3),
        print_batch_every=emergency.get('print_batch_every', 10),
        
        # ✅ NEW: Temperature Calibration
        temperature_grid=temperature_calib.get('temperature_grid', [0.6, 0.8, 1.0, 1.2, 1.5, 1.7, 2.0]),
        default_temperature=temperature_calib.get('default_temperature', 1.0),
        
        # 🎯 NEW: Validation Strategy (Phase 3) - Simplified
        validation_strategy=config.get('validation_strategy', 'line_f1'),
        
        # Features
        head_ssm_enabled=head_ssm.get('enabled', True),
        head_ssm_dimension=head_ssm.get('output_dim', 12),
        head_ssm_words=head_ssm.get('head_words', 2),
        tail_ssm_enabled=tail_ssm.get('enabled', True),
        tail_ssm_dimension=tail_ssm.get('output_dim', 12),
        tail_ssm_words=tail_ssm.get('tail_words', 2),
        phonetic_ssm_enabled=phonetic_ssm.get('enabled', True),
        phonetic_ssm_dimension=phonetic_ssm.get('output_dim', 12),
        phonetic_ssm_mode=phonetic_ssm.get('mode', 'rhyme'),
        phonetic_ssm_similarity_method=phonetic_ssm.get('similarity_method', 'binary'),
        phonetic_ssm_normalize=phonetic_ssm.get('normalize', False),
        phonetic_ssm_normalize_method=phonetic_ssm.get('normalize_method', 'zscore'),
        phonetic_ssm_high_sim_threshold=phonetic_ssm.get('high_sim_threshold', 0.8),
        pos_ssm_enabled=pos_ssm.get('enabled', False),
        pos_ssm_dimension=pos_ssm.get('output_dim', 12),
        pos_ssm_tagset=pos_ssm.get('tagset', 'simplified'),
        pos_ssm_similarity_method=pos_ssm.get('similarity_method', 'combined'),
        pos_ssm_high_sim_threshold=pos_ssm.get('high_sim_threshold', 0.7),
        string_ssm_enabled=string_ssm.get('enabled', True),
        string_ssm_dimension=string_ssm.get('output_dim', 12),
        string_ssm_case_sensitive=string_ssm.get('case_sensitive', False),
        string_ssm_remove_punctuation=string_ssm.get('remove_punctuation', True),
        string_ssm_similarity_threshold=string_ssm.get('similarity_threshold', 0.0),
        string_ssm_similarity_method=string_ssm.get('similarity_method', 'word_overlap'),
        
        # Output
        output_base_dir=output.get('base_dir', 'training_sessions'),
        save_best_model=output.get('save_best_model', True),
        save_final_model=output.get('save_final_model', True),
        save_training_metrics=output.get('save_training_metrics', True),
        
        # System
        seed=system.get('seed', 42),
        device=system.get('device', 'auto'),
        num_workers=system.get('num_workers', 4),
        
        # Experiment
        experiment_name=experiment.get('name', 'blstm_experiment'),
        experiment_description=experiment.get('description', 'BiLSTM training experiment'),
        experiment_tags=experiment.get('tags', []),
        experiment_notes=experiment.get('notes', '')
    )


def resolve_device(device_config: str) -> str:
    """Resolve device configuration to actual device."""
    
    if device_config == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    else:
        return device_config


def load_training_config(config_path: str) -> TrainingConfig:
    """Load and validate complete training configuration."""
    
    print(f"🔧 Loading training configuration...")
    
    # Load YAML
    config_dict = load_yaml_config(config_path)
    
    # Validate
    validated_config = validate_config(config_dict)
    
    # Convert to structured config
    training_config = flatten_config(validated_config)
    
    # Resolve device
    training_config.device = resolve_device(training_config.device)
    
    print(f"📋 Configuration Summary:")
    print(f"   Experiment: {training_config.experiment_name}")
    print(f"   Model: hidden_dim={training_config.hidden_dim}, dropout={training_config.dropout}")
    print(f"   Training: batch_size={training_config.batch_size}, lr={training_config.learning_rate}, epochs={training_config.max_epochs}")
    print(f"   ✅ Scheduler: {training_config.scheduler} (min_lr={training_config.min_lr})")
    print(f"   Anti-collapse: smoothing={training_config.label_smoothing}, entropy_λ={training_config.entropy_lambda}")
    print(f"   ✅ Emergency monitoring: {len([x for x in [training_config.max_confidence_threshold, training_config.val_overconf_threshold] if x])} thresholds")
    
    # Feature summary
    enabled_features = []
    total_dim = 0
    if training_config.head_ssm_enabled:
        enabled_features.append(f"Head-SSM({training_config.head_ssm_dimension}D)")
        total_dim += training_config.head_ssm_dimension
    if training_config.tail_ssm_enabled:
        enabled_features.append(f"Tail-SSM({training_config.tail_ssm_dimension}D)")
        total_dim += training_config.tail_ssm_dimension
    if training_config.phonetic_ssm_enabled:
        phon_desc = f"Phonetic-SSM({training_config.phonetic_ssm_dimension}D,{training_config.phonetic_ssm_mode}"
        if training_config.phonetic_ssm_similarity_method != "binary":
            phon_desc += f",{training_config.phonetic_ssm_similarity_method}"
        if training_config.phonetic_ssm_normalize:
            phon_desc += f",norm"
        phon_desc += f",th={training_config.phonetic_ssm_high_sim_threshold})"
        enabled_features.append(phon_desc)
        total_dim += training_config.phonetic_ssm_dimension
    if training_config.pos_ssm_enabled:
        pos_desc = f"POS-SSM({training_config.pos_ssm_dimension}D,{training_config.pos_ssm_tagset},{training_config.pos_ssm_similarity_method},th={training_config.pos_ssm_high_sim_threshold})"
        enabled_features.append(pos_desc)
        total_dim += training_config.pos_ssm_dimension
    if training_config.string_ssm_enabled:
        string_desc = f"String-SSM({training_config.string_ssm_dimension}D"
        if training_config.string_ssm_case_sensitive:
            string_desc += ",case_sens"
        if not training_config.string_ssm_remove_punctuation:
            string_desc += ",keep_punct"
        if training_config.string_ssm_similarity_threshold > 0:
            string_desc += f",th={training_config.string_ssm_similarity_threshold}"
        string_desc += ")"
        enabled_features.append(string_desc)
        total_dim += training_config.string_ssm_dimension
    
    if enabled_features:
        print(f"   Features: {', '.join(enabled_features)} = {total_dim}D total")
    else:
        print(f"   Features: None enabled (⚠️ Warning)")
    
    print(f"   Device: {training_config.device}")
    
    return training_config


def merge_with_args(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    """Merge configuration with command line argument overrides."""
    
    overrides = []
    
    # Override with command line args if provided
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.batch_size = args.batch_size
        overrides.append(f"batch_size={args.batch_size}")
    
    if hasattr(args, 'learning_rate') and args.learning_rate is not None:
        config.learning_rate = args.learning_rate
        overrides.append(f"learning_rate={args.learning_rate}")
    
    if hasattr(args, 'epochs') and args.epochs is not None:
        config.max_epochs = args.epochs
        overrides.append(f"max_epochs={args.epochs}")
    
    if hasattr(args, 'label_smoothing') and args.label_smoothing is not None:
        config.label_smoothing = args.label_smoothing
        overrides.append(f"label_smoothing={args.label_smoothing}")
    
    if hasattr(args, 'entropy_lambda') and args.entropy_lambda is not None:
        config.entropy_lambda = args.entropy_lambda
        overrides.append(f"entropy_lambda={args.entropy_lambda}")
    
    if hasattr(args, 'disable_emergency_monitoring') and args.disable_emergency_monitoring:
        config.emergency_monitoring_enabled = False
        overrides.append("emergency_monitoring=disabled")
    
    # Data file overrides
    if hasattr(args, 'train') and args.train is not None:
        config.train_file = args.train
        overrides.append(f"train_file={args.train}")
        
    if hasattr(args, 'val') and args.val is not None:
        config.val_file = args.val
        overrides.append(f"val_file={args.val}")
        
    if hasattr(args, 'test') and args.test is not None:
        config.test_file = args.test
        overrides.append(f"test_file={args.test}")
    
    if overrides:
        print(f"🔄 Command line overrides: {', '.join(overrides)}")
    
    return config


def save_config_snapshot(config: TrainingConfig, output_dir: Path):
    """Save configuration snapshot for reproducibility."""
    
    config_snapshot = asdict(config)
    config_file = output_dir / "training_config_snapshot.yaml"
    
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config_snapshot, f, default_flow_style=False, indent=2)
        
        print(f"📝 Config snapshot saved: {config_file}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save config snapshot: {e}")


if __name__ == "__main__":
    # Test configuration loading
    
    print("🧪 Testing configuration loading...")
    
    try:
        # Test default config
        config = load_training_config("configs/training_config.yaml")
        print(f"✅ Successfully loaded config: {config.experiment_name}")
        
        # Test quick config
        quick_config = load_training_config("configs/quick_test.yaml") 
        print(f"✅ Successfully loaded quick config: {quick_config.experiment_name}")
        
        print("🎉 All configuration tests passed!")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
