"""
CNN training script with YAML configuration support.

This script loads training parameters from YAML configuration files
and supports command line overrides for key parameters.
Optimized for CNN models with CNN-specific training strategies.
"""

import argparse
import torch
from torch.optim import AdamW
from pathlib import Path
import time
from datetime import datetime

# Import our modules
from segmodel.utils import TrainingConfig, load_training_config, merge_with_args, save_config_snapshot
from segmodel.data import SongsDataset, create_dataloader
from segmodel.features import FeatureExtractor
from segmodel.train.cnn_trainer import CNNTrainer


def setup_data_loaders(config: TrainingConfig, feature_extractor):
    """Setup data loaders from configuration."""
    
    print(f"📚 Loading datasets for CNN training...")
    
    # Load datasets
    train_dataset = SongsDataset(config.train_file)
    val_dataset = SongsDataset(config.val_file)
    test_dataset = SongsDataset(config.test_file)
    
    print(f"✅ Loaded {len(train_dataset)} training songs")
    print(f"✅ Loaded {len(val_dataset)} validation songs")  
    print(f"✅ Loaded {len(test_dataset)} test songs")
    
    # Create data loaders - CNN can handle larger batches efficiently
    batch_size = config.batch_size
    if hasattr(config, 'cnn_batch_multiplier'):
        batch_size = int(batch_size * config.cnn_batch_multiplier)
        print(f"🏃‍♂️ CNN batch size multiplier: {config.cnn_batch_multiplier}x -> {batch_size}")
    
    train_loader = create_dataloader(
        train_dataset,
        feature_extractor,
        batch_size=batch_size,
        shuffle=True,
        use_weighted_sampling=config.weighted_sampling,
        num_workers=config.num_workers
    )
    
    val_loader = create_dataloader(
        val_dataset,
        feature_extractor,
        batch_size=batch_size, 
        shuffle=False,
        use_weighted_sampling=False,
        num_workers=config.num_workers
    )
    
    test_loader = create_dataloader(
        test_dataset,
        feature_extractor,
        batch_size=batch_size,
        shuffle=False, 
        use_weighted_sampling=False,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader, test_loader, train_dataset


def setup_cnn_model_and_training(config: TrainingConfig, train_dataset: SongsDataset, device: str, feature_dim: int):
    """Setup CNN model, loss function, and optimizer from configuration."""
    
    # Create CNN model using YAML configuration
    from segmodel.models.cnn_tagger import CNNTagger
    
    kernel_sizes = None
    dilation_rates = None
    use_residual = None
    
    if hasattr(config, 'cnn') and config.cnn is not None:
        cnn_config = config.cnn
        kernel_sizes = getattr(cnn_config, 'kernel_sizes', None)
        dilation_rates = getattr(cnn_config, 'dilation_rates', None) 
        use_residual = getattr(cnn_config, 'use_residual', None)
    
    if kernel_sizes is None:
        kernel_sizes = getattr(config, 'cnn_kernel_sizes', [3, 5, 7])
    if dilation_rates is None:
        dilation_rates = getattr(config, 'cnn_dilation_rates', [1, 2, 4])
    if use_residual is None:
        use_residual = getattr(config, 'cnn_use_residual', True)
    
    print(f"🤖 Creating CNN model...")
    model = CNNTagger(
        feat_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        layer_dropout=config.layer_dropout,
        attention_enabled=config.attention_enabled,
        attention_type=config.attention_type,
        attention_heads=config.attention_heads,
        attention_dropout=config.attention_dropout,
        attention_dim=config.attention_dim,
        positional_encoding=config.positional_encoding,
        max_seq_length=config.max_seq_length,
        window_size=config.window_size,
        boundary_temperature=config.boundary_temperature,
        # CNN-specific parameters
        kernel_sizes=tuple(kernel_sizes),
        dilation_rates=tuple(dilation_rates),
        use_residual=use_residual
    ).to(device)
    
    print(f"\n🤖 CNN Architecture Details:")
    model.print_model_info()
    
    print(f"\n📊 CNN Architecture Summary:")
    print(f"   Input features: {feature_dim}D")
    print(f"   Hidden dimension: {config.hidden_dim}D")
    print(f"   CNN layers: {config.num_layers}")
    print(f"   Kernel sizes: {kernel_sizes}")
    print(f"   Dilation rates: {dilation_rates}")
    print(f"   Residual connections: {use_residual}")
    
    if config.num_layers > 1:
        print(f"   ✅ Multi-layer CNN: {config.num_layers} blocks")
        if config.layer_dropout > 0:
            print(f"      Inter-layer dropout: {config.layer_dropout}")
        else:
            print(f"      No inter-layer dropout")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"      Total parameters: {total_params:,}")
    else:
        print(f"   Single CNN block")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"      Total parameters: {total_params:,}")
    
    # Attention information
    if config.attention_enabled:
        print(f"   🎯 Attention mechanism:")
        print(f"      Type: {config.attention_type}")
        print(f"      Heads: {config.attention_heads}")
        print(f"      Dropout: {config.attention_dropout}")
        if config.attention_dim:
            print(f"      Dimension: {config.attention_dim}")
        else:
            print(f"      Dimension: auto (CNN output = {config.hidden_dim})")
        print(f"      Positional encoding: {config.positional_encoding}")
        if hasattr(model, 'attention') and model.attention is not None:
            attention_params = sum(p.numel() for p in model.attention.parameters())
            total_params = sum(p.numel() for p in model.parameters())
            print(f"      Attention parameters: {attention_params:,} ({(attention_params/total_params*100):.1f}% of total)")
    else:
        print(f"   🎯 Attention: disabled")
    
    print(f"   Output dropout: {config.dropout}")
    print(f"   Output classes: {config.num_classes}")
    
    # Loss Function Configuration
    print(f"\n🎯 CNN Loss Function Configuration:")
    print(f"=" * 70)
    
    if not hasattr(config, 'loss'):
        raise ValueError(
            "❌ Config must include 'loss' section!\n"
            "   Please update your YAML config to include:\n"
            "   loss:\n"
            "     type: 'boundary_aware_cross_entropy'  # or 'cross_entropy'\n"
            "     label_smoothing: 0.16\n"
            "     # ... other loss parameters\n"
        )
    
    loss_config = config.loss
    loss_type = loss_config.get('type', 'boundary_aware_cross_entropy')
    
    print(f"   ✅ CNN Loss Configuration Found")
    print(f"      Loss Type: {loss_type}")
    
    class_weights = train_dataset.get_class_weights().to(device)
    label_smoothing = loss_config.get('label_smoothing', 0.2)
    
    if loss_type == 'boundary_aware_cross_entropy':
        # Boundary-aware loss parameters
        entropy_lambda = loss_config.get('entropy_lambda', 0.0)
        boundary_weight = loss_config.get('boundary_weight', 2.0)
        segment_consistency_lambda = loss_config.get('segment_consistency_lambda', 0.03)
        conf_penalty_lambda = loss_config.get('conf_penalty_lambda', 0.005)
        conf_threshold = loss_config.get('conf_threshold', 0.95)
        use_boundary_as_primary = loss_config.get('use_boundary_as_primary', True)
        
        from segmodel.losses.boundary_aware_cross_entropy import BoundaryAwareCrossEntropy
        loss_function = BoundaryAwareCrossEntropy(
            num_classes=config.num_classes,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
            boundary_weight=boundary_weight,
            segment_consistency_lambda=segment_consistency_lambda,
            conf_penalty_lambda=conf_penalty_lambda,
            conf_threshold=conf_threshold,
            entropy_lambda=entropy_lambda,
            use_boundary_as_primary=use_boundary_as_primary
        )
        
        print(f"\n   🏗️ CNN Boundary-Aware Loss Architecture:")
        print(f"      Base Loss: Cross-Entropy (label_smoothing={label_smoothing:.3f})")
        print(f"      Class Weights: {class_weights.cpu().numpy()}")
        print(f"      ├── Boundary Weight: {boundary_weight:.1f}x {'✅ ACTIVE' if boundary_weight > 1.0 else '❌ DISABLED'}")
        print(f"      ├── Segment Consistency: λ={segment_consistency_lambda:.3f} {'✅ ACTIVE' if segment_consistency_lambda > 0 else '❌ DISABLED'}")
        print(f"      ├── Confidence Penalty: λ={conf_penalty_lambda:.3f} {'✅ ACTIVE' if conf_penalty_lambda > 0 else '❌ DISABLED'}")
        print(f"      ├── Entropy Regularization: λ={entropy_lambda:.3f} {'✅ ACTIVE' if entropy_lambda > 0 else '❌ DISABLED'}")
        print(f"      └── Architecture: {'🎯 Boundary-Primary' if use_boundary_as_primary else '📊 Cross-Entropy Primary'}")
        
        print(f"\n   📈 Expected CNN + Boundary-Aware Improvements:")
        print(f"      • CNN local pattern detection + boundary awareness")
        print(f"      • Enhanced segment boundary detection (+15-25% boundary F1)")
        print(f"      • Reduced segmentation fragmentation")
        print(f"      • Improved confidence calibration")
            
    elif loss_type == 'cross_entropy':
        entropy_lambda = loss_config.get('entropy_lambda', 0.0)
        
        from segmodel.losses.cross_entropy import create_loss_function as create_legacy_loss
        loss_function = create_legacy_loss(
            num_classes=config.num_classes,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
            entropy_lambda=entropy_lambda
        )
        
        print(f"\n   🏗️ CNN Legacy Cross-Entropy Loss:")
        print(f"      Base Loss: Cross-Entropy (label_smoothing={label_smoothing:.3f})")
        print(f"      Class Weights: {class_weights.cpu().numpy()}")
        print(f"      Entropy Regularization: λ={entropy_lambda:.3f} {'✅ ACTIVE' if entropy_lambda > 0 else '❌ DISABLED'}")
        print(f"\n   ⚠️  Consider upgrading to 'boundary_aware_cross_entropy' for CNN:")
        print(f"      • CNN excels at local pattern detection")
        print(f"      • Boundary-aware loss complements CNN strengths")
        
    else:
        raise ValueError(f"❌ Unknown loss type: {loss_type}")
    
    print(f"=" * 70)
    
    weight_decay = getattr(config, 'weight_decay', 0.01)
    if hasattr(config, 'cnn_weight_decay'):
        weight_decay = config.cnn_weight_decay
        print(f"🏃‍♂️ Using CNN-specific weight decay: {weight_decay}")
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=weight_decay
    )
    
    print(f"⚙️  CNN Training setup:")
    if config.learning_rate > 0.0005:
        print(f"   ⚠️ WARNING: Learning rate {config.learning_rate} may be too high for CNN models!")
        print(f"   ⚠️ Consider using a lower learning rate (1e-4 to 5e-4) for stability")
        
    print(f"   Optimizer: AdamW (lr={config.learning_rate}, wd={weight_decay})")
    print(f"   Scheduler: {getattr(config, 'scheduler', 'onecycle')} (CNN-optimized)")
    print(f"   Gradient clipping: {config.gradient_clip_norm}")
    
    return model, loss_function, optimizer


def create_cnn_session_directory(config: TrainingConfig) -> Path:
    """Create timestamped CNN session directory."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"cnn_session_{timestamp}_{config.experiment_name}"
    session_dir = Path(config.output_base_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 CNN Session directory: {session_dir}")
    
    return session_dir


def print_cnn_experiment_header(config: TrainingConfig):
    """Print CNN experiment information."""
    
    print(f"\n🎵 CNN Training with YAML Config")
    print("=" * 50)
    print(f"🧪 Experiment: {config.experiment_name}")
    print(f"📄 Description: {config.experiment_description}")
    
    if config.experiment_tags:
        tags = list(config.experiment_tags)
        if 'CNN' not in tags:
            tags.append('CNN')
        print(f"🏷️  Tags: {', '.join(tags)}")
    else:
        print(f"🏷️  Tags: CNN")
    
    if config.experiment_notes:
        print(f"📝 Notes: {config.experiment_notes}")
    
    print(f"🍎 Device: {config.device}")
    print(f"🏗️  Architecture: CNN (Convolutional Neural Network)")


def main():
    """Main CNN training function."""
    
    parser = argparse.ArgumentParser(
        description="CNN training with YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
CNN Training Examples:
  # Use CNN config
  python train_cnn_with_config.py configs/cnn_training_config.yaml
  
  # Quick CNN test
  python train_cnn_with_config.py configs/cnn_quick_test.yaml
  
  # With CNN-specific overrides
  python train_cnn_with_config.py configs/cnn_config.yaml --batch-size 64 --epochs 30
  
  # Disable emergency monitoring
  python train_cnn_with_config.py configs/cnn_config.yaml --disable-emergency-monitoring
        """
    )
    
    parser.add_argument('config', 
                       help='Path to YAML configuration file')
    
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size from config')
    parser.add_argument('--learning-rate', type=float,
                       help='Override learning rate from config')
    parser.add_argument('--epochs', type=int,
                       help='Override max epochs from config')
    parser.add_argument('--label-smoothing', type=float,
                       help='Override label smoothing from config')
    parser.add_argument('--entropy-lambda', type=float,
                       help='Override entropy lambda from config')
    parser.add_argument('--disable-emergency-monitoring', action='store_true',
                       help='Disable emergency monitoring (overrides config)')
    
    parser.add_argument('--cnn-kernel-sizes', nargs='+', type=int,
                       help='Override CNN kernel sizes (e.g., --cnn-kernel-sizes 3 5 7)')
    parser.add_argument('--cnn-layers', type=int,
                       help='Override number of CNN layers')
    parser.add_argument('--cnn-scheduler', choices=['onecycle', 'cosine', 'cosine_restarts', 'plateau'],
                       help='Override CNN scheduler type')
    
    parser.add_argument('--train',
                       help='Override training data file')
    parser.add_argument('--val', 
                       help='Override validation data file')
    parser.add_argument('--test',
                       help='Override test data file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_training_config(args.config)
        
        import yaml
        with open(args.config, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        config = merge_with_args(config, args)
        
        if 'cnn' in raw_config and raw_config['cnn']:
            cnn_config = raw_config['cnn']
            print(f"🔧 Loading CNN-specific configuration...")
            
            if 'kernel_sizes' in cnn_config:
                config.cnn_kernel_sizes = cnn_config['kernel_sizes']
                print(f"   Kernel sizes: {cnn_config['kernel_sizes']}")
            
            if 'dilation_rates' in cnn_config:
                config.cnn_dilation_rates = cnn_config['dilation_rates'] 
                print(f"   Dilation rates: {cnn_config['dilation_rates']}")
            
            if 'use_residual' in cnn_config:
                config.cnn_use_residual = cnn_config['use_residual']
                print(f"   Residual connections: {cnn_config['use_residual']}")
            
            if 'batch_multiplier' in cnn_config:
                config.cnn_batch_multiplier = cnn_config['batch_multiplier']
                print(f"   Batch multiplier: {cnn_config['batch_multiplier']}")
            
            if 'weight_decay' in cnn_config:
                config.cnn_weight_decay = cnn_config['weight_decay']
                print(f"   Weight decay: {cnn_config['weight_decay']}")
        else:
            print(f"⚠️  No CNN section in config, using defaults")
            config.cnn_kernel_sizes = getattr(config, 'cnn_kernel_sizes', [3, 5, 7])
            config.cnn_dilation_rates = getattr(config, 'cnn_dilation_rates', [1, 2, 4])
            config.cnn_use_residual = getattr(config, 'cnn_use_residual', True)
            config.cnn_batch_multiplier = getattr(config, 'cnn_batch_multiplier', 1.5)
            config.cnn_weight_decay = getattr(config, 'cnn_weight_decay', 0.005)
        
        if args.cnn_kernel_sizes:
            config.cnn_kernel_sizes = args.cnn_kernel_sizes
            print(f"🏃‍♂️ CNN kernel sizes override: {args.cnn_kernel_sizes}")
        
        if args.cnn_layers:
            config.num_layers = args.cnn_layers
            print(f"🏃‍♂️ CNN layers override: {args.cnn_layers}")
        
        if args.cnn_scheduler:
            config.scheduler = args.cnn_scheduler
            print(f"🏃‍♂️ CNN scheduler override: {args.cnn_scheduler}")
        
        torch.manual_seed(config.seed)
        
        print(f"\n📋 CNN Configuration Summary:")
        print(f"   Experiment: {config.experiment_name}")
        print(f"   Model: hidden_dim={config.hidden_dim}, layers={config.num_layers}, dropout={config.dropout}")
        print(f"   ✅ Multi-layer CNN: {config.num_layers} blocks, inter-block_dropout={config.layer_dropout}")
        
        if config.attention_enabled:
            print(f"   🎯 Attention: {config.attention_type} type, {config.attention_heads} heads, dropout={config.attention_dropout}")
            if config.attention_type == 'boundary_aware':
                print(f"      🎯 Boundary-aware: temperature={config.boundary_temperature}")
            if config.positional_encoding:
                print(f"      ✅ Positional encoding enabled (max_len={config.max_seq_length})")
        
        print(f"   Training: batch_size={config.batch_size}, lr={config.learning_rate}, epochs={config.max_epochs}")
        
        if hasattr(config, 'scheduler'):
            print(f"   ✅ Scheduler: {config.scheduler} (min_lr={config.min_lr})")
        
        if config.weighted_sampling:
            print(f"   Anti-collapse: weighted_sampling={config.weighted_sampling}")
        
        if config.emergency_monitoring_enabled:
            print(f"   ✅ Emergency monitoring: CNN-optimized thresholds")
        
        if hasattr(config, 'calibration_methods') and config.calibration_methods:
            print(f"   🎯 Calibration: {', '.join(config.calibration_methods)}")
        
        print(f"   🏗️  CNN Architecture:")
        print(f"      Kernel sizes: {getattr(config, 'cnn_kernel_sizes', [3, 5, 7])}")
        print(f"      Dilation rates: {getattr(config, 'cnn_dilation_rates', [1, 2, 4])}")
        print(f"      Residual connections: {getattr(config, 'cnn_use_residual', True)}")
        print(f"      Batch multiplier: {getattr(config, 'cnn_batch_multiplier', 1.5)}")
        
        print(f"   Device: {config.device}")
        
        print_cnn_experiment_header(config)
        
        session_dir = create_cnn_session_directory(config)
        
        save_config_snapshot(config, session_dir)
        
        enhanced_config_path = session_dir / "cnn_enhanced_config_snapshot.yaml"
        import yaml
        
        enhanced_config = config.__dict__.copy()
        
        enhanced_config['model_type'] = 'CNN'
        enhanced_config['architecture'] = 'CNNTagger'
        enhanced_config['cnn_kernel_sizes'] = getattr(config, 'cnn_kernel_sizes', [3, 5, 7])
        enhanced_config['cnn_dilation_rates'] = getattr(config, 'cnn_dilation_rates', [1, 2, 4])
        enhanced_config['cnn_use_residual'] = getattr(config, 'cnn_use_residual', True)
        enhanced_config['cnn_batch_multiplier'] = getattr(config, 'cnn_batch_multiplier', 1.5)
        enhanced_config['cnn_weight_decay'] = getattr(config, 'cnn_weight_decay', 0.005)
        
        # Add training strategy indicators
        enhanced_config['scheduler_type'] = getattr(config, 'scheduler', 'onecycle')
        enhanced_config['validation_strategy'] = getattr(config, 'validation_strategy', 'cnn_composite')
        enhanced_config['convergence_window'] = getattr(config, 'convergence_window', 3)
        
        # Add CNN optimization flags
        enhanced_config['cnn_optimized'] = True
        enhanced_config['emergency_monitoring_cnn_tuned'] = config.emergency_monitoring_enabled
        
        with open(enhanced_config_path, 'w') as f:
            yaml.dump(enhanced_config, f, default_flow_style=False, indent=2)
        
        print(f"📝 Enhanced CNN config snapshot saved: cnn_enhanced_config_snapshot.yaml")
        
        # Setup feature extractor
        print(f"🧩 Setting up feature extraction for CNN...")
        feature_config = {
            'head_ssm': {
                'enabled': config.head_ssm_enabled,
                'head_words': config.head_ssm_words,
                'output_dim': config.head_ssm_dimension
            },
            'tail_ssm': {
                'enabled': config.tail_ssm_enabled,
                'tail_words': config.tail_ssm_words,
                'output_dim': config.tail_ssm_dimension
            },
            'phonetic_ssm': {
                'enabled': config.phonetic_ssm_enabled,
                'mode': config.phonetic_ssm_mode,
                'output_dim': config.phonetic_ssm_dimension,
                'similarity_method': config.phonetic_ssm_similarity_method,
                'normalize': config.phonetic_ssm_normalize,
                'normalize_method': config.phonetic_ssm_normalize_method,
                'high_sim_threshold': config.phonetic_ssm_high_sim_threshold
            },
            'pos_ssm': {
                'enabled': config.pos_ssm_enabled,
                'tagset': config.pos_ssm_tagset,
                'similarity_method': config.pos_ssm_similarity_method,
                'high_sim_threshold': config.pos_ssm_high_sim_threshold,
                'output_dim': config.pos_ssm_dimension
            },
            'string_ssm': {
                'enabled': config.string_ssm_enabled,
                'case_sensitive': config.string_ssm_case_sensitive,
                'remove_punctuation': config.string_ssm_remove_punctuation,
                'similarity_threshold': config.string_ssm_similarity_threshold,
                'similarity_method': config.string_ssm_similarity_method,
                'output_dim': config.string_ssm_dimension
            },
            'syllable_pattern_ssm': {
                'enabled': config.syllable_pattern_ssm_enabled,
                'similarity_method': config.syllable_pattern_ssm_similarity_method,
                'levenshtein_weight': config.syllable_pattern_ssm_levenshtein_weight,
                'cosine_weight': config.syllable_pattern_ssm_cosine_weight,
                'normalize': config.syllable_pattern_ssm_normalize,
                'normalize_method': config.syllable_pattern_ssm_normalize_method,
                'dimension': config.syllable_pattern_ssm_dimension
            },
            'line_syllable_ssm': {
                'enabled': config.line_syllable_ssm_enabled,
                'similarity_method': config.line_syllable_ssm_similarity_method,
                'ratio_threshold': config.line_syllable_ssm_ratio_threshold,
                'normalize': config.line_syllable_ssm_normalize,
                'normalize_method': config.line_syllable_ssm_normalize_method,
                'dimension': config.line_syllable_ssm_dimension
            },
            # Embedding features
            'word2vec_enabled': config.word2vec_enabled,
            'word2vec_model': config.word2vec_model,
            'word2vec_mode': config.word2vec_mode,
            'word2vec_normalize': config.word2vec_normalize,
            'word2vec_similarity_metric': config.word2vec_similarity_metric,
            'word2vec_high_sim_threshold': config.word2vec_high_sim_threshold,
            'contextual_enabled': config.contextual_enabled,
            'contextual_model': config.contextual_model,
            'contextual_mode': config.contextual_mode,
            'contextual_normalize': config.contextual_normalize,
            'contextual_similarity_metric': config.contextual_similarity_metric,
            'contextual_high_sim_threshold': config.contextual_high_sim_threshold
        }
        feature_extractor = FeatureExtractor(feature_config)
        feature_dim = feature_extractor.get_feature_dimension()
        print(f"✅ CNN Feature dimension: {feature_dim}")
        
        print(f"🔧 CNN Feature configuration details:")
        enabled_features = []
        if config.head_ssm_enabled:
            enabled_features.append(f"Head-SSM({config.head_ssm_dimension}D)")
        if config.tail_ssm_enabled:
            enabled_features.append(f"Tail-SSM({config.tail_ssm_dimension}D)")
        if config.phonetic_ssm_enabled:
            enabled_features.append(f"Phonetic-SSM({config.phonetic_ssm_dimension}D)")
        if config.pos_ssm_enabled:
            enabled_features.append(f"POS-SSM({config.pos_ssm_dimension}D)")
        if config.string_ssm_enabled:
            enabled_features.append(f"String-SSM({config.string_ssm_dimension}D)")
        if config.syllable_pattern_ssm_enabled:
            enabled_features.append(f"SyllablePattern-SSM({config.syllable_pattern_ssm_dimension}D)")
        if config.line_syllable_ssm_enabled:
            enabled_features.append(f"LineSyllable-SSM({config.line_syllable_ssm_dimension}D)")
        if config.word2vec_enabled:
            w2v_dim = 12 if config.word2vec_mode == "summary" else 300
            enabled_features.append(f"Word2Vec({w2v_dim}D)")
        if config.contextual_enabled:
            ctx_dim = 12 if config.contextual_mode == "summary" else 384
            enabled_features.append(f"Contextual({ctx_dim}D)")
        
        if enabled_features:
            print(f"   CNN Active features: {', '.join(enabled_features)}")
        else:
            print(f"   ⚠️ No features enabled for CNN!")
        
        train_loader, val_loader, test_loader, train_dataset = setup_data_loaders(config, feature_extractor)
        
        model, loss_function, optimizer = setup_cnn_model_and_training(
            config, train_dataset, config.device, feature_dim
        )
        
        # Create CNN trainer
        print(f"🚀 Setting up CNN trainer...")
        trainer = CNNTrainer(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            device=config.device,
            config=config,
            output_dir=session_dir,
            disable_emergency_monitoring=not config.emergency_monitoring_enabled
        )
        
        # Start CNN training
        print(f"🎓 Starting CNN training...")
        start_time = time.time()
        
        best_model, calibration_info = trainer.train(train_loader, val_loader)
        
        training_time = time.time() - start_time
        print(f"⏱️  Total CNN training time: {training_time/60:.1f} minutes")
        
        print(f"📊 Final CNN evaluation on test set...")
        
        calibrator = None
        if calibration_info:
            try:
                from segmodel.train.calibration import load_calibration
                calibration_path = session_dir / 'calibration.json'
                if calibration_path.exists():
                    method, calibrator = load_calibration(calibration_path)
                    print(f"✅ Using CNN calibration for test evaluation: {method}")
                else:
                    print("⚠️  No CNN calibration.json found for test evaluation")
            except Exception as e:
                print(f"⚠️  Could not load CNN calibration: {e}")
        
        test_results = trainer.evaluate(test_loader, calibrator=calibrator)
        
        print(f"\n🎯 Final CNN Test Results:")
        print(f"   Macro F1: {test_results['macro_f1']:.4f}")
        print(f"   Verse F1: {test_results['verse_f1']:.4f}")  
        print(f"   Chorus F1: {test_results['chorus_f1']:.4f}")
        print(f"   Boundary F1: {test_results.get('boundary_f1', 0.0):.4f}")
        print(f"   Segment Overlap: {test_results.get('avg_segment_overlap', 0.0):.4f}")
        print(f"   Avg confidence: {test_results['max_prob_mean']:.3f}")
        print(f"   Chorus rate: {test_results['chorus_rate']:.2%}")
        print(f"   CNN Calibration: {list(calibration_info.keys()) if calibration_info else 'none'}")
        
        results_file = session_dir / "final_cnn_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"CNN Training Results - {config.experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Configuration: {args.config}\n")
            f.write(f"Training time: {training_time/60:.1f} minutes\n")
            f.write(f"Feature dimension: {feature_dim}\n\n")
            
            f.write("CNN Model Architecture:\n")
            f.write("-" * 22 + "\n")
            f.write(f"  Architecture: CNN (Convolutional Neural Network)\n")
            f.write(f"  Input dimension: {feature_dim}D\n")
            f.write(f"  Hidden dimension: {config.hidden_dim}D\n")
            f.write(f"  CNN layers: {config.num_layers}\n")
            f.write(f"  Kernel sizes: {getattr(config, 'cnn_kernel_sizes', [3, 5, 7])}\n")
            f.write(f"  Dilation rates: {getattr(config, 'cnn_dilation_rates', [1, 2, 4])}\n")
            f.write(f"  Residual connections: {getattr(config, 'cnn_use_residual', True)}\n")
            f.write(f"  Output dropout: {config.dropout}\n")
            f.write(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
            
            if config.attention_enabled:
                f.write(f"  Attention type: {config.attention_type}\n")
                f.write(f"  Attention heads: {config.attention_heads}\n")
                f.write(f"  Positional encoding: {config.positional_encoding}\n")
            else:
                f.write(f"  Attention: disabled\n")
            
            f.write(f"\nCNN Training Configuration:\n")
            f.write("-" * 27 + "\n")
            f.write(f"  Batch size: {config.batch_size}\n")
            f.write(f"  Learning rate: {config.learning_rate}\n")
            f.write(f"  Scheduler: {getattr(config, 'scheduler', 'onecycle')}\n")
            f.write(f"  Max epochs: {config.max_epochs}\n")
            f.write(f"  Validation strategy: {getattr(config, 'validation_strategy', 'cnn_composite')}\n")
            
            # CNN Test Results
            f.write(f"\nCNN Test Results:\n")
            f.write("-" * 17 + "\n")
            f.write(f"  Macro F1: {test_results['macro_f1']:.4f}\n")
            f.write(f"  Verse F1: {test_results['verse_f1']:.4f}\n")
            f.write(f"  Chorus F1: {test_results['chorus_f1']:.4f}\n")
            f.write(f"  Boundary F1: {test_results.get('boundary_f1', 0.0):.4f}\n")
            f.write(f"  Segment Overlap: {test_results.get('avg_segment_overlap', 0.0):.4f}\n")
            f.write(f"  WindowDiff: {test_results.get('window_diff', 1.0):.4f}\n")
            f.write(f"  Pk metric: {test_results.get('pk_metric', 1.0):.4f}\n")
            f.write(f"  Confidence: {test_results['max_prob_mean']:.3f}\n")
            f.write(f"  Chorus rate: {test_results['chorus_rate']:.2%}\n")
            f.write(f"  Calibration: {list(calibration_info.keys()) if calibration_info else 'none'}\n")
            
            if 'receptive_field_usage' in test_results:
                f.write(f"\nCNN-Specific Metrics:\n")
                f.write("-" * 21 + "\n")
                f.write(f"  Receptive field usage: {test_results['receptive_field_usage']:.1%}\n")
                f.write(f"  Effective layers: {test_results.get('effective_layers', 0)}\n")
                f.write(f"  Gradient stability: {test_results.get('gradient_stability', 0.0):.3f}\n")
        
        print(f"\n✅ CNN Training completed successfully!")
        print(f"📁 All CNN files saved to: {session_dir}")
        print(f"🏆 Best CNN model: {session_dir}/best_cnn_model.pt")
        
        final_model_name = f"cnn_{config.experiment_name}_{config.hidden_dim}d_{config.num_layers}layers_final.pt"
        final_model_path = session_dir / final_model_name
        
        import shutil
        best_model_path = session_dir / "best_cnn_model.pt"
        if best_model_path.exists():
            shutil.copy2(best_model_path, final_model_path)
            print(f"🎯 Final CNN model also saved as: {final_model_name}")
        
        metadata_file = session_dir / "cnn_model_metadata.json"
        import json
        metadata = {
            "model_type": "CNN",
            "architecture": "CNNTagger",
            "experiment_name": config.experiment_name,
            "model_files": {
                "best_model": "best_cnn_model.pt",
                "final_model": final_model_name,
                "config_snapshot": "training_config_snapshot.yaml"
            },
            "architecture_config": {
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "kernel_sizes": getattr(config, 'cnn_kernel_sizes', [3, 5, 7]),
                "dilation_rates": getattr(config, 'cnn_dilation_rates', [1, 2, 4]),
                "use_residual": getattr(config, 'cnn_use_residual', True),
                "batch_multiplier": getattr(config, 'cnn_batch_multiplier', 1.5)
            },
            "attention_config": {
                "enabled": config.attention_enabled,
                "type": getattr(config, 'attention_type', None),
                "heads": getattr(config, 'attention_heads', None),
                "dropout": getattr(config, 'attention_dropout', None)
            },
            "training_config": {
                "scheduler": getattr(config, 'scheduler', 'onecycle'),
                "validation_strategy": getattr(config, 'validation_strategy', 'cnn_composite'),
                "emergency_monitoring": config.emergency_monitoring_enabled
            },
            "feature_dimension": feature_dim,
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "final_results": test_results,
            "training_time_minutes": training_time/60
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 CNN metadata saved: cnn_model_metadata.json")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ CNN Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
