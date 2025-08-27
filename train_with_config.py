"""
BLSTM training script with YAML configuration support.

This script loads training parameters from YAML configuration files
and supports command line overrides for key parameters.
"""

import argparse
import torch
from torch.optim import AdamW
from pathlib import Path
import time
from datetime import datetime

from segmodel.utils import TrainingConfig, load_training_config, merge_with_args, save_config_snapshot
from segmodel.data import SongsDataset, create_dataloader
from segmodel.features import FeatureExtractor
from segmodel.train import Trainer


def setup_data_loaders(config: TrainingConfig, feature_extractor):
    """Setup data loaders from configuration."""
    
    print(f"📚 Loading datasets...")
    
    # Load datasets
    train_dataset = SongsDataset(config.train_file)
    val_dataset = SongsDataset(config.val_file)
    test_dataset = SongsDataset(config.test_file)
    
    print(f"✅ Loaded {len(train_dataset)} training songs")
    print(f"✅ Loaded {len(val_dataset)} validation songs")  
    print(f"✅ Loaded {len(test_dataset)} test songs")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        feature_extractor,
        batch_size=config.batch_size,
        shuffle=True,
        use_weighted_sampling=config.weighted_sampling,
        num_workers=config.num_workers
    )
    
    val_loader = create_dataloader(
        val_dataset,
        feature_extractor,
        batch_size=config.batch_size, 
        shuffle=False,
        use_weighted_sampling=False,
        num_workers=config.num_workers
    )
    
    test_loader = create_dataloader(
        test_dataset,
        feature_extractor,
        batch_size=config.batch_size,
        shuffle=False, 
        use_weighted_sampling=False,
        num_workers=config.num_workers
    )
    
    return train_loader, val_loader, test_loader, train_dataset


def setup_model_and_training(config: TrainingConfig, train_dataset: SongsDataset, device: str, feature_dim: int):
    """Setup model, loss function, and optimizer from configuration."""
    
    from segmodel.models import BLSTMTagger
    
    # Create model
    print(f"🤖 Creating model...")
    model = BLSTMTagger(
        feat_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        dropout=config.dropout,
        layer_dropout=config.layer_dropout,
        # NEW: Attention parameters
        attention_enabled=config.attention_enabled,
        attention_type=config.attention_type,
        attention_heads=config.attention_heads,
        attention_dropout=config.attention_dropout,
        attention_dim=config.attention_dim,
        positional_encoding=config.positional_encoding,
        max_seq_length=config.max_seq_length,
        window_size=config.window_size,
        boundary_temperature=config.boundary_temperature
    ).to(device)
    
    # Print detailed model information
    print(f"\n🤖 Model Architecture Details:")
    model.print_model_info()
    
    print(f"\n📊 Architecture Summary:")
    print(f"   Input features: {feature_dim}D")
    print(f"   Hidden dimension: {config.hidden_dim}D (shared across all layers)")
    print(f"   LSTM layers: {config.num_layers}")
    if config.num_layers > 1:
        print(f"   ✅ Multi-layer BiLSTM: {config.num_layers} layers")
        if config.layer_dropout > 0:
            print(f"      Inter-layer dropout: {config.layer_dropout}")
        else:
            print(f"      No inter-layer dropout")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"      Total parameters: {total_params:,}")
    else:
        print(f"   Single-layer BiLSTM (backward compatible)")
    
    if config.attention_enabled:
        print(f"   🎯 Attention mechanism:")
        print(f"      Heads: {config.attention_heads}")
        print(f"      Dropout: {config.attention_dropout}")
        if config.attention_dim:
            print(f"      Dimension: {config.attention_dim}")
        else:
            print(f"      Dimension: auto (LSTM output = {config.hidden_dim * 2})")
        print(f"      Positional encoding: {config.positional_encoding}")
        if hasattr(model, 'attention') and model.attention is not None:
            attention_params = sum(p.numel() for p in model.attention.parameters())
            total_params = sum(p.numel() for p in model.parameters())
            print(f"      Attention parameters: {attention_params:,} ({(attention_params/total_params*100):.1f}% of total)")
    else:
        print(f"   🎯 Attention: disabled")
    
    print(f"   Output dropout: {config.dropout}")
    print(f"   Output classes: {config.num_classes}")
    
    print(f"\n🎯 Loss Function Configuration:")
    print(f"=" * 70)
    
    if not hasattr(config, 'loss'):
        raise ValueError(
            "❌ Config must include 'loss' section!\n"
            "   Please update your YAML config to include:\n"
            "   loss:\n"
            "     type: 'boundary_aware_cross_entropy'  # or 'cross_entropy'\n"
            "     label_smoothing: 0.16\n"
            "     # ... other loss parameters\n"
            "   See: documentation/boundary_aware_loss_migration_roadmap.md"
        )
    
    loss_config = config.loss
    loss_type = loss_config.get('type', 'boundary_aware_cross_entropy')
    
    print(f"   ✅ Loss Configuration Found")
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
        
        # Create boundary-aware loss function
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
        
        print(f"\n   � Boundary-Aware Loss Architecture:")
        print(f"      Base Loss: Cross-Entropy (label_smoothing={label_smoothing:.3f})")
        print(f"      Class Weights: {class_weights.cpu().numpy()}")
        print(f"      ├── Boundary Weight: {boundary_weight:.1f}x {'✅ ACTIVE' if boundary_weight > 1.0 else '❌ DISABLED'}")
        print(f"      ├── Segment Consistency: λ={segment_consistency_lambda:.3f} {'✅ ACTIVE' if segment_consistency_lambda > 0 else '❌ DISABLED'}")
        print(f"      ├── Confidence Penalty: λ={conf_penalty_lambda:.3f} (threshold={conf_threshold:.2f}) {'✅ ACTIVE' if conf_penalty_lambda > 0 else '❌ DISABLED'}")
        print(f"      ├── Entropy Regularization: λ={entropy_lambda:.3f} {'✅ ACTIVE' if entropy_lambda > 0 else '❌ DISABLED'}")
        print(f"      └── Architecture: {'🎯 Boundary-Primary' if use_boundary_as_primary else '📊 Cross-Entropy Primary'}")
            
    elif loss_type == 'cross_entropy':
        entropy_lambda = loss_config.get('entropy_lambda', 0.0)
        
        from segmodel.losses.cross_entropy import create_loss_function as create_legacy_loss
        loss_function = create_legacy_loss(
            num_classes=config.num_classes,
            label_smoothing=label_smoothing,
            class_weights=class_weights,
            entropy_lambda=entropy_lambda
        )
        
        print(f"\n   � Legacy Cross-Entropy Loss:")
        print(f"      Base Loss: Cross-Entropy (label_smoothing={label_smoothing:.3f})")
        print(f"      Class Weights: {class_weights.cpu().numpy()}")
        print(f"      Entropy Regularization: λ={entropy_lambda:.3f} {'✅ ACTIVE' if entropy_lambda > 0 else '❌ DISABLED'}")
        print(f"\n   ⚠️  Consider upgrading to 'boundary_aware_cross_entropy' for:")
        print(f"      • Better boundary detection")
        print(f"      • Improved segmentation quality")
        print(f"      • Enhanced confidence calibration")
        
    else:
        raise ValueError(f"❌ Unknown loss type: {loss_type}. Supported: 'boundary_aware_cross_entropy', 'cross_entropy'")
    
    print(f"=" * 70)
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    print(f"⚙️  Training setup:")
    print(f"   Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
    print(f"   Scheduler: {getattr(config, 'scheduler', 'plateau')} (configured in YAML)")
    print(f"   Gradient clipping: {config.gradient_clip_norm}")
    
    return model, loss_function, optimizer


def create_session_directory(config: TrainingConfig) -> Path:
    """Create timestamped session directory."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"session_{timestamp}_{config.experiment_name}"
    session_dir = Path(config.output_base_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Session directory: {session_dir}")
    
    return session_dir


def print_experiment_header(config: TrainingConfig):
    """Print experiment information."""
    
    print(f"\n🎵 BLSTM Training with YAML Config")
    print("=" * 50)
    print(f"🧪 Experiment: {config.experiment_name}")
    print(f"📄 Description: {config.experiment_description}")
    
    if config.experiment_tags:
        print(f"🏷️  Tags: {', '.join(config.experiment_tags)}")
    
    if config.experiment_notes:
        print(f"📝 Notes: {config.experiment_notes}")
    
    print(f"🍎 Device: {config.device}")


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(
        description="BLSTM training with YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python train_with_config.py configs/training_config.yaml
  
  # Quick test
  python train_with_config.py configs/quick_test.yaml
  
  # With overrides
  python train_with_config.py configs/training_config.yaml --batch-size 32 --epochs 100
  
  # Disable emergency monitoring
  python train_with_config.py configs/training_config.yaml --disable-emergency-monitoring
        """
    )
    
    # Required config file argument
    parser.add_argument('config', 
                       help='Path to YAML configuration file')
    
    # Optional overrides
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
    
    # Data file overrides
    parser.add_argument('--train',
                       help='Override training data file')
    parser.add_argument('--val', 
                       help='Override validation data file')
    parser.add_argument('--test',
                       help='Override test data file')
    
    args = parser.parse_args()
    
    try:
        
        config = load_training_config(args.config)
        
        config = merge_with_args(config, args)
        
        torch.manual_seed(config.seed)
        
        print_experiment_header(config)
        
        session_dir = create_session_directory(config)
        
        save_config_snapshot(config, session_dir)
        
        print(f"🧩 Setting up feature extraction...")
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
            # Embedding features in flattened format expected by FeatureExtractor
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
        print(f"✅ Feature dimension: {feature_dim}")
        
        print(f"🔧 Feature configuration details:")
        enabled_features = []
        if config.head_ssm_enabled:
            enabled_features.append(f"Head-SSM({config.head_ssm_dimension}D)")
        if config.tail_ssm_enabled:
            enabled_features.append(f"Tail-SSM({config.tail_ssm_dimension}D)")
        if config.phonetic_ssm_enabled:
            phon_desc = f"Phonetic-SSM({config.phonetic_ssm_dimension}D,{config.phonetic_ssm_mode},{config.phonetic_ssm_similarity_method}"
            if config.phonetic_ssm_normalize:
                phon_desc += f",norm-{config.phonetic_ssm_normalize_method}"
            phon_desc += f",th={config.phonetic_ssm_high_sim_threshold})"
            enabled_features.append(phon_desc)
        if config.pos_ssm_enabled:
            pos_desc = f"POS-SSM({config.pos_ssm_dimension}D,{config.pos_ssm_tagset},{config.pos_ssm_similarity_method},th={config.pos_ssm_high_sim_threshold})"
            enabled_features.append(pos_desc)
        if config.string_ssm_enabled:
            string_desc = f"String-SSM({config.string_ssm_dimension}D"
            if config.string_ssm_case_sensitive:
                string_desc += ",case_sens"
            if not config.string_ssm_remove_punctuation:
                string_desc += ",keep_punct"
            if config.string_ssm_similarity_threshold > 0:
                string_desc += f",th={config.string_ssm_similarity_threshold}"
            string_desc += ")"
            enabled_features.append(string_desc)
        
        # Syllable SSM features
        if config.syllable_pattern_ssm_enabled:
            syl_desc = f"SyllablePattern-SSM({config.syllable_pattern_ssm_dimension}D,{config.syllable_pattern_ssm_similarity_method}"
            if config.syllable_pattern_ssm_similarity_method == "combined":
                syl_desc += f",lev={config.syllable_pattern_ssm_levenshtein_weight},cos={config.syllable_pattern_ssm_cosine_weight}"
            if config.syllable_pattern_ssm_normalize:
                syl_desc += f",norm-{config.syllable_pattern_ssm_normalize_method}"
            syl_desc += ")"
            enabled_features.append(syl_desc)
        
        if config.line_syllable_ssm_enabled:
            line_syl_desc = f"LineSyllable-SSM({config.line_syllable_ssm_dimension}D,{config.line_syllable_ssm_similarity_method}"
            if config.line_syllable_ssm_normalize:
                line_syl_desc += f",norm-{config.line_syllable_ssm_normalize_method}"
            line_syl_desc += f",ratio={config.line_syllable_ssm_ratio_threshold})"
            enabled_features.append(line_syl_desc)
        
        # Embedding features
        if config.word2vec_enabled:
            w2v_dim = 12 if config.word2vec_mode == "summary" else 300
            w2v_desc = f"Word2Vec({w2v_dim}D,{config.word2vec_model},{config.word2vec_mode}"
            if config.word2vec_normalize:
                w2v_desc += ",norm"
            w2v_desc += f",{config.word2vec_similarity_metric},th={config.word2vec_high_sim_threshold})"
            enabled_features.append(w2v_desc)
        
        if config.contextual_enabled:
            ctx_dim = 12 if config.contextual_mode == "summary" else 384
            ctx_desc = f"Contextual({ctx_dim}D,{config.contextual_model},{config.contextual_mode}"
            if config.contextual_normalize:
                ctx_desc += ",norm"
            ctx_desc += f",{config.contextual_similarity_metric},th={config.contextual_high_sim_threshold})"
            enabled_features.append(ctx_desc)
        
        if enabled_features:
            print(f"   Active features: {', '.join(enabled_features)}")
        else:
            print(f"   ⚠️ No features enabled!")
        
        if config.pos_ssm_enabled:
            print(f"   🧩 POS-SSM config: {config.pos_ssm_tagset} tags, {config.pos_ssm_similarity_method} similarity, threshold={config.pos_ssm_high_sim_threshold}")
        if config.string_ssm_enabled:
            print(f"   🧩 String-SSM config: case_sensitive={config.string_ssm_case_sensitive}, remove_punctuation={config.string_ssm_remove_punctuation}, threshold={config.string_ssm_similarity_threshold}")
        if config.word2vec_enabled:
            print(f"   🔤 Word2Vec config: {config.word2vec_model}, mode={config.word2vec_mode}, normalize={config.word2vec_normalize}, {config.word2vec_similarity_metric}, threshold={config.word2vec_high_sim_threshold}")
        if config.contextual_enabled:
            print(f"   🤖 Contextual config: {config.contextual_model}, mode={config.contextual_mode}, normalize={config.contextual_normalize}, {config.contextual_similarity_metric}, threshold={config.contextual_high_sim_threshold}")
        
        if config.syllable_pattern_ssm_enabled:
            print(f"   🎵 SyllablePattern-SSM config: method={config.syllable_pattern_ssm_similarity_method}, normalize={config.syllable_pattern_ssm_normalize}")
            if config.syllable_pattern_ssm_similarity_method == "combined":
                print(f"      Combined weights: Levenshtein={config.syllable_pattern_ssm_levenshtein_weight}, Cosine={config.syllable_pattern_ssm_cosine_weight}")
        if config.line_syllable_ssm_enabled:
            print(f"   🎼 LineSyllable-SSM config: method={config.line_syllable_ssm_similarity_method}, ratio_threshold={config.line_syllable_ssm_ratio_threshold}, normalize={config.line_syllable_ssm_normalize}")
        
        train_loader, val_loader, test_loader, train_dataset = setup_data_loaders(config, feature_extractor)
        
        model, loss_function, optimizer = setup_model_and_training(
            config, train_dataset, config.device, feature_dim
        )
        
        print(f"🚀 Setting up trainer...")
        trainer = Trainer(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            device=config.device,
            config=config,
            output_dir=session_dir,
            disable_emergency_monitoring=not config.emergency_monitoring_enabled
        )
        
        print(f"🎓 Starting training...")
        start_time = time.time()
        
        best_model, calibration_info = trainer.train(train_loader, val_loader)
        
        training_time = time.time() - start_time
        print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
        
        print(f"📊 Final evaluation on test set...")
        
        calibrator = None
        if calibration_info:
            try:
                from segmodel.train.calibration import load_calibration
                calibration_path = session_dir / 'calibration.json'
                if calibration_path.exists():
                    method, calibrator = load_calibration(calibration_path)
                    print(f"✅ Using calibration for test evaluation: {method}")
                else:
                    print("⚠️  No calibration.json found for test evaluation")
            except Exception as e:
                print(f"⚠️  Could not load calibration for test evaluation: {e}")
        
        test_results = trainer.evaluate(test_loader, calibrator=calibrator)
        
        print(f"\n🎯 Final Test Results:")
        print(f"   Macro F1: {test_results['macro_f1']:.4f}")
        print(f"   Verse F1: {test_results['verse_f1']:.4f}")  
        print(f"   Chorus F1: {test_results['chorus_f1']:.4f}")
        print(f"   Avg confidence: {test_results['max_prob_mean']:.3f}")
        print(f"   Chorus rate: {test_results['chorus_rate']:.2%}")
        print(f"   Calibration: {list(calibration_info.keys()) if calibration_info else 'none'}")
        
        results_file = session_dir / "final_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"BLSTM Training Results - {config.experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Configuration: {args.config}\n")
            f.write(f"Training time: {training_time/60:.1f} minutes\n")
            f.write(f"Feature dimension: {feature_dim}\n\n")
            
            f.write("Feature Configuration:\n")
            f.write("-" * 25 + "\n")
            
            if config.head_ssm_enabled:
                f.write(f"  Head-SSM: Enabled ({config.head_ssm_dimension}D)\n")
                f.write(f"    Head words: {config.head_ssm_words}\n")
            else:
                f.write(f"  Head-SSM: Disabled\n")
            
            if config.tail_ssm_enabled:
                f.write(f"  Tail-SSM: Enabled ({config.tail_ssm_dimension}D)\n")
                f.write(f"    Tail words: {config.tail_ssm_words}\n")
            else:
                f.write(f"  Tail-SSM: Disabled\n")
            
            if config.phonetic_ssm_enabled:
                f.write(f"  Phonetic-SSM: Enabled ({config.phonetic_ssm_dimension}D)\n")
                f.write(f"    Mode: {config.phonetic_ssm_mode}\n")
                f.write(f"    Similarity: {config.phonetic_ssm_similarity_method}\n")
                f.write(f"    Normalize: {config.phonetic_ssm_normalize}")
                if config.phonetic_ssm_normalize:
                    f.write(f" ({config.phonetic_ssm_normalize_method})")
                f.write("\n")
                f.write(f"    High sim threshold: {config.phonetic_ssm_high_sim_threshold}\n")
            else:
                f.write(f"  Phonetic-SSM: Disabled\n")
            
            if config.pos_ssm_enabled:
                f.write(f"  POS-SSM: Enabled ({config.pos_ssm_dimension}D)\n")
                f.write(f"    Tagset: {config.pos_ssm_tagset}\n")
                f.write(f"    Similarity: {config.pos_ssm_similarity_method}\n")
                f.write(f"    High sim threshold: {config.pos_ssm_high_sim_threshold}\n")
            else:
                f.write(f"  POS-SSM: Disabled\n")
            
            if config.string_ssm_enabled:
                f.write(f"  String-SSM: Enabled ({config.string_ssm_dimension}D)\n")
                f.write(f"    Case sensitive: {config.string_ssm_case_sensitive}\n")
                f.write(f"    Remove punctuation: {config.string_ssm_remove_punctuation}\n")
                f.write(f"    Similarity threshold: {config.string_ssm_similarity_threshold}\n")
                f.write(f"    Similarity method: {config.string_ssm_similarity_method}\n")
            else:
                f.write(f"  String-SSM: Disabled\n")
            
            if config.syllable_pattern_ssm_enabled:
                f.write(f"  SyllablePattern-SSM: Enabled ({config.syllable_pattern_ssm_dimension}D)\n")
                f.write(f"    Similarity method: {config.syllable_pattern_ssm_similarity_method}\n")
                if config.syllable_pattern_ssm_similarity_method == "combined":
                    f.write(f"    Levenshtein weight: {config.syllable_pattern_ssm_levenshtein_weight}\n")
                    f.write(f"    Cosine weight: {config.syllable_pattern_ssm_cosine_weight}\n")
                f.write(f"    Normalize: {config.syllable_pattern_ssm_normalize}")
                if config.syllable_pattern_ssm_normalize:
                    f.write(f" ({config.syllable_pattern_ssm_normalize_method})")
                f.write("\n")
            else:
                f.write(f"  SyllablePattern-SSM: Disabled\n")
            
            if config.line_syllable_ssm_enabled:
                f.write(f"  LineSyllable-SSM: Enabled ({config.line_syllable_ssm_dimension}D)\n")
                f.write(f"    Similarity method: {config.line_syllable_ssm_similarity_method}\n")
                f.write(f"    Ratio threshold: {config.line_syllable_ssm_ratio_threshold}\n")
                f.write(f"    Normalize: {config.line_syllable_ssm_normalize}")
                if config.line_syllable_ssm_normalize:
                    f.write(f" ({config.line_syllable_ssm_normalize_method})")
                f.write("\n")
            else:
                f.write(f"  LineSyllable-SSM: Disabled\n")
            
            f.write(f"\n")
            
            if config.word2vec_enabled:
                w2v_dim = 12 if config.word2vec_mode == "summary" else 300
                f.write(f"  Word2Vec Embeddings: Enabled ({w2v_dim}D)\n")
                f.write(f"    Model: {config.word2vec_model}\n")
                f.write(f"    Mode: {config.word2vec_mode} ({w2v_dim}D {'statistical features' if config.word2vec_mode == 'summary' else 'full embeddings'})\n")
                f.write(f"    Normalize: {config.word2vec_normalize}\n")
                f.write(f"    Similarity metric: {config.word2vec_similarity_metric}\n")
                f.write(f"    High similarity threshold: {config.word2vec_high_sim_threshold}\n")
            else:
                f.write(f"  Word2Vec Embeddings: Disabled\n")
            
            if config.contextual_enabled:
                ctx_dim = 12 if config.contextual_mode == "summary" else 384
                f.write(f"  Contextual Embeddings: Enabled ({ctx_dim}D)\n")
                f.write(f"    Model: {config.contextual_model}\n")
                f.write(f"    Mode: {config.contextual_mode} ({ctx_dim}D {'statistical features' if config.contextual_mode == 'summary' else 'full embeddings'})\n")
                f.write(f"    Normalize: {config.contextual_normalize}\n")
                f.write(f"    Similarity metric: {config.contextual_similarity_metric}\n")
                f.write(f"    High similarity threshold: {config.contextual_high_sim_threshold}\n")
            else:
                f.write(f"  Contextual Embeddings: Disabled\n")
            
            f.write(f"\n")
            enabled_feature_dims = []
            if config.head_ssm_enabled:
                enabled_feature_dims.append(f"Head-SSM: {config.head_ssm_dimension}D")
            if config.tail_ssm_enabled:
                enabled_feature_dims.append(f"Tail-SSM: {config.tail_ssm_dimension}D")
            if config.phonetic_ssm_enabled:
                enabled_feature_dims.append(f"Phonetic-SSM: {config.phonetic_ssm_dimension}D")
            if config.pos_ssm_enabled:
                enabled_feature_dims.append(f"POS-SSM: {config.pos_ssm_dimension}D")
            if config.string_ssm_enabled:
                enabled_feature_dims.append(f"String-SSM: {config.string_ssm_dimension}D")
            if config.syllable_pattern_ssm_enabled:
                enabled_feature_dims.append(f"SyllablePattern-SSM: {config.syllable_pattern_ssm_dimension}D")
            if config.line_syllable_ssm_enabled:
                enabled_feature_dims.append(f"LineSyllable-SSM: {config.line_syllable_ssm_dimension}D")
            if config.word2vec_enabled:
                w2v_dim = 12 if config.word2vec_mode == "summary" else 300
                enabled_feature_dims.append(f"Word2Vec: {w2v_dim}D")
            if config.contextual_enabled:
                ctx_dim = 12 if config.contextual_mode == "summary" else 384
                enabled_feature_dims.append(f"Contextual: {ctx_dim}D")
            
            if enabled_feature_dims:
                f.write(f"  Total Feature Dimension: {feature_dim}D ({' + '.join(enabled_feature_dims)})\n")
            
            f.write(f"\nModel Architecture:\n")
            f.write("-" * 18 + "\n")
            f.write(f"  Input dimension: {feature_dim}D\n")
            f.write(f"  Hidden dimension: {config.hidden_dim}D\n")
            f.write(f"  LSTM layers: {config.num_layers}\n")
            if config.num_layers > 1:
                f.write(f"  ✅ Multi-layer BiLSTM architecture\n")
                if config.layer_dropout > 0:
                    f.write(f"      Inter-layer dropout: {config.layer_dropout}\n")
                else:
                    f.write(f"      No inter-layer dropout\n")
                total_params = sum(p.numel() for p in model.parameters())
                f.write(f"      Total parameters: {total_params:,}\n")
            else:
                f.write(f"  Single-layer BiLSTM (backward compatible)\n")
                total_params = sum(p.numel() for p in model.parameters())
                f.write(f"      Total parameters: {total_params:,}\n")
            
            f.write(f"  Attention type: {config.attention_type}\n")
            if config.attention_type == 'localized':
                f.write(f"    Window size: {config.window_size}\n")
            elif config.attention_type == 'boundary_aware':
                f.write(f"    Boundary temperature: {config.boundary_temperature}\n")
            f.write(f"  Positional encoding: {config.positional_encoding}\n")
            if config.positional_encoding:
                f.write(f"    PE max length: {config.max_seq_length}\n")
            
            f.write(f"  Output dropout: {config.dropout}\n")
            f.write(f"  Batch size: {config.batch_size}\n")
            f.write(f"  Learning rate: {config.learning_rate}\n")

            if config.loss_config.loss_type == 'cross_entropy':
                f.write(f"  Loss type: {config.loss_config.loss_type}\n")
                f.write(f"  Label smoothing: {config.loss_config.label_smoothing}\n")
            elif config.loss_config.loss_type == 'boundary_aware_cross_entropy':
                f.write(f"  Loss type: {config.loss_config.loss_type}\n")
                f.write(f"  Boundary weight: {config.loss_config.boundary_weight}\n")
                f.write(f"  Entropy lambda: {config.loss_config.entropy_lambda}\n")
                f.write(f"  Segment consistency lambda: {config.loss_config.segment_consistency_lambda}\n")
            f.write(f"  Weighted sampling: {config.weighted_sampling}\n")
            
            f.write(f"\nTest Results:\n")
            f.write("-" * 13 + "\n")
            f.write(f"  Macro F1: {test_results['macro_f1']:.4f}\n")
            f.write(f"  Verse F1: {test_results['verse_f1']:.4f}\n")
            f.write(f"  Chorus F1: {test_results['chorus_f1']:.4f}\n")
            f.write(f"  Confidence: {test_results['max_prob_mean']:.3f}\n")
            f.write(f"  Chorus rate: {test_results['chorus_rate']:.2%}\n")
            f.write(f"  Calibration: {list(calibration_info.keys()) if calibration_info else 'none'}\n")
            
            if calibration_info:
                f.write(f"\nCalibration Details:\n")
                f.write("-" * 20 + "\n")
                for method, result in calibration_info.items():
                    if 'ece_before' in result and 'ece_after' in result:
                        ece_before = result['ece_before']
                        ece_after = result['ece_after']
                        improvement = ece_before - ece_after
                        f.write(f"  {method}: ECE {ece_before:.4f} → {ece_after:.4f} (Δ{improvement:+.4f})\n")
                        if method == 'temperature' and 'params' in result:
                            f.write(f"    T = {result['params']['temperature']:.3f}\n")
                        elif method == 'platt' and 'params' in result:
                            A = result['params']['A']; B = result['params']['B']
                            f.write(f"    A = {A:.3f}, B = {B:.3f}\n")
                        elif method == 'isotonic' and 'params' in result:
                            knots = result['params']['knots']
                            f.write(f"    Knots = {knots}\n")
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 All files saved to: {session_dir}")
        print(f"🏆 Best model: {session_dir}/best_model.pt")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
