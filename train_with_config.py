#!/usr/bin/env python3
"""
BLSTM training script with YAML configuration support.

This script loads training parameters from YAML configuration files
and supports command line overrides for key parameters.
"""

import argparse
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import time
from datetime import datetime

# Import our modules
from segmodel.utils import TrainingConfig, load_training_config, merge_with_args, save_config_snapshot
from segmodel.data import SongsDataset, create_dataloader
from segmodel.features import FeatureExtractor
from segmodel.models import create_model
from segmodel.losses import create_loss_function
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
    
    # Create model using YAML configuration directly
    from segmodel.models import BLSTMTagger
    
    # Create model
    print(f"🤖 Creating model...")
    model = BLSTMTagger(
        feat_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(device)
    
    print(f"🤖 Model Architecture:")
    print(f"   Input dimension: {feature_dim}")
    print(f"   Hidden dimension: {config.hidden_dim}")
    print(f"   Output classes: {config.num_classes}")
    print(f"   Dropout: {config.dropout}")
    
    # Create loss function
    print(f"🎯 Setting up loss function...")
    class_weights = train_dataset.get_class_weights().to(device)
    loss_function = create_loss_function(
        num_classes=config.num_classes,
        label_smoothing=config.label_smoothing,
        class_weights=class_weights,
        entropy_lambda=config.entropy_lambda
    )
    
    print(f"🔧 Loss function configuration:")
    print(f"   Label smoothing: {config.label_smoothing}")
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    print(f"   Entropy lambda: {config.entropy_lambda}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=config.patience // 2
    )
    
    print(f"⚙️  Training setup:")
    print(f"   Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
    print(f"   Scheduler: ReduceLROnPlateau")
    print(f"   Gradient clipping: {config.gradient_clip_norm}")
    
    return model, loss_function, optimizer, scheduler


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
        # Load configuration
        config = load_training_config(args.config)
        
        # Apply command line overrides
        config = merge_with_args(config, args)
        
        # Set random seed
        torch.manual_seed(config.seed)
        
        # Print experiment info
        print_experiment_header(config)
        
        # Create session directory
        session_dir = create_session_directory(config)
        
        # Save config snapshot for reproducibility
        save_config_snapshot(config, session_dir)
        
        # Setup feature extractor first
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
            }
        }
        feature_extractor = FeatureExtractor(feature_config)
        feature_dim = feature_extractor.get_feature_dimension()
        print(f"✅ Feature dimension: {feature_dim}")
        
        # Show detailed feature configuration
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
        
        if enabled_features:
            print(f"   Active features: {', '.join(enabled_features)}")
        else:
            print(f"   ⚠️ No features enabled!")
        
        if config.pos_ssm_enabled:
            print(f"   🧩 POS-SSM config: {config.pos_ssm_tagset} tags, {config.pos_ssm_similarity_method} similarity, threshold={config.pos_ssm_high_sim_threshold}")
        if config.string_ssm_enabled:
            print(f"   🧩 String-SSM config: case_sensitive={config.string_ssm_case_sensitive}, remove_punctuation={config.string_ssm_remove_punctuation}, threshold={config.string_ssm_similarity_threshold}")
        
        # Setup data loaders
        train_loader, val_loader, test_loader, train_dataset = setup_data_loaders(config, feature_extractor)
        
        # Setup model and training components
        model, loss_function, optimizer, scheduler = setup_model_and_training(
            config, train_dataset, config.device, feature_dim
        )
        
        # Create trainer
        print(f"🚀 Setting up trainer...")
        trainer = Trainer(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            device=config.device,
            config=config,  # Use YAML config directly
            output_dir=session_dir,
            disable_emergency_monitoring=not config.emergency_monitoring_enabled
        )
        
        # Start training
        print(f"🎓 Starting training...")
        start_time = time.time()
        
        best_model, best_temperature = trainer.train(train_loader, val_loader)
        
        training_time = time.time() - start_time
        print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
        
        # Final evaluation
        print(f"📊 Final evaluation on test set...")
        test_results = trainer.evaluate(test_loader)
        
        # Print final results
        print(f"\n🎯 Final Test Results:")
        print(f"   Macro F1: {test_results['macro_f1']:.4f}")
        print(f"   Verse F1: {test_results['verse_f1']:.4f}")  
        print(f"   Chorus F1: {test_results['chorus_f1']:.4f}")
        print(f"   Avg confidence: {test_results['max_prob_mean']:.3f}")
        print(f"   Chorus rate: {test_results['chorus_rate']:.2%}")
        print(f"   Optimal temperature: {best_temperature:.2f}")
        
        # Save final results
        results_file = session_dir / "final_results.txt"
        with open(results_file, 'w') as f:
            f.write(f"BLSTM Training Results - {config.experiment_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Configuration: {args.config}\n")
            f.write(f"Training time: {training_time/60:.1f} minutes\n")
            f.write(f"Feature dimension: {feature_dim}\n\n")
            
            # Feature Configuration Summary
            f.write("Feature Configuration:\n")
            f.write("-" * 25 + "\n")
            
            # Head-SSM
            if config.head_ssm_enabled:
                f.write(f"  Head-SSM: Enabled ({config.head_ssm_dimension}D)\n")
                f.write(f"    Head words: {config.head_ssm_words}\n")
            else:
                f.write(f"  Head-SSM: Disabled\n")
            
            # Tail-SSM  
            if config.tail_ssm_enabled:
                f.write(f"  Tail-SSM: Enabled ({config.tail_ssm_dimension}D)\n")
                f.write(f"    Tail words: {config.tail_ssm_words}\n")
            else:
                f.write(f"  Tail-SSM: Disabled\n")
            
            # Phonetic-SSM
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
            
            # POS-SSM
            if config.pos_ssm_enabled:
                f.write(f"  POS-SSM: Enabled ({config.pos_ssm_dimension}D)\n")
                f.write(f"    Tagset: {config.pos_ssm_tagset}\n")
                f.write(f"    Similarity: {config.pos_ssm_similarity_method}\n")
                f.write(f"    High sim threshold: {config.pos_ssm_high_sim_threshold}\n")
            else:
                f.write(f"  POS-SSM: Disabled\n")
            
            f.write(f"\nModel Architecture:\n")
            f.write("-" * 18 + "\n")
            f.write(f"  Hidden dimension: {config.hidden_dim}\n")
            f.write(f"  Dropout: {config.dropout}\n")
            f.write(f"  Batch size: {config.batch_size}\n")
            f.write(f"  Learning rate: {config.learning_rate}\n")
            f.write(f"  Label smoothing: {config.label_smoothing}\n")
            f.write(f"  Weighted sampling: {config.weighted_sampling}\n")
            
            f.write(f"\nTest Results:\n")
            f.write("-" * 13 + "\n")
            f.write(f"  Macro F1: {test_results['macro_f1']:.4f}\n")
            f.write(f"  Verse F1: {test_results['verse_f1']:.4f}\n")
            f.write(f"  Chorus F1: {test_results['chorus_f1']:.4f}\n")
            f.write(f"  Confidence: {test_results['max_prob_mean']:.3f}\n")
            f.write(f"  Chorus rate: {test_results['chorus_rate']:.2%}\n")
            f.write(f"  Temperature: {best_temperature:.2f}\n")
        
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
