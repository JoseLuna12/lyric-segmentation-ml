#!/usr/bin/env python3
"""
Run calibration on a trained model from a specific training session.

This script loads a trained model and runs calibration methods (temperature, Platt, isotonic)
on the validation set, producing identical results to the original training calibration.
Results are saved to calibration.json in the session directory.

Usage:
    python scripts/recalibrate_session.py <session_name>
    
Example:
    python scripts/recalibrate_session.py session_20250818_194333_training_validation_2layer_optimized_new_calibration_v1
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
from segmodel.models.blstm_tagger import BLSTMTagger
from segmodel.data import SongsDataset, create_dataloader
from segmodel.features.extractor import create_feature_extractor, FeatureExtractor
from segmodel.train.calibration import fit_calibration
import yaml
import json
import numpy as np


def find_session_directory(session_name: str) -> Path:
    """Find the training session directory."""
    sessions_dir = project_root / 'training_sessions'
    if not sessions_dir.exists():
        raise FileNotFoundError(f"Training sessions directory not found: {sessions_dir}")
    
    # Try exact match first
    session_path = sessions_dir / session_name
    if session_path.exists():
        return session_path
    
    # Try partial match
    matching_dirs = [d for d in sessions_dir.iterdir() 
                    if d.is_dir() and session_name in d.name]
    
    if not matching_dirs:
        available = [d.name for d in sessions_dir.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"No session found matching '{session_name}'\n"
            f"Available sessions: {available}"
        )
    
    if len(matching_dirs) > 1:
        matches = [d.name for d in matching_dirs]
        raise ValueError(
            f"Multiple sessions match '{session_name}': {matches}\n"
            f"Please be more specific."
        )
    
    return matching_dirs[0]


def load_model_from_session(session_dir: Path) -> tuple[BLSTMTagger, dict]:
    """Load model and config from session directory."""
    
    # Find model file (prefer best_model.pt, fallback to final_model.pt)
    model_path = session_dir / 'best_model.pt'
    if not model_path.exists():
        model_path = session_dir / 'final_model.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"No model found in {session_dir}")
    
    # Find config file
    config_path = session_dir / 'training_config_snapshot.yaml'
    if not config_path.exists():
        config_path = session_dir / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"No config found in {session_dir}")
    
    print(f"📁 Loading from session: {session_dir.name}")
    print(f"🔧 Model: {model_path.name}")
    print(f"⚙️  Config: {config_path.name}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert flattened config to feature config format expected by FeatureExtractor
    feature_config = {}
    
    # SSM features
    if config.get('head_ssm_enabled', False):
        feature_config['head_ssm'] = {
            'enabled': True,
            'output_dim': config.get('head_ssm_dimension', 12),
            'head_words': config.get('head_ssm_words', 2)
        }
    
    if config.get('tail_ssm_enabled', False):
        feature_config['tail_ssm'] = {
            'enabled': True,
            'output_dim': config.get('tail_ssm_dimension', 12),
            'tail_words': config.get('tail_ssm_words', 2)
        }
    
    if config.get('phonetic_ssm_enabled', False):
        feature_config['phonetic_ssm'] = {
            'enabled': True,
            'output_dim': config.get('phonetic_ssm_dimension', 12),
            'mode': config.get('phonetic_ssm_mode', 'rhyme'),
            'similarity_method': config.get('phonetic_ssm_similarity_method', 'binary'),
            'normalize': config.get('phonetic_ssm_normalize', False),
            'normalize_method': config.get('phonetic_ssm_normalize_method', 'zscore'),
            'high_sim_threshold': config.get('phonetic_ssm_high_sim_threshold', 0.8)
        }
    
    if config.get('pos_ssm_enabled', False):
        feature_config['pos_ssm'] = {
            'enabled': True,
            'output_dim': config.get('pos_ssm_dimension', 12),
            'tagset': config.get('pos_ssm_tagset', 'simplified'),
            'similarity_method': config.get('pos_ssm_similarity_method', 'combined'),
            'high_sim_threshold': config.get('pos_ssm_high_sim_threshold', 0.7)
        }
    
    if config.get('string_ssm_enabled', False):
        feature_config['string_ssm'] = {
            'enabled': True,
            'output_dim': config.get('string_ssm_dimension', 12),
            'case_sensitive': config.get('string_ssm_case_sensitive', False),
            'remove_punctuation': config.get('string_ssm_remove_punctuation', True),
            'similarity_threshold': config.get('string_ssm_similarity_threshold', 0.0),
            'similarity_method': config.get('string_ssm_similarity_method', 'word_overlap')
        }
    
    if config.get('syllable_pattern_ssm_enabled', False):
        feature_config['syllable_pattern_ssm'] = {
            'enabled': True,
            'dimension': config.get('syllable_pattern_ssm_dimension', 12),
            'similarity_method': config.get('syllable_pattern_ssm_similarity_method', 'levenshtein'),
            'levenshtein_weight': config.get('syllable_pattern_ssm_levenshtein_weight', 0.7),
            'cosine_weight': config.get('syllable_pattern_ssm_cosine_weight', 0.3),
            'normalize': config.get('syllable_pattern_ssm_normalize', False),
            'normalize_method': config.get('syllable_pattern_ssm_normalize_method', 'zscore')
        }
    
    if config.get('line_syllable_ssm_enabled', False):
        feature_config['line_syllable_ssm'] = {
            'enabled': True,
            'dimension': config.get('line_syllable_ssm_dimension', 12),
            'similarity_method': config.get('line_syllable_ssm_similarity_method', 'cosine'),
            'ratio_threshold': config.get('line_syllable_ssm_ratio_threshold', 0.1),
            'normalize': config.get('line_syllable_ssm_normalize', False),
            'normalize_method': config.get('line_syllable_ssm_normalize_method', 'minmax')
        }
    
    # Embeddings - converted to flattened format FeatureExtractor expects
    if config.get('word2vec_enabled', False):
        feature_config['word2vec_enabled'] = True
        feature_config['word2vec_model'] = config.get('word2vec_model', 'word2vec-google-news-300')
        feature_config['word2vec_mode'] = config.get('word2vec_mode', 'summary')
        feature_config['word2vec_normalize'] = config.get('word2vec_normalize', True)
        feature_config['word2vec_similarity_metric'] = config.get('word2vec_similarity_metric', 'cosine')
        feature_config['word2vec_high_sim_threshold'] = config.get('word2vec_high_sim_threshold', 0.8)
    
    if config.get('contextual_enabled', False):
        feature_config['contextual_enabled'] = True
        feature_config['contextual_model'] = config.get('contextual_model', 'all-MiniLM-L6-v2')
        feature_config['contextual_mode'] = config.get('contextual_mode', 'summary')
        feature_config['contextual_normalize'] = config.get('contextual_normalize', True)
        feature_config['contextual_similarity_metric'] = config.get('contextual_similarity_metric', 'cosine')
        feature_config['contextual_high_sim_threshold'] = config.get('contextual_high_sim_threshold', 0.7)
    
    # Create feature extractor to calculate dimension automatically
    from segmodel.features.extractor import FeatureExtractor
    feature_extractor = FeatureExtractor(feature_config=feature_config)
    feat_dim = feature_extractor.get_feature_dimension()
    
    print(f"🔧 Calculated feature dimension: {feat_dim}D")
    
    # Create model with correct parameters including attention
    model = BLSTMTagger(
        feat_dim=feat_dim,
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 2),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.1),
        layer_dropout=config.get('layer_dropout', 0.0),
        # Attention parameters
        attention_enabled=config.get('attention_enabled', False),
        attention_type=config.get('attention_type', 'self'),
        attention_heads=config.get('attention_heads', 8),
        attention_dim=config.get('attention_dim', 256),
        attention_dropout=config.get('attention_dropout', 0.1),
        positional_encoding=config.get('positional_encoding', False),
        max_seq_length=config.get('max_seq_length', 1000)
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✅ Model loaded successfully")
    return model, config, feature_extractor


def create_validation_loader(config: dict, feature_extractor: FeatureExtractor) -> torch.utils.data.DataLoader:
    """Create validation data loader with feature extractor from training config."""
    
    # Load validation dataset
    val_file = config.get('val_file', 'data/val.jsonl')
    val_dataset = SongsDataset(val_file)
    
    # Create data loader using the provided feature extractor
    val_loader = create_dataloader(
        val_dataset,
        feature_extractor,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        use_weighted_sampling=False,
        num_workers=0
    )
    
    print(f"📊 Validation: {len(val_dataset)} songs, {len(val_loader)} batches")
    return val_loader


def run_calibration(session_name: str) -> bool:
    """Run calibration for the specified session."""
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {device}")
        
        # Find and load session
        session_dir = find_session_directory(session_name)
        model, config, feature_extractor = load_model_from_session(session_dir)
        model.to(device)
        model.eval()
        
        # Create validation loader
        val_loader = create_validation_loader(config, feature_extractor)
        
        # Run calibration
        methods = ['temperature', 'platt', 'isotonic']
        print(f"\n🌡️  Running calibration with methods: {methods}")
        
        calibration_info = fit_calibration(
            model=model,
            val_loader=val_loader,
            device=device,
            methods=methods,
            output_dir=session_dir
        )
        
        # Display results
        print(f"\n📈 Calibration Results:")
        best_improvement = 0
        best_method = None
        
        for method, result in calibration_info.items():
            improvement = result['improvement']
            print(f"   {method.upper()}:")
            print(f"     ECE Before: {result['ece_before']:.4f}")
            print(f"     ECE After:  {result['ece_after']:.4f}")
            print(f"     Improvement: {improvement:+.4f}")
            
            status = "✅ SUCCESS" if improvement > 0 else "❌ NO IMPROVEMENT"
            print(f"     Status: {status}")
            
            if method == 'isotonic' and improvement > 0:
                print(f"     🎉 ISOTONIC CALIBRATION WORKING!")
            
            # Track best method
            if improvement > best_improvement:
                best_improvement = improvement
                best_method = method
        
        # Summary
        if best_method:
            print(f"\n🏆 Best method: {best_method.upper()} (improvement: {best_improvement:+.4f})")
        else:
            print(f"\n⚠️  No calibration method improved performance")
        
        print(f"\n💾 Calibration results saved to: calibration.json")
        print("\n✅ Recalibration completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Recalibration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run calibration on a trained model from a specific training session.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate latest session (if unique substring)
  python scripts/recalibrate_session.py 2layer_optimized
  
  # Calibrate specific session
  python scripts/recalibrate_session.py session_20250818_194333_training_validation_2layer_optimized_new_calibration_v1
        """
    )
    
    parser.add_argument(
        'session_name',
        nargs='?',  # Make session_name optional
        help='Training session name or unique substring'
    )
    
    parser.add_argument(
        '--list-sessions',
        action='store_true',
        help='List available training sessions'
    )
    
    args = parser.parse_args()
    
    # List sessions if requested
    if args.list_sessions:
        sessions_dir = project_root / 'training_sessions'
        if sessions_dir.exists():
            sessions = [d.name for d in sessions_dir.iterdir() if d.is_dir()]
            sessions.sort()
            print("Available training sessions:")
            for session in sessions:
                print(f"  {session}")
        else:
            print("No training sessions directory found")
        return
    
    # Check that session_name is provided when not listing
    if not args.session_name:
        parser.error("session_name is required unless --list-sessions is used")
    
    # Run calibration
    print(f"🔄 Starting recalibration for session: {args.session_name}")
    success = run_calibration(args.session_name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
