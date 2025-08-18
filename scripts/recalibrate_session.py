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
from segmodel.features.extractor import create_feature_extractor
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
    
    # Create model with correct parameters
    model = BLSTMTagger(
        feat_dim=60,  # Feature dimension from your system
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 2),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.1),
        layer_dropout=config.get('layer_dropout', 0.0)
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✅ Model loaded successfully")
    return model, config


def create_validation_loader(config: dict) -> torch.utils.data.DataLoader:
    """Create validation data loader with feature extractor."""
    
    # Create feature extractor (standard configuration)
    feature_config = {
        'head_ssm': {'enabled': True, 'output_dim': 12},
        'tail_ssm': {'enabled': True, 'output_dim': 12},
        'string_ssm': {'enabled': True, 'output_dim': 12},
        'pos_ssm': {'enabled': True, 'output_dim': 12},
        'phonetic_ssm': {'enabled': True, 'output_dim': 12}
    }
    
    feature_extractor = create_feature_extractor(feature_config)
    
    # Load validation dataset
    val_file = config.get('val_file', 'data/val.jsonl')
    val_dataset = SongsDataset(val_file)
    
    # Create data loader
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
        model, config = load_model_from_session(session_dir)
        model.to(device)
        model.eval()
        
        # Create validation loader
        val_loader = create_validation_loader(config)
        
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
