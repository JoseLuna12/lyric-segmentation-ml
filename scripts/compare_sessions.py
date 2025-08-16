#!/usr/bin/env python3
"""
Compare multiple training sessions and find the best results.
"""

import json
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime


def load_session_summary(session_dir: Path) -> dict:
    """Load summary info for a training session."""
    
    metrics_file = session_dir / "training_metrics.json"
    config_file = session_dir / "config.txt"
    results_file = session_dir / "final_results.txt"
    
    summary = {
        'session_name': session_dir.name,
        'session_path': str(session_dir),
        'timestamp': session_dir.name.split('_')[-1] if '_' in session_dir.name else 'unknown',
        'status': 'unknown'
    }
    
    # Load metrics
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            if metrics:
                final_metrics = metrics[-1]
                summary.update({
                    'epochs': len(metrics),
                    'final_train_loss': final_metrics.get('train_loss', 0),
                    'final_val_loss': final_metrics.get('val_loss', 0),
                    'final_macro_f1': final_metrics.get('val_macro_f1', 0),
                    'final_verse_f1': final_metrics.get('val_verse_f1', 0),
                    'final_chorus_f1': final_metrics.get('val_chorus_f1', 0),
                    'final_confidence': final_metrics.get('val_max_prob', 0),
                    'final_chorus_rate': final_metrics.get('val_chorus_rate', 0),
                    'total_time': sum(m.get('epoch_time', 0) for m in metrics),
                    'best_macro_f1': max(m.get('val_macro_f1', 0) for m in metrics),
                    'status': 'completed'
                })
            else:
                summary['status'] = 'empty_metrics'
                
        except Exception as e:
            summary['status'] = f'error_loading_metrics: {e}'
    
    else:
        summary['status'] = 'no_metrics_file'
    
    # Add config info if available
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_content = f.read()
            
            # Extract key config values
            for line in config_content.split('\n'):
                if 'epochs:' in line:
                    summary['config_epochs'] = line.split(':')[-1].strip()
                elif 'batch_size:' in line:
                    summary['config_batch_size'] = line.split(':')[-1].strip()
                elif 'lr:' in line and 'learning' not in line.lower():
                    summary['config_lr'] = line.split(':')[-1].strip()
                    
        except Exception:
            pass
    
    return summary


def find_training_sessions(base_dir: str = "training_sessions") -> list:
    """Find all training sessions in the base directory."""
    
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    sessions = []
    for session_dir in base_path.iterdir():
        if session_dir.is_dir() and session_dir.name.startswith('session_'):
            summary = load_session_summary(session_dir)
            sessions.append(summary)
    
    # Sort by timestamp (newest first)
    sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    return sessions


def print_session_comparison(sessions: list):
    """Print a comparison table of training sessions."""
    
    if not sessions:
        print("No training sessions found.")
        return
    
    print(f"\n🎵 Training Sessions Comparison")
    print("=" * 100)
    
    # Filter completed sessions for the main table
    completed_sessions = [s for s in sessions if s['status'] == 'completed']
    
    if completed_sessions:
        print(f"\n✅ Completed Sessions ({len(completed_sessions)} found):")
        print("-" * 100)
        
        # Create header
        header = f"{'Session':<20} {'Epochs':<7} {'Best F1':<8} {'Chorus F1':<9} {'Confidence':<11} {'Time':<8} {'Status':<10}"
        print(header)
        print("-" * 100)
        
        # Print each completed session
        for session in completed_sessions:
            session_name = session['session_name'][-15:]  # Show last 15 chars
            epochs = session.get('epochs', 0)
            best_f1 = session.get('best_macro_f1', 0)
            chorus_f1 = session.get('final_chorus_f1', 0)
            confidence = session.get('final_confidence', 0)
            time_mins = session.get('total_time', 0) / 60
            status = "✅ Complete"
            
            row = f"{session_name:<20} {epochs:<7} {best_f1:<8.3f} {chorus_f1:<9.3f} {confidence:<11.3f} {time_mins:<8.1f} {status:<10}"
            print(row)
    
    # Show problematic sessions
    problematic_sessions = [s for s in sessions if s['status'] != 'completed']
    if problematic_sessions:
        print(f"\n⚠️  Problematic Sessions ({len(problematic_sessions)} found):")
        print("-" * 60)
        
        for session in problematic_sessions:
            session_name = session['session_name'][-20:]
            status = session['status']
            print(f"  {session_name:<25} {status}")
    
    # Show best results
    if completed_sessions:
        best_f1_session = max(completed_sessions, key=lambda x: x.get('best_macro_f1', 0))
        best_chorus_session = max(completed_sessions, key=lambda x: x.get('final_chorus_f1', 0))
        
        print(f"\n🏆 Best Results:")
        print(f"   Best Macro F1: {best_f1_session['best_macro_f1']:.4f} in {best_f1_session['session_name']}")
        print(f"   Best Chorus F1: {best_chorus_session['final_chorus_f1']:.4f} in {best_chorus_session['session_name']}")
        print(f"   Best session path: {best_f1_session['session_path']}")


def print_detailed_session_info(session_name: str, base_dir: str = "training_sessions"):
    """Print detailed information about a specific session."""
    
    session_path = Path(base_dir) / session_name
    if not session_path.exists():
        print(f"❌ Session not found: {session_name}")
        return
    
    summary = load_session_summary(session_path)
    
    print(f"\n🔍 Detailed Session Info: {session_name}")
    print("=" * 50)
    
    print(f"📁 Path: {summary['session_path']}")
    print(f"📅 Timestamp: {summary['timestamp']}")
    print(f"📊 Status: {summary['status']}")
    
    if summary['status'] == 'completed':
        print(f"\n📈 Performance:")
        print(f"   Epochs trained: {summary['epochs']}")
        print(f"   Final Macro F1: {summary['final_macro_f1']:.4f}")
        print(f"   Best Macro F1: {summary['best_macro_f1']:.4f}")
        print(f"   Final Verse F1: {summary['final_verse_f1']:.4f}")
        print(f"   Final Chorus F1: {summary['final_chorus_f1']:.4f}")
        
        print(f"\n🛡️  Anti-Collapse Metrics:")
        print(f"   Final confidence: {summary['final_confidence']:.3f}")
        print(f"   Final chorus rate: {summary['final_chorus_rate']:.2%}")
        
        print(f"\n⏱️  Training Time:")
        print(f"   Total time: {summary['total_time']:.1f}s ({summary['total_time']/60:.1f} minutes)")
        print(f"   Avg time/epoch: {summary['total_time']/summary['epochs']:.1f}s")
        
        # Show files available
        files = list(session_path.iterdir())
        print(f"\n📄 Available files:")
        for file in sorted(files):
            if file.is_file():
                print(f"   {file.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare BLSTM training sessions"
    )
    
    parser.add_argument('--base-dir', default='training_sessions',
                       help='Base directory containing training sessions')
    parser.add_argument('--detail', 
                       help='Show detailed info for specific session')
    
    args = parser.parse_args()
    
    if args.detail:
        print_detailed_session_info(args.detail, args.base_dir)
    else:
        sessions = find_training_sessions(args.base_dir)
        print_session_comparison(sessions)
        
        if sessions:
            print(f"\n💡 Tips:")
            print(f"   • Use --detail <session_name> for detailed info")
            print(f"   • Use scripts/analyze_training.py <session>/training_metrics.json for plots")
            print(f"   • Best model is typically in <session>/best_model.pt")


if __name__ == "__main__":
    main()
