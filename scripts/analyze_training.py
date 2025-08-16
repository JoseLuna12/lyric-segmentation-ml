#!/usr/bin/env python3
"""
Analyze and visualize training metrics from training_metrics.json files.
"""

import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np


def load_training_metrics(metrics_file: str) -> pd.DataFrame:
    """Load training metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        raise ValueError(f"No metrics found in {metrics_file}")
    
    df = pd.DataFrame(data)
    return df


def plot_training_metrics(df: pd.DataFrame, output_dir: Path = None):
    """Create comprehensive training analysis plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('BLSTM Training Analysis', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, df['train_loss'], 'b-', label='Training', linewidth=2)
    ax.plot(epochs, df['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. F1 Scores
    ax = axes[0, 1]
    ax.plot(epochs, df['val_macro_f1'], 'g-', label='Macro F1', linewidth=2)
    ax.plot(epochs, df['val_verse_f1'], 'b-', label='Verse F1', linewidth=2)
    ax.plot(epochs, df['val_chorus_f1'], 'r-', label='Chorus F1', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Scores (Validation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 3. Chorus Rate (Class Balance)
    ax = axes[0, 2]
    ax.plot(epochs, df['train_chorus_rate'], 'b-', label='Training', linewidth=2)
    ax.plot(epochs, df['val_chorus_rate'], 'r-', label='Validation', linewidth=2)
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.7, label='Expected (~30%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Chorus Rate')
    ax.set_title('Chorus Rate (Class Balance)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 4. Confidence Calibration
    ax = axes[1, 0]
    ax.plot(epochs, df['train_max_prob'], 'b-', label='Training', linewidth=2)
    ax.plot(epochs, df['val_max_prob'], 'r-', label='Validation', linewidth=2)
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Overconfidence threshold')
    ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='Target confidence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Max Probability')
    ax.set_title('Confidence Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    
    # 5. Overconfidence Tracking
    ax = axes[1, 1]
    ax.plot(epochs, df['train_conf_over_90'] * 100, 'b-', label='Train >90%', linewidth=2)
    ax.plot(epochs, df['val_conf_over_90'] * 100, 'r-', label='Val >90%', linewidth=2)
    ax.plot(epochs, df['train_conf_over_95'] * 100, 'b--', label='Train >95%', linewidth=2)
    ax.plot(epochs, df['val_conf_over_95'] * 100, 'r--', label='Val >95%', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Overconfident Predictions (%)')
    ax.set_title('Overconfidence Monitoring')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Training Efficiency
    ax = axes[1, 2]
    ax.plot(epochs, df['epoch_time'], 'purple', linewidth=2, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Epoch Training Time')
    ax.grid(True, alpha=0.3)
    
    # Add learning rate if available
    if 'learning_rate' in df.columns:
        ax2 = ax.twinx()
        ax2.plot(epochs, df['learning_rate'], 'orange', linewidth=1, alpha=0.7, label='LR')
        ax2.set_ylabel('Learning Rate', color='orange')
        ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if output_dir:
        plot_file = output_dir / "training_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Training analysis plot saved to: {plot_file}")
    
    plt.show()


def print_training_summary(df: pd.DataFrame):
    """Print a comprehensive training summary."""
    
    print("\n🎵 BLSTM Training Summary")
    print("=" * 50)
    
    # Basic info
    total_epochs = len(df)
    final_metrics = df.iloc[-1]
    
    print(f"📅 Training Duration:")
    print(f"   Total epochs: {total_epochs}")
    print(f"   Total time: {df['epoch_time'].sum():.1f}s ({df['epoch_time'].sum()/60:.1f} minutes)")
    print(f"   Avg time per epoch: {df['epoch_time'].mean():.1f}s")
    
    # Performance metrics
    print(f"\n📈 Final Performance:")
    print(f"   Validation Macro F1: {final_metrics['val_macro_f1']:.4f}")
    print(f"   Validation Verse F1: {final_metrics['val_verse_f1']:.4f}")
    print(f"   Validation Chorus F1: {final_metrics['val_chorus_f1']:.4f}")
    print(f"   Training Loss: {final_metrics['train_loss']:.4f}")
    print(f"   Validation Loss: {final_metrics['val_loss']:.4f}")
    
    # Best metrics
    best_f1_epoch = df['val_macro_f1'].idxmax() + 1
    best_f1 = df['val_macro_f1'].max()
    
    print(f"\n🏆 Best Results:")
    print(f"   Best Macro F1: {best_f1:.4f} (epoch {best_f1_epoch})")
    print(f"   Best Chorus F1: {df['val_chorus_f1'].max():.4f}")
    print(f"   Lowest Train Loss: {df['train_loss'].min():.4f}")
    print(f"   Lowest Val Loss: {df['val_loss'].min():.4f}")
    
    # Anti-collapse metrics
    print(f"\n🛡️  Anti-Collapse Status:")
    print(f"   Final train confidence: {final_metrics['train_max_prob']:.3f}")
    print(f"   Final val confidence: {final_metrics['val_max_prob']:.3f}")
    print(f"   Max overconf (>95%): {df['train_conf_over_95'].max():.2%} train, {df['val_conf_over_95'].max():.2%} val")
    print(f"   Final chorus rate: {final_metrics['train_chorus_rate']:.2%} train, {final_metrics['val_chorus_rate']:.2%} val")
    
    # Training stability
    loss_variance = df['train_loss'].std()
    f1_trend = df['val_macro_f1'].iloc[-1] - df['val_macro_f1'].iloc[0] if len(df) > 1 else 0
    
    print(f"\n📊 Training Stability:")
    print(f"   Loss std deviation: {loss_variance:.4f}")
    print(f"   F1 improvement: {f1_trend:+.4f}")
    print(f"   Learning rate: {final_metrics['learning_rate']:.2e}")
    
    # Warnings and recommendations
    print(f"\n⚠️  Analysis:")
    
    if final_metrics['val_max_prob'] > 0.95:
        print("   🚨 High confidence detected - consider temperature scaling")
    elif final_metrics['val_max_prob'] > 0.90:
        print("   ⚠️  Moderately high confidence - monitor for overconfidence")
    else:
        print("   ✅ Confidence well-calibrated")
    
    if final_metrics['val_chorus_rate'] < 0.15:
        print("   🚨 Low chorus rate - possible class collapse")
    elif final_metrics['val_chorus_rate'] > 0.60:
        print("   ⚠️  High chorus rate - check class balance")
    else:
        print("   ✅ Chorus rate balanced")
    
    if best_f1 > 0.75:
        print("   🎉 Excellent performance achieved!")
    elif best_f1 > 0.65:
        print("   ✅ Good performance - consider longer training")
    else:
        print("   📈 Room for improvement - consider architecture changes")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BLSTM training metrics and create visualizations"
    )
    
    parser.add_argument('metrics_file', 
                       help='Path to training_metrics.json file')
    parser.add_argument('--output-dir', 
                       help='Directory to save plots (optional)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting, just show summary')
    
    args = parser.parse_args()
    
    # Load metrics
    try:
        df = load_training_metrics(args.metrics_file)
        print(f"✅ Loaded metrics from {args.metrics_file}")
        print(f"   Found {len(df)} epochs of training data")
        
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
        return
    
    # Print summary
    print_training_summary(df)
    
    # Create plots
    if not args.no_plot:
        try:
            output_dir = Path(args.output_dir) if args.output_dir else None
            plot_training_metrics(df, output_dir)
            
        except ImportError:
            print("\n⚠️  matplotlib not available - skipping plots")
            print("   Install with: pip install matplotlib pandas")
        
        except Exception as e:
            print(f"\n⚠️  Error creating plots: {e}")


if __name__ == "__main__":
    main()
