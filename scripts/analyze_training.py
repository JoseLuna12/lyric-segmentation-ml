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
import os
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')


def load_training_metrics(metrics_file: str) -> tuple[pd.DataFrame, dict]:
    """Load training metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    
    if not data:
        raise ValueError(f"No metrics found in {metrics_file}")
    
    # Handle new format with metadata and metrics
    if 'metrics' in data and 'metadata' in data:
        metadata = data['metadata']
        df = pd.DataFrame(data['metrics'])
    else:
        # Legacy format - just metrics
        metadata = {}
        df = pd.DataFrame(data)
    
    return df, metadata


def plot_training_metrics(df: pd.DataFrame, metadata: dict, output_dir: Path = None):
    """Create comprehensive training analysis plots."""
    
    # Create multiple figure windows for better organization
    
    # Figure 1: Core Training Metrics (2x2)
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Core Training Metrics', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # 1. Loss curves
    ax = axes1[0, 0]
    ax.plot(epochs, df['train_loss'], 'b-', label='Training', linewidth=2)
    ax.plot(epochs, df['val_loss'], 'r-', label='Validation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. F1 Scores
    ax = axes1[0, 1]
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
    ax = axes1[1, 0]
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
    ax = axes1[1, 1]
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
    
    plt.tight_layout()
    
    if output_dir:
        plot_file = output_dir / "core_training_metrics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Core training metrics plot saved to: {plot_file}")
    
    plt.show()
    
    # Figure 2: Boundary and Segmentation Metrics (2x2)
    if 'val_boundary_f1' in df.columns:
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle('Boundary & Segmentation Metrics', fontsize=16, fontweight='bold')
        
        # 1. Boundary Metrics
        ax = axes2[0, 0]
        ax.plot(epochs, df['val_boundary_f1'], 'g-', label='Boundary F1', linewidth=2)
        ax.plot(epochs, df['val_boundary_precision'], 'b-', label='Boundary Precision', linewidth=2)
        ax.plot(epochs, df['val_boundary_recall'], 'r-', label='Boundary Recall', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Boundary Detection Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 2. Segmentation Quality
        ax = axes2[0, 1]
        ax.plot(epochs, df['val_complete_segments'], 'purple', label='Complete Segments', linewidth=2)
        ax.plot(epochs, df['val_avg_segment_overlap'], 'orange', label='Avg Segment Overlap', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Segmentation Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 3. Transition Accuracy
        ax = axes2[1, 0]
        ax.plot(epochs, df['val_verse_to_chorus_acc'], 'b-', label='Verse → Chorus', linewidth=2)
        ax.plot(epochs, df['val_chorus_to_verse_acc'], 'r-', label='Chorus → Verse', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Transition Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 4. Distance Metrics
        ax = axes2[1, 1]
        ax.plot(epochs, df['val_window_diff'], 'red', label='WindowDiff', linewidth=2)
        ax.plot(epochs, df['val_pk_metric'], 'orange', label='Pk Metric', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error Rate')
        ax.set_title('Distance-based Metrics (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if output_dir:
            plot_file = output_dir / "boundary_segmentation_metrics.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📊 Boundary/segmentation metrics plot saved to: {plot_file}")
        
        plt.show()
    
    # Figure 3: Advanced Analysis (2x2)
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    fig3.suptitle('Advanced Training Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overconfidence Tracking
    ax = axes3[0, 0]
    ax.plot(epochs, df['train_conf_over_90'] * 100, 'b-', label='Train >90%', linewidth=2)
    ax.plot(epochs, df['val_conf_over_90'] * 100, 'r-', label='Val >90%', linewidth=2)
    ax.plot(epochs, df['train_conf_over_95'] * 100, 'b--', label='Train >95%', linewidth=2)
    ax.plot(epochs, df['val_conf_over_95'] * 100, 'r--', label='Val >95%', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Overconfident Predictions (%)')
    ax.set_title('Overconfidence Monitoring')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Training Efficiency
    ax = axes3[0, 1]
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
        ax2.legend(loc='upper right')
    
    # 3. Model Performance vs Confidence
    ax = axes3[1, 0]
    ax.scatter(df['val_max_prob'], df['val_macro_f1'], c=epochs, cmap='viridis', alpha=0.7)
    ax.set_xlabel('Max Confidence')
    ax.set_ylabel('Macro F1')
    ax.set_title('Performance vs Confidence')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Epoch')
    
    # 4. Loss vs F1 Evolution
    ax = axes3[1, 1]
    ax.scatter(df['val_loss'], df['val_macro_f1'], c=epochs, cmap='plasma', alpha=0.7)
    ax.set_xlabel('Validation Loss')
    ax.set_ylabel('Macro F1')
    ax.set_title('Loss vs F1 Evolution')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Epoch')
    
    plt.tight_layout()
    
    if output_dir:
        plot_file = output_dir / "advanced_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Advanced analysis plot saved to: {plot_file}")
    
    plt.show()


def plot_model_dynamics(df: pd.DataFrame, metadata: dict, output_dir: Path = None):
    """Plot model dynamics and training health metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Dynamics & Training Health', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # 1. Gradient Health (if available - using loss variance as proxy)
    ax = axes[0, 0]
    # Loss smoothness as gradient health indicator
    loss_diff = df['train_loss'].diff().abs()
    loss_smooth = df['train_loss'].rolling(window=3).std()
    
    ax.plot(epochs[1:], loss_diff[1:], 'b-', label='Loss Gradient', alpha=0.7)
    ax.plot(epochs, loss_smooth, 'r-', label='Loss Smoothness (3-epoch)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Change Magnitude')
    ax.set_title('Training Stability Indicators')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. Learning Rate Schedule Analysis
    ax = axes[0, 1]
    if 'learning_rate' in df.columns:
        lr = df['learning_rate']
        ax.plot(epochs, lr, 'orange', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Detect schedule phases
        lr_changes = lr.diff().abs()
        if lr_changes.sum() > 0:
            # Find warmup, stable, and decay phases
            warmup_end = lr.idxmax() if lr.idxmax() > 0 else 0
            if warmup_end > 0:
                ax.axvline(x=epochs[warmup_end], color='green', linestyle='--', alpha=0.7, label='Warmup End')
            
            # Find significant decay points
            significant_drops = lr_changes > lr_changes.quantile(0.9)
            for idx in epochs[significant_drops]:
                if idx > warmup_end:
                    ax.axvline(x=idx, color='red', linestyle=':', alpha=0.5)
            
            ax.legend()
    else:
        ax.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Learning Rate Schedule')
    
    # 3. Training Efficiency
    ax = axes[1, 0]
    time_smooth = df['epoch_time'].rolling(window=5).mean()
    ax.plot(epochs, df['epoch_time'], 'purple', alpha=0.5, label='Raw')
    ax.plot(epochs, time_smooth, 'purple', linewidth=2, label='Smoothed (5-epoch)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add efficiency trend
    if len(df) > 10:
        z = np.polyfit(epochs, df['epoch_time'], 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), 'red', linestyle='--', alpha=0.7, 
                label=f'Trend: {z[0]:.2f}s/epoch')
        ax.legend()
    
    # 4. Model Confidence Evolution
    ax = axes[1, 1]
    ax.plot(epochs, df['train_max_prob'], 'b-', label='Train Confidence', linewidth=2)
    ax.plot(epochs, df['val_max_prob'], 'r-', label='Val Confidence', linewidth=2)
    
    # Confidence bands
    ax.fill_between(epochs, 0.5, 0.7, alpha=0.1, color='green', label='Underconfident')
    ax.fill_between(epochs, 0.85, 0.95, alpha=0.1, color='orange', label='Target Range')
    ax.fill_between(epochs, 0.95, 1.0, alpha=0.1, color='red', label='Overconfident')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Max Probability')
    ax.set_title('Confidence Calibration Evolution')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    
    if output_dir:
        plot_file = output_dir / "model_dynamics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Model dynamics plot saved to: {plot_file}")
    
    plt.show()


def plot_calibration_analysis(df: pd.DataFrame, metadata: dict, output_dir: Path = None):
    """Plot calibration and reliability analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Calibration & Reliability Analysis', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # 1. Calibration Error Evolution (using confidence as proxy)
    ax = axes[0, 0]
    # ECE approximation using overconfidence metrics
    train_ece_proxy = df['train_conf_over_90'] + 2 * df['train_conf_over_95']
    val_ece_proxy = df['val_conf_over_90'] + 2 * df['val_conf_over_95']
    
    ax.plot(epochs, train_ece_proxy * 100, 'b-', label='Train ECE (proxy)', linewidth=2)
    ax.plot(epochs, val_ece_proxy * 100, 'r-', label='Val ECE (proxy)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Expected Calibration Error (%)')
    ax.set_title('Calibration Error Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Confidence Distribution Evolution
    ax = axes[0, 1]
    # Create confidence bins visualization
    conf_90 = df['val_conf_over_90'] * 100
    conf_95 = df['val_conf_over_95'] * 100
    conf_80_90 = conf_90 - conf_95  # Approximation
    conf_under_80 = 100 - conf_90
    
    ax.stackplot(epochs, conf_under_80, conf_80_90, conf_95, 
                labels=['<80%', '80-90%', '>95%'], alpha=0.7,
                colors=['lightgreen', 'orange', 'red'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Percentage of Predictions')
    ax.set_title('Confidence Distribution Evolution')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. Performance vs Confidence Correlation
    ax = axes[1, 0]
    scatter = ax.scatter(df['val_max_prob'], df['val_macro_f1'], 
                        c=epochs, cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('Average Confidence')
    ax.set_ylabel('Macro F1')
    ax.set_title('Performance vs Confidence')
    ax.grid(True, alpha=0.3)
    
    # Add ideal calibration line (diagonal)
    min_conf = df['val_max_prob'].min()
    max_conf = df['val_max_prob'].max()
    ax.plot([min_conf, max_conf], [min_conf, max_conf], 'r--', alpha=0.5, label='Perfect Calibration')
    ax.legend()
    
    plt.colorbar(scatter, ax=ax, label='Epoch')
    
    # 4. Reliability Trends
    ax = axes[1, 1]
    # Brier Score approximation (using validation loss as proxy)
    brier_proxy = df['val_loss'] ** 2  # Simplified approximation
    ax.plot(epochs, brier_proxy, 'purple', linewidth=2, label='Brier Score (proxy)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Brier Score (proxy)')
    ax.set_title('Reliability Score Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add secondary axis for calibration
    ax2 = ax.twinx()
    ax2.plot(epochs, val_ece_proxy * 100, 'orange', linewidth=2, alpha=0.7, label='ECE')
    ax2.set_ylabel('ECE (%)', color='orange')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_dir:
        plot_file = output_dir / "calibration_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Calibration analysis plot saved to: {plot_file}")
    
    plt.show()


def plot_error_analysis(df: pd.DataFrame, metadata: dict, output_dir: Path = None):
    """Plot error analysis and class-specific metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Error Analysis & Class Performance', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # 1. Class Performance Evolution
    ax = axes[0, 0]
    ax.plot(epochs, df['val_verse_f1'], 'b-', label='Verse F1', linewidth=2)
    ax.plot(epochs, df['val_chorus_f1'], 'r-', label='Chorus F1', linewidth=2)
    ax.plot(epochs, df['val_macro_f1'], 'g-', label='Macro F1', linewidth=2)
    
    # Add performance gap
    f1_gap = df['val_verse_f1'] - df['val_chorus_f1']
    ax.fill_between(epochs, df['val_chorus_f1'], df['val_verse_f1'], 
                   alpha=0.2, color='gray', label='Performance Gap')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class Performance Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 2. Class Balance Analysis
    ax = axes[0, 1]
    ax.plot(epochs, df['train_chorus_rate'] * 100, 'b-', label='Train Chorus %', linewidth=2)
    ax.plot(epochs, df['val_chorus_rate'] * 100, 'r-', label='Val Chorus %', linewidth=2)
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.7, label='Expected 30%')
    
    # Highlight imbalance regions
    ax.fill_between(epochs, 0, 15, alpha=0.1, color='red', label='Severe Imbalance')
    ax.fill_between(epochs, 15, 25, alpha=0.1, color='orange', label='Moderate Imbalance')
    ax.fill_between(epochs, 35, 60, alpha=0.1, color='orange')
    ax.fill_between(epochs, 60, 100, alpha=0.1, color='red')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Chorus Rate (%)')
    ax.set_title('Class Balance Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # 3. Transition Analysis (if available)
    ax = axes[1, 0]
    if 'val_verse_to_chorus_acc' in df.columns:
        ax.plot(epochs, df['val_verse_to_chorus_acc'], 'b-', label='Verse → Chorus', linewidth=2)
        ax.plot(epochs, df['val_chorus_to_verse_acc'], 'r-', label='Chorus → Verse', linewidth=2)
        
        # Calculate transition balance
        transition_balance = df['val_verse_to_chorus_acc'] - df['val_chorus_to_verse_acc']
        ax.fill_between(epochs, df['val_chorus_to_verse_acc'], df['val_verse_to_chorus_acc'], 
                       alpha=0.2, color='gray', label='Transition Bias')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Transition Accuracy')
        ax.set_title('Transition Confusion Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'Transition Data\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Transition Analysis')
    
    # 4. Loss Components Analysis
    ax = axes[1, 1]
    # Analyze loss behavior
    loss_ratio = df['val_loss'] / df['train_loss']
    ax.plot(epochs, loss_ratio, 'purple', linewidth=2, label='Val/Train Loss Ratio')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Perfect Generalization')
    ax.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Slight Overfitting')
    ax.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, label='Significant Overfitting')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Ratio')
    ax.set_title('Generalization Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add secondary axis for absolute losses
    ax2 = ax.twinx()
    ax2.plot(epochs, df['train_loss'], 'b:', alpha=0.5, label='Train Loss')
    ax2.plot(epochs, df['val_loss'], 'r:', alpha=0.5, label='Val Loss')
    ax2.set_ylabel('Absolute Loss', alpha=0.5)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    if output_dir:
        plot_file = output_dir / "error_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Error analysis plot saved to: {plot_file}")
    
    plt.show()


def plot_segmentation_analysis(df: pd.DataFrame, metadata: dict, output_dir: Path = None):
    """Plot detailed segmentation-specific analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Segmentation Quality Analysis', fontsize=16, fontweight='bold')
    
    epochs = df['epoch']
    
    # 1. Boundary Detection Quality
    ax = axes[0, 0]
    if 'val_boundary_f1' in df.columns:
        ax.plot(epochs, df['val_boundary_f1'], 'g-', label='Boundary F1', linewidth=3)
        ax.plot(epochs, df['val_boundary_precision'], 'b-', label='Boundary Precision', linewidth=2)
        ax.plot(epochs, df['val_boundary_recall'], 'r-', label='Boundary Recall', linewidth=2)
        
        # Highlight trade-offs
        precision_recall_diff = df['val_boundary_precision'] - df['val_boundary_recall']
        ax.fill_between(epochs, df['val_boundary_recall'], df['val_boundary_precision'], 
                       alpha=0.2, color='gray', label='P-R Trade-off')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Boundary Detection Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'Boundary Metrics\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Boundary Detection')
    
    # 2. Segmentation Quality Metrics
    ax = axes[0, 1]
    if 'val_complete_segments' in df.columns:
        ax.plot(epochs, df['val_complete_segments'], 'purple', label='Complete Segments', linewidth=2)
        ax.plot(epochs, df['val_avg_segment_overlap'], 'orange', label='Avg Overlap', linewidth=2)
        
        # Quality indicators
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Target Overlap')
        ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Min Complete')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Segmentation Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'Segmentation Metrics\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Segmentation Quality')
    
    # 3. Distance-based Error Analysis
    ax = axes[1, 0]
    if 'val_window_diff' in df.columns:
        ax.plot(epochs, df['val_window_diff'], 'red', label='WindowDiff', linewidth=2)
        ax.plot(epochs, df['val_pk_metric'], 'orange', label='Pk Metric', linewidth=2)
        
        # Error thresholds
        ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.7, label='Acceptable Threshold')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Poor Performance')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error Rate')
        ax.set_title('Distance-based Segmentation Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, 'Distance Metrics\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Distance-based Errors')
    
    # 4. Comprehensive Segmentation Health
    ax = axes[1, 1]
    if all(col in df.columns for col in ['val_boundary_f1', 'val_complete_segments', 'val_window_diff']):
        # Create a composite segmentation health score
        boundary_norm = df['val_boundary_f1']
        complete_norm = df['val_complete_segments']
        window_norm = 1 - df['val_window_diff']  # Invert so higher is better
        
        composite_score = (boundary_norm + complete_norm + window_norm) / 3
        
        ax.plot(epochs, composite_score, 'purple', linewidth=3, label='Composite Health')
        ax.plot(epochs, boundary_norm, 'g--', alpha=0.7, label='Boundary (norm)')
        ax.plot(epochs, complete_norm, 'b--', alpha=0.7, label='Complete (norm)')
        ax.plot(epochs, window_norm, 'r--', alpha=0.7, label='WindowDiff (inv)')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized Score')
        ax.set_title('Composite Segmentation Health')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Highlight best performing regions
        best_score = composite_score.max()
        best_epochs = epochs[composite_score > best_score * 0.95]
        if len(best_epochs) > 0:
            ax.axvspan(best_epochs.min(), best_epochs.max(), alpha=0.2, color='green', label='Peak Performance')
    else:
        ax.text(0.5, 0.5, 'Insufficient Data\nfor Composite Analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Composite Segmentation Health')
    
    plt.tight_layout()
    
    if output_dir:
        plot_file = output_dir / "segmentation_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Segmentation analysis plot saved to: {plot_file}")
    
    plt.show()


def plot_metric_correlations(df: pd.DataFrame, metadata: dict, output_dir: Path = None):
    """Plot metric correlations and best epoch analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Metric Correlations & Peak Performance Analysis', fontsize=16, fontweight='bold')
    
    # Select relevant metrics for correlation analysis
    metric_cols = [col for col in df.columns if col.startswith('val_') and 
                   col not in ['val_conf_over_90', 'val_conf_over_95', 'val_max_prob']]
    
    # 1. Correlation Heatmap
    ax = axes[0, 0]
    if len(metric_cols) > 3:
        corr_data = df[metric_cols].corr()
        sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, 
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title('Metric Correlation Matrix')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    else:
        ax.text(0.5, 0.5, 'Insufficient Metrics\nfor Correlation Analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Metric Correlations')
    
    # 2. Best Epoch Heatmap
    ax = axes[0, 1]
    if len(metric_cols) > 3:
        best_epochs = {}
        for col in metric_cols:
            if 'loss' in col.lower() or 'window' in col.lower() or 'pk' in col.lower():
                best_epochs[col] = df[col].idxmin()  # Lower is better
            else:
                best_epochs[col] = df[col].idxmax()  # Higher is better
        
        # Create heatmap data
        epoch_range = range(1, len(df) + 1)
        heatmap_data = np.zeros((len(metric_cols), len(epoch_range)))
        
        for i, metric in enumerate(metric_cols):
            best_epoch = best_epochs[metric]
            heatmap_data[i, best_epoch] = 1
        
        im = ax.imshow(heatmap_data, cmap='Reds', aspect='auto')
        ax.set_yticks(range(len(metric_cols)))
        ax.set_yticklabels([col.replace('val_', '') for col in metric_cols])
        ax.set_xlabel('Epoch')
        ax.set_title('Best Epoch per Metric')
        
        # Add epoch labels for significant points
        epoch_step = max(1, len(epoch_range) // 10)
        ax.set_xticks(range(0, len(epoch_range), epoch_step))
        ax.set_xticklabels(range(1, len(epoch_range) + 1, epoch_step))
    else:
        ax.text(0.5, 0.5, 'Insufficient Metrics\nfor Best Epoch Analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Best Epoch Analysis')
    
    # 3. Performance vs Training Time
    ax = axes[1, 0]
    cumulative_time = df['epoch_time'].cumsum() / 60  # Convert to minutes
    ax.scatter(cumulative_time, df['val_macro_f1'], c=df['epoch'], 
              cmap='viridis', alpha=0.7, s=50)
    ax.set_xlabel('Cumulative Training Time (minutes)')
    ax.set_ylabel('Macro F1')
    ax.set_title('Performance vs Training Time')
    ax.grid(True, alpha=0.3)
    
    # Add efficiency frontier
    if len(df) > 5:
        # Find Pareto frontier (best F1 for each time point)
        pareto_mask = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            # Point is on frontier if no other point has both better time and performance
            better_mask = (cumulative_time <= cumulative_time.iloc[i]) & (df['val_macro_f1'] >= df['val_macro_f1'].iloc[i])
            if better_mask.sum() == 1:  # Only itself
                pareto_mask[i] = True
        
        if pareto_mask.any():
            pareto_times = cumulative_time[pareto_mask]
            pareto_f1s = df['val_macro_f1'][pareto_mask]
            ax.plot(pareto_times, pareto_f1s, 'r-', alpha=0.7, linewidth=2, label='Efficiency Frontier')
            ax.legend()
    
    # 4. Metric Stability Analysis
    ax = axes[1, 1]
    # Calculate coefficient of variation for key metrics
    key_metrics = ['val_macro_f1', 'val_loss']
    if 'val_boundary_f1' in df.columns:
        key_metrics.append('val_boundary_f1')
    
    stability_data = []
    stability_labels = []
    
    for metric in key_metrics:
        if metric in df.columns:
            # Calculate stability in the last 20% of training
            last_20_pct = int(len(df) * 0.8)
            recent_data = df[metric].iloc[last_20_pct:]
            cv = recent_data.std() / recent_data.mean() if recent_data.mean() != 0 else 0
            stability_data.append(cv * 100)  # Convert to percentage
            stability_labels.append(metric.replace('val_', ''))
    
    if stability_data:
        bars = ax.bar(stability_labels, stability_data, 
                     color=['green' if x < 5 else 'orange' if x < 10 else 'red' for x in stability_data])
        ax.set_ylabel('Coefficient of Variation (%)')
        ax.set_title('Metric Stability (Last 20% of Training)')
        ax.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Stable (<5%)')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Moderate (5-10%)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, stability_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.1f}%', ha='center', va='bottom')
    else:
        ax.text(0.5, 0.5, 'Insufficient Data\nfor Stability Analysis', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Metric Stability')
    
    plt.tight_layout()
    
    if output_dir:
        plot_file = output_dir / "metric_correlations.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Metric correlations plot saved to: {plot_file}")
    
    plt.show()


def print_training_summary(df: pd.DataFrame, metadata: dict):
    """Print a comprehensive training summary."""
    
    print("\n🎵 BLSTM Training Summary")
    print("=" * 60)
    
    # Model Information
    if metadata and 'model_info' in metadata:
        model_info = metadata['model_info']
        print(f"\n🧠 Model Architecture:")
        print(f"   Hidden dim: {model_info.get('hidden_dim', 'N/A')}")
        print(f"   Layers: {model_info.get('num_layers', 'N/A')}")
        print(f"   Dropout: {model_info.get('dropout', 'N/A')}")
        if model_info.get('attention_enabled', False):
            print(f"   Attention: ✅ ({model_info.get('attention_heads', 'N/A')} heads)")
            print(f"   Attention dim: {model_info.get('attention_dim', 'N/A')}")
        else:
            print(f"   Attention: ❌")
        
        total_params = model_info.get('total_params', 'N/A')
        if isinstance(total_params, (int, float)):
            print(f"   Total params: {total_params:,}")
        else:
            print(f"   Total params: {total_params}")
    
    # Training Configuration
    if metadata and 'training_info' in metadata:
        train_info = metadata['training_info']
        print(f"\n⚙️  Training Configuration:")
        print(f"   Batch size: {train_info.get('batch_size', 'N/A')}")
        print(f"   Learning rate: {train_info.get('learning_rate', 'N/A')}")
        print(f"   Scheduler: {train_info.get('scheduler', 'N/A')}")
        print(f"   Label smoothing: {train_info.get('label_smoothing', 'N/A')}")
        print(f"   Validation strategy: {train_info.get('validation_strategy', 'N/A')}")
        print(f"   Weight decay: {train_info.get('weight_decay', 'N/A')}")
    
    # Basic info
    total_epochs = len(df)
    final_metrics = df.iloc[-1]
    
    print(f"\n📅 Training Duration:")
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
    
    # Boundary and segmentation metrics if available
    if 'val_boundary_f1' in df.columns:
        print(f"\n🎯 Boundary & Segmentation:")
        print(f"   Boundary F1: {final_metrics['val_boundary_f1']:.4f}")
        print(f"   Boundary Precision: {final_metrics['val_boundary_precision']:.4f}")
        print(f"   Boundary Recall: {final_metrics['val_boundary_recall']:.4f}")
        print(f"   Complete Segments: {final_metrics['val_complete_segments']:.4f}")
        print(f"   Avg Segment Overlap: {final_metrics['val_avg_segment_overlap']:.4f}")
        print(f"   WindowDiff: {final_metrics['val_window_diff']:.4f}")
        print(f"   Pk Metric: {final_metrics['val_pk_metric']:.4f}")
    
    # Best metrics
    best_f1_epoch = df['val_macro_f1'].idxmax() + 1
    best_f1 = df['val_macro_f1'].max()
    
    print(f"\n🏆 Best Results:")
    print(f"   Best Macro F1: {best_f1:.4f} (epoch {best_f1_epoch})")
    print(f"   Best Chorus F1: {df['val_chorus_f1'].max():.4f}")
    print(f"   Lowest Train Loss: {df['train_loss'].min():.4f}")
    print(f"   Lowest Val Loss: {df['val_loss'].min():.4f}")
    
    if 'val_boundary_f1' in df.columns:
        best_boundary_f1_epoch = df['val_boundary_f1'].idxmax() + 1
        print(f"   Best Boundary F1: {df['val_boundary_f1'].max():.4f} (epoch {best_boundary_f1_epoch})")
        print(f"   Best Complete Segments: {df['val_complete_segments'].max():.4f}")
        print(f"   Lowest WindowDiff: {df['val_window_diff'].min():.4f}")
    
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
    
    # Transition analysis if available
    if 'val_verse_to_chorus_acc' in df.columns:
        print(f"\n🔄 Transition Analysis:")
        print(f"   Verse → Chorus accuracy: {final_metrics['val_verse_to_chorus_acc']:.4f}")
        print(f"   Chorus → Verse accuracy: {final_metrics['val_chorus_to_verse_acc']:.4f}")
    
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
    
    # Boundary-specific recommendations
    if 'val_boundary_f1' in df.columns:
        boundary_f1 = final_metrics['val_boundary_f1']
        if boundary_f1 > 0.15:
            print("   🎯 Good boundary detection capability")
        elif boundary_f1 > 0.05:
            print("   ⚠️  Moderate boundary detection - room for improvement")
        else:
            print("   🚨 Poor boundary detection - consider architecture changes")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BLSTM training metrics and create visualizations"
    )
    
    # Support both session folder and direct metrics file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--session-folder', 
                       help='Path to training session folder (will look for training_metrics.json)')
    group.add_argument('--metrics-file',
                       help='Direct path to training_metrics.json file')
    
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting, just show summary')
    
    args = parser.parse_args()
    
    # Determine metrics file path and output directory
    if args.session_folder:
        session_path = Path(args.session_folder)
        if not session_path.exists():
            print(f"❌ Session folder does not exist: {args.session_folder}")
            return
        
        metrics_file = session_path / "training_metrics.json"
        if not metrics_file.exists():
            print(f"❌ training_metrics.json not found in: {args.session_folder}")
            return
        
        # Output plots to session_folder/plots
        output_dir = session_path / "plots"
        output_dir.mkdir(exist_ok=True)
        
    else:
        metrics_file = Path(args.metrics_file)
        if not metrics_file.exists():
            print(f"❌ Metrics file does not exist: {args.metrics_file}")
            return
        
        # Output plots to same directory as metrics file
        output_dir = metrics_file.parent / "plots"
        output_dir.mkdir(exist_ok=True)
    
    # Load metrics
    try:
        df, metadata = load_training_metrics(str(metrics_file))
        print(f"✅ Loaded metrics from {metrics_file}")
        print(f"   Found {len(df)} epochs of training data")
        
        if metadata:
            print(f"   Model has {metadata.get('model_info', {}).get('total_params', 'unknown')} parameters")
            if metadata.get('model_info', {}).get('attention_enabled', False):
                print(f"   Attention mechanism: ✅")
            else:
                print(f"   Attention mechanism: ❌")
        
    except Exception as e:
        print(f"❌ Error loading metrics: {e}")
        return
    
    # Print summary
    print_training_summary(df, metadata)
    
    # Create plots
    if not args.no_plot:
        try:
            # Core plots
            plot_training_metrics(df, metadata, output_dir)
            
            # Advanced analysis dashboards
            plot_model_dynamics(df, metadata, output_dir)
            plot_calibration_analysis(df, metadata, output_dir)
            plot_error_analysis(df, metadata, output_dir)
            plot_segmentation_analysis(df, metadata, output_dir)
            plot_metric_correlations(df, metadata, output_dir)
            
            print(f"\n📁 All plots saved to: {output_dir}")
            
        except ImportError as e:
            print(f"\n⚠️  Missing required packages - skipping plots: {e}")
            print("   Install with: pip install matplotlib pandas seaborn")
        
        except Exception as e:
            print(f"\n⚠️  Error creating plots: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
