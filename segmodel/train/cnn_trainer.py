"""
CNN Training module with anti-collapse guardrails and temperature calibration.
Implements the complete training loop optimized for CNN models with monitoring and early stopping.
CNN-specific optimizations for parallel processing and convolutional architectures.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau, 
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts,
    StepLR,
    LinearLR,
    OneCycleLR
)

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
from dataclasses import dataclass, asdict
import json
from collections import deque

from ..losses import batch_guardrails, sequence_f1_score
from ..metrics import (
    compute_boundary_metrics,
    compute_segment_metrics,
    compute_transition_metrics,
    format_boundary_metrics_report,
    compute_segmentation_metrics,
    format_segmentation_metrics_report
)
from .calibration import fit_calibration, load_calibration


class CNNEarlyStopper:
    """
    CNN-optimized early stopping with faster convergence awareness.
    
    CNN models often converge faster than RNNs, so we use:
    - Shorter patience windows for faster decisions
    - Gradient-aware stopping to detect convergence plateaus
    - Multi-metric tracking for CNN-specific patterns
    """
    
    def __init__(self, patience=8, min_delta=0.0, convergence_window=5):
        self.patience = patience
        self.min_delta = min_delta
        self.convergence_window = convergence_window
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        
        # CNN-specific convergence tracking
        self.score_history = deque(maxlen=convergence_window)
        self.f1_hist = deque(maxlen=3)
        self.ece_hist = deque(maxlen=3)
        self.mode = "uninitialized"
        
        print(f"🏃‍♂️ CNN Early Stopper: patience={patience}, convergence_window={convergence_window}")
    
    def step(self, metrics: Dict[str, float]) -> bool:
        """
        CNN-optimized stepping with convergence detection.
        
        Args:
            metrics: Dictionary containing validation metrics
        
        Returns:
            bool: True if training should stop
        """
        
        # --- Priority: CNN-optimized calibration-aware stopping ---
        if "val_f1" in metrics and "val_ece" in metrics:
            if self.mode != "cnn_calibration":
                self.mode = "cnn_calibration"
                print(f"  🎯 CNN EarlyStopper: Using CNN-optimized calibration mode")
            
            self.f1_hist.append(metrics["val_f1"])
            self.ece_hist.append(metrics["val_ece"])
            
            # CNN convergence detection: faster decisions with shorter history
            if len(self.f1_hist) == self.f1_hist.maxlen:
                ece_increasing = all(self.ece_hist[i] <= self.ece_hist[i+1] 
                                   for i in range(len(self.ece_hist)-1))
                f1_gain = max(self.f1_hist) - min(self.f1_hist)
                
                # More aggressive stopping for CNNs (faster convergence)
                if ece_increasing and f1_gain < 0.003:  # Tighter threshold than BiLSTM
                    self.should_stop = True
                    print(f"  🛑 CNN calibration-aware early stop:")
                    print(f"     ECE trend: increasing")
                    print(f"     F1 gain: {f1_gain:.4f} < 0.003 (CNN threshold)")
                    return True
                else:
                    self.counter = 0
                    return False
            
            return False
        
        # --- Fallback: CNN-optimized traditional stopping ---
        if self.mode != "cnn_traditional":
            self.mode = "cnn_traditional"
            print(f"  📉 CNN EarlyStopper: Using CNN-optimized traditional mode")
        
        score = -metrics.get("val_loss", 0.0)
        self.score_history.append(score)
        
        # Check for convergence plateau (CNN-specific)
        if len(self.score_history) == self.convergence_window:
            score_variance = np.var(list(self.score_history))
            if score_variance < 1e-6:  # Very low variance indicates convergence
                print(f"  🛑 CNN convergence plateau detected (variance: {score_variance:.2e})")
                return True
        
        # Traditional patience logic with CNN adjustments
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  🛑 CNN traditional early stop: {self.patience} epochs without improvement")
                return True
        else:
            self.best_score = score
            self.counter = 0
        
        return False
    
    def reset(self):
        """Reset early stopper state."""
        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.score_history.clear()
        self.f1_hist.clear()
        self.ece_hist.clear()
        self.mode = "uninitialized"


@dataclass
class CNNTrainingMetrics:
    """Container for CNN training metrics with additional CNN-specific fields."""
    epoch: int
    train_loss: float
    train_chorus_rate: float
    train_max_prob: float
    train_conf_over_90: float
    train_conf_over_95: float
    val_loss: float
    val_macro_f1: float
    val_verse_f1: float
    val_chorus_f1: float
    val_chorus_rate: float
    val_max_prob: float
    val_conf_over_90: float
    val_conf_over_95: float
    learning_rate: float
    epoch_time: float
    # Boundary-aware metrics
    val_boundary_f1: float = 0.0
    val_boundary_precision: float = 0.0
    val_boundary_recall: float = 0.0
    val_complete_segments: float = 0.0
    val_avg_segment_overlap: float = 0.0
    val_verse_to_chorus_acc: float = 0.0
    val_chorus_to_verse_acc: float = 0.0
    # Segmentation metrics
    val_window_diff: float = 1.0
    val_pk_metric: float = 1.0
    # CNN-specific metrics
    cnn_receptive_field_usage: float = 0.0
    cnn_gradient_norm: float = 0.0
    cnn_effective_layers: int = 0


class CNNEmergencyMonitor:
    """
    CNN-specific emergency monitoring system.
    
    Optimized for CNN patterns:
    - Faster overconfidence detection (CNNs can overfit quickly)
    - Gradient explosion monitoring (important for deep CNNs)
    - Receptive field utilization tracking
    """
    
    def __init__(
        self,
        max_confidence: float = 0.93,  # Slightly lower for CNNs
        min_chorus_rate: float = 0.05,
        max_chorus_rate: float = 0.85,
        max_conf_over_95: float = 0.08,  # Tighter for CNNs
        val_overconf_threshold: float = 0.94,  # More aggressive
        val_f1_collapse_threshold: float = 0.1,
        emergency_overconf_threshold: float = 0.97,  # Slightly lower
        emergency_conf95_ratio: float = 0.7,  # More aggressive
        emergency_f1_threshold: float = 0.05,
        max_gradient_norm: float = 5.0  # CNN-specific
    ):
        self.max_confidence = max_confidence
        self.min_chorus_rate = min_chorus_rate
        self.max_chorus_rate = max_chorus_rate
        self.max_conf_over_95 = max_conf_over_95
        self.val_overconf_threshold = val_overconf_threshold
        self.val_f1_collapse_threshold = val_f1_collapse_threshold
        self.emergency_overconf_threshold = emergency_overconf_threshold
        self.emergency_conf95_ratio = emergency_conf95_ratio
        self.emergency_f1_threshold = emergency_f1_threshold
        self.max_gradient_norm = max_gradient_norm
        
        print(f"🛡️  CNN Emergency Monitor Activated:")
        print(f"   Max confidence: {max_confidence} (CNN-optimized)")
        print(f"   Chorus rate range: [{min_chorus_rate:.2f}, {max_chorus_rate:.2f}]")
        print(f"   Max conf>0.95: {max_conf_over_95} (tighter for CNNs)")
        print(f"   Max gradient norm: {max_gradient_norm}")
    
    def check_batch(self, guardrails: Dict[str, float], batch_idx: int, gradient_norm: float = None) -> Tuple[bool, str]:
        """
        CNN-optimized batch checking with gradient monitoring.
        """
        warnings = []
        
        # Standard confidence checks (with CNN-specific thresholds)
        if guardrails['max_prob_mean'] > self.max_confidence:
            warnings.append(f"CNN_OVERCONF: avg_conf={guardrails['max_prob_mean']:.3f}")
        
        if guardrails['confidence_over_95'] > self.max_conf_over_95:
            warnings.append(f"CNN_HIGH_CONF: conf>0.95={guardrails['confidence_over_95']:.2%}")
        
        # Chorus rate monitoring
        if guardrails['chorus_rate'] < self.min_chorus_rate:
            warnings.append(f"CNN_CHORUS_COLLAPSE: rate={guardrails['chorus_rate']:.2%}")
        elif guardrails['chorus_rate'] > self.max_chorus_rate:
            warnings.append(f"CNN_CHORUS_OVER: rate={guardrails['chorus_rate']:.2%}")
        
        # CNN-specific: gradient monitoring
        if gradient_norm is not None and gradient_norm > self.max_gradient_norm:
            warnings.append(f"CNN_GRAD_EXPLOSION: norm={gradient_norm:.2f}")
        
        # Emergency conditions (using configurable thresholds, same as BiLSTM approach)
        conditions = {
            'extreme_overconf': guardrails['max_prob_mean'] > self.emergency_overconf_threshold,
            'high_conf_ratio': guardrails['confidence_over_95'] > self.emergency_conf95_ratio,
            'chorus_collapse': guardrails['chorus_rate'] < 0.01,  # Keep hardcoded for safety
            'all_chorus': guardrails['chorus_rate'] > 0.99,  # Keep hardcoded for safety
            'gradient_explosion': gradient_norm is not None and gradient_norm > self.max_gradient_norm * 2  # CNN-specific
        }
        
        should_stop = any(conditions.values())
        
        if should_stop:
            triggered = [k for k, v in conditions.items() if v]
            warnings.append(f"CNN_EMERGENCY: {triggered}")
        
        warning_msg = " | ".join(warnings) if warnings else ""
        return should_stop, warning_msg
    
    def check_epoch(self, metrics) -> Tuple[bool, str]:
        """CNN-optimized epoch checking."""
        warnings = []
        
        # More aggressive validation monitoring for CNNs
        if metrics.val_max_prob > self.val_overconf_threshold:
            warnings.append(f"CNN_VAL_OVERCONF: {metrics.val_max_prob:.3f}")
        
        if metrics.val_macro_f1 < self.val_f1_collapse_threshold:
            warnings.append(f"CNN_F1_COLLAPSE: {metrics.val_macro_f1:.3f}")
        
        # CNN-specific gradient monitoring
        if hasattr(metrics, 'cnn_gradient_norm') and metrics.cnn_gradient_norm > self.max_gradient_norm:
            warnings.append(f"CNN_HIGH_GRADIENTS: {metrics.cnn_gradient_norm:.2f}")
        
        should_stop = (
            metrics.val_max_prob > self.emergency_overconf_threshold or
            metrics.val_conf_over_95 > self.emergency_conf95_ratio or
            metrics.val_macro_f1 < self.emergency_f1_threshold
        )
        
        warning_msg = " | ".join(warnings) if warnings else ""
        return should_stop, warning_msg


def create_cnn_scheduler(optimizer, config, total_steps: int = None):
    """
    CNN-optimized scheduler factory with aggressive learning rate schedules.
    
    CNNs often benefit from:
    - OneCycleLR for faster convergence
    - Cosine annealing with restarts
    - More aggressive learning rate policies
    """
    scheduler_name = getattr(config, 'scheduler', 'onecycle')
    
    if scheduler_name == 'onecycle':
        # OneCycleLR is excellent for CNNs
        max_lr = getattr(config, 'learning_rate', 0.001)
        epochs = getattr(config, 'max_epochs', 50)
        steps_per_epoch = total_steps // epochs if total_steps else 100
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=getattr(config, 'onecycle_pct_start', 0.3),
            anneal_strategy=getattr(config, 'onecycle_anneal', 'cos')
        )
        return scheduler, 'step'
    
    elif scheduler_name == 'cosine':
        T_max = getattr(config, 'cosine_t_max', config.max_epochs)
        eta_min = float(getattr(config, 'min_lr', 1e-6))
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        return scheduler, 'epoch'
    
    elif scheduler_name == 'cosine_restarts':
        # Excellent for CNNs - allows escaping local minima
        T_0 = getattr(config, 'cosine_t0', 8)  # Shorter for CNNs
        T_mult = getattr(config, 'cosine_t_mult', 2)
        eta_min = float(getattr(config, 'min_lr', 1e-6))
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        return scheduler, 'epoch'
    
    elif scheduler_name == 'plateau':
        # More aggressive for CNNs
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=getattr(config, 'lr_factor', 0.3),  # More aggressive
            patience=getattr(config, 'lr_patience', 3),  # Shorter patience
            min_lr=float(getattr(config, 'min_lr', 1e-6))
        )
        return scheduler, 'epoch'
    
    else:
        # Fallback to cosine
        print(f"⚠️  Unknown CNN scheduler '{scheduler_name}', using cosine")
        return create_cnn_scheduler(optimizer, config, total_steps)


def compute_cnn_validation_score(metrics, config: Any) -> float:
    """
    CNN-optimized validation score computation.
    
    CNNs excel at local pattern detection, so we might emphasize:
    - Boundary detection (local transitions)
    - Segment quality (local consistency)
    """
    strategy = getattr(config, 'validation_strategy', 'cnn_composite')
    
    if strategy == 'cnn_composite':
        # CNN-optimized composite score
        composite_score = (
            0.20 * metrics.val_macro_f1 +                    # Line-level (reduced weight)
            0.45 * metrics.val_boundary_f1 +                 # Boundary detection (CNNs excel here)
            0.30 * metrics.val_avg_segment_overlap +         # Segment quality (local consistency)
            0.05 * (1.0 - metrics.val_window_diff)          # Segmentation quality
        )
        return composite_score
    
    elif strategy == 'boundary_focused':
        # Emphasize boundary detection for CNNs
        return 0.7 * metrics.val_boundary_f1 + 0.3 * metrics.val_macro_f1
    
    else:
        # Fall back to standard strategies
        if strategy == 'line_f1':
            return metrics.val_macro_f1
        elif strategy == 'boundary_f1':
            return metrics.val_boundary_f1
        elif strategy == 'windowdiff':
            return 1.0 - metrics.val_window_diff
        elif strategy == 'pk':
            return 1.0 - metrics.val_pk_metric
        elif strategy == 'segment_iou':
            return metrics.val_avg_segment_overlap
        else:
            return metrics.val_macro_f1


class CNNTrainer:
    """
    CNN-optimized trainer class with specialized monitoring and optimization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_function: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Any,
        output_dir: Path,
        disable_emergency_monitoring: bool = False
    ):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.output_dir = output_dir
        
        # CNN-specific attributes
        self.is_cnn = hasattr(model, 'cnn_blocks') or 'CNN' in str(type(model))
        self.gradient_history = deque(maxlen=10)
        
        # Initialize calibration info storage
        self.calibration_info = None
        
        # Set validation strategy (CNN-optimized default)
        self.validation_strategy = getattr(config, 'validation_strategy', 'cnn_composite')
        
        # CNN-optimized scheduler
        total_steps = getattr(config, 'max_epochs', 50) * getattr(config, 'steps_per_epoch', 100)
        self.scheduler, self.scheduler_type = create_cnn_scheduler(
            optimizer, 
            config,
            total_steps=total_steps
        )
        
        print(f"🎯 CNN Scheduler: {getattr(config, 'scheduler', 'onecycle')} ({self.scheduler_type}-based)")
        print(f"🎯 CNN Validation Strategy: {self.validation_strategy}")
        
        # CNN-optimized emergency monitoring
        if not disable_emergency_monitoring:
            self.emergency_monitor = CNNEmergencyMonitor(
                max_confidence=getattr(config, 'max_confidence_threshold', 0.93),
                min_chorus_rate=getattr(config, 'min_chorus_rate', 0.05),
                max_chorus_rate=getattr(config, 'max_chorus_rate', 0.85),
                max_conf_over_95=getattr(config, 'max_conf_over_95_ratio', 0.08),
                val_overconf_threshold=getattr(config, 'val_overconf_threshold', 0.94),
                val_f1_collapse_threshold=getattr(config, 'val_f1_collapse_threshold', 0.1),
                emergency_overconf_threshold=getattr(config, 'emergency_overconf_threshold', 0.97),
                emergency_conf95_ratio=getattr(config, 'emergency_conf95_ratio', 0.7),
                emergency_f1_threshold=getattr(config, 'emergency_f1_threshold', 0.05),
                max_gradient_norm=getattr(config, 'max_gradient_norm', 5.0)
            )
        else:
            self.emergency_monitor = None
            print("⚠️  CNN Emergency monitoring DISABLED")
        
        # Training state
        self.best_val_score = -1.0
        self.patience_counter = 0
        self.training_metrics = []
        self._best_epoch = 0
        
        # CNN-optimized early stopper
        self.early_stopper = CNNEarlyStopper(
            patience=getattr(config, 'patience', 6),  # Shorter for CNNs
            min_delta=getattr(config, 'min_delta', 0.0),
            convergence_window=getattr(config, 'convergence_window', 5)
        )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_metrics_file()
        
        print(f"🚀 CNN Trainer initialized:")
        print(f"   Device: {device}")
        print(f"   Model type: {'CNN' if self.is_cnn else 'Unknown'}")
        print(f"   Output dir: {output_dir}")
        print(f"   Max epochs: {config.max_epochs}")
        print(f"   Patience: {getattr(config, 'patience', 6)} (CNN-optimized)")
        print(f"   Convergence detection: {getattr(config, 'convergence_window', 5)} epochs")
    
    def _compute_gradient_norm(self) -> float:
        """Compute gradient norm for monitoring."""
        total_norm = 0.0
        param_count = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        self.gradient_history.append(total_norm)
        
        return total_norm
    
    def _analyze_cnn_utilization(self) -> Dict[str, float]:
        """Analyze CNN-specific utilization metrics."""
        metrics = {
            'receptive_field_usage': 0.0,
            'effective_layers': 0,
            'gradient_stability': 0.0
        }
        
        # Gradient stability analysis
        if len(self.gradient_history) > 1:
            recent_grads = list(self.gradient_history)[-5:]
            metrics['gradient_stability'] = 1.0 / (1.0 + np.std(recent_grads))
        
        # Count effective layers (CNN blocks)
        if hasattr(self.model, 'cnn_blocks'):
            metrics['effective_layers'] = len(self.model.cnn_blocks)
        
        # Receptive field usage (simplified)
        if hasattr(self.model, 'kernel_sizes'):
            avg_kernel = np.mean(self.model.kernel_sizes)
            max_seq_len = getattr(self.model, 'max_seq_length', 1000)
            metrics['receptive_field_usage'] = min(1.0, avg_kernel / max_seq_len * 100)
        
        return metrics
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """CNN-optimized training epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_guardrails = []
        num_batches = len(train_loader)
        gradient_norms = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            features = batch.features.to(self.device)
            labels = batch.labels.to(self.device)
            mask = batch.mask.to(self.device)
            
            # Forward pass
            logits = self.model(features, mask)
            loss = self.loss_function(logits, labels, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient monitoring (CNN-specific)
            gradient_norm = self._compute_gradient_norm()
            gradient_norms.append(gradient_norm)
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Step-based scheduler update (for OneCycleLR)
            if self.scheduler_type == 'step':
                self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Batch-level guardrails
            guardrails = batch_guardrails(logits, mask)
            all_guardrails.append(guardrails)
            
            # CNN-optimized emergency monitoring
            should_stop, warning = False, ""
            skip_batches = getattr(self.config, 'skip_batches', 30)  # Shorter for CNNs
            print_batch_every = getattr(self.config, 'print_batch_every', 10)
            
            if self.emergency_monitor is not None and batch_idx >= skip_batches:
                should_stop, warning = self.emergency_monitor.check_batch(
                    guardrails, batch_idx, gradient_norm
                )
            
            # Print batch info with CNN-specific metrics
            if batch_idx % print_batch_every == 0 or batch_idx == num_batches - 1:
                print(f"  CNN Batch {batch_idx+1:3d}/{num_batches}: "
                      f"loss={loss.item():.4f}, "
                      f"grad_norm={gradient_norm:.3f}, "
                      f"chorus%={guardrails['chorus_rate']:.2f}, "
                      f"conf={guardrails['max_prob_mean']:.3f}")
                
                if warning:
                    print(f"    ⚠️  {warning}")
            
            if should_stop:
                print(f"🛑 CNN EMERGENCY STOP at batch {batch_idx+1}")
                raise RuntimeError("CNN emergency training stop")
        
        # Aggregate metrics with CNN-specific additions
        avg_metrics = {}
        for key in all_guardrails[0].keys():
            avg_metrics[key] = np.mean([g[key] for g in all_guardrails])
        
        avg_metrics['loss'] = total_loss / num_batches
        avg_metrics['avg_gradient_norm'] = np.mean(gradient_norms)
        avg_metrics['gradient_stability'] = 1.0 / (1.0 + np.std(gradient_norms))
        
        return avg_metrics
    
    def evaluate(self, val_loader: DataLoader, calibrator=None) -> Dict[str, float]:
        """CNN-optimized evaluation with specialized metrics."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_masks = []
        all_guardrails = []
        
        print(f"🔍 CNN Validating on {len(val_loader)} batches...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader, 1):
                features = batch.features.to(self.device)
                labels = batch.labels.to(self.device)
                mask = batch.mask.to(self.device)
                
                # Forward pass
                logits = self.model(features, mask)
                loss = self.loss_function(logits, labels, mask)
                
                # Track loss
                total_loss += loss.item()
                
                # Apply calibration if provided
                probs = (calibrator.apply(logits) if calibrator else torch.softmax(logits, dim=-1))
                predictions = probs.argmax(dim=-1)
                
                # Store for metrics computation
                all_predictions.append(predictions)
                all_targets.append(labels)
                all_masks.append(mask)
                
                # Track guardrails
                guardrails = batch_guardrails(torch.log(probs + 1e-8), mask)
                all_guardrails.append(guardrails)
                
                # Print validation progress (more frequent for CNNs)
                if batch_idx == 1 or batch_idx % max(1, len(val_loader) // 5) == 0 or batch_idx == len(val_loader):
                    avg_loss = total_loss / batch_idx
                    print(f"  CNN Val {batch_idx:2d}/{len(val_loader)}: "
                          f"loss={avg_loss:.4f}, "
                          f"chorus%={guardrails['chorus_rate']:.2f}, "
                          f"conf={guardrails['max_prob_mean']:.3f}")
        
        print(f"✅ CNN Validation complete, computing metrics...")
        
        # Standard metrics computation (same as BiLSTM trainer)
        avg_loss = total_loss / len(val_loader)
        
        # Aggregate guardrails
        avg_guardrails = {}
        for key in all_guardrails[0].keys():
            avg_guardrails[key] = np.mean([g[key] for g in all_guardrails])
        
        # Compute line-level F1 scores
        all_pred_flat = torch.cat([pred.flatten() for pred in all_predictions], dim=0)
        all_targ_flat = torch.cat([targ.flatten() for targ in all_targets], dim=0) 
        all_mask_flat = torch.cat([mask.flatten() for mask in all_masks], dim=0)
        
        f1_scores = sequence_f1_score(all_pred_flat, all_targ_flat, all_mask_flat)
        
        # Compute boundary-aware metrics
        boundary_metrics_batch = []
        segment_metrics_batch = []
        transition_metrics_batch = []
        
        for pred_batch, targ_batch, mask_batch in zip(all_predictions, all_targets, all_masks):
            batch_boundary = compute_boundary_metrics(pred_batch, targ_batch, mask_batch)
            batch_segment = compute_segment_metrics(pred_batch, targ_batch, mask_batch)
            batch_transition = compute_transition_metrics(pred_batch, targ_batch, mask_batch)
            
            boundary_metrics_batch.append(batch_boundary)
            segment_metrics_batch.append(batch_segment)
            transition_metrics_batch.append(batch_transition)
        
        # Aggregate boundary metrics
        total_pred_boundaries = sum(bm.total_boundaries_predicted for bm in boundary_metrics_batch)
        total_true_boundaries = sum(bm.total_boundaries_actual for bm in boundary_metrics_batch)  
        total_correct_boundaries = sum(bm.correct_boundaries for bm in boundary_metrics_batch)
        
        boundary_precision = total_correct_boundaries / total_pred_boundaries if total_pred_boundaries > 0 else 0.0
        boundary_recall = total_correct_boundaries / total_true_boundaries if total_true_boundaries > 0 else 0.0
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0.0
        
        # Aggregate segment metrics
        avg_complete_segments = np.mean([sm.complete_segments_detected for sm in segment_metrics_batch]) if segment_metrics_batch else 0.0
        avg_segment_overlap = np.mean([sm.avg_segment_overlap for sm in segment_metrics_batch]) if segment_metrics_batch else 0.0
        
        # Aggregate transition metrics  
        total_v2c = sum(tm.verse_to_chorus_count for tm in transition_metrics_batch)
        total_c2v = sum(tm.chorus_to_verse_count for tm in transition_metrics_batch)
        total_v2c_correct = sum(tm.verse_to_chorus_correct for tm in transition_metrics_batch)
        total_c2v_correct = sum(tm.chorus_to_verse_correct for tm in transition_metrics_batch)
        
        v2c_accuracy = total_v2c_correct / total_v2c if total_v2c > 0 else 0.0
        c2v_accuracy = total_c2v_correct / total_c2v if total_c2v > 0 else 0.0
        
        # Compute segmentation metrics
        segmentation_metrics_batch = []
        for pred_batch, targ_batch, mask_batch in zip(all_predictions, all_targets, all_masks):
            batch_segmentation = compute_segmentation_metrics(pred_batch, targ_batch, mask_batch)
            segmentation_metrics_batch.append(batch_segmentation)
        
        # Aggregate segmentation metrics
        avg_window_diff = np.mean([sm.window_diff for sm in segmentation_metrics_batch]) if segmentation_metrics_batch else 1.0
        avg_pk_metric = np.mean([sm.pk_metric for sm in segmentation_metrics_batch]) if segmentation_metrics_batch else 1.0
        
        # CNN-specific utilization metrics
        cnn_utilization = self._analyze_cnn_utilization()
        
        # Combine all metrics
        metrics = {
            'loss': avg_loss,
            **avg_guardrails,
            **f1_scores,
            # Boundary metrics
            'boundary_f1': boundary_f1,
            'boundary_precision': boundary_precision,
            'boundary_recall': boundary_recall,
            # Segment metrics
            'complete_segments': avg_complete_segments,
            'avg_segment_overlap': avg_segment_overlap,
            # Transition metrics
            'verse_to_chorus_acc': v2c_accuracy,
            'chorus_to_verse_acc': c2v_accuracy,
            # Segmentation metrics
            'window_diff': avg_window_diff,
            'pk_metric': avg_pk_metric,
            # CNN-specific metrics
            **cnn_utilization,
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Tuple[nn.Module, float]:
        """
        CNN-optimized main training loop.
        """
        print(f"🚀 Starting CNN training for {self.config.max_epochs} epochs...")
        print(f"   Model: {type(self.model).__name__}")
        print(f"   Scheduler: {getattr(self.config, 'scheduler', 'onecycle')}")
        print(f"   Validation: {self.validation_strategy}")
        
        best_model_state = None
        best_epoch = 0
        
        try:
            for epoch in range(1, self.config.max_epochs + 1):
                epoch_start = time.time()
                
                print(f"\n📅 CNN Epoch {epoch}/{self.config.max_epochs}")
                print("-" * 50)
                
                try:
                    # Training
                    train_metrics = self.train_epoch(train_loader)
                    
                    # Validation  
                    val_metrics = self.evaluate(val_loader)
                    
                    # Calculate epoch time
                    epoch_time = time.time() - epoch_start
                    
                    # Create metrics object with CNN-specific fields
                    metrics = CNNTrainingMetrics(
                        epoch=epoch,
                        train_loss=train_metrics['loss'],
                        train_chorus_rate=train_metrics['chorus_rate'],
                        train_max_prob=train_metrics['max_prob_mean'],
                        train_conf_over_90=train_metrics['confidence_over_90'],
                        train_conf_over_95=train_metrics['confidence_over_95'],
                        val_loss=val_metrics['loss'],
                        val_macro_f1=val_metrics['macro_f1'],
                        val_verse_f1=val_metrics['verse_f1'],
                        val_chorus_f1=val_metrics['chorus_f1'],
                        val_chorus_rate=val_metrics['chorus_rate'],
                        val_max_prob=val_metrics['max_prob_mean'],
                        val_conf_over_90=val_metrics['confidence_over_90'],
                        val_conf_over_95=val_metrics['confidence_over_95'],
                        learning_rate=0.0,  # Will be set below
                        epoch_time=epoch_time,
                        val_boundary_f1=val_metrics.get('boundary_f1', 0.0),
                        val_boundary_precision=val_metrics.get('boundary_precision', 0.0),
                        val_boundary_recall=val_metrics.get('boundary_recall', 0.0),
                        val_window_diff=val_metrics.get('window_diff', 1.0),
                        val_pk_metric=val_metrics.get('pk_metric', 1.0),
                        val_complete_segments=val_metrics.get('complete_segments', 0.0),
                        val_avg_segment_overlap=val_metrics.get('avg_segment_overlap', 0.0),
                        val_verse_to_chorus_acc=val_metrics.get('verse_to_chorus_acc', 0.0),
                        val_chorus_to_verse_acc=val_metrics.get('chorus_to_verse_acc', 0.0),
                        # CNN-specific metrics
                        cnn_receptive_field_usage=val_metrics.get('receptive_field_usage', 0.0),
                        cnn_gradient_norm=train_metrics.get('avg_gradient_norm', 0.0),
                        cnn_effective_layers=val_metrics.get('effective_layers', 0)
                    )
                    
                    current_val_score = compute_cnn_validation_score(metrics, self.config)
                    
                    # Learning rate scheduling
                    if self.scheduler_type == 'epoch':
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(current_val_score)
                        else:
                            self.scheduler.step()
                    
                    current_lr = self.optimizer.param_groups[0]['lr']
                    metrics.learning_rate = current_lr
                    
                    self._save_epoch_metrics(metrics)
                    
                    # Print CNN-optimized epoch summary
                    print(f"\n📊 CNN Epoch {epoch} Summary:")
                    print(f"  Train: loss={metrics.train_loss:.4f}, "
                          f"grad_norm={metrics.cnn_gradient_norm:.3f}, "
                          f"chorus%={metrics.train_chorus_rate:.2f}")
                    print(f"  Val:   Score={current_val_score:.4f} ({self.validation_strategy}), "
                          f"F1={metrics.val_macro_f1:.4f}, "
                          f"conf={metrics.val_max_prob:.3f}")
                    print(f"  📏 Boundary: F1={metrics.val_boundary_f1:.3f}, "
                          f"Precision={metrics.val_boundary_precision:.3f}, "
                          f"Recall={metrics.val_boundary_recall:.3f}")
                    print(f"  🎯 Segments: Complete={metrics.val_complete_segments:.1%}, "
                          f"Overlap={metrics.val_avg_segment_overlap:.3f}")
                    print(f"  🏗️  CNN: Layers={metrics.cnn_effective_layers}, "
                          f"RF_usage={metrics.cnn_receptive_field_usage:.1%}")
                    print(f"  Time: {epoch_time:.1f}s, LR: {current_lr:.2e}")
                    
                    # Emergency monitoring
                    should_stop, warning = False, ""
                    skip_epochs = getattr(self.config, 'skip_epochs', 2)  # Shorter for CNNs
                    if self.emergency_monitor is not None and epoch >= skip_epochs:
                        should_stop, warning = self.emergency_monitor.check_epoch(metrics)
                    if warning:
                        print(f"  ⚠️  {warning}")
                    if should_stop:
                        print(f"🛑 CNN EMERGENCY STOP at epoch {epoch}")
                        break
                    
                    # CNN-optimized early stopping
                    stopper_metrics = {
                        'val_loss': metrics.val_loss,
                        'val_f1': metrics.val_macro_f1,
                        'val_ece': metrics.val_conf_over_95
                    }
                    
                    should_early_stop = self.early_stopper.step(stopper_metrics)
                    
                    # Best model tracking
                    if current_val_score > self.best_val_score:
                        self.best_val_score = current_val_score
                        self._best_epoch = epoch
                        best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                        best_epoch = epoch
                        
                        # Save best model
                        torch.save(best_model_state, self.output_dir / "best_cnn_model.pt")
                        print(f"  ✅ New best CNN {self.validation_strategy}: {current_val_score:.4f} (epoch {epoch})")
                    else:
                        print(f"  📊 CNN Early stopping: {self.early_stopper.mode} mode")
                    
                    print(f"  🏆 Best CNN model: {self.validation_strategy}={self.best_val_score:.4f} from epoch {self._best_epoch}")
                    
                    if should_early_stop:
                        print(f"🛑 CNN early stopping after {epoch} epochs")
                        break
                
                except RuntimeError as e:
                    if "CNN emergency training stop" in str(e):
                        print(f"🚨 CNN training stopped due to emergency condition")
                        break
                    else:
                        raise e
        
        except KeyboardInterrupt:
            print(f"\n⚠️  CNN training interrupted at epoch {epoch}")
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
            self._run_calibration(val_loader)
            torch.save(self.model.state_dict(), self.output_dir / "final_cnn_model.pt")
            self._save_training_log(final=True)
            return self.model, self.calibration_info
        
        # Load best model and run calibration
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        print(f"\n🌡️ Running CNN model calibration...")
        calibration_methods = None
        if hasattr(self.config, 'calibration') and hasattr(self.config.calibration, 'methods'):
            calibration_methods = self.config.calibration.methods
        
        self.calibration_info = fit_calibration(
            model=self.model,
            val_loader=val_loader,
            device=self.device,
            methods=calibration_methods,
            output_dir=self.output_dir
        )
        
        # Save final model and metrics
        torch.save(self.model.state_dict(), self.output_dir / "final_cnn_model.pt")
        self._save_training_log(final=True)
        
        print(f"\n✅ CNN Training completed!")
        print(f"   Best validation score ({self.validation_strategy}): {self.best_val_score:.4f}")
        print(f"   Early stopping mode: {self.early_stopper.mode}")
        print(f"   CNN Models saved to: {self.output_dir}")
        
        return self.model, self.calibration_info
    
    def _run_calibration(self, val_loader):
        """Run CNN model calibration."""
        print(f"\n🌡️ Running CNN model calibration...")
        
        calibration_methods = None
        if hasattr(self.config, 'calibration') and hasattr(self.config.calibration, 'methods'):
            calibration_methods = self.config.calibration.methods
        
        try:
            self.calibration_info = fit_calibration(
                model=self.model,
                val_loader=val_loader,
                device=self.device,
                methods=calibration_methods,
                output_dir=self.output_dir
            )
            print(f"✅ CNN Calibration completed")
        except Exception as e:
            print(f"⚠️  CNN Calibration failed: {e}")
            self.calibration_info = {}
        
        return self.calibration_info
    
    def _save_training_log(self, final: bool = False):
        """Save CNN training metrics to file."""
        metrics_file = self.output_dir / "cnn_training_metrics.json"
        
        # Convert to serializable format
        metrics_data = [asdict(m) for m in self.training_metrics]
        
        # Create CNN-specific training log
        training_log = {
            "metadata": {
                "model_info": {
                    "architecture": "CNN",
                    "hidden_dim": getattr(self.model, 'hidden_dim', 128),
                    "num_layers": getattr(self.model, 'num_layers', 1),
                    "dropout": getattr(self.model, 'dropout_p', 0.2),
                    "layer_dropout": getattr(self.model, 'layer_dropout_p', 0.0),
                    # CNN-specific parameters
                    "kernel_sizes": getattr(self.model, 'kernel_sizes', [3, 5, 7]),
                    "dilation_rates": getattr(self.model, 'dilation_rates', [1, 2, 4]),
                    "use_residual": getattr(self.model, 'use_residual', True),
                    # Attention parameters
                    "attention_enabled": getattr(self.model, 'attention_enabled', False),
                    "attention_type": getattr(self.model, 'attention_type', None),
                    "attention_heads": getattr(self.model, 'attention_heads', 8),
                    # Parameter counts
                    "total_params": sum(p.numel() for p in self.model.parameters()),
                    "cnn_params": (sum(sum(p.numel() for p in block.parameters()) 
                                     for block in self.model.cnn_blocks) 
                                   if hasattr(self.model, 'cnn_blocks') else 0),
                    "attention_params": (sum(p.numel() for p in self.model.attention.parameters()) 
                                       if hasattr(self.model, 'attention') and self.model.attention is not None else 0),
                },
                "training_info": {
                    "batch_size": getattr(self.config, 'batch_size', 16),
                    "learning_rate": getattr(self.config, 'learning_rate', 0.001),
                    "scheduler": getattr(self.config, 'scheduler', 'onecycle'),
                    "scheduler_type": self.scheduler_type,
                    "validation_strategy": self.validation_strategy,
                    "total_epochs": len(self.training_metrics),
                    "early_stopping_mode": getattr(self.early_stopper, 'mode', 'unknown'),
                    "convergence_window": getattr(self.config, 'convergence_window', 5),
                },
                "calibration_info": self.calibration_info
            },
            "metrics": metrics_data
        }
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(training_log, f, indent=2)
            
            if final:
                print(f"📝 Final CNN training log saved to: {metrics_file}")
        
        except Exception as e:
            print(f"⚠️  Error saving CNN training metrics: {e}")
    
    def _initialize_metrics_file(self):
        """Initialize empty CNN metrics file."""
        metrics_file = self.output_dir / "cnn_training_metrics.json"
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump([], f, indent=2)
            print(f"📝 Initialized CNN metrics file: {metrics_file}")
        except Exception as e:
            print(f"⚠️  Error initializing CNN metrics file: {e}")
    
    def _save_epoch_metrics(self, metrics: CNNTrainingMetrics):
        """Save CNN metrics after each epoch."""
        self.training_metrics.append(metrics)
        self._save_training_log()


if __name__ == "__main__":
    print("🧪 Testing CNN trainer components...")
    
    # Test CNN emergency monitor
    monitor = CNNEmergencyMonitor()
    
    # Test normal batch
    normal_guardrails = {
        'chorus_rate': 0.35,
        'max_prob_mean': 0.75,
        'confidence_over_90': 0.05,
        'confidence_over_95': 0.01
    }
    
    should_stop, warning = monitor.check_batch(normal_guardrails, 0, gradient_norm=2.0)
    print(f"CNN Normal batch: stop={should_stop}, warning='{warning}'")
    
    # Test problematic batch with high gradients
    problem_guardrails = {
        'chorus_rate': 0.05,
        'max_prob_mean': 0.95,
        'confidence_over_90': 0.7,
        'confidence_over_95': 0.5
    }
    
    should_stop, warning = monitor.check_batch(problem_guardrails, 0, gradient_norm=8.0)
    print(f"CNN Problem batch: stop={should_stop}, warning='{warning}'")
    
    # Test CNN early stopper
    stopper = CNNEarlyStopper(patience=5, convergence_window=3)
    
    # Simulate converging metrics
    for i in range(5):
        metrics = {'val_loss': 0.5 + 0.001 * i, 'val_f1': 0.8, 'val_ece': 0.1}
        should_stop = stopper.step(metrics)
        print(f"CNN Early stopper step {i+1}: stop={should_stop}")
        if should_stop:
            break
    
    print("✅ CNN trainer components test passed!")
